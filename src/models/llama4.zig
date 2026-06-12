//! Llama 4 transformer model implementation.
//!
//! Supports the Llama 4 architecture with:
//! * iRoPE: alternating RoPE (local) and NoPE (global) attention layers
//! * Chunked attention: RoPE layers attend within fixed-size non-overlapping chunks
//! * Temperature scaling on NoPE (global) layers
//! * MoE routing with top-1 expert + optional shared expert
//! * Dense FFN fallback for layers without a router tensor
//! * Per-head QK RMSNorm applied AFTER RoPE (only on RoPE layers)
//! * SiLU+SwiGLU activation (NOT GELU)
//! * PagedKvCache with optional KV quantization

const std = @import("std");
const backend_mod = @import("../backend/backend.zig");
const format_mod = @import("../format/format.zig");
const model_mod = @import("model.zig");
const math_ops = @import("../ops/math.zig");
const attn_ops = @import("../ops/attention.zig");
const quant = @import("../ops/quant.zig");
const perf = @import("../perf.zig");
const kv_quant = @import("../ops/kv_quant.zig");
const kvcache = @import("../kvcache/manager.zig");
const block_alloc_mod = @import("../kvcache/block_allocator.zig");
const BlockAllocator = block_alloc_mod.BlockAllocator;
const TieredBlockAllocator = block_alloc_mod.TieredBlockAllocator;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;
const Backend = backend_mod.Backend;
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;
const PagedKvCache = kvcache.PagedKvCache;
const SeqBlockTable = kvcache.SeqBlockTable;

// ── Named constants ─────────────────────────────────────────────

/// Default number of transformer layers.
const default_n_layers: u32 = 48;
/// Default embedding dimension.
const default_n_embd: u32 = 5120;
/// Default attention head count.
const default_n_head: u32 = 40;
/// Default KV head count (GQA).
const default_n_head_kv: u32 = 8;
/// Default head dimension.
const default_head_dim: u32 = 128;
/// Default feed-forward intermediate dimension.
const default_n_ff: u32 = 14336;
/// Default vocabulary size.
const default_vocab_size: u32 = 202048;
/// Default RoPE frequency base.
const default_rope_theta: f32 = 500_000.0;
/// Default RMS layer-norm epsilon.
const default_rms_eps: f32 = 1e-5;
/// Default NoPE interval: every Nth layer is global (NoPE). Value 4 means
/// layers 3, 7, 11, ... are NoPE (3 RoPE then 1 NoPE repeating).
const default_nope_interval: u32 = 4;
/// Default chunk size for local (RoPE) chunked attention.
const default_chunk_size: u32 = 8192;
/// Default prefill chunk size (tokens per batched GEMM).
const default_pf_chunk_size: u32 = 512;
/// Default fallback maximum sequence length.
const default_max_seq_len: usize = 131072;
/// Floor scale for NoPE temperature computation.
const default_floor_scale: f32 = 8192.0;
/// Maximum cached norm weight entries.
const max_norm_entries: usize = 512;
/// Maximum top-k experts for stack-allocated selection arrays.
const max_active_experts: usize = 16;

/// Llama 4 transformer model with iRoPE (interleaved RoPE/NoPE), chunked
/// attention, temperature-scaled NoPE global attention, and MoE routing.
///
/// The iRoPE pattern alternates between local (RoPE + chunked attention) layers
/// and global (NoPE + temperature scaling) layers. A layer is NoPE when
/// `(layer_id + 1) % nope_interval == 0`.
///
/// MoE layers use top-1 expert routing with an optional shared expert. Some
/// layers may be dense (no router tensor present).
pub const Llama4Model = struct {
    const NormCacheEntry = model_mod.NormCacheEntry;

    // ── Configuration ───────────────────────────────────────────
    n_layers: u32,
    n_embd: u32,
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    n_ff: u32,
    vocab_size: u32,
    rope_theta: f32,
    rms_eps: f32,
    eos_token_id: u32,
    /// NoPE interval: every Nth layer uses global NoPE attention.
    nope_interval: u32,
    /// Chunk size for local (RoPE) chunked attention.
    chunk_size: u32,
    /// Number of experts (0 = all layers dense).
    n_experts: u32,
    /// Number of active experts per token (top-k, always 1 for Llama 4).
    n_experts_active: u32,
    /// Attention temperature floor scale for NoPE layers.
    attn_floor_scale: f32,
    /// Attention temperature scale factor for NoPE layers (from metadata).
    attn_temp_scale: f32,
    max_seq_len: usize,

    // ── Dependencies ────────────────────────────────────────────
    fmt: Format,
    be: Backend,
    allocator: Allocator,

    // ── Working buffers ─────────────────────────────────────────
    hidden: []f32 = &.{},
    hidden2: []f32 = &.{},
    q_buf: []f32 = &.{},
    k_buf: []f32 = &.{},
    v_buf: []f32 = &.{},
    attn_out: []f32 = &.{},
    ff_gate: []f32 = &.{},
    ff_up: []f32 = &.{},
    logits_buf: []f32 = &.{},
    scores: []f32 = &.{},
    /// Router logits buffer (allocated only when n_experts > 0).
    router_logits: []f32 = &.{},
    /// MoE weighted output accumulator (allocated only when n_experts > 0).
    moe_out: []f32 = &.{},

    // ── Prefill buffers (sized to pf_chunk_size * dim, page-aligned for GPU) ──
    pf_hidden: []f32 = &.{},
    pf_hidden2: []f32 = &.{},
    pf_q: []f32 = &.{},
    pf_k: []f32 = &.{},
    pf_v: []f32 = &.{},
    pf_attn_out: []f32 = &.{},
    pf_ff_gate: []f32 = &.{},
    pf_ff_up: []f32 = &.{},
    pf_positions: []u32 = &.{},
    /// Prefill chunk size (tokens per batched GEMM, separate from attention chunk_size).
    pf_chunk_size: u32 = default_pf_chunk_size,

    /// Cached pre-computed f32 norm weights (lazily populated on first token).
    norm_cache: [max_norm_entries]NormCacheEntry = undefined,
    norm_cache_len: usize = 0,

    // ── KV cache ────────────────────────────────────────────────
    paged_cache: PagedKvCache = undefined,
    seq_table: SeqBlockTable = undefined,
    block_allocator: BlockAllocator = undefined,
    tiered_cache: ?*TieredKvCache = null,
    tiered_block_allocator: ?TieredBlockAllocator = null,
    kv_type_k: kv_quant.KvQuantType,
    kv_type_v: kv_quant.KvQuantType,
    /// Number of boundary layers (first/last N) that use f16 V.
    kv_boundary_v: u32 = 0,
    kv_seq_len: usize = 0,
    layer_skip_start: u32 = 0,
    layer_skip_end: u32 = 0,
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    perf: perf.PerfCounters = .{},
    /// Enable fused megakernel for single-dispatch forward pass.
    megakernel_enabled: bool = false,

    // ── Initialization ──────────────────────────────────────────

    /// Initialize the model from format metadata and allocate all working buffers.
    /// When `tiered_cache` is provided, the model uses tiered block allocation
    /// instead of creating its own PagedKvCache (the tiered cache is owned externally).
    pub fn init(allocator: Allocator, f: Format, be: Backend, ctx_size: u32, kv_type_k: kv_quant.KvQuantType, kv_type_v: kv_quant.KvQuantType, tiered_cache: ?*TieredKvCache) !Llama4Model {
        const arch = f.getMetaStr("general.architecture") orelse "llama4";
        const n_layers = f.getArchU32(arch, "block_count") orelse default_n_layers;
        const n_embd = f.getArchU32(arch, "embedding_length") orelse default_n_embd;
        const n_head = f.getArchU32(arch, "attention.head_count") orelse default_n_head;
        const n_head_kv = f.getArchU32(arch, "attention.head_count_kv") orelse default_n_head_kv;
        const head_dim = f.getArchU32(arch, "attention.key_length") orelse default_head_dim;
        const n_ff = f.getArchU32(arch, "feed_forward_length") orelse default_n_ff;
        const vocab_size: u32 = if (f.getVocab()) |v| @intCast(v.len) else default_vocab_size;

        const qkv_dim = n_head * head_dim;
        const kv_dim = n_head_kv * head_dim;
        const nl: usize = n_layers;

        var max_sl: usize = default_max_seq_len;
        if (f.getArchU32(arch, "context_length")) |cl| max_sl = cl;
        if (ctx_size > 0) max_sl = ctx_size;

        // Expert count: 0 means all layers are dense.
        const n_experts = f.getArchU32(arch, "expert_count") orelse 0;
        const n_experts_active = f.getArchU32(arch, "expert_used_count") orelse 1;

        // Determine the maximum FFN buffer size needed — fits both dense and expert paths.
        const expert_ff = f.getArchU32(arch, "expert_feed_forward_length") orelse n_ff;
        const max_ff: usize = @max(n_ff, expert_ff);

        var self = Llama4Model{
            .n_layers = n_layers,
            .n_embd = n_embd,
            .n_head = n_head,
            .n_head_kv = n_head_kv,
            .head_dim = head_dim,
            .n_ff = n_ff,
            .vocab_size = vocab_size,
            .rope_theta = f.getArchF32(arch, "rope.freq_base") orelse default_rope_theta,
            .rms_eps = f.getArchF32(arch, "attention.layer_norm_rms_epsilon") orelse default_rms_eps,
            .eos_token_id = f.getMetaU32("tokenizer.ggml.eos_token_id") orelse 128009,
            .nope_interval = f.getArchU32(arch, "attention.sliding_window_pattern") orelse default_nope_interval,
            .chunk_size = f.getArchU32(arch, "attention.sliding_window") orelse default_chunk_size,
            .n_experts = n_experts,
            .n_experts_active = n_experts_active,
            .attn_floor_scale = default_floor_scale,
            .attn_temp_scale = f.getArchF32(arch, "attention.temperature_scale") orelse
                f.getMetaF32("attn_temperature_scale") orelse 0.1,
            .max_seq_len = max_sl,
            .fmt = f,
            .be = be,
            .allocator = allocator,
            .kv_type_k = kv_type_k,
            .kv_type_v = kv_type_v,
            .tiered_cache = tiered_cache,
        };

        // KV cache: use TieredKvCache if provided, otherwise flat PagedKvCache.
        if (tiered_cache) |tc| {
            var ta = TieredBlockAllocator.init(tc, allocator);
            self.seq_table = try ta.allocateSeqTable(nl);
            errdefer ta.freeSeqTable(&self.seq_table);
            try ta.appendBlock(&self.seq_table);
            self.tiered_block_allocator = ta;
        } else {
            const paged_block_size: u16 = 256;
            const blocks_per_layer = (max_sl + paged_block_size - 1) / paged_block_size;
            const num_blocks = nl * blocks_per_layer;
            self.paged_cache = try PagedKvCache.init(allocator, nl, kv_dim, num_blocks, paged_block_size);
            errdefer self.paged_cache.deinit();
            self.block_allocator = BlockAllocator.init(&self.paged_cache, allocator);
            self.seq_table = try self.block_allocator.allocateSeqTable(nl);
            errdefer self.block_allocator.freeSeqTable(&self.seq_table);
            try self.block_allocator.appendBlock(&self.seq_table);
        }

        self.hidden = try allocator.alloc(f32, n_embd);
        errdefer allocator.free(self.hidden);
        self.hidden2 = try allocator.alloc(f32, n_embd);
        errdefer allocator.free(self.hidden2);
        self.q_buf = try allocator.alloc(f32, qkv_dim);
        errdefer allocator.free(self.q_buf);
        self.k_buf = try allocator.alloc(f32, kv_dim);
        errdefer allocator.free(self.k_buf);
        self.v_buf = try allocator.alloc(f32, kv_dim);
        errdefer allocator.free(self.v_buf);
        self.attn_out = try allocator.alloc(f32, qkv_dim);
        errdefer allocator.free(self.attn_out);
        self.ff_gate = try allocator.alloc(f32, max_ff);
        errdefer allocator.free(self.ff_gate);
        self.ff_up = try allocator.alloc(f32, max_ff);
        errdefer allocator.free(self.ff_up);
        self.logits_buf = try allocator.alloc(f32, vocab_size);
        errdefer allocator.free(self.logits_buf);
        self.scores = try allocator.alloc(f32, max_sl);
        errdefer allocator.free(self.scores);

        // MoE-specific buffers
        if (n_experts > 0) {
            self.router_logits = try allocator.alloc(f32, n_experts);
            errdefer allocator.free(self.router_logits);
            self.moe_out = try allocator.alloc(f32, n_embd);
            errdefer allocator.free(self.moe_out);
        }

        // Prefill buffers use page allocator for GPU compatibility (Metal's
        // newBufferWithBytesNoCopy requires page-aligned pointers).
        const pa = std.heap.page_allocator;
        const pcs: usize = default_pf_chunk_size;
        self.pf_hidden = try pa.alloc(f32, pcs * n_embd);
        errdefer pa.free(self.pf_hidden);
        self.pf_hidden2 = try pa.alloc(f32, pcs * n_embd);
        errdefer pa.free(self.pf_hidden2);
        self.pf_q = try pa.alloc(f32, pcs * qkv_dim);
        errdefer pa.free(self.pf_q);
        self.pf_k = try pa.alloc(f32, pcs * kv_dim);
        errdefer pa.free(self.pf_k);
        self.pf_v = try pa.alloc(f32, pcs * kv_dim);
        errdefer pa.free(self.pf_v);
        self.pf_attn_out = try pa.alloc(f32, pcs * qkv_dim);
        errdefer pa.free(self.pf_attn_out);
        self.pf_ff_gate = try pa.alloc(f32, pcs * max_ff);
        errdefer pa.free(self.pf_ff_gate);
        self.pf_ff_up = try pa.alloc(f32, pcs * max_ff);
        errdefer pa.free(self.pf_ff_up);
        self.pf_positions = try pa.alloc(u32, pcs);
        errdefer pa.free(self.pf_positions);

        // Pre-populate norm cache so no allocations happen during inference.
        self.warmNormCache();

        return self;
    }

    /// Release all heap allocations owned by this model.
    pub fn deinit(self: *Llama4Model) void {
        self.allocator.free(self.hidden);
        self.allocator.free(self.hidden2);
        self.allocator.free(self.q_buf);
        self.allocator.free(self.k_buf);
        self.allocator.free(self.v_buf);
        self.allocator.free(self.attn_out);
        self.allocator.free(self.ff_gate);
        self.allocator.free(self.ff_up);
        self.allocator.free(self.logits_buf);
        self.allocator.free(self.scores);
        if (self.n_experts > 0) {
            self.allocator.free(self.router_logits);
            self.allocator.free(self.moe_out);
        }
        const pa = std.heap.page_allocator;
        const pf_bufs = .{
            &self.pf_hidden, &self.pf_hidden2,  &self.pf_q,       &self.pf_k,
            &self.pf_v,      &self.pf_attn_out, &self.pf_ff_gate, &self.pf_ff_up,
        };
        inline for (pf_bufs) |buf| if (buf.len > 0) pa.free(buf.*);
        if (self.pf_positions.len > 0) pa.free(self.pf_positions);
        for (self.norm_cache[0..self.norm_cache_len]) |entry| self.allocator.free(entry.data);
        if (self.tiered_block_allocator) |*ta| {
            ta.freeSeqTable(&self.seq_table);
        } else {
            self.block_allocator.freeSeqTable(&self.seq_table);
            self.paged_cache.deinit();
        }
    }

    /// Wrap this model in the generic `Model` interface.
    pub fn model(self: *Llama4Model) Model {
        return Model.from(Llama4Model, self);
    }

    // ── Forward pass ────────────────────────────────────────────

    /// Run one decode step: process `token_id` through all layers and return
    /// the argmax next-token ID.
    ///
    /// Each layer applies:
    ///   1. Pre-attention RMSNorm
    ///   2. QKV projections
    ///   3. iRoPE: RoPE on local layers, skip on NoPE global layers
    ///   4. QK RMSNorm (local layers only, applied after RoPE)
    ///   5. Temperature scaling (NoPE layers only)
    ///   6. SDPA (chunked for local, full for global)
    ///   7. Output projection + residual
    ///   8. FFN (dense SwiGLU or MoE top-1 + shared expert)
    pub fn forward(self: *Llama4Model, token_id: u32) !u32 {
        if (self.kv_seq_len >= self.max_seq_len) return error.KVCacheFull;

        try model_mod.ensureKvBlock(self);

        // Embedding lookup
        var t = self.perf.start();
        self.embLookup(token_id);
        self.perf.end(.emb_lookup, t);

        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.monotonic)) return error.Cancelled;
            const l: u32 = @intCast(li);
            if (l >= self.layer_skip_start and l < self.layer_skip_end) continue;
            self.fmt.prefetchLayer(@intCast(li + 1));
            try self.attention(l);
            try self.feedForward(l);
        }

        // Final norm -> logits -> argmax
        t = self.perf.start();
        const norm_w = self.fmt.getTensor("output_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(norm_w, self.n_embd), self.hidden.ptr, self.hidden.len, self.rms_eps);
        self.perf.end(.rms_norm, t);

        t = self.perf.start();
        const out_w = self.fmt.getTensor("output.weight") orelse
            self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        self.doGemv(self.hidden.ptr, out_w, self.logits_buf.ptr, self.vocab_size, self.n_embd);
        self.perf.end(.gemv_ffn, t);

        self.be.sync();

        self.kv_seq_len += 1;
        self.perf.addToken();
        return math_ops.argmax(self.logits_buf);
    }

    /// Batched prefill: process all token_ids through all layers using batched
    /// GEMM. Splits into chunks of `pf_chunk_size` tokens. Returns argmax of
    /// the last token's logits.
    pub fn prefill(self: *Llama4Model, token_ids: []const u32) !u32 {
        if (token_ids.len == 0) return error.MissingTensor;
        if (token_ids.len > self.max_seq_len) return error.KVCacheFull;

        // MLX models: sequential per-token processing (no batched MLX GEMM kernel).
        const is_mlx = (self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor).dtype == .mlx_q;
        const cs: usize = if (is_mlx) 1 else self.pf_chunk_size;
        if (cs <= 1 or token_ids.len == 1) {
            var last: u32 = 0;
            for (token_ids) |tid| last = try self.forward(tid);
            return last;
        }

        var offset: usize = 0;
        while (offset < token_ids.len) {
            const chunk_len = @min(cs, token_ids.len - offset);
            try self.prefillChunk(token_ids[offset..][0..chunk_len], @intCast(offset));
            offset += chunk_len;
        }

        // Final: rmsNorm + logits on the LAST token only
        const last_in_chunk = (token_ids.len - 1) % cs;
        const e: usize = self.n_embd;
        @memcpy(self.hidden, self.pf_hidden[last_in_chunk * e ..][0..e]);

        const norm_w = self.fmt.getTensor("output_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(norm_w, e), self.hidden.ptr, e, self.rms_eps);

        const out_w = self.fmt.getTensor("output.weight") orelse
            self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        self.doGemv(self.hidden.ptr, out_w, self.logits_buf.ptr, self.vocab_size, e);

        self.be.sync();

        self.kv_seq_len = token_ids.len;
        self.perf.addToken();
        return math_ops.argmax(self.logits_buf);
    }

    /// Process one chunk of tokens through all layers during prefill.
    fn prefillChunk(self: *Llama4Model, token_ids: []const u32, base_pos: u32) !void {
        const n_tok = token_ids.len;
        const e: usize = self.n_embd;

        // Ensure KV blocks allocated for all new positions
        for (0..n_tok) |t| {
            self.kv_seq_len = base_pos + t;
            try model_mod.ensureKvBlock(self);
        }

        // Embedding lookup for all tokens
        for (token_ids, 0..) |tid, t| {
            self.embLookup(tid);
            @memcpy(self.pf_hidden[t * e ..][0..e], self.hidden);
        }

        // Build position array
        for (0..n_tok) |t| {
            self.pf_positions[t] = base_pos + @as(u32, @intCast(t));
        }

        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.monotonic)) return error.Cancelled;
            const l: u32 = @intCast(li);
            if (l >= self.layer_skip_start and l < self.layer_skip_end) continue;
            self.fmt.prefetchLayer(@intCast(li + 1));
            try self.prefillAttention(l, n_tok);
            try self.prefillFeedForward(l, n_tok);
        }

        self.kv_seq_len = base_pos + n_tok;
    }

    /// Batched attention for prefill: pre-norm, QKV GEMM, iRoPE, QK norm,
    /// temperature scaling, SDPA, output projection, residual.
    fn prefillAttention(self: *Llama4Model, li: u32, n_tok: usize) !void {
        const e: usize = self.n_embd;
        const nh: usize = self.n_head;
        const nkv: usize = self.n_head_kv;
        const hd: usize = self.head_dim;
        const is_nope = self.isNopeLayer(li);

        // Pre-attention norm (batched)
        const norm_w = self.fmt.layerTensor(li, "attn_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNormBatched(self.pf_hidden.ptr, self.normAsF32(norm_w, e), self.pf_hidden2.ptr, n_tok, e, self.rms_eps);

        // QKV projections (batched GEMM)
        const qw = self.fmt.layerTensor(li, "attn_q.weight") orelse return error.MissingTensor;
        const kw = self.fmt.layerTensor(li, "attn_k.weight") orelse return error.MissingTensor;
        const vw = self.fmt.layerTensor(li, "attn_v.weight") orelse return error.MissingTensor;
        self.doGemm(self.pf_hidden2.ptr, qw, self.pf_q.ptr, n_tok, nh * hd, e);
        self.doGemm(self.pf_hidden2.ptr, kw, self.pf_k.ptr, n_tok, nkv * hd, e);
        self.doGemm(self.pf_hidden2.ptr, vw, self.pf_v.ptr, n_tok, nkv * hd, e);

        // iRoPE: apply RoPE only on local (non-NoPE) layers
        if (!is_nope) {
            self.be.ropeBatched(self.pf_q.ptr, self.pf_positions.ptr, n_tok, nh, hd, hd, self.rope_theta);
            self.be.ropeBatched(self.pf_k.ptr, self.pf_positions.ptr, n_tok, nkv, hd, hd, self.rope_theta);

            // QK RMSNorm — applied AFTER RoPE on local layers only
            if (self.fmt.layerTensor(li, "attn_q_norm.weight")) |qn| {
                const kn = self.fmt.layerTensor(li, "attn_k_norm.weight") orelse return error.MissingTensor;
                self.be.rmsNormMulti(self.pf_q.ptr, self.normAsF32(qn, hd), n_tok * nh, hd, self.rms_eps);
                self.be.rmsNormMulti(self.pf_k.ptr, self.normAsF32(kn, hd), n_tok * nkv, hd, self.rms_eps);
            }
        }

        // Temperature scaling for NoPE layers: scale Q per-position
        if (is_nope) {
            self.be.sync();
            const qkv_dim = nh * hd;
            for (0..n_tok) |t| {
                const pos_f: f32 = @floatFromInt(self.pf_positions[t] + 1);
                const temp = @log(@floor(pos_f / self.attn_floor_scale) + 1.0) * self.attn_temp_scale + 1.0;
                if (temp != 1.0) {
                    math_ops.simdScaleF32(self.pf_q.ptr + t * qkv_dim, temp, qkv_dim);
                }
            }
        }

        // SDPA: fused causal prefill attention
        const kv_view = self.getLayerKvView(li);
        const kv_keys_bytes: []u8 = std.mem.sliceAsBytes(kv_view.keys);
        const kv_values_bytes: []u8 = std.mem.sliceAsBytes(kv_view.values);
        const attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const prev_len: usize = self.pf_positions[0];
        self.be.sdpaPrefill(self.pf_q.ptr, self.pf_k.ptr, self.pf_v.ptr, kv_keys_bytes, kv_values_bytes, self.pf_attn_out.ptr, nh, nkv, hd, prev_len, n_tok, attn_scale, .f32, .f32);

        // Output projection (batched GEMM)
        const ow = self.fmt.layerTensor(li, "attn_output.weight") orelse return error.MissingTensor;
        self.doGemm(self.pf_attn_out.ptr, ow, self.pf_hidden2.ptr, n_tok, e, nh * hd);

        // Residual: hidden += attn_output
        self.be.add(self.pf_hidden.ptr, self.pf_hidden2.ptr, self.pf_hidden.ptr, n_tok * e);
    }

    /// Batched feed-forward for prefill. Dense layers use batched GEMM;
    /// MoE layers fall back to per-token sequential routing since each token
    /// may route to different experts.
    fn prefillFeedForward(self: *Llama4Model, li: u32, n_tok: usize) !void {
        const e: usize = self.n_embd;

        // Pre-FFN norm (batched)
        const norm_w = self.fmt.layerTensor(li, "ffn_norm.weight") orelse
            self.fmt.layerTensor(li, "post_attention_layernorm.weight") orelse return error.MissingTensor;
        self.be.rmsNormBatched(self.pf_hidden.ptr, self.normAsF32(norm_w, e), self.pf_hidden2.ptr, n_tok, e, self.rms_eps);

        if (self.fmt.layerTensor(li, "ffn_gate_inp.weight")) |_| {
            // MoE layer: process tokens sequentially (different expert routing per token)
            try self.prefillMoeLayer(li, n_tok);
        } else {
            // Dense FFN: fully batched
            self.prefillDenseFFN(li, n_tok);
        }
    }

    /// Dense SwiGLU FFN during prefill: batched gate+up GEMM, siluMul, down GEMM, residual.
    fn prefillDenseFFN(self: *Llama4Model, li: u32, n_tok: usize) void {
        const e: usize = self.n_embd;
        const ff: usize = self.n_ff;

        const gw = self.fmt.layerTensor(li, "ffn_gate.weight") orelse return;
        const uw = self.fmt.layerTensor(li, "ffn_up.weight") orelse return;
        const dw = self.fmt.layerTensor(li, "ffn_down.weight") orelse return;
        self.doGemm(self.pf_hidden2.ptr, gw, self.pf_ff_gate.ptr, n_tok, ff, e);
        self.doGemm(self.pf_hidden2.ptr, uw, self.pf_ff_up.ptr, n_tok, ff, e);

        self.be.siluMul(self.pf_ff_gate.ptr, self.pf_ff_up.ptr, self.pf_ff_gate.ptr, n_tok * ff);

        self.doGemm(self.pf_ff_gate.ptr, dw, self.pf_hidden2.ptr, n_tok, e, ff);

        // Residual: hidden += ffn_output
        self.be.add(self.pf_hidden.ptr, self.pf_hidden2.ptr, self.pf_hidden.ptr, n_tok * e);
    }

    /// MoE FFN during prefill: each token is routed independently through top-K
    /// experts plus optional shared expert. Uses per-token sequential processing
    /// because each token routes to different experts.
    fn prefillMoeLayer(self: *Llama4Model, li: u32, n_tok: usize) !void {
        const e: usize = self.n_embd;
        const n_exp: usize = self.n_experts;
        const n_active: usize = self.n_experts_active;

        // Fetch packed expert tensor metadata
        const gate_exps = self.fmt.layerTensor(li, "ffn_gate_exps.weight") orelse return error.MissingTensor;
        const up_exps = self.fmt.layerTensor(li, "ffn_up_exps.weight") orelse return error.MissingTensor;
        const down_exps = self.fmt.layerTensor(li, "ffn_down_exps.weight") orelse return error.MissingTensor;
        const gate_stride = model_mod.expertWeightStride(gate_exps);
        const up_stride = model_mod.expertWeightStride(up_exps);
        const down_stride = model_mod.expertWeightStride(down_exps);
        const expert_ff: usize = @intCast(gate_exps.dims[0]);

        // Router weight tensor
        const rw = self.fmt.layerTensor(li, "ffn_gate_inp.weight") orelse return error.MissingTensor;

        // Shared expert tensors (optional, fetched once)
        const shared_gate = self.fmt.layerTensor(li, "ffn_gate_shexp.weight");
        const shared_up = self.fmt.layerTensor(li, "ffn_up_shexp.weight");
        const shared_down = self.fmt.layerTensor(li, "ffn_down_shexp.weight");

        for (0..n_tok) |t| {
            const hidden2_ptr: [*]f32 = self.pf_hidden2.ptr + t * e;

            // Router: logits = router_weight @ hidden2[t]
            self.doGemv(hidden2_ptr, rw, self.router_logits.ptr, n_exp, e);
            self.be.sync();

            // Softmax over router logits
            {
                var max_logit: f32 = self.router_logits[0];
                for (1..n_exp) |i| if (self.router_logits[i] > max_logit) {
                    max_logit = self.router_logits[i];
                };
                var sum_e: f32 = 0.0;
                for (0..n_exp) |i| {
                    self.router_logits[i] = @exp(self.router_logits[i] - max_logit);
                    sum_e += self.router_logits[i];
                }
                const inv = if (sum_e > 0.0) 1.0 / sum_e else 0.0;
                for (0..n_exp) |i| self.router_logits[i] *= inv;
            }

            var top_experts: [max_active_experts]usize = undefined;
            var top_scores: [max_active_experts]f32 = undefined;
            math_ops.topKExperts(self.router_logits[0..n_exp], n_active, top_experts[0..n_active], top_scores[0..n_active]);

            // Renormalize selected weights
            {
                var sel_sum: f32 = 0.0;
                for (0..n_active) |i| sel_sum += top_scores[i];
                const inv = if (sel_sum > 0.0) 1.0 / sel_sum else 0.0;
                for (0..n_active) |i| top_scores[i] *= inv;
            }

            // Accumulate weighted expert outputs into moe_out
            @memset(self.moe_out[0..e], 0);

            for (0..n_active) |ti| {
                const ei = top_experts[ti];
                const mix_weight = top_scores[ti];

                const gate_data = gate_exps.data_ptr + ei * gate_stride;
                const up_data = up_exps.data_ptr + ei * up_stride;
                const GemvOp = backend_mod.GemvOp;
                const exp_ops = [_]GemvOp{
                    .{ .w = .{ .data = gate_data, .dtype = gate_exps.dtype }, .y = self.ff_gate.ptr, .n = expert_ff },
                    .{ .w = .{ .data = up_data, .dtype = up_exps.dtype }, .y = self.ff_up.ptr, .n = expert_ff },
                };
                self.be.gemvMulti(hidden2_ptr, &exp_ops, e);

                self.be.siluMul(self.ff_gate.ptr, self.ff_up.ptr, self.ff_gate.ptr, expert_ff);

                const down_data = down_exps.data_ptr + ei * down_stride;
                self.be.gemv(self.ff_gate.ptr, .{ .data = down_data, .dtype = down_exps.dtype }, self.attn_out.ptr, e, expert_ff);
                self.be.sync();

                self.be.addScaled(self.attn_out.ptr, self.moe_out.ptr, mix_weight, e);
            }

            // Shared expert (optional)
            if (shared_gate) |sg| {
                const su = shared_up orelse continue;
                const sd = shared_down orelse continue;
                const shared_ff: usize = @intCast(sg.dims[0]);

                self.be.beginBatch();
                self.doGemv(hidden2_ptr, sg, self.ff_gate.ptr, shared_ff, e);
                self.doGemv(hidden2_ptr, su, self.ff_up.ptr, shared_ff, e);
                self.be.endBatch();
                self.be.siluMul(self.ff_gate.ptr, self.ff_up.ptr, self.ff_gate.ptr, shared_ff);
                self.doGemv(self.ff_gate.ptr, sd, self.attn_out.ptr, e, shared_ff);
                self.be.sync();

                self.be.addScaled(self.attn_out.ptr, self.moe_out.ptr, 1.0, e);
            }

            // Residual: pf_hidden[t] += moe_out
            const hidden_ptr: [*]f32 = self.pf_hidden.ptr + t * e;
            self.be.add(hidden_ptr, self.moe_out.ptr, hidden_ptr, e);
        }
    }

    /// Reset the KV cache position for a new conversation.
    pub fn resetCache(self: *Llama4Model) void {
        model_mod.resetKvCache(self);
    }

    /// Signal an in-progress forward pass to abort. Thread-safe.
    pub fn cancel(self: *Llama4Model) void {
        model_mod.signalCancel(&self.cancelled);
    }

    /// Return physical block IDs from layer 0 of the current sequence table.
    pub fn getBlockTable(self: *Llama4Model) []const u32 {
        return self.seq_table.block_table[0];
    }

    // ── Layer implementations ───────────────────────────────────

    /// Returns true when the given layer is a NoPE (global attention) layer.
    /// NoPE layers skip RoPE and use temperature scaling + full context attention.
    inline fn isNopeLayer(self: *const Llama4Model, layer: u32) bool {
        if (self.nope_interval == 0) return false;
        return ((layer + 1) % self.nope_interval) == 0;
    }

    /// Attention layer: pre-norm, QKV projection, iRoPE, QK norm, temperature
    /// scaling, SDPA, output projection, residual.
    fn attention(self: *Llama4Model, li: u32) !void {
        const e: usize = self.n_embd;
        const nh: usize = self.n_head;
        const nkv: usize = self.n_head_kv;
        const hd: usize = self.head_dim;
        const is_nope = self.isNopeLayer(li);

        // Pre-attention norm
        var t = self.perf.start();
        const norm_w = self.fmt.layerTensor(li, "attn_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(norm_w, e), self.hidden2.ptr, self.hidden.len, self.rms_eps);
        self.perf.end(.rms_norm, t);

        // QKV projections
        t = self.perf.start();
        const qw = self.fmt.layerTensor(li, "attn_q.weight") orelse return error.MissingTensor;
        const kw = self.fmt.layerTensor(li, "attn_k.weight") orelse return error.MissingTensor;
        const vw = self.fmt.layerTensor(li, "attn_v.weight") orelse return error.MissingTensor;
        self.be.beginBatch();
        self.doGemv(self.hidden2.ptr, qw, self.q_buf.ptr, nh * hd, e);
        self.doGemv(self.hidden2.ptr, kw, self.k_buf.ptr, nkv * hd, e);
        self.doGemv(self.hidden2.ptr, vw, self.v_buf.ptr, nkv * hd, e);
        self.be.endBatch();
        self.perf.end(.gemv_qkv, t);

        // iRoPE: apply RoPE only on local (non-NoPE) layers
        if (!is_nope) {
            t = self.perf.start();
            self.be.beginBatch();
            self.be.rope(self.q_buf.ptr, self.kv_seq_len, nh, hd, hd, self.rope_theta);
            self.be.rope(self.k_buf.ptr, self.kv_seq_len, nkv, hd, hd, self.rope_theta);
            self.be.endBatch();
            self.perf.end(.rope, t);

            // QK RMSNorm — applied AFTER RoPE on local layers only
            if (self.fmt.layerTensor(li, "attn_q_norm.weight")) |qn| {
                t = self.perf.start();
                const kn = self.fmt.layerTensor(li, "attn_k_norm.weight") orelse return error.MissingTensor;
                self.be.beginBatch();
                self.be.rmsNormMulti(self.q_buf.ptr, self.normAsF32(qn, hd), nh, hd, self.rms_eps);
                self.be.rmsNormMulti(self.k_buf.ptr, self.normAsF32(kn, hd), nkv, hd, self.rms_eps);
                self.be.endBatch();
                self.perf.end(.rms_norm, t);
            }
        }

        // Temperature scaling for NoPE layers:
        // scale = log(floor((pos+1) / floor_scale) + 1) * attn_scale + 1.0
        // Applied to all Q values (per-position, same scale for all heads/dims)
        if (is_nope) {
            self.be.sync();
            const pos_f: f32 = @floatFromInt(self.kv_seq_len + 1);
            const temp = @log(@floor(pos_f / self.attn_floor_scale) + 1.0) * self.attn_temp_scale + 1.0;
            if (temp != 1.0) {
                const qkv_dim = nh * hd;
                math_ops.simdScaleF32(self.q_buf.ptr, temp, qkv_dim);
            }
        }

        // SDPA: chunked for local (RoPE) layers, full for global (NoPE) layers.
        // For local layers, pass window config to restrict attention to current chunk.
        t = self.perf.start();
        const kv_view = self.getLayerKvView(li);
        const kv_keys_bytes: []u8 = std.mem.sliceAsBytes(kv_view.keys);
        const kv_values_bytes: []u8 = std.mem.sliceAsBytes(kv_view.values);
        const attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

        if (!is_nope and self.chunk_size > 0) {
            // Chunked attention: attend within non-overlapping chunks of chunk_size.
            // Window start = floor(pos / chunk_size) * chunk_size
            const cs: usize = self.chunk_size;
            const win_start = (self.kv_seq_len / cs) * cs;
            const win_len = self.kv_seq_len + 1 - win_start;
            attn_ops.scaledDotProductAttention(
                self.q_buf.ptr,
                kv_keys_bytes,
                kv_values_bytes,
                self.k_buf,
                self.v_buf,
                self.attn_out.ptr,
                self.scores.ptr,
                nh,
                nkv,
                hd,
                self.kv_seq_len,
                attn_scale,
                self.be,
                .{ .start = win_start, .len = win_len },
                0,
                .f32,
                .f32,
            );
        } else if (self.isMultiBlock(li)) {
            // Full attention with paged KV cache
            self.be.sdpaPaged(
                self.q_buf.ptr,
                self.getPagedKvView(li),
                self.k_buf.ptr,
                self.v_buf.ptr,
                self.attn_out.ptr,
                nh,
                nkv,
                hd,
                attn_scale,
                .f32,
                .f32,
            );
        } else {
            // Full attention with single-block fast path
            attn_ops.scaledDotProductAttention(
                self.q_buf.ptr,
                kv_keys_bytes,
                kv_values_bytes,
                self.k_buf,
                self.v_buf,
                self.attn_out.ptr,
                self.scores.ptr,
                nh,
                nkv,
                hd,
                self.kv_seq_len,
                attn_scale,
                self.be,
                null,
                0,
                .f32,
                .f32,
            );
        }
        self.perf.end(.sdpa, t);

        // Output projection + residual
        t = self.perf.start();
        const ow = self.fmt.layerTensor(li, "attn_output.weight") orelse return error.MissingTensor;
        self.doGemv(self.attn_out.ptr, ow, self.hidden2.ptr, e, nh * hd);
        self.perf.end(.gemv_out, t);

        t = self.perf.start();
        // Residual add deferred: feedForward/denseFFN fuses add(hidden, hidden2) + rmsNorm.
        self.perf.end(.add, t);
    }

    /// Feed-forward layer: pre-FFN norm, then either dense SwiGLU or MoE routing.
    /// Detects per-layer whether the layer is MoE (has ffn_gate_inp) or dense.
    fn feedForward(self: *Llama4Model, li: u32) !void {
        // Check for MoE router tensor to determine layer type
        if (self.fmt.layerTensor(li, "ffn_gate_inp.weight")) |_| {
            try self.moeLayer(li);
        } else {
            try self.denseFFN(li);
        }
    }

    /// Dense SwiGLU FFN: fused (residual-add + pre-norm), gate+up projections, SiLU*mul,
    /// down projection, residual. hidden2 holds the deferred attention output on entry.
    fn denseFFN(self: *Llama4Model, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.n_ff;

        // Fused residual add + pre-FFN norm (hidden2 = deferred attention output).
        var t = self.perf.start();
        const norm_w = self.fmt.layerTensor(li, "ffn_norm.weight") orelse
            self.fmt.layerTensor(li, "post_attention_layernorm.weight") orelse return error.MissingTensor;
        self.be.addRmsNorm(self.hidden.ptr, self.hidden2.ptr, self.normAsF32(norm_w, e), self.hidden2.ptr, self.hidden.len, self.rms_eps);
        self.perf.end(.rms_norm, t);

        t = self.perf.start();
        const gw = self.fmt.layerTensor(li, "ffn_gate.weight") orelse return error.MissingTensor;
        const uw = self.fmt.layerTensor(li, "ffn_up.weight") orelse return error.MissingTensor;
        const dw = self.fmt.layerTensor(li, "ffn_down.weight") orelse return error.MissingTensor;
        self.be.beginBatch();
        self.doGemv(self.hidden2.ptr, gw, self.ff_gate.ptr, ff, e);
        self.doGemv(self.hidden2.ptr, uw, self.ff_up.ptr, ff, e);
        self.be.endBatch();
        self.perf.end(.gemv_ffn, t);

        t = self.perf.start();
        self.be.siluMul(self.ff_gate.ptr, self.ff_up.ptr, self.ff_gate.ptr, ff);
        self.perf.end(.gemv_ffn, t);

        t = self.perf.start();
        self.doGemv(self.ff_gate.ptr, dw, self.hidden2.ptr, e, ff);
        self.perf.end(.gemv_ffn, t);

        t = self.perf.start();
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
        self.perf.end(.add, t);
    }

    /// MoE FFN layer: router + top-1 expert + optional shared expert + residual.
    fn moeLayer(self: *Llama4Model, li: u32) !void {
        const e: usize = self.n_embd;
        const n_exp: usize = self.n_experts;
        const n_active: usize = self.n_experts_active;

        // Fused residual add + pre-FFN norm (hidden2 = deferred attention output).
        var t = self.perf.start();
        const norm_w = self.fmt.layerTensor(li, "ffn_norm.weight") orelse
            self.fmt.layerTensor(li, "post_attention_layernorm.weight") orelse return error.MissingTensor;
        self.be.addRmsNorm(self.hidden.ptr, self.hidden2.ptr, self.normAsF32(norm_w, e), self.hidden2.ptr, self.hidden.len, self.rms_eps);
        self.perf.end(.rms_norm, t);

        // Router: logits = router_weight @ hidden2
        t = self.perf.start();
        const rw = self.fmt.layerTensor(li, "ffn_gate_inp.weight") orelse return error.MissingTensor;
        self.doGemv(self.hidden2.ptr, rw, self.router_logits.ptr, n_exp, e);
        self.be.sync();
        self.perf.end(.gemv_ffn, t);

        // Softmax over router logits, then top-K selection
        {
            var max_logit: f32 = self.router_logits[0];
            for (1..n_exp) |i| if (self.router_logits[i] > max_logit) {
                max_logit = self.router_logits[i];
            };
            var sum_e: f32 = 0.0;
            for (0..n_exp) |i| {
                self.router_logits[i] = @exp(self.router_logits[i] - max_logit);
                sum_e += self.router_logits[i];
            }
            const inv = if (sum_e > 0.0) 1.0 / sum_e else 0.0;
            for (0..n_exp) |i| self.router_logits[i] *= inv;
        }

        var top_experts: [max_active_experts]usize = undefined;
        var top_scores: [max_active_experts]f32 = undefined;
        math_ops.topKExperts(self.router_logits[0..n_exp], n_active, top_experts[0..n_active], top_scores[0..n_active]);

        // Renormalize selected weights to sum to 1.0
        {
            var sel_sum: f32 = 0.0;
            for (0..n_active) |i| sel_sum += top_scores[i];
            const inv = if (sel_sum > 0.0) 1.0 / sel_sum else 0.0;
            for (0..n_active) |i| top_scores[i] *= inv;
        }

        // Fetch packed expert tensor metadata
        const gate_exps = self.fmt.layerTensor(li, "ffn_gate_exps.weight") orelse return error.MissingTensor;
        const up_exps = self.fmt.layerTensor(li, "ffn_up_exps.weight") orelse return error.MissingTensor;
        const down_exps = self.fmt.layerTensor(li, "ffn_down_exps.weight") orelse return error.MissingTensor;
        const gate_stride = model_mod.expertWeightStride(gate_exps);
        const up_stride = model_mod.expertWeightStride(up_exps);
        const down_stride = model_mod.expertWeightStride(down_exps);

        // Infer expert FFN dim from the gate tensor shape
        // GGUF 3D: dims = [cols, rows, n_experts], expert output = dims[0]
        const expert_ff: usize = @intCast(gate_exps.dims[0]);

        // Accumulate weighted expert outputs
        @memset(self.moe_out[0..e], 0);

        for (0..n_active) |ti| {
            const ei = top_experts[ti];
            const mix_weight = top_scores[ti];

            t = self.perf.start();
            const gate_data = gate_exps.data_ptr + ei * gate_stride;
            const up_data = up_exps.data_ptr + ei * up_stride;
            const GemvOp = backend_mod.GemvOp;
            const exp_ops = [_]GemvOp{
                .{ .w = .{ .data = gate_data, .dtype = gate_exps.dtype }, .y = self.ff_gate.ptr, .n = expert_ff },
                .{ .w = .{ .data = up_data, .dtype = up_exps.dtype }, .y = self.ff_up.ptr, .n = expert_ff },
            };
            self.be.gemvMulti(self.hidden2.ptr, &exp_ops, e);
            self.perf.end(.gemv_ffn, t);

            // SwiGLU: silu(gate) * up
            t = self.perf.start();
            self.be.siluMul(self.ff_gate.ptr, self.ff_up.ptr, self.ff_gate.ptr, expert_ff);

            // Down projection -> attn_out (reused as scratch, >= n_embd)
            const down_data = down_exps.data_ptr + ei * down_stride;
            self.be.gemv(self.ff_gate.ptr, .{ .data = down_data, .dtype = down_exps.dtype }, self.attn_out.ptr, e, expert_ff);
            self.be.sync();
            self.perf.end(.gemv_ffn, t);

            // Weighted accumulation
            self.be.addScaled(self.attn_out.ptr, self.moe_out.ptr, mix_weight, e);
        }

        // Shared expert (optional)
        if (self.fmt.layerTensor(li, "ffn_gate_shexp.weight")) |sg| {
            t = self.perf.start();
            const su = self.fmt.layerTensor(li, "ffn_up_shexp.weight") orelse return error.MissingTensor;
            const sd = self.fmt.layerTensor(li, "ffn_down_shexp.weight") orelse return error.MissingTensor;

            // Infer shared expert FFN dim from gate tensor
            const shared_ff: usize = @intCast(sg.dims[0]);

            self.be.beginBatch();
            self.doGemv(self.hidden2.ptr, sg, self.ff_gate.ptr, shared_ff, e);
            self.doGemv(self.hidden2.ptr, su, self.ff_up.ptr, shared_ff, e);
            self.be.endBatch();
            self.be.siluMul(self.ff_gate.ptr, self.ff_up.ptr, self.ff_gate.ptr, shared_ff);
            self.doGemv(self.ff_gate.ptr, sd, self.attn_out.ptr, e, shared_ff);
            self.be.sync();
            self.perf.end(.gemv_ffn, t);

            // Add shared expert output to MoE accumulator
            self.be.addScaled(self.attn_out.ptr, self.moe_out.ptr, 1.0, e);
        }

        // Residual: hidden += moe_out
        t = self.perf.start();
        self.be.add(self.hidden.ptr, self.moe_out.ptr, self.hidden.ptr, e);
        self.perf.end(.add, t);
    }

    // ── Helpers ──────────────────────────────────────────────────

    /// First/last `kv_boundary_v` layers use f16 to protect attention quality;
    /// middle layers use the configured kv_type_v.
    inline fn layerVType(self: *const Llama4Model, li: u32) kv_quant.KvQuantType {
        if (self.kv_boundary_v == 0) return self.kv_type_v;
        const b = self.kv_boundary_v;
        if (li < b or li >= self.n_layers - b) return .f16;
        return self.kv_type_v;
    }

    /// Get flat f32 view of KV cache for a layer from paged/tiered blocks.
    fn getLayerKvView(self: *Llama4Model, layer: usize) struct { keys: []f32, values: []f32 } {
        const num_blocks = self.seq_table.block_table[layer].len;
        if (num_blocks == 0) return .{ .keys = &[_]f32{}, .values = &[_]f32{} };

        const block_id = self.seq_table.block_table[layer][0];
        if (self.tiered_cache) |tc| {
            return .{
                .keys = tc.blocks[block_id].base.keys,
                .values = tc.blocks[block_id].base.values,
            };
        }
        return .{
            .keys = self.paged_cache.blocks[block_id].keys,
            .values = self.paged_cache.blocks[block_id].values,
        };
    }

    const PagedKvView = kvcache.PagedKvView;

    fn getPagedKvView(self: *Llama4Model, layer: usize) PagedKvView {
        return PagedKvView.initView(
            self.seq_table.block_table[layer],
            self.paged_cache.blocks,
            self.paged_cache.block_size,
            self.paged_cache.kv_dim,
            self.kv_seq_len,
        );
    }

    fn isMultiBlock(self: *Llama4Model, layer: usize) bool {
        return self.seq_table.block_table[layer].len > 1;
    }

    /// Embedding lookup: fetch token embedding from weight table.
    fn embLookup(self: *Llama4Model, tok: u32) void {
        const t = self.fmt.getTensor("token_embd.weight") orelse {
            @memset(self.hidden, 0);
            return;
        };
        self.be.embLookup(.{ .data = t.data_ptr, .dtype = t.dtype }, tok, self.hidden.ptr, self.n_embd);
        self.be.sync();
    }

    /// GEMV dispatch: routes through model_mod.dispatchGemv for MLX/NVFP4/GPTQ
    /// support in addition to standard quantized formats.
    fn doGemv(self: *Llama4Model, x: [*]const f32, t: TensorInfo, y: [*]f32, n: usize, k: usize) void {
        model_mod.dispatchGemv(self.be, self.fmt, x, t, y, n, k);
    }

    /// Batched GEMM dispatch: for MLX quantized formats, falls back to
    /// per-token GEMV since no batched MLX GEMM kernel exists.
    fn doGemm(self: *Llama4Model, x: [*]const f32, t: TensorInfo, y: [*]f32, n_tok: usize, n_out: usize, n_in: usize) void {
        if (t.dtype == .mlx_q) {
            for (0..n_tok) |i| {
                self.doGemv(x + i * n_in, t, y + i * n_out, n_out, n_in);
            }
            return;
        }
        self.be.gemm(x, .{ .data = t.data_ptr, .dtype = t.dtype }, y, n_tok, n_out, n_in);
    }

    /// Pre-populate the norm weight cache during init so no allocations occur
    /// in the hot path.
    fn warmNormCache(self: *Llama4Model) void {
        const e: usize = self.n_embd;
        const hd: usize = self.head_dim;
        if (self.fmt.getTensor("output_norm.weight")) |t| _ = self.normAsF32(t, e);
        for (0..self.n_layers) |i| {
            const li: u32 = @intCast(i);
            if (self.fmt.layerTensor(li, "attn_norm.weight")) |t| _ = self.normAsF32(t, e);
            if (self.fmt.layerTensor(li, "attn_q_norm.weight")) |t| _ = self.normAsF32(t, hd);
            if (self.fmt.layerTensor(li, "attn_k_norm.weight")) |t| _ = self.normAsF32(t, hd);
            if (self.fmt.layerTensor(li, "ffn_norm.weight")) |t| _ = self.normAsF32(t, e);
            if (self.fmt.layerTensor(li, "post_attention_layernorm.weight")) |t| _ = self.normAsF32(t, e);
        }
    }

    /// Get norm weights as f32 pointer. Caches converted weights on first access
    /// so subsequent tokens return a stable pointer with zero work.
    fn normAsF32(self: *Llama4Model, t: TensorInfo, n: usize) [*]const f32 {
        if (t.dtype == .f32) return @ptrCast(@alignCast(t.data_ptr));

        // Check cache (linear scan — bounded by max_norm_entries)
        const key = @intFromPtr(t.data_ptr);
        for (self.norm_cache[0..self.norm_cache_len]) |entry| {
            if (entry.key == key) return entry.data.ptr;
        }

        // Cache miss: allocate, convert, store permanently.
        if (self.norm_cache_len >= max_norm_entries)
            @panic("normAsF32: norm cache overflow — increase max_norm_entries");
        const buf = self.allocator.alloc(f32, n) catch |err| {
            std.log.warn("normAsF32: alloc failed ({s}), using raw pointer", .{@errorName(err)});
            return @ptrCast(@alignCast(t.data_ptr));
        };
        if (t.dtype == .bf16) {
            const src: [*]const u16 = @ptrCast(@alignCast(t.data_ptr));
            for (0..n) |i| buf[i] = quant.bf16ToF32(src[i]);
        } else {
            quant.dequantToF32(buf, t.data_ptr, t.dtype, n);
        }
        self.norm_cache[self.norm_cache_len] = .{ .key = key, .data = buf };
        self.norm_cache_len += 1;
        return buf.ptr;
    }
};

// ── Tests ─────────────────────────────────────────────────────────

test "Llama4 isNopeLayer with interval 4" {
    var m: Llama4Model = undefined;
    m.nope_interval = 4;
    // NoPE when (layer+1) % 4 == 0 → layers 3, 7, 11, ...
    try std.testing.expect(!m.isNopeLayer(0)); // 1 % 4 != 0
    try std.testing.expect(!m.isNopeLayer(1)); // 2 % 4 != 0
    try std.testing.expect(!m.isNopeLayer(2)); // 3 % 4 != 0
    try std.testing.expect(m.isNopeLayer(3)); // 4 % 4 == 0
    try std.testing.expect(!m.isNopeLayer(4)); // 5 % 4 != 0
    try std.testing.expect(m.isNopeLayer(7)); // 8 % 4 == 0
    try std.testing.expect(m.isNopeLayer(11)); // 12 % 4 == 0
}

test "Llama4 isNopeLayer with interval 0 — no NoPE layers" {
    var m: Llama4Model = undefined;
    m.nope_interval = 0;
    try std.testing.expect(!m.isNopeLayer(0));
    try std.testing.expect(!m.isNopeLayer(3));
    try std.testing.expect(!m.isNopeLayer(47));
}

test "Llama4 layerVType boundary protection" {
    var m: Llama4Model = undefined;
    m.n_layers = 48;
    m.kv_type_v = .turbo3;

    // No boundary
    m.kv_boundary_v = 0;
    try std.testing.expectEqual(kv_quant.KvQuantType.turbo3, m.layerVType(0));
    try std.testing.expectEqual(kv_quant.KvQuantType.turbo3, m.layerVType(47));

    // Boundary = 4: first/last 4 layers use f16
    m.kv_boundary_v = 4;
    try std.testing.expectEqual(kv_quant.KvQuantType.f16, m.layerVType(0));
    try std.testing.expectEqual(kv_quant.KvQuantType.f16, m.layerVType(3));
    try std.testing.expectEqual(kv_quant.KvQuantType.turbo3, m.layerVType(4));
    try std.testing.expectEqual(kv_quant.KvQuantType.turbo3, m.layerVType(24));
    try std.testing.expectEqual(kv_quant.KvQuantType.turbo3, m.layerVType(43));
    try std.testing.expectEqual(kv_quant.KvQuantType.f16, m.layerVType(44));
    try std.testing.expectEqual(kv_quant.KvQuantType.f16, m.layerVType(47));
}

test "Llama4 NoPE temperature scaling formula" {
    // temp = log(floor((pos+1) / floor_scale) + 1) * attn_temp_scale + 1.0
    const floor_scale: f32 = 8192.0;
    const attn_temp_scale: f32 = 0.1;

    // Position 0: log(floor(1/8192) + 1) * 0.1 + 1 = log(0+1)*0.1 + 1 = 0 + 1 = 1.0
    const pos0: f32 = @floatFromInt(@as(u32, 0) + 1);
    const temp0 = @log(@floor(pos0 / floor_scale) + 1.0) * attn_temp_scale + 1.0;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), temp0, 1e-5);

    // Position 8190 (last in first chunk): floor(8191/8192) = 0 → same as pos 0
    const pos8190: f32 = @floatFromInt(@as(u32, 8190) + 1);
    const temp8191 = @log(@floor(pos8190 / floor_scale) + 1.0) * attn_temp_scale + 1.0;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), temp8191, 1e-5);

    // Position 8192: log(floor(8193/8192) + 1) * 0.1 + 1 = log(2)*0.1 + 1
    const pos8192: f32 = @floatFromInt(@as(u32, 8192) + 1);
    const temp8192 = @log(@floor(pos8192 / floor_scale) + 1.0) * attn_temp_scale + 1.0;
    const expected = @log(@as(f32, 2.0)) * 0.1 + 1.0;
    try std.testing.expectApproxEqAbs(expected, temp8192, 1e-5);
}

test "Llama4 chunked attention window" {
    // Window start = floor(pos / chunk_size) * chunk_size
    // Window length = pos + 1 - window_start
    const chunk_size: usize = 8192;

    // Position 100: window [0, 101)
    const pos0: usize = 100;
    const win_start0 = (pos0 / chunk_size) * chunk_size;
    const win_len0 = pos0 + 1 - win_start0;
    try std.testing.expectEqual(@as(usize, 0), win_start0);
    try std.testing.expectEqual(@as(usize, 101), win_len0);

    // Position 8192: new chunk starts, window [8192, 8193)
    const pos1: usize = 8192;
    const win_start1 = (pos1 / chunk_size) * chunk_size;
    const win_len1 = pos1 + 1 - win_start1;
    try std.testing.expectEqual(@as(usize, 8192), win_start1);
    try std.testing.expectEqual(@as(usize, 1), win_len1);

    // Position 8300: window [8192, 8301)
    const pos2: usize = 8300;
    const win_start2 = (pos2 / chunk_size) * chunk_size;
    const win_len2 = pos2 + 1 - win_start2;
    try std.testing.expectEqual(@as(usize, 8192), win_start2);
    try std.testing.expectEqual(@as(usize, 109), win_len2);
}

test "Llama4 model vtable compiles" {
    try std.testing.expect(@hasDecl(Llama4Model, "forward"));
    try std.testing.expect(@hasDecl(Llama4Model, "prefill"));
    try std.testing.expect(@hasDecl(Llama4Model, "resetCache"));
    try std.testing.expect(@hasDecl(Llama4Model, "cancel"));
    try std.testing.expect(@hasDecl(Llama4Model, "model"));
}

test "Llama4 isNopeLayer with interval 1 — all NoPE" {
    var m: Llama4Model = undefined;
    m.nope_interval = 1;
    // (layer+1) % 1 == 0 is always true
    try std.testing.expect(m.isNopeLayer(0));
    try std.testing.expect(m.isNopeLayer(1));
    try std.testing.expect(m.isNopeLayer(47));
}

test "Llama4 isNopeLayer with interval 2 — alternating" {
    var m: Llama4Model = undefined;
    m.nope_interval = 2;
    // NoPE when (layer+1) % 2 == 0 -> layers 1, 3, 5, ...
    try std.testing.expect(!m.isNopeLayer(0)); // 1 % 2 = 1
    try std.testing.expect(m.isNopeLayer(1)); // 2 % 2 = 0
    try std.testing.expect(!m.isNopeLayer(2)); // 3 % 2 = 1
    try std.testing.expect(m.isNopeLayer(3)); // 4 % 2 = 0
}

test "Llama4 layerVType boundary covers all layers" {
    var m: Llama4Model = undefined;
    m.n_layers = 48;
    m.kv_type_v = .turbo3;
    // Boundary = 24: half the layers on each side -> all layers get f16
    m.kv_boundary_v = 24;
    for (0..48) |i| {
        try std.testing.expectEqual(kv_quant.KvQuantType.f16, m.layerVType(@intCast(i)));
    }
}

test "Llama4 layerVType boundary = 1 — only first and last layer" {
    var m: Llama4Model = undefined;
    m.n_layers = 48;
    m.kv_type_v = .turbo3;
    m.kv_boundary_v = 1;
    try std.testing.expectEqual(kv_quant.KvQuantType.f16, m.layerVType(0));
    try std.testing.expectEqual(kv_quant.KvQuantType.turbo3, m.layerVType(1));
    try std.testing.expectEqual(kv_quant.KvQuantType.turbo3, m.layerVType(46));
    try std.testing.expectEqual(kv_quant.KvQuantType.f16, m.layerVType(47));
}

test "Llama4 default constants are consistent" {
    // Verify head_dim * n_head = QKV dimension (embedding dim for standard Llama 4)
    try std.testing.expectEqual(@as(u32, 5120), default_n_head * default_head_dim);
    // KV dimension should be n_head_kv * head_dim
    try std.testing.expectEqual(@as(u32, 1024), default_n_head_kv * default_head_dim);
    // chunk_size should be a power of two
    try std.testing.expect(default_chunk_size > 0);
    try std.testing.expectEqual(@as(u32, 0), default_chunk_size & (default_chunk_size - 1));
    // nope_interval should divide evenly into typical layer counts
    try std.testing.expectEqual(@as(u32, 0), default_n_layers % default_nope_interval);
}

test "Llama4 chunked attention window — chunk boundary edge cases" {
    const chunk_size: usize = 8192;

    // Position 0: first token
    const pos0: usize = 0;
    const win_start0 = (pos0 / chunk_size) * chunk_size;
    const win_len0 = pos0 + 1 - win_start0;
    try std.testing.expectEqual(@as(usize, 0), win_start0);
    try std.testing.expectEqual(@as(usize, 1), win_len0);

    // Last position in first chunk (8191): window covers full chunk
    const pos_last: usize = 8191;
    const win_start_last = (pos_last / chunk_size) * chunk_size;
    const win_len_last = pos_last + 1 - win_start_last;
    try std.testing.expectEqual(@as(usize, 0), win_start_last);
    try std.testing.expectEqual(@as(usize, 8192), win_len_last);

    // Third chunk: position 16384
    const pos3: usize = 16384;
    const win_start3 = (pos3 / chunk_size) * chunk_size;
    const win_len3 = pos3 + 1 - win_start3;
    try std.testing.expectEqual(@as(usize, 16384), win_start3);
    try std.testing.expectEqual(@as(usize, 1), win_len3);
}

test "Llama4 NoPE temperature scaling — large position" {
    const floor_scale: f32 = 8192.0;
    const attn_temp_scale: f32 = 0.1;

    // Position 131071 (max default - 1): floor(131072/8192) = 16
    // temp = log(16 + 1) * 0.1 + 1 = log(17)*0.1 + 1
    const pos: f32 = @floatFromInt(@as(u32, 131071) + 1);
    const temp = @log(@floor(pos / floor_scale) + 1.0) * attn_temp_scale + 1.0;
    const expected = @log(@as(f32, 17.0)) * 0.1 + 1.0;
    try std.testing.expectApproxEqAbs(expected, temp, 1e-5);
    // Temperature should be > 1.0 for long contexts
    try std.testing.expect(temp > 1.0);
}

test "Llama4 function signatures type check" {
    // Verify key method signatures exist and have expected types at comptime
    comptime {
        _ = @TypeOf(Llama4Model.init);
        _ = @TypeOf(Llama4Model.deinit);
        _ = @TypeOf(Llama4Model.forward);
        _ = @TypeOf(Llama4Model.prefill);
        _ = @TypeOf(Llama4Model.resetCache);
        _ = @TypeOf(Llama4Model.cancel);
        _ = @TypeOf(Llama4Model.model);
        _ = @TypeOf(Llama4Model.getBlockTable);
    }
}

test "fuzz: all llama4 pub functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // ── comptime: verify every pub function is reachable ────
            comptime {
                _ = &Llama4Model.init;
                _ = &Llama4Model.deinit;
                _ = &Llama4Model.model;
                _ = &Llama4Model.forward;
                _ = &Llama4Model.prefill;
                _ = &Llama4Model.resetCache;
                _ = &Llama4Model.cancel;
                _ = &Llama4Model.getBlockTable;
            }

            // ── runtime: cancel is the only pub fn callable without
            //    a full Format+Backend init ──────────────────────────
            var m: Llama4Model = undefined;
            m.cancelled = std.atomic.Value(bool).init(false);

            // Fuzz isNopeLayer via cancel-gate pattern
            var raw: [8]u8 = undefined;
            smith.bytesWithHash(&raw, 0);
            const fuzz_interval: u32 = @as(u32, raw[0]) +| 1;
            const fuzz_layer: u32 = @as(u32, raw[1]);
            m.nope_interval = fuzz_interval;

            // isNopeLayer invariant: result matches modular arithmetic
            const expected_nope = ((fuzz_layer +% 1) % fuzz_interval) == 0;
            try std.testing.expectEqual(expected_nope, m.isNopeLayer(fuzz_layer));

            // layerVType invariant: boundary=0 always returns kv_type_v
            m.n_layers = @as(u32, raw[2]) +| 2;
            m.kv_type_v = .turbo3;
            m.kv_boundary_v = 0;
            const layer_idx: u32 = raw[3] % m.n_layers;
            try std.testing.expectEqual(kv_quant.KvQuantType.turbo3, m.layerVType(layer_idx));

            // layerVType with boundary: first/last b layers get f16
            const b = raw[4] % (m.n_layers / 2 + 1);
            m.kv_boundary_v = b;
            const vt = m.layerVType(layer_idx);
            if (b > 0 and (layer_idx < b or layer_idx >= m.n_layers - b)) {
                try std.testing.expectEqual(kv_quant.KvQuantType.f16, vt);
            } else if (b == 0) {
                try std.testing.expectEqual(kv_quant.KvQuantType.turbo3, vt);
            }

            // cancel: sets the atomic flag
            try std.testing.expect(!m.cancelled.load(.monotonic));
            m.cancel();
            try std.testing.expect(m.cancelled.load(.monotonic));

            // cancel is idempotent
            m.cancel();
            try std.testing.expect(m.cancelled.load(.monotonic));
        }
    }.f, .{});
}
