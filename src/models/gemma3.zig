//! Gemma 3 transformer model implementation.
//! Supports GQA (Grouped Query Attention), GELU activation, post-norms,
//! RoPE position embeddings, and per-head QK normalization.

const std = @import("std");
const math = std.math;
const backend_mod = @import("../backend/backend.zig");
const format_mod = @import("../format/format.zig");
const model_mod = @import("model.zig");
const math_ops = @import("../ops/math.zig");
const attn_ops = @import("../ops/attention.zig");
const mlx_ops = @import("../ops/mlx.zig");
const quant = @import("../ops/quant.zig");
const perf = @import("../perf.zig");
const kv_quant = @import("../ops/kv_quant.zig");
const kvcache = @import("../kvcache/manager.zig");
const block_alloc_mod = @import("../kvcache/block_allocator.zig");
const BlockAllocator = block_alloc_mod.BlockAllocator;
const TieredBlockAllocator = block_alloc_mod.TieredBlockAllocator;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;
const split_attn = @import("../ops/split_attention.zig");
const Backend = backend_mod.Backend;
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;
const PagedKvCache = kvcache.PagedKvCache;
const SeqBlockTable = kvcache.SeqBlockTable;

/// Default prefill chunk size (tokens per batch).
const default_chunk_size: u32 = 512;
/// Default sliding window pattern (every Nth layer uses full attention).
const default_sliding_window_pattern: u32 = 6;
/// Default RoPE frequency base for global attention layers.
const default_rope_freq_base: f32 = 1_000_000.0;
/// Default RoPE frequency base for local (sliding-window) attention layers.
const default_rope_local_freq_base: f32 = 10_000.0;
/// Default RMS layer-norm epsilon.
const default_rms_eps: f32 = 1e-6;
/// Fallback layer count for Gemma3 1B when metadata is missing.
const default_n_layers: u32 = 26;
/// Fallback embedding dimension for Gemma3 1B.
const default_n_embd: u32 = 1152;
/// Fallback head dimension for Gemma3.
const default_head_dim: u32 = 256;
/// Fallback feed-forward dimension for Gemma3 1B.
const default_n_ff: u32 = 6912;
/// Fallback vocabulary size for Gemma3.
const default_vocab_size: u32 = 262144;
/// Fallback attention head count for Gemma3 1B.
const default_n_head: u32 = 4;
/// Fallback KV head count for Gemma3 1B.
const default_n_head_kv: u32 = 1;
/// Fallback maximum sequence length when metadata is missing.
const default_max_seq_len: usize = 4096;

/// Gemma 3 transformer model with GQA, GELU activation, and per-head QK normalization.
/// Supports both GGUF and SafeTensors/MLX quantized weights.
pub const Gemma3Model = struct {
    const NormCacheEntry = model_mod.NormCacheEntry;
    const max_norm_entries: usize = 256;

    // Configuration (read from GGUF metadata)
    n_layers: u32,
    n_embd: u32,
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    n_ff: u32,
    vocab_size: u32,
    rope_theta: f32,
    rope_local_theta: f32,
    rope_freq_scale: f32,
    rope_dim: u32,
    sliding_window_pattern: u32,
    rms_eps: f32,
    eos_token_id: u32,
    attn_scale: f32,
    embd_scale: f32,
    final_logit_softcap: f32,
    mlx_bits: u32,
    /// Whether to add +1.0 to norm weights during RMS norm (true for SafeTensors/MLX
    /// which store raw weights; false for GGUF which bakes +1.0 in at conversion time).
    norm_add_one: bool,
    max_seq_len: usize = 4096,

    // Dependencies
    fmt: Format,
    be: Backend,
    allocator: Allocator,

    // Working buffers
    hidden: []f32,
    hidden2: []f32,
    q_buf: []f32,
    k_buf: []f32,
    v_buf: []f32,
    attn_out: []f32,
    ff_gate: []f32,
    ff_up: []f32,
    logits_buf: []f32,
    scores: []f32,

    // Prefill buffers (sized to chunk_size × dim, allocated at init)
    pf_hidden: []f32 = &[_]f32{},
    pf_hidden2: []f32 = &[_]f32{},
    pf_q: []f32 = &[_]f32{},
    pf_k: []f32 = &[_]f32{},
    pf_v: []f32 = &[_]f32{},
    pf_attn_out: []f32 = &[_]f32{},
    pf_ff_gate: []f32 = &[_]f32{},
    pf_ff_up: []f32 = &[_]f32{},
    pf_positions: []u32 = &[_]u32{},
    chunk_size: u32 = default_chunk_size,

    /// Cached pre-computed f32 norm weights (lazily populated on first token).
    /// Eliminates per-token dequantization and all norm-buffer GPU sync points.
    norm_cache: [max_norm_entries]NormCacheEntry = undefined,
    norm_cache_len: usize = 0,
    /// MLX companion tensor cache (scales/biases pointers keyed by weight ptr).
    mlx_cc_keys: [mlx_companion_cache_size]usize = [_]usize{0} ** mlx_companion_cache_size,
    mlx_cc_vals: [mlx_companion_cache_size]MlxCompanion = undefined,

    // KV cache (PagedAttention or TieredKvCache)
    paged_cache: PagedKvCache = undefined,
    seq_table: SeqBlockTable = undefined,
    block_allocator: BlockAllocator = undefined,
    tiered_cache: ?*TieredKvCache = null,
    tiered_block_allocator: ?TieredBlockAllocator = null,
    /// Split-attention output buffers (allocated only when tiered cache is active).
    /// Used by splitAttention() for concurrent GPU + CPU SDPA merge.
    split_gpu_out: []f32 = &[_]f32{},
    split_cpu_out: []f32 = &[_]f32{},
    kv_type_k: kv_quant.KvQuantType,
    kv_type_v: kv_quant.KvQuantType,
    kv_seq_len: usize = 0,
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    perf: perf.PerfCounters = .{},

    /// Initialize the model from format metadata and allocate all working buffers.
    /// When `tiered_cache` is provided, the model uses tiered block allocation instead
    /// of creating its own PagedKvCache (the tiered cache is owned externally).
    pub fn init(allocator: Allocator, f: Format, be: Backend, ctx_size: u32, kv_type_k: kv_quant.KvQuantType, kv_type_v: kv_quant.KvQuantType, tiered_cache: ?*TieredKvCache) !Gemma3Model {
        const arch = f.getMetaStr("general.architecture") orelse "gemma3";
        const n_layers = f.getArchU32(arch, "block_count") orelse default_n_layers;
        const n_embd = f.getArchU32(arch, "embedding_length") orelse default_n_embd;
        const n_head = f.getArchU32(arch, "attention.head_count") orelse default_n_head;
        const n_head_kv = f.getArchU32(arch, "attention.head_count_kv") orelse default_n_head_kv;
        const head_dim = f.getArchU32(arch, "attention.key_length") orelse default_head_dim;
        const n_ff = f.getArchU32(arch, "feed_forward_length") orelse default_n_ff;
        const rope_dim: u32 = head_dim;
        const vocab_size: u32 = if (f.getVocab()) |v| @intCast(v.len) else default_vocab_size;

        const qkv_dim = n_head * head_dim;
        const kv_dim = n_head_kv * head_dim;
        const nl: usize = n_layers;

        var max_sl: usize = default_max_seq_len;
        if (f.getArchU32(arch, "context_length")) |cl| max_sl = cl;
        if (ctx_size > 0) max_sl = ctx_size;

        var self = Gemma3Model{
            .n_layers = n_layers,
            .n_embd = n_embd,
            .n_head = n_head,
            .n_head_kv = n_head_kv,
            .head_dim = head_dim,
            .n_ff = n_ff,
            .vocab_size = vocab_size,
            .rope_theta = f.getArchF32(arch, "rope.freq_base") orelse default_rope_freq_base,
            .rope_local_theta = f.getArchF32(arch, "rope.freq_base_swa") orelse
                f.getMetaF32("rope_local_base_freq") orelse default_rope_local_freq_base,
            .rope_freq_scale = blk: {
                const factor = f.getArchF32(arch, "rope.scaling.factor") orelse
                    f.getMetaF32("rope_scaling_factor") orelse 1.0;
                break :blk if (factor > 0) 1.0 / factor else 1.0;
            },
            .rope_dim = rope_dim,
            .sliding_window_pattern = f.getArchU32(arch, "attention.sliding_window_pattern") orelse
                f.getMetaU32("sliding_window_pattern") orelse
                if (f.getArchU32(arch, "attention.sliding_window")) |_| default_sliding_window_pattern else 0,
            .rms_eps = f.getArchF32(arch, "attention.layer_norm_rms_epsilon") orelse default_rms_eps,
            .eos_token_id = f.getMetaU32("tokenizer.ggml.eos_token_id") orelse 1,
            .attn_scale = blk: {
                // Gemma uses query_pre_attn_scalar (config) or head_dim as the scaling denominator
                const scalar = f.getMetaU32("query_pre_attn_scalar") orelse head_dim;
                break :blk 1.0 / @sqrt(@as(f32, @floatFromInt(scalar)));
            },
            .embd_scale = @sqrt(@as(f32, @floatFromInt(n_embd))),
            .max_seq_len = max_sl,
            .final_logit_softcap = 0.0,
            .mlx_bits = f.getMetaU32("bits") orelse 6,
            // GGUF bakes +1.0 into norm weights (norm_add_one=false).
            // SafeTensors/MLX stores raw weights needing +1.0 (norm_add_one=true).
            // Detect SafeTensors by presence of HF config key "model_type".
            .norm_add_one = f.getMetaStr("model_type") != null,
            .fmt = f,
            .be = be,
            .allocator = allocator,
            .hidden = undefined,
            .hidden2 = undefined,
            .q_buf = undefined,
            .k_buf = undefined,
            .v_buf = undefined,
            .attn_out = undefined,
            .ff_gate = undefined,
            .ff_up = undefined,
            .logits_buf = undefined,
            .scores = undefined,
            .tiered_cache = tiered_cache,
            .kv_type_k = kv_type_k,
            .kv_type_v = kv_type_v,
        };

        // KV cache: use TieredKvCache if provided, otherwise flat PagedKvCache.
        if (tiered_cache) |tc| {
            var ta = TieredBlockAllocator.init(tc, allocator);
            self.seq_table = try ta.allocateSeqTable(nl);
            errdefer ta.freeSeqTable(&self.seq_table);
            try ta.appendBlock(&self.seq_table);
            self.tiered_block_allocator = ta;
            // Allocate split-attention buffers for concurrent GPU + CPU SDPA
            self.split_gpu_out = try allocator.alloc(f32, qkv_dim);
            errdefer allocator.free(self.split_gpu_out);
            self.split_cpu_out = try allocator.alloc(f32, qkv_dim);
            errdefer allocator.free(self.split_cpu_out);
        } else {
            // Use full-sequence blocks for contiguous KV access in both prefill and decode.
            // One block per layer, each holding max_sl positions. Same total memory.
            const block_size: u16 = @intCast(@min(max_sl, std.math.maxInt(u16)));
            const num_blocks = nl;
            self.paged_cache = try PagedKvCache.init(allocator, nl, kv_dim, num_blocks, block_size);
            errdefer self.paged_cache.deinit();
            // BlockAllocator stores a pointer — must point to self.paged_cache (not a local copy).
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
        self.ff_gate = try allocator.alloc(f32, n_ff);
        errdefer allocator.free(self.ff_gate);
        self.ff_up = try allocator.alloc(f32, n_ff);
        errdefer allocator.free(self.ff_up);
        self.logits_buf = try allocator.alloc(f32, vocab_size);
        errdefer allocator.free(self.logits_buf);
        self.scores = try allocator.alloc(f32, max_sl);
        errdefer allocator.free(self.scores);

        // Prefill buffers use page allocator for GPU compatibility (Metal's
        // newBufferWithBytesNoCopy requires page-aligned pointers).
        const pa = std.heap.page_allocator;
        const cs: usize = default_chunk_size;
        self.pf_hidden = try pa.alloc(f32, cs * n_embd);
        errdefer pa.free(self.pf_hidden);
        self.pf_hidden2 = try pa.alloc(f32, cs * n_embd);
        errdefer pa.free(self.pf_hidden2);
        self.pf_q = try pa.alloc(f32, cs * qkv_dim);
        errdefer pa.free(self.pf_q);
        self.pf_k = try pa.alloc(f32, cs * kv_dim);
        errdefer pa.free(self.pf_k);
        self.pf_v = try pa.alloc(f32, cs * kv_dim);
        errdefer pa.free(self.pf_v);
        self.pf_attn_out = try pa.alloc(f32, cs * qkv_dim);
        errdefer pa.free(self.pf_attn_out);
        self.pf_ff_gate = try pa.alloc(f32, cs * n_ff);
        errdefer pa.free(self.pf_ff_gate);
        self.pf_ff_up = try pa.alloc(f32, cs * n_ff);
        errdefer pa.free(self.pf_ff_up);
        self.pf_positions = try pa.alloc(u32, cs);
        errdefer pa.free(self.pf_positions);

        // Pre-populate norm cache so no allocations happen during inference.
        self.warmNormCache();

        return self;
    }

    /// Release all heap allocations owned by this model.
    pub fn deinit(self: *Gemma3Model) void {
        const bufs = .{
            &self.hidden,     &self.hidden2,  &self.q_buf,   &self.k_buf,
            &self.v_buf,      &self.attn_out, &self.ff_gate, &self.ff_up,
            &self.logits_buf, &self.scores,
        };
        inline for (bufs) |buf| self.allocator.free(buf.*);
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
            if (self.split_gpu_out.len > 0) self.allocator.free(self.split_gpu_out);
            if (self.split_cpu_out.len > 0) self.allocator.free(self.split_cpu_out);
        } else {
            self.block_allocator.freeSeqTable(&self.seq_table);
            self.paged_cache.deinit();
        }
    }

    /// Wrap this model in the generic `Model` interface.
    pub fn model(self: *Gemma3Model) Model {
        return Model.from(Gemma3Model, self);
    }

    // ── Forward pass ──────────────────────────────────────────────

    /// Run one decode step, returning the argmax next-token ID.
    pub fn forward(self: *Gemma3Model, token_id: u32) !u32 {
        if (self.kv_seq_len >= self.max_seq_len) return error.KVCacheFull;

        try model_mod.ensureKvBlock(self);

        // Embedding lookup + Gemma scaling
        var t = self.perf.start();
        self.embLookup(token_id);
        self.perf.end(.emb_lookup, t);

        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.monotonic)) return error.Cancelled;
            self.fmt.prefetchLayer(@intCast(li + 1));
            try self.attention(@intCast(li));
            try self.feedForward(@intCast(li));
        }

        // Final norm → logits → argmax
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
        if (self.final_logit_softcap > 0.0) self.applySoftcap();

        self.kv_seq_len += 1;
        self.perf.addToken();
        return math_ops.argmax(self.logits_buf);
    }

    /// Batched prefill: process all token_ids through all layers.
    /// Splits into chunks of `chunk_size` tokens. Returns argmax of
    /// the last token's logits.
    pub fn prefill(self: *Gemma3Model, token_ids: []const u32) !u32 {
        if (token_ids.len == 0) return error.MissingTensor;
        if (token_ids.len > self.max_seq_len) return error.KVCacheFull;

        // MLX models: sequential per-token processing (no batched MLX GEMM kernel).
        const is_mlx = (self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor).dtype == .mlx_q;
        const cs: usize = if (is_mlx) 1 else self.chunk_size;
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
        if (self.final_logit_softcap > 0.0) self.applySoftcap();

        self.kv_seq_len = token_ids.len;
        self.perf.addToken();
        return math_ops.argmax(self.logits_buf);
    }

    fn prefillChunk(self: *Gemma3Model, token_ids: []const u32, base_pos: u32) !void {
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
            self.fmt.prefetchLayer(@intCast(li + 1));
            try self.prefillAttention(@intCast(li), n_tok);
            try self.prefillFeedForward(@intCast(li), n_tok);
        }

        self.kv_seq_len = base_pos + n_tok;
    }

    fn prefillAttention(self: *Gemma3Model, li: u32, n_tok: usize) !void {
        const e: usize = self.n_embd;
        const nh: usize = self.n_head;
        const nkv: usize = self.n_head_kv;
        const hd: usize = self.head_dim;

        const norm_w = self.fmt.layerTensor(li, "attn_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNormBatched(self.pf_hidden.ptr, self.normAsF32(norm_w, e), self.pf_hidden2.ptr, n_tok, e, self.rms_eps);

        const qw = self.fmt.layerTensor(li, "attn_q.weight") orelse return error.MissingTensor;
        const kw = self.fmt.layerTensor(li, "attn_k.weight") orelse return error.MissingTensor;
        const vw = self.fmt.layerTensor(li, "attn_v.weight") orelse return error.MissingTensor;
        self.doGemm(self.pf_hidden2.ptr, qw, self.pf_q.ptr, n_tok, nh * hd, e);
        self.doGemm(self.pf_hidden2.ptr, kw, self.pf_k.ptr, n_tok, nkv * hd, e);
        self.doGemm(self.pf_hidden2.ptr, vw, self.pf_v.ptr, n_tok, nkv * hd, e);

        // Per-head QK norms — treat n_tok*n_heads as total heads
        const qn = self.fmt.layerTensor(li, "attn_q_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNormMulti(self.pf_q.ptr, self.normAsF32(qn, hd), n_tok * nh, hd, self.rms_eps);
        const kn = self.fmt.layerTensor(li, "attn_k_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNormMulti(self.pf_k.ptr, self.normAsF32(kn, hd), n_tok * nkv, hd, self.rms_eps);

        // RoPE
        const rd: usize = self.rope_dim;
        const is_local = self.sliding_window_pattern > 0 and (li + 1) % self.sliding_window_pattern != 0;
        if (is_local) {
            self.be.ropeBatched(self.pf_q.ptr, self.pf_positions.ptr, n_tok, nh, hd, rd, self.rope_local_theta);
            self.be.ropeBatched(self.pf_k.ptr, self.pf_positions.ptr, n_tok, nkv, hd, rd, self.rope_local_theta);
        } else if (self.rope_freq_scale != 1.0) {
            // CPU RoPE with frequency scaling — must sync before reading GPU-written pf_q/pf_k.
            self.be.sync();
            for (0..n_tok) |t| {
                applyRopeScaled(self.pf_q.ptr + t * nh * hd, self.pf_positions[t], nh, hd, rd, self.rope_theta, self.rope_freq_scale);
                applyRopeScaled(self.pf_k.ptr + t * nkv * hd, self.pf_positions[t], nkv, hd, rd, self.rope_theta, self.rope_freq_scale);
            }
        } else {
            self.be.ropeBatched(self.pf_q.ptr, self.pf_positions.ptr, n_tok, nh, hd, rd, self.rope_theta);
            self.be.ropeBatched(self.pf_k.ptr, self.pf_positions.ptr, n_tok, nkv, hd, rd, self.rope_theta);
        }

        // Fused causal attention
        const kv_view = self.getLayerKvView(li);
        const kv_keys_bytes: []u8 = std.mem.sliceAsBytes(kv_view.keys);
        const kv_values_bytes: []u8 = std.mem.sliceAsBytes(kv_view.values);
        const prev_len: usize = self.pf_positions[0];
        // Use .f32 — PagedKvCache blocks store f32. Must match decode path.
        self.be.sdpaPrefill(self.pf_q.ptr, self.pf_k.ptr, self.pf_v.ptr, kv_keys_bytes, kv_values_bytes, self.pf_attn_out.ptr, nh, nkv, hd, prev_len, n_tok, self.attn_scale, .f32, .f32);

        // Output projection
        const ow = self.fmt.layerTensor(li, "attn_output.weight") orelse return error.MissingTensor;
        self.doGemm(self.pf_attn_out.ptr, ow, self.pf_hidden2.ptr, n_tok, e, nh * hd);

        // Post-attention norm + residual
        const post_norm = self.fmt.layerTensor(li, "post_attention_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNormBatched(self.pf_hidden2.ptr, self.normAsF32(post_norm, e), self.pf_hidden2.ptr, n_tok, e, self.rms_eps);
        self.be.add(self.pf_hidden.ptr, self.pf_hidden2.ptr, self.pf_hidden.ptr, n_tok * e);
    }

    fn prefillFeedForward(self: *Gemma3Model, li: u32, n_tok: usize) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.n_ff;

        const norm_w = self.fmt.layerTensor(li, "ffn_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNormBatched(self.pf_hidden.ptr, self.normAsF32(norm_w, e), self.pf_hidden2.ptr, n_tok, e, self.rms_eps);

        const gw = self.fmt.layerTensor(li, "ffn_gate.weight") orelse return error.MissingTensor;
        const uw = self.fmt.layerTensor(li, "ffn_up.weight") orelse return error.MissingTensor;
        const dw = self.fmt.layerTensor(li, "ffn_down.weight") orelse return error.MissingTensor;
        self.doGemm(self.pf_hidden2.ptr, gw, self.pf_ff_gate.ptr, n_tok, ff, e);
        self.doGemm(self.pf_hidden2.ptr, uw, self.pf_ff_up.ptr, n_tok, ff, e);

        self.be.geluMul(self.pf_ff_gate.ptr, self.pf_ff_up.ptr, self.pf_ff_gate.ptr, n_tok * ff);

        self.doGemm(self.pf_ff_gate.ptr, dw, self.pf_hidden2.ptr, n_tok, e, ff);

        const post_norm = self.fmt.layerTensor(li, "post_ffw_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNormBatched(self.pf_hidden2.ptr, self.normAsF32(post_norm, e), self.pf_hidden2.ptr, n_tok, e, self.rms_eps);
        self.be.add(self.pf_hidden.ptr, self.pf_hidden2.ptr, self.pf_hidden.ptr, n_tok * e);
    }

    /// Reset the KV cache position for a new conversation.
    pub fn resetCache(self: *Gemma3Model) void {
        model_mod.resetKvCache(self);
    }

    /// Signal an in-progress forward pass to abort. Thread-safe.
    pub fn cancel(self: *Gemma3Model) void {
        model_mod.signalCancel(&self.cancelled);
    }

    /// Return physical block IDs from layer 0 of the current sequence table.
    /// All layers share the same block IDs, so layer 0 is sufficient.
    pub fn getBlockTable(self: *Gemma3Model) []const u32 {
        return self.seq_table.block_table[0];
    }

    // ── Layer implementations ─────────────────────────────────────

    /// Helper: get flat f32 view of KV cache for a layer (assembled from paged blocks).
    /// Returns slices pointing into paged or tiered cache blocks.
    fn getLayerKvView(self: *Gemma3Model, layer: usize) struct { keys: []f32, values: []f32 } {
        const num_blocks = self.seq_table.block_table[layer].len;
        if (num_blocks == 0) return .{ .keys = &[_]f32{}, .values = &[_]f32{} };

        // For now, assume single block (will extend for multi-block later)
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

    fn attention(self: *Gemma3Model, li: u32) !void {
        const e: usize = self.n_embd;
        const nh: usize = self.n_head;
        const nkv: usize = self.n_head_kv;
        const hd: usize = self.head_dim;
        // Pre-norm
        var t = self.perf.start();
        const norm_w = self.fmt.layerTensor(li, "attn_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(norm_w, e), self.hidden2.ptr, self.hidden.len, self.rms_eps);
        self.perf.end(.rms_norm, t);

        // QKV projections
        t = self.perf.start();
        const qw = self.fmt.layerTensor(li, "attn_q.weight") orelse return error.MissingTensor;
        const kw = self.fmt.layerTensor(li, "attn_k.weight") orelse return error.MissingTensor;
        const vw = self.fmt.layerTensor(li, "attn_v.weight") orelse return error.MissingTensor;
        self.doGemv(self.hidden2.ptr, qw, self.q_buf.ptr, nh * hd, e);
        self.doGemv(self.hidden2.ptr, kw, self.k_buf.ptr, nkv * hd, e);
        self.doGemv(self.hidden2.ptr, vw, self.v_buf.ptr, nkv * hd, e);
        self.perf.end(.gemv_qkv, t);

        // Per-head QK norms (single dispatch per Q/K, not per-head)
        t = self.perf.start();
        const qn = self.fmt.layerTensor(li, "attn_q_norm.weight") orelse return error.MissingTensor;
        const qn_w = self.normAsF32(qn, hd);
        self.be.rmsNormMulti(self.q_buf.ptr, qn_w, nh, hd, self.rms_eps);
        const kn = self.fmt.layerTensor(li, "attn_k_norm.weight") orelse return error.MissingTensor;
        const kn_w = self.normAsF32(kn, hd);
        self.be.rmsNormMulti(self.k_buf.ptr, kn_w, nkv, hd, self.rms_eps);
        self.perf.end(.rms_norm, t);

        // RoPE — partial rotation (rope_dim may be < head_dim)
        // Local layers: theta=10K unscaled; global layers: theta=1M with freq_scale
        t = self.perf.start();
        const rd: usize = self.rope_dim;
        const is_local = self.sliding_window_pattern > 0 and (li + 1) % self.sliding_window_pattern != 0;
        if (is_local) {
            self.be.rope(self.q_buf.ptr, self.kv_seq_len, nh, hd, rd, self.rope_local_theta);
            self.be.rope(self.k_buf.ptr, self.kv_seq_len, nkv, hd, rd, self.rope_local_theta);
        } else if (self.rope_freq_scale != 1.0) {
            // CPU RoPE with frequency scaling — must sync before reading GPU-written q_buf/k_buf.
            self.be.sync();
            applyRopeScaled(self.q_buf.ptr, self.kv_seq_len, nh, hd, rd, self.rope_theta, self.rope_freq_scale);
            applyRopeScaled(self.k_buf.ptr, self.kv_seq_len, nkv, hd, rd, self.rope_theta, self.rope_freq_scale);
        } else {
            self.be.rope(self.q_buf.ptr, self.kv_seq_len, nh, hd, rd, self.rope_theta);
            self.be.rope(self.k_buf.ptr, self.kv_seq_len, nkv, hd, rd, self.rope_theta);
        }
        self.perf.end(.rope, t);

        // KV cache append + scaled dot-product attention
        t = self.perf.start();
        const kv_view = self.getLayerKvView(li);
        const kv_keys_bytes: []u8 = std.mem.sliceAsBytes(kv_view.keys);
        const kv_values_bytes: []u8 = std.mem.sliceAsBytes(kv_view.values);

        // Tiered cache path: partition blocks by tier, run split-attention
        // when KV spans both GPU and CPU memory.
        if (self.tiered_cache) |tc| {
            const partition = split_attn.partitionBlocks(
                self.seq_table.block_table[li],
                tc.blocks,
                tc.block_size,
                self.kv_seq_len + 1,
            );
            // Extract thread pool from CPU backend for parallel CPU SDPA heads
            const pool: ?*@import("../thread_pool.zig").ThreadPool = switch (self.be) {
                .cpu => |cpu| cpu.pool,
                else => null,
            };

            attn_ops.scaledDotProductAttentionTiered(
                self.q_buf.ptr,
                kv_keys_bytes,
                kv_values_bytes,
                self.k_buf,
                self.v_buf,
                self.attn_out.ptr,
                nh,
                nkv,
                hd,
                self.kv_seq_len,
                self.attn_scale,
                self.be,
                .f32,
                .f32,
                .{
                    .partition = partition,
                    .pool = pool,
                    .gpu_out = self.split_gpu_out.ptr,
                    .cpu_out = self.split_cpu_out.ptr,
                },
            );
        } else {
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
                self.attn_scale,
                self.be,
                null,
                0,
                .f32, // PagedKvCache uses f32 blocks
                .f32,
            );
        }
        self.perf.end(.sdpa, t);

        // Output projection + post-norm + residual
        t = self.perf.start();
        const ow = self.fmt.layerTensor(li, "attn_output.weight") orelse return error.MissingTensor;
        self.doGemv(self.attn_out.ptr, ow, self.hidden2.ptr, e, nh * hd);
        self.perf.end(.gemv_out, t);

        t = self.perf.start();
        const post_norm = self.fmt.layerTensor(li, "post_attention_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden2.ptr, self.normAsF32(post_norm, e), self.hidden2.ptr, self.hidden2.len, self.rms_eps);
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
        self.perf.end(.add, t);
    }

    fn feedForward(self: *Gemma3Model, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.n_ff;

        var t = self.perf.start();
        const norm_w = self.fmt.layerTensor(li, "ffn_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(norm_w, e), self.hidden2.ptr, self.hidden.len, self.rms_eps);
        self.perf.end(.rms_norm, t);

        t = self.perf.start();
        const gw = self.fmt.layerTensor(li, "ffn_gate.weight") orelse return error.MissingTensor;
        const uw = self.fmt.layerTensor(li, "ffn_up.weight") orelse return error.MissingTensor;
        const dw = self.fmt.layerTensor(li, "ffn_down.weight") orelse return error.MissingTensor;
        self.doGemv(self.hidden2.ptr, gw, self.ff_gate.ptr, ff, e);
        self.doGemv(self.hidden2.ptr, uw, self.ff_up.ptr, ff, e);
        self.perf.end(.gemv_ffn, t);

        t = self.perf.start();
        self.be.geluMul(self.ff_gate.ptr, self.ff_up.ptr, self.ff_gate.ptr, ff);
        self.perf.end(.gelu_mul, t);

        t = self.perf.start();
        self.doGemv(self.ff_gate.ptr, dw, self.hidden2.ptr, e, ff);
        self.perf.end(.gemv_ffn, t);

        t = self.perf.start();
        const post_norm = self.fmt.layerTensor(li, "post_ffw_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden2.ptr, self.normAsF32(post_norm, e), self.hidden2.ptr, self.hidden2.len, self.rms_eps);
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
        self.perf.end(.add, t);
    }

    /// RoPE with linear frequency scaling for context extension.
    /// angle = (pos * freq_scale) * theta^(-2i/d)
    fn applyRopeScaled(x: [*]f32, pos: usize, n_heads: usize, head_dim: usize, rope_dim_arg: usize, theta: f32, freq_scale: f32) void {
        const half = rope_dim_arg / 2;
        const p: f32 = @as(f32, @floatFromInt(pos)) * freq_scale;
        const inv_rd: f32 = 1.0 / @as(f32, @floatFromInt(rope_dim_arg));
        const neg_log_theta: f32 = -@log(theta);
        for (0..n_heads) |h| {
            const base = h * head_dim;
            for (0..half) |i| {
                const freq = @exp(neg_log_theta * @as(f32, @floatFromInt(2 * i)) * inv_rd);
                const angle = p * freq;
                const c = @cos(angle);
                const s = @sin(angle);
                const r = x[base + i];
                const im = x[base + i + half];
                x[base + i] = r * c - im * s;
                x[base + i + half] = r * s + im * c;
            }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────

    /// Pre-populate the norm weight cache during init so no allocations occur
    /// in the hot path. Iterates all norm tensors and triggers conversion.
    fn warmNormCache(self: *Gemma3Model) void {
        const e: usize = self.n_embd;
        const hd: usize = self.head_dim;
        if (self.fmt.getTensor("output_norm.weight")) |t| _ = self.normAsF32(t, e);
        for (0..self.n_layers) |i| {
            const li: u32 = @intCast(i);
            if (self.fmt.layerTensor(li, "attn_norm.weight")) |t| _ = self.normAsF32(t, e);
            if (self.fmt.layerTensor(li, "attn_q_norm.weight")) |t| _ = self.normAsF32(t, hd);
            if (self.fmt.layerTensor(li, "attn_k_norm.weight")) |t| _ = self.normAsF32(t, hd);
            if (self.fmt.layerTensor(li, "post_attention_norm.weight")) |t| _ = self.normAsF32(t, e);
            if (self.fmt.layerTensor(li, "ffn_norm.weight")) |t| _ = self.normAsF32(t, e);
            if (self.fmt.layerTensor(li, "post_ffw_norm.weight")) |t| _ = self.normAsF32(t, e);
        }
    }

    /// Get norm weights as f32 pointer. Caches converted weights on first access
    /// so subsequent tokens return a stable pointer with zero work and no GPU syncs.
    fn normAsF32(self: *Gemma3Model, t: TensorInfo, n: usize) [*]const f32 {
        const needs_convert = t.dtype == .bf16 or self.norm_add_one;
        if (!needs_convert) return @ptrCast(@alignCast(t.data_ptr));

        // Check cache (linear scan — at most ~170 entries, first-token only on miss)
        const key = @intFromPtr(t.data_ptr);
        for (self.norm_cache[0..self.norm_cache_len]) |entry| {
            if (entry.key == key) return entry.data.ptr;
        }

        // Cache miss: allocate, convert, and store permanently.
        // Guard capacity before allocating to avoid leaking uncached buffers.
        if (self.norm_cache_len >= max_norm_entries)
            @panic("normAsF32: norm cache overflow — increase max_norm_entries");
        const buf = self.allocator.alloc(f32, n) catch |err| {
            std.log.warn("normAsF32: alloc failed ({s}), using unconverted weights", .{@errorName(err)});
            return @ptrCast(@alignCast(t.data_ptr));
        };
        const offset: f32 = if (self.norm_add_one) 1.0 else 0.0;
        if (t.dtype == .bf16) {
            const src: [*]const u16 = @ptrCast(@alignCast(t.data_ptr));
            for (0..n) |i| buf[i] = quant.bf16ToF32(src[i]) + offset;
        } else {
            const src: [*]const f32 = @ptrCast(@alignCast(t.data_ptr));
            for (0..n) |i| buf[i] = src[i] + offset;
        }
        self.norm_cache[self.norm_cache_len] = .{ .key = key, .data = buf };
        self.norm_cache_len += 1;
        return buf.ptr;
    }

    /// Cached companion pointers for MLX quantized weight tensors.
    /// Avoids per-GEMV string formatting + HashMap lookups for .scales/.biases.
    const MlxCompanion = struct { scales: [*]const u8, biases: [*]const u8 };
    const mlx_companion_cache_size: usize = 256;

    /// GEMV dispatch that handles both regular and MLX-quantized weights.
    /// For MLX quantized tensors (dtype == .mlx_q), looks up companion
    /// .scales and .biases tensors via a pointer-keyed cache (first call
    /// per tensor does the HashMap lookup; subsequent calls are O(1)).
    fn doGemv(self: *Gemma3Model, x: [*]const f32, t: TensorInfo, y: [*]f32, n: usize, k: usize) void {
        if (t.dtype != .mlx_q) {
            self.be.gemv(x, .{ .data = t.data_ptr, .dtype = t.dtype }, y, n, k);
            return;
        }
        // Look up companion scale/bias pointers from cache
        const key = @intFromPtr(t.data_ptr);
        const slot = key % mlx_companion_cache_size;
        var companion: MlxCompanion = undefined;
        if (self.mlx_cc_keys[slot] == key) {
            companion = self.mlx_cc_vals[slot];
        } else {
            // Cache miss — resolve via name lookup (once per unique tensor)
            const base_name = t.name;
            const prefix_len = if (std.mem.endsWith(u8, base_name, ".weight"))
                base_name.len - ".weight".len
            else
                base_name.len;
            var sbuf: [model_mod.tensor_name_buf_size]u8 = undefined;
            var bbuf: [model_mod.tensor_name_buf_size]u8 = undefined;
            const s_name = std.fmt.bufPrint(&sbuf, "{s}.scales", .{base_name[0..prefix_len]}) catch return;
            const b_name = std.fmt.bufPrint(&bbuf, "{s}.biases", .{base_name[0..prefix_len]}) catch return;
            const st = self.fmt.getTensor(s_name) orelse return;
            const bt = self.fmt.getTensor(b_name) orelse return;
            companion = .{ .scales = st.data_ptr, .biases = bt.data_ptr };
            self.mlx_cc_keys[slot] = key;
            self.mlx_cc_vals[slot] = companion;
        }
        self.be.gemvMlxQ(
            x,
            t.data_ptr,
            companion.scales,
            companion.biases,
            y,
            n,
            k,
            self.mlx_bits,
        );
    }

    /// GEMM dispatch handling both regular and MLX-quantized weights.
    /// For MLX-Q, falls back to loop-of-doGemv (no native MLX GEMM yet).
    fn doGemm(self: *Gemma3Model, x: [*]const f32, t: TensorInfo, y: [*]f32, n_tok: usize, n_out: usize, n_in: usize) void {
        if (t.dtype == .mlx_q) {
            for (0..n_tok) |i| {
                self.doGemv(x + i * n_in, t, y + i * n_out, n_out, n_in);
            }
            return;
        }
        self.be.gemm(x, .{ .data = t.data_ptr, .dtype = t.dtype }, y, n_tok, n_out, n_in);
    }

    fn embLookup(self: *Gemma3Model, tok: u32) void {
        const t = self.fmt.getTensor("token_embd.weight") orelse {
            @memset(self.hidden, 0);
            return;
        };
        if (t.dtype == .mlx_q) {
            const st = self.fmt.getTensor("token_embd.scales") orelse return;
            const bt = self.fmt.getTensor("token_embd.biases") orelse return;
            mlx_ops.mlxEmbLookup(
                self.hidden.ptr,
                @ptrCast(@alignCast(t.data_ptr)),
                @ptrCast(@alignCast(st.data_ptr)),
                @ptrCast(@alignCast(bt.data_ptr)),
                tok,
                self.n_embd,
                self.mlx_bits,
            );
        } else {
            self.be.embLookup(.{ .data = t.data_ptr, .dtype = t.dtype }, tok, self.hidden.ptr, self.n_embd);
            self.be.sync();
        }
        const V8 = @Vector(8, f32);
        const scale_v: V8 = @splat(self.embd_scale);
        const n: usize = self.n_embd;
        var i: usize = 0;
        while (i + 8 <= n) : (i += 8) {
            self.hidden[i..][0..8].* = @as(V8, self.hidden[i..][0..8].*) * scale_v;
        }
        while (i < n) : (i += 1) self.hidden[i] *= self.embd_scale;
    }

    /// SIMD-vectorized logit softcapping: logits = cap * tanh(logits / cap).
    fn applySoftcap(self: *Gemma3Model) void {
        const V8 = @Vector(8, f32);
        const inv_v: V8 = @splat(1.0 / self.final_logit_softcap);
        const cap_v: V8 = @splat(self.final_logit_softcap);
        const one: V8 = @splat(1.0);
        const two: V8 = @splat(2.0);
        const n = self.logits_buf.len;
        var i: usize = 0;
        while (i + 8 <= n) : (i += 8) {
            const x: V8 = self.logits_buf[i..][0..8].*;
            const t = x * inv_v;
            const e2t = @exp(two * t);
            self.logits_buf[i..][0..8].* = cap_v * (e2t - one) / (e2t + one);
        }
        const inv = 1.0 / self.final_logit_softcap;
        const cap = self.final_logit_softcap;
        while (i < n) : (i += 1) {
            self.logits_buf[i] = cap * @as(f32, math.tanh(self.logits_buf[i] * inv));
        }
    }
};
