//! NVIDIA Nemotron-H — Hybrid Mamba-2 + Attention + FFN decoder.
//!
//! Architecture overview
//! ---------------------
//! * 42 layers of three distinct types, detected at init from tensor presence:
//!   - SSM layers (e.g. 21 in the 8B variant): Mamba-2 selective-state-space with causal conv1d.
//!   - Attention layers (e.g. 4): standard GQA with partial RoPE (rope_dim=78).
//!   - FFN-only layers (e.g. 17): squared-ReLU MLP (no gate tensor).
//! * Embedding dim n_embd=3136, vocab=131072.
//! * All layers share a pre-norm named `attn_norm.weight`.
//! * No attention bias, no MLP bias.
//! * KV cache pre-allocated only for the 4 attention layers.
//! * Mamba-2 SSM state pre-allocated per SSM layer (zero-initialised).
const std = @import("std");
const backend_mod = @import("../backend/backend.zig");
const format_mod = @import("../format/format.zig");
const model_mod = @import("model.zig");
const math_ops = @import("../ops/math.zig");
const attn_ops = @import("../ops/attention.zig");
const ssm_ops = @import("../ops/ssm.zig");
const kvcache = @import("../kvcache/manager.zig");
const block_alloc_mod = @import("../kvcache/block_allocator.zig");
const BlockAllocator = block_alloc_mod.BlockAllocator;
const TieredBlockAllocator = block_alloc_mod.TieredBlockAllocator;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;

const Backend = backend_mod.Backend;
const Format = format_mod.Format;
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;
const kv_quant = @import("../ops/kv_quant.zig");
const PagedKvCache = kvcache.PagedKvCache;
const SeqBlockTable = kvcache.SeqBlockTable;

/// Layer variants in Nemotron-H.
const LayerType = enum { ssm, attention, ffn_only };

/// Maximum supported layer count (controls static array sizes for layer_types).
const max_layers: usize = 128;
/// Norm weight cache (output + attn_norm + ssm_norm per SSM layer).
const max_norm_entries: usize = 512;
const NormCacheEntry = model_mod.NormCacheEntry;
const TensorInfo = format_mod.TensorInfo;
const quant_ops = @import("../ops/quant.zig");

/// Buffer size for tensor name formatting.
const name_buf_size: usize = model_mod.tensor_name_buf_size;

/// Nemotron-H hybrid model state.
pub const NemotronHModel = struct {
    // ── Configuration ─────────────────────────────────────────────
    fmt: Format,
    be: Backend,
    allocator: Allocator,

    /// Number of transformer blocks.
    n_layers: u32 = 42,
    /// Hidden embedding dimension.
    n_embd: u32 = 3136,
    /// Number of query heads (attention layers only).
    n_head: u32 = 40,
    /// Number of KV heads (GQA, attention layers only).
    n_head_kv: u32 = 8,
    /// Per-head key/value dimension.
    head_dim: u32 = 128,
    /// FFN intermediate size (FFN-only layers).
    n_ff: u32 = 12544,
    /// Vocabulary size.
    vocab_size: u32 = 131072,
    /// RoPE base frequency.
    rope_theta: f32 = 10000.0,
    /// Number of head dimensions to rotate with RoPE (partial RoPE).
    rope_dim: u32 = 78,
    /// RMS-norm epsilon.
    rms_eps: f32 = 1e-5,
    /// End-of-sequence token identifier.
    eos_token_id: u32 = 11,
    /// Maximum sequence length for the pre-allocated KV cache.
    max_seq_len: usize = 4096,

    // ── Mamba-2 SSM parameters ────────────────────────────────────
    /// Causal conv kernel size.
    ssm_d_conv: u32 = 4,
    /// SSM state size per head.
    ssm_d_state: u32 = 128,
    /// Number of SSM groups (B/C vectors shared within group).
    ssm_n_group: u32 = 8,
    /// Number of Mamba-2 heads (= dt_rank).
    ssm_dt_rank: u32 = 96,
    /// Mamba-2 inner dimension (= num_heads * head_dim).
    ssm_d_inner: u32 = 7680,

    // ── Layer-type map (populated at init) ────────────────────────
    /// Per-layer type, indexed [0..n_layers).
    layer_types: [max_layers]LayerType = [_]LayerType{.ffn_only} ** max_layers,

    // ── Working buffers (allocated once, reused every token) ──────
    hidden: []f32 = &.{},
    hidden2: []f32 = &.{},
    /// Q projection output — n_head * head_dim elements.
    q_buf: []f32 = &.{},
    /// K projection output — n_head_kv * head_dim elements.
    k_buf: []f32 = &.{},
    /// V projection output — n_head_kv * head_dim elements.
    v_buf: []f32 = &.{},
    /// Attention / SSM output before output projection.
    attn_out: []f32 = &.{},
    /// Dot-product attention score buffer — max_seq_len elements.
    scores_buf: []f32 = &.{},
    /// FFN first half (gate or up).
    ff_buf1: []f32 = &.{},
    /// SSM input projection output — [z(ssm_d_inner) | conv_in(conv_ch) | dt(ssm_dt_rank)].
    ssm_proj_buf: []f32 = &.{},
    /// Causal conv1d output — conv_ch elements.
    ssm_conv_out: []f32 = &.{},
    /// SSM output (y) before gating — ssm_d_inner elements.
    ssm_y_buf: []f32 = &.{},
    /// Final vocabulary logits — vocab_size elements.
    logits_buf: []f32 = &.{},

    // ── Batched-prefill buffers (page_allocator, chunk_size tokens) ──
    /// Prefill chunk size (tokens per batch).
    chunk_size: usize = 256,
    /// Batched hidden states [chunk_size * n_embd].
    pf_hidden: []f32 = &.{},
    /// Batched scratch for norm/projection output [chunk_size * n_embd].
    pf_hidden2: []f32 = &.{},
    /// Batched Q projection [chunk_size * n_head * head_dim].
    pf_q: []f32 = &.{},
    /// Batched K projection [chunk_size * n_head_kv * head_dim].
    pf_k: []f32 = &.{},
    /// Batched V projection [chunk_size * n_head_kv * head_dim].
    pf_v: []f32 = &.{},
    /// Batched attention output [chunk_size * n_head * head_dim].
    pf_attn_out: []f32 = &.{},
    /// Batched FFN up projection [chunk_size * n_ff].
    pf_ff_buf1: []f32 = &.{},
    /// Position indices for batched RoPE [chunk_size].
    pf_positions: []u32 = &.{},

    // ── Per-layer SSM state ───────────────────────────────────────
    /// conv_states[i] = ring buffer [(d_conv-1) * conv_ch] f32, zero-init.
    /// Empty slice for non-SSM layers.
    conv_states: [][]f32 = &.{},
    /// ssm_states[i] = flat array [dt_rank * mamba_head_dim * ssm_d_state] f32, zero-init.
    /// Empty slice for non-SSM layers.
    ssm_states: [][]f32 = &.{},

    // ── Per-layer KV cache (PagedAttention or TieredKvCache) ────────
    paged_cache: PagedKvCache = undefined,
    seq_table: SeqBlockTable = undefined,
    block_allocator: BlockAllocator = undefined,
    tiered_cache: ?*TieredKvCache = null,
    tiered_block_allocator: ?TieredBlockAllocator = null,
    /// KV cache quantization type for keys.
    kv_type_k: kv_quant.KvQuantType = .f32,
    /// KV cache quantization type for values.
    kv_type_v: kv_quant.KvQuantType = .f32,
    /// Number of tokens committed to the KV cache.
    kv_seq_len: usize = 0,
    layer_skip_start: u32 = 0,
    layer_skip_end: u32 = 0,
    /// When true, hidden2 holds an unmerged attention/FFN output. The next layer's pre-norm
    /// fuses the residual add into addRmsNorm(hidden, hidden2, w, hidden2) instead of two dispatches.
    pending_residual: bool = false,
    /// Set to true from another thread to abort an in-progress `forward` call.
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    /// Enable fused megakernel for single-dispatch forward pass.
    megakernel_enabled: bool = false,

    /// Permanent f32 cache for non-f32 RMS norm weights.
    norm_cache: [max_norm_entries]NormCacheEntry = undefined,
    norm_cache_len: usize = 0,

    // ── Lifecycle ─────────────────────────────────────────────────

    /// Initialize the model from format metadata and pre-allocate all buffers.
    /// Caller owns the returned value and must call `deinit` when done.
    pub fn init(allocator: Allocator, f: Format, be: Backend, ctx_size: u32, kv_type_k: kv_quant.KvQuantType, kv_type_v: kv_quant.KvQuantType, tiered_cache: ?*TieredKvCache) !NemotronHModel {
        var self = NemotronHModel{ .fmt = f, .be = be, .allocator = allocator };
        self.kv_type_k = kv_type_k;
        self.kv_type_v = kv_type_v;

        const arch = f.getMetaStr("general.architecture") orelse "nemotron-h";
        if (f.getArchU32(arch, "block_count")) |v| self.n_layers = v;
        if (f.getArchU32(arch, "embedding_length")) |v| self.n_embd = v;
        if (f.getArchU32(arch, "attention.head_count")) |v| self.n_head = v;
        // head_count_kv and feed_forward_length are per-layer arrays in nemotron_h GGUF.
        // getArchU32 returns 0 for array-type metadata, so fall back to struct defaults.
        if (f.getArchU32(arch, "attention.head_count_kv")) |v| {
            if (v > 0) self.n_head_kv = v;
        }
        if (f.getArchU32(arch, "attention.key_length")) |v| self.head_dim = v;
        if (f.getArchU32(arch, "feed_forward_length")) |v| {
            if (v > 0) self.n_ff = v;
        }
        if (f.getArchU32(arch, "ssm.conv_kernel")) |v| self.ssm_d_conv = v;
        if (f.getArchU32(arch, "ssm.state_size")) |v| self.ssm_d_state = v;
        if (f.getArchU32(arch, "ssm.group_count")) |v| self.ssm_n_group = v;
        if (f.getArchU32(arch, "ssm.time_step_rank")) |v| self.ssm_dt_rank = v;
        if (f.getArchU32(arch, "ssm.inner_size")) |v| self.ssm_d_inner = v;
        if (f.getArchU32(arch, "rope.dimension_count")) |v| self.rope_dim = v;
        if (f.getArchF32(arch, "rope.freq_base")) |v| self.rope_theta = v;
        if (f.getArchF32(arch, "attention.layer_norm_rms_epsilon")) |v| self.rms_eps = v;
        if (f.getMetaU32("tokenizer.ggml.eos_token_id")) |v| self.eos_token_id = v;
        if (f.getVocab()) |v| self.vocab_size = @intCast(v.len);
        if (f.getArchU32(arch, "context_length")) |cl| self.max_seq_len = cl;
        if (ctx_size > 0) self.max_seq_len = ctx_size;

        if (self.n_head_kv == 0 or self.n_head % self.n_head_kv != 0) {
            std.log.err("nemotron_h: n_head ({d}) not divisible by n_head_kv ({d})", .{ self.n_head, self.n_head_kv });
            return error.MissingTensor;
        }
        if (self.ssm_dt_rank == 0 or self.ssm_d_inner % self.ssm_dt_rank != 0) {
            std.log.err("nemotron_h: ssm_d_inner ({d}) not divisible by ssm_dt_rank ({d})", .{ self.ssm_d_inner, self.ssm_dt_rank });
            return error.MissingTensor;
        }
        if (self.rope_dim > self.head_dim) {
            std.log.err("nemotron_h: rope_dim ({d}) exceeds head_dim ({d})", .{ self.rope_dim, self.head_dim });
            return error.MissingTensor;
        }
        if (self.rope_dim % 2 != 0) {
            std.log.err("nemotron_h: rope_dim ({d}) is not even", .{self.rope_dim});
            return error.MissingTensor;
        }
        if (self.n_layers > max_layers) {
            std.log.err("nemotron_h: n_layers ({d}) exceeds max_layers ({d})", .{ self.n_layers, max_layers });
            return error.MissingTensor;
        }
        if (self.ssm_n_group == 0 or self.ssm_dt_rank % self.ssm_n_group != 0) {
            std.log.err("nemotron_h: ssm_dt_rank ({d}) not divisible by ssm_n_group ({d})", .{ self.ssm_dt_rank, self.ssm_n_group });
            return error.MissingTensor;
        }

        // ── Layer type detection ──────────────────────────────────
        // Check tensor presence to classify each layer.
        var nb: [name_buf_size]u8 = undefined;
        var n_ssm: usize = 0;
        var n_attn: usize = 0;
        var n_ffn: usize = 0;

        for (0..self.n_layers) |li| {
            const l: u32 = @intCast(li);
            const ssm_name = std.fmt.bufPrint(&nb, "blk.{d}.ssm_in.weight", .{l}) catch break;
            if (f.getTensor(ssm_name) != null) {
                self.layer_types[li] = .ssm;
                n_ssm += 1;
                continue;
            }
            const attn_name = std.fmt.bufPrint(&nb, "blk.{d}.attn_q.weight", .{l}) catch break;
            if (f.getTensor(attn_name) != null) {
                self.layer_types[li] = .attention;
                n_attn += 1;
                continue;
            }
            self.layer_types[li] = .ffn_only;
            n_ffn += 1;
        }

        std.log.info("[nemotron_h] Detected {} SSM, {} attention, {} ffn_only layers", .{ n_ssm, n_attn, n_ffn });

        // ── Derived sizes ─────────────────────────────────────────
        const qd: usize = @as(usize, self.n_head) * self.head_dim;
        const kvd: usize = @as(usize, self.n_head_kv) * self.head_dim;
        const conv_ch: usize = self.convChannels();
        const proj_size: usize = @as(usize, self.ssm_d_inner) + conv_ch + self.ssm_dt_rank;
        const nl: usize = self.n_layers;

        // ── Working buffers ───────────────────────────────────────
        self.hidden = try allocator.alloc(f32, self.n_embd);
        errdefer allocator.free(self.hidden);
        self.hidden2 = try allocator.alloc(f32, self.n_embd);
        errdefer allocator.free(self.hidden2);
        self.q_buf = try allocator.alloc(f32, qd);
        errdefer allocator.free(self.q_buf);
        self.k_buf = try allocator.alloc(f32, kvd);
        errdefer allocator.free(self.k_buf);
        self.v_buf = try allocator.alloc(f32, kvd);
        errdefer allocator.free(self.v_buf);
        self.attn_out = try allocator.alloc(f32, @max(qd, self.ssm_d_inner));
        errdefer allocator.free(self.attn_out);
        self.scores_buf = try allocator.alloc(f32, self.max_seq_len);
        errdefer allocator.free(self.scores_buf);
        self.ff_buf1 = try allocator.alloc(f32, self.n_ff);
        errdefer allocator.free(self.ff_buf1);
        self.ssm_proj_buf = try allocator.alloc(f32, proj_size);
        errdefer allocator.free(self.ssm_proj_buf);
        self.ssm_conv_out = try allocator.alloc(f32, conv_ch);
        errdefer allocator.free(self.ssm_conv_out);
        self.ssm_y_buf = try allocator.alloc(f32, self.ssm_d_inner);
        errdefer allocator.free(self.ssm_y_buf);
        self.logits_buf = try allocator.alloc(f32, self.vocab_size);
        errdefer allocator.free(self.logits_buf);

        // ── Per-layer state ───────────────────────────────────────
        const mamba_head_dim: usize = @as(usize, self.ssm_d_inner) / self.ssm_dt_rank;
        const state_per_layer: usize = @as(usize, self.ssm_dt_rank) * mamba_head_dim * self.ssm_d_state;
        const conv_state_len: usize = (@as(usize, self.ssm_d_conv) - 1) * conv_ch;

        // KV cache: use TieredKvCache if provided, otherwise flat PagedKvCache.
        if (tiered_cache) |tc| {
            var ta = TieredBlockAllocator.init(tc, allocator);
            self.seq_table = try ta.allocateSeqTable(nl);
            errdefer ta.freeSeqTable(&self.seq_table);
            try ta.appendBlock(&self.seq_table);
            self.tiered_cache = tc;
            self.tiered_block_allocator = ta;
        } else {
            // Paged KV cache: small fixed-size blocks allocated on demand.
            // Memory scales with actual sequence length, not max context.
            const paged_block_size: u16 = 256;
            const blocks_per_layer = (self.max_seq_len + paged_block_size - 1) / paged_block_size;
            const num_blocks = nl * blocks_per_layer;
            const block_size = paged_block_size;
            self.paged_cache = try PagedKvCache.init(allocator, nl, kvd, num_blocks, block_size);
            errdefer self.paged_cache.deinit();
            // BlockAllocator stores a pointer — must point to self.paged_cache (not a local copy).
            self.block_allocator = BlockAllocator.init(&self.paged_cache, allocator);
            self.seq_table = try self.block_allocator.allocateSeqTable(nl);
            errdefer self.block_allocator.freeSeqTable(&self.seq_table);
            try self.block_allocator.appendBlock(&self.seq_table);
        }

        self.conv_states = try allocator.alloc([]f32, nl);
        errdefer allocator.free(self.conv_states);
        self.ssm_states = try allocator.alloc([]f32, nl);
        errdefer allocator.free(self.ssm_states);

        var layer_init_count: usize = 0;
        errdefer {
            for (0..layer_init_count) |i| {
                if (self.conv_states[i].len > 0) allocator.free(self.conv_states[i]);
                if (self.ssm_states[i].len > 0) allocator.free(self.ssm_states[i]);
            }
        }

        for (0..nl) |i| {
            switch (self.layer_types[i]) {
                .ssm => {
                    self.conv_states[i] = try allocator.alloc(f32, conv_state_len);
                    @memset(self.conv_states[i], 0);
                    self.ssm_states[i] = try allocator.alloc(f32, state_per_layer);
                    @memset(self.ssm_states[i], 0);
                },
                .attention, .ffn_only => {
                    self.conv_states[i] = &.{};
                    self.ssm_states[i] = &.{};
                },
            }
            layer_init_count = i + 1;
        }

        // ── Prefill buffers (page_allocator for GPU zero-copy) ─────
        {
            const pa = std.heap.page_allocator;
            const cs = self.chunk_size;
            self.pf_hidden = try pa.alloc(f32, cs * self.n_embd);
            errdefer pa.free(self.pf_hidden);
            self.pf_hidden2 = try pa.alloc(f32, cs * self.n_embd);
            errdefer pa.free(self.pf_hidden2);
            self.pf_q = try pa.alloc(f32, cs * qd);
            errdefer pa.free(self.pf_q);
            self.pf_k = try pa.alloc(f32, cs * kvd);
            errdefer pa.free(self.pf_k);
            self.pf_v = try pa.alloc(f32, cs * kvd);
            errdefer pa.free(self.pf_v);
            self.pf_attn_out = try pa.alloc(f32, cs * qd);
            errdefer pa.free(self.pf_attn_out);
            self.pf_ff_buf1 = try pa.alloc(f32, cs * self.n_ff);
            errdefer pa.free(self.pf_ff_buf1);
            self.pf_positions = try pa.alloc(u32, cs);
            errdefer pa.free(self.pf_positions);
        }

        self.warmNormCache();
        return self;
    }

    /// Release all heap allocations owned by this model.
    pub fn deinit(self: *NemotronHModel) void {
        self.be.sync();
        for (self.norm_cache[0..self.norm_cache_len]) |entry| self.allocator.free(entry.data);
        const nl: usize = self.n_layers;
        for (0..nl) |i| {
            if (self.conv_states[i].len > 0) self.allocator.free(self.conv_states[i]);
            if (self.ssm_states[i].len > 0) self.allocator.free(self.ssm_states[i]);
        }
        self.allocator.free(self.conv_states);
        self.allocator.free(self.ssm_states);

        if (self.tiered_block_allocator) |*ta| {
            ta.freeSeqTable(&self.seq_table);
        } else {
            self.block_allocator.freeSeqTable(&self.seq_table);
            self.paged_cache.deinit();
        }

        const bufs = .{
            &self.hidden,       &self.hidden2,   &self.q_buf,
            &self.k_buf,        &self.v_buf,     &self.attn_out,
            &self.scores_buf,   &self.ff_buf1,   &self.ssm_proj_buf,
            &self.ssm_conv_out, &self.ssm_y_buf, &self.logits_buf,
        };
        inline for (bufs) |buf| self.allocator.free(buf.*);

        // Prefill buffers (page_allocator — must match init allocation).
        {
            const pa = std.heap.page_allocator;
            const pf_bufs = .{
                &self.pf_hidden, &self.pf_hidden2, &self.pf_q,
                &self.pf_k,      &self.pf_v,       &self.pf_attn_out,
                &self.pf_ff_buf1,
            };
            inline for (pf_bufs) |buf| if (buf.len > 0) pa.free(buf.*);
            if (self.pf_positions.len > 0) pa.free(self.pf_positions);
        }
    }

    /// Wrap this model in the generic `Model` interface.
    pub fn model(self: *NemotronHModel) Model {
        return Model.from(NemotronHModel, self);
    }

    // ── Public interface ──────────────────────────────────────────

    /// Run one decode step.  Returns the argmax next-token id.
    /// Errors: `error.MissingTensor` if a required weight is absent,
    ///         `error.KVCacheFull` if max_seq_len is reached,
    ///         `error.Cancelled` if `cancel()` was called concurrently.
    pub fn forward(self: *NemotronHModel, token_id: u32) !u32 {
        if (self.kv_seq_len >= self.max_seq_len) return error.KVCacheFull;

        try model_mod.ensureKvBlock(self);

        // Embedding lookup — zero-copy read from mmap.
        const emb_t = self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        self.be.embLookup(
            .{ .data = emb_t.data_ptr, .dtype = emb_t.dtype },
            token_id,
            self.hidden.ptr,
            self.n_embd,
        );

        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.monotonic)) return error.Cancelled;
            const l: u32 = @intCast(li);
            if (l >= self.layer_skip_start and l < self.layer_skip_end) continue;
            self.fmt.prefetchLayer(@intCast(li + 1));

            switch (self.layer_types[li]) {
                .ssm => try self.ssmLayer(l),
                .attention => try self.attentionLayer(l),
                .ffn_only => try self.ffnLayer(l),
            }
        }

        // Final norm → LM head (tied) → argmax.
        const nw = self.fmt.getTensor("output_norm.weight") orelse return error.MissingTensor;
        const ow = self.fmt.getTensor("output.weight") orelse
            self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        self.kv_seq_len += 1;

        // Flush deferred residual (if last layer left it pending), then final norm.
        if (self.pending_residual) {
            self.be.addRmsNorm(self.hidden.ptr, self.hidden2.ptr, self.normAsF32(nw, self.n_embd), self.hidden.ptr, self.n_embd, self.rms_eps);
            self.pending_residual = false;
        } else {
            self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, self.n_embd), self.hidden.ptr, self.n_embd, self.rms_eps);
        }
        self.be.gemv(self.hidden.ptr, .{ .data = ow.data_ptr, .dtype = ow.dtype }, self.logits_buf.ptr, self.vocab_size, self.n_embd);
        self.be.sync(); // GPU wrote logits — sync before CPU argmax
        return math_ops.argmax(self.logits_buf);
    }

    /// Batched prefill with chunked processing.
    ///
    /// Attention and FFN-only layers use batched GEMM (one dispatch per chunk).
    /// SSM layers require sequential state updates, so within each chunk they
    /// fall back to per-token ssmLayer.
    ///
    /// Falls back to naive per-token forward when prefill buffers are absent
    /// or the prompt is a single token.
    pub fn prefill(self: *NemotronHModel, token_ids: []const u32) !u32 {
        if (token_ids.len == 0) return error.MissingTensor;
        if (token_ids.len > self.max_seq_len) return error.KVCacheFull;

        const cs: usize = if (self.pf_hidden.len > 0) self.chunk_size else 1;
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

        // Final: rmsNorm + logits on the LAST token only.
        // In batched mode, residuals are never deferred — pf_hidden has the
        // complete residual stream, so plain rmsNorm suffices.
        const last_in_chunk = (token_ids.len - 1) % cs;
        const e: usize = self.n_embd;
        @memcpy(self.hidden, self.pf_hidden[last_in_chunk * e ..][0..e]);

        const nw = self.fmt.getTensor("output_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden.ptr, e, self.rms_eps);

        const ow = self.fmt.getTensor("output.weight") orelse
            self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        self.be.gemv(self.hidden.ptr, .{ .data = ow.data_ptr, .dtype = ow.dtype }, self.logits_buf.ptr, self.vocab_size, e);

        self.be.sync();
        self.kv_seq_len = token_ids.len;
        return math_ops.argmax(self.logits_buf);
    }

    /// Reset all SSM states, KV cache, and the cancellation flag for a new conversation.
    pub fn resetCache(self: *NemotronHModel) void {
        for (0..self.n_layers) |i| {
            if (self.conv_states[i].len > 0) @memset(self.conv_states[i], 0);
            if (self.ssm_states[i].len > 0) @memset(self.ssm_states[i], 0);
        }
        model_mod.resetKvCache(self);
    }

    /// Signal an in-progress `forward` call to abort.  Thread-safe.
    pub fn cancel(self: *NemotronHModel) void {
        model_mod.signalCancel(&self.cancelled);
    }

    /// Return physical block IDs from layer 0 of the current sequence table.
    /// All layers share the same block IDs, so layer 0 is sufficient.
    pub fn getBlockTable(self: *NemotronHModel) []const u32 {
        return self.seq_table.block_table[0];
    }

    // ── Layer implementations ─────────────────────────────────────

    /// Helper: get KV cache byte slices for a layer from the first paged/tiered block.
    fn getLayerKvView(self: *NemotronHModel, layer: usize) struct { keys: []u8, values: []u8 } {
        const num_blocks = self.seq_table.block_table[layer].len;
        if (num_blocks == 0) return .{ .keys = &[_]u8{}, .values = &[_]u8{} };

        const block_id = self.seq_table.block_table[layer][0];
        if (self.tiered_cache) |tc| {
            return .{
                .keys = std.mem.sliceAsBytes(tc.blocks[block_id].base.keys),
                .values = std.mem.sliceAsBytes(tc.blocks[block_id].base.values),
            };
        }
        const keys_f32 = self.paged_cache.blocks[block_id].keys;
        const values_f32 = self.paged_cache.blocks[block_id].values;
        return .{
            .keys = std.mem.sliceAsBytes(keys_f32),
            .values = std.mem.sliceAsBytes(values_f32),
        };
    }

    const PagedKvView = kvcache.PagedKvView;

    fn getPagedKvView(self: *NemotronHModel, layer: usize) PagedKvView {
        return PagedKvView.initView(
            self.seq_table.block_table[layer],
            self.paged_cache.blocks,
            self.paged_cache.block_size,
            self.paged_cache.kv_dim,
            self.kv_seq_len,
        );
    }

    fn isMultiBlock(self: *NemotronHModel, layer: usize) bool {
        return self.paged_cache.block_size > 0 and self.seq_table.block_table[layer].len > 1;
    }

    /// Mamba-2 SSM layer: pre-norm → input projection → causal conv1d →
    /// selective state space recurrence → group norm → SiLU gate →
    /// output projection → residual add.
    fn ssmLayer(self: *NemotronHModel, li: u32) !void {
        const e: usize = self.n_embd;
        const d_inner: usize = self.ssm_d_inner;
        const num_heads: usize = self.ssm_dt_rank;
        const mamba_head_dim: usize = d_inner / num_heads;
        const d_state: usize = self.ssm_d_state;
        const n_group: usize = self.ssm_n_group;
        const heads_per_group: usize = num_heads / n_group;
        const group_state: usize = d_state; // B/C size per group
        const conv_ch: usize = self.convChannels();
        const d_conv: usize = self.ssm_d_conv; // 4

        if (num_heads % n_group != 0) @panic("nemotron_h: ssm_dt_rank must be divisible by ssm_n_group");

        // 1. Pre-norm (fused with previous layer's deferred residual when pending_residual=true).
        const nw = self.fmt.layerTensor(li, "attn_norm.weight") orelse return error.MissingTensor;

        if (self.pending_residual) {
            self.be.addRmsNorm(self.hidden.ptr, self.hidden2.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
            self.pending_residual = false;
        } else {
            self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        }

        // 2. Input projection: [z(d_inner) | conv_in(conv_ch) | dt(num_heads)]
        const iw = self.fmt.layerTensor(li, "ssm_in.weight") orelse return error.MissingTensor;
        const proj_size: usize = d_inner + conv_ch + num_heads;
        self.be.gemv(
            self.hidden2.ptr,
            .{ .data = iw.data_ptr, .dtype = iw.dtype },
            self.ssm_proj_buf.ptr,
            proj_size,
            e,
        );

        // Split projection.
        self.be.sync(); // GPU gemv wrote ssm_proj_buf — flush before CPU reads
        const z_ptr = self.ssm_proj_buf.ptr; // [d_inner] gate
        const conv_in_ptr = self.ssm_proj_buf.ptr + d_inner; // [conv_ch] conv input
        const dt_raw_ptr = self.ssm_proj_buf.ptr + d_inner + conv_ch; // [num_heads] dt

        // 3. Causal conv1d (compute + update ring buffer via shared SSM op).
        const cs = self.conv_states[li];
        const conv_w_t = self.fmt.layerTensor(li, "ssm_conv1d.weight") orelse return error.MissingTensor;
        const conv_b_t = self.fmt.layerTensor(li, "ssm_conv1d.bias") orelse return error.MissingTensor;
        const conv_w = requireF32Ptr(conv_w_t, "ssm_conv1d.weight");
        const conv_b = requireF32Ptr(conv_b_t, "ssm_conv1d.bias");
        ssm_ops.causalConv1dSilu(self.ssm_conv_out.ptr, cs.ptr, conv_in_ptr, conv_w, conv_b, conv_ch, d_conv);

        // 4. Split conv output: x[0:d_inner] | B[d_inner:d_inner+n_group*d_state] | C[...].
        const x_ptr = self.ssm_conv_out.ptr; // [d_inner] = [96*80]
        const B_ptr = self.ssm_conv_out.ptr + d_inner; // [n_group * d_state] = [1024]
        const C_ptr = self.ssm_conv_out.ptr + d_inner + n_group * group_state; // [1024]

        // 5. Load per-head A and D scalars.
        const ssm_a_t = self.fmt.layerTensor(li, "ssm_a") orelse return error.MissingTensor;
        const ssm_d_t = self.fmt.layerTensor(li, "ssm_d") orelse return error.MissingTensor;
        const dt_bias_t = self.fmt.layerTensor(li, "ssm_dt.bias") orelse return error.MissingTensor;
        const ssm_a = requireF32Ptr(ssm_a_t, "ssm_a");
        const ssm_d = requireF32Ptr(ssm_d_t, "ssm_d");
        const dt_bias = requireF32Ptr(dt_bias_t, "ssm_dt.bias");

        // 6. Mamba-2 autoregressive recurrence, per head.
        const state = self.ssm_states[li]; // [num_heads * mamba_head_dim * d_state]
        const y_ptr = self.ssm_y_buf.ptr; // [d_inner]

        ssm_ops.mamba2Recurrence(y_ptr, state, x_ptr, B_ptr, C_ptr, dt_raw_ptr, dt_bias, ssm_a, ssm_d, num_heads, mamba_head_dim, d_state, heads_per_group);

        // 7. Group RMS norm on y, then SiLU gate.
        const norm_w_t = self.fmt.layerTensor(li, "ssm_norm.weight") orelse return error.MissingTensor;
        const norm_w = self.normAsF32(norm_w_t, d_inner);

        ssm_ops.groupRmsNormSiluGate(y_ptr, z_ptr, norm_w, d_inner, n_group, self.rms_eps);

        // 8. Output projection.
        const out_w = self.fmt.layerTensor(li, "ssm_out.weight") orelse return error.MissingTensor;
        self.be.gemv(
            y_ptr,
            .{ .data = out_w.data_ptr, .dtype = out_w.dtype },
            self.hidden2.ptr,
            e,
            d_inner,
        );

        // 9. Defer residual add: will fuse with next layer's pre-norm via addRmsNorm.
        self.pending_residual = true;
    }

    /// GQA attention layer: pre-norm → Q/K/V projections → partial RoPE →
    /// KV cache append → scaled dot-product attention → output projection → residual add.
    fn attentionLayer(self: *NemotronHModel, li: u32) !void {
        const e: usize = self.n_embd;
        const nh: usize = self.n_head;
        const nkv: usize = self.n_head_kv;
        const hd: usize = self.head_dim;
        const qd: usize = nh * hd;
        const kvd: usize = nkv * hd;

        // 1. Pre-norm (fused with previous layer's deferred residual when pending_residual=true).
        const nw = self.fmt.layerTensor(li, "attn_norm.weight") orelse return error.MissingTensor;
        if (self.pending_residual) {
            self.be.addRmsNorm(self.hidden.ptr, self.hidden2.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
            self.pending_residual = false;
        } else {
            self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        }

        // 2. Q/K/V projections (no bias).
        const qw = self.fmt.layerTensor(li, "attn_q.weight") orelse return error.MissingTensor;
        const kw = self.fmt.layerTensor(li, "attn_k.weight") orelse return error.MissingTensor;
        const vw = self.fmt.layerTensor(li, "attn_v.weight") orelse return error.MissingTensor;
        self.be.beginBatch();
        self.be.gemv(self.hidden2.ptr, .{ .data = qw.data_ptr, .dtype = qw.dtype }, self.q_buf.ptr, qd, e);
        self.be.gemv(self.hidden2.ptr, .{ .data = kw.data_ptr, .dtype = kw.dtype }, self.k_buf.ptr, kvd, e);
        self.be.gemv(self.hidden2.ptr, .{ .data = vw.data_ptr, .dtype = vw.dtype }, self.v_buf.ptr, kvd, e);
        self.be.endBatch();

        // 3. Partial RoPE: rotate only the first rope_dim dimensions of each head.
        self.be.beginBatch();
        self.be.rope(self.q_buf.ptr, self.kv_seq_len, nh, hd, self.rope_dim, self.rope_theta);
        self.be.rope(self.k_buf.ptr, self.kv_seq_len, nkv, hd, self.rope_dim, self.rope_theta);
        self.be.endBatch();

        // 4/5. KV cache append + scaled dot-product attention.
        // (backend handles sync and KV append internally)
        const kv_view = self.getLayerKvView(li);
        if (self.isMultiBlock(li)) {
            self.be.sdpaPaged(
                self.q_buf.ptr,
                self.getPagedKvView(li),
                self.k_buf.ptr,
                self.v_buf.ptr,
                self.attn_out.ptr,
                nh,
                nkv,
                hd,
                1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
                .f32,
                .f32,
            );
        } else {
            attn_ops.scaledDotProductAttention(
                self.q_buf.ptr,
                kv_view.keys,
                kv_view.values,
                self.k_buf,
                self.v_buf,
                self.attn_out.ptr,
                self.scores_buf.ptr,
                nh,
                nkv,
                hd,
                self.kv_seq_len,
                1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
                self.be,
                null,
                0,
                .f32, // PagedKvCache uses f32 blocks
                .f32,
            );
        }

        // 6. Output projection.
        const ow = self.fmt.layerTensor(li, "attn_output.weight") orelse return error.MissingTensor;
        self.be.gemv(
            self.attn_out.ptr,
            .{ .data = ow.data_ptr, .dtype = ow.dtype },
            self.hidden2.ptr,
            e,
            qd,
        );

        // 7. Defer residual add: will fuse with next layer's pre-norm via addRmsNorm.
        self.pending_residual = true;
    }

    /// FFN-only layer: pre-norm → squared-ReLU MLP (up → relu² → down) → residual add.
    /// No gate projection; activation is relu(x)² not SwiGLU.
    fn ffnLayer(self: *NemotronHModel, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.n_ff;

        // 1. Pre-norm (fused with previous layer's deferred residual when pending_residual=true).
        const nw = self.fmt.layerTensor(li, "attn_norm.weight") orelse return error.MissingTensor;
        if (self.pending_residual) {
            self.be.addRmsNorm(self.hidden.ptr, self.hidden2.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
            self.pending_residual = false;
        } else {
            self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        }

        // 2. Up projection → squared ReLU.
        const uw = self.fmt.layerTensor(li, "ffn_up.weight") orelse return error.MissingTensor;
        self.be.gemv(self.hidden2.ptr, .{ .data = uw.data_ptr, .dtype = uw.dtype }, self.ff_buf1.ptr, ff, e);
        self.be.sync(); // GPU gemv wrote ff_buf1 — flush before CPU squared-ReLU
        math_ops.applyReluSquared(self.ff_buf1[0..ff]);

        // 3. Down projection.
        const dw = self.fmt.layerTensor(li, "ffn_down.weight") orelse return error.MissingTensor;
        self.be.gemv(self.ff_buf1.ptr, .{ .data = dw.data_ptr, .dtype = dw.dtype }, self.hidden2.ptr, e, ff);

        // 4. Defer residual add: will fuse with next layer's pre-norm via addRmsNorm.
        self.pending_residual = true;
    }

    // ── Batched prefill ──────────────────────────────────────────

    /// Batched GEMM: n_tok rows × weight matrix.
    fn doGemm(self: *NemotronHModel, x: [*]const f32, t: TensorInfo, y: [*]f32, n_tok: usize, n_out: usize, n_in: usize) void {
        self.be.gemm(x, .{ .data = t.data_ptr, .dtype = t.dtype }, y, n_tok, n_out, n_in);
    }

    /// Process one chunk of tokens through all layers using batched GEMM
    /// for attention and FFN-only layers, per-token fallback for SSM layers.
    fn prefillChunk(self: *NemotronHModel, token_ids: []const u32, base_pos: u32) !void {
        const n_tok = token_ids.len;
        const e: usize = self.n_embd;

        // Ensure KV blocks allocated for all new positions.
        for (0..n_tok) |t| {
            self.kv_seq_len = base_pos + t;
            try model_mod.ensureKvBlock(self);
        }

        // Embedding lookup for all tokens into batched buffer.
        const emb_t = self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        for (token_ids, 0..) |tid, t| {
            self.be.embLookup(
                .{ .data = emb_t.data_ptr, .dtype = emb_t.dtype },
                tid,
                self.hidden.ptr,
                self.n_embd,
            );
            @memcpy(self.pf_hidden[t * e ..][0..e], self.hidden);
        }

        // Build position array.
        for (0..n_tok) |t| {
            self.pf_positions[t] = base_pos + @as(u32, @intCast(t));
        }

        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.monotonic)) return error.Cancelled;
            const l: u32 = @intCast(li);
            if (l >= self.layer_skip_start and l < self.layer_skip_end) continue;
            self.fmt.prefetchLayer(@intCast(li + 1));

            switch (self.layer_types[li]) {
                .attention => {
                    try self.prefillAttention(l, n_tok);
                },
                .ssm => {
                    // SSM recurrence is inherently sequential — fall back to per-token.
                    self.be.sync();
                    for (0..n_tok) |t| {
                        @memcpy(self.hidden, self.pf_hidden[t * e ..][0..e]);
                        self.pending_residual = false;
                        try self.ssmLayer(l);
                        // ssmLayer defers residual; flush it now so pf_hidden stays complete.
                        self.be.sync();
                        if (self.pending_residual) {
                            for (0..e) |j| self.hidden[j] += self.hidden2[j];
                            self.pending_residual = false;
                        }
                        @memcpy(self.pf_hidden[t * e ..][0..e], self.hidden);
                    }
                },
                .ffn_only => {
                    try self.prefillFeedForward(l, n_tok);
                },
            }
        }

        self.kv_seq_len = base_pos + n_tok;
    }

    /// Batched attention for one layer: norm → QKV GEMM → partial RoPE →
    /// sdpaPrefill → output GEMM → residual add.
    fn prefillAttention(self: *NemotronHModel, li: u32, n_tok: usize) !void {
        const e: usize = self.n_embd;
        const nh: usize = self.n_head;
        const nkv: usize = self.n_head_kv;
        const hd: usize = self.head_dim;
        const qd: usize = nh * hd;
        const kvd: usize = nkv * hd;

        // Pre-attention norm (batched).
        const nw = self.fmt.layerTensor(li, "attn_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNormBatched(self.pf_hidden.ptr, self.normAsF32(nw, e), self.pf_hidden2.ptr, n_tok, e, self.rms_eps);

        // Q/K/V projections (batched GEMM, no bias).
        const qw = self.fmt.layerTensor(li, "attn_q.weight") orelse return error.MissingTensor;
        const kw = self.fmt.layerTensor(li, "attn_k.weight") orelse return error.MissingTensor;
        const vw = self.fmt.layerTensor(li, "attn_v.weight") orelse return error.MissingTensor;
        self.doGemm(self.pf_hidden2.ptr, qw, self.pf_q.ptr, n_tok, qd, e);
        self.doGemm(self.pf_hidden2.ptr, kw, self.pf_k.ptr, n_tok, kvd, e);
        self.doGemm(self.pf_hidden2.ptr, vw, self.pf_v.ptr, n_tok, kvd, e);

        // Partial RoPE (batched).
        self.be.ropeBatched(self.pf_q.ptr, self.pf_positions.ptr, n_tok, nh, hd, self.rope_dim, self.rope_theta);
        self.be.ropeBatched(self.pf_k.ptr, self.pf_positions.ptr, n_tok, nkv, hd, self.rope_dim, self.rope_theta);

        // Fused causal attention (sdpaPrefill writes KV into cache and computes output).
        const kv_view = self.getLayerKvView(li);
        const prev_len: usize = self.pf_positions[0];
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        self.be.sdpaPrefill(self.pf_q.ptr, self.pf_k.ptr, self.pf_v.ptr, kv_view.keys, kv_view.values, self.pf_attn_out.ptr, nh, nkv, hd, prev_len, n_tok, scale, .f32, .f32);

        // Output projection (batched GEMM).
        const ow = self.fmt.layerTensor(li, "attn_output.weight") orelse return error.MissingTensor;
        self.doGemm(self.pf_attn_out.ptr, ow, self.pf_hidden2.ptr, n_tok, e, qd);

        // Residual add (no deferred residuals in batched mode).
        self.be.add(self.pf_hidden.ptr, self.pf_hidden2.ptr, self.pf_hidden.ptr, n_tok * e);
    }

    /// Batched feed-forward for one layer: norm → up GEMM → squared ReLU → down GEMM → residual.
    fn prefillFeedForward(self: *NemotronHModel, li: u32, n_tok: usize) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.n_ff;

        // Pre-FFN norm (batched). Nemotron-H uses attn_norm for all layer types.
        const nw = self.fmt.layerTensor(li, "attn_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNormBatched(self.pf_hidden.ptr, self.normAsF32(nw, e), self.pf_hidden2.ptr, n_tok, e, self.rms_eps);

        // Up projection (batched GEMM).
        const uw = self.fmt.layerTensor(li, "ffn_up.weight") orelse return error.MissingTensor;
        self.doGemm(self.pf_hidden2.ptr, uw, self.pf_ff_buf1.ptr, n_tok, ff, e);

        // Squared ReLU activation (element-wise on entire batch buffer).
        self.be.sync();
        math_ops.applyReluSquared(self.pf_ff_buf1[0 .. n_tok * ff]);

        // Down projection (batched GEMM).
        const dw = self.fmt.layerTensor(li, "ffn_down.weight") orelse return error.MissingTensor;
        self.doGemm(self.pf_ff_buf1.ptr, dw, self.pf_hidden2.ptr, n_tok, e, ff);

        // Residual add (no deferred residuals in batched mode).
        self.be.add(self.pf_hidden.ptr, self.pf_hidden2.ptr, self.pf_hidden.ptr, n_tok * e);
    }

    // ── Helpers ───────────────────────────────────────────────────

    /// Number of channels entering the conv1d:
    /// ssm_d_inner + 2 * ssm_n_group * ssm_d_state = 7680 + 2*8*128 = 9728.
    fn convChannels(self: *const NemotronHModel) usize {
        return @as(usize, self.ssm_d_inner) +
            2 * @as(usize, self.ssm_n_group) * @as(usize, self.ssm_d_state);
    }

    /// Return f32 view of a norm tensor, converting+caching non-f32 weights once.
    fn normAsF32(self: *NemotronHModel, t: TensorInfo, n: usize) [*]const f32 {
        if (t.dtype == .f32) return @ptrCast(@alignCast(t.data_ptr));

        const key = @intFromPtr(t.data_ptr);
        for (self.norm_cache[0..self.norm_cache_len]) |entry| {
            if (entry.key == key) return entry.data.ptr;
        }

        if (self.norm_cache_len >= max_norm_entries)
            @panic("normAsF32: norm cache overflow — increase max_norm_entries");
        const buf = self.allocator.alloc(f32, n) catch @panic("normAsF32: out of memory converting norm weights");
        quant_ops.dequantToF32(buf, t.data_ptr, t.dtype, n);
        self.norm_cache[self.norm_cache_len] = .{ .key = key, .data = buf };
        self.norm_cache_len += 1;
        return buf.ptr;
    }

    fn warmNormCache(self: *NemotronHModel) void {
        const e: usize = self.n_embd;
        if (self.fmt.getTensor("output_norm.weight")) |t| _ = self.normAsF32(t, e);
        for (0..self.n_layers) |li| {
            const l: u32 = @intCast(li);
            if (self.fmt.layerTensor(l, "attn_norm.weight")) |t| _ = self.normAsF32(t, e);
            if (self.layer_types[li] == .ssm) {
                if (self.fmt.layerTensor(l, "ssm_norm.weight")) |t| {
                    _ = self.normAsF32(t, self.ssm_d_inner);
                }
            }
        }
    }
};

/// SSM parameter tensors must be f32; quantized SSM weights are not supported.
fn requireF32Ptr(t: TensorInfo, name: []const u8) [*]const f32 {
    if (t.dtype != .f32) {
        std.debug.panic("nemotron_h: {s} must be f32 (got {s})", .{ name, @tagName(t.dtype) });
    }
    return @ptrCast(@alignCast(t.data_ptr));
}

// ── Tests ─────────────────────────────────────────────────────────

test "NemotronHModel convChannels default" {
    // Verify derived sizes with default config values.
    // conv_ch = 7680 + 2*8*128 = 7680 + 2048 = 9728
    // proj_size = 7680 + 9728 + 96 = 17504
    const ch: usize = 7680 + 2 * 8 * 128;
    try std.testing.expectEqual(@as(usize, 9728), ch);
    const proj: usize = 7680 + ch + 96;
    try std.testing.expectEqual(@as(usize, 17504), proj);
}

test "NemotronH convChannels method" {
    // Test the actual method on a struct instance
    var m: NemotronHModel = undefined;
    m.ssm_d_inner = 7680;
    m.ssm_n_group = 8;
    m.ssm_d_state = 128;
    try std.testing.expectEqual(@as(usize, 9728), m.convChannels());

    // Smaller config
    m.ssm_d_inner = 1024;
    m.ssm_n_group = 4;
    m.ssm_d_state = 64;
    // 1024 + 2*4*64 = 1024 + 512 = 1536
    try std.testing.expectEqual(@as(usize, 1536), m.convChannels());
}

test "NemotronH LayerType enum" {
    // Verify the three layer types are distinct
    try std.testing.expect(LayerType.ssm != LayerType.attention);
    try std.testing.expect(LayerType.ssm != LayerType.ffn_only);
    try std.testing.expect(LayerType.attention != LayerType.ffn_only);
}

test "NemotronH layer type pattern" {
    // Simulate the 8B variant's layer distribution
    var types: [42]LayerType = [_]LayerType{.ffn_only} ** 42;
    var n_ssm: usize = 0;
    var n_attn: usize = 0;
    var n_ffn: usize = 0;

    // Example pattern: layers 0,2,4,... are SSM; 1,9,17,25 are attention; rest FFN
    const ssm_layers = [_]usize{ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40 };
    const attn_layers = [_]usize{ 1, 9, 17, 25 };
    for (ssm_layers) |l| types[l] = .ssm;
    for (attn_layers) |l| types[l] = .attention;

    for (types) |t| switch (t) {
        .ssm => n_ssm += 1,
        .attention => n_attn += 1,
        .ffn_only => n_ffn += 1,
    };
    try std.testing.expectEqual(@as(usize, 21), n_ssm);
    try std.testing.expectEqual(@as(usize, 4), n_attn);
    try std.testing.expectEqual(@as(usize, 17), n_ffn);
}

test "NemotronH model vtable compiles" {
    try std.testing.expect(@hasDecl(NemotronHModel, "forward"));
    try std.testing.expect(@hasDecl(NemotronHModel, "prefill"));
    try std.testing.expect(@hasDecl(NemotronHModel, "resetCache"));
    try std.testing.expect(@hasDecl(NemotronHModel, "cancel"));
    try std.testing.expect(@hasDecl(NemotronHModel, "model"));
}

test "NemotronH default config values" {
    var m: NemotronHModel = undefined;
    m.n_layers = 42;
    m.n_embd = 3136;
    m.n_head = 40;
    m.n_head_kv = 8;
    m.head_dim = 128;
    m.n_ff = 12544;
    m.vocab_size = 131072;
    m.rope_theta = 10000.0;
    m.rope_dim = 78;
    m.rms_eps = 1e-5;
    m.ssm_d_conv = 4;
    m.ssm_d_state = 128;
    m.ssm_n_group = 8;
    m.ssm_dt_rank = 96;
    m.ssm_d_inner = 7680;

    // Verify GQA ratio
    try std.testing.expectEqual(@as(u32, 5), m.n_head / m.n_head_kv);
    // Verify mamba_head_dim
    const mamba_hd: u32 = m.ssm_d_inner / m.ssm_dt_rank;
    try std.testing.expectEqual(@as(u32, 80), mamba_hd);
    // Verify partial RoPE (< head_dim)
    try std.testing.expect(m.rope_dim < m.head_dim);
    try std.testing.expect(m.rope_dim % 2 == 0);
}

test "NemotronH layer_types default" {
    var m: NemotronHModel = undefined;
    m.layer_types = [_]LayerType{.ffn_only} ** max_layers;
    // All layers default to ffn_only
    for (0..max_layers) |i| {
        try std.testing.expectEqual(LayerType.ffn_only, m.layer_types[i]);
    }
}

test "NemotronH getBlockTable compiles" {
    try std.testing.expect(@hasDecl(NemotronHModel, "getBlockTable"));
}

// argmax is tested in src/ops/math.zig — no need to duplicate here.

test "fuzz: all nemotron_h functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // ── LayerType enum ──────────────────────────────────────
            const lt_idx = smith.valueWithHash(u8, 0) % 3;
            const lt: LayerType = @enumFromInt(lt_idx);
            switch (lt) {
                .ssm, .attention, .ffn_only => {},
            }

            // ── NemotronHModel.cancel (exercises atomic store) ──────
            var m: NemotronHModel = undefined;
            m.cancelled = std.atomic.Value(bool).init(false);
            m.cancel();
            try std.testing.expect(m.cancelled.load(.monotonic) == true);

            // ── NemotronHModel.convChannels (private but callable via struct) ──
            m.ssm_d_inner = smith.valueWithHash(u16, 1) | 1; // ensure non-zero
            m.ssm_n_group = @as(u32, smith.valueWithHash(u8, 2) % 16) + 1;
            m.ssm_d_state = @as(u32, smith.valueWithHash(u8, 3) % 64) + 1;
            const ch = m.convChannels();
            // conv_ch = d_inner + 2 * n_group * d_state
            const expected = @as(usize, m.ssm_d_inner) +
                2 * @as(usize, m.ssm_n_group) * @as(usize, m.ssm_d_state);
            try std.testing.expectEqual(expected, ch);

            // ── NemotronHModel default field values ─────────────────
            const n_head_val = smith.valueWithHash(u8, 4) | 1;
            const n_head_kv_val: u32 = @as(u32, smith.valueWithHash(u8, 5) % 8) + 1;
            m.n_head = n_head_val;
            m.n_head_kv = n_head_kv_val;
            m.head_dim = @as(u32, smith.valueWithHash(u8, 6) % 64) + 1;
            m.ssm_dt_rank = @as(u32, smith.valueWithHash(u8, 7) % 32) + 1;
            // Verify convChannels still consistent after field changes
            _ = m.convChannels();

            // ── Comptime verification: all pub functions exist ──────
            // These require Format/Backend/KV infrastructure to call at runtime.
            // Verify they compile and have the expected signatures.
            comptime {
                _ = &NemotronHModel.init;
                _ = &NemotronHModel.deinit;
                _ = &NemotronHModel.model;
                _ = &NemotronHModel.forward;
                _ = &NemotronHModel.prefill;
                _ = &NemotronHModel.resetCache;
                _ = &NemotronHModel.cancel;
                _ = &NemotronHModel.getBlockTable;
            }

            // ── layer_types array fuzz ──────────────────────────────
            m.layer_types = [_]LayerType{.ffn_only} ** max_layers;
            const layer_count = smith.valueWithHash(u8, 8) % max_layers;
            var ssm_count: usize = 0;
            var attn_count: usize = 0;
            var ffn_count: usize = 0;
            for (0..layer_count) |i| {
                const kind = smith.valueWithHash(u8, @intCast(i + 100)) % 3;
                m.layer_types[i] = @enumFromInt(kind);
                switch (m.layer_types[i]) {
                    .ssm => ssm_count += 1,
                    .attention => attn_count += 1,
                    .ffn_only => ffn_count += 1,
                }
            }
            try std.testing.expectEqual(layer_count, ssm_count + attn_count + ffn_count);
        }
    }.f, .{});
}
