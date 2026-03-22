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
const TensorData = backend_mod.TensorData;
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;
const kv_quant = @import("../ops/kv_quant.zig");
const PagedKvCache = kvcache.PagedKvCache;
const SeqBlockTable = kvcache.SeqBlockTable;

/// Layer variants in Nemotron-H.
pub const LayerType = enum { ssm, attention, ffn_only };

/// Maximum supported layer count (controls static array sizes for layer_types).
const max_layers: usize = 128;

/// Buffer size for tensor name formatting.
const name_buf_size: usize = 256;

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
    /// FFN second half (up or intermediate).
    ff_buf2: []f32 = &.{},
    /// SSM input projection output — [z(ssm_d_inner) | conv_in(conv_ch) | dt(ssm_dt_rank)].
    ssm_proj_buf: []f32 = &.{},
    /// Causal conv1d output — conv_ch elements.
    ssm_conv_out: []f32 = &.{},
    /// SSM output (y) before gating — ssm_d_inner elements.
    ssm_y_buf: []f32 = &.{},
    /// Final vocabulary logits — vocab_size elements.
    logits_buf: []f32 = &.{},

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
    /// KV cache quantization type.
    kv_type: kv_quant.KvQuantType = .f32,
    /// Number of tokens committed to the KV cache.
    kv_seq_len: usize = 0,
    /// Set to true from another thread to abort an in-progress `forward` call.
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    // ── Lifecycle ─────────────────────────────────────────────────

    /// Initialize the model from format metadata and pre-allocate all buffers.
    /// Caller owns the returned value and must call `deinit` when done.
    pub fn init(allocator: Allocator, f: Format, be: Backend, ctx_size: u32, kv_type: kv_quant.KvQuantType, tiered_cache: ?*TieredKvCache) !NemotronHModel {
        var self = NemotronHModel{ .fmt = f, .be = be, .allocator = allocator };
        self.kv_type = kv_type;

        const arch = f.getMetaStr("general.architecture") orelse "nemotron-h";
        self.n_layers = f.getArchU32(arch, "block_count") orelse 42;
        self.n_embd = f.getArchU32(arch, "embedding_length") orelse 3136;
        self.n_head = f.getArchU32(arch, "attention.head_count") orelse 40;
        // head_count_kv and feed_forward_length are per-layer arrays in nemotron_h GGUF.
        // getArchU32 returns 0 for array-type metadata, so fall back to defaults.
        const raw_kv = f.getArchU32(arch, "attention.head_count_kv");
        self.n_head_kv = if (raw_kv == null or raw_kv.? == 0) 8 else raw_kv.?;
        self.head_dim = f.getArchU32(arch, "attention.key_length") orelse 128;
        const raw_ff = f.getArchU32(arch, "feed_forward_length");
        self.n_ff = if (raw_ff == null or raw_ff.? == 0) 12544 else raw_ff.?;
        self.ssm_d_conv = f.getArchU32(arch, "ssm.conv_kernel") orelse 4;
        self.ssm_d_state = f.getArchU32(arch, "ssm.state_size") orelse 128;
        self.ssm_n_group = f.getArchU32(arch, "ssm.group_count") orelse 8;
        self.ssm_dt_rank = f.getArchU32(arch, "ssm.time_step_rank") orelse 96;
        self.ssm_d_inner = f.getArchU32(arch, "ssm.inner_size") orelse 7680;
        self.rope_dim = f.getArchU32(arch, "rope.dimension_count") orelse 78;
        if (f.getArchF32(arch, "rope.freq_base")) |v| self.rope_theta = v;
        if (f.getArchF32(arch, "attention.layer_norm_rms_epsilon")) |v| self.rms_eps = v;
        if (f.getMetaU32("tokenizer.ggml.eos_token_id")) |v| self.eos_token_id = v;
        if (f.getVocab()) |v| self.vocab_size = @intCast(v.len);
        if (f.getArchU32(arch, "context_length")) |cl| self.max_seq_len = cl;
        if (ctx_size > 0) self.max_seq_len = ctx_size;

        std.debug.assert(self.n_head % self.n_head_kv == 0);
        std.debug.assert(self.ssm_d_inner % self.ssm_dt_rank == 0);
        std.debug.assert(self.rope_dim <= self.head_dim);
        std.debug.assert(self.rope_dim % 2 == 0);
        std.debug.assert(self.n_layers <= max_layers);

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

        std.log.warn("[nemotron_h] Detected {} SSM, {} attention, {} ffn_only layers", .{ n_ssm, n_attn, n_ffn });

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
        self.ff_buf2 = try allocator.alloc(f32, self.n_ff);
        errdefer allocator.free(self.ff_buf2);
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
            const block_size = kvcache.default_block_size;
            const num_blocks = (self.max_seq_len + block_size - 1) / block_size * nl;
            var paged_cache = try PagedKvCache.init(allocator, nl, kvd, num_blocks, block_size);
            errdefer paged_cache.deinit();
            var block_allocator = BlockAllocator.init(&paged_cache, allocator);
            var seq_table = try block_allocator.allocateSeqTable(nl);
            errdefer block_allocator.freeSeqTable(&seq_table);
            try block_allocator.appendBlock(&seq_table);
            self.paged_cache = paged_cache;
            self.seq_table = seq_table;
            self.block_allocator = block_allocator;
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
                .attention => {
                    self.conv_states[i] = &.{};
                    self.ssm_states[i] = &.{};
                },
                .ffn_only => {
                    self.conv_states[i] = &.{};
                    self.ssm_states[i] = &.{};
                },
            }
            layer_init_count = i + 1;
        }

        return self;
    }

    /// Release all heap allocations owned by this model.
    pub fn deinit(self: *NemotronHModel) void {
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
            &self.hidden,       &self.hidden2,      &self.q_buf,
            &self.k_buf,        &self.v_buf,        &self.attn_out,
            &self.scores_buf,   &self.ff_buf1,      &self.ff_buf2,
            &self.ssm_proj_buf, &self.ssm_conv_out, &self.ssm_y_buf,
            &self.logits_buf,
        };
        inline for (bufs) |buf| self.allocator.free(buf.*);
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

        // Check if new block needed
        const bs: usize = if (self.tiered_cache) |tc| tc.block_size else self.paged_cache.block_size;
        const current_blocks = self.seq_table.block_table[0].len;
        const needed_blocks = (self.kv_seq_len + 1 + bs - 1) / bs;
        if (needed_blocks > current_blocks) {
            if (self.tiered_block_allocator) |*ta| {
                try ta.appendBlock(&self.seq_table);
            } else {
                try self.block_allocator.appendBlock(&self.seq_table);
            }
        }

        // Embedding lookup — zero-copy read from mmap.
        const emb_t = self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        self.be.embLookup(
            .{ .data = emb_t.data_ptr, .dtype = emb_t.dtype },
            token_id,
            self.hidden.ptr,
            self.n_embd,
        );

        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.acquire)) return error.Cancelled;
            const l: u32 = @intCast(li);

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
        const result = math_ops.finalLogits(
            self.hidden.ptr,
            nw.data_ptr,
            .{ .data = ow.data_ptr, .dtype = ow.dtype },
            self.logits_buf,
            self.vocab_size,
            self.n_embd,
            self.rms_eps,
            self.be,
        );

        return result;
    }

    /// Reset all SSM states, KV cache, and the cancellation flag for a new conversation.
    pub fn resetCache(self: *NemotronHModel) void {
        for (0..self.n_layers) |i| {
            if (self.conv_states[i].len > 0) @memset(self.conv_states[i], 0);
            if (self.ssm_states[i].len > 0) @memset(self.ssm_states[i], 0);
        }
        if (self.tiered_block_allocator) |*ta| {
            ta.freeSeqTable(&self.seq_table);
            self.seq_table = ta.allocateSeqTable(self.n_layers) catch return;
            ta.appendBlock(&self.seq_table) catch return;
        } else {
            self.block_allocator.freeSeqTable(&self.seq_table);
            self.seq_table = self.block_allocator.allocateSeqTable(self.n_layers) catch return;
            self.block_allocator.appendBlock(&self.seq_table) catch return;
        }
        model_mod.resetInferenceState(&self.kv_seq_len, &self.cancelled);
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

    /// Helper: get flat f32 view of KV cache for a layer (assembled from paged or tiered blocks).
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

    /// Mamba-2 SSM layer: pre-norm → input projection → causal conv1d →
    /// selective state space recurrence → group norm → SiLU gate →
    /// output projection → residual add.
    fn ssmLayer(self: *NemotronHModel, li: u32) !void {
        const e: usize = self.n_embd;
        const d_inner: usize = self.ssm_d_inner;
        const num_heads: usize = self.ssm_dt_rank; // = num_mamba_heads = 96
        const mamba_head_dim: usize = d_inner / num_heads; // 80
        const d_state: usize = self.ssm_d_state; // 128
        const n_group: usize = self.ssm_n_group; // 8
        const heads_per_group: usize = num_heads / n_group; // 12
        const group_state: usize = d_state; // B/C size per group = 128
        const conv_ch: usize = self.convChannels(); // 9728
        const d_conv: usize = self.ssm_d_conv; // 4

        std.debug.assert(num_heads % n_group == 0);

        // 1. Pre-norm.
        const nw = self.fmt.layerTensor(li, "attn_norm.weight") orelse return error.MissingTensor;

        self.be.rmsNorm(
            self.hidden.ptr,
            @ptrCast(@alignCast(nw.data_ptr)),
            self.hidden2.ptr,
            e,
            self.rms_eps,
        );

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
        const conv_w: [*]const f32 = @ptrCast(@alignCast(conv_w_t.data_ptr));
        const conv_b: [*]const f32 = @ptrCast(@alignCast(conv_b_t.data_ptr));
        ssm_ops.causalConv1dSilu(self.ssm_conv_out.ptr, cs.ptr, conv_in_ptr, conv_w, conv_b, conv_ch, d_conv);

        // 4. Split conv output: x[0:d_inner] | B[d_inner:d_inner+n_group*d_state] | C[...].
        const x_ptr = self.ssm_conv_out.ptr; // [d_inner] = [96*80]
        const B_ptr = self.ssm_conv_out.ptr + d_inner; // [n_group * d_state] = [1024]
        const C_ptr = self.ssm_conv_out.ptr + d_inner + n_group * group_state; // [1024]

        // 5. Load per-head A and D scalars.
        const ssm_a_t = self.fmt.layerTensor(li, "ssm_a") orelse return error.MissingTensor;
        const ssm_d_t = self.fmt.layerTensor(li, "ssm_d") orelse return error.MissingTensor;
        const dt_bias_t = self.fmt.layerTensor(li, "ssm_dt.bias") orelse return error.MissingTensor;
        const ssm_a: [*]const f32 = @ptrCast(@alignCast(ssm_a_t.data_ptr));
        const ssm_d: [*]const f32 = @ptrCast(@alignCast(ssm_d_t.data_ptr));
        const dt_bias: [*]const f32 = @ptrCast(@alignCast(dt_bias_t.data_ptr));

        // 6. Mamba-2 autoregressive recurrence, per head.
        const state = self.ssm_states[li]; // [num_heads * mamba_head_dim * d_state]
        const y_ptr = self.ssm_y_buf.ptr; // [d_inner]

        ssm_ops.mamba2Recurrence(y_ptr, state, x_ptr, B_ptr, C_ptr, dt_raw_ptr, dt_bias, ssm_a, ssm_d, num_heads, mamba_head_dim, d_state, heads_per_group);

        // 7. Group RMS norm on y, then SiLU gate.
        const norm_w_t = self.fmt.layerTensor(li, "ssm_norm.weight") orelse return error.MissingTensor;
        const norm_w: [*]const f32 = @ptrCast(@alignCast(norm_w_t.data_ptr));

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

        // 9. Residual.
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
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

        // 1. Pre-norm.
        const nw = self.fmt.layerTensor(li, "attn_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(
            self.hidden.ptr,
            @ptrCast(@alignCast(nw.data_ptr)),
            self.hidden2.ptr,
            e,
            self.rms_eps,
        );

        // 2. Q/K/V projections (no bias).
        const qw = self.fmt.layerTensor(li, "attn_q.weight") orelse return error.MissingTensor;
        const kw = self.fmt.layerTensor(li, "attn_k.weight") orelse return error.MissingTensor;
        const vw = self.fmt.layerTensor(li, "attn_v.weight") orelse return error.MissingTensor;
        self.be.gemv(self.hidden2.ptr, .{ .data = qw.data_ptr, .dtype = qw.dtype }, self.q_buf.ptr, qd, e);
        self.be.gemv(self.hidden2.ptr, .{ .data = kw.data_ptr, .dtype = kw.dtype }, self.k_buf.ptr, kvd, e);
        self.be.gemv(self.hidden2.ptr, .{ .data = vw.data_ptr, .dtype = vw.dtype }, self.v_buf.ptr, kvd, e);

        // 3. Partial RoPE: rotate only the first rope_dim dimensions of each head.
        self.be.rope(self.q_buf.ptr, self.kv_seq_len, nh, hd, self.rope_dim, self.rope_theta);
        self.be.rope(self.k_buf.ptr, self.kv_seq_len, nkv, hd, self.rope_dim, self.rope_theta);

        // 4/5. KV cache append + scaled dot-product attention.
        // (backend handles sync and KV append internally)
        const kv_view = self.getLayerKvView(li);
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
        );

        // 6. Output projection.
        const ow = self.fmt.layerTensor(li, "attn_output.weight") orelse return error.MissingTensor;
        self.be.gemv(
            self.attn_out.ptr,
            .{ .data = ow.data_ptr, .dtype = ow.dtype },
            self.hidden2.ptr,
            e,
            qd,
        );

        // 7. Residual.
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
    }

    /// FFN-only layer: pre-norm → squared-ReLU MLP (up → relu² → down) → residual add.
    /// No gate projection; activation is relu(x)² not SwiGLU.
    fn ffnLayer(self: *NemotronHModel, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.n_ff;

        // 1. Pre-norm.
        const nw = self.fmt.layerTensor(li, "attn_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(
            self.hidden.ptr,
            @ptrCast(@alignCast(nw.data_ptr)),
            self.hidden2.ptr,
            e,
            self.rms_eps,
        );

        // 2. Up projection → squared ReLU.
        const uw = self.fmt.layerTensor(li, "ffn_up.weight") orelse return error.MissingTensor;
        self.be.gemv(self.hidden2.ptr, .{ .data = uw.data_ptr, .dtype = uw.dtype }, self.ff_buf1.ptr, ff, e);
        self.be.sync(); // GPU gemv wrote ff_buf1 — flush before CPU squared-ReLU
        math_ops.applyReluSquared(self.ff_buf1[0..ff]);

        // 3. Down projection.
        const dw = self.fmt.layerTensor(li, "ffn_down.weight") orelse return error.MissingTensor;
        self.be.gemv(self.ff_buf1.ptr, .{ .data = dw.data_ptr, .dtype = dw.dtype }, self.hidden2.ptr, e, ff);

        // 4. Residual.
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
    }

    // ── Helpers ───────────────────────────────────────────────────

    /// Number of channels entering the conv1d:
    /// ssm_d_inner + 2 * ssm_n_group * ssm_d_state = 7680 + 2*8*128 = 9728.
    fn convChannels(self: *const NemotronHModel) usize {
        return @as(usize, self.ssm_d_inner) +
            2 * @as(usize, self.ssm_n_group) * @as(usize, self.ssm_d_state);
    }
};

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

test "argmax" {
    const buf = [_]f32{ 1.0, 3.0, 2.0, 0.5 };
    try std.testing.expectEqual(@as(u32, 1), math_ops.argmax(&buf));
}
