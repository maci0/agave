//! GPT-OSS 20B — Mixture-of-Experts decoder with sliding-window attention,
//! learned attention sinks, and clamped SwiGLU.
//!
//! Architecture highlights
//! -----------------------
//! * 24 transformer layers, n_embd=2880, 64 Q-heads / 8 KV-heads, head_dim=64
//! * Alternating attention: even layers = sliding window (128 tokens), odd = full
//! * Per-layer learned attention sinks: one scalar bias per Q-head
//! * MoE FFN: 32 experts, top-4 active, expert size = 2880 hidden units
//! * SwiGLU activation with hard clamp at ±7.0
//! * Post-attention norm before residual add; optional pre-FFN norm (ffn_norm) when present
//! * All attention projections carry a bias term
//! * Packed expert weights stored as mxfp4 in GGUF
const std = @import("std");
const math = std.math;
const backend_mod = @import("../backend/backend.zig");
const format_mod = @import("../format/format.zig");
const model_mod = @import("model.zig");
const math_ops = @import("../ops/math.zig");
const kvcache = @import("../kvcache/manager.zig");
const block_alloc_mod = @import("../kvcache/block_allocator.zig");
const BlockAllocator = block_alloc_mod.BlockAllocator;
const TieredBlockAllocator = block_alloc_mod.TieredBlockAllocator;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;

const Backend = backend_mod.Backend;
const TensorData = backend_mod.TensorData;
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const DType = format_mod.DType;
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;
const quant = @import("../ops/quant.zig");
const kv_quant = @import("../ops/kv_quant.zig");
const mlx_ops = @import("../ops/mlx.zig");
const PagedKvCache = kvcache.PagedKvCache;
const SeqBlockTable = kvcache.SeqBlockTable;

/// Maximum top-k experts for stack-allocated selection arrays.
const max_active_experts: usize = 8;
/// Maximum attention heads for stack-allocated sink bias array.
const max_sink_heads: usize = 64;
/// Maximum cached norm weight entries (≥ 3 norms/layer × 24 layers + 1 output norm).
const max_norm_entries: usize = 128;
/// Cached f32 norm weight entry, keyed by source data pointer.
const NormCacheEntry = struct { key: usize, data: []f32 };

/// Compute the byte stride between consecutive experts in a packed weight tensor.
/// Handles both GGUF layout [inner, outer, n_experts] and SafeTensors MLX
/// layout [n_experts, rows, words_per_row].
fn expertStride(t: TensorInfo) usize {
    if (t.dtype == .mlx_q) {
        // SafeTensors MLX: [n_experts, rows, words_per_row] U32
        // Per-expert: rows * words_per_row * sizeof(u32)
        std.debug.assert(t.n_dims >= 3);
        return @as(usize, @intCast(t.dims[0])) * @as(usize, @intCast(t.dims[1])) * @sizeOf(u32);
    }
    // GGUF: dims = [inner, outer, n_experts] — use shared helper
    return expertWeightStride(t);
}

/// Compute the byte stride between consecutive experts in a companion scale/bias tensor.
/// MLX companion tensors (scales, biases) have dims [n_experts, rows, groups_per_row],
/// stored as bf16. This computes the byte offset per expert slice.
fn expertScaleStride(scale_t: TensorInfo, weight_t: TensorInfo) usize {
    _ = weight_t;
    if (scale_t.n_dims >= 3) {
        const elems: usize = @as(usize, @intCast(scale_t.dims[0])) * @as(usize, @intCast(scale_t.dims[1]));
        return elems * @sizeOf(u16); // bf16 = 2 bytes per element
    }
    // Fallback: divide total size by number of experts
    const total = scale_t.numElements() * @sizeOf(u16);
    const n_experts = if (scale_t.n_dims >= 3) @as(usize, @intCast(scale_t.dims[2])) else 1;
    return if (n_experts > 0) total / n_experts else total;
}
/// Extra score slot reserved for the learned attention sink.
const attention_sink_slots: usize = 1;
/// Minimum valid MLX quantization bits.
const min_mlx_bits: u64 = 1;
/// Maximum valid MLX quantization bits.
const max_mlx_bits: u64 = 32;
/// Default MLX quantization bits when inference fails.
const default_mlx_bits: u32 = 4;

/// GPT-OSS 20B model state.
pub const GptOssModel = struct {
    // ── Configuration ────────────────────────────────────────────
    fmt: Format,
    be: Backend,
    allocator: Allocator,

    /// Number of transformer blocks.
    n_layers: u32 = 24,
    /// Hidden embedding dimension.
    n_embd: u32 = 2880,
    /// Number of query heads.
    n_head: u32 = 64,
    /// Number of KV heads (GQA).
    n_head_kv: u32 = 8,
    /// Per-head key/value dimension.
    head_dim: u32 = 64,
    /// FFN intermediate size per expert.
    n_ff: u32 = 2880,
    /// Vocabulary size.
    vocab_size: u32 = 201088,
    /// Total expert count.
    n_experts: u32 = 32,
    /// Top-k experts selected per token.
    n_experts_active: u32 = 4,
    /// RoPE base frequency.
    rope_theta: f32 = 150000.0,
    /// RMS-norm epsilon.
    rms_eps: f32 = 1e-5,
    /// Sliding-window size for even-numbered layers.
    sliding_window: u32 = 128,
    /// Hard clamp applied after SwiGLU gate*up product.
    swiglu_limit: f32 = 7.0,
    /// MLX quantization bit width (4, 6, or 8). 0 = not MLX format.
    mlx_bits: u32 = 0,
    /// End-of-sequence token identifier.
    eos_token_id: u32 = 200002,
    /// Maximum sequence length for the pre-allocated KV cache.
    max_seq_len: usize = 4096,

    // ── Working buffers (allocated once, reused every token) ──────
    hidden: []f32 = &.{},
    hidden2: []f32 = &.{},
    /// Q projection output — n_head * head_dim elements.
    q_buf: []f32 = &.{},
    /// K projection output — n_head_kv * head_dim elements.
    k_buf: []f32 = &.{},
    /// V projection output — n_head_kv * head_dim elements.
    v_buf: []f32 = &.{},
    /// Attention output before output projection — n_head * head_dim.
    attn_out: []f32 = &.{},
    /// Attention score buffer — max_seq_len + 1 (extra slot for sink).
    scores_buf: []f32 = &.{},
    /// Router logits — one per expert.
    router_logits: []f32 = &.{},
    /// Gate path buffer inside active expert — n_ff elements.
    expert_gate: []f32 = &.{},
    /// Up path buffer inside active expert — n_ff elements.
    expert_up: []f32 = &.{},
    /// Down projection output of active expert — n_embd elements.
    expert_down: []f32 = &.{},
    /// Accumulated weighted MoE output — n_embd elements.
    moe_out: []f32 = &.{},
    /// Final vocabulary logits.
    logits_buf: []f32 = &.{},

    // ── KV cache (PagedAttention or TieredKvCache) ────────────────
    paged_cache: PagedKvCache = undefined,
    seq_table: SeqBlockTable = undefined,
    block_allocator: BlockAllocator = undefined,
    tiered_cache: ?*TieredKvCache = null,
    tiered_block_allocator: ?TieredBlockAllocator = null,
    /// KV cache quantization format for keys.
    kv_type_k: kv_quant.KvQuantType = .f32,
    /// KV cache quantization format for values.
    kv_type_v: kv_quant.KvQuantType = .f32,
    /// Number of tokens currently stored in the KV cache.
    kv_seq_len: usize = 0,
    /// Set to true to abort a running `forward` call from another thread.
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    // ── Norm weight cache (BF16 → f32, converted once) ────────────
    norm_cache: [max_norm_entries]NormCacheEntry = undefined,
    norm_cache_len: usize = 0,

    // ── Lifecycle ─────────────────────────────────────────────────

    /// Initialize the model from format metadata and pre-allocate all buffers.
    /// Caller owns the returned struct and must call `deinit` when done.
    pub fn init(allocator: Allocator, f: Format, be: Backend, ctx_size: u32, kv_type_k: kv_quant.KvQuantType, kv_type_v: kv_quant.KvQuantType, tiered_cache: ?*TieredKvCache) !GptOssModel {
        var self = GptOssModel{ .fmt = f, .be = be, .allocator = allocator };
        self.kv_type_k = kv_type_k;
        self.kv_type_v = kv_type_v;

        const arch = f.getMetaStr("general.architecture") orelse "gpt-oss";
        if (f.getArchU32(arch, "block_count")) |v| self.n_layers = v;
        if (f.getArchU32(arch, "embedding_length")) |v| self.n_embd = v;
        if (f.getArchU32(arch, "attention.head_count")) |v| self.n_head = v;
        if (f.getArchU32(arch, "attention.head_count_kv")) |v| self.n_head_kv = v;
        if (f.getArchU32(arch, "attention.key_length")) |v| self.head_dim = v;
        self.n_ff = f.getArchU32(arch, "expert_feed_forward_length") orelse
            f.getArchU32(arch, "feed_forward_length") orelse self.n_ff;
        if (f.getArchU32(arch, "expert_count")) |v| self.n_experts = v;
        if (f.getArchU32(arch, "expert_used_count")) |v| self.n_experts_active = v;
        if (f.getArchU32(arch, "attention.sliding_window")) |v| self.sliding_window = v;
        if (f.getArchF32(arch, "rope.freq_base")) |v| self.rope_theta = v;
        if (f.getArchF32(arch, "attention.layer_norm_rms_epsilon")) |v| self.rms_eps = v;
        if (f.getMetaU32("tokenizer.ggml.eos_token_id")) |v| self.eos_token_id = v;
        if (f.getVocab()) |v| self.vocab_size = @intCast(v.len);
        if (f.getMetaU32("bits")) |b| self.mlx_bits = b;
        if (f.getArchU32(arch, "context_length")) |cl| self.max_seq_len = cl;
        if (ctx_size > 0) self.max_seq_len = ctx_size;

        std.debug.assert(self.n_head % self.n_head_kv == 0);
        std.debug.assert(self.n_experts_active <= self.n_experts);
        std.debug.assert(self.n_experts_active <= max_active_experts);
        std.debug.assert(self.head_dim % 2 == 0);

        const qd: usize = @as(usize, self.n_head) * self.head_dim;
        const kvd: usize = @as(usize, self.n_head_kv) * self.head_dim;
        const nl: usize = self.n_layers;

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
        self.attn_out = try allocator.alloc(f32, qd);
        errdefer allocator.free(self.attn_out);
        self.scores_buf = try allocator.alloc(f32, self.max_seq_len + attention_sink_slots);
        errdefer allocator.free(self.scores_buf);
        self.router_logits = try allocator.alloc(f32, self.n_experts);
        errdefer allocator.free(self.router_logits);
        self.expert_gate = try allocator.alloc(f32, self.n_ff);
        errdefer allocator.free(self.expert_gate);
        self.expert_up = try allocator.alloc(f32, self.n_ff);
        errdefer allocator.free(self.expert_up);
        self.expert_down = try allocator.alloc(f32, self.n_embd);
        errdefer allocator.free(self.expert_down);
        self.moe_out = try allocator.alloc(f32, self.n_embd);
        errdefer allocator.free(self.moe_out);
        // Pad logits_buf to match lm_head weight rows (may be > vocab_size due to padding).
        const lm_head_rows = if (f.getTensor("output.weight")) |ow| blk: {
            break :blk if (ow.n_dims >= 1) @as(usize, @intCast(ow.dims[0])) else self.vocab_size;
        } else self.vocab_size;
        self.logits_buf = try allocator.alloc(f32, @max(lm_head_rows, self.vocab_size));
        errdefer allocator.free(self.logits_buf);

        // KV cache: use TieredKvCache if provided, otherwise flat PagedKvCache.
        if (tiered_cache) |tc| {
            var ta = TieredBlockAllocator.init(tc, allocator);
            self.seq_table = try ta.allocateSeqTable(nl);
            errdefer ta.freeSeqTable(&self.seq_table);
            try ta.appendBlock(&self.seq_table);
            self.tiered_cache = tc;
            self.tiered_block_allocator = ta;
        } else {
            // GPT-OSS uses inline attention with flat KV pointers — one block must hold
            // the entire sequence. Use max_seq_len as block_size so getLayerKvView returns
            // a single contiguous buffer per layer.
            const block_size: u16 = @intCast(self.max_seq_len);
            const num_blocks = nl; // one block per layer
            self.paged_cache = try PagedKvCache.init(allocator, nl, kvd, num_blocks, block_size);
            errdefer self.paged_cache.deinit();
            // BlockAllocator stores a pointer — must point to self.paged_cache (not a local copy).
            self.block_allocator = BlockAllocator.init(&self.paged_cache, allocator);
            self.seq_table = try self.block_allocator.allocateSeqTable(nl);
            errdefer self.block_allocator.freeSeqTable(&self.seq_table);
            try self.block_allocator.appendBlock(&self.seq_table);
        }

        // Pre-populate norm cache so no allocations happen during inference.
        self.warmNormCache();

        return self;
    }

    /// Release all heap allocations owned by this model.
    pub fn deinit(self: *GptOssModel) void {
        self.be.sync();
        // Free KV cache first (may have been modified by forward)
        if (self.tiered_block_allocator) |*ta| {
            ta.freeSeqTable(&self.seq_table);
        } else {
            self.block_allocator.freeSeqTable(&self.seq_table);
            self.paged_cache.deinit();
        }
        // Free cached norm weight conversions
        for (self.norm_cache[0..self.norm_cache_len]) |entry| self.allocator.free(entry.data);
        // Free working buffers
        const bufs = .{
            &self.hidden,     &self.hidden2,       &self.q_buf,
            &self.k_buf,      &self.v_buf,         &self.attn_out,
            &self.scores_buf, &self.router_logits, &self.expert_gate,
            &self.expert_up,  &self.expert_down,   &self.moe_out,
            &self.logits_buf,
        };
        inline for (bufs) |buf| self.allocator.free(buf.*);
    }

    /// Wrap this model in the generic `Model` interface.
    pub fn model(self: *GptOssModel) Model {
        return Model.from(GptOssModel, self);
    }

    // ── Public interface ──────────────────────────────────────────

    /// Run one decode step.  Returns the argmax next-token id.
    /// Errors: `error.MissingTensor` if a required weight is absent,
    ///         `error.KVCacheFull` if max_seq_len is reached,
    ///         `error.Cancelled` if `cancel()` was called concurrently.
    pub fn forward(self: *GptOssModel, token_id: u32) !u32 {
        if (self.kv_seq_len >= self.max_seq_len) return error.KVCacheFull;

        try model_mod.ensureKvBlock(self);

        // Embedding lookup — no allocation, direct mmap read.
        const emb_t = self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        if (emb_t.dtype == .mlx_q) {
            const emb_s = self.fmt.getTensor("token_embd.scales") orelse return error.MissingTensor;
            const emb_b = self.fmt.getTensor("token_embd.biases") orelse return error.MissingTensor;
            // Embedding scales are BF16 → affine; detect bits from scale dtype
            const bits: u32 = if (emb_s.dtype == .unknown) 4 else 8;
            mlx_ops.mlxEmbLookup(
                self.hidden.ptr,
                @ptrCast(@alignCast(emb_t.data_ptr)),
                @ptrCast(@alignCast(emb_s.data_ptr)),
                @ptrCast(@alignCast(emb_b.data_ptr)),
                token_id,
                self.n_embd,
                bits,
            );
        } else {
            self.be.embLookup(
                .{ .data = emb_t.data_ptr, .dtype = emb_t.dtype },
                token_id,
                self.hidden.ptr,
                self.n_embd,
            );
        }

        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.acquire)) return error.Cancelled;
            try self.attentionLayer(@intCast(li));
            try self.moeLayer(@intCast(li));
        }

        // Final norm → LM head → argmax.
        const nw = self.fmt.getTensor("output_norm.weight") orelse return error.MissingTensor;
        const ow = self.fmt.getTensor("output.weight") orelse return error.MissingTensor;
        self.kv_seq_len += 1;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, self.n_embd), self.hidden.ptr, self.n_embd, self.rms_eps);
        self.be.sync();
        self.doGemv(self.hidden.ptr, ow, self.logits_buf.ptr, self.vocab_size, self.n_embd);
        self.be.sync();
        return math_ops.argmax(self.logits_buf);
    }

    /// Batched prefill — sequential. MoE routing selects different experts
    /// per token, preventing straightforward batched FFN. Attention layers
    /// could be batched but MoE dominates compute.
    pub fn prefill(self: *GptOssModel, token_ids: []const u32) !u32 {
        var last: u32 = 0;
        for (token_ids) |tid| last = try self.forward(tid);
        return last;
    }

    /// Reset the KV cache and cancellation flag for a new conversation.
    pub fn resetCache(self: *GptOssModel) void {
        model_mod.resetKvCache(self);
    }

    /// Signal an in-progress `forward` call to abort.  Thread-safe.
    pub fn cancel(self: *GptOssModel) void {
        model_mod.signalCancel(&self.cancelled);
    }

    /// Return physical block IDs from layer 0 of the current sequence table.
    /// All layers share the same block IDs, so layer 0 is sufficient.
    pub fn getBlockTable(self: *GptOssModel) []const u32 {
        return self.seq_table.block_table[0];
    }

    // ── Layer implementations ─────────────────────────────────────

    /// Helper: get flat f32 view of KV cache for a layer (assembled from paged or tiered blocks).
    fn getLayerKvView(self: *GptOssModel, layer: usize) struct { keys: []f32, values: []f32 } {
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

    /// Dispatch GEMV through the backend. For mlx_q weights, looks up companion
    /// scale/bias tensors and routes to gemvMlxQ (affine) or gemvMxfp4St (MXFP4).
    fn doGemv(self: *GptOssModel, x: [*]const f32, t: TensorInfo, y: [*]f32, n: usize, k: usize) void {
        if (t.dtype != .mlx_q) {
            self.be.gemv(x, .{ .data = t.data_ptr, .dtype = t.dtype }, y, n, k);
            return;
        }
        var comp = self.findCompanion(t) orelse return;
        // Override bits using actual k dimension (findCompanion uses n_embd which may differ)
        if (!comp.is_mxfp4) comp.bits = inferMlxBits(t, @intCast(k));
        if (comp.is_mxfp4) {
            self.be.gemvMxfp4St(x, t.data_ptr, comp.scales, y, n, k);
        } else {
            self.be.gemvMlxQ(x, t.data_ptr, comp.scales, comp.biases, y, n, k, comp.bits);
        }
    }

    /// Dispatch GEMV for a single expert slice from a packed expert tensor.
    /// SafeTensors stores experts as [n_experts, rows, cols] (expert axis first).
    fn doGemvExpert(self: *GptOssModel, x: [*]const f32, exp_t: TensorInfo, ei: usize, stride: usize, y: [*]f32, n: usize, k: usize) void {
        const data = exp_t.data_ptr + ei * stride;
        if (exp_t.dtype != .mlx_q) {
            self.be.gemv(x, .{ .data = data, .dtype = exp_t.dtype }, y, n, k);
            return;
        }
        const comp = self.findCompanion(exp_t) orelse return;
        if (comp.is_mxfp4) {
            // MXFP4: U8 scales only, no quantization bias
            const s_stride = comp.expertScaleStrideMxfp4();
            self.be.gemvMxfp4St(x, data, comp.scales + ei * s_stride, y, n, k);
        } else {
            // MLX affine: BF16 scales + biases (2 bytes/group each)
            const s_stride = comp.expertScaleStrideAffine();
            self.be.gemvMlxQ(x, data, comp.scales + ei * s_stride, comp.biases + ei * s_stride, y, n, k, comp.bits);
        }
    }

    /// Companion tensor info for an MLX-quantized weight tensor.
    const Companion = struct {
        scales: [*]const u8,
        biases: [*]const u8, // undefined for MXFP4 (no quantization bias)
        scale_t: TensorInfo,
        bias_t: TensorInfo, // undefined for MXFP4
        is_mxfp4: bool,
        bits: u32,

        /// Per-expert stride for U8 MXFP4 scales: rows × groups_per_row × 1 byte.
        fn expertScaleStrideMxfp4(self: Companion) usize {
            if (self.scale_t.n_dims >= 3)
                return @as(usize, @intCast(self.scale_t.dims[0])) * @as(usize, @intCast(self.scale_t.dims[1]));
            return self.scale_t.numElements();
        }
        /// Per-expert stride for BF16 affine scales: rows × groups_per_row × 2 bytes.
        fn expertScaleStrideAffine(self: Companion) usize {
            if (self.scale_t.n_dims >= 3)
                return @as(usize, @intCast(self.scale_t.dims[0])) * @as(usize, @intCast(self.scale_t.dims[1])) * 2;
            return self.scale_t.numElements() * 2;
        }
    };

    /// Look up companion .scales/.biases tensors for an mlx_q weight tensor.
    /// MXFP4 has .scales (U8 FP8) but NO .biases — the .bias tensor (singular)
    /// is the model's linear layer bias, handled separately by addBias().
    /// MLX affine has both .scales (BF16) and .biases (BF16).
    fn findCompanion(self: *const GptOssModel, t: TensorInfo) ?Companion {
        const wi = std.mem.lastIndexOf(u8, t.name, ".weight") orelse return null;
        var sbuf: [model_mod.tensor_name_buf_size]u8 = undefined;
        const prefix = t.name[0..wi];
        const s_name = std.fmt.bufPrint(&sbuf, "{s}.scales", .{prefix}) catch return null;
        const st = self.fmt.getTensor(s_name) orelse return null;
        // U8 scales (DType.unknown) = MXFP4 FP8; BF16 scales = affine
        const is_mxfp4 = (st.dtype == .unknown);
        if (is_mxfp4) {
            // MXFP4: no quantization biases — only weight * fp8_scale
            return .{
                .scales = st.data_ptr,
                .biases = undefined,
                .scale_t = st,
                .bias_t = undefined,
                .is_mxfp4 = true,
                .bits = 4,
            };
        }
        // MLX affine: needs .biases companion tensor
        var bbuf: [model_mod.tensor_name_buf_size]u8 = undefined;
        const b_name = std.fmt.bufPrint(&bbuf, "{s}.biases", .{prefix}) catch return null;
        const bt = self.fmt.getTensor(b_name) orelse return null;
        return .{
            .scales = st.data_ptr,
            .biases = bt.data_ptr,
            .scale_t = st,
            .bias_t = bt,
            .is_mxfp4 = false,
            // Detect bits per-tensor from weight words: bits = last_dim * 32 / k
            // where last_dim is words_per_row. Handles mixed-quant (8-bit attn + 4-bit experts).
            .bits = inferMlxBits(t, self.n_embd),
        };
    }

    /// One attention layer: pre-norm → QKV + bias → RoPE → KV cache →
    /// masked dot-product attention (sliding-window on even layers) →
    /// learned sink logit → output proj + bias → post-attention norm →
    /// residual add.
    fn attentionLayer(self: *GptOssModel, li: u32) !void {
        const e: usize = self.n_embd;
        const nh: usize = self.n_head;
        const nkv: usize = self.n_head_kv;
        const hd: usize = self.head_dim;
        const qd: usize = nh * hd;
        const kvd: usize = nkv * hd;

        // 1. Pre-attention RMS norm: hidden2 = norm(hidden, attn_norm)
        const nw = self.fmt.layerTensor(li, "attn_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        // 2. QKV projections.
        const qw = self.fmt.layerTensor(li, "attn_q.weight") orelse return error.MissingTensor;
        const kw = self.fmt.layerTensor(li, "attn_k.weight") orelse return error.MissingTensor;
        const vw = self.fmt.layerTensor(li, "attn_v.weight") orelse return error.MissingTensor;
        self.doGemv(self.hidden2.ptr, qw, self.q_buf.ptr, qd, e);
        self.doGemv(self.hidden2.ptr, kw, self.k_buf.ptr, kvd, e);
        self.doGemv(self.hidden2.ptr, vw, self.v_buf.ptr, kvd, e);

        // 3. Add projection biases.
        const qb = self.fmt.layerTensor(li, "attn_q.bias") orelse return error.MissingTensor;
        const kb = self.fmt.layerTensor(li, "attn_k.bias") orelse return error.MissingTensor;
        const vb = self.fmt.layerTensor(li, "attn_v.bias") orelse return error.MissingTensor;
        self.be.sync();
        addBiasTyped(self.q_buf[0..qd], qb.data_ptr, qd, qb.dtype);
        addBiasTyped(self.k_buf[0..kvd], kb.data_ptr, kvd, kb.dtype);
        addBiasTyped(self.v_buf[0..kvd], vb.data_ptr, kvd, vb.dtype);

        // 4. Rotary position embeddings (full head rotation).
        self.be.rope(self.q_buf.ptr, self.kv_seq_len, nh, hd, hd, self.rope_theta);
        self.be.rope(self.k_buf.ptr, self.kv_seq_len, nkv, hd, hd, self.rope_theta);

        // 5. Append current token to KV cache.
        self.be.sync();
        const kv_view = self.getLayerKvView(li);
        const pos = self.kv_seq_len;
        @memcpy(kv_view.keys[pos * kvd ..][0..kvd], self.k_buf[0..kvd]);
        @memcpy(kv_view.values[pos * kvd ..][0..kvd], self.v_buf[0..kvd]);

        // 6. Dot-product attention with optional sliding window.
        const sl = self.kv_seq_len + 1; // sequence length including current token
        const is_sliding = (li % 2 == 0);
        const win: usize = if (is_sliding) @min(sl, self.sliding_window) else sl;
        const start: usize = if (is_sliding and sl > self.sliding_window) sl - self.sliding_window else 0;

        // Learned attention sinks — optional per-head scalar logit prepended to scores.
        // SafeTensors stores sinks as BF16; dequant inline.
        const sinks_t = self.fmt.layerTensor(li, "attn_sinks.weight");
        var sinks_f32: [max_sink_heads]f32 = undefined;
        const sinks: ?[*]const f32 = if (sinks_t) |st| blk: {
            if (st.dtype == .bf16) {
                const bf16_ptr: [*]const u16 = @ptrCast(@alignCast(st.data_ptr));
                for (0..nh) |i| sinks_f32[i] = quant.bf16ToF32(bf16_ptr[i]);
                break :blk &sinks_f32;
            }
            break :blk @ptrCast(@alignCast(st.data_ptr));
        } else null;

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const hpg = nh / nkv; // heads per KV group
        const ck = kv_view.keys.ptr;
        const cv = kv_view.values.ptr;

        for (0..nh) |h| {
            const kvh = h / hpg;
            var score_off: usize = 0;

            // Prepend the learned sink logit (if present) at scores_buf[0].
            if (sinks) |s| {
                self.scores_buf[0] = s[h];
                score_off = 1;
            }

            // Compute QK^T dot products for the visible window.
            const q_base = h * hd;
            for (0..win) |wi| {
                const t = start + wi;
                const elem_off = t * kvd + kvh * hd;
                const k_ptr: [*]const f32 = @ptrCast(@alignCast(ck + elem_off * @sizeOf(f32)));
                var dot: f32 = 0;
                for (0..hd) |i| dot += self.q_buf[q_base + i] * k_ptr[i];
                self.scores_buf[score_off + wi] = dot * scale;
            }

            // Inline CPU softmax over sink + window scores — avoids backend
            // dispatch overhead since QK dot products are CPU SIMD.
            const n_scores = score_off + win;
            {
                var max_val: f32 = self.scores_buf[0];
                for (1..n_scores) |i| if (self.scores_buf[i] > max_val) {
                    max_val = self.scores_buf[i];
                };
                var sm_sum: f32 = 0;
                for (0..n_scores) |i| {
                    self.scores_buf[i] = @exp(self.scores_buf[i] - max_val);
                    sm_sum += self.scores_buf[i];
                }
                const inv_sum = 1.0 / sm_sum;
                for (0..n_scores) |i| self.scores_buf[i] *= inv_sum;
            }

            // Weighted value accumulation (sink contributes no value vector).
            @memset(self.attn_out[q_base..][0..hd], 0);
            for (0..win) |wi| {
                const t = start + wi;
                const elem_off = t * kvd + kvh * hd;
                const v_ptr: [*]const f32 = @ptrCast(@alignCast(cv + elem_off * @sizeOf(f32)));
                const weight = self.scores_buf[score_off + wi];
                for (0..hd) |i| self.attn_out[q_base + i] += weight * v_ptr[i];
            }
        }

        // 7. Output projection (with bias).
        const ow = self.fmt.layerTensor(li, "attn_output.weight") orelse return error.MissingTensor;
        self.doGemv(self.attn_out.ptr, ow, self.hidden2.ptr, e, qd);
        const ob = self.fmt.layerTensor(li, "attn_output.bias") orelse return error.MissingTensor;
        self.be.sync();
        addBiasTyped(self.hidden2[0..e], ob.data_ptr, e, ob.dtype);

        // 8. Post-attention norm applied to the attention output.
        const pan = self.fmt.layerTensor(li, "post_attention_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden2.ptr, self.normAsF32(pan, e), self.hidden2.ptr, e, self.rms_eps);

        // 9. Residual: hidden += attn_out (post-normed).
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
    }

    /// MoE FFN layer: optional pre-FFN norm → router → top-k selection →
    /// per-expert SwiGLU (with clamp) → weighted accumulation → residual add.
    fn moeLayer(self: *GptOssModel, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.n_ff;
        const n_exp: usize = self.n_experts;
        const n_active: usize = self.n_experts_active;

        // Optional pre-FFN norm (may be absent — model uses residual stream directly).
        if (self.fmt.layerTensor(li, "ffn_norm.weight")) |fnw| {
            self.be.rmsNorm(self.hidden.ptr, self.normAsF32(fnw, e), self.hidden2.ptr, e, self.rms_eps);
        } else {
            @memcpy(self.hidden2[0..e], self.hidden[0..e]);
        }

        // 1. Router: logits = router_weight @ hidden2 (+ optional bias)
        const rw = self.fmt.layerTensor(li, "ffn_gate_inp.weight") orelse return error.MissingTensor;
        self.doGemv(self.hidden2.ptr, rw, self.router_logits.ptr, n_exp, e);
        self.be.sync();
        if (self.fmt.layerTensor(li, "ffn_gate_inp.bias")) |rb| {
            addBiasTyped(self.router_logits[0..n_exp], rb.data_ptr, n_exp, rb.dtype);
        }

        // 2. Top-k expert selection (stack-allocated, no heap).
        var top_experts: [max_active_experts]usize = undefined;
        var top_scores: [max_active_experts]f32 = undefined;
        std.debug.assert(n_active <= max_active_experts);
        math_ops.topKExperts(self.router_logits[0..n_exp], n_active, top_experts[0..n_active], top_scores[0..n_active]);

        // 3. Softmax over selected expert scores to get mixing weights.
        var max_score = top_scores[0];
        for (1..n_active) |i| if (top_scores[i] > max_score) {
            max_score = top_scores[i];
        };
        var sum_exp: f32 = 0.0;
        for (0..n_active) |i| {
            top_scores[i] = @exp(top_scores[i] - max_score);
            sum_exp += top_scores[i];
        }
        const inv_sum = 1.0 / sum_exp;
        for (0..n_active) |i| top_scores[i] *= inv_sum;

        // 4. Fetch packed expert tensor metadata once.
        const gate_exps = self.fmt.layerTensor(li, "ffn_gate_exps.weight") orelse return error.MissingTensor;
        const up_exps = self.fmt.layerTensor(li, "ffn_up_exps.weight") orelse return error.MissingTensor;
        const down_exps = self.fmt.layerTensor(li, "ffn_down_exps.weight") orelse return error.MissingTensor;
        const gate_bias_t = self.fmt.layerTensor(li, "ffn_gate_exps.bias");
        const up_bias_t = self.fmt.layerTensor(li, "ffn_up_exps.bias");
        const down_bias_t = self.fmt.layerTensor(li, "ffn_down_exps.bias");

        // Byte stride per expert inside the packed weight tensor.
        // GGUF: dims = [inner, outer, n_experts] — expertWeightStride uses dims[0]*dims[1]
        // SafeTensors MLX: dims = [n_experts, rows, cols] — stride = dims[0]*dims[1]*sizeof(u32)
        const gate_stride = expertStride(gate_exps);
        const up_stride = expertStride(up_exps);
        const down_stride = expertStride(down_exps);

        // 5. Accumulate weighted expert outputs.
        @memset(self.moe_out[0..e], 0);

        for (0..n_active) |ti| {
            const ei = top_experts[ti];
            const mix_weight = top_scores[ti];

            // Gate + Up projections. Use batched gemvMulti for non-MLX,
            // sequential doGemvExpert for MLX (needs companion tensors).
            if (gate_exps.dtype == .mlx_q and self.mlx_bits > 0) {
                self.doGemvExpert(self.hidden2.ptr, gate_exps, ei, gate_stride, self.expert_gate.ptr, ff, e);
                self.doGemvExpert(self.hidden2.ptr, up_exps, ei, up_stride, self.expert_up.ptr, ff, e);
            } else {
                const gate_data = gate_exps.data_ptr + ei * gate_stride;
                const up_data = up_exps.data_ptr + ei * up_stride;
                const GemvOp = backend_mod.GemvOp;
                const exp_ops = [_]GemvOp{
                    .{ .w = .{ .data = gate_data, .dtype = gate_exps.dtype }, .y = self.expert_gate.ptr, .n = ff },
                    .{ .w = .{ .data = up_data, .dtype = up_exps.dtype }, .y = self.expert_up.ptr, .n = ff },
                };
                self.be.gemvMulti(self.hidden2.ptr, &exp_ops, e);
            }
            self.be.sync();
            if (gate_bias_t) |gbt| {
                const bpe: usize = if (gbt.dtype == .bf16) backend_mod.f16_elem_bytes else @sizeOf(f32);
                addBiasTyped(self.expert_gate[0..ff], gbt.data_ptr + ei * ff * bpe, ff, gbt.dtype);
            }
            if (up_bias_t) |ubt| {
                const bpe: usize = if (ubt.dtype == .bf16) backend_mod.f16_elem_bytes else @sizeOf(f32);
                addBiasTyped(self.expert_up[0..ff], ubt.data_ptr + ei * ff * bpe, ff, ubt.dtype);
            }

            // SwiGLU with hard clamp: out = clamp(silu(gate) * up, ±limit)
            const limit = self.swiglu_limit;
            for (0..ff) |i| {
                const g = self.expert_gate[i];
                const silu_g = math_ops.silu(g);
                const prod = silu_g * self.expert_up[i];
                self.expert_gate[i] = @min(@max(prod, -limit), limit);
            }

            // Down projection: expert_down = down_exps[ei] @ expert_gate
            self.doGemvExpert(self.expert_gate.ptr, down_exps, ei, down_stride, self.expert_down.ptr, e, ff);
            self.be.sync();
            if (down_bias_t) |dbt| {
                const bpe: usize = if (dbt.dtype == .bf16) backend_mod.f16_elem_bytes else @sizeOf(f32);
                addBiasTyped(self.expert_down[0..e], dbt.data_ptr + ei * e * bpe, e, dbt.dtype);
            }

            // Weighted accumulation.
            for (0..e) |i| self.moe_out[i] += mix_weight * self.expert_down[i];
        }

        // 6. Residual: hidden += moe_out
        self.be.add(self.hidden.ptr, self.moe_out.ptr, self.hidden.ptr, e);
    }

    /// Pre-populate the norm weight cache during init so no allocations occur
    /// in the hot path. Iterates all norm tensors and triggers conversion.
    fn warmNormCache(self: *GptOssModel) void {
        const e: usize = self.n_embd;
        if (self.fmt.getTensor("output_norm.weight")) |t| _ = self.normAsF32(t, e);
        for (0..self.n_layers) |i| {
            const li: u32 = @intCast(i);
            if (self.fmt.layerTensor(li, "attn_norm.weight")) |t| _ = self.normAsF32(t, e);
            if (self.fmt.layerTensor(li, "post_attention_norm.weight")) |t| _ = self.normAsF32(t, e);
            if (self.fmt.layerTensor(li, "ffn_norm.weight")) |t| _ = self.normAsF32(t, e);
        }
    }

    /// Get norm weights as f32 pointer. Caches converted weights on first access
    /// so subsequent tokens return a stable pointer with zero work and no GPU syncs.
    fn normAsF32(self: *GptOssModel, t: TensorInfo, n: usize) [*]const f32 {
        if (t.dtype != .bf16) return @ptrCast(@alignCast(t.data_ptr));

        // Check cache (linear scan — at most ~73 entries, first-token only on miss)
        const key = @intFromPtr(t.data_ptr);
        for (self.norm_cache[0..self.norm_cache_len]) |entry| {
            if (entry.key == key) return entry.data.ptr;
        }

        // Cache miss: allocate, convert, and store permanently
        if (self.norm_cache_len >= max_norm_entries) {
            // Cache full — fall back to raw pointer (BF16 read directly).
            // This should not happen in practice (max ~73 entries needed).
            return @ptrCast(@alignCast(t.data_ptr));
        }
        const buf = self.allocator.alloc(f32, n) catch return @ptrCast(@alignCast(t.data_ptr));
        const src: [*]const u16 = @ptrCast(@alignCast(t.data_ptr));
        for (0..n) |i| buf[i] = quant.bf16ToF32(src[i]);
        self.norm_cache[self.norm_cache_len] = .{ .key = key, .data = buf };
        self.norm_cache_len += 1;
        return buf.ptr;
    }
};

// ── Free functions ────────────────────────────────────────────────

/// Add f32 bias (pointed to by `bias_bytes`) into `dst`.
/// For bf16 bias data, call `addBiasTyped()` directly with `.bf16`.
inline fn addBias(dst: []f32, bias_bytes: [*]const u8, n: usize) void {
    addBiasTyped(dst, bias_bytes, n, .f32);
}

/// Add bias with explicit dtype handling.
fn addBiasTyped(dst: []f32, bias_bytes: [*]const u8, n: usize, dtype: format_mod.DType) void {
    if (dtype == .bf16) {
        const bias_bf16: [*]const u16 = @ptrCast(@alignCast(bias_bytes));
        for (0..n) |i| dst[i] += quant.bf16ToF32(bias_bf16[i]);
    } else {
        const bias: [*]const f32 = @ptrCast(@alignCast(bias_bytes));
        for (0..n) |i| dst[i] += bias[i];
    }
}

/// Infer MLX quantization bits from weight tensor dimensions.
/// bits = last_dim * 32 / k, where last_dim is u32 words per row.
/// For 8-bit: 720 words * 32 / 2880 = 8. For 4-bit: 360 * 32 / 2880 = 4.
fn inferMlxBits(t: TensorInfo, k: u32) u32 {
    if (t.n_dims < 2 or k == 0) return 4;
    // For expert tensors [n_experts, rows, words], use dims[n_dims-1]
    const words_dim = @as(u64, @intCast(t.dims[t.n_dims - 1]));
    const result = words_dim * 32 / @as(u64, k);
    return if (result >= min_mlx_bits and result <= max_mlx_bits) @intCast(result) else default_mlx_bits;
}

const expertWeightStride = model_mod.expertWeightStride;
