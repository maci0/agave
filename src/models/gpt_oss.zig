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
const BlockAllocator = @import("../kvcache/block_allocator.zig").BlockAllocator;

const Backend = backend_mod.Backend;
const TensorData = backend_mod.TensorData;
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;
const kv_quant = @import("../ops/kv_quant.zig");
const PagedKvCache = kvcache.PagedKvCache;
const SeqBlockTable = kvcache.SeqBlockTable;

/// Maximum top-k experts for stack-allocated selection arrays.
const max_active_experts: usize = 8;
/// Extra score slot reserved for the learned attention sink.
const attention_sink_slots: usize = 1;

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

    // ── KV cache (PagedAttention) ────────────────────────────────
    paged_cache: PagedKvCache = undefined,
    seq_table: SeqBlockTable = undefined,
    block_allocator: BlockAllocator = undefined,
    /// KV cache quantization format.
    kv_type: kv_quant.KvQuantType = .f32,
    /// Number of tokens currently stored in the KV cache.
    kv_seq_len: usize = 0,
    /// Set to true to abort a running `forward` call from another thread.
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    // ── Lifecycle ─────────────────────────────────────────────────

    /// Initialize the model from format metadata and pre-allocate all buffers.
    /// Caller owns the returned struct and must call `deinit` when done.
    pub fn init(allocator: Allocator, f: Format, be: Backend, ctx_size: u32, kv_type: kv_quant.KvQuantType) !GptOssModel {
        var self = GptOssModel{ .fmt = f, .be = be, .allocator = allocator };
        self.kv_type = kv_type;

        const arch = f.getMetaStr("general.architecture") orelse "gpt-oss";
        self.n_layers = f.getArchU32(arch, "block_count") orelse 24;
        self.n_embd = f.getArchU32(arch, "embedding_length") orelse 2880;
        self.n_head = f.getArchU32(arch, "attention.head_count") orelse 64;
        self.n_head_kv = f.getArchU32(arch, "attention.head_count_kv") orelse 8;
        self.head_dim = f.getArchU32(arch, "attention.key_length") orelse 64;
        self.n_ff = f.getArchU32(arch, "expert_feed_forward_length") orelse
            f.getArchU32(arch, "feed_forward_length") orelse 2880;
        self.n_experts = f.getArchU32(arch, "expert_count") orelse 32;
        self.n_experts_active = f.getArchU32(arch, "expert_used_count") orelse 4;
        self.sliding_window = f.getArchU32(arch, "attention.sliding_window") orelse 128;
        if (f.getArchF32(arch, "rope.freq_base")) |v| self.rope_theta = v;
        if (f.getArchF32(arch, "attention.layer_norm_rms_epsilon")) |v| self.rms_eps = v;
        if (f.getMetaU32("tokenizer.ggml.eos_token_id")) |v| self.eos_token_id = v;
        if (f.getVocab()) |v| self.vocab_size = @intCast(v.len);
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
        self.logits_buf = try allocator.alloc(f32, self.vocab_size);
        errdefer allocator.free(self.logits_buf);

        // PagedAttention: block_size=16, calculate num_blocks for max_seq_len
        const block_size: u16 = 16;
        const num_blocks = (self.max_seq_len + block_size - 1) / block_size * nl;

        var paged_cache = try PagedKvCache.init(allocator, nl, kvd, num_blocks, block_size);
        errdefer paged_cache.deinit();

        var block_allocator = BlockAllocator.init(&paged_cache, allocator);

        var seq_table = try block_allocator.allocateSeqTable(nl);
        errdefer block_allocator.freeSeqTable(&seq_table);

        // Allocate first block
        try block_allocator.appendBlock(&seq_table);

        self.paged_cache = paged_cache;
        self.seq_table = seq_table;
        self.block_allocator = block_allocator;

        return self;
    }

    /// Release all heap allocations owned by this model.
    pub fn deinit(self: *GptOssModel) void {
        const bufs = .{
            &self.hidden,     &self.hidden2,       &self.q_buf,
            &self.k_buf,      &self.v_buf,         &self.attn_out,
            &self.scores_buf, &self.router_logits, &self.expert_gate,
            &self.expert_up,  &self.expert_down,   &self.moe_out,
            &self.logits_buf,
        };
        inline for (bufs) |buf| self.allocator.free(buf.*);

        self.block_allocator.freeSeqTable(&self.seq_table);
        self.paged_cache.deinit();
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

        // Check if new block needed
        const current_blocks = self.seq_table.block_table[0].len;
        const needed_blocks = (self.kv_seq_len + 1 + self.paged_cache.block_size - 1) / self.paged_cache.block_size;
        if (needed_blocks > current_blocks) {
            try self.block_allocator.appendBlock(&self.seq_table);
        }

        // Embedding lookup — no allocation, direct mmap read.
        const emb_t = self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        self.be.embLookup(
            .{ .data = emb_t.data_ptr, .dtype = emb_t.dtype },
            token_id,
            self.hidden.ptr,
            self.n_embd,
        );

        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.acquire)) return error.Cancelled;
            try self.attentionLayer(@intCast(li));
            try self.moeLayer(@intCast(li));
        }

        // Final norm → LM head → argmax.
        const nw = self.fmt.getTensor("output_norm.weight") orelse return error.MissingTensor;
        const ow = self.fmt.getTensor("output.weight") orelse return error.MissingTensor;
        self.kv_seq_len += 1;
        return math_ops.finalLogits(
            self.hidden.ptr,
            nw.data_ptr,
            .{ .data = ow.data_ptr, .dtype = ow.dtype },
            self.logits_buf,
            self.vocab_size,
            self.n_embd,
            self.rms_eps,
            self.be,
        );
    }

    /// Reset the KV cache and cancellation flag for a new conversation.
    pub fn resetCache(self: *GptOssModel) void {
        self.block_allocator.freeSeqTable(&self.seq_table);
        self.seq_table = self.block_allocator.allocateSeqTable(self.n_layers) catch {
            return;
        };
        self.block_allocator.appendBlock(&self.seq_table) catch {
            return;
        };
        model_mod.resetInferenceState(&self.kv_seq_len, &self.cancelled);
    }

    /// Signal an in-progress `forward` call to abort.  Thread-safe.
    pub fn cancel(self: *GptOssModel) void {
        model_mod.signalCancel(&self.cancelled);
    }

    // ── Layer implementations ─────────────────────────────────────

    /// Helper: get flat f32 view of KV cache for a layer (assembled from paged blocks).
    /// Returns slices pointing into paged cache blocks.
    fn getLayerKvView(self: *GptOssModel, layer: usize) struct { keys: []f32, values: []f32 } {
        const num_blocks = self.seq_table.block_table[layer].len;
        if (num_blocks == 0) return .{ .keys = &[_]f32{}, .values = &[_]f32{} };

        // For now, assume single block (will extend for multi-block later)
        const block_id = self.seq_table.block_table[layer][0];
        return .{
            .keys = self.paged_cache.blocks[block_id].keys,
            .values = self.paged_cache.blocks[block_id].values,
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
        self.be.rmsNorm(
            self.hidden.ptr,
            @ptrCast(@alignCast(nw.data_ptr)),
            self.hidden2.ptr,
            e,
            self.rms_eps,
        );

        // 2. QKV projections.
        const qw = self.fmt.layerTensor(li, "attn_q.weight") orelse return error.MissingTensor;
        const kw = self.fmt.layerTensor(li, "attn_k.weight") orelse return error.MissingTensor;
        const vw = self.fmt.layerTensor(li, "attn_v.weight") orelse return error.MissingTensor;
        self.be.gemv(self.hidden2.ptr, .{ .data = qw.data_ptr, .dtype = qw.dtype }, self.q_buf.ptr, qd, e);
        self.be.gemv(self.hidden2.ptr, .{ .data = kw.data_ptr, .dtype = kw.dtype }, self.k_buf.ptr, kvd, e);
        self.be.gemv(self.hidden2.ptr, .{ .data = vw.data_ptr, .dtype = vw.dtype }, self.v_buf.ptr, kvd, e);

        // 3. Add projection biases.
        const qb = self.fmt.layerTensor(li, "attn_q.bias") orelse return error.MissingTensor;
        const kb = self.fmt.layerTensor(li, "attn_k.bias") orelse return error.MissingTensor;
        const vb = self.fmt.layerTensor(li, "attn_v.bias") orelse return error.MissingTensor;
        self.be.sync();
        addBias(self.q_buf[0..qd], qb.data_ptr, qd);
        addBias(self.k_buf[0..kvd], kb.data_ptr, kvd);
        addBias(self.v_buf[0..kvd], vb.data_ptr, kvd);

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
        const sinks_t = self.fmt.layerTensor(li, "attn_sinks.weight");
        const sinks: ?[*]const f32 = if (sinks_t) |st| @ptrCast(@alignCast(st.data_ptr)) else null;

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
        self.be.gemv(self.attn_out.ptr, .{ .data = ow.data_ptr, .dtype = ow.dtype }, self.hidden2.ptr, e, qd);
        const ob = self.fmt.layerTensor(li, "attn_output.bias") orelse return error.MissingTensor;
        self.be.sync();
        addBias(self.hidden2[0..e], ob.data_ptr, e);

        // 8. Post-attention norm applied to the attention output.
        const pan = self.fmt.layerTensor(li, "post_attention_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(
            self.hidden2.ptr,
            @ptrCast(@alignCast(pan.data_ptr)),
            self.hidden2.ptr,
            e,
            self.rms_eps,
        );

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
            self.be.rmsNorm(
                self.hidden.ptr,
                @ptrCast(@alignCast(fnw.data_ptr)),
                self.hidden2.ptr,
                e,
                self.rms_eps,
            );
        } else {
            @memcpy(self.hidden2[0..e], self.hidden[0..e]);
        }

        // 1. Router: logits = router_weight @ hidden2 (+ optional bias)
        const rw = self.fmt.layerTensor(li, "ffn_gate_inp.weight") orelse return error.MissingTensor;
        self.be.gemv(
            self.hidden2.ptr,
            .{ .data = rw.data_ptr, .dtype = rw.dtype },
            self.router_logits.ptr,
            n_exp,
            e,
        );
        self.be.sync();
        if (self.fmt.layerTensor(li, "ffn_gate_inp.bias")) |rb| {
            addBias(self.router_logits[0..n_exp], rb.data_ptr, n_exp);
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
        // GGUF dims = [n_ff_or_embd, n_embd_or_ff, n_experts] (innermost first).
        const gate_stride = expertWeightStride(gate_exps);
        const up_stride = expertWeightStride(up_exps);
        const down_stride = expertWeightStride(down_exps);

        // 5. Accumulate weighted expert outputs.
        @memset(self.moe_out[0..e], 0);

        for (0..n_active) |ti| {
            const ei = top_experts[ti];
            const mix_weight = top_scores[ti];

            // Gate + Up projections batched into a single gemvMulti dispatch.
            // Saves one GPU sync vs. two separate gemv calls.
            const gate_data = gate_exps.data_ptr + ei * gate_stride;
            const up_data = up_exps.data_ptr + ei * up_stride;
            const GemvOp = backend_mod.GemvOp;
            const exp_ops = [_]GemvOp{
                .{ .w = .{ .data = gate_data, .dtype = gate_exps.dtype }, .y = self.expert_gate.ptr, .n = ff },
                .{ .w = .{ .data = up_data, .dtype = up_exps.dtype }, .y = self.expert_up.ptr, .n = ff },
            };
            self.be.gemvMulti(self.hidden2.ptr, &exp_ops, e);
            self.be.sync();
            if (gate_bias_t) |gbt| addBias(self.expert_gate[0..ff], gbt.data_ptr + ei * ff * @sizeOf(f32), ff);
            if (up_bias_t) |ubt| addBias(self.expert_up[0..ff], ubt.data_ptr + ei * ff * @sizeOf(f32), ff);

            // SwiGLU with hard clamp: out = clamp(silu(gate) * up, ±limit)
            const limit = self.swiglu_limit;
            for (0..ff) |i| {
                const g = self.expert_gate[i];
                const silu_g = math_ops.silu(g);
                const prod = silu_g * self.expert_up[i];
                self.expert_gate[i] = @min(@max(prod, -limit), limit);
            }

            // Down projection: expert_down = down_exps[ei] @ expert_gate
            const down_data = down_exps.data_ptr + ei * down_stride;
            self.be.gemv(
                self.expert_gate.ptr,
                .{ .data = down_data, .dtype = down_exps.dtype },
                self.expert_down.ptr,
                e,
                ff,
            );
            self.be.sync();
            if (down_bias_t) |dbt| addBias(self.expert_down[0..e], dbt.data_ptr + ei * e * @sizeOf(f32), e);

            // Weighted accumulation.
            for (0..e) |i| self.moe_out[i] += mix_weight * self.expert_down[i];
        }

        // 6. Residual: hidden += moe_out
        self.be.add(self.hidden.ptr, self.moe_out.ptr, self.hidden.ptr, e);
    }
};

// ── Free functions ────────────────────────────────────────────────

/// Add a raw f32 bias array (pointed to by `bias_bytes`) into `dst`.
/// `bias_bytes` must point to at least `n * 4` valid bytes of f32 data.
inline fn addBias(dst: []f32, bias_bytes: [*]const u8, n: usize) void {
    const bias: [*]const f32 = @ptrCast(@alignCast(bias_bytes));
    for (0..n) |i| dst[i] += bias[i];
}

const expertWeightStride = model_mod.expertWeightStride;
