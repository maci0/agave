//! Nemotron Nano 30B-A3B — NVFP4-quantized hybrid Mamba-2 + MoE + Attention decoder.
//!
//! Loads from SafeTensors format with `backbone.layers.*` tensor naming.
//! Architecture: 52 layers following `hybrid_override_pattern` from config.json:
//!   M = Mamba-2 SSM, E = MoE FFN (128 routed + 1 shared expert), * = GQA Attention
//!
//! Key differences from the GGUF NemotronHModel:
//! - NVFP4 quantization: weights as packed U32 nibbles + separate FP8 E4M3 scales
//! - BF16 norm weights, biases, and attention projections (not f32)
//! - MoE layers (not plain FFN) with sigmoid routing and shared experts
//! - A_log stored as log values; code applies -exp(A_log) during inference

const std = @import("std");
const Allocator = std.mem.Allocator;
const backend_mod = @import("../backend/backend.zig");
const format_mod = @import("../format/format.zig");
const model_mod = @import("model.zig");
const math_ops = @import("../ops/math.zig");
const attn_ops = @import("../ops/attention.zig");
const ssm_ops = @import("../ops/ssm.zig");
const quant = @import("../ops/quant.zig");
const kvcache = @import("../kvcache/manager.zig");
const BlockAllocator = @import("../kvcache/block_allocator.zig").BlockAllocator;

const Backend = backend_mod.Backend;
const TensorData = backend_mod.TensorData;
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const Model = model_mod.Model;
const kv_quant = @import("../ops/kv_quant.zig");
const PagedKvCache = kvcache.PagedKvCache;
const SeqBlockTable = kvcache.SeqBlockTable;

/// Layer variants in Nemotron Nano.
const LayerType = enum { ssm, moe, attention };

/// Maximum supported layer count.
const max_layers: usize = 64;

/// Stack allocation limit for expert selection arrays (actual per-token count is num_experts_per_tok).
const max_active_experts: usize = 8;

/// Buffer size for tensor name formatting (layer prefix + suffix).
const name_buf_size: usize = 256;

/// NVFP4 packing: 2 values per byte (4 bits each).
const nvfp4_values_per_byte: usize = 2;
/// NVFP4 scaling: 1 FP8 scale per 16 elements.
const nvfp4_scale_group_size: usize = 16;

/// Nemotron-3 Nano 30B hybrid model state.
pub const NemotronNanoModel = struct {
    fmt: Format,
    be: Backend,
    allocator: Allocator,

    // ── Configuration ─────────────────────────────────────────────
    n_layers: u32 = 52,
    n_embd: u32 = 2688,
    n_head: u32 = 32,
    n_head_kv: u32 = 2,
    head_dim: u32 = 128,
    vocab_size: u32 = 131072,
    rope_theta: f32 = 10000.0,
    rope_dim: u32 = 128,
    rms_eps: f32 = 1e-5,
    eos_token_id: u32 = 2,
    max_seq_len: usize = 4096,

    // Mamba-2 SSM parameters
    ssm_d_conv: u32 = 4,
    ssm_d_state: u32 = 128,
    ssm_n_groups: u32 = 8,
    mamba_num_heads: u32 = 64,
    mamba_head_dim: u32 = 64,

    // MoE parameters
    moe_intermediate_size: u32 = 1856,
    shared_expert_size: u32 = 3712,
    num_experts_per_tok: u32 = 6,
    n_routed_experts: u32 = 128,
    routed_scaling_factor: f32 = 2.5,

    // ── Layer-type map ────────────────────────────────────────────
    layer_types: [max_layers]LayerType = [_]LayerType{.moe} ** max_layers,

    // ── Working buffers ───────────────────────────────────────────
    hidden: []f32 = &.{},
    hidden2: []f32 = &.{},
    q_buf: []f32 = &.{},
    k_buf: []f32 = &.{},
    v_buf: []f32 = &.{},
    attn_out: []f32 = &.{},
    scores_buf: []f32 = &.{},
    ssm_proj_buf: []f32 = &.{},
    ssm_conv_out: []f32 = &.{},
    ssm_y_buf: []f32 = &.{},
    router_buf: []f32 = &.{},
    expert_buf: []f32 = &.{},
    moe_out: []f32 = &.{},
    logits_buf: []f32 = &.{},
    /// Large temp buffer for BF16→f32 conversion and SSM scalar staging.
    bf16_buf_large: []f32 = &.{},
    /// Small temp buffer for BF16→f32 conversion (layer norms, biases).
    bf16_buf_small: []f32 = &.{},

    // ── Per-layer state ───────────────────────────────────────────
    conv_states: [][]f32 = &.{},
    ssm_states: [][]f32 = &.{},

    // KV cache (PagedAttention, attention layers only)
    paged_cache: PagedKvCache = undefined,
    seq_table: SeqBlockTable = undefined,
    block_allocator: BlockAllocator = undefined,
    kv_type: kv_quant.KvQuantType = .f32,
    kv_seq_len: usize = 0,
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    // ── Lifecycle ─────────────────────────────────────────────────

    /// Initialize the model, reading hyperparameters from SafeTensors config.json
    /// and allocating all working buffers and per-layer state.
    pub fn init(allocator: Allocator, f: Format, be: Backend, ctx_size: u32, kv_type: kv_quant.KvQuantType) !NemotronNanoModel {
        var self = NemotronNanoModel{ .fmt = f, .be = be, .allocator = allocator };
        self.kv_type = kv_type;

        // Read hyperparameters from config.json metadata
        self.n_layers = f.getMetaU32("num_hidden_layers") orelse 52;
        self.n_embd = f.getMetaU32("hidden_size") orelse 2688;
        self.n_head = f.getMetaU32("num_attention_heads") orelse 32;
        self.n_head_kv = f.getMetaU32("num_key_value_heads") orelse 2;
        self.head_dim = f.getMetaU32("head_dim") orelse 128;
        self.mamba_num_heads = f.getMetaU32("mamba_num_heads") orelse 64;
        self.mamba_head_dim = f.getMetaU32("mamba_head_dim") orelse 64;
        self.ssm_d_conv = f.getMetaU32("conv_kernel") orelse 4;
        self.ssm_d_state = f.getMetaU32("ssm_state_size") orelse 128;
        self.ssm_n_groups = f.getMetaU32("n_groups") orelse 8;
        self.moe_intermediate_size = f.getMetaU32("moe_intermediate_size") orelse
            f.getMetaU32("intermediate_size") orelse 1856;
        self.shared_expert_size = f.getMetaU32("moe_shared_expert_intermediate_size") orelse 3712;
        self.num_experts_per_tok = f.getMetaU32("num_experts_per_tok") orelse 6;
        self.n_routed_experts = f.getMetaU32("n_routed_experts") orelse 128;
        self.vocab_size = f.getMetaU32("vocab_size") orelse 131072;
        self.rms_eps = f.getMetaF32("norm_eps") orelse f.getMetaF32("layer_norm_epsilon") orelse 1e-5;
        self.eos_token_id = f.getMetaU32("eos_token_id") orelse 2;
        self.rope_theta = f.getMetaF32("rope_theta") orelse 10000.0;
        self.rope_dim = self.head_dim; // partial_rotary_factor = 1.0
        if (f.getMetaF32("routed_scaling_factor")) |v| self.routed_scaling_factor = v;
        if (f.getVocab()) |v| self.vocab_size = @intCast(v.len);
        self.max_seq_len = f.getMetaU32("max_position_embeddings") orelse 4096;
        if (ctx_size > 0) self.max_seq_len = ctx_size;

        std.debug.assert(self.n_layers <= max_layers);
        std.debug.assert(self.num_experts_per_tok <= max_active_experts);

        // ── Layer type detection from hybrid_override_pattern ────
        if (f.getMetaStr("hybrid_override_pattern")) |pattern| {
            for (pattern, 0..) |c, i| {
                if (i >= self.n_layers) break;
                self.layer_types[i] = switch (c) {
                    'M' => .ssm,
                    'E' => .moe,
                    '*' => .attention,
                    else => .moe,
                };
            }
        }

        // ── Derived sizes ────────────────────────────────────────
        const e: usize = self.n_embd;
        const qd: usize = @as(usize, self.n_head) * self.head_dim;
        const kvd: usize = @as(usize, self.n_head_kv) * self.head_dim;
        const d_inner: usize = @as(usize, self.mamba_num_heads) * self.mamba_head_dim;
        const conv_ch: usize = d_inner + 2 * @as(usize, self.ssm_n_groups) * self.ssm_d_state;
        const proj_size: usize = d_inner + conv_ch + self.mamba_num_heads;
        const nl: usize = self.n_layers;
        const max_ff: usize = @max(self.shared_expert_size, self.moe_intermediate_size);

        // ── Allocate working buffers ─────────────────────────────
        self.hidden = try allocator.alloc(f32, e);
        errdefer allocator.free(self.hidden);
        self.hidden2 = try allocator.alloc(f32, e);
        errdefer allocator.free(self.hidden2);
        self.q_buf = try allocator.alloc(f32, qd);
        errdefer allocator.free(self.q_buf);
        self.k_buf = try allocator.alloc(f32, kvd);
        errdefer allocator.free(self.k_buf);
        self.v_buf = try allocator.alloc(f32, kvd);
        errdefer allocator.free(self.v_buf);
        self.attn_out = try allocator.alloc(f32, @max(qd, e));
        errdefer allocator.free(self.attn_out);
        self.scores_buf = try allocator.alloc(f32, self.max_seq_len);
        errdefer allocator.free(self.scores_buf);
        self.ssm_proj_buf = try allocator.alloc(f32, proj_size);
        errdefer allocator.free(self.ssm_proj_buf);
        self.ssm_conv_out = try allocator.alloc(f32, conv_ch);
        errdefer allocator.free(self.ssm_conv_out);
        self.ssm_y_buf = try allocator.alloc(f32, d_inner);
        errdefer allocator.free(self.ssm_y_buf);
        self.router_buf = try allocator.alloc(f32, self.n_routed_experts);
        errdefer allocator.free(self.router_buf);
        self.expert_buf = try allocator.alloc(f32, max_ff);
        errdefer allocator.free(self.expert_buf);
        self.moe_out = try allocator.alloc(f32, e);
        errdefer allocator.free(self.moe_out);
        self.logits_buf = try allocator.alloc(f32, self.vocab_size);
        errdefer allocator.free(self.logits_buf);
        self.bf16_buf_large = try allocator.alloc(f32, conv_ch * self.ssm_d_conv);
        errdefer allocator.free(self.bf16_buf_large);
        self.bf16_buf_small = try allocator.alloc(f32, conv_ch);
        errdefer allocator.free(self.bf16_buf_small);

        // ── Per-layer state ──────────────────────────────────────
        const d_state: usize = self.ssm_d_state;
        const state_per_layer: usize = @as(usize, self.mamba_num_heads) * self.mamba_head_dim * d_state;
        const conv_state_len: usize = (@as(usize, self.ssm_d_conv) - 1) * conv_ch;

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
                .moe => {
                    self.conv_states[i] = &.{};
                    self.ssm_states[i] = &.{};
                },
            }
            layer_init_count = i + 1;
        }

        return self;
    }

    /// Release all allocated buffers and per-layer state.
    pub fn deinit(self: *NemotronNanoModel) void {
        const nl: usize = self.n_layers;
        for (0..nl) |i| {
            if (self.conv_states[i].len > 0) self.allocator.free(self.conv_states[i]);
            if (self.ssm_states[i].len > 0) self.allocator.free(self.ssm_states[i]);
        }
        self.allocator.free(self.conv_states);
        self.allocator.free(self.ssm_states);

        self.block_allocator.freeSeqTable(&self.seq_table);
        self.paged_cache.deinit();

        const bufs = .{
            &self.hidden,         &self.hidden2,       &self.q_buf,
            &self.k_buf,          &self.v_buf,         &self.attn_out,
            &self.scores_buf,     &self.ssm_proj_buf,  &self.ssm_conv_out,
            &self.ssm_y_buf,      &self.router_buf,    &self.expert_buf,
            &self.moe_out,        &self.logits_buf,    &self.bf16_buf_large,
            &self.bf16_buf_small,
        };
        inline for (bufs) |buf| self.allocator.free(buf.*);
    }

    /// Return a type-erased Model interface.
    pub fn model(self: *NemotronNanoModel) Model {
        return Model.from(NemotronNanoModel, self);
    }

    // ── Public interface ──────────────────────────────────────────

    /// Run one decode step. Returns the argmax next-token ID.
    pub fn forward(self: *NemotronNanoModel, token_id: u32) !u32 {
        if (self.kv_seq_len >= self.max_seq_len) return error.KVCacheFull;

        // Check if new block needed
        const current_blocks = self.seq_table.block_table[0].len;
        const needed_blocks = (self.kv_seq_len + 1 + self.paged_cache.block_size - 1) / self.paged_cache.block_size;
        if (needed_blocks > current_blocks) {
            try self.block_allocator.appendBlock(&self.seq_table);
        }

        const e: usize = self.n_embd;

        // Embedding lookup (BF16)
        const emb = self.fmt.getTensor("backbone.embeddings.weight") orelse return error.MissingTensor;
        self.be.embLookup(
            .{ .data = emb.data_ptr, .dtype = emb.dtype },
            token_id,
            self.hidden.ptr,
            e,
        );

        // Transformer layers
        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.acquire)) return error.Cancelled;
            const l: u32 = @intCast(li);
            switch (self.layer_types[li]) {
                .ssm => try self.ssmLayer(l),
                .moe => try self.moeLayer(l),
                .attention => try self.attentionLayer(l),
            }
        }

        // Final norm (BF16) → LM head (BF16) → argmax
        const nf = self.fmt.getTensor("backbone.norm_f.weight") orelse return error.MissingTensor;
        bf16ToF32Buf(nf.data_ptr, self.bf16_buf_small[0..e]);
        self.be.rmsNorm(self.hidden.ptr, self.bf16_buf_small.ptr, self.hidden.ptr, e, self.rms_eps);

        const lm = self.fmt.getTensor("lm_head.weight") orelse return error.MissingTensor;
        self.be.gemv(self.hidden.ptr, .{ .data = lm.data_ptr, .dtype = lm.dtype }, self.logits_buf.ptr, self.vocab_size, e);

        self.kv_seq_len += 1;
        self.be.sync();

        const result = math_ops.argmax(self.logits_buf);
        return result;
    }

    /// Reset all SSM states, KV cache, and cancellation flag.
    pub fn resetCache(self: *NemotronNanoModel) void {
        for (0..self.n_layers) |i| {
            if (self.conv_states[i].len > 0) @memset(self.conv_states[i], 0);
            if (self.ssm_states[i].len > 0) @memset(self.ssm_states[i], 0);
        }
        self.block_allocator.freeSeqTable(&self.seq_table);
        self.seq_table = self.block_allocator.allocateSeqTable(self.n_layers) catch {
            return;
        };
        self.block_allocator.appendBlock(&self.seq_table) catch {
            return;
        };
        model_mod.resetInferenceState(&self.kv_seq_len, &self.cancelled);
    }

    /// Signal an in-progress forward pass to abort.
    pub fn cancel(self: *NemotronNanoModel) void {
        model_mod.signalCancel(&self.cancelled);
    }

    // ── Layer implementations ─────────────────────────────────────

    /// Helper: get flat f32 view of KV cache for a layer (assembled from paged blocks).
    fn getLayerKvView(self: *NemotronNanoModel, layer: usize) struct { keys: []u8, values: []u8 } {
        const num_blocks = self.seq_table.block_table[layer].len;
        if (num_blocks == 0) return .{ .keys = &[_]u8{}, .values = &[_]u8{} };

        // For now, assume single block (will extend for multi-block later)
        const block_id = self.seq_table.block_table[layer][0];
        const keys_f32 = self.paged_cache.blocks[block_id].keys;
        const values_f32 = self.paged_cache.blocks[block_id].values;
        return .{
            .keys = std.mem.sliceAsBytes(keys_f32),
            .values = std.mem.sliceAsBytes(values_f32),
        };
    }

    /// Mamba-2 SSM layer: pre-norm → NVFP4 in_proj → causal conv1d →
    /// selective state space recurrence → group norm → SiLU gate →
    /// NVFP4 out_proj → residual add.
    fn ssmLayer(self: *NemotronNanoModel, li: u32) !void {
        const e: usize = self.n_embd;
        const d_inner: usize = @as(usize, self.mamba_num_heads) * self.mamba_head_dim;
        const num_heads: usize = self.mamba_num_heads;
        const mhd: usize = self.mamba_head_dim;
        const d_state: usize = self.ssm_d_state;
        const n_groups: usize = self.ssm_n_groups;
        const heads_per_group: usize = num_heads / n_groups;
        const conv_ch: usize = d_inner + 2 * n_groups * d_state;
        const d_conv: usize = self.ssm_d_conv;
        const proj_size: usize = d_inner + conv_ch + num_heads;

        // 1. Pre-norm (BF16 weights → f32 conversion)
        const nw = self.stLayerTensor(li, "norm.weight") orelse return error.MissingTensor;
        bf16ToF32Buf(nw.data_ptr, self.bf16_buf_small[0..e]);
        self.be.rmsNorm(self.hidden.ptr, self.bf16_buf_small.ptr, self.hidden2.ptr, e, self.rms_eps);
        self.be.sync();

        // 2. Input projection (NVFP4 or BF16 depending on layer)
        const ip_w = self.stLayerTensor(li, "mixer.in_proj.weight") orelse return error.MissingTensor;
        if (self.stLayerTensor(li, "mixer.in_proj.scales")) |ip_s| {
            self.be.gemvNvfp4St(self.hidden2.ptr, ip_w.data_ptr, ip_s.data_ptr, self.ssm_proj_buf.ptr, proj_size, e);
        } else {
            self.be.gemv(self.hidden2.ptr, .{ .data = ip_w.data_ptr, .dtype = ip_w.dtype }, self.ssm_proj_buf.ptr, proj_size, e);
        }
        self.be.sync(); // CPU reads ssm_proj_buf next

        // Split projection: z | conv_in | dt
        const z_ptr = self.ssm_proj_buf.ptr;
        const conv_in_ptr = self.ssm_proj_buf.ptr + d_inner;
        const dt_raw_ptr = self.ssm_proj_buf.ptr + d_inner + conv_ch;

        // 3. Causal conv1d (BF16 weights → f32)
        const cw_t = self.stLayerTensor(li, "mixer.conv1d.weight") orelse return error.MissingTensor;
        const cb_t = self.stLayerTensor(li, "mixer.conv1d.bias") orelse return error.MissingTensor;
        bf16ToF32Buf(cw_t.data_ptr, self.bf16_buf_large[0 .. conv_ch * d_conv]);
        bf16ToF32Buf(cb_t.data_ptr, self.bf16_buf_small[0..conv_ch]);
        ssm_ops.causalConv1dSilu(
            self.ssm_conv_out.ptr,
            self.conv_states[li].ptr,
            conv_in_ptr,
            self.bf16_buf_large.ptr,
            self.bf16_buf_small.ptr,
            conv_ch,
            d_conv,
        );

        // 4. Split conv output: x | B | C
        const x_ptr = self.ssm_conv_out.ptr;
        const B_ptr = self.ssm_conv_out.ptr + d_inner;
        const C_ptr = self.ssm_conv_out.ptr + d_inner + n_groups * d_state;

        // 5. Load SSM scalars (BF16 → f32, with A_log → -exp transform)
        const a_log_t = self.stLayerTensor(li, "mixer.A_log") orelse return error.MissingTensor;
        const d_t = self.stLayerTensor(li, "mixer.D") orelse return error.MissingTensor;
        const dt_bias_t = self.stLayerTensor(li, "mixer.dt_bias") orelse return error.MissingTensor;
        bf16ToF32Buf(a_log_t.data_ptr, self.bf16_buf_large[0..num_heads]);
        // Convert A_log → A = -exp(A_log)
        for (0..num_heads) |h| self.bf16_buf_large[h] = -@exp(self.bf16_buf_large[h]);
        const ssm_a = self.bf16_buf_large.ptr;
        bf16ToF32Buf(d_t.data_ptr, self.bf16_buf_large[num_heads .. 2 * num_heads]);
        const ssm_d = self.bf16_buf_large.ptr + num_heads;
        bf16ToF32Buf(dt_bias_t.data_ptr, self.bf16_buf_large[2 * num_heads .. 3 * num_heads]);
        const dt_bias = self.bf16_buf_large.ptr + 2 * num_heads;

        // 6. Mamba-2 autoregressive recurrence
        const state = self.ssm_states[li];
        const y_ptr = self.ssm_y_buf.ptr;

        ssm_ops.mamba2Recurrence(y_ptr, state, x_ptr, B_ptr, C_ptr, dt_raw_ptr, dt_bias, ssm_a, ssm_d, num_heads, mhd, d_state, heads_per_group);

        // 7. Group RMS norm + SiLU gate (BF16 norm weight)
        const snw_t = self.stLayerTensor(li, "mixer.norm.weight") orelse return error.MissingTensor;
        bf16ToF32Buf(snw_t.data_ptr, self.bf16_buf_large[0..d_inner]);
        const norm_w = self.bf16_buf_large.ptr;

        ssm_ops.groupRmsNormSiluGate(y_ptr, z_ptr, norm_w, d_inner, n_groups, self.rms_eps);

        // 8. Output projection (NVFP4 or BF16 depending on layer)
        const op_w = self.stLayerTensor(li, "mixer.out_proj.weight") orelse return error.MissingTensor;
        if (self.stLayerTensor(li, "mixer.out_proj.scales")) |op_s| {
            self.be.gemvNvfp4St(y_ptr, op_w.data_ptr, op_s.data_ptr, self.hidden2.ptr, e, d_inner);
        } else {
            self.be.gemv(y_ptr, .{ .data = op_w.data_ptr, .dtype = op_w.dtype }, self.hidden2.ptr, e, d_inner);
        }
        self.be.sync();

        // 9. Residual
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
    }

    /// MoE FFN layer: pre-norm → sigmoid router → top-k expert selection →
    /// routed NVFP4 FFN (relu²) → shared expert NVFP4 FFN (relu²) → residual add.
    fn moeLayer(self: *NemotronNanoModel, li: u32) !void {
        const e: usize = self.n_embd;
        const n_exp: usize = self.n_routed_experts;
        const k_exp: usize = self.num_experts_per_tok;
        const ff: usize = self.moe_intermediate_size;
        const shared_ff: usize = self.shared_expert_size;

        // 1. Pre-norm (BF16)
        const nw = self.stLayerTensor(li, "norm.weight") orelse return error.MissingTensor;
        bf16ToF32Buf(nw.data_ptr, self.bf16_buf_small[0..e]);
        self.be.rmsNorm(self.hidden.ptr, self.bf16_buf_small.ptr, self.hidden2.ptr, e, self.rms_eps);

        // 2. Router: BF16 GEMV → sigmoid + bias correction
        const gate_w = self.stLayerTensor(li, "mixer.gate.weight") orelse return error.MissingTensor;
        self.be.sync(); // ensure hidden2 (from rmsNorm) is ready

        self.be.gemv(self.hidden2.ptr, .{ .data = gate_w.data_ptr, .dtype = .bf16 }, self.router_buf.ptr, n_exp, e);
        self.be.sync();

        const bias_t = self.stLayerTensor(li, "mixer.gate.e_score_correction_bias") orelse return error.MissingTensor;
        bf16ToF32Buf(bias_t.data_ptr, self.bf16_buf_small[0..n_exp]);
        for (0..n_exp) |i| {
            const sig = math_ops.sigmoid(self.router_buf[i]);
            self.router_buf[i] = sig + self.bf16_buf_small[i];
        }

        // 3. Top-k selection
        var top_experts: [max_active_experts]usize = undefined;
        var top_scores: [max_active_experts]f32 = undefined;
        math_ops.topKExperts(self.router_buf[0..n_exp], k_exp, &top_experts, &top_scores);

        // Normalize + scale
        var score_sum: f32 = 0;
        for (0..k_exp) |i| score_sum += top_scores[i];
        if (score_sum > 0) {
            const inv_sum = self.routed_scaling_factor / score_sum;
            for (0..k_exp) |i| top_scores[i] *= inv_sum;
        }

        // 4. Routed expert computation
        @memset(self.moe_out, 0);
        const fc1_w = self.stLayerTensor(li, "mixer.switch_mlp.fc1.weight") orelse return error.MissingTensor;
        const fc1_s = self.stLayerTensor(li, "mixer.switch_mlp.fc1.scales") orelse return error.MissingTensor;
        const fc2_w = self.stLayerTensor(li, "mixer.switch_mlp.fc2.weight") orelse return error.MissingTensor;
        const fc2_s = self.stLayerTensor(li, "mixer.switch_mlp.fc2.scales") orelse return error.MissingTensor;

        const fc1_w_stride = ff * (e / nvfp4_values_per_byte);
        const fc1_s_stride = ff * (e / nvfp4_scale_group_size);
        const fc2_w_stride = e * (ff / nvfp4_values_per_byte);
        const fc2_s_stride = e * (ff / nvfp4_scale_group_size);

        for (0..k_exp) |t| {
            const exp_idx = top_experts[t];
            const w = top_scores[t];

            // fc1: up projection [ff, e] NVFP4 → expert_buf [ff]
            self.be.gemvNvfp4St(
                self.hidden2.ptr,
                fc1_w.data_ptr + exp_idx * fc1_w_stride,
                fc1_s.data_ptr + exp_idx * fc1_s_stride,
                self.expert_buf.ptr,
                ff,
                e,
            );
            self.be.sync(); // CPU reads expert_buf next

            // relu²
            math_ops.applyReluSquared(self.expert_buf[0..ff]);

            // fc2: down projection [e, ff] NVFP4 → attn_out [e]
            self.be.gemvNvfp4St(
                self.expert_buf.ptr,
                fc2_w.data_ptr + exp_idx * fc2_w_stride,
                fc2_s.data_ptr + exp_idx * fc2_s_stride,
                self.attn_out.ptr,
                e,
                ff,
            );
            self.be.sync(); // CPU reads attn_out next

            // Accumulate weighted output
            for (0..e) |i| self.moe_out[i] += w * self.attn_out[i];
        }

        // 5. Shared expert
        const sup_w = self.stLayerTensor(li, "mixer.shared_experts.up_proj.weight") orelse return error.MissingTensor;
        const sup_s = self.stLayerTensor(li, "mixer.shared_experts.up_proj.scales") orelse return error.MissingTensor;
        self.be.gemvNvfp4St(self.hidden2.ptr, sup_w.data_ptr, sup_s.data_ptr, self.expert_buf.ptr, shared_ff, e);
        self.be.sync(); // CPU reads expert_buf next

        math_ops.applyReluSquared(self.expert_buf[0..shared_ff]);

        const sdp_w = self.stLayerTensor(li, "mixer.shared_experts.down_proj.weight") orelse return error.MissingTensor;
        const sdp_s = self.stLayerTensor(li, "mixer.shared_experts.down_proj.scales") orelse return error.MissingTensor;
        self.be.gemvNvfp4St(self.expert_buf.ptr, sdp_w.data_ptr, sdp_s.data_ptr, self.attn_out.ptr, e, shared_ff);
        self.be.sync(); // CPU reads attn_out next

        for (0..e) |i| self.moe_out[i] += self.attn_out[i];

        // 6. Residual
        self.be.add(self.hidden.ptr, self.moe_out.ptr, self.hidden.ptr, e);
    }

    /// GQA attention layer: pre-norm → BF16 Q/K/V projections → RoPE →
    /// KV cache append → SDPA → BF16 output projection → residual add.
    fn attentionLayer(self: *NemotronNanoModel, li: u32) !void {
        const e: usize = self.n_embd;
        const nh: usize = self.n_head;
        const nkv: usize = self.n_head_kv;
        const hd: usize = self.head_dim;

        // 1. Pre-norm (BF16)
        const nw = self.stLayerTensor(li, "norm.weight") orelse return error.MissingTensor;
        bf16ToF32Buf(nw.data_ptr, self.bf16_buf_small[0..e]);
        self.be.rmsNorm(self.hidden.ptr, self.bf16_buf_small.ptr, self.hidden2.ptr, e, self.rms_eps);

        // 2. Q/K/V projections (BF16, via backend)
        const qw = self.stLayerTensor(li, "mixer.q_proj.weight") orelse return error.MissingTensor;
        const kw = self.stLayerTensor(li, "mixer.k_proj.weight") orelse return error.MissingTensor;
        const vw = self.stLayerTensor(li, "mixer.v_proj.weight") orelse return error.MissingTensor;
        self.be.gemv(self.hidden2.ptr, .{ .data = qw.data_ptr, .dtype = .bf16 }, self.q_buf.ptr, nh * hd, e);
        self.be.gemv(self.hidden2.ptr, .{ .data = kw.data_ptr, .dtype = .bf16 }, self.k_buf.ptr, nkv * hd, e);
        self.be.gemv(self.hidden2.ptr, .{ .data = vw.data_ptr, .dtype = .bf16 }, self.v_buf.ptr, nkv * hd, e);

        // 3. RoPE
        self.be.rope(self.q_buf.ptr, self.kv_seq_len, nh, hd, self.rope_dim, self.rope_theta);
        self.be.rope(self.k_buf.ptr, self.kv_seq_len, nkv, hd, self.rope_dim, self.rope_theta);

        // 4. KV cache + scaled dot-product attention
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

        // 5. Output projection (BF16)
        const ow = self.stLayerTensor(li, "mixer.o_proj.weight") orelse return error.MissingTensor;
        self.be.gemv(self.attn_out.ptr, .{ .data = ow.data_ptr, .dtype = .bf16 }, self.hidden2.ptr, e, nh * hd);

        // 6. Residual
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
    }

    // ── Helpers ───────────────────────────────────────────────────

    /// Look up a SafeTensors layer tensor by index and suffix.
    /// E.g., stLayerTensor(3, "mixer.q_proj.weight") → "backbone.layers.3.mixer.q_proj.weight"
    fn stLayerTensor(self: *NemotronNanoModel, li: u32, comptime suffix: []const u8) ?TensorInfo {
        var buf: [name_buf_size]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, "backbone.layers.{d}.{s}", .{ li, suffix }) catch return null;
        return self.fmt.getTensor(name);
    }
};

// ── Free functions ────────────────────────────────────────────────

/// Convert a BF16 byte array to f32 values in the given output buffer.
fn bf16ToF32Buf(data: [*]const u8, out: []f32) void {
    const u16s: [*]const u16 = @ptrCast(@alignCast(data));
    for (0..out.len) |i| out[i] = quant.bf16ToF32(u16s[i]);
}

// ── Tests ─────────────────────────────────────────────────────────

test "bf16ToF32Buf" {
    // BF16 1.0 = 0x3F80, -1.0 = 0xBF80
    const bf16_data = [_]u16{ 0x3F80, 0xBF80 };
    var out = [_]f32{ 0, 0 };
    bf16ToF32Buf(@ptrCast(&bf16_data), &out);
    try std.testing.expectEqual(@as(f32, 1.0), out[0]);
    try std.testing.expectEqual(@as(f32, -1.0), out[1]);
}
