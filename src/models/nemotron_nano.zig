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
const mlx_ops = @import("../ops/mlx.zig");
const kvcache = @import("../kvcache/manager.zig");
const block_alloc_mod = @import("../kvcache/block_allocator.zig");
const BlockAllocator = block_alloc_mod.BlockAllocator;
const TieredBlockAllocator = block_alloc_mod.TieredBlockAllocator;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;

const Backend = backend_mod.Backend;
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

/// Maximum routed experts for stack-allocated sigmoid buffer in MoE routing.
const max_routed_experts: usize = 256;
/// Default MLX quantization bit width (4-bit). Canonical source: model.zig.
const default_mlx_bits = model_mod.default_mlx_bits;

/// Maximum norm cache entries. 52 layers × up to 4 norms per layer + 1 final = ~209.
const max_norm_entries: usize = 256;

/// Buffer size for tensor name formatting (layer prefix + suffix).
const name_buf_size: usize = model_mod.tensor_name_buf_size;

/// NVFP4 packing: 2 values per byte (4 bits each).
const nvfp4_values_per_byte: usize = 2;
/// NVFP4 scaling: 1 FP8 scale per 16 elements.
const nvfp4_scale_group_size: usize = 16;
/// MLX quantization group size (elements per scale/bias pair).
const mlx_group_size: usize = mlx_ops.mlx_group_size;

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

    // MLX quantization
    mlx_bits: u32 = 4,

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
    /// Pre-computed SSM constants per layer (one-time BF16→f32 at init).
    /// Indexed by layer_idx * mamba_num_heads. Avoids per-token conversion.
    ssm_a_cache: []f32 = &.{},
    ssm_d_cache: []f32 = &.{},
    ssm_dt_bias_cache: []f32 = &.{},

    // ── Norm weight cache (avoids per-token BF16→f32 conversion) ──
    norm_cache: [max_norm_entries]NormCacheEntry = undefined,
    norm_cache_len: usize = 0,

    // ── Per-layer state ───────────────────────────────────────────
    conv_states: [][]f32 = &.{},
    ssm_states: [][]f32 = &.{},

    // KV cache (PagedAttention or TieredKvCache)
    paged_cache: PagedKvCache = undefined,
    seq_table: SeqBlockTable = undefined,
    block_allocator: BlockAllocator = undefined,
    tiered_cache: ?*TieredKvCache = null,
    tiered_block_allocator: ?TieredBlockAllocator = null,
    kv_type_k: kv_quant.KvQuantType = .f32,
    kv_type_v: kv_quant.KvQuantType = .f32,
    kv_seq_len: usize = 0,
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    /// Enable fused megakernel for single-dispatch forward pass.
    megakernel_enabled: bool = false,

    // ── Lifecycle ─────────────────────────────────────────────────

    /// Initialize the model, reading hyperparameters from SafeTensors config.json
    /// and allocating all working buffers and per-layer state.
    pub fn init(allocator: Allocator, f: Format, be: Backend, ctx_size: u32, kv_type_k: kv_quant.KvQuantType, kv_type_v: kv_quant.KvQuantType, tiered_cache: ?*TieredKvCache) !NemotronNanoModel {
        var self = NemotronNanoModel{ .fmt = f, .be = be, .allocator = allocator };
        self.kv_type_k = kv_type_k;
        self.kv_type_v = kv_type_v;

        // Read hyperparameters from config.json metadata
        if (f.getMetaU32("num_hidden_layers")) |v| self.n_layers = v;
        if (f.getMetaU32("hidden_size")) |v| self.n_embd = v;
        if (f.getMetaU32("num_attention_heads")) |v| self.n_head = v;
        if (f.getMetaU32("num_key_value_heads")) |v| self.n_head_kv = v;
        if (f.getMetaU32("head_dim")) |v| self.head_dim = v;
        if (f.getMetaU32("mamba_num_heads")) |v| self.mamba_num_heads = v;
        if (f.getMetaU32("mamba_head_dim")) |v| self.mamba_head_dim = v;
        if (f.getMetaU32("conv_kernel")) |v| self.ssm_d_conv = v;
        if (f.getMetaU32("ssm_state_size")) |v| self.ssm_d_state = v;
        if (f.getMetaU32("n_groups")) |v| self.ssm_n_groups = v;
        self.moe_intermediate_size = f.getMetaU32("moe_intermediate_size") orelse
            f.getMetaU32("intermediate_size") orelse self.moe_intermediate_size;
        if (f.getMetaU32("moe_shared_expert_intermediate_size")) |v| self.shared_expert_size = v;
        if (f.getMetaU32("num_experts_per_tok")) |v| self.num_experts_per_tok = v;
        if (f.getMetaU32("n_routed_experts")) |v| self.n_routed_experts = v;
        if (f.getMetaU32("vocab_size")) |v| self.vocab_size = v;
        self.rms_eps = f.getMetaF32("norm_eps") orelse f.getMetaF32("layer_norm_epsilon") orelse self.rms_eps;
        if (f.getMetaU32("eos_token_id")) |v| self.eos_token_id = v;
        if (f.getMetaF32("rope_theta")) |v| self.rope_theta = v;
        self.rope_dim = self.head_dim; // partial_rotary_factor = 1.0
        if (f.getMetaF32("routed_scaling_factor")) |v| self.routed_scaling_factor = v;
        self.mlx_bits = f.getMetaU32("bits") orelse default_mlx_bits;
        if (f.getVocab()) |v| self.vocab_size = @intCast(v.len);
        if (f.getMetaU32("max_position_embeddings")) |v| self.max_seq_len = v;
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

        // KV cache: use TieredKvCache if provided, otherwise flat PagedKvCache.
        if (tiered_cache) |tc| {
            var ta = TieredBlockAllocator.init(tc, allocator);
            self.seq_table = try ta.allocateSeqTable(nl);
            errdefer ta.freeSeqTable(&self.seq_table);
            try ta.appendBlock(&self.seq_table);
            self.tiered_cache = tc;
            self.tiered_block_allocator = ta;
        } else {
            // Use full-sequence blocks for contiguous KV access in both prefill and decode.
            const block_size: u16 = @intCast(@min(self.max_seq_len, std.math.maxInt(u16)));
            const num_blocks = nl;
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

        // Pre-compute SSM constants (A = -exp(A_log), D, dt_bias) per layer per head
        // at init to avoid per-token BF16→f32 conversion and -exp() in the hot path.
        const nh: usize = self.mamba_num_heads;
        self.ssm_a_cache = try allocator.alloc(f32, nl * nh);
        errdefer allocator.free(self.ssm_a_cache);
        self.ssm_d_cache = try allocator.alloc(f32, nl * nh);
        errdefer allocator.free(self.ssm_d_cache);
        self.ssm_dt_bias_cache = try allocator.alloc(f32, nl * nh);
        errdefer allocator.free(self.ssm_dt_bias_cache);

        for (0..nl) |i| {
            if (self.layer_types[i] != .ssm) continue;
            const li: u32 = @intCast(i);
            const off = i * nh;
            if (self.stLayerTensor(li, "mixer.A_log")) |t| {
                tensorToF32Buf(t, self.ssm_a_cache[off..][0..nh]);
                for (self.ssm_a_cache[off..][0..nh]) |*v| v.* = -@exp(v.*);
            }
            if (self.stLayerTensor(li, "mixer.D")) |t| {
                tensorToF32Buf(t, self.ssm_d_cache[off..][0..nh]);
            }
            if (self.stLayerTensor(li, "mixer.dt_bias")) |t| {
                tensorToF32Buf(t, self.ssm_dt_bias_cache[off..][0..nh]);
            }
        }

        // Pre-populate norm cache so no BF16→f32 conversions happen during inference.
        self.warmNormCache();

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

        for (self.norm_cache[0..self.norm_cache_len]) |entry| self.allocator.free(entry.data);

        if (self.tiered_block_allocator) |*ta| {
            ta.freeSeqTable(&self.seq_table);
        } else {
            self.block_allocator.freeSeqTable(&self.seq_table);
            self.paged_cache.deinit();
        }

        const bufs = .{
            &self.hidden,         &self.hidden2,      &self.q_buf,
            &self.k_buf,          &self.v_buf,        &self.attn_out,
            &self.scores_buf,     &self.ssm_proj_buf, &self.ssm_conv_out,
            &self.ssm_y_buf,      &self.router_buf,   &self.expert_buf,
            &self.moe_out,        &self.logits_buf,   &self.bf16_buf_large,
            &self.bf16_buf_small, &self.ssm_a_cache,   &self.ssm_d_cache,
            &self.ssm_dt_bias_cache,
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

        try model_mod.ensureKvBlock(self);

        const e: usize = self.n_embd;

        // Embedding lookup (BF16 or MLX quantized)
        const emb = self.fmt.getTensor("backbone.embeddings.weight") orelse return error.MissingTensor;
        if (emb.dtype == .mlx_q) {
            const emb_s = self.fmt.getTensor("backbone.embeddings.scales") orelse return error.MissingTensor;
            const emb_b = self.fmt.getTensor("backbone.embeddings.biases") orelse return error.MissingTensor;
            mlx_ops.mlxEmbLookup(
                self.hidden.ptr,
                @ptrCast(@alignCast(emb.data_ptr)),
                @ptrCast(@alignCast(emb_s.data_ptr)),
                @ptrCast(@alignCast(emb_b.data_ptr)),
                token_id,
                e,
                self.mlx_bits,
            );
        } else {
            self.be.embLookup(
                .{ .data = emb.data_ptr, .dtype = emb.dtype },
                token_id,
                self.hidden.ptr,
                e,
            );
        }

        // Transformer layers
        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.monotonic)) return error.Cancelled;
            self.fmt.prefetchLayer(@intCast(li + 1));
            const l: u32 = @intCast(li);

            switch (self.layer_types[li]) {
                .ssm => try self.ssmLayer(l),
                .moe => try self.moeLayer(l),
                .attention => try self.attentionLayer(l),
            }
        }

        // Final norm (cached BF16→f32) → LM head → argmax
        const nf = self.fmt.getTensor("backbone.norm_f.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nf, e), self.hidden.ptr, e, self.rms_eps);

        const lm = self.fmt.getTensor("lm_head.weight") orelse return error.MissingTensor;
        if (lm.dtype == .mlx_q) {
            const lm_s = self.fmt.getTensor("lm_head.scales") orelse return error.MissingTensor;
            const lm_b = self.fmt.getTensor("lm_head.biases") orelse return error.MissingTensor;
            self.be.gemvMlxQ(self.hidden.ptr, lm.data_ptr, lm_s.data_ptr, lm_b.data_ptr, self.logits_buf.ptr, self.vocab_size, e, self.mlx_bits);
        } else {
            self.be.gemv(self.hidden.ptr, .{ .data = lm.data_ptr, .dtype = lm.dtype }, self.logits_buf.ptr, self.vocab_size, e);
        }

        self.kv_seq_len += 1;
        self.be.sync();

        const result = math_ops.argmax(self.logits_buf);
        return result;
    }

    /// Batched prefill — sequential. Hybrid SSM/MoE/attention pattern
    /// requires sequential SSM state updates. MoE routing is per-token.
    pub fn prefill(self: *NemotronNanoModel, token_ids: []const u32) !u32 {
        var last: u32 = 0;
        for (token_ids) |tid| last = try self.forward(tid);
        return last;
    }

    /// Reset all SSM states, KV cache, and cancellation flag.
    pub fn resetCache(self: *NemotronNanoModel) void {
        for (0..self.n_layers) |i| {
            if (self.conv_states[i].len > 0) @memset(self.conv_states[i], 0);
            if (self.ssm_states[i].len > 0) @memset(self.ssm_states[i], 0);
        }
        model_mod.resetKvCache(self);
    }

    /// Signal an in-progress forward pass to abort.
    pub fn cancel(self: *NemotronNanoModel) void {
        model_mod.signalCancel(&self.cancelled);
    }

    /// Return physical block IDs from layer 0 of the current sequence table.
    /// All layers share the same block IDs, so layer 0 is sufficient.
    pub fn getBlockTable(self: *NemotronNanoModel) []const u32 {
        return self.seq_table.block_table[0];
    }

    // ── Layer implementations ─────────────────────────────────────

    /// Helper: get KV cache byte slices for a layer from the first paged/tiered block.
    fn getLayerKvView(self: *NemotronNanoModel, layer: usize) struct { keys: []u8, values: []u8 } {
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

        // 1. Pre-norm (cached BF16→f32)
        const nw = self.stLayerTensor(li, "norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);

        // 2. Input projection (MLX, NVFP4, or BF16 depending on format)
        const ip_w = self.stLayerTensor(li, "mixer.in_proj.weight") orelse return error.MissingTensor;
        if (ip_w.dtype == .mlx_q) {
            try self.doGemv(self.hidden2.ptr, ip_w, self.ssm_proj_buf.ptr, proj_size, e, li, "mixer.in_proj");
        } else if (self.stLayerTensor(li, "mixer.in_proj.scales")) |ip_s| {
            self.be.gemvNvfp4St(self.hidden2.ptr, ip_w.data_ptr, ip_s.data_ptr, self.ssm_proj_buf.ptr, proj_size, e);
        } else {
            self.be.gemv(self.hidden2.ptr, .{ .data = ip_w.data_ptr, .dtype = ip_w.dtype }, self.ssm_proj_buf.ptr, proj_size, e);
        }
        self.be.sync(); // CPU reads ssm_proj_buf next

        // Split projection: z | conv_in | dt
        const z_ptr = self.ssm_proj_buf.ptr;
        const conv_in_ptr = self.ssm_proj_buf.ptr + d_inner;
        const dt_raw_ptr = self.ssm_proj_buf.ptr + d_inner + conv_ch;

        // 3. Causal conv1d (cached BF16→f32 weights and bias)
        const cw_t = self.stLayerTensor(li, "mixer.conv1d.weight") orelse return error.MissingTensor;
        const cb_t = self.stLayerTensor(li, "mixer.conv1d.bias") orelse return error.MissingTensor;
        ssm_ops.causalConv1dSilu(
            self.ssm_conv_out.ptr,
            self.conv_states[li].ptr,
            conv_in_ptr,
            self.normAsF32(cw_t, conv_ch * d_conv),
            self.normAsF32(cb_t, conv_ch),
            conv_ch,
            d_conv,
        );

        // 4. Split conv output: x | B | C
        const x_ptr = self.ssm_conv_out.ptr;
        const B_ptr = self.ssm_conv_out.ptr + d_inner;
        const C_ptr = self.ssm_conv_out.ptr + d_inner + n_groups * d_state;

        // 5. SSM scalars from init-time cache (no per-token BF16 conversion or -exp)
        const li_off = @as(usize, li) * num_heads;
        const ssm_a = self.ssm_a_cache.ptr + li_off;
        const ssm_d = self.ssm_d_cache.ptr + li_off;
        const dt_bias = self.ssm_dt_bias_cache.ptr + li_off;

        // 6. Mamba-2 autoregressive recurrence
        const state = self.ssm_states[li];
        const y_ptr = self.ssm_y_buf.ptr;

        ssm_ops.mamba2Recurrence(y_ptr, state, x_ptr, B_ptr, C_ptr, dt_raw_ptr, dt_bias, ssm_a, ssm_d, num_heads, mhd, d_state, heads_per_group);

        // 7. Group RMS norm + SiLU gate (cached BF16→f32 norm weight)
        const snw_t = self.stLayerTensor(li, "mixer.norm.weight") orelse return error.MissingTensor;
        ssm_ops.groupRmsNormSiluGate(y_ptr, z_ptr, self.normAsF32(snw_t, d_inner), d_inner, n_groups, self.rms_eps);

        // 8. Output projection (MLX, NVFP4, or BF16 depending on format)
        const op_w = self.stLayerTensor(li, "mixer.out_proj.weight") orelse return error.MissingTensor;
        if (op_w.dtype == .mlx_q) {
            try self.doGemv(y_ptr, op_w, self.hidden2.ptr, e, d_inner, li, "mixer.out_proj");
        } else if (self.stLayerTensor(li, "mixer.out_proj.scales")) |op_s| {
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

        // 1. Pre-norm (cached BF16→f32)
        const nw = self.stLayerTensor(li, "norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);

        // 2. Router: GEMV → sigmoid + bias correction
        const gate_w = self.stLayerTensor(li, "mixer.gate.weight") orelse return error.MissingTensor;
        self.be.sync(); // ensure hidden2 (from rmsNorm) is ready

        try self.doGemv(self.hidden2.ptr, gate_w, self.router_buf.ptr, n_exp, e, li, "mixer.gate");
        self.be.sync();

        const bias_t = self.stLayerTensor(li, "mixer.gate.e_score_correction_bias") orelse return error.MissingTensor;
        // Bias is F32 (not BF16 like other tensors) — read directly
        const bias_ptr: [*]const f32 = @ptrCast(@alignCast(bias_t.data_ptr));

        // Apply sigmoid to all router logits and save raw scores for weighting.
        std.debug.assert(n_exp <= max_routed_experts);
        var raw_sigmoid: [max_routed_experts]f32 = undefined;
        for (0..n_exp) |i| {
            raw_sigmoid[i] = math_ops.sigmoid(self.router_buf[i]);
        }

        // Use sigmoid + bias for top-k SELECTION only (bias shifts selection, not weights).
        // Reference: HF NemotronHMoEGate.forward() applies bias only to get_topk_indices,
        // then gathers raw sigmoid scores for the selected experts.
        for (0..n_exp) |i| {
            self.router_buf[i] = raw_sigmoid[i] + bias_ptr[i];
        }

        // 3. Top-k selection (using bias-corrected scores)
        var top_experts: [max_active_experts]usize = undefined;
        var top_scores_biased: [max_active_experts]f32 = undefined;
        math_ops.topKExperts(self.router_buf[0..n_exp], k_exp, &top_experts, &top_scores_biased);

        // Gather raw sigmoid scores for the selected experts (NO bias, NO float round-trip)
        var top_scores: [max_active_experts]f32 = undefined;
        for (0..k_exp) |i| {
            top_scores[i] = raw_sigmoid[top_experts[i]];
        }

        // Normalize + scale (using raw sigmoid scores)
        var score_sum: f32 = 0;
        for (0..k_exp) |i| score_sum += top_scores[i];
        if (score_sum > 0) {
            for (0..k_exp) |i| top_scores[i] /= score_sum;
        }
        for (0..k_exp) |i| top_scores[i] *= self.routed_scaling_factor;

        // 4. Routed expert computation
        @memset(self.moe_out, 0);
        const fc1_w = self.stLayerTensor(li, "mixer.switch_mlp.fc1.weight") orelse return error.MissingTensor;
        const fc2_w = self.stLayerTensor(li, "mixer.switch_mlp.fc2.weight") orelse return error.MissingTensor;

        for (0..k_exp) |t| {
            const exp_idx = top_experts[t];
            const w = top_scores[t];

            // fc1: up projection [ff, e] → expert_buf [ff]
            try self.doExpertGemv(li, fc1_w, "mixer.switch_mlp.fc1", self.hidden2.ptr, self.expert_buf.ptr, ff, e, exp_idx);
            self.be.sync(); // CPU reads expert_buf next

            // relu²
            math_ops.applyReluSquared(self.expert_buf[0..ff]);

            // fc2: down projection [e, ff] → attn_out [e]
            try self.doExpertGemv(li, fc2_w, "mixer.switch_mlp.fc2", self.expert_buf.ptr, self.attn_out.ptr, e, ff, exp_idx);
            self.be.sync(); // CPU reads attn_out next

            // Accumulate weighted output (SIMD via backend addScaled).
            self.be.addScaled(self.attn_out.ptr, self.moe_out.ptr, w, e);
        }

        // 5. Shared expert
        const sup_w = self.stLayerTensor(li, "mixer.shared_experts.up_proj.weight") orelse return error.MissingTensor;
        try self.doSharedExpertGemv(li, sup_w, "mixer.shared_experts.up_proj", self.hidden2.ptr, self.expert_buf.ptr, shared_ff, e);
        self.be.sync(); // CPU reads expert_buf next

        math_ops.applyReluSquared(self.expert_buf[0..shared_ff]);

        const sdp_w = self.stLayerTensor(li, "mixer.shared_experts.down_proj.weight") orelse return error.MissingTensor;
        try self.doSharedExpertGemv(li, sdp_w, "mixer.shared_experts.down_proj", self.expert_buf.ptr, self.attn_out.ptr, e, shared_ff);
        self.be.sync(); // CPU reads attn_out next

        self.be.addScaled(self.attn_out.ptr, self.moe_out.ptr, 1.0, e);

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

        // 1. Pre-norm (cached BF16→f32)
        const nw = self.stLayerTensor(li, "norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);

        // 2. Q/K/V projections (BF16 or MLX quantized)
        const qw = self.stLayerTensor(li, "mixer.q_proj.weight") orelse return error.MissingTensor;
        const kw = self.stLayerTensor(li, "mixer.k_proj.weight") orelse return error.MissingTensor;
        const vw = self.stLayerTensor(li, "mixer.v_proj.weight") orelse return error.MissingTensor;
        try self.doGemv(self.hidden2.ptr, qw, self.q_buf.ptr, nh * hd, e, li, "mixer.q_proj");
        try self.doGemv(self.hidden2.ptr, kw, self.k_buf.ptr, nkv * hd, e, li, "mixer.k_proj");
        try self.doGemv(self.hidden2.ptr, vw, self.v_buf.ptr, nkv * hd, e, li, "mixer.v_proj");

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
            .f32,
        );

        // 5. Output projection (BF16 or MLX quantized)
        const ow = self.stLayerTensor(li, "mixer.o_proj.weight") orelse return error.MissingTensor;
        try self.doGemv(self.attn_out.ptr, ow, self.hidden2.ptr, e, nh * hd, li, "mixer.o_proj");

        // 6. Residual
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
    }

    // ── Helpers ───────────────────────────────────────────────────

    const NormCacheEntry = model_mod.NormCacheEntry;

    /// Get norm weights as f32 pointer. Caches converted BF16 weights on first access
    /// so subsequent tokens return a stable pointer with zero conversion work.
    fn normAsF32(self: *NemotronNanoModel, t: TensorInfo, n: usize) [*]const f32 {
        if (t.dtype == .f32) return @ptrCast(@alignCast(t.data_ptr));

        const key = @intFromPtr(t.data_ptr);
        for (self.norm_cache[0..self.norm_cache_len]) |entry| {
            if (entry.key == key) return entry.data.ptr;
        }

        // Cache miss: allocate, convert, store permanently.
        if (self.norm_cache_len >= max_norm_entries) {
            // Fallback: convert into bf16_buf_small (no caching)
            bf16ToF32Buf(t.data_ptr, self.bf16_buf_small[0..n]);
            return self.bf16_buf_small.ptr;
        }
        const buf = self.allocator.alloc(f32, n) catch {
            bf16ToF32Buf(t.data_ptr, self.bf16_buf_small[0..n]);
            return self.bf16_buf_small.ptr;
        };
        bf16ToF32Buf(t.data_ptr, buf);
        self.norm_cache[self.norm_cache_len] = .{ .key = key, .data = buf };
        self.norm_cache_len += 1;
        return buf.ptr;
    }

    /// Pre-populate norm cache at init so no allocations happen during inference.
    fn warmNormCache(self: *NemotronNanoModel) void {
        const e: usize = self.n_embd;
        const d_inner: usize = @as(usize, self.mamba_num_heads) * self.mamba_head_dim;
        const n_groups: usize = self.ssm_n_groups;
        const d_state: usize = self.ssm_d_state;
        const conv_ch: usize = d_inner + 2 * n_groups * d_state;
        const d_conv: usize = self.ssm_d_conv;

        // Final norm
        if (self.fmt.getTensor("backbone.norm_f.weight")) |t| _ = self.normAsF32(t, e);

        for (0..self.n_layers) |i| {
            const li: u32 = @intCast(i);
            // Layer norm (all layer types)
            if (self.stLayerTensor(li, "norm.weight")) |t| _ = self.normAsF32(t, e);

            if (self.layer_types[i] == .ssm) {
                // SSM conv1d weights and bias
                if (self.stLayerTensor(li, "mixer.conv1d.weight")) |t| _ = self.normAsF32(t, conv_ch * d_conv);
                if (self.stLayerTensor(li, "mixer.conv1d.bias")) |t| _ = self.normAsF32(t, conv_ch);
                // SSM group norm weight
                if (self.stLayerTensor(li, "mixer.norm.weight")) |t| _ = self.normAsF32(t, d_inner);
            }
        }
    }

    /// Look up a SafeTensors layer tensor by index and suffix.
    /// E.g., stLayerTensor(3, "mixer.q_proj.weight") → "backbone.layers.3.mixer.q_proj.weight"
    fn stLayerTensor(self: *NemotronNanoModel, li: u32, comptime suffix: []const u8) ?TensorInfo {
        var buf: [name_buf_size]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, "backbone.layers.{d}.{s}", .{ li, suffix }) catch return null;
        return self.fmt.getTensor(name);
    }

    /// GEMV dispatch: MLX quantized, NVFP4, or standard be.gemv for BF16/f32.
    /// U32 dtype (`.mlx_q`) is shared by both MLX and NVFP4 packed formats.
    /// MLX has `.scales` + `.biases` companions; NVFP4 has `.scales` only.
    fn doGemv(self: *NemotronNanoModel, x: [*]const f32, w: TensorInfo, y: [*]f32, n: usize, k: usize, li: u32, comptime prefix: []const u8) !void {
        if (w.dtype == .mlx_q) {
            if (self.stLayerTensor(li, prefix ++ ".biases")) |b_t| {
                // MLX path: U32 packed weights + BF16 scales + BF16 biases
                const s_t = self.stLayerTensor(li, prefix ++ ".scales") orelse return error.MissingTensor;
                self.be.gemvMlxQ(x, w.data_ptr, s_t.data_ptr, b_t.data_ptr, y, n, k, self.mlx_bits);
            } else if (self.stLayerTensor(li, prefix ++ ".scales")) |s_t| {
                // NVFP4 path: U32 packed weights + FP8 scales, no biases
                self.be.gemvNvfp4St(x, w.data_ptr, s_t.data_ptr, y, n, k);
            } else {
                self.be.gemv(x, .{ .data = w.data_ptr, .dtype = w.dtype }, y, n, k);
            }
        } else if (self.stLayerTensor(li, prefix ++ ".scales")) |s_t| {
            self.be.gemvNvfp4St(x, w.data_ptr, s_t.data_ptr, y, n, k);
        } else {
            self.be.gemv(x, .{ .data = w.data_ptr, .dtype = w.dtype }, y, n, k);
        }
    }

    /// GEMV dispatch for stacked expert weights (indexed by expert_id).
    /// For NVFP4: uses strides into packed weight/scale tensors.
    /// For MLX: uses strides into packed weight/scale/bias tensors.
    /// For BF16/f32: uses byte stride into weight tensor.
    fn doExpertGemv(
        self: *NemotronNanoModel,
        li: u32,
        w: TensorInfo,
        comptime prefix: []const u8,
        x: [*]const f32,
        y: [*]f32,
        n: usize,
        k: usize,
        exp_idx: usize,
    ) !void {
        if (w.dtype == .mlx_q and self.stLayerTensor(li, prefix ++ ".biases") != null) {
            // MLX path: U32 packed weights + BF16 scales + BF16 biases
            const s_t = self.stLayerTensor(li, prefix ++ ".scales") orelse return error.MissingTensor;
            const b_t = self.stLayerTensor(li, prefix ++ ".biases").?;
            const gs = mlx_group_size;
            const gpr = (k + gs - 1) / gs;
            const wpg = mlx_ops.wordsPerGroup(self.mlx_bits);
            const wpr = gpr * wpg;
            // Byte offsets: weights are u32 words, scales/biases are u16 (bf16)
            const w_byte_offset = exp_idx * n * wpr * @sizeOf(u32);
            const s_byte_offset = exp_idx * n * gpr * @sizeOf(u16);
            self.be.gemvMlxQ(x, w.data_ptr + w_byte_offset, s_t.data_ptr + s_byte_offset, b_t.data_ptr + s_byte_offset, y, n, k, self.mlx_bits);
        } else {
            // NVFP4 path: U32 or non-U32 packed weights + FP8/U8 scales
            const s_t = self.stLayerTensor(li, prefix ++ ".scales") orelse return error.MissingTensor;
            const w_stride = n * (k / nvfp4_values_per_byte);
            const s_stride = n * (k / nvfp4_scale_group_size);
            self.be.gemvNvfp4St(x, w.data_ptr + exp_idx * w_stride, s_t.data_ptr + exp_idx * s_stride, y, n, k);
        }
    }

    /// GEMV dispatch for non-stacked (shared) expert weights.
    /// For NVFP4: uses separate scale tensor. For MLX: uses companion tensors.
    fn doSharedExpertGemv(
        self: *NemotronNanoModel,
        li: u32,
        w: TensorInfo,
        comptime prefix: []const u8,
        x: [*]const f32,
        y: [*]f32,
        n: usize,
        k: usize,
    ) !void {
        if (w.dtype == .mlx_q and self.stLayerTensor(li, prefix ++ ".biases") != null) {
            // MLX path: U32 packed weights + BF16 scales + BF16 biases
            const s_t = self.stLayerTensor(li, prefix ++ ".scales") orelse return error.MissingTensor;
            const b_t = self.stLayerTensor(li, prefix ++ ".biases").?;
            self.be.gemvMlxQ(x, w.data_ptr, s_t.data_ptr, b_t.data_ptr, y, n, k, self.mlx_bits);
        } else {
            // NVFP4 path
            const s_t = self.stLayerTensor(li, prefix ++ ".scales") orelse return error.MissingTensor;
            self.be.gemvNvfp4St(x, w.data_ptr, s_t.data_ptr, y, n, k);
        }
    }
};

// ── Free functions ────────────────────────────────────────────────

/// Convert a BF16 byte array to f32 values in the given output buffer.
fn bf16ToF32Buf(data: [*]const u8, out: []f32) void {
    const u16s: [*]const u16 = @ptrCast(@alignCast(data));
    for (0..out.len) |i| out[i] = quant.bf16ToF32(u16s[i]);
}

/// Convert a tensor's data to f32, dispatching on its dtype.
/// Handles F32 (direct copy) and BF16 (conversion). Other types
/// (e.g. unknown scalars) are treated as BF16 since that's the
/// most common non-f32 scalar type in NemotronH SafeTensors.
fn tensorToF32Buf(t: TensorInfo, out: []f32) void {
    if (t.dtype == .f32) {
        const f32s: [*]const f32 = @ptrCast(@alignCast(t.data_ptr));
        @memcpy(out, f32s[0..out.len]);
    } else {
        bf16ToF32Buf(t.data_ptr, out);
    }
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
