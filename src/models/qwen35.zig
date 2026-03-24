//! Qwen 3.5 hybrid model with DeltaNet SSM, full attention layers, and optional MoE FFN.
//! Alternates between DeltaNet (linear attention with delta rule) and
//! standard GQA layers based on full_attention_interval.

const std = @import("std");
const math = std.math;
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
const TensorData = backend_mod.TensorData;
const Format = format_mod.Format;
const FormatTensorInfo = format_mod.TensorInfo;
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;
const PagedKvCache = kvcache.PagedKvCache;
const SeqBlockTable = kvcache.SeqBlockTable;

/// Maximum DeltaNet V-heads validated at runtime (assert).
const max_ssm_v_heads: usize = 128;

/// Maximum top-k experts for stack-allocated selection arrays (MoE variant).
const max_active_experts: usize = 16;

/// Qwen3.5 hybrid model with DeltaNet SSM, full attention layers, and optional MoE FFN.
pub const Qwen35Model = struct {
    fmt: Format,
    be: Backend,
    allocator: Allocator,

    n_layers: u32 = 32,
    n_embd: u32 = 4096,
    n_head: u32 = 16,
    n_head_kv: u32 = 4,
    head_dim: u32 = 256,
    n_ff: u32 = 12288,
    vocab_size: u32 = 248320,
    rope_theta: f32 = 10000000.0,
    rope_dim: u32 = 64,
    rms_eps: f32 = 1e-6,
    full_attn_interval: u32 = 4,
    eos_token_id: u32 = 248046,
    max_seq_len: usize = 4096,

    ssm_d_conv: u32 = 4,
    ssm_d_state: u32 = 128,
    ssm_n_group: u32 = 16,
    ssm_dt_rank: u32 = 16,
    ssm_d_inner: u32 = 2048,

    // MoE configuration (populated when is_moe == true)
    // Architecture variant detection (Qwen3.5 vs Qwen2/3)
    has_gate: bool = true, // Q projection includes interleaved gate (Qwen3.5 only)
    has_qk_norm: bool = true, // Per-head Q/K RMS norms (Qwen3/3.5 only, not Qwen2)
    has_post_attn_norm: bool = true, // Qwen3.5 fused addRmsNorm; Qwen3 uses separate ffn_norm

    /// True when weights are MLX quantized (SafeTensors U32 packed).
    is_mlx: bool = false,

    is_moe: bool = false,
    n_experts: u32 = 0,
    n_experts_active: u32 = 0,
    expert_ff_dim: u32 = 0,
    shared_expert_ff_dim: u32 = 0,

    hidden: []f32 = &.{},
    hidden2: []f32 = &.{},
    q_buf: []f32 = &.{},
    k_buf: []f32 = &.{},
    v_buf: []f32 = &.{},
    attn_out: []f32 = &.{},
    ff_buf1: []f32 = &.{},
    ff_buf2: []f32 = &.{},
    logits_buf: []f32 = &.{},
    scores_buf: []f32 = &.{},
    ssm_qkv_buf: []f32 = &.{},
    ssm_z_buf: []f32 = &.{},
    ssm_conv_out: []f32 = &.{},
    ssm_alpha_buf: []f32 = &.{},
    ssm_beta_buf: []f32 = &.{},
    dequant_buf: []f32 = &.{}, // scratch for dequantizing non-F32 tensors at runtime (e.g. norm weights via asF32)

    // MoE buffers (allocated only when is_moe == true)
    router_logits: []f32 = &.{},
    moe_out: []f32 = &.{},

    // Pre-dequantized per-DeltaNet-layer constant weights (populated at init, avoids
    // per-token dequant and ensures GPU buffer lifetime for Metal deferred dispatch).
    dn_ssm_a: [][]f32 = &.{}, // [n_delta_layers][num_v_heads]
    dn_dt_bias: [][]f32 = &.{}, // [n_delta_layers][num_v_heads]
    dn_conv_w: [][]f32 = &.{}, // [n_delta_layers][conv_ch * d_conv]
    dn_ssm_norm_w: [][]f32 = &.{}, // [n_delta_layers][head_v_dim]

    // Per-layer state: conv_states[layer] = ring buffer [conv_channels * (d_conv-1)]
    conv_states: [][]f32 = &.{},
    ssm_states: [][]f32 = &.{},

    // KV cache (PagedAttention or TieredKvCache)
    paged_cache: PagedKvCache = undefined,
    seq_table: SeqBlockTable = undefined,
    block_allocator: BlockAllocator = undefined,
    tiered_cache: ?*TieredKvCache = null,
    tiered_block_allocator: ?TieredBlockAllocator = null,
    kv_type: kv_quant.KvQuantType = .f32,
    kv_seq_len: usize = 0,
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    perf: perf.PerfCounters = .{},

    /// Returns the generic Model interface for this Qwen3.5 instance.
    pub fn model(self: *Qwen35Model) Model {
        return Model.from(Qwen35Model, self);
    }

    /// Initialize a Qwen3.5 model from format metadata and weights.
    /// When `tiered_cache` is provided, uses tiered block allocation instead of PagedKvCache.
    pub fn init(allocator: Allocator, f: Format, be: Backend, ctx_size: u32, kv_type: kv_quant.KvQuantType, tiered_cache: ?*TieredKvCache) !Qwen35Model {
        var self = Qwen35Model{ .fmt = f, .be = be, .allocator = allocator };
        self.kv_type = kv_type;
        const arch = f.getMetaStr("general.architecture") orelse "qwen35";
        self.n_layers = f.getArchU32(arch, "block_count") orelse 32;
        self.n_embd = f.getArchU32(arch, "embedding_length") orelse 4096;
        self.n_head = f.getArchU32(arch, "attention.head_count") orelse 16;
        self.n_head_kv = f.getArchU32(arch, "attention.head_count_kv") orelse 4;
        self.head_dim = f.getArchU32(arch, "attention.key_length") orelse 256;
        self.n_ff = f.getArchU32(arch, "feed_forward_length") orelse 12288;
        self.full_attn_interval = f.getArchU32(arch, "full_attention_interval") orelse 4;
        self.ssm_d_conv = f.getArchU32(arch, "ssm.conv_kernel") orelse 4;
        self.ssm_d_state = f.getArchU32(arch, "ssm.state_size") orelse 128;
        self.ssm_n_group = f.getArchU32(arch, "ssm.group_count") orelse 16;
        self.ssm_dt_rank = f.getArchU32(arch, "ssm.time_step_rank") orelse 16;
        self.ssm_d_inner = f.getArchU32(arch, "ssm.inner_size") orelse 2048;
        self.rope_dim = f.getArchU32(arch, "rope.dimension_count") orelse self.head_dim;

        // MoE configuration (e.g., Qwen3.5-35B-A3B uses 256 experts, top-8 + shared expert)
        if (f.getArchU32(arch, "expert_count")) |ec| {
            self.is_moe = true;
            self.n_experts = ec;
            self.n_experts_active = f.getArchU32(arch, "expert_used_count") orelse 8;
            self.expert_ff_dim = f.getArchU32(arch, "expert_feed_forward_length") orelse 512;
            self.shared_expert_ff_dim = f.getArchU32(arch, "expert_shared_feed_forward_length") orelse self.expert_ff_dim;
            // For MoE, n_ff is repurposed as max buffer size (must fit both expert FFN and attention de-interleave)
            self.n_ff = @max(self.expert_ff_dim, self.n_head * self.head_dim);
        }
        if (f.getArchF32(arch, "rope.freq_base")) |v| self.rope_theta = v;
        if (f.getArchF32(arch, "attention.layer_norm_rms_epsilon")) |v| self.rms_eps = v;
        if (f.getMetaU32("tokenizer.ggml.eos_token_id")) |v| self.eos_token_id = v;
        if (f.getVocab()) |v| self.vocab_size = @intCast(v.len);
        if (f.getArchU32(arch, "context_length")) |cl| self.max_seq_len = cl;
        if (ctx_size > 0) self.max_seq_len = ctx_size;

        // Auto-detect architecture variant from weight tensors.
        // Qwen3.5: DeltaNet layers use attn_qkv.weight; full attention Q has gate (2× head_dim).
        // Qwen2/3: Pure attention — no DeltaNet, no gate in Q, possibly no Q/K norms.
        if (self.full_attn_interval > 1) {
            if (f.layerTensor(0, "attn_qkv.weight") == null) {
                self.full_attn_interval = 1; // No DeltaNet tensors — pure attention model
            }
        }

        // Find first full-attention layer to check Q weight dimensions.
        var check_layer: u32 = 0;
        for (0..self.n_layers) |i| {
            if (self.isFullAttn(@intCast(i))) {
                check_layer = @intCast(i);
                break;
            }
        }

        // Detect gate in Q: Qwen3.5 Q weight has n_head * head_dim * 2 output elements.
        if (f.layerTensor(check_layer, "attn_q.weight")) |qw| {
            const q_elems = qw.numElements();
            const expected_gate = @as(usize, self.n_head) * @as(usize, self.head_dim) * 2 * @as(usize, self.n_embd);
            self.has_gate = (q_elems == expected_gate);
        }

        // Detect Q/K per-head norms (present in Qwen3/3.5, absent in Qwen2).
        self.has_qk_norm = f.layerTensor(check_layer, "attn_q_norm.weight") != null;

        // Detect Qwen3.5 vs Qwen3 residual structure.
        // Qwen3.5: "post_attention_norm" (fused addRmsNorm before MLP).
        // Qwen3/2: "ffn_norm" (separate pre-norm, standard residual after attention).
        self.has_post_attn_norm = f.layerTensor(check_layer, "post_attention_norm.weight") != null;

        // Detect MLX quantized weights (SafeTensors U32 packed with companion scale/bias).
        // Check multiple tensor names since some layers might not have attention.
        const mlx_check_names = [_][]const u8{ "attn_q.weight", "ffn_gate.weight", "ffn_up.weight" };
        for (mlx_check_names) |name| {
            for (0..self.n_layers) |li| {
                if (f.layerTensor(@intCast(li), name)) |tw| {
                    if (tw.dtype == .mlx_q) self.is_mlx = true;
                    break;
                }
            }
            if (self.is_mlx) break;
        }

        std.debug.assert(self.n_head % self.n_head_kv == 0);
        std.debug.assert(self.ssm_d_inner % self.ssm_dt_rank == 0);
        std.debug.assert(self.rope_dim <= self.head_dim);
        std.debug.assert(self.rope_dim % 2 == 0);
        if (self.is_moe) {
            std.debug.assert(self.n_experts_active <= self.n_experts);
            std.debug.assert(self.n_experts_active <= max_active_experts);
        }

        const qd: usize = self.n_head * self.head_dim;
        const kvd: usize = self.n_head_kv * self.head_dim;
        const conv_ch = self.ssmConvChannels();
        self.hidden = try allocator.alloc(f32, self.n_embd);
        errdefer allocator.free(self.hidden);
        self.hidden2 = try allocator.alloc(f32, self.n_embd);
        errdefer allocator.free(self.hidden2);
        self.q_buf = try allocator.alloc(f32, if (self.has_gate) qd * 2 else qd);
        errdefer allocator.free(self.q_buf);
        self.k_buf = try allocator.alloc(f32, kvd);
        errdefer allocator.free(self.k_buf);
        self.v_buf = try allocator.alloc(f32, kvd);
        errdefer allocator.free(self.v_buf);
        self.attn_out = try allocator.alloc(f32, @max(qd, self.ssm_d_inner));
        errdefer allocator.free(self.attn_out);
        self.ff_buf1 = try allocator.alloc(f32, self.n_ff);
        errdefer allocator.free(self.ff_buf1);
        self.ff_buf2 = try allocator.alloc(f32, self.n_ff);
        errdefer allocator.free(self.ff_buf2);
        self.logits_buf = try allocator.alloc(f32, self.vocab_size);
        errdefer allocator.free(self.logits_buf);
        self.scores_buf = try allocator.alloc(f32, self.max_seq_len);
        errdefer allocator.free(self.scores_buf);
        self.ssm_qkv_buf = try allocator.alloc(f32, conv_ch);
        errdefer allocator.free(self.ssm_qkv_buf);
        self.ssm_z_buf = try allocator.alloc(f32, self.ssm_d_inner);
        errdefer allocator.free(self.ssm_z_buf);
        self.ssm_conv_out = try allocator.alloc(f32, conv_ch);
        errdefer allocator.free(self.ssm_conv_out);
        self.ssm_alpha_buf = try allocator.alloc(f32, self.ssm_dt_rank);
        errdefer allocator.free(self.ssm_alpha_buf);
        self.ssm_beta_buf = try allocator.alloc(f32, self.ssm_dt_rank);
        errdefer allocator.free(self.ssm_beta_buf);
        // Scratch for dequantizing non-F32 tensors: largest is conv1d weight (d_conv * conv_ch)
        const dequant_size = @max(self.ssm_d_conv * conv_ch, self.n_embd);
        self.dequant_buf = try allocator.alloc(f32, dequant_size);
        errdefer allocator.free(self.dequant_buf);

        // MoE-specific buffers
        if (self.is_moe) {
            self.router_logits = try allocator.alloc(f32, self.n_experts);
            errdefer allocator.free(self.router_logits);
            self.moe_out = try allocator.alloc(f32, self.n_embd);
            errdefer allocator.free(self.moe_out);
        }

        const nl: usize = self.n_layers;
        const num_v_heads: usize = self.ssm_dt_rank;
        const head_v_dim: usize = self.ssm_d_inner / num_v_heads;
        const head_k_dim: usize = self.ssm_d_state;
        const nkv_dim: usize = @as(usize, self.n_head_kv) * @as(usize, self.head_dim);

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
            self.paged_cache = try PagedKvCache.init(allocator, nl, nkv_dim, num_blocks, block_size);
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
            if (!self.isFullAttn(@intCast(i))) {
                // Conv state: (d_conv-1) columns, each of conv_channels
                self.conv_states[i] = try allocator.alloc(f32, (self.ssm_d_conv - 1) * conv_ch);
                @memset(self.conv_states[i], 0);
                // SSM state: per v-head, a [head_v_dim x head_k_dim] matrix
                self.ssm_states[i] = try allocator.alloc(f32, num_v_heads * head_v_dim * head_k_dim);
                @memset(self.ssm_states[i], 0);
            } else {
                self.conv_states[i] = &.{};
                self.ssm_states[i] = &.{};
            }
            layer_init_count = i + 1;
        }

        // Pre-dequantize per-DeltaNet-layer constant weights (avoids per-token dequant
        // and ensures GPU buffer lifetime for Metal deferred dispatch).
        self.dn_ssm_a = try allocator.alloc([]f32, nl);
        errdefer allocator.free(self.dn_ssm_a);
        self.dn_dt_bias = try allocator.alloc([]f32, nl);
        errdefer allocator.free(self.dn_dt_bias);
        self.dn_conv_w = try allocator.alloc([]f32, nl);
        errdefer allocator.free(self.dn_conv_w);
        self.dn_ssm_norm_w = try allocator.alloc([]f32, nl);
        errdefer allocator.free(self.dn_ssm_norm_w);
        @memset(self.dn_ssm_a, &.{});
        @memset(self.dn_dt_bias, &.{});
        @memset(self.dn_conv_w, &.{});
        @memset(self.dn_ssm_norm_w, &.{});

        var dn_init_count: usize = 0;
        errdefer {
            for (0..dn_init_count) |i| {
                if (self.dn_ssm_a[i].len > 0) allocator.free(self.dn_ssm_a[i]);
                if (self.dn_dt_bias[i].len > 0) allocator.free(self.dn_dt_bias[i]);
                if (self.dn_conv_w[i].len > 0) allocator.free(self.dn_conv_w[i]);
                if (self.dn_ssm_norm_w[i].len > 0) allocator.free(self.dn_ssm_norm_w[i]);
            }
        }
        for (0..nl) |i| {
            if (self.isFullAttn(@intCast(i))) {
                dn_init_count = i + 1;
                continue;
            }
            const li: u32 = @intCast(i);
            const ssm_a_t = f.layerTensor(li, "ssm_a") orelse return error.MissingTensor;
            const dt_bias_t = f.layerTensor(li, "ssm_dt.bias") orelse return error.MissingTensor;
            const conv_w_t = f.layerTensor(li, "ssm_conv1d.weight") orelse return error.MissingTensor;
            const ssm_norm_t = f.layerTensor(li, "ssm_norm.weight") orelse return error.MissingTensor;
            self.dn_ssm_a[i] = try allocator.alloc(f32, num_v_heads);
            self.dn_dt_bias[i] = try allocator.alloc(f32, num_v_heads);
            self.dn_conv_w[i] = try allocator.alloc(f32, conv_ch * self.ssm_d_conv);
            self.dn_ssm_norm_w[i] = try allocator.alloc(f32, head_v_dim);
            quant.dequantToF32(self.dn_ssm_a[i], ssm_a_t.data_ptr, ssm_a_t.dtype, num_v_heads);
            quant.dequantToF32(self.dn_dt_bias[i], dt_bias_t.data_ptr, dt_bias_t.dtype, num_v_heads);
            quant.dequantToF32(self.dn_conv_w[i], conv_w_t.data_ptr, conv_w_t.dtype, conv_ch * self.ssm_d_conv);
            quant.dequantToF32(self.dn_ssm_norm_w[i], ssm_norm_t.data_ptr, ssm_norm_t.dtype, head_v_dim);
            dn_init_count = i + 1;
        }

        return self;
    }

    fn ssmConvChannels(self: *const Qwen35Model) usize {
        return self.ssm_d_inner + 2 * @as(usize, self.ssm_n_group) * @as(usize, self.ssm_d_state);
    }

    /// Free all allocated buffers and KV cache.
    pub fn deinit(self: *Qwen35Model) void {
        self.allocator.free(self.hidden);
        self.allocator.free(self.hidden2);
        self.allocator.free(self.q_buf);
        self.allocator.free(self.k_buf);
        self.allocator.free(self.v_buf);
        self.allocator.free(self.attn_out);
        self.allocator.free(self.ff_buf1);
        self.allocator.free(self.ff_buf2);
        self.allocator.free(self.logits_buf);
        self.allocator.free(self.scores_buf);
        self.allocator.free(self.ssm_qkv_buf);
        self.allocator.free(self.ssm_z_buf);
        self.allocator.free(self.ssm_conv_out);
        self.allocator.free(self.ssm_alpha_buf);
        self.allocator.free(self.ssm_beta_buf);
        self.allocator.free(self.dequant_buf);
        if (self.is_moe) {
            self.allocator.free(self.router_logits);
            self.allocator.free(self.moe_out);
        }
        for (0..self.n_layers) |i| {
            if (self.conv_states[i].len > 0) self.allocator.free(self.conv_states[i]);
            if (self.ssm_states[i].len > 0) self.allocator.free(self.ssm_states[i]);
            if (self.dn_ssm_a[i].len > 0) self.allocator.free(self.dn_ssm_a[i]);
            if (self.dn_dt_bias[i].len > 0) self.allocator.free(self.dn_dt_bias[i]);
            if (self.dn_conv_w[i].len > 0) self.allocator.free(self.dn_conv_w[i]);
            if (self.dn_ssm_norm_w[i].len > 0) self.allocator.free(self.dn_ssm_norm_w[i]);
        }
        self.allocator.free(self.conv_states);
        self.allocator.free(self.ssm_states);
        self.allocator.free(self.dn_ssm_a);
        self.allocator.free(self.dn_dt_bias);
        self.allocator.free(self.dn_conv_w);
        self.allocator.free(self.dn_ssm_norm_w);

        if (self.tiered_block_allocator) |*ta| {
            ta.freeSeqTable(&self.seq_table);
        } else {
            self.block_allocator.freeSeqTable(&self.seq_table);
            self.paged_cache.deinit();
        }
    }

    /// Return tensor weight data as [*]const f32.
    /// If the tensor is already F32, returns the raw pointer (zero-copy).
    /// Otherwise, dequantizes into dequant_buf and returns that.
    fn asF32(self: *Qwen35Model, t: FormatTensorInfo, n: usize) [*]const f32 {
        if (t.dtype == .f32) return @ptrCast(@alignCast(t.data_ptr));
        quant.dequantToF32(self.dequant_buf, t.data_ptr, t.dtype, n);
        return self.dequant_buf.ptr;
    }

    fn embLookup(self: *Qwen35Model, tok: u32) !void {
        const t = self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        if (t.dtype == .mlx_q) {
            const mlx_ops = @import("../ops/mlx.zig");
            const st = self.fmt.getTensor("token_embd.scales") orelse return error.MissingTensor;
            const bt = self.fmt.getTensor("token_embd.biases") orelse return error.MissingTensor;
            const bits: u32 = if (st.dtype == .unknown) 4 else (self.fmt.getMetaU32("bits") orelse 4);
            mlx_ops.mlxEmbLookup(self.hidden.ptr, @ptrCast(@alignCast(t.data_ptr)), @ptrCast(@alignCast(st.data_ptr)), @ptrCast(@alignCast(bt.data_ptr)), tok, self.n_embd, bits);
        } else {
            self.be.embLookup(.{ .data = t.data_ptr, .dtype = t.dtype }, tok, self.hidden.ptr, self.n_embd);
        }
    }
    fn isFullAttn(self: *const Qwen35Model, layer: u32) bool {
        if (self.full_attn_interval == 0) return true;
        return ((layer + 1) % self.full_attn_interval) == 0;
    }

    /// Flush GPU work for accurate profiling timestamps.
    fn syncProfile(self: *Qwen35Model) void {
        if (self.perf.enabled) self.be.sync();
    }

    /// Helper: get KV cache byte slices for a layer from the first paged/tiered block.
    fn getLayerKvView(self: *Qwen35Model, layer: usize) struct { keys: []u8, values: []u8 } {
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

    // ---- MLX-aware GEMV dispatch ----

    /// Dispatch GEMV: handles MLX quantized weights via model_mod.dispatchGemv.
    fn doGemv(self: *Qwen35Model, x: [*]const f32, t: FormatTensorInfo, y: [*]f32, n: usize, k: usize) void {
        model_mod.dispatchGemv(self.be, self.fmt, x, t, y, n, k);
    }

    /// Batched GEMV: dispatches 2 or 3 ops. For MLX, uses sequential doGemv
    /// with TensorInfo (needed for companion tensor lookup).
    fn doGemvBatch2(self: *Qwen35Model, x: [*]const f32, t0: FormatTensorInfo, y0: [*]f32, n0: usize, t1: FormatTensorInfo, y1: [*]f32, n1: usize, k: usize) void {
        if (!self.is_mlx) {
            const GemvOp = backend_mod.GemvOp;
            const ops = [_]GemvOp{
                .{ .w = .{ .data = t0.data_ptr, .dtype = t0.dtype }, .y = y0, .n = n0 },
                .{ .w = .{ .data = t1.data_ptr, .dtype = t1.dtype }, .y = y1, .n = n1 },
            };
            self.be.gemvMulti(x, &ops, k);
        } else {
            self.doGemv(x, t0, y0, n0, k);
            self.doGemv(x, t1, y1, n1, k);
        }
    }

    fn doGemvBatch3(self: *Qwen35Model, x: [*]const f32, t0: FormatTensorInfo, y0: [*]f32, n0: usize, t1: FormatTensorInfo, y1: [*]f32, n1: usize, t2: FormatTensorInfo, y2: [*]f32, n2: usize, k: usize) void {
        if (!self.is_mlx) {
            const GemvOp = backend_mod.GemvOp;
            const ops = [_]GemvOp{
                .{ .w = .{ .data = t0.data_ptr, .dtype = t0.dtype }, .y = y0, .n = n0 },
                .{ .w = .{ .data = t1.data_ptr, .dtype = t1.dtype }, .y = y1, .n = n1 },
                .{ .w = .{ .data = t2.data_ptr, .dtype = t2.dtype }, .y = y2, .n = n2 },
            };
            self.be.gemvMulti(x, &ops, k);
        } else {
            self.doGemv(x, t0, y0, n0, k);
            self.doGemv(x, t1, y1, n1, k);
            self.doGemv(x, t2, y2, n2, k);
        }
    }

    /// Dispatch expert slice GEMV for MLX quantized expert tensors.
    fn doGemvExpert(self: *Qwen35Model, x: [*]const f32, exp_t: FormatTensorInfo, ei: usize, stride: usize, y: [*]f32, n: usize, k: usize) void {
        const data = exp_t.data_ptr + ei * stride;
        if (exp_t.dtype != .mlx_q) {
            self.be.gemv(x, .{ .data = data, .dtype = exp_t.dtype }, y, n, k);
            return;
        }
        const wi = std.mem.lastIndexOf(u8, exp_t.name, ".weight") orelse return;
        var sbuf: [128]u8 = undefined;
        const prefix = exp_t.name[0..wi];
        const s_name = std.fmt.bufPrint(&sbuf, "{s}.scales", .{prefix}) catch return;
        const st = self.fmt.getTensor(s_name) orelse return;
        if (st.dtype == .unknown) {
            // MXFP4
            const s_stride = if (st.n_dims >= 3) @as(usize, @intCast(st.dims[0])) * @as(usize, @intCast(st.dims[1])) else st.numElements();
            self.be.gemvMxfp4St(x, data, st.data_ptr + ei * s_stride, y, n, k);
        } else {
            // MLX affine
            var bbuf: [128]u8 = undefined;
            const b_name = std.fmt.bufPrint(&bbuf, "{s}.biases", .{prefix}) catch return;
            const bt = self.fmt.getTensor(b_name) orelse return;
            const s_stride = if (st.n_dims >= 3) @as(usize, @intCast(st.dims[0])) * @as(usize, @intCast(st.dims[1])) * 2 else st.numElements() * 2;
            self.be.gemvMlxQ(x, data, st.data_ptr + ei * s_stride, bt.data_ptr + ei * s_stride, y, n, k, 8);
        }
    }

    // ---- Full attention layer ----
    // Qwen3.5 full attention: Q projection outputs Q+gate interleaved (2*head_dim per head)
    // gate is applied as sigmoid(gate) * attention_output before output projection
    fn fullAttnLayer(self: *Qwen35Model, li: u32) !void {
        const e: usize = self.n_embd;
        const nh: usize = self.n_head;
        const nkv: usize = self.n_head_kv;
        const hd: usize = self.head_dim;
        const qd: usize = nh * hd;

        var t = self.perf.start();
        const nw = self.fmt.layerTensor(li, "attn_norm.weight") orelse return;
        self.be.rmsNorm(self.hidden.ptr, self.asF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        self.syncProfile();
        self.perf.end(.rms_norm, t);

        // Q/K/V projections — Q output size depends on gate presence
        t = self.perf.start();
        const qw = self.fmt.layerTensor(li, "attn_q.weight") orelse return;
        const kw = self.fmt.layerTensor(li, "attn_k.weight") orelse return;
        const vw = self.fmt.layerTensor(li, "attn_v.weight") orelse return;

        const q_out: usize = if (self.has_gate) qd * 2 else qd;
        self.doGemvBatch3(self.hidden2.ptr, qw, self.q_buf.ptr, q_out, kw, self.k_buf.ptr, nkv * hd, vw, self.v_buf.ptr, nkv * hd, e);
        self.syncProfile();
        self.perf.end(.gemv_qkv, t);

        // Q processing: with gate (Qwen3.5) → deinterleave Q+gate; without → use q_buf directly
        const q_ptr: [*]f32 = if (self.has_gate) blk: {
            t = self.perf.start();
            const gate_buf = self.ff_buf1.ptr;
            const q_deint = self.ff_buf2.ptr;
            self.be.deinterleave(self.q_buf.ptr, q_deint, gate_buf, hd, nh);
            self.syncProfile();
            self.perf.end(.deinterleave, t);
            break :blk q_deint;
        } else self.q_buf.ptr;

        // Q/K norms — per-head rmsNorm (Qwen3/3.5 only, absent in Qwen2)
        if (self.has_qk_norm) {
            t = self.perf.start();
            const qnw = self.fmt.layerTensor(li, "attn_q_norm.weight") orelse return;
            const qnd = self.asF32(qnw, hd);
            self.be.rmsNormMulti(q_ptr, qnd, nh, hd, self.rms_eps);
            const knw = self.fmt.layerTensor(li, "attn_k_norm.weight") orelse return;
            const knd = self.asF32(knw, hd);
            self.be.rmsNormMulti(self.k_buf.ptr, knd, nkv, hd, self.rms_eps);
            self.syncProfile();
            self.perf.end(.rms_norm, t);
        }

        // RoPE
        t = self.perf.start();
        self.be.rope(q_ptr, self.kv_seq_len, nh, hd, self.rope_dim, self.rope_theta);
        self.be.rope(self.k_buf.ptr, self.kv_seq_len, nkv, hd, self.rope_dim, self.rope_theta);
        self.syncProfile();
        self.perf.end(.rope, t);

        // SDPA
        t = self.perf.start();
        const kv_view = self.getLayerKvView(li);
        attn_ops.scaledDotProductAttention(
            q_ptr,
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
        self.syncProfile();
        self.perf.end(.sdpa, t);

        // Gate: attn_out *= sigmoid(gate) — Qwen3.5 only
        if (self.has_gate) {
            t = self.perf.start();
            self.be.sigmoidMul(self.attn_out.ptr, self.ff_buf1.ptr, qd);
            self.syncProfile();
            self.perf.end(.sigmoid_mul, t);
        }

        // Output projection
        t = self.perf.start();
        const ow = self.fmt.layerTensor(li, "attn_output.weight") orelse return;
        self.doGemv(self.attn_out.ptr, ow, self.hidden2.ptr, e, qd);
        self.syncProfile();
        self.perf.end(.gemv_out, t);

        // Qwen3/2: standard residual after attention (no fused addRmsNorm in MLP).
        if (!self.has_post_attn_norm) {
            t = self.perf.start();
            self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
            self.syncProfile();
            self.perf.end(.add, t);
        }
    }

    // ---- DeltaNet SSM layer ----
    // Reference: llama.cpp build_layer_attn_linear in src/models/qwen35.cpp
    fn deltaNetLayer(self: *Qwen35Model, li: u32) !void {
        const e: usize = self.n_embd;
        const d_inner: usize = self.ssm_d_inner;
        const num_k_heads: usize = self.ssm_n_group;
        const head_k_dim: usize = self.ssm_d_state;
        const num_v_heads: usize = self.ssm_dt_rank;
        const head_v_dim: usize = d_inner / num_v_heads;
        const conv_ch: usize = self.ssmConvChannels();
        const d_conv: usize = self.ssm_d_conv;

        std.debug.assert(num_v_heads <= max_ssm_v_heads);

        // 1. Attention norm
        var t = self.perf.start();
        const nw = self.fmt.layerTensor(li, "attn_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.asF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        self.syncProfile();
        self.perf.end(.rms_norm, t);

        // 2. Input projections: QKV, gate(z), alpha, beta
        t = self.perf.start();
        const qkv_w = self.fmt.layerTensor(li, "attn_qkv.weight") orelse return error.MissingTensor;
        const gate_w = self.fmt.layerTensor(li, "attn_gate.weight") orelse return error.MissingTensor;
        const alpha_w = self.fmt.layerTensor(li, "ssm_alpha.weight") orelse return error.MissingTensor;
        const beta_w = self.fmt.layerTensor(li, "ssm_beta.weight") orelse return error.MissingTensor;
        if (self.is_mlx) {
            self.doGemv(self.hidden2.ptr, qkv_w, self.ssm_qkv_buf.ptr, conv_ch, e);
            self.doGemv(self.hidden2.ptr, gate_w, self.ssm_z_buf.ptr, d_inner, e);
            self.doGemv(self.hidden2.ptr, alpha_w, self.ssm_alpha_buf.ptr, num_v_heads, e);
            self.doGemv(self.hidden2.ptr, beta_w, self.ssm_beta_buf.ptr, num_v_heads, e);
        } else {
            const GemvOp = backend_mod.GemvOp;
            const delta_ops = [_]GemvOp{
                .{ .w = .{ .data = qkv_w.data_ptr, .dtype = qkv_w.dtype }, .y = self.ssm_qkv_buf.ptr, .n = conv_ch },
                .{ .w = .{ .data = gate_w.data_ptr, .dtype = gate_w.dtype }, .y = self.ssm_z_buf.ptr, .n = d_inner },
                .{ .w = .{ .data = alpha_w.data_ptr, .dtype = alpha_w.dtype }, .y = self.ssm_alpha_buf.ptr, .n = num_v_heads },
                .{ .w = .{ .data = beta_w.data_ptr, .dtype = beta_w.dtype }, .y = self.ssm_beta_buf.ptr, .n = num_v_heads },
            };
            self.be.gemvMulti(self.hidden2.ptr, &delta_ops, e);
        }
        self.syncProfile();
        self.perf.end(.gemv_qkv, t);

        // 3-8. DeltaNet recurrence
        t = self.perf.start();
        const q_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_k_dim)));
        self.be.deltaNet(
            self.ssm_qkv_buf.ptr,
            self.ssm_conv_out.ptr,
            self.ssm_z_buf.ptr,
            self.ssm_alpha_buf.ptr,
            self.ssm_beta_buf.ptr,
            self.attn_out.ptr,
            self.conv_states[li].ptr,
            self.ssm_states[li],
            self.dn_ssm_a[li].ptr,
            self.dn_dt_bias[li].ptr,
            self.dn_conv_w[li].ptr,
            self.dn_ssm_norm_w[li].ptr,
            .{
                .conv_ch = @intCast(conv_ch),
                .d_conv = @intCast(d_conv),
                .d_inner = @intCast(d_inner),
                .num_k_heads = @intCast(num_k_heads),
                .head_k_dim = @intCast(head_k_dim),
                .num_v_heads = @intCast(num_v_heads),
                .head_v_dim = @intCast(head_v_dim),
                .q_scale = q_scale,
                .rms_eps = self.rms_eps,
            },
        );
        self.syncProfile();
        self.perf.end(.deltanet, t);

        // 9. Output projection
        t = self.perf.start();
        const out_w = self.fmt.layerTensor(li, "ssm_out.weight") orelse return error.MissingTensor;
        self.doGemv(self.attn_out.ptr, out_w, self.hidden2.ptr, e, d_inner);
        self.syncProfile();
        self.perf.end(.gemv_out, t);
    }

    /// MLP layer with post-attention norm applied to the residual stream.
    /// Fuses the attention residual add with the post-attention norm into a
    /// single addRmsNorm dispatch (saves one GPU kernel launch per layer).
    fn mlpLayer(self: *Qwen35Model, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.n_ff;

        // Pre-MLP norm: Qwen3.5 fuses residual add + norm (addRmsNorm with post_attention_norm),
        // Qwen3/2 uses standard separate pre-norm (rmsNorm with ffn_norm).
        var t = self.perf.start();
        if (self.has_post_attn_norm) {
            const nw = self.fmt.layerTensor(li, "post_attention_norm.weight") orelse return error.MissingTensor;
            self.be.addRmsNorm(self.hidden.ptr, self.hidden2.ptr, self.asF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        } else {
            const nw = self.fmt.layerTensor(li, "ffn_norm.weight") orelse return error.MissingTensor;
            self.be.rmsNorm(self.hidden.ptr, self.asF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        }
        self.syncProfile();
        self.perf.end(.rms_norm, t);

        // SwiGLU FFN — gate+up projections
        t = self.perf.start();
        const gw = self.fmt.layerTensor(li, "ffn_gate.weight") orelse return error.MissingTensor;
        const uw = self.fmt.layerTensor(li, "ffn_up.weight") orelse return error.MissingTensor;
        self.doGemvBatch2(self.hidden2.ptr, gw, self.ff_buf1.ptr, ff, uw, self.ff_buf2.ptr, ff, e);
        self.syncProfile();
        self.perf.end(.gemv_ffn, t);

        t = self.perf.start();
        self.be.siluMul(self.ff_buf1.ptr, self.ff_buf2.ptr, self.ff_buf1.ptr, ff);
        self.syncProfile();
        self.perf.end(.silu_mul, t);

        // Down projection
        t = self.perf.start();
        const dw = self.fmt.layerTensor(li, "ffn_down.weight") orelse return error.MissingTensor;
        self.doGemv(self.ff_buf1.ptr, dw, self.hidden2.ptr, e, ff);
        self.syncProfile();
        self.perf.end(.gemv_ffn, t);

        // FFN residual
        t = self.perf.start();
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
        self.syncProfile();
        self.perf.end(.add, t);
    }

    /// MoE FFN layer — router + top-K experts + shared expert + residual.
    /// Used by Qwen3.5-35B-A3B and similar MoE variants.
    fn moeLayer(self: *Qwen35Model, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.expert_ff_dim;
        const n_exp: usize = self.n_experts;
        const n_active: usize = self.n_experts_active;

        // Post-attention norm
        var t = self.perf.start();
        const nw = self.fmt.layerTensor(li, "post_attention_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.asF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        self.syncProfile();
        self.perf.end(.rms_norm, t);

        // 1. Router: logits = router_weight @ hidden2
        t = self.perf.start();
        const rw = self.fmt.layerTensor(li, "ffn_gate_inp.weight") orelse return error.MissingTensor;
        self.doGemv(self.hidden2.ptr, rw, self.router_logits.ptr, n_exp, e);
        self.be.sync();
        self.perf.end(.gemv_ffn, t);

        // 2. Softmax over all router logits, then top-K selection.
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
            const inv = 1.0 / sum_e;
            for (0..n_exp) |i| self.router_logits[i] *= inv;
        }

        var top_experts: [max_active_experts]usize = undefined;
        var top_scores: [max_active_experts]f32 = undefined;
        math_ops.topKExperts(self.router_logits[0..n_exp], n_active, top_experts[0..n_active], top_scores[0..n_active]);

        // Renormalize selected weights to sum to 1.0.
        {
            var sel_sum: f32 = 0.0;
            for (0..n_active) |i| sel_sum += top_scores[i];
            const inv = 1.0 / sel_sum;
            for (0..n_active) |i| top_scores[i] *= inv;
        }

        // 4. Fetch packed expert tensor metadata.
        const gate_exps = self.fmt.layerTensor(li, "ffn_gate_exps.weight") orelse return error.MissingTensor;
        const up_exps = self.fmt.layerTensor(li, "ffn_up_exps.weight") orelse return error.MissingTensor;
        const down_exps = self.fmt.layerTensor(li, "ffn_down_exps.weight") orelse return error.MissingTensor;
        const gate_stride = expertWeightStride(gate_exps);
        const up_stride = expertWeightStride(up_exps);
        const down_stride = expertWeightStride(down_exps);

        // 5. Accumulate weighted expert outputs.
        @memset(self.moe_out[0..e], 0);

        for (0..n_active) |ti| {
            const ei = top_experts[ti];
            const mix_weight = top_scores[ti];

            t = self.perf.start();
            // Gate + up projections
            const gate_data = gate_exps.data_ptr + ei * gate_stride;
            if (self.is_mlx and gate_exps.dtype == .mlx_q) {
                self.doGemvExpert(self.hidden2.ptr, gate_exps, ei, gate_stride, self.ff_buf1.ptr, ff, e);
                self.doGemvExpert(self.hidden2.ptr, up_exps, ei, up_stride, self.ff_buf2.ptr, ff, e);
            } else {
                const up_data = up_exps.data_ptr + ei * up_stride;
                const GemvOp = backend_mod.GemvOp;
                const exp_ops = [_]GemvOp{
                    .{ .w = .{ .data = gate_data, .dtype = gate_exps.dtype }, .y = self.ff_buf1.ptr, .n = ff },
                    .{ .w = .{ .data = up_data, .dtype = up_exps.dtype }, .y = self.ff_buf2.ptr, .n = ff },
                };
                self.be.gemvMulti(self.hidden2.ptr, &exp_ops, e);
            }
            self.perf.end(.gemv_ffn, t);

            // SwiGLU: silu(gate) * up — GPU-accelerated, chains with gemvMulti
            t = self.perf.start();
            self.be.siluMul(self.ff_buf1.ptr, self.ff_buf2.ptr, self.ff_buf1.ptr, ff);

            // Down projection → attn_out (reused as scratch, ≥ n_embd)
            const down_data = down_exps.data_ptr + ei * down_stride;
            if (self.is_mlx and down_exps.dtype == .mlx_q) {
                self.doGemvExpert(self.ff_buf1.ptr, down_exps, ei, down_stride, self.attn_out.ptr, e, ff);
            } else {
                self.be.gemv(self.ff_buf1.ptr, .{ .data = down_data, .dtype = down_exps.dtype }, self.attn_out.ptr, e, ff);
            }
            self.be.sync();
            self.perf.end(.gemv_ffn, t);

            // Weighted accumulation
            for (0..e) |i| self.moe_out[i] += mix_weight * self.attn_out[i];
        }

        // 6. Shared expert
        t = self.perf.start();
        const sg = self.fmt.layerTensor(li, "ffn_gate_shexp.weight") orelse return error.MissingTensor;
        const su = self.fmt.layerTensor(li, "ffn_up_shexp.weight") orelse return error.MissingTensor;
        const shared_ff: usize = self.shared_expert_ff_dim;
        self.doGemvBatch2(self.hidden2.ptr, sg, self.ff_buf1.ptr, shared_ff, su, self.ff_buf2.ptr, shared_ff, e);
        self.perf.end(.gemv_ffn, t);

        // SwiGLU for shared expert — GPU-accelerated, chains with gemvMulti
        t = self.perf.start();
        self.be.siluMul(self.ff_buf1.ptr, self.ff_buf2.ptr, self.ff_buf1.ptr, shared_ff);

        const sd = self.fmt.layerTensor(li, "ffn_down_shexp.weight") orelse return error.MissingTensor;
        self.doGemv(self.ff_buf1.ptr, sd, self.attn_out.ptr, e, shared_ff);
        self.be.sync();
        self.perf.end(.gemv_ffn, t);

        // Shared expert gate: sigmoid(dot(gate_weight, hidden2)) * shared_out
        if (self.fmt.layerTensor(li, "ffn_gate_inp_shexp.weight")) |gw| {
            const gate_ptr: [*]const f32 = @ptrCast(@alignCast(gw.data_ptr));
            var dot: f32 = 0.0;
            for (0..e) |i| dot += gate_ptr[i] * self.hidden2[i];
            const gate_val = math_ops.sigmoid(dot);
            for (0..e) |i| self.moe_out[i] += gate_val * self.attn_out[i];
        } else {
            for (0..e) |i| self.moe_out[i] += self.attn_out[i];
        }

        // 7. Residual: hidden += moe_out
        t = self.perf.start();
        self.be.add(self.hidden.ptr, self.moe_out.ptr, self.hidden.ptr, e);
        self.syncProfile();
        self.perf.end(.add, t);
    }

    /// Signal cancellation of the current forward pass (thread-safe).
    pub fn cancel(self: *Qwen35Model) void {
        model_mod.signalCancel(&self.cancelled);
    }

    /// Return physical block IDs from layer 0 of the current sequence table.
    /// All layers share the same block IDs, so layer 0 is sufficient.
    pub fn getBlockTable(self: *Qwen35Model) []const u32 {
        return self.seq_table.block_table[0];
    }

    /// Run one token through the model, returning the argmax next token ID.
    pub fn forward(self: *Qwen35Model, token_id: u32) !u32 {
        if (self.kv_seq_len >= self.max_seq_len) return error.KVCacheFull;

        try model_mod.ensureKvBlock(self);

        const t = self.perf.start();
        try self.embLookup(token_id);
        self.syncProfile();
        self.perf.end(.emb_lookup, t);

        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.acquire)) return error.Cancelled;
            const l: u32 = @intCast(li);
            if (self.isFullAttn(l)) try self.fullAttnLayer(l) else try self.deltaNetLayer(l);
            if (self.is_moe) try self.moeLayer(l) else try self.mlpLayer(l);
        }

        const nw = self.fmt.getTensor("output_norm.weight") orelse return error.MissingTensor;
        const ow = self.fmt.getTensor("output.weight") orelse self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        self.kv_seq_len += 1;
        self.perf.addToken();
        const norm_ptr: [*]const u8 = if (nw.dtype == .f32)
            nw.data_ptr
        else blk: {
            quant.dequantToF32(self.dequant_buf, nw.data_ptr, nw.dtype, self.n_embd);
            break :blk @ptrCast(self.dequant_buf.ptr);
        };
        if (ow.dtype == .mlx_q) {
            // MLX output weight: inline RMSNorm + doGemv + argmax
            self.be.rmsNorm(self.hidden.ptr, @ptrCast(@alignCast(norm_ptr)), self.hidden.ptr, self.n_embd, self.rms_eps);
            self.be.sync();
            self.doGemv(self.hidden.ptr, ow, self.logits_buf.ptr, self.vocab_size, self.n_embd);
            self.be.sync();
            return math_ops.argmax(self.logits_buf);
        }
        return math_ops.finalLogits(
            self.hidden.ptr,
            norm_ptr,
            .{ .data = ow.data_ptr, .dtype = ow.dtype },
            self.logits_buf,
            self.vocab_size,
            self.n_embd,
            self.rms_eps,
            self.be,
        );
    }

    /// Reset all KV cache and SSM state for a new conversation.
    pub fn resetCache(self: *Qwen35Model) void {
        for (0..self.n_layers) |i| {
            if (self.conv_states[i].len > 0) @memset(self.conv_states[i], 0);
            if (self.ssm_states[i].len > 0) @memset(self.ssm_states[i], 0);
        }
        model_mod.resetKvCache(self);
    }
};

const expertWeightStride = model_mod.expertWeightStride;
