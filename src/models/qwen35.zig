//! Qwen 3.5 hybrid model with DeltaNet SSM, full attention layers, and optional MoE FFN.
//! Alternates between DeltaNet (linear attention with delta rule) and
//! standard GQA layers based on full_attention_interval.

const std = @import("std");
const builtin = @import("builtin");
const math = std.math;
const backend_mod = @import("../backend/backend.zig");
const format_mod = @import("../format/format.zig");
const model_mod = @import("model.zig");
const math_ops = @import("../ops/math.zig");
const attn_ops = @import("../ops/attention.zig");
const quant = @import("../ops/quant.zig");
const mlx_ops = @import("../ops/mlx.zig");
const perf = @import("../perf.zig");
const kv_quant = @import("../ops/kv_quant.zig");
const kvcache = @import("../kvcache/manager.zig");
const block_alloc_mod = @import("../kvcache/block_allocator.zig");
const BlockAllocator = block_alloc_mod.BlockAllocator;
const TieredBlockAllocator = block_alloc_mod.TieredBlockAllocator;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;
const TransportMod = @import("../parallel/transport.zig");
const Backend = backend_mod.Backend;
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;
const PagedKvCache = kvcache.PagedKvCache;
const SeqBlockTable = kvcache.SeqBlockTable;

/// Maximum DeltaNet V-heads validated at runtime (assert).
const max_ssm_v_heads: usize = 128;

/// Maximum top-k experts for stack-allocated selection arrays (MoE variant).
const max_active_experts: usize = 16;

/// Default top-k experts for Qwen3.5 MoE (35B-A3B variant).
const default_moe_experts_active: u32 = 8;
/// Default per-expert FFN dimension for Qwen3.5 MoE.
const default_moe_expert_ff_dim: u32 = 512;
/// Default MLX quantization bit width (4-bit). Canonical source: model.zig.
const default_mlx_bits = model_mod.default_mlx_bits;

/// Qwen3.5 hybrid model with DeltaNet SSM, full attention layers, and optional MoE FFN.
pub const Qwen35Model = struct {
    /// Norm weight cache: permanently dequantized BF16 norm weights keyed by data pointer.
    /// Avoids reusing dequant_buf for GPU ops (Metal buf_cache would serve stale data).
    const max_norm_entries: usize = 256;
    const NormCacheEntry = model_mod.NormCacheEntry;

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
    has_attn_bias: bool = false, // Q/K/V bias (Qwen2/2.5 only, not Qwen3/3.5)
    has_full_attn_output_gate: bool = false, // Full attention output gate (Nex-N2-Pro: attn_output_gate)

    // Tensor parallelism
    tp_rank: u32 = 0,
    tp_degree: u32 = 1,
    tp_peer_buf: ?[*]const f32 = null, // peer rank's partial output for all-reduce
    tp_row_shard_buf: []u8 = &.{}, // scratch for row-split weight column extraction
    tp_transport: ?*TransportMod.Transport = null, // network transport for distributed TP
    // Pipeline parallelism
    pp_rank: u32 = 0,
    pp_degree: u32 = 1,
    pp_transport: ?*TransportMod.Transport = null,
    tp_kv_cache_rank1: ?PagedKvCache = null, // second KV cache for TP rank 1
    tp_seq_table_rank1: ?kvcache.SeqBlockTable = null,

    /// True when weights are MLX quantized (SafeTensors U32 packed).
    is_mlx: bool = false,
    /// True when loaded from SafeTensors (HF conventions for Q/K/V split, GQA, A_log).
    /// Set from Format.is_safetensors during init.
    is_safetensors: bool = false,

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
    dequant_buf: []f32 = &.{}, // scratch for dequantizing non-F32 tensors (CPU-only, not GPU-safe)
    norm_cache: [max_norm_entries]NormCacheEntry = undefined,
    norm_cache_len: usize = 0,

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
    /// Per-layer DeltaNet flag: true if this layer is a DeltaNet SSM layer.
    /// Populated during init; uses tensor presence for MTP-aware detection.
    layer_is_deltanet: []bool = &.{},

    // KV cache (PagedAttention or TieredKvCache)
    paged_cache: PagedKvCache = undefined,
    seq_table: SeqBlockTable = undefined,
    block_allocator: BlockAllocator = undefined,
    tiered_cache: ?*TieredKvCache = null,
    tiered_block_allocator: ?TieredBlockAllocator = null,
    kv_type_k: kv_quant.KvQuantType = .f32,
    kv_type_v: kv_quant.KvQuantType = .f32,
    /// Number of boundary layers (first/last N) that use f16 V to protect attention quality.
    kv_boundary_v: u32 = 0,
    kv_seq_len: usize = 0,
    layer_skip_start: u32 = 0,
    layer_skip_end: u32 = 0,
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    perf: perf.PerfCounters = .{},

    // ── Visual token embeddings (multimodal) ───────────────────
    /// Visual token embeddings from vision encoder, or null if text-only.
    image_embeddings: ?[]const f32 = null,
    /// Number of visual tokens.
    n_visual_tokens: u32 = 0,
    /// Image pad token ID — the placeholder token that gets replaced with
    /// visual embeddings during forward(). Set by setImageEmbeddings().
    image_pad_token_id: u32 = 0,
    /// Index into image_embeddings for the next visual token injection.
    /// Incremented each time a pad token is encountered during forward().
    visual_token_idx: u32 = 0,

    /// Enable fused megakernel for single-dispatch forward pass.
    megakernel_enabled: bool = false,

    // ── MTP (Multi-Token Prediction) ──────────────────────────
    n_mtp_layers: u32 = 0,
    mtp_hidden_pre_norm: []f32 = &.{},
    mtp_concat_buf: []f32 = &.{},
    mtp_logits_buf: []f32 = &.{},
    mtp_kv_keys: []u8 = &.{},
    mtp_kv_values: []u8 = &.{},
    mtp_kv_seq_len: usize = 0,

    /// Returns the generic Model interface for this Qwen3.5 instance.
    pub fn model(self: *Qwen35Model) Model {
        return Model.from(Qwen35Model, self);
    }

    /// Initialize a Qwen3.5 model from format metadata and weights.
    /// When `tiered_cache` is provided, uses tiered block allocation instead of PagedKvCache.
    pub fn init(allocator: Allocator, f: Format, be: Backend, ctx_size: u32, kv_type_k: kv_quant.KvQuantType, kv_type_v: kv_quant.KvQuantType, tiered_cache: ?*TieredKvCache) !Qwen35Model {
        var self = Qwen35Model{ .fmt = f, .be = be, .allocator = allocator, .is_safetensors = f.is_safetensors };
        self.kv_type_k = kv_type_k;
        self.kv_type_v = kv_type_v;
        const arch = f.getMetaStr("general.architecture") orelse "qwen35";
        if (f.getArchU32(arch, "block_count")) |v| self.n_layers = v;
        if (f.getArchU32(arch, "embedding_length")) |v| self.n_embd = v;
        if (f.getArchU32(arch, "attention.head_count")) |v| self.n_head = v;
        if (f.getArchU32(arch, "attention.head_count_kv")) |v| self.n_head_kv = v;
        if (f.getArchU32(arch, "attention.key_length")) |v| {
            self.head_dim = v;
        } else if (self.n_embd > 0 and self.n_head > 0) {
            self.head_dim = self.n_embd / self.n_head;
        }
        if (f.getArchU32(arch, "feed_forward_length")) |v| {
            self.n_ff = v;
        } else if (f.layerTensor(0, "ffn_up.weight")) |t| {
            // Infer from tensor shape when metadata is missing
            self.n_ff = @intCast(t.dims[0]);
        } else {
            // Pure MoE (no dense FFN) — use shared expert dim or 0
            self.n_ff = 0;
        }
        if (f.getArchU32(arch, "full_attention_interval")) |v| self.full_attn_interval = v;
        if (f.getArchU32(arch, "ssm.conv_kernel")) |v| self.ssm_d_conv = v;
        if (f.getArchU32(arch, "ssm.state_size")) |v| self.ssm_d_state = v;
        if (f.getArchU32(arch, "ssm.group_count")) |v| self.ssm_n_group = v;
        if (f.getArchU32(arch, "ssm.time_step_rank")) |v| self.ssm_dt_rank = v;
        self.ssm_d_inner = f.getArchU32(arch, "ssm.inner_size") orelse blk: {
            // SafeTensors: compute from linear_num_value_heads × linear_value_head_dim.
            if (f.getMetaU32("linear_value_head_dim")) |vhd| {
                break :blk self.ssm_dt_rank * vhd;
            }
            break :blk self.ssm_dt_rank * self.ssm_d_state;
        };
        self.rope_dim = f.getArchU32(arch, "rope.dimension_count") orelse blk: {
            // SafeTensors: compute from head_dim × partial_rotary_factor
            if (f.getArchF32(arch, "partial_rotary_factor")) |prf| {
                break :blk @intFromFloat(@as(f32, @floatFromInt(self.head_dim)) * prf);
            }
            break :blk self.head_dim;
        };

        // MoE configuration (e.g., Qwen3.5-35B-A3B: 256 experts; Nex-N2-Pro: 512 experts).
        // Infer expert count from metadata or from tensor dimensions when metadata is absent.
        const expert_count_meta = f.getArchU32(arch, "expert_count") orelse blk: {
            if (f.layerTensor(0, "ffn_gate_exps.weight")) |t| {
                // Tensor shape for stacked expert weights: dims[0] = n_experts (outermost).
                const n_from_tensor: u32 = if (t.n_dims >= 1) @intCast(t.dims[0]) else 256;
                break :blk n_from_tensor;
            }
            break :blk null;
        };
        if (expert_count_meta) |ec| {
            self.is_moe = true;
            self.n_experts = ec;
            self.n_experts_active = f.getArchU32(arch, "expert_used_count") orelse default_moe_experts_active;
            self.expert_ff_dim = f.getArchU32(arch, "expert_feed_forward_length") orelse default_moe_expert_ff_dim;
            self.shared_expert_ff_dim = f.getArchU32(arch, "expert_shared_feed_forward_length") orelse self.expert_ff_dim;
            // For MoE, n_ff is repurposed as max buffer size (must fit both expert FFN and attention de-interleave)
            self.n_ff = @max(self.expert_ff_dim, self.n_head * self.head_dim);
        }
        if (f.getArchF32(arch, "rope.freq_base")) |v| self.rope_theta = v;
        if (f.getArchF32(arch, "attention.layer_norm_rms_epsilon")) |v| self.rms_eps = v;

        if (f.getMetaU32("tokenizer.ggml.eos_token_id")) |v| self.eos_token_id = v;
        if (f.getVocab()) |v| self.vocab_size = @intCast(v.len);
        // Config.json vocab_size overrides tokenizer count (may include special tokens)
        if (f.getArchU32(arch, "vocab_size")) |vs| {
            if (vs > self.vocab_size) self.vocab_size = vs;
        } else if (f.getMetaU32("vocab_size")) |vs| {
            if (vs > self.vocab_size) self.vocab_size = vs;
        }
        // SafeTensors: weight tensor rows may exceed both — use the largest
        for ([_][]const u8{ "token_embd.weight", "output.weight" }) |tname| {
            if (f.getTensor(tname)) |t| {
                const rows: u32 = @intCast(t.dims[0]);
                if (rows > self.vocab_size) self.vocab_size = rows;
            }
        }
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

        // Detect gate in Q: Qwen3.5 Q weight output dim = n_head * head_dim * 2.
        // When attn_output_gate is true AND Q dim / n_head == head_dim (not 2*head_dim),
        // the Q projection has NO embedded gate — the output gate is applied separately.
        // (Nex-N2-Pro uses output-only gate; Qwen3.5 standard embeds gate in Q proj.)
        if (f.layerTensor(check_layer, "attn_q.weight")) |qw| {
            const q_out_dim: usize = if (qw.n_dims >= 1) @intCast(qw.dims[0]) else 0;
            const expected_gate = @as(usize, self.n_head) * @as(usize, self.head_dim) * 2;
            const expected_no_gate = @as(usize, self.n_head) * @as(usize, self.head_dim);
            if (self.has_full_attn_output_gate and q_out_dim == expected_no_gate) {
                // attn_output_gate model where gate is NOT embedded in Q proj.
                // Q heads use head_dim directly; output gate applied separately.
                self.has_gate = false;
            } else {
                self.has_gate = (q_out_dim == expected_gate);
            }
        }

        // Detect Q/K per-head norms (present in Qwen3/3.5, absent in Qwen2).
        self.has_qk_norm = f.layerTensor(check_layer, "attn_q_norm.weight") != null;

        // Detect attention Q/K/V biases (present in Qwen2/2.5, absent in Qwen3/3.5).
        self.has_attn_bias = f.layerTensor(check_layer, "attn_q.bias") != null;

        // Detect Qwen3.5 vs Qwen3 residual structure.
        // Qwen3.5: "post_attention_norm" (fused addRmsNorm before MLP).
        // Qwen3/2: "ffn_norm" (separate pre-norm, standard residual after attention).
        self.has_post_attn_norm = f.layerTensor(check_layer, "post_attention_norm.weight") != null;

        // Detect full-attention output gate (attn_output_gate: true in Nex-N2-Pro config).
        // The full attention output is gated: attn_out *= sigmoid(gate(hidden2))
        // Use metadata flag or tensor presence as fallback.
        self.has_full_attn_output_gate = blk: {
            // Check config metadata first
            if (f.getMetaU32("attn_output_gate")) |v| break :blk v != 0;
            // Fallback: check for output gate weight tensor in a full attention layer
            if (f.layerTensor(check_layer, "attn_output_gate.weight")) |_| break :blk true;
            break :blk false;
        };

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

        // MTP detection: two GGUF layouts exist.
        // Layout A (block_count excludes MTP): nextn head at blk.{n_layers}
        // Layout B (block_count includes MTP): nextn head at blk.{n_layers-1}, shared transformer
        //   also counted in block_count. Subtract nextn_predict_layers from n_layers so the
        //   regular forward pass runs blk.0..n_layers-1 and mtpForward uses mtp_lid = n_layers+depth.
        const nextn_at_n = f.layerTensor(self.n_layers, "nextn.eh_proj") != null or
            f.layerTensor(self.n_layers, "nextn.eh_proj.weight") != null;
        const nextn_at_n1 = f.layerTensor(self.n_layers - 1, "nextn.eh_proj") != null or
            f.layerTensor(self.n_layers - 1, "nextn.eh_proj.weight") != null;
        if (nextn_at_n) {
            self.n_mtp_layers = f.getArchU32(arch, "nextn_predict_layers") orelse 1;
        } else if (nextn_at_n1) {
            // block_count includes MTP heads: last block is nextn head, adjust n_layers down
            const nc = f.getArchU32(arch, "nextn_predict_layers") orelse 1;
            self.n_mtp_layers = nc;
            self.n_layers -= nc;
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

        // MTP buffers: flat KV cache for single transformer layer
        if (self.n_mtp_layers > 0) {
            self.mtp_hidden_pre_norm = try allocator.alloc(f32, self.n_embd);
            errdefer allocator.free(self.mtp_hidden_pre_norm);
            self.mtp_concat_buf = try allocator.alloc(f32, self.n_embd * 2);
            errdefer allocator.free(self.mtp_concat_buf);
            self.mtp_logits_buf = try allocator.alloc(f32, self.vocab_size);
            errdefer allocator.free(self.mtp_logits_buf);
            const kvd_bytes = @as(usize, self.n_head_kv) * @as(usize, self.head_dim) * @sizeOf(f32);
            self.mtp_kv_keys = try allocator.alloc(u8, self.max_seq_len * kvd_bytes);
            errdefer allocator.free(self.mtp_kv_keys);
            self.mtp_kv_values = try allocator.alloc(u8, self.max_seq_len * kvd_bytes);
            errdefer allocator.free(self.mtp_kv_values);
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
            // Paged KV cache: small fixed-size blocks allocated on demand.
            // Memory scales with actual sequence length, not max context.
            const paged_block_size: u16 = 256;
            const blocks_per_layer = (self.max_seq_len + paged_block_size - 1) / paged_block_size;
            const num_blocks = nl * blocks_per_layer;
            const block_size = paged_block_size;
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
        self.layer_is_deltanet = try allocator.alloc(bool, nl);
        errdefer allocator.free(self.layer_is_deltanet);
        // Pre-populate from tensor presence: a layer is DeltaNet if it has attn_qkv.weight.
        // This correctly handles irregular patterns (layer_types arrays) and MTP boundary layers.
        for (0..nl) |i| {
            self.layer_is_deltanet[i] = f.layerTensor(@intCast(i), "attn_qkv.weight") != null;
        }

        var layer_init_count: usize = 0;
        errdefer {
            for (0..layer_init_count) |i| {
                if (self.conv_states[i].len > 0) allocator.free(self.conv_states[i]);
                if (self.ssm_states[i].len > 0) allocator.free(self.ssm_states[i]);
            }
        }
        for (0..nl) |i| {
            if (self.layer_is_deltanet[i]) {
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
            // If SSM tensors don't exist for this layer, promote to full attention.
            // Handles models where block_count includes MTP heads using standard attention
            // tensor names rather than nextn.* (e.g. Qwopus3.5-9B-Coder-MTP).
            const ssm_a_t = f.layerTensor(li, "ssm_a") orelse {
                self.layer_is_deltanet[i] = false; // Treat as full attention
                dn_init_count = i + 1;
                continue;
            };
            const dt_bias_t = f.layerTensor(li, "ssm_dt.bias") orelse return error.MissingTensor;
            const conv_w_t = f.layerTensor(li, "ssm_conv1d.weight") orelse return error.MissingTensor;
            const ssm_norm_t = f.layerTensor(li, "ssm_norm.weight") orelse return error.MissingTensor;
            // Set init_count early so the outer errdefer covers partial allocations
            // at index i (the .len > 0 guards prevent freeing unallocated empty slices).
            dn_init_count = i + 1;
            self.dn_ssm_a[i] = try allocator.alloc(f32, num_v_heads);
            self.dn_dt_bias[i] = try allocator.alloc(f32, num_v_heads);
            self.dn_conv_w[i] = try allocator.alloc(f32, conv_ch * self.ssm_d_conv);
            self.dn_ssm_norm_w[i] = try allocator.alloc(f32, head_v_dim);
            quant.dequantToF32(self.dn_ssm_a[i], ssm_a_t.data_ptr, ssm_a_t.dtype, num_v_heads);
            // SafeTensors stores raw A_log; GGUF stores -exp(A_log) (pre-converted by llama.cpp).
            // Our kernel expects -exp(A_log), so convert SafeTensors values here.
            if (self.is_safetensors) {
                for (self.dn_ssm_a[i]) |*v| v.* = -@exp(v.*);
            }
            quant.dequantToF32(self.dn_dt_bias[i], dt_bias_t.data_ptr, dt_bias_t.dtype, num_v_heads);
            quant.dequantToF32(self.dn_conv_w[i], conv_w_t.data_ptr, conv_w_t.dtype, conv_ch * self.ssm_d_conv);
            quant.dequantToF32(self.dn_ssm_norm_w[i], ssm_norm_t.data_ptr, ssm_norm_t.dtype, head_v_dim);
        }

        // Pre-populate norm cache so no allocations happen during inference.
        self.warmNormCache();

        return self;
    }

    fn ssmConvChannels(self: *const Qwen35Model) usize {
        return self.ssm_d_inner + 2 * @as(usize, self.ssm_n_group) * @as(usize, self.ssm_d_state);
    }

    /// Free all allocated buffers and KV cache.
    pub fn deinit(self: *Qwen35Model) void {
        for (self.norm_cache[0..self.norm_cache_len]) |entry| self.allocator.free(entry.data);
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
        if (self.n_mtp_layers > 0) {
            self.allocator.free(self.mtp_hidden_pre_norm);
            self.allocator.free(self.mtp_concat_buf);
            self.allocator.free(self.mtp_logits_buf);
            self.allocator.free(self.mtp_kv_keys);
            self.allocator.free(self.mtp_kv_values);
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
        if (self.layer_is_deltanet.len > 0) self.allocator.free(self.layer_is_deltanet);
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

    /// Pre-populate the norm weight cache during init so no allocations occur
    /// in the hot path. Iterates all norm tensors and triggers conversion.
    fn warmNormCache(self: *Qwen35Model) void {
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
        }
    }

    /// Return tensor weight data as [*]const f32 with permanent caching.
    /// F32 tensors: returns raw pointer (zero-copy).
    /// BF16 tensors: dequantizes once, caches permanently (GPU-safe — avoids
    /// stale Metal buf_cache entries from reused dequant_buf).
    fn normAsF32(self: *Qwen35Model, t: TensorInfo, n: usize) [*]const f32 {
        if (t.dtype == .f32) return @ptrCast(@alignCast(t.data_ptr));

        // Check cache (linear scan — at most ~200 entries, first-token only on miss)
        const key = @intFromPtr(t.data_ptr);
        for (self.norm_cache[0..self.norm_cache_len]) |entry| {
            if (entry.key == key) return entry.data.ptr;
        }

        // Cache miss: allocate, convert, store permanently.
        // Guard capacity before allocating to avoid leaking uncached buffers.
        if (self.norm_cache_len >= max_norm_entries) {
            quant.dequantToF32(self.dequant_buf, t.data_ptr, t.dtype, n);
            return self.dequant_buf.ptr;
        }
        const buf = self.allocator.alloc(f32, n) catch {
            // Fallback to dequant_buf (CPU-only, not GPU-safe)
            quant.dequantToF32(self.dequant_buf, t.data_ptr, t.dtype, n);
            return self.dequant_buf.ptr;
        };
        quant.dequantToF32(buf, t.data_ptr, t.dtype, n);
        self.norm_cache[self.norm_cache_len] = .{ .key = key, .data = buf };
        self.norm_cache_len += 1;
        return buf.ptr;
    }

    fn embLookup(self: *Qwen35Model, tok: u32) !void {
        const t = self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        if (t.dtype == .mlx_q) {
            const st = self.fmt.getTensor("token_embd.scales") orelse return error.MissingTensor;
            const bt = self.fmt.getTensor("token_embd.biases") orelse return error.MissingTensor;
            const bits: u32 = if (st.dtype == .unknown) default_mlx_bits else (self.fmt.getMetaU32("bits") orelse default_mlx_bits);
            mlx_ops.mlxEmbLookup(self.hidden.ptr, @ptrCast(@alignCast(t.data_ptr)), @ptrCast(@alignCast(st.data_ptr)), @ptrCast(@alignCast(bt.data_ptr)), tok, self.n_embd, bits);
        } else {
            self.be.embLookup(.{ .data = t.data_ptr, .dtype = t.dtype }, tok, self.hidden.ptr, self.n_embd);
        }
    }
    fn isFullAttn(self: *const Qwen35Model, layer: u32) bool {
        if (self.full_attn_interval == 0) return true;
        return ((layer + 1) % self.full_attn_interval) == 0;
    }

    /// First/last `kv_boundary_v` layers use f16 to protect attention quality;
    /// middle layers use the configured kv_type_v (which may be turbo4/turbo3/etc).
    inline fn layerVType(self: *const Qwen35Model, li: u32) kv_quant.KvQuantType {
        if (self.kv_boundary_v == 0) return self.kv_type_v;
        const b = self.kv_boundary_v;
        if (li < b or li >= self.n_layers - b) return .f16;
        return self.kv_type_v;
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

    const PagedKvView = kvcache.PagedKvView;

    fn getPagedKvView(self: *Qwen35Model, layer: usize) PagedKvView {
        return PagedKvView.initView(
            self.seq_table.block_table[layer],
            self.paged_cache.blocks,
            self.paged_cache.block_size,
            self.paged_cache.kv_dim,
            self.kv_seq_len,
        );
    }

    fn isMultiBlock(self: *Qwen35Model, layer: usize) bool {
        // Tiered KV uses its own block management; paged_cache.block_size=0 when tiered.
        return self.paged_cache.block_size > 0 and self.seq_table.block_table[layer].len > 1;
    }

    // ---- MLX-aware GEMV dispatch ----

    /// Dispatch GEMV: handles MLX quantized weights via model_mod.dispatchGemv.
    fn doGemv(self: *Qwen35Model, x: [*]const f32, t: TensorInfo, y: [*]f32, n: usize, k: usize) void {
        model_mod.dispatchGemv(self.be, self.fmt, x, t, y, n, k);
    }

    /// Column-shard (output-dim split): returns TensorInfo pointing to
    /// this rank's slice of rows. Each rank gets n_total/tp_degree consecutive rows.
    fn shardColumnWeight(self: *const Qwen35Model, t: TensorInfo, n_total: usize, k: usize) TensorInfo {
        if (self.tp_degree <= 1) return t;
        const n_local = n_total / self.tp_degree;
        const row_bytes = backend_mod.gemvRowBytes(t.dtype, k);
        if (row_bytes == 0) return t;
        var shard = t;
        shard.data_ptr = t.data_ptr + self.tp_rank * n_local * row_bytes;
        return shard;
    }

    /// Row-shard (input-dim split): extracts this rank's column slice into
    /// a pre-allocated contiguous buffer. For weight W[n, k], rank r gets
    /// columns r*local_k:(r+1)*local_k from each row.
    /// Returns a TensorInfo pointing to the contiguous shard buffer.
    fn shardRowWeight(self: *Qwen35Model, t: TensorInfo, n: usize, k_total: usize, shard_buf: []u8) TensorInfo {
        if (self.tp_degree <= 1) return t;
        const local_k = k_total / self.tp_degree;
        const full_row_bytes = backend_mod.gemvRowBytes(t.dtype, k_total);
        const local_row_bytes = backend_mod.gemvRowBytes(t.dtype, local_k);
        if (full_row_bytes == 0 or local_row_bytes == 0) return t;
        const col_offset = self.tp_rank * local_row_bytes;
        for (0..n) |row| {
            const src = t.data_ptr + row * full_row_bytes + col_offset;
            const dst_off = row * local_row_bytes;
            @memcpy(shard_buf[dst_off..][0..local_row_bytes], src[0..local_row_bytes]);
        }
        // Evict GPU weight cache — shard_buf address reused with different data per rank
        self.be.invalidateWeight(shard_buf.ptr);
        var shard = t;
        shard.data_ptr = shard_buf.ptr;
        return shard;
    }

    /// Build a GemvOp from a TensorInfo, populating MLX companion pointers
    /// when the tensor is MLX-quantized. This enables gemvMulti to dispatch
    /// MLX kernels without barriers between batched ops.
    fn makeOp(self: *Qwen35Model, t: TensorInfo, y: [*]f32, n: usize, k: usize) backend_mod.GemvOp {
        var op = backend_mod.GemvOp{ .w = .{ .data = t.data_ptr, .dtype = t.dtype }, .y = y, .n = n };
        if (t.dtype == .mlx_q) {
            const comp = model_mod.findMlxCompanion(self.fmt, t, k);
            if (comp) |c| {
                op.mlx_scales = c.scales;
                op.mlx_biases = c.biases;
                op.mlx_bits = c.bits;
            }
        }
        return op;
    }

    /// Batched GEMV: dispatches 2 or 3 ops via gemvMulti.
    /// MLX companion pointers are resolved per-op so all dispatches
    /// (including MLX-Q) run without inter-dispatch barriers.
    fn doGemvBatch2(self: *Qwen35Model, x: [*]const f32, t0: TensorInfo, y0: [*]f32, n0: usize, t1: TensorInfo, y1: [*]f32, n1: usize, k: usize) void {
        if (t0.dtype == .nvfp4 or t1.dtype == .nvfp4 or t0.dtype == .gptq or t1.dtype == .gptq or t0.dtype == .awq or t1.dtype == .awq) {
            self.doGemv(x, t0, y0, n0, k);
            self.doGemv(x, t1, y1, n1, k);
            return;
        }
        const ops = [_]backend_mod.GemvOp{
            self.makeOp(t0, y0, n0, k),
            self.makeOp(t1, y1, n1, k),
        };
        self.be.gemvMulti(x, &ops, k);
    }

    fn doGemvBatch3(self: *Qwen35Model, x: [*]const f32, t0: TensorInfo, y0: [*]f32, n0: usize, t1: TensorInfo, y1: [*]f32, n1: usize, t2: TensorInfo, y2: [*]f32, n2: usize, k: usize) void {
        if (t0.dtype == .nvfp4 or t1.dtype == .nvfp4 or t2.dtype == .nvfp4 or
            t0.dtype == .gptq or t1.dtype == .gptq or t2.dtype == .gptq or
            t0.dtype == .awq or t1.dtype == .awq or t2.dtype == .awq)
        {
            self.doGemv(x, t0, y0, n0, k);
            self.doGemv(x, t1, y1, n1, k);
            self.doGemv(x, t2, y2, n2, k);
            return;
        }
        const ops = [_]backend_mod.GemvOp{
            self.makeOp(t0, y0, n0, k),
            self.makeOp(t1, y1, n1, k),
            self.makeOp(t2, y2, n2, k),
        };
        self.be.gemvMulti(x, &ops, k);
    }

    /// Dispatch expert slice GEMV for MLX quantized expert tensors.
    fn doGemvExpert(self: *Qwen35Model, x: [*]const f32, exp_t: TensorInfo, ei: usize, stride: usize, y: [*]f32, n: usize, k: usize) void {
        const data = exp_t.data_ptr + ei * stride;
        if (exp_t.dtype != .mlx_q) {
            self.be.gemv(x, .{ .data = data, .dtype = exp_t.dtype }, y, n, k);
            return;
        }
        const wi = std.mem.lastIndexOf(u8, exp_t.name, ".weight") orelse return;
        var sbuf: [model_mod.tensor_name_buf_size]u8 = undefined;
        const prefix = exp_t.name[0..wi];
        const s_name = std.fmt.bufPrint(&sbuf, "{s}.scales", .{prefix}) catch return;
        const st = self.fmt.getTensor(s_name) orelse return;
        if (st.dtype == .unknown) {
            // MXFP4: dims [n_experts, rows, groups_per_row], U8 — per-expert = dims[1]*dims[2]
            const s_stride = if (st.n_dims >= 3) @as(usize, @intCast(st.dims[1])) * @as(usize, @intCast(st.dims[2])) else st.numElements();
            self.be.gemvMxfp4St(x, data, st.data_ptr + ei * s_stride, y, n, k);
        } else {
            // MLX affine: dims [n_experts, rows, groups_per_row], BF16 — per-expert = dims[1]*dims[2]*2
            var bbuf: [model_mod.tensor_name_buf_size]u8 = undefined;
            const b_name = std.fmt.bufPrint(&bbuf, "{s}.biases", .{prefix}) catch return;
            const bt = self.fmt.getTensor(b_name) orelse return;
            const s_stride = if (st.n_dims >= 3) @as(usize, @intCast(st.dims[1])) * @as(usize, @intCast(st.dims[2])) * 2 else st.numElements() * 2;
            self.be.gemvMlxQ(x, data, st.data_ptr + ei * s_stride, bt.data_ptr + ei * s_stride, y, n, k, 8);
        }
    }

    // ---- Full attention layer ----
    // Qwen3.5 full attention: Q projection outputs Q+gate interleaved (2*head_dim per head)
    // gate is applied as sigmoid(gate) * attention_output before output projection
    /// Full attention layer. When `fuse_ffn_residual` is true, the pre-attention
    /// norm fuses the previous layer's deferred FFN residual (hidden2) into
    /// hidden via addRmsNorm instead of a separate add + rmsNorm.
    fn fullAttnLayer(self: *Qwen35Model, li: u32, fuse_ffn_residual: bool) !void {
        const e: usize = self.n_embd;
        const nh: usize = if (self.tp_degree > 1) self.n_head / self.tp_degree else self.n_head;
        const nkv: usize = if (self.tp_degree > 1) self.n_head_kv / self.tp_degree else self.n_head_kv;
        const hd: usize = self.head_dim;
        const qd: usize = nh * hd;

        var t = self.perf.start();
        const nw = self.fmt.layerTensor(li, "attn_norm.weight") orelse return;
        if (fuse_ffn_residual) {
            // Fused: hidden += hidden2 (prior FFN residual) + normalize
            self.be.addRmsNorm(self.hidden.ptr, self.hidden2.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        } else {
            self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        }
        self.syncProfile();
        self.perf.end(.rms_norm, t);

        // Q/K/V projections — Q output size depends on gate presence
        t = self.perf.start();
        const qw = self.fmt.layerTensor(li, "attn_q.weight") orelse return;
        const kw = self.fmt.layerTensor(li, "attn_k.weight") orelse return;
        const vw = self.fmt.layerTensor(li, "attn_v.weight") orelse return;

        const q_out: usize = if (self.has_gate) qd * 2 else qd;
        const qw_s = self.shardColumnWeight(qw, if (self.has_gate) self.n_head * hd * 2 else self.n_head * hd, e);
        const kw_s = self.shardColumnWeight(kw, self.n_head_kv * hd, e);
        const vw_s = self.shardColumnWeight(vw, self.n_head_kv * hd, e);
        self.doGemvBatch3(self.hidden2.ptr, qw_s, self.q_buf.ptr, q_out, kw_s, self.k_buf.ptr, nkv * hd, vw_s, self.v_buf.ptr, nkv * hd, e);
        self.syncProfile();
        self.perf.end(.gemv_qkv, t);

        // Attention Q/K/V biases (Qwen2/2.5)
        if (self.has_attn_bias) {
            const kvd = nkv * hd;
            if (self.fmt.layerTensor(li, "attn_q.bias")) |qb|
                self.be.addScaled(self.normAsF32(qb, q_out), self.q_buf.ptr, 1.0, q_out);
            if (self.fmt.layerTensor(li, "attn_k.bias")) |kb|
                self.be.addScaled(self.normAsF32(kb, kvd), self.k_buf.ptr, 1.0, kvd);
            if (self.fmt.layerTensor(li, "attn_v.bias")) |vb|
                self.be.addScaled(self.normAsF32(vb, kvd), self.v_buf.ptr, 1.0, kvd);
        }

        // Q processing: with gate (Qwen3.5) → split Q+gate; without → use q_buf directly
        const q_ptr: [*]f32 = if (self.has_gate) blk: {
            t = self.perf.start();
            const gate_buf = self.ff_buf1.ptr;
            const q_deint = self.ff_buf2.ptr;
            if (self.is_safetensors) {
                // SafeTensors/HF: per-head interleaved [Q_h0, G_h0, Q_h1, G_h1, ...].
                // Each head has head_dim Q values followed by head_dim gate values.
                self.be.splitQGate(self.q_buf.ptr, q_deint, gate_buf, hd, nh);
            } else {
                // GGUF: Q+gate element-wise interleaved [Q0,G0,Q1,G1,...] per head
                self.be.deinterleave(self.q_buf.ptr, q_deint, gate_buf, hd, nh);
            }
            self.syncProfile();
            self.perf.end(.deinterleave, t);
            break :blk q_deint;
        } else self.q_buf.ptr;

        // Q/K norms — per-head rmsNorm (Qwen3/3.5 only, absent in Qwen2)
        // Q and K norms write to independent buffers — batch without barriers.
        if (self.has_qk_norm) {
            t = self.perf.start();
            const qnw = self.fmt.layerTensor(li, "attn_q_norm.weight") orelse return;
            const qnd = self.normAsF32(qnw, hd);
            const knw = self.fmt.layerTensor(li, "attn_k_norm.weight") orelse return;
            const knd = self.normAsF32(knw, hd);
            self.be.beginBatch();
            self.be.rmsNormMulti(q_ptr, qnd, nh, hd, self.rms_eps);
            self.be.rmsNormMulti(self.k_buf.ptr, knd, nkv, hd, self.rms_eps);
            self.be.endBatch();
            self.syncProfile();
            self.perf.end(.rms_norm, t);
        }

        // RoPE — Q and K write to independent buffers, batch without barriers.
        t = self.perf.start();
        self.be.beginBatch();
        self.be.rope(q_ptr, self.kv_seq_len, nh, hd, self.rope_dim, self.rope_theta);
        self.be.rope(self.k_buf.ptr, self.kv_seq_len, nkv, hd, self.rope_dim, self.rope_theta);
        self.be.endBatch();
        self.syncProfile();
        self.perf.end(.rope, t);

        // SDPA
        t = self.perf.start();
        const kv_view = self.getLayerKvView(li);
        if (self.isMultiBlock(li)) {
            self.be.sdpaPaged(
                q_ptr,
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
                .f32,
            );
        }
        self.syncProfile();
        self.perf.end(.sdpa, t);

        // Gate: attn_out *= sigmoid(gate) — Qwen3.5 only
        if (self.has_gate) {
            t = self.perf.start();
            self.be.sigmoidMul(self.attn_out.ptr, self.ff_buf1.ptr, qd);
            self.syncProfile();
            self.perf.end(.sigmoid_mul, t);
        }

        // Full-attention output gate (Nex-N2-Pro attn_output_gate: true).
        // Distinct from the Q-projection gate (has_gate): applies to the attention output
        // before the output projection — attn_out *= sigmoid(gate_proj(hidden2)).
        if (self.has_full_attn_output_gate) {
            if (self.fmt.layerTensor(li, "attn_output_gate.weight")) |gw| {
                t = self.perf.start();
                // Compute gate using original (pre-attn) normed hidden2 → stored in ff_buf2
                // (ff_buf2 was not used for Q processing since has_gate handled the Q gate)
                self.doGemv(self.hidden2.ptr, gw, self.ff_buf2.ptr, qd, e);
                self.be.sigmoidMul(self.attn_out.ptr, self.ff_buf2.ptr, qd);
                self.syncProfile();
                self.perf.end(.sigmoid_mul, t);
            }
        }

        // Output projection (row-split for TP: each rank uses local_qd input columns)
        t = self.perf.start();
        const ow_raw = self.fmt.layerTensor(li, "attn_output.weight") orelse return;
        if (self.tp_degree > 1 and self.tp_row_shard_buf.len > 0) {
            const ow_s = self.shardRowWeight(ow_raw, e, self.n_head * hd, self.tp_row_shard_buf);
            self.doGemv(self.attn_out.ptr, ow_s, self.hidden2.ptr, e, qd);
        } else {
            self.doGemv(self.attn_out.ptr, ow_raw, self.hidden2.ptr, e, qd);
        }
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
    /// DeltaNet SSM layer. When `fuse_ffn_residual` is true, the pre-attention
    /// norm fuses the previous layer's deferred FFN residual (hidden2) into
    /// hidden via addRmsNorm instead of a separate add + rmsNorm.
    fn deltaNetLayer(self: *Qwen35Model, li: u32, fuse_ffn_residual: bool) !void {
        const e: usize = self.n_embd;
        const d_inner: usize = self.ssm_d_inner;
        const num_k_heads: usize = self.ssm_n_group;
        const head_k_dim: usize = self.ssm_d_state;
        const num_v_heads: usize = self.ssm_dt_rank;
        const head_v_dim: usize = d_inner / num_v_heads;
        const conv_ch: usize = self.ssmConvChannels();
        const d_conv: usize = self.ssm_d_conv;

        std.debug.assert(num_v_heads <= max_ssm_v_heads);

        // 1. Attention norm (fused with prior FFN residual when available)
        var t = self.perf.start();
        const nw = self.fmt.layerTensor(li, "attn_norm.weight") orelse return error.MissingTensor;
        if (fuse_ffn_residual) {
            self.be.addRmsNorm(self.hidden.ptr, self.hidden2.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        } else {
            self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        }
        self.syncProfile();
        self.perf.end(.rms_norm, t);

        // 2. Input projections: QKV, gate(z), alpha, beta
        t = self.perf.start();
        const qkv_w = self.fmt.layerTensor(li, "attn_qkv.weight") orelse return error.MissingTensor;
        const gate_w = self.fmt.layerTensor(li, "attn_gate.weight") orelse return error.MissingTensor;
        const alpha_w = self.fmt.layerTensor(li, "ssm_alpha.weight") orelse return error.MissingTensor;
        const beta_w = self.fmt.layerTensor(li, "ssm_beta.weight") orelse return error.MissingTensor;
        {
            const delta_ops = [_]backend_mod.GemvOp{
                self.makeOp(qkv_w, self.ssm_qkv_buf.ptr, conv_ch, e),
                self.makeOp(gate_w, self.ssm_z_buf.ptr, d_inner, e),
                self.makeOp(alpha_w, self.ssm_alpha_buf.ptr, num_v_heads, e),
                self.makeOp(beta_w, self.ssm_beta_buf.ptr, num_v_heads, e),
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
                .kqv_order = false, // Q,K,V order for both GGUF and HF SafeTensors
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

    /// FFN compute without norm: gate/up GEMV → SiLU → down GEMV → hidden2.
    /// Input: hidden2 (normed). Output: hidden2 (partial FFN output for this rank).
    fn ffnCompute(self: *Qwen35Model, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = if (self.tp_degree > 1) self.n_ff / self.tp_degree else self.n_ff;

        const gw_raw = self.fmt.layerTensor(li, "ffn_gate.weight") orelse return error.MissingTensor;
        const uw_raw = self.fmt.layerTensor(li, "ffn_up.weight") orelse return error.MissingTensor;
        const gw = self.shardColumnWeight(gw_raw, self.n_ff, e);
        const uw = self.shardColumnWeight(uw_raw, self.n_ff, e);

        self.doGemvBatch2(self.hidden2.ptr, gw, self.ff_buf1.ptr, ff, uw, self.ff_buf2.ptr, ff, e);
        self.be.siluMul(self.ff_buf1.ptr, self.ff_buf2.ptr, self.ff_buf1.ptr, ff);

        const dw_raw = self.fmt.layerTensor(li, "ffn_down.weight") orelse return error.MissingTensor;
        if (self.tp_degree > 1 and self.tp_row_shard_buf.len > 0) {
            const dw = self.shardRowWeight(dw_raw, e, self.n_ff, self.tp_row_shard_buf);
            self.doGemv(self.ff_buf1.ptr, dw, self.hidden2.ptr, e, ff);
        } else {
            self.doGemv(self.ff_buf1.ptr, dw_raw, self.hidden2.ptr, e, ff);
        }
    }

    /// MLP layer with post-attention norm applied to the residual stream.
    /// Fuses the attention residual add with the post-attention norm into a
    /// single addRmsNorm dispatch (saves one GPU kernel launch per layer).
    /// When `defer_residual` is true, the final FFN residual add is skipped —
    /// the caller fuses it with the next layer's pre-attention norm.
    fn mlpLayer(self: *Qwen35Model, li: u32, defer_residual: bool) !void {
        const e: usize = self.n_embd;
        const ff: usize = if (self.tp_degree > 1) self.n_ff / self.tp_degree else self.n_ff;

        // Pre-MLP norm: Qwen3.5 fuses residual add + norm (addRmsNorm with post_attention_norm),
        // Qwen3/2 uses standard separate pre-norm (rmsNorm with ffn_norm).
        var t = self.perf.start();
        if (self.has_post_attn_norm) {
            const nw = self.fmt.layerTensor(li, "post_attention_norm.weight") orelse return error.MissingTensor;
            self.be.addRmsNorm(self.hidden.ptr, self.hidden2.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        } else {
            const nw = self.fmt.layerTensor(li, "ffn_norm.weight") orelse return error.MissingTensor;
            self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        }
        self.syncProfile();
        self.perf.end(.rms_norm, t);

        // SwiGLU FFN — gate+up projections + SiLU*mul
        t = self.perf.start();
        const gw_raw = self.fmt.layerTensor(li, "ffn_gate.weight") orelse return error.MissingTensor;
        const uw_raw = self.fmt.layerTensor(li, "ffn_up.weight") orelse return error.MissingTensor;
        const gw = self.shardColumnWeight(gw_raw, self.n_ff, e);
        const uw = self.shardColumnWeight(uw_raw, self.n_ff, e);
        if (self.megakernel_enabled and self.tp_degree <= 1 and (gw.dtype == .q8_0 or gw.dtype == .q4_k or gw.dtype == .q4_0 or gw.dtype == .q5_k or gw.dtype == .q6_k)) {
            // Fused: gate GEMV + up GEMV + SiLU*mul in a single dispatch (3→1)
            // Use inline else to avoid compiling Metal-specific code on Linux
            switch (self.be) {
                inline else => |be| {
                    if (comptime @hasDecl(@TypeOf(be.*), "fusedFfnGateUpSiluQ8")) {
                        switch (gw.dtype) {
                            .q8_0 => be.fusedFfnGateUpSiluQ8(self.hidden2.ptr, gw.data_ptr, uw.data_ptr, self.ff_buf1.ptr, ff, e),
                            .q4_k => if (comptime @hasDecl(@TypeOf(be.*), "fusedFfnGateUpSiluQ4K"))
                                be.fusedFfnGateUpSiluQ4K(self.hidden2.ptr, gw.data_ptr, uw.data_ptr, self.ff_buf1.ptr, ff, e),
                            .q5_k => if (comptime @hasDecl(@TypeOf(be.*), "fusedFfnGateUpSiluQ5K"))
                                be.fusedFfnGateUpSiluQ5K(self.hidden2.ptr, gw.data_ptr, uw.data_ptr, self.ff_buf1.ptr, ff, e),
                            .q6_k => if (comptime @hasDecl(@TypeOf(be.*), "fusedFfnGateUpSiluQ6K"))
                                be.fusedFfnGateUpSiluQ6K(self.hidden2.ptr, gw.data_ptr, uw.data_ptr, self.ff_buf1.ptr, ff, e),
                            .q4_0 => if (comptime @hasDecl(@TypeOf(be.*), "fusedFfnGateUpSiluQ40"))
                                be.fusedFfnGateUpSiluQ40(self.hidden2.ptr, gw.data_ptr, uw.data_ptr, self.ff_buf1.ptr, ff, e),
                            else => {},
                        }
                    }
                },
            }
        } else {
            // Standard path: 3 dispatches (gate GEMV + up GEMV + siluMul)
            self.doGemvBatch2(self.hidden2.ptr, gw, self.ff_buf1.ptr, ff, uw, self.ff_buf2.ptr, ff, e);
            self.syncProfile();
            self.perf.end(.gemv_ffn, t);
            t = self.perf.start();
            self.be.siluMul(self.ff_buf1.ptr, self.ff_buf2.ptr, self.ff_buf1.ptr, ff);
        }
        self.syncProfile();
        self.perf.end(.gemv_ffn, t);

        // Down projection (row-split for TP)
        t = self.perf.start();
        const dw_raw = self.fmt.layerTensor(li, "ffn_down.weight") orelse return error.MissingTensor;
        if (self.tp_degree > 1 and self.tp_row_shard_buf.len > 0) {
            const dw = self.shardRowWeight(dw_raw, e, self.n_ff, self.tp_row_shard_buf);
            self.doGemv(self.ff_buf1.ptr, dw, self.hidden2.ptr, e, ff);
        } else {
            self.doGemv(self.ff_buf1.ptr, dw_raw, self.hidden2.ptr, e, ff);
        }
        self.syncProfile();
        self.perf.end(.gemv_ffn, t);

        if (defer_residual) return; // Caller fuses with next layer's pre-attn norm

        // FFN residual (only on last layer or when deferral is disabled)
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

        // Post-attention norm: fuse residual add when has_post_attn_norm
        // (same as mlpLayer — attention output in hidden2 needs residual add to hidden)
        var t = self.perf.start();
        if (self.has_post_attn_norm) {
            const nw = self.fmt.layerTensor(li, "post_attention_norm.weight") orelse return error.MissingTensor;
            self.be.addRmsNorm(self.hidden.ptr, self.hidden2.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        } else {
            const nw = self.fmt.layerTensor(li, "ffn_norm.weight") orelse return error.MissingTensor;
            self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        }
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
            const inv = if (sum_e > 0.0) 1.0 / sum_e else 0.0;
            for (0..n_exp) |i| self.router_logits[i] *= inv;
        }

        var top_experts: [max_active_experts]usize = undefined;
        var top_scores: [max_active_experts]f32 = undefined;
        math_ops.topKExperts(self.router_logits[0..n_exp], n_active, top_experts[0..n_active], top_scores[0..n_active]);

        // Renormalize selected weights to sum to 1.0.
        {
            var sel_sum: f32 = 0.0;
            for (0..n_active) |i| sel_sum += top_scores[i];
            const inv = if (sel_sum > 0.0) 1.0 / sel_sum else 0.0;
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
            } else if (gate_exps.dtype == .nvfp4) {
                // Compressed-tensors NVFP4: GPU dispatch with companion scale tensor
                const gate_scales = self.fmt.layerTensor(li, "ffn_gate_exps.scales") orelse return error.MissingTensor;
                const up_scales = self.fmt.layerTensor(li, "ffn_up_exps.scales") orelse return error.MissingTensor;
                const gate_s_stride = model_mod.expertStride(.{ .dtype = .fp8_e4m3, .n_dims = gate_scales.n_dims, .dims = gate_scales.dims, .name = "", .data_ptr = undefined });
                const up_s_stride = model_mod.expertStride(.{ .dtype = .fp8_e4m3, .n_dims = up_scales.n_dims, .dims = up_scales.dims, .name = "", .data_ptr = undefined });
                self.be.gemvNvfp4St(self.hidden2.ptr, gate_data, gate_scales.data_ptr + ei * gate_s_stride, self.ff_buf1.ptr, ff, e);
                const up_data = up_exps.data_ptr + ei * up_stride;
                self.be.gemvNvfp4St(self.hidden2.ptr, up_data, up_scales.data_ptr + ei * up_s_stride, self.ff_buf2.ptr, ff, e);
                self.be.sync();
                // Apply input_global_scale / weight_global_scale
                applyNvfp4Scale(self.fmt, self.ff_buf1[0..ff], li, ei, "gate_proj");
                applyNvfp4Scale(self.fmt, self.ff_buf2[0..ff], li, ei, "up_proj");
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
            } else if (down_exps.dtype == .nvfp4) {
                const down_scales = self.fmt.layerTensor(li, "ffn_down_exps.scales") orelse return error.MissingTensor;
                const down_s_stride = model_mod.expertStride(.{ .dtype = .fp8_e4m3, .n_dims = down_scales.n_dims, .dims = down_scales.dims, .name = "", .data_ptr = undefined });
                self.be.gemvNvfp4St(self.ff_buf1.ptr, down_data, down_scales.data_ptr + ei * down_s_stride, self.attn_out.ptr, e, ff);
                self.be.sync();
                applyNvfp4Scale(self.fmt, self.attn_out[0..e], li, ei, "down_proj");
            } else {
                self.be.gemv(self.ff_buf1.ptr, .{ .data = down_data, .dtype = down_exps.dtype }, self.attn_out.ptr, e, ff);
            }
            self.be.sync();
            self.perf.end(.gemv_ffn, t);

            // Weighted accumulation (SIMD via backend addScaled).
            self.be.addScaled(self.attn_out.ptr, self.moe_out.ptr, mix_weight, e);
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
            const gate_val = math_ops.sigmoid(math_ops.simdDotF32(gate_ptr, self.hidden2.ptr, e));
            self.be.addScaled(self.attn_out.ptr, self.moe_out.ptr, gate_val, e);
        } else {
            self.be.addScaled(self.attn_out.ptr, self.moe_out.ptr, 1.0, e);
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

    /// RMSNorm with +1 weight offset: output[i] = (1 + w[i]) * x[i] / rms(x)
    fn rmsNormPlusOne(input: []const f32, output: []f32, weight: [*]const f32, n: usize, eps: f32) void {
        const V8 = @Vector(8, f32);
        const sum_sq = math_ops.simdDotF32(input.ptr, input.ptr, n);
        const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(n)) + eps);
        const inv_v: V8 = @splat(inv_rms);
        const one_v: V8 = @splat(@as(f32, 1.0));
        var i: usize = 0;
        while (i + 8 <= n) : (i += 8) {
            output[i..][0..8].* = (one_v + @as(V8, weight[i..][0..8].*)) * @as(V8, input[i..][0..8].*) * inv_v;
        }
        while (i < n) : (i += 1) {
            output[i] = (1.0 + weight[i]) * input[i] * inv_rms;
        }
    }

    /// MTP head forward: run token through a single MTP transformer layer.
    /// Uses saved pre-norm hidden state from the main model's last forward().
    /// Returns argmax of MTP logits. `depth` selects which MTP head (0-based).
    pub fn mtpForward(self: *Qwen35Model, token_id: u32, depth: u32) !u32 {
        if (self.n_mtp_layers == 0 or depth >= self.n_mtp_layers) return error.MissingTensor;
        if (self.mtp_kv_seq_len >= self.max_seq_len) return error.KVCacheFull;

        const e: usize = self.n_embd;
        const nh: usize = self.n_head;
        const nkv: usize = self.n_head_kv;
        const hd: usize = self.head_dim;
        const qd: usize = nh * hd;
        const mtp_lid: u32 = self.n_layers + depth;

        // 1. Embed token — per-depth embed table if present, else share main embedding
        const emb_t = self.fmt.layerTensor(mtp_lid, "nextn.embed_tokens.weight") orelse
            self.fmt.layerTensor(mtp_lid, "nextn.embed_tokens") orelse
            self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        self.be.embLookup(.{ .data = emb_t.data_ptr, .dtype = emb_t.dtype }, token_id, self.hidden2.ptr, e);

        // 2. RMSNorm both branches with +1 offset: output = (1 + w) * rmsNorm(x)
        // Standard rmsNorm computes w * x / rms. We need (1 + w) * x / rms.
        // Strategy: rmsNorm with weight w → result, then add x / rms (unweighted).
        // Equivalently: compute rms, apply (1+w) manually on CPU.
        const enorm_t = self.fmt.layerTensor(mtp_lid, "nextn.enorm.weight") orelse
            self.fmt.layerTensor(mtp_lid, "nextn.enorm") orelse return error.MissingTensor;
        const hnorm_t = self.fmt.layerTensor(mtp_lid, "nextn.hnorm.weight") orelse
            self.fmt.layerTensor(mtp_lid, "nextn.hnorm") orelse return error.MissingTensor;
        const enorm_w = self.normAsF32(enorm_t, e);
        const hnorm_w = self.normAsF32(hnorm_t, e);
        self.be.sync();

        // Embed branch → first half of concat buf
        rmsNormPlusOne(self.hidden2, self.mtp_concat_buf[0..e], enorm_w, e, self.rms_eps);
        // Hidden branch → second half of concat buf
        rmsNormPlusOne(self.mtp_hidden_pre_norm, self.mtp_concat_buf[e..][0..e], hnorm_w, e, self.rms_eps);

        // 3. eh_proj: [2*n_embd] → [n_embd]
        const eh_proj = self.fmt.layerTensor(mtp_lid, "nextn.eh_proj.weight") orelse
            self.fmt.layerTensor(mtp_lid, "nextn.eh_proj") orelse return error.MissingTensor;
        self.doGemv(self.mtp_concat_buf.ptr, eh_proj, self.hidden.ptr, e, e * 2);

        // 4. Transformer block: attention + FFN at mtp_lid
        // Pre-attention norm
        const attn_nw = self.fmt.layerTensor(mtp_lid, "attn_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(attn_nw, e), self.hidden2.ptr, e, self.rms_eps);

        // Q/K/V projections
        const qw = self.fmt.layerTensor(mtp_lid, "attn_q.weight") orelse return error.MissingTensor;
        const kw = self.fmt.layerTensor(mtp_lid, "attn_k.weight") orelse return error.MissingTensor;
        const vw = self.fmt.layerTensor(mtp_lid, "attn_v.weight") orelse return error.MissingTensor;
        const q_out: usize = if (self.has_gate) qd * 2 else qd;
        self.doGemvBatch3(self.hidden2.ptr, qw, self.q_buf.ptr, q_out, kw, self.k_buf.ptr, nkv * hd, vw, self.v_buf.ptr, nkv * hd, e);
        self.be.sync();

        // Q gate split (Qwen3.5 has gated Q)
        const q_ptr: [*]f32 = if (self.has_gate) blk: {
            const gate_buf = self.ff_buf1.ptr;
            const q_deint = self.ff_buf2.ptr;
            if (self.is_safetensors) {
                // SafeTensors/HF: per-head interleaved [Q_h0, G_h0, Q_h1, G_h1, ...]
                self.be.splitQGate(self.q_buf.ptr, q_deint, gate_buf, hd, nh);
            } else {
                self.be.deinterleave(self.q_buf.ptr, q_deint, gate_buf, hd, nh);
            }
            self.be.sync();
            break :blk q_deint;
        } else self.q_buf.ptr;

        // Q/K norms
        if (self.has_qk_norm) {
            const qnw = self.fmt.layerTensor(mtp_lid, "attn_q_norm.weight") orelse return error.MissingTensor;
            const knw = self.fmt.layerTensor(mtp_lid, "attn_k_norm.weight") orelse return error.MissingTensor;
            self.be.rmsNormMulti(q_ptr, self.normAsF32(qnw, hd), nh, hd, self.rms_eps);
            self.be.rmsNormMulti(self.k_buf.ptr, self.normAsF32(knw, hd), nkv, hd, self.rms_eps);
        }

        // RoPE at MTP KV position
        self.be.rope(q_ptr, self.mtp_kv_seq_len, nh, hd, self.rope_dim, self.rope_theta);
        self.be.rope(self.k_buf.ptr, self.mtp_kv_seq_len, nkv, hd, self.rope_dim, self.rope_theta);

        // SDPA with MTP flat KV cache
        attn_ops.scaledDotProductAttention(
            q_ptr,
            self.mtp_kv_keys,
            self.mtp_kv_values,
            self.k_buf,
            self.v_buf,
            self.attn_out.ptr,
            self.scores_buf.ptr,
            nh,
            nkv,
            hd,
            self.mtp_kv_seq_len,
            1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
            self.be,
            null,
            0,
            .f32,
            .f32,
        );
        self.mtp_kv_seq_len += 1;

        // Gate: attn_out *= sigmoid(gate)
        if (self.has_gate) {
            self.be.sigmoidMul(self.attn_out.ptr, self.ff_buf1.ptr, qd);
        }

        // Output projection → hidden2 (residual target)
        const ow = self.fmt.layerTensor(mtp_lid, "attn_output.weight") orelse return error.MissingTensor;
        self.doGemv(self.attn_out.ptr, ow, self.hidden2.ptr, e, qd);

        // Residual add: hidden += attn_out
        self.be.addScaled(self.hidden2.ptr, self.hidden.ptr, 1.0, e);

        // FFN: pre-norm + gate/up + silu*mul + down
        const ffn_nw = self.fmt.layerTensor(mtp_lid, "ffn_norm.weight") orelse
            self.fmt.layerTensor(mtp_lid, "post_attention_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(ffn_nw, e), self.hidden2.ptr, e, self.rms_eps);

        const gw = self.fmt.layerTensor(mtp_lid, "ffn_gate.weight") orelse return error.MissingTensor;
        const uw = self.fmt.layerTensor(mtp_lid, "ffn_up.weight") orelse return error.MissingTensor;
        const ff: usize = self.n_ff;
        self.doGemvBatch2(self.hidden2.ptr, gw, self.ff_buf1.ptr, ff, uw, self.ff_buf2.ptr, ff, e);
        self.be.siluMul(self.ff_buf1.ptr, self.ff_buf2.ptr, self.ff_buf1.ptr, ff);
        const dw = self.fmt.layerTensor(mtp_lid, "ffn_down.weight") orelse return error.MissingTensor;
        self.doGemv(self.ff_buf1.ptr, dw, self.hidden2.ptr, e, ff);

        // Residual add: hidden += ffn_out
        self.be.addScaled(self.hidden2.ptr, self.hidden.ptr, 1.0, e);

        // 5. Output head: shared_head_norm → shared_head_head → logits
        const sh_norm = self.fmt.layerTensor(mtp_lid, "nextn.shared_head_norm.weight") orelse
            self.fmt.layerTensor(mtp_lid, "nextn.shared_head_norm") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(sh_norm, e), self.hidden.ptr, e, self.rms_eps);
        // Shared output head — use per-depth nextn head if present, else share main output
        const sh_head = self.fmt.layerTensor(mtp_lid, "nextn.shared_head_head") orelse
            self.fmt.getTensor("output.weight") orelse return error.MissingTensor;
        self.doGemv(self.hidden.ptr, sh_head, self.mtp_logits_buf.ptr, self.vocab_size, e);
        self.be.sync();

        return math_ops.argmax(self.mtp_logits_buf);
    }

    pub fn getMtpLogits(self: *Qwen35Model) []f32 {
        return self.mtp_logits_buf;
    }

    pub fn resetMtpCache(self: *Qwen35Model) void {
        self.mtp_kv_seq_len = 0;
    }

    /// Run one token through the model, returning the argmax next token ID.
    pub fn forward(self: *Qwen35Model, token_id: u32) !u32 {
        if (self.kv_seq_len >= self.max_seq_len) return error.KVCacheFull;

        try model_mod.ensureKvBlock(self);

        if (self.perf.enabled) {
            if (comptime builtin.os.tag == .macos) switch (self.be) {
                .metal => |be| be.resetCounters(),
                else => {},
            };
        }

        const t = self.perf.start();
        // Check if this is an image pad token — if so, inject the pre-computed
        // visual embedding instead of looking up from the token embedding table.
        var is_image_token = false;
        if (self.image_embeddings) |vis_embd| {
            if (self.image_pad_token_id != 0 and token_id == self.image_pad_token_id) {
                const idx = self.visual_token_idx;
                if (idx < self.n_visual_tokens) {
                    const offset = @as(usize, idx) * self.n_embd;
                    const end = offset + self.n_embd;
                    if (end <= vis_embd.len) {
                        @memcpy(self.hidden, vis_embd[offset..end]);
                        is_image_token = true;
                    }
                    self.visual_token_idx = idx + 1;
                }
            }
        }
        if (!is_image_token) {
            try self.embLookup(token_id);
        }
        self.syncProfile();
        self.perf.end(.emb_lookup, t);

        // Pipeline parallelism: determine this rank's layer range
        const pp_layers_per_rank = if (self.pp_degree > 1) self.n_layers / self.pp_degree else self.n_layers;
        const pp_layer_start = self.pp_rank * pp_layers_per_rank;
        const pp_layer_end = if (self.pp_rank == self.pp_degree - 1) self.n_layers else pp_layer_start + pp_layers_per_rank;

        // PP: receive activations from previous stage (batched NCCL group)
        if (self.pp_degree > 1 and self.pp_rank > 0) {
            if (self.pp_transport) |transport| {
                const e = self.n_embd;
                const recv_bufs = [_][*]f32{ self.hidden.ptr, self.hidden2.ptr };
                const recv_lens = [_]usize{ e, e };
                transport.recvBufs(&recv_bufs, &recv_lens);
            }
        }

        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.monotonic)) return error.Cancelled;
            const l: u32 = @intCast(li);
            if (l >= self.layer_skip_start and l < self.layer_skip_end) continue;
            // PP: skip layers not owned by this rank
            if (self.pp_degree > 1 and (li < pp_layer_start or li >= pp_layer_end)) continue;
            self.fmt.prefetchLayer(@intCast(li + 1));
            const fuse = li > 0 and !self.is_moe;

            if (self.tp_degree > 1 and self.tp_row_shard_buf.len > 0 and !self.is_moe) {
                const e = self.n_embd;

                // Attention: run with TP=1 (full heads, single KV cache)
                // Full attention TP pending — needs per-rank KV cache refactor
                const saved_tp = self.tp_degree;
                self.tp_degree = 1;
                if (!self.layer_is_deltanet[l]) try self.fullAttnLayer(l, fuse) else try self.deltaNetLayer(l, fuse);
                self.tp_degree = saved_tp;

                // FFN TP: norm → per-rank compute → all-reduce
                // Distributed path: single rank + network all-reduce
                if (self.tp_transport) |transport| {
                    if (self.has_post_attn_norm) {
                        const nw = self.fmt.layerTensor(l, "post_attention_norm.weight") orelse return error.MissingTensor;
                        self.be.addRmsNorm(self.hidden.ptr, self.hidden2.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
                    } else {
                        const nw = self.fmt.layerTensor(l, "ffn_norm.weight") orelse return error.MissingTensor;
                        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
                    }
                    // For NCCL: no sync needed — data stays on GPU throughout.
                    // For TCP/shm: sync after norm so ffnCompute reads correct host data.
                    self.be.sync();
                    try self.ffnCompute(l);
                    transport.allReduceAdd(self.hidden2.ptr, e) catch |err| {
                        std.log.err("allReduceAdd failed: {}", .{err});
                        return error.MissingTensor;
                    };
                    continue;
                }

                // Local TP: dual-rank sequential + local all-reduce
                // Step 1: Pre-MLP norm (shared)
                {
                    const nt = self.perf.start();
                    if (self.has_post_attn_norm) {
                        const nw = self.fmt.layerTensor(l, "post_attention_norm.weight") orelse return error.MissingTensor;
                        self.be.addRmsNorm(self.hidden.ptr, self.hidden2.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
                    } else {
                        const nw = self.fmt.layerTensor(l, "ffn_norm.weight") orelse return error.MissingTensor;
                        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
                    }
                    self.syncProfile();
                    self.perf.end(.rms_norm, nt);
                }
                self.be.sync();

                // Save normed hidden2 for rank 1
                @memcpy(self.scores_buf[0..e], self.hidden2[0..e]);

                // Step 2: Rank 0 FFN compute
                self.tp_rank = 0;
                try self.ffnCompute(l);
                self.be.sync();
                @memcpy(self.attn_out[0..e], self.hidden2[0..e]);

                // Restore normed hidden2 for rank 1
                @memcpy(self.hidden2[0..e], self.scores_buf[0..e]);
                self.be.invalidateActivation(self.hidden2.ptr);

                // Step 3: Rank 1 FFN compute
                self.tp_rank = 1;
                try self.ffnCompute(l);
                self.be.sync();

                // Step 4: All-reduce: hidden2 += rank0_partial
                for (0..e) |i| self.hidden2[i] += self.attn_out[i];
                self.be.invalidateActivation(self.hidden2.ptr);
                self.tp_rank = 0;
            } else {
                if (!self.layer_is_deltanet[l]) try self.fullAttnLayer(l, fuse) else try self.deltaNetLayer(l, fuse);
                if (self.is_moe) {
                    try self.moeLayer(l);
                } else {
                    try self.mlpLayer(l, true);
                }
            }
        }

        // PP: send activations to next stage / receive logits
        if (self.pp_degree > 1 and self.pp_transport != null) {
            const transport = self.pp_transport.?;
            const e = self.n_embd;
            self.be.sync();
            if (self.pp_rank < self.pp_degree - 1) {
                // Not last stage: send hidden+hidden2 to next rank (batched), receive token back
                const send_bufs = [_][*]const f32{ self.hidden.ptr, self.hidden2.ptr };
                const send_lens = [_]usize{ e, e };
                transport.sendBufs(&send_bufs, &send_lens);
                // Receive the argmax'd token from last rank
                var result_token: [1]f32 = undefined;
                transport.recvBuf(&result_token, 1);
                self.kv_seq_len += 1;
                return @intFromFloat(result_token[0]);
            }
            // Last stage: runs output projection below, sends token back to rank 0
        }

        // Save pre-norm hidden state for MTP heads before final norm destroys it
        if (self.n_mtp_layers > 0) {
            self.be.sync();
            @memcpy(self.mtp_hidden_pre_norm, self.hidden[0..self.n_embd]);
        }

        // Fuse final FFN residual (hidden2) into output norm.
        const nw = self.fmt.getTensor("output_norm.weight") orelse return error.MissingTensor;
        const ow = self.fmt.getTensor("output.weight") orelse self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        self.kv_seq_len += 1;
        self.perf.addToken();
        const e = self.n_embd;
        const norm_w = self.normAsF32(nw, e);
        if (self.is_moe) {
            self.be.rmsNorm(self.hidden.ptr, norm_w, self.hidden.ptr, e, self.rms_eps);
        } else {
            // Fused: hidden += hidden2 (last FFN residual) + output norm
            self.be.addRmsNorm(self.hidden.ptr, self.hidden2.ptr, norm_w, self.hidden.ptr, e, self.rms_eps);
        }
        self.be.sync();
        if (ow.dtype == .mlx_q) {
            self.doGemv(self.hidden.ptr, ow, self.logits_buf.ptr, self.vocab_size, e);
        } else {
            self.be.gemv(self.hidden.ptr, .{ .data = ow.data_ptr, .dtype = ow.dtype }, self.logits_buf.ptr, self.vocab_size, e);
        }
        self.be.sync();
        // Log dispatch stats on first generated token
        if (self.perf.enabled and self.kv_seq_len == 1) {
            if (comptime builtin.os.tag == .macos) switch (self.be) {
                .metal => |be| std.log.warn("Metal stats: {d} dispatches, {d} barriers, {d} syncs", .{
                    be.dispatch_count, be.barrier_count, be.sync_count,
                }),
                else => {},
            };
        }
        const result = math_ops.argmax(self.logits_buf);
        // PP: last rank sends result token back to rank 0
        if (self.pp_degree > 1 and self.pp_rank == self.pp_degree - 1) {
            if (self.pp_transport) |transport| {
                var tok_f32 = [1]f32{@floatFromInt(result)};
                transport.sendBuf(&tok_f32, 1);
            }
        }
        return result;
    }

    /// Batched prefill — sequential. DeltaNet SSM layers require sequential
    /// state updates (recurrence depends on previous token). Only every
    /// full_attn_interval-th layer is pure attention — batching those alone
    /// would add complexity for marginal gain.
    pub fn prefill(self: *Qwen35Model, token_ids: []const u32) !u32 {
        var last: u32 = 0;
        for (token_ids) |tid| last = try self.forward(tid);
        return last;
    }

    /// Send KV cache to a peer via transport (for disaggregated prefill/decode).
    pub fn sendKvCache(self: *Qwen35Model, transport: *TransportMod.Transport) void {
        const kvd = self.paged_cache.kv_dim;
        const bs = self.paged_cache.block_size;
        const elems_per_block = @as(usize, bs) * kvd;
        // Send seq_len and n_layers
        var meta = [3]f32{ @floatFromInt(self.kv_seq_len), @floatFromInt(self.n_layers), 0 };
        transport.sendBuf(&meta, 3);
        // Send block data for each layer
        for (0..self.n_layers) |li| {
            const bt = self.seq_table.block_table[li];
            const n_blocks = (self.kv_seq_len + bs - 1) / bs;
            for (0..n_blocks) |bi| {
                const block_id = bt[bi];
                const blk = self.paged_cache.blocks[block_id];
                transport.sendBuf(blk.keys.ptr, elems_per_block);
                transport.sendBuf(blk.values.ptr, elems_per_block);
            }
        }
    }

    /// Receive KV cache from a peer via transport.
    pub fn recvKvCache(self: *Qwen35Model, transport: *TransportMod.Transport) void {
        const kvd = self.paged_cache.kv_dim;
        const bs = self.paged_cache.block_size;
        const elems_per_block = @as(usize, bs) * kvd;
        // Receive seq_len and n_layers
        var meta: [3]f32 = undefined;
        transport.recvBuf(&meta, 3);
        const seq_len: usize = @intFromFloat(meta[0]);
        // Allocate blocks for the received sequence
        const n_blocks = (seq_len + bs - 1) / bs;
        // Ensure we have enough blocks
        while (self.kv_seq_len < seq_len) {
            if (self.kv_seq_len > 0 and self.kv_seq_len % bs == 0) {
                self.block_allocator.appendBlock(&self.seq_table) catch break;
            }
            self.kv_seq_len += 1;
        }
        self.kv_seq_len = seq_len;
        // Receive block data
        for (0..self.n_layers) |li| {
            const bt = self.seq_table.block_table[li];
            for (0..n_blocks) |bi| {
                const block_id = bt[bi];
                const blk = &self.paged_cache.blocks[block_id];
                transport.recvBuf(blk.keys.ptr, elems_per_block);
                transport.recvBuf(blk.values.ptr, elems_per_block);
                blk.used = if (bi < n_blocks - 1) bs else @intCast(seq_len % bs);
            }
        }
    }

    /// Reset all KV cache and SSM state for a new conversation.
    /// Save all SSM states (conv + recurrence) for prefix caching.
    /// Returns owned memory that must be freed by the caller.
    pub fn saveSsmState(self: *const Qwen35Model, allocator: std.mem.Allocator) ![]u8 {
        var total_bytes: usize = 0;
        for (0..self.n_layers) |i| {
            total_bytes += self.conv_states[i].len * @sizeOf(f32);
            total_bytes += self.ssm_states[i].len * @sizeOf(f32);
        }
        if (total_bytes == 0) return &.{};
        const buf = try allocator.alloc(u8, total_bytes);
        var pos: usize = 0;
        for (0..self.n_layers) |i| {
            const conv_bytes = self.conv_states[i].len * @sizeOf(f32);
            if (conv_bytes > 0) {
                @memcpy(buf[pos..][0..conv_bytes], std.mem.sliceAsBytes(self.conv_states[i]));
                pos += conv_bytes;
            }
            const ssm_bytes = self.ssm_states[i].len * @sizeOf(f32);
            if (ssm_bytes > 0) {
                @memcpy(buf[pos..][0..ssm_bytes], std.mem.sliceAsBytes(self.ssm_states[i]));
                pos += ssm_bytes;
            }
        }
        return buf;
    }

    /// Restore SSM states from a previously saved snapshot.
    pub fn restoreSsmState(self: *Qwen35Model, snapshot: []const u8) void {
        var pos: usize = 0;
        for (0..self.n_layers) |i| {
            const conv_bytes = self.conv_states[i].len * @sizeOf(f32);
            if (conv_bytes > 0 and pos + conv_bytes <= snapshot.len) {
                @memcpy(std.mem.sliceAsBytes(self.conv_states[i]), snapshot[pos..][0..conv_bytes]);
                pos += conv_bytes;
            }
            const ssm_bytes = self.ssm_states[i].len * @sizeOf(f32);
            if (ssm_bytes > 0 and pos + ssm_bytes <= snapshot.len) {
                @memcpy(std.mem.sliceAsBytes(self.ssm_states[i]), snapshot[pos..][0..ssm_bytes]);
                pos += ssm_bytes;
            }
        }
    }

    pub fn resetCache(self: *Qwen35Model) void {
        for (0..self.n_layers) |i| {
            if (self.conv_states[i].len > 0) @memset(self.conv_states[i], 0);
            if (self.ssm_states[i].len > 0) @memset(self.ssm_states[i], 0);
        }
        model_mod.resetKvCache(self);
    }
};

const expertWeightStride = model_mod.expertWeightStride;

/// Apply NVFP4 weight_global_scale to a buffer (DIVIDE).
/// The fp8 block scales are quantized as scale_fp8 = cast_fp8(max/6/gs),
/// so the GEMV output needs to be divided by weight_global_scale to
/// recover the true dot product magnitude.
fn applyNvfp4Scale(fmt: Format, buf: []f32, layer: u32, expert: usize, proj: []const u8) void {
    var nb: [256]u8 = undefined;
    const n = std.fmt.bufPrint(&nb, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.weight_global_scale", .{ layer, expert, proj }) catch return;
    const t = fmt.getTensor(n) orelse return;
    const gs = @as(*const f32, @ptrCast(@alignCast(t.data_ptr))).*;
    if (gs != 1.0 and gs != 0.0) {
        const inv = 1.0 / gs;
        for (buf) |*v| {
            v.* *= inv;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

test "isFullAttn with interval 4" {
    var m: Qwen35Model = undefined;
    m.full_attn_interval = 4;
    try std.testing.expect(!m.isFullAttn(0)); // (0+1) % 4 != 0
    try std.testing.expect(!m.isFullAttn(1)); // (1+1) % 4 != 0
    try std.testing.expect(!m.isFullAttn(2)); // (2+1) % 4 != 0
    try std.testing.expect(m.isFullAttn(3)); // (3+1) % 4 == 0
    try std.testing.expect(!m.isFullAttn(4)); // (4+1) % 4 != 0
    try std.testing.expect(m.isFullAttn(7)); // (7+1) % 4 == 0
}

test "isFullAttn with interval 1 — all layers are full attention" {
    var m: Qwen35Model = undefined;
    m.full_attn_interval = 1;
    try std.testing.expect(m.isFullAttn(0));
    try std.testing.expect(m.isFullAttn(1));
    try std.testing.expect(m.isFullAttn(31));
}

test "isFullAttn with interval 0 — all layers are full attention" {
    var m: Qwen35Model = undefined;
    m.full_attn_interval = 0;
    try std.testing.expect(m.isFullAttn(0));
    try std.testing.expect(m.isFullAttn(5));
    try std.testing.expect(m.isFullAttn(31));
}

test "layerVType boundary protection" {
    var m: Qwen35Model = undefined;
    m.n_layers = 32;
    m.kv_type_v = .turbo4;

    // No boundary — all layers use configured type
    m.kv_boundary_v = 0;
    try std.testing.expectEqual(kv_quant.KvQuantType.turbo4, m.layerVType(0));
    try std.testing.expectEqual(kv_quant.KvQuantType.turbo4, m.layerVType(15));
    try std.testing.expectEqual(kv_quant.KvQuantType.turbo4, m.layerVType(31));

    // Boundary = 2: first 2 and last 2 layers use f16
    m.kv_boundary_v = 2;
    try std.testing.expectEqual(kv_quant.KvQuantType.f16, m.layerVType(0));
    try std.testing.expectEqual(kv_quant.KvQuantType.f16, m.layerVType(1));
    try std.testing.expectEqual(kv_quant.KvQuantType.turbo4, m.layerVType(2));
    try std.testing.expectEqual(kv_quant.KvQuantType.turbo4, m.layerVType(15));
    try std.testing.expectEqual(kv_quant.KvQuantType.turbo4, m.layerVType(29));
    try std.testing.expectEqual(kv_quant.KvQuantType.f16, m.layerVType(30));
    try std.testing.expectEqual(kv_quant.KvQuantType.f16, m.layerVType(31));
}

test "ssmConvChannels default config" {
    var m: Qwen35Model = undefined;
    m.ssm_d_inner = 2048;
    m.ssm_n_group = 16;
    m.ssm_d_state = 128;
    // conv_ch = 2048 + 2 * 16 * 128 = 2048 + 4096 = 6144
    try std.testing.expectEqual(@as(usize, 6144), m.ssmConvChannels());
}

test "shardColumnWeight no-op for single rank" {
    var m: Qwen35Model = undefined;
    m.tp_degree = 1;
    m.tp_rank = 0;
    const dummy = format_mod.TensorInfo{
        .name = "test",
        .data_ptr = @ptrFromInt(0x1000),
        .dtype = .f32,
        .n_dims = 2,
        .dims = .{ 4096, 4096, 0, 0 },
    };
    const result = m.shardColumnWeight(dummy, 4096, 4096);
    // With tp_degree=1, should return the tensor unchanged
    try std.testing.expectEqual(dummy.data_ptr, result.data_ptr);
}

test "shardColumnWeight with 2-way TP" {
    var m: Qwen35Model = undefined;
    m.tp_degree = 2;
    m.tp_rank = 1;
    const base_ptr: [*]const u8 = @ptrFromInt(0x1000);
    const dummy = format_mod.TensorInfo{
        .name = "test",
        .data_ptr = base_ptr,
        .dtype = .f32,
        .n_dims = 2,
        .dims = .{ 4096, 4096, 0, 0 },
    };
    const n_total: usize = 4096;
    const k: usize = 4096;
    const result = m.shardColumnWeight(dummy, n_total, k);
    // Rank 1 should offset by n_local * row_bytes = 2048 * 16384 bytes
    const n_local = n_total / 2;
    const row_bytes = backend_mod.gemvRowBytes(.f32, k);
    const expected_offset = 1 * n_local * row_bytes;
    try std.testing.expectEqual(base_ptr + expected_offset, result.data_ptr);
}

test "rmsNormPlusOne correctness" {
    // (1 + w) * x / rms(x) where rms = sqrt(mean(x^2) + eps)
    var input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output = [_]f32{ 0, 0, 0, 0 };
    var weight = [_]f32{ 0.0, 0.0, 0.0, 0.0 }; // w=0 → (1+0)=1 → plain rmsNorm

    Qwen35Model.rmsNormPlusOne(&input, &output, &weight, 4, 1e-6);

    // With w=0, output should be x / rms(x)
    // rms = sqrt((1+4+9+16)/4 + eps) = sqrt(7.5 + eps) ~ 2.7386
    const sum_sq: f32 = 1.0 + 4.0 + 9.0 + 16.0;
    const inv_rms = 1.0 / @sqrt(sum_sq / 4.0 + 1e-6);
    try std.testing.expectApproxEqAbs(1.0 * inv_rms, output[0], 1e-5);
    try std.testing.expectApproxEqAbs(2.0 * inv_rms, output[1], 1e-5);
    try std.testing.expectApproxEqAbs(3.0 * inv_rms, output[2], 1e-5);
    try std.testing.expectApproxEqAbs(4.0 * inv_rms, output[3], 1e-5);
}

test "model vtable compiles" {
    // Verify that Qwen35Model implements the Model vtable interface
    const M = model_mod.Model;
    _ = M.from;
    // Verify the struct has the expected public API surface
    try std.testing.expect(@hasDecl(Qwen35Model, "forward"));
    try std.testing.expect(@hasDecl(Qwen35Model, "prefill"));
    try std.testing.expect(@hasDecl(Qwen35Model, "resetCache"));
    try std.testing.expect(@hasDecl(Qwen35Model, "cancel"));
    try std.testing.expect(@hasDecl(Qwen35Model, "model"));
}

test "layerVType boundary = 1 single-layer protection" {
    var m: Qwen35Model = undefined;
    m.n_layers = 8;
    m.kv_type_v = .turbo3;
    m.kv_boundary_v = 1;
    // Only first and last layer use f16
    try std.testing.expectEqual(kv_quant.KvQuantType.f16, m.layerVType(0));
    try std.testing.expectEqual(kv_quant.KvQuantType.turbo3, m.layerVType(1));
    try std.testing.expectEqual(kv_quant.KvQuantType.turbo3, m.layerVType(3));
    try std.testing.expectEqual(kv_quant.KvQuantType.turbo3, m.layerVType(6));
    try std.testing.expectEqual(kv_quant.KvQuantType.f16, m.layerVType(7));
}

test "layerVType boundary covers all layers" {
    var m: Qwen35Model = undefined;
    m.n_layers = 8;
    m.kv_type_v = .turbo4;
    // boundary = 4, n_layers = 8: first 4 + last 4 = all 8 layers use f16
    m.kv_boundary_v = 4;
    for (0..8) |i| {
        try std.testing.expectEqual(kv_quant.KvQuantType.f16, m.layerVType(@intCast(i)));
    }
}

test "isFullAttn with interval 2 — alternating" {
    var m: Qwen35Model = undefined;
    m.full_attn_interval = 2;
    // (layer+1) % 2 == 0 → odd layers are full attention
    try std.testing.expect(!m.isFullAttn(0)); // (0+1)%2 = 1
    try std.testing.expect(m.isFullAttn(1)); // (1+1)%2 = 0
    try std.testing.expect(!m.isFullAttn(2)); // (2+1)%2 = 1
    try std.testing.expect(m.isFullAttn(3)); // (3+1)%2 = 0
}

test "rmsNormPlusOne with non-zero weights" {
    var input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output = [_]f32{ 0, 0, 0, 0 };
    var weight = [_]f32{ 1.0, 0.5, -0.5, 0.0 };

    Qwen35Model.rmsNormPlusOne(&input, &output, &weight, 4, 1e-6);

    const sum_sq: f32 = 1.0 + 4.0 + 9.0 + 16.0;
    const inv_rms = 1.0 / @sqrt(sum_sq / 4.0 + 1e-6);
    // output[i] = (1 + w[i]) * input[i] * inv_rms
    try std.testing.expectApproxEqAbs(2.0 * 1.0 * inv_rms, output[0], 1e-5);
    try std.testing.expectApproxEqAbs(1.5 * 2.0 * inv_rms, output[1], 1e-5);
    try std.testing.expectApproxEqAbs(0.5 * 3.0 * inv_rms, output[2], 1e-5);
    try std.testing.expectApproxEqAbs(1.0 * 4.0 * inv_rms, output[3], 1e-5);
}

test "ssmConvChannels with different configs" {
    var m: Qwen35Model = undefined;
    // Minimal config: d_inner=256, n_group=4, d_state=64
    m.ssm_d_inner = 256;
    m.ssm_n_group = 4;
    m.ssm_d_state = 64;
    // conv_ch = 256 + 2 * 4 * 64 = 256 + 512 = 768
    try std.testing.expectEqual(@as(usize, 768), m.ssmConvChannels());
}

test "applyNvfp4Scale no-op when tensor missing" {
    // applyNvfp4Scale should not modify the buffer when the scale tensor is missing.
    var mock = @import("model.zig").MockFormat{ .tensors = &.{} };
    var buf = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    applyNvfp4Scale(mock.format(), &buf, 0, 0, "gate_proj");
    // Buffer should be unchanged — no matching tensor found.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), buf[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), buf[3], 1e-6);
}

test "fuzz: all qwen35 functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // ── 1. isFullAttn: pure field-based, no alloc ──
            {
                var m: Qwen35Model = undefined;
                const interval = smith.valueWithHash(u32, 0) % 17;
                m.full_attn_interval = interval;
                const layer = smith.valueWithHash(u32, 1) % 64;
                const result = m.isFullAttn(layer);
                if (interval == 0) {
                    try std.testing.expect(result);
                }
            }

            // ── 2. layerVType: boundary layer protection ──
            {
                var m: Qwen35Model = undefined;
                m.n_layers = 32;
                m.kv_type_v = .turbo4;
                m.kv_boundary_v = smith.valueWithHash(u32, 2) % 17;
                const li = smith.valueWithHash(u32, 3) % 32;
                const vt = m.layerVType(li);
                _ = vt;
            }

            // ── 3. ssmConvChannels: arithmetic helper ──
            {
                var m: Qwen35Model = undefined;
                m.ssm_d_inner = @as(u32, smith.valueWithHash(u16, 4)) | 1;
                m.ssm_n_group = @as(u32, smith.valueWithHash(u8, 5) % 64) | 1;
                m.ssm_d_state = @as(u32, smith.valueWithHash(u8, 6) % 128) | 1;
                const ch = m.ssmConvChannels();
                try std.testing.expect(ch >= m.ssm_d_inner);
            }

            // ── 4. shardColumnWeight: no-op for tp_degree=1 ──
            {
                var m: Qwen35Model = undefined;
                m.tp_degree = 1;
                m.tp_rank = 0;
                const dummy = format_mod.TensorInfo{
                    .name = "test",
                    .data_ptr = @ptrFromInt(0x1000),
                    .dtype = .f32,
                    .n_dims = 2,
                    .dims = .{ 64, 64, 0, 0 },
                };
                const result = m.shardColumnWeight(dummy, 64, 64);
                try std.testing.expectEqual(dummy.data_ptr, result.data_ptr);
            }

            // ── 5. rmsNormPlusOne: static function, fuzz-safe ──
            {
                const n: usize = 16;
                var input: [n]f32 = undefined;
                var output: [n]f32 = undefined;
                var weight: [n]f32 = undefined;
                for (0..n) |i| {
                    input[i] = @as(f32, @floatFromInt(smith.valueWithHash(i16, @intCast(7 + i)))) / 100.0;
                    weight[i] = @as(f32, @floatFromInt(smith.valueWithHash(i8, @intCast(23 + i)))) / 127.0;
                }
                const eps_raw = smith.valueWithHash(u16, 39);
                const eps: f32 = @as(f32, @floatFromInt(eps_raw | 1)) * 1e-8;
                Qwen35Model.rmsNormPlusOne(&input, &output, &weight, n, eps);
                for (0..n) |i| {
                    try std.testing.expect(math.isFinite(output[i]));
                }
            }

            // ── 6. cancel: atomic store, no deps ──
            {
                var m: Qwen35Model = undefined;
                m.cancelled = std.atomic.Value(bool).init(false);
                m.cancel();
                try std.testing.expect(m.cancelled.load(.monotonic));
            }

            // ── 7. getMtpLogits: returns slice ──
            {
                var m: Qwen35Model = undefined;
                var buf = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
                m.mtp_logits_buf = &buf;
                const logits = m.getMtpLogits();
                try std.testing.expectEqual(@as(usize, 4), logits.len);
            }

            // ── 8. resetMtpCache: resets counter ──
            {
                var m: Qwen35Model = undefined;
                m.mtp_kv_seq_len = 42;
                m.resetMtpCache();
                try std.testing.expectEqual(@as(usize, 0), m.mtp_kv_seq_len);
            }

            // ── 9. applyNvfp4Scale: file-scope fn with MockFormat ──
            {
                var mock = model_mod.MockFormat{ .tensors = &.{} };
                var buf = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
                const layer = smith.valueWithHash(u8, 40);
                const expert = smith.valueWithHash(u8, 41);
                applyNvfp4Scale(mock.format(), &buf, layer, expert, "gate_proj");
                // No matching tensor — buffer unchanged
                try std.testing.expectApproxEqAbs(@as(f32, 1.0), buf[0], 1e-6);
            }

            // ── 10. saveSsmState + restoreSsmState: round-trip ──
            {
                const allocator = std.testing.allocator;
                var m: Qwen35Model = undefined;
                m.n_layers = 2;
                m.conv_states = allocator.alloc([]f32, 2) catch return;
                defer allocator.free(m.conv_states);
                m.ssm_states = allocator.alloc([]f32, 2) catch return;
                defer allocator.free(m.ssm_states);

                var conv0 = [_]f32{ 1.0, 2.0 };
                var conv1 = [_]f32{ 3.0, 4.0 };
                m.conv_states[0] = &conv0;
                m.conv_states[1] = &conv1;
                var ssm0 = [_]f32{ 5.0, 6.0 };
                var ssm1 = [_]f32{ 7.0, 8.0 };
                m.ssm_states[0] = &ssm0;
                m.ssm_states[1] = &ssm1;

                const snapshot = m.saveSsmState(allocator) catch return;
                defer if (snapshot.len > 0) allocator.free(snapshot);
                try std.testing.expect(snapshot.len > 0);

                // Zero out, then restore
                conv0 = .{ 0, 0 };
                conv1 = .{ 0, 0 };
                ssm0 = .{ 0, 0 };
                ssm1 = .{ 0, 0 };
                m.restoreSsmState(snapshot);
                try std.testing.expectApproxEqAbs(@as(f32, 1.0), conv0[0], 1e-6);
                try std.testing.expectApproxEqAbs(@as(f32, 8.0), ssm1[1], 1e-6);
            }

            // ── 11. getBlockTable: needs seq_table — verify via comptime ──
            // ── 12. model: returns Model vtable — verify via comptime ──
            // ── 13. init/deinit: need Format+Backend — verify via comptime ──
            // ── 14. forward/prefill: need full model — verify via comptime ──
            // ── 15. mtpForward: needs full model — verify via comptime ──
            // ── 16. sendKvCache/recvKvCache: need transport — verify via comptime ──
            // ── 17. resetCache: needs KV cache — verify via comptime ──
            comptime {
                // Verify all pub functions exist and are callable
                _ = &Qwen35Model.model;
                _ = &Qwen35Model.init;
                _ = &Qwen35Model.deinit;
                _ = &Qwen35Model.cancel;
                _ = &Qwen35Model.getBlockTable;
                _ = &Qwen35Model.mtpForward;
                _ = &Qwen35Model.getMtpLogits;
                _ = &Qwen35Model.resetMtpCache;
                _ = &Qwen35Model.forward;
                _ = &Qwen35Model.prefill;
                _ = &Qwen35Model.sendKvCache;
                _ = &Qwen35Model.recvKvCache;
                _ = &Qwen35Model.saveSsmState;
                _ = &Qwen35Model.restoreSsmState;
                _ = &Qwen35Model.resetCache;
                // Verify the type itself
                _ = @sizeOf(Qwen35Model);
            }
        }
    }.f, .{});
}
