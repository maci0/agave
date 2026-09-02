//! DeepSeek V4 Flash 0731 inference model.
//! Uses GGUF-style tensor names (blk.N.*).
//! Architecture: 4-stream hyper connections, modified MLA (K=V single compressed head,
//! no separate V projection), hash routing (layers 0-2), sqrt_softplus routing (3+),
//! grouped output LoRA (8 groups × 1024 rank).
//! KV compressors: CSA (ratio=4, overlap coff=2, 21 layers) and HCA (ratio=128, 20 layers).
//! CSA softmax-pools 8 candidates (previous-group low half + current-group high half).
//! Every layer uses a 128-token raw sliding window plus learned attention sinks.
//! Lightning Indexer (LID): implemented for CSA layers. When >index_topk compressed
//! blocks exist, scores all blocks via multi-head ReLU dot-product and selects top-k
//! for sparse attention. Gracefully disabled when indexer tensors are absent.

const std = @import("std");
const build_options = @import("build_options");
const Allocator = std.mem.Allocator;
const backend_mod = @import("../backend/backend.zig");
const format_mod = @import("../format/format.zig");
const model_mod = @import("model.zig");
const math_ops = @import("../ops/math.zig");

const quant_ops = @import("../ops/quant.zig");
const mlx_ops = @import("../ops/mlx.zig");
const kv_quant = @import("../ops/kv_quant.zig");
const attn_ops = @import("../ops/attention.zig");
const Backend = backend_mod.Backend;
const gemvMXFP8 = backend_mod.CpuGemv.gemvMXFP8;
const TransportMod = @import("../parallel/transport.zig");
const KvQuantType = kv_quant.KvQuantType;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;
const ExpertCache = @import("../expert_cache.zig").ExpertCache;
const ExpertProfile = @import("../expert_profile.zig").ExpertProfile;
const MtpWeights = @import("ds4_mtp.zig").MtpWeights;
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const DType = format_mod.DType;
const Model = model_mod.Model;
const TensorData = backend_mod.TensorData;

const name_buf_size: usize = model_mod.tensor_name_buf_size;
const n_hc: usize = 4;

/// Monotonic milliseconds (CLOCK_MONOTONIC, interval timing only).
fn perfMonoMs() u64 {
    var ts: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1000 + @as(u64, @intCast(ts.nsec)) / std.time.ns_per_ms;
}
const hc_mix_dim: usize = (2 + n_hc) * n_hc; // = 24
/// Must match GGUF deepseek4.hyper_connection.sinkhorn_iterations (default 20).
const hc_sinkhorn_iters: usize = 20;
const hc_eps: f32 = 1e-6;
const max_norm_entries: usize = 512;

const NormCacheEntry = struct { key: usize, data: []f32 };

/// Sparse V threshold: skip V dequant+accumulation for positions where softmax
/// weight is below this value. At 1e-6, skipped positions contribute < 0.0001%
/// to the output, zero measured PPL impact. Matches attention.zig threshold.
const sparse_v_threshold: f32 = 1e-6;

/// Raw sliding-window length (`sliding_window` in Flash 0731 config). Every
/// layer attends the latest 128 raw KV rows; compressed layers also attend
/// completed CSA/HCA groups. Learned per-head sinks participate in softmax.
const ds4_raw_attn_window: usize = 128;

/// FP4 group size of the official DeepSeek-V4-Flash-0731 routed experts
/// (E2M1 packed 2 nibbles/byte with E8M0 scales, fp4_block_size=32). Used
/// when the fused expert tensors are pointer tables whose scale shape cannot
/// carry the group size.
const ds4_flash_fp4_group_size: usize = 32;

/// First raw KV index visible at `pos` under `ds4_raw_attn_window`.
fn rawAttnStart(pos: usize) usize {
    return if (pos + 1 > ds4_raw_attn_window) pos + 1 - ds4_raw_attn_window else 0;
}

/// Flash 0731 fallback when GGUF/ST metadata has no compress_ratios array.
/// Layers 0–1: SWA only. Even layers from 2: CSA (4). Odd from 3: HCA (128).
fn defaultCompressRatio(layer_i: usize) u32 {
    if (layer_i < 2) return 0;
    return if (layer_i % 2 == 0) 4 else 128;
}

/// CSA compress_ratio. Overlap (coff=2) is enabled only at this ratio.
const csa_compress_ratio: usize = 4;

/// L1-normalize expert weights, then multiply by `routed_scaling_factor`.
/// Matches HuggingFace `DeepseekV4TopKRouter` / `norm_topk_prob=true`:
/// `weights / weights.sum() * routed_scaling_factor`.
fn scaleExpertWeights(weights: []f32, scale: f32) void {
    var sum: f32 = 0;
    for (weights) |w| sum += w;
    if (sum <= 0) return;
    const inv = scale / sum;
    for (weights) |*w| w.* *= inv;
}

/// Softmax-pool one CSA group with official overlap (coff=2).
/// Each token is projected to `2 * head_dim`. The pool mixes the previous
/// group's low half with the current group's high half, producing `head_dim`
/// outputs. First group has no previous window: those four scores are -inf
/// so only the current high half contributes (reference decode path).
/// `token_stride` is the circular-buffer stride (max of CSA/HCA proj dims).
fn csaOverlapPool(
    out: []f32,
    curr_kv: [*]const f32,
    curr_score: [*]const f32,
    prev_kv: ?[*]const f32,
    prev_score: ?[*]const f32,
    token_stride: usize,
) void {
    const head_dim = out.len;
    const V8 = @Vector(8, f32);
    const neg_inf_v: V8 = @splat(-std.math.inf(f32));
    const zeros: V8 = @splat(0.0);
    const has_prev = prev_kv != null and prev_score != null;
    var d: usize = 0;
    while (d + 8 <= head_dim) : (d += 8) {
        var mx: V8 = neg_inf_v;
        var sp: [csa_compress_ratio]V8 = .{ neg_inf_v, neg_inf_v, neg_inf_v, neg_inf_v };
        var kp: [csa_compress_ratio]V8 = .{ zeros, zeros, zeros, zeros };
        if (has_prev) {
            const ps = prev_score.?;
            const pk = prev_kv.?;
            inline for (0..csa_compress_ratio) |t| {
                const off = t * token_stride + d;
                sp[t] = ps[off..][0..8].*;
                kp[t] = pk[off..][0..8].*;
                mx = @max(mx, sp[t]);
            }
        }
        var sc: [csa_compress_ratio]V8 = undefined;
        var kc: [csa_compress_ratio]V8 = undefined;
        inline for (0..csa_compress_ratio) |t| {
            const off = t * token_stride + head_dim + d;
            sc[t] = curr_score[off..][0..8].*;
            kc[t] = curr_kv[off..][0..8].*;
            mx = @max(mx, sc[t]);
        }
        var sum: V8 = zeros;
        var acc: V8 = zeros;
        inline for (0..csa_compress_ratio) |t| {
            const ev = @exp(sp[t] - mx);
            sum += ev;
            acc = @mulAdd(V8, ev, kp[t], acc);
        }
        inline for (0..csa_compress_ratio) |t| {
            const ev = @exp(sc[t] - mx);
            sum += ev;
            acc = @mulAdd(V8, ev, kc[t], acc);
        }
        out[d..][0..8].* = acc / sum;
    }
    while (d < head_dim) : (d += 1) {
        var mx: f32 = -std.math.inf(f32);
        var sp: [csa_compress_ratio]f32 = .{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) };
        var kp: [csa_compress_ratio]f32 = .{ 0, 0, 0, 0 };
        if (has_prev) {
            const ps = prev_score.?;
            const pk = prev_kv.?;
            inline for (0..csa_compress_ratio) |t| {
                sp[t] = ps[t * token_stride + d];
                kp[t] = pk[t * token_stride + d];
                mx = @max(mx, sp[t]);
            }
        }
        var sc: [csa_compress_ratio]f32 = undefined;
        var kc: [csa_compress_ratio]f32 = undefined;
        inline for (0..csa_compress_ratio) |t| {
            sc[t] = curr_score[t * token_stride + head_dim + d];
            kc[t] = curr_kv[t * token_stride + head_dim + d];
            mx = @max(mx, sc[t]);
        }
        var sm: f32 = 0;
        var acc: f32 = 0;
        inline for (0..csa_compress_ratio) |t| {
            const ev = @exp(sp[t] - mx);
            sm += ev;
            acc = @mulAdd(f32, ev, kp[t], acc);
        }
        inline for (0..csa_compress_ratio) |t| {
            const ev = @exp(sc[t] - mx);
            sm += ev;
            acc = @mulAdd(f32, ev, kc[t], acc);
        }
        out[d] = acc / sm;
    }
}

/// Max compressed groups per layer. Uses ratio=4 (smallest) to size the shared stride,
/// since CSA and HCA layers share the same `csa_k` buffer with per-layer offsets.
fn compSlotsPerLayer(max_seq_len: usize) usize {
    return max_seq_len / 4 + 1;
}

/// Pipeline stage `[start, end)` for this rank. Last rank takes the remainder
/// so `n_layers` that do not divide `pp_degree` still cover every layer.
fn ppLayerRange(n_layers: u32, pp_rank: u32, pp_degree: u32) struct { start: u32, end: u32 } {
    if (pp_degree <= 1) return .{ .start = 0, .end = n_layers };
    const per = n_layers / pp_degree;
    const start = pp_rank * per;
    const end = if (pp_rank + 1 == pp_degree) n_layers else start + per;
    return .{ .start = start, .end = end };
}

/// Expert-parallel owner: routed expert `eid` runs on this TP rank.
/// Shared experts are replicated on every rank (not gated by this helper).
fn isLocalExpert(eid: usize, tp_rank: u32, tp_degree: u32) bool {
    if (tp_degree <= 1) return true;
    return eid % @as(usize, tp_degree) == @as(usize, tp_rank);
}

/// After an in-place all-reduce of `hidden = shexp + local_routed`, every rank
/// has `2*shexp + all_routed`. Subtract the local shared-expert contribution
/// so the residual is `shexp + all_routed`.
fn undoDuplicatedShared(hidden: []f32, shexp_out: []const f32, shexp_weight: f32) void {
    std.debug.assert(hidden.len == shexp_out.len);
    const V8 = @Vector(8, f32);
    const wv: V8 = @splat(shexp_weight);
    var i: usize = 0;
    while (i + 8 <= hidden.len) : (i += 8) {
        const h: V8 = hidden[i..][0..8].*;
        const s: V8 = shexp_out[i..][0..8].*;
        hidden[i..][0..8].* = h - s * wv;
    }
    while (i < hidden.len) : (i += 1) hidden[i] -= shexp_out[i] * shexp_weight;
}

/// DeepSeek V4 Flash 0731 inference model.
pub const Ds4Model = struct {
    fmt: Format,
    be: Backend,
    allocator: Allocator,

    // Architecture params
    n_layers: u32 = 43,
    n_embd: u32 = 4096,
    n_head: u32 = 64,
    n_head_kv: u32 = 1,
    q_lora_rank: u32 = 1024,
    kv_lora_rank: u32 = 512,
    rope_dim: u32 = 64,
    n_experts: u32 = 256,
    n_expert_used: u32 = 6,
    /// Expert budget for verification mode (MoE-Spec, arXiv 2602.16052).
    /// When > 0: use this instead of n_expert_used during forward().
    /// Reduces SSD reads during speculative verification by loading fewer experts.
    /// Set to 0 for normal decode (uses n_expert_used).
    expert_budget: u32 = 0,
    n_expert_shared: u32 = 1,
    ff_exp: u32 = 2048,
    hash_layer_count: u32 = 3,
    o_groups: u32 = 8,
    o_lora_rank: u32 = 1024,
    index_head_dim: u32 = 128,
    index_n_heads: u32 = 64,
    index_topk: u32 = 512,
    rms_eps: f32 = 1e-6,
    rope_freq: f32 = 10000.0,
    expert_weights_scale: f32 = 1.5,
    compress_rope_freq: f32 = 160000.0,
    vocab_size: u32 = 129280,
    max_seq_len: u32 = 512,
    eos_token_id: u32 = 1,
    /// Per-layer compression ratios: 0=none, 4=CSA, 128=HCA. Drives rope freq selection.
    compress_ratios: [64]u32 = [_]u32{0} ** 64,

    // Vtable compatibility
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    megakernel_enabled: bool = false,
    /// True after prefaultLocalExperts device-copied the EP-local experts.
    /// When set, the FFN skips its per-token madvise(WILLNEED) prefetch,
    /// the host pages were DONTNEED'd after the copy, so prefetching them
    /// again would re-read the whole expert working set from disk per token.
    experts_resident: bool = false,
    /// True after the first forward released the large repacked host buffers
    /// (their device copies are permanent — see fmt.releaseRepacked).
    repacked_freed: bool = false,
    layer_skip_start: u32 = 0,
    layer_skip_end: u32 = 0,
    pool: ?*@import("../thread_pool.zig").ThreadPool = null,
    /// Dedicated CPU backend for the DS4 hot path on every selected backend.
    /// CSA/HCA, LID, and HC mixing read host f32; GPU GEMV would leave those
    /// slices stale. `setPool()` copies the shared thread pool onto `cpu.pool`.
    cpu: backend_mod.CpuBackend = .{},
    // Tensor parallelism (expert-parallel MoE + NCCL all-reduce of routed output)
    tp_rank: u32 = 0,
    tp_degree: u32 = 1,
    tp_row_shard_buf: []u8 = &.{},
    tp_transport: ?*TransportMod.Transport = null,
    // Pipeline parallelism (layer split; HC state is the activation)
    pp_rank: u32 = 0,
    pp_degree: u32 = 1,
    pp_transport: ?*TransportMod.Transport = null,
    kv_seq_len: usize = 0,
    kv_type: KvQuantType = .q8_0,

    // HC buffers
    hc_state: []f32 = &.{}, // [n_hc * n_embd]
    new_hc: []f32 = &.{}, // [n_hc * n_embd] temp
    hc_mixes: []f32 = &.{}, // [hc_mix_dim = 24]
    hc_pre_w: []f32 = &.{}, // [n_hc]
    hc_post_w: []f32 = &.{}, // [n_hc]
    hc_comb: []f32 = &.{}, // [n_hc * n_hc]

    // Attention buffers
    hidden: []f32 = &.{}, // [n_embd]
    hidden2: []f32 = &.{}, // [n_embd]
    flat_norm: []f32 = &.{}, // [n_hc * n_embd] for HC rms norm
    q_compressed: []f32 = &.{}, // [q_lora_rank]
    q_full: []f32 = &.{}, // [n_head * kv_lora_rank]
    kv_proj: []f32 = &.{}, // [kv_lora_rank]
    scores_buf: []f32 = &.{}, // [max_seq_len]
    attn_out: []f32 = &.{}, // [n_head * kv_lora_rank]
    lora_out: []f32 = &.{}, // [o_groups * o_lora_rank]
    attn_result: []f32 = &.{}, // [n_embd]

    // FFN buffers
    ff_gate: []f32 = &.{}, // [ff_exp]
    ff_up: []f32 = &.{}, // [ff_exp]
    ff_down: []f32 = &.{}, // [n_embd]
    expert_accum: []f32 = &.{}, // [n_embd]
    expert_scratch: []f32 = &.{}, // [max_total_experts * n_embd] for batched down GEMVs
    ff_gate_scratch: []f32 = &.{}, // [max_total_experts * ff_exp] gate outputs pre-siluMul
    ff_up_scratch: []f32 = &.{}, // [max_total_experts * ff_exp] up outputs pre-siluMul
    /// Scratch buffer for dequantized expert weights (AMX acceleration path).
    /// Sized for one expert weight matrix: ff_exp × n_embd elements.
    amx_dequant_buf: []f32 = &.{},
    router_logits: []f32 = &.{}, // [n_experts]
    logits_buf: []f32 = &.{}, // [vocab_size]
    mtp_hidden_buf: []f32 = &.{}, // [n_embd], saved MTP hidden state between depths
    /// MTP KV cache: stores kv_proj (kv_lora_rank=512 dims) per position.
    /// Shared across MTP depths. Positions 0..mtp_kv_len-1 are populated.
    mtp_kv_cache: []f32 = &.{}, // [max_seq_len * kv_lora_rank]
    mtp_kv_len: usize = 0, // number of populated positions
    /// MTP HC state: 4 streams × n_embd. Initialized from target's hc_state.
    mtp_hc_state: []f32 = &.{}, // [n_hc * n_embd]
    score_stride: usize = 0, // per-head score buffer stride

    // SSD streaming: expert cache and activation profiler (set by main.zig).
    expert_cache: ?*ExpertCache = null,
    /// GGUF file descriptor for pread-based expert loading (SSD streaming).
    gguf_fd: i32 = -1,
    /// Tensor data override table: maps tensor name → heap-copied data pointer.
    /// Non-expert weights are heap-copied at first access for Metal safety.
    /// Expert weights use the pread pool instead.
    tensor_overrides: std.StringHashMap([*]const u8) = undefined,
    tensor_overrides_inited: bool = false,

    /// Expert data pool: heap-allocated buffers for pread-loaded expert weights.
    /// Pool holds one layer's worth of experts (7 × gate/up/down weights).
    /// Metal can safely wrap these heap pointers (no page fault risk).
    expert_pool: []u8 = &.{},
    expert_pool_slots: u32 = 0,
    expert_pool_slot_size: u32 = 0,
    /// Base address of mmap'd GGUF data (for computing file offsets from data_ptr).
    gguf_mmap_base: ?[*]const u8 = null,
    expert_profile: ?*ExpertProfile = null,
    /// MTP (multi-token prediction) weights loaded from separate safetensors.
    mtp_weights: ?*MtpWeights = null,
    /// Number of available MTP depths. Set when mtp_weights are loaded.
    /// Used by the Model vtable to expose getMtpDepth().
    n_mtp_layers: u32 = 0,

    // Pre-computed RoPE frequency bases [rope_dim/2]. Eliminates pow() per token.
    rope_freqs: [32]f32 = undefined, // freq_base = rope_freq (layers with ratio=0)
    compress_rope_freqs: [32]f32 = undefined, // freq_base = compress_rope_freq (ratio≠0 layers)

    // GPU-visible rope cos/sin buffers for Metal dispatch.
    rope_cos_buf: []f32 = &.{}, // [rope_dim/2]
    rope_sin_buf: []f32 = &.{}, // [rope_dim/2]
    // GPU-visible slot_weights buffer for Metal weighted accumulation.
    gpu_slot_weights: []f32 = &.{}, // [max_total_experts + 1]
    // GPU-visible routing buffers for zero-sync MoE dispatch.
    gpu_top_ids: []u32 = &.{}, // [max_total_experts]
    gpu_top_weights: []f32 = &.{}, // [max_total_experts]

    // KV cache as quantized bytes: [n_layers * ctx * kv_lora_rank].
    // K=V in DS4 MLA (single compressed head), one buffer serves both K and V.
    kv_k_bytes: []u8 = &.{},

    // Compressor state for CSA (ratio=4) and HCA (ratio=128) layers.
    // Per-token projected KV and score [n_layers * ctx * max_comp_dim], max_comp_dim = 2*kv_lora_rank
    // Compressed KV after softmax+sum: [n_layers * compSlotsPerLayer(ctx) * kv_lora_rank]
    csa_comp_kv: []f32 = &.{},
    csa_comp_score: []f32 = &.{},
    csa_k: []f32 = &.{},
    csa_score_scratch: []f32 = &.{}, // [2 * kv_lora_rank] temp scratch

    // Lightning Indexer state (CSA layers only).
    // Compressed indexer keys: [n_layers * compSlotsPerLayer(ctx) * index_head_dim]
    // Indexer query: [index_n_heads * index_head_dim] (per-token scratch)
    // Head weights: [index_n_heads] (per-token scratch from W^w projection)
    // Block scores: [compSlotsPerLayer(ctx)] (per-token scratch)
    // Top-k indices: [index_topk] (selected compressed block indices)
    lid_comp_k: []f32 = &.{},
    lid_query: []f32 = &.{},
    lid_head_w: []f32 = &.{},
    lid_scores: []f32 = &.{},
    lid_topk_ids: []u32 = &.{},
    lid_enabled: bool = false,

    // Prefill buffers (page_allocator for GPU zero-copy, Metal's
    // newBufferWithBytesNoCopy requires page-aligned pointers).
    // Allocated at init but not yet used: batched prefill is deferred
    // for this model due to hyper connection complexity (see prefill()).
    chunk_size: usize = 256,
    pf_hidden: []f32 = &.{},
    pf_hidden2: []f32 = &.{},
    pf_q_a: []f32 = &.{}, // [cs * q_lora_rank]
    pf_q: []f32 = &.{}, // [cs * n_head * kv_lora_rank]
    pf_kv_proj: []f32 = &.{}, // [cs * kv_lora_rank]
    pf_attn_out: []f32 = &.{}, // [cs * n_head * kv_lora_rank]
    pf_positions: []u32 = &.{},

    /// Pre-dequantized f32 attention weights for forwardTree fast path.
    /// Allocated at first forwardTree call, covers skip..n_layers.
    pf_dequant_q_a: []f32 = &.{}, // [n_verify_layers * q_lora_rank * n_embd]
    pf_dequant_kv: []f32 = &.{}, // [n_verify_layers * kv_lora_rank * n_embd]
    pf_dequant_q_b: []f32 = &.{}, // [n_verify_layers * n_head * kv_lora_rank * q_lora_rank]
    pf_dequant_wo_b: []f32 = &.{}, // [n_verify_layers * n_embd * o_groups * o_lora_rank]
    pf_dequant_ready: bool = false,

    // Norm weight cache (dequantized to f32)
    norm_cache: [max_norm_entries]NormCacheEntry = undefined,
    norm_cache_len: usize = 0,
    name_buf: [name_buf_size]u8 = undefined,

    /// Initialize the model from GGUF format metadata and allocate all working buffers.
    /// Reads architecture hyperparameters (layers, heads, MoE experts, compress ratios),
    /// pre-computes RoPE frequencies, allocates KV cache, compressor buffers, HC state,
    /// and optionally sets up Lightning Indexer (LID) tensors when present.
    pub fn init(
        allocator: Allocator,
        f: Format,
        be: Backend,
        ctx_size: u32,
        kv_type_k: kv_quant.KvQuantType,
        _: kv_quant.KvQuantType, // kv_type_v (K=V shared, use kv_type_k for both)
        _: ?*TieredKvCache, // tiered_cache (not supported yet)
    ) !Ds4Model {
        // DS4 MLA compressed attention uses kvDot/kvMulAccum. f16/f32 are
        // remapped to q8_0. `nvfp4_ds_mla` packs NoPE as NVFP4 and keeps the
        // RoPE tail in f16; other block types pass through from CLI.
        const effective_kv = switch (kv_type_k) {
            .f16, .f32 => .q8_0, // unsupported → fall back to q8_0
            else => kv_type_k,
        };
        var self = Ds4Model{
            .fmt = f,
            .be = be,
            .allocator = allocator,
            .kv_type = effective_kv,
            .gguf_fd = f.file_fd,
            .gguf_mmap_base = f.mmap_base,
        };

        const arch = "deepseek4";
        if (f.getArchU32(arch, "block_count")) |v| self.n_layers = v;
        if (f.getArchU32(arch, "embedding_length")) |v| self.n_embd = v;
        if (f.getArchU32(arch, "attention.head_count")) |v| self.n_head = v;
        if (f.getArchU32(arch, "attention.key_length")) |v| self.kv_lora_rank = v;
        if (f.getArchU32(arch, "rope.dimension_count")) |v| self.rope_dim = v;
        if (f.getArchU32(arch, "expert_count")) |v| self.n_experts = v;
        if (f.getArchU32(arch, "expert_used_count")) |v| self.n_expert_used = v;
        if (f.getArchU32(arch, "expert_shared_count")) |v| self.n_expert_shared = v;
        if (f.getArchU32(arch, "expert_feed_forward_length")) |v| self.ff_exp = v;
        if (f.getArchU32(arch, "hash_layer_count")) |v| self.hash_layer_count = v;
        if (f.getArchU32(arch, "attention.output_group_count")) |v| self.o_groups = v;
        if (f.getArchU32(arch, "attention.output_lora_rank")) |v| self.o_lora_rank = v;
        if (f.getArchU32(arch, "attention.q_lora_rank")) |v| self.q_lora_rank = v;
        if (f.getArchF32(arch, "rope.freq_base")) |v| self.rope_freq = v;
        if (f.getArchF32(arch, "attention.compress_rope_freq_base")) |v| self.compress_rope_freq = v;
        if (f.getArchF32(arch, "attention.layernorm_rms_epsilon")) |v| self.rms_eps = v;
        if (f.getArchU32(arch, "attention.index_head_dim")) |v| self.index_head_dim = v;
        if (f.getArchU32(arch, "attention.index_n_heads")) |v| self.index_n_heads = v;
        if (f.getArchU32(arch, "attention.index_topk")) |v| self.index_topk = v;

        // DS4 Flash 43-layer pattern: [0,0,4,128,4,128,...,4] (layers 2+ alternate).
        // Prefer a metadata array when present (GGUF); SafeTensors currently
        // only stores the first compress_ratios element, so use the fallback.
        for (0..@as(usize, self.n_layers)) |i| {
            self.compress_ratios[i] = defaultCompressRatio(i);
        }
        if (f.getMetaU32Array("deepseek4.compress_ratios") orelse
            f.getMetaU32Array("compress_ratios")) |arr|
        {
            const n = @min(arr.len, @as(usize, self.n_layers));
            for (0..n) |i| self.compress_ratios[i] = arr[i];
        }
        if (f.getMetaF32("deepseek4.expert_weights_scale")) |v| self.expert_weights_scale = v;
        if (f.getArchU32(arch, "context_length")) |v| self.max_seq_len = @min(v, 65536);
        if (ctx_size > 0) self.max_seq_len = ctx_size;

        if (f.getTensor("token_embd.weight")) |t| {
            // Vocab size is the larger dimension (GGUF dim ordering varies by converter).
            if (t.n_dims >= 2) self.vocab_size = @intCast(@max(t.dims[0], t.dims[1]));
        }
        if (f.getMetaU32("tokenizer.ggml.eos_token_id")) |v| self.eos_token_id = v;

        const e: usize = self.n_embd;
        const nh: usize = self.n_head;
        const kd: usize = self.kv_lora_rank;
        const ff: usize = self.ff_exp;
        const nl: usize = self.n_layers;
        const ql: usize = self.q_lora_rank;
        const og: usize = self.o_groups;
        const olr: usize = self.o_lora_rank;
        const ctx: usize = self.max_seq_len;
        const rd: usize = self.rope_dim;
        const nd: usize = rd / 2;
        if (self.kv_type == .nvfp4_ds_mla) {
            if (kd != kv_quant.ds_mla_latent_dim or rd != kv_quant.ds_mla_rope_dim)
                @panic("nvfp4_ds_mla requires kv_lora_rank=512 and rope_dim=64");
        }

        // Pre-compute RoPE frequency bases once (eliminates pow() per token per layer).
        for (0..nd) |i| {
            self.rope_freqs[i] = std.math.pow(f32, self.rope_freq, -@as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(rd)));
            self.compress_rope_freqs[i] = std.math.pow(f32, self.compress_rope_freq, -@as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(rd)));
        }

        self.hc_state = try allocator.alloc(f32, n_hc * e);
        errdefer allocator.free(self.hc_state);
        self.new_hc = try allocator.alloc(f32, n_hc * e);
        errdefer allocator.free(self.new_hc);
        self.hc_mixes = try allocator.alloc(f32, hc_mix_dim);
        errdefer allocator.free(self.hc_mixes);
        self.hc_pre_w = try allocator.alloc(f32, n_hc);
        errdefer allocator.free(self.hc_pre_w);
        self.hc_post_w = try allocator.alloc(f32, n_hc);
        errdefer allocator.free(self.hc_post_w);
        self.hc_comb = try allocator.alloc(f32, n_hc * n_hc);
        errdefer allocator.free(self.hc_comb);
        self.hidden = try allocator.alloc(f32, e);
        errdefer allocator.free(self.hidden);
        self.hidden2 = try allocator.alloc(f32, e);
        errdefer allocator.free(self.hidden2);
        self.flat_norm = try allocator.alloc(f32, n_hc * e);
        errdefer allocator.free(self.flat_norm);
        self.q_compressed = try allocator.alloc(f32, ql);
        errdefer allocator.free(self.q_compressed);
        self.q_full = try allocator.alloc(f32, nh * kd);
        errdefer allocator.free(self.q_full);
        self.kv_proj = try allocator.alloc(f32, kd);
        errdefer allocator.free(self.kv_proj);
        // scores_buf: per-head slices for parallel attention (64 heads × score_stride).
        self.score_stride = std.math.add(usize, ctx, std.math.add(usize, compSlotsPerLayer(ctx), 1) catch return error.OutOfMemory) catch return error.OutOfMemory;
        const scores_elems = std.math.mul(usize, nh, self.score_stride) catch return error.OutOfMemory;
        self.scores_buf = try allocator.alloc(f32, scores_elems);
        errdefer allocator.free(self.scores_buf);
        self.attn_out = try allocator.alloc(f32, nh * kd);
        errdefer allocator.free(self.attn_out);
        self.lora_out = try allocator.alloc(f32, og * olr);
        errdefer allocator.free(self.lora_out);
        self.attn_result = try allocator.alloc(f32, e);
        errdefer allocator.free(self.attn_result);
        self.ff_gate = try allocator.alloc(f32, ff);
        errdefer allocator.free(self.ff_gate);
        self.ff_up = try allocator.alloc(f32, ff);
        errdefer allocator.free(self.ff_up);
        self.ff_down = try allocator.alloc(f32, e);
        errdefer allocator.free(self.ff_down);
        self.expert_accum = try allocator.alloc(f32, e);
        errdefer allocator.free(self.expert_accum);
        // Scratch for batched expert down GEMVs (max_experts = n_expert_used + n_expert_shared)
        const max_experts: usize = @as(usize, self.n_expert_used) + @as(usize, self.n_expert_shared);
        const expert_scratch_elems = std.math.mul(usize, max_experts, e) catch return error.OutOfMemory;
        self.expert_scratch = try allocator.alloc(f32, expert_scratch_elems);
        errdefer allocator.free(self.expert_scratch);
        const ff_gate_elems = std.math.mul(usize, max_experts, ff) catch return error.OutOfMemory;
        self.ff_gate_scratch = try allocator.alloc(f32, ff_gate_elems);
        errdefer allocator.free(self.ff_gate_scratch);
        // AMX dequant scratch: one expert weight matrix (ff × e for gate/up, e × ff for down)
        const amx_elems = std.math.mul(usize, ff, e) catch return error.OutOfMemory;
        self.amx_dequant_buf = try allocator.alloc(f32, amx_elems);
        errdefer allocator.free(self.amx_dequant_buf);

        // Expert data pool: heap staging for GPU-safe expert weight access.
        // 21 slots (7 experts × 3 weights), each sized for max expert weight.
        // For GGUF: pread from file into pool. For SafeTensors: memcpy from mmap.
        // Total: ~92MB on MXFP4. Heap-allocated → Metal GPU can safely read.
        {
            const ff_e_elems = std.math.mul(usize, ff, e) catch return error.OutOfMemory;
            const mxfp4_bytes = backend_mod.weightBytes(.mxfp4, 1, ff_e_elems);
            const max_expert_bytes: u32 = @intCast(@max(mxfp4_bytes, ff_e_elems / 2)); // MXFP4 or MLX-Q4
            const n_pool_slots: u32 = (self.n_expert_used + self.n_expert_shared) * 3;
            const pool_size = std.math.mul(usize, n_pool_slots, max_expert_bytes) catch return error.OutOfMemory;
            self.expert_pool = try allocator.alloc(u8, pool_size);
            errdefer allocator.free(self.expert_pool);
            self.expert_pool_slots = n_pool_slots;
            self.expert_pool_slot_size = max_expert_bytes;
        }
        self.tensor_overrides = std.StringHashMap([*]const u8).init(allocator);
        self.tensor_overrides_inited = true;
        self.ff_up_scratch = try allocator.alloc(f32, ff_gate_elems);
        errdefer allocator.free(self.ff_up_scratch);
        self.router_logits = try allocator.alloc(f32, self.n_experts);
        errdefer allocator.free(self.router_logits);
        self.logits_buf = try allocator.alloc(f32, self.vocab_size);
        errdefer allocator.free(self.logits_buf);
        self.mtp_hidden_buf = try allocator.alloc(f32, e);
        errdefer allocator.free(self.mtp_hidden_buf);
        @memset(self.mtp_hidden_buf, 0);
        // MTP KV cache: max_seq_len positions × kv_lora_rank dims
        self.mtp_kv_cache = try allocator.alloc(f32, @as(usize, self.max_seq_len) * self.kv_lora_rank);
        errdefer allocator.free(self.mtp_kv_cache);
        @memset(self.mtp_kv_cache, 0);
        self.mtp_hc_state = try allocator.alloc(f32, n_hc * e);
        errdefer allocator.free(self.mtp_hc_state);
        @memset(self.mtp_hc_state, 0);

        // GPU-visible rope cos/sin and slot_weights buffers for Metal dispatch.
        self.rope_cos_buf = try allocator.alloc(f32, self.rope_dim / 2);
        errdefer allocator.free(self.rope_cos_buf);
        self.rope_sin_buf = try allocator.alloc(f32, self.rope_dim / 2);
        errdefer allocator.free(self.rope_sin_buf);
        self.gpu_slot_weights = try allocator.alloc(f32, max_experts + 1);
        errdefer allocator.free(self.gpu_slot_weights);
        self.gpu_top_ids = try allocator.alloc(u32, max_experts);
        errdefer allocator.free(self.gpu_top_ids);
        self.gpu_top_weights = try allocator.alloc(f32, max_experts);
        errdefer allocator.free(self.gpu_top_weights);

        // KV cache bytes: K=V shared buffer (MLA single compressed head, halves KV memory).
        const kv_bytes_per_layer = kv_quant.kvByteOffset(self.kv_type, ctx * kd);
        self.kv_k_bytes = try allocator.alloc(u8, nl * kv_bytes_per_layer);
        errdefer allocator.free(self.kv_k_bytes);

        // Compressor buffers: per-token projections for the current compression group only.
        // CSA groups have ratio=4, HCA groups have ratio=128. We use max_ratio=128 as a
        // circular buffer per layer, completed groups are compressed into csa_k and the
        // per-token slots are reused. This is O(layers × max_ratio) instead of O(layers × ctx),
        // reducing memory from ~92GB to ~44MB at 256K context.
        const max_comp_dim: usize = 2 * kd;
        const comp_slots = compSlotsPerLayer(ctx);
        const max_ratio: usize = 128; // HCA max group size
        self.csa_comp_kv = try allocator.alloc(f32, nl * max_ratio * max_comp_dim);
        errdefer allocator.free(self.csa_comp_kv);
        self.csa_comp_score = try allocator.alloc(f32, nl * max_ratio * max_comp_dim);
        errdefer allocator.free(self.csa_comp_score);
        self.csa_k = try allocator.alloc(f32, nl * comp_slots * kd);
        errdefer allocator.free(self.csa_k);
        self.csa_score_scratch = try allocator.alloc(f32, max_comp_dim);
        errdefer allocator.free(self.csa_score_scratch);

        // Lightning Indexer: probe for tensors to detect availability.
        // Indexer is only used on CSA layers (ratio=4) when compressed block count > index_topk.
        const ihd: usize = self.index_head_dim;
        const inh: usize = self.index_n_heads;
        const itk: usize = self.index_topk;
        if (f.getTensor("blk.2.attn_indexer_q_b.weight") != null) {
            self.lid_comp_k = try allocator.alloc(f32, nl * comp_slots * ihd);
            self.lid_query = try allocator.alloc(f32, inh * ihd);
            self.lid_head_w = try allocator.alloc(f32, inh);
            self.lid_scores = try allocator.alloc(f32, comp_slots);
            self.lid_topk_ids = try allocator.alloc(u32, itk);
            self.lid_enabled = true;
        }
        // Block-scoped errdefers above expire when the if-block ends normally.
        // Re-register at function scope so later allocation failures still free these.
        errdefer if (self.lid_comp_k.len > 0) allocator.free(self.lid_comp_k);
        errdefer if (self.lid_query.len > 0) allocator.free(self.lid_query);
        errdefer if (self.lid_head_w.len > 0) allocator.free(self.lid_head_w);
        errdefer if (self.lid_scores.len > 0) allocator.free(self.lid_scores);
        errdefer if (self.lid_topk_ids.len > 0) allocator.free(self.lid_topk_ids);

        // Prefill buffers: deferred, DS4 prefill is sequential (see prefill() doc comment).
        // Fields stay as empty slices; allocate on first batched-prefill use (future).
        // Saves ~64 MB page_allocator memory that was allocated but never touched.

        self.warmNormCache();
        return self;
    }

    /// Pre-dequantize all norm + HC weights at init time to avoid per-token allocations.
    /// HC fn weights (hc_attn_fn, hc_ffn_fn, output_hc_fn) are dequantized to f32
    /// so the GPU HC mixing kernel can access them directly.
    fn warmNormCache(self: *Ds4Model) void {
        const e = self.n_embd;
        _ = self.normAsF32OrNull(self.fmt.getTensor("output_norm.weight"), e);
        // Output HC head: fn[n_hc × flat_size], base[n_hc], scale[1]
        _ = self.normAsF32OrNull(self.fmt.getTensor("output_hc_fn.weight"), n_hc * n_hc * e);
        _ = self.normAsF32OrNull(self.fmt.getTensor("output_hc_base.weight"), n_hc);
        _ = self.normAsF32OrNull(self.fmt.getTensor("output_hc_scale.weight"), 1);
        for (0..self.n_layers) |li| {
            _ = self.normAsF32OrNull(self.layerTensor(li, "attn_norm.weight"), e);
            _ = self.normAsF32OrNull(self.layerTensor(li, "ffn_norm.weight"), e);
            _ = self.normAsF32OrNull(self.layerTensor(li, "attn_q_a_norm.weight"), self.q_lora_rank);
            _ = self.normAsF32OrNull(self.layerTensor(li, "attn_kv_a_norm.weight"), self.kv_lora_rank);
            // HC weights: dequant fn to f32 for GPU kernel (24 × 4*n_embd = 393K elems)
            _ = self.normAsF32OrNull(self.layerTensor(li, "hc_attn_fn.weight"), hc_mix_dim * n_hc * e);
            _ = self.normAsF32OrNull(self.layerTensor(li, "hc_attn_base.weight"), hc_mix_dim);
            _ = self.normAsF32OrNull(self.layerTensor(li, "hc_attn_scale.weight"), 3);
            _ = self.normAsF32OrNull(self.layerTensor(li, "hc_ffn_fn.weight"), hc_mix_dim * n_hc * e);
            _ = self.normAsF32OrNull(self.layerTensor(li, "hc_ffn_base.weight"), hc_mix_dim);
            _ = self.normAsF32OrNull(self.layerTensor(li, "hc_ffn_scale.weight"), 3);
            _ = self.normAsF32OrNull(self.layerTensor(li, "attn_compressor_norm.weight"), self.kv_lora_rank);
            _ = self.normAsF32OrNull(self.layerTensor(li, "attn_sinks.weight"), self.n_head);
        }
    }

    /// Release all heap allocations owned by this model, including norm cache,
    /// working buffers, KV cache, compressor state, and LID indexer buffers.
    pub fn deinit(self: *Ds4Model) void {
        self.be.sync();
        for (self.norm_cache[0..self.norm_cache_len]) |e| self.allocator.free(e.data);
        const a = self.allocator;
        inline for (.{
            &self.hc_state,        &self.new_hc,           &self.hc_mixes,       &self.hc_pre_w,       &self.hc_post_w,
            &self.hc_comb,         &self.hidden,           &self.hidden2,        &self.flat_norm,      &self.q_compressed,
            &self.q_full,          &self.kv_proj,          &self.scores_buf,     &self.attn_out,       &self.lora_out,
            &self.attn_result,     &self.ff_gate,          &self.ff_up,          &self.ff_down,        &self.expert_accum,
            &self.expert_scratch,  &self.ff_gate_scratch,  &self.ff_up_scratch,  &self.router_logits,  &self.logits_buf,
            &self.kv_k_bytes,      &self.csa_comp_kv,      &self.csa_comp_score, &self.csa_k,          &self.csa_score_scratch,
            &self.lid_comp_k,      &self.lid_query,        &self.lid_head_w,     &self.lid_scores,     &self.rope_cos_buf,
            &self.rope_sin_buf,    &self.gpu_slot_weights, &self.expert_pool,    &self.gpu_top_ids,
            &self.gpu_top_weights,
        }) |buf| a.free(buf.*);
        if (self.lid_topk_ids.len > 0) a.free(self.lid_topk_ids);
        // Prefill buffers (page_allocator), currently empty slices (allocation deferred).
        // Guards remain for forward compatibility when batched prefill is implemented.
        {
            const pa = std.heap.page_allocator;
            const pf_bufs = .{
                &self.pf_hidden, &self.pf_hidden2, &self.pf_q_a,
                &self.pf_q,      &self.pf_kv_proj, &self.pf_attn_out,
            };
            inline for (pf_bufs) |buf| if (buf.len > 0) pa.free(buf.*);
            if (self.pf_positions.len > 0) pa.free(self.pf_positions);
        }
    }

    /// Wrap this model in the generic `Model` vtable interface for backend-agnostic dispatch.
    pub fn model(self: *Ds4Model) Model {
        return Model.from(Ds4Model, self);
    }

    // ── Tensor lookup ─────────────────────────────────────────────

    fn layerTensor(self: *Ds4Model, li: usize, suffix: []const u8) ?TensorInfo {
        const name = std.fmt.bufPrint(&self.name_buf, "blk.{d}.{s}", .{ li, suffix }) catch return null;
        return self.fmt.getTensor(name);
    }

    fn layerTensorReq(self: *Ds4Model, li: usize, suffix: []const u8) !TensorInfo {
        return self.layerTensor(li, suffix) orelse {
            std.log.err("ds4: missing blk.{d}.{s}", .{ li, suffix });
            return error.MissingTensor;
        };
    }

    /// Learned per-head attention sink logits, dequantized to f32. Null if the
    /// layer has no `attn_sinks` tensor. Sinks absorb softmax mass and have no V.
    fn layerSinks(self: *Ds4Model, li: usize) ?[*]const f32 {
        const t = self.layerTensor(li, "attn_sinks.weight") orelse return null;
        return self.normAsF32(t, self.n_head);
    }

    fn getTensorReq(self: *Ds4Model, name: []const u8) !TensorInfo {
        return self.fmt.getTensor(name) orelse {
            std.log.err("ds4: missing {s}", .{name});
            return error.MissingTensor;
        };
    }

    // ── Norm weight cache ─────────────────────────────────────────

    fn normAsF32OrNull(self: *Ds4Model, t_opt: ?TensorInfo, n: usize) ?[*]const f32 {
        const t = t_opt orelse return null;
        return self.normAsF32(t, n);
    }

    fn normAsF32(self: *Ds4Model, t: TensorInfo, n: usize) [*]const f32 {
        if (t.dtype == .f32) return @ptrCast(@alignCast(t.data_ptr));
        const key = @intFromPtr(t.data_ptr);
        for (self.norm_cache[0..self.norm_cache_len]) |entry| {
            if (entry.key == key) return entry.data.ptr;
        }
        if (self.norm_cache_len >= max_norm_entries) @panic("ds4: norm cache overflow");
        const buf = self.allocator.alloc(f32, n) catch @panic("ds4: norm alloc");
        quant_ops.dequantToF32(buf, t.data_ptr, t.dtype, n);
        self.norm_cache[self.norm_cache_len] = .{ .key = key, .data = buf };
        self.norm_cache_len += 1;
        return buf.ptr;
    }

    // ── KV cache helpers ──────────────────────────────────────────

    fn kvLayerBytes(self: *Ds4Model) usize {
        return kv_quant.kvByteOffset(self.kv_type, self.max_seq_len * self.kv_lora_rank);
    }

    fn kvKLayer(self: *Ds4Model, li: usize) []u8 {
        const layer_bytes = self.kvLayerBytes();
        return self.kv_k_bytes[li * layer_bytes ..][0..layer_bytes];
    }

    // ── Hyper Connection ──────────────────────────────────────────

    /// Compute HC pre-weights and sublayer input in `self.hidden`.
    fn hcPre(
        self: *Ds4Model,
        hc_fn: TensorInfo,
        hc_base: TensorInfo,
        hc_scale: TensorInfo,
    ) void {
        self.be.sync();
        self.hcPreCpu(hc_fn, hc_base, hc_scale);
    }

    /// CPU fallback for hcPre (non-Metal backends).
    fn hcPreCpu(
        self: *Ds4Model,
        hc_fn: TensorInfo,
        hc_base: TensorInfo,
        hc_scale: TensorInfo,
    ) void {
        const e = self.n_embd;
        const flat_size = n_hc * e;
        const rms_inv = blk: {
            const V8 = @Vector(8, f32);
            var acc: V8 = @splat(0.0);
            var ri: usize = 0;
            while (ri + 8 <= flat_size) : (ri += 8) {
                const v: V8 = self.hc_state[ri..][0..8].*;
                acc = @mulAdd(V8, v, v, acc);
            }
            var ss: f32 = @reduce(.Add, acc);
            while (ri < flat_size) : (ri += 1) ss += self.hc_state[ri] * self.hc_state[ri];
            break :blk 1.0 / @sqrt(ss / @as(f32, @floatFromInt(flat_size)) + self.rms_eps);
        };

        if (self.be == .cuda) {
            // CUDA: run the HC GEMV on the GPU (weights cached in buf_cache
            // after first touch). The CPU branches below read the mmap'd
            // weights directly, which demand-page at 4KB-fault speed when
            // memory pressure evicts them (measured 10s/token on GB10).
            self.doGemv(self.hc_state.ptr, hc_fn, self.hc_mixes.ptr, hc_mix_dim, flat_size);
            self.be.sync();
        } else if (hc_fn.dtype == .q8_0) {
            cpuGemvQ8_0(hc_fn.data_ptr, self.hc_state, self.hc_mixes, flat_size);
        } else if (hc_fn.dtype == .f32) {
            cpuGemvF32(hc_fn.data_ptr, self.hc_state, self.hc_mixes[0..hc_mix_dim], flat_size);
        } else if (hc_fn.dtype == .bf16) {
            cpuGemvBf16(hc_fn.data_ptr, self.hc_state, self.hc_mixes[0..hc_mix_dim], flat_size);
        } else if (hc_fn.dtype == .f16) {
            cpuGemvF16(hc_fn.data_ptr, self.hc_state, self.hc_mixes[0..hc_mix_dim], flat_size);
        } else {
            @memcpy(self.flat_norm, self.hc_state);
            self.doGemv(self.flat_norm.ptr, hc_fn, self.hc_mixes.ptr, hc_mix_dim, flat_size);
            self.be.sync();
        }
        for (self.hc_mixes[0..hc_mix_dim]) |*m| m.* *= rms_inv;

        const base = self.normAsF32(hc_base, hc_mix_dim);
        const scale = self.normAsF32(hc_scale, 3);
        const mixes = self.hc_mixes;
        for (0..n_hc) |s| {
            self.hc_pre_w[s] = sigmoid(mixes[s] * scale[0] + base[s]) + hc_eps;
            self.hc_post_w[s] = sigmoid(mixes[n_hc + s] * scale[1] + base[n_hc + s]) * 2.0;
        }
        for (0..n_hc * n_hc) |s| {
            self.hc_comb[s] = mixes[2 * n_hc + s] * scale[2] + base[2 * n_hc + s];
        }
        hcSinkhorn(self.hc_comb);
        const V8 = @Vector(8, f32);
        var i: usize = 0;
        while (i + 8 <= e) : (i += 8) {
            var acc: V8 = @splat(@as(f32, 0.0));
            for (0..n_hc) |s| {
                const w: V8 = @splat(self.hc_pre_w[s]);
                acc = @mulAdd(V8, @as(V8, self.hc_state[s * e + i ..][0..8].*), w, acc);
            }
            self.hidden[i..][0..8].* = acc;
        }
        while (i < e) : (i += 1) {
            var v: f32 = 0.0;
            for (0..n_hc) |s| v += self.hc_state[s * e + i] * self.hc_pre_w[s];
            self.hidden[i] = v;
        }
    }

    /// Update HC state after a sublayer. Sublayer output must be in `self.hidden`.
    fn hcPost(self: *Ds4Model) void {
        const e = self.n_embd;

        self.be.sync();
        const sub = self.hidden;
        const V8 = @Vector(8, f32);
        for (0..n_hc) |dst| {
            const ns = self.new_hc[dst * e ..][0..e];
            const pw: V8 = @splat(self.hc_post_w[dst]);
            var cvec: [n_hc]V8 = undefined;
            for (0..n_hc) |src| cvec[src] = @splat(self.hc_comb[dst + src * n_hc]);
            var i: usize = 0;
            while (i + 8 <= e) : (i += 8) {
                var acc: V8 = @as(V8, sub[i..][0..8].*) * pw;
                for (0..n_hc) |src| {
                    acc = @mulAdd(V8, @as(V8, self.hc_state[src * e + i ..][0..8].*), cvec[src], acc);
                }
                ns[i..][0..8].* = acc;
            }
            while (i < e) : (i += 1) {
                var v = sub[i] * self.hc_post_w[dst];
                for (0..n_hc) |src| v += self.hc_state[src * e + i] * self.hc_comb[dst + src * n_hc];
                ns[i] = v;
            }
        }
        const tmp = self.hc_state;
        self.hc_state = self.new_hc;
        self.new_hc = tmp;
    }

    /// HC head: merge 4 streams → self.hidden.
    fn hcHead(self: *Ds4Model, hc_fn: TensorInfo, hc_base: TensorInfo, hc_scale: TensorInfo) void {
        const e = self.n_embd;

        self.be.sync();
        const flat_size_h = n_hc * e;
        const rms_inv_h = blk: {
            const V8 = @Vector(8, f32);
            var acc: V8 = @splat(0.0);
            var ri: usize = 0;
            while (ri + 8 <= flat_size_h) : (ri += 8) {
                const v: V8 = self.hc_state[ri..][0..8].*;
                acc = @mulAdd(V8, v, v, acc);
            }
            var ss: f32 = @reduce(.Add, acc);
            while (ri < flat_size_h) : (ri += 1) ss += self.hc_state[ri] * self.hc_state[ri];
            break :blk 1.0 / @sqrt(ss / @as(f32, @floatFromInt(flat_size_h)) + self.rms_eps);
        };
        if (hc_fn.dtype == .q8_0) {
            cpuGemvQ8_0(hc_fn.data_ptr, self.hc_state, self.hc_pre_w, flat_size_h);
        } else if (hc_fn.dtype == .f32) {
            cpuGemvF32(hc_fn.data_ptr, self.hc_state, self.hc_pre_w[0..n_hc], flat_size_h);
        } else if (hc_fn.dtype == .bf16) {
            cpuGemvBf16(hc_fn.data_ptr, self.hc_state, self.hc_pre_w[0..n_hc], flat_size_h);
        } else if (hc_fn.dtype == .f16) {
            cpuGemvF16(hc_fn.data_ptr, self.hc_state, self.hc_pre_w[0..n_hc], flat_size_h);
        } else {
            @memcpy(self.flat_norm, self.hc_state);
            self.doGemv(self.flat_norm.ptr, hc_fn, self.hc_pre_w.ptr, n_hc, flat_size_h);
            self.be.sync();
        }
        for (self.hc_pre_w[0..n_hc]) |*m| m.* *= rms_inv_h;
        const base = self.normAsF32(hc_base, n_hc);
        const scale = self.normAsF32(hc_scale, 1);
        for (0..n_hc) |s| {
            self.hc_pre_w[s] = sigmoid(self.hc_pre_w[s] * scale[0] + base[s]) + hc_eps;
        }
        @memset(self.hidden, 0.0);
        for (0..n_hc) |s| {
            const stream = self.hc_state[s * e ..][0..e];
            const w = self.hc_pre_w[s];
            for (0..e) |i| self.hidden[i] += stream[i] * w;
        }
    }

    // ── Attention layer ───────────────────────────────────────────

    fn attentionLayer(self: *Ds4Model, li: usize) !void {
        const e = self.n_embd;
        const nh: usize = self.n_head;
        const kd: usize = self.kv_lora_rank;
        const ql: usize = self.q_lora_rank;
        const rd: usize = self.rope_dim;
        const nope: usize = kd - rd;
        const pos = self.kv_seq_len;
        const max_comp_dim: usize = 2 * kd; // buffer stride (max of CSA=1024 and HCA=512)

        // All Q, KV, and (if CSA) compressor GPU ops in ONE command buffer → 1 sync
        const nw = try self.layerTensorReq(li, "attn_norm.weight");
        self.cpu.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        const q_a = try self.layerTensorReq(li, "attn_q_a.weight");
        self.doGemv(self.hidden2.ptr, q_a, self.q_compressed.ptr, ql, e);
        const q_an = try self.layerTensorReq(li, "attn_q_a_norm.weight");
        self.cpu.rmsNorm(self.q_compressed.ptr, self.normAsF32(q_an, ql), self.q_compressed.ptr, ql, self.rms_eps);
        const q_b = try self.layerTensorReq(li, "attn_q_b.weight");
        self.doGemv(self.q_compressed.ptr, q_b, self.q_full.ptr, nh * kd, ql);
        const kv_a = try self.layerTensorReq(li, "attn_kv.weight");
        self.doGemv(self.hidden2.ptr, kv_a, self.kv_proj.ptr, kd, e);
        const kv_an = try self.layerTensorReq(li, "attn_kv_a_norm.weight");
        self.cpu.rmsNorm(self.kv_proj.ptr, self.normAsF32(kv_an, kd), self.kv_proj.ptr, kd, self.rms_eps);

        // Compressor projections for all compressed layers (CSA ratio=4, HCA ratio=128).
        // Both batched with Q+KV in same GPU command buffer, single sync covers all.
        // Circular buffer: per-token projections indexed by pos % comp_buf_ratio (128).
        const comp_buf_ratio: usize = 128; // circular buffer stride per layer
        const comp_layer_stride = comp_buf_ratio * max_comp_dim;
        var comp_kv_pos: []f32 = &.{};
        var comp_score_pos: []f32 = &.{};
        var actual_comp_dim: usize = 0;
        const ratio = self.compress_ratios[li];
        if (ratio != 0) {
            if (self.layerTensor(li, "attn_compressor_kv.weight")) |wkv| {
                const kwgate = self.layerTensor(li, "attn_compressor_gate.weight") orelse return error.MissingTensor;
                actual_comp_dim = @min(@as(usize, @intCast(wkv.dims[0])), @as(usize, @intCast(wkv.dims[1])));
                const circ_pos = pos % comp_buf_ratio;
                comp_kv_pos = self.csa_comp_kv[li * comp_layer_stride + circ_pos * max_comp_dim ..][0..actual_comp_dim];
                comp_score_pos = self.csa_comp_score[li * comp_layer_stride + circ_pos * max_comp_dim ..][0..actual_comp_dim];
                self.doGemv(self.hidden2.ptr, wkv, comp_kv_pos.ptr, actual_comp_dim, e);
                self.doGemv(self.hidden2.ptr, kwgate, comp_score_pos.ptr, actual_comp_dim, e);
            }
        }

        // Pre-dispatch LID GEMVs into same GPU command buffer, eliminates 1 sync per CSA layer.
        // Inputs (q_compressed, hidden) are already in the pipeline; GPU ordering guarantees correctness.
        const lid_pre_dispatched: bool = blk: {
            if (!self.lid_enabled or ratio != @as(u32, csa_compress_ratio)) break :blk false;
            const n_comp_early: usize = (pos + 1) / ratio;
            if (n_comp_early <= self.index_topk) break :blk false;
            const inh: usize = self.index_n_heads;
            const ihd: usize = self.index_head_dim;
            if (self.layerTensor(li, "attn_indexer_q_b.weight")) |wiq| {
                self.doGemv(self.q_compressed.ptr, wiq, self.lid_query.ptr, inh * ihd, self.q_lora_rank);
            } else break :blk false;
            if (self.layerTensor(li, "attn_indexer_proj.weight")) |ww| {
                self.doGemv(self.hidden.ptr, ww, self.lid_head_w.ptr, inh, self.n_embd);
            } else {
                for (self.lid_head_w[0..inh]) |*w| w.* = 1.0 / @as(f32, @floatFromInt(inh));
            }
            break :blk true;
        };

        self.be.sync();
        for (0..nh) |h| plainRmsNorm(self.q_full[h * kd ..][0..kd], self.rms_eps);

        // RoPE cos/sin from pre-computed freq bases, SIMD vectorized.
        const nd = rd / 2;
        // Compressed layers (ratio != 0) use compress_rope_freq (160000) for main attention
        // Q and KV RoPE. Non-compressed layers use the standard rope_freq (10000).
        // This matches llama.cpp's use_compress_rope logic.
        const freqs = if (self.compress_ratios[li] != 0) &self.compress_rope_freqs else &self.rope_freqs;
        var rope_cos: [32]f32 = undefined;
        var rope_sin: [32]f32 = undefined;
        {
            const V8f = @Vector(8, f32);
            const pos_v: V8f = @splat(@floatFromInt(pos));
            var i: usize = 0;
            while (i + 8 <= nd) : (i += 8) {
                const fv: V8f = freqs[i..][0..8].*;
                const theta: V8f = pos_v * fv;
                const cv: V8f = @cos(theta);
                const sv: V8f = @sin(theta);
                rope_cos[i..][0..8].* = cv;
                rope_sin[i..][0..8].* = sv;
            }
            while (i < nd) : (i += 1) {
                const theta = @as(f32, @floatFromInt(pos)) * freqs[i];
                rope_cos[i] = @cos(theta);
                rope_sin[i] = @sin(theta);
            }
        }

        for (0..nh) |h| applyRopeTable(self.q_full[h * kd + nope ..][0..rd], rope_cos[0..nd], rope_sin[0..nd]);
        applyRopeTable(self.kv_proj[nope..][0..rd], rope_cos[0..nd], rope_sin[0..nd]);

        // Compressor: GPU projections done above; do APE + group compression here.
        // Works for both CSA (ratio=4, actual_comp_dim=1024) and HCA (ratio=128, actual_comp_dim=512).
        if (ratio != 0 and actual_comp_dim > 0) {
            {
                const ape = self.layerTensor(li, "attn_compressor_ape.weight");

                // Add APE (absolute position encoding) for position within group
                if (ape) |a| {
                    // ape dims=[actual_comp_dim, ratio]; column pos%ratio is the encoding
                    const group_pos = pos % ratio;
                    // APE stored column-major: ape[:, group_pos]
                    const ape_col: [*]const u8 = a.data_ptr + group_pos * backend_mod.weightBytes(a.dtype, 1, actual_comp_dim);
                    var ape_f32: [2048]f32 = undefined;
                    std.debug.assert(actual_comp_dim <= ape_f32.len); // model comp_dim exceeds buffer
                    quant_ops.dequantToF32(&ape_f32, ape_col, a.dtype, actual_comp_dim);
                    for (comp_score_pos, ape_f32[0..actual_comp_dim]) |*s, av| s.* += av;
                }

                // Compress complete group of `ratio` tokens
                if ((pos + 1) % ratio == 0 and pos >= ratio - 1) {
                    const group_start = pos + 1 - ratio;
                    const group_idx = (pos + 1) / ratio - 1;
                    const comp_slots = compSlotsPerLayer(self.max_seq_len);

                    // Softmax-weighted compression into compressed[0..kd].
                    // CSA (ratio=4): overlap pool, 8 candidates. HCA (ratio=128): per-dim softmax.
                    var compressed: [2048]f32 = undefined;
                    std.debug.assert(actual_comp_dim <= compressed.len);
                    // Circular buffer: group_start % comp_buf_ratio gives the start slot.
                    // Groups are ratio-aligned, so slots within a group are consecutive
                    // modulo comp_buf_ratio (which is >= max(ratio)=128).
                    const circ_base = li * comp_layer_stride + (group_start % comp_buf_ratio) * max_comp_dim;
                    if (ratio == @as(u32, csa_compress_ratio)) {
                        std.debug.assert(actual_comp_dim >= kd * 2);
                        // Overlap pool: previous low-half + current high-half → kd outputs.
                        var prev_kv: ?[*]const f32 = null;
                        var prev_score: ?[*]const f32 = null;
                        if (group_start >= csa_compress_ratio) {
                            const prev_start = group_start - csa_compress_ratio;
                            const prev_circ = li * comp_layer_stride + (prev_start % comp_buf_ratio) * max_comp_dim;
                            prev_kv = self.csa_comp_kv[prev_circ..].ptr;
                            prev_score = self.csa_comp_score[prev_circ..].ptr;
                        }
                        csaOverlapPool(
                            compressed[0..kd],
                            self.csa_comp_kv[circ_base..].ptr,
                            self.csa_comp_score[circ_base..].ptr,
                            prev_kv,
                            prev_score,
                            max_comp_dim,
                        );
                    } else {
                        // HCA path (ratio=128): SIMD across 8 dimensions at a time.
                        // Two-pass per chunk: (1) find max over tokens, (2) fused exp+sum+weighted-kv.
                        // 12× fewer iterations than scalar per-dimension loop with better cache behavior.
                        const V8c = @Vector(8, f32);
                        const neg_inf_v: V8c = @splat(-std.math.inf(f32));
                        var d: usize = 0;
                        while (d + 8 <= actual_comp_dim) : (d += 8) {
                            // Pass 1: max across all tokens (contiguous 32B loads per token)
                            var mx_v: V8c = neg_inf_v;
                            for (0..ratio) |t| {
                                const off = circ_base + t * max_comp_dim + d;
                                const sv: V8c = self.csa_comp_score[off..][0..8].*;
                                mx_v = @max(mx_v, sv);
                            }
                            // Pass 2: fused exp + sum + weighted KV accumulation
                            var sum_v: V8c = @splat(@as(f32, 0.0));
                            var acc_v: V8c = @splat(@as(f32, 0.0));
                            for (0..ratio) |t| {
                                const off = circ_base + t * max_comp_dim + d;
                                const ev = @exp(@as(V8c, self.csa_comp_score[off..][0..8].*) - mx_v);
                                sum_v += ev;
                                acc_v = @mulAdd(V8c, ev, @as(V8c, self.csa_comp_kv[off..][0..8].*), acc_v);
                            }
                            compressed[d..][0..8].* = acc_v / sum_v;
                        }
                        // Scalar tail for non-8-aligned remainder
                        while (d < actual_comp_dim) : (d += 1) {
                            var mx: f32 = -std.math.inf(f32);
                            for (0..ratio) |t| {
                                const sv = self.csa_comp_score[circ_base + t * max_comp_dim + d];
                                if (sv > mx) mx = sv;
                            }
                            var sm: f32 = 0;
                            var acc: f32 = 0;
                            for (0..ratio) |t| {
                                const off = circ_base + t * max_comp_dim + d;
                                const ev = @exp(self.csa_comp_score[off] - mx);
                                sm += ev;
                                acc += ev * self.csa_comp_kv[off];
                            }
                            compressed[d] = acc / sm;
                        }
                    }

                    // Apply RMS norm + scale to first [0..kd=512] of compressed vector
                    if (self.layerTensor(li, "attn_compressor_norm.weight")) |cn| {
                        const norm_w = self.normAsF32(cn, kd);
                        var comp_first = compressed[0..kd];
                        plainRmsNorm(comp_first, self.rms_eps);
                        for (0..kd) |i| comp_first[i] *= norm_w[i];
                    }
                    const comp_rope = compressed[nope..kd];

                    // RoPE on compressed rope portion using group start position.
                    // Uses pre-computed compress_rope_freqs (no pow() per token).
                    const comp_pos = group_start;
                    if (comp_pos == pos) {
                        applyRopeTable(comp_rope[0..rd], rope_cos[0..nd], rope_sin[0..nd]);
                    } else {
                        var cg_cos: [32]f32 = undefined;
                        var cg_sin: [32]f32 = undefined;
                        const V8f = @Vector(8, f32);
                        const cp_v: V8f = @splat(@floatFromInt(comp_pos));
                        var ci: usize = 0;
                        while (ci + 8 <= nd) : (ci += 8) {
                            const fv: V8f = self.compress_rope_freqs[ci..][0..8].*;
                            const theta: V8f = cp_v * fv;
                            const cv: V8f = @cos(theta);
                            const sv: V8f = @sin(theta);
                            cg_cos[ci..][0..8].* = cv;
                            cg_sin[ci..][0..8].* = sv;
                        }
                        while (ci < nd) : (ci += 1) {
                            const theta = @as(f32, @floatFromInt(comp_pos)) * self.compress_rope_freqs[ci];
                            cg_cos[ci] = @cos(theta);
                            cg_sin[ci] = @sin(theta);
                        }
                        applyRopeTable(comp_rope[0..rd], cg_cos[0..nd], cg_sin[0..nd]);
                    }

                    // Store final compressed KV [kd=512] in CSA cache
                    const csa_off = (li * comp_slots + group_idx) * kd;
                    @memcpy(self.csa_k[csa_off..][0..kd], compressed[0..kd]);

                    // Lightning Indexer: compress indexer keys (CSA layers only).
                    // Same compression procedure as main compressor but with separate
                    // indexer_compressor weights → produces K^IComp [index_head_dim].
                    if (self.lid_enabled and ratio == @as(u32, csa_compress_ratio)) {
                        self.lidCompressGroup(li, group_start, group_idx, comp_slots);
                    }
                }
            }
        }

        // Mixed attention when compressed groups exist: SWA raw KV + compressed + sinks.
        // Uncompressed (ratio=0) and HCA before the first completed group use backend
        // SDPA, which matches the quality-verified Flash 0731 path. Sinks on that
        // path collapsed the first-layer residual (L2 32→26) and destroyed greedy
        // factoid accuracy, so they stay on the compressed path only.
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));
        const kv_k_layer = self.kvKLayer(li);
        const n_comp_groups: usize = if (ratio != 0) (pos + 1) / ratio else 0;

        if (n_comp_groups > 0) {
            const k_byte_off = kv_quant.kvByteOffset(self.kv_type, pos * kd);
            kv_quant.kvStore(kv_k_layer[k_byte_off..].ptr, self.kv_proj.ptr, kd, self.kv_type);

            const comp_slots = compSlotsPerLayer(self.max_seq_len);
            const itk: usize = self.index_topk;
            const use_lid = self.lid_enabled and ratio == @as(u32, csa_compress_ratio) and n_comp_groups > itk;
            var n_attend_comp: usize = n_comp_groups;
            if (use_lid) {
                self.lidScoreAndSelect(li, n_comp_groups, comp_slots, lid_pre_dispatched);
                n_attend_comp = @min(itk, n_comp_groups);
            }

            const raw_start = rawAttnStart(pos);
            const raw_count = pos + 1 - raw_start;
            const sl_total = raw_count + n_attend_comp;
            const kv_elem_bytes = kv_quant.kvByteOffset(self.kv_type, kd);
            var ctx = CompressedAttnCtx{
                .q_full = self.q_full,
                .scores_buf = self.scores_buf,
                .attn_out = self.attn_out,
                .kv_k_layer = kv_k_layer,
                .kv_v_layer = kv_k_layer,
                .csa_k = self.csa_k,
                .lid_topk_ids = if (use_lid) self.lid_topk_ids else &.{},
                .sink_data = self.layerSinks(li),
                .kd = kd,
                .pos = pos,
                .raw_start = raw_start,
                .raw_count = raw_count,
                .ss = self.score_stride,
                .kv_elem_bytes = kv_elem_bytes,
                .n_attend_comp = n_attend_comp,
                .sl_total = sl_total,
                .comp_slots = comp_slots,
                .li = li,
                .scale = scale,
                .use_lid = use_lid,
                .kv_quant_type = self.kv_type,
            };
            if (self.pool) |pool| {
                pool.parallelFor(nh, 1, @ptrCast(&ctx), CompressedAttnCtx.perHeadFn);
            } else {
                for (0..nh) |h| ctx.processHead(h);
            }
        } else {
            if (pos + 1 > ds4_raw_attn_window) {
                attn_ops.scaledDotProductAttention(
                    self.q_full.ptr,
                    kv_k_layer,
                    kv_k_layer,
                    self.kv_proj,
                    self.kv_proj,
                    self.attn_out.ptr,
                    self.scores_buf.ptr,
                    nh,
                    1,
                    kd,
                    pos,
                    scale,
                    self.computeBackend(),
                    .{ .start = pos + 1 - ds4_raw_attn_window, .len = ds4_raw_attn_window },
                    0,
                    self.kv_type,
                    self.kv_type,
                );
            } else {
                attn_ops.scaledDotProductAttention(
                    self.q_full.ptr,
                    kv_k_layer,
                    kv_k_layer,
                    self.kv_proj,
                    self.kv_proj,
                    self.attn_out.ptr,
                    self.scores_buf.ptr,
                    nh,
                    1,
                    kd,
                    pos,
                    scale,
                    self.computeBackend(),
                    null,
                    0,
                    self.kv_type,
                    self.kv_type,
                );
            }
        }

        self.be.sync();
        for (0..nh) |h| applyRopeInverseTable(self.attn_out[h * kd + nope ..][0..rd], rope_cos[0..nd], rope_sin[0..nd]);

        // Output LoRA: grouped wo_a [n_in=4096 per group, n_out=o_lora_rank] × 8 groups
        const og: usize = self.o_groups;
        const olr: usize = self.o_lora_rank;
        const group_in: usize = nh * kd / og; // = 64*512/8 = 4096
        const wo_a = try self.layerTensorReq(li, "attn_output_a.weight");
        // Per-group stride: MLX-Q packs as u32 words (dims-based), others use weightBytes.
        const group_stride = if (wo_a.dtype == .mlx_q)
            ds4ExpertStride(wo_a, og)
        else blk: {
            const row_bytes = backend_mod.weightBytes(wo_a.dtype, 1, group_in);
            break :blk olr * row_bytes;
        };
        // wo_a groups: 8 independent GEMVs, batch for no barriers.
        self.gemvBackend().beginBatch();
        for (0..og) |g| {
            const xp = self.attn_out.ptr + g * group_in;
            const yp = self.lora_out.ptr + g * olr;
            self.doGemvExpert(xp, wo_a, g, group_stride, yp, olr, group_in);
        }
        self.gemvBackend().endBatch();
        const wo_b = try self.layerTensorReq(li, "attn_output_b.weight");
        // Write wo_b output directly to hidden (avoids 16KB attn_result → hidden copy)
        self.doGemv(self.lora_out.ptr, wo_b, self.hidden.ptr, e, og * olr);
        self.be.sync();
    }

    // ── Compressed attention parallel dispatch context ─────────

    const CompressedAttnCtx = struct {
        q_full: []f32,
        scores_buf: []f32,
        attn_out: []f32,
        kv_k_layer: []u8,
        kv_v_layer: []u8,
        csa_k: []f32,
        lid_topk_ids: []u32,
        sink_data: ?[*]const f32,
        kd: usize,
        pos: usize,
        raw_start: usize,
        raw_count: usize,
        ss: usize,
        kv_elem_bytes: usize,
        n_attend_comp: usize,
        sl_total: usize,
        comp_slots: usize,
        li: usize,
        scale: f32,
        use_lid: bool,
        kv_quant_type: KvQuantType,

        fn perHeadFn(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *CompressedAttnCtx = @ptrCast(@alignCast(ctx_ptr));
            for (start..end) |h| {
                ctx.processHead(h);
            }
        }

        fn processHead(ctx: *const CompressedAttnCtx, h: usize) void {
            const kd = ctx.kd;
            const V8 = @Vector(8, f32);
            const q_h = ctx.q_full[h * kd ..][0..kd];
            const scores_h = ctx.scores_buf[h * ctx.ss ..];
            const raw_start = ctx.raw_start;
            const raw_count = ctx.raw_count;
            // Packed scores: [0, raw_count) = SWA raw KV, then compressed groups.
            var running_max: f32 = -std.math.inf(f32);
            const eb = ctx.kv_elem_bytes;
            for (0..raw_count) |ri| {
                const t = raw_start + ri;
                const k_ptr = ctx.kv_k_layer[t * eb ..].ptr;
                if (ri + 1 < raw_count) {
                    @prefetch(ctx.kv_k_layer[(t + 1) * eb ..].ptr, .{ .locality = 3 });
                }
                const s = kv_quant.kvDot(q_h.ptr, k_ptr, kd, ctx.kv_quant_type) * ctx.scale;
                scores_h[ri] = s;
                running_max = @max(running_max, s);
            }
            for (0..ctx.n_attend_comp) |gi| {
                const g = if (ctx.use_lid) ctx.lid_topk_ids[gi] else @as(u32, @intCast(gi));
                const ck = ctx.csa_k[(ctx.li * ctx.comp_slots + g) * kd ..][0..kd];
                var acc: V8 = @splat(0.0);
                var i: usize = 0;
                while (i + 8 <= kd) : (i += 8) {
                    acc = @mulAdd(V8, @as(V8, q_h[i..][0..8].*), @as(V8, ck[i..][0..8].*), acc);
                }
                var dot = @reduce(.Add, acc);
                while (i < kd) : (i += 1) dot = @mulAdd(f32, q_h[i], ck[i], dot);
                const s = dot * ctx.scale;
                scores_h[raw_count + gi] = s;
                running_max = @max(running_max, s);
            }
            var sm: f32 = undefined;
            {
                const scores_sl = scores_h[0..ctx.sl_total];
                var mx = running_max;
                if (ctx.sink_data) |sd| mx = @max(mx, sd[h]);
                const mx_splat: V8 = @splat(mx);
                var sum_v: V8 = @splat(@as(f32, 0.0));
                var si: usize = 0;
                while (si + 8 <= ctx.sl_total) : (si += 8) {
                    const ev = @exp(@as(V8, scores_sl[si..][0..8].*) - mx_splat);
                    scores_sl[si..][0..8].* = ev;
                    sum_v += ev;
                }
                sm = @reduce(.Add, sum_v);
                while (si < ctx.sl_total) : (si += 1) {
                    scores_sl[si] = @exp(scores_sl[si] - mx);
                    sm += scores_sl[si];
                }
                if (ctx.sink_data) |sd| sm += @exp(sd[h] - mx);
            }
            const sparse_threshold_unnorm = sparse_v_threshold * sm;
            const ao_h = ctx.attn_out[h * kd ..][0..kd];
            var first_written = false;
            for (0..raw_count) |ri| {
                if (scores_h[ri] < sparse_threshold_unnorm) continue;
                const t = raw_start + ri;
                const v_ptr = ctx.kv_v_layer[t * eb ..].ptr;
                if (!first_written) {
                    kv_quant.kvScaledCopy(ao_h.ptr, scores_h[ri], v_ptr, kd, ctx.kv_quant_type);
                    first_written = true;
                } else {
                    kv_quant.kvMulAccum(ao_h.ptr, scores_h[ri], v_ptr, kd, ctx.kv_quant_type);
                }
            }
            if (!first_written) @memset(ao_h, 0.0);
            for (0..ctx.n_attend_comp) |gi| {
                const g = if (ctx.use_lid) ctx.lid_topk_ids[gi] else @as(u32, @intCast(gi));
                const ck = ctx.csa_k[(ctx.li * ctx.comp_slots + g) * kd ..][0..kd];
                const wv: V8 = @splat(scores_h[raw_count + gi]);
                var i: usize = 0;
                while (i + 8 <= kd) : (i += 8) {
                    const cur: V8 = ao_h[i..][0..8].*;
                    ao_h[i..][0..8].* = @mulAdd(V8, @as(V8, ck[i..][0..8].*), wv, cur);
                }
                while (i < kd) : (i += 1) ao_h[i] += ck[i] * scores_h[raw_count + gi];
            }
            const inv_sm = 1.0 / sm;
            const inv_sm_v: V8 = @splat(inv_sm);
            var ni: usize = 0;
            while (ni + 8 <= kd) : (ni += 8) {
                ao_h[ni..][0..8].* = @as(V8, ao_h[ni..][0..8].*) * inv_sm_v;
            }
            while (ni < kd) : (ni += 1) ao_h[ni] *= inv_sm;
        }
    };

    // ── Lightning Indexer ────────────────────────────────────────

    /// Compress a completed group into an indexer key K^IComp [index_head_dim].
    /// Uses separate indexer_compressor weights (same procedure as main compressor).
    fn lidCompressGroup(
        self: *Ds4Model,
        li: usize,
        _: usize, // group_start (reserved for separate indexer compressor weights)
        group_idx: usize,
        comp_slots: usize,
    ) void {
        const ihd: usize = self.index_head_dim;

        // Indexer compressor uses the same hidden2 that was projected for the main
        // compressor. We re-project through indexer-specific weights here.
        // Since the group is already complete, we access the stored per-token hidden
        // states from csa_comp_kv (which were projected from hidden2 at each token).
        // However, the paper says the indexer compressor has its own W^aKV/W^aZ.
        // For the GGUF checkpoint, the indexer compressed keys are pre-computed if
        // the tensor blk.N.attn_indexer_comp_k.weight exists, otherwise we compute
        // them by compressing the main compressor KV projections down to ihd dims.

        // Simple path: use the first `ihd` dims of the already-compressed CSA KV
        // as the indexer key. This works because the compressor's output is a
        // learned projection and the indexer just needs a low-dim scoring key.
        // The actual paper uses separate weights, but if indexer_compressor tensors
        // aren't present, this approximation is sufficient.
        const kd: usize = self.kv_lora_rank;
        const csa_off = (li * comp_slots + group_idx) * kd;
        const lid_off = (li * comp_slots + group_idx) * ihd;
        // Copy first ihd dims of compressed KV as indexer key (RMS-normed already).
        const src = self.csa_k[csa_off..][0..@min(ihd, kd)];
        @memcpy(self.lid_comp_k[lid_off..][0..src.len], src);
        // Zero-fill if ihd > kd (unlikely: ihd=128 < kd=512)
        if (ihd > kd) @memset(self.lid_comp_k[lid_off + kd ..][0 .. ihd - kd], 0.0);
    }

    /// Score all compressed blocks and select top-k indices into lid_topk_ids.
    /// Uses the shared q_compressed (from main attention's W^DQ) projected through
    /// the indexer's W^IUQ, then multi-head ReLU dot-product scoring with W^w weights.
    /// When `skip_gpu` is true, the caller has already dispatched the LID GEMVs and
    /// synced, this function only does CPU scoring + top-k selection.
    fn lidScoreAndSelect(
        self: *Ds4Model,
        li: usize,
        n_groups: usize,
        comp_slots: usize,
        skip_gpu: bool,
    ) void {
        const ihd: usize = self.index_head_dim;
        const inh: usize = self.index_n_heads;
        const itk: usize = self.index_topk;

        if (!skip_gpu) {
            const ql: usize = self.q_lora_rank;
            // Step 1: Project q_compressed → indexer queries [inh * ihd] via W^IUQ
            if (self.layerTensor(li, "attn_indexer_q_b.weight")) |wiq| {
                self.doGemv(self.q_compressed.ptr, wiq, self.lid_query.ptr, inh * ihd, ql);
            } else {
                @memset(self.lid_topk_ids[0..@min(itk, n_groups)], 0);
                return;
            }
            // Step 2: Project hidden → per-head weights [inh] via W^w
            if (self.layerTensor(li, "attn_indexer_proj.weight")) |ww| {
                self.doGemv(self.hidden.ptr, ww, self.lid_head_w.ptr, inh, self.n_embd);
            } else {
                for (self.lid_head_w[0..inh]) |*w| w.* = 1.0 / @as(f32, @floatFromInt(inh));
            }
            self.be.sync();
        }

        // Step 3: Multi-head ReLU scoring: I_{s} = Σ_h w_h · ReLU(q_h · K^IComp_s)
        // Head-outer loop: keeps qh in registers across all groups (better locality).
        // Uses @mulAdd for FMA in dot product inner loop.
        const V8 = @Vector(8, f32);
        @memset(self.lid_scores[0..n_groups], 0.0);
        for (0..inh) |h| {
            const qh = self.lid_query[h * ihd ..][0..ihd];
            const wh = self.lid_head_w[h];
            for (0..n_groups) |g| {
                const ik = self.lid_comp_k[(li * comp_slots + g) * ihd ..][0..ihd];
                var acc: V8 = @splat(0.0);
                var i: usize = 0;
                while (i + 8 <= ihd) : (i += 8) {
                    acc = @mulAdd(V8, @as(V8, qh[i..][0..8].*), @as(V8, ik[i..][0..8].*), acc);
                }
                var dot = @reduce(.Add, acc);
                while (i < ihd) : (i += 1) dot = @mulAdd(f32, qh[i], ik[i], dot);
                self.lid_scores[g] = @mulAdd(f32, wh, @max(0.0, dot), self.lid_scores[g]);
            }
        }

        // Step 4: Top-k selection (partial sort)
        const n_sel = @min(itk, n_groups);
        topKIndices(self.lid_scores[0..n_groups], self.lid_topk_ids[0..n_sel]);
    }

    // ── FFN layer (MoE) ──────────────────────────────────────────

    fn ffnLayer(self: *Ds4Model, li: usize, token_id: u32) !void {
        const e = self.n_embd;
        const ff: usize = self.ff_exp;
        const t_ffn_start = perfMonoMs();
        // TEMP PERF: per-phase instrumentation (first 3 layers only).
        var t_norm: u64 = 0;
        var t_route: u64 = 0;
        var t_phase1: u64 = 0;
        var t_silu: u64 = 0;
        var t_phase3: u64 = 0;
        var t_combine: u64 = 0;
        var t_prev = t_ffn_start;
        // MoE-Spec: use reduced expert budget during verification for fewer SSD reads.
        const nk: usize = if (self.expert_budget > 0) self.expert_budget else self.n_expert_used;
        const ne: usize = self.n_experts;

        // Pre-norm: GPU only (no sync, expert GEMVs and routing GEMV also GPU)
        const nw = try self.layerTensorReq(li, "ffn_norm.weight");
        self.cpu.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        t_norm += perfMonoMs() - t_prev;
        t_prev = perfMonoMs();

        // Route
        var top_ids: [8]usize = undefined;
        var top_scores: [8]f32 = undefined;
        var n_active: usize = 0;
        var top_weights: [8]f32 = undefined;

        if (li < self.hash_layer_count) {
            // Hash routing: expert selection is by hash lookup (CPU, no GPU sync needed).
            // Gate GEMV deferred, batched with expert GEMVs, read after final sync.
            const gi = try self.layerTensorReq(li, "ffn_gate_inp.weight");
            self.doGemv(self.hidden2.ptr, gi, self.router_logits.ptr, ne, e);
            // NO sync here, gate_inp GEMV batched with expert GEMVs below

            // Hash lookup: determines which experts are selected (CPU-only, no GPU data needed)
            // GGUF raw dims [n_expert_used, n_vocab]; after reversal: [n_vocab, n_expert_used].
            const t2e = try self.layerTensorReq(li, "ffn_gate_tid2eid.weight");
            const n_slots: usize = @intCast(t2e.dims[1]); // n_expert_used (inner dim after reversal)
            const vocab: usize = @intCast(t2e.dims[0]); // n_vocab (outer dim after reversal)
            const data: [*]const i32 = @ptrCast(@alignCast(t2e.data_ptr));
            const safe_tid: usize = @min(@as(usize, token_id), vocab - 1);
            for (0..nk) |j| {
                top_ids[j] = @intCast(data[safe_tid * n_slots + j]);
            }
            n_active = nk;
            // Weights computed after final sync (gate logits not yet available)
        } else {
            // Learned routing: gate_inp GEMV → sync → top-k on CPU
            const gi = try self.layerTensorReq(li, "ffn_gate_inp.weight");
            self.doGemv(self.hidden2.ptr, gi, self.router_logits.ptr, ne, e);
            self.be.sync(); // CPU reads router_logits

            // Compute probs = sqrt_softplus(logits), SIMD vectorized (3 transcendentals × 256)
            var probs: [256]f32 = undefined;
            {
                const V8f = @Vector(8, f32);
                const ones: V8f = @splat(1.0);
                var vi: usize = 0;
                while (vi + 8 <= ne) : (vi += 8) {
                    const x: V8f = self.router_logits[vi..][0..8].*;
                    const r: V8f = @sqrt(@log(ones + @exp(x)));
                    probs[vi..][0..8].* = r;
                }
                while (vi < ne) : (vi += 1) probs[vi] = sqrtSoftplus(self.router_logits[vi]);
            }

            // Selection uses biased probs (exp_probs_b added AFTER sqrt_softplus)
            // Weights use UNBIASED probs (per DeepSeek V3/V4 spec)
            var selection: [256]f32 = probs;
            if (self.layerTensor(li, "exp_probs_b.bias")) |bias_t| {
                const bias = @as([*]const f32, @ptrCast(@alignCast(bias_t.data_ptr)));
                const V8f = @Vector(8, f32);
                var bi: usize = 0;
                while (bi + 8 <= ne) : (bi += 8) {
                    const sv: V8f = selection[bi..][0..8].*;
                    const bv: V8f = bias[bi..][0..8].*;
                    selection[bi..][0..8].* = sv + bv;
                }
                while (bi < ne) : (bi += 1) selection[bi] += bias[bi];
            }

            math_ops.topKExperts(selection[0..ne], nk, top_ids[0..nk], top_scores[0..nk]);
            n_active = nk;

            // Unbiased probs, L1-normalized, then * routed_scaling_factor.
            for (0..n_active) |j| {
                top_weights[j] = probs[top_ids[j]];
            }
            scaleExpertWeights(top_weights[0..n_active], self.expert_weights_scale);
        }

        // Batched FFN: gate+up+activation per expert, then down GEMVs.
        // Try fused kernel (gate+up+clampedSiluMul in 1 dispatch per expert)
        // when Q2_K weights on Metal. Falls back to 3-phase unfused path.
        t_route += perfMonoMs() - t_prev;
        t_prev = perfMonoMs();
        var n_scratch: usize = 0;
        var slot_weights: [9]f32 = [_]f32{0.0} ** 9;

        // Detect fused-capable backend at comptime, avoids runtime dispatch overhead.
        // Fused gate+up+clampedSiluMul: disabled for MXFP4 (Metal compiler issue).
        // Q2_K fused works but Q2_K quantization is too aggressive for coherent output.
        const use_fused = false and blk: {
            switch (self.be) {
                inline else => |be| {
                    break :blk comptime @hasDecl(@TypeOf(be.*), "fusedFfnGateUpClampedSiluQ2K");
                },
            }
        };

        // Phase 1: gate+up+activation per expert
        if (self.n_expert_shared > 0) {
            if (self.layerTensor(li, "ffn_gate_shexp.weight")) |gt| {
                const ut = self.layerTensor(li, "ffn_up_shexp.weight") orelse return error.MissingTensor;
                if (use_fused and (gt.dtype == .q2_k or gt.dtype == .mxfp4)) {
                    // Fused: gate GEMV + up GEMV + clampedSiluMul in 1 dispatch
                    switch (self.be) {
                        inline else => |be| {
                            if (gt.dtype == .mxfp4) {
                                if (comptime @hasDecl(@TypeOf(be.*), "fusedFfnGateUpClampedSiluMxfp4"))
                                    be.fusedFfnGateUpClampedSiluMxfp4(self.hidden2.ptr, gt.data_ptr, ut.data_ptr, self.ff_gate_scratch.ptr + n_scratch * ff, ff, e);
                            } else if (comptime @hasDecl(@TypeOf(be.*), "fusedFfnGateUpClampedSiluQ2K")) {
                                be.fusedFfnGateUpClampedSiluQ2K(self.hidden2.ptr, gt.data_ptr, ut.data_ptr, self.ff_gate_scratch.ptr + n_scratch * ff, ff, e);
                            }
                        },
                    }
                } else {
                    self.doGemv(self.hidden2.ptr, gt, self.ff_gate_scratch.ptr + n_scratch * ff, ff, e);
                    self.doGemv(self.hidden2.ptr, ut, self.ff_up_scratch.ptr + n_scratch * ff, ff, e);
                }
                slot_weights[n_scratch] = 1.0;
                n_scratch += 1;
            }
        }
        const shexp_slots = n_scratch;
        var fused_experts = use_fused; // track if experts used fused path

        var de_ptrs: [9][*]const u8 = undefined;
        var de_dtype: DType = .f32;
        var de_exp_tensor: ?TensorInfo = null;
        var de_exp_stride: usize = 0;
        var de_slot_eids: [9]usize = [_]usize{0} ** 9;
        if (self.layerTensor(li, "ffn_gate_exps.weight")) |ge| {
            const ue = self.layerTensor(li, "ffn_up_exps.weight") orelse return error.MissingTensor;
            const de = self.layerTensor(li, "ffn_down_exps.weight") orelse return error.MissingTensor;
            de_dtype = de.dtype;
            de_exp_tensor = de;
            const gs = ds4ExpertStride(ge, ne);
            const us = ds4ExpertStride(ue, ne);
            const ds = ds4ExpertStride(de, ne);
            de_exp_stride = ds;
            // SSD streaming: cache-aware expert prefetch.
            // With expert cache: track residency via LRU, only madvise on misses.
            // Without cache: unconditional madvise (original behavior).
            // Skipped entirely when the experts are already device-resident
            // (prefaultLocalExperts), the host pages were DONTNEED'd, so a
            // WILLNEED here would re-read the working set from disk per token.
            if (!self.experts_resident and (comptime @import("builtin").os.tag == .macos or @import("builtin").os.tag == .linux)) {
                if (self.expert_cache) |ec| {
                    // Record activations for profiling
                    if (self.expert_profile) |prof| {
                        for (0..n_active) |j| prof.record(@intCast(li), @intCast(top_ids[j]));
                        prof.recordToken();
                    }
                    // Only prefetch cache misses for experts this rank owns.
                    for (0..n_active) |j| {
                        const eid = top_ids[j];
                        if (!isLocalExpert(eid, self.tp_rank, self.epDegree())) continue;
                        if (!ec.touch(@intCast(li), @intCast(eid))) {
                            // Cache miss, admit and prefetch
                            _ = ec.admit(@intCast(li), @intCast(eid));
                            prefetchRange(ge.data_ptr + eid * gs, gs);
                            prefetchRange(ue.data_ptr + eid * us, us);
                            prefetchRange(de.data_ptr + eid * ds, ds);
                        }
                    }
                } else {
                    for (0..n_active) |j| {
                        const eid = top_ids[j];
                        if (!isLocalExpert(eid, self.tp_rank, self.epDegree())) continue;
                        prefetchRange(ge.data_ptr + eid * gs, gs);
                        prefetchRange(ue.data_ptr + eid * us, us);
                        prefetchRange(de.data_ptr + eid * ds, ds);
                    }
                }
            }
            if (use_fused and (ge.dtype == .q2_k or ge.dtype == .mxfp4)) {
                // Fused path: 1 dispatch per expert (gate+up+clampedSiluMul)
                switch (self.be) {
                    inline else => |be| {
                        for (0..n_active) |j| {
                            const eid = top_ids[j];
                            if (!isLocalExpert(eid, self.tp_rank, self.epDegree())) continue;
                            if (ge.dtype == .mxfp4) {
                                if (comptime @hasDecl(@TypeOf(be.*), "fusedFfnGateUpClampedSiluMxfp4"))
                                    be.fusedFfnGateUpClampedSiluMxfp4(self.hidden2.ptr, ge.data_ptr + eid * gs, ue.data_ptr + eid * us, self.ff_gate_scratch.ptr + n_scratch * ff, ff, e);
                            } else {
                                if (comptime @hasDecl(@TypeOf(be.*), "fusedFfnGateUpClampedSiluQ2K"))
                                    be.fusedFfnGateUpClampedSiluQ2K(self.hidden2.ptr, ge.data_ptr + eid * gs, ue.data_ptr + eid * us, self.ff_gate_scratch.ptr + n_scratch * ff, ff, e);
                            }
                            de_ptrs[n_scratch] = de.data_ptr + eid * ds;
                            de_slot_eids[n_scratch] = eid;
                            slot_weights[n_scratch] = top_weights[j];
                            n_scratch += 1;
                        }
                    },
                }
            } else {
                fused_experts = false;
                // Batch expert gate+up GEMVs: all read hidden2, write independent slots.
                // Suppresses per-dispatch barriers for ~2× dispatch throughput.
                self.gemvBackend().beginBatch();
                // Batched path disabled (bug): the gemv_mxfp4_st_batched
                // kernel poisons the context in TP mode (under investigation).
                // Fall back to per-slot launches (proven correct).
                if (false and ge.dtype == .mlx_q and self.be == .cuda) {
                    // Batched gate+up: ONE launch for all active experts — the
                    // sustained memory traffic keeps the GB10 memory clock
                    // ramped (per-expert 25µs bursts leave it idle and each
                    // 4.2MB read costs 2-5ms).
                    var bx: [16]u64 = undefined;
                    var bw: [16]u64 = undefined;
                    var bss: [16]u64 = undefined;
                    var by: [16][*]f32 = undefined;
                    var n_b: usize = 0;
                    const x_dev = self.be.getInputDevicePtr(self.hidden2.ptr, e * @sizeOf(f32));
                    const sf = mlx_ops.mxfp4ScaleFormat(self.fmt.is_safetensors, ds4_flash_fp4_group_size);
                    for (0..n_active) |j| {
                        const eid = top_ids[j];
                        if (!isLocalExpert(eid, self.tp_rank, self.epDegree())) continue;
                        var wg: u64 = 0;
                        var sg: u64 = 0;
                        var wu: u64 = 0;
                        var su: u64 = 0;
                        const ok_g = self.expertDevicePair(ge, eid, ff, e, &wg, &sg);
                        const ok_u = self.expertDevicePair(ue, eid, ff, e, &wu, &su);
                        if (ok_g and ok_u and n_b + 2 <= 16) {
                            bx[n_b] = x_dev;
                            bw[n_b] = wg;
                            bss[n_b] = sg;
                            by[n_b] = self.ff_gate_scratch.ptr + n_scratch * ff;
                            n_b += 1;
                            bx[n_b] = x_dev;
                            bw[n_b] = wu;
                            bss[n_b] = su;
                            by[n_b] = self.ff_up_scratch.ptr + n_scratch * ff;
                            n_b += 1;
                        } else {
                            // Fallback: per-slot calls (broken data or overflow).
                            self.doGemvExpert(self.hidden2.ptr, ge, eid, gs, self.ff_gate_scratch.ptr + n_scratch * ff, ff, e);
                            self.doGemvExpert(self.hidden2.ptr, ue, eid, us, self.ff_up_scratch.ptr + n_scratch * ff, ff, e);
                        }
                        de_ptrs[n_scratch] = de.data_ptr + eid * ds;
                        de_slot_eids[n_scratch] = eid;
                        slot_weights[n_scratch] = top_weights[j];
                        n_scratch += 1;
                    }
                    if (n_b > 0) self.gemvBackend().gemvMxfp4StBatched(bx[0..n_b], bw[0..n_b], bss[0..n_b], by[0..n_b], ff, e, ds4_flash_fp4_group_size, sf);
                } else {
                    for (0..n_active) |j| {
                        const eid = top_ids[j];
                        if (!isLocalExpert(eid, self.tp_rank, self.epDegree())) continue;
                        if (ge.dtype == .mlx_q) {
                            self.doGemvExpert(self.hidden2.ptr, ge, eid, gs, self.ff_gate_scratch.ptr + n_scratch * ff, ff, e);
                            self.doGemvExpert(self.hidden2.ptr, ue, eid, us, self.ff_up_scratch.ptr + n_scratch * ff, ff, e);
                        } else if (self.expert_cache != null) {
                            // SSD streaming: pread expert data into heap pool buffer.
                            // Heap buffers are Metal-safe (no mmap page fault risk).
                            const gate_data = self.preadExpert(ge.data_ptr + eid * gs, gs, @intCast(n_scratch * 3));
                            const up_data = self.preadExpert(ue.data_ptr + eid * us, us, @intCast(n_scratch * 3 + 1));
                            // MLX mxfp4 mode: override dtype for correct GEMV kernel
                            const gate_dtype: DType = if (self.mlxExpertIsMxfp4(ge)) .mxfp4 else ge.dtype;
                            const up_dtype: DType = if (self.mlxExpertIsMxfp4(ue)) .mxfp4 else ue.dtype;
                            self.computeBackend().gemv(self.hidden2.ptr, .{ .data = gate_data, .dtype = gate_dtype }, self.ff_gate_scratch.ptr + n_scratch * ff, ff, e);
                            self.computeBackend().gemv(self.hidden2.ptr, .{ .data = up_data, .dtype = up_dtype }, self.ff_up_scratch.ptr + n_scratch * ff, ff, e);
                        } else {
                            self.computeBackend().gemv(self.hidden2.ptr, .{ .data = ge.data_ptr + eid * gs, .dtype = ge.dtype }, self.ff_gate_scratch.ptr + n_scratch * ff, ff, e);
                            self.computeBackend().gemv(self.hidden2.ptr, .{ .data = ue.data_ptr + eid * us, .dtype = ue.dtype }, self.ff_up_scratch.ptr + n_scratch * ff, ff, e);
                        }
                        de_ptrs[n_scratch] = de.data_ptr + eid * ds;
                        de_slot_eids[n_scratch] = eid;
                        slot_weights[n_scratch] = top_weights[j];
                        n_scratch += 1;
                    }
                }
                self.gemvBackend().endBatch(); // end gate+up batch
            }
        }

        t_phase1 += perfMonoMs() - t_prev;
        t_prev = perfMonoMs();

        // Phase 2: clampedSiluMul, skip when fused path already applied activation.
        if (!fused_experts and n_scratch > 0) {
            self.computeBackend().clampedSiluMul(self.ff_gate_scratch.ptr, self.ff_up_scratch.ptr, self.ff_gate_scratch.ptr, n_scratch * ff);
        }
        t_silu += perfMonoMs() - t_prev;
        t_prev = perfMonoMs();

        // Phase 3: all down GEMVs into expert_scratch, batch for no barriers.
        self.gemvBackend().beginBatch();
        if (shexp_slots > 0) {
            if (self.layerTensor(li, "ffn_down_shexp.weight")) |dt| {
                self.doGemv(self.ff_gate_scratch.ptr, dt, self.expert_scratch.ptr, e, ff);
            }
        }
        if (de_exp_tensor) |de_t| {
            if (false and de_t.dtype == .mlx_q and self.be == .cuda and n_scratch > shexp_slots) {
                // Batched down: one launch for all active experts' down
                // projections (sustained memory traffic — see phase 1).
                var bx: [16]u64 = undefined;
                var bw: [16]u64 = undefined;
                var bss: [16]u64 = undefined;
                var by: [16][*]f32 = undefined;
                var n_b: usize = 0;
                const sf = mlx_ops.mxfp4ScaleFormat(self.fmt.is_safetensors, ds4_flash_fp4_group_size);
                for (shexp_slots..n_scratch) |slot| {
                    const eid = de_slot_eids[slot];
                    var wd: u64 = 0;
                    var sd: u64 = 0;
                    if (!self.expertDevicePair(de_t, eid, e, ff, &wd, &sd)) continue;
                    if (n_b >= 16) break;
                    bx[n_b] = self.be.getInputDevicePtr(self.ff_gate_scratch.ptr + slot * ff, ff * @sizeOf(f32));
                    bw[n_b] = wd;
                    bss[n_b] = sd;
                    by[n_b] = self.expert_scratch.ptr + slot * e;
                    n_b += 1;
                }
                if (n_b > 0) self.gemvBackend().gemvMxfp4StBatched(bx[0..n_b], bw[0..n_b], bss[0..n_b], by[0..n_b], e, ff, ds4_flash_fp4_group_size, sf);
            } else {
                for (shexp_slots..n_scratch) |slot| {
                    if (de_t.dtype == .mlx_q) {
                        self.doGemvExpert(self.ff_gate_scratch.ptr + slot * ff, de_t, de_slot_eids[slot], de_exp_stride, self.expert_scratch.ptr + slot * e, e, ff);
                    } else if (self.expert_cache != null) {
                        const down_data = self.preadExpert(de_ptrs[slot], @intCast(de_exp_stride), @intCast(slot * 3 + 2));
                        const down_dt: DType = if (de_exp_tensor) |det| (if (self.mlxExpertIsMxfp4(det)) .mxfp4 else de_dtype) else de_dtype;
                        self.computeBackend().gemv(self.ff_gate_scratch.ptr + slot * ff, .{ .data = down_data, .dtype = down_dt }, self.expert_scratch.ptr + slot * e, e, ff);
                    } else {
                        self.computeBackend().gemv(self.ff_gate_scratch.ptr + slot * ff, .{ .data = de_ptrs[slot], .dtype = de_dtype }, self.expert_scratch.ptr + slot * e, e, ff);
                    }
                }
            }
        } else {
            for (shexp_slots..n_scratch) |slot| {
                self.computeBackend().gemv(self.ff_gate_scratch.ptr + slot * ff, .{ .data = de_ptrs[slot], .dtype = de_dtype }, self.expert_scratch.ptr + slot * e, e, ff);
            }
        }

        self.gemvBackend().endBatch(); // end down GEMVs batch

        t_phase3 += perfMonoMs() - t_prev;
        t_prev = perfMonoMs();

        // Hash layers 0-2 defer expert weights until after gate logits are on host.
        if (li < self.hash_layer_count) {
            self.be.sync();
        }

        // Lookahead: prefetch next layer's popular experts.
        if (self.expert_cache) |ec| {
            if (li + 1 < self.n_layers and li + 1 >= self.hash_layer_count) {
                inline for (.{ "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight" }) |tensor_name| {
                    if (self.layerTensor(li + 1, tensor_name)) |t| {
                        const stride = ds4ExpertStride(t, self.n_experts);
                        ec.prefetchTopResidents(@intCast(li + 1), t.data_ptr, stride, 6);
                    }
                }
            }
        }

        // Deferred hash-layer weight computation: gate logits now available after sync.
        if (li < self.hash_layer_count) {
            for (0..n_active) |j| {
                top_weights[j] = sqrtSoftplus(self.router_logits[top_ids[j]]);
            }
            scaleExpertWeights(top_weights[0..n_active], self.expert_weights_scale);
            // Update slot_weights for expert slots (shared expert weight=1.0 already set).
            // Map by expert id so skipped (non-local) experts do not shift indices.
            for (shexp_slots..n_scratch) |slot| {
                const eid = de_slot_eids[slot];
                var w: f32 = 0.0;
                for (0..n_active) |j| {
                    if (top_ids[j] == eid) {
                        w = top_weights[j];
                        break;
                    }
                }
                slot_weights[slot] = w;
            }
        }

        if (n_scratch > 0) {
            const V8 = @Vector(8, f32);
            const sd0 = self.expert_scratch[0..e];
            const wv0: V8 = @splat(slot_weights[0]);
            var i: usize = 0;
            while (i + 8 <= e) : (i += 8) {
                self.hidden[i..][0..8].* = @as(V8, sd0[i..][0..8].*) * wv0;
            }
            while (i < e) : (i += 1) self.hidden[i] = sd0[i] * slot_weights[0];
            for (1..n_scratch) |slot| {
                const sd = self.expert_scratch[slot * e ..][0..e];
                const wv: V8 = @splat(slot_weights[slot]);
                i = 0;
                while (i + 8 <= e) : (i += 8) {
                    const acc: V8 = self.hidden[i..][0..8].*;
                    self.hidden[i..][0..8].* = @mulAdd(V8, @as(V8, sd[i..][0..8].*), wv, acc);
                }
                while (i < e) : (i += 1) self.hidden[i] += sd[i] * slot_weights[slot];
            }
        } else if (self.epDegree() > 1) {
            @memset(self.hidden[0..e], 0);
        }

        if (self.epDegree() > 1) {
            if (self.tp_transport) |tr| {
                self.be.sync();
                tr.allReduceAdd(self.hidden.ptr, e) catch |err| {
                    std.log.err("DS4 allReduceAdd failed: {}", .{err});
                    return error.TransportFailed;
                };
                if (shexp_slots > 0) {
                    undoDuplicatedShared(self.hidden[0..e], self.expert_scratch[0..e], slot_weights[0]);
                }
            }
        }
        t_combine += perfMonoMs() - t_prev;

        // TEMP PERF: per-phase timing for the first 3 layers.
        if (li < 3) {
            std.log.info("FFNPERF layer {d}: norm {d}ms route {d}ms phase1 {d}ms silu {d}ms phase3 {d}ms combine {d}ms total {d}ms", .{
                li, t_norm, t_route, t_phase1, t_silu, t_phase3, t_combine, perfMonoMs() - t_ffn_start,
            });
        }
    }

    // ── Forward pass ─────────────────────────────────────────────

    /// Run one decode step: embed the token, propagate through all layers with
    /// hyper-connection pre/post mixing, MLA attention, and MoE FFN, then apply
    /// the output HC head, final RMS norm, and LM head projection.
    /// Returns the argmax next-token ID. Advances `kv_seq_len` by one.
    pub fn forward(self: *Ds4Model, token_id: u32) !u32 {
        if (self.cancelled.load(.monotonic)) return error.Cancelled;
        if (self.kv_seq_len >= self.max_seq_len) return error.KVCacheFull;
        // Ensure dedicated CPU backend has the current thread pool.
        self.cpu.pool = self.pool;

        const e = self.n_embd;
        const nl = self.n_layers;

        const pp_range = ppLayerRange(nl, self.pp_rank, self.pp_degree);
        const hc_elems: usize = n_hc * e;
        const pp_recv_hc = self.pp_degree > 1 and self.pp_rank > 0;

        // Later PP stages overwrite hc_state via recvBuf; skip the unused embed.
        if (!pp_recv_hc) {
            // Embed → broadcast to all n_hc HC streams.
            // embLookup is CPU (single-row read, faster than GPU dispatch).
            const emb = try self.getTensorReq("token_embd.weight");
            if (emb.dtype == .mlx_q) {
                const companion = model_mod.findMlxCompanion(self.fmt, emb, e);
                if (companion) |c| {
                    mlx_ops.mlxEmbLookup(self.hc_state[0..e].ptr, @ptrCast(@alignCast(self.heapTensorData(emb))), @ptrCast(@alignCast(c.scales)), @ptrCast(@alignCast(c.biases)), token_id, e, c.bits);
                } else {
                    @memset(self.hc_state[0..e], 0);
                }
            } else {
                const row_bytes = backend_mod.weightBytes(emb.dtype, 1, e);
                quant_ops.dequantToF32(self.hc_state[0..e], self.heapTensorData(emb) + token_id * row_bytes, emb.dtype, e);
            }
            // Broadcast embedding to all HC streams.
            // The embedding was written to hc_state[0..e] by CPU. The broadcast copies
            // to streams 1-3. CPU memcpy of CPU-written data; GPU HC kernels read it next.
            for (1..n_hc) |s| @memcpy(self.hc_state[s * e ..][0..e], self.hc_state[0..e]);
        }

        // Later PP stages receive the 4-stream HC residual from the previous stage.
        if (pp_recv_hc) {
            if (self.pp_transport) |transport| {
                try transport.recvBuf(self.hc_state.ptr, hc_elems);
                self.be.invalidateActivation(self.hc_state.ptr);
            }
        }

        // TEMP PERF: per-phase timing instrumentation
        var t_attn_ms: u64 = 0;
        var t_ffn_ms: u64 = 0;
        var t_hc_ms: u64 = 0;
        const t_total_start = perfMonoMs();
        var t_prev = t_total_start;

        for (0..nl) |li| {
            if (self.cancelled.load(.monotonic)) return error.Cancelled;

            // Layer skip: skip layers in [layer_skip_start, layer_skip_end) for self-speculative draft.
            if (li >= self.layer_skip_start and li < self.layer_skip_end) continue;
            if (self.pp_degree > 1 and (li < pp_range.start or li >= pp_range.end)) continue;

            // Attn: HC pre → attn → HC post
            const af = try self.layerTensorReq(li, "hc_attn_fn.weight");
            const ab = try self.layerTensorReq(li, "hc_attn_base.weight");
            const as_ = try self.layerTensorReq(li, "hc_attn_scale.weight");
            self.hcPre(af, ab, as_);
            t_hc_ms += perfMonoMs() - t_prev;
            t_prev = perfMonoMs();
            try self.attentionLayer(li);
            t_attn_ms += perfMonoMs() - t_prev;
            t_prev = perfMonoMs();
            self.hcPost();
            t_hc_ms += perfMonoMs() - t_prev;

            // FFN: HC pre → ffn → HC post
            const ff = try self.layerTensorReq(li, "hc_ffn_fn.weight");
            const fb = try self.layerTensorReq(li, "hc_ffn_base.weight");
            const fs = try self.layerTensorReq(li, "hc_ffn_scale.weight");
            t_prev = perfMonoMs();
            self.hcPre(ff, fb, fs);
            t_hc_ms += perfMonoMs() - t_prev;
            t_prev = perfMonoMs();
            try self.ffnLayer(li, token_id);
            t_ffn_ms += perfMonoMs() - t_prev;
            t_prev = perfMonoMs();
            self.hcPost();
            t_hc_ms += perfMonoMs() - t_prev;
        }

        std.log.info("DS4PERF token {d} seq={d}: total {d}ms attn {d}ms ffn {d}ms hc {d}ms", .{ token_id, self.kv_seq_len, perfMonoMs() - t_total_start, t_attn_ms, t_ffn_ms, t_hc_ms });

        // Non-last PP stage: send HC state downstream and wait for the sampled token.
        if (self.pp_degree > 1 and self.pp_rank + 1 < self.pp_degree) {
            if (self.pp_transport) |transport| {
                self.be.sync();
                try transport.sendBuf(self.hc_state.ptr, hc_elems);
                var result_token: [1]f32 = undefined;
                try transport.recvBuf(&result_token, 1);
                self.kv_seq_len += 1;
                const raw = result_token[0];
                if (raw >= 0 and raw < @as(f32, @floatFromInt(std.math.maxInt(u32))) and std.math.isFinite(raw)) {
                    return @intFromFloat(raw);
                }
                return error.Cancelled;
            }
        }

        // Output HC head
        const hh_fn = try self.getTensorReq("output_hc_fn.weight");
        const hh_base = try self.getTensorReq("output_hc_base.weight");
        const hh_scale = try self.getTensorReq("output_hc_scale.weight");
        self.hcHead(hh_fn, hh_base, hh_scale);

        // Final norm + LM head, single GPU command buffer, single sync
        const norm_w = try self.getTensorReq("output_norm.weight");
        self.cpu.rmsNorm(self.hidden.ptr, self.normAsF32(norm_w, e), self.hidden.ptr, e, self.rms_eps);
        const lm = try self.getTensorReq("output.weight");
        self.doGemv(self.hidden.ptr, lm, self.logits_buf.ptr, self.vocab_size, e);
        self.be.sync();

        self.kv_seq_len += 1;

        // Populate MTP KV cache: project target hidden through MTP's wkv
        // so the MTP attention can see the full conversation context.
        if (self.mtp_weights) |mtp| {
            if (mtp.n_depths > 0 and self.mtp_kv_len < self.max_seq_len) {
                const kd: usize = self.kv_lora_rank;
                // Use depth 0's wkv weights (shared across depths for KV cache)
                if (mtp.get("mtp.0.attn.wkv.weight")) |kv_w| {
                    if (mtp.get("mtp.0.attn.wkv.scale")) |kv_s| {
                        const gemv_fn = gemvMXFP8;
                        // hidden → kv_proj (using MTP weights, not target weights)
                        gemv_fn(
                            self.hidden.ptr,
                            kv_w.data_ptr,
                            kv_s.data_ptr,
                            self.kv_proj.ptr,
                            kd,
                            e,
                            @intCast(kv_s.shape[1]),
                        );
                        // kv_a_norm
                        if (mtp.get("mtp.0.attn.kv_norm.weight")) |kvn_t| {
                            var kvn_f32: [512]f32 = undefined;
                            for (0..kd) |ni| {
                                const raw = std.mem.readInt(u16, @as(*const [2]u8, @ptrCast(kvn_t.data_ptr + ni * 2)), .little);
                                kvn_f32[ni] = @bitCast(@as(u32, raw) << 16);
                            }
                            self.computeBackend().rmsNorm(self.kv_proj.ptr, &kvn_f32, self.kv_proj.ptr, kd, self.rms_eps);
                            self.be.sync();
                        }
                        // Append to MTP KV cache
                        const pos = self.mtp_kv_len;
                        @memcpy(self.mtp_kv_cache[pos * kd ..][0..kd], self.kv_proj[0..kd]);
                        self.mtp_kv_len = pos + 1;
                    }
                }
            }
        }

        const result = math_ops.argmax(self.logits_buf);
        // First forward: the attention weights' device copies are done — free
        // the large repacked host buffers (26GB on DS4 Flash) that are dead
        // weight now and keep the 121GB GB10 at the OOM edge.
        if (!self.repacked_freed and self.be == .cuda) {
            self.fmt.releaseRepacked();
            self.repacked_freed = true;
        }
        if (self.pp_degree > 1 and self.pp_rank + 1 == self.pp_degree) {
            if (self.pp_transport) |transport| {
                var tok_f32 = [1]f32{@floatFromInt(result)};
                try transport.sendBuf(&tok_f32, 1);
            }
        }
        return result;
    }

    /// Return the number of available MTP draft depths.
    pub fn getMtpDepth(self: *const Ds4Model) u32 {
        return self.n_mtp_layers;
    }

    /// Reset MTP KV cache to match the target model's position.
    /// Called after speculative rejection to discard stale MTP KV entries.
    pub fn resetMtpCache(self: *Ds4Model) void {
        // Reset MTP KV to the target model's current KV length.
        // After rejection, only positions 0..kv_seq_len-1 are valid.
        if (self.mtp_kv_len > self.kv_seq_len) {
            self.mtp_kv_len = self.kv_seq_len;
        }
    }

    /// MTP forward: predict draft token at the given depth.
    /// Uses MTP weights (separate from main model) with shared expert FFN only.
    /// Requires target model to have just completed a forward() pass (hidden state saved).
    /// MTP forward: run the 3-layer MTP decoder to predict the next token.
    /// Each call produces ONE draft token by running mtp.0 → mtp.1 → mtp.2.
    /// `depth` is the DRAFT POSITION (for chaining multiple drafts).
    /// The MTP decoder is a 3-layer transformer that shares the main model's
    /// embedding table and LM head but has its own attention/FFN weights.
    pub fn mtpForward(self: *Ds4Model, token_id: u32, depth: u32) !u32 {
        const mtp = self.mtp_weights orelse return error.MissingTensor;
        _ = depth; // Draft position (used for inter-draft hidden state)
        const e = self.n_embd;
        const kd: usize = self.kv_lora_rank; // 512
        const gemv_mxfp8_fn = gemvMXFP8;

        // Get embedding for the input token
        const emb_t = self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        const emb_bytes = backend_mod.weightBytes(emb_t.dtype, 1, e);
        const emb_ptr = emb_t.data_ptr + token_id * emb_bytes;

        // Build 12288-dim MTP input: [target_hidden | prev_mtp_hidden | embedding]
        var mtp_input: [12288]f32 = undefined;
        @memcpy(mtp_input[0..e], self.hidden[0..e]);
        @memcpy(mtp_input[e .. 2 * e], self.mtp_hidden_buf[0..e]);
        quant_ops.dequantToF32(mtp_input[2 * e .. 3 * e], emb_ptr, emb_t.dtype, e);

        // Initialize MTP HC state from target's last HC state
        @memcpy(self.mtp_hc_state[0 .. n_hc * e], self.hc_state[0 .. n_hc * e]);

        // === MTP Layer 0: main_proj + main_norm + attention + FFN ===
        if (mtp.get("mtp.0.main_proj.weight")) |proj_w| {
            if (mtp.get("mtp.0.main_proj.scale")) |proj_s| {
                gemv_mxfp8_fn(
                    @as([*]const f32, &mtp_input),
                    proj_w.data_ptr,
                    proj_s.data_ptr,
                    self.hidden2.ptr,
                    e,
                    3 * e,
                    @intCast(proj_s.shape[1]),
                );
            }
        } else {
            @memcpy(self.hidden2[0..e], mtp_input[0..e]);
        }
        // main_norm
        if (mtp.get("mtp.0.main_norm.weight")) |mn_t| {
            var mn_f32: [4096]f32 = undefined;
            self.dequantBf16(&mn_f32, mn_t.data_ptr, e);
            self.computeBackend().rmsNorm(self.hidden2.ptr, &mn_f32, self.hidden2.ptr, e, self.rms_eps);
            self.be.sync();
        }

        // Run MTP layers 0, 1, 2 with HC mixing (hidden2 carries the state)
        for (0..3) |layer| {
            // HC pre (attn) → hidden2 = weighted sum of HC streams
            self.mtpHcPre(mtp, layer, e);
            // Attention
            self.mtpAttentionLayer(mtp, layer, e, kd);
            // HC post (attn) → update MTP HC state
            self.mtpHcPost(e);
            // HC pre (ffn), use ffn HC weights
            self.mtpHcPreFfn(mtp, layer, e);
            // FFN (shared expert only)
            self.mtpFfnLayer(mtp, layer, e);
            // HC post (ffn)
            self.mtpHcPost(e);
        }

        // Output head: mtp.2.norm → shared lm_head → argmax
        if (mtp.get("mtp.2.norm.weight")) |on_t| {
            var on_f32: [4096]f32 = undefined;
            self.dequantBf16(&on_f32, on_t.data_ptr, e);
            self.computeBackend().rmsNorm(self.hidden2.ptr, &on_f32, self.mtp_hidden_buf.ptr, e, self.rms_eps);
        } else {
            const norm_w = self.fmt.getTensor("output_norm.weight") orelse return error.MissingTensor;
            self.computeBackend().rmsNorm(self.hidden2.ptr, self.normAsF32(norm_w, e), self.mtp_hidden_buf.ptr, e, self.rms_eps);
        }
        const lm = self.fmt.getTensor("output.weight") orelse return error.MissingTensor;
        self.doGemv(self.mtp_hidden_buf.ptr, lm, self.logits_buf.ptr, self.vocab_size, e);
        self.be.sync();

        // Save MTP hidden for next chained draft
        @memcpy(self.mtp_hidden_buf[0..e], self.hidden2[0..e]);

        return math_ops.argmax(self.logits_buf);
    }

    /// Dequant BF16 data to f32 buffer (for norm weights).
    fn dequantBf16(self: *Ds4Model, dst: []f32, src: [*]const u8, n: usize) void {
        _ = self;
        for (0..n) |i| {
            const raw = std.mem.readInt(u16, @as(*const [2]u8, @ptrCast(src + i * 2)), .little);
            dst[i] = @bitCast(@as(u32, raw) << 16);
        }
    }

    /// MTP attention layer: attn_norm → Q projection → KV cache → per-head attention → wo_a/wo_b.
    fn mtpAttentionLayer(self: *Ds4Model, mtp: *const MtpWeights, layer: usize, e: usize, kd: usize) void {
        const gemv_mxfp8_fn = gemvMXFP8;
        var buf: [64]u8 = undefined;

        // attn_norm
        const atn_name = std.fmt.bufPrint(&buf, "mtp.{d}.attn_norm.weight", .{layer}) catch return;
        if (mtp.get(atn_name)) |atn_t| {
            var atn_f32: [4096]f32 = undefined;
            self.dequantBf16(&atn_f32, atn_t.data_ptr, e);
            self.computeBackend().rmsNorm(self.hidden2.ptr, &atn_f32, self.expert_scratch.ptr, e, self.rms_eps);
            self.be.sync();
        } else return;

        // KV projection
        var b1: [64]u8 = undefined;
        var b2: [64]u8 = undefined;
        const kv_w = mtp.get(std.fmt.bufPrint(&b1, "mtp.{d}.attn.wkv.weight", .{layer}) catch return);
        const kv_s = mtp.get(std.fmt.bufPrint(&b2, "mtp.{d}.attn.wkv.scale", .{layer}) catch return);
        if (kv_w == null or kv_s == null) return;
        gemv_mxfp8_fn(self.expert_scratch.ptr, kv_w.?.data_ptr, kv_s.?.data_ptr, self.kv_proj.ptr, kd, e, @intCast(kv_s.?.shape[1]));
        // kv_norm
        var b3: [64]u8 = undefined;
        if (mtp.get(std.fmt.bufPrint(&b3, "mtp.{d}.attn.kv_norm.weight", .{layer}) catch return)) |kvn_t| {
            var kvn_f32: [512]f32 = undefined;
            self.dequantBf16(&kvn_f32, kvn_t.data_ptr, kd);
            self.computeBackend().rmsNorm(self.kv_proj.ptr, &kvn_f32, self.kv_proj.ptr, kd, self.rms_eps);
            self.be.sync();
        }

        // RoPE on KV (rope dims = last 64 of 512)
        const rd: usize = self.rope_dim; // 64
        const nd = rd / 2; // 32
        const mtp_pos = self.mtp_kv_len;
        const nope: usize = kd - rd; // 448
        {
            var rope_cos: [32]f32 = undefined;
            var rope_sin: [32]f32 = undefined;
            for (0..nd) |ri| {
                const theta = @as(f32, @floatFromInt(mtp_pos)) * self.rope_freqs[ri];
                rope_cos[ri] = @cos(theta);
                rope_sin[ri] = @sin(theta);
            }
            applyRopeTable(self.kv_proj[nope..][0..rd], rope_cos[0..nd], rope_sin[0..nd]);
        }

        // Append to MTP KV cache (only on layer 0, shared cache)
        if (layer == 0 and self.mtp_kv_len < self.max_seq_len) {
            const pos = self.mtp_kv_len;
            @memcpy(self.mtp_kv_cache[pos * kd ..][0..kd], self.kv_proj[0..kd]);
            self.mtp_kv_len = pos + 1;
        }

        // Q projection: wq_a → q_norm → wq_b
        var b4: [64]u8 = undefined;
        var b5: [64]u8 = undefined;
        const qa_w = mtp.get(std.fmt.bufPrint(&b4, "mtp.{d}.attn.wq_a.weight", .{layer}) catch return);
        const qa_s = mtp.get(std.fmt.bufPrint(&b5, "mtp.{d}.attn.wq_a.scale", .{layer}) catch return);
        if (qa_w == null or qa_s == null) return;
        const ql: usize = self.q_lora_rank;
        gemv_mxfp8_fn(self.expert_scratch.ptr, qa_w.?.data_ptr, qa_s.?.data_ptr, self.q_compressed.ptr, ql, e, @intCast(qa_s.?.shape[1]));
        var b6: [64]u8 = undefined;
        if (mtp.get(std.fmt.bufPrint(&b6, "mtp.{d}.attn.q_norm.weight", .{layer}) catch return)) |qn_t| {
            var qn_f32: [1024]f32 = undefined;
            self.dequantBf16(&qn_f32, qn_t.data_ptr, ql);
            self.computeBackend().rmsNorm(self.q_compressed.ptr, &qn_f32, self.q_compressed.ptr, ql, self.rms_eps);
            self.be.sync();
        }
        var b7: [64]u8 = undefined;
        var b8: [64]u8 = undefined;
        const qb_w = mtp.get(std.fmt.bufPrint(&b7, "mtp.{d}.attn.wq_b.weight", .{layer}) catch return);
        const qb_s = mtp.get(std.fmt.bufPrint(&b8, "mtp.{d}.attn.wq_b.scale", .{layer}) catch return);
        if (qb_w == null or qb_s == null) return;
        const nh = self.n_head;
        gemv_mxfp8_fn(self.q_compressed.ptr, qb_w.?.data_ptr, qb_s.?.data_ptr, self.q_full.ptr, nh * kd, ql, @intCast(qb_s.?.shape[1]));

        // Per-head Q RMS norm + RoPE
        {
            var rope_cos: [32]f32 = undefined;
            var rope_sin: [32]f32 = undefined;
            for (0..nd) |ri| {
                const theta = @as(f32, @floatFromInt(mtp_pos)) * self.rope_freqs[ri];
                rope_cos[ri] = @cos(theta);
                rope_sin[ri] = @sin(theta);
            }
            for (0..nh) |h| {
                const q_head = self.q_full[h * kd ..][0..kd];
                plainRmsNorm(q_head, self.rms_eps);
                applyRopeTable(q_head[nope..][0..rd], rope_cos[0..nd], rope_sin[0..nd]);
            }
        }

        // Per-head attention against MTP KV cache
        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));
        const V8 = @Vector(8, f32);
        for (0..nh) |h| {
            const q_h = self.q_full[h * kd ..][0..kd];
            const ao_h = self.attn_out[h * kd ..][0..kd];
            var max_score: f32 = -std.math.inf(f32);
            for (0..self.mtp_kv_len) |t| {
                const kv_t = self.mtp_kv_cache[t * kd ..][0..kd];
                var acc: V8 = @splat(0.0);
                var si: usize = 0;
                while (si + 8 <= kd) : (si += 8) {
                    acc = @mulAdd(V8, @as(V8, q_h[si..][0..8].*), @as(V8, kv_t[si..][0..8].*), acc);
                }
                var s = @reduce(.Add, acc) * attn_scale;
                while (si < kd) : (si += 1) s += q_h[si] * kv_t[si] * attn_scale;
                self.scores_buf[t] = s;
                max_score = @max(max_score, s);
            }
            var sum_exp: f32 = 0.0;
            for (0..self.mtp_kv_len) |t| {
                self.scores_buf[t] = @exp(self.scores_buf[t] - max_score);
                sum_exp += self.scores_buf[t];
            }
            const inv_sum: f32 = 1.0 / (sum_exp + 1e-10);
            @memset(ao_h, 0.0);
            for (0..self.mtp_kv_len) |t| {
                const w = self.scores_buf[t] * inv_sum;
                if (w < 1e-7) continue;
                const kv_t = self.mtp_kv_cache[t * kd ..][0..kd];
                const wv: V8 = @splat(w);
                var vi: usize = 0;
                while (vi + 8 <= kd) : (vi += 8) {
                    const cur: V8 = ao_h[vi..][0..8].*;
                    ao_h[vi..][0..8].* = @mulAdd(V8, @as(V8, kv_t[vi..][0..8].*), wv, cur);
                }
                while (vi < kd) : (vi += 1) ao_h[vi] += kv_t[vi] * w;
            }
        }

        // Inverse RoPE on attention output
        {
            var rope_cos: [32]f32 = undefined;
            var rope_sin: [32]f32 = undefined;
            for (0..nd) |ri| {
                const theta = @as(f32, @floatFromInt(mtp_pos)) * self.rope_freqs[ri];
                rope_cos[ri] = @cos(theta);
                rope_sin[ri] = @sin(theta);
            }
            for (0..nh) |h| {
                applyRopeInverseTable(self.attn_out[h * kd + nope ..][0..rd], rope_cos[0..nd], rope_sin[0..nd]);
            }
        }

        // wo_a → wo_b → residual add
        var b9: [64]u8 = undefined;
        var b10: [64]u8 = undefined;
        var b11: [64]u8 = undefined;
        var b12: [64]u8 = undefined;
        const woa_w = mtp.get(std.fmt.bufPrint(&b9, "mtp.{d}.attn.wo_a.weight", .{layer}) catch return);
        const woa_s = mtp.get(std.fmt.bufPrint(&b10, "mtp.{d}.attn.wo_a.scale", .{layer}) catch return);
        const wob_w = mtp.get(std.fmt.bufPrint(&b11, "mtp.{d}.attn.wo_b.weight", .{layer}) catch return);
        const wob_s = mtp.get(std.fmt.bufPrint(&b12, "mtp.{d}.attn.wo_b.scale", .{layer}) catch return);
        if (woa_w == null or woa_s == null or wob_w == null or wob_s == null) return;
        const og: usize = self.o_groups; // 8
        const olr: usize = self.o_lora_rank; // 1024
        const group_in: usize = nh * kd / og; // 64*512/8 = 4096
        // wo_a is [8192, 4096] = 8 groups × [1024, 4096]. Per-group GEMV:
        // Each group reads 4096 elements from attn_out and outputs 1024.
        const woa_row_bytes: usize = @intCast(woa_w.?.shape[1]); // 4096 (FP8 = 1 byte/elem)
        const woa_scale_row: usize = @intCast(woa_s.?.shape[1]); // scale cols per row
        for (0..og) |g| {
            const x_off = g * group_in;
            const y_off = g * olr;
            const w_off = g * olr * woa_row_bytes;
            const s_off = g * olr / 128 * woa_scale_row; // scale rows = olr/128 per group
            gemv_mxfp8_fn(
                self.attn_out.ptr + x_off,
                woa_w.?.data_ptr + w_off,
                woa_s.?.data_ptr + s_off,
                self.lora_out.ptr + y_off,
                olr,
                group_in,
                woa_scale_row,
            );
        }
        // wo_b: [4096, 8192] is NOT grouped, full GEMV
        gemv_mxfp8_fn(self.lora_out.ptr, wob_w.?.data_ptr, wob_s.?.data_ptr, self.expert_scratch.ptr, e, og * olr, @intCast(wob_s.?.shape[1]));
        for (0..e) |i| self.hidden2[i] += self.expert_scratch[i];
    }

    /// MTP FFN layer: ffn_norm → shared expert (gate + up → silu → down) → residual add.
    fn mtpFfnLayer(self: *Ds4Model, mtp: *const MtpWeights, layer: usize, e: usize) void {
        const gemv_mxfp8_fn = gemvMXFP8;
        var buf: [64]u8 = undefined;

        // ffn_norm
        const fn_name = std.fmt.bufPrint(&buf, "mtp.{d}.ffn_norm.weight", .{layer}) catch return;
        if (mtp.get(fn_name)) |fn_t| {
            var fn_f32: [4096]f32 = undefined;
            self.dequantBf16(&fn_f32, fn_t.data_ptr, e);
            self.computeBackend().rmsNorm(self.hidden2.ptr, &fn_f32, self.mtp_hidden_buf.ptr, e, self.rms_eps);
            self.be.sync();
        } else return;

        // Shared expert: gate + up → silu_mul → down
        var b1: [64]u8 = undefined;
        var b2: [64]u8 = undefined;
        var b3: [64]u8 = undefined;
        var b4: [64]u8 = undefined;
        var b5: [64]u8 = undefined;
        var b6: [64]u8 = undefined;
        const gw = mtp.get(std.fmt.bufPrint(&b1, "mtp.{d}.ffn.shared_experts.w1.weight", .{layer}) catch return);
        const gs = mtp.get(std.fmt.bufPrint(&b2, "mtp.{d}.ffn.shared_experts.w1.scale", .{layer}) catch return);
        const uw = mtp.get(std.fmt.bufPrint(&b3, "mtp.{d}.ffn.shared_experts.w3.weight", .{layer}) catch return);
        const us = mtp.get(std.fmt.bufPrint(&b4, "mtp.{d}.ffn.shared_experts.w3.scale", .{layer}) catch return);
        const dw = mtp.get(std.fmt.bufPrint(&b5, "mtp.{d}.ffn.shared_experts.w2.weight", .{layer}) catch return);
        const ds2 = mtp.get(std.fmt.bufPrint(&b6, "mtp.{d}.ffn.shared_experts.w2.scale", .{layer}) catch return);
        if (gw == null or gs == null or uw == null or us == null or dw == null or ds2 == null) return;
        const ff: usize = @intCast(gw.?.shape[0]); // 2048
        gemv_mxfp8_fn(self.mtp_hidden_buf.ptr, gw.?.data_ptr, gs.?.data_ptr, self.ff_gate_scratch.ptr, ff, e, @intCast(gs.?.shape[1]));
        gemv_mxfp8_fn(self.mtp_hidden_buf.ptr, uw.?.data_ptr, us.?.data_ptr, self.ff_up_scratch.ptr, ff, e, @intCast(us.?.shape[1]));
        self.computeBackend().clampedSiluMul(self.ff_gate_scratch.ptr, self.ff_up_scratch.ptr, self.ff_gate_scratch.ptr, ff);
        gemv_mxfp8_fn(self.ff_gate_scratch.ptr, dw.?.data_ptr, ds2.?.data_ptr, self.expert_scratch.ptr, e, ff, @intCast(ds2.?.shape[1]));
        for (0..e) |i| self.hidden2[i] += self.expert_scratch[i];
    }

    /// Prefill: process all token IDs sequentially through forward().
    /// Returns the argmax of the last token's logits.
    ///
    /// NOTE: Batched (chunked) prefill is deferred for DeepSeek V4.
    /// Unlike standard transformers where layers are independent residual blocks,
    /// DS4 has several features that make batched prefill a research-level task:
    ///
    /// 1. **4-stream Hyper Connections (HC):** Every layer is wrapped by hcPre/hcPost
    ///    which mix 4 residual streams via learned combination matrices. Batching
    ///    would require [4 × chunk_size × n_embd] HC state buffers and careful
    ///    per-token sequential HC updates (the streams carry state across tokens).
    ///
    /// 2. **MoE routing is per-token:** 256-expert top-6 routing (with hash routing
    ///    on layers 0-2) produces different expert sets per token, same as GLM-4
    ///    and GPT-OSS, but combined with HC makes the bookkeeping much harder.
    ///
    /// 3. **CSA/HCA compressors have per-position state:** Compressed KV blocks
    ///    accumulate over positions with softmax scoring. The write targets are
    ///    sequential and position-dependent.
    ///
    /// 4. **Grouped output LoRA:** 8-group × 1024-rank attention output projection
    ///    is non-standard and would need its own batched path.
    ///
    /// The pf_* buffers are allocated for forward compatibility, a future
    /// implementation can batch the MLA attention within each layer (as GLM-4 does)
    /// while keeping HC, MoE, and compressor passes per-token.
    pub fn prefill(self: *Ds4Model, token_ids: []const u32) !u32 {
        var last: u32 = 0;
        for (token_ids) |tid| last = try self.forward(tid);
        return last;
    }

    /// Reset the KV cache position for a new conversation and clear the cancellation flag.
    pub fn resetCache(self: *Ds4Model) void {
        self.kv_seq_len = 0;
        self.cancelled.store(false, .release);
    }

    /// Signal an in-progress forward pass to abort. Thread-safe.
    pub fn cancel(self: *Ds4Model) void {
        self.cancelled.store(true, .release);
    }

    // ---- MLX-aware GEMV dispatch ----

    /// Dedicated CpuBackend for CSA/HCA, LID, HC, rmsNorm, and q8_0 SDPA.
    /// Those ops read host f32. Vulkan and WebGPU run MLX-Q / MXFP4 GEMV on
    /// native shaders via `gemvBackend()`, then download so this CPU path sees
    /// fresh activations. Metal/CUDA/ROCm keep the full CpuBackend bypass
    /// (UMA incoherence / unverified GPU GEMV).
    fn computeBackend(self: *Ds4Model) Backend {
        return .{ .cpu = &self.cpu };
    }

    /// GEMV backend: native Vulkan/WebGPU/CUDA shaders for Flash MLX-Q attention
    /// and MXFP4 experts. Other backends stay on the dedicated CpuBackend.
    /// CUDA GEMVs copy outputs back synchronously (syncGemvOutput), matching the
    /// Vulkan per-call upload/download semantics so the CPU-side pooling, LID,
    /// and HC passes see fresh activations between GEMVs.
    fn gemvBackend(self: *Ds4Model) Backend {
        return switch (self.be) {
            .vulkan, .webgpu, .cuda => self.be,
            else => .{ .cpu = &self.cpu },
        };
    }

    /// Expert-parallel degree. Without a transport the rank still holds every
    /// expert (mmap pages for unused shards stay cold once EP is active).
    fn epDegree(self: *const Ds4Model) u32 {
        return if (self.tp_transport != null) self.tp_degree else 1;
    }

    /// Prefault (CPU-touch + device-copy) the routed-expert pages this TP
    /// rank owns, releasing the host pages after each copy. On a unified-
    /// memory part the device copy becomes the source of truth, so the
    /// EP-local working set (~half the model) is the only resident footprint
    /// (measured: the demand-paged mmap thrashed at 8-30s/token of FFN
    /// page-fault stalls). Both the FP4 packed weights AND their E8M0 scale
    /// tensors are made resident, the scales are read on every expert gemv
    /// and would otherwise demand-page during decode.
    ///
    /// Ranges are processed in address order (file order within each shard)
    /// so the disk reads during the copy are sequential rather than jumping
    /// between 48 shards per layer.
    pub fn prefaultLocalExperts(self: *Ds4Model) void {
        if (self.epDegree() <= 1) return;
        const ne = self.n_experts;
        const Range = struct { ptr: [*]const u8, len: usize };
        var ranges: std.ArrayListUnmanaged(Range) = .empty;
        defer ranges.deinit(self.allocator);
        for (0..self.n_layers) |li| {
            const projs = [_][]const u8{ "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight" };
            for (projs) |tn| {
                const t = self.layerTensor(li, tn) orelse continue;
                if (t.n_dims != 2 or t.dims[1] != 1) continue; // fused pointer table
                const tbl: [*]const [*]const u8 = @ptrCast(@alignCast(t.data_ptr));
                const w_size = self.expertPackSize(tn);
                // Companion E8M0 scale table (MLX FP4 gs=32): scales are
                // [out, in/32] bytes vs [out, in/2] for the packed weights.
                var s_tbl: ?[*]const [*]const u8 = null;
                var s_buf: [name_buf_size]u8 = undefined;
                const prefix = tn[0 .. tn.len - ".weight".len];
                const s_name = std.fmt.bufPrint(&s_buf, "{s}.scales", .{prefix}) catch null;
                if (s_name) |sn| {
                    if (self.fmt.getTensor(sn)) |st| {
                        if (st.n_dims == 2 and st.dims[1] == 1) {
                            s_tbl = @ptrCast(@alignCast(st.data_ptr));
                        }
                    }
                }
                const s_size = w_size / (ds4_flash_fp4_group_size / 2); // E8M0 [out, in/32] for gs=32 FP4
                for (0..ne) |eid| {
                    if (!isLocalExpert(eid, self.tp_rank, self.epDegree())) continue;
                    // Broken checkpoints (tiny-random) emit NULL scale
                    // entries — skip them like the gemv path.
                    if (s_tbl) |stp| {
                        if (@intFromPtr(stp[eid]) == 0) continue;
                    }
                    if (@intFromPtr(tbl[eid]) == 0) continue; // NULL weight row
                    ranges.append(self.allocator, .{ .ptr = tbl[eid], .len = w_size }) catch continue;
                    if (s_tbl) |stp| ranges.append(self.allocator, .{ .ptr = stp[eid], .len = s_size }) catch continue;
                }
            }
        }
        std.mem.sort(Range, ranges.items, {}, struct {
            fn lt(_: void, a: Range, b: Range) bool {
                return @intFromPtr(a.ptr) < @intFromPtr(b.ptr);
            }
        }.lt);
        var n_ranges: usize = 0;
        var n_bytes: usize = 0;
        for (ranges.items) |r| {
            self.be.residentWeight(r.ptr, r.len);
            n_ranges += 1;
            n_bytes += r.len;
        }
        // Residual mmap'd weights the GPU reads every token (router, HC
        // streams, LM head): make them resident too so their host pages can
        // be released below. Otherwise the decode's first-touch uploads would
        // re-fault them from disk at 4KB speed and their resident pages would
        // keep the 121GB machine under constant memory pressure (measured:
        // multi-second cuMemAlloc stalls at every layer's combine sync).
        const residual_names = [_][]const u8{
            "ffn_gate_inp.weight",
            "hc_attn_fn.weight",
            "hc_attn_base.weight",
            "hc_attn_scale.weight",
            "hc_ffn_fn.weight",
            "hc_ffn_base.weight",
            "hc_ffn_scale.weight",
        };
        for (0..self.n_layers) |li| {
            for (residual_names) |tn| {
                if (self.layerTensor(li, tn)) |t| {
                    if (t.n_dims > 0) {
                        self.be.residentWeight(t.data_ptr, t.dataByteLen());
                        n_ranges += 1;
                        n_bytes += t.dataByteLen();
                    }
                }
            }
        }
        if (self.fmt.getTensor("output.weight")) |t| {
            if (t.n_dims > 0) {
                self.be.residentWeight(t.data_ptr, t.dataByteLen());
                n_ranges += 1;
                n_bytes += t.dataByteLen();
            }
        }
        // The copy set SEQUENTIAL readahead on the shard mmaps; decode must
        // go back to RANDOM so scattered reads don't over-read whole shards.
        // The non-expert shard pages are also DONTNEED'd, everything the GPU
        // needs is device-resident or heap-repacked, so keeping ~50GB of dead
        // mmap pages resident only fuels the allocation-pressure stalls.
        self.be.restoreMmapHints();
        self.experts_resident = true;
        std.log.info("DS4: made {d} local expert ranges ({d} MiB) resident (rank {d})", .{ n_ranges, n_bytes >> 20, self.tp_rank });
    }

    /// Packed byte size of one routed-expert tensor slice (FP4: out × in/2).
    fn expertPackSize(self: *const Ds4Model, tensor_name: []const u8) usize {
        const e = self.n_embd;
        const ff = self.ff_exp;
        if (std.mem.indexOf(u8, tensor_name, "down") != null) return e * (ff / 2);
        return ff * (e / 2);
    }

    /// Dispatch a single GEMV through the format-aware path.
    /// Handles MLX-Q (affine with companion scales/biases), NVFP4, GPTQ, AWQ, HQQ,
    /// and standard GGUF quantized weights transparently.
    /// Check if an MLX-Q tensor should use MXFP4 decode instead of affine.
    /// MLX 4-bit models use mxfp4 mode for expert weights (E2M1 LUT)
    /// and affine mode for attention weights (scale × int + bias).
    fn mlxExpertIsMxfp4(self: *const Ds4Model, t: TensorInfo) bool {
        if (t.dtype != .mlx_q) return false;
        if (!self.fmt.is_safetensors) return false;
        // Expert weights use mxfp4 mode (check tensor name for "exps" or "switch_mlp")
        const name = t.name;
        return std.mem.indexOf(u8, name, "exps") != null or
            std.mem.indexOf(u8, name, "switch_mlp") != null;
    }

    fn doGemv(self: *Ds4Model, x: [*]const f32, t_raw: TensorInfo, y: [*]f32, n: usize, k: usize) void {
        // MLX/MXFP4 dtypes use the native GPU backend with per-call sync
        // copy-back. Other dtypes (bf16/f32 attention projections, shared
        // experts): on CUDA the GPU runs the GEMV followed by a per-call
        // copy-back (Backend.syncGemvOutput) so DS4's interleaved CPU reads
        // (rmsNorm between projections) see fresh data; on other backends
        // the cache-based GPU gemv would leave stale host copies, so the
        // dedicated CpuBackend is used.
        const be = self.gemvBackend();
        if (t_raw.dtype == .mlx_q) {
            if (model_mod.mlxGemv(be, self.fmt, x, t_raw, y, n, k)) return;
        }
        const t = self.heapTensor(t_raw);
        switch (self.be) {
            .cuda => {
                model_mod.dispatchGemv(self.gemvBackend(), self.fmt, x, t, y, n, k);
                self.be.syncGemvOutput(y, n);
                // The device copy is cached now — the large repacked host
                // buffer is dead weight. Free it incrementally so the first
                // forward's ~26GB of attention uploads don't OOM the node.
                self.fmt.freeRepackedTensor(t.data_ptr);
            },
            else => {
                model_mod.dispatchGemv(self.computeBackend(), self.fmt, x, t, y, n, k);
            },
        }
    }

    /// Resolve a routed expert's packed-weight + scale DEVICE pointers from
    /// the fused pointer table (batched-CUDA path). Returns false when the
    /// expert is invalid (broken routing ids, NULL entries) or not the
    /// MLX-FP4 table form.
    fn expertDevicePair(self: *Ds4Model, exp_t: TensorInfo, ei: usize, n: usize, k: usize, out_w: *u64, out_s: *u64) bool {
        const is_tbl = (exp_t.n_dims == 2 and exp_t.dims[1] == 1);
        if (!is_tbl) return false;
        if (ei >= exp_t.dims[0]) return false; // broken routing id
        const tbl: [*]const [*]const u8 = @ptrCast(@alignCast(exp_t.data_ptr));
        const data = tbl[ei];
        if (@intFromPtr(data) == 0) return false;
        const wi = std.mem.lastIndexOf(u8, exp_t.name, ".weight") orelse return false;
        var sbuf: [name_buf_size]u8 = undefined;
        const prefix = exp_t.name[0..wi];
        const s_name = std.fmt.bufPrint(&sbuf, "{s}.scales", .{prefix}) catch return false;
        const st = self.fmt.getTensor(s_name) orelse return false;
        if (!(st.dtype == .unknown or st.dtype == .nvfp4)) return false;
        if (!(st.n_dims == 2 and st.dims[1] == 1)) return false;
        if (ei >= st.dims[0]) return false;
        const stbl: [*]const [*]const u8 = @ptrCast(@alignCast(st.data_ptr));
        const s_data = stbl[ei];
        if (@intFromPtr(s_data) == 0) return false;
        const mxfp4_gs: usize = ds4_flash_fp4_group_size;
        const gpr = (k + mxfp4_gs - 1) / mxfp4_gs;
        const wpg: usize = mxfp4_gs * 4 / 32;
        const w_bytes = n * gpr * wpg * @sizeOf(u32);
        out_w.* = self.be.getWeightDevicePtr(data, w_bytes);
        out_s.* = self.be.getWeightDevicePtr(s_data, n * gpr);
        return true;
    }

    /// Dispatch a GEMV for a single expert slice from a packed expert tensor.
    /// Handles MLX-Q companion tensor slicing for per-expert scale/bias offsets.
    /// Expert tensors fused by the official DeepSeek-V4-Flash-0731 loader are
    /// POINTER TABLES ([n_experts, 1] u64 mmap addresses, the checkpoint's
    /// experts are at non-uniform file offsets, so direct stride math is
    /// impossible) and are dereferenced per expert here.
    fn doGemvExpert(self: *Ds4Model, x: [*]const f32, exp_t: TensorInfo, ei: usize, stride: usize, y: [*]f32, n: usize, k: usize) void {
        const is_tbl = (exp_t.n_dims == 2 and exp_t.dims[1] == 1);
        // Out-of-range routing id (broken checkpoints: the tiny-random test
        // model emits expert ids beyond the table size). Skip like the CPU
        // path, the fused tables only cover [0, n_experts).
        if (is_tbl and ei >= exp_t.dims[0]) return;
        const data = if (is_tbl)
            (@as([*]const [*]const u8, @ptrCast(@alignCast(exp_t.data_ptr))))[ei]
        else
            exp_t.data_ptr + ei * stride;
        if (@intFromPtr(data) == 0) {
            std.log.err("DS4 doGemvExpert: NULL weight pointer for {s} eid={d} is_tbl={}", .{ exp_t.name, ei, is_tbl });
            return;
        }
        if (exp_t.dtype != .mlx_q) {
            self.computeBackend().gemv(x, .{ .data = data, .dtype = exp_t.dtype }, y, n, k);
            return;
        }
        const wi = std.mem.lastIndexOf(u8, exp_t.name, ".weight") orelse return;
        var sbuf: [name_buf_size]u8 = undefined;
        const prefix = exp_t.name[0..wi];
        const s_name = std.fmt.bufPrint(&sbuf, "{s}.scales", .{prefix}) catch return;
        const st = self.fmt.getTensor(s_name) orelse return;
        if (st.dtype == .unknown or st.dtype == .nvfp4) {
            const s_is_tbl = (st.n_dims == 2 and st.dims[1] == 1);
            const s_stride = if (st.n_dims >= 3)
                @as(usize, @intCast(st.dims[1])) * @as(usize, @intCast(st.dims[2]))
            else
                n * @as(usize, @intCast(st.dims[st.n_dims - 1]));
            const s_data = if (s_is_tbl)
                (@as([*]const [*]const u8, @ptrCast(@alignCast(st.data_ptr))))[ei]
            else
                st.data_ptr + ei * s_stride;
            if (@intFromPtr(s_data) == 0) {
                std.log.err("DS4 doGemvExpert: NULL scale pointer for {s} eid={d}", .{ exp_t.name, ei });
                return;
            }
            const mxfp4_gs = if (s_is_tbl) ds4_flash_fp4_group_size else model_mod.inferMxfp4GroupSize(st, k);
            const sf = mlx_ops.mxfp4ScaleFormat(self.fmt.is_safetensors, mxfp4_gs);
            const mlx = @import("../ops/mlx.zig");
            switch (self.be) {
                .vulkan, .webgpu, .cuda => {
                    self.gemvBackend().gemvMxfp4StGpu(x, data, s_data, y, n, k, mxfp4_gs, sf);
                },
                else => {
                    if (self.pool) |pool| {
                        var ctx = struct {
                            xp: [*]const f32,
                            wp: [*]const u8,
                            sp: [*]const u8,
                            yp: [*]f32,
                            kv: usize,
                            gs_v: usize,
                            sf_v: mlx.Mxfp4ScaleFormat,
                            fn work(c_ptr: *anyopaque, start: usize, end: usize) void {
                                const c: *const @This() = @ptrCast(@alignCast(c_ptr));
                                mlx.mlxMxfp4GemvRows(c.xp, @ptrCast(@alignCast(c.wp)), c.sp, @ptrCast(c.yp), start, end - start, c.kv, c.gs_v, c.sf_v);
                            }
                        }{ .xp = x, .wp = data, .sp = s_data, .yp = y, .kv = k, .gs_v = mxfp4_gs, .sf_v = sf };
                        pool.parallelFor(n, 128, @ptrCast(&ctx), @TypeOf(ctx).work);
                    } else mlx.mlxMxfp4GemvRows(x, @ptrCast(@alignCast(data)), s_data, @ptrCast(y), 0, n, k, mxfp4_gs, sf);
                },
            }
        } else {
            var bbuf: [name_buf_size]u8 = undefined;
            const b_name = std.fmt.bufPrint(&bbuf, "{s}.biases", .{prefix}) catch return;
            const bt = self.fmt.getTensor(b_name) orelse return;
            const s_stride = if (st.n_dims >= 3)
                @as(usize, @intCast(st.dims[1])) * @as(usize, @intCast(st.dims[2])) * 2
            else
                n * @as(usize, @intCast(st.dims[st.n_dims - 1])) * 2;
            const bits: u32 = if (exp_t.n_dims >= 2 and k > 0)
                @intCast(@as(u64, exp_t.dims[exp_t.n_dims - 1]) * 32 / @as(u64, @intCast(k)))
            else
                8;
            const gs_e = model_mod.inferMlxGroupSize(st, k);
            const mlx2 = @import("../ops/mlx.zig");
            switch (self.be) {
                .vulkan, .webgpu, .cuda => {
                    self.gemvBackend().gemvMlxQGpu(x, data, st.data_ptr + ei * s_stride, bt.data_ptr + ei * s_stride, y, n, k, bits, gs_e);
                },
                else => {
                    if (self.pool) |pool| {
                        var ctx = struct {
                            xp: [*]const f32,
                            wp: [*]const u8,
                            sp: [*]const u8,
                            bp: [*]const u8,
                            yp: [*]f32,
                            kv: usize,
                            b: u32,
                            g: u32,
                            fn work(c_ptr: *anyopaque, start: usize, end: usize) void {
                                const c: *const @This() = @ptrCast(@alignCast(c_ptr));
                                mlx2.mlxGemvRows(c.xp, @ptrCast(@alignCast(c.wp)), @ptrCast(@alignCast(c.sp)), @ptrCast(@alignCast(c.bp)), @ptrCast(c.yp), start, end - start, c.kv, c.b, c.g);
                            }
                        }{ .xp = x, .wp = data, .sp = st.data_ptr + ei * s_stride, .bp = bt.data_ptr + ei * s_stride, .yp = y, .kv = k, .b = bits, .g = gs_e };
                        pool.parallelFor(n, 128, @ptrCast(&ctx), @TypeOf(ctx).work);
                    } else mlx2.mlxGemvRaw(x, @ptrCast(@alignCast(data)), @ptrCast(@alignCast(st.data_ptr + ei * s_stride)), @ptrCast(@alignCast(bt.data_ptr + ei * s_stride)), @ptrCast(y), n, k, bits, gs_e);
                },
            }
        }
    }

    /// Enable or disable the fused megakernel dispatch path.
    pub fn setMegakernel(self: *Ds4Model, en: bool) void {
        self.megakernel_enabled = en;
    }
    /// Set the layer range to skip during self-speculative decoding.
    /// Layers in `[s, end)` are bypassed in the draft pass.
    pub fn setLayerSkip(self: *Ds4Model, s: u32, end: u32) void {
        self.layer_skip_start = s;
        self.layer_skip_end = end;
    }
    /// Return physical block IDs for the paged KV cache. DeepSeek V4 uses a flat
    /// KV layout (not paged), so this always returns an empty slice.
    pub fn getBlockTable(_: *const Ds4Model) []const u32 {
        return &.{};
    }
    /// MTP HC pre: compute weighted sum of MTP HC streams → hidden2.
    /// Uses MTP-specific HC weights from safetensors.
    fn mtpHcPre(self: *Ds4Model, mtp: *const MtpWeights, layer: usize, e: usize) void {
        var b1: [64]u8 = undefined;
        var b2: [64]u8 = undefined;
        var b3: [64]u8 = undefined;
        const fn_name = std.fmt.bufPrint(&b1, "mtp.{d}.hc_attn_fn", .{layer}) catch return;
        const base_name = std.fmt.bufPrint(&b2, "mtp.{d}.hc_attn_base", .{layer}) catch return;
        const scale_name = std.fmt.bufPrint(&b3, "mtp.{d}.hc_attn_scale", .{layer}) catch return;
        const fn_t = mtp.get(fn_name) orelse return;
        const base_t = mtp.get(base_name) orelse return;
        const scale_t = mtp.get(scale_name) orelse return;

        const flat_size = n_hc * e;
        // RMS scale factor from mtp_hc_state
        const V8 = @Vector(8, f32);
        const rms_inv = blk: {
            var acc: V8 = @splat(0.0);
            var ri: usize = 0;
            while (ri + 8 <= flat_size) : (ri += 8) {
                const v: V8 = self.mtp_hc_state[ri..][0..8].*;
                acc = @mulAdd(V8, v, v, acc);
            }
            var ss: f32 = @reduce(.Add, acc);
            while (ri < flat_size) : (ri += 1) ss += self.mtp_hc_state[ri] * self.mtp_hc_state[ri];
            break :blk 1.0 / @sqrt(ss / @as(f32, @floatFromInt(flat_size)) + self.rms_eps);
        };

        // mixes[24] = hc_fn @ mtp_hc_state (F32 GEMV, tiny)
        var mixes: [hc_mix_dim]f32 = undefined;
        cpuGemvF32(fn_t.data_ptr, self.mtp_hc_state, &mixes, flat_size);
        for (&mixes) |*m| m.* *= rms_inv;

        const base: [*]const f32 = @ptrCast(@alignCast(base_t.data_ptr));
        const scale: [*]const f32 = @ptrCast(@alignCast(scale_t.data_ptr));

        // pre/post/comb weights
        for (0..n_hc) |s| {
            self.hc_pre_w[s] = sigmoid(mixes[s] * scale[0] + base[s]) + hc_eps;
            self.hc_post_w[s] = sigmoid(mixes[n_hc + s] * scale[1] + base[n_hc + s]) * 2.0;
        }
        for (0..n_hc * n_hc) |s| {
            self.hc_comb[s] = mixes[2 * n_hc + s] * scale[2] + base[2 * n_hc + s];
        }
        hcSinkhorn(self.hc_comb);

        // Weighted sum: hidden2 = Σ pre_w[s] * mtp_hc_state[s]
        var i: usize = 0;
        while (i + 8 <= e) : (i += 8) {
            var acc2: V8 = @splat(@as(f32, 0.0));
            for (0..n_hc) |s| {
                const w: V8 = @splat(self.hc_pre_w[s]);
                acc2 = @mulAdd(V8, @as(V8, self.mtp_hc_state[s * e + i ..][0..8].*), w, acc2);
            }
            self.hidden2[i..][0..8].* = acc2;
        }
        while (i < e) : (i += 1) {
            var v: f32 = 0.0;
            for (0..n_hc) |s| v += self.mtp_hc_state[s * e + i] * self.hc_pre_w[s];
            self.hidden2[i] = v;
        }
    }

    /// MTP HC post: update MTP HC state from sublayer output in hidden2.
    fn mtpHcPost(self: *Ds4Model, e: usize) void {
        const sub = self.hidden2;
        const V8 = @Vector(8, f32);
        for (0..n_hc) |dst| {
            const ns = self.new_hc[dst * e ..][0..e];
            const pw: V8 = @splat(self.hc_post_w[dst]);
            var cvec: [n_hc]V8 = undefined;
            for (0..n_hc) |src| cvec[src] = @splat(self.hc_comb[dst + src * n_hc]);
            var i: usize = 0;
            while (i + 8 <= e) : (i += 8) {
                var acc: V8 = @as(V8, sub[i..][0..8].*) * pw;
                for (0..n_hc) |src| {
                    acc = @mulAdd(V8, @as(V8, self.mtp_hc_state[src * e + i ..][0..8].*), cvec[src], acc);
                }
                ns[i..][0..8].* = acc;
            }
            while (i < e) : (i += 1) {
                var v = sub[i] * self.hc_post_w[dst];
                for (0..n_hc) |src| v += self.mtp_hc_state[src * e + i] * self.hc_comb[dst + src * n_hc];
                ns[i] = v;
            }
        }
        // Swap: mtp_hc_state ← new_hc
        @memcpy(self.mtp_hc_state, self.new_hc[0 .. n_hc * e]);
    }

    /// MTP HC pre for FFN: same as mtpHcPre but uses hc_ffn_* weights.
    fn mtpHcPreFfn(self: *Ds4Model, mtp: *const MtpWeights, layer: usize, e: usize) void {
        var b1: [64]u8 = undefined;
        var b2: [64]u8 = undefined;
        var b3: [64]u8 = undefined;
        const fn_name = std.fmt.bufPrint(&b1, "mtp.{d}.hc_ffn_fn", .{layer}) catch return;
        const base_name = std.fmt.bufPrint(&b2, "mtp.{d}.hc_ffn_base", .{layer}) catch return;
        const scale_name = std.fmt.bufPrint(&b3, "mtp.{d}.hc_ffn_scale", .{layer}) catch return;
        const fn_t = mtp.get(fn_name) orelse return;
        const base_t = mtp.get(base_name) orelse return;
        const scale_t = mtp.get(scale_name) orelse return;

        const flat_size = n_hc * e;
        const V8 = @Vector(8, f32);
        const rms_inv = blk: {
            var acc: V8 = @splat(0.0);
            var ri: usize = 0;
            while (ri + 8 <= flat_size) : (ri += 8) {
                const v: V8 = self.mtp_hc_state[ri..][0..8].*;
                acc = @mulAdd(V8, v, v, acc);
            }
            var ss: f32 = @reduce(.Add, acc);
            while (ri < flat_size) : (ri += 1) ss += self.mtp_hc_state[ri] * self.mtp_hc_state[ri];
            break :blk 1.0 / @sqrt(ss / @as(f32, @floatFromInt(flat_size)) + self.rms_eps);
        };

        var mixes: [hc_mix_dim]f32 = undefined;
        cpuGemvF32(fn_t.data_ptr, self.mtp_hc_state, &mixes, flat_size);
        for (&mixes) |*m| m.* *= rms_inv;

        const base: [*]const f32 = @ptrCast(@alignCast(base_t.data_ptr));
        const scale: [*]const f32 = @ptrCast(@alignCast(scale_t.data_ptr));

        for (0..n_hc) |s| {
            self.hc_pre_w[s] = sigmoid(mixes[s] * scale[0] + base[s]) + hc_eps;
            self.hc_post_w[s] = sigmoid(mixes[n_hc + s] * scale[1] + base[n_hc + s]) * 2.0;
        }
        for (0..n_hc * n_hc) |s| {
            self.hc_comb[s] = mixes[2 * n_hc + s] * scale[2] + base[2 * n_hc + s];
        }
        hcSinkhorn(self.hc_comb);

        var i: usize = 0;
        while (i + 8 <= e) : (i += 8) {
            var acc2: V8 = @splat(@as(f32, 0.0));
            for (0..n_hc) |s| {
                const w: V8 = @splat(self.hc_pre_w[s]);
                acc2 = @mulAdd(V8, @as(V8, self.mtp_hc_state[s * e + i ..][0..8].*), w, acc2);
            }
            self.hidden2[i..][0..8].* = acc2;
        }
        while (i < e) : (i += 1) {
            var v: f32 = 0.0;
            for (0..n_hc) |s| v += self.mtp_hc_state[s * e + i] * self.hc_pre_w[s];
            self.hidden2[i] = v;
        }
    }

    /// Batched forward for speculative verification.
    /// Processes n_nodes tokens through all layers simultaneously.
    /// Uses GEMM (matrix-matrix) instead of GEMV for weight projections.
    /// Simplified: shared expert only, no HC mixing, no compressors.
    /// Maximum f32 temp buffer for dequant+SGEMM path (32MB = 8M floats).
    const max_dequant_f32_elems: usize = 8 * 1024 * 1024;

    /// Dispatch batched GEMM for MLX-Q tensors.
    /// For small-to-medium weight matrices: dequant to f32, then Accelerate SGEMM (AMX).
    /// Falls back to be.gemm (N×GEMV) for large matrices or non-MLX-Q dtypes.
    fn batchedGemm(self: *Ds4Model, x: [*]const f32, t: TensorInfo, y: [*]f32, n_tok: usize, n_out: usize, n_in: usize) void {
        // Direct N×GEMV path. Dequant+SGEMM was tested (iters 33-36) but dequant overhead
        // cancels AMX speedup when weights are page-cached (8× memory expansion).
        // Dequant+SGEMM disabled: dequant overhead cancels AMX speedup on page-cached weights.
        // Enable with comptime flag when page cache is cold (SSD-streamed models).
        const use_dequant_sgemm = false;
        if (use_dequant_sgemm and comptime @import("builtin").os.tag == .macos and build_options.enable_metal) {
            if (n_tok > 1 and t.dtype == .mlx_q and n_out * n_in <= max_dequant_f32_elems) {
                if (model_mod.findMlxCompanion(self.fmt, t, n_in)) |comp| {
                    if (comp.bits == 4) {
                        // Dequant full weight matrix [n_out, n_in] to f32 temp buffer.
                        // Uses pf_q as temp (largest prefill buffer: cs * nh * kd).
                        const needed = n_out * n_in;
                        if (self.pf_q.len >= needed) {
                            const pw: [*]const u32 = @ptrCast(@alignCast(t.data_ptr));
                            const sc: [*]const u16 = @ptrCast(@alignCast(comp.scales));
                            const bi: [*]const u16 = @ptrCast(@alignCast(comp.biases));
                            // Dequant each row: could parallelize but dequant is fast vs SGEMM
                            if (self.pool) |pool| {
                                var dctx = DequantCtx{ .out = self.pf_q.ptr, .pw = pw, .sc = sc, .bi = bi, .k = n_in, .bits = comp.bits, .gs = comp.group_size };
                                pool.parallelFor(n_out, 64, @ptrCast(&dctx), DequantCtx.work);
                            } else {
                                for (0..n_out) |row| {
                                    mlx_ops.mlxEmbLookup(self.pf_q.ptr + row * n_in, pw, sc, bi, row, n_in, comp.bits);
                                }
                            }
                            // SGEMM: y[n_tok, n_out] = x[n_tok, n_in] × W[n_out, n_in]^T
                            const accel = backend_mod.accelerate;
                            accel.sgemm(n_tok, n_out, n_in, x, self.pf_q.ptr, y);
                            return;
                        }
                    }
                }
            }
        }
        // Tiled dequant+SGEMM for large MLX-Q 4-bit matrices.
        // Process in row-tiles that fit in the temp buffer.
        if (comptime @import("builtin").os.tag == .macos and build_options.enable_metal) {
            if (n_tok > 1 and t.dtype == .mlx_q and n_in <= max_dequant_f32_elems) {
                if (model_mod.findMlxCompanion(self.fmt, t, n_in)) |comp| {
                    if (comp.bits == 4) {
                        const tile_rows = max_dequant_f32_elems / n_in; // rows per tile
                        if (tile_rows > 0 and self.pf_q.len >= tile_rows * n_in) {
                            const pw: [*]const u32 = @ptrCast(@alignCast(t.data_ptr));
                            const sc: [*]const u16 = @ptrCast(@alignCast(comp.scales));
                            const bi: [*]const u16 = @ptrCast(@alignCast(comp.biases));
                            const accel = backend_mod.accelerate;
                            var row_start: usize = 0;
                            while (row_start < n_out) {
                                const tile_n = @min(tile_rows, n_out - row_start);
                                // Dequant this tile
                                if (self.pool) |pool| {
                                    var dctx = DequantRowCtx{ .out = self.pf_q.ptr, .pw = pw, .sc = sc, .bi = bi, .k = n_in, .bits = comp.bits, .gs = comp.group_size, .row_off = row_start };
                                    pool.parallelFor(tile_n, 64, @ptrCast(&dctx), DequantRowCtx.work);
                                } else {
                                    for (0..tile_n) |r| {
                                        mlx_ops.mlxEmbLookup(self.pf_q.ptr + r * n_in, pw, sc, bi, row_start + r, n_in, comp.bits);
                                    }
                                }
                                // SGEMM: y_tile[n_tok, tile_n] = x[n_tok, n_in] × tile_W[tile_n, n_in]^T
                                // Output goes to y[t * n_out + row_start..]
                                // Accelerate sgemm writes y = x @ W^T (row-major)
                                // y[n_tok, tile_n] written contiguously, then copy to strided output
                                if (tile_n == n_out) {
                                    // Full matrix, write directly to y
                                    accel.sgemm(n_tok, n_out, n_in, x, self.pf_q.ptr, y);
                                } else {
                                    // Tile, need to write to correct columns in y
                                    // Use a temp output buffer (reuse pf_kv_proj if big enough)
                                    if (self.pf_kv_proj.len >= n_tok * tile_n) {
                                        accel.sgemm(n_tok, tile_n, n_in, x, self.pf_q.ptr, self.pf_kv_proj.ptr);
                                        // Copy tile output to strided y
                                        for (0..n_tok) |tok| {
                                            @memcpy(y[tok * n_out + row_start ..][0..tile_n], self.pf_kv_proj[tok * tile_n ..][0..tile_n]);
                                        }
                                    } else {
                                        // Buffer too small, fall back to per-tile GEMV
                                        for (0..n_tok) |tok| {
                                            for (0..tile_n) |r| {
                                                var sum: f32 = 0;
                                                const wr = self.pf_q.ptr + r * n_in;
                                                for (0..n_in) |i| sum += x[tok * n_in + i] * wr[i];
                                                y[tok * n_out + row_start + r] = sum;
                                            }
                                        }
                                    }
                                }
                                row_start += tile_n;
                            }
                            return;
                        }
                    }
                }
            }
        }
        self.computeBackend().gemm(x, .{ .data = t.data_ptr, .dtype = t.dtype }, y, n_tok, n_out, n_in);
    }

    /// Thread-pool context for parallel dequant with row offset.
    const DequantRowCtx = struct {
        out: [*]f32,
        pw: [*]const u32,
        sc: [*]const u16,
        bi: [*]const u16,
        k: usize,
        bits: u32,
        gs: u32,
        row_off: usize,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *const DequantRowCtx = @ptrCast(@alignCast(ctx_ptr));
            for (start..end) |r| {
                mlx_ops.mlxEmbLookup(ctx.out + r * ctx.k, ctx.pw, ctx.sc, ctx.bi, ctx.row_off + r, ctx.k, ctx.bits);
            }
        }
    };

    /// Thread-pool context for parallel MLX-Q dequantization.
    const DequantCtx = struct {
        out: [*]f32,
        pw: [*]const u32,
        sc: [*]const u16,
        bi: [*]const u16,
        k: usize,
        bits: u32,
        gs: u32,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *const DequantCtx = @ptrCast(@alignCast(ctx_ptr));
            for (start..end) |row| {
                mlx_ops.mlxEmbLookup(ctx.out + row * ctx.k, ctx.pw, ctx.sc, ctx.bi, row, ctx.k, ctx.bits);
            }
        }
    };

    /// Attention window for forwardTree verification.
    /// Same raw window as decode: latest `ds4_raw_attn_window` positions.
    const ft_attn_window: usize = ds4_raw_attn_window;

    /// Per-head attention for a single token in forwardTree.
    fn pf_attn_head(self: *Ds4Model, t: usize, h: usize, pos: usize, kv_k_layer: [*]const u8, kd: usize, nh: usize, scale: f32, kv_elem_bytes: usize, ss: usize) void {
        const q_h = self.pf_q[t * nh * kd + h * kd ..][0..kd];
        const ao_h = self.pf_attn_out[t * nh * kd + h * kd ..][0..kd];
        const scores = self.scores_buf[h * ss ..];
        // Approximate attention: only score against the last ft_attn_window positions.
        const start_p: usize = if (ft_attn_window > 0 and pos + 1 > ft_attn_window) pos + 1 - ft_attn_window else 0;
        const n_score = pos + 1 - start_p;
        var running_max: f32 = -std.math.inf(f32);
        for (0..n_score) |pi| {
            const p = start_p + pi;
            const s = kv_quant.kvDot(q_h.ptr, kv_k_layer + p * kv_elem_bytes, kd, self.kv_type) * scale;
            scores[p] = s;
            running_max = @max(running_max, s);
        }
        var sm: f32 = 0;
        for (0..n_score) |pi| {
            scores[pi] = @exp(scores[pi] - running_max);
            sm += scores[pi];
        }
        const inv = 1.0 / sm;
        @memset(ao_h, 0.0);
        for (0..n_score) |pi| {
            const w = scores[pi] * inv;
            if (w < 1e-6) continue;
            kv_quant.kvMulAccum(ao_h.ptr, w, kv_k_layer + (start_p + pi) * kv_elem_bytes, kd, self.kv_type);
        }
    }

    /// Thread-pool context for parallel attention over heads in forwardTree.
    const PfAttnCtx = struct {
        pf_q: [*]const f32,
        pf_attn_out: [*]f32,
        kv_k_layer: [*]const u8,
        scores_buf: [*]f32,
        kd: usize,
        nh: usize,
        pos: usize,
        t: usize,
        scale: f32,
        kv_elem_bytes: usize,
        ss: usize,
        kv_type: @import("../ops/kv_quant.zig").KvQuantType,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *const PfAttnCtx = @ptrCast(@alignCast(ctx_ptr));
            for (start..end) |h| {
                const q_h = ctx.pf_q + ctx.t * ctx.nh * ctx.kd + h * ctx.kd;
                const ao_h = ctx.pf_attn_out + ctx.t * ctx.nh * ctx.kd + h * ctx.kd;
                const scores = ctx.scores_buf + h * ctx.ss;
                const start_p2: usize = if (ft_attn_window > 0 and ctx.pos + 1 > ft_attn_window) ctx.pos + 1 - ft_attn_window else 0;
                const n_score2 = ctx.pos + 1 - start_p2;
                var running_max: f32 = -std.math.inf(f32);
                for (0..n_score2) |pi| {
                    const p = start_p2 + pi;
                    const s = kv_quant.kvDot(q_h, ctx.kv_k_layer + p * ctx.kv_elem_bytes, ctx.kd, ctx.kv_type) * ctx.scale;
                    scores[pi] = s;
                    running_max = @max(running_max, s);
                }
                var sm: f32 = 0;
                for (0..n_score2) |pi| {
                    scores[pi] = @exp(scores[pi] - running_max);
                    sm += scores[pi];
                }
                const inv = 1.0 / sm;
                @memset(ao_h[0..ctx.kd], 0.0);
                for (0..n_score2) |pi| {
                    const w = scores[pi] * inv;
                    if (w < 1e-6) continue;
                    kv_quant.kvMulAccum(ao_h, w, ctx.kv_k_layer + (start_p2 + pi) * ctx.kv_elem_bytes, ctx.kd, ctx.kv_type);
                }
            }
        }
    };

    /// Thread-pool context for parallelized batched MLX-Q4 GEMM.
    const BatchedGemmQ4Ctx = struct {
        x: [*]const f32,
        pw: [*]const u32,
        sc: [*]const u16,
        bi: [*]const u16,
        y: [*]f32,
        n_tok: usize,
        n_out: usize,
        k: usize,
        gs: usize,
        gpr: usize,
        wpr: usize,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *const BatchedGemmQ4Ctx = @ptrCast(@alignCast(ctx_ptr));
            // Process output rows [start, end) for all n_tok input vectors
            for (start..end) |row| {
                const wr = ctx.pw + row * ctx.wpr;
                const sr = ctx.sc + row * ctx.gpr;
                const br = ctx.bi + row * ctx.gpr;

                // Per-token accumulators
                var q_dots: [128]f32 = undefined;
                var x_sums: [128]f32 = undefined;
                for (0..ctx.n_tok) |t_idx| {
                    q_dots[t_idx] = 0;
                    x_sums[t_idx] = 0;
                }

                const V = 8; // nibbles_per_u32
                for (0..ctx.gpr) |g| {
                    const scale = @import("../ops/quant.zig").bf16ToF32(sr[g]);
                    const bias = @import("../ops/quant.zig").bf16ToF32(br[g]);
                    const xo = g * ctx.gs;
                    const wo = g * (ctx.gs * 4 / 32); // wpg for 4-bit
                    const elems = @min(ctx.gs, ctx.k - xo);
                    const full_words = elems / V;

                    for (0..full_words) |wi| {
                        const word = wr[wo + wi];
                        // Dequant 8 nibbles
                        var vals: [8]f32 = undefined;
                        inline for (0..8) |ni| {
                            vals[ni] = @as(f32, @floatFromInt((word >> @as(u5, @intCast(ni * 4))) & 0xF));
                        }
                        // Dot with each token's x vector
                        for (0..ctx.n_tok) |t_idx| {
                            const xbase = ctx.x + t_idx * ctx.k + xo + wi * V;
                            var qd: f32 = 0;
                            var xs: f32 = 0;
                            inline for (0..8) |vi| {
                                qd += xbase[vi] * vals[vi];
                                xs += xbase[vi];
                            }
                            q_dots[t_idx] += qd;
                            x_sums[t_idx] += xs;
                        }
                    }

                    // Apply scale+bias
                    for (0..ctx.n_tok) |t_idx| {
                        ctx.y[t_idx * ctx.n_out + row] += scale * q_dots[t_idx] + bias * x_sums[t_idx];
                        q_dots[t_idx] = 0;
                        x_sums[t_idx] = 0;
                    }
                }
            }
        }
    };

    pub fn forwardTree(
        self: *Ds4Model,
        token_ids: []const u32,
        position_ids: []const u32,
        _: [*]const [8]u64, // ancestor_masks (unused, we use standard causal attention)
        n_nodes: u32,
    ) !void {
        if (n_nodes == 0) return;
        self.cpu.pool = self.pool;
        const n: usize = n_nodes;
        const e = self.n_embd;
        const kd: usize = self.kv_lora_rank;
        const ql: usize = self.q_lora_rank;
        const nh: usize = self.n_head;
        const ff: usize = self.ff_exp;
        const rd: usize = self.rope_dim;
        const nope: usize = kd - rd;
        const nd = rd / 2;

        // Lazy-allocate prefill buffers
        if (self.pf_hidden.len == 0) {
            const pa = std.heap.page_allocator;
            const cs = self.chunk_size;
            self.pf_hidden = try pa.alloc(f32, cs * e);
            self.pf_hidden2 = try pa.alloc(f32, cs * e);
            self.pf_q_a = try pa.alloc(f32, cs * ql);
            self.pf_q = try pa.alloc(f32, cs * nh * kd);
            self.pf_kv_proj = try pa.alloc(f32, cs * kd);
            self.pf_attn_out = try pa.alloc(f32, cs * nh * kd);
            self.pf_positions = try pa.alloc(u32, cs);
        }

        // Embedding: lookup each token
        const emb_t = try self.getTensorReq("token_embd.weight");
        const emb_bytes = backend_mod.weightBytes(emb_t.dtype, 1, e);
        for (0..n) |t| {
            const tid = token_ids[t];
            quant_ops.dequantToF32(self.pf_hidden[t * e ..][0..e], emb_t.data_ptr + tid * emb_bytes, emb_t.dtype, e);
        }
        @memcpy(self.pf_positions[0..n], position_ids[0..n]);

        // Process layers, optionally skip early layers for faster verification.
        // forwardTree has no HC, so skipping early layers loses representation depth
        // but doesn't corrupt state propagation.
        const ft_skip: usize = if (self.layer_skip_end > 0) @min(self.layer_skip_end, self.n_layers / 2) else 0;
        const pp_range = ppLayerRange(self.n_layers, self.pp_rank, self.pp_degree);
        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.monotonic)) return error.Cancelled;
            if (self.pp_degree > 1 and (li < pp_range.start or li >= pp_range.end)) continue;

            // Attention norm (batched), always needed for KV projection
            const nw = try self.layerTensorReq(li, "attn_norm.weight");
            self.computeBackend().rmsNormBatched(self.pf_hidden.ptr, self.normAsF32(nw, e), self.pf_hidden2.ptr, n, e, self.rms_eps);

            // KV projection + RoPE + KV cache store (ALWAYS, even for skipped layers).
            // This ensures generation forward() has valid KV data for all layers.
            {
                const kv_a2 = try self.layerTensorReq(li, "attn_kv.weight");
                self.batchedGemm(self.pf_hidden2.ptr, kv_a2, self.pf_kv_proj.ptr, n, kd, e);
                const kv_an2 = try self.layerTensorReq(li, "attn_kv_a_norm.weight");
                self.computeBackend().rmsNormBatched(self.pf_kv_proj.ptr, self.normAsF32(kv_an2, kd), self.pf_kv_proj.ptr, n, kd, self.rms_eps);
                self.be.sync();
                const freqs2 = if (self.compress_ratios[li] != 0) &self.compress_rope_freqs else &self.rope_freqs;
                for (0..n) |t| {
                    var rc: [32]f32 = undefined;
                    var rs: [32]f32 = undefined;
                    for (0..nd) |i| {
                        const theta = @as(f32, @floatFromInt(position_ids[t])) * freqs2[i];
                        rc[i] = @cos(theta);
                        rs[i] = @sin(theta);
                    }
                    applyRopeTable(self.pf_kv_proj[t * kd + nope ..][0..rd], rc[0..nd], rs[0..nd]);
                    const kv_k_l = self.kv_k_bytes[li * self.kvLayerBytes() ..];
                    const k_off = kv_quant.kvByteOffset(self.kv_type, (self.kv_seq_len + t) * kd);
                    kv_quant.kvStore(kv_k_l[k_off..].ptr, self.pf_kv_proj[t * kd ..].ptr, kd, self.kv_type);
                }
            }

            // Skip layers below ft_skip, KV cache already populated above.
            if (li < ft_skip) continue;

            // Q projection: [n, e] → [n, ql] → norm → [n, nh*kd]
            const q_a = try self.layerTensorReq(li, "attn_q_a.weight");
            if (self.pf_dequant_ready and li >= ft_skip) {
                // Use pre-dequanted f32 weights + Accelerate SGEMM
                const rel = li - ft_skip;
                if (comptime @import("builtin").os.tag == .macos and build_options.enable_metal) {
                    const accel = backend_mod.accelerate;
                    accel.sgemm(n, ql, e, self.pf_hidden2.ptr, self.pf_dequant_q_a.ptr + rel * ql * e, self.pf_q_a.ptr);
                } else {
                    self.batchedGemm(self.pf_hidden2.ptr, q_a, self.pf_q_a.ptr, n, ql, e);
                }
            } else {
                self.batchedGemm(self.pf_hidden2.ptr, q_a, self.pf_q_a.ptr, n, ql, e);
            }
            const q_an = try self.layerTensorReq(li, "attn_q_a_norm.weight");
            self.computeBackend().rmsNormBatched(self.pf_q_a.ptr, self.normAsF32(q_an, ql), self.pf_q_a.ptr, n, ql, self.rms_eps);
            const q_b = try self.layerTensorReq(li, "attn_q_b.weight");
            // Batched Q_b: use weight-stationary mlxGemmQ4 with thread pool
            // to read the [32768×1024] weight matrix ONCE for all N tokens.
            if (n > 1 and q_b.dtype == .mlx_q) blk: {
                const comp = model_mod.findMlxCompanion(self.fmt, q_b, ql) orelse break :blk;
                if (comp.bits != 4) break :blk;
                @memset(self.pf_q[0 .. n * nh * kd], 0);
                if (self.pool) |pool| {
                    var ctx = BatchedGemmQ4Ctx{
                        .x = self.pf_q_a.ptr,
                        .pw = @ptrCast(@alignCast(q_b.data_ptr)),
                        .sc = @ptrCast(@alignCast(comp.scales)),
                        .bi = @ptrCast(@alignCast(comp.biases)),
                        .y = self.pf_q.ptr,
                        .n_tok = n,
                        .n_out = nh * kd,
                        .k = ql,
                        .gs = comp.group_size,
                        .gpr = (ql + comp.group_size - 1) / comp.group_size,
                        .wpr = ((ql + comp.group_size - 1) / comp.group_size) * (comp.group_size * 4 / 32),
                    };
                    pool.parallelFor(nh * kd, 128, @ptrCast(&ctx), BatchedGemmQ4Ctx.work);
                } else {
                    mlx_ops.mlxGemmQ4(self.pf_q_a.ptr, @ptrCast(@alignCast(q_b.data_ptr)), @ptrCast(@alignCast(comp.scales)), @ptrCast(@alignCast(comp.biases)), self.pf_q.ptr, n, nh * kd, ql, comp.group_size);
                }
            } else {
                self.batchedGemm(self.pf_q_a.ptr, q_b, self.pf_q.ptr, n, nh * kd, ql);
            }

            // KV projection was done above (early KV-store for all layers).

            self.be.sync();

            // Per-position: RoPE on Q, per-head Q norm
            const freqs = if (self.compress_ratios[li] != 0) &self.compress_rope_freqs else &self.rope_freqs;
            for (0..n) |t| {
                const pos = position_ids[t];
                var rope_cos: [32]f32 = undefined;
                var rope_sin: [32]f32 = undefined;
                for (0..nd) |i| {
                    const theta = @as(f32, @floatFromInt(pos)) * freqs[i];
                    rope_cos[i] = @cos(theta);
                    rope_sin[i] = @sin(theta);
                }
                // Per-head Q norm + RoPE
                for (0..nh) |h| {
                    const q_head = self.pf_q[t * nh * kd + h * kd ..][0..kd];
                    plainRmsNorm(q_head, self.rms_eps);
                    applyRopeTable(q_head[nope..][0..rd], rope_cos[0..nd], rope_sin[0..nd]);
                }
            }

            // Attention: for each position, parallel dot-product over heads against full KV cache.
            // Uses per-head score buffer slices for thread safety.
            const kv_k_layer = self.kv_k_bytes[li * self.kvLayerBytes() ..];
            const kv_elem_bytes = kv_quant.kvByteOffset(self.kv_type, kd);
            const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));
            const ss = self.score_stride;
            // Head subsampling: compute every head_stride-th head for faster approximate attention.
            // Zero all attn_out first, then compute only selected heads.
            const head_stride: usize = 1; // compute 8 of 64 heads
            @memset(self.pf_attn_out[0 .. n * nh * kd], 0);
            for (0..n) |t| {
                const pos = self.kv_seq_len + t;
                // Compute subsampled heads
                var h: usize = 0;
                while (h < nh) : (h += head_stride) {
                    self.pf_attn_head(t, h, pos, kv_k_layer.ptr, kd, nh, scale, kv_elem_bytes, ss);
                }
                // Inverse RoPE
                const pos_u = self.kv_seq_len + t;
                var rope_cos2: [32]f32 = undefined;
                var rope_sin2: [32]f32 = undefined;
                for (0..nd) |i| {
                    const theta = @as(f32, @floatFromInt(pos_u)) * freqs[i];
                    rope_cos2[i] = @cos(theta);
                    rope_sin2[i] = @sin(theta);
                }
                for (0..nh) |hi| {
                    applyRopeInverseTable(self.pf_attn_out[t * nh * kd + hi * kd + nope ..][0..rd], rope_cos2[0..nd], rope_sin2[0..nd]);
                }
            }

            // wo_a (grouped LoRA) + wo_b (batched per position)
            const wo_a = try self.layerTensorReq(li, "attn_output_a.weight");
            const wo_b = try self.layerTensorReq(li, "attn_output_b.weight");
            const og: usize = self.o_groups;
            const olr: usize = self.o_lora_rank;
            const group_in: usize = nh * kd / og;
            const wo_a_group_stride = if (wo_a.dtype == .mlx_q)
                ds4ExpertStride(wo_a, og)
            else blk: {
                const row_bytes2 = backend_mod.weightBytes(wo_a.dtype, 1, group_in);
                break :blk olr * row_bytes2;
            };
            for (0..n) |t| {
                for (0..og) |g| {
                    const xp = self.pf_attn_out.ptr + t * nh * kd + g * group_in;
                    const yp = self.lora_out.ptr + g * olr;
                    self.computeBackend().gemv(xp, .{ .data = wo_a.data_ptr + g * wo_a_group_stride, .dtype = wo_a.dtype }, yp, olr, group_in);
                }
                // wo_b: use pre-dequanted f32 + SGEMM per-token (single-row SGEMM = GEMV)
                if (self.pf_dequant_ready) {
                    if (comptime @import("builtin").os.tag == .macos and build_options.enable_metal) {
                        const rel3 = li - ft_skip;
                        const accel3 = backend_mod.accelerate;
                        accel3.sgemm(1, e, og * olr, self.lora_out.ptr, self.pf_dequant_wo_b.ptr + rel3 * e * og * olr, self.pf_hidden.ptr + t * e);
                    } else {
                        self.doGemv(self.lora_out.ptr, wo_b, self.pf_hidden.ptr + t * e, e, og * olr);
                    }
                } else {
                    self.doGemv(self.lora_out.ptr, wo_b, self.pf_hidden.ptr + t * e, e, og * olr);
                }
            }
            self.be.sync();

            // FFN: shared expert only (simplified for batched verification).
            // Skip FFN for early active layers to reduce compute, attention-only is
            // approximate but sufficient for suffix argmax verification.
            const ft_ffn_skip: usize = 0; // skip FFN for first 15 active layers
            if (li < ft_skip + ft_ffn_skip) {
                self.be.sync();
                continue;
            }
            const fnw = try self.layerTensorReq(li, "ffn_norm.weight");
            self.computeBackend().rmsNormBatched(self.pf_hidden.ptr, self.normAsF32(fnw, e), self.pf_hidden2.ptr, n, e, self.rms_eps);

            if (self.layerTensor(li, "ffn_gate_shexp.weight")) |gt| {
                const ut = self.layerTensor(li, "ffn_up_shexp.weight") orelse continue;
                const dt = self.layerTensor(li, "ffn_down_shexp.weight") orelse continue;
                // Batched FFN: gate+up GEMM for all tokens at once, then per-token silu+down.
                // gate GEMM: [n, e] × [e, ff] → pf_q_a[0..n*ff] (reuse prefill buffer)
                // up GEMM: [n, e] × [e, ff] → pf_kv_proj[0..n*ff] (reuse prefill buffer)
                // This reads gate/up weights ONCE instead of N times.
                if (n > 1 and n <= 128 and self.pf_q_a.len >= n * ff and self.pf_kv_proj.len >= n * ff and gt.dtype == .mlx_q) {
                    // Weight-stationary batched MLX-Q4 GEMM: read weights ONCE for all N tokens.
                    // Requires companion scale/bias tensors (MLX affine format).
                    const g_comp = model_mod.findMlxCompanion(self.fmt, gt, e);
                    const u_comp = model_mod.findMlxCompanion(self.fmt, ut, e);
                    const d_comp = model_mod.findMlxCompanion(self.fmt, dt, ff);
                    if (g_comp != null and u_comp != null and d_comp != null and g_comp.?.bits == 4 and u_comp.?.bits == 4 and d_comp.?.bits == 4) {
                        // Zero output buffers
                        @memset(self.pf_q_a[0 .. n * ff], 0);
                        @memset(self.pf_kv_proj[0 .. n * ff], 0);
                        // Batched gate: [n, e] × [ff, e]^T → [n, ff]
                        mlx_ops.mlxGemmQ4(
                            self.pf_hidden2.ptr,
                            @ptrCast(@alignCast(gt.data_ptr)),
                            @ptrCast(@alignCast(g_comp.?.scales)),
                            @ptrCast(@alignCast(g_comp.?.biases)),
                            self.pf_q_a.ptr,
                            n,
                            ff,
                            e,
                            g_comp.?.group_size,
                        );
                        // Batched up: [n, e] × [ff, e]^T → [n, ff]
                        mlx_ops.mlxGemmQ4(
                            self.pf_hidden2.ptr,
                            @ptrCast(@alignCast(ut.data_ptr)),
                            @ptrCast(@alignCast(u_comp.?.scales)),
                            @ptrCast(@alignCast(u_comp.?.biases)),
                            self.pf_kv_proj.ptr,
                            n,
                            ff,
                            e,
                            u_comp.?.group_size,
                        );
                        // Per-token: silu(gate) * up, then batched down
                        for (0..n) |t| {
                            self.computeBackend().clampedSiluMul(self.pf_q_a.ptr + t * ff, self.pf_kv_proj.ptr + t * ff, self.pf_q_a.ptr + t * ff, ff);
                        }
                        // Batched down: [n, ff] × [e, ff]^T → [n, e]
                        // pf_q_a has contiguous silu'd outputs for all tokens.
                        // Write to pf_kv_proj (reusable), then add to pf_hidden.
                        self.computeBackend().gemm(self.pf_q_a.ptr, .{ .data = dt.data_ptr, .dtype = dt.dtype }, self.pf_kv_proj.ptr, n, e, ff);
                        for (0..n * e) |i| self.pf_hidden[i] += self.pf_kv_proj[i];
                    } else {
                        // No companion tensors, fall back to be.gemm
                        self.computeBackend().gemm(self.pf_hidden2.ptr, .{ .data = gt.data_ptr, .dtype = gt.dtype }, self.pf_q_a.ptr, n, ff, e);
                        self.computeBackend().gemm(self.pf_hidden2.ptr, .{ .data = ut.data_ptr, .dtype = ut.dtype }, self.pf_kv_proj.ptr, n, ff, e);
                        for (0..n) |t| {
                            self.computeBackend().clampedSiluMul(self.pf_q_a.ptr + t * ff, self.pf_kv_proj.ptr + t * ff, self.pf_q_a.ptr + t * ff, ff);
                            self.computeBackend().gemv(self.pf_q_a.ptr + t * ff, .{ .data = dt.data_ptr, .dtype = dt.dtype }, self.expert_scratch.ptr, e, ff);
                            for (0..e) |i| self.pf_hidden[t * e + i] += self.expert_scratch[i];
                        }
                    }
                } else {
                    // Fallback: per-token (n=1 or buffers too small)
                    for (0..n) |t| {
                        self.computeBackend().gemv(self.pf_hidden2.ptr + t * e, .{ .data = gt.data_ptr, .dtype = gt.dtype }, self.ff_gate_scratch.ptr, ff, e);
                        self.computeBackend().gemv(self.pf_hidden2.ptr + t * e, .{ .data = ut.data_ptr, .dtype = ut.dtype }, self.ff_up_scratch.ptr, ff, e);
                        self.computeBackend().clampedSiluMul(self.ff_gate_scratch.ptr, self.ff_up_scratch.ptr, self.ff_gate_scratch.ptr, ff);
                        self.computeBackend().gemv(self.ff_gate_scratch.ptr, .{ .data = dt.data_ptr, .dtype = dt.dtype }, self.expert_scratch.ptr, e, ff);
                        for (0..e) |i| self.pf_hidden[t * e + i] += self.expert_scratch[i];
                    }
                }
            }
            self.be.sync();
        }

        // Update KV sequence length
        self.kv_seq_len += n;

        // Final norm + lm_head for each position → store logits
        const norm_w = try self.getTensorReq("output_norm.weight");
        // Store per-position logits in a temporary: use pf_hidden2 to hold normed hidden,
        // then compute logits position-by-position into logits_buf (overwritten each time,
        // but treeLogits reads them one at a time anyway).
        self.computeBackend().rmsNormBatched(self.pf_hidden.ptr, self.normAsF32(norm_w, e), self.pf_hidden2.ptr, n, e, self.rms_eps);
        // Save the normed hidden states, treeLogits will compute logits on demand
        // (pf_hidden2[t*e..] holds the normed hidden for position t)
    }

    /// Return argmax token for tree node i (after forwardTree).
    pub fn treeLogits(self: *Ds4Model, node_i: u32) u32 {
        const e = self.n_embd;
        const lm = self.fmt.getTensor("output.weight") orelse return 0;
        self.doGemv(self.pf_hidden2.ptr + @as(usize, node_i) * e, lm, self.logits_buf.ptr, self.vocab_size, e);
        self.be.sync();
        return math_ops.argmax(self.logits_buf);
    }

    /// Return a TensorInfo with heap-overridden data pointer (Metal-safe).
    /// If the tensor has been heap-copied, returns a copy with the heap pointer.
    /// Otherwise returns the original TensorInfo unchanged.
    fn heapTensor(self: *Ds4Model, t: TensorInfo) TensorInfo {
        if (!self.tensor_overrides_inited) return t;
        const heap_ptr = self.heapTensorData(t);
        if (heap_ptr == t.data_ptr) return t;
        var result = t;
        result.data_ptr = heap_ptr;
        return result;
    }

    /// Get tensor data with heap override (Metal-safe).
    /// On first access: copies mmap data to heap. Returns heap pointer.
    /// On subsequent access: returns cached heap pointer.
    /// Expert weights use preadExpert instead (separate pool).
    fn heapTensorData(self: *Ds4Model, t: format_mod.TensorInfo) [*]const u8 {
        if (!self.tensor_overrides_inited) return t.data_ptr;

        // Check override table
        if (self.tensor_overrides.get(t.name)) |heap_ptr| {
            return heap_ptr;
        }

        // First access: copy to heap
        const size = t.dataByteLen();
        if (size == 0 or size > 256 * 1024 * 1024) return t.data_ptr; // skip huge/empty

        const heap = self.allocator.alloc(u8, size) catch return t.data_ptr;
        // Pre-fault: touch every page from CPU before memcpy.
        // Ensures page faults are resolved before Metal accesses the copy.
        const src = @as([*]const u8, t.data_ptr);
        {
            const page_size: usize = 16384;
            var off: usize = 0;
            while (off < size) : (off += page_size) {
                const p: *const volatile u8 = @ptrCast(src + off);
                _ = p.*;
            }
        }
        @memcpy(heap, src[0..size]);

        self.tensor_overrides.put(t.name, heap.ptr) catch {
            self.allocator.free(heap);
            return t.data_ptr;
        };

        return heap.ptr;
    }

    /// Read expert weight data via pread into a pool buffer slot (GGUF only).
    fn preadExpert(self: *Ds4Model, data_ptr: [*]const u8, size: usize, slot: u32) [*]const u8 {
        if (self.gguf_fd < 0 or self.expert_pool.len == 0 or self.gguf_mmap_base == null)
            return data_ptr;
        if (slot >= self.expert_pool_slots) return data_ptr;

        // Compute file offset from mmap pointer
        const mmap_base = self.gguf_mmap_base.?;
        const offset = @intFromPtr(data_ptr) - @intFromPtr(mmap_base);
        const pool_ptr = self.expert_pool.ptr + @as(usize, slot) * self.expert_pool_slot_size;

        // pread: read directly from file into heap buffer
        const actual_size = @min(size, self.expert_pool_slot_size);
        var total_read: usize = 0;
        while (total_read < actual_size) {
            const n = std.posix.system.pread(
                self.gguf_fd,
                pool_ptr + total_read,
                actual_size - total_read,
                @intCast(offset + total_read),
            );
            if (n <= 0) break;
            total_read += @intCast(n);
        }

        if (total_read == actual_size) return pool_ptr;
        return data_ptr; // fallback to mmap if pread failed
    }

    /// Pre-fault expert weights for ALL layers using PARALLEL page touching.
    /// Uses the thread pool to touch expert pages across multiple threads,
    /// forcing synchronous page faults in parallel (~54ms vs ~585ms sequential).
    /// Inspired by SP-MoE (arXiv 2510.10302) and PreScope (arXiv 2509.23638).
    pub fn prefetchAllLayers(self: *Ds4Model) void {
        const ec = self.expert_cache orelse return;
        if (comptime @import("builtin").os.tag != .macos and @import("builtin").os.tag != .linux) return;
        // First: madvise for ALL layers (async hint to OS)
        for (self.hash_layer_count..self.n_layers) |li| {
            inline for (.{ "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight" }) |tensor_name| {
                if (self.layerTensor(li, tensor_name)) |t| {
                    const stride = ds4ExpertStride(t, self.n_experts);
                    ec.prefetchTopResidents(@intCast(li), t.data_ptr, stride, 6);
                }
            }
        }
        // Then: synchronous pre-fault of gate weights across all layers.
        // Touch first byte of each expert to force page fault NOW.
        // Uses thread pool for parallel faulting (14 threads → 14× faster).
        if (self.pool) |pool| {
            const n_layers = self.n_layers - self.hash_layer_count;
            const PrefaultCtx = struct {
                model: *Ds4Model,
                hash_offset: u32,
                fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
                    const ctx: *const @This() = @ptrCast(@alignCast(ctx_ptr));
                    for (start..end) |rel_li| {
                        const li = rel_li + ctx.hash_offset;
                        if (ctx.model.layerTensor(li, "ffn_gate_exps.weight")) |t| {
                            const stride = ds4ExpertStride(t, ctx.model.n_experts);
                            // Touch first byte of each of the top-6 MRU experts
                            if (ctx.model.expert_cache) |ec2| {
                                var top_ids: [6]u32 = undefined;
                                const n_top = ec2.getTopResidents(@intCast(li), &top_ids);
                                for (0..n_top) |i| {
                                    if (!isLocalExpert(top_ids[i], ctx.model.tp_rank, ctx.model.epDegree())) continue;
                                    const ptr: *const volatile u8 = @ptrCast(t.data_ptr + top_ids[i] * stride);
                                    _ = ptr.*;
                                }
                            }
                        }
                    }
                }
            };
            var ctx = PrefaultCtx{ .model = self, .hash_offset = self.hash_layer_count };
            pool.parallelFor(n_layers, 4, @ptrCast(&ctx), PrefaultCtx.work);
        }
    }
};

// ── Math helpers ─────────────────────────────────────────────────

/// Partial-sort top-k selection: writes the k highest-scoring indices into `out`.
/// Uses a simple insertion-sort approach suitable for k ≤ 512 (the LID index_topk).
fn topKIndices(scores: []const f32, out: []u32) void {
    const k = out.len;
    if (k == 0) return;
    // Initialize with first k indices (unsorted threshold)
    const init_k = @min(k, scores.len);
    for (0..init_k) |i| out[i] = @intCast(i);
    // Sort initial k by score descending (insertion sort, k is small)
    for (1..init_k) |i| {
        const val = out[i];
        const vs = scores[val];
        var j: usize = i;
        while (j > 0 and scores[out[j - 1]] < vs) : (j -= 1) {
            out[j] = out[j - 1];
        }
        out[j] = val;
    }
    // Scan remaining elements, insert if larger than current k-th
    if (scores.len > k) {
        var min_score = scores[out[k - 1]];
        for (k..scores.len) |i| {
            if (scores[i] > min_score) {
                // Insert i into sorted out[], displacing the smallest
                const idx: u32 = @intCast(i);
                var j: usize = k - 1;
                while (j > 0 and scores[out[j - 1]] < scores[i]) : (j -= 1) {
                    out[j] = out[j - 1];
                }
                out[j] = idx;
                min_score = scores[out[k - 1]];
            }
        }
    }
}

inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

inline fn sqrtSoftplus(x: f32) f32 {
    return @sqrt(@log(1.0 + @exp(x)));
}

/// In-place RMS normalization, no learned weight. SIMD-optimized.
/// Inlined for tight per-head loops (called 64× per layer).
inline fn plainRmsNorm(x: []f32, eps: f32) void {
    const V8 = @Vector(8, f32);
    var acc: V8 = @splat(@as(f32, 0.0));
    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) {
        const v: V8 = x[i..][0..8].*;
        acc = @mulAdd(V8, v, v, acc);
    }
    var ss: f32 = @reduce(.Add, acc);
    while (i < x.len) : (i += 1) ss += x[i] * x[i];
    const scale = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    const sv: V8 = @splat(scale);
    i = 0;
    while (i + 8 <= x.len) : (i += 8) x[i..][0..8].* = @as(V8, x[i..][0..8].*) * sv;
    while (i < x.len) : (i += 1) x[i] *= scale;
}

/// Apply RoPE using a pre-computed cos/sin table. SIMD-vectorized: processes 4
/// complex rotations per iteration (loads 8 consecutive f32, deinterleaves to
/// even/odd, applies rotation matrix, interleaves back).
inline fn applyRopeTable(x: []f32, cos_t: []const f32, sin_t: []const f32) void {
    const V4 = @Vector(4, f32);
    const n = cos_t.len;
    var i: usize = 0;
    while (i + 4 <= n) : (i += 4) {
        const c: V4 = cos_t[i..][0..4].*;
        const s: V4 = sin_t[i..][0..4].*;
        // Deinterleave: x[2i], x[2i+2], x[2i+4], x[2i+6] → evens
        //               x[2i+1], x[2i+3], x[2i+5], x[2i+7] → odds
        const base = i * 2;
        const x0 = V4{ x[base], x[base + 2], x[base + 4], x[base + 6] };
        const x1 = V4{ x[base + 1], x[base + 3], x[base + 5], x[base + 7] };
        const r0 = @mulAdd(V4, x0, c, -x1 * s);
        const r1 = @mulAdd(V4, x0, s, x1 * c);
        // Interleave back
        x[base] = r0[0];
        x[base + 1] = r1[0];
        x[base + 2] = r0[1];
        x[base + 3] = r1[1];
        x[base + 4] = r0[2];
        x[base + 5] = r1[2];
        x[base + 6] = r0[3];
        x[base + 7] = r1[3];
    }
    while (i < n) : (i += 1) {
        const x0 = x[i * 2];
        const x1 = x[i * 2 + 1];
        x[i * 2] = x0 * cos_t[i] - x1 * sin_t[i];
        x[i * 2 + 1] = x0 * sin_t[i] + x1 * cos_t[i];
    }
}

/// Inverse RoPE using pre-computed table (negate sin). SIMD-vectorized.
inline fn applyRopeInverseTable(x: []f32, cos_t: []const f32, sin_t: []const f32) void {
    const V4 = @Vector(4, f32);
    const n = cos_t.len;
    var i: usize = 0;
    while (i + 4 <= n) : (i += 4) {
        const c: V4 = cos_t[i..][0..4].*;
        const s: V4 = sin_t[i..][0..4].*;
        const base = i * 2;
        const x0 = V4{ x[base], x[base + 2], x[base + 4], x[base + 6] };
        const x1 = V4{ x[base + 1], x[base + 3], x[base + 5], x[base + 7] };
        const r0 = @mulAdd(V4, x0, c, x1 * s);
        const r1 = @mulAdd(V4, x1, c, -x0 * s);
        x[base] = r0[0];
        x[base + 1] = r1[0];
        x[base + 2] = r0[1];
        x[base + 3] = r1[1];
        x[base + 4] = r0[2];
        x[base + 5] = r1[2];
        x[base + 6] = r0[3];
        x[base + 7] = r1[3];
    }
    while (i < n) : (i += 1) {
        const x0 = x[i * 2];
        const x1 = x[i * 2 + 1];
        x[i * 2] = x0 * cos_t[i] + x1 * sin_t[i];
        x[i * 2 + 1] = -x0 * sin_t[i] + x1 * cos_t[i];
    }
}

/// Sinkhorn normalization for [n_hc × n_hc] matrix.
/// m[r*n+c] maps to comb(dst=c, src=r). ggml_soft_max runs over ne[0]=dst for each src.
fn hcSinkhorn(m: []f32) void {
    const n = n_hc;
    // Initial softmax: for each src=r, normalize over dst=c (matches ggml_soft_max over ne[0]=dst)
    for (0..n) |r| { // for each src=r
        var mx = m[r * n + 0];
        for (1..n) |c| if (m[r * n + c] > mx) {
            mx = m[r * n + c];
        };
        var sm: f32 = 0.0;
        for (0..n) |c| {
            m[r * n + c] = @exp(m[r * n + c] - mx);
            sm += m[r * n + c];
        }
        for (0..n) |c| m[r * n + c] /= sm;
    }
    for (m) |*v| v.* += hc_eps;

    var col_s: [n_hc]f32 = undefined;
    var row_s: [n_hc]f32 = undefined;
    for (0..hc_sinkhorn_iters) |_| {
        @memset(&col_s, 0.0);
        for (0..n) |r| for (0..n) |c| {
            col_s[c] += m[r * n + c];
        };
        for (&col_s) |*v| v.* += hc_eps;
        for (0..n) |r| for (0..n) |c| {
            m[r * n + c] /= col_s[c];
        };

        @memset(&row_s, 0.0);
        for (0..n) |r| for (0..n) |c| {
            row_s[r] += m[r * n + c];
        };
        for (&row_s) |*v| v.* += hc_eps;
        for (0..n) |r| for (0..n) |c| {
            m[r * n + c] /= row_s[r];
        };
    }
}

/// CPU f32 GEMV for small matrices (e.g. HC mixing weights from safetensors).
/// y[n_out] += w[n_out × n_in] @ x[n_in]. Simple dot product, no quantization.
fn cpuGemvF32(w_ptr: [*]const u8, x: []const f32, y: []f32, n_in: usize) void {
    const V8 = @Vector(8, f32);
    const w: [*]const f32 = @ptrCast(@alignCast(w_ptr));
    for (y, 0..) |*out, row| {
        var acc: V8 = @splat(0);
        const row_w = w + row * n_in;
        var i: usize = 0;
        while (i + 8 <= n_in) : (i += 8) {
            const xv: V8 = x[i..][0..8].*;
            const wv: V8 = row_w[i..][0..8].*;
            acc = @mulAdd(V8, xv, wv, acc);
        }
        var sum: f32 = @reduce(.Add, acc);
        while (i < n_in) : (i += 1) sum += x[i] * row_w[i];
        out.* = sum;
    }
}

/// CPU bf16 GEMV for small matrices (e.g. HC mixing weights from safetensors).
/// y[n_out] += w_bf16[n_out × n_in] @ x[n_in].
fn cpuGemvBf16(w_ptr: [*]const u8, x: []const f32, y: []f32, n_in: usize) void {
    const w: [*]const u16 = @ptrCast(@alignCast(w_ptr));
    for (y, 0..) |*out, row| {
        var sum: f32 = 0;
        const row_w = w + row * n_in;
        for (0..n_in) |i| {
            sum += x[i] * quant_ops.bf16ToF32(row_w[i]);
        }
        out.* = sum;
    }
}

/// CPU f16 GEMV for small matrices (e.g. HC mixing weights from ds4 GGUF).
/// y[n_out] += w_f16[n_out × n_in] @ x[n_in].
fn cpuGemvF16(w_ptr: [*]const u8, x: []const f32, y: []f32, n_in: usize) void {
    const w: [*]const f16 = @ptrCast(@alignCast(w_ptr));
    for (y, 0..) |*out, row| {
        var sum: f32 = 0;
        const row_w = w + row * n_in;
        for (0..n_in) |i| {
            sum += x[i] * @as(f32, row_w[i]);
        }
        out.* = sum;
    }
}

/// CPU Q8_0 GEMV: y[n_out] = w[n_out rows × n_in cols] @ x[n_in].
/// Avoids Metal GPU dispatch overhead for tiny output dims (like HC pre's 24-output GEMV).
/// Q8_0 block: 2-byte f16 scale + 32 i8 values = 34 bytes.
/// Processes 2 rows at a time to share x[] loads and improve ILP.
fn cpuGemvQ8_0(w_ptr: [*]const u8, x: []const f32, y: []f32, n_in: usize) void {
    const V8 = @Vector(8, f32);
    const block_size: usize = 32;
    const block_bytes: usize = 34;
    const n_out = y.len;
    const blocks_per_row = n_in / block_size;
    const row_stride = blocks_per_row * block_bytes;
    // 2-row interleaved path: share x[] loads between row pairs
    var i: usize = 0;
    while (i + 2 <= n_out) : (i += 2) {
        var acc0: V8 = @splat(0.0);
        var acc1: V8 = @splat(0.0);
        const rp0 = w_ptr + i * row_stride;
        const rp1 = w_ptr + (i + 1) * row_stride;
        for (0..blocks_per_row) |b| {
            const blk0 = rp0 + b * block_bytes;
            const blk1 = rp1 + b * block_bytes;
            const s0: f32 = @floatCast(@as(f16, @bitCast(@as(u16, blk0[0]) | (@as(u16, blk0[1]) << 8))));
            const s1: f32 = @floatCast(@as(f16, @bitCast(@as(u16, blk1[0]) | (@as(u16, blk1[1]) << 8))));
            const sv0: V8 = @splat(s0);
            const sv1: V8 = @splat(s1);
            const xb = x[b * block_size ..][0..32];
            var k: usize = 0;
            while (k + 8 <= 32) : (k += 8) {
                const xv: V8 = xb[k..][0..8].*;
                var qv0: V8 = undefined;
                var qv1: V8 = undefined;
                inline for (0..8) |idx| {
                    qv0[idx] = @floatFromInt(@as(i8, @bitCast((blk0 + 2)[k + idx])));
                    qv1[idx] = @floatFromInt(@as(i8, @bitCast((blk1 + 2)[k + idx])));
                }
                acc0 = @mulAdd(V8, qv0 * sv0, xv, acc0);
                acc1 = @mulAdd(V8, qv1 * sv1, xv, acc1);
            }
        }
        y[i] = @reduce(.Add, acc0);
        y[i + 1] = @reduce(.Add, acc1);
    }
    // Scalar tail for odd n_out
    while (i < n_out) : (i += 1) {
        var acc: V8 = @splat(0.0);
        const row_ptr = w_ptr + i * row_stride;
        for (0..blocks_per_row) |b| {
            const blk = row_ptr + b * block_bytes;
            const scale: f32 = @floatCast(@as(f16, @bitCast(@as(u16, blk[0]) | (@as(u16, blk[1]) << 8))));
            const sv: V8 = @splat(scale);
            const xb = x[b * block_size ..][0..32];
            var k: usize = 0;
            while (k + 8 <= 32) : (k += 8) {
                const xv: V8 = xb[k..][0..8].*;
                var qv: V8 = undefined;
                inline for (0..8) |idx| qv[idx] = @floatFromInt(@as(i8, @bitCast((blk + 2)[k + idx])));
                acc = @mulAdd(V8, qv * sv, xv, acc);
            }
        }
        y[i] = @reduce(.Add, acc);
    }
}

/// Hint the OS to start paging in a memory range from the mmap'd model file.
/// Non-blocking, the kernel reads pages in the background while we continue.
fn prefetchRange(ptr: [*]const u8, len: usize) void {
    const page_size: usize = std.heap.page_size_min;
    const addr = @intFromPtr(ptr);
    const aligned = addr & ~(page_size - 1);
    const total = (addr - aligned) + len;
    std.posix.madvise(@ptrFromInt(aligned), total, std.posix.system.MADV.WILLNEED) catch {};
}

/// Per-expert stride for ds4 expert tensors.
/// 3D tensors: GGUF raw [n_in, n_ff, n_experts] → reversed [n_experts, n_ff, n_in].
///   Stride = dims[1] × dims[2] = n_ff × n_in.
/// 2D tensors (some converters flatten): raw [n_in, n_ff * n_experts] → reversed [n_ff * n_experts, n_in].
///   Stride = dims[0] / n_experts × dims[1], but we don't know n_experts here.
///   Instead, use the total bytes / n_experts where n_experts comes from the model config.
fn ds4ExpertStride(t: TensorInfo, n_experts: usize) usize {
    if (t.dtype == .mlx_q) {
        // MLX-Q: weights packed as u32 words. Shape is [n_experts, rows, words_per_row].
        if (t.n_dims >= 3) return @as(usize, @intCast(t.dims[1])) * @as(usize, @intCast(t.dims[2])) * @sizeOf(u32);
        // 2D: total / n_experts
        return (@as(usize, @intCast(t.dims[0])) * @as(usize, @intCast(t.dims[1])) * @sizeOf(u32)) / n_experts;
    }
    if (t.n_dims >= 3) {
        // 3D: dims are normalized to [outermost, ..., innermost] = [n_experts, n_out, n_in].
        // (GGUF stores innermost-first but gguf.zig reverses at load time.)
        // Per-expert slice = dims[1] × dims[2] elements (n_out × n_in).
        const elems = @as(usize, @intCast(t.dims[1])) * @as(usize, @intCast(t.dims[2]));
        return backend_mod.weightBytes(t.dtype, 1, elems);
    } else {
        // 2D: total elements / n_experts
        const total = @as(usize, @intCast(t.dims[0])) * @as(usize, @intCast(t.dims[1]));
        const per_expert = total / n_experts;
        return backend_mod.weightBytes(t.dtype, 1, per_expert);
    }
}

// ── Tests ─────────────────────────────────────────────────────────

test "topKIndices basic" {
    const scores = [_]f32{ 1.0, 5.0, 3.0, 8.0, 2.0, 7.0, 4.0, 6.0 };
    var out: [3]u32 = undefined;
    topKIndices(&scores, &out);
    // Top 3 scores: 8.0 (idx 3), 7.0 (idx 5), 6.0 (idx 7)
    try std.testing.expectEqual(@as(u32, 3), out[0]);
    try std.testing.expectEqual(@as(u32, 5), out[1]);
    try std.testing.expectEqual(@as(u32, 7), out[2]);
}

test "topKIndices k equals n" {
    const scores = [_]f32{ 2.0, 1.0, 3.0 };
    var out: [3]u32 = undefined;
    topKIndices(&scores, &out);
    try std.testing.expectEqual(@as(u32, 2), out[0]); // 3.0
    try std.testing.expectEqual(@as(u32, 0), out[1]); // 2.0
    try std.testing.expectEqual(@as(u32, 1), out[2]); // 1.0
}

test "topKIndices k=1" {
    const scores = [_]f32{ 10.0, 20.0, 5.0, 15.0 };
    var out: [1]u32 = undefined;
    topKIndices(&scores, &out);
    try std.testing.expectEqual(@as(u32, 1), out[0]); // 20.0
}

test "compSlotsPerLayer" {
    try std.testing.expectEqual(@as(usize, 1), compSlotsPerLayer(0));
    try std.testing.expectEqual(@as(usize, 2), compSlotsPerLayer(4));
    try std.testing.expectEqual(@as(usize, 129), compSlotsPerLayer(512));
    try std.testing.expectEqual(@as(usize, 16385), compSlotsPerLayer(65536));
}

test "defaultCompressRatio Flash 0731 pattern" {
    try std.testing.expectEqual(@as(u32, 0), defaultCompressRatio(0));
    try std.testing.expectEqual(@as(u32, 0), defaultCompressRatio(1));
    try std.testing.expectEqual(@as(u32, 4), defaultCompressRatio(2));
    try std.testing.expectEqual(@as(u32, 128), defaultCompressRatio(3));
    try std.testing.expectEqual(@as(u32, 4), defaultCompressRatio(42));
    try std.testing.expectEqual(@as(u32, 128), defaultCompressRatio(41));
}

test "rawAttnStart matches sliding_window 128" {
    try std.testing.expectEqual(@as(usize, 0), rawAttnStart(0));
    try std.testing.expectEqual(@as(usize, 0), rawAttnStart(127));
    try std.testing.expectEqual(@as(usize, 1), rawAttnStart(128));
    try std.testing.expectEqual(@as(usize, 385), rawAttnStart(512));
}

test "scaleExpertWeights L1 then routed scale" {
    var w = [_]f32{ 1.0, 2.0, 3.0 };
    scaleExpertWeights(w[0..], 1.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), w[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.50), w[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), w[2], 1e-5);
}

test "csaOverlapPool first group uses current high half" {
    const head_dim: usize = 8;
    const stride: usize = 16;
    var curr_kv: [csa_compress_ratio * stride]f32 = [_]f32{0} ** (csa_compress_ratio * stride);
    var curr_score: [csa_compress_ratio * stride]f32 = [_]f32{0} ** (csa_compress_ratio * stride);
    for (0..csa_compress_ratio) |t| {
        const base = t * stride;
        for (0..head_dim) |d| curr_kv[base + d] = 100.0; // low half must not win
        for (0..head_dim) |d| curr_kv[base + head_dim + d] = 1.0 + @as(f32, @floatFromInt(d));
        for (0..stride) |d| curr_score[base + d] = 0.0;
    }
    var out: [head_dim]f32 = undefined;
    csaOverlapPool(&out, &curr_kv, &curr_score, null, null, stride);
    for (0..head_dim) |d| {
        try std.testing.expectApproxEqAbs(1.0 + @as(f32, @floatFromInt(d)), out[d], 1e-5);
    }
}

test "csaOverlapPool mixes previous low half with current high half" {
    const head_dim: usize = 8;
    const stride: usize = 16;
    var curr_kv: [csa_compress_ratio * stride]f32 = [_]f32{0} ** (csa_compress_ratio * stride);
    var curr_score: [csa_compress_ratio * stride]f32 = [_]f32{0} ** (csa_compress_ratio * stride);
    var prev_kv: [csa_compress_ratio * stride]f32 = [_]f32{0} ** (csa_compress_ratio * stride);
    var prev_score: [csa_compress_ratio * stride]f32 = [_]f32{0} ** (csa_compress_ratio * stride);
    for (0..csa_compress_ratio) |t| {
        const base = t * stride;
        for (0..head_dim) |d| {
            prev_kv[base + d] = 2.0;
            curr_kv[base + head_dim + d] = 4.0;
        }
        for (0..stride) |d| {
            prev_score[base + d] = 0.0;
            curr_score[base + d] = 0.0;
        }
    }
    var out: [head_dim]f32 = undefined;
    csaOverlapPool(&out, &curr_kv, &curr_score, &prev_kv, &prev_score, stride);
    // 8 equal scores: mean of 4×2 + 4×4 = 3
    for (out) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 3.0), v, 1e-5);
    }
}

test "sigmoid boundaries" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid(0.0), 1e-6);
    try std.testing.expect(sigmoid(10.0) > 0.999);
    try std.testing.expect(sigmoid(-10.0) < 0.001);
    // Monotonicity
    try std.testing.expect(sigmoid(1.0) > sigmoid(0.0));
    try std.testing.expect(sigmoid(0.0) > sigmoid(-1.0));
}

test "sqrtSoftplus matches sqrt(log(1+exp(x)))" {
    const vals = [_]f32{ -5.0, -1.0, 0.0, 1.0, 5.0, 10.0 };
    for (vals) |x| {
        const expected = @sqrt(@log(1.0 + @exp(x)));
        try std.testing.expectApproxEqAbs(expected, sqrtSoftplus(x), 1e-5);
    }
}

test "plainRmsNorm normalizes to unit RMS" {
    var x = [_]f32{ 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    plainRmsNorm(&x, 1e-6);
    // RMS of result should be ~1.0
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const rms = @sqrt(ss / 8.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), rms, 1e-5);
    // Relative magnitudes preserved: x[0]/x[1] should still be 3/4
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), x[0] / x[1], 1e-5);
}

test "applyRopeTable and inverse are inverses" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const original = x;
    const cos_t = [_]f32{ 0.5403, -0.4161, -0.9900, -0.6536 }; // cos(1), cos(2), cos(3), cos(4)
    const sin_t = [_]f32{ 0.8415, 0.9093, 0.1411, -0.7568 }; // sin(1), sin(2), sin(3), sin(4)
    applyRopeTable(&x, &cos_t, &sin_t);
    // Values should have changed
    try std.testing.expect(x[0] != original[0]);
    // Apply inverse to recover original
    applyRopeInverseTable(&x, &cos_t, &sin_t);
    for (x, original) |got, exp| {
        try std.testing.expectApproxEqAbs(exp, got, 1e-3);
    }
}

test "hcSinkhorn produces doubly stochastic matrix" {
    // Input: 4x4 matrix of log-scale values (n_hc=4)
    var m = [_]f32{
        1.0, 2.0, 0.5, 1.5,
        0.3, 1.8, 2.2, 0.7,
        1.1, 0.4, 1.6, 1.9,
        2.0, 1.0, 0.8, 0.2,
    };
    hcSinkhorn(&m);

    // After Sinkhorn iterations, rows and columns should each sum to ~1.0
    for (0..n_hc) |r| {
        var row_sum: f32 = 0;
        for (0..n_hc) |c| row_sum += m[r * n_hc + c];
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), row_sum, 0.02);
    }
    for (0..n_hc) |c| {
        var col_sum: f32 = 0;
        for (0..n_hc) |r| col_sum += m[r * n_hc + c];
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), col_sum, 0.02);
    }

    // All entries must be non-negative
    for (m) |v| try std.testing.expect(v >= 0);
}

test "cpuGemvQ8_0 basic correctness" {
    // Q8_0 block: 2-byte f16 scale + 32 i8 values = 34 bytes.
    // Create a simple 2-output × 32-input weight matrix (1 block per row).
    const block_size: usize = 32;
    const block_bytes: usize = 34;
    const n_out: usize = 2;
    const n_in: usize = block_size;

    var w: [n_out * block_bytes]u8 = undefined;

    // Row 0: scale=1.0, all quants=1 → dot([1,1,...,1], x) = sum(x)
    const scale_one: u16 = @bitCast(@as(f16, 1.0));
    w[0] = @truncate(scale_one);
    w[1] = @truncate(scale_one >> 8);
    @memset(w[2..block_bytes], 1); // i8 = 1

    // Row 1: scale=2.0, all quants=1 → 2.0 * sum(x)
    const scale_two: u16 = @bitCast(@as(f16, 2.0));
    w[block_bytes] = @truncate(scale_two);
    w[block_bytes + 1] = @truncate(scale_two >> 8);
    @memset(w[block_bytes + 2 .. 2 * block_bytes], 1);

    // Input: x = [1.0, 1.0, ..., 1.0] (32 ones)
    var x: [n_in]f32 = undefined;
    @memset(&x, 1.0);

    var y: [n_out]f32 = undefined;
    cpuGemvQ8_0(&w, &x, &y, n_in);

    // Row 0: 1.0 * (1*1 + 1*1 + ... 32 times) = 32.0
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), y[0], 0.1);
    // Row 1: 2.0 * 32 = 64.0
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), y[1], 0.1);
}

test "cpuGemvQ8_0 odd output count" {
    // Test the scalar tail path (odd n_out)
    const block_bytes: usize = 34;
    const n_in: usize = 32;

    var w: [3 * block_bytes]u8 = undefined;
    const scale_one: u16 = @bitCast(@as(f16, 1.0));
    // All 3 rows: scale=1.0, quants=2
    for (0..3) |r| {
        w[r * block_bytes] = @truncate(scale_one);
        w[r * block_bytes + 1] = @truncate(scale_one >> 8);
        @memset(w[r * block_bytes + 2 .. (r + 1) * block_bytes], 2); // i8 = 2
    }

    var x: [n_in]f32 = undefined;
    @memset(&x, 0.5);

    var y: [3]f32 = undefined;
    cpuGemvQ8_0(&w, &x, &y, n_in);

    // Each row: 1.0 * (2 * 0.5) * 32 = 32.0
    for (y) |v| try std.testing.expectApproxEqAbs(@as(f32, 32.0), v, 0.1);
}

test "ppLayerRange covers all layers without overlap" {
    const nl: u32 = 43; // DeepSeek V4 Flash 0731
    const d: u32 = 2;
    const r0 = ppLayerRange(nl, 0, d);
    const r1 = ppLayerRange(nl, 1, d);
    try std.testing.expectEqual(@as(u32, 0), r0.start);
    try std.testing.expectEqual(@as(u32, 21), r0.end);
    try std.testing.expectEqual(@as(u32, 21), r1.start);
    try std.testing.expectEqual(nl, r1.end);
    try std.testing.expectEqual(nl, r0.end - r0.start + r1.end - r1.start);
    const single = ppLayerRange(nl, 0, 1);
    try std.testing.expectEqual(@as(u32, 0), single.start);
    try std.testing.expectEqual(nl, single.end);
}

test "isLocalExpert partitions 256 experts across two ranks" {
    var r0: u32 = 0;
    var r1: u32 = 0;
    for (0..256) |eid| {
        const a = isLocalExpert(eid, 0, 2);
        const b = isLocalExpert(eid, 1, 2);
        try std.testing.expect(a != b);
        if (a) r0 += 1 else r1 += 1;
        try std.testing.expect(isLocalExpert(eid, 0, 1));
    }
    try std.testing.expectEqual(@as(u32, 128), r0);
    try std.testing.expectEqual(@as(u32, 128), r1);
}

test "undoDuplicatedShared restores shexp plus routed sum" {
    // Rank 0: shexp=1, routed=3. Rank 1: shexp=1, routed=5.
    // All-reduce hidden: 1+3 + 1+5 = 10. Subtract local shexp → 9 = shexp+all_routed.
    var hidden = [_]f32{ 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0 };
    const shexp = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    undoDuplicatedShared(&hidden, &shexp, 1.0);
    for (hidden) |v| try std.testing.expectApproxEqAbs(@as(f32, 9.0), v, 1e-6);
}
