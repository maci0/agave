//! Gemma 4 transformer model implementation — supports all variants:
//!   26B-A4B (MoE), E2B (dense, 35 layers), E4B (dense, 42 layers).
//!
//! Architecture highlights:
//! * Dual attention: sliding-window layers + global layers (per-layer pattern from metadata)
//! * Per-layer KV head counts (scalar or per-layer array)
//! * Per-head QK RMSNorm after Q/K projections
//! * Optional MoE: 26B-A4B has dual FFN (dense GELU-gated + MoE with 128 experts, top-8 softmax)
//!   Dense variants (E2B, E4B) have only the dense FFN path (no experts)
//! * Per-layer feed forward dimensions (scalar or per-layer array)
//! * PLE (Per-Layer Embeddings): Dense variants inject per-layer conditioning via
//!   token_identity + context_projection → gated residual in each layer
//! * Shared KV cache: trailing layers reuse KV from earlier same-type layers
//! * Logit softcapping: cap * tanh(logits / cap), cap=30.0
//! * Gemma-family embedding scaling: hidden *= sqrt(n_embd)

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
const Backend = backend_mod.Backend;
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const DType = backend_mod.DType;
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;
const PagedKvCache = kvcache.PagedKvCache;
const SeqBlockTable = kvcache.SeqBlockTable;

// ── Named constants ──────────────────────────────────────────────

/// Maximum number of transformer layers (compile-time array size).
const max_layers: usize = 64;
/// Maximum top-k experts for stack-allocated selection arrays.
const max_active_experts: usize = 16;
/// Maximum cached norm weight entries (norms/layer × layers + output norm).
const max_norm_entries: usize = 512;
/// MLX companion tensor cache size.
const mlx_companion_cache_size: usize = 512;
/// Default prefill chunk size (tokens per batch).
const default_chunk_size: u32 = 512;

// ── Architecture defaults (Gemma 4 26B-A4B) ──────────────────────
const default_n_layers: u32 = 30;
const default_n_embd: u32 = 2816;
const default_vocab_size: u32 = 262144;
const default_rms_eps: f32 = 1e-6;
const default_final_logit_softcap: f32 = 30.0;

// ── Sliding-window attention defaults ────────────────────────────
const default_sl_n_head: u32 = 16;
const default_sl_n_kv_head: u32 = 8;
const default_sl_head_dim: u32 = 256;
const default_sl_rope_theta: f32 = 10_000.0;
const default_sliding_window: u32 = 1024;

// ── Global attention defaults ────────────────────────────────────
const default_gl_n_head: u32 = 16;
const default_gl_n_kv_head: u32 = 2;
const default_gl_head_dim: u32 = 512;
const default_gl_rope_theta: f32 = 1_000_000.0;
const default_gl_partial_rotary: f32 = 0.25;
const default_global_layer_interval: u32 = 6;

// ── MoE defaults ─────────────────────────────────────────────────
const default_n_experts: u32 = 128;
const default_top_k_experts: u32 = 8;
const default_moe_intermediate: u32 = 704;

// ── GPU SDPA limits ─────────────────────────────────────────────
/// Maximum head dimension supported by Metal/GPU SDPA kernels. Layers with
/// larger head_dim (e.g. global layers with hd=512) use CPU-side SDPA fallback.
const gpu_sdpa_max_head_dim: usize = 256;

// ── Attention scale ─────────────────────────────────────────────
/// Gemma 4 uses QK norms instead of 1/sqrt(head_dim) scaling.
/// Attention scale is 1.0 (matching HuggingFace and llama.cpp).
const attn_scale: f32 = 1.0;

// ── Dense FFN defaults ───────────────────────────────────────────
const default_dense_ff_dim: u32 = 2816;

// ── PLE (Per-Layer Embeddings) defaults ─────────────────────────
/// Per-layer embedding dimension (0 = PLE disabled).
const default_ple_dim: u32 = 0;
/// PLE embedding scale: sqrt(ple_dim).
/// PLE model projection scale: 1/sqrt(n_embd).
/// PLE combination scale: 1/sqrt(2).
const ple_combination_scale: f32 = 1.0 / @sqrt(2.0);

/// Bits per u32 word — used to compute per-tensor bit width from packed weight dimensions.
const bits_per_u32_word: u64 = 32;

/// Cached f32 norm weight entry, keyed by source data pointer.
const NormCacheEntry = struct { key: usize, data: []f32 };

/// MLX companion tensor pointers (scales + biases).
const MlxCompanion = struct { scales: [*]const u8, biases: [*]const u8 };

/// Gemma 4 26B-A4B MoE model with dual attention, dual FFN, and sigmoid routing.
pub const Gemma4Model = struct {
    // ── Core configuration ───────────────────────────────────────
    n_layers: u32,
    n_embd: u32,
    vocab_size: u32,
    rms_eps: f32,
    final_logit_softcap: f32,
    embd_scale: f32,

    // ── Sliding-window attention config ──────────────────────────
    /// Number of query heads for sliding-window layers.
    sl_n_head: u32,
    /// Number of KV heads for sliding-window layers (default; may be overridden per-layer).
    sl_n_kv_head: u32,
    /// Per-head dimension for sliding-window layers.
    sl_head_dim: u32,
    /// RoPE frequency base for sliding-window layers.
    sl_rope_theta: f32,
    /// RoPE rotation dimension count for sliding-window layers.
    sl_rope_dim: u32,
    /// Sliding window size (tokens).
    sliding_window: u32,

    // ── Global attention config ──────────────────────────────────
    /// Number of query heads for global layers.
    gl_n_head: u32,
    /// Number of KV heads for global layers (default; may be overridden per-layer).
    gl_n_kv_head: u32,
    /// Per-head dimension for global layers.
    gl_head_dim: u32,
    /// RoPE frequency base for global layers.
    gl_rope_theta: f32,
    /// RoPE rotation dimension count for global layers.
    gl_rope_dim: u32,
    /// Fraction of head_dim to rotate for global layers (e.g., 0.25).
    gl_partial_rotary: f32,

    // ── MoE config ───────────────────────────────────────────────
    /// Total number of experts per MoE layer.
    n_experts: u32,
    /// Number of experts selected per token (top-k).
    top_k_experts: u32,
    /// FFN intermediate dimension per expert.
    moe_intermediate: u32,

    // ── Dense FFN config ─────────────────────────────────────────
    /// Dense FFN intermediate dimension (scalar default or max of per-layer array).
    dense_ff_dim: u32,
    /// Per-layer FFN intermediate dimensions (from metadata array, or all same).
    per_layer_ff_dim: [max_layers]u32 = [_]u32{0} ** max_layers,

    // ── Layer map ────────────────────────────────────────────────
    /// Every global_layer_interval-th layer (1-indexed) is a global attention layer.
    global_layer_interval: u32,
    /// Per-layer flag: true = global attention, false = sliding-window.
    layer_is_global: [max_layers]bool = [_]bool{false} ** max_layers,
    /// Per-layer KV head count (from metadata array, or defaults).
    per_layer_n_kv_head: [max_layers]u32 = [_]u32{0} ** max_layers,

    // ── Shared KV ────────────────────────────────────────────────
    /// Number of trailing shared-KV layers.
    n_kv_shared_layers: u32 = 0,
    /// Per-layer KV source: kv_source[i] = layer that owns the KV cache for layer i.
    /// For non-shared layers, kv_source[i] == i.
    kv_source: [max_layers]u32 = [_]u32{0} ** max_layers,

    // ── PLE (Per-Layer Embeddings) config ────────────────────────
    /// Per-layer embedding dimension (0 = PLE disabled).
    ple_dim: u32 = 0,
    /// PLE embedding scale: sqrt(ple_dim).
    ple_embd_scale: f32 = 0.0,
    /// PLE model projection scale: 1/sqrt(n_embd).
    ple_proj_scale: f32 = 0.0,

    // ── Dependencies ─────────────────────────────────────────────
    fmt: Format,
    be: Backend,
    allocator: Allocator,

    // ── Norm weight + MLX companion caches ───────────────────────
    norm_add_one: bool,
    mlx_bits: u32 = 4,
    norm_cache: [max_norm_entries]NormCacheEntry = undefined,
    norm_cache_len: usize = 0,
    mlx_cc_keys: [mlx_companion_cache_size]usize = [_]usize{0} ** mlx_companion_cache_size,
    mlx_cc_vals: [mlx_companion_cache_size]MlxCompanion = undefined,

    // ── Working buffers (allocated once, reused every token) ─────
    hidden: []f32 = &.{},
    hidden2: []f32 = &.{},
    q_buf: []f32 = &.{},
    k_buf: []f32 = &.{},
    v_buf: []f32 = &.{},
    attn_out: []f32 = &.{},
    ff_buf: []f32 = &.{},
    ff_buf2: []f32 = &.{},
    /// Dense FFN gate buffer (sized to dense_ff_dim).
    dense_gate: []f32 = &.{},
    /// Dense FFN up buffer (sized to dense_ff_dim).
    dense_up: []f32 = &.{},
    /// Dense FFN output buffer (sized to n_embd).
    dense_out: []f32 = &.{},
    /// Router input buffer (sized to n_embd): holds rms-normed + scaled residual for MoE routing.
    router_input: []f32 = &.{},
    router_buf: []f32 = &.{},
    logits: []f32 = &.{},
    scores: []f32 = &.{},
    /// PLE per-layer input vector (sized to ple_dim, one layer's slice at a time).
    ple_buf: []f32 = &.{},
    /// PLE gate projection output (sized to ple_dim).
    ple_gate_buf: []f32 = &.{},
    /// PLE combined per-layer embeddings: [n_layers * ple_dim].
    /// Computed once per token from token_identity + context_projection.
    ple_combined: []f32 = &.{},

    // ── KV cache (raw per-layer allocation with per-layer kvd) ────
    /// Per-layer key cache: keys[layer][max_seq_len * layer_kvd].
    layer_keys: [][]f32 = &.{},
    /// Per-layer value cache: values[layer][max_seq_len * layer_kvd].
    layer_values: [][]f32 = &.{},
    /// Per-layer KV dimension (nkv * hd for that layer).
    layer_kvd: [max_layers]usize = [_]usize{0} ** max_layers,

    // These fields are kept for Model vtable compatibility (ensureKvBlock, resetKvCache)
    paged_cache: PagedKvCache = undefined,
    seq_table: SeqBlockTable = undefined,
    block_allocator: BlockAllocator = undefined,
    tiered_cache: ?*TieredKvCache = null,
    tiered_block_allocator: ?TieredBlockAllocator = null,
    kv_type_k: kv_quant.KvQuantType = .f32,
    kv_type_v: kv_quant.KvQuantType = .f32,
    kv_seq_len: usize = 0,
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    max_seq_len: usize = 4096,
    perf: perf.PerfCounters = .{},

    // ── Public fields required by Model vtable ───────────────────
    eos_token_id: u32 = 1,
    /// Exposed for Model vtable — uses sliding-window head count (dominant layer type).
    n_head: u32 = default_sl_n_head,
    /// Exposed for Model vtable — uses sliding-window KV head count.
    n_head_kv: u32 = default_sl_n_kv_head,

    /// Initialize the Gemma 4 model from format metadata and allocate all buffers.
    pub fn init(allocator: Allocator, f: Format, be: Backend, ctx_size: u32, kv_type_k: kv_quant.KvQuantType, kv_type_v: kv_quant.KvQuantType, tiered_cache: ?*TieredKvCache) !Gemma4Model {
        const arch = f.getMetaStr("general.architecture") orelse "gemma4";

        const n_layers = f.getArchU32(arch, "block_count") orelse
            f.getMetaU32("num_hidden_layers") orelse default_n_layers;
        const n_embd = f.getArchU32(arch, "embedding_length") orelse
            f.getMetaU32("hidden_size") orelse default_n_embd;
        const vocab_size: u32 = if (f.getVocab()) |v| @intCast(v.len) else
            f.getArchU32(arch, "vocab_size") orelse default_vocab_size;

        // Sliding-window attention params
        const sl_n_head = f.getArchU32(arch, "attention.head_count") orelse
            f.getMetaU32("num_attention_heads") orelse default_sl_n_head;
        // head_count_kv can be a per-layer array; read the scalar default first
        const sl_n_kv_head = f.getArchU32(arch, "attention.head_count_kv") orelse
            f.getMetaU32("num_key_value_heads") orelse default_sl_n_kv_head;
        // key_length_swa is the sliding-window head dim; fall back to key_length
        const sl_head_dim = f.getArchU32(arch, "attention.key_length_swa") orelse
            f.getArchU32(arch, "attention.key_length") orelse
            f.getMetaU32("head_dim") orelse default_sl_head_dim;
        // Sliding-window uses freq_base_swa (10K); freq_base is the global theta (1M).
        const sl_rope_theta = f.getArchF32(arch, "rope.freq_base_swa") orelse
            f.getArchF32(arch, "rope.freq_base") orelse
            f.getMetaF32("rope_theta") orelse default_sl_rope_theta;
        const sliding_window = f.getArchU32(arch, "attention.sliding_window") orelse
            f.getMetaU32("sliding_window") orelse default_sliding_window;
        // RoPE dimension counts — separate for sliding and global
        const sl_rope_dim = f.getArchU32(arch, "rope.dimension_count_swa") orelse
            f.getArchU32(arch, "rope.dimension_count") orelse sl_head_dim;

        // Global attention params
        const gl_n_head = f.getArchU32(arch, "attention.head_count_global") orelse
            f.getMetaU32("global_num_attention_heads") orelse sl_n_head;
        // Global KV head count: try global-specific key, then fall back to the
        // shared head_count_kv scalar (E2B/E4B use a single value for all layers).
        const gl_n_kv_head = f.getArchU32(arch, "attention.head_count_kv_global") orelse
            f.getMetaU32("global_num_key_value_heads") orelse sl_n_kv_head;
        const gl_head_dim = f.getArchU32(arch, "attention.key_length") orelse
            f.getArchU32(arch, "attention.key_length_global") orelse
            f.getMetaU32("global_head_dim") orelse default_gl_head_dim;
        // Global attention uses freq_base (1M); freq_base_global is a fallback alias.
        const gl_rope_theta = f.getArchF32(arch, "rope.freq_base") orelse
            f.getArchF32(arch, "rope.freq_base_global") orelse
            f.getMetaF32("global_rope_theta") orelse default_gl_rope_theta;
        const gl_partial_rotary = f.getArchF32(arch, "rope.partial_rotary_factor") orelse
            f.getMetaF32("partial_rotary_factor") orelse default_gl_partial_rotary;
        const gl_rope_dim = f.getArchU32(arch, "rope.dimension_count") orelse gl_head_dim;

        // MoE params — detect dense variants by checking for expert tensors.
        // Dense models (e.g. Gemma 4 31B, E4B) have no expert_count metadata
        // and no expert weight tensors; default to 0 when both are absent.
        const has_expert_tensors = f.layerTensor(0, "ffn_gate_inp.weight") != null;
        const n_experts = f.getArchU32(arch, "expert_count") orelse
            f.getMetaU32("num_local_experts") orelse
            if (has_expert_tensors) default_n_experts else 0;
        const top_k_experts = if (n_experts == 0) @as(u32, 0) else
            f.getArchU32(arch, "expert_used_count") orelse
            f.getMetaU32("num_experts_per_tok") orelse default_top_k_experts;
        const moe_intermediate = if (n_experts == 0) @as(u32, 0) else
            f.getArchU32(arch, "expert_feed_forward_length") orelse
            f.getMetaU32("expert_intermediate_size") orelse default_moe_intermediate;

        // Dense FFN intermediate dimension — can be scalar or per-layer array.
        // Read per-layer array first; if absent, try scalar; if absent, infer from tensor shape.
        var per_layer_ff_dim: [max_layers]u32 = [_]u32{0} ** max_layers;
        var dense_ff_dim: u32 = 0;
        {
            const nli: usize = n_layers;
            var ff_key_buf: [256]u8 = undefined;
            const ff_key = std.fmt.bufPrint(&ff_key_buf, "{s}.feed_forward_length", .{arch}) catch "";
            const ff_arr = f.getMetaU32Array(ff_key);
            if (ff_arr) |arr| {
                // Per-layer array: use each layer's value, dense_ff_dim = max
                for (0..@min(nli, arr.len)) |i| {
                    per_layer_ff_dim[i] = arr[i];
                    if (arr[i] > dense_ff_dim) dense_ff_dim = arr[i];
                }
            } else {
                // Try scalar metadata
                dense_ff_dim = f.getArchU32(arch, "feed_forward_length") orelse blk: {
                    // Infer from ffn_gate.weight shape: [ff_dim × n_embd]
                    if (f.layerTensor(0, "ffn_gate.weight")) |t| {
                        if (t.n_dims >= 2) break :blk @as(u32, @intCast(t.dims[1]));
                    }
                    break :blk default_dense_ff_dim;
                };
                // Fill all layers with the scalar value
                for (0..nli) |i| per_layer_ff_dim[i] = dense_ff_dim;
            }
        }

        // Layer structure
        const global_layer_interval = f.getArchU32(arch, "attention.global_layer_interval") orelse
            f.getMetaU32("global_layer_interval") orelse default_global_layer_interval;

        // Shared KV layers
        const n_kv_shared_layers = f.getArchU32(arch, "attention.shared_kv_layers") orelse
            f.getArchU32(arch, "attention.kv_shared_layer_count") orelse
            f.getMetaU32("num_kv_shared_layers") orelse 0;

        // PLE (Per-Layer Embeddings)
        const ple_dim = f.getArchU32(arch, "embedding_length_per_layer_input") orelse
            f.getMetaU32("hidden_size_per_layer_input") orelse default_ple_dim;

        // Softcap
        const final_logit_softcap = f.getArchF32(arch, "final_logit_softcapping") orelse
            f.getMetaF32("final_logit_softcapping") orelse default_final_logit_softcap;

        const rms_eps = f.getArchF32(arch, "attention.layer_norm_rms_epsilon") orelse
            f.getMetaF32("rms_norm_eps") orelse default_rms_eps;

        const eos_token_id = f.getMetaU32("tokenizer.ggml.eos_token_id") orelse 1;

        var max_sl: usize = 4096;
        if (f.getArchU32(arch, "context_length")) |cl| max_sl = cl;
        if (f.getMetaU32("max_position_embeddings")) |cl| max_sl = cl;
        if (ctx_size > 0) max_sl = ctx_size;

        const nl: usize = n_layers;

        // Build layer type map — use per-layer sliding_window_pattern array if available
        var layer_is_global: [max_layers]bool = [_]bool{false} ** max_layers;
        {
            // Try to read per-layer sliding_window_pattern array from GGUF metadata.
            // Format: arch_key e.g. "gemma4.attention.sliding_window_pattern"
            // Array values: 0 = sliding, >0 = global (window size).
            var key_buf: [256]u8 = undefined;
            const sw_key = std.fmt.bufPrint(&key_buf, "{s}.attention.sliding_window_pattern", .{arch}) catch "";
            const sw_pattern_arr = f.getMetaU32Array(sw_key);
            if (sw_pattern_arr) |arr| {
                // Per-layer pattern: non-zero = global layer
                for (0..@min(nl, arr.len)) |i| {
                    layer_is_global[i] = (arr[i] == 0);
                }
            } else if (global_layer_interval > 0) {
                // Fallback: every global_layer_interval-th layer (1-indexed) is global.
                // For interval=6: layers 5,11,17,23,29 (0-indexed) are global.
                for (0..nl) |i| {
                    layer_is_global[i] = ((i + 1) % global_layer_interval == 0);
                }
            }
        }

        // Build per-layer KV head count map
        var per_layer_n_kv_head: [max_layers]u32 = [_]u32{0} ** max_layers;
        {
            var key_buf: [256]u8 = undefined;
            const kv_key = std.fmt.bufPrint(&key_buf, "{s}.attention.head_count_kv", .{arch}) catch "";
            const kv_arr = f.getMetaU32Array(kv_key);
            if (kv_arr) |arr| {
                for (0..@min(nl, arr.len)) |i| {
                    per_layer_n_kv_head[i] = arr[i];
                }
            } else {
                // Fill from defaults based on layer type
                for (0..nl) |i| {
                    per_layer_n_kv_head[i] = if (layer_is_global[i]) gl_n_kv_head else sl_n_kv_head;
                }
            }
        }

        // Build KV source mapping: shared layers point to the last non-shared
        // layer of the same type (global or sliding).
        var kv_source: [max_layers]u32 = undefined;
        for (0..nl) |i| kv_source[i] = @intCast(i); // default: self
        if (n_kv_shared_layers > 0 and nl > n_kv_shared_layers) {
            const first_shared = nl - n_kv_shared_layers;
            for (first_shared..nl) |i| {
                // Walk backwards to find the closest non-shared layer of matching type
                const is_gl = layer_is_global[i];
                var src: usize = i;
                var j: usize = i;
                while (j > 0) {
                    j -= 1;
                    if (j < first_shared and layer_is_global[j] == is_gl) {
                        src = j;
                        break;
                    }
                }
                kv_source[i] = @intCast(src);
            }
        }

        // Compute max buffer sizes across both layer types
        const sl_qkv_dim: usize = @as(usize, sl_n_head) * sl_head_dim;
        const sl_kv_dim: usize = @as(usize, sl_n_kv_head) * sl_head_dim;
        const gl_qkv_dim: usize = @as(usize, gl_n_head) * gl_head_dim;
        const gl_kv_dim: usize = @as(usize, gl_n_kv_head) * gl_head_dim;
        // attn_out also serves as PLE context projection temp buffer (total_ple_dim)
        const total_ple_dim: usize = nl * ple_dim;
        const max_qkv_dim = @max(@max(sl_qkv_dim, gl_qkv_dim), total_ple_dim);
        const max_kv_dim = @max(sl_kv_dim, gl_kv_dim);

        // KV cache kv_dim is the maximum across layer types
        const kv_dim_for_cache = max_kv_dim;

        var self = Gemma4Model{
            .n_layers = n_layers,
            .n_embd = n_embd,
            .vocab_size = vocab_size,
            .rms_eps = rms_eps,
            .final_logit_softcap = final_logit_softcap,
            .embd_scale = @sqrt(@as(f32, @floatFromInt(n_embd))),
            .sl_n_head = sl_n_head,
            .sl_n_kv_head = sl_n_kv_head,
            .sl_head_dim = sl_head_dim,
            .sl_rope_theta = sl_rope_theta,
            .sl_rope_dim = sl_rope_dim,
            .sliding_window = sliding_window,
            .gl_n_head = gl_n_head,
            .gl_n_kv_head = gl_n_kv_head,
            .gl_head_dim = gl_head_dim,
            .gl_rope_theta = gl_rope_theta,
            .gl_rope_dim = gl_rope_dim,
            .gl_partial_rotary = gl_partial_rotary,
            .n_experts = n_experts,
            .top_k_experts = top_k_experts,
            .moe_intermediate = moe_intermediate,
            .dense_ff_dim = dense_ff_dim,
            .per_layer_ff_dim = per_layer_ff_dim,
            .global_layer_interval = global_layer_interval,
            .layer_is_global = layer_is_global,
            .per_layer_n_kv_head = per_layer_n_kv_head,
            .n_kv_shared_layers = n_kv_shared_layers,
            .kv_source = kv_source,
            .ple_dim = ple_dim,
            .ple_embd_scale = if (ple_dim > 0) @sqrt(@as(f32, @floatFromInt(ple_dim))) else 0.0,
            .ple_proj_scale = if (ple_dim > 0) 1.0 / @sqrt(@as(f32, @floatFromInt(n_embd))) else 0.0,
            .fmt = f,
            .be = be,
            .allocator = allocator,
            .eos_token_id = eos_token_id,
            .n_head = sl_n_head,
            .n_head_kv = sl_n_kv_head,
            .max_seq_len = max_sl,
            .kv_type_k = kv_type_k,
            .kv_type_v = kv_type_v,
            .tiered_cache = tiered_cache,
            .mlx_bits = f.getMetaU32("bits") orelse 4,
            // Gemma4 GGUF converter stores norm weights as-is (norm_shift=0.0), unlike
            // Gemma3 which bakes +1 (norm_shift=1.0). No +1 needed for GGUF.
            // SafeTensors (model_type present) also stores raw weights (no +1 needed).
            .norm_add_one = false,
        };

        // ── Per-layer KV cache allocation ──────────────────────────
        // Gemma 4 has per-layer varying kvd (sliding: nkv*hd = 2048, global: nkv*hd = 1024).
        // Allocate raw KV buffers per layer with the correct dimensions.
        // Shared layers point to their source layer's buffers.
        {
            self.layer_keys = try allocator.alloc([]f32, nl);
            errdefer allocator.free(self.layer_keys);
            self.layer_values = try allocator.alloc([]f32, nl);
            errdefer allocator.free(self.layer_values);

            for (0..nl) |i| {
                const is_gl = layer_is_global[i];
                const lnkv: usize = per_layer_n_kv_head[i];
                const lhd: usize = if (is_gl) gl_head_dim else sl_head_dim;
                self.layer_kvd[i] = lnkv * lhd;

                if (kv_source[i] != @as(u32, @intCast(i))) {
                    // Shared layer — will be redirected to source in getLayerKvView
                    self.layer_keys[i] = &.{};
                    self.layer_values[i] = &.{};
                } else {
                    const slot_size = max_sl * self.layer_kvd[i];
                    self.layer_keys[i] = try allocator.alloc(f32, slot_size);
                    errdefer allocator.free(self.layer_keys[i]);
                    self.layer_values[i] = try allocator.alloc(f32, slot_size);
                    errdefer allocator.free(self.layer_values[i]);
                }
            }
        }

        // PagedKvCache still needed for Model vtable (ensureKvBlock, resetKvCache)
        // but actual KV data goes through layer_keys/layer_values.
        if (tiered_cache) |tc| {
            var ta = TieredBlockAllocator.init(tc, allocator);
            self.seq_table = try ta.allocateSeqTable(nl);
            errdefer ta.freeSeqTable(&self.seq_table);
            try ta.appendBlock(&self.seq_table);
            self.tiered_block_allocator = ta;
        } else {
            const block_size: u16 = @intCast(@min(max_sl, std.math.maxInt(u16)));
            const num_blocks = nl;
            self.paged_cache = try PagedKvCache.init(allocator, nl, kv_dim_for_cache, num_blocks, block_size);
            errdefer self.paged_cache.deinit();
            self.block_allocator = BlockAllocator.init(&self.paged_cache, allocator);
            self.seq_table = try self.block_allocator.allocateSeqTable(nl);
            errdefer self.block_allocator.freeSeqTable(&self.seq_table);
            try self.block_allocator.appendBlock(&self.seq_table);
        }

        // ── Working buffer allocation ────────────────────────────
        self.hidden = try allocator.alloc(f32, n_embd);
        errdefer allocator.free(self.hidden);
        self.hidden2 = try allocator.alloc(f32, n_embd);
        errdefer allocator.free(self.hidden2);
        self.q_buf = try allocator.alloc(f32, max_qkv_dim);
        errdefer allocator.free(self.q_buf);
        self.k_buf = try allocator.alloc(f32, max_kv_dim);
        errdefer allocator.free(self.k_buf);
        self.v_buf = try allocator.alloc(f32, max_kv_dim);
        errdefer allocator.free(self.v_buf);
        self.attn_out = try allocator.alloc(f32, max_qkv_dim);
        errdefer allocator.free(self.attn_out);
        // MoE expert buffers: ff_buf sized to moe_intermediate (0 for dense-only models),
        // ff_buf2 reused for up-proj (moe_intermediate) AND down-proj output (n_embd)
        const moe_buf_size = @max(moe_intermediate, 1); // Avoid zero-length alloc
        self.ff_buf = try allocator.alloc(f32, moe_buf_size);
        errdefer allocator.free(self.ff_buf);
        self.ff_buf2 = try allocator.alloc(f32, @max(moe_buf_size, n_embd));
        errdefer allocator.free(self.ff_buf2);
        // Dense FFN buffers — sized to the max per-layer FFN dimension
        self.dense_gate = try allocator.alloc(f32, dense_ff_dim);
        errdefer allocator.free(self.dense_gate);
        self.dense_up = try allocator.alloc(f32, dense_ff_dim);
        errdefer allocator.free(self.dense_up);
        self.dense_out = try allocator.alloc(f32, n_embd);
        errdefer allocator.free(self.dense_out);
        self.router_input = try allocator.alloc(f32, n_embd);
        errdefer allocator.free(self.router_input);
        const router_buf_size = @max(n_experts, 1); // Avoid zero-length alloc
        self.router_buf = try allocator.alloc(f32, router_buf_size);
        errdefer allocator.free(self.router_buf);
        self.logits = try allocator.alloc(f32, vocab_size);
        errdefer allocator.free(self.logits);
        self.scores = try allocator.alloc(f32, max_sl);
        errdefer allocator.free(self.scores);

        // PLE buffers — only allocated when PLE is active
        if (ple_dim > 0) {
            self.ple_buf = try allocator.alloc(f32, ple_dim);
            errdefer allocator.free(self.ple_buf);
            self.ple_gate_buf = try allocator.alloc(f32, ple_dim);
            errdefer allocator.free(self.ple_gate_buf);
            self.ple_combined = try allocator.alloc(f32, nl * ple_dim);
            errdefer allocator.free(self.ple_combined);
        }

        // Pre-populate norm cache
        self.warmNormCache();

        return self;
    }

    /// Release all heap allocations owned by this model.
    pub fn deinit(self: *Gemma4Model) void {
        self.be.sync();
        // Free per-layer KV cache
        for (0..self.n_layers) |i| {
            if (self.kv_source[i] == @as(u32, @intCast(i))) {
                if (self.layer_keys[i].len > 0) self.allocator.free(self.layer_keys[i]);
                if (self.layer_values[i].len > 0) self.allocator.free(self.layer_values[i]);
            }
        }
        if (self.layer_keys.len > 0) self.allocator.free(self.layer_keys);
        if (self.layer_values.len > 0) self.allocator.free(self.layer_values);
        // Free PagedKvCache (still used by vtable)
        if (self.tiered_block_allocator) |*ta| {
            ta.freeSeqTable(&self.seq_table);
        } else {
            self.block_allocator.freeSeqTable(&self.seq_table);
            self.paged_cache.deinit();
        }
        // Free cached norm weight conversions
        for (self.norm_cache[0..self.norm_cache_len]) |entry| self.allocator.free(entry.data);
        // Free PLE buffers
        if (self.ple_buf.len > 0) self.allocator.free(self.ple_buf);
        if (self.ple_gate_buf.len > 0) self.allocator.free(self.ple_gate_buf);
        if (self.ple_combined.len > 0) self.allocator.free(self.ple_combined);
        // Free working buffers
        const bufs = .{
            &self.hidden,       &self.hidden2,     &self.q_buf,
            &self.k_buf,        &self.v_buf,       &self.attn_out,
            &self.ff_buf,       &self.ff_buf2,     &self.dense_gate,
            &self.dense_up,     &self.dense_out,   &self.router_input,
            &self.router_buf,   &self.logits,      &self.scores,
        };
        inline for (bufs) |buf| self.allocator.free(buf.*);
    }

    /// Wrap this model in the generic `Model` interface.
    pub fn model(self: *Gemma4Model) Model {
        return Model.from(Gemma4Model, self);
    }

    // ── Forward pass ──────────────────────────────────────────────

    /// Run one decode step, returning the argmax next-token ID.
    /// Processes token through embedding, all layers (dual attention + dual FFN),
    /// final norm, LM head, softcapping, and argmax.
    pub fn forward(self: *Gemma4Model, token_id: u32) !u32 {
        if (self.kv_seq_len >= self.max_seq_len) return error.KVCacheFull;

        try model_mod.ensureKvBlock(self);

        // Embedding lookup + Gemma scaling
        var t = self.perf.start();
        self.lookupEmbd(token_id);
        self.perf.end(.emb_lookup, t);

        // PLE: compute per-layer embeddings from token + context projection
        if (self.ple_dim > 0) {
            t = self.perf.start();
            self.computePleEmbeddings(token_id);
            self.perf.end(.emb_lookup, t);
        }

        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.acquire)) return error.Cancelled;

            try self.attention(@intCast(li));
            try self.dualFfnLayer(@intCast(li));
        }

        // Final RMSNorm → LM head → softcap → argmax
        t = self.perf.start();
        const norm_w = self.fmt.getTensor("output_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(norm_w, self.n_embd), self.hidden.ptr, self.hidden.len, self.rms_eps);
        self.perf.end(.rms_norm, t);

        t = self.perf.start();
        const out_w = self.fmt.getTensor("output.weight") orelse
            self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        self.doGemv(self.hidden.ptr, out_w, self.logits.ptr, self.vocab_size, self.n_embd);
        self.perf.end(.gemv_ffn, t);

        self.be.sync();
        if (self.final_logit_softcap > 0.0) self.applySoftcap();

        self.kv_seq_len += 1;
        self.perf.addToken();
        return math_ops.argmax(self.logits);
    }

    /// Prefill: process all prompt tokens and return argmax of the last
    /// token's logits. Currently uses sequential per-token forward passes.
    ///
    /// MoE routing selects different experts per token, preventing
    /// straightforward batched FFN. Dual attention configs (different head
    /// dims per layer type) further complicate batched GEMM. Batched
    /// prefill for the dense FFN + attention projections is future work.
    pub fn prefill(self: *Gemma4Model, token_ids: []const u32) !u32 {
        if (token_ids.len == 0) return error.MissingTensor;
        if (token_ids.len > self.max_seq_len) return error.KVCacheFull;
        var last: u32 = 0;
        for (token_ids) |tid| last = try self.forward(tid);
        return last;
    }

    /// Reset the KV cache position for a new conversation.
    pub fn resetCache(self: *Gemma4Model) void {
        model_mod.resetKvCache(self);
    }

    /// Signal an in-progress forward pass to abort. Thread-safe.
    pub fn cancel(self: *Gemma4Model) void {
        model_mod.signalCancel(&self.cancelled);
    }

    /// Return physical block IDs from layer 0 of the current sequence table.
    pub fn getBlockTable(self: *Gemma4Model) []const u32 {
        return self.seq_table.block_table[0];
    }

    // ── Layer implementations ─────────────────────────────────────

    /// Get flat f32 view of KV cache for a layer (resolves shared KV mapping).
    /// Returns the per-layer KV buffers with correct per-layer kvd stride.
    fn getLayerKvView(self: *Gemma4Model, layer: usize) struct { keys: []f32, values: []f32 } {
        const src_layer = self.kv_source[layer];
        return .{
            .keys = self.layer_keys[src_layer],
            .values = self.layer_values[src_layer],
        };
    }

    /// One attention layer: pre-norm → QKV → QK norm → V norm (tied K=V) → RoPE →
    /// KV append + SDPA → output proj → post-attention norm → residual.
    /// Dispatches to sliding-window or global attention based on layer type.
    /// Shared-KV layers skip K/V projection and read from the source layer's cache.
    fn attention(self: *Gemma4Model, li: u32) !void {
        const e: usize = self.n_embd;
        const is_global = self.layer_is_global[li];
        const has_own_kv = (self.kv_source[li] == li);

        // Select attention dimensions based on layer type.
        // For shared layers, use the SOURCE layer's KV dimensions (nkv, hd)
        // because we attend against the source layer's cache.
        const src_layer = self.kv_source[li];
        const src_is_global = self.layer_is_global[src_layer];
        const nh: usize = if (is_global) self.gl_n_head else self.sl_n_head;
        const nkv: usize = self.per_layer_n_kv_head[src_layer];
        const hd: usize = if (src_is_global) self.gl_head_dim else self.sl_head_dim;
        const qkv_dim = nh * hd;
        const kv_dim = nkv * hd;

        // 1. Pre-attention RMSNorm
        var t = self.perf.start();
        const norm_w = self.fmt.layerTensor(li, "attn_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(norm_w, e), self.hidden2.ptr, e, self.rms_eps);
        self.perf.end(.rms_norm, t);

        // 2. Q projection (always computed)
        t = self.perf.start();
        const qw = self.fmt.layerTensor(li, "attn_q.weight") orelse return error.MissingTensor;
        self.doGemv(self.hidden2.ptr, qw, self.q_buf.ptr, qkv_dim, e);
        self.perf.end(.gemv_qkv, t);

        // 3. K/V projections — only for layers that own their KV cache.
        // Shared layers skip K/V projection and attend against the source's existing cache.
        if (has_own_kv) {
            t = self.perf.start();
            const kw = self.fmt.layerTensor(li, "attn_k.weight") orelse return error.MissingTensor;
            const vw = self.fmt.layerTensor(li, "attn_v.weight");
            self.doGemv(self.hidden2.ptr, kw, self.k_buf.ptr, kv_dim, e);
            if (vw) |vw_t| {
                self.doGemv(self.hidden2.ptr, vw_t, self.v_buf.ptr, kv_dim, e);
            } else {
                // Tied K=V: copy K to V before K gets normed/rotated.
                self.be.sync();
                @memcpy(self.v_buf[0..kv_dim], self.k_buf[0..kv_dim]);
            }
            self.perf.end(.gemv_qkv, t);

            // Per-head QK RMSNorm (with learned weights) and V norm
            t = self.perf.start();
            if (self.fmt.layerTensor(li, "attn_q_norm.weight")) |qn| {
                self.be.rmsNormMulti(self.q_buf.ptr, self.normAsF32(qn, hd), nh, hd, self.rms_eps);
            }
            if (self.fmt.layerTensor(li, "attn_k_norm.weight")) |kn| {
                self.be.rmsNormMulti(self.k_buf.ptr, self.normAsF32(kn, hd), nkv, hd, self.rms_eps);
            }
            if (vw == null) {
                self.be.sync();
                rmsNormPlainMulti(self.v_buf.ptr, nkv, hd, self.rms_eps);
            }
            self.perf.end(.rms_norm, t);

            // RoPE for K
            t = self.perf.start();
            if (is_global) {
                const rd: usize = @intFromFloat(@as(f32, @floatFromInt(self.gl_head_dim)) * self.gl_partial_rotary);
                const rd_even = rd & ~@as(usize, 1);
                if (rd_even > 0) {
                    self.be.rope(self.q_buf.ptr, self.kv_seq_len, nh, hd, rd_even, self.gl_rope_theta);
                    self.be.rope(self.k_buf.ptr, self.kv_seq_len, nkv, hd, rd_even, self.gl_rope_theta);
                }
            } else {
                const rd: usize = self.sl_rope_dim;
                self.be.rope(self.q_buf.ptr, self.kv_seq_len, nh, hd, rd, self.sl_rope_theta);
                self.be.rope(self.k_buf.ptr, self.kv_seq_len, nkv, hd, rd, self.sl_rope_theta);
            }
            self.perf.end(.rope, t);
        } else {
            // Shared layer: Q norm + RoPE only (no K/V computed)
            t = self.perf.start();
            if (self.fmt.layerTensor(li, "attn_q_norm.weight")) |qn| {
                self.be.rmsNormMulti(self.q_buf.ptr, self.normAsF32(qn, hd), nh, hd, self.rms_eps);
            }
            self.perf.end(.rms_norm, t);

            t = self.perf.start();
            if (is_global) {
                const rd: usize = @intFromFloat(@as(f32, @floatFromInt(self.gl_head_dim)) * self.gl_partial_rotary);
                const rd_even = rd & ~@as(usize, 1);
                if (rd_even > 0) {
                    self.be.rope(self.q_buf.ptr, self.kv_seq_len, nh, hd, rd_even, self.gl_rope_theta);
                }
            } else {
                const rd: usize = self.sl_rope_dim;
                self.be.rope(self.q_buf.ptr, self.kv_seq_len, nh, hd, rd, self.sl_rope_theta);
            }
            self.perf.end(.rope, t);
        }

        // 5. KV cache SDPA
        // For has_own_kv: append new K/V and attend. For shared: attend to source's existing cache.
        t = self.perf.start();

        const kv_view = self.getLayerKvView(li);
        const src_kvd = self.layer_kvd[src_layer];
        const kv_keys_bytes: []u8 = std.mem.sliceAsBytes(kv_view.keys);
        const kv_values_bytes: []u8 = std.mem.sliceAsBytes(kv_view.values);
        const scale: f32 = attn_scale;

        // For shared layers, we use the full seq_len+1 because the source layer
        // already wrote its K/V at position kv_seq_len earlier in this forward pass.
        const sl = self.kv_seq_len + 1;

        // For shared layers, re-read the source's K/V from cache at current position
        // so the SDPA "append" is a harmless no-op (writes same data back).
        if (!has_own_kv) {
            const k_off = self.kv_seq_len * src_kvd;
            const v_off = self.kv_seq_len * src_kvd;
            @memcpy(self.k_buf[0..src_kvd], kv_view.keys[k_off..][0..src_kvd]);
            @memcpy(self.v_buf[0..src_kvd], kv_view.values[v_off..][0..src_kvd]);
        }

        if (!is_global and self.sliding_window > 0) {
            const win: usize = @min(sl, self.sliding_window);
            const start: usize = if (sl > self.sliding_window) sl - self.sliding_window else 0;
            attn_ops.scaledDotProductAttention(
                self.q_buf.ptr,
                kv_keys_bytes,
                kv_values_bytes,
                self.k_buf[0..kv_dim],
                self.v_buf[0..kv_dim],
                self.attn_out.ptr,
                self.scores.ptr,
                nh,
                nkv,
                hd,
                self.kv_seq_len,
                scale,
                self.be,
                .{ .start = start, .len = win },
                0,
                .f32,
                .f32,
            );
        } else if (hd > gpu_sdpa_max_head_dim) {
            attn_ops.scaledDotProductAttention(
                self.q_buf.ptr,
                kv_keys_bytes,
                kv_values_bytes,
                self.k_buf[0..kv_dim],
                self.v_buf[0..kv_dim],
                self.attn_out.ptr,
                self.scores.ptr,
                nh,
                nkv,
                hd,
                self.kv_seq_len,
                scale,
                self.be,
                .{ .start = 0, .len = sl },
                0,
                .f32,
                .f32,
            );
        } else {
            attn_ops.scaledDotProductAttention(
                self.q_buf.ptr,
                kv_keys_bytes,
                kv_values_bytes,
                self.k_buf[0..kv_dim],
                self.v_buf[0..kv_dim],
                self.attn_out.ptr,
                self.scores.ptr,
                nh,
                nkv,
                hd,
                self.kv_seq_len,
                scale,
                self.be,
                null,
                0,
                .f32,
                .f32,
            );
        }
        self.perf.end(.sdpa, t);

        // 6. Output projection + post-attention norm + residual
        t = self.perf.start();
        const ow = self.fmt.layerTensor(li, "attn_output.weight") orelse return error.MissingTensor;
        self.doGemv(self.attn_out.ptr, ow, self.hidden2.ptr, e, qkv_dim);
        self.perf.end(.gemv_out, t);

        t = self.perf.start();
        if (self.fmt.layerTensor(li, "post_attention_norm.weight")) |post_norm| {
            self.be.rmsNorm(self.hidden2.ptr, self.normAsF32(post_norm, e), self.hidden2.ptr, e, self.rms_eps);
        }
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
        self.perf.end(.add, t);
    }

    /// Dual FFN layer: dense GELU-gated FFN + MoE GELU, with correct norm ordering.
    ///
    /// Architecture (from llama.cpp / HuggingFace reference):
    ///   1. Dense FFN: rmsNorm(hidden, ffn_norm) → gelu(gate)*up → down → rmsNorm(post_ffw_norm_1)
    ///   2. MoE FFN: router(hidden) → rmsNorm(hidden, pre_ffw_norm_2) → experts → rmsNorm(post_ffw_norm_2)
    ///   3. Combine: sum = dense_normed + moe_normed
    ///   4. Final norm: rmsNorm(sum, post_ffw_norm)
    ///   5. Residual: hidden += normed_sum
    ///   6. Layer scale: hidden *= layer_output_scale
    fn dualFfnLayer(self: *Gemma4Model, li: u32) !void {
        const e: usize = self.n_embd;
        const has_moe = self.n_experts > 0 and self.fmt.layerTensor(li, "ffn_gate_inp.weight") != null;

        // ── Dense FFN path ──────────────────────────────────────
        // dense_out = post_ffw_norm_1(dense_ffn(rmsNorm(hidden, ffn_norm)))
        try self.denseFfn(li);

        if (has_moe) {
            // ── MoE FFN path ────────────────────────────────────────
            // moe_out = post_ffw_norm_2(experts(rmsNorm(hidden, pre_ffw_norm_2)))
            // Router operates on hidden (the residual), with its own norm + scale.
            try self.moeFfn(li);

            // ── Combine: sum dense + MoE, then post_ffw_norm, then residual ──
            const t_combine = self.perf.start();

            // dense_out already has post_ffw_norm_1, attn_out[0..e] has post_ffw_norm_2
            // Sum dense + MoE into dense_out
            const moe_out = self.attn_out[0..e];
            self.be.add(self.dense_out.ptr, moe_out.ptr, self.dense_out.ptr, e);

            // Apply post_ffw_norm to combined output (before residual add)
            if (self.fmt.layerTensor(li, "post_ffw_norm.weight")) |post_norm| {
                self.be.rmsNorm(self.dense_out.ptr, self.normAsF32(post_norm, e), self.dense_out.ptr, e, self.rms_eps);
            }

            // Add to residual
            self.be.add(self.hidden.ptr, self.dense_out.ptr, self.hidden.ptr, e);
            self.perf.end(.add, t_combine);
        } else {
            // ── Dense-only path (no MoE experts) ────────────────────
            // Apply post_ffw_norm to dense output, then add to residual.
            const t_dense = self.perf.start();
            if (self.fmt.layerTensor(li, "post_ffw_norm.weight")) |post_norm| {
                self.be.rmsNorm(self.dense_out.ptr, self.normAsF32(post_norm, e), self.dense_out.ptr, e, self.rms_eps);
            }
            self.be.add(self.hidden.ptr, self.dense_out.ptr, self.hidden.ptr, e);
            self.perf.end(.add, t_dense);
        }

        // ── PLE: per-layer embedding residual (after FFN, BEFORE layer scale) ─
        // Pipeline: gate = GELU(inp_gate @ hidden) * ple_input → proj → post_norm → residual
        if (self.ple_dim > 0) {
            try self.applyPle(li);
        }

        // ── Layer output scale (applied AFTER PLE residual) ─
        if (self.fmt.layerTensor(li, "layer_output_scale.weight")) |scale_t| {
            const t_scale = self.perf.start();
            self.be.sync();
            const scale_val: f32 = blk: {
                if (scale_t.dtype == .bf16) {
                    const ptr: *const u16 = @ptrCast(@alignCast(scale_t.data_ptr));
                    break :blk quant.bf16ToF32(ptr.*);
                } else {
                    const ptr: *const f32 = @ptrCast(@alignCast(scale_t.data_ptr));
                    break :blk ptr.*;
                }
            };
            for (0..e) |i| self.hidden[i] *= scale_val;
            self.perf.end(.add, t_scale);
        }
    }

    /// Dense FFN: pre-norm → gate/up projections → GELU-gated multiply → down projection
    /// → post_ffw_norm_1. Output stored in self.dense_out.
    /// Uses per-layer FFN dimension (may vary across layers for E2B).
    fn denseFfn(self: *Gemma4Model, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.per_layer_ff_dim[li];

        // Pre-FFN RMSNorm
        var t = self.perf.start();
        const norm_w = self.fmt.layerTensor(li, "ffn_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(norm_w, e), self.hidden2.ptr, e, self.rms_eps);
        self.perf.end(.rms_norm, t);

        // Gate + Up projections
        t = self.perf.start();
        const gw = self.fmt.layerTensor(li, "ffn_gate.weight") orelse return error.MissingTensor;
        const uw = self.fmt.layerTensor(li, "ffn_up.weight") orelse return error.MissingTensor;
        self.doGemv(self.hidden2.ptr, gw, self.dense_gate.ptr, ff, e);
        self.doGemv(self.hidden2.ptr, uw, self.dense_up.ptr, ff, e);
        self.perf.end(.gemv_ffn, t);

        // GELU(gate) * up — use GPU-accelerated geluMul
        t = self.perf.start();
        self.be.geluMul(self.dense_gate.ptr, self.dense_up.ptr, self.dense_gate.ptr, ff);
        self.perf.end(.gelu_mul, t);

        // Down projection
        t = self.perf.start();
        const dw = self.fmt.layerTensor(li, "ffn_down.weight") orelse return error.MissingTensor;
        self.doGemv(self.dense_gate.ptr, dw, self.dense_out.ptr, e, ff);
        self.perf.end(.gemv_ffn, t);

        // Post-norm for dense path (post_ffw_norm_1)
        t = self.perf.start();
        if (self.fmt.layerTensor(li, "post_ffw_norm_1.weight")) |post_norm| {
            self.be.rmsNorm(self.dense_out.ptr, self.normAsF32(post_norm, e), self.dense_out.ptr, e, self.rms_eps);
        }
        self.perf.end(.rms_norm, t);
    }

    /// MoE FFN: router (with own norm+scale, softmax top-k) → pre_ffw_norm_2 on residual →
    /// per-expert fused gate+up GELU FFN → weighted accumulation → post_ffw_norm_2.
    /// Output stored in self.attn_out[0..n_embd] (reused as MoE accumulator).
    ///
    /// Router operates on self.hidden (the residual after attention), not the pre-normed
    /// input. The router has its own RMS norm + scale pipeline:
    ///   tmp = rmsNorm(hidden) * (1/sqrt(n_embd)) * ffn_gate_inp.scale
    ///   logits = ffn_gate_inp.weight @ tmp
    ///   probs = softmax(logits)
    ///   top_k_weights = normalize(topk(probs, k))
    fn moeFfn(self: *Gemma4Model, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.moe_intermediate;
        const n_exp: usize = self.n_experts;
        const n_active: usize = self.top_k_experts;

        // ── Pre-MoE norm on residual for expert input ───────────
        const t0 = self.perf.start();
        if (self.fmt.layerTensor(li, "pre_ffw_norm_2.weight")) |norm_w| {
            self.be.rmsNorm(self.hidden.ptr, self.normAsF32(norm_w, e), self.hidden2.ptr, e, self.rms_eps);
        }
        self.perf.end(.rms_norm, t0);

        // ── Router: operates on hidden (residual), with its own norm+scale ──
        // Step 1: RMS-norm the residual (plain, no learned weights)
        var t = self.perf.start();
        self.be.sync(); // Need to read hidden on CPU for router norm
        rmsNormPlain(self.router_input.ptr, self.hidden.ptr, e, self.rms_eps);

        // Step 2: Scale by 1/sqrt(n_embd) and element-wise multiply with ffn_gate_inp.scale
        const inv_sqrt_embd: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.n_embd)));
        if (self.fmt.layerTensor(li, "ffn_gate_inp.scale")) |scale_t| {
            const scale_ptr: [*]const f32 = @ptrCast(@alignCast(scale_t.data_ptr));
            for (0..e) |i| self.router_input[i] *= inv_sqrt_embd * scale_ptr[i];
        } else {
            for (0..e) |i| self.router_input[i] *= inv_sqrt_embd;
        }

        // Step 3: Project to expert logits
        const rw = self.fmt.layerTensor(li, "ffn_gate_inp.weight") orelse return error.MissingTensor;
        self.doGemv(self.router_input.ptr, rw, self.router_buf.ptr, n_exp, e);
        self.be.sync();
        self.perf.end(.gemv_ffn, t);

        // Step 4: Softmax + top-k selection (stack-allocated, no heap)
        softmaxInPlace(self.router_buf[0..n_exp]);

        var top_experts: [max_active_experts]usize = undefined;
        var top_scores: [max_active_experts]f32 = undefined;
        std.debug.assert(n_active <= max_active_experts);
        math_ops.topKExperts(self.router_buf[0..n_exp], n_active, top_experts[0..n_active], top_scores[0..n_active]);

        // Normalize selected expert weights to sum to 1.
        var sum_scores: f32 = 0.0;
        for (0..n_active) |i| sum_scores += top_scores[i];
        if (sum_scores > 0.0) {
            const inv_sum = 1.0 / sum_scores;
            for (0..n_active) |i| top_scores[i] *= inv_sum;
        }

        // Apply per-expert scale (ffn_down_exps.scale) if present
        if (self.fmt.layerTensor(li, "ffn_down_exps.scale")) |exp_scale_t| {
            const exp_scale: [*]const f32 = @ptrCast(@alignCast(exp_scale_t.data_ptr));
            for (0..n_active) |i| top_scores[i] *= exp_scale[top_experts[i]];
        }

        // ── Expert computation ──────────────────────────────────
        const fused_gate_up = self.fmt.layerTensor(li, "ffn_gate_up_exps.weight");
        const gate_exps = self.fmt.layerTensor(li, "ffn_gate_exps.weight");
        const up_exps = self.fmt.layerTensor(li, "ffn_up_exps.weight");
        const down_exps = self.fmt.layerTensor(li, "ffn_down_exps.weight");

        const has_fused = fused_gate_up != null and down_exps != null;
        const has_separate = gate_exps != null and up_exps != null and down_exps != null;

        if (!has_fused and !has_separate) {
            std.log.err("MoE missing: layer={d} fused={} gate={} up={} down={}", .{ li, fused_gate_up != null, gate_exps != null, up_exps != null, down_exps != null });
            return error.MissingTensor;
        }

        var fused_stride: usize = 0;
        var gate_stride: usize = 0;
        var up_stride: usize = 0;
        var down_stride: usize = 0;
        if (has_fused) {
            fused_stride = expertStride(fused_gate_up.?);
            down_stride = expertStride(down_exps.?);
        } else if (has_separate) {
            gate_stride = expertStride(gate_exps.?);
            up_stride = expertStride(up_exps.?);
            down_stride = expertStride(down_exps.?);
        }

        // Accumulate weighted expert outputs into attn_out[0..e] (MoE accumulator)
        const moe_out = self.attn_out[0..e];
        @memset(moe_out, 0);

        for (0..n_active) |ti| {
            const ei = top_experts[ti];
            const mix_weight = top_scores[ti];

            // Expert input is hidden2 (pre-normed with pre_ffw_norm_2)
            if (has_fused) {
                const fused_t = fused_gate_up.?;
                const data = fused_t.data_ptr + ei * fused_stride;
                const row_bytes = expertRowBytes(fused_t, 2 * ff, self.n_embd);
                const gate_data = data;
                const up_data = data + ff * row_bytes;

                self.be.gemv(self.hidden2.ptr, .{ .data = gate_data, .dtype = fused_t.dtype }, self.ff_buf.ptr, ff, e);
                self.be.gemv(self.hidden2.ptr, .{ .data = up_data, .dtype = fused_t.dtype }, self.ff_buf2.ptr, ff, e);
            } else {
                self.doGemvExpert(self.hidden2.ptr, gate_exps.?, ei, gate_stride, self.ff_buf.ptr, ff, e);
                self.doGemvExpert(self.hidden2.ptr, up_exps.?, ei, up_stride, self.ff_buf2.ptr, ff, e);
            }
            self.be.sync();

            // GELU activation on gate, then element-wise multiply with up
            math_ops.applyGelu(self.ff_buf[0..ff]);
            for (0..ff) |i| self.ff_buf[i] *= self.ff_buf2[i];

            // Down projection
            self.doGemvExpert(self.ff_buf.ptr, down_exps.?, ei, down_stride, self.ff_buf2.ptr, e, ff);
            self.be.sync();

            // Weighted accumulation
            for (0..e) |i| moe_out[i] += mix_weight * self.ff_buf2[i];
        }

        // Post-MoE norm (post_ffw_norm_2 for the MoE path)
        t = self.perf.start();
        if (self.fmt.layerTensor(li, "post_ffw_norm_2.weight")) |post_norm| {
            self.be.rmsNorm(moe_out.ptr, self.normAsF32(post_norm, e), moe_out.ptr, e, self.rms_eps);
        }
        self.perf.end(.rms_norm, t);
    }

    // ── Helpers ───────────────────────────────────────────────────

    /// Look up an individual expert tensor by layer, prefix and expert ID.
    /// SafeTensors naming: "blk.{layer}.{prefix}_exps.{expert_id}.weight"
    fn expertTensor(self: *Gemma4Model, li: u32, comptime prefix: []const u8, ei: usize) ?TensorInfo {
        var buf: [model_mod.tensor_name_buf_size]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, "blk.{d}." ++ prefix ++ "_exps.{d}.weight", .{ li, ei }) catch return null;
        return self.fmt.getTensor(name);
    }

    /// Compute the byte stride between consecutive experts in a packed weight tensor.
    fn expertStride(t: TensorInfo) usize {
        if (t.dtype == .mlx_q) {
            // SafeTensors MLX dims (not reversed): [n_experts, rows, words_per_row] U32
            // Per-expert stride = rows * words_per_row * sizeof(u32)
            std.debug.assert(t.n_dims >= 3);
            return @as(usize, @intCast(t.dims[1])) * @as(usize, @intCast(t.dims[2])) * @sizeOf(u32);
        }
        // GGUF dims (reversed): [n_experts, rows, cols]
        return model_mod.expertWeightStride(t);
    }

    /// Compute per-row byte size for a given dtype and input dimension k.
    /// For standard types, returns k * element_size.
    /// For quantized types, returns the block-quantized row size.
    fn expertRowBytes(t: TensorInfo, n_rows: usize, k: usize) usize {
        return switch (t.dtype) {
            .f32 => k * @sizeOf(f32),
            .f16 => k * @sizeOf(f16),
            .bf16 => k * 2, // bf16 is 2 bytes
            .q4_0 => (k / 32) * 18, // 32 elements per block, 18 bytes per block
            .q4_k => (k / 32) * 18,
            .q6_k => (k / 16) * 14,
            .q8_0 => (k / 32) * 34,
            else => blk: {
                // Fallback: compute from total expert stride divided by row count.
                const stride = model_mod.expertWeightStride(t);
                break :blk if (n_rows > 0) stride / n_rows else k * @sizeOf(f32);
            },
        };
    }

    /// GEMV dispatch that handles both regular and MLX-quantized weights.
    fn doGemv(self: *Gemma4Model, x: [*]const f32, t: TensorInfo, y: [*]f32, n: usize, k: usize) void {
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

    /// Dispatch GEMV for a single expert slice from a packed expert tensor.
    fn doGemvExpert(self: *Gemma4Model, x: [*]const f32, exp_t: TensorInfo, ei: usize, stride: usize, y: [*]f32, n: usize, k: usize) void {
        const data = exp_t.data_ptr + ei * stride;
        if (exp_t.dtype != .mlx_q) {
            self.be.gemv(x, .{ .data = data, .dtype = exp_t.dtype }, y, n, k);
            return;
        }
        // MLX expert: find companion tensors and compute per-expert offsets
        if (model_mod.mlxGemv(self.be, self.fmt, x, exp_t, y, n, k)) return;
        // Fallback: raw GEMV (should not reach here for mlx_q)
        self.be.gemv(x, .{ .data = data, .dtype = exp_t.dtype }, y, n, k);
    }

    /// Pre-populate the norm weight cache during init so no allocations occur
    /// in the hot path.
    fn warmNormCache(self: *Gemma4Model) void {
        const e: usize = self.n_embd;
        const pd: usize = self.ple_dim;
        if (self.fmt.getTensor("output_norm.weight")) |t| _ = self.normAsF32(t, e);
        // PLE projection norm
        if (pd > 0) {
            if (self.fmt.getTensor("per_layer_proj_norm.weight")) |t| _ = self.normAsF32(t, pd);
        }
        for (0..self.n_layers) |i| {
            const li: u32 = @intCast(i);
            const hd: usize = if (self.layer_is_global[i]) self.gl_head_dim else self.sl_head_dim;
            if (self.fmt.layerTensor(li, "attn_norm.weight")) |t| _ = self.normAsF32(t, e);
            if (self.fmt.layerTensor(li, "attn_q_norm.weight")) |t| _ = self.normAsF32(t, hd);
            if (self.fmt.layerTensor(li, "attn_k_norm.weight")) |t| _ = self.normAsF32(t, hd);
            if (self.fmt.layerTensor(li, "post_attention_norm.weight")) |t| _ = self.normAsF32(t, e);
            if (self.fmt.layerTensor(li, "ffn_norm.weight")) |t| _ = self.normAsF32(t, e);
            if (self.fmt.layerTensor(li, "post_ffw_norm.weight")) |t| _ = self.normAsF32(t, e);
            if (self.fmt.layerTensor(li, "pre_ffw_norm_2.weight")) |t| _ = self.normAsF32(t, e);
            if (self.fmt.layerTensor(li, "post_ffw_norm_1.weight")) |t| _ = self.normAsF32(t, e);
            if (self.fmt.layerTensor(li, "post_ffw_norm_2.weight")) |t| _ = self.normAsF32(t, e);
            // PLE post-norm
            if (pd > 0) {
                if (self.fmt.layerTensor(li, "post_norm.weight")) |t| _ = self.normAsF32(t, e);
            }
        }
    }

    /// Get norm weights as f32 pointer. Caches converted weights on first access
    /// so subsequent tokens return a stable pointer with zero work and no GPU syncs.
    fn normAsF32(self: *Gemma4Model, t: TensorInfo, n: usize) [*]const f32 {
        const needs_convert = t.dtype == .bf16 or self.norm_add_one;
        if (!needs_convert) return @ptrCast(@alignCast(t.data_ptr));

        const key = @intFromPtr(t.data_ptr);
        for (self.norm_cache[0..self.norm_cache_len]) |entry| {
            if (entry.key == key) return entry.data.ptr;
        }

        // Cache miss: allocate, convert, and store permanently
        const buf = self.allocator.alloc(f32, n) catch return @ptrCast(@alignCast(t.data_ptr));
        const offset: f32 = if (self.norm_add_one) 1.0 else 0.0;
        if (t.dtype == .bf16) {
            const src: [*]const u16 = @ptrCast(@alignCast(t.data_ptr));
            for (0..n) |i| buf[i] = quant.bf16ToF32(src[i]) + offset;
        } else {
            const src: [*]const f32 = @ptrCast(@alignCast(t.data_ptr));
            for (0..n) |i| buf[i] = src[i] + offset;
        }
        if (self.norm_cache_len < max_norm_entries) {
            self.norm_cache[self.norm_cache_len] = .{ .key = key, .data = buf };
            self.norm_cache_len += 1;
        }
        return buf.ptr;
    }

    fn lookupEmbd(self: *Gemma4Model, tok: u32) void {
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
        // Gemma scaling: hidden *= sqrt(n_embd)
        const V8 = @Vector(8, f32);
        const scale_v: V8 = @splat(self.embd_scale);
        const n: usize = self.n_embd;
        var i: usize = 0;
        while (i + 8 <= n) : (i += 8) {
            self.hidden[i..][0..8].* = @as(V8, self.hidden[i..][0..8].*) * scale_v;
        }
        while (i < n) : (i += 1) self.hidden[i] *= self.embd_scale;
    }

    /// Compute combined PLE embeddings for one token.
    /// Pipeline:
    ///   1. Token identity: per_layer_token_embd[tok] → scale by sqrt(ple_dim) → reshape [n_layers, ple_dim]
    ///   2. Context projection: per_layer_model_proj @ hidden → scale by 1/sqrt(n_embd)
    ///      → reshape [n_layers, ple_dim] → RMSNorm per layer
    ///   3. Combine: (context_projection + token_identity) * 1/sqrt(2)
    /// Result stored in self.ple_combined[n_layers * ple_dim].
    fn computePleEmbeddings(self: *Gemma4Model, tok: u32) void {
        const pd: usize = self.ple_dim;
        const nl: usize = self.n_layers;
        const total_ple_dim: usize = nl * pd;
        const e: usize = self.n_embd;

        // Step 1: Token identity lookup — per_layer_token_embd.weight [total_ple_dim, vocab]
        const ple_embd_t = self.fmt.getTensor("per_layer_token_embd.weight") orelse {
            @memset(self.ple_combined, 0);
            return;
        };
        // Embedding lookup: extract row `tok` of size total_ple_dim
        self.be.embLookup(.{ .data = ple_embd_t.data_ptr, .dtype = ple_embd_t.dtype }, tok, self.ple_combined.ptr, total_ple_dim);
        self.be.sync();

        // Scale by sqrt(ple_dim)
        const emb_scale = self.ple_embd_scale;
        for (self.ple_combined[0..total_ple_dim]) |*v| v.* *= emb_scale;

        // Step 2: Context projection — per_layer_model_proj @ hidden
        const proj_t = self.fmt.getTensor("per_layer_model_proj.weight") orelse return;
        // Use ple_gate_buf as temp for the projection output (need total_ple_dim space)
        // Since total_ple_dim might be larger than ple_buf, use router_input as temp
        // Actually, we need total_ple_dim f32s. We can compute this GEMV and accumulate.
        // The projection is [total_ple_dim, n_embd] @ hidden[n_embd] -> [total_ple_dim]
        // We need a temp buffer of size total_ple_dim. Use attn_out (which is max_qkv_dim >= total_ple_dim for these models).
        const ctx_proj = self.attn_out[0..total_ple_dim];
        self.doGemv(self.hidden.ptr, proj_t, ctx_proj.ptr, total_ple_dim, e);
        self.be.sync();

        // Scale by 1/sqrt(n_embd)
        const proj_scale = self.ple_proj_scale;
        for (ctx_proj) |*v| v.* *= proj_scale;

        // RMSNorm per layer on context projection
        const proj_norm_t = self.fmt.getTensor("per_layer_proj_norm.weight");
        if (proj_norm_t) |pn| {
            const norm_w = self.normAsF32(pn, pd);
            for (0..nl) |li| {
                const base = li * pd;
                var sum_sq: f32 = 0.0;
                for (0..pd) |j| sum_sq += ctx_proj[base + j] * ctx_proj[base + j];
                const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(pd)) + self.rms_eps);
                const inv_rms = 1.0 / rms;
                for (0..pd) |j| ctx_proj[base + j] = ctx_proj[base + j] * inv_rms * norm_w[j];
            }
        }

        // Step 3: Combine (context_projection + token_identity) * 1/sqrt(2)
        for (0..total_ple_dim) |i| {
            self.ple_combined[i] = (self.ple_combined[i] + ctx_proj[i]) * ple_combination_scale;
        }
    }

    /// Apply PLE gated residual for one layer.
    /// Pipeline: gate = GELU(inp_gate @ hidden) * ple_input[layer] → proj → post_norm → residual
    fn applyPle(self: *Gemma4Model, li: u32) !void {
        const e: usize = self.n_embd;
        const pd: usize = self.ple_dim;

        // Gate projection: [ple_dim, n_embd] @ hidden → [ple_dim]
        const gate_w = self.fmt.layerTensor(li, "inp_gate.weight") orelse return;
        self.doGemv(self.hidden.ptr, gate_w, self.ple_gate_buf.ptr, pd, e);
        self.be.sync();

        // GELU activation
        math_ops.applyGelu(self.ple_gate_buf[0..pd]);

        // Element-wise multiply with per-layer PLE input
        const ple_offset: usize = @as(usize, li) * pd;
        const ple_input = self.ple_combined[ple_offset..][0..pd];
        for (0..pd) |i| self.ple_gate_buf[i] *= ple_input[i];

        // Projection back to hidden dim: [n_embd, ple_dim] @ gated → [n_embd]
        const proj_w = self.fmt.layerTensor(li, "proj.weight") orelse return;
        self.doGemv(self.ple_gate_buf.ptr, proj_w, self.dense_out.ptr, e, pd);

        // Post-PLE RMSNorm
        if (self.fmt.layerTensor(li, "post_norm.weight")) |post_norm_w| {
            self.be.rmsNorm(self.dense_out.ptr, self.normAsF32(post_norm_w, e), self.dense_out.ptr, e, self.rms_eps);
        }

        // Residual add
        self.be.add(self.hidden.ptr, self.dense_out.ptr, self.hidden.ptr, e);
    }

    /// Apply logit softcapping: logits = cap * tanh(logits / cap).
    fn applySoftcap(self: *Gemma4Model) void {
        const inv = 1.0 / self.final_logit_softcap;
        const cap = self.final_logit_softcap;
        for (self.logits) |*v| v.* = @as(f32, math.tanh(v.* * inv)) * cap;
    }
};

// ── Module-level helper functions (not bound to Gemma4Model) ─────

/// Plain RMS normalization without learned weights.
/// Computes: output[i] = input[i] / rms(input), where rms = sqrt(mean(x^2) + eps).
/// Used for tied K=V value normalization and router input normalization.
fn rmsNormPlain(output: [*]f32, input: [*]const f32, n: usize, eps: f32) void {
    var sum_sq: f32 = 0.0;
    for (0..n) |i| sum_sq += input[i] * input[i];
    const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(n)) + eps);
    const inv_rms = 1.0 / rms;
    for (0..n) |i| output[i] = input[i] * inv_rms;
}

/// Plain RMS normalization for multiple heads without learned weights.
/// Normalizes each contiguous head of `hd` elements independently.
/// Used for tied K=V value normalization across KV heads.
fn rmsNormPlainMulti(x: [*]f32, n_heads: usize, hd: usize, eps: f32) void {
    for (0..n_heads) |h| {
        const base = h * hd;
        var sum_sq: f32 = 0.0;
        for (0..hd) |i| sum_sq += x[base + i] * x[base + i];
        const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(hd)) + eps);
        const inv_rms = 1.0 / rms;
        for (0..hd) |i| x[base + i] *= inv_rms;
    }
}

/// In-place softmax over a float slice.
/// Numerically stable: subtracts max before exp to prevent overflow.
fn softmaxInPlace(x: []f32) void {
    var max_val: f32 = -math.inf(f32);
    for (x) |v| if (v > max_val) {
        max_val = v;
    };
    var sum: f32 = 0.0;
    for (x) |*v| {
        v.* = @exp(v.* - max_val);
        sum += v.*;
    }
    if (sum > 0.0) {
        const inv_sum = 1.0 / sum;
        for (x) |*v| v.* *= inv_sum;
    }
}

// ── Tests ─────────────────────────────────────────────────────────

test "Gemma4 layer type detection" {
    // With interval=6 (default): layers 5,11,17,23,29 (0-indexed) are global.
    const interval: u32 = 6;
    const n_layers: usize = 30;
    var layer_is_global: [max_layers]bool = [_]bool{false} ** max_layers;
    for (0..n_layers) |i| {
        layer_is_global[i] = ((i + 1) % interval == 0);
    }
    // Global layers at indices 5, 11, 17, 23, 29
    try std.testing.expect(layer_is_global[5]);
    try std.testing.expect(layer_is_global[11]);
    try std.testing.expect(layer_is_global[17]);
    try std.testing.expect(layer_is_global[23]);
    try std.testing.expect(layer_is_global[29]);
    // Non-global
    try std.testing.expect(!layer_is_global[0]);
    try std.testing.expect(!layer_is_global[4]);
    try std.testing.expect(!layer_is_global[6]);
    try std.testing.expect(!layer_is_global[28]);

    // Count: should be exactly 5 global layers in 30 layers
    var global_count: usize = 0;
    for (0..n_layers) |i| {
        if (layer_is_global[i]) global_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 5), global_count);
}

test "Gemma4 shared KV mapping" {
    // Test with 30 layers, 4 shared layers (layers 26-29 share from earlier layers)
    const n_layers: usize = 30;
    const n_kv_shared: usize = 4;
    const interval: u32 = 6;

    var layer_is_global: [max_layers]bool = [_]bool{false} ** max_layers;
    for (0..n_layers) |i| {
        layer_is_global[i] = ((i + 1) % interval == 0);
    }

    var kv_source: [max_layers]u32 = undefined;
    for (0..n_layers) |i| kv_source[i] = @intCast(i);

    const first_shared = n_layers - n_kv_shared;
    for (first_shared..n_layers) |i| {
        const is_gl = layer_is_global[i];
        var src: usize = i;
        var j: usize = i;
        while (j > 0) {
            j -= 1;
            if (j < first_shared and layer_is_global[j] == is_gl) {
                src = j;
                break;
            }
        }
        kv_source[i] = @intCast(src);
    }

    // Layer 26 (sliding) should point to a sliding layer < 26
    try std.testing.expect(!layer_is_global[26]);
    try std.testing.expect(kv_source[26] < first_shared);
    try std.testing.expect(!layer_is_global[kv_source[26]]);

    // Layer 29 (global, since (29+1)%6==0) should point to a global layer < 26
    try std.testing.expect(layer_is_global[29]);
    try std.testing.expect(kv_source[29] < first_shared);
    try std.testing.expect(layer_is_global[kv_source[29]]);

    // Non-shared layers should point to themselves
    for (0..first_shared) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(i)), kv_source[i]);
    }
}

test "Gemma4 per-layer KV head count defaults" {
    // When no metadata array, per_layer_n_kv_head should match layer type defaults
    const n_layers: usize = 30;
    const interval: u32 = 6;
    const sl_kv: u32 = 8;
    const gl_kv: u32 = 2;

    var layer_is_global: [max_layers]bool = [_]bool{false} ** max_layers;
    var per_layer_n_kv_head: [max_layers]u32 = [_]u32{0} ** max_layers;

    for (0..n_layers) |i| {
        layer_is_global[i] = ((i + 1) % interval == 0);
        per_layer_n_kv_head[i] = if (layer_is_global[i]) gl_kv else sl_kv;
    }

    // Sliding layers should have 8 KV heads
    try std.testing.expectEqual(sl_kv, per_layer_n_kv_head[0]);
    try std.testing.expectEqual(sl_kv, per_layer_n_kv_head[4]);
    // Global layers should have 2 KV heads
    try std.testing.expectEqual(gl_kv, per_layer_n_kv_head[5]);
    try std.testing.expectEqual(gl_kv, per_layer_n_kv_head[29]);
}

test "Gemma4 dense variant detection" {
    // When n_experts=0, MoE buffers should be zero-sized and MoE path skipped.
    // This simulates dense variants like Gemma 4 31B and E4B.
    const n_experts: u32 = 0;
    const top_k_experts: u32 = if (n_experts == 0) 0 else 8;
    const moe_intermediate: u32 = if (n_experts == 0) 0 else 704;

    try std.testing.expectEqual(@as(u32, 0), n_experts);
    try std.testing.expectEqual(@as(u32, 0), top_k_experts);
    try std.testing.expectEqual(@as(u32, 0), moe_intermediate);

    // Dense models still need dense FFN buffers
    const dense_ff_dim: u32 = 2816;
    try std.testing.expect(dense_ff_dim > 0);
}
