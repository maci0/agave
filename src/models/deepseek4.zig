//! DeepSeek V4 Flash 0731 inference model.
//! Uses GGUF-style tensor names (blk.N.*).
//! Architecture: 4-stream hyper connections, modified MLA (K=V single compressed head,
//! no separate V projection), hash routing (layers 0-2), sqrt_softplus routing (3+),
//! grouped output LoRA (8 groups × 1024 rank).
//! KV compressors: CSA (ratio=4, 21 layers) and HCA (ratio=128, 20 layers) — both
//! fully implemented with per-ratio APE, group compression, and compressed attention.
//! Lightning Indexer (LID): implemented for CSA layers. When >index_topk compressed
//! blocks exist, scores all blocks via multi-head ReLU dot-product and selects top-k
//! for sparse attention. Gracefully disabled when indexer tensors are absent.

const std = @import("std");
const Allocator = std.mem.Allocator;
const backend_mod = @import("../backend/backend.zig");
const format_mod = @import("../format/format.zig");
const model_mod = @import("model.zig");
const math_ops = @import("../ops/math.zig");

const quant_ops = @import("../ops/quant.zig");
const kv_quant = @import("../ops/kv_quant.zig");
const attn_ops = @import("../ops/attention.zig");
const Backend = backend_mod.Backend;
const KvQuantType = kv_quant.KvQuantType;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const DType = format_mod.DType;
const Model = model_mod.Model;
const TensorData = backend_mod.TensorData;

const name_buf_size: usize = model_mod.tensor_name_buf_size;
const n_hc: usize = 4;
const hc_mix_dim: usize = (2 + n_hc) * n_hc; // = 24
/// 8 iterations suffice for 4×4 doubly-stochastic convergence (was 20, no quality change).
const hc_sinkhorn_iters: usize = 8;
const hc_eps: f32 = 1e-6;
const max_norm_entries: usize = 512;

const NormCacheEntry = struct { key: usize, data: []f32 };

/// Sparse V threshold: skip V dequant+accumulation for positions where softmax
/// weight is below this value. At 1e-6, skipped positions contribute < 0.0001%
/// to the output — zero measured PPL impact. Matches attention.zig threshold.
const sparse_v_threshold: f32 = 1e-6;

/// Max compressed groups per layer. Uses ratio=4 (smallest) to size the shared stride,
/// since CSA and HCA layers share the same `csa_k` buffer with per-layer offsets.
fn compSlotsPerLayer(max_seq_len: usize) usize {
    return max_seq_len / 4 + 1;
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
    layer_skip_start: u32 = 0,
    layer_skip_end: u32 = 0,
    pool: ?*@import("../thread_pool.zig").ThreadPool = null,
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
    router_logits: []f32 = &.{}, // [n_experts]
    logits_buf: []f32 = &.{}, // [vocab_size]
    score_stride: usize = 0, // per-head score buffer stride

    // Pre-computed RoPE frequency bases [rope_dim/2]. Eliminates pow() per token.
    rope_freqs: [32]f32 = undefined, // freq_base = rope_freq (layers with ratio=0)
    compress_rope_freqs: [32]f32 = undefined, // freq_base = compress_rope_freq (ratio≠0 layers)

    // KV cache as quantized bytes: [n_layers * ctx * kv_lora_rank].
    // K=V in DS4 MLA (single compressed head) — one buffer serves both K and V.
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

    // Prefill buffers (page_allocator for GPU zero-copy — Metal's
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
        // DS4 MLA compressed attention uses kvDot/kvMulAccum which require block-quantized
        // KV types. f16/f32 are unsupported (no block structure for the dequant inner loop).
        // Default to q8_0 for maximum quality; accept any block-quantized type from CLI.
        const effective_kv = switch (kv_type_k) {
            .f16, .f32 => .q8_0, // unsupported → fall back to q8_0
            else => kv_type_k,
        };
        var self = Ds4Model{ .fmt = f, .be = be, .allocator = allocator, .kv_type = effective_kv };

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

        // Read compress_ratios via per-element metadata access.
        // GGUF stores as array; read each element using arch key with index.
        // DS4 Flash 43-layer pattern: [0,0,4,128,4,128,...,4] (layers 2+ alternate)
        for (0..@as(usize, self.n_layers)) |i| {
            // Fallback: layers 0-1 have no compression; layers 2+ alternate 4/128
            self.compress_ratios[i] = if (i < 2) 0 else if (i % 2 == 0) 4 else 128;
        }
        if (f.getMetaF32("deepseek4.expert_weights_scale")) |v| self.expert_weights_scale = v;
        if (f.getArchU32(arch, "context_length")) |v| self.max_seq_len = @min(v, 65536);
        if (ctx_size > 0) self.max_seq_len = ctx_size;

        if (f.getTensor("token_embd.weight")) |t| {
            if (t.n_dims >= 2) self.vocab_size = @intCast(t.dims[1]);
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
        self.score_stride = ctx + compSlotsPerLayer(ctx) + 1;
        self.scores_buf = try allocator.alloc(f32, nh * self.score_stride);
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
        self.expert_scratch = try allocator.alloc(f32, max_experts * e);
        errdefer allocator.free(self.expert_scratch);
        self.ff_gate_scratch = try allocator.alloc(f32, max_experts * ff);
        errdefer allocator.free(self.ff_gate_scratch);
        self.ff_up_scratch = try allocator.alloc(f32, max_experts * ff);
        errdefer allocator.free(self.ff_up_scratch);
        self.router_logits = try allocator.alloc(f32, self.n_experts);
        errdefer allocator.free(self.router_logits);
        self.logits_buf = try allocator.alloc(f32, self.vocab_size);
        errdefer allocator.free(self.logits_buf);

        // KV cache bytes: K=V shared buffer (MLA single compressed head, halves KV memory).
        const kv_bytes_per_layer = kv_quant.kvByteOffset(self.kv_type, ctx * kd);
        self.kv_k_bytes = try allocator.alloc(u8, nl * kv_bytes_per_layer);
        errdefer allocator.free(self.kv_k_bytes);

        // Compressor buffers: per-token projections for the current compression group only.
        // CSA groups have ratio=4, HCA groups have ratio=128. We use max_ratio=128 as a
        // circular buffer per layer — completed groups are compressed into csa_k and the
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

        // Prefill buffers: deferred — DS4 prefill is sequential (see prefill() doc comment).
        // Fields stay as empty slices; allocate on first batched-prefill use (future).
        // Saves ~64 MB page_allocator memory that was allocated but never touched.

        self.warmNormCache();
        return self;
    }

    /// Pre-dequantize all norm weights at init time to avoid per-token allocations.
    fn warmNormCache(self: *Ds4Model) void {
        const e = self.n_embd;
        _ = self.normAsF32OrNull(self.fmt.getTensor("output_norm.weight"), e);
        _ = self.normAsF32OrNull(self.fmt.getTensor("output_hc_fn.weight"), n_hc);
        _ = self.normAsF32OrNull(self.fmt.getTensor("output_hc_base.weight"), n_hc);
        _ = self.normAsF32OrNull(self.fmt.getTensor("output_hc_scale.weight"), 1);
        for (0..self.n_layers) |li| {
            _ = self.normAsF32OrNull(self.layerTensor(li, "attn_norm.weight"), e);
            _ = self.normAsF32OrNull(self.layerTensor(li, "ffn_norm.weight"), e);
            _ = self.normAsF32OrNull(self.layerTensor(li, "attn_q_a_norm.weight"), self.q_lora_rank);
            _ = self.normAsF32OrNull(self.layerTensor(li, "attn_kv_a_norm.weight"), self.kv_lora_rank);
            _ = self.normAsF32OrNull(self.layerTensor(li, "hc_attn_base.weight"), hc_mix_dim);
            _ = self.normAsF32OrNull(self.layerTensor(li, "hc_attn_scale.weight"), 3);
            _ = self.normAsF32OrNull(self.layerTensor(li, "hc_ffn_base.weight"), hc_mix_dim);
            _ = self.normAsF32OrNull(self.layerTensor(li, "hc_ffn_scale.weight"), 3);
            _ = self.normAsF32OrNull(self.layerTensor(li, "attn_compressor_norm.weight"), self.kv_lora_rank);
        }
    }

    /// Release all heap allocations owned by this model, including norm cache,
    /// working buffers, KV cache, compressor state, and LID indexer buffers.
    pub fn deinit(self: *Ds4Model) void {
        self.be.sync();
        for (self.norm_cache[0..self.norm_cache_len]) |e| self.allocator.free(e.data);
        const a = self.allocator;
        inline for (.{
            &self.hc_state,          &self.new_hc,          &self.hc_mixes,      &self.hc_pre_w,       &self.hc_post_w,
            &self.hc_comb,           &self.hidden,          &self.hidden2,       &self.flat_norm,      &self.q_compressed,
            &self.q_full,            &self.kv_proj,         &self.scores_buf,    &self.attn_out,       &self.lora_out,
            &self.attn_result,       &self.ff_gate,         &self.ff_up,         &self.ff_down,        &self.expert_accum,
            &self.expert_scratch,    &self.ff_gate_scratch, &self.ff_up_scratch, &self.router_logits,  &self.logits_buf,
            &self.kv_k_bytes,        &self.csa_comp_kv,     &self.csa_comp_score, &self.csa_k,
            &self.csa_score_scratch, &self.lid_comp_k,      &self.lid_query,     &self.lid_head_w,     &self.lid_scores,
        }) |buf| a.free(buf.*);
        if (self.lid_topk_ids.len > 0) a.free(self.lid_topk_ids);
        // Prefill buffers (page_allocator) — currently empty slices (allocation deferred).
        // Guards remain for forward compatibility when batched prefill is implemented.
        {
            const pa = std.heap.page_allocator;
            const pf_bufs = .{
                &self.pf_hidden,  &self.pf_hidden2, &self.pf_q_a,
                &self.pf_q,       &self.pf_kv_proj, &self.pf_attn_out,
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

    /// K=V in DS4 MLA — V cache is the same buffer as K cache.
    fn kvVLayer(self: *Ds4Model, li: usize) []u8 {
        return self.kvKLayer(li);
    }

    // ── Hyper Connection ──────────────────────────────────────────

    /// Debug: disable HC, use mean of streams as identity
    const debug_disable_hc = false;

    /// Compute HC pre-weights and sublayer input in `self.hidden`.
    fn hcPre(
        self: *Ds4Model,
        hc_fn: TensorInfo,
        hc_base: TensorInfo,
        hc_scale: TensorInfo,
    ) void {
        if (debug_disable_hc) {
            // Identity: just mean of HC streams
            const e = self.n_embd;
            @memset(self.hidden, 0.0);
            for (0..n_hc) |s| {
                const stream = self.hc_state[s * e ..][0..e];
                for (0..e) |i| self.hidden[i] += stream[i] * (1.0 / n_hc);
            }
            return;
        }
        const e = self.n_embd;
        const flat_size = n_hc * e;

        // Compute RMS scale factor from hc_state without copying.
        // Then run GEMV on raw hc_state and post-scale the output.
        // Saves 16KB memcpy + in-place norm pass (was: copy → norm → GEMV).
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

        // mixes[24] = hc_fn @ hc_state — then post-scale by rms_inv
        if (hc_fn.dtype == .q8_0) {
            cpuGemvQ8_0(hc_fn.data_ptr, self.hc_state, self.hc_mixes, flat_size);
        } else {
            @memcpy(self.flat_norm, self.hc_state); // GPU path still needs stable buffer
            self.be.gemv(self.flat_norm.ptr, .{ .data = hc_fn.data_ptr, .dtype = hc_fn.dtype }, self.hc_mixes.ptr, hc_mix_dim, flat_size);
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
        // Comb: raw affine values — sinkhorn applies its own softmax
        for (0..n_hc * n_hc) |s| {
            self.hc_comb[s] = mixes[2 * n_hc + s] * scale[2] + base[2 * n_hc + s];
        }
        hcSinkhorn(self.hc_comb);

        // Weighted sum of HC streams → sublayer input (SIMD-optimized)
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
        if (debug_disable_hc) {
            const e = self.n_embd;
            for (0..n_hc) |s| @memcpy(self.hc_state[s * e ..][0..e], self.hidden);
            return;
        }
        const e = self.n_embd;
        const sub = self.hidden;
        const V8 = @Vector(8, f32);
        for (0..n_hc) |dst| {
            const ns = self.new_hc[dst * e ..][0..e];
            const pw: V8 = @splat(self.hc_post_w[dst]);
            // Pre-load comb coefficients for this dst
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
        // Swap buffers instead of copying 16KB
        const tmp = self.hc_state;
        self.hc_state = self.new_hc;
        self.new_hc = tmp;
    }

    /// HC head: merge 4 streams → self.hidden.
    fn hcHead(self: *Ds4Model, hc_fn: TensorInfo, hc_base: TensorInfo, hc_scale: TensorInfo) void {
        if (debug_disable_hc) {
            // Identity: mean of streams
            const e = self.n_embd;
            @memset(self.hidden, 0.0);
            for (0..n_hc) |s| {
                const stream = self.hc_state[s * e ..][0..e];
                for (0..e) |i| self.hidden[i] += stream[i] * (1.0 / n_hc);
            }
            return;
        }
        const e = self.n_embd;
        const flat_size_h = n_hc * e;
        // Same post-scale RMS optimization as hcPre
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
        } else {
            @memcpy(self.flat_norm, self.hc_state);
            self.be.gemv(self.flat_norm.ptr, .{ .data = hc_fn.data_ptr, .dtype = hc_fn.dtype }, self.hc_pre_w.ptr, n_hc, flat_size_h);
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
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        const q_a = try self.layerTensorReq(li, "attn_q_a.weight");
        self.be.gemv(self.hidden2.ptr, .{ .data = q_a.data_ptr, .dtype = q_a.dtype }, self.q_compressed.ptr, ql, e);
        const q_an = try self.layerTensorReq(li, "attn_q_a_norm.weight");
        self.be.rmsNorm(self.q_compressed.ptr, self.normAsF32(q_an, ql), self.q_compressed.ptr, ql, self.rms_eps);
        const q_b = try self.layerTensorReq(li, "attn_q_b.weight");
        self.be.gemv(self.q_compressed.ptr, .{ .data = q_b.data_ptr, .dtype = q_b.dtype }, self.q_full.ptr, nh * kd, ql);
        const kv_a = try self.layerTensorReq(li, "attn_kv.weight");
        self.be.gemv(self.hidden2.ptr, .{ .data = kv_a.data_ptr, .dtype = kv_a.dtype }, self.kv_proj.ptr, kd, e);
        const kv_an = try self.layerTensorReq(li, "attn_kv_a_norm.weight");
        self.be.rmsNorm(self.kv_proj.ptr, self.normAsF32(kv_an, kd), self.kv_proj.ptr, kd, self.rms_eps);

        // Compressor projections for all compressed layers (CSA ratio=4, HCA ratio=128).
        // Both batched with Q+KV in same GPU command buffer — single sync covers all.
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
                self.be.gemv(self.hidden2.ptr, .{ .data = wkv.data_ptr, .dtype = wkv.dtype }, comp_kv_pos.ptr, actual_comp_dim, e);
                self.be.gemv(self.hidden2.ptr, .{ .data = kwgate.data_ptr, .dtype = kwgate.dtype }, comp_score_pos.ptr, actual_comp_dim, e);
            }
        }

        // Pre-dispatch LID GEMVs into same GPU command buffer — eliminates 1 sync per CSA layer.
        // Inputs (q_compressed, hidden) are already in the pipeline; GPU ordering guarantees correctness.
        const lid_pre_dispatched: bool = blk: {
            if (!self.lid_enabled or ratio != 4) break :blk false;
            const n_comp_early: usize = (pos + 1) / ratio;
            if (n_comp_early <= self.index_topk) break :blk false;
            const inh: usize = self.index_n_heads;
            const ihd: usize = self.index_head_dim;
            if (self.layerTensor(li, "attn_indexer_q_b.weight")) |wiq| {
                self.be.gemv(self.q_compressed.ptr, .{ .data = wiq.data_ptr, .dtype = wiq.dtype }, self.lid_query.ptr, inh * ihd, self.q_lora_rank);
            } else break :blk false;
            if (self.layerTensor(li, "attn_indexer_proj.weight")) |ww| {
                self.be.gemv(self.hidden.ptr, .{ .data = ww.data_ptr, .dtype = ww.dtype }, self.lid_head_w.ptr, inh, self.n_embd);
            } else {
                for (self.lid_head_w[0..inh]) |*w| w.* = 1.0 / @as(f32, @floatFromInt(inh));
            }
            break :blk true;
        };

        self.be.sync(); // single sync: Q, KV, CSA/HCA, and (if batched) LID GEMVs

        // RoPE cos/sin from pre-computed freq bases — SIMD vectorized.
        const nd = rd / 2;
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

        // CPU: per-head Q RMS norm + Q RoPE (using pre-computed table)
        for (0..nh) |h| {
            const q_head = self.q_full[h * kd ..][0..kd];
            plainRmsNorm(q_head, self.rms_eps);
            applyRopeTable(q_head[nope..][0..rd], rope_cos[0..nd], rope_sin[0..nd]);
        }
        // CPU: KV RoPE
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

                    // Softmax-weighted compression: for each dimension d, compute
                    // compressed[d] = Σ_t softmax(score[t,d]) × kv[t,d].
                    // CSA (ratio=4): unrolled 4-token softmax with SIMD across dimensions.
                    // HCA (ratio=128): scalar per-dimension loop (ratio too large to unroll).
                    var compressed: [2048]f32 = undefined;
                    std.debug.assert(actual_comp_dim <= compressed.len);
                    // Circular buffer: group_start % comp_buf_ratio gives the start slot.
                    // Groups are ratio-aligned, so slots within a group are consecutive
                    // modulo comp_buf_ratio (which is >= max(ratio)=128).
                    const circ_base = li * comp_layer_stride + (group_start % comp_buf_ratio) * max_comp_dim;
                    if (ratio == 4) {
                        // SIMD-optimized: process 8 dimensions at a time across 4 tokens
                        const s0 = self.csa_comp_score[circ_base ..][0..actual_comp_dim];
                        const s1 = self.csa_comp_score[circ_base + max_comp_dim ..][0..actual_comp_dim];
                        const s2 = self.csa_comp_score[circ_base + 2 * max_comp_dim ..][0..actual_comp_dim];
                        const s3 = self.csa_comp_score[circ_base + 3 * max_comp_dim ..][0..actual_comp_dim];
                        const k0 = self.csa_comp_kv[circ_base ..][0..actual_comp_dim];
                        const k1 = self.csa_comp_kv[circ_base + max_comp_dim ..][0..actual_comp_dim];
                        const k2 = self.csa_comp_kv[circ_base + 2 * max_comp_dim ..][0..actual_comp_dim];
                        const k3 = self.csa_comp_kv[circ_base + 3 * max_comp_dim ..][0..actual_comp_dim];
                        const V8 = @Vector(8, f32);
                        var d: usize = 0;
                        while (d + 8 <= actual_comp_dim) : (d += 8) {
                            const sv0: V8 = s0[d..][0..8].*;
                            const sv1: V8 = s1[d..][0..8].*;
                            const sv2: V8 = s2[d..][0..8].*;
                            const sv3: V8 = s3[d..][0..8].*;
                            // Max across 4 tokens
                            const mx = @max(@max(sv0, sv1), @max(sv2, sv3));
                            // Exp
                            const e0 = @exp(sv0 - mx);
                            const e1 = @exp(sv1 - mx);
                            const e2 = @exp(sv2 - mx);
                            const e3 = @exp(sv3 - mx);
                            const ones: V8 = @splat(1.0);
                            const sm_inv: V8 = ones / (e0 + e1 + e2 + e3);
                            // Weighted sum of KV
                            const kv0: V8 = k0[d..][0..8].*;
                            const kv1: V8 = k1[d..][0..8].*;
                            const kv2: V8 = k2[d..][0..8].*;
                            const kv3: V8 = k3[d..][0..8].*;
                            compressed[d..][0..8].* = (e0 * kv0 + e1 * kv1 + e2 * kv2 + e3 * kv3) * sm_inv;
                        }
                        // Scalar tail
                        while (d < actual_comp_dim) : (d += 1) {
                            const mx = @max(@max(s0[d], s1[d]), @max(s2[d], s3[d]));
                            const e0 = @exp(s0[d] - mx);
                            const e1 = @exp(s1[d] - mx);
                            const e2 = @exp(s2[d] - mx);
                            const e3 = @exp(s3[d] - mx);
                            const inv = 1.0 / (e0 + e1 + e2 + e3);
                            compressed[d] = (e0 * k0[d] + e1 * k1[d] + e2 * k2[d] + e3 * k3[d]) * inv;
                        }
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
                    if (self.lid_enabled and ratio == 4) {
                        self.lidCompressGroup(li, group_start, group_idx, comp_slots);
                    }
                }
            }
        }

        // SDPA: head_dim=512 exceeds Metal GPU limit (256), force CPU SIMD path via window.
        // K=V in DS4 MLA (single compressed head, GQA 64:1).
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));
        const kv_k_layer = self.kvKLayer(li);
        // K=V shared buffer — no separate V layer needed (halves KV cache + eliminates duplicate kvStore).
        // Compressed group count: works for any ratio (CSA=4, HCA=128).
        const n_comp_groups: usize = if (ratio != 0) (pos + 1) / ratio else 0;

        if (n_comp_groups > 0) {
            // Compressed attention: append current KV to cache, then attend raw + compressed.
            const k_byte_off = kv_quant.kvByteOffset(self.kv_type, pos * kd);
            kv_quant.kvStore(kv_k_layer[k_byte_off..].ptr, self.kv_proj.ptr, kd, self.kv_type);

            const comp_slots = compSlotsPerLayer(self.max_seq_len);

            // Lightning Indexer: select top-k compressed blocks for CSA layers.
            // When n_comp_groups > index_topk and LID tensors are present, score all
            // compressed blocks and attend only to the top-k (sparse).
            // Otherwise fall back to dense attention over all compressed groups.
            const itk: usize = self.index_topk;
            const use_lid = self.lid_enabled and ratio == 4 and n_comp_groups > itk;
            var n_attend_comp: usize = n_comp_groups;

            if (use_lid) {
                self.lidScoreAndSelect(li, n_comp_groups, comp_slots, lid_pre_dispatched);
                n_attend_comp = @min(itk, n_comp_groups);
            }

            const sl_total = pos + 1 + n_attend_comp;

            const kv_elem_bytes = kv_quant.kvByteOffset(self.kv_type, kd);
            // Hoist sink tensor lookup outside per-head loop (avoids 128 hash lookups)
            const sink_data: ?[*]const f32 = if (self.layerTensor(li, "attn_sinks.weight")) |st|
                @ptrCast(@alignCast(st.data_ptr))
            else
                null;
            const V8 = @Vector(8, f32);
            const ss = self.score_stride;

            // Dispatch per-head attention via thread pool when available (64 independent heads).
            if (self.pool) |pool| {
                var ctx = CompressedAttnCtx{
                    .q_full = self.q_full,
                    .scores_buf = self.scores_buf,
                    .attn_out = self.attn_out,
                    .kv_k_layer = kv_k_layer,
                    .kv_v_layer = kv_k_layer, // K=V shared
                    .csa_k = self.csa_k,
                    .lid_topk_ids = if (use_lid) self.lid_topk_ids else &.{},
                    .sink_data = sink_data,
                    .kd = kd,
                    .pos = pos,
                    .ss = ss,
                    .kv_elem_bytes = kv_elem_bytes,
                    .n_attend_comp = n_attend_comp,
                    .sl_total = sl_total,
                    .comp_slots = comp_slots,
                    .li = li,
                    .scale = scale,
                    .use_lid = use_lid,
                    .kv_quant_type = self.kv_type,
                };
                pool.parallelFor(nh, 1, @ptrCast(&ctx), CompressedAttnCtx.perHeadFn);
            } else for (0..nh) |h| {
                const q_h = self.q_full[h * kd ..][0..kd];
                const scores_h = self.scores_buf[h * ss ..];
                // QK dot products — fuse running max to skip a separate max pass.
                // Prefetch next KV block while computing current dot product.
                var running_max: f32 = -std.math.inf(f32);
                for (0..pos + 1) |t| {
                    const k_ptr = kv_k_layer[t * kv_elem_bytes ..].ptr;
                    if (t + 1 <= pos) @prefetch(kv_k_layer[(t + 1) * kv_elem_bytes ..].ptr, .{ .locality = 3 });
                    const s = kv_quant.kvDot(q_h.ptr, k_ptr, kd, self.kv_type) * scale;
                    scores_h[t] = s;
                    running_max = @max(running_max, s);
                }
                // QK for compressed positions (f32 SIMD with @mulAdd, tracking max)
                for (0..n_attend_comp) |gi| {
                    const g = if (use_lid) self.lid_topk_ids[gi] else @as(u32, @intCast(gi));
                    const ck = self.csa_k[(li * comp_slots + g) * kd ..][0..kd];
                    var acc: V8 = @splat(0.0);
                    var i: usize = 0;
                    while (i + 8 <= kd) : (i += 8) {
                        acc = @mulAdd(V8, @as(V8, q_h[i..][0..8].*), @as(V8, ck[i..][0..8].*), acc);
                    }
                    var dot = @reduce(.Add, acc);
                    while (i < kd) : (i += 1) dot = @mulAdd(f32, q_h[i], ck[i], dot);
                    const s = dot * scale;
                    scores_h[pos + 1 + gi] = s;
                    running_max = @max(running_max, s);
                }
                // Softmax — max already known from scoring, skip max-finding pass.
                {
                    const scores_sl = scores_h[0..sl_total];
                    var mx = running_max;
                    if (sink_data) |sd| mx = @max(mx, sd[h]);
                    const mx_splat: V8 = @splat(mx);
                    var sum_v: V8 = @splat(@as(f32, 0.0));
                    var si: usize = 0;
                    while (si + 8 <= sl_total) : (si += 8) {
                        const ev = @exp(@as(V8, scores_sl[si..][0..8].*) - mx_splat);
                        scores_sl[si..][0..8].* = ev;
                        sum_v += ev;
                    }
                    var sm = @reduce(.Add, sum_v);
                    while (si < sl_total) : (si += 1) {
                        scores_sl[si] = @exp(scores_sl[si] - mx);
                        sm += scores_sl[si];
                    }
                    if (sink_data) |sd| sm += @exp(sd[h] - mx);
                    const inv = 1.0 / sm;
                    const inv_v: V8 = @splat(inv);
                    si = 0;
                    while (si + 8 <= sl_total) : (si += 8) {
                        scores_sl[si..][0..8].* = @as(V8, scores_sl[si..][0..8].*) * inv_v;
                    }
                    while (si < sl_total) : (si += 1) scores_sl[si] *= inv;
                }
                // V accumulation — first-slot direct write avoids 2KB memset.
                const ao_h = self.attn_out[h * kd ..][0..kd];
                var first_written = false;
                for (0..pos + 1) |t| {
                    if (scores_h[t] < sparse_v_threshold) continue;
                    const v_ptr = kv_k_layer[t * kv_elem_bytes ..].ptr; // K=V shared
                    if (!first_written) {
                        kv_quant.kvScaledCopy(ao_h.ptr, scores_h[t], v_ptr, kd, self.kv_type);
                        first_written = true;
                    } else {
                        kv_quant.kvMulAccum(ao_h.ptr, scores_h[t], v_ptr, kd, self.kv_type);
                    }
                }
                if (!first_written) @memset(ao_h, 0.0);
                // V accumulation: compressed (always f32)
                for (0..n_attend_comp) |gi| {
                    const g = if (use_lid) self.lid_topk_ids[gi] else @as(u32, @intCast(gi));
                    const ck = self.csa_k[(li * comp_slots + g) * kd ..][0..kd];
                    const wv: V8 = @splat(scores_h[pos + 1 + gi]);
                    var i: usize = 0;
                    if (!first_written) {
                        // First accumulation: direct write
                        while (i + 8 <= kd) : (i += 8) {
                            ao_h[i..][0..8].* = @as(V8, ck[i..][0..8].*) * wv;
                        }
                        while (i < kd) : (i += 1) ao_h[i] = ck[i] * scores_h[pos + 1 + gi];
                        first_written = true;
                    } else {
                        while (i + 8 <= kd) : (i += 8) {
                            const cur: V8 = ao_h[i..][0..8].*;
                            ao_h[i..][0..8].* = @mulAdd(V8, @as(V8, ck[i..][0..8].*), wv, cur);
                        }
                        while (i < kd) : (i += 1) ao_h[i] += ck[i] * scores_h[pos + 1 + gi];
                    }
                }
            }
        } else {
            // Standard attention (no compressed KVs).
            // GPU SDPA supports f32, q8_0, turbo2/3/4. For other KV types,
            // use the CPU compressed attention path with n_comp_groups=0.
            const sdpa_ok = switch (self.kv_type) {
                .f32, .q8_0, .turbo2, .turbo3, .turbo4 => true,
                else => false,
            };
            if (sdpa_ok) {
                attn_ops.scaledDotProductAttention(
                    self.q_full.ptr,
                    kv_k_layer,
                    kv_k_layer, // K=V shared
                    self.kv_proj,
                    self.kv_proj,
                    self.attn_out.ptr,
                    self.scores_buf.ptr,
                    nh,
                    1,
                    kd,
                    pos,
                    scale,
                    self.be,
                    null,
                    0,
                    self.kv_type,
                    self.kv_type,
                );
            } else {
                // CPU fallback for KV types not supported by GPU SDPA
                const k_byte_off = kv_quant.kvByteOffset(self.kv_type, pos * kd);
                kv_quant.kvStore(kv_k_layer[k_byte_off..].ptr, self.kv_proj.ptr, kd, self.kv_type);
                const kv_elem_bytes = kv_quant.kvByteOffset(self.kv_type, kd);
                const V8 = @Vector(8, f32);
                for (0..nh) |h| {
                    const q_h = self.q_full[h * kd ..][0..kd];
                    const ao_h = self.attn_out[h * kd ..][0..kd];
                    const scores_h = self.scores_buf[h * self.score_stride ..];
                    var running_max: f32 = -std.math.inf(f32);
                    for (0..pos + 1) |t| {
                        const k_ptr = kv_k_layer[t * kv_elem_bytes ..].ptr;
                        const s = kv_quant.kvDot(q_h.ptr, k_ptr, kd, self.kv_type) * scale;
                        scores_h[t] = s;
                        running_max = @max(running_max, s);
                    }
                    const mx_splat: V8 = @splat(running_max);
                    var sum_v: V8 = @splat(@as(f32, 0.0));
                    var si: usize = 0;
                    while (si + 8 <= pos + 1) : (si += 8) {
                        const ev = @exp(@as(V8, scores_h[si..][0..8].*) - mx_splat);
                        scores_h[si..][0..8].* = ev;
                        sum_v += ev;
                    }
                    var sm = @reduce(.Add, sum_v);
                    while (si < pos + 1) : (si += 1) {
                        scores_h[si] = @exp(scores_h[si] - running_max);
                        sm += scores_h[si];
                    }
                    const inv = 1.0 / sm;
                    const inv_v: V8 = @splat(inv);
                    si = 0;
                    while (si + 8 <= pos + 1) : (si += 8) {
                        scores_h[si..][0..8].* = @as(V8, scores_h[si..][0..8].*) * inv_v;
                    }
                    while (si < pos + 1) : (si += 1) scores_h[si] *= inv;
                    var first_written = false;
                    for (0..pos + 1) |t| {
                        if (scores_h[t] < sparse_v_threshold) continue;
                        const v_ptr = kv_k_layer[t * kv_elem_bytes ..].ptr;
                        if (!first_written) {
                            kv_quant.kvScaledCopy(ao_h.ptr, scores_h[t], v_ptr, kd, self.kv_type);
                            first_written = true;
                        } else {
                            kv_quant.kvMulAccum(ao_h.ptr, scores_h[t], v_ptr, kd, self.kv_type);
                        }
                    }
                    if (!first_written) @memset(ao_h, 0.0);
                }
            }
        }

        // Apply inverse RoPE (derope) using the cached cos/sin table (same table, negate sin).
        for (0..nh) |h| {
            applyRopeInverseTable(self.attn_out[h * kd + nope ..][0..rd], rope_cos[0..nd], rope_sin[0..nd]);
        }

        // Output LoRA: grouped wo_a [n_in=4096 per group, n_out=o_lora_rank] × 8 groups
        const og: usize = self.o_groups;
        const olr: usize = self.o_lora_rank;
        const group_in: usize = nh * kd / og; // = 64*512/8 = 4096
        const wo_a = try self.layerTensorReq(li, "attn_output_a.weight");
        const row_bytes = backend_mod.weightBytes(wo_a.dtype, 1, group_in);
        // wo_a groups + wo_b in one GPU command buffer: lora_out feeds directly into wo_b
        for (0..og) |g| {
            const xp = self.attn_out.ptr + g * group_in;
            const wp = wo_a.data_ptr + g * olr * row_bytes;
            const yp = self.lora_out.ptr + g * olr;
            self.be.gemv(xp, .{ .data = wp, .dtype = wo_a.dtype }, yp, olr, group_in);
        }
        const wo_b = try self.layerTensorReq(li, "attn_output_b.weight");
        // Write wo_b output directly to hidden (avoids 16KB attn_result → hidden copy)
        self.be.gemv(self.lora_out.ptr, .{ .data = wo_b.data_ptr, .dtype = wo_b.dtype }, self.hidden.ptr, e, og * olr);
        self.be.sync(); // single sync covers all 9 GEMVs (8 wo_a + wo_b)
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
            // QK dot products for raw KV positions — track running max to fuse
            // with softmax (avoids a separate max-finding pass over all scores).
            // Prefetch next KV block while computing current dot product.
            var running_max: f32 = -std.math.inf(f32);
            const eb = ctx.kv_elem_bytes;
            for (0..ctx.pos + 1) |t| {
                const k_ptr = ctx.kv_k_layer[t * eb ..].ptr;
                // Prefetch next position's KV data into L1 cache (T0 = all cache levels)
                if (t + 1 <= ctx.pos) {
                    @prefetch(ctx.kv_k_layer[(t + 1) * eb ..].ptr, .{ .locality = 3 });
                }
                const s = kv_quant.kvDot(q_h.ptr, k_ptr, kd, ctx.kv_quant_type) * ctx.scale;
                scores_h[t] = s;
                running_max = @max(running_max, s);
            }
            // QK for compressed positions (f32 SIMD, also tracking max)
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
                scores_h[ctx.pos + 1 + gi] = s;
                running_max = @max(running_max, s);
            }
            // Softmax exp+sum — skip normalize pass, fold 1/sum into V weights.
            // Saves one full traversal of scores (O(sl_total) SIMD ops).
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
            // V accumulation with unnormalized exp weights — multiply by 1/sum at the end.
            // Sparse threshold adjusted: exp(s) < threshold * sum ≡ normalized_w < threshold.
            const sparse_threshold_unnorm = sparse_v_threshold * sm;
            const ao_h = ctx.attn_out[h * kd ..][0..kd];
            var first_written = false;
            for (0..ctx.pos + 1) |t| {
                if (scores_h[t] < sparse_threshold_unnorm) continue;
                const v_ptr = ctx.kv_v_layer[t * ctx.kv_elem_bytes ..].ptr;
                if (!first_written) {
                    kv_quant.kvScaledCopy(ao_h.ptr, scores_h[t], v_ptr, kd, ctx.kv_quant_type);
                    first_written = true;
                } else {
                    kv_quant.kvMulAccum(ao_h.ptr, scores_h[t], v_ptr, kd, ctx.kv_quant_type);
                }
            }
            if (!first_written) @memset(ao_h, 0.0);
            for (0..ctx.n_attend_comp) |gi| {
                const g = if (ctx.use_lid) ctx.lid_topk_ids[gi] else @as(u32, @intCast(gi));
                const ck = ctx.csa_k[(ctx.li * ctx.comp_slots + g) * kd ..][0..kd];
                const wv: V8 = @splat(scores_h[ctx.pos + 1 + gi]);
                var i: usize = 0;
                while (i + 8 <= kd) : (i += 8) {
                    const cur: V8 = ao_h[i..][0..8].*;
                    ao_h[i..][0..8].* = @mulAdd(V8, @as(V8, ck[i..][0..8].*), wv, cur);
                }
                while (i < kd) : (i += 1) ao_h[i] += ck[i] * scores_h[ctx.pos + 1 + gi];
            }
            // Final normalize: out was accumulated with unnormalized exp weights.
            // Multiply by 1/sum to get correct softmax-weighted output.
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
    /// synced — this function only does CPU scoring + top-k selection.
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
                self.be.gemv(self.q_compressed.ptr, .{ .data = wiq.data_ptr, .dtype = wiq.dtype }, self.lid_query.ptr, inh * ihd, ql);
            } else {
                @memset(self.lid_topk_ids[0..@min(itk, n_groups)], 0);
                return;
            }
            // Step 2: Project hidden → per-head weights [inh] via W^w
            if (self.layerTensor(li, "attn_indexer_proj.weight")) |ww| {
                self.be.gemv(self.hidden.ptr, .{ .data = ww.data_ptr, .dtype = ww.dtype }, self.lid_head_w.ptr, inh, self.n_embd);
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
        const nk: usize = self.n_expert_used;
        const ne: usize = self.n_experts;

        // Pre-norm: GPU only (no sync — expert GEMVs and routing GEMV also GPU)
        const nw = try self.layerTensorReq(li, "ffn_norm.weight");
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);

        // Route
        var top_ids: [8]usize = undefined;
        var top_scores: [8]f32 = undefined;
        var n_active: usize = 0;
        var top_weights: [8]f32 = undefined;

        if (li < self.hash_layer_count) {
            // Hash routing: expert selection is by hash lookup (CPU, no GPU sync needed).
            // Gate GEMV deferred — batched with expert GEMVs, read after final sync.
            const gi = try self.layerTensorReq(li, "ffn_gate_inp.weight");
            self.be.gemv(self.hidden2.ptr, .{ .data = gi.data_ptr, .dtype = gi.dtype }, self.router_logits.ptr, ne, e);
            // NO sync here — gate_inp GEMV batched with expert GEMVs below

            // Hash lookup: determines which experts are selected (CPU-only, no GPU data needed)
            const t2e = try self.layerTensorReq(li, "ffn_gate_tid2eid.weight");
            const n_slots: usize = @intCast(t2e.dims[0]);
            const vocab: usize = @intCast(t2e.dims[1]);
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
            self.be.gemv(self.hidden2.ptr, .{ .data = gi.data_ptr, .dtype = gi.dtype }, self.router_logits.ptr, ne, e);
            self.be.sync(); // CPU reads router_logits

            // Compute probs = sqrt_softplus(logits) — SIMD vectorized (3 transcendentals × 256)
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

            // Weights from unbiased probs, normalized
            var wsum: f32 = 0.0;
            for (0..n_active) |j| {
                top_weights[j] = probs[top_ids[j]]; // unbiased prob for selected expert
                wsum += top_weights[j];
            }
            if (wsum > 0.0) {
                const inv = self.expert_weights_scale / wsum;
                for (0..n_active) |j| top_weights[j] *= inv;
            }
        }

        // Batched FFN: gate+up+activation per expert, then down GEMVs.
        // Try fused kernel (gate+up+clampedSiluMul in 1 dispatch per expert)
        // when Q2_K weights on Metal. Falls back to 3-phase unfused path.
        var n_scratch: usize = 0;
        var slot_weights: [9]f32 = [_]f32{0.0} ** 9;

        // Detect fused-capable backend at comptime — avoids runtime dispatch overhead.
        const use_fused = blk: {
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
                    self.be.gemv(self.hidden2.ptr, .{ .data = gt.data_ptr, .dtype = gt.dtype }, self.ff_gate_scratch.ptr + n_scratch * ff, ff, e);
                    self.be.gemv(self.hidden2.ptr, .{ .data = ut.data_ptr, .dtype = ut.dtype }, self.ff_up_scratch.ptr + n_scratch * ff, ff, e);
                }
                slot_weights[n_scratch] = 1.0;
                n_scratch += 1;
            }
        }
        const shexp_slots = n_scratch;
        var fused_experts = use_fused; // track if experts used fused path

        var de_ptrs: [9][*]const u8 = undefined;
        var de_dtype: DType = .f32;
        if (self.layerTensor(li, "ffn_gate_exps.weight")) |ge| {
            const ue = self.layerTensor(li, "ffn_up_exps.weight") orelse return error.MissingTensor;
            const de = self.layerTensor(li, "ffn_down_exps.weight") orelse return error.MissingTensor;
            de_dtype = de.dtype;
            const gs = ds4ExpertStride(ge);
            const us = ds4ExpertStride(ue);
            const ds = ds4ExpertStride(de);
            if (use_fused and (ge.dtype == .q2_k or ge.dtype == .mxfp4)) {
                // Fused path: 1 dispatch per expert (gate+up+clampedSiluMul)
                switch (self.be) {
                    inline else => |be| {
                        for (0..n_active) |j| {
                            const eid = top_ids[j];
                            if (ge.dtype == .mxfp4) {
                                if (comptime @hasDecl(@TypeOf(be.*), "fusedFfnGateUpClampedSiluMxfp4"))
                                    be.fusedFfnGateUpClampedSiluMxfp4(self.hidden2.ptr, ge.data_ptr + eid * gs, ue.data_ptr + eid * us, self.ff_gate_scratch.ptr + n_scratch * ff, ff, e);
                            } else {
                                if (comptime @hasDecl(@TypeOf(be.*), "fusedFfnGateUpClampedSiluQ2K"))
                                    be.fusedFfnGateUpClampedSiluQ2K(self.hidden2.ptr, ge.data_ptr + eid * gs, ue.data_ptr + eid * us, self.ff_gate_scratch.ptr + n_scratch * ff, ff, e);
                            }
                            de_ptrs[n_scratch] = de.data_ptr + eid * ds;
                            slot_weights[n_scratch] = top_weights[j];
                            n_scratch += 1;
                        }
                    },
                }
            } else {
                fused_experts = false;
                for (0..n_active) |j| {
                    const eid = top_ids[j];
                    self.be.gemv(self.hidden2.ptr, .{ .data = ge.data_ptr + eid * gs, .dtype = ge.dtype }, self.ff_gate_scratch.ptr + n_scratch * ff, ff, e);
                    self.be.gemv(self.hidden2.ptr, .{ .data = ue.data_ptr + eid * us, .dtype = ue.dtype }, self.ff_up_scratch.ptr + n_scratch * ff, ff, e);
                    de_ptrs[n_scratch] = de.data_ptr + eid * ds;
                    slot_weights[n_scratch] = top_weights[j];
                    n_scratch += 1;
                }
            }
        }

        // Phase 2: clampedSiluMul — skip when fused path already applied activation.
        if (!fused_experts and n_scratch > 0) {
            self.be.clampedSiluMul(self.ff_gate_scratch.ptr, self.ff_up_scratch.ptr, self.ff_gate_scratch.ptr, n_scratch * ff);
        }

        // Phase 3: all down GEMVs into expert_scratch (same cmd buffer as siluMul)
        if (shexp_slots > 0) {
            if (self.layerTensor(li, "ffn_down_shexp.weight")) |dt| {
                self.be.gemv(self.ff_gate_scratch.ptr, .{ .data = dt.data_ptr, .dtype = dt.dtype }, self.expert_scratch.ptr, e, ff);
            }
        }
        for (shexp_slots..n_scratch) |slot| {
            self.be.gemv(self.ff_gate_scratch.ptr + slot * ff, .{ .data = de_ptrs[slot], .dtype = de_dtype }, self.expert_scratch.ptr + slot * e, e, ff);
        }

        self.be.sync(); // all down GEMVs complete (+ gate_inp for hash layers)

        // Deferred hash-layer weight computation: gate logits now available after sync.
        if (li < self.hash_layer_count) {
            var wsum: f32 = 0.0;
            for (0..n_active) |j| {
                top_weights[j] = sqrtSoftplus(self.router_logits[top_ids[j]]);
                wsum += top_weights[j];
            }
            if (wsum > 0.0) {
                const inv = self.expert_weights_scale / wsum;
                for (0..n_active) |j| top_weights[j] *= inv;
            }
            // Update slot_weights for expert slots (shared expert weight=1.0 already set)
            for (shexp_slots..n_scratch) |slot| {
                slot_weights[slot] = top_weights[slot - shexp_slots];
            }
        }

        // CPU: SIMD weighted accumulation directly into hidden.
        // First slot: direct scaled write (skip memset + mulAdd into zeros).
        // Remaining slots: fused multiply-add.
        const V8 = @Vector(8, f32);
        if (n_scratch > 0) {
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

        const e = self.n_embd;
        const nl = self.n_layers;

        // Embed → broadcast to all n_hc HC streams.
        // CPU-side dequant avoids GPU dispatch + sync for single-row read.
        const emb = try self.getTensorReq("token_embd.weight");
        const row_bytes = backend_mod.weightBytes(emb.dtype, 1, e);
        quant_ops.dequantToF32(self.hc_state[0..e], emb.data_ptr + token_id * row_bytes, emb.dtype, e);
        for (1..n_hc) |s| @memcpy(self.hc_state[s * e ..][0..e], self.hc_state[0..e]);

        for (0..nl) |li| {
            if (self.cancelled.load(.monotonic)) return error.Cancelled;

            // Attn: HC pre → attn → HC post
            const af = try self.layerTensorReq(li, "hc_attn_fn.weight");
            const ab = try self.layerTensorReq(li, "hc_attn_base.weight");
            const as_ = try self.layerTensorReq(li, "hc_attn_scale.weight");
            self.hcPre(af, ab, as_);
            try self.attentionLayer(li);
            self.hcPost();

            // FFN: HC pre → ffn → HC post
            const ff = try self.layerTensorReq(li, "hc_ffn_fn.weight");
            const fb = try self.layerTensorReq(li, "hc_ffn_base.weight");
            const fs = try self.layerTensorReq(li, "hc_ffn_scale.weight");
            self.hcPre(ff, fb, fs);
            try self.ffnLayer(li, token_id);
            self.hcPost();
        }

        // Output HC head
        const hh_fn = try self.getTensorReq("output_hc_fn.weight");
        const hh_base = try self.getTensorReq("output_hc_base.weight");
        const hh_scale = try self.getTensorReq("output_hc_scale.weight");
        self.hcHead(hh_fn, hh_base, hh_scale);

        // Final norm + LM head — single GPU command buffer, single sync
        const norm_w = try self.getTensorReq("output_norm.weight");
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(norm_w, e), self.hidden.ptr, e, self.rms_eps);
        const lm = try self.getTensorReq("output.weight");
        self.be.gemv(self.hidden.ptr, .{ .data = lm.data_ptr, .dtype = lm.dtype }, self.logits_buf.ptr, self.vocab_size, e);
        self.be.sync();

        self.kv_seq_len += 1;
        return math_ops.argmax(self.logits_buf);
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
    ///    on layers 0-2) produces different expert sets per token — same as GLM-4
    ///    and GPT-OSS, but combined with HC makes the bookkeeping much harder.
    ///
    /// 3. **CSA/HCA compressors have per-position state:** Compressed KV blocks
    ///    accumulate over positions with softmax scoring. The write targets are
    ///    sequential and position-dependent.
    ///
    /// 4. **Grouped output LoRA:** 8-group × 1024-rank attention output projection
    ///    is non-standard and would need its own batched path.
    ///
    /// The pf_* buffers are allocated for forward compatibility — a future
    /// implementation can batch the MLA attention within each layer (as GLM-4 does)
    /// while keeping HC, MoE, and compressor passes per-token.
    pub fn prefill(self: *Ds4Model, token_ids: []const u32) !u32 {
        // Sequential fallback — see doc comment above for rationale.
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
    /// Return the current hidden state after the last forward pass.
    pub fn getHidden(self: *const Ds4Model) []const f32 {
        return self.hidden;
    }
    /// Return physical block IDs for the paged KV cache. DeepSeek V4 uses a flat
    /// KV layout (not paged), so this always returns an empty slice.
    pub fn getBlockTable(_: *const Ds4Model) []const u32 {
        return &.{};
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
    // Sort initial k by score descending (insertion sort — k is small)
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

/// Per-expert stride for ds4 expert tensors: dims=[input_dim, ff_dim, n_experts].
fn ds4ExpertStride(t: TensorInfo) usize {
    if (t.n_dims < 3) @panic("ds4ExpertStride: expected >= 3D tensor for expert weights");
    const elems = @as(usize, @intCast(t.dims[0])) * @as(usize, @intCast(t.dims[1]));
    return backend_mod.weightBytes(t.dtype, 1, elems);
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
