//! DeepSeek V4 Flash 0731 inference model.
//! Uses GGUF-style tensor names (blk.N.*).
//! Architecture: 4-stream hyper connections, modified MLA (K=V compressed,
//! no separate V projection), hash routing (layers 0-2), sqrt_softplus routing
//! (layers 3+), grouped output LoRA (8 groups × 1024 rank).
//! KV compressor / indexer attention not implemented (short-context only).

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
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const DType = format_mod.DType;
const Model = model_mod.Model;
const TensorData = backend_mod.TensorData;

const name_buf_size: usize = model_mod.tensor_name_buf_size;
const n_hc: usize = 4;
const hc_mix_dim: usize = (2 + n_hc) * n_hc; // = 24
const hc_sinkhorn_iters: usize = 20;
const hc_eps: f32 = 1e-6;
const max_norm_entries: usize = 512;

const NormCacheEntry = struct { key: usize, data: []f32 };

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

    // HC buffers
    hc_state: []f32 = &.{},   // [n_hc * n_embd]
    new_hc: []f32 = &.{},     // [n_hc * n_embd] temp
    hc_mixes: []f32 = &.{},   // [hc_mix_dim = 24]
    hc_pre_w: []f32 = &.{},   // [n_hc]
    hc_post_w: []f32 = &.{},  // [n_hc]
    hc_comb: []f32 = &.{},    // [n_hc * n_hc]

    // Attention buffers
    hidden: []f32 = &.{},       // [n_embd]
    hidden2: []f32 = &.{},      // [n_embd]
    flat_norm: []f32 = &.{},    // [n_hc * n_embd] for HC rms norm
    q_compressed: []f32 = &.{}, // [q_lora_rank]
    q_full: []f32 = &.{},       // [n_head * kv_lora_rank]
    kv_proj: []f32 = &.{},      // [kv_lora_rank]
    scores_buf: []f32 = &.{},   // [max_seq_len]
    attn_out: []f32 = &.{},     // [n_head * kv_lora_rank]
    lora_out: []f32 = &.{},     // [o_groups * o_lora_rank]
    attn_result: []f32 = &.{},  // [n_embd]

    // FFN buffers
    ff_gate: []f32 = &.{},          // [ff_exp]
    ff_up: []f32 = &.{},            // [ff_exp]
    ff_down: []f32 = &.{},          // [n_embd]
    expert_accum: []f32 = &.{},     // [n_embd]
    expert_scratch: []f32 = &.{},    // [max_total_experts * n_embd] for batched down GEMVs
    ff_gate_scratch: []f32 = &.{},  // [max_total_experts * ff_exp] gate outputs pre-siluMul
    ff_up_scratch: []f32 = &.{},    // [max_total_experts * ff_exp] up outputs pre-siluMul
    router_logits: []f32 = &.{},    // [n_experts]
    logits_buf: []f32 = &.{},       // [vocab_size]

    // KV cache as f16 bytes for GPU SDPA: [n_layers * ctx * kv_lora_rank * 2]
    kv_k_bytes: []u8 = &.{},
    kv_v_bytes: []u8 = &.{},

    // CSA compressor state for ratio=4 layers.
    // Per-token projected KV and score [n_layers * ctx * comp_dim], comp_dim = 2*kv_lora_rank
    // Compressed KV after softmax+sum: [n_layers * (ctx/4+1) * kv_lora_rank]
    csa_comp_kv: []f32 = &.{},
    csa_comp_score: []f32 = &.{},
    csa_k: []f32 = &.{},
    csa_score_scratch: []f32 = &.{},  // [2 * kv_lora_rank] temp scratch

    // Norm weight cache (dequantized to f32)
    norm_cache: [max_norm_entries]NormCacheEntry = undefined,
    norm_cache_len: usize = 0,
    name_buf: [name_buf_size]u8 = undefined,

    pub fn init(
        allocator: Allocator,
        f: Format,
        be: Backend,
        ctx_size: u32,
        _: anytype, // kv_type_k (ignored, using f32)
        _: anytype, // kv_type_v (ignored, using f32)
        _: anytype, // tiered_cache (not supported yet)
    ) !Ds4Model {
        var self = Ds4Model{ .fmt = f, .be = be, .allocator = allocator };

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

        self.hc_state = try allocator.alloc(f32, n_hc * e);
        self.new_hc = try allocator.alloc(f32, n_hc * e);
        self.hc_mixes = try allocator.alloc(f32, hc_mix_dim);
        self.hc_pre_w = try allocator.alloc(f32, n_hc);
        self.hc_post_w = try allocator.alloc(f32, n_hc);
        self.hc_comb = try allocator.alloc(f32, n_hc * n_hc);
        self.hidden = try allocator.alloc(f32, e);
        self.hidden2 = try allocator.alloc(f32, e);
        self.flat_norm = try allocator.alloc(f32, n_hc * e);
        self.q_compressed = try allocator.alloc(f32, ql);
        self.q_full = try allocator.alloc(f32, nh * kd);
        self.kv_proj = try allocator.alloc(f32, kd);
        self.scores_buf = try allocator.alloc(f32, ctx);
        self.attn_out = try allocator.alloc(f32, nh * kd);
        self.lora_out = try allocator.alloc(f32, og * olr);
        self.attn_result = try allocator.alloc(f32, e);
        self.ff_gate = try allocator.alloc(f32, ff);
        self.ff_up = try allocator.alloc(f32, ff);
        self.ff_down = try allocator.alloc(f32, e);
        self.expert_accum = try allocator.alloc(f32, e);
        // Scratch for batched expert down GEMVs (max_experts = n_expert_used + n_expert_shared)
        const max_experts: usize = @as(usize, self.n_expert_used) + @as(usize, self.n_expert_shared);
        self.expert_scratch = try allocator.alloc(f32, max_experts * e);
        self.ff_gate_scratch = try allocator.alloc(f32, max_experts * ff);
        self.ff_up_scratch = try allocator.alloc(f32, max_experts * ff);
        self.router_logits = try allocator.alloc(f32, self.n_experts);
        self.logits_buf = try allocator.alloc(f32, self.vocab_size);

        // KV cache bytes: each layer stores ctx × nkv=1 × kd elements in kv_type format
        const kv_bytes_per_layer = kv_quant.kvByteOffset(kv_type, ctx * kd);
        self.kv_k_bytes = try allocator.alloc(u8, nl * kv_bytes_per_layer);
        self.kv_v_bytes = try allocator.alloc(u8, nl * kv_bytes_per_layer);

        // CSA compressor: comp_dim = 2 * kv_lora_rank (coff=2 for ratio=4 layers)
        const comp_dim: usize = 2 * kd;
        const csa_groups = (ctx + 3) / 4 + 1;  // max compressed groups per layer
        self.csa_comp_kv = try allocator.alloc(f32, nl * ctx * comp_dim);
        self.csa_comp_score = try allocator.alloc(f32, nl * ctx * comp_dim);
        self.csa_k = try allocator.alloc(f32, nl * csa_groups * kd);
        self.csa_score_scratch = try allocator.alloc(f32, comp_dim);

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
        }
    }

    pub fn deinit(self: *Ds4Model) void {
        for (self.norm_cache[0..self.norm_cache_len]) |e| self.allocator.free(e.data);
        const a = self.allocator;
        inline for (.{
            &self.hc_state, &self.new_hc, &self.hc_mixes, &self.hc_pre_w, &self.hc_post_w,
            &self.hc_comb, &self.hidden, &self.hidden2, &self.flat_norm, &self.q_compressed,
            &self.q_full, &self.kv_proj, &self.scores_buf, &self.attn_out, &self.lora_out,
            &self.attn_result, &self.ff_gate, &self.ff_up, &self.ff_down, &self.expert_accum,
            &self.expert_scratch, &self.ff_gate_scratch, &self.ff_up_scratch,
            &self.router_logits, &self.logits_buf, &self.kv_k_bytes, &self.kv_v_bytes,
            &self.csa_comp_kv, &self.csa_comp_score, &self.csa_k, &self.csa_score_scratch,
        }) |buf| a.free(buf.*);
    }

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

    const kv_type: KvQuantType = .f32;

    fn kvLayerBytes(self: *Ds4Model) usize {
        return kv_quant.kvByteOffset(kv_type, self.max_seq_len * self.kv_lora_rank);
    }

    fn kvKLayer(self: *Ds4Model, li: usize) []u8 {
        const layer_bytes = self.kvLayerBytes();
        return self.kv_k_bytes[li * layer_bytes ..][0..layer_bytes];
    }

    fn kvVLayer(self: *Ds4Model, li: usize) []u8 {
        const layer_bytes = self.kvLayerBytes();
        return self.kv_v_bytes[li * layer_bytes ..][0..layer_bytes];
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

        // RMS-norm the flat HC state (no learned weight)
        @memcpy(self.flat_norm, self.hc_state);
        plainRmsNorm(self.flat_norm, self.rms_eps);

        // mixes[24] = hc_fn @ flat_norm — CPU GEMV avoids GPU dispatch overhead for 24-output case
        if (hc_fn.dtype == .q8_0) {
            cpuGemvQ8_0(hc_fn.data_ptr, self.flat_norm, self.hc_mixes, flat_size);
        } else {
            self.be.gemv(self.flat_norm.ptr, .{ .data = hc_fn.data_ptr, .dtype = hc_fn.dtype }, self.hc_mixes.ptr, hc_mix_dim, flat_size);
            self.be.sync();
        }

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
        @memcpy(self.hc_state, self.new_hc);
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
        @memcpy(self.flat_norm, self.hc_state);
        plainRmsNorm(self.flat_norm, self.rms_eps);
        if (hc_fn.dtype == .q8_0) {
            cpuGemvQ8_0(hc_fn.data_ptr, self.flat_norm, self.hc_pre_w, n_hc * e);
        } else {
            self.be.gemv(self.flat_norm.ptr, .{ .data = hc_fn.data_ptr, .dtype = hc_fn.dtype }, self.hc_pre_w.ptr, n_hc, n_hc * e);
            self.be.sync();
        }

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
        const comp_dim: usize = 2 * kd; // CSA compressor output dim (coff=2 for ratio=4)
        const csa_ratio_4: usize = 4; // CSA compression ratio
        // Use compressed rope freq for layers with ratio≠0 (most layers in DS4)
        const rope_freq = if (self.compress_ratios[li] != 0) self.compress_rope_freq else self.rope_freq;

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
        // CSA: batched with Q+KV in same cmd buffer. HCA: computed after sync (separate).
        const layer_stride = self.max_seq_len * comp_dim;
        var comp_kv_pos: []f32 = &.{};
        var comp_score_pos: []f32 = &.{};
        var actual_comp_dim: usize = 0;
        const ratio = self.compress_ratios[li];
        if (ratio == csa_ratio_4) {
            // CSA: batch with Q+KV (no extra sync)
            comp_kv_pos = self.csa_comp_kv[li * layer_stride + pos * comp_dim ..][0..comp_dim];
            comp_score_pos = self.csa_comp_score[li * layer_stride + pos * comp_dim ..][0..comp_dim];
            actual_comp_dim = comp_dim;
            if (self.layerTensor(li, "attn_compressor_kv.weight")) |wkv| {
                const kwgate = self.layerTensor(li, "attn_compressor_gate.weight") orelse @panic("CSA gate missing");
                self.be.gemv(self.hidden2.ptr, .{ .data = wkv.data_ptr, .dtype = wkv.dtype }, comp_kv_pos.ptr, comp_dim, e);
                self.be.gemv(self.hidden2.ptr, .{ .data = kwgate.data_ptr, .dtype = kwgate.dtype }, comp_score_pos.ptr, comp_dim, e);
            }
        }

        self.be.sync(); // sync: CPU reads q_full + kv_proj + CSA projections

        if (ratio != 0 and ratio != csa_ratio_4) {
            // HCA (ratio=128): separate GPU dispatch after main sync to avoid command buffer overflow.
            // comp_dim for HCA is 512 (coff=1 vs coff=2 for CSA).
            if (self.layerTensor(li, "attn_compressor_kv.weight")) |wkv| {
                const kwgate = self.layerTensor(li, "attn_compressor_gate.weight") orelse return error.MissingTensor;
                actual_comp_dim = @intCast(wkv.dims[1]); // 512 for HCA
                comp_kv_pos = self.csa_comp_kv[li * layer_stride + pos * comp_dim ..][0..actual_comp_dim];
                comp_score_pos = self.csa_comp_score[li * layer_stride + pos * comp_dim ..][0..actual_comp_dim];
                self.be.gemv(self.hidden2.ptr, .{ .data = wkv.data_ptr, .dtype = wkv.dtype }, comp_kv_pos.ptr, actual_comp_dim, e);
                self.be.gemv(self.hidden2.ptr, .{ .data = kwgate.data_ptr, .dtype = kwgate.dtype }, comp_score_pos.ptr, actual_comp_dim, e);
                self.be.sync(); // HCA projections need separate sync
            }
        }

        // Pre-compute RoPE cos/sin table once for this (pos, rope_freq) — shared by all heads.
        const nd = rd / 2;
        var rope_cos: [32]f32 = undefined;
        var rope_sin: [32]f32 = undefined;
        for (0..nd) |i| {
            const freq = std.math.pow(f32, rope_freq, -@as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(rd)));
            const theta = @as(f32, @floatFromInt(pos)) * freq;
            rope_cos[i] = @cos(theta);
            rope_sin[i] = @sin(theta);
        }

        // CPU: per-head Q RMS norm + Q RoPE (using pre-computed table)
        for (0..nh) |h| {
            const q_head = self.q_full[h * kd ..][0..kd];
            plainRmsNorm(q_head, self.rms_eps);
            applyRopeTable(q_head[nope..][0..rd], rope_cos[0..nd], rope_sin[0..nd]);
        }
        // CPU: KV RoPE
        applyRopeTable(self.kv_proj[nope..][0..rd], rope_cos[0..nd], rope_sin[0..nd]);

        // CSA compressor (ratio=4): GPU projections already done above; do APE + compression here.
        const csa_ratio: usize = if (ratio != 0) ratio else csa_ratio_4;
        if (ratio != 0 and actual_comp_dim > 0) {
            {
                const ape = self.layerTensor(li, "attn_compressor_ape.weight");

                // Add APE (absolute position encoding) for position within group
                if (ape) |a| {
                    // ape dims=[comp_dim, ratio] in Q8_0; column pos%ratio is the encoding
                    const group_pos = pos % csa_ratio;
                    // APE stored column-major: ape[:, group_pos] at offset group_pos * comp_dim elements
                    const ape_col: [*]const u8 = a.data_ptr + group_pos * backend_mod.weightBytes(a.dtype, 1, comp_dim);
                    // Dequantize APE column
                    var ape_f32: [1024]f32 = undefined;
                    quant_ops.dequantToF32(&ape_f32, ape_col, a.dtype, comp_dim);
                    for (comp_score_pos, ape_f32) |*s, av| s.* += av;
                }

                // Compress complete group of 4 tokens
                if ((pos + 1) % csa_ratio == 0 and pos >= csa_ratio - 1) {
                    const group_start = pos + 1 - csa_ratio;
                    const group_idx = (pos + 1) / csa_ratio - 1;
                    const csa_slots = (self.max_seq_len / csa_ratio_4 + 1); // fixed stride for all layers

                    // Softmax over the 4-token dimension for each of comp_dim dimensions independently
                    // Then weighted sum → compressed KV [comp_dim]
                    var compressed: [1024]f32 = undefined;
                    for (0..comp_dim) |d| {
                        // Collect scores for this dimension across 4 tokens
                        var sc: [4]f32 = undefined;
                        for (0..csa_ratio) |t| {
                            sc[t] = self.csa_comp_score[(li * self.max_seq_len + group_start + t) * comp_dim + d];
                        }
                        // Softmax
                        var mx = sc[0];
                        for (sc[1..]) |sv| if (sv > mx) { mx = sv; };
                        var sm: f32 = 0;
                        for (&sc) |*sv| { sv.* = @exp(sv.* - mx); sm += sv.*; }
                        const inv = 1.0 / sm;
                        // Weighted sum of KV values
                        var acc: f32 = 0;
                        for (0..csa_ratio) |t| {
                            acc += sc[t] * inv * self.csa_comp_kv[(li * self.max_seq_len + group_start + t) * comp_dim + d];
                        }
                        compressed[d] = acc;
                    }

                    // Apply RMS norm + scale to full [0..kd=512] compressed vector
                    // Then split into nope [0..448] and rope [448..512] for RoPE
                    if (self.layerTensor(li, "attn_compressor_norm.weight")) |cn| {
                        const norm_w = self.normAsF32(cn, kd);
                        var comp_first = compressed[0..kd];
                        plainRmsNorm(comp_first, self.rms_eps);
                        for (0..kd) |i| comp_first[i] *= norm_w[i];
                    }
                    const comp_rope = compressed[nope..kd];

                    // RoPE on compressed rope portion using group position
                    // Use compressed_rope_freq since this is a ratio≠0 layer
                    const comp_pos = group_start; // position of the group (first token)
                    var cg_cos: [32]f32 = undefined;
                    var cg_sin: [32]f32 = undefined;
                    for (0..nd) |i| {
                        const freq = std.math.pow(f32, self.compress_rope_freq, -@as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(rd)));
                        const theta = @as(f32, @floatFromInt(comp_pos)) * freq;
                        cg_cos[i] = @cos(theta);
                        cg_sin[i] = @sin(theta);
                    }
                    applyRopeTable(comp_rope[0..rd], cg_cos[0..nd], cg_sin[0..nd]);

                    // Store final compressed KV [kd=512] in CSA cache
                    const csa_off = (li * csa_slots + group_idx) * kd;
                    @memcpy(self.csa_k[csa_off..][0..kd], compressed[0..kd]);
                }
            }
        }

        // SDPA: head_dim=512 exceeds Metal GPU limit (256), force CPU SIMD path via window.
        // K=V in DS4 MLA (single compressed head, GQA 64:1).
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));
        const kv_k_layer = self.kvKLayer(li);
        const kv_v_layer = self.kvVLayer(li);
        const n_csa_groups: usize = if (self.compress_ratios[li] == csa_ratio_4) (pos + 1) / csa_ratio_4 else 0;

        if (n_csa_groups > 0) {
            // CSA attention: append current KV to cache, then attend raw + compressed.
            // First: store kv_proj in raw cache
            const k_byte_off = kv_quant.kvByteOffset(kv_type, pos * kd);
            kv_quant.kvStore(kv_k_layer[k_byte_off..].ptr, self.kv_proj.ptr, kd, kv_type);
            kv_quant.kvStore(kv_v_layer[k_byte_off..].ptr, self.kv_proj.ptr, kd, kv_type);

            // Manual attention over raw (0..pos+1) + compressed (n_csa_groups) positions
            const csa_slots = (self.max_seq_len / csa_ratio_4 + 1); // fixed stride for all layers
            const sl_total = pos + 1 + n_csa_groups;

            @memset(self.attn_out, 0.0);
            const V8 = @Vector(8, f32);
            for (0..nh) |h| {
                const q_h = self.q_full[h * kd ..][0..kd];
                const q_v8: []const f32 = q_h;
                const f32_k: [*]const f32 = @ptrCast(@alignCast(kv_k_layer.ptr));
                // SIMD QK dot products for raw positions
                for (0..pos + 1) |t| {
                    const k_t = f32_k[t * kd ..][0..kd];
                    var acc: V8 = @splat(0.0);
                    var i: usize = 0;
                    while (i + 8 <= kd) : (i += 8) {
                        acc += @as(V8, q_v8[i..][0..8].*) * @as(V8, k_t[i..][0..8].*);
                    }
                    var dot = @reduce(.Add, acc);
                    while (i < kd) : (i += 1) dot += q_h[i] * k_t[i];
                    self.scores_buf[t] = dot * scale;
                }
                // SIMD QK for compressed positions
                for (0..n_csa_groups) |g| {
                    const ck = self.csa_k[(li * csa_slots + g) * kd ..][0..kd];
                    var acc: V8 = @splat(0.0);
                    var i: usize = 0;
                    while (i + 8 <= kd) : (i += 8) {
                        acc += @as(V8, q_v8[i..][0..8].*) * @as(V8, ck[i..][0..8].*);
                    }
                    var dot = @reduce(.Add, acc);
                    while (i < kd) : (i += 1) dot += q_h[i] * ck[i];
                    self.scores_buf[pos + 1 + g] = dot * scale;
                }
                // Softmax over all positions
                {
                    var mx = self.scores_buf[0];
                    for (self.scores_buf[1..sl_total]) |v| if (v > mx) { mx = v; };
                    if (n_csa_groups > 0) {
                        // sink virtual token included in denominator
                        const sinks_t2 = self.layerTensor(li, "attn_sinks.weight");
                        if (sinks_t2) |st| {
                            const sd: [*]const f32 = @ptrCast(@alignCast(st.data_ptr));
                            if (sd[h] > mx) mx = sd[h];
                        }
                    }
                    var sm: f32 = 0;
                    for (self.scores_buf[0..sl_total]) |*v| { v.* = @exp(v.* - mx); sm += v.*; }
                    if (self.layerTensor(li, "attn_sinks.weight")) |st| {
                        const sd: [*]const f32 = @ptrCast(@alignCast(st.data_ptr));
                        sm += @exp(sd[h] - mx);
                    }
                    const inv = 1.0 / sm;
                    for (self.scores_buf[0..sl_total]) |*v| v.* *= inv;
                }
                // V accumulation: raw
                const ao_h = self.attn_out[h * kd ..][0..kd];
                const f32_v: [*]const f32 = @ptrCast(@alignCast(kv_v_layer.ptr));
                // SIMD V accumulation: raw
                for (0..pos + 1) |t| {
                    const v_t = f32_v[t * kd ..][0..kd];
                    const wv: V8 = @splat(self.scores_buf[t]);
                    var i: usize = 0;
                    while (i + 8 <= kd) : (i += 8) {
                        const cur: V8 = ao_h[i..][0..8].*;
                        ao_h[i..][0..8].* = @mulAdd(V8, @as(V8, v_t[i..][0..8].*), wv, cur);
                    }
                    while (i < kd) : (i += 1) ao_h[i] += v_t[i] * self.scores_buf[t];
                }
                // SIMD V accumulation: compressed
                for (0..n_csa_groups) |g| {
                    const ck = self.csa_k[(li * csa_slots + g) * kd ..][0..kd];
                    const wv: V8 = @splat(self.scores_buf[pos + 1 + g]);
                    var i: usize = 0;
                    while (i + 8 <= kd) : (i += 8) {
                        const cur: V8 = ao_h[i..][0..8].*;
                        ao_h[i..][0..8].* = @mulAdd(V8, @as(V8, ck[i..][0..8].*), wv, cur);
                    }
                    while (i < kd) : (i += 1) ao_h[i] += ck[i] * self.scores_buf[pos + 1 + g];
                }
            }
        } else {
            // Standard attention (no compressed KVs)
            attn_ops.scaledDotProductAttention(
                self.q_full.ptr,
                kv_k_layer,
                kv_v_layer,
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
                .{ .start = 0, .len = pos + 1 },
                0,
                kv_type,
                kv_type,
            );
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
        self.be.gemv(self.lora_out.ptr, .{ .data = wo_b.data_ptr, .dtype = wo_b.dtype }, self.attn_result.ptr, e, og * olr);
        self.be.sync(); // single sync covers all 9 GEMVs (8 wo_a + wo_b)

        @memcpy(self.hidden, self.attn_result);
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
            // Hash routing: gate GEMV still runs (weights come from logits), only selection is hash-based.
            // No exp_probs_b for hash layers.
            const gi = try self.layerTensorReq(li, "ffn_gate_inp.weight");
            self.be.gemv(self.hidden2.ptr, .{ .data = gi.data_ptr, .dtype = gi.dtype }, self.router_logits.ptr, ne, e);
            self.be.sync();

            // Hash lookup: determines which experts are selected
            const t2e = try self.layerTensorReq(li, "ffn_gate_tid2eid.weight");
            const n_slots: usize = @intCast(t2e.dims[0]);
            const vocab: usize = @intCast(t2e.dims[1]);
            const data: [*]const i32 = @ptrCast(@alignCast(t2e.data_ptr));
            const safe_tid: usize = @min(@as(usize, token_id), vocab - 1);
            for (0..nk) |j| {
                top_ids[j] = @intCast(data[safe_tid * n_slots + j]);
            }
            n_active = nk;

            // Weights from gate logits at the selected expert positions (no bias for hash layers)
            var wsum: f32 = 0.0;
            for (0..n_active) |j| {
                top_weights[j] = sqrtSoftplus(self.router_logits[top_ids[j]]);
                wsum += top_weights[j];
            }
            if (wsum > 0.0) {
                const inv = self.expert_weights_scale / wsum;
                for (0..n_active) |j| top_weights[j] *= inv;
            }
        } else {
            // Learned routing: gate_inp GEMV → sync → top-k on CPU
            const gi = try self.layerTensorReq(li, "ffn_gate_inp.weight");
            self.be.gemv(self.hidden2.ptr, .{ .data = gi.data_ptr, .dtype = gi.dtype }, self.router_logits.ptr, ne, e);
            self.be.sync(); // CPU reads router_logits

            // Compute probs = sqrt_softplus(logits) — used for final weights (unbiased)
            var probs: [256]f32 = undefined;
            for (0..ne) |i| probs[i] = sqrtSoftplus(self.router_logits[i]);

            // Selection uses biased probs (exp_probs_b added AFTER sqrt_softplus)
            // Weights use UNBIASED probs (per DeepSeek V3/V4 spec)
            var selection: [256]f32 = probs;
            if (self.layerTensor(li, "exp_probs_b.bias")) |bias_t| {
                const bias = @as([*]const f32, @ptrCast(@alignCast(bias_t.data_ptr)))[0..ne];
                for (0..ne) |i| selection[i] += bias[i];
            }

            math_ops.topKExperts(selection[0..ne], nk, top_ids[0..nk], top_scores[0..nk]);
            n_active = nk;

            // Weights from unbiased probs, normalized
            var wsum: f32 = 0.0;
            for (0..n_active) |j| {
                top_weights[j] = probs[top_ids[j]];  // unbiased prob for selected expert
                wsum += top_weights[j];
            }
            if (wsum > 0.0) {
                const inv = self.expert_weights_scale / wsum;
                for (0..n_active) |j| top_weights[j] *= inv;
            }
        }

        // 3-phase batched FFN: all expert gate+up GEMVs, clamped siluMul, down GEMVs in one GPU cmd buffer.
        // DS4 SwiGLU clamp (gate≤10, up±10) applied via clamped_silu_mul_f32 Metal kernel.
        var n_scratch: usize = 0;
        var slot_weights: [9]f32 = [_]f32{0.0} ** 9;

        // Phase 1: all gate+up GEMVs into scratch buffers (no siluMul yet)
        if (self.n_expert_shared > 0) {
            if (self.layerTensor(li, "ffn_gate_shexp.weight")) |gt| {
                const ut = self.layerTensor(li, "ffn_up_shexp.weight") orelse return error.MissingTensor;
                self.be.gemv(self.hidden2.ptr, .{ .data = gt.data_ptr, .dtype = gt.dtype }, self.ff_gate_scratch.ptr + n_scratch * ff, ff, e);
                self.be.gemv(self.hidden2.ptr, .{ .data = ut.data_ptr, .dtype = ut.dtype }, self.ff_up_scratch.ptr + n_scratch * ff, ff, e);
                slot_weights[n_scratch] = 1.0;
                n_scratch += 1;
            }
        }
        const shexp_slots = n_scratch;

        var de_ptrs: [9][*]const u8 = undefined;
        var de_dtype: DType = .f32;
        if (self.layerTensor(li, "ffn_gate_exps.weight")) |ge| {
            const ue = self.layerTensor(li, "ffn_up_exps.weight") orelse return error.MissingTensor;
            const de = self.layerTensor(li, "ffn_down_exps.weight") orelse return error.MissingTensor;
            de_dtype = de.dtype;
            const gs = ds4ExpertStride(ge);
            const us = ds4ExpertStride(ue);
            const ds = ds4ExpertStride(de);
            for (0..n_active) |j| {
                const eid = top_ids[j];
                self.be.gemv(self.hidden2.ptr, .{ .data = ge.data_ptr + eid * gs, .dtype = ge.dtype }, self.ff_gate_scratch.ptr + n_scratch * ff, ff, e);
                self.be.gemv(self.hidden2.ptr, .{ .data = ue.data_ptr + eid * us, .dtype = ue.dtype }, self.ff_up_scratch.ptr + n_scratch * ff, ff, e);
                de_ptrs[n_scratch] = de.data_ptr + eid * ds;
                slot_weights[n_scratch] = top_weights[j];
                n_scratch += 1;
            }
        }

        // Phase 2: clamped SiLU×mul (gate≤10, up±10) — no sync needed (same cmd buffer)
        for (0..n_scratch) |slot| {
            self.be.clampedSiluMul(
                self.ff_gate_scratch.ptr + slot * ff,
                self.ff_up_scratch.ptr + slot * ff,
                self.ff_gate_scratch.ptr + slot * ff, ff);
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

        self.be.sync(); // all down GEMVs complete

        // CPU: SIMD weighted accumulation from all scratch slots
        const V8 = @Vector(8, f32);
        @memset(self.expert_accum, 0.0);
        for (0..n_scratch) |slot| {
            const sd = self.expert_scratch[slot * e ..][0..e];
            const wv: V8 = @splat(slot_weights[slot]);
            var i: usize = 0;
            while (i + 8 <= e) : (i += 8) {
                const acc: V8 = self.expert_accum[i..][0..8].*;
                self.expert_accum[i..][0..8].* = @mulAdd(V8, @as(V8, sd[i..][0..8].*), wv, acc);
            }
            while (i < e) : (i += 1) self.expert_accum[i] += sd[i] * slot_weights[slot];
        }

        @memcpy(self.hidden, self.expert_accum);
    }

    // ── Forward pass ─────────────────────────────────────────────

    pub fn forward(self: *Ds4Model, token_id: u32) !u32 {
        if (self.cancelled.load(.acquire)) return error.Cancelled;
        if (self.kv_seq_len >= self.max_seq_len) return error.KVCacheFull;

        const e = self.n_embd;
        const nl = self.n_layers;

        // Embed → broadcast to all n_hc HC streams
        const emb = try self.getTensorReq("token_embd.weight");
        self.be.embLookup(.{ .data = emb.data_ptr, .dtype = emb.dtype }, token_id, self.hc_state.ptr, e);
        self.be.sync();
        for (1..n_hc) |s| @memcpy(self.hc_state[s * e ..][0..e], self.hc_state[0..e]);

        for (0..nl) |li| {
            if (self.cancelled.load(.acquire)) return error.Cancelled;

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

        // Final norm + LM head
        const norm_w = try self.getTensorReq("output_norm.weight");
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(norm_w, e), self.hidden.ptr, e, self.rms_eps);
        self.be.sync();
        const lm = try self.getTensorReq("output.weight");
        self.be.gemv(self.hidden.ptr, .{ .data = lm.data_ptr, .dtype = lm.dtype }, self.logits_buf.ptr, self.vocab_size, e);
        self.be.sync();

        self.kv_seq_len += 1;
        return math_ops.argmax(self.logits_buf);
    }

    pub fn prefill(self: *Ds4Model, token_ids: []const u32) !u32 {
        var last: u32 = 0;
        for (token_ids) |tid| last = try self.forward(tid);
        return last;
    }

    pub fn resetCache(self: *Ds4Model) void {
        self.kv_seq_len = 0;
        self.cancelled.store(false, .release);
    }

    pub fn cancel(self: *Ds4Model) void {
        self.cancelled.store(true, .release);
    }

    pub fn setMegakernel(self: *Ds4Model, en: bool) void { self.megakernel_enabled = en; }
    pub fn setLayerSkip(self: *Ds4Model, s: u32, end: u32) void { self.layer_skip_start = s; self.layer_skip_end = end; }
    pub fn getHidden(self: *const Ds4Model) []const f32 { return self.hidden; }
    pub fn getBlockTable(_: *const Ds4Model) []const u32 { return &.{}; }
};

// ── Math helpers ─────────────────────────────────────────────────

inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

inline fn sqrtSoftplus(x: f32) f32 {
    return @sqrt(@log(1.0 + @exp(x)));
}

/// In-place RMS normalization, no learned weight. SIMD-optimized.
fn plainRmsNorm(x: []f32, eps: f32) void {
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

/// SiGLU with per-DS4 clamping: up clamped [-10,10], gate upper-clamped [−∞,10].
/// Clamp prevents outlier explosion from Q2_K dequantization.
fn siluMulClamped(gate: []f32, up: []f32, clamp: f32) void {
    for (gate, up) |*g, *u| {
        u.* = @min(clamp, @max(-clamp, u.*));  // two-sided clamp on up
        g.* = @min(clamp, g.*);               // upper-only clamp on gate
        g.* = g.* * (1.0 / (1.0 + @exp(-g.*))) * u.*;
    }
}

fn siluMul(gate: []f32, up: []f32) void {
    for (gate, up) |*g, u| g.* = g.* * (1.0 / (1.0 + @exp(-g.*))) * u;
}

/// Apply RoPE using a pre-computed cos/sin table (avoids recomputing per head).
fn applyRopeTable(x: []f32, cos_t: []const f32, sin_t: []const f32) void {
    for (cos_t, sin_t, 0..) |c, s, i| {
        const x0 = x[i * 2];
        const x1 = x[i * 2 + 1];
        x[i * 2] = x0 * c - x1 * s;
        x[i * 2 + 1] = x0 * s + x1 * c;
    }
}

/// Inverse RoPE using pre-computed table (negate sin).
fn applyRopeInverseTable(x: []f32, cos_t: []const f32, sin_t: []const f32) void {
    for (cos_t, sin_t, 0..) |c, s, i| {
        const x0 = x[i * 2];
        const x1 = x[i * 2 + 1];
        x[i * 2] = x0 * c + x1 * s;
        x[i * 2 + 1] = -x0 * s + x1 * c;
    }
}

fn applyRope(x: []f32, pos: usize, freq_base: f32, rope_dim: usize) void {
    const nd = rope_dim / 2;
    for (0..nd) |i| {
        const freq = std.math.pow(f32, freq_base, -@as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(rope_dim)));
        const theta = @as(f32, @floatFromInt(pos)) * freq;
        const c = @cos(theta);
        const s = @sin(theta);
        const x0 = x[i * 2];
        const x1 = x[i * 2 + 1];
        x[i * 2] = x0 * c - x1 * s;
        x[i * 2 + 1] = x0 * s + x1 * c;
    }
}

/// Inverse RoPE (rope_ext_back): rotation by -theta. Applied to attention output rope portion.
fn applyRopeInverse(x: []f32, pos: usize, freq_base: f32, rope_dim: usize) void {
    const nd = rope_dim / 2;
    for (0..nd) |i| {
        const freq = std.math.pow(f32, freq_base, -@as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(rope_dim)));
        const theta = @as(f32, @floatFromInt(pos)) * freq;
        const c = @cos(theta);
        const s = @sin(theta);
        const x0 = x[i * 2];
        const x1 = x[i * 2 + 1];
        // Inverse: transpose of rotation matrix = rotation by -theta
        x[i * 2] = x0 * c + x1 * s;
        x[i * 2 + 1] = -x0 * s + x1 * c;
    }
}

/// Sinkhorn normalization for [n_hc × n_hc] matrix.
/// m[r*n+c] maps to comb(dst=c, src=r). ggml_soft_max runs over ne[0]=dst for each src.
fn hcSinkhorn(m: []f32) void {
    const n = n_hc;
    // Initial softmax: for each src=r, normalize over dst=c (matches ggml_soft_max over ne[0]=dst)
    for (0..n) |r| { // for each src=r
        var mx = m[r * n + 0];
        for (1..n) |c| if (m[r * n + c] > mx) { mx = m[r * n + c]; };
        var sm: f32 = 0.0;
        for (0..n) |c| { m[r * n + c] = @exp(m[r * n + c] - mx); sm += m[r * n + c]; }
        for (0..n) |c| m[r * n + c] /= sm;
    }
    for (m) |*v| v.* += hc_eps;

    var col_s: [n_hc]f32 = undefined;
    var row_s: [n_hc]f32 = undefined;
    for (0..hc_sinkhorn_iters) |_| {
        @memset(&col_s, 0.0);
        for (0..n) |r| for (0..n) |c| { col_s[c] += m[r * n + c]; };
        for (&col_s) |*v| v.* += hc_eps;
        for (0..n) |r| for (0..n) |c| { m[r * n + c] /= col_s[c]; };

        @memset(&row_s, 0.0);
        for (0..n) |r| for (0..n) |c| { row_s[r] += m[r * n + c]; };
        for (&row_s) |*v| v.* += hc_eps;
        for (0..n) |r| for (0..n) |c| { m[r * n + c] /= row_s[r]; };
    }
}

/// CPU Q8_0 GEMV: y[n_out] = w[n_out rows × n_in cols] @ x[n_in].
/// Avoids Metal GPU dispatch overhead for tiny output dims (like HC pre's 24-output GEMV).
/// Q8_0 block: 2-byte f16 scale + 32 i8 values = 34 bytes.
fn cpuGemvQ8_0(w_ptr: [*]const u8, x: []const f32, y: []f32, n_in: usize) void {
    const V8 = @Vector(8, f32);
    const block_size: usize = 32;
    const block_bytes: usize = 34;
    const n_out = y.len;
    const blocks_per_row = n_in / block_size;
    for (0..n_out) |i| {
        var acc: V8 = @splat(0.0);
        const row_ptr = w_ptr + i * blocks_per_row * block_bytes;
        for (0..blocks_per_row) |b| {
            const blk = row_ptr + b * block_bytes;
            const scale: f32 = @floatCast(@as(f16, @bitCast(@as(u16, blk[0]) | (@as(u16, blk[1]) << 8))));
            const sv: V8 = @splat(scale);
            const q = blk + 2;
            const xb = x[b * block_size ..][0..32];
            var k: usize = 0;
            while (k + 8 <= 32) : (k += 8) {
                var qv: V8 = undefined;
                inline for (0..8) |idx| qv[idx] = @floatFromInt(@as(i8, @bitCast(q[k + idx])));
                const xv: V8 = xb[k..][0..8].*;
                acc = @mulAdd(V8, qv * sv, xv, acc);
            }
        }
        y[i] = @reduce(.Add, acc);
    }
}

/// Per-expert stride for ds4 expert tensors: dims=[input_dim, ff_dim, n_experts].
fn ds4ExpertStride(t: TensorInfo) usize {
    std.debug.assert(t.n_dims >= 3);
    const elems = @as(usize, @intCast(t.dims[0])) * @as(usize, @intCast(t.dims[1]));
    return backend_mod.weightBytes(t.dtype, 1, elems);
}
