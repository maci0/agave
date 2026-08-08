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
const Backend = backend_mod.Backend;
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
    expert_scratch: []f32 = &.{},   // [max_total_experts * n_embd] for batched down GEMVs
    router_logits: []f32 = &.{},    // [n_experts]
    logits_buf: []f32 = &.{},       // [vocab_size]

    // Simple flat f32 KV cache: [n_layers * ctx * kv_lora_rank]
    kv_k: []f32 = &.{},
    kv_v: []f32 = &.{},

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
        self.router_logits = try allocator.alloc(f32, self.n_experts);
        self.logits_buf = try allocator.alloc(f32, self.vocab_size);

        // Simple flat KV cache: nl layers × ctx positions × kd dims × 2 (K and V same size)
        self.kv_k = try allocator.alloc(f32, nl * ctx * kd);
        self.kv_v = try allocator.alloc(f32, nl * ctx * kd);

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
            &self.expert_scratch, &self.router_logits, &self.logits_buf, &self.kv_k, &self.kv_v,
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

    fn kvKSlice(self: *Ds4Model, li: usize, pos: usize) []f32 {
        const kd = self.kv_lora_rank;
        const ctx = self.max_seq_len;
        return self.kv_k[li * ctx * kd + pos * kd ..][0..kd];
    }

    fn kvVSlice(self: *Ds4Model, li: usize, pos: usize) []f32 {
        const kd = self.kv_lora_rank;
        const ctx = self.max_seq_len;
        return self.kv_v[li * ctx * kd + pos * kd ..][0..kd];
    }

    // ── Hyper Connection ──────────────────────────────────────────

    /// Compute HC pre-weights and sublayer input in `self.hidden`.
    fn hcPre(
        self: *Ds4Model,
        hc_fn: TensorInfo,
        hc_base: TensorInfo,
        hc_scale: TensorInfo,
    ) void {
        const e = self.n_embd;
        const flat_size = n_hc * e;

        // RMS-norm the flat HC state (no learned weight)
        @memcpy(self.flat_norm, self.hc_state);
        plainRmsNorm(self.flat_norm, self.rms_eps);

        // mixes[24] = hc_fn @ flat_norm
        self.be.gemv(self.flat_norm.ptr, .{ .data = hc_fn.data_ptr, .dtype = hc_fn.dtype }, self.hc_mixes.ptr, hc_mix_dim, flat_size);
        self.be.sync();

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

        // sublayer input = weighted sum of streams
        @memset(self.hidden, 0.0);
        for (0..n_hc) |s| {
            const stream = self.hc_state[s * e ..][0..e];
            const w = self.hc_pre_w[s];
            for (0..e) |i| self.hidden[i] += stream[i] * w;
        }
    }

    /// Update HC state after a sublayer. Sublayer output must be in `self.hidden`.
    fn hcPost(self: *Ds4Model) void {
        const e = self.n_embd;
        const sub = self.hidden;
        for (0..n_hc) |dst| {
            const ns = self.new_hc[dst * e ..][0..e];
            const pw = self.hc_post_w[dst];
            for (0..e) |i| ns[i] = sub[i] * pw;
            for (0..n_hc) |src| {
                const ss = self.hc_state[src * e ..][0..e];
                // Column-major storage: comb[dst, src] = hc_comb[dst + src*n_hc]
                const c = self.hc_comb[dst + src * n_hc];
                for (0..e) |i| ns[i] += ss[i] * c;
            }
        }
        @memcpy(self.hc_state, self.new_hc);
    }

    /// HC head: merge 4 streams → self.hidden.
    fn hcHead(self: *Ds4Model, hc_fn: TensorInfo, hc_base: TensorInfo, hc_scale: TensorInfo) void {
        const e = self.n_embd;
        @memcpy(self.flat_norm, self.hc_state);
        plainRmsNorm(self.flat_norm, self.rms_eps);
        self.be.gemv(self.flat_norm.ptr, .{ .data = hc_fn.data_ptr, .dtype = hc_fn.dtype }, self.hc_pre_w.ptr, n_hc, n_hc * e);
        self.be.sync();

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
        // Use compressed rope freq for layers with ratio≠0 (most layers in DS4)
        const rope_freq = if (self.compress_ratios[li] != 0) self.compress_rope_freq else self.rope_freq;

        // Pre-norm → q_a → q_a_norm → q_b all on GPU in one batch
        const nw = try self.layerTensorReq(li, "attn_norm.weight");
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);
        const q_a = try self.layerTensorReq(li, "attn_q_a.weight");
        self.be.gemv(self.hidden2.ptr, .{ .data = q_a.data_ptr, .dtype = q_a.dtype }, self.q_compressed.ptr, ql, e);
        const q_an = try self.layerTensorReq(li, "attn_q_a_norm.weight");
        self.be.rmsNorm(self.q_compressed.ptr, self.normAsF32(q_an, ql), self.q_compressed.ptr, ql, self.rms_eps);
        const q_b = try self.layerTensorReq(li, "attn_q_b.weight");
        self.be.gemv(self.q_compressed.ptr, .{ .data = q_b.data_ptr, .dtype = q_b.dtype }, self.q_full.ptr, nh * kd, ql);
        self.be.sync(); // CPU per-head norm + RoPE read q_full

        // Per-head Q RMS norm + RoPE
        for (0..nh) |h| {
            const q_head = self.q_full[h * kd ..][0..kd];
            plainRmsNorm(q_head, self.rms_eps);
            applyRope(q_head[nope..][0..rd], pos, rope_freq, rd);
        }

        // KV: hidden2 → kv_a → kv_a_norm on GPU, then CPU RoPE
        const kv_a = try self.layerTensorReq(li, "attn_kv.weight");
        self.be.gemv(self.hidden2.ptr, .{ .data = kv_a.data_ptr, .dtype = kv_a.dtype }, self.kv_proj.ptr, kd, e);
        const kv_an = try self.layerTensorReq(li, "attn_kv_a_norm.weight");
        self.be.rmsNorm(self.kv_proj.ptr, self.normAsF32(kv_an, kd), self.kv_proj.ptr, kd, self.rms_eps);
        self.be.sync(); // CPU RoPE reads kv_proj
        applyRope(self.kv_proj[nope..][0..rd], pos, rope_freq, rd);

        // Store KV (K = V in this MLA variant)
        @memcpy(self.kvKSlice(li, pos), self.kv_proj);
        @memcpy(self.kvVSlice(li, pos), self.kv_proj);

        // Attention: Q [nh, kd] × K [1, kd, sl] → scores → × V
        const sl = pos + 1;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));
        // Per-head sink biases added to position 0 score (prevents attention collapse)
        const sinks_t = self.layerTensor(li, "attn_sinks.weight");
        const sinks_data: ?[*]const f32 = if (sinks_t) |t| @ptrCast(@alignCast(t.data_ptr)) else null;
        @memset(self.attn_out, 0.0);
        for (0..nh) |h| {
            const q_h = self.q_full[h * kd ..][0..kd];
            // QK scores
            for (0..sl) |t| {
                const k_t = self.kvKSlice(li, t);
                var dot: f32 = 0.0;
                for (0..kd) |i| dot += q_h[i] * k_t[i];
                self.scores_buf[t] = dot * scale;
            }
            // Add sink bias to position 0 (attention sink)
            if (sinks_data) |sd| self.scores_buf[0] += sd[h];
            // Softmax
            {
                var mx = self.scores_buf[0];
                for (self.scores_buf[1..sl]) |v| if (v > mx) { mx = v; };
                var sm: f32 = 0.0;
                for (self.scores_buf[0..sl]) |*v| { v.* = @exp(v.* - mx); sm += v.*; }
                const inv = 1.0 / sm;
                for (self.scores_buf[0..sl]) |*v| v.* *= inv;
            }
            // V accumulation
            const ao_h = self.attn_out[h * kd ..][0..kd];
            for (0..sl) |t| {
                const v_t = self.kvVSlice(li, t);
                const w = self.scores_buf[t];
                for (0..kd) |i| ao_h[i] += v_t[i] * w;
            }
        }

        // Apply inverse RoPE (derope) to rope portion of each attention head output.
        // llama.cpp applies ggml_rope_ext_back before the output LoRA projection.
        for (0..nh) |h| {
            applyRopeInverse(self.attn_out[h * kd + nope ..][0..rd], pos, rope_freq, rd);
        }

        // Output LoRA: grouped wo_a [n_in=4096 per group, n_out=o_lora_rank] × 8 groups
        const og: usize = self.o_groups;
        const olr: usize = self.o_lora_rank;
        const group_in: usize = nh * kd / og; // = 64*512/8 = 4096
        const wo_a = try self.layerTensorReq(li, "attn_output_a.weight");
        const row_bytes = backend_mod.weightBytes(wo_a.dtype, 1, group_in);
        for (0..og) |g| {
            const xp = self.attn_out.ptr + g * group_in;
            const wp = wo_a.data_ptr + g * olr * row_bytes;
            const yp = self.lora_out.ptr + g * olr;
            self.be.gemv(xp, .{ .data = wp, .dtype = wo_a.dtype }, yp, olr, group_in);
        }
        self.be.sync();

        const wo_b = try self.layerTensorReq(li, "attn_output_b.weight");
        self.be.gemv(self.lora_out.ptr, .{ .data = wo_b.data_ptr, .dtype = wo_b.dtype }, self.attn_result.ptr, e, og * olr);
        self.be.sync();

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
            // Hash routing: table lookup, no GPU data needed
            const t2e = try self.layerTensorReq(li, "ffn_gate_tid2eid.weight");
            const n_slots: usize = @intCast(t2e.dims[0]);
            const vocab: usize = @intCast(t2e.dims[1]);
            const data: [*]const i32 = @ptrCast(@alignCast(t2e.data_ptr));
            const safe_tid: usize = @min(@as(usize, token_id), vocab - 1);
            for (0..nk) |j| {
                top_ids[j] = @intCast(data[safe_tid * n_slots + j]);
                top_weights[j] = 1.0 / @as(f32, @floatFromInt(nk));
            }
            n_active = nk;
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

        // All expert gate+up+siluMul+down GEMVs batched on GPU — single sync at end.
        // Results in expert_scratch[slot*e..]; slot_weights tracks scaling per slot.
        var n_scratch: usize = 0;
        var slot_weights: [9]f32 = [_]f32{0.0} ** 9; // max = n_expert_used+n_expert_shared

        // Shared expert (weight 1.0, unscaled)
        if (self.n_expert_shared > 0) {
            if (self.layerTensor(li, "ffn_gate_shexp.weight")) |gt| {
                const ut = self.layerTensor(li, "ffn_up_shexp.weight") orelse return error.MissingTensor;
                const dt = self.layerTensor(li, "ffn_down_shexp.weight") orelse return error.MissingTensor;
                self.be.gemv(self.hidden2.ptr, .{ .data = gt.data_ptr, .dtype = gt.dtype }, self.ff_gate.ptr, ff, e);
                self.be.gemv(self.hidden2.ptr, .{ .data = ut.data_ptr, .dtype = ut.dtype }, self.ff_up.ptr, ff, e);
                self.be.siluMul(self.ff_gate.ptr, self.ff_up.ptr, self.ff_gate.ptr, ff);
                self.be.gemv(self.ff_gate.ptr, .{ .data = dt.data_ptr, .dtype = dt.dtype }, self.expert_scratch.ptr + n_scratch * e, e, ff);
                slot_weights[n_scratch] = 1.0;
                n_scratch += 1;
            }
        }

        // Routed experts — all on GPU, no intermediate syncs
        if (self.layerTensor(li, "ffn_gate_exps.weight")) |ge| {
            const ue = self.layerTensor(li, "ffn_up_exps.weight") orelse return error.MissingTensor;
            const de = self.layerTensor(li, "ffn_down_exps.weight") orelse return error.MissingTensor;
            const gs = ds4ExpertStride(ge);
            const us = ds4ExpertStride(ue);
            const ds = ds4ExpertStride(de);
            for (0..n_active) |j| {
                const eid = top_ids[j];
                const gp = ge.data_ptr + eid * gs;
                const up = ue.data_ptr + eid * us;
                const dp = de.data_ptr + eid * ds;
                self.be.gemv(self.hidden2.ptr, .{ .data = gp, .dtype = ge.dtype }, self.ff_gate.ptr, ff, e);
                self.be.gemv(self.hidden2.ptr, .{ .data = up, .dtype = ue.dtype }, self.ff_up.ptr, ff, e);
                self.be.siluMul(self.ff_gate.ptr, self.ff_up.ptr, self.ff_gate.ptr, ff);
                self.be.gemv(self.ff_gate.ptr, .{ .data = dp, .dtype = de.dtype }, self.expert_scratch.ptr + n_scratch * e, e, ff);
                slot_weights[n_scratch] = top_weights[j];
                n_scratch += 1;
            }
        }

        self.be.sync(); // single sync — all expert down GEMVs complete

        // CPU: weighted accumulation from scratch slots
        @memset(self.expert_accum, 0.0);
        for (0..n_scratch) |slot| {
            const sd = self.expert_scratch[slot * e ..][0..e];
            const w = slot_weights[slot];
            for (0..e) |i| self.expert_accum[i] += sd[i] * w;
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

/// In-place RMS normalization, no learned weight.
fn plainRmsNorm(x: []f32, eps: f32) void {
    var ss: f32 = 0.0;
    for (x) |v| ss += v * v;
    const s = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    for (x) |*v| v.* *= s;
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

/// Sinkhorn normalization for [n_hc × n_hc] matrix stored row-major [dst*n_hc+src].
/// Initial softmax is column-wise (over dst for each src), matching ggml column-major convention.
fn hcSinkhorn(m: []f32) void {
    const n = n_hc;
    // Softmax over columns (dst dimension) for each src — matches ggml ggml_soft_max on [ne0=dst, ne1=src]
    for (0..n) |c| { // for each src
        var mx = m[0 * n + c];
        for (1..n) |r| if (m[r * n + c] > mx) { mx = m[r * n + c]; };
        var sm: f32 = 0.0;
        for (0..n) |r| { m[r * n + c] = @exp(m[r * n + c] - mx); sm += m[r * n + c]; }
        for (0..n) |r| m[r * n + c] /= sm;
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

/// Per-expert stride for ds4 expert tensors: dims=[input_dim, ff_dim, n_experts].
fn ds4ExpertStride(t: TensorInfo) usize {
    std.debug.assert(t.n_dims >= 3);
    const elems = @as(usize, @intCast(t.dims[0])) * @as(usize, @intCast(t.dims[1]));
    return backend_mod.weightBytes(t.dtype, 1, elems);
}
