//! Qwen3.8-Flash-Next (GGUF architecture `qwen4exp`).
//!
//! 4-stream hyper-connections, Gated DeltaNet (sigmoid output gate), QSA
//! (indexer + GQA every 4th layer), n-gram PLE with a lazily gathered table,
//! and 512-expert MoE. The PLE table is never uploaded or prefaulted; IQ2/IQ3
//! expert GEMV runs on a dedicated CpuBackend because GPU kernels panic on
//! those dtypes.

const std = @import("std");
const backend_mod = @import("../backend/backend.zig");
const format_mod = @import("../format/format.zig");
const model_mod = @import("model.zig");
const math_ops = @import("../ops/math.zig");
const attn_ops = @import("../ops/attention.zig");
const quant = @import("../ops/quant.zig");
const kv_quant = @import("../ops/kv_quant.zig");
const Backend = backend_mod.Backend;
const CpuBackend = backend_mod.CpuBackend;
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const DType = format_mod.DType;
const Allocator = std.mem.Allocator;
const ThreadPool = @import("../thread_pool.zig").ThreadPool;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;

const max_active_experts: usize = 16;
const max_ple_ngram: usize = 8;
const max_ple_heads: usize = 32;
const max_norm_entries: usize = 768;
const qsa_select_cap: usize = 4096;
const default_hc: u32 = 4;
const default_full_attn_interval: u32 = 4;
const default_moe_active: u32 = 10;
const ple_null_tok: i32 = -1;
const ple_gate_floor: f32 = 1e-6;
const expertWeightStride = model_mod.expertWeightStride;

/// Host-side PLE n-gram hash. `prev` is oldest-first, length `n_gram-1`, with
/// `ple_null_tok` for a missing predecessor. EOS in the window resets older
/// slots; the current token's own EOS does not cut its context.
pub fn pleRowIndices(
    cur: u32,
    prev: []const i32,
    n_gram: u32,
    heads_per_ngram: u32,
    multipliers: []const u64,
    offsets: []const u64,
    vocabs: []const u64,
    eos: u32,
    out: []u32,
) void {
    std.debug.assert(n_gram >= 2 and n_gram <= max_ple_ngram);
    const n_prev: usize = n_gram - 1;
    std.debug.assert(prev.len >= n_prev);
    std.debug.assert(multipliers.len >= n_gram);
    var ctx: [max_ple_ngram]u32 = undefined;
    ctx[0] = cur;
    var cut = false;
    var s: usize = 1;
    while (s < n_gram) : (s += 1) {
        const t: i32 = if (cut) ple_null_tok else prev[n_prev - s];
        cut = cut or t < 0 or t == @as(i32, @intCast(eos));
        ctx[s] = if (cut) eos else @intCast(t);
    }
    const n_heads: usize = (n_gram - 1) * heads_per_ngram;
    std.debug.assert(out.len >= n_heads);
    std.debug.assert(offsets.len >= n_heads and vocabs.len >= n_heads);
    var n: u32 = 2;
    while (n <= n_gram) : (n += 1) {
        var mixed: u64 = @as(u64, ctx[0]) *% multipliers[0];
        var j: usize = 1;
        while (j < n) : (j += 1) {
            mixed ^= @as(u64, ctx[j]) *% multipliers[j];
        }
        const base: usize = (n - 2) * heads_per_ngram;
        var g: usize = 0;
        while (g < heads_per_ngram) : (g += 1) {
            const h = base + g;
            const vs = vocabs[h];
            const row = if (vs == 0) offsets[h] else (mixed % vs) + offsets[h];
            out[h] = @intCast(row);
        }
    }
}

fn dtypeNeedsCpuGemv(dt: DType) bool {
    return switch (dt) {
        .iq2_s, .iq2_xs, .iq2_xxs, .iq3_s, .iq3_xxs, .iq4_nl, .iq1_s, .iq1_m => true,
        else => false,
    };
}

/// Qwen3.8-Flash-Next hybrid MoE with hyper-connections, GDN, QSA, and PLE.
pub const Qwen4ExpModel = struct {
    const NormCacheEntry = model_mod.NormCacheEntry;

    fmt: Format,
    be: Backend,
    cpu: CpuBackend = .{},
    pool: ?*ThreadPool = null,
    allocator: Allocator,

    n_layers: u32 = 48,
    n_embd: u32 = 2560,
    n_head: u32 = 24,
    n_head_kv: u32 = 2,
    head_dim: u32 = 256,
    vocab_size: u32 = 248320,
    rope_theta: f32 = 10_000_000.0,
    rope_dim: u32 = 64,
    rms_eps: f32 = 1e-6,
    full_attn_interval: u32 = default_full_attn_interval,
    eos_token_id: u32 = 248046,
    max_seq_len: usize = 4096,
    kv_seq_len: usize = 0,

    n_hc: u32 = default_hc,
    hc_low_rank: u32 = 320,
    n_experts: u32 = 512,
    n_experts_active: u32 = default_moe_active,
    expert_ff_dim: u32 = 640,
    shared_expert_ff_dim: u32 = 640,

    ssm_d_conv: u32 = 4,
    ssm_d_state: u32 = 128,
    ssm_n_group: u32 = 16,
    ssm_dt_rank: u32 = 48,
    ssm_d_inner: u32 = 6144,

    indexer_n_head: u32 = 4,
    indexer_head_size: u32 = 128,
    indexer_top_k: u32 = 2048,

    ple_ngram: u32 = 3,
    ple_heads_per_ngram: u32 = 8,
    ple_n_heads: u32 = 16,
    ple_head_dim: u32 = 160,
    ple_conv_kernel: u32 = 4,
    ple_eos: u32 = 248046,
    ple_image_token: u32 = 0,
    ple_layer_mask: u64 = 0,
    ple_multipliers: []const u64 = &.{},
    ple_offsets: []const u64 = &.{},
    ple_vocabs: []const u64 = &.{},

    compress_ratios: []u32 = &.{},

    hidden: []f32 = &.{},
    hidden2: []f32 = &.{},
    res_hc: []f32 = &.{},
    hc_lo: []f32 = &.{},
    hc_gate: []f32 = &.{},
    hc_inject: []f32 = &.{},
    q_buf: []f32 = &.{},
    k_buf: []f32 = &.{},
    v_buf: []f32 = &.{},
    gate_buf: []f32 = &.{},
    attn_out: []f32 = &.{},
    scores_buf: []f32 = &.{},
    ff_buf1: []f32 = &.{},
    ff_buf2: []f32 = &.{},
    moe_out: []f32 = &.{},
    router_logits: []f32 = &.{},
    logits_buf: []f32 = &.{},
    ssm_qkv_buf: []f32 = &.{},
    ssm_z_buf: []f32 = &.{},
    ssm_alpha_buf: []f32 = &.{},
    ssm_beta_buf: []f32 = &.{},
    ssm_conv_out: []f32 = &.{},
    idx_q_buf: []f32 = &.{},
    idx_k_buf: []f32 = &.{},
    idx_pooled: []f32 = &.{},
    idx_scores: []f32 = &.{},
    gather_k: []u8 = &.{},
    gather_v: []u8 = &.{},
    ple_emb: []f32 = &.{},
    ple_key: []f32 = &.{},
    ple_value: []f32 = &.{},
    ple_query: []f32 = &.{},
    ple_gated: []f32 = &.{},
    ple_conv_in: []f32 = &.{},
    ple_conv_out: []f32 = &.{},
    ple_conv_w: []f32 = &.{},
    ple_conv_hist: []f32 = &.{},
    ple_tok_hist: []i32 = &.{},
    ple_rows: []u32 = &.{},

    conv_states: [][]f32 = &.{},
    ssm_states: [][]f32 = &.{},
    dn_ssm_a: [][]f32 = &.{},
    dn_dt_bias: [][]f32 = &.{},
    dn_conv_w: [][]f32 = &.{},
    dn_ssm_norm_w: [][]f32 = &.{},
    qsa_k: [][]u8 = &.{},
    qsa_v: [][]u8 = &.{},
    qsa_idx_k: [][]f32 = &.{},
    qsa_slot: []u8 = &.{},
    n_qsa: u32 = 0,

    norm_cache: [max_norm_entries]NormCacheEntry = undefined,
    norm_cache_len: usize = 0,

    cancelled: std.atomic.Value(bool) = .init(false),
    kv_type_k: kv_quant.KvQuantType = .f32,
    kv_type_v: kv_quant.KvQuantType = .f32,
    megakernel_enabled: bool = false,

    fn isQsa(self: *const Qwen4ExpModel, layer: u32) bool {
        if (self.full_attn_interval == 0) return true;
        return ((layer + 1) % self.full_attn_interval) == 0;
    }

    fn isPle(self: *const Qwen4ExpModel, layer: u32) bool {
        if (layer >= 64) return false;
        return (self.ple_layer_mask & (@as(u64, 1) << @intCast(layer))) != 0;
    }

    fn compressRatio(self: *const Qwen4ExpModel, layer: u32) u32 {
        if (layer < self.compress_ratios.len) return self.compress_ratios[layer];
        return if (self.isQsa(layer)) 4 else 0;
    }

    fn hcDim(self: *const Qwen4ExpModel) usize {
        return @as(usize, self.n_hc) * self.n_embd;
    }

    fn convCh(self: *const Qwen4ExpModel) usize {
        return @as(usize, self.ssm_d_inner) + 2 * @as(usize, self.ssm_n_group) * @as(usize, self.ssm_d_state);
    }

    fn missing(self: *const Qwen4ExpModel, li: u32, name: []const u8) error{MissingTensor} {
        _ = self;
        std.log.err("qwen4exp: missing blk.{d}.{s}", .{ li, name });
        return error.MissingTensor;
    }

    fn layerT(self: *const Qwen4ExpModel, li: u32, name: []const u8) !TensorInfo {
        return self.fmt.layerTensor(li, name) orelse self.missing(li, name);
    }

    fn gemvW(self: *Qwen4ExpModel, x: [*]const f32, t: TensorInfo, y: [*]f32, n: usize, k: usize) void {
        if (dtypeNeedsCpuGemv(t.dtype)) {
            self.be.sync();
            self.cpu.gemv(x, .{ .data = t.data_ptr, .dtype = t.dtype }, y, n, k);
            return;
        }
        model_mod.dispatchGemv(self.be, self.fmt, x, t, y, n, k);
    }

    fn gemvExpert(self: *Qwen4ExpModel, x: [*]const f32, t: TensorInfo, ei: usize, stride: usize, y: [*]f32, n: usize, k: usize) void {
        const data = t.data_ptr + ei * stride;
        if (dtypeNeedsCpuGemv(t.dtype)) {
            self.be.sync();
            self.cpu.gemv(x, .{ .data = data, .dtype = t.dtype }, y, n, k);
            return;
        }
        self.be.gemv(x, .{ .data = data, .dtype = t.dtype }, y, n, k);
    }

    fn normAsF32(self: *Qwen4ExpModel, t: TensorInfo, n: usize) [*]const f32 {
        const key = @intFromPtr(t.data_ptr);
        for (self.norm_cache[0..self.norm_cache_len]) |entry| {
            if (entry.key == key) return entry.data.ptr;
        }
        if (t.dtype == .f32) return @ptrCast(@alignCast(t.data_ptr));
        if (self.norm_cache_len >= max_norm_entries)
            @panic("qwen4exp: norm cache overflow");
        const buf = self.allocator.alloc(f32, n) catch @panic("qwen4exp: norm cache OOM");
        quant.dequantToF32(buf, t.data_ptr, t.dtype, n);
        self.norm_cache[self.norm_cache_len] = .{ .key = key, .data = buf };
        self.norm_cache_len += 1;
        return buf.ptr;
    }

    fn groupedRms(self: *const Qwen4ExpModel, x: []const f32, w: [*]const f32, out: []f32) void {
        const e: usize = self.n_embd;
        const hc: usize = self.n_hc;
        const eps = self.rms_eps;
        var s: usize = 0;
        while (s < hc) : (s += 1) {
            const off = s * e;
            const sum_sq = math_ops.simdDotF32(x.ptr + off, x.ptr + off, e);
            const inv = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(e)) + eps);
            var i: usize = 0;
            while (i < e) : (i += 1) {
                out[off + i] = x[off + i] * inv * w[off + i];
            }
        }
    }

    /// Initialize from GGUF metadata. `tiered_cache` is unused (linear QSA KV).
    pub fn init(allocator: Allocator, f: Format, be: Backend, ctx_size: u32, kv_type_k: kv_quant.KvQuantType, kv_type_v: kv_quant.KvQuantType, tiered_cache: ?*TieredKvCache) !Qwen4ExpModel {
        _ = tiered_cache;
        var self = Qwen4ExpModel{
            .fmt = f,
            .be = be,
            .allocator = allocator,
            .kv_type_k = kv_type_k,
            .kv_type_v = kv_type_v,
        };
        const arch = f.getMetaStr("general.architecture") orelse "qwen4exp";
        if (f.getArchU32(arch, "block_count")) |v| self.n_layers = v;
        if (f.getArchU32(arch, "embedding_length")) |v| self.n_embd = v;
        if (f.getArchU32(arch, "attention.head_count")) |v| self.n_head = v;
        if (f.getArchU32(arch, "attention.head_count_kv")) |v| self.n_head_kv = v;
        if (f.getArchU32(arch, "attention.key_length")) |v| self.head_dim = v;
        if (f.getArchU32(arch, "vocab_size")) |v| self.vocab_size = v;
        if (f.getVocab()) |v| {
            if (v.len > self.vocab_size) self.vocab_size = @intCast(v.len);
        }
        if (f.getArchF32(arch, "rope.freq_base")) |v| self.rope_theta = v;
        if (f.getArchU32(arch, "rope.dimension_count")) |v| self.rope_dim = v;
        if (f.getArchF32(arch, "attention.layer_norm_rms_epsilon")) |v| self.rms_eps = v;
        if (f.getArchU32(arch, "full_attention_interval")) |v| self.full_attn_interval = v;
        if (f.getMetaU32("tokenizer.ggml.eos_token_id")) |v| self.eos_token_id = v;
        if (f.getArchU32(arch, "context_length")) |cl| self.max_seq_len = cl;
        if (ctx_size > 0) self.max_seq_len = ctx_size;

        self.n_hc = f.getArchU32(arch, "hyper_connection.count") orelse default_hc;
        self.hc_low_rank = f.getArchU32(arch, "hyper_connection.low_rank") orelse 320;
        self.n_experts = f.getArchU32(arch, "expert_count") orelse 512;
        self.n_experts_active = f.getArchU32(arch, "expert_used_count") orelse default_moe_active;
        self.expert_ff_dim = f.getArchU32(arch, "expert_feed_forward_length") orelse 640;
        self.shared_expert_ff_dim = f.getArchU32(arch, "expert_shared_feed_forward_length") orelse self.expert_ff_dim;
        if (self.n_experts_active > max_active_experts) return error.MissingTensor;

        if (f.getArchU32(arch, "ssm.conv_kernel")) |v| self.ssm_d_conv = v;
        if (f.getArchU32(arch, "ssm.state_size")) |v| self.ssm_d_state = v;
        if (f.getArchU32(arch, "ssm.group_count")) |v| self.ssm_n_group = v;
        if (f.layerTensor(0, "ssm_a")) |a_t| {
            if (a_t.n_dims >= 1 and a_t.dims[0] > 0) self.ssm_dt_rank = @intCast(a_t.dims[0]);
        } else if (f.getArchU32(arch, "ssm.time_step_rank")) |v| {
            self.ssm_dt_rank = v;
        }
        self.ssm_d_inner = f.getArchU32(arch, "ssm.inner_size") orelse self.ssm_dt_rank * self.ssm_d_state;

        self.indexer_n_head = f.getArchU32(arch, "attention.indexer.head_count") orelse 4;
        self.indexer_head_size = f.getArchU32(arch, "attention.indexer.key_length") orelse 128;
        self.indexer_top_k = f.getArchU32(arch, "attention.indexer.top_k") orelse 2048;

        self.ple_ngram = f.getArchU32(arch, "ple.ngram_size") orelse 3;
        self.ple_heads_per_ngram = f.getArchU32(arch, "ple.heads_per_ngram") orelse 8;
        self.ple_conv_kernel = f.getArchU32(arch, "ple.conv_kernel") orelse 4;
        self.ple_eos = f.getArchU32(arch, "ple.eos_token_id") orelse self.eos_token_id;
        self.ple_image_token = f.getArchU32(arch, "ple.image_token_id") orelse 0;
        self.ple_head_dim = f.getArchU32(arch, "embedding_length_per_layer_input") orelse 160;
        self.ple_n_heads = (self.ple_ngram - 1) * self.ple_heads_per_ngram;
        if (self.ple_ngram < 2 or self.ple_n_heads > max_ple_heads) return error.MissingTensor;

        self.ple_multipliers = f.getArchU64Array(arch, "ple.layer_multipliers") orelse &.{};
        self.ple_offsets = f.getArchU64Array(arch, "ple.head_offsets") orelse &.{};
        self.ple_vocabs = f.getArchU64Array(arch, "ple.head_vocab_sizes") orelse &.{};
        if (self.ple_multipliers.len < self.ple_ngram or self.ple_offsets.len < self.ple_n_heads or self.ple_vocabs.len < self.ple_n_heads)
            return error.MissingTensor;

        if (f.getArchU32Array(arch, "ple.layers")) |layers| {
            for (layers) |li| {
                if (li < 64) self.ple_layer_mask |= @as(u64, 1) << @intCast(li);
            }
        } else {
            self.ple_layer_mask = 1 << 1; // layer 1 is PLE in the 125B Flash-Next GGUF
        }

        self.compress_ratios = try allocator.alloc(u32, self.n_layers);
        errdefer self.deinit();
        if (f.getArchU32Array(arch, "attention.compress_ratios")) |arr| {
            const n = @min(arr.len, self.compress_ratios.len);
            @memcpy(self.compress_ratios[0..n], arr[0..n]);
            if (n < self.compress_ratios.len) {
                for (n..self.compress_ratios.len) |i| {
                    self.compress_ratios[i] = if (self.isQsa(@intCast(i))) 4 else 0;
                }
            }
        } else {
            for (self.compress_ratios, 0..) |*c, i| {
                c.* = if (self.isQsa(@intCast(i))) 4 else 0;
            }
        }

        try self.allocScratch(allocator);
        try self.allocStates(allocator);
        return self;
    }

    fn allocScratch(self: *Qwen4ExpModel, allocator: Allocator) !void {
        const e: usize = self.n_embd;
        const hc_dim = self.hcDim();
        const nh: usize = self.n_head;
        const nkv: usize = self.n_head_kv;
        const hd: usize = self.head_dim;
        const qd = nh * hd;
        const kvd = nkv * hd;
        const conv_ch = self.convCh();
        const d_inner: usize = self.ssm_d_inner;
        const ff = @max(self.expert_ff_dim, self.shared_expert_ff_dim);
        const qg = qd * 2;
        const scratch = @max(qg, @max(hc_dim, @max(d_inner, ff)));
        const ratio = blk: {
            var r: u32 = 4;
            for (self.compress_ratios) |c| r = @max(r, c);
            break :blk r;
        };
        const width = @min(self.max_seq_len, self.indexer_top_k + ratio -| 1);
        const n_blocks = (self.max_seq_len + ratio - 1) / ratio;
        const idx_dim: usize = self.indexer_head_size;
        const idx_qh: usize = self.indexer_n_head;

        const fields = .{
            .{ &self.hidden, e },
            .{ &self.hidden2, e },
            .{ &self.res_hc, hc_dim },
            .{ &self.hc_lo, self.hc_low_rank },
            .{ &self.hc_gate, hc_dim },
            .{ &self.hc_inject, self.n_hc },
            .{ &self.q_buf, qg },
            .{ &self.k_buf, kvd },
            .{ &self.v_buf, kvd },
            .{ &self.gate_buf, qd },
            .{ &self.attn_out, @max(qd, d_inner) },
            .{ &self.scores_buf, @max(nh * self.max_seq_len, n_blocks * idx_qh) },
            .{ &self.ff_buf1, scratch },
            .{ &self.ff_buf2, scratch },
            .{ &self.moe_out, e },
            .{ &self.router_logits, self.n_experts },
            .{ &self.logits_buf, self.vocab_size },
            .{ &self.ssm_qkv_buf, conv_ch },
            .{ &self.ssm_z_buf, d_inner },
            .{ &self.ssm_alpha_buf, self.ssm_dt_rank },
            .{ &self.ssm_beta_buf, self.ssm_dt_rank },
            .{ &self.ssm_conv_out, conv_ch },
            .{ &self.idx_q_buf, idx_qh * idx_dim },
            .{ &self.idx_k_buf, idx_dim },
            .{ &self.idx_pooled, n_blocks * idx_dim },
            .{ &self.idx_scores, @max(n_blocks, self.max_seq_len) },
            .{ &self.ple_emb, self.ple_n_heads * self.ple_head_dim },
            .{ &self.ple_key, hc_dim },
            .{ &self.ple_value, e },
            .{ &self.ple_query, hc_dim },
            .{ &self.ple_gated, hc_dim },
            .{ &self.ple_conv_in, hc_dim },
            .{ &self.ple_conv_out, hc_dim },
        };
        inline for (fields) |pair| {
            pair[0].* = try allocator.alloc(f32, pair[1]);
        }
        self.gather_k = try allocator.alloc(u8, width * kvd * 4);
        self.gather_v = try allocator.alloc(u8, width * kvd * 4);

        const hist = (self.ple_conv_kernel - 1) * self.ple_ngram;
        self.ple_conv_hist = try allocator.alloc(f32, hist * hc_dim);
        @memset(self.ple_conv_hist, 0);
        self.ple_tok_hist = try allocator.alloc(i32, self.ple_ngram - 1);
        @memset(self.ple_tok_hist, ple_null_tok);
        self.ple_rows = try allocator.alloc(u32, self.ple_n_heads);
        self.ple_conv_w = try allocator.alloc(f32, self.ple_conv_kernel * hc_dim);
    }

    fn allocStates(self: *Qwen4ExpModel, allocator: Allocator) !void {
        const nl: usize = self.n_layers;
        const conv_ch = self.convCh();
        const num_v: usize = self.ssm_dt_rank;
        const head_v: usize = self.ssm_d_inner / self.ssm_dt_rank;
        const head_k: usize = self.ssm_d_state;
        const kvd = @as(usize, self.n_head_kv) * self.head_dim;
        const idx_dim: usize = self.indexer_head_size;

        self.qsa_slot = try allocator.alloc(u8, nl);
        var n_qsa: u32 = 0;
        for (0..nl) |i| {
            if (self.isQsa(@intCast(i))) {
                self.qsa_slot[i] = @intCast(n_qsa);
                n_qsa += 1;
            } else {
                self.qsa_slot[i] = 0xff;
            }
        }
        self.n_qsa = n_qsa;

        self.conv_states = try allocator.alloc([]f32, nl);
        self.ssm_states = try allocator.alloc([]f32, nl);
        self.dn_ssm_a = try allocator.alloc([]f32, nl);
        self.dn_dt_bias = try allocator.alloc([]f32, nl);
        self.dn_conv_w = try allocator.alloc([]f32, nl);
        self.dn_ssm_norm_w = try allocator.alloc([]f32, nl);
        @memset(self.conv_states, &.{});
        @memset(self.ssm_states, &.{});
        @memset(self.dn_ssm_a, &.{});
        @memset(self.dn_dt_bias, &.{});
        @memset(self.dn_conv_w, &.{});
        @memset(self.dn_ssm_norm_w, &.{});

        self.qsa_k = try allocator.alloc([]u8, n_qsa);
        self.qsa_v = try allocator.alloc([]u8, n_qsa);
        self.qsa_idx_k = try allocator.alloc([]f32, n_qsa);
        @memset(self.qsa_k, &.{});
        @memset(self.qsa_v, &.{});
        @memset(self.qsa_idx_k, &.{});

        const kv_bytes = kv_quant.kvByteOffset(.f32, self.max_seq_len * kvd);
        for (0..n_qsa) |i| {
            self.qsa_k[i] = try allocator.alloc(u8, kv_bytes);
            self.qsa_v[i] = try allocator.alloc(u8, kv_bytes);
            self.qsa_idx_k[i] = try allocator.alloc(f32, self.max_seq_len * idx_dim);
        }

        for (0..nl) |i| {
            const li: u32 = @intCast(i);
            if (self.isQsa(li)) continue;
            self.conv_states[i] = try allocator.alloc(f32, conv_ch * (self.ssm_d_conv - 1));
            @memset(self.conv_states[i], 0);
            self.ssm_states[i] = try allocator.alloc(f32, num_v * head_v * head_k);
            @memset(self.ssm_states[i], 0);
            const ssm_a_t = try self.layerT(li, "ssm_a");
            const dt_t = try self.layerT(li, "ssm_dt.bias");
            const conv_t = try self.layerT(li, "ssm_conv1d.weight");
            const norm_t = try self.layerT(li, "ssm_norm.weight");
            self.dn_ssm_a[i] = try allocator.alloc(f32, num_v);
            self.dn_dt_bias[i] = try allocator.alloc(f32, num_v);
            self.dn_conv_w[i] = try allocator.alloc(f32, conv_ch * self.ssm_d_conv);
            self.dn_ssm_norm_w[i] = try allocator.alloc(f32, head_v);
            quant.dequantToF32(self.dn_ssm_a[i], ssm_a_t.data_ptr, ssm_a_t.dtype, num_v);
            quant.dequantToF32(self.dn_dt_bias[i], dt_t.data_ptr, dt_t.dtype, num_v);
            quant.dequantToF32(self.dn_conv_w[i], conv_t.data_ptr, conv_t.dtype, conv_ch * self.ssm_d_conv);
            quant.dequantToF32(self.dn_ssm_norm_w[i], norm_t.data_ptr, norm_t.dtype, head_v);
        }

        if (self.ple_layer_mask != 0) {
            const li: u32 = @ctz(self.ple_layer_mask);
            if (self.fmt.layerTensor(li, "ple_conv1d.weight")) |cw| {
                quant.dequantToF32(self.ple_conv_w, cw.data_ptr, cw.dtype, self.ple_conv_kernel * self.hcDim());
            }
        }
    }

    pub fn deinit(self: *Qwen4ExpModel) void {
        const a = self.allocator;
        inline for (.{
            &self.hidden,        &self.hidden2,      &self.res_hc,     &self.hc_lo,
            &self.hc_gate,       &self.hc_inject,    &self.q_buf,      &self.k_buf,
            &self.v_buf,         &self.gate_buf,     &self.attn_out,   &self.scores_buf,
            &self.ff_buf1,       &self.ff_buf2,      &self.moe_out,    &self.router_logits,
            &self.logits_buf,    &self.ssm_qkv_buf,  &self.ssm_z_buf,  &self.ssm_alpha_buf,
            &self.ssm_beta_buf,  &self.ssm_conv_out, &self.idx_q_buf,  &self.idx_k_buf,
            &self.idx_pooled,    &self.idx_scores,   &self.ple_emb,    &self.ple_key,
            &self.ple_value,     &self.ple_query,    &self.ple_gated,  &self.ple_conv_in,
            &self.ple_conv_out,  &self.ple_conv_w,   &self.ple_conv_hist,
        }) |buf| {
            if (buf.len > 0) a.free(buf.*);
        }
        if (self.gather_k.len > 0) a.free(self.gather_k);
        if (self.gather_v.len > 0) a.free(self.gather_v);
        if (self.ple_tok_hist.len > 0) a.free(self.ple_tok_hist);
        if (self.ple_rows.len > 0) a.free(self.ple_rows);
        if (self.compress_ratios.len > 0) a.free(self.compress_ratios);
        if (self.qsa_slot.len > 0) a.free(self.qsa_slot);
        for (self.conv_states) |s| if (s.len > 0) a.free(s);
        for (self.ssm_states) |s| if (s.len > 0) a.free(s);
        for (self.dn_ssm_a) |s| if (s.len > 0) a.free(s);
        for (self.dn_dt_bias) |s| if (s.len > 0) a.free(s);
        for (self.dn_conv_w) |s| if (s.len > 0) a.free(s);
        for (self.dn_ssm_norm_w) |s| if (s.len > 0) a.free(s);
        if (self.conv_states.len > 0) a.free(self.conv_states);
        if (self.ssm_states.len > 0) a.free(self.ssm_states);
        if (self.dn_ssm_a.len > 0) a.free(self.dn_ssm_a);
        if (self.dn_dt_bias.len > 0) a.free(self.dn_dt_bias);
        if (self.dn_conv_w.len > 0) a.free(self.dn_conv_w);
        if (self.dn_ssm_norm_w.len > 0) a.free(self.dn_ssm_norm_w);
        for (self.qsa_k) |s| if (s.len > 0) a.free(s);
        for (self.qsa_v) |s| if (s.len > 0) a.free(s);
        for (self.qsa_idx_k) |s| if (s.len > 0) a.free(s);
        if (self.qsa_k.len > 0) a.free(self.qsa_k);
        if (self.qsa_v.len > 0) a.free(self.qsa_v);
        if (self.qsa_idx_k.len > 0) a.free(self.qsa_idx_k);
        for (self.norm_cache[0..self.norm_cache_len]) |entry| a.free(entry.data);
    }

    pub fn cancel(self: *Qwen4ExpModel) void {
        model_mod.signalCancel(&self.cancelled);
    }

    pub fn getBlockTable(_: *Qwen4ExpModel) []const u32 {
        return &.{};
    }

    pub fn resetCache(self: *Qwen4ExpModel) void {
        self.kv_seq_len = 0;
        @memset(self.ple_tok_hist, ple_null_tok);
        @memset(self.ple_conv_hist, 0);
        for (self.conv_states) |s| if (s.len > 0) @memset(s, 0);
        for (self.ssm_states) |s| if (s.len > 0) @memset(s, 0);
        self.cancelled.store(false, .release);
    }

    fn embToken(self: *Qwen4ExpModel, tok: u32) !void {
        const t = self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        self.be.embLookup(.{ .data = t.data_ptr, .dtype = t.dtype }, tok, self.hidden.ptr, self.n_embd);
        const e: usize = self.n_embd;
        const hc: usize = self.n_hc;
        var s: usize = 0;
        while (s < hc) : (s += 1) {
            @memcpy(self.res_hc[s * e ..][0..e], self.hidden);
        }
    }

    fn hcMix(self: *Qwen4ExpModel, li: u32, comptime kind: []const u8, inject_out: ?[]f32) ![]f32 {
        const e: usize = self.n_embd;
        const hc_dim = self.hcDim();
        const lr: usize = self.hc_low_rank;
        const hc: usize = self.n_hc;
        const nw = try self.layerT(li, kind ++ "_norm.weight");
        const dw = try self.layerT(li, kind ++ "_down.weight");
        const uw = try self.layerT(li, kind ++ "_up.weight");
        self.groupedRms(self.res_hc, self.normAsF32(nw, hc_dim), self.hc_gate);
        // Inject is W @ xn, not the gated stream (llama.cpp build_hc_mix).
        if (inject_out) |inj| {
            const iw = try self.layerT(li, kind ++ "_inject.weight");
            self.gemvW(self.hc_gate.ptr, iw, inj.ptr, hc, hc_dim);
        }
        self.gemvW(self.hc_gate.ptr, dw, self.hc_lo.ptr, lr, hc_dim);
        self.be.sync();
        const inv_hc = 1.0 / @as(f32, @floatFromInt(hc));
        for (self.hc_lo[0..lr]) |*v| v.* *= inv_hc;
        self.be.silu(self.hc_lo.ptr, self.hc_lo.ptr, lr);
        self.gemvW(self.hc_lo.ptr, uw, self.ff_buf1.ptr, hc_dim, lr);
        self.be.sync();
        for (0..hc_dim) |i| {
            self.hc_gate[i] *= math_ops.sigmoid(self.ff_buf1[i]);
        }
        @memset(self.hidden2[0..e], 0);
        var s: usize = 0;
        while (s < hc) : (s += 1) {
            const off = s * e;
            var i: usize = 0;
            while (i < e) : (i += 1) self.hidden2[i] += self.hc_gate[off + i];
        }
        const inv = 1.0 / @as(f32, @floatFromInt(hc));
        for (self.hidden2[0..e]) |*v| v.* *= inv;
        return self.hidden2;
    }

    fn hcMixHead(self: *Qwen4ExpModel) !void {
        const e: usize = self.n_embd;
        const hc_dim = self.hcDim();
        const lr: usize = self.hc_low_rank;
        const hc: usize = self.n_hc;
        const nw = self.fmt.getTensor("output_hc_norm.weight") orelse return error.MissingTensor;
        const dw = self.fmt.getTensor("output_hc_down.weight") orelse return error.MissingTensor;
        const uw = self.fmt.getTensor("output_hc_up.weight") orelse return error.MissingTensor;
        self.groupedRms(self.res_hc, self.normAsF32(nw, hc_dim), self.hc_gate);
        self.gemvW(self.hc_gate.ptr, dw, self.hc_lo.ptr, lr, hc_dim);
        self.be.sync();
        const inv_hc = 1.0 / @as(f32, @floatFromInt(hc));
        for (self.hc_lo[0..lr]) |*v| v.* *= inv_hc;
        self.be.silu(self.hc_lo.ptr, self.hc_lo.ptr, lr);
        self.gemvW(self.hc_lo.ptr, uw, self.ff_buf1.ptr, hc_dim, lr);
        self.be.sync();
        for (0..hc_dim) |i| {
            self.hc_gate[i] *= math_ops.sigmoid(self.ff_buf1[i]);
        }
        @memset(self.hidden[0..e], 0);
        var s: usize = 0;
        while (s < hc) : (s += 1) {
            const off = s * e;
            var i: usize = 0;
            while (i < e) : (i += 1) self.hidden[i] += self.hc_gate[off + i];
        }
        const inv = 1.0 / @as(f32, @floatFromInt(hc));
        for (self.hidden[0..e]) |*v| v.* *= inv;
    }

    fn hcCombine(self: *Qwen4ExpModel, block_out: []const f32, inject: []const f32) void {
        const e: usize = self.n_embd;
        const hc: usize = self.n_hc;
        const inv_hc = 1.0 / @as(f32, @floatFromInt(hc));
        var s: usize = 0;
        while (s < hc) : (s += 1) {
            const w = 2.0 * math_ops.sigmoid(inject[s] * inv_hc);
            const off = s * e;
            var i: usize = 0;
            while (i < e) : (i += 1) self.res_hc[off + i] += block_out[i] * w;
        }
    }

    fn applyPle(self: *Qwen4ExpModel, li: u32, tok: u32) !void {
        const e: usize = self.n_embd;
        const hc_dim = self.hcDim();
        const n_heads: usize = self.ple_n_heads;
        const hd: usize = self.ple_head_dim;
        const table = self.fmt.getTensor("per_layer_token_embd.weight") orelse return error.MissingTensor;
        const hash_tok: u32 = if (self.ple_image_token != 0 and tok == self.ple_image_token)
            self.ple_image_token
        else
            tok;
        pleRowIndices(
            hash_tok,
            self.ple_tok_hist,
            self.ple_ngram,
            self.ple_heads_per_ngram,
            self.ple_multipliers,
            self.ple_offsets,
            self.ple_vocabs,
            self.ple_eos,
            self.ple_rows,
        );
        var h: usize = 0;
        while (h < n_heads) : (h += 1) {
            self.cpu.embLookup(.{ .data = table.data_ptr, .dtype = table.dtype }, self.ple_rows[h], self.ple_emb.ptr + h * hd, hd);
        }
        const key_w = try self.layerT(li, "ple_key.weight");
        const val_w = try self.layerT(li, "ple_value.weight");
        self.gemvW(self.ple_emb.ptr, key_w, self.ple_key.ptr, hc_dim, n_heads * hd);
        self.gemvW(self.ple_emb.ptr, val_w, self.ple_value.ptr, e, n_heads * hd);
        self.be.sync();
        const kn = try self.layerT(li, "ple_norm_key.weight");
        const qn = try self.layerT(li, "ple_norm_query.weight");
        const cn = try self.layerT(li, "ple_norm_conv.weight");
        self.groupedRms(self.ple_key, self.normAsF32(kn, hc_dim), self.ple_key);
        self.groupedRms(self.res_hc, self.normAsF32(qn, hc_dim), self.ple_query);
        const inv_sqrt = 1.0 / @sqrt(@as(f32, @floatFromInt(e)));
        const hc: usize = self.n_hc;
        var s: usize = 0;
        while (s < hc) : (s += 1) {
            const off = s * e;
            const dot = math_ops.simdDotF32(self.ple_key.ptr + off, self.ple_query.ptr + off, e) * inv_sqrt;
            const mag = @sqrt(@max(@abs(dot), ple_gate_floor));
            const sgn: f32 = if (dot > 0) 1.0 else if (dot < 0) -1.0 else 0.0;
            const g = math_ops.sigmoid(sgn * mag);
            var i: usize = 0;
            while (i < e) : (i += 1) self.ple_gated[off + i] = self.ple_value[i] * g;
        }
        self.groupedRms(self.ple_gated, self.normAsF32(cn, hc_dim), self.ple_conv_in);
        const kern: usize = self.ple_conv_kernel;
        const dil: usize = self.ple_ngram;
        const hist = (kern - 1) * dil;
        @memset(self.ple_conv_out[0..hc_dim], 0);
        var k: usize = 0;
        while (k < kern) : (k += 1) {
            const start = hist - (kern - 1 - k) * dil;
            const src: [*]const f32 = if (start < hist)
                self.ple_conv_hist.ptr + start * hc_dim
            else
                self.ple_conv_in.ptr;
            var c: usize = 0;
            while (c < hc_dim) : (c += 1) {
                const w = self.ple_conv_w[c * kern + k];
                self.ple_conv_out[c] += w * src[c];
            }
        }
        self.be.silu(self.ple_conv_out.ptr, self.ple_conv_out.ptr, hc_dim);
        if (hist > 0) {
            if (hist > 1) {
                std.mem.copyForwards(f32, self.ple_conv_hist[0 .. (hist - 1) * hc_dim], self.ple_conv_hist[hc_dim..hist * hc_dim]);
            }
            @memcpy(self.ple_conv_hist[(hist - 1) * hc_dim ..][0..hc_dim], self.ple_conv_in);
        }
        var i: usize = 0;
        while (i < hc_dim) : (i += 1) {
            self.res_hc[i] += self.ple_gated[i] + self.ple_conv_out[i];
        }
    }

    fn shiftPleHist(self: *Qwen4ExpModel, tok: u32) void {
        if (self.ple_tok_hist.len == 0) return;
        var i: usize = 0;
        while (i + 1 < self.ple_tok_hist.len) : (i += 1) {
            self.ple_tok_hist[i] = self.ple_tok_hist[i + 1];
        }
        self.ple_tok_hist[self.ple_tok_hist.len - 1] = @intCast(tok);
    }

    fn gdnLayer(self: *Qwen4ExpModel, li: u32) !void {
        const e: usize = self.n_embd;
        const conv_ch = self.convCh();
        const d_inner: usize = self.ssm_d_inner;
        const num_v: usize = self.ssm_dt_rank;
        const num_k: usize = self.ssm_n_group;
        const head_k: usize = self.ssm_d_state;
        const head_v: usize = d_inner / num_v;
        const qkv_w = try self.layerT(li, "attn_qkv.weight");
        const gate_w = try self.layerT(li, "attn_gate.weight");
        const alpha_w = try self.layerT(li, "ssm_alpha.weight");
        const beta_w = try self.layerT(li, "ssm_beta.weight");
        self.gemvW(self.hidden2.ptr, qkv_w, self.ssm_qkv_buf.ptr, conv_ch, e);
        self.gemvW(self.hidden2.ptr, gate_w, self.ssm_z_buf.ptr, d_inner, e);
        self.gemvW(self.hidden2.ptr, alpha_w, self.ssm_alpha_buf.ptr, num_v, e);
        self.gemvW(self.hidden2.ptr, beta_w, self.ssm_beta_buf.ptr, num_v, e);
        self.be.sync();
        const q_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_k)));
        self.cpu.deltaNet(
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
                .d_conv = self.ssm_d_conv,
                .d_inner = @intCast(d_inner),
                .num_k_heads = @intCast(num_k),
                .head_k_dim = @intCast(head_k),
                .num_v_heads = @intCast(num_v),
                .head_v_dim = @intCast(head_v),
                .q_scale = q_scale,
                .rms_eps = self.rms_eps,
                .kqv_order = false,
                .out_gate_sigmoid = true,
            },
        );
        const out_w = try self.layerT(li, "ssm_out.weight");
        self.gemvW(self.attn_out.ptr, out_w, self.hidden2.ptr, e, d_inner);
        self.be.sync();
    }

    fn qsaLayer(self: *Qwen4ExpModel, li: u32) !void {
        const e: usize = self.n_embd;
        const nh: usize = self.n_head;
        const nkv: usize = self.n_head_kv;
        const hd: usize = self.head_dim;
        const qd = nh * hd;
        const kvd = nkv * hd;
        const slot: usize = self.qsa_slot[li];
        const qw = try self.layerT(li, "attn_q.weight");
        const kw = try self.layerT(li, "attn_k.weight");
        const vw = try self.layerT(li, "attn_v.weight");
        self.gemvW(self.hidden2.ptr, qw, self.q_buf.ptr, qd * 2, e);
        self.gemvW(self.hidden2.ptr, kw, self.k_buf.ptr, kvd, e);
        self.gemvW(self.hidden2.ptr, vw, self.v_buf.ptr, kvd, e);
        self.be.sync();
        // GGUF qwen4exp packs Q then gate per head (HF splitQGate layout).
        self.be.splitQGate(self.q_buf.ptr, self.ff_buf2.ptr, self.gate_buf.ptr, hd, nh);
        const q_ptr: [*]f32 = self.ff_buf2.ptr;
        const qnw = try self.layerT(li, "attn_q_norm.weight");
        const knw = try self.layerT(li, "attn_k_norm.weight");
        self.be.rmsNormMulti(q_ptr, self.normAsF32(qnw, hd), nh, hd, self.rms_eps);
        self.be.rmsNormMulti(self.k_buf.ptr, self.normAsF32(knw, hd), nkv, hd, self.rms_eps);
        self.be.rope(q_ptr, self.kv_seq_len, nh, hd, self.rope_dim, self.rope_theta);
        self.be.rope(self.k_buf.ptr, self.kv_seq_len, nkv, hd, self.rope_dim, self.rope_theta);

        const ratio = self.compressRatio(li);
        const n_after = self.kv_seq_len + 1;
        const width_cap = self.indexer_top_k + ratio -| 1;
        const use_sparse = ratio > 0 and n_after > width_cap;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        if (!use_sparse) {
            attn_ops.scaledDotProductAttention(
                q_ptr,
                self.qsa_k[slot],
                self.qsa_v[slot],
                self.k_buf,
                self.v_buf,
                self.attn_out.ptr,
                self.scores_buf.ptr,
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
        } else {
            try self.qsaSparse(li, slot, q_ptr, scale, ratio, n_after, width_cap);
        }
        self.be.sigmoidMul(self.attn_out.ptr, self.gate_buf.ptr, qd);
        const ow = try self.layerT(li, "attn_output.weight");
        self.gemvW(self.attn_out.ptr, ow, self.hidden2.ptr, e, qd);
        self.be.sync();
    }

    fn qsaSparse(self: *Qwen4ExpModel, li: u32, slot: usize, q_ptr: [*]const f32, scale: f32, ratio: u32, n_after: usize, width_cap: u32) !void {
        const nkv: usize = self.n_head_kv;
        const hd: usize = self.head_dim;
        const kvd = nkv * hd;
        const idx_dim: usize = self.indexer_head_size;
        const n_idx_h: usize = self.indexer_n_head;
        const pos = self.kv_seq_len;
        const kq = try self.layerT(li, "indexer.k_proj.weight");
        const qq = try self.layerT(li, "indexer.q_proj.weight");
        const kn = try self.layerT(li, "indexer.k_norm.weight");
        const qn = try self.layerT(li, "indexer.q_norm.weight");
        self.gemvW(self.hidden2.ptr, kq, self.idx_k_buf.ptr, idx_dim, self.n_embd);
        self.be.sync();
        @memcpy(self.qsa_idx_k[slot][pos * idx_dim ..][0..idx_dim], self.idx_k_buf[0..idx_dim]);
        kv_quant.kvStore(self.qsa_k[slot][kv_quant.kvByteOffset(.f32, pos * kvd)..].ptr, self.k_buf.ptr, kvd, .f32);
        kv_quant.kvStore(self.qsa_v[slot][kv_quant.kvByteOffset(.f32, pos * kvd)..].ptr, self.v_buf.ptr, kvd, .f32);

        const n_blocks = (n_after + ratio - 1) / ratio;
        const knw = self.normAsF32(kn, idx_dim);
        var b: usize = 0;
        while (b < n_blocks) : (b += 1) {
            const start = b * ratio;
            const end = @min(start + ratio, n_after);
            const count = end - start;
            @memset(self.idx_pooled[b * idx_dim ..][0..idx_dim], 0);
            var t: usize = start;
            while (t < end) : (t += 1) {
                const src = self.qsa_idx_k[slot][t * idx_dim ..][0..idx_dim];
                var d: usize = 0;
                while (d < idx_dim) : (d += 1) self.idx_pooled[b * idx_dim + d] += src[d];
            }
            const inv = 1.0 / @as(f32, @floatFromInt(count));
            var d: usize = 0;
            while (d < idx_dim) : (d += 1) self.idx_pooled[b * idx_dim + d] *= inv;
            self.be.rmsNorm(self.idx_pooled.ptr + b * idx_dim, knw, self.idx_pooled.ptr + b * idx_dim, idx_dim, self.rms_eps);
            self.be.rope(self.idx_pooled.ptr + b * idx_dim, start, 1, idx_dim, self.rope_dim, self.rope_theta);
        }
        self.gemvW(self.hidden2.ptr, qq, self.idx_q_buf.ptr, n_idx_h * idx_dim, self.n_embd);
        self.be.sync();
        self.be.rmsNormMulti(self.idx_q_buf.ptr, self.normAsF32(qn, idx_dim), n_idx_h, idx_dim, self.rms_eps);
        self.be.rope(self.idx_q_buf.ptr, pos, n_idx_h, idx_dim, self.rope_dim, self.rope_theta);

        b = 0;
        while (b < n_blocks) : (b += 1) {
            var acc: f32 = 0;
            var h: usize = 0;
            while (h < n_idx_h) : (h += 1) {
                const dot = math_ops.simdDotF32(self.idx_q_buf.ptr + h * idx_dim, self.idx_pooled.ptr + b * idx_dim, idx_dim);
                acc += @max(dot, 0.0);
            }
            self.idx_scores[b] = acc;
        }
        const width: usize = @min(n_after, width_cap);
        var selected: [qsa_select_cap]usize = undefined;
        std.debug.assert(width <= selected.len);
        // Force-include the current token, then fill remaining slots by block score.
        selected[width - 1] = pos;
        var filled: usize = 0;
        while (filled + 1 < width) : (filled += 1) {
            var best: usize = 0;
            var best_s: f32 = -std.math.inf(f32);
            var t: usize = 0;
            while (t < n_after) : (t += 1) {
                if (t == pos) continue;
                var dup = false;
                for (selected[0..filled]) |s| if (s == t) {
                    dup = true;
                    break;
                };
                if (dup) continue;
                const sc = self.idx_scores[t / ratio];
                if (sc > best_s) {
                    best_s = sc;
                    best = t;
                }
            }
            selected[filled] = best;
        }
        std.mem.sort(usize, selected[0..width], {}, std.sort.asc(usize));
        const keys_f: [*]f32 = @ptrCast(@alignCast(self.gather_k.ptr));
        const vals_f: [*]f32 = @ptrCast(@alignCast(self.gather_v.ptr));
        const src_k: [*]const f32 = @ptrCast(@alignCast(self.qsa_k[slot].ptr));
        const src_v: [*]const f32 = @ptrCast(@alignCast(self.qsa_v[slot].ptr));
        var gi: usize = 0;
        while (gi < width) : (gi += 1) {
            const t = selected[gi];
            @memcpy(keys_f[gi * kvd ..][0..kvd], src_k[t * kvd ..][0..kvd]);
            @memcpy(vals_f[gi * kvd ..][0..kvd], src_v[t * kvd ..][0..kvd]);
        }
        // Attend over gathered prefix; last gathered row is treated as k_new/v_new.
        const last = width - 1;
        const k_off = last * kvd;
        const k_new: []const f32 = keys_f[k_off .. k_off + kvd];
        const v_new: []const f32 = vals_f[k_off .. k_off + kvd];
        attn_ops.scaledDotProductAttention(
            q_ptr,
            self.gather_k,
            self.gather_v,
            k_new,
            v_new,
            self.attn_out.ptr,
            self.scores_buf.ptr,
            self.n_head,
            nkv,
            hd,
            last,
            scale,
            self.be,
            null,
            0,
            .f32,
            .f32,
        );
    }

    fn moeLayer(self: *Qwen4ExpModel, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.expert_ff_dim;
        const n_exp: usize = self.n_experts;
        const n_active: usize = self.n_experts_active;
        const rw = try self.layerT(li, "ffn_gate_inp.weight");
        self.gemvW(self.hidden2.ptr, rw, self.router_logits.ptr, n_exp, e);
        self.be.sync();
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

        var top_experts: [max_active_experts]usize = undefined;
        var top_scores: [max_active_experts]f32 = undefined;
        math_ops.topKExperts(self.router_logits[0..n_exp], n_active, top_experts[0..n_active], top_scores[0..n_active]);
        var sel_sum: f32 = 0.0;
        for (0..n_active) |i| sel_sum += top_scores[i];
        const inv_sel = if (sel_sum > 0.0) 1.0 / sel_sum else 0.0;
        for (0..n_active) |i| top_scores[i] *= inv_sel;

        const gate_exps = try self.layerT(li, "ffn_gate_exps.weight");
        const up_exps = try self.layerT(li, "ffn_up_exps.weight");
        const down_exps = try self.layerT(li, "ffn_down_exps.weight");
        const gate_stride = expertWeightStride(gate_exps);
        const up_stride = expertWeightStride(up_exps);
        const down_stride = expertWeightStride(down_exps);
        @memset(self.moe_out[0..e], 0);
        for (0..n_active) |ti| {
            const ei = top_experts[ti];
            self.gemvExpert(self.hidden2.ptr, gate_exps, ei, gate_stride, self.ff_buf1.ptr, ff, e);
            self.gemvExpert(self.hidden2.ptr, up_exps, ei, up_stride, self.ff_buf2.ptr, ff, e);
            self.be.siluMul(self.ff_buf1.ptr, self.ff_buf2.ptr, self.ff_buf1.ptr, ff);
            self.gemvExpert(self.ff_buf1.ptr, down_exps, ei, down_stride, self.attn_out.ptr, e, ff);
            self.be.sync();
            self.be.addScaled(self.attn_out.ptr, self.moe_out.ptr, top_scores[ti], e);
        }
        const sg = try self.layerT(li, "ffn_gate_shexp.weight");
        const su = try self.layerT(li, "ffn_up_shexp.weight");
        const sd = try self.layerT(li, "ffn_down_shexp.weight");
        const shared_ff: usize = self.shared_expert_ff_dim;
        self.gemvW(self.hidden2.ptr, sg, self.ff_buf1.ptr, shared_ff, e);
        self.gemvW(self.hidden2.ptr, su, self.ff_buf2.ptr, shared_ff, e);
        self.be.siluMul(self.ff_buf1.ptr, self.ff_buf2.ptr, self.ff_buf1.ptr, shared_ff);
        self.gemvW(self.ff_buf1.ptr, sd, self.attn_out.ptr, e, shared_ff);
        self.be.sync();
        var gate_val: f32 = 1.0;
        if (self.fmt.layerTensor(li, "ffn_gate_inp_shexp.weight")) |gw| {
            if (gw.dtype == .f32) {
                const gp: [*]const f32 = @ptrCast(@alignCast(gw.data_ptr));
                gate_val = math_ops.sigmoid(math_ops.simdDotF32(gp, self.hidden2.ptr, e));
            } else {
                self.gemvW(self.hidden2.ptr, gw, self.hc_inject.ptr, 1, e);
                self.be.sync();
                gate_val = math_ops.sigmoid(self.hc_inject[0]);
            }
        }
        self.be.addScaled(self.attn_out.ptr, self.moe_out.ptr, gate_val, e);
        @memcpy(self.hidden2[0..e], self.moe_out[0..e]);
    }

    fn step(self: *Qwen4ExpModel, token_id: u32, want_logits: bool) !u32 {
        if (self.cancelled.load(.monotonic)) return error.Cancelled;
        if (self.kv_seq_len >= self.max_seq_len) return error.KVCacheFull;
        self.cpu.pool = self.pool;
        try self.embToken(token_id);
        var li: u32 = 0;
        while (li < self.n_layers) : (li += 1) {
            if (self.cancelled.load(.monotonic)) return error.Cancelled;
            self.fmt.prefetchLayer(li + 1);
            if (self.isPle(li)) try self.applyPle(li, token_id);
            _ = try self.hcMix(li, "hc_attn", self.hc_inject);
            if (self.isQsa(li)) {
                try self.qsaLayer(li);
            } else {
                try self.gdnLayer(li);
            }
            self.hcCombine(self.hidden2, self.hc_inject);
            _ = try self.hcMix(li, "hc_ffn", self.hc_inject);
            try self.moeLayer(li);
            self.hcCombine(self.hidden2, self.hc_inject);
        }
        self.shiftPleHist(token_id);
        self.kv_seq_len += 1;
        if (!want_logits) return 0;
        try self.hcMixHead();
        const ow = self.fmt.getTensor("output.weight") orelse self.fmt.getTensor("token_embd.weight") orelse return error.MissingTensor;
        self.gemvW(self.hidden.ptr, ow, self.logits_buf.ptr, self.vocab_size, self.n_embd);
        self.be.sync();
        return math_ops.argmax(self.logits_buf);
    }

    pub fn forward(self: *Qwen4ExpModel, token_id: u32) !u32 {
        return self.step(token_id, true);
    }

    pub fn prefill(self: *Qwen4ExpModel, token_ids: []const u32) !u32 {
        if (token_ids.len == 0) return 0;
        var i: usize = 0;
        while (i + 1 < token_ids.len) : (i += 1) {
            _ = try self.step(token_ids[i], false);
        }
        return self.step(token_ids[token_ids.len - 1], true);
    }
};

test "pleRowIndices empty window hashes current against EOS" {
    const n_gram: u32 = 3;
    const hpn: u32 = 2;
    const eos: u32 = 7;
    const multipliers = [_]u64{ 3, 5, 11 };
    const offsets = [_]u64{ 0, 10, 20, 30 };
    const vocabs = [_]u64{ 10, 10, 10, 10 };
    const prev = [_]i32{ ple_null_tok, ple_null_tok };
    var out: [4]u32 = undefined;
    pleRowIndices(4, &prev, n_gram, hpn, &multipliers, &offsets, &vocabs, eos, &out);
    // n=2: 4*3 xor 7*5 = 12 xor 35 = 39. n=3 also xors 7*11 → 98.
    const mixed2: u64 = (@as(u64, 4) *% 3) ^ (@as(u64, 7) *% 5);
    const mixed3: u64 = mixed2 ^ (@as(u64, 7) *% 11);
    try std.testing.expectEqual(@as(u32, @intCast(mixed2 % 10 + 0)), out[0]);
    try std.testing.expectEqual(@as(u32, @intCast(mixed2 % 10 + 10)), out[1]);
    try std.testing.expectEqual(@as(u32, @intCast(mixed3 % 10 + 20)), out[2]);
    try std.testing.expectEqual(@as(u32, @intCast(mixed3 % 10 + 30)), out[3]);
}

test "pleRowIndices predecessor EOS cuts older context" {
    const multipliers = [_]u64{ 3, 5, 11 };
    const offsets = [_]u64{ 0, 10, 20, 30 };
    const vocabs = [_]u64{ 10, 10, 10, 10 };
    const prev = [_]i32{ 9, 7 }; // oldest=9, newest predecessor=EOS
    var out: [4]u32 = undefined;
    pleRowIndices(4, &prev, 3, 2, &multipliers, &offsets, &vocabs, 7, &out);
    // ctx = [4, EOS, EOS] because predecessor EOS cuts
    var out2: [4]u32 = undefined;
    const prev_empty = [_]i32{ ple_null_tok, ple_null_tok };
    pleRowIndices(4, &prev_empty, 3, 2, &multipliers, &offsets, &vocabs, 7, &out2);
    try std.testing.expectEqualSlices(u32, &out2, &out);
}

test "hcCombine zero inject is residual add" {
    var res = [_]f32{ 1, 2, 3, 4 };
    const block = [_]f32{ 10, 20 };
    const inject = [_]f32{ 0, 0 };
    // 2*sigmoid(0) = 1, so each stream gets +block
    const e: usize = 2;
    const hc: usize = 2;
    const inv_hc = 0.5;
    var s: usize = 0;
    while (s < hc) : (s += 1) {
        const w = 2.0 * math_ops.sigmoid(inject[s] * inv_hc);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), w, 1e-5);
        const off = s * e;
        var i: usize = 0;
        while (i < e) : (i += 1) res[off + i] += block[i] * w;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 11), res[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 22), res[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 13), res[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 24), res[3], 1e-5);
}
