//! DFlash2 block-diffusion draft model (Inco AI "DFlash 2"; z-lab checkpoints).
//!
//! Predicts a whole block of draft tokens in ONE parallel forward pass:
//! slot 0 carries the last verified token, remaining slots carry mask-token
//! embeddings. Attention is bidirectional within the block (config
//! `is_causal: false`) over two key sets per layer:
//!   1. injected context: fused target-model hidden features projected through
//!      each layer's own k/v weights into a rotating KV cache (one entry per
//!      processed context position, bounded by `sliding_window - 1`), and
//!   2. ephemeral block keys (never written to the cache).
//! Two-tap grouped dynamic depthwise convolutions wrap every attention and MLP
//! sublayer, and a lightweight candidate path selector picks one coherent
//! token per slot from the per-slot top-K vocabulary candidates.
//!
//! The checkpoint ships no embeddings or LM head; both bind from the TARGET
//! model at runtime (`bindTarget`), matching the reference implementation.
//!
//! Weights load through the standard quantization-aware storage path: any GGUF
//! K-quant, bf16/f16/f32 safetensors, NVFP4/MXFP4/MLX work via
//! `dispatchGemv` (dequantization happens inside the kernel).

const std = @import("std");
const math_ops = @import("../ops/math.zig");
const quant = @import("../ops/quant.zig");
const kv_quant = @import("../ops/kv_quant.zig");
const backend_mod = @import("../backend/backend.zig");
const format_mod = @import("../format/format.zig");
const model_mod = @import("model.zig");
const dflash2_alg = @import("../spec/dflash2.zig");

const Backend = backend_mod.Backend;
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const Allocator = std.mem.Allocator;
const ThreadPool = @import("../thread_pool.zig").ThreadPool;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;

/// Largest supported draft block (slots incl. anchor). Published checkpoints
/// use 8; headroom avoids reallocating fixed scratch for future blocks.
const max_block_size: usize = 64;

/// DFlash2 block-diffusion drafter.
pub const DFlash2Model = struct {
    fmt: Format,
    be: Backend,
    allocator: Allocator,

    // ── Geometry (from config.json / GGUF metadata) ──────────────
    n_layers: u32 = 5,
    n_embd: u32 = 5120,
    n_head: u32 = 32,
    n_head_kv: u32 = 8,
    head_dim: u32 = 128,
    n_ff: u32 = 17408,
    vocab_size: u32 = 248320,
    rope_theta: f32 = 10000000.0,
    rms_eps: f32 = 1e-6,
    sliding_window: u32 = 2048,
    block_size: u32 = 8,
    mask_token_id: u32 = 248070,
    conv_kernel: u32 = 2,
    conv_group: u32 = 16,
    sel_rank: u32 = 256,
    sel_top_k: u32 = 16,
    /// Target-model layers whose hidden states feed the context injection.
    target_layer_ids: []u32 = &.{},
    input_embedding_scale: f32 = 1.0,
    output_multiplier: f32 = 1.0,
    softcap: f32 = 0, // 0 disables final_logit_softcapping
    eos_token_id: u32 = 248046,
    max_seq_len: usize = 262144,

    /// True when loaded from SafeTensors (HF RMSNorm weights need +1 baked).
    is_safetensors: bool = false,
    /// True when weights are MLX-quantized (norm shifts already baked).
    is_mlx: bool = false,

    // ── Required Model-interface fields ──────────────────────────
    /// Mirrors ingested context position count; external writes (specDecode
    /// finishRound syncing an AR draft KV cache) are ignored by this model.
    kv_seq_len: usize = 0,
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    logits_buf: []f32 = &.{},
    pool: ?*ThreadPool = null,

    // ── Target binding (embeddings + LM head live in the target) ─
    target_emb: ?TensorInfo = null,
    target_lm_head: ?TensorInfo = null,
    target_fmt: ?Format = null,

    // ── Rotating injected-context KV cache ───────────────────────
    cap: usize = 0, // entries retained per layer (= sliding_window - 1)
    ctx_k: [][]f32 = &.{}, // [layer][cap * kvd]
    ctx_v: [][]f32 = &.{},
    /// Number of context positions ingested (absolute position counter).
    ctx_count: usize = 0,

    // ── Dequantized small weights (init-time; hot path never allocates) ──
    norm_fc: []f32 = &.{},
    norm_final: []f32 = &.{},
    norm_input: [][]f32 = &.{},
    norm_post: [][]f32 = &.{},
    norm_q: [][]f32 = &.{},
    norm_k: [][]f32 = &.{},
    /// Convolution base kernels, dequantized: [layer][stage][K*E].
    conv_base: [][]f32 = &.{},

    // ── Block scratch buffers ([block_size][dim], allocated at init) ────
    blk_x: []f32 = &.{}, // residual stream [S*e]
    blk_h: []f32 = &.{}, // normed input / o_proj output [S*e]
    blk_q: []f32 = &.{}, // queries [S*qd]
    blk_k: []f32 = &.{}, // block keys [S*kvd]
    blk_v: []f32 = &.{}, // block values [S*kvd]
    blk_attn: []f32 = &.{}, // attention scores out / MLP output [S*max(qd,e)]
    cnv_a: []f32 = &.{}, // tap-0 conv output [S*e]
    cnv_b: []f32 = &.{}, // tap-1 conv output [S*e]
    conv_dyn0: []f32 = &.{}, // tap-0 dynamic coefficients [S*K*groups]
    conv_dyn1: []f32 = &.{}, // tap-1 dynamic coefficients [S*K*groups]
    ff_gate: []f32 = &.{}, // MLP gate activations [S*ff]
    ff_up: []f32 = &.{}, // MLP up activations [S*ff]
    attn_scores: []f32 = &.{}, // SDPA row scratch [(cap+S) capped]

    // ── Selector scratch ─────────────────────────────────────────
    sel_hid: []f32 = &.{}, // projected hidden [S*rank]
    sel_cand: []u32 = &.{}, // candidate ids [S*K]
    sel_unary: []f32 = &.{}, // candidate logits [S*K]
    sel_edge: []f32 = &.{}, // edge score scratch [K]
    sel_g: []f32 = &.{}, // gated predecessor embedding [rank]
    sel_b_row: []f32 = &.{}, // dequantized successor row [rank]
    sel_a_row: []f32 = &.{}, // dequantized predecessor row [rank]

    // ── Ingestion scratch ────────────────────────────────────────
    ing_fused: []f32 = &.{}, // fused context feature [e]
    ing_k: []f32 = &.{}, // context key [kvd]
    ing_v: []f32 = &.{}, // context value [kvd]

    /// Returns the generic Model interface for this DFlash2 instance.
    pub fn model(self: *DFlash2Model) model_mod.Model {
        return model_mod.Model.from(DFlash2Model, self);
    }

    fn metaFirst(comptime T: type, f: Format, keys: []const []const u8) ?T {
        for (keys) |k| {
            const v: ?T = switch (T) {
                u32 => f.getMetaU32(k),
                f32 => f.getMetaF32(k),
                else => unreachable,
            };
            if (v) |val| return val;
        }
        return null;
    }

    /// Initialize from format metadata + weights. Does not touch KV state.
    /// On any error everything allocated so far is released.
    /// The KV-type / tiered-cache parameters are accepted for uniformity with
    /// ModelStorage.initFromArch; the drafter keeps its own f32 context cache.
    pub fn init(allocator: Allocator, f: Format, be: Backend, ctx_size: u32, kv_type_k: kv_quant.KvQuantType, kv_type_v: kv_quant.KvQuantType, tiered_cache: ?*TieredKvCache) !DFlash2Model {
        _ = kv_type_k;
        _ = kv_type_v;
        _ = tiered_cache;
        var self = DFlash2Model{
            .fmt = f,
            .be = be,
            .allocator = allocator,
            .is_safetensors = f.is_safetensors,
        };
        errdefer self.deinit();

        self.n_layers = metaFirst(u32, f, &.{ "dflash.num_layers", "dflash2.block_count", "num_hidden_layers" }) orelse self.n_layers;
        self.n_embd = metaFirst(u32, f, &.{ "dflash.hidden_size", "hidden_size" }) orelse self.n_embd;
        self.n_head = metaFirst(u32, f, &.{ "dflash.num_attention_heads", "num_attention_heads" }) orelse self.n_head;
        self.n_head_kv = metaFirst(u32, f, &.{ "dflash.num_key_value_heads", "num_key_value_heads" }) orelse self.n_head_kv;
        if (metaFirst(u32, f, &.{ "dflash.head_dim", "head_dim", "attention.key_length" })) |v| {
            self.head_dim = v;
        } else if (self.n_head > 0 and self.n_embd % self.n_head == 0) {
            self.head_dim = self.n_embd / self.n_head;
        }
        self.n_ff = metaFirst(u32, f, &.{ "dflash.intermediate_size", "intermediate_size" }) orelse self.n_ff;
        self.vocab_size = metaFirst(u32, f, &.{ "dflash.vocab_size", "vocab_size" }) orelse self.vocab_size;
        if (metaFirst(f32, f, &.{ "rope_theta", "dflash2.rope.freq_base" })) |v| self.rope_theta = v;
        if (metaFirst(f32, f, &.{ "rms_norm_eps", "dflash2.attention.layer_norm_rms_epsilon" })) |v| self.rms_eps = v;
        self.sliding_window = metaFirst(u32, f, &.{"sliding_window"}) orelse self.sliding_window;
        self.block_size = metaFirst(u32, f, &.{ "dflash.block_size", "dflash2.block_size" }) orelse self.block_size;
        self.mask_token_id = metaFirst(u32, f, &.{ "dflash.mask_token_id", "dflash2.mask_token_id" }) orelse self.mask_token_id;
        self.conv_kernel = metaFirst(u32, f, &.{ "dflash.conv_kernel_size", "dflash2.conv_kernel_size" }) orelse self.conv_kernel;
        self.conv_group = metaFirst(u32, f, &.{ "dflash.conv_group_size", "dflash2.conv_group_size" }) orelse self.conv_group;
        self.sel_rank = metaFirst(u32, f, &.{ "dflash.selector_rank", "dflash2.selector_rank" }) orelse self.sel_rank;
        self.sel_top_k = metaFirst(u32, f, &.{ "dflash.selector_top_k", "dflash2.selector_top_k" }) orelse self.sel_top_k;
        if (metaFirst(u32, f, &.{"max_position_embeddings"})) |v| self.max_seq_len = v;
        if (ctx_size > 0) self.max_seq_len = @min(self.max_seq_len, ctx_size);
        if (metaFirst(f32, f, &.{ "dflash.input_embedding_scale", "dflash2.input_embedding_scale" })) |v| self.input_embedding_scale = v;
        if (metaFirst(f32, f, &.{ "dflash.output_multiplier", "dflash2.output_multiplier" })) |v| self.output_multiplier = v;
        if (metaFirst(f32, f, &.{ "dflash.final_logit_softcapping", "final_logit_softcapping" })) |v| self.softcap = v;
        if (f.getMetaU32("tokenizer.ggml.eos_token_id")) |v| self.eos_token_id = v;

        // Causality: published DFlash2 checkpoints declare is_causal=false
        // (bools surface through getMetaU32 as 0/1). A causal (DFlash1-style)
        // drafter cannot run through this path — fail loudly rather than
        // drafting silently wrong.
        if (f.getMetaU32("is_causal")) |causal| {
            if (causal != 0) {
                std.log.err("dflash2: checkpoint requests causal drafting (DFlash1 behavior); unsupported", .{});
                return error.MissingTensor;
            }
        }

        // MLX detection: any mlx_q probe tensor marks the whole checkpoint.
        {
            const probes = [_][]const u8{
                "layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.q_proj.scales",
                "fc.weight",
            };
            for (probes) |p| {
                if (f.getTensor(p)) |t| {
                    if (t.dtype == .mlx_q) self.is_mlx = true;
                    if (std.mem.endsWith(u8, p, ".scales")) self.is_mlx = true;
                }
            }
        }

        // Target-layer capture ids. Safetensors stores them as a comma string
        // under dflash.target_layer_ids; synthesize even spacing otherwise
        // (reference build_target_layer_ids behavior).
        if (f.getMetaStr("dflash.target_layer_ids")) |ids_str| {
            var ids = std.ArrayList(u32).empty;
            defer ids.deinit(allocator);
            var it = std.mem.tokenizeScalar(u8, ids_str, ',');
            while (it.next()) |tok| {
                const id = std.fmt.parseInt(u32, std.mem.trim(u8, tok, " \t"), 10) catch continue;
                ids.append(allocator, id) catch break;
            }
            if (ids.items.len != self.n_layers) {
                std.log.err("dflash2: target_layer_ids has {d} entries, expected {d}", .{ ids.items.len, self.n_layers });
                return error.MissingTensor;
            }
            self.target_layer_ids = try allocator.dupe(u32, ids.items);
        } else {
            const num_target = metaFirst(u32, f, &.{"num_target_layers"}) orelse {
                std.log.err("dflash2: config lacks target_layer_ids and num_target_layers", .{});
                return error.MissingTensor;
            };
            const ids = try allocator.alloc(u32, self.n_layers);
            for (ids, 0..) |*id, i| {
                id.* = @intFromFloat(@as(f32, @floatFromInt((i + 1) * num_target)) / @as(f32, @floatFromInt(self.n_layers + 1)));
            }
            self.target_layer_ids = ids;
        }

        // Validate geometry against actual tensors before allocating.
        const concat_dim: usize = @as(usize, self.target_layer_ids.len) * self.n_embd;
        _ = try self.tensor(&.{ "fc.weight", "dflash.fc.weight" }, concat_dim * self.n_embd, "fusion fc");
        for (0..self.n_layers) |li| {
            const lid: u32 = @intCast(li);
            _ = try self.layerTensor(lid, "attn_q");
            _ = try self.layerTensor(lid, "attn_k");
            _ = try self.layerTensor(lid, "attn_v");
            _ = try self.layerTensor(lid, "attn_o");
            _ = try self.layerTensor(lid, "conv_a_base");
            _ = try self.layerTensor(lid, "conv_a_proj");
            _ = try self.layerTensor(lid, "conv_b_base");
            _ = try self.layerTensor(lid, "conv_b_proj");
            _ = try self.layerTensor(lid, "ffn_gate");
            _ = try self.layerTensor(lid, "ffn_up");
            _ = try self.layerTensor(lid, "ffn_down");
        }
        _ = try self.tensor(&.{"candidate_selector.hidden_projection.weight"}, self.sel_rank * self.n_embd, "selector hidden_projection");
        const vocab_rank: usize = @as(usize, self.vocab_size) * self.sel_rank;
        _ = try self.tensor(&.{"candidate_selector.predecessor_codebook"}, vocab_rank, "predecessor_codebook");
        _ = try self.tensor(&.{"candidate_selector.successor_codebook"}, vocab_rank, "successor_codebook");

        if (self.block_size == 0 or self.block_size > max_block_size) {
            std.log.err("dflash2: unsupported block_size {d} (max {d})", .{ self.block_size, max_block_size });
            return error.MissingTensor;
        }
        if (self.conv_group == 0 or self.n_embd % self.conv_group != 0) {
            std.log.err("dflash2: conv_group_size {d} must divide hidden_size {d}", .{ self.conv_group, self.n_embd });
            return error.MissingTensor;
        }
        if (self.sel_top_k == 0 or self.sel_rank == 0 or self.sel_top_k > self.vocab_size) {
            std.log.err("dflash2: invalid selector config (rank={d}, top_k={d})", .{ self.sel_rank, self.sel_top_k });
            return error.MissingTensor;
        }
        if (self.n_head_kv == 0 or self.n_head % self.n_head_kv != 0) {
            std.log.err("dflash2: n_head ({d}) not divisible by n_head_kv ({d})", .{ self.n_head, self.n_head_kv });
            return error.MissingTensor;
        }

        // Context rotating cache: sliding_window - 1 entries, clamped by ctx.
        self.cap = @min(@as(usize, self.sliding_window -| 1), self.max_seq_len);
        if (self.cap == 0) self.cap = 1;

        const a = allocator;
        const nl: usize = self.n_layers;
        const S: usize = self.block_size;
        const e: usize = self.n_embd;
        const hd: usize = self.head_dim;
        const nh: usize = self.n_head;
        const nkv: usize = self.n_head_kv;
        const qd: usize = nh * hd;
        const kvd: usize = nkv * hd;
        const ff: usize = self.n_ff;
        const groups: usize = e / self.conv_group;
        const k_groups: usize = @as(usize, self.conv_kernel) * groups;

        // Small weights, eagerly dequantized so inference never allocates.
        self.norm_fc = try self.loadNormAny(&.{ "hidden_norm.weight", "fc_norm.weight" }, e);
        self.norm_final = try self.loadNormAny(&.{ "norm.weight", "output_norm.weight" }, e);
        self.norm_input = try a.alloc([]f32, nl);
        self.norm_post = try a.alloc([]f32, nl);
        self.norm_q = try a.alloc([]f32, nl);
        self.norm_k = try a.alloc([]f32, nl);
        self.conv_base = try a.alloc([]f32, nl);
        for (0..nl) |li| {
            const lid: u32 = @intCast(li);
            self.norm_input[li] = try self.loadLayerNorm(lid, .input, e);
            self.norm_post[li] = try self.loadLayerNorm(lid, .post, e);
            self.norm_q[li] = try self.loadLayerNorm(lid, .q_norm, hd);
            self.norm_k[li] = try self.loadLayerNorm(lid, .k_norm, hd);
            // Base kernels stacked as [stage][K*E]; dequantized to f32 (source
            // may be bf16 — raw pointer casts are not alignment-safe).
            const cb_t = try self.layerTensor(lid, "conv_a_base");
            const elems: usize = 2 * @as(usize, self.conv_kernel) * e;
            if (cb_t.numElements() != elems) {
                std.log.err("dflash2: conv base kernel has {d} elems, expected {d}", .{ cb_t.numElements(), elems });
                return error.MissingTensor;
            }
            const cb = try a.alloc(f32, elems);
            quant.dequantToF32(cb, cb_t.data_ptr, cb_t.dtype, elems);
            self.conv_base[li] = cb;
        }

        // Context caches.
        self.ctx_k = try a.alloc([]f32, nl);
        self.ctx_v = try a.alloc([]f32, nl);
        for (0..nl) |li| {
            self.ctx_k[li] = try a.alloc(f32, self.cap * kvd);
            @memset(self.ctx_k[li], 0);
            self.ctx_v[li] = try a.alloc(f32, self.cap * kvd);
            @memset(self.ctx_v[li], 0);
        }

        // Block scratch.
        self.blk_x = try a.alloc(f32, S * e);
        self.blk_h = try a.alloc(f32, S * e);
        self.blk_q = try a.alloc(f32, S * qd);
        self.blk_k = try a.alloc(f32, S * kvd);
        self.blk_v = try a.alloc(f32, S * kvd);
        self.blk_attn = try a.alloc(f32, S * @max(qd, ff));
        self.cnv_a = try a.alloc(f32, S * e);
        self.cnv_b = try a.alloc(f32, S * e);
        self.conv_dyn0 = try a.alloc(f32, S * k_groups);
        self.conv_dyn1 = try a.alloc(f32, S * k_groups);
        self.ff_gate = try a.alloc(f32, S * ff);
        self.ff_up = try a.alloc(f32, S * ff);
        self.attn_scores = try a.alloc(f32, self.cap + S);
        self.logits_buf = try a.alloc(f32, self.vocab_size);

        // Selector scratch.
        self.sel_hid = try a.alloc(f32, S * self.sel_rank);
        self.sel_cand = try a.alloc(u32, S * self.sel_top_k);
        self.sel_unary = try a.alloc(f32, S * self.sel_top_k);
        self.sel_edge = try a.alloc(f32, self.sel_top_k);
        self.sel_g = try a.alloc(f32, self.sel_rank);
        self.sel_b_row = try a.alloc(f32, self.sel_rank);
        self.sel_a_row = try a.alloc(f32, self.sel_rank);

        // Ingestion scratch.
        self.ing_fused = try a.alloc(f32, e);
        self.ing_k = try a.alloc(f32, kvd);
        self.ing_v = try a.alloc(f32, kvd);

        std.log.info(
            "dflash2: {d} layers · hidden {d} · heads {d}/{d}x{d} · block {d} · window {d} · selector top-{d}/rank-{d} · capture layers {any}",
            .{ self.n_layers, self.n_embd, self.n_head, self.n_head_kv, self.head_dim, self.block_size, self.sliding_window, self.sel_top_k, self.sel_rank, self.target_layer_ids },
        );
        return self;
    }

    pub fn deinit(self: *DFlash2Model) void {
        self.be.sync();
        const a = self.allocator;
        if (self.target_layer_ids.len > 0) a.free(self.target_layer_ids);
        if (self.norm_fc.len > 0) a.free(self.norm_fc);
        if (self.norm_final.len > 0) a.free(self.norm_final);
        for (0..self.norm_input.len) |i| {
            if (self.norm_input[i].len > 0) a.free(self.norm_input[i]);
            if (self.norm_post[i].len > 0) a.free(self.norm_post[i]);
            if (self.norm_q[i].len > 0) a.free(self.norm_q[i]);
            if (self.norm_k[i].len > 0) a.free(self.norm_k[i]);
            if (self.conv_base[i].len > 0) a.free(self.conv_base[i]);
        }
        if (self.norm_input.len > 0) a.free(self.norm_input);
        if (self.norm_post.len > 0) a.free(self.norm_post);
        if (self.norm_q.len > 0) a.free(self.norm_q);
        if (self.norm_k.len > 0) a.free(self.norm_k);
        if (self.conv_base.len > 0) a.free(self.conv_base);
        for (0..self.ctx_k.len) |i| {
            if (self.ctx_k[i].len > 0) a.free(self.ctx_k[i]);
            if (self.ctx_v[i].len > 0) a.free(self.ctx_v[i]);
        }
        if (self.ctx_k.len > 0) a.free(self.ctx_k);
        if (self.ctx_v.len > 0) a.free(self.ctx_v);
        inline for (.{
            &self.blk_x,       &self.blk_h,      &self.blk_q,     &self.blk_k,
            &self.blk_v,       &self.blk_attn,   &self.cnv_a,     &self.cnv_b,
            &self.conv_dyn0,   &self.conv_dyn1,  &self.ff_gate,   &self.ff_up,
            &self.attn_scores, &self.logits_buf, &self.sel_hid,   &self.sel_edge,
            &self.sel_g,       &self.sel_b_row,  &self.sel_a_row, &self.ing_fused,
            &self.ing_k,       &self.ing_v,
        }) |buf| {
            if (buf.len > 0) a.free(buf.*);
        }
        if (self.sel_cand.len > 0) a.free(self.sel_cand);
        if (self.sel_unary.len > 0) a.free(self.sel_unary);
    }

    // ── Tensor-name resolution ───────────────────────────────────────
    //
    // HuggingFace safetensors checkpoints store literal names such as
    // "layers.0.self_attn.q_proj.weight" which resolve through the format's
    // exact-match lookup. GGUF conversions use "blk.N.attn_q.weight" naming;
    // both spellings are attempted for every tensor.

    const NormKind = enum { input, post, q_norm, k_norm };

    fn layerTensor(self: *const DFlash2Model, li: u32, comptime component: []const u8) !TensorInfo {
        const f = self.fmt;
        switch (comptime componentIdx(component)) {
            0 => { // attn_q
                if (try self.tryTensorHF(li, "self_attn.q_proj")) |t| return t;
                return f.layerTensor(li, "attn_q.weight") orelse f.layerTensor(li, "attn_q") orelse self.missing(component, li);
            },
            1 => { // attn_k
                if (try self.tryTensorHF(li, "self_attn.k_proj")) |t| return t;
                return f.layerTensor(li, "attn_k.weight") orelse f.layerTensor(li, "attn_k") orelse self.missing(component, li);
            },
            2 => { // attn_v
                if (try self.tryTensorHF(li, "self_attn.v_proj")) |t| return t;
                return f.layerTensor(li, "attn_v.weight") orelse f.layerTensor(li, "attn_v") orelse self.missing(component, li);
            },
            3 => { // attn_o
                if (try self.tryTensorHF(li, "self_attn.o_proj")) |t| return t;
                return f.layerTensor(li, "attn_output.weight") orelse f.layerTensor(li, "attn_output") orelse self.missing(component, li);
            },
            4 => { // conv_a_base
                if (try self.tryTensorHF(li, "attention_conv.base_kernel")) |t| return t;
                return f.layerTensor(li, "conv_attn.base") orelse self.missing(component, li);
            },
            5 => { // conv_a_proj
                if (try self.tryTensorHF(li, "attention_conv.kernel_projection")) |t| return t;
                return f.layerTensor(li, "conv_attn.kernel_proj") orelse self.missing(component, li);
            },
            6 => { // conv_b_base
                if (try self.tryTensorHF(li, "mlp_conv.base_kernel")) |t| return t;
                return f.layerTensor(li, "conv_ffn.base") orelse self.missing(component, li);
            },
            7 => { // conv_b_proj
                if (try self.tryTensorHF(li, "mlp_conv.kernel_projection")) |t| return t;
                return f.layerTensor(li, "conv_ffn.kernel_proj") orelse self.missing(component, li);
            },
            8 => { // ffn_gate
                if (try self.tryTensorHF(li, "mlp.gate_proj")) |t| return t;
                return f.layerTensor(li, "ffn_gate.weight") orelse f.layerTensor(li, "ffn_gate") orelse self.missing(component, li);
            },
            9 => { // ffn_up
                if (try self.tryTensorHF(li, "mlp.up_proj")) |t| return t;
                return f.layerTensor(li, "ffn_up.weight") orelse f.layerTensor(li, "ffn_up") orelse self.missing(component, li);
            },
            10 => { // ffn_down
                if (try self.tryTensorHF(li, "mlp.down_proj")) |t| return t;
                return f.layerTensor(li, "ffn_down.weight") orelse f.layerTensor(li, "ffn_down") orelse self.missing(component, li);
            },
            else => unreachable,
        }
    }

    /// Look up "layers.{li}.{hf}" plus its bare (suffix-stripped) spelling.
    fn tryTensorHF(self: *const DFlash2Model, li: u32, comptime hf_suffix: []const u8) !?TensorInfo {
        var buf: [160]u8 = undefined;
        const full = std.fmt.bufPrint(&buf, "layers.{d}." ++ hf_suffix ++ ".weight", .{li}) catch return error.MissingTensor;
        if (self.fmt.getTensor(full)) |t| return t;
        var buf2: [160]u8 = undefined;
        const bare = std.fmt.bufPrint(&buf2, "layers.{d}." ++ hf_suffix, .{li}) catch return error.MissingTensor;
        if (self.fmt.getTensor(bare)) |t| return t;
        return null;
    }

    fn missing(self: *const DFlash2Model, comptime component: []const u8, li: u32) error{MissingTensor} {
        std.log.err("dflash2: tensor '{s}' missing for layer {d}", .{ component, li });
        _ = self;
        return error.MissingTensor;
    }

    fn componentIdx(comptime component: []const u8) usize {
        const names = [_][]const u8{
            "attn_q",      "attn_k",      "attn_v",      "attn_o",
            "conv_a_base", "conv_a_proj", "conv_b_base", "conv_b_proj",
            "ffn_gate",    "ffn_up",      "ffn_down",
        };
        comptime for (names, 0..) |n, i| {
            if (std.mem.eql(u8, component, n)) return i;
        };
        @compileError("unknown dflash2 component: " ++ component);
    }

    /// Public comptime accessor used by the compile-time dispatch sanity check.
    pub fn componentIdxPublic(comptime component: []const u8) usize {
        return componentIdx(component);
    }

    fn tensor(self: *const DFlash2Model, names: []const []const u8, expect_elems: usize, what: []const u8) !TensorInfo {
        for (names) |n| {
            if (self.fmt.getTensor(n)) |t| {
                if (expect_elems != 0 and t.numElements() != expect_elems) {
                    std.log.err("dflash2: {s} has {d} elements, expected {d}", .{ what, t.numElements(), expect_elems });
                    return error.MissingTensor;
                }
                return t;
            }
        }
        std.log.err("dflash2: {s} tensor missing (tried {any})", .{ what, names });
        return error.MissingTensor;
    }

    fn loadNormAny(self: *DFlash2Model, names: []const []const u8, n: usize) ![]f32 {
        const t = try self.tensor(names, n, "norm");
        return self.dequantNorm(t, n);
    }

    fn loadLayerNorm(self: *DFlash2Model, li: u32, kind: NormKind, n: usize) ![]f32 {
        var buf: [160]u8 = undefined;
        const hf_suffix = switch (kind) {
            .input => "input_layernorm.weight",
            .post => "post_attention_layernorm.weight",
            .q_norm => "self_attn.q_norm.weight",
            .k_norm => "self_attn.k_norm.weight",
        };
        const name = std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ li, hf_suffix }) catch return error.MissingTensor;
        if (self.fmt.getTensor(name)) |t| return self.dequantNorm(t, n);
        const gguf_suffix = switch (kind) {
            .input => "attn_norm.weight",
            .post => "post_attention_norm.weight",
            .q_norm => "attn_q_norm.weight",
            .k_norm => "attn_k_norm.weight",
        };
        const t = self.fmt.layerTensor(li, gguf_suffix) orelse {
            std.log.err("dflash2: norm '{s}' missing for layer {d}", .{ gguf_suffix, li });
            return error.MissingTensor;
        };
        return self.dequantNorm(t, n);
    }

    /// Dequantize one RMSNorm weight. SafeTensors checkpoints store raw HF
    /// RMSNorm weights (out = x/rms * (1+w)); bake the +1 here. GGUF converters
    /// and MLX sanitizers pre-bake it.
    fn dequantNorm(self: *DFlash2Model, t: TensorInfo, n: usize) ![]f32 {
        const buf = try self.allocator.alloc(f32, n);
        errdefer self.allocator.free(buf);
        quant.dequantToF32(buf, t.data_ptr, t.dtype, n);
        if (self.is_safetensors and !self.is_mlx) {
            for (buf) |*v| v.* += 1.0;
        }
        return buf;
    }

    // ── Runtime API ──────────────────────────────────────────────

    /// Bind target-model embeddings and LM head. Required before proposeBlock;
    /// the DFlash2 checkpoint ships neither.
    pub fn bindTarget(self: *DFlash2Model, emb: TensorInfo, lm_head: TensorInfo, target_fmt: Format) void {
        self.target_emb = emb;
        self.target_lm_head = lm_head;
        self.target_fmt = target_fmt;
    }

    /// Reset injected-context cache for a fresh conversation.
    pub fn resetCache(self: *DFlash2Model) void {
        self.ctx_count = 0;
        self.kv_seq_len = 0;
        self.cancelled.store(false, .release);
    }

    /// Number of ingested context positions (absolute position counter).
    pub fn contextLen(self: *const DFlash2Model) usize {
        return self.ctx_count;
    }

    // ── Model-interface stubs ────────────────────────────────────
    //
    // The drafter is not an autoregressive language model: it never runs
    // standalone forwards or prefills. The orchestrator drives it exclusively
    // through ingestContext/proposeBlock. The generic vtable still requires
    // these methods to exist.

    /// Unsupported: DFlash2 drafts whole blocks in one pass, never token-by-token.
    pub fn forward(self: *DFlash2Model, token_id: u32) !u32 {
        _ = self;
        _ = token_id;
        return error.MissingTensor;
    }

    /// Unsupported: context enters via ingestContext, not AR prefill.
    pub fn prefill(self: *DFlash2Model, token_ids: []const u32) !u32 {
        _ = self;
        _ = token_ids;
        return error.MissingTensor;
    }

    /// Signal cancellation; checked between backbone layers during proposeBlock.
    pub fn cancel(self: *DFlash2Model) void {
        self.cancelled.store(true, .release);
    }

    /// No paged block table — the context cache is a fixed rotating ring.
    pub fn getBlockTable(self: *DFlash2Model) []const u32 {
        _ = self;
        return &.{};
    }

    /// Ingest `n_pos` context positions starting at absolute position `start`.
    ///
    /// `features` holds `[n_pos][n_capture * n_embd]` f32 values: for each
    /// position the concatenation of the captured target hidden states ordered
    /// by `target_layer_ids`. Projects each position through the fusion fc +
    /// hidden-norm and appends its k/v (after k-norm and RoPE at the absolute
    /// position) to every layer's rotating context cache.
    pub fn ingestContext(self: *DFlash2Model, features: []const f32, start: usize, n_pos: usize) !void {
        if (start + n_pos <= start) return;
        if (start != self.ctx_count) {
            std.log.warn("dflash2: context gap (ingest at {d}, have {d})", .{ start, self.ctx_count });
        }
        const e: usize = self.n_embd;
        const kvd: usize = @as(usize, self.n_head_kv) * self.head_dim;
        const concat_dim: usize = @as(usize, self.target_layer_ids.len) * e;
        const fc = try self.tensor(&.{ "fc.weight", "dflash.fc.weight" }, concat_dim * e, "fusion fc");

        for (0..n_pos) |pi| {
            const abs_pos = start + pi;
            if (abs_pos >= self.max_seq_len) return error.KVCacheFull;
            if (self.cancelled.load(.monotonic)) return error.Cancelled;
            const feat = features[pi * concat_dim ..][0..concat_dim];

            // Fusion: h_ctx = hidden_norm(fc(concat(features)))
            self.dispatchGemv(feat.ptr, fc, self.ing_fused.ptr, e, concat_dim);
            self.be.sync();
            self.be.rmsNorm(self.ing_fused.ptr, self.norm_fc.ptr, self.ing_fused.ptr, e, self.rms_eps);

            const slot = abs_pos % self.cap;
            for (0..self.n_layers) |li| {
                const lid: u32 = @intCast(li);
                const kw = try self.layerTensor(lid, "attn_k");
                const vw = try self.layerTensor(lid, "attn_v");
                const ops = [_]backend_mod.GemvOp{
                    self.makeOp(kw, self.ing_k.ptr, kvd),
                    self.makeOp(vw, self.ing_v.ptr, kvd),
                };
                self.be.gemvMulti(self.ing_fused.ptr, &ops, e);
                self.be.sync();
                // Per-KV-head k-norm (values are NOT normalized), then RoPE keys.
                self.be.rmsNormMulti(self.ing_k.ptr, self.norm_k[li].ptr, self.n_head_kv, self.head_dim, self.rms_eps);
                self.be.rope(self.ing_k.ptr, abs_pos, self.n_head_kv, self.head_dim, self.head_dim, self.rope_theta);
                self.be.sync();
                @memcpy(self.ctx_k[li][slot * kvd ..][0..kvd], self.ing_k[0..kvd]);
                @memcpy(self.ctx_v[li][slot * kvd ..][0..kvd], self.ing_v[0..kvd]);
            }
        }
        self.ctx_count = start + n_pos;
        self.kv_seq_len = self.ctx_count;
    }

    /// Draft one block: `gamma` proposal tokens following `anchor`.
    ///
    /// anchor sits at absolute position `anchor_pos` (must equal the current
    /// context length). Raw per-slot logits are written through CALLER-supplied
    /// `logit_rows`: logit_rows[t] receives the vocabulary-sized lm_head output
    /// for proposal slot t; at temperature > 0 the selector rewrites each row
    /// into a q distribution (log-prob form) consumed by the lossless verifier.
    /// Returns the number of drafted tokens (== gamma on success).
    pub fn proposeBlock(
        self: *DFlash2Model,
        anchor: u32,
        anchor_pos: usize,
        gamma: u32,
        temperature: f32,
        rng: std.Random,
        tokens_out: []u32,
        logit_rows: []const []f32,
    ) !u32 {
        if (self.target_emb == null or self.target_lm_head == null) {
            std.log.err("dflash2: proposeBlock before bindTarget", .{});
            return error.MissingTensor;
        }
        if (gamma == 0 or gamma >= self.block_size or gamma > max_block_size - 1) return error.MissingTensor;
        if (anchor_pos != self.ctx_count) {
            std.log.err("dflash2: propose anchor_pos {d} != ctx_count {d}", .{ anchor_pos, self.ctx_count });
            return error.KVCacheFull;
        }
        const slots: usize = @as(usize, gamma) + 1; // anchor + proposals
        const e: usize = self.n_embd;
        const groups: usize = e / self.conv_group;
        const k_groups: usize = @as(usize, self.conv_kernel) * groups;
        const K: usize = self.sel_top_k;
        const rank: usize = self.sel_rank;

        // Embeddings: slot 0 = anchor, others = mask token.
        const emb = self.target_emb.?;
        for (0..slots) |s| {
            const tok: u32 = if (s == 0) anchor else self.mask_token_id;
            self.be.embLookup(.{ .data = emb.data_ptr, .dtype = emb.dtype }, tok, self.blk_x.ptr + s * e, e);
        }
        self.be.sync();
        if (self.input_embedding_scale != 1.0) {
            const scale = self.input_embedding_scale;
            for (0..slots * e) |i| self.blk_x[i] *= scale;
        }

        // ── Backbone: conv-wrapped transformer layers ────────────
        // Attention sublayer: x = resid + conv1(attn(conv0(norm(resid))))
        // MLP sublayer:      x = resid + conv1(mlp(conv0(norm(resid))))
        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.monotonic)) return error.Cancelled;
            const lid: u32 = @intCast(li);

            self.be.rmsNormBatched(self.blk_x.ptr, self.norm_input[li].ptr, self.blk_h.ptr, slots, e, self.rms_eps);
            try self.runConvProj(lid, 0, slots, k_groups);
            self.cnvStage(0, slots, li, self.blk_h, self.cnv_a);
            try self.attention(lid, slots, anchor_pos); // consumes cnv_a, writes blk_h
            self.cnvStage(1, slots, li, self.blk_h, self.cnv_b);
            for (0..slots * e) |i| self.blk_x[i] += self.cnv_b[i];

            self.be.rmsNormBatched(self.blk_x.ptr, self.norm_post[li].ptr, self.blk_h.ptr, slots, e, self.rms_eps);
            try self.runConvProj(lid, 1, slots, k_groups);
            self.cnvStage(0, slots, li, self.blk_h, self.cnv_a);
            try self.mlp(lid, slots); // consumes cnv_a, writes blk_attn[..S*e]
            self.cnvStage(1, slots, li, self.blk_attn, self.cnv_b);
            for (0..slots * e) |i| self.blk_x[i] += self.cnv_b[i];
        }

        // Final norm (all rows; proposal rows consumed below).
        self.be.rmsNormBatched(self.blk_x.ptr, self.norm_final.ptr, self.blk_h.ptr, slots, e, self.rms_eps);
        self.be.sync();

        // Selector hidden projections: hid[t] = H(h_slot_{t+1}).
        const hp = try self.tensor(&.{"candidate_selector.hidden_projection.weight"}, rank * e, "selector hidden_projection");
        for (0..@as(usize, gamma)) |t| {
            const row = self.blk_h[(t + 1) * e ..][0..e];
            self.dispatchGemv(row.ptr, hp, self.sel_hid[t * rank ..].ptr, rank, e);
        }
        self.be.sync();

        // Per-slot logits from the TARGET LM head.
        const head = self.target_lm_head.?;
        const t_fmt = self.target_fmt.?;
        for (0..@as(usize, gamma)) |t| {
            const row = self.blk_h[(t + 1) * e ..][0..e];
            model_mod.dispatchGemv(self.be, t_fmt, row.ptr, head, logit_rows[t].ptr, self.vocab_size, e);
        }
        self.be.sync();

        // Post-process logits, select top-K candidates.
        for (0..@as(usize, gamma)) |t| {
            const dst = logit_rows[t];
            if (self.softcap > 0) {
                const cap = self.softcap;
                for (dst) |*v| v.* = std.math.tanh(v.* / cap) * cap;
            }
            if (self.output_multiplier != 1.0) {
                const mul = self.output_multiplier;
                for (dst) |*v| v.* *= mul;
            }
            const n_sel = dflash2_alg.topK(dst, self.sel_cand[t * K ..][0..K], self.sel_unary[t * K ..][0..K]);
            if (n_sel != K) return error.MissingTensor;
        }

        const ac = try self.tensor(&.{"candidate_selector.predecessor_codebook"}, 0, "predecessor_codebook");
        const bc = try self.tensor(&.{"candidate_selector.successor_codebook"}, 0, "successor_codebook");

        // Path walk: score adjacent edges, pick greedily or sample at T > 0.
        // (Inline rather than via spec/dflash2.selectPath: quantized codebook
        // rows must dequantize through the format-aware helper.)
        var pred = anchor;
        for (0..@as(usize, gamma)) |t| {
            try self.codebookRow(ac, pred, self.sel_a_row);
            const hid_row = self.sel_hid[t * rank ..][0..rank];
            for (0..rank) |r| self.sel_g[r] = self.sel_a_row[r] * hid_row[r];
            for (0..K) |ci| {
                try self.codebookRow(bc, self.sel_cand[t * K + ci], self.sel_b_row);
                var dot: f32 = 0;
                for (0..rank) |r| dot += self.sel_g[r] * self.sel_b_row[r];
                self.sel_edge[ci] = self.sel_unary[t * K + ci] + dot;
            }
            if (temperature <= 0) {
                var best: usize = 0;
                var best_val = self.sel_edge[0];
                for (1..K) |ci| {
                    if (self.sel_edge[ci] > best_val) {
                        best_val = self.sel_edge[ci];
                        best = ci;
                    }
                }
                pred = self.sel_cand[t * K + best];
            } else {
                // Softmax over edge scores, inverse-CDF draw, export q row.
                var max_v = self.sel_edge[0];
                for (self.sel_edge[1..K]) |s| max_v = @max(max_v, s);
                var sum: f32 = 0;
                for (0..K) |ci| {
                    const ex = @exp((self.sel_edge[ci] - max_v) / temperature);
                    self.sel_edge[ci] = ex;
                    sum += ex;
                }
                const inv_sum = if (sum > 0) 1.0 / sum else 0;
                const r = rng.float(f32);
                var chosen: usize = K - 1;
                var cdf: f32 = 0;
                for (0..K) |ci| {
                    cdf += self.sel_edge[ci] * inv_sum;
                    if (r < cdf) {
                        chosen = ci;
                        break;
                    }
                }
                // Rewrite the verifier's q row: the chosen candidate keeps its
                // probability, everything else becomes 0 (log −inf). The
                // lossless verifier reads exp(log_probs[token]).
                const row = logit_rows[t];
                const q_chosen = self.sel_edge[chosen] * inv_sum;
                @memset(row, -std.math.inf(f32));
                row[self.sel_cand[t * K + chosen]] = @log(@max(q_chosen, 1e-30));
                pred = self.sel_cand[t * K + chosen];
            }
            tokens_out[t] = pred;
        }
        return gamma;
    }

    /// Apply one convolution stage: groupedDynamicConv over `src` rows using
    /// this layer's dequantized base kernel and the staged dynamic coefficients.
    fn cnvStage(self: *const DFlash2Model, stage: usize, slots: usize, li: usize, src: []const f32, dst: []f32) void {
        const e: usize = self.n_embd;
        const k_groups: usize = @as(usize, self.conv_kernel) * (e / self.conv_group);
        const dyn = if (stage == 0) self.conv_dyn0 else self.conv_dyn1;
        const base_off = stage * @as(usize, self.conv_kernel) * e;
        const base = self.conv_base[li][base_off..][0 .. @as(usize, self.conv_kernel) * e];
        dflash2_alg.groupedDynamicConv(
            src[0 .. slots * e],
            dyn[0 .. slots * k_groups],
            base,
            dst[0 .. slots * e],
            slots,
            e,
            self.conv_kernel,
            self.conv_group,
        );
    }

    /// kernel_projection for one conv stage: per-row GEMV producing [K*groups]
    /// dynamic coefficients. Stage-1 weights are the upper row-block of the
    /// projection matrix (reference packs [2 taps] along the output dim).
    fn runConvProj(self: *DFlash2Model, lid: u32, stage: usize, slots: usize, k_groups: usize) !void {
        const proj_full = switch (stage) {
            0 => try self.layerTensor(lid, "conv_a_proj"),
            else => try self.layerTensor(lid, "conv_b_proj"),
        };
        const e: usize = self.n_embd;
        const out_dim: usize = k_groups;
        const row_bytes = backend_mod.gemvRowBytes(proj_full.dtype, e);
        if (row_bytes == 0) return error.MissingTensor;
        var proj = proj_full;
        proj.data_ptr += @as(usize, stage) * out_dim * row_bytes;
        const dyn = if (stage == 0) &self.conv_dyn0 else &self.conv_dyn1;
        for (0..slots) |s| {
            self.dispatchGemv(self.blk_h[s * e ..].ptr, proj, dyn.*[s * k_groups ..].ptr, out_dim, e);
        }
        self.be.sync();
    }

    /// Non-causal block attention with sliding-window masked context.
    /// Consumes cnv_a[..slots*e] as input; writes o_proj output to blk_h.
    fn attention(self: *DFlash2Model, lid: u32, slots: usize, anchor_pos: usize) !void {
        const e: usize = self.n_embd;
        const hd: usize = self.head_dim;
        const nh: usize = self.n_head;
        const nkv: usize = self.n_head_kv;
        const qd: usize = nh * hd;
        const kvd: usize = nkv * hd;
        const heads_per_kv = nh / nkv;

        const qw = try self.layerTensor(lid, "attn_q");
        const kw = try self.layerTensor(lid, "attn_k");
        const vw = try self.layerTensor(lid, "attn_v");
        const ow = try self.layerTensor(lid, "attn_o");

        // Q/K/V for all block slots.
        for (0..slots) |s| {
            const x = self.cnv_a[s * e ..][0..e];
            const ops = [_]backend_mod.GemvOp{
                self.makeOp(qw, self.blk_q.ptr + s * qd, qd),
                self.makeOp(kw, self.blk_k.ptr + s * kvd, kvd),
                self.makeOp(vw, self.blk_v.ptr + s * kvd, kvd),
            };
            self.be.gemvMulti(x.ptr, &ops, e);
        }
        self.be.sync();

        // Q/K norms + RoPE at absolute block positions.
        self.be.rmsNormMulti(self.blk_q.ptr, self.norm_q[lid].ptr, slots * nh, hd, self.rms_eps);
        self.be.rmsNormMulti(self.blk_k.ptr, self.norm_k[lid].ptr, slots * nkv, hd, self.rms_eps);
        for (0..slots) |s| {
            const pos = anchor_pos + s;
            self.be.rope(self.blk_q.ptr + s * qd, pos, nh, hd, hd, self.rope_theta);
            self.be.rope(self.blk_k.ptr + s * kvd, pos, nkv, hd, hd, self.rope_theta);
        }
        self.be.sync();

        // Manual SDPA: queries attend to (context ∪ block) keys. Context keys
        // satisfy the sliding-window constraint positionally; block keys are
        // fully visible (is_causal=false).
        const S_ctx = @min(self.cap, self.ctx_count);
        const oldest = self.ctx_count - S_ctx;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const window: i64 = @intCast(self.sliding_window);
        const n_keys = S_ctx + slots;
        const scores = self.attn_scores[0..n_keys];

        for (0..slots) |si| {
            const q_abs: i64 = @intCast(anchor_pos + si);
            for (0..nh) |h| {
                const kv_head = h / heads_per_kv;
                const q = self.blk_q[si * qd + h * hd ..][0..hd];
                var max_v: f32 = -std.math.inf(f32);
                for (0..n_keys) |j| {
                    if (j < S_ctx) {
                        const k_abs: i64 = @intCast(oldest + j);
                        if (q_abs - k_abs >= window) {
                            scores[j] = -std.math.inf(f32);
                            continue;
                        }
                    }
                    const k = self.keyAt(lid, j, S_ctx, kv_head, kvd, hd);
                    var dot: f32 = 0;
                    for (0..hd) |d| dot += q[d] * k[d];
                    const s_val = dot * scale;
                    scores[j] = s_val;
                    if (s_val > max_v) max_v = s_val;
                }
                var sum: f32 = 0;
                for (0..n_keys) |j| {
                    if (scores[j] == -std.math.inf(f32)) continue;
                    const ex = @exp(scores[j] - max_v);
                    scores[j] = ex;
                    sum += ex;
                }
                const inv_sum = if (sum > 0) 1.0 / sum else 0;
                // Head output accumulates into blk_attn ([S*qd] layout);
                // blk_h stays reserved for the o_proj result.
                const out_row = self.blk_attn[si * qd + h * hd ..][0..hd];
                @memset(out_row, 0);
                for (0..n_keys) |j| {
                    if (scores[j] == -std.math.inf(f32)) continue;
                    const w = scores[j] * inv_sum;
                    if (w == 0) continue;
                    const v = self.valueAt(lid, j, S_ctx, kv_head, kvd, hd);
                    for (0..hd) |d| out_row[d] += w * v[d];
                }
            }
        }

        // Output projection: attention heads were accumulated into blk_attn
        // (per-slot qd layout); map [S*qd] -> [S*e] into blk_h.
        for (0..slots) |s| {
            self.dispatchGemv(self.blk_attn[s * qd ..].ptr, ow, self.blk_h[s * e ..].ptr, e, qd);
        }
        self.be.sync();
    }

    inline fn keyAt(self: *const DFlash2Model, lid: usize, j: usize, S_ctx: usize, kv_head: usize, kvd: usize, hd: usize) [*]const f32 {
        if (j < S_ctx) {
            const idx = (self.ctx_count - S_ctx + j) % self.cap;
            return self.ctx_k[lid][idx * kvd + kv_head * hd ..].ptr;
        }
        return self.blk_k[(j - S_ctx) * kvd + kv_head * hd ..].ptr;
    }

    inline fn valueAt(self: *const DFlash2Model, lid: usize, j: usize, S_ctx: usize, kv_head: usize, kvd: usize, hd: usize) [*]const f32 {
        if (j < S_ctx) {
            const idx = (self.ctx_count - S_ctx + j) % self.cap;
            return self.ctx_v[lid][idx * kvd + kv_head * hd ..].ptr;
        }
        return self.blk_v[(j - S_ctx) * kvd + kv_head * hd ..].ptr;
    }

    /// SwiGLU MLP over all block slots: result lands in blk_attn[..slots*e].
    fn mlp(self: *DFlash2Model, lid: u32, slots: usize) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.n_ff;
        const gw = try self.layerTensor(lid, "ffn_gate");
        const uw = try self.layerTensor(lid, "ffn_up");
        const dw = try self.layerTensor(lid, "ffn_down");
        for (0..slots) |s| {
            const x = self.cnv_a[s * e ..][0..e];
            const ops = [_]backend_mod.GemvOp{
                self.makeOp(gw, self.ff_gate.ptr + s * ff, ff),
                self.makeOp(uw, self.ff_up.ptr + s * ff, ff),
            };
            self.be.gemvMulti(x.ptr, &ops, e);
        }
        self.be.siluMul(self.ff_gate.ptr, self.ff_up.ptr, self.ff_gate.ptr, slots * ff);
        for (0..slots) |s| {
            self.dispatchGemv(self.ff_gate[s * ff ..].ptr, dw, self.blk_attn[s * e ..].ptr, e, ff);
        }
        self.be.sync();
    }

    /// Dequantize one codebook row (token id -> rank floats). Row addressing is
    /// format-aware so quantized codebooks (GGUF K-quants etc.) work.
    fn codebookRow(self: *DFlash2Model, t: TensorInfo, token: u32, out: []f32) !void {
        const rank: usize = self.sel_rank;
        const row_bytes = backend_mod.gemvRowBytes(t.dtype, rank);
        if (row_bytes == 0) return error.MissingTensor;
        const row_ptr = t.data_ptr + @as(usize, token) * row_bytes;
        quant.dequantToF32(out, row_ptr, t.dtype, rank);
    }

    fn makeOp(self: *const DFlash2Model, t: TensorInfo, y: [*]f32, n: usize) backend_mod.GemvOp {
        var op = backend_mod.GemvOp{ .w = .{ .data = t.data_ptr, .dtype = t.dtype }, .y = y, .n = n };
        if (t.dtype == .mlx_q) {
            if (model_mod.findMlxCompanion(self.fmt, t, 0)) |c| {
                op.mlx_scales = c.scales;
                op.mlx_biases = c.biases;
                op.mlx_bits = c.bits;
                op.mlx_group_size = c.group_size;
            }
        }
        return op;
    }

    fn dispatchGemv(self: *DFlash2Model, x: [*]const f32, t: TensorInfo, y: [*]f32, n: usize, k: usize) void {
        model_mod.dispatchGemv(self.be, self.fmt, x, t, y, n, k);
    }
};

comptime {
    // Component-name dispatch table sanity: unique indices for all components.
    const names = [_][]const u8{
        "attn_q",      "attn_k",      "attn_v",      "attn_o",
        "conv_a_base", "conv_a_proj", "conv_b_base", "conv_b_proj",
        "ffn_gate",    "ffn_up",      "ffn_down",
    };
    for (names, 0..) |n, i| {
        if (DFlash2Model.componentIdxPublic(n) != i) @compileError("componentIdx mismatch");
    }
}

test "dflash2 geometry defaults match Qwen3.8-27B-DFlash2 config" {
    // Documented checkpoint values; guards against accidental drift.
    const m = DFlash2Model{ .fmt = undefined, .be = undefined, .allocator = undefined };
    try std.testing.expectEqual(@as(u32, 5), m.n_layers);
    try std.testing.expectEqual(@as(u32, 5120), m.n_embd);
    try std.testing.expectEqual(@as(u32, 32), m.n_head);
    try std.testing.expectEqual(@as(u32, 8), m.n_head_kv);
    try std.testing.expectEqual(@as(u32, 128), m.head_dim);
    try std.testing.expectEqual(@as(u32, 17408), m.n_ff);
    try std.testing.expectEqual(@as(u32, 248320), m.vocab_size);
    try std.testing.expectEqual(@as(u32, 2048), m.sliding_window);
    try std.testing.expectEqual(@as(u32, 8), m.block_size);
    try std.testing.expectEqual(@as(u32, 248070), m.mask_token_id);
    try std.testing.expectEqual(@as(u32, 2), m.conv_kernel);
    try std.testing.expectEqual(@as(u32, 16), m.conv_group);
    try std.testing.expectEqual(@as(u32, 256), m.sel_rank);
    try std.testing.expectEqual(@as(u32, 16), m.sel_top_k);
}
