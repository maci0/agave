//! DiffusionGemma — Google's discrete text diffusion model.
//!
//! Architecture overview
//! ---------------------
//! Based on Gemma 4 26B A4B with block-autoregressive masked diffusion:
//! * 30 layers: pattern of 5 sliding-window + 1 global attention (×5 blocks)
//! * 128 experts MoE (top-8 active); shared expert per token
//! * Fused expert weights: `experts.gate_up_proj` (gate+up concat) and `experts.down_proj`
//! * Per-layer `layer_scalar` applied to attention output
//! * Canvas = 256 tokens; denoising uses bidirectional attention within canvas
//! * Tensor prefix: `model.decoder.layers.N.` (SafeTensors BF16 only)
//! * Self-conditioning block (`model.self_cond_block.*`) — optional, skipped in v1
//!
//! Inference flow:
//!   1. Encoder prefill: causal forward pass over prompt → KV cache
//!   2. Denoising loop: iteratively forward the 256-token canvas with bidirectional
//!      attention, accepting high-confidence tokens and re-noising low-confidence ones
//!   3. Block autoregressive: chain 256-token canvases for long outputs

const std = @import("std");
const build_options = @import("build_options");

const backend_mod = @import("../backend/backend.zig");
const format_mod = @import("../format/format.zig");
const model_mod = @import("model.zig");
const math_ops = @import("../ops/math.zig");
const attn_ops = @import("../ops/attention.zig");
const kvcache = @import("../kvcache/manager.zig");
const block_alloc_mod = @import("../kvcache/block_allocator.zig");

const Backend = backend_mod.Backend;
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;
const kv_quant = @import("../ops/kv_quant.zig");
const quant = @import("../ops/quant.zig");
const PagedKvCache = kvcache.PagedKvCache;
const SeqBlockTable = kvcache.SeqBlockTable;
const BlockAllocator = block_alloc_mod.BlockAllocator;
const TieredBlockAllocator = block_alloc_mod.TieredBlockAllocator;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;

// ── Default architecture constants (Gemma 4 26B A4B / DiffusionGemma 26B A4B) ──

const default_n_layers: u32 = 30;
const default_n_embd: u32 = 2816;
const default_rms_eps: f32 = 1e-6;
const default_vocab_size: u32 = 262144;

/// Sliding-window attention params.
const default_sl_n_head: u32 = 16;
const default_sl_n_kv_head: u32 = 8;
const default_sl_head_dim: u32 = 256;
const default_sl_rope_theta: f32 = 10_000.0;
const default_sliding_window: u32 = 1024;

/// Global attention params.
const default_gl_n_head: u32 = 16;
const default_gl_n_kv_head: u32 = 8;
const default_gl_head_dim: u32 = 512;
const default_gl_rope_theta: f32 = 1_000_000.0;
const default_gl_partial_rotary: f32 = 0.25;

/// MoE params.
const default_n_experts: u32 = 128;
const default_top_k_experts: u32 = 8;
const default_moe_ff: u32 = 704;

/// Global attention every N layers (e.g. every 6th: 5 sliding + 1 global).
const global_attn_stride: u32 = 6;
/// Canvas size for block diffusion (tokens per denoising block).
pub const default_canvas_length: u32 = 256;
/// Maximum layers supported.
const max_layers: usize = 64;
/// Buffer size for constructing tensor names.
const name_buf_size: usize = 256;
/// Norm weight cache entries: ~6 norms/layer × 30 layers + final ≈ 181; leave headroom.
const max_norm_entries: usize = 256;

/// GPU SDPA threshold: head dims above this fall back to CPU-side SDPA.
const gpu_sdpa_max_head_dim: usize = 256;

/// DiffusionGemma model state.
pub const DiffusionGemmaModel = struct {
    fmt: Format,
    be: Backend,
    allocator: Allocator,

    // ── Architecture parameters ───────────────────────────────────
    n_layers: u32 = default_n_layers,
    n_embd: u32 = default_n_embd,
    rms_eps: f32 = default_rms_eps,
    vocab_size: u32 = default_vocab_size,
    emb_scale: f32 = 1.0,

    // Sliding-window attention
    sl_n_head: u32 = default_sl_n_head,
    sl_n_kv_head: u32 = default_sl_n_kv_head,
    sl_head_dim: u32 = default_sl_head_dim,
    sl_rope_theta: f32 = default_sl_rope_theta,
    sl_rope_dim: u32 = default_sl_head_dim,
    sliding_window: u32 = default_sliding_window,

    // Global attention
    gl_n_head: u32 = default_gl_n_head,
    gl_n_kv_head: u32 = default_gl_n_kv_head,
    gl_head_dim: u32 = default_gl_head_dim,
    gl_rope_theta: f32 = default_gl_rope_theta,
    gl_rope_dim: u32 = default_gl_head_dim,
    gl_partial_rotary: f32 = default_gl_partial_rotary,

    // MoE
    n_experts: u32 = default_n_experts,
    top_k_experts: u32 = default_top_k_experts,
    moe_ff: u32 = default_moe_ff,

    // Diffusion
    canvas_length: u32 = default_canvas_length,

    // Model vtable compatibility aliases.
    n_head: u32 = default_sl_n_head,
    n_head_kv: u32 = default_sl_n_kv_head,
    eos_token_id: u32 = 1,

    // ── Working buffers ───────────────────────────────────────────
    hidden: []f32 = &.{},
    hidden2: []f32 = &.{},
    /// Q projection buffer: sl_n_head * sl_head_dim (max) elements.
    q_buf: []f32 = &.{},
    /// K projection buffer: sl_n_kv_head * max(sl_head_dim, gl_head_dim) elements.
    k_buf: []f32 = &.{},
    /// V projection buffer: sl_n_kv_head * max(sl_head_dim, gl_head_dim) elements.
    v_buf: []f32 = &.{},
    /// Attention output: max(sl_n_head * sl_head_dim, gl_n_head * gl_head_dim) elements.
    attn_out: []f32 = &.{},
    /// Attention scores scratch: max_seq_len elements.
    scores_buf: []f32 = &.{},
    /// Gate output (first half of fused gate_up_proj): moe_ff elements.
    gate_buf: []f32 = &.{},
    /// Up output (second half of fused gate_up_proj): moe_ff elements.
    up_buf: []f32 = &.{},
    /// Expert output accumulator: n_embd elements.
    expert_out: []f32 = &.{},
    /// Router logits: n_experts elements.
    router_logits: []f32 = &.{},
    /// Final logits: vocab_size elements.
    logits_buf: []f32 = &.{},
    /// Canvas token IDs for diffusion (canvas_length elements).
    canvas_tokens: []u32 = &.{},

    // ── Prefill batch buffers (canvas denoising) ──────────────────
    /// Batch hidden states: canvas_length * n_embd.
    pf_hidden: []f32 = &.{},
    /// Batch scratch: canvas_length * n_embd.
    pf_scratch: []f32 = &.{},
    /// Batch logits: canvas_length * vocab_size (set lazily during diffusion).
    pf_logits: []f32 = &.{},
    /// Pre-allocated canvas K buffer: canvas_length * max_kv_dim (avoids per-layer alloc).
    canvas_k_buf: []f32 = &.{},
    /// Pre-allocated canvas V buffer: canvas_length * max_kv_dim.
    canvas_v_buf: []f32 = &.{},

    // ── KV cache ──────────────────────────────────────────────────
    paged_cache: PagedKvCache = undefined,
    seq_table: SeqBlockTable = undefined,
    block_allocator: BlockAllocator = undefined,
    tiered_cache: ?*TieredKvCache = null,
    tiered_block_allocator: ?TieredBlockAllocator = null,
    kv_type_k: kv_quant.KvQuantType = .f32,
    kv_type_v: kv_quant.KvQuantType = .f32,
    kv_seq_len: usize = 0,
    max_seq_len: usize = 4096,

    // ── Spec-decode / control ─────────────────────────────────────
    layer_skip_start: u32 = 0,
    layer_skip_end: u32 = 0,
    hidden_pre_norm: []f32 = &.{},
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    megakernel_enabled: bool = false,

    /// Permanently dequantized BF16 norm weights (GPU-safe stable pointers).
    norm_cache: [max_norm_entries]model_mod.NormCacheEntry = undefined,
    norm_cache_len: usize = 0,

    // ── Lifecycle ─────────────────────────────────────────────────

    /// Initialize from SafeTensors metadata. Caller must call `deinit` when done.
    pub fn init(
        allocator: Allocator,
        f: Format,
        be: Backend,
        ctx_size: u32,
        kv_type_k: kv_quant.KvQuantType,
        kv_type_v: kv_quant.KvQuantType,
        tiered_cache: ?*TieredKvCache,
    ) !DiffusionGemmaModel {
        var self = DiffusionGemmaModel{ .fmt = f, .be = be, .allocator = allocator };
        self.kv_type_k = kv_type_k;
        self.kv_type_v = kv_type_v;

        // Load architecture from SafeTensors config (JSON metadata).
        self.n_layers = f.getMetaU32("num_hidden_layers") orelse default_n_layers;
        self.n_embd = f.getMetaU32("hidden_size") orelse default_n_embd;
        self.rms_eps = f.getMetaF32("rms_norm_eps") orelse default_rms_eps;
        self.vocab_size = f.getMetaU32("vocab_size") orelse default_vocab_size;
        self.emb_scale = @sqrt(@as(f32, @floatFromInt(self.n_embd)));

        // Sliding-window attention.
        const text_cfg_n_head = f.getMetaU32("text_config.num_attention_heads") orelse
            f.getMetaU32("num_attention_heads");
        self.sl_n_head = text_cfg_n_head orelse default_sl_n_head;
        self.sl_n_kv_head = f.getMetaU32("text_config.num_key_value_heads") orelse
            f.getMetaU32("num_key_value_heads") orelse default_sl_n_kv_head;
        self.sl_head_dim = f.getMetaU32("text_config.head_dim") orelse
            f.getMetaU32("head_dim") orelse default_sl_head_dim;
        self.sl_rope_theta = f.getMetaF32("text_config.rope_theta") orelse
            f.getMetaF32("rope_theta") orelse default_sl_rope_theta;
        self.sliding_window = f.getMetaU32("text_config.sliding_window") orelse
            f.getMetaU32("sliding_window") orelse default_sliding_window;
        self.sl_rope_dim = self.sl_head_dim;

        // Global attention.
        self.gl_n_head = f.getMetaU32("text_config.global_num_attention_heads") orelse
            f.getMetaU32("global_num_attention_heads") orelse self.sl_n_head;
        self.gl_n_kv_head = f.getMetaU32("text_config.global_num_key_value_heads") orelse
            f.getMetaU32("global_num_key_value_heads") orelse self.sl_n_kv_head;
        self.gl_head_dim = f.getMetaU32("text_config.global_head_dim") orelse
            f.getMetaU32("global_head_dim") orelse default_gl_head_dim;
        self.gl_rope_theta = f.getMetaF32("text_config.global_rope_theta") orelse
            f.getMetaF32("global_rope_theta") orelse default_gl_rope_theta;
        self.gl_partial_rotary = f.getMetaF32("text_config.partial_rotary_factor") orelse
            f.getMetaF32("partial_rotary_factor") orelse default_gl_partial_rotary;
        self.gl_rope_dim = @as(u32, @intFromFloat(@as(f32, @floatFromInt(self.gl_head_dim)) * self.gl_partial_rotary));

        // MoE.
        self.n_experts = f.getMetaU32("text_config.num_local_experts") orelse
            f.getMetaU32("num_local_experts") orelse default_n_experts;
        self.top_k_experts = f.getMetaU32("text_config.num_experts_per_tok") orelse
            f.getMetaU32("num_experts_per_tok") orelse default_top_k_experts;
        self.moe_ff = f.getMetaU32("text_config.moe_intermediate_size") orelse
            f.getMetaU32("moe_intermediate_size") orelse default_moe_ff;

        // Canvas.
        self.canvas_length = f.getMetaU32("canvas_length") orelse default_canvas_length;

        // Vtable aliases.
        self.n_head = self.sl_n_head;
        self.n_head_kv = self.sl_n_kv_head;
        self.eos_token_id = f.getMetaU32("eos_token_id") orelse 1;

        const max_sl_hd = @max(self.sl_head_dim, self.gl_head_dim);
        const max_sl_nkv = @max(self.sl_n_kv_head, self.gl_n_kv_head);
        const max_sl_nh = @max(self.sl_n_head, self.gl_n_head);

        // KV cache setup.
        const max_seq = if (ctx_size > 0) @as(usize, ctx_size) else 4096;
        self.max_seq_len = max_seq;
        const nl: usize = self.n_layers;
        // kv_dim = elements per token position = max(sl_nkv*sl_hd, gl_nkv*gl_hd)
        const sl_kv_dim = @as(usize, self.sl_n_kv_head) * @as(usize, self.sl_head_dim);
        const gl_kv_dim = @as(usize, self.gl_n_kv_head) * @as(usize, self.gl_head_dim);
        const kv_dim = @max(sl_kv_dim, gl_kv_dim);
        const paged_block_size: u16 = 256;

        if (tiered_cache) |tc| {
            self.tiered_cache = tc;
            var ta = TieredBlockAllocator.init(tc, allocator);
            self.seq_table = try ta.allocateSeqTable(nl);
            errdefer ta.freeSeqTable(&self.seq_table);
            try ta.appendBlock(&self.seq_table);
            self.tiered_block_allocator = ta;
        } else {
            const blocks_per_layer = (max_seq + paged_block_size - 1) / paged_block_size;
            const num_blocks = nl * blocks_per_layer;
            self.paged_cache = try PagedKvCache.init(allocator, nl, kv_dim, num_blocks, paged_block_size);
            errdefer self.paged_cache.deinit();
            self.block_allocator = BlockAllocator.init(&self.paged_cache, allocator);
            self.seq_table = try self.block_allocator.allocateSeqTable(nl);
            errdefer self.block_allocator.freeSeqTable(&self.seq_table);
            try self.block_allocator.appendBlock(&self.seq_table);
        }

        const e: usize = self.n_embd;
        const qd_sl: usize = self.sl_n_head * self.sl_head_dim;
        const kvd_sl: usize = max_sl_nkv * max_sl_hd;
        const attn_out_dim: usize = @max(max_sl_nh * max_sl_hd, self.sl_n_head * self.sl_head_dim);
        const cl: usize = self.canvas_length;

        self.hidden = try allocator.alloc(f32, e);
        errdefer allocator.free(self.hidden);
        self.hidden2 = try allocator.alloc(f32, e);
        errdefer allocator.free(self.hidden2);
        self.hidden_pre_norm = try allocator.alloc(f32, e);
        errdefer allocator.free(self.hidden_pre_norm);
        self.q_buf = try allocator.alloc(f32, @max(qd_sl, self.gl_n_head * self.gl_head_dim));
        errdefer allocator.free(self.q_buf);
        self.k_buf = try allocator.alloc(f32, kvd_sl);
        errdefer allocator.free(self.k_buf);
        self.v_buf = try allocator.alloc(f32, kvd_sl);
        errdefer allocator.free(self.v_buf);
        self.attn_out = try allocator.alloc(f32, attn_out_dim);
        errdefer allocator.free(self.attn_out);
        self.scores_buf = try allocator.alloc(f32, max_seq + cl + 16); // prompt + canvas
        errdefer allocator.free(self.scores_buf);
        self.gate_buf = try allocator.alloc(f32, self.moe_ff);
        errdefer allocator.free(self.gate_buf);
        self.up_buf = try allocator.alloc(f32, self.moe_ff);
        errdefer allocator.free(self.up_buf);
        self.expert_out = try allocator.alloc(f32, e);
        errdefer allocator.free(self.expert_out);
        self.router_logits = try allocator.alloc(f32, self.n_experts);
        errdefer allocator.free(self.router_logits);
        self.logits_buf = try allocator.alloc(f32, self.vocab_size);
        errdefer allocator.free(self.logits_buf);
        self.canvas_tokens = try allocator.alloc(u32, cl);
        errdefer allocator.free(self.canvas_tokens);
        // Prefill batch buffers for canvas denoising.
        self.pf_hidden = try allocator.alloc(f32, cl * e);
        errdefer allocator.free(self.pf_hidden);
        self.pf_scratch = try allocator.alloc(f32, cl * e);
        errdefer allocator.free(self.pf_scratch);
        // pf_logits: allocated lazily in forwardCanvas (canvas_length * vocab_size can be large).
        self.pf_logits = try allocator.alloc(f32, cl * self.vocab_size);
        errdefer allocator.free(self.pf_logits);
        const max_kv_dim = @max(
            @as(usize, self.sl_n_kv_head) * @as(usize, self.sl_head_dim),
            @as(usize, self.gl_n_kv_head) * @as(usize, self.gl_head_dim),
        );
        self.canvas_k_buf = try allocator.alloc(f32, cl * max_kv_dim);
        errdefer allocator.free(self.canvas_k_buf);
        self.canvas_v_buf = try allocator.alloc(f32, cl * max_kv_dim);
        errdefer allocator.free(self.canvas_v_buf);

        self.warmNormCache();
        return self;
    }

    pub fn deinit(self: *DiffusionGemmaModel) void {
        const allocator = self.allocator;
        for (self.norm_cache[0..self.norm_cache_len]) |entry| allocator.free(entry.data);
        allocator.free(self.hidden);
        allocator.free(self.hidden2);
        allocator.free(self.hidden_pre_norm);
        allocator.free(self.q_buf);
        allocator.free(self.k_buf);
        allocator.free(self.v_buf);
        allocator.free(self.attn_out);
        allocator.free(self.scores_buf);
        allocator.free(self.gate_buf);
        allocator.free(self.up_buf);
        allocator.free(self.expert_out);
        allocator.free(self.router_logits);
        allocator.free(self.logits_buf);
        allocator.free(self.pf_hidden);
        allocator.free(self.pf_scratch);
        allocator.free(self.pf_logits);
        allocator.free(self.canvas_k_buf);
        allocator.free(self.canvas_v_buf);
        allocator.free(self.canvas_tokens);
        if (self.tiered_block_allocator) |*tba| {
            tba.freeSeqTable(&self.seq_table);
        } else {
            self.block_allocator.freeSeqTable(&self.seq_table);
            self.paged_cache.deinit();
        }
    }

    // ── Public API ────────────────────────────────────────────────

    /// Autoregressive single-token forward pass (encoder / prefill phase).
    /// Adds token to KV cache and returns the next-token argmax.
    pub fn forward(self: *DiffusionGemmaModel, token_id: u32) !u32 {
        if (self.kv_seq_len >= self.max_seq_len) return error.KVCacheFull;
        try model_mod.ensureKvBlock(self);

        const emb = self.fmt.getTensor("model.decoder.embed_tokens.weight") orelse return error.MissingTensor;
        self.be.embLookup(.{ .data = emb.data_ptr, .dtype = emb.dtype }, token_id, self.hidden.ptr, self.n_embd);
        self.be.sync();
        // Gemma-family embedding scaling: hidden *= sqrt(n_embd).
        for (self.hidden) |*v| v.* *= self.emb_scale;

        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.monotonic)) return error.Cancelled;
            const l: u32 = @intCast(li);
            if (l >= self.layer_skip_start and l < self.layer_skip_end) continue;
            try self.encoderLayer(l);
        }

        const nw = self.fmt.getTensor("model.decoder.norm.weight") orelse return error.MissingTensor;
        const ow = self.fmt.getTensor("model.decoder.embed_tokens.weight") orelse return error.MissingTensor;
        self.kv_seq_len += 1;

        self.be.sync();
        @memcpy(self.hidden_pre_norm, self.hidden);
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, self.n_embd), self.hidden.ptr, self.n_embd, self.rms_eps);
        self.be.gemv(self.hidden.ptr, .{ .data = ow.data_ptr, .dtype = ow.dtype }, self.logits_buf.ptr, self.vocab_size, self.n_embd);
        self.be.sync();
        return math_ops.argmax(self.logits_buf);
    }

    /// Sequential token prefill — runs forward() for each token.
    pub fn prefill(self: *DiffusionGemmaModel, token_ids: []const u32) !u32 {
        var last: u32 = 0;
        for (token_ids) |tid| last = try self.forward(tid);
        return last;
    }

    /// Canvas denoising pass: run all canvas_length tokens through the model with
    /// bidirectional attention within the canvas block. Writes per-position logits
    /// into `logits_out` (shape [canvas_length, vocab_size]).
    /// The prompt KV cache (from prior prefill) is used for attention.
    pub fn forwardCanvas(self: *DiffusionGemmaModel, canvas: []const u32, logits_out: []f32) !void {
        const e: usize = self.n_embd;
        const cl: usize = canvas.len;
        const emb_t = self.fmt.getTensor("model.decoder.embed_tokens.weight") orelse return error.MissingTensor;

        // Embed each canvas token into pf_hidden[0..cl*e] with Gemma scaling.
        for (canvas, 0..) |tok, i| {
            const out = self.pf_hidden[i * e ..][0..e];
            self.be.embLookup(.{ .data = emb_t.data_ptr, .dtype = emb_t.dtype }, tok, out.ptr, e);
            self.be.sync();
            for (out) |*v| v.* *= self.emb_scale;
        }

        // Run all layers with bidirectional attention for canvas range.
        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.monotonic)) return error.Cancelled;
            const l: u32 = @intCast(li);
            try self.canvasLayer(l, cl);
        }

        // Final norm + LM head for each canvas position.
        const nw = self.fmt.getTensor("model.decoder.norm.weight") orelse return error.MissingTensor;
        const ow = emb_t; // tied weights
        const nw_f32 = self.normAsF32(nw, e);

        for (0..cl) |i| {
            const h = self.pf_hidden[i * e ..][0..e];
            self.be.sync();
            self.be.rmsNorm(h.ptr, nw_f32, h.ptr, e, self.rms_eps);
            self.be.gemv(h.ptr, .{ .data = ow.data_ptr, .dtype = ow.dtype }, logits_out[i * self.vocab_size ..].ptr, self.vocab_size, e);
            self.be.sync();
        }
    }

    /// Reset KV cache and SSM state for a new conversation.
    pub fn resetCache(self: *DiffusionGemmaModel) void {
        model_mod.resetKvCache(self);
    }

    pub fn cancel(self: *DiffusionGemmaModel) void {
        model_mod.signalCancel(&self.cancelled);
    }

    pub fn getLogits(self: *const DiffusionGemmaModel) []const f32 {
        return self.logits_buf;
    }

    pub fn getHiddenState(self: *const DiffusionGemmaModel) []const f32 {
        return self.hidden;
    }

    pub fn getPreNormHiddenState(self: *const DiffusionGemmaModel) []const f32 {
        if (self.hidden_pre_norm.len > 0) return self.hidden_pre_norm;
        return self.hidden;
    }

    pub fn getBlockTable(self: *DiffusionGemmaModel) []const u32 {
        return self.seq_table.block_table[0];
    }

    // ── Internal helpers ──────────────────────────────────────────

    /// True when layer li uses global (non-sliding-window) attention.
    inline fn isGlobalLayer(li: u32) bool {
        return (li + 1) % global_attn_stride == 0;
    }

    /// Get a tensor from model.decoder.layers.{li}.{suffix}.
    fn lt(self: *DiffusionGemmaModel, li: u32, suffix: []const u8) ?TensorInfo {
        var buf: [name_buf_size]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, "model.decoder.layers.{d}.{s}", .{ li, suffix }) catch return null;
        return self.fmt.getTensor(name);
    }

    /// Convert norm weights to a stable f32 pointer. Caches BF16→f32 conversions
    /// so GPU backends never bind a reused scratch buffer.
    fn normAsF32(self: *DiffusionGemmaModel, t: TensorInfo, n: usize) [*]const f32 {
        if (t.dtype == .f32) return @ptrCast(@alignCast(t.data_ptr));

        const key = @intFromPtr(t.data_ptr);
        for (self.norm_cache[0..self.norm_cache_len]) |entry| {
            if (entry.key == key) return entry.data.ptr;
        }

        if (self.norm_cache_len >= max_norm_entries)
            @panic("normAsF32: norm cache overflow — increase max_norm_entries");
        const buf = self.allocator.alloc(f32, n) catch @panic("normAsF32: out of memory converting norm weights");
        quant.dequantToF32(buf, t.data_ptr, t.dtype, n);
        self.norm_cache[self.norm_cache_len] = .{ .key = key, .data = buf };
        self.norm_cache_len += 1;
        return buf.ptr;
    }

    /// Pre-convert BF16 norms at init so inference never allocates.
    fn warmNormCache(self: *DiffusionGemmaModel) void {
        const e: usize = self.n_embd;
        const sl_hd: usize = self.sl_head_dim;
        const gl_hd: usize = self.gl_head_dim;
        if (self.fmt.getTensor("model.decoder.norm.weight")) |t| _ = self.normAsF32(t, e);
        for (0..self.n_layers) |li| {
            const l: u32 = @intCast(li);
            const hd: usize = if (isGlobalLayer(l)) gl_hd else sl_hd;
            if (self.lt(l, "input_layernorm.weight")) |t| _ = self.normAsF32(t, e);
            if (self.lt(l, "self_attn.q_norm.weight")) |t| _ = self.normAsF32(t, hd);
            if (self.lt(l, "self_attn.k_norm.weight")) |t| _ = self.normAsF32(t, hd);
            if (self.lt(l, "post_attention_layernorm.weight")) |t| _ = self.normAsF32(t, e);
            if (self.lt(l, "pre_feedforward_layernorm.weight")) |t| _ = self.normAsF32(t, e);
            if (self.lt(l, "post_feedforward_layernorm.weight")) |t| _ = self.normAsF32(t, e);
        }
    }

    // ── KV cache view helpers ─────────────────────────────────────

    fn getLayerKvView(self: *DiffusionGemmaModel, layer: usize) struct { keys: []u8, values: []u8 } {
        if (self.seq_table.block_table[layer].len == 0) return .{ .keys = &[_]u8{}, .values = &[_]u8{} };
        const block_id = self.seq_table.block_table[layer][0];
        if (self.tiered_cache) |tc| {
            return .{
                .keys = std.mem.sliceAsBytes(tc.blocks[block_id].base.keys),
                .values = std.mem.sliceAsBytes(tc.blocks[block_id].base.values),
            };
        }
        return .{
            .keys = std.mem.sliceAsBytes(self.paged_cache.blocks[block_id].keys),
            .values = std.mem.sliceAsBytes(self.paged_cache.blocks[block_id].values),
        };
    }

    const PagedKvView = kvcache.PagedKvView;

    fn getPagedKvView(self: *DiffusionGemmaModel, layer: usize) PagedKvView {
        return PagedKvView.initView(
            self.seq_table.block_table[layer],
            self.paged_cache.blocks,
            self.paged_cache.block_size,
            self.paged_cache.kv_dim,
            self.kv_seq_len,
        );
    }

    fn isMultiBlock(self: *DiffusionGemmaModel, layer: usize) bool {
        return self.paged_cache.block_size > 0 and self.seq_table.block_table[layer].len > 1;
    }

    // ── Layer implementations (encoder / autoregressive) ──────────

    /// Single-token encoder layer: attention + MoE/dense FFN with residual.
    fn encoderLayer(self: *DiffusionGemmaModel, li: u32) !void {
        const e: usize = self.n_embd;
        const is_global = isGlobalLayer(li);
        const nh: usize = if (is_global) self.gl_n_head else self.sl_n_head;
        const nkv: usize = if (is_global) self.gl_n_kv_head else self.sl_n_kv_head;
        const hd: usize = if (is_global) self.gl_head_dim else self.sl_head_dim;
        const qd: usize = nh * hd;
        const kvd: usize = nkv * hd;
        const rope_theta = if (is_global) self.gl_rope_theta else self.sl_rope_theta;
        const rope_dim: u32 = if (is_global) self.gl_rope_dim else self.sl_rope_dim;

        // 1. Pre-attention norm.
        const nw_in = self.lt(li, "input_layernorm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw_in, e), self.hidden2.ptr, e, self.rms_eps);

        // 2. Q/K/V projections.
        const qw = self.lt(li, "self_attn.q_proj.weight") orelse return error.MissingTensor;
        const kw = self.lt(li, "self_attn.k_proj.weight") orelse return error.MissingTensor;
        const vw = self.lt(li, "self_attn.v_proj.weight") orelse return error.MissingTensor;
        self.be.beginBatch();
        self.be.gemv(self.hidden2.ptr, .{ .data = qw.data_ptr, .dtype = qw.dtype }, self.q_buf.ptr, qd, e);
        self.be.gemv(self.hidden2.ptr, .{ .data = kw.data_ptr, .dtype = kw.dtype }, self.k_buf.ptr, kvd, e);
        self.be.gemv(self.hidden2.ptr, .{ .data = vw.data_ptr, .dtype = vw.dtype }, self.v_buf.ptr, kvd, e);
        self.be.endBatch();

        // QK norms (Gemma 4 style).
        if (self.lt(li, "self_attn.q_norm.weight")) |qnw| {
            self.be.rmsNormBatched(self.q_buf.ptr, self.normAsF32(qnw, hd), self.q_buf.ptr, nh, hd, self.rms_eps);
        }
        if (self.lt(li, "self_attn.k_norm.weight")) |knw| {
            self.be.rmsNormBatched(self.k_buf.ptr, self.normAsF32(knw, hd), self.k_buf.ptr, nkv, hd, self.rms_eps);
        }

        // 3. RoPE.
        self.be.beginBatch();
        self.be.rope(self.q_buf.ptr, self.kv_seq_len, nh, hd, rope_dim, rope_theta);
        self.be.rope(self.k_buf.ptr, self.kv_seq_len, nkv, hd, rope_dim, rope_theta);
        self.be.endBatch();

        // 4. KV cache + attention.
        const kv_view = self.getLayerKvView(li);
        if (self.isMultiBlock(li)) {
            self.be.sdpaPaged(
                self.q_buf.ptr,
                self.getPagedKvView(li),
                self.k_buf.ptr,
                self.v_buf.ptr,
                self.attn_out.ptr,
                nh,
                nkv,
                hd,
                1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
                self.kv_type_k,
                self.kv_type_v,
            );
        } else {
            const sliding: usize = if (!is_global) self.sliding_window else 0;
            // Full attention (no sliding window) — simpler for v1 DiffusionGemma.
            // DiffusionGemma uses sliding window 1024 for sliding layers, but we use full
            // attention here since the encoder is short (prompt only).
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
                self.kv_type_k,
                self.kv_type_v,
            );
            _ = sliding; // sliding window enforcement not needed for short prompts
        }

        // 5. Optional layer scalar on attention output.
        if (self.lt(li, "layer_scalar")) |ls| {
            self.be.sync();
            const scalar_f32: f32 = if (ls.dtype == .f32)
                @as(*const f32, @ptrCast(@alignCast(ls.data_ptr))).*
            else blk: {
                // BF16 scalar — convert.
                const bits = @as(*const u16, @ptrCast(@alignCast(ls.data_ptr))).*;
                break :blk @bitCast(@as(u32, bits) << 16);
            };
            if (scalar_f32 != 1.0) {
                for (self.attn_out[0..qd]) |*v| v.* *= scalar_f32;
            }
        }

        // 6. Output projection + post-attn norm.
        const ow = self.lt(li, "self_attn.o_proj.weight") orelse return error.MissingTensor;
        self.be.gemv(self.attn_out.ptr, .{ .data = ow.data_ptr, .dtype = ow.dtype }, self.hidden2.ptr, e, qd);

        if (self.lt(li, "post_attention_layernorm.weight")) |pan| {
            self.be.rmsNorm(self.hidden2.ptr, self.normAsF32(pan, e), self.hidden2.ptr, e, self.rms_eps);
        }

        // 7. Residual: hidden += hidden2.
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);

        // 8. Pre-FFN norm.
        const nw_pre_ffn = self.lt(li, "pre_feedforward_layernorm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw_pre_ffn, e), self.hidden2.ptr, e, self.rms_eps);

        // 9. MoE forward (if this is an expert layer).
        if (self.n_experts > 0 and self.lt(li, "router.proj.weight") != null) {
            try self.encoderMoeLayer(li);
        } else {
            try self.encoderDenseFfn(li);
        }

        // 10. Post-FFN norm + residual.
        if (self.lt(li, "post_feedforward_layernorm.weight")) |pffn| {
            self.be.rmsNorm(self.hidden2.ptr, self.normAsF32(pffn, e), self.hidden2.ptr, e, self.rms_eps);
        }
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
    }

    /// MoE forward for encoder (single token).
    fn encoderMoeLayer(self: *DiffusionGemmaModel, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.moe_ff;
        const k: usize = self.top_k_experts;
        const n_exp: usize = self.n_experts;

        // Router: compute logits and select top-k experts.
        const rw = self.lt(li, "router.proj.weight") orelse return error.MissingTensor;
        self.be.gemv(self.hidden2.ptr, .{ .data = rw.data_ptr, .dtype = rw.dtype }, self.router_logits.ptr, n_exp, e);
        self.be.sync();

        var top_ids: [16]usize = undefined;
        var top_weights: [16]f32 = undefined;
        math_ops.topKExperts(self.router_logits[0..n_exp], k, top_ids[0..k], top_weights[0..k]);
        const n_sel = k;

        // Softmax over selected weights.
        var w_sum: f32 = 0;
        for (top_weights[0..n_sel]) |*w| {
            w.* = @exp(w.*);
            w_sum += w.*;
        }
        const inv_w = 1.0 / w_sum;
        for (top_weights[0..n_sel]) |*w| w.* *= inv_w;

        // Accumulate expert outputs.
        @memset(self.expert_out, 0);

        // Fused gate_up_proj: shape [n_experts * 2 * ff, e] laid out row-major.
        // Expert ei occupies rows [ei * 2*ff, (ei+1) * 2*ff).
        const gu = self.lt(li, "experts.gate_up_proj") orelse return error.MissingTensor;
        const dw = self.lt(li, "experts.down_proj") orelse return error.MissingTensor;
        const expert_gate_up_stride: usize = 2 * ff * e; // bytes per expert in gate_up (for u8 indexing)
        const expert_down_stride: usize = e * ff;

        for (top_ids[0..n_sel], top_weights[0..n_sel]) |ei, weight| {
            // Gate rows: [ei*2*ff + 0, ei*2*ff + ff) × hidden.
            const gu_base: [*]const u8 = @ptrCast(gu.data_ptr);
            const gate_ptr: [*]const u8 = gu_base + ei * expert_gate_up_stride * @sizeOf(u16);
            const up_ptr: [*]const u8 = gu_base + (ei * 2 * ff + ff) * e * @sizeOf(u16);

            self.be.gemv(self.hidden2.ptr, .{ .data = gate_ptr, .dtype = gu.dtype }, self.gate_buf.ptr, ff, e);
            self.be.gemv(self.hidden2.ptr, .{ .data = up_ptr, .dtype = gu.dtype }, self.up_buf.ptr, ff, e);
            self.be.sync();

            self.be.siluMul(self.gate_buf.ptr, self.up_buf.ptr, self.gate_buf.ptr, ff);
            self.be.sync();

            // Down projection.
            const dw_base: [*]const u8 = @ptrCast(dw.data_ptr);
            const down_ptr: [*]const u8 = dw_base + ei * expert_down_stride * @sizeOf(u16);
            self.be.gemv(self.gate_buf.ptr, .{ .data = down_ptr, .dtype = dw.dtype }, self.hidden2.ptr, e, ff);
            self.be.sync();

            // Weighted accumulate.
            for (self.expert_out, self.hidden2) |*o, v| o.* += weight * v;
        }

        @memcpy(self.hidden2, self.expert_out);
    }

    /// Dense FFN forward for encoder (single token). Used for layers without MoE.
    fn encoderDenseFfn(self: *DiffusionGemmaModel, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.moe_ff * 2; // dense uses larger intermediate

        const gw = self.lt(li, "mlp.gate_proj.weight") orelse return error.MissingTensor;
        const uw = self.lt(li, "mlp.up_proj.weight") orelse return error.MissingTensor;
        const dw = self.lt(li, "mlp.down_proj.weight") orelse return error.MissingTensor;

        self.be.beginBatch();
        self.be.gemv(self.hidden2.ptr, .{ .data = gw.data_ptr, .dtype = gw.dtype }, self.gate_buf.ptr, ff, e);
        self.be.gemv(self.hidden2.ptr, .{ .data = uw.data_ptr, .dtype = uw.dtype }, self.up_buf.ptr, ff, e);
        self.be.endBatch();
        self.be.siluMul(self.gate_buf.ptr, self.up_buf.ptr, self.gate_buf.ptr, ff);
        self.be.gemv(self.gate_buf.ptr, .{ .data = dw.data_ptr, .dtype = dw.dtype }, self.hidden2.ptr, e, ff);
    }

    // ── Canvas denoising layer (bidirectional attention) ──────────

    /// Canvas layer: processes cl tokens with bidirectional attention in the canvas
    /// range. pf_hidden[0..cl*e] holds the per-token hidden states.
    ///
    /// v1 implementation: iterates tokens sequentially (cl calls per layer).
    /// Future optimization: batch all cl tokens through a single GEMM dispatch
    /// using be.gemm() with batch size cl, reducing GPU dispatch overhead
    /// from O(cl * n_layers) to O(n_layers) per denoising step.
    fn canvasLayer(self: *DiffusionGemmaModel, li: u32, cl: usize) !void {
        const e: usize = self.n_embd;
        const is_global = isGlobalLayer(li);
        const nh: usize = if (is_global) self.gl_n_head else self.sl_n_head;
        const nkv: usize = if (is_global) self.gl_n_kv_head else self.sl_n_kv_head;
        const hd: usize = if (is_global) self.gl_head_dim else self.sl_head_dim;
        const qd: usize = nh * hd;
        const rope_theta = if (is_global) self.gl_rope_theta else self.sl_rope_theta;
        const rope_dim: u32 = if (is_global) self.gl_rope_dim else self.sl_rope_dim;

        // 1. Pre-attention norm for all canvas tokens (batched: cl vectors of length e).
        const nw_in = self.lt(li, "input_layernorm.weight") orelse return error.MissingTensor;
        self.be.rmsNormBatched(self.pf_hidden.ptr, self.normAsF32(nw_in, e), self.pf_scratch.ptr, cl, e, self.rms_eps);

        // 2. QKV projection for all canvas tokens via batched GEMM (cl tokens at once).
        const qw = self.lt(li, "self_attn.q_proj.weight") orelse return error.MissingTensor;
        const kw = self.lt(li, "self_attn.k_proj.weight") orelse return error.MissingTensor;
        const vw = self.lt(li, "self_attn.v_proj.weight") orelse return error.MissingTensor;
        const kvd: usize = nkv * hd;

        // Use pre-allocated canvas K/V buffers (sized for max kvd = gl_n_kv_head * gl_head_dim).
        const canvas_k = self.canvas_k_buf[0 .. cl * kvd];
        const canvas_v = self.canvas_v_buf[0 .. cl * kvd];

        // Batched GEMM: Q[cl, qd] = pf_scratch[cl, e] @ Wq^T[qd, e]
        self.be.beginBatch();
        self.be.gemm(self.pf_scratch.ptr, .{ .data = qw.data_ptr, .dtype = qw.dtype }, self.q_buf.ptr, cl, qd, e);
        self.be.gemm(self.pf_scratch.ptr, .{ .data = kw.data_ptr, .dtype = kw.dtype }, canvas_k.ptr, cl, kvd, e);
        self.be.gemm(self.pf_scratch.ptr, .{ .data = vw.data_ptr, .dtype = vw.dtype }, canvas_v.ptr, cl, kvd, e);
        self.be.endBatch();

        // QK norms (batched: process cl*nh heads, each of size hd).
        if (self.lt(li, "self_attn.q_norm.weight")) |qnw| {
            self.be.rmsNormBatched(self.q_buf.ptr, self.normAsF32(qnw, hd), self.q_buf.ptr, cl * nh, hd, self.rms_eps);
        }
        if (self.lt(li, "self_attn.k_norm.weight")) |knw| {
            self.be.rmsNormBatched(canvas_k.ptr, self.normAsF32(knw, hd), canvas_k.ptr, cl * nkv, hd, self.rms_eps);
        }

        // 3. RoPE — canvas tokens start at kv_seq_len in position space.
        for (0..cl) |i| {
            const pos = self.kv_seq_len + i;
            self.be.rope(self.q_buf.ptr + i * qd, pos, nh, hd, rope_dim, rope_theta);
            self.be.rope(canvas_k.ptr + i * kvd, pos, nkv, hd, rope_dim, rope_theta);
        }

        // 4. Attention with bidirectional canvas (treat canvas as image region).
        // Each canvas token attends to: all prompt KV (in cache) + all other canvas tokens.
        // We use CPU-side SDPA with the image region mechanism (no causal mask in canvas).
        const ow = self.lt(li, "self_attn.o_proj.weight") orelse return error.MissingTensor;

        const kv_view = self.getLayerKvView(li);
        self.be.sync();

        for (0..cl) |i| {
            // Build combined K (prompt KV cache + canvas K) and V (prompt KV cache + canvas V).
            // The canvas token sees all prompt tokens causally + all canvas tokens bidirectionally.
            // We implement this as: full attention over [prompt_kv | canvas_kv],
            // where causal mask is disabled for canvas-to-canvas attention.
            // For simplicity in v1: use full attention over combined KV (slight over-attending for later canvas).
            const q_ptr = self.q_buf.ptr + i * qd;
            attn_ops.scaledDotProductAttentionCanvas(
                q_ptr,
                kv_view.keys,
                kv_view.values,
                canvas_k,
                canvas_v,
                self.attn_out.ptr,
                self.scores_buf.ptr,
                nh,
                nkv,
                hd,
                self.kv_seq_len,
                cl,
                1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
                self.be,
                self.kv_type_k,
                self.kv_type_v,
            );

            // Output projection.
            const attn_out_dim = qd;
            self.be.gemv(self.attn_out.ptr, .{ .data = ow.data_ptr, .dtype = ow.dtype }, self.pf_scratch[i * e ..].ptr, e, attn_out_dim);
        }

        // Post-attn norm (batched) + residual add (batched).
        if (self.lt(li, "post_attention_layernorm.weight")) |pan| {
            self.be.rmsNormBatched(self.pf_scratch.ptr, self.normAsF32(pan, e), self.pf_scratch.ptr, cl, e, self.rms_eps);
        }
        // Element-wise add: pf_hidden += pf_scratch (cl * e elements).
        self.be.sync();
        for (0..cl * e) |i| self.pf_hidden[i] += self.pf_scratch[i];

        // Pre-FFN norm (batched) for all canvas tokens.
        const nw_pre_ffn = self.lt(li, "pre_feedforward_layernorm.weight") orelse return error.MissingTensor;
        self.be.rmsNormBatched(self.pf_hidden.ptr, self.normAsF32(nw_pre_ffn, e), self.pf_scratch.ptr, cl, e, self.rms_eps);

        // MoE / dense FFN for each canvas token.
        for (0..cl) |i| {
            if (self.n_experts > 0 and self.lt(li, "router.proj.weight") != null) {
                try self.canvasMoeToken(li, self.pf_scratch[i * e ..].ptr, self.pf_hidden[i * e ..].ptr);
            } else {
                try self.canvasDenseFfnToken(li, self.pf_scratch[i * e ..].ptr, self.pf_hidden[i * e ..].ptr);
            }
        }

        // Post-FFN norm (batched) + residual add.
        if (self.lt(li, "post_feedforward_layernorm.weight")) |pffn| {
            self.be.sync();
            self.be.rmsNormBatched(self.pf_scratch.ptr, self.normAsF32(pffn, e), self.pf_scratch.ptr, cl, e, self.rms_eps);
            self.be.sync();
            for (0..cl * e) |i| self.pf_hidden[i] += self.pf_scratch[i];
        }
    }

    /// MoE for a single canvas token. inp = normed input, out = residual accumulation.
    fn canvasMoeToken(self: *DiffusionGemmaModel, li: u32, inp: [*]const f32, out: [*]f32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.moe_ff;
        const k: usize = self.top_k_experts;
        const n_exp: usize = self.n_experts;

        const rw = self.lt(li, "router.proj.weight") orelse return error.MissingTensor;
        self.be.gemv(inp, .{ .data = rw.data_ptr, .dtype = rw.dtype }, self.router_logits.ptr, n_exp, e);
        self.be.sync();

        var top_ids: [16]usize = undefined;
        var top_weights: [16]f32 = undefined;
        math_ops.topKExperts(self.router_logits[0..n_exp], k, top_ids[0..k], top_weights[0..k]);
        const n_sel = k;

        var w_sum: f32 = 0;
        for (top_weights[0..n_sel]) |*w| {
            w.* = @exp(w.*);
            w_sum += w.*;
        }
        const inv_w = 1.0 / w_sum;
        for (top_weights[0..n_sel]) |*w| w.* *= inv_w;

        @memset(self.expert_out, 0);

        const gu = self.lt(li, "experts.gate_up_proj") orelse return error.MissingTensor;
        const dw = self.lt(li, "experts.down_proj") orelse return error.MissingTensor;
        const expert_gate_up_stride: usize = 2 * ff * e;
        const expert_down_stride: usize = e * ff;

        for (top_ids[0..n_sel], top_weights[0..n_sel]) |ei, weight| {
            const gu_base: [*]const u8 = @ptrCast(gu.data_ptr);
            const gate_ptr2: [*]const u8 = gu_base + ei * expert_gate_up_stride * @sizeOf(u16);
            const up_ptr2: [*]const u8 = gu_base + (ei * 2 * ff + ff) * e * @sizeOf(u16);

            self.be.gemv(inp, .{ .data = gate_ptr2, .dtype = gu.dtype }, self.gate_buf.ptr, ff, e);
            self.be.gemv(inp, .{ .data = up_ptr2, .dtype = gu.dtype }, self.up_buf.ptr, ff, e);
            self.be.sync();
            self.be.siluMul(self.gate_buf.ptr, self.up_buf.ptr, self.gate_buf.ptr, ff);
            self.be.sync();

            const dw_base: [*]const u8 = @ptrCast(dw.data_ptr);
            const down_ptr2: [*]const u8 = dw_base + ei * expert_down_stride * @sizeOf(u16);
            self.be.gemv(self.gate_buf.ptr, .{ .data = down_ptr2, .dtype = dw.dtype }, self.hidden2.ptr, e, ff);
            self.be.sync();

            for (self.expert_out, self.hidden2) |*o, v| o.* += weight * v;
        }

        self.be.add(out, self.expert_out.ptr, out, e);
    }

    /// Dense FFN for a single canvas token.
    fn canvasDenseFfnToken(self: *DiffusionGemmaModel, li: u32, inp: [*]const f32, out: [*]f32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.moe_ff * 2;

        const gw = self.lt(li, "mlp.gate_proj.weight") orelse return error.MissingTensor;
        const uw = self.lt(li, "mlp.up_proj.weight") orelse return error.MissingTensor;
        const dw = self.lt(li, "mlp.down_proj.weight") orelse return error.MissingTensor;

        self.be.beginBatch();
        self.be.gemv(inp, .{ .data = gw.data_ptr, .dtype = gw.dtype }, self.gate_buf.ptr, ff, e);
        self.be.gemv(inp, .{ .data = uw.data_ptr, .dtype = uw.dtype }, self.up_buf.ptr, ff, e);
        self.be.endBatch();
        self.be.siluMul(self.gate_buf.ptr, self.up_buf.ptr, self.gate_buf.ptr, ff);
        self.be.gemv(self.gate_buf.ptr, .{ .data = dw.data_ptr, .dtype = dw.dtype }, self.hidden2.ptr, e, ff);
        self.be.sync();
        self.be.add(out, self.hidden2.ptr, out, e);
    }
};

// ── Tests ───────────────────────────────────────────────────────────────────

test "isGlobalLayer pattern: every 6th layer" {
    // global_attn_stride = 6, so global layers are at (li+1) % 6 == 0.
    // Layer 5 (6th, 0-indexed): global. Layer 11: global. Layer 0-4: sliding.
    try std.testing.expect(DiffusionGemmaModel.isGlobalLayer(5));
    try std.testing.expect(DiffusionGemmaModel.isGlobalLayer(11));
    try std.testing.expect(DiffusionGemmaModel.isGlobalLayer(17));
    try std.testing.expect(DiffusionGemmaModel.isGlobalLayer(23));
    try std.testing.expect(DiffusionGemmaModel.isGlobalLayer(29));

    // Sliding-window layers:
    try std.testing.expect(!DiffusionGemmaModel.isGlobalLayer(0));
    try std.testing.expect(!DiffusionGemmaModel.isGlobalLayer(1));
    try std.testing.expect(!DiffusionGemmaModel.isGlobalLayer(4));
    try std.testing.expect(!DiffusionGemmaModel.isGlobalLayer(6));
    try std.testing.expect(!DiffusionGemmaModel.isGlobalLayer(10));
}

test "architecture constants are consistent" {
    // Default 30 layers with stride 6 → 5 global layers (5,11,17,23,29)
    var n_global: u32 = 0;
    for (0..default_n_layers) |li| {
        if (DiffusionGemmaModel.isGlobalLayer(@intCast(li))) n_global += 1;
    }
    try std.testing.expectEqual(@as(u32, 5), n_global);

    // Canvas length must be a power of 2 (for alignment in batch processing)
    try std.testing.expect(std.math.isPowerOfTwo(default_canvas_length));

    // Global head dim > sliding-window head dim (512 vs 256)
    try std.testing.expect(default_gl_head_dim > default_sl_head_dim);

    // Sliding-window rope theta < global rope theta
    try std.testing.expect(default_sl_rope_theta < default_gl_rope_theta);
}

test "default model field defaults match constants" {
    const m = DiffusionGemmaModel{
        .fmt = undefined,
        .be = undefined,
        .allocator = undefined,
    };
    try std.testing.expectEqual(default_n_layers, m.n_layers);
    try std.testing.expectEqual(default_n_embd, m.n_embd);
    try std.testing.expectEqual(default_vocab_size, m.vocab_size);
    try std.testing.expectEqual(default_canvas_length, m.canvas_length);
    try std.testing.expectEqual(default_n_experts, m.n_experts);
    try std.testing.expectEqual(default_top_k_experts, m.top_k_experts);
    try std.testing.expectEqual(default_sl_n_head, m.sl_n_head);
    try std.testing.expectEqual(default_gl_n_head, m.gl_n_head);
}

test "global_attn_stride divides default_n_layers" {
    // Ensures the architecture has a clean global/sliding pattern with no remainder
    try std.testing.expectEqual(@as(u32, 0), default_n_layers % global_attn_stride);
}
