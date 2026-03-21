//! Gemma 3 transformer model implementation.
//! Supports GQA (Grouped Query Attention), GELU activation, post-norms,
//! RoPE position embeddings, and per-head QK normalization.

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
const Backend = backend_mod.Backend;
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const DType = backend_mod.DType;
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;

/// Gemma 3 transformer model with GQA, GELU activation, and per-head QK normalization.
/// Supports both GGUF and SafeTensors/MLX quantized weights.
pub const Gemma3Model = struct {
    const NormCacheEntry = struct { key: usize, data: []f32 };
    const max_norm_entries: usize = 256;

    // Configuration (read from GGUF metadata)
    n_layers: u32,
    n_embd: u32,
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    n_ff: u32,
    vocab_size: u32,
    rope_theta: f32,
    rope_local_theta: f32,
    sliding_window_pattern: u32,
    rms_eps: f32,
    eos_token_id: u32,
    attn_scale: f32,
    embd_scale: f32,
    final_logit_softcap: f32,
    mlx_bits: u32,
    /// Whether norm weights need +1.0 offset (SafeTensors Gemma stores raw weights;
    /// GGUF bakes +1.0 into norm weights).
    norm_add_one: bool,
    max_seq_len: usize = 4096,

    // Dependencies
    fmt: Format,
    be: Backend,
    allocator: Allocator,

    // Working buffers
    hidden: []f32,
    hidden2: []f32,
    q_buf: []f32,
    k_buf: []f32,
    v_buf: []f32,
    attn_out: []f32,
    ff_gate: []f32,
    ff_up: []f32,
    logits: []f32,
    scores: []f32,
    /// Cached pre-computed f32 norm weights (lazily populated on first token).
    /// Eliminates per-token dequantization and all norm-buffer GPU sync points.
    norm_cache: [max_norm_entries]NormCacheEntry = undefined,
    norm_cache_len: usize = 0,
    /// MLX companion tensor cache (scales/biases pointers keyed by weight ptr).
    mlx_cc_keys: [mlx_companion_cache_size]usize = [_]usize{0} ** mlx_companion_cache_size,
    mlx_cc_vals: [mlx_companion_cache_size]MlxCompanion = undefined,

    // KV cache
    kv_keys: [][]f32,
    kv_values: [][]f32,
    kv_seq_len: usize = 0,
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    perf: perf.PerfCounters = .{},

    /// Initialize the model from format metadata and allocate all working buffers.
    pub fn init(allocator: Allocator, f: Format, be: Backend, ctx_size: u32) !Gemma3Model {
        const arch = f.getMetaStr("general.architecture") orelse "gemma3";
        const n_layers = f.getArchU32(arch, "block_count") orelse 26;
        const n_embd = f.getArchU32(arch, "embedding_length") orelse 1152;
        const n_head = f.getArchU32(arch, "attention.head_count") orelse 4;
        const n_head_kv = f.getArchU32(arch, "attention.head_count_kv") orelse 1;
        const head_dim = f.getArchU32(arch, "attention.key_length") orelse 256;
        const n_ff = f.getArchU32(arch, "feed_forward_length") orelse 6912;
        const vocab_size: u32 = if (f.getVocab()) |v| @intCast(v.len) else 262144;

        const qkv_dim = n_head * head_dim;
        const kv_dim = n_head_kv * head_dim;
        const nl: usize = n_layers;

        var max_sl: usize = 4096;
        if (f.getArchU32(arch, "context_length")) |cl| max_sl = cl;
        if (ctx_size > 0) max_sl = ctx_size;

        const kv_keys = try allocator.alloc([]f32, nl);
        errdefer allocator.free(kv_keys);
        const kv_values = try allocator.alloc([]f32, nl);
        errdefer allocator.free(kv_values);

        var layer_init_count: usize = 0;
        errdefer {
            for (0..layer_init_count) |i| {
                be.freeKvSlice(allocator, kv_keys[i]);
                be.freeKvSlice(allocator, kv_values[i]);
            }
        }
        for (0..nl) |i| {
            kv_keys[i] = try be.allocKvSlice(allocator, max_sl * kv_dim);
            kv_values[i] = try be.allocKvSlice(allocator, max_sl * kv_dim);
            layer_init_count += 1;
        }

        const hidden = try allocator.alloc(f32, n_embd);
        errdefer allocator.free(hidden);
        const hidden2 = try allocator.alloc(f32, n_embd);
        errdefer allocator.free(hidden2);
        const q_buf = try allocator.alloc(f32, qkv_dim);
        errdefer allocator.free(q_buf);
        const k_buf = try allocator.alloc(f32, kv_dim);
        errdefer allocator.free(k_buf);
        const v_buf = try allocator.alloc(f32, kv_dim);
        errdefer allocator.free(v_buf);
        const attn_out = try allocator.alloc(f32, qkv_dim);
        errdefer allocator.free(attn_out);
        const ff_gate = try allocator.alloc(f32, n_ff);
        errdefer allocator.free(ff_gate);
        const ff_up = try allocator.alloc(f32, n_ff);
        errdefer allocator.free(ff_up);
        const logits = try allocator.alloc(f32, vocab_size);
        errdefer allocator.free(logits);
        const scores = try allocator.alloc(f32, max_sl);
        errdefer allocator.free(scores);
        return .{
            .n_layers = n_layers,
            .n_embd = n_embd,
            .n_head = n_head,
            .n_head_kv = n_head_kv,
            .head_dim = head_dim,
            .n_ff = n_ff,
            .vocab_size = vocab_size,
            .rope_theta = f.getArchF32(arch, "rope.freq_base") orelse 1_000_000.0,
            .rope_local_theta = f.getMetaF32("rope_local_base_freq") orelse
                f.getArchF32(arch, "rope.freq_base") orelse 1_000_000.0,
            .sliding_window_pattern = f.getMetaU32("sliding_window_pattern") orelse 0,
            .rms_eps = f.getArchF32(arch, "attention.layer_norm_rms_epsilon") orelse 1e-6,
            .eos_token_id = f.getMetaU32("tokenizer.ggml.eos_token_id") orelse 1,
            .attn_scale = blk: {
                // Gemma uses query_pre_attn_scalar (config) or head_dim as the scaling denominator
                const scalar = f.getMetaU32("query_pre_attn_scalar") orelse head_dim;
                break :blk 1.0 / @sqrt(@as(f32, @floatFromInt(scalar)));
            },
            .embd_scale = @sqrt(@as(f32, @floatFromInt(n_embd))),
            .max_seq_len = max_sl,
            .final_logit_softcap = 0.0,
            .mlx_bits = f.getMetaU32("bits") orelse 6,
            // GGUF bakes +1.0 into norm weights (norm_add_one=false).
            // SafeTensors/MLX stores raw weights needing +1.0 (norm_add_one=true).
            // Detect SafeTensors by presence of HF config key "model_type".
            .norm_add_one = f.getMetaStr("model_type") != null,
            .fmt = f,
            .be = be,
            .allocator = allocator,
            .hidden = hidden,
            .hidden2 = hidden2,
            .q_buf = q_buf,
            .k_buf = k_buf,
            .v_buf = v_buf,
            .attn_out = attn_out,
            .ff_gate = ff_gate,
            .ff_up = ff_up,
            .logits = logits,
            .scores = scores,
            .kv_keys = kv_keys,
            .kv_values = kv_values,
        };
    }

    /// Release all heap allocations owned by this model.
    pub fn deinit(self: *Gemma3Model) void {
        const bufs = .{
            &self.hidden, &self.hidden2,  &self.q_buf,   &self.k_buf,
            &self.v_buf,  &self.attn_out, &self.ff_gate, &self.ff_up,
            &self.logits, &self.scores,
        };
        inline for (bufs) |buf| self.allocator.free(buf.*);
        for (self.norm_cache[0..self.norm_cache_len]) |entry| self.allocator.free(entry.data);
        for (self.kv_keys, self.kv_values) |k, v| {
            self.be.freeKvSlice(self.allocator, k);
            self.be.freeKvSlice(self.allocator, v);
        }
        self.allocator.free(self.kv_keys);
        self.allocator.free(self.kv_values);
    }

    /// Wrap this model in the generic `Model` interface.
    pub fn model(self: *Gemma3Model) Model {
        return Model.from(Gemma3Model, self);
    }

    // ── Forward pass ──────────────────────────────────────────────

    /// Run one decode step, returning the argmax next-token ID.
    pub fn forward(self: *Gemma3Model, token_id: u32) !u32 {
        if (self.kv_seq_len >= self.max_seq_len) return error.KVCacheFull;
        // Embedding lookup + Gemma scaling
        var t = self.perf.start();
        self.lookupEmbd(token_id);
        self.perf.end(.emb_lookup, t);

        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.acquire)) return error.Cancelled;
            try self.attention(@intCast(li));
            try self.feedForward(@intCast(li));
        }

        // Final norm → logits → argmax
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

    /// Reset the KV cache position for a new conversation.
    pub fn resetCache(self: *Gemma3Model) void {
        model_mod.resetInferenceState(&self.kv_seq_len, &self.cancelled);
    }

    /// Signal an in-progress forward pass to abort. Thread-safe.
    pub fn cancel(self: *Gemma3Model) void {
        model_mod.signalCancel(&self.cancelled);
    }

    // ── Layer implementations ─────────────────────────────────────

    fn attention(self: *Gemma3Model, li: u32) !void {
        const e: usize = self.n_embd;
        const nh: usize = self.n_head;
        const nkv: usize = self.n_head_kv;
        const hd: usize = self.head_dim;
        // Pre-norm
        var t = self.perf.start();
        const norm_w = self.fmt.layerTensor(li, "attn_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(norm_w, e), self.hidden2.ptr, self.hidden.len, self.rms_eps);
        self.perf.end(.rms_norm, t);

        // QKV projections
        t = self.perf.start();
        const qw = self.fmt.layerTensor(li, "attn_q.weight") orelse return error.MissingTensor;
        const kw = self.fmt.layerTensor(li, "attn_k.weight") orelse return error.MissingTensor;
        const vw = self.fmt.layerTensor(li, "attn_v.weight") orelse return error.MissingTensor;
        self.doGemv(self.hidden2.ptr, qw, self.q_buf.ptr, nh * hd, e);
        self.doGemv(self.hidden2.ptr, kw, self.k_buf.ptr, nkv * hd, e);
        self.doGemv(self.hidden2.ptr, vw, self.v_buf.ptr, nkv * hd, e);
        self.perf.end(.gemv_qkv, t);

        // Per-head QK norms (single dispatch per Q/K, not per-head)
        t = self.perf.start();
        const qn = self.fmt.layerTensor(li, "attn_q_norm.weight") orelse return error.MissingTensor;
        const qn_w = self.normAsF32(qn, hd);
        self.be.rmsNormMulti(self.q_buf.ptr, qn_w, nh, hd, self.rms_eps);
        const kn = self.fmt.layerTensor(li, "attn_k_norm.weight") orelse return error.MissingTensor;
        const kn_w = self.normAsF32(kn, hd);
        self.be.rmsNormMulti(self.k_buf.ptr, kn_w, nkv, hd, self.rms_eps);
        self.perf.end(.rms_norm, t);

        // RoPE
        t = self.perf.start();
        const theta = if (self.sliding_window_pattern > 0 and (li + 1) % self.sliding_window_pattern != 0)
            self.rope_local_theta
        else
            self.rope_theta;
        self.be.rope(self.q_buf.ptr, self.kv_seq_len, nh, hd, hd, theta);
        self.be.rope(self.k_buf.ptr, self.kv_seq_len, nkv, hd, hd, theta);
        self.perf.end(.rope, t);

        // KV cache append + scaled dot-product attention
        t = self.perf.start();
        attn_ops.scaledDotProductAttention(
            self.q_buf.ptr,
            self.kv_keys[li],
            self.kv_values[li],
            self.k_buf,
            self.v_buf,
            self.attn_out.ptr,
            self.scores.ptr,
            nh,
            nkv,
            hd,
            self.kv_seq_len,
            self.attn_scale,
            self.be,
            null,
            0,
        );
        self.perf.end(.sdpa, t);

        // Output projection + post-norm + residual
        t = self.perf.start();
        const ow = self.fmt.layerTensor(li, "attn_output.weight") orelse return error.MissingTensor;
        self.doGemv(self.attn_out.ptr, ow, self.hidden2.ptr, e, nh * hd);
        self.perf.end(.gemv_out, t);

        t = self.perf.start();
        const post_norm = self.fmt.layerTensor(li, "post_attention_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden2.ptr, self.normAsF32(post_norm, e), self.hidden2.ptr, self.hidden2.len, self.rms_eps);
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
        self.perf.end(.add, t);
    }

    fn feedForward(self: *Gemma3Model, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.n_ff;

        var t = self.perf.start();
        const norm_w = self.fmt.layerTensor(li, "ffn_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(norm_w, e), self.hidden2.ptr, self.hidden.len, self.rms_eps);
        self.perf.end(.rms_norm, t);

        t = self.perf.start();
        const gw = self.fmt.layerTensor(li, "ffn_gate.weight") orelse return error.MissingTensor;
        const uw = self.fmt.layerTensor(li, "ffn_up.weight") orelse return error.MissingTensor;
        const dw = self.fmt.layerTensor(li, "ffn_down.weight") orelse return error.MissingTensor;
        self.doGemv(self.hidden2.ptr, gw, self.ff_gate.ptr, ff, e);
        self.doGemv(self.hidden2.ptr, uw, self.ff_up.ptr, ff, e);
        self.perf.end(.gemv_ffn, t);

        t = self.perf.start();
        self.be.gelu(self.ff_gate.ptr, self.ff_gate.ptr, ff);
        self.be.mul(self.ff_gate.ptr, self.ff_up.ptr, self.ff_gate.ptr, ff);
        self.perf.end(.silu_mul, t);

        t = self.perf.start();
        self.doGemv(self.ff_gate.ptr, dw, self.hidden2.ptr, e, ff);
        self.perf.end(.gemv_ffn, t);

        t = self.perf.start();
        const post_norm = self.fmt.layerTensor(li, "post_ffw_norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden2.ptr, self.normAsF32(post_norm, e), self.hidden2.ptr, self.hidden2.len, self.rms_eps);
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
        self.perf.end(.add, t);
    }

    // ── Helpers ───────────────────────────────────────────────────

    /// Get norm weights as f32 pointer. Caches converted weights on first access
    /// so subsequent tokens return a stable pointer with zero work and no GPU syncs.
    fn normAsF32(self: *Gemma3Model, t: TensorInfo, n: usize) [*]const f32 {
        const needs_convert = t.dtype == .bf16 or self.norm_add_one;
        if (!needs_convert) return @ptrCast(@alignCast(t.data_ptr));

        // Check cache (linear scan — at most ~170 entries, first-token only on miss)
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

    /// Cached companion pointers for MLX quantized weight tensors.
    /// Avoids per-GEMV string formatting + HashMap lookups for .scales/.biases.
    const MlxCompanion = struct { scales: [*]const u8, biases: [*]const u8 };
    const mlx_companion_cache_size: usize = 256;

    /// GEMV dispatch that handles both regular and MLX-quantized weights.
    /// For MLX quantized tensors (dtype == .mlx_q), looks up companion
    /// .scales and .biases tensors via a pointer-keyed cache (first call
    /// per tensor does the HashMap lookup; subsequent calls are O(1)).
    fn doGemv(self: *Gemma3Model, x: [*]const f32, t: TensorInfo, y: [*]f32, n: usize, k: usize) void {
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
            var sbuf: [256]u8 = undefined;
            var bbuf: [256]u8 = undefined;
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

    fn lookupEmbd(self: *Gemma3Model, tok: u32) void {
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
        const V8 = @Vector(8, f32);
        const scale_v: V8 = @splat(self.embd_scale);
        const n: usize = self.n_embd;
        var i: usize = 0;
        while (i + 8 <= n) : (i += 8) {
            self.hidden[i..][0..8].* = @as(V8, self.hidden[i..][0..8].*) * scale_v;
        }
        while (i < n) : (i += 1) self.hidden[i] *= self.embd_scale;
    }

    fn applySoftcap(self: *Gemma3Model) void {
        const inv = 1.0 / self.final_logit_softcap;
        const cap = self.final_logit_softcap;
        for (self.logits) |*v| v.* = @as(f32, math.tanh(v.* * inv)) * cap;
    }
};

