//! Composable Megakernel Generator
//!
//! Generates model-specific megakernel MSL source at runtime from model metadata.
//! Instead of hand-written per-model kernel files, this module composes the kernel
//! from reusable building blocks based on the model's layer structure.
//!
//! Architecture:
//!   Model metadata (GGUF) → ModelDesc → composeMSL() → MSL source string
//!   Metal backend compiles the source at init via newLibraryWithSource.
//!
//! The generated kernel processes ALL layers in a single GPU dispatch using
//! atomic grid sync between stages. Building blocks from mega_common.metal
//! are referenced by the generated code (they're concatenated before it).

const std = @import("std");

/// Layer type for megakernel composition.
pub const LayerKind = enum {
    /// Standard GQA attention (Q/K/V projection, RoPE, SDPA, output projection).
    attention,
    /// DeltaNet SSM — breaks out of megakernel (sequential recurrence).
    deltanet,
    /// MoE expert routing — breaks out (CPU-side routing).
    moe,
    /// FFN-only (no attention, single up projection + activation + down).
    ffn_only,
};

/// Activation function for the FFN.
pub const Activation = enum {
    silu,
    gelu,
    relu_squared,
};

/// Quantization format for weight GEMV dispatch.
pub const QuantKind = enum {
    q8_0,
    q4_k,
    q5_k,
    q6_k,
    q4_0,
};

/// Maximum supported layers for static arrays.
const max_layers: usize = 64;

/// Model descriptor — everything needed to compose a megakernel.
/// Populated from GGUF/SafeTensors metadata at model init time.
pub const ModelDesc = struct {
    /// Model architecture name (for kernel function naming).
    name: []const u8,
    /// Number of transformer layers.
    n_layers: u32,
    /// Embedding dimension.
    n_embd: u32,
    /// FFN intermediate dimension (dense layers). Per-layer override via layer_n_ff.
    n_ff: u32,
    /// Number of query heads (default). Per-layer override via layer_n_head.
    n_head: u32,
    /// Number of KV heads (default). Per-layer override via layer_n_kv.
    n_kv: u32,
    /// Per-head dimension (default). Per-layer override via layer_head_dim.
    head_dim: u32,
    /// RoPE dimension (may be < head_dim for partial rotation).
    rope_dim: u32,
    /// RoPE base frequency (default). Per-layer override via layer_rope_theta.
    rope_theta: f32,
    /// RMS norm epsilon.
    rms_eps: f32,
    /// Maximum sequence length.
    max_seq_len: u32,
    /// Sliding window size (0 = full attention). Per-layer window via layer_sliding_window.
    sliding_window: u32 = 0,
    /// FFN activation function.
    activation: Activation,
    /// Weight quantization format.
    quant: QuantKind,
    /// Per-layer type (attention, deltanet, moe, ffn_only).
    layer_types: [max_layers]LayerKind,

    // ── Per-layer overrides (0 = use default) ────────────────
    layer_n_head: [max_layers]u32 = [_]u32{0} ** max_layers,
    layer_n_kv: [max_layers]u32 = [_]u32{0} ** max_layers,
    layer_head_dim: [max_layers]u32 = [_]u32{0} ** max_layers,
    layer_n_ff: [max_layers]u32 = [_]u32{0} ** max_layers,
    layer_rope_theta: [max_layers]f32 = [_]f32{0} ** max_layers,
    layer_sliding_window: [max_layers]u32 = [_]u32{0} ** max_layers,

    // ── Model-specific flags ──────────────────────────────────
    /// Q projection includes interleaved gate (Qwen 3.5 only).
    has_gate: bool = false,
    /// Per-head Q/K RMS norms after projection.
    has_qk_norm: bool = false,
    /// Post-attention norm (fused addRmsNorm with FFN).
    has_post_attn_norm: bool = false,
    /// Fuse residual add into pre-norm (deferred residual pattern).
    fuse_residual: bool = false,
    /// Gemma-style embedding scaling (hidden *= sqrt(n_embd)).
    embd_scale: bool = false,
    /// Logit softcap value (0 = disabled). Gemma uses 30.0.
    logit_softcap: f32 = 0,

    /// Get effective n_head for layer (uses per-layer override or default).
    pub fn layerNHead(self: ModelDesc, li: usize) u32 {
        return if (self.layer_n_head[li] != 0) self.layer_n_head[li] else self.n_head;
    }
    /// Get effective n_kv for layer.
    pub fn layerNKv(self: ModelDesc, li: usize) u32 {
        return if (self.layer_n_kv[li] != 0) self.layer_n_kv[li] else self.n_kv;
    }
    /// Get effective head_dim for layer.
    pub fn layerHeadDim(self: ModelDesc, li: usize) u32 {
        return if (self.layer_head_dim[li] != 0) self.layer_head_dim[li] else self.head_dim;
    }
    /// Get effective n_ff for layer.
    pub fn layerNFf(self: ModelDesc, li: usize) u32 {
        return if (self.layer_n_ff[li] != 0) self.layer_n_ff[li] else self.n_ff;
    }
    /// Get effective rope_theta for layer.
    pub fn layerRopeTheta(self: ModelDesc, li: usize) f32 {
        return if (self.layer_rope_theta[li] != 0) self.layer_rope_theta[li] else self.rope_theta;
    }
    /// Get sliding window for layer (0 = full attention).
    pub fn layerWindow(self: ModelDesc, li: usize) u32 {
        return if (self.layer_sliding_window[li] != 0) self.layer_sliding_window[li] else self.sliding_window;
    }
    /// Returns true if any layer has per-layer overrides (non-uniform model).
    pub fn hasPerLayerVariation(self: ModelDesc) bool {
        for (0..self.n_layers) |i| {
            if (self.layer_n_head[i] != 0 or self.layer_head_dim[i] != 0 or
                self.layer_rope_theta[i] != 0 or self.layer_n_ff[i] != 0) return true;
        }
        return false;
    }

    /// Create a uniform layer pattern (all layers same type).
    pub fn uniform(n_layers: u32, kind: LayerKind) [max_layers]LayerKind {
        var types: [max_layers]LayerKind = undefined;
        for (0..max_layers) |i| {
            types[i] = if (i < n_layers) kind else .attention;
        }
        return types;
    }

    /// Create Qwen 3.5 hybrid pattern: DeltaNet except every Nth is attention.
    pub fn qwenHybrid(n_layers: u32, full_attn_interval: u32) [max_layers]LayerKind {
        var types: [max_layers]LayerKind = undefined;
        for (0..max_layers) |i| {
            if (i >= n_layers) {
                types[i] = .attention;
            } else if (full_attn_interval > 0 and ((i + 1) % full_attn_interval) == 0) {
                types[i] = .attention;
            } else {
                types[i] = .deltanet;
            }
        }
        return types;
    }
};

// ── MSL Template Fragments ───────────────────────────────────────────────
// These are composed into the final MSL source. Each fragment is a complete
// stage that calls the building blocks from mega_common.metal.

const msl_header =
    \\// Auto-composed megakernel — generated by mega_compose.zig
    \\// Single GPU dispatch for all layers. DO NOT EDIT.
    \\
    \\
;

/// Append `s` into `b` at `*p`, truncating silently if the buffer is full.
fn appendSlice(b: []u8, p: *usize, s: []const u8) void {
    if (p.* + s.len > b.len) return;
    @memcpy(b[p.*..][0..s.len], s);
    p.* += s.len;
}

/// Emit `MegaAutoParams` and optional per-layer constant arrays.
/// Must run before the kernel body so `composeMSL` output is compilable when
/// concatenated with the Metal `msl_source` blob (which already defines
/// `MegaLayerOffsets` via the hand-written mega_*.metal files).
fn emitParamsStruct(buf: []u8, pos: *usize, desc: ModelDesc) void {
    appendSlice(buf, pos,
        \\struct MegaAutoParams {
        \\    uint n_layers;
        \\    uint n_embd;
        \\    uint n_head;
        \\    uint n_kv;
        \\    uint head_dim;
        \\    uint n_ff;
        \\    uint rope_dim;
        \\    float rope_theta;
        \\    float rms_eps;
        \\    float embd_scale;
        \\    float logit_softcap;
        \\    uint sliding_window;
        \\    uint max_seq_len;
        \\    uint seq_pos;
        \\    uint n_tgs;
        \\};
        \\
        \\
    );

    if (!desc.hasPerLayerVariation()) return;

    appendSlice(buf, pos, "// Per-layer dimension overrides (baked at compose time)\n");

    const EmitU32 = struct {
        fn array(b: []u8, p: *usize, name: []const u8, desc_inner: ModelDesc, get: *const fn (ModelDesc, usize) u32) void {
            appendSlice(b, p, "constant uint ");
            appendSlice(b, p, name);
            appendSlice(b, p, "[] = {");
            var i: usize = 0;
            while (i < desc_inner.n_layers) : (i += 1) {
                if (i > 0) appendSlice(b, p, ",");
                var num_buf: [16]u8 = undefined;
                const s = std.fmt.bufPrint(&num_buf, "{d}", .{get(desc_inner, i)}) catch return;
                appendSlice(b, p, s);
            }
            appendSlice(b, p, "};\n");
        }
    };
    const EmitF32 = struct {
        fn array(b: []u8, p: *usize, name: []const u8, desc_inner: ModelDesc, get: *const fn (ModelDesc, usize) f32) void {
            appendSlice(b, p, "constant float ");
            appendSlice(b, p, name);
            appendSlice(b, p, "[] = {");
            var i: usize = 0;
            while (i < desc_inner.n_layers) : (i += 1) {
                if (i > 0) appendSlice(b, p, ",");
                var num_buf: [32]u8 = undefined;
                const s = std.fmt.bufPrint(&num_buf, "{d:.1}", .{get(desc_inner, i)}) catch return;
                appendSlice(b, p, s);
            }
            appendSlice(b, p, "};\n");
        }
    };

    EmitU32.array(buf, pos, "layer_n_head", desc, ModelDesc.layerNHead);
    EmitU32.array(buf, pos, "layer_n_kv", desc, ModelDesc.layerNKv);
    EmitU32.array(buf, pos, "layer_head_dim", desc, ModelDesc.layerHeadDim);
    EmitU32.array(buf, pos, "layer_n_ff", desc, ModelDesc.layerNFf);
    EmitF32.array(buf, pos, "layer_rope_theta", desc, ModelDesc.layerRopeTheta);
    EmitU32.array(buf, pos, "layer_window", desc, ModelDesc.layerWindow);
    appendSlice(buf, pos, "\n");
}

/// GEMV function name for the given quant kind.
fn gemvFn(quant: QuantKind) []const u8 {
    return switch (quant) {
        .q8_0 => "mega_gemv_q8",
        .q4_k => "mega_gemv_q4k",
        .q5_k => "mega_gemv_q5k",
        .q6_k => "mega_gemv_q6k",
        .q4_0 => "mega_gemv_q4_0",
    };
}

/// Weight cast expression for typed vs raw pointers.
fn weightCast(quant: QuantKind) []const u8 {
    return switch (quant) {
        .q8_0 => "(device const block_q8_0*)(weights + lo.{s})",
        .q4_k, .q5_k, .q6_k => "weights + lo.{s}",
        .q4_0 => "(device const block_q4_0*)(weights + lo.{s})",
    };
}

/// Activation function call name.
fn activationFn(activation: Activation) []const u8 {
    return switch (activation) {
        .silu => "mega_silu_mul",
        .gelu => "mega_gelu_mul",
        .relu_squared => "mega_relu_squared",
    };
}

/// Compose a complete megakernel MSL source from a model descriptor.
/// The generated kernel is a single entry point that processes all layers.
/// Returns the generated MSL as a slice of the provided buffer.
pub fn composeMSL(buf: []u8, desc: ModelDesc) []const u8 {
    var pos: usize = 0;

    // Header + params/offset structs (required for Metal compile)
    appendSlice(buf, &pos, msl_header);
    emitParamsStruct(buf, &pos, desc);

    // Kernel signature
    const sig =
        \\kernel void megakernel_auto(
        \\    device const uchar*             weights     [[buffer(0)]],
        \\    device const MegaLayerOffsets*   layer_off   [[buffer(1)]],
        \\    device float*                    kv_keys     [[buffer(2)]],
        \\    device float*                    kv_values   [[buffer(3)]],
        \\    device float*                    hidden      [[buffer(4)]],
        \\    device float*                    scratch     [[buffer(5)]],
        \\    device atomic_uint*              sync_ctrs   [[buffer(6)]],
        \\    constant MegaAutoParams&         p           [[buffer(7)]],
        \\    uint tgid    [[threadgroup_position_in_grid]],
        \\    uint tid     [[thread_index_in_threadgroup]],
        \\    uint tg_size [[threads_per_threadgroup]])
        \\{
        \\    threadgroup float shared[8];
        \\    device float* hidden2    = scratch;
        \\    device float* ff_gate    = scratch + p.n_embd;
        \\    device float* ff_up      = scratch + p.n_embd + p.n_ff;
        \\    device float* qkv_buf    = scratch + p.n_embd + 2 * p.n_ff;
        \\    device float* ss_scratch = scratch + p.n_embd + 2 * p.n_ff +
        \\                               (p.n_head + 2 * p.n_kv) * p.head_dim;
        \\    uint sync_idx = 0;
        \\
        \\
    ;
    appendSlice(buf, &pos, sig);

    // Layer loop
    appendSlice(buf, &pos, "    for (uint li = 0; li < p.n_layers; li++) {\n");
    appendSlice(buf, &pos, "        device const MegaLayerOffsets& lo = layer_off[li];\n");

    // Per-layer dimension variables (from baked constant arrays or uniform params)
    if (desc.hasPerLayerVariation()) {
        appendSlice(buf, &pos,
            \\        uint cur_n_head = layer_n_head[li];
            \\        uint cur_n_kv = layer_n_kv[li];
            \\        uint cur_head_dim = layer_head_dim[li];
            \\        uint cur_n_ff = layer_n_ff[li];
            \\        float cur_rope_theta = layer_rope_theta[li];
            \\        uint cur_window = layer_window[li];
            \\
            \\
        );
    } else {
        appendSlice(buf, &pos,
            \\        uint cur_n_head = p.n_head;
            \\        uint cur_n_kv = p.n_kv;
            \\        uint cur_head_dim = p.head_dim;
            \\        uint cur_n_ff = p.n_ff;
            \\        float cur_rope_theta = p.rope_theta;
            \\        uint cur_window = p.sliding_window;
            \\
            \\
        );
    }

    // Pre-attention norm
    if (desc.fuse_residual) {
        appendSlice(buf, &pos,
            \\        device const float* norm_w = (device const float*)(weights + lo.attn_norm);
            \\        if (li > 0) {
            \\            mega_add_rms_norm(hidden, hidden2, norm_w, hidden2,
            \\                ss_scratch, &sync_ctrs[sync_idx++ % 32],
            \\                p.n_embd, p.n_tgs, p.rms_eps, shared, tgid, tid, tg_size);
            \\        } else {
            \\            mega_rms_norm(hidden, norm_w, hidden2,
            \\                ss_scratch, &sync_ctrs[sync_idx++ % 32],
            \\                p.n_embd, p.n_tgs, p.rms_eps, shared, tgid, tid, tg_size);
            \\        }
            \\
            \\
        );
    } else {
        appendSlice(buf, &pos,
            \\        device const float* norm_w = (device const float*)(weights + lo.attn_norm);
            \\        mega_rms_norm(hidden, norm_w, hidden2,
            \\            ss_scratch, &sync_ctrs[sync_idx++ % 32],
            \\            p.n_embd, p.n_tgs, p.rms_eps, shared, tgid, tid, tg_size);
            \\
            \\
        );
    }

    // Emit per-layer attention block for layers that have it
    // Check if any layers are attention
    var has_attn = false;
    for (0..desc.n_layers) |i| {
        if (desc.layer_types[i] == .attention) {
            has_attn = true;
            break;
        }
    }

    if (has_attn) {
        // Attention: Q/K/V GEMV → RoPE → KV append → SDPA → output

        // Use a runtime layer-type check since we can't comptime-unroll in MSL
        // The layer_types are encoded as a bitfield in the params or checked via
        // the weight offsets (attention layers have attn_q != 0)
        appendSlice(buf, &pos,
            \\        // Attention: skip if no attention weights for this layer
            \\        if (lo.attn_q != 0) {
            \\            uint qd = cur_n_head * cur_head_dim;
            \\            uint kvd = cur_n_kv * cur_head_dim;
            \\            device float* q_buf = qkv_buf;
            \\            device float* k_buf = qkv_buf + qd;
            \\            device float* v_buf = qkv_buf + qd + kvd;
            \\
            \\
        );

        // Q/K/V projections — emit with correct GEMV function
        const q_gemv = switch (desc.quant) {
            .q8_0 =>
            \\            mega_gemv_q8(hidden2, (device const block_q8_0*)(weights + lo.attn_q),
            \\                q_buf, qd, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q4_k =>
            \\            mega_gemv_q4k(hidden2, weights + lo.attn_q,
            \\                q_buf, qd, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q5_k =>
            \\            mega_gemv_q5k(hidden2, weights + lo.attn_q,
            \\                q_buf, qd, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q6_k =>
            \\            mega_gemv_q6k(hidden2, weights + lo.attn_q,
            \\                q_buf, qd, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q4_0 =>
            \\            mega_gemv_q4_0(hidden2, (device const block_q4_0*)(weights + lo.attn_q),
            \\                q_buf, qd, p.n_embd, shared, tgid, tid, tg_size);
            ,
        };
        appendSlice(buf, &pos, q_gemv);
        appendSlice(buf, &pos, "\n            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);\n            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);\n\n");

        // K projection (same quant)
        const k_gemv = switch (desc.quant) {
            .q8_0 =>
            \\            mega_gemv_q8(hidden2, (device const block_q8_0*)(weights + lo.attn_k),
            \\                k_buf, kvd, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q4_k =>
            \\            mega_gemv_q4k(hidden2, weights + lo.attn_k,
            \\                k_buf, kvd, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q5_k =>
            \\            mega_gemv_q5k(hidden2, weights + lo.attn_k,
            \\                k_buf, kvd, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q6_k =>
            \\            mega_gemv_q6k(hidden2, weights + lo.attn_k,
            \\                k_buf, kvd, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q4_0 =>
            \\            mega_gemv_q4_0(hidden2, (device const block_q4_0*)(weights + lo.attn_k),
            \\                k_buf, kvd, p.n_embd, shared, tgid, tid, tg_size);
            ,
        };
        appendSlice(buf, &pos, k_gemv);
        appendSlice(buf, &pos, "\n            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);\n            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);\n\n");

        // V projection
        const v_gemv = switch (desc.quant) {
            .q8_0 =>
            \\            mega_gemv_q8(hidden2, (device const block_q8_0*)(weights + lo.attn_v),
            \\                v_buf, kvd, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q4_k =>
            \\            mega_gemv_q4k(hidden2, weights + lo.attn_v,
            \\                v_buf, kvd, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q5_k =>
            \\            mega_gemv_q5k(hidden2, weights + lo.attn_v,
            \\                v_buf, kvd, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q6_k =>
            \\            mega_gemv_q6k(hidden2, weights + lo.attn_v,
            \\                v_buf, kvd, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q4_0 =>
            \\            mega_gemv_q4_0(hidden2, (device const block_q4_0*)(weights + lo.attn_v),
            \\                v_buf, kvd, p.n_embd, shared, tgid, tid, tg_size);
            ,
        };
        appendSlice(buf, &pos, v_gemv);
        appendSlice(buf, &pos, "\n            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);\n            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);\n\n");

        // RoPE
        appendSlice(buf, &pos,
            \\            mega_rope(q_buf, cur_n_head, cur_head_dim, p.rope_dim, cur_rope_theta, p.seq_pos,
            \\                tgid, tid, tg_size);
            \\            mega_rope(k_buf, cur_n_kv, cur_head_dim, p.rope_dim, cur_rope_theta, p.seq_pos,
            \\                tgid, tid, tg_size);
            \\            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            \\            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);
            \\
            \\
        );

        // KV cache append + inline SDPA
        appendSlice(buf, &pos,
            \\            uint kv_layer_stride = p.max_seq_len * cur_n_kv * cur_head_dim;
            \\            device float* layer_keys = kv_keys + li * kv_layer_stride;
            \\            device float* layer_values = kv_values + li * kv_layer_stride;
            \\            mega_kv_append_f32(k_buf, v_buf, layer_keys, layer_values,
            \\                cur_n_kv * cur_head_dim, p.seq_pos, tgid, tid, tg_size);
            \\            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            \\            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);
            \\
            \\            device float* attn_out_buf = qkv_buf;
            \\            mega_sdpa_inline(q_buf, (device const uchar*)layer_keys,
            \\                (device const uchar*)layer_values, attn_out_buf,
            \\                cur_n_head, cur_n_kv, cur_head_dim, p.seq_pos + 1,
            \\                1.0f / sqrt(float(cur_head_dim)),
            \\                0, 0, 0, 0, shared, tgid, tid, tg_size);
            \\            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            \\            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);
            \\
            \\
        );

        // Output projection
        const out_gemv = switch (desc.quant) {
            .q8_0 =>
            \\            mega_gemv_q8(attn_out_buf, (device const block_q8_0*)(weights + lo.attn_output),
            \\                hidden2, p.n_embd, cur_n_head * cur_head_dim, shared, tgid, tid, tg_size);
            ,
            .q4_k =>
            \\            mega_gemv_q4k(attn_out_buf, weights + lo.attn_output,
            \\                hidden2, p.n_embd, cur_n_head * cur_head_dim, shared, tgid, tid, tg_size);
            ,
            .q5_k =>
            \\            mega_gemv_q5k(attn_out_buf, weights + lo.attn_output,
            \\                hidden2, p.n_embd, cur_n_head * cur_head_dim, shared, tgid, tid, tg_size);
            ,
            .q6_k =>
            \\            mega_gemv_q6k(attn_out_buf, weights + lo.attn_output,
            \\                hidden2, p.n_embd, cur_n_head * cur_head_dim, shared, tgid, tid, tg_size);
            ,
            .q4_0 =>
            \\            mega_gemv_q4_0(attn_out_buf, (device const block_q4_0*)(weights + lo.attn_output),
            \\                hidden2, p.n_embd, cur_n_head * cur_head_dim, shared, tgid, tid, tg_size);
            ,
        };
        appendSlice(buf, &pos, out_gemv);
        appendSlice(buf, &pos, "\n            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);\n            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);\n");

        appendSlice(buf, &pos, "        }\n\n"); // close attention if
    }

    // Post-attention norm (if applicable)
    if (desc.has_post_attn_norm) {
        appendSlice(buf, &pos,
            \\        device const float* post_norm_w = (device const float*)(weights + lo.post_attn_norm);
            \\        mega_add_rms_norm(hidden, hidden2, post_norm_w, hidden2,
            \\            ss_scratch, &sync_ctrs[sync_idx++ % 32],
            \\            p.n_embd, p.n_tgs, p.rms_eps, shared, tgid, tid, tg_size);
            \\
            \\
        );
    } else {
        appendSlice(buf, &pos,
            \\        mega_add(hidden, hidden2, p.n_embd, tgid, tid, tg_size);
            \\        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            \\        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);
            \\
            \\
        );
    }

    // FFN: gate + up + activation + down
    if (desc.activation != .relu_squared) {
        // SwiGLU/GeGLU: gate + up GEMV → activation*mul → down GEMV
        const gate_gemv = switch (desc.quant) {
            .q8_0 =>
            \\        mega_gemv_q8(hidden2, (device const block_q8_0*)(weights + lo.ffn_gate),
            \\            ff_gate, cur_n_ff, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q4_k =>
            \\        mega_gemv_q4k(hidden2, weights + lo.ffn_gate,
            \\            ff_gate, cur_n_ff, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q5_k =>
            \\        mega_gemv_q5k(hidden2, weights + lo.ffn_gate,
            \\            ff_gate, cur_n_ff, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q6_k =>
            \\        mega_gemv_q6k(hidden2, weights + lo.ffn_gate,
            \\            ff_gate, cur_n_ff, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q4_0 =>
            \\        mega_gemv_q4_0(hidden2, (device const block_q4_0*)(weights + lo.ffn_gate),
            \\            ff_gate, cur_n_ff, p.n_embd, shared, tgid, tid, tg_size);
            ,
        };
        appendSlice(buf, &pos, gate_gemv);
        appendSlice(buf, &pos, "\n        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);\n        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);\n\n");

        // Up GEMV (same quant)
        const up_gemv = switch (desc.quant) {
            .q8_0 =>
            \\        mega_gemv_q8(hidden2, (device const block_q8_0*)(weights + lo.ffn_up),
            \\            ff_up, cur_n_ff, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q4_k =>
            \\        mega_gemv_q4k(hidden2, weights + lo.ffn_up,
            \\            ff_up, cur_n_ff, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q5_k =>
            \\        mega_gemv_q5k(hidden2, weights + lo.ffn_up,
            \\            ff_up, cur_n_ff, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q6_k =>
            \\        mega_gemv_q6k(hidden2, weights + lo.ffn_up,
            \\            ff_up, cur_n_ff, p.n_embd, shared, tgid, tid, tg_size);
            ,
            .q4_0 =>
            \\        mega_gemv_q4_0(hidden2, (device const block_q4_0*)(weights + lo.ffn_up),
            \\            ff_up, cur_n_ff, p.n_embd, shared, tgid, tid, tg_size);
            ,
        };
        appendSlice(buf, &pos, up_gemv);
        appendSlice(buf, &pos, "\n        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);\n        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);\n\n");

        // Activation
        const act_call = switch (desc.activation) {
            .silu => "        mega_silu_mul(ff_gate, ff_up, cur_n_ff, tgid, tid, tg_size);\n",
            .gelu => "        mega_gelu_mul(ff_gate, ff_up, cur_n_ff, tgid, tid, tg_size);\n",
            .relu_squared => unreachable,
        };
        appendSlice(buf, &pos, act_call);
        appendSlice(buf, &pos, "        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);\n        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);\n\n");
    } else {
        // ReLU²: single up GEMV → relu² → down
        const up_gemv = switch (desc.quant) {
            .q8_0 =>
            \\        mega_gemv_q8(hidden2, (device const block_q8_0*)(weights + lo.ffn_up),
            \\            ff_gate, cur_n_ff, p.n_embd, shared, tgid, tid, tg_size);
            ,
            else =>
            \\        mega_gemv_q4k(hidden2, weights + lo.ffn_up,
            \\            ff_gate, cur_n_ff, p.n_embd, shared, tgid, tid, tg_size);
            ,
        };
        appendSlice(buf, &pos, up_gemv);
        appendSlice(buf, &pos, "\n        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);\n        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);\n");
        appendSlice(buf, &pos, "        mega_relu_squared(ff_gate, cur_n_ff, tgid, tid, tg_size);\n");
        appendSlice(buf, &pos, "        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);\n        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);\n\n");
    }

    // Down projection
    const down_gemv = switch (desc.quant) {
        .q8_0 =>
        \\        mega_gemv_q8(ff_gate, (device const block_q8_0*)(weights + lo.ffn_down),
        \\            hidden2, p.n_embd, cur_n_ff, shared, tgid, tid, tg_size);
        ,
        .q4_k =>
        \\        mega_gemv_q4k(ff_gate, weights + lo.ffn_down,
        \\            hidden2, p.n_embd, cur_n_ff, shared, tgid, tid, tg_size);
        ,
        .q5_k =>
        \\        mega_gemv_q5k(ff_gate, weights + lo.ffn_down,
        \\            hidden2, p.n_embd, cur_n_ff, shared, tgid, tid, tg_size);
        ,
        .q6_k =>
        \\        mega_gemv_q6k(ff_gate, weights + lo.ffn_down,
        \\            hidden2, p.n_embd, cur_n_ff, shared, tgid, tid, tg_size);
        ,
        .q4_0 =>
        \\        mega_gemv_q4_0(ff_gate, (device const block_q4_0*)(weights + lo.ffn_down),
        \\            hidden2, p.n_embd, cur_n_ff, shared, tgid, tid, tg_size);
        ,
    };
    appendSlice(buf, &pos, down_gemv);
    appendSlice(buf, &pos, "\n        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);\n        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);\n");

    // End of layer loop
    if (!desc.fuse_residual) {
        appendSlice(buf, &pos, "\n        mega_add(hidden, hidden2, p.n_embd, tgid, tid, tg_size);\n");
        appendSlice(buf, &pos, "        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);\n        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);\n");
    }
    appendSlice(buf, &pos, "    }\n");

    // Final residual (for fused residual models)
    if (desc.fuse_residual) {
        appendSlice(buf, &pos, "    mega_add(hidden, hidden2, p.n_embd, tgid, tid, tg_size);\n");
    }

    // Close kernel
    appendSlice(buf, &pos, "}\n");

    return buf[0..pos];
}

// ── Tests ────────────────────────────────────────────────────────────────

test "ModelDesc.uniform creates all-same-kind layer array" {
    const types = ModelDesc.uniform(5, .attention);
    for (0..5) |i| try std.testing.expectEqual(LayerKind.attention, types[i]);
    // Beyond n_layers should default to .attention
    try std.testing.expectEqual(LayerKind.attention, types[5]);
    try std.testing.expectEqual(LayerKind.attention, types[63]);
}

test "ModelDesc.uniform with deltanet" {
    const types = ModelDesc.uniform(3, .deltanet);
    try std.testing.expectEqual(LayerKind.deltanet, types[0]);
    try std.testing.expectEqual(LayerKind.deltanet, types[1]);
    try std.testing.expectEqual(LayerKind.deltanet, types[2]);
    // Past n_layers
    try std.testing.expectEqual(LayerKind.attention, types[3]);
}

test "ModelDesc.qwenHybrid creates correct attention pattern" {
    // Qwen: every 4th layer is attention, rest deltanet
    const types = ModelDesc.qwenHybrid(12, 4);
    // Layers 3, 7, 11 (0-indexed: (i+1) % 4 == 0) should be attention
    try std.testing.expectEqual(LayerKind.deltanet, types[0]);
    try std.testing.expectEqual(LayerKind.deltanet, types[1]);
    try std.testing.expectEqual(LayerKind.deltanet, types[2]);
    try std.testing.expectEqual(LayerKind.attention, types[3]); // (3+1)%4==0
    try std.testing.expectEqual(LayerKind.deltanet, types[4]);
    try std.testing.expectEqual(LayerKind.deltanet, types[5]);
    try std.testing.expectEqual(LayerKind.deltanet, types[6]);
    try std.testing.expectEqual(LayerKind.attention, types[7]); // (7+1)%4==0
    try std.testing.expectEqual(LayerKind.attention, types[11]); // (11+1)%4==0
}

test "ModelDesc.qwenHybrid with zero interval is all deltanet" {
    const types = ModelDesc.qwenHybrid(8, 0);
    for (0..8) |i| try std.testing.expectEqual(LayerKind.deltanet, types[i]);
}

test "ModelDesc per-layer accessors use override or default" {
    var desc = ModelDesc{
        .name = "test",
        .n_layers = 4,
        .n_embd = 2048,
        .n_ff = 8192,
        .n_head = 16,
        .n_kv = 4,
        .head_dim = 128,
        .rope_dim = 128,
        .rope_theta = 10000.0,
        .rms_eps = 1e-6,
        .max_seq_len = 4096,
        .activation = .silu,
        .quant = .q4_k,
        .layer_types = ModelDesc.uniform(4, .attention),
    };

    // Defaults: all layers return the base values
    try std.testing.expectEqual(@as(u32, 16), desc.layerNHead(0));
    try std.testing.expectEqual(@as(u32, 4), desc.layerNKv(0));
    try std.testing.expectEqual(@as(u32, 128), desc.layerHeadDim(0));
    try std.testing.expectEqual(@as(u32, 8192), desc.layerNFf(0));
    try std.testing.expectApproxEqAbs(@as(f32, 10000.0), desc.layerRopeTheta(0), 0.1);
    try std.testing.expectEqual(@as(u32, 0), desc.layerWindow(0));

    // Override layer 2
    desc.layer_n_head[2] = 32;
    desc.layer_head_dim[2] = 64;
    desc.layer_n_ff[2] = 4096;
    desc.layer_rope_theta[2] = 500000.0;
    desc.layer_sliding_window[2] = 1024;

    try std.testing.expectEqual(@as(u32, 32), desc.layerNHead(2));
    try std.testing.expectEqual(@as(u32, 64), desc.layerHeadDim(2));
    try std.testing.expectEqual(@as(u32, 4096), desc.layerNFf(2));
    try std.testing.expectApproxEqAbs(@as(f32, 500000.0), desc.layerRopeTheta(2), 0.1);
    try std.testing.expectEqual(@as(u32, 1024), desc.layerWindow(2));

    // Other layers unchanged
    try std.testing.expectEqual(@as(u32, 16), desc.layerNHead(0));
    try std.testing.expectEqual(@as(u32, 128), desc.layerHeadDim(1));
}

test "ModelDesc.hasPerLayerVariation detects overrides" {
    var desc = ModelDesc{
        .name = "test",
        .n_layers = 4,
        .n_embd = 2048,
        .n_ff = 8192,
        .n_head = 16,
        .n_kv = 4,
        .head_dim = 128,
        .rope_dim = 128,
        .rope_theta = 10000.0,
        .rms_eps = 1e-6,
        .max_seq_len = 4096,
        .activation = .silu,
        .quant = .q4_k,
        .layer_types = ModelDesc.uniform(4, .attention),
    };

    // No overrides
    try std.testing.expect(!desc.hasPerLayerVariation());

    // Set one override
    desc.layer_n_head[1] = 32;
    try std.testing.expect(desc.hasPerLayerVariation());
}

test "gemvFn returns correct function name per quant" {
    try std.testing.expectEqualStrings("mega_gemv_q8", gemvFn(.q8_0));
    try std.testing.expectEqualStrings("mega_gemv_q4k", gemvFn(.q4_k));
    try std.testing.expectEqualStrings("mega_gemv_q5k", gemvFn(.q5_k));
    try std.testing.expectEqualStrings("mega_gemv_q6k", gemvFn(.q6_k));
    try std.testing.expectEqualStrings("mega_gemv_q4_0", gemvFn(.q4_0));
}

test "activationFn returns correct function name" {
    try std.testing.expectEqualStrings("mega_silu_mul", activationFn(.silu));
    try std.testing.expectEqualStrings("mega_gelu_mul", activationFn(.gelu));
    try std.testing.expectEqualStrings("mega_relu_squared", activationFn(.relu_squared));
}

test "composeMSL uniform model does not emit per-layer arrays" {
    var buf: [32768]u8 = undefined;
    const desc = ModelDesc{
        .name = "uniform_test",
        .n_layers = 4,
        .n_embd = 2048,
        .n_ff = 8192,
        .n_head = 16,
        .n_kv = 4,
        .head_dim = 128,
        .rope_dim = 128,
        .rope_theta = 10000.0,
        .rms_eps = 1e-6,
        .max_seq_len = 4096,
        .activation = .silu,
        .quant = .q4_k,
        .layer_types = ModelDesc.uniform(4, .attention),
    };
    const msl = composeMSL(&buf, desc);
    // Uniform model should use p.n_head, not per-layer arrays
    try std.testing.expect(std.mem.indexOf(u8, msl, "struct MegaAutoParams") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl, "cur_n_head = p.n_head") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl, "cur_n_head = layer_n_head[li]") == null);
    try std.testing.expect(std.mem.indexOf(u8, msl, "constant uint layer_n_head[]") == null);
}

test "composeMSL per-layer variation uses baked arrays" {
    var buf: [32768]u8 = undefined;
    var desc = ModelDesc{
        .name = "iRoPE",
        .n_layers = 4,
        .n_embd = 2048,
        .n_ff = 8192,
        .n_head = 16,
        .n_kv = 4,
        .head_dim = 128,
        .rope_dim = 128,
        .rope_theta = 10000.0,
        .rms_eps = 1e-6,
        .max_seq_len = 4096,
        .activation = .silu,
        .quant = .q4_k,
        .layer_types = ModelDesc.uniform(4, .attention),
    };
    desc.layer_head_dim[2] = 64;

    const msl = composeMSL(&buf, desc);
    // With per-layer variation, kernel should use layer arrays
    try std.testing.expect(std.mem.indexOf(u8, msl, "struct MegaAutoParams") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl, "constant uint layer_n_head[]") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl, "constant uint layer_head_dim[]") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl, "cur_n_head = layer_n_head[li]") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl, "cur_head_dim = layer_head_dim[li]") != null);
}

test "composeMSL Q5_K and Q6_K quant variants" {
    var buf: [32768]u8 = undefined;

    // Q5_K
    const desc5 = ModelDesc{
        .name = "q5k_test",
        .n_layers = 2,
        .n_embd = 1024,
        .n_ff = 4096,
        .n_head = 8,
        .n_kv = 4,
        .head_dim = 128,
        .rope_dim = 128,
        .rope_theta = 10000.0,
        .rms_eps = 1e-6,
        .max_seq_len = 2048,
        .activation = .silu,
        .quant = .q5_k,
        .layer_types = ModelDesc.uniform(2, .attention),
    };
    const msl5 = composeMSL(&buf, desc5);
    try std.testing.expect(std.mem.indexOf(u8, msl5, "mega_gemv_q5k") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl5, "mega_gemv_q4k") == null);

    // Q6_K
    const desc6 = ModelDesc{
        .name = "q6k_test",
        .n_layers = 2,
        .n_embd = 1024,
        .n_ff = 4096,
        .n_head = 8,
        .n_kv = 4,
        .head_dim = 128,
        .rope_dim = 128,
        .rope_theta = 10000.0,
        .rms_eps = 1e-6,
        .max_seq_len = 2048,
        .activation = .silu,
        .quant = .q6_k,
        .layer_types = ModelDesc.uniform(2, .attention),
    };
    const msl6 = composeMSL(&buf, desc6);
    try std.testing.expect(std.mem.indexOf(u8, msl6, "mega_gemv_q6k") != null);
}

test "composeMSL generates valid kernel for Gemma Q4_K" {
    var buf: [32768]u8 = undefined;
    const desc = ModelDesc{
        .name = "gemma",
        .n_layers = 28,
        .n_embd = 2304,
        .n_ff = 9216,
        .n_head = 8,
        .n_kv = 4,
        .head_dim = 256,
        .rope_dim = 256,
        .rope_theta = 10000.0,
        .rms_eps = 1e-6,
        .max_seq_len = 4096,
        .activation = .gelu,
        .quant = .q4_k,
        .layer_types = ModelDesc.uniform(28, .attention),
    };
    const msl = composeMSL(&buf, desc);
    // Should contain the kernel entry point
    try std.testing.expect(std.mem.indexOf(u8, msl, "kernel void megakernel_auto") != null);
    // Should contain GELU activation
    try std.testing.expect(std.mem.indexOf(u8, msl, "mega_gelu_mul") != null);
    // Should contain Q4_K GEMV
    try std.testing.expect(std.mem.indexOf(u8, msl, "mega_gemv_q4k") != null);
    // Should contain SDPA
    try std.testing.expect(std.mem.indexOf(u8, msl, "mega_sdpa_inline") != null);
}

test "composeMSL generates Qwen SiLU with fused residual" {
    var buf: [32768]u8 = undefined;
    const desc = ModelDesc{
        .name = "qwen35",
        .n_layers = 24,
        .n_embd = 1536,
        .n_ff = 4096,
        .n_head = 16,
        .n_kv = 4,
        .head_dim = 128,
        .rope_dim = 64,
        .rope_theta = 10000000.0,
        .rms_eps = 1e-6,
        .max_seq_len = 4096,
        .activation = .silu,
        .quant = .q8_0,
        .has_post_attn_norm = true,
        .fuse_residual = true,
        .layer_types = ModelDesc.qwenHybrid(24, 4),
    };
    const msl = composeMSL(&buf, desc);
    try std.testing.expect(std.mem.indexOf(u8, msl, "kernel void megakernel_auto") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl, "mega_silu_mul") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl, "mega_add_rms_norm") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl, "mega_gemv_q8") != null);
}

test "fuzz: all mega_compose functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            _ = smith;
            // ModelDesc methods
            const desc = ModelDesc{
                .name = "fuzz",
                .n_layers = 2,
                .n_embd = 64,
                .n_ff = 128,
                .n_head = 4,
                .n_kv = 2,
                .head_dim = 16,
                .rope_dim = 16,
                .rope_theta = 10000.0,
                .rms_eps = 1e-6,
                .max_seq_len = 64,
                .activation = .silu,
                .quant = .q8_0,
                .layer_types = ModelDesc.uniform(2, .attention),
            };
            _ = desc.layerNHead(0);
            _ = desc.layerNKv(0);
            _ = desc.layerHeadDim(0);
            _ = desc.layerNFf(0);
            _ = desc.layerRopeTheta(0);
            _ = desc.layerWindow(0);
            _ = desc.hasPerLayerVariation();
            _ = ModelDesc.uniform(2, .attention);
            _ = ModelDesc.qwenHybrid(2, 1);

            // composeMSL
            var buf: [32768]u8 = undefined;
            _ = composeMSL(&buf, desc);

            // gemvFn / activationFn / weightCast (module-level fns)
            _ = gemvFn(.q8_0);
            _ = activationFn(.silu);
            _ = weightCast(.q8_0);
        }
    }.f, .{});
}

test "composeMSL generates Nemotron-H ReLU² FFN" {
    var buf: [32768]u8 = undefined;
    var layer_types: [max_layers]LayerKind = undefined;
    for (0..42) |i| layer_types[i] = .ffn_only;
    layer_types[5] = .attention;
    layer_types[12] = .attention;

    const desc = ModelDesc{
        .name = "nemotron_h",
        .n_layers = 42,
        .n_embd = 3136,
        .n_ff = 12544,
        .n_head = 40,
        .n_kv = 8,
        .head_dim = 128,
        .rope_dim = 78,
        .rope_theta = 10000.0,
        .rms_eps = 1e-6,
        .max_seq_len = 4096,
        .activation = .relu_squared,
        .quant = .q8_0,
        .layer_types = layer_types,
    };
    const msl = composeMSL(&buf, desc);
    try std.testing.expect(std.mem.indexOf(u8, msl, "kernel void megakernel_auto") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl, "mega_relu_squared") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl, "mega_gemv_q8") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl, "mega_rms_norm") != null);
}
