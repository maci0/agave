//! Megakernel weight packing: computes per-layer byte offsets into the GGUF
//! memory-mapped weight buffer for single-dispatch GPU execution.

const std = @import("std");
const format_mod = @import("../format/format.zig");
const Format = format_mod.Format;

/// Maximum layers supported in a megakernel weight pack.
const max_layers: usize = 64;

/// Per-layer weight offset table. All offsets are byte offsets from the
/// earliest tensor data pointer in the model file.
pub const LayerOffsets = struct {
    /// Attention/DeltaNet norm
    attn_norm: usize = 0,
    /// Full attention Q projection
    attn_q: usize = 0,
    /// Full attention K projection
    attn_k: usize = 0,
    /// Full attention V projection
    attn_v: usize = 0,
    /// Q head norm (post-projection)
    attn_q_norm: usize = 0,
    /// K head norm (post-projection)
    attn_k_norm: usize = 0,
    /// Attention output projection
    attn_output: usize = 0,
    /// Fused QKV projection (DeltaNet/Mamba layers)
    attn_qkv: usize = 0,
    /// Gating projection (DeltaNet)
    attn_gate: usize = 0,
    /// SSM alpha parameter
    ssm_alpha: usize = 0,
    /// SSM beta parameter
    ssm_beta: usize = 0,
    /// SSM A matrix (state transition)
    ssm_a: usize = 0,
    /// SSM dt bias
    ssm_dt_bias: usize = 0,
    /// SSM 1D convolution weights
    ssm_conv1d: usize = 0,
    /// SSM output norm
    ssm_norm: usize = 0,
    /// SSM output projection
    ssm_out: usize = 0,
    /// Post-attention layernorm
    post_attn_norm: usize = 0,
    /// FFN gate projection (SwiGLU first half)
    ffn_gate: usize = 0,
    /// FFN up projection (SwiGLU second half)
    ffn_up: usize = 0,
    /// FFN down projection
    ffn_down: usize = 0,
};

/// Packed weight metadata with per-layer offset table.
pub const WeightPack = struct {
    /// Base pointer — earliest tensor data address in the mmap.
    base_ptr: [*]const u8,
    /// Per-layer byte offsets relative to `base_ptr`.
    layer_offsets: [max_layers]LayerOffsets,
    /// Output norm weight offset.
    output_norm: usize,
    /// Output projection weight offset.
    output_weight: usize,
    /// Token embedding table offset.
    token_embd: usize,
    /// Total number of layers packed.
    n_layers: u32,
};

/// Scan all layer weight tensors and record their byte offsets relative to
/// the earliest tensor data pointer. GGUF files memory-map weights contiguously,
/// so no copying is needed — the kernel accesses `base_ptr + offset`.
pub fn computeOffsets(fmt: Format, n_layers: u32) WeightPack {
    std.debug.assert(n_layers <= max_layers);
    var pack = WeightPack{
        .base_ptr = undefined,
        .layer_offsets = undefined,
        .output_norm = 0,
        .output_weight = 0,
        .token_embd = 0,
        .n_layers = n_layers,
    };
    @memset(&pack.layer_offsets, LayerOffsets{});

    // Find base pointer (earliest tensor data address)
    var base_addr: usize = std.math.maxInt(usize);

    const layer_names = .{
        "attn_norm.weight",   "attn_q.weight",      "attn_k.weight",
        "attn_v.weight",      "attn_q_norm.weight",  "attn_k_norm.weight",
        "attn_output.weight", "attn_qkv.weight",    "attn_gate.weight",
        "ssm_alpha.weight",   "ssm_beta.weight",    "ssm_a",
        "ssm_dt.bias",        "ssm_conv1d.weight",  "ssm_norm.weight",
        "ssm_out.weight",     "post_attention_norm.weight",
        "ffn_gate.weight",    "ffn_up.weight",      "ffn_down.weight",
    };

    for (0..n_layers) |li| {
        const l: u32 = @intCast(li);
        for (layer_names) |name| {
            if (fmt.layerTensor(l, name)) |t| {
                base_addr = @min(base_addr, @intFromPtr(t.data_ptr));
            }
        }
    }
    // Global tensors
    for (.{ "output_norm.weight", "token_embd.weight" }) |name| {
        if (fmt.getTensor(name)) |t| {
            base_addr = @min(base_addr, @intFromPtr(t.data_ptr));
        }
    }
    if (fmt.getTensor("output.weight")) |t| {
        base_addr = @min(base_addr, @intFromPtr(t.data_ptr));
    }

    if (base_addr == std.math.maxInt(usize)) {
        @panic("megakernel: no weight tensors found — wrong model format?");
    }
    pack.base_ptr = @ptrFromInt(base_addr);

    // Record per-layer offsets relative to base
    for (0..n_layers) |li| {
        const l: u32 = @intCast(li);
        const lo = &pack.layer_offsets[li];
        if (fmt.layerTensor(l, "attn_norm.weight")) |t| lo.attn_norm = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "attn_q.weight")) |t| lo.attn_q = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "attn_k.weight")) |t| lo.attn_k = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "attn_v.weight")) |t| lo.attn_v = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "attn_q_norm.weight")) |t| lo.attn_q_norm = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "attn_k_norm.weight")) |t| lo.attn_k_norm = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "attn_output.weight")) |t| lo.attn_output = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "attn_qkv.weight")) |t| lo.attn_qkv = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "attn_gate.weight")) |t| lo.attn_gate = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "ssm_alpha.weight")) |t| lo.ssm_alpha = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "ssm_beta.weight")) |t| lo.ssm_beta = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "ssm_a")) |t| lo.ssm_a = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "ssm_dt.bias")) |t| lo.ssm_dt_bias = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "ssm_conv1d.weight")) |t| lo.ssm_conv1d = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "ssm_norm.weight")) |t| lo.ssm_norm = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "ssm_out.weight")) |t| lo.ssm_out = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "post_attention_norm.weight")) |t| lo.post_attn_norm = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "ffn_gate.weight")) |t| lo.ffn_gate = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "ffn_up.weight")) |t| lo.ffn_up = @intFromPtr(t.data_ptr) - base_addr;
        if (fmt.layerTensor(l, "ffn_down.weight")) |t| lo.ffn_down = @intFromPtr(t.data_ptr) - base_addr;
    }

    // Global tensor offsets
    if (fmt.getTensor("output_norm.weight")) |t| pack.output_norm = @intFromPtr(t.data_ptr) - base_addr;
    if (fmt.getTensor("output.weight") orelse fmt.getTensor("token_embd.weight")) |t| pack.output_weight = @intFromPtr(t.data_ptr) - base_addr;
    if (fmt.getTensor("token_embd.weight")) |t| pack.token_embd = @intFromPtr(t.data_ptr) - base_addr;

    return pack;
}

// ── Tests ─────────────────────────────────────────────────────────

test "LayerOffsets default zero" {
    const lo = LayerOffsets{};
    try std.testing.expectEqual(@as(usize, 0), lo.attn_norm);
    try std.testing.expectEqual(@as(usize, 0), lo.ffn_down);
}
