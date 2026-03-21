//! Sigmoid-gated multiply kernel: out[i] = a[i] * sigmoid(b[i])
//! Used by Qwen3.5 attention gate. When a == out, this is in-place.
//! Grid: ceil(n / 256) workgroups of 256 threads.

const cu = @import("common.zig");

export fn sigmoid_mul_kernel(a: [*]const f32, b: [*]const f32, output: [*]f32, n: u32) callconv(.kernel) void {
    const idx = cu.globalIdx();
    if (idx >= n) return;
    output[idx] = a[idx] * cu.sigmoidf(b[idx]);
}

/// Fused SiLU + multiply: out[i] = silu(a[i]) * b[i]
/// Used in SwiGLU FFN to avoid separate silu + mul dispatches.
export fn silu_mul_kernel(a: [*]const f32, b: [*]const f32, output: [*]f32, n: u32) callconv(.kernel) void {
    const idx = cu.globalIdx();
    if (idx >= n) return;
    const x = a[idx];
    output[idx] = (x * cu.sigmoidf(x)) * b[idx];
}
