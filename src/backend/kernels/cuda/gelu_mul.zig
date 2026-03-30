//! Fused GELU + multiply kernel: out[i] = gelu(a[i]) * b[i]
//! Used by GeGLU FFN in Gemma3 models.
//! Grid: ceil(n / 256) blocks of 256 threads.

const cu = @import("common.zig");

const sqrt_2_over_pi: f32 = 0.7978845608028654;
const gelu_coeff: f32 = 0.044715;
/// GELU tanh-argument clamp bound (prevents exp overflow in tanhf).
const gelu_clamp_bound: f32 = 10.0;

export fn gelu_mul_kernel(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32) callconv(.kernel) void {
    const idx = cu.globalIdx();
    if (idx >= n) return;
    const x = a[idx];
    const t = sqrt_2_over_pi * (x + gelu_coeff * x * x * x);
    const t_clamped = @min(@max(t, -gelu_clamp_bound), gelu_clamp_bound);
    out[idx] = 0.5 * x * (1.0 + cu.tanhf(t_clamped)) * b[idx];
}
