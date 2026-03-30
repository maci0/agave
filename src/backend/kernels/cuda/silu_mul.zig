//! Fused SiLU + multiply kernel: out[i] = silu(a[i]) * b[i]
//! Used by SwiGLU FFN in all transformer models.
//! Grid: ceil(n / 256) blocks of 256 threads.

const cu = @import("common.zig");

export fn silu_mul_kernel(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32) callconv(.kernel) void {
    const idx = cu.globalIdx();
    if (idx >= n) return;
    const x = a[idx];
    out[idx] = x * cu.sigmoidf(x) * b[idx];
}
