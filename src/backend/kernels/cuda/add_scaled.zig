//! Scaled accumulate kernel: dst[i] += src[i] * scale
//! Used for MoE expert output accumulation.
//! Grid: ceil(n / 256) blocks of 256 threads.

const cu = @import("common.zig");

export fn add_scaled_kernel(src: [*]const f32, dst: [*]f32, scale: f32, n: u32) callconv(.kernel) void {
    const idx = cu.globalIdx();
    if (idx >= n) return;
    dst[idx] += src[idx] * scale;
}
