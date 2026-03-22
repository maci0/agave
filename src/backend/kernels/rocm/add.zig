//! Element-wise add kernel: out[i] = a[i] + b[i]
//! Grid: ceil(n / 256) workgroups of 256 threads.

const cu = @import("common.zig");

export fn add_kernel(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32) callconv(.kernel) void {
    const idx = cu.globalIdx();
    if (idx >= n) return;
    out[idx] = a[idx] + b[idx];
}
