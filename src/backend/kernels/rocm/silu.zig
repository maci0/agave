//! SiLU activation kernel: y[i] = x[i] * sigmoid(x[i])
//! Grid: ceil(n / 256) workgroups of 256 threads.

const cu = @import("common.zig");

export fn silu_kernel(input: [*]const f32, output: [*]f32, n: u32) callconv(.kernel) void {
    const idx = cu.globalIdx();
    if (idx >= n) return;
    const x = input[idx];
    output[idx] = x * cu.sigmoidf(x);
}
