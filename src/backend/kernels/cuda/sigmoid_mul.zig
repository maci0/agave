//! Sigmoid-gated multiply kernel: data[i] *= sigmoid(gate[i])
//! Used for DeltaNet attention gating.
//! Grid: ceil(n / 256) blocks of 256 threads.

const cu = @import("common.zig");

export fn sigmoid_mul_kernel(data: [*]f32, gate: [*]const f32, n: u32) callconv(.kernel) void {
    const idx = cu.globalIdx();
    if (idx >= n) return;
    data[idx] *= cu.sigmoidf(gate[idx]);
}
