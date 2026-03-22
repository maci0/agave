//! GELU activation kernel: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//! Grid: ceil(n / 256) workgroups of 256 threads.

const cu = @import("common.zig");

const sqrt_2_over_pi: f32 = 0.7978845608028654;
const gelu_coeff: f32 = 0.044715;

export fn gelu_kernel(input: [*]const f32, output: [*]f32, n: u32) callconv(.kernel) void {
    const idx = cu.globalIdx();
    if (idx >= n) return;
    const x = input[idx];
    const inner = sqrt_2_over_pi * @mulAdd(f32, gelu_coeff * x * x, x, x);
    output[idx] = 0.5 * x * (1.0 + cu.tanhf(inner));
}
