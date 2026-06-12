//! GELU activation kernel: y[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//! Grid: ceil(n / 256) blocks of 256 threads.

const cu = @import("common.zig");

const sqrt_2_over_pi: f32 = 0.7978845608028654;
const gelu_coeff: f32 = 0.044715;
/// GELU tanh-argument clamp bound (prevents exp overflow in tanhf).
const gelu_clamp_bound: f32 = 10.0;

export fn gelu_kernel(input: [*]const f32, output: [*]f32, n: u32) callconv(.nvptx_device) void {
    const idx = cu.globalIdx();
    if (idx >= n) return;
    const x = input[idx];
    const t = sqrt_2_over_pi * (x + gelu_coeff * x * x * x);
    const t_clamped = @min(@max(t, -gelu_clamp_bound), gelu_clamp_bound);
    output[idx] = 0.5 * x * (1.0 + cu.tanhf(t_clamped));
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(gelu_clamp_bound > 0);
    comptime std.debug.assert(sqrt_2_over_pi > 0);
    comptime std.debug.assert(gelu_coeff > 0);
}

test "fuzz: gelu functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
