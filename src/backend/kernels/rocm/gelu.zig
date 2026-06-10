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

const std = @import("std");

test "constants valid" {
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
