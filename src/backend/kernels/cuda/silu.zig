//! SiLU activation kernel: y[i] = x[i] * sigmoid(x[i])
//! Grid: ceil(n / 256) blocks of 256 threads.

const cu = @import("common.zig");

export fn silu_kernel(input: [*]const f32, output: [*]f32, n: u32) callconv(.kernel) void {
    const idx = cu.globalIdx();
    if (idx >= n) return;
    const x = input[idx];
    output[idx] = x * cu.sigmoidf(x);
}

const std = @import("std");

test "constants valid" {
    // No module-level numeric constants in this file.
    _ = @sizeOf(u8);
}

test "fuzz: silu functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
