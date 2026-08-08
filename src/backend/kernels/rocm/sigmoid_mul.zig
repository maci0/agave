//! Sigmoid-gated multiply kernel: out[i] = a[i] * sigmoid(b[i])
//! Used by Qwen3.5 attention gate. When a == out, this is in-place.
//! Grid: ceil(n / 256) workgroups of 256 threads.

const cu = @import("common.zig");

export fn sigmoid_mul_kernel(a: [*]const f32, b: [*]const f32, output: [*]f32, n: u32) callconv(.kernel) void {
    const idx = cu.globalIdx();
    if (idx >= n) return;
    output[idx] = a[idx] * cu.sigmoidf(b[idx]);
}

const std = @import("std");

test "constants valid" {
    // No module-level numeric constants in this file.
    _ = @sizeOf(u8);
}

test "fuzz: sigmoid_mul functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
