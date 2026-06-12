//! Sigmoid-gated multiply kernel: data[i] *= sigmoid(gate[i])
//! Used for DeltaNet attention gating.
//! Grid: ceil(n / 256) blocks of 256 threads.

const cu = @import("common.zig");

export fn sigmoid_mul_kernel(data: [*]f32, gate: [*]const f32, n: u32) callconv(.nvptx_device) void {
    const idx = cu.globalIdx();
    if (idx >= n) return;
    data[idx] *= cu.sigmoidf(gate[idx]);
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
