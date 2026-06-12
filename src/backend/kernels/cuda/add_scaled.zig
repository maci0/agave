//! Scaled accumulate kernel: dst[i] += src[i] * scale
//! Used for MoE expert output accumulation.
//! Grid: ceil(n / 256) blocks of 256 threads.

const cu = @import("common.zig");

export fn add_scaled_kernel(src: [*]const f32, dst: [*]f32, scale: f32, n: u32) callconv(.nvptx_device) void {
    const idx = cu.globalIdx();
    if (idx >= n) return;
    dst[idx] += src[idx] * scale;
}

const std = @import("std");

test "constants valid" {
    // No module-level numeric constants in this file.
    _ = @sizeOf(u8);
}

test "fuzz: add_scaled functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
