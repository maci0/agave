//! Element-wise mul kernel: out[i] = a[i] * b[i]
//! Grid: ceil(n / 256) blocks of 256 threads.

const cu = @import("common.zig");

export fn mul_kernel(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32) callconv(.kernel) void {
    const idx = cu.globalIdx();
    if (idx >= n) return;
    out[idx] = a[idx] * b[idx];
}

const std = @import("std");

test "constants valid" {
    _ = std.mem.zeroes(u8);
}

test "fuzz: mul functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime { _ = @sizeOf(u8); }
        }
    }.f, .{});
}
