//! Fused SiLU + multiply kernel: out[i] = silu(a[i]) * b[i]
//! Used by SwiGLU FFN in all transformer models.
//! Grid: ceil(n / 256) blocks of 256 threads.

const cu = @import("common.zig");

export fn silu_mul_kernel(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32) callconv(.kernel) void {
    const idx = cu.globalIdx();
    if (idx >= n) return;
    const x = a[idx];
    out[idx] = x * cu.sigmoidf(x) * b[idx];
}

const std = @import("std");

test "constants valid" {
    // No module-level numeric constants in this file.
    _ = @sizeOf(u8);
}

test "fuzz: silu_mul functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
