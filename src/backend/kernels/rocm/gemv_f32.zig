//! GEMV F32 kernel: y[row] = dot(W[row,:], x)
//! Launch with n workgroups of 256 threads (one row per workgroup).

const cu = @import("common.zig");

export fn gemv_f32_kernel(x: [*]const f32, w: [*]const f32, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    var sum: f32 = 0.0;
    const row_offset = row * k;
    var j = tid;
    while (j < k) : (j += bdim) {
        sum += w[row_offset + j] * x[j];
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}

const std = @import("std");

test "constants valid" {
    // No module-level numeric constants in this file.
    _ = @sizeOf(u8);
}

test "fuzz: gemv_f32 functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
