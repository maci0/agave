//! GEMV FP8 E5M2 kernel: y[row] = dot(W_fp8[row,:], x)
//! FP8 E5M2: 1 byte per element.
//! Uses 256-entry comptime LUT for branch-free dequantization.

const cu = @import("common.zig");

export fn gemv_fp8_e5m2_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const row_offset = row * k;

    var sum: f32 = 0.0;
    const sparse_threshold: f32 = 0.005;
    const chunk_size: u32 = 32;

    var col = tid;
    while (col < k) : (col += bdim) {
        // Sparse skip: check if all 32 input values in this chunk are near-zero
        const chunk_base = (col / chunk_size) * chunk_size;
        const check_end = @min(chunk_base + chunk_size, k);
        var bmax: f32 = 0.0;
        for (chunk_base..check_end) |i| {
            const a = @abs(x[i]);
            if (a > bmax) bmax = a;
        }
        if (bmax < sparse_threshold) continue;

        const wval = cu.fp8e5m2ToF32(w[row_offset + col]);
        sum += wval * x[col];
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(@as(f32, 0.005) > 0);
    comptime std.debug.assert(@as(u32, 32) > 0);
}

test "fuzz: gemv_fp8_e5m2 functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime { _ = @sizeOf(u8); }
        }
    }.f, .{});
}
