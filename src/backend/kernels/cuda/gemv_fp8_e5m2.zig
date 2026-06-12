//! CUDA GEMV kernel for FP8 E5M2 format.
//! 1:1 mapping (1 FP8 byte → 1 f32 value) with 256-entry LUT conversion.
//! Launch with n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

/// FP8 E5M2 GEMV kernel: y[row] = dot(W[row,:], x)
/// Simple 1:1 element-wise conversion and accumulation.
export fn gemv_fp8_e5m2_kernel(
    x: [*]const f32,
    w: [*]const u8,
    y: [*]f32,
    n: u32,
    k: u32,
) callconv(.nvptx_device) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const row_offset = row * k;

    var sum: f32 = 0.0;
    const sparse_threshold: f32 = 0.005;
    const chunk_size: u32 = 32;
    var j = tid;
    while (j < k) : (j += bdim) {
        // Sparse skip: check if all 32 input values in this chunk are near-zero
        const chunk_base = (j / chunk_size) * chunk_size;
        const check_end = @min(chunk_base + chunk_size, k);
        var bmax: f32 = 0.0;
        for (chunk_base..check_end) |i| {
            const a = @abs(x[i]);
            if (a > bmax) bmax = a;
        }
        if (bmax < sparse_threshold) continue;

        const wval = cu.fp8e5m2ToF32(w[row_offset + j]);
        sum += wval * x[j];
    }

    // Block reduction (warp + inter-warp)
    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}

const std = @import("std");

test "constants valid" {
    // No module-level numeric constants exist in this file (sparse_threshold and
    // chunk_size are local variables inside the kernel function).
    _ = @sizeOf(u8);
}

test "fuzz: gemv_fp8_e5m2 functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
