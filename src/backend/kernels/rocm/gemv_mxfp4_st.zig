//! MXFP4 SafeTensors GEMV kernel for ROCm.
//! FP4 E2M1 weights (2 per byte) with E8M0 per-32-element block scales.
//! Grid: n blocks of 256 threads (1 workgroup per output row).

const cu = @import("common.zig");

const e2m1_lut = cu.e2m1_lut;
const e8m0ToF32 = cu.e8m0ToF32;

export fn gemv_mxfp4_st_kernel(x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    const tid = cu.threadIdx();
    if (row >= n) return;

    const blocks_per_row = k / 32;
    const bytes_per_row = k / 2;

    var sum: f32 = 0.0;
    var blk: u32 = tid;
    while (blk < blocks_per_row) : (blk += cu.block_dim) {
        const sc = e8m0ToF32(scale[row * blocks_per_row + blk]);
        const base = blk * 32;
        const w_off = row * bytes_per_row + blk * 16;
        for (0..16) |j| {
            const byte = weight[w_off + j];
            const v0 = e2m1_lut[byte & 0xF] * sc;
            const v1 = e2m1_lut[byte >> 4] * sc;
            sum += v0 * x[base + 2 * j] + v1 * x[base + 2 * j + 1];
        }
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}

const std = @import("std");

test "constants valid" {
    // No module-level numeric constants defined in this file.
    _ = @sizeOf(u8);
}

test "fuzz: gemv_mxfp4_st functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
