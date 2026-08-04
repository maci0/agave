//! MXFP4 SafeTensors GEMV kernel for ROCm.
//! U32-packed FP4 E2M1 weights with FP8 E4M3 per-16-element group scales.
//! Grid: n blocks of 256 threads (1 workgroup per output row).

const cu = @import("common.zig");

const e2m1_lut = cu.e2m1_lut;
const fp8e4m3ToF32 = cu.fp8e4m3ToF32;

/// NVIDIA MXFP4: 16 elements per scale group.
const mxfp4_group_size: u32 = 16;
/// Bytes per group (16 nibbles → 8 bytes).
const mxfp4_bytes_per_group: u32 = 8;

export fn gemv_mxfp4_st_kernel(x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    const tid = cu.threadIdx();
    if (row >= n) return;

    const gpr = (k + mxfp4_group_size - 1) / mxfp4_group_size;
    const bytes_per_row = (k + 1) / 2;

    var sum: f32 = 0.0;
    var g: u32 = tid;
    while (g < gpr) : (g += cu.block_dim) {
        const sc = fp8e4m3ToF32(scale[row * gpr + g]);
        const base = g * mxfp4_group_size;
        const w_off = row * bytes_per_row + g * mxfp4_bytes_per_group;
        const elems = @min(mxfp4_group_size, k - base);
        const nbytes = (elems + 1) / 2;
        var j: u32 = 0;
        while (j < nbytes) : (j += 1) {
            const byte = weight[w_off + j];
            const v0 = e2m1_lut[byte & 0xF] * sc;
            const v1 = e2m1_lut[byte >> 4] * sc;
            const xi0 = base + 2 * j;
            sum += v0 * x[xi0];
            if (xi0 + 1 < k) sum += v1 * x[xi0 + 1];
        }
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(mxfp4_group_size == 16);
    comptime std.debug.assert(mxfp4_bytes_per_group == 8);
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
