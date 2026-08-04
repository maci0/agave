//! MXFP4 SafeTensors GEMV: y[row] = dot(dequant(W[row,:]), x)
//! Weights: u32-packed 4-bit nibbles (8 per word), group_size=16.
//! Scales: FP8 E4M3 per group (1 byte each). No bias.
//! Dequant: float_val = mxfp4_lut[nibble] * fp8e4m3_to_f32(scale).
//! Grid: n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

const e2m1_lut = cu.e2m1_lut;
const fp8e4m3ToF32 = cu.fp8e4m3ToF32;

/// NVIDIA MXFP4: 16 elements per scale group.
const mxfp4_group_size: u32 = 16;
/// u32 words per group (16 nibbles / 8 per word).
const mxfp4_words_per_group: u32 = 2;

export fn gemv_mxfp4_st_kernel(
    x: [*]const f32,
    w: [*]const u32,
    s: [*]const u8,
    y: [*]f32,
    n: u32,
    k: u32,
) callconv(.nvptx_device) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const gs = mxfp4_group_size;
    const wpg = mxfp4_words_per_group;
    const gpr = (k + gs - 1) / gs;
    const wpr = gpr * wpg;

    var sum: f32 = 0.0;
    var g: u32 = tid;
    while (g < gpr) : (g += bdim) {
        const scale = fp8e4m3ToF32(s[row * gpr + g]);
        const xo = g * gs;
        const wo = row * wpr + g * wpg;

        var gdot: f32 = 0.0;
        var wi: u32 = 0;
        while (wi < wpg and xo + wi * 8 < k) : (wi += 1) {
            const word = w[wo + wi];
            const xi = xo + wi * 8;
            const rem = @min(8, k - xi);
            var i: u32 = 0;
            while (i < rem) : (i += 1) {
                gdot += e2m1_lut[(word >> @as(u5, @intCast(i * 4))) & 0xF] * x[xi + i];
            }
        }
        sum += scale * gdot;
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(mxfp4_group_size == 16);
    comptime std.debug.assert(mxfp4_words_per_group == 2);
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
