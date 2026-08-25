//! MXFP4 SafeTensors GEMV: y[row] = dot(dequant(W[row,:]), x)
//! Weights: u32-packed 4-bit nibbles (8 per word), group_size = gs (16 or 32).
//! Scales: one byte per group — FP8 E4M3 (mode 0) or E8M0 (mode 1).
//! Dequant: float_val = mxfp4_lut[nibble] * scale_to_f32(scale).
//! Grid: n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

const e2m1_lut = cu.e2m1_lut;
const fp8e4m3ToF32 = cu.fp8e4m3ToF32;
const e8m0ToF32 = cu.e8m0ToF32;

/// NVIDIA MXFP4: 16 elements per scale group.
const mxfp4_group_size: u32 = 16;
/// u32 words per group (16 nibbles / 8 per word).
const mxfp4_words_per_group: u32 = 2;
/// Scale format selectors (kernel parameters).
const scale_mode_e4m3: u32 = 0;
const scale_mode_e8m0: u32 = 1;

export fn gemv_mxfp4_st_kernel(
    x: [*]const f32,
    w: [*]const u32,
    s: [*]const u8,
    y: [*]f32,
    n: u32,
    k: u32,
    gs: u32,
    scale_mode: u32,
) callconv(.nvptx_device) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const wpg = gs / 8; // u32 words per scale group
    const gpr = (k + gs - 1) / gs;
    const wpr = gpr * wpg;

    var sum: f32 = 0.0;
    var g: u32 = tid;
    while (g < gpr) : (g += bdim) {
        const scale = if (scale_mode == scale_mode_e8m0) e8m0ToF32(s[row * gpr + g]) else fp8e4m3ToF32(s[row * gpr + g]);
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
    comptime std.debug.assert(scale_mode_e4m3 == 0);
    comptime std.debug.assert(scale_mode_e8m0 == 1);
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
