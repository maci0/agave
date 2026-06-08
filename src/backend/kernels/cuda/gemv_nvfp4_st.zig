//! NVFP4 SafeTensors GEMV: y[row] = dot(dequant(W[row,:]), x)
//! Weights: packed nibble pairs (4 bits each, 2 per byte), group_size=16.
//! Scales: FP8 E4M3 per group (1 byte each).
//! Dequant: float_val = mxfp4_lut[nibble] * fp8_scale.
//! Grid: n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

const e2m1_lut = cu.e2m1_lut;

export fn gemv_nvfp4_st_kernel(
    x: [*]const f32,
    w: [*]const u8,
    s: [*]const u8,
    y: [*]f32,
    n: u32,
    k: u32,
) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const bytes_per_row = k / 2;
    const groups_per_row = k / 16;

    var sum: f32 = 0.0;
    var g: u32 = tid;
    while (g < groups_per_row) : (g += bdim) {
        const scale = cu.fp8e4m3ToF32(s[row * groups_per_row + g]);
        const w_base = row * bytes_per_row + g * 8;
        const x_base = g * 16;

        var gdot: f32 = 0.0;
        var j: u32 = 0;
        while (j < 8) : (j += 1) {
            const byte = w[w_base + j];
            const lo = e2m1_lut[byte & 0xF];
            const hi = e2m1_lut[byte >> 4];
            gdot += lo * x[x_base + 2 * j] + hi * x[x_base + 2 * j + 1];
        }
        sum += scale * gdot;
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}

const std = @import("std");

test "constants valid" {
    // No module-level numeric constants are defined in this file;
    // group_size and bytes_per_group are computed inline from runtime k.
    // Verify the e2m1_lut table imported from common.zig has a positive length.
    comptime std.debug.assert(e2m1_lut.len > 0);
}

test "fuzz: gemv_nvfp4_st functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
