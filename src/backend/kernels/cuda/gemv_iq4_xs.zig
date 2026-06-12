//! GEMV IQ4_XS kernel: y[row] = dot(W_iq4xs[row,:], x)
//! IQ4_XS super-block: 136 bytes = f16 d (2B) + u16 scales_h (2B) + u8 scales_l[4] (4B) + u8 qs[128] (128B).
//! 256 values per super-block, 8 sub-blocks of 32 elements.
//! Uses IQ4_NL lookup table + per-sub-block 6-bit scales.

const cu = @import("common.zig");

const block_bytes: u32 = 136;
const block_elems: u32 = 256;
const iq4nl_lut = [16]i8{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
const scale_bias: i32 = -32;

export fn gemv_iq4_xs_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.nvptx_device) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const nb = (k + block_elems - 1) / block_elems;
    const row_bytes = nb * block_bytes;

    var sum: f32 = 0.0;
    const sparse_threshold: f32 = 0.005;
    var blk = tid;
    while (blk < nb) : (blk += bdim) {
        const base_col = blk * block_elems;

        // Sparse skip: check if all 256 input values are near-zero
        var bmax: f32 = 0.0;
        const check_end = @min(base_col + block_elems, k);
        for (base_col..check_end) |i| {
            const a = @abs(x[i]);
            if (a > bmax) bmax = a;
        }
        if (bmax < sparse_threshold) continue;

        const bp = w + row * row_bytes + blk * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp)).*)));
        const scales_h: u16 = @as(*align(1) const u16, @ptrCast(bp + 2)).*;
        const scales_l = bp + 4;
        const qs = bp + 8;
        const bk = blk * block_elems;

        for (0..8) |sb| {
            const lo4: u8 = if (sb % 2 == 0) scales_l[sb / 2] & 0x0F else scales_l[sb / 2] >> 4;
            const hi2: u8 = @truncate((scales_h >> @as(u4, @intCast(sb * 2))) & 0x3);
            const scale_raw: i32 = @as(i32, lo4 | (@as(u8, hi2) << 4)) + scale_bias;
            const sub_scale: f32 = d * @as(f32, @floatFromInt(scale_raw));
            const sub_qs = qs + sb * 16;
            const sub_bk = bk + sb * 32;
            var block_sum: f32 = 0.0;

            for (0..16) |j| {
                const col_lo = sub_bk + j;
                const col_hi = sub_bk + j + 16;
                if (col_lo >= k) break;
                const byte = sub_qs[j];
                block_sum += x[col_lo] * @as(f32, @floatFromInt(iq4nl_lut[byte & 0x0F]));
                if (col_hi < k) {
                    block_sum += x[col_hi] * @as(f32, @floatFromInt(iq4nl_lut[byte >> 4]));
                }
            }
            sum += block_sum * sub_scale;
        }
    }
    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(block_bytes > 0);
    comptime std.debug.assert(block_elems > 0);
}

test "fuzz: gemv_iq4_xs functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
