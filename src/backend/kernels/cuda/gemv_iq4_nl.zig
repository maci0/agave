//! GEMV IQ4_NL kernel: y[row] = dot(W_iq4nl[row,:], x)
//! IQ4_NL block: 18 bytes = f16 scale (2B) + 16 nibble bytes (32 values).
//! Uses non-linear 16-entry lookup table for dequant.

const cu = @import("common.zig");

const block_bytes: u32 = 18;
const block_elems: u32 = 32;

const iq4nl_lut = [16]i8{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

inline fn iq4nlBlockDot(x: [*]const f32, bp: [*]const u8, k: u32, base: u32) f32 {
    const scale_bits = @as(u16, bp[0]) | (@as(u16, bp[1]) << 8);
    const d: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));
    const qs = bp + 2;
    var sum: f32 = 0.0;

    for (0..16) |j| {
        const col_lo = base + j;
        const col_hi = base + j + 16;
        if (col_lo >= k) break;
        const byte = qs[j];
        sum += x[col_lo] * @as(f32, @floatFromInt(iq4nl_lut[byte & 0x0F]));
        if (col_hi < k) {
            sum += x[col_hi] * @as(f32, @floatFromInt(iq4nl_lut[byte >> 4]));
        }
    }
    return sum * d;
}

export fn gemv_iq4_nl_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
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

        // Sparse skip: check if all 32 input values are near-zero
        var bmax: f32 = 0.0;
        for (0..block_elems) |i| {
            if (base_col + i < k) {
                const a = @abs(x[base_col + i]);
                if (a > bmax) bmax = a;
            }
        }
        if (bmax < sparse_threshold) continue;

        sum += iq4nlBlockDot(x, w + row * row_bytes + blk * block_bytes, k, base_col);
    }
    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(block_bytes > 0);
    comptime std.debug.assert(block_elems > 0);
}

test "fuzz: gemv_iq4_nl functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = &iq4nlBlockDot;
            }
        }
    }.f, .{});
}
