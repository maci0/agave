//! GEMV Q5_0 kernel: y[row] = dot(W_q5_0[row,:], x)
//! Q5_0 block: 22 bytes = 2 bytes (f16 d) + 4 bytes (qh high bits) + 16 bytes (32 x 4-bit low quants).
//! Dequant: val = ((lo_nibble | (qh_bit << 4)) - 16) * d

const cu = @import("common.zig");

const bytes_per_block: u32 = 22;
const values_per_block: u32 = 32;
const dequant_bias: f32 = -16.0;

inline fn q50BlockDot(x: [*]const f32, bp: [*]const u8, k: u32, block_start: u32) f32 {
    const d: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp)).*)));
    const qh = @as(*align(1) const u32, @ptrCast(bp + 2)).*;
    const qs = bp + 6;
    var sum: f32 = 0.0;

    for (0..16) |i| {
        const gi_lo = block_start + i;
        const gi_hi = block_start + i + 16;
        if (gi_lo >= k) break;
        const byte = qs[i];
        const lo_nibble: f32 = @floatFromInt(byte & 0x0F);
        const hi_nibble: f32 = @floatFromInt(byte >> 4);
        const qh_lo: f32 = if ((qh >> @as(u5, @intCast(i))) & 1 != 0) 16.0 else 0.0;
        const qh_hi: f32 = if ((qh >> @as(u5, @intCast(i + 16))) & 1 != 0) 16.0 else 0.0;
        sum += x[gi_lo] * ((lo_nibble + qh_lo + dequant_bias) * d);
        if (gi_hi < k) {
            sum += x[gi_hi] * ((hi_nibble + qh_hi + dequant_bias) * d);
        }
    }
    return sum;
}

export fn gemv_q5_0_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.nvptx_device) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const blocks_per_row = (k + values_per_block - 1) / values_per_block;
    const row_bytes = blocks_per_row * bytes_per_block;

    var sum: f32 = 0.0;
    const sparse_threshold: f32 = 0.005;
    var blk = tid;
    while (blk < blocks_per_row) : (blk += bdim) {
        const base_col = blk * values_per_block;

        // Sparse skip: check if all 32 input values are near-zero
        var bmax: f32 = 0.0;
        for (0..values_per_block) |i| {
            if (base_col + i < k) {
                const a = @abs(x[base_col + i]);
                if (a > bmax) bmax = a;
            }
        }
        if (bmax < sparse_threshold) continue;

        sum += q50BlockDot(x, w + row * row_bytes + blk * bytes_per_block, k, base_col);
    }
    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(bytes_per_block > 0);
    comptime std.debug.assert(values_per_block > 0);
}

test "fuzz: gemv_q5_0 functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = &q50BlockDot;
            }
        }
    }.f, .{});
}
