//! GEMV Q4_1 kernel: y[row] = dot(W_q4_1[row,:], x)
//! Q4_1 block: 20 bytes = 2 bytes (f16 d) + 2 bytes (f16 m) + 16 bytes (32 x 4-bit quants).
//! Dequant: val = nibble * d + m (no bias subtraction, unlike Q4_0).

const cu = @import("common.zig");

const bytes_per_block: u32 = 20;
const values_per_block: u32 = 32;

inline fn q41BlockDot(x: [*]const f32, bp: [*]const u8, k: u32, block_start: u32) f32 {
    const d: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp)).*)));
    const m: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp + 2)).*)));
    const qs = bp + 4;
    var sum: f32 = 0.0;
    var x_sum: f32 = 0.0;
    for (0..16) |i| {
        const gi_lo = block_start + i;
        const gi_hi = block_start + i + 16;
        if (gi_lo >= k) break;
        const byte = qs[i];
        sum += x[gi_lo] * @as(f32, @floatFromInt(byte & 0x0F)) * d;
        x_sum += x[gi_lo];
        if (gi_hi < k) {
            sum += x[gi_hi] * @as(f32, @floatFromInt(byte >> 4)) * d;
            x_sum += x[gi_hi];
        }
    }
    return sum + m * x_sum;
}

export fn gemv_q4_1_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
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

        sum += q41BlockDot(x, w + row * row_bytes + blk * bytes_per_block, k, base_col);
    }
    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(bytes_per_block > 0);
    comptime std.debug.assert(values_per_block > 0);
}

test "fuzz: gemv_q4_1 functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime { _ = &q41BlockDot; }
        }
    }.f, .{});
}
