//! GEMV Q2_K kernel: y[row] = dot(W_q2k[row,:], x)
//! Q2_K super-block: 84 bytes = 16 bytes (scales) + 64 bytes (qs) + 2 bytes (f16 d) + 2 bytes (f16 dmin).
//! 256 values per super-block, 16 groups of 16 elements each.
//! Dequant: val = d * sc * q - dmin * m

const cu = @import("common.zig");

const bytes_per_block: u32 = 84;
const values_per_block: u32 = 256;

inline fn q2kBlockDot(x: [*]const f32, bp: [*]const u8, k: u32, block_start: u32) f32 {
    const scales = bp;
    const qs = bp + 16;
    const d: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp + 80)).*)));
    const dmin: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp + 82)).*)));
    var sum: f32 = 0.0;

    for (0..16) |g| {
        const sc: f32 = @floatFromInt(scales[g] & 0x0F);
        const m: f32 = @floatFromInt(scales[g] >> 4);
        const d_sc = d * sc;
        const dm_m = dmin * m;
        const base = block_start + g * 16;

        for (0..16) |l| {
            const gi = base + l;
            if (gi >= k) break;
            const byte_idx = g * 4 + l / 4;
            const bit_shift: u3 = @intCast((l % 4) * 2);
            const q: f32 = @floatFromInt((qs[byte_idx] >> bit_shift) & 0x3);
            sum += x[gi] * (d_sc * q - dm_m);
        }
    }
    return sum;
}

export fn gemv_q2_k_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
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

        // Sparse skip: check if all 256 input values are near-zero
        var bmax: f32 = 0.0;
        const check_end = @min(base_col + values_per_block, k);
        for (base_col..check_end) |i| {
            const a = @abs(x[i]);
            if (a > bmax) bmax = a;
        }
        if (bmax < sparse_threshold) continue;

        sum += q2kBlockDot(x, w + row * row_bytes + blk * bytes_per_block, k, base_col);
    }
    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(bytes_per_block > 0);
    comptime std.debug.assert(values_per_block > 0);
}

test "fuzz: gemv_q2_k functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime { _ = &q2kBlockDot; }
        }
    }.f, .{});
}
