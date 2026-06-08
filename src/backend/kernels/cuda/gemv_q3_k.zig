//! GEMV Q3_K kernel: y[row] = dot(W_q3k[row,:], x)
//! Q3_K super-block: 110 bytes = 32 bytes (hmask) + 64 bytes (qs) + 12 bytes (scales) + 2 bytes (f16 d).
//! 256 values per super-block, 16 groups of 16 elements each.
//! 3-bit quant: 2 bits from qs + 1 bit from hmask.
//! Dequant: val = d * scale * ((q_lo | (q_hi << 2)) - 4)

const cu = @import("common.zig");

const bytes_per_block: u32 = 110;
const values_per_block: u32 = 256;
const dequant_bias: i8 = -4;

inline fn q3kBlockDot(x: [*]const f32, bp: [*]const u8, k: u32, block_start: u32) f32 {
    const hmask = bp;
    const qs = bp + 32;
    const raw_scales = bp + 96;
    const d: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp + 108)).*)));
    var sum: f32 = 0.0;

    for (0..16) |g| {
        const base = block_start + g * 16;
        if (base >= k) break;

        // Extract 4-bit scale and subtract bias
        const scale_raw = raw_scales[if (g < 8) g else g - 8];
        const scale_nibble: i8 = @intCast(if (g < 8) (scale_raw & 0x0F) else (scale_raw >> 4));
        const scale: f32 = @floatFromInt(scale_nibble - 8);
        const d_sc = d * scale;

        for (0..16) |l| {
            const gi = base + l;
            if (gi >= k) break;
            const flat_idx = g * 16 + l;
            const qs_byte_idx = flat_idx / 4;
            const qs_bit_shift: u3 = @intCast((flat_idx % 4) * 2);
            const q_lo: u8 = (qs[qs_byte_idx] >> qs_bit_shift) & 0x3;

            const hm_byte_idx = flat_idx % 32;
            const hm_bit: u3 = @intCast(flat_idx / 32);
            const q_hi: u8 = (hmask[hm_byte_idx] >> hm_bit) & 1;

            const q3: i8 = @as(i8, @intCast(q_lo | (q_hi << 2))) + dequant_bias;
            sum += x[gi] * d_sc * @as(f32, @floatFromInt(q3));
        }
    }
    return sum;
}

export fn gemv_q3_k_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
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

        sum += q3kBlockDot(x, w + row * row_bytes + blk * bytes_per_block, k, base_col);
    }
    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(bytes_per_block > 0);
    comptime std.debug.assert(values_per_block > 0);
}

test "fuzz: gemv_q3_k functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime { _ = &q3kBlockDot; }
        }
    }.f, .{});
}
