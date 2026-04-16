//! GEMV Q5_K kernel: y[row] = dot(W_q5k[row,:], x)
//! Q5_K block: 176 bytes = d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128].
//! 256 values per super-block, 8 sub-blocks of 32 values each.
//! 5-bit values: 4 low bits in qs[], 1 high bit in qh[].
//! NR=2: Launch with ceil(n/2) workgroups, each processes 2 output rows.

const cu = @import("common.zig");

const getScaleMinK4 = cu.getScaleMinK4;

/// Bytes per Q5_K super-block (256 elements).
const q5_k_block_size: u32 = 176;
/// Elements per Q5_K super-block.
const q5_k_group_size: u32 = 256;
/// Q5_K high-bit contribution: the 5th bit adds 2^4 = 16 to the value.
const q5_k_high_bit_int: u8 = 16;
/// Number of output rows per workgroup.
const nr: u32 = 2;

/// Compute one super-block's dot product for a single row.
inline fn q5kBlockDot(x: [*]const f32, blk_addr: usize, k: u32, base_col: u32) f32 {
    const d: f32 = @floatCast(@as(f16, @bitCast(@as(
        *align(1) const u16,
        @ptrFromInt(blk_addr),
    ).*)));
    const dmin: f32 = @floatCast(@as(f16, @bitCast(@as(
        *align(1) const u16,
        @ptrFromInt(blk_addr + 2),
    ).*)));

    const scales = @as([*]const u8, @ptrFromInt(blk_addr + 4));
    const qh = @as([*]const u8, @ptrFromInt(blk_addr + 16));
    const qs = @as([*]const u8, @ptrFromInt(blk_addr + 48));

    var sum: f32 = 0.0;

    for (0..4) |group| {
        const j = group * 64;
        const is = group * 2;
        const umask1: u8 = @as(u8, 1) << @intCast(group * 2);
        const umask2: u8 = @as(u8, 2) << @intCast(group * 2);
        const ql_off = group * 32;

        var sc1: u8 = undefined;
        var m1: u8 = undefined;
        var sc2: u8 = undefined;
        var m2: u8 = undefined;
        getScaleMinK4(is + 0, scales, &sc1, &m1);
        getScaleMinK4(is + 1, scales, &sc2, &m2);

        const gi_base = base_col + j;
        if (gi_base >= k) break;

        const d_sc1 = d * @as(f32, @floatFromInt(sc1));
        const dm_m1 = dmin * @as(f32, @floatFromInt(m1));
        const d_sc2 = d * @as(f32, @floatFromInt(sc2));
        const dm_m2 = dmin * @as(f32, @floatFromInt(m2));

        // First half: low nibbles + umask1
        {
            var sum1: f32 = 0.0;
            var x_sum1: f32 = 0.0;
            const count1 = @min(32, k - gi_base);
            for (0..count1) |l| {
                const lo = qs[ql_off + l] & 0x0F;
                const hi: u8 = if ((qh[l] & umask1) != 0) q5_k_high_bit_int else 0;
                const qval: f32 = @floatFromInt(lo + hi);
                sum1 += x[gi_base + l] * qval;
                x_sum1 += x[gi_base + l];
            }
            sum += d_sc1 * sum1 - dm_m1 * x_sum1;
        }

        // Second half: high nibbles + umask2
        const gi_base2 = gi_base + 32;
        if (gi_base2 < k) {
            var sum2: f32 = 0.0;
            var x_sum2: f32 = 0.0;
            const count2 = @min(32, k - gi_base2);
            for (0..count2) |l| {
                const lo = qs[ql_off + l] >> 4;
                const hi: u8 = if ((qh[l] & umask2) != 0) q5_k_high_bit_int else 0;
                const qval: f32 = @floatFromInt(lo + hi);
                sum2 += x[gi_base2 + l] * qval;
                x_sum2 += x[gi_base2 + l];
            }
            sum += d_sc2 * sum2 - dm_m2 * x_sum2;
        }
    }
    return sum;
}

export fn gemv_q5_k_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row_base = cu.blockIdx() * nr;
    if (row_base >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const nr_active = @min(nr, n - row_base);

    const blocks_per_row = (k + q5_k_group_size - 1) / q5_k_group_size;
    const row_bytes = blocks_per_row * q5_k_block_size;

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;

    var blk = tid;
    while (blk < blocks_per_row) : (blk += bdim) {
        const base_col = blk * q5_k_group_size;

        // Row 0
        sum0 += q5kBlockDot(x, @intFromPtr(w) + row_base * row_bytes + blk * q5_k_block_size, k, base_col);

        // Row 1 (if active)
        if (nr_active > 1)
            sum1 += q5kBlockDot(x, @intFromPtr(w) + (row_base + 1) * row_bytes + blk * q5_k_block_size, k, base_col);
    }

    sum0 = cu.blockReduceAdd(sum0);
    if (tid == 0) y[row_base] = sum0;

    if (nr_active > 1) {
        cu.syncthreads();
        sum1 = cu.blockReduceAdd(sum1);
        if (tid == 0) y[row_base + 1] = sum1;
    }
}
