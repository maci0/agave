//! GEMV Q4_K kernel: y[row] = dot(W_q4k[row,:], x)
//! Q4_K block: 144 bytes = d(f16) + dmin(f16) + scales[12] + qs[128].
//! 256 values per super-block, 4 groups of 64 values (low/high nibbles).
//! NR=2: Launch with ceil(n/2) workgroups, each processes 2 output rows.

const cu = @import("common.zig");

const getScaleMinK4 = cu.getScaleMinK4;

/// Bytes per Q4_K super-block (256 elements).
const q4_k_block_size: u32 = 144;
/// Elements per Q4_K super-block.
const q4_k_group_size: u32 = 256;
/// Number of output rows per workgroup.
const nr: u32 = 2;

export fn gemv_q4_k_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row_base = cu.blockIdx() * nr;
    if (row_base >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const nr_active = @min(nr, n - row_base);

    const blocks_per_row = (k + q4_k_group_size - 1) / q4_k_group_size;
    const row_bytes = blocks_per_row * q4_k_block_size;

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;

    var blk = tid;
    while (blk < blocks_per_row) : (blk += bdim) {
        const base_col = blk * q4_k_group_size;

        // Row 0
        {
            const blk_addr = @intFromPtr(w) + row_base * row_bytes + blk * q4_k_block_size;
            const d: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrFromInt(blk_addr)).*)));
            const dmin: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrFromInt(blk_addr + 2)).*)));
            const scales = @as([*]const u8, @ptrFromInt(blk_addr + 4));
            const qs = @as([*]const u8, @ptrFromInt(blk_addr + 16));

            for (0..4) |g| {
                const gi_lo = base_col + g * 64;
                if (gi_lo >= k) break;
                const ql_off = g * 32;
                var sc_lo: u8 = undefined;
                var m_lo: u8 = undefined;
                var sc_hi: u8 = undefined;
                var m_hi: u8 = undefined;
                getScaleMinK4(g * 2, scales, &sc_lo, &m_lo);
                getScaleMinK4(g * 2 + 1, scales, &sc_hi, &m_hi);
                {
                    const d_sc = d * @as(f32, @floatFromInt(sc_lo));
                    const dm_m = dmin * @as(f32, @floatFromInt(m_lo));
                    var q_dot: f32 = 0.0;
                    var x_sum: f32 = 0.0;
                    for (0..32) |l| {
                        const gi = gi_lo + l;
                        if (gi >= k) break;
                        q_dot += x[gi] * @as(f32, @floatFromInt(qs[ql_off + l] & 0x0F));
                        x_sum += x[gi];
                    }
                    sum0 += d_sc * q_dot - dm_m * x_sum;
                }
                {
                    const d_sc = d * @as(f32, @floatFromInt(sc_hi));
                    const dm_m = dmin * @as(f32, @floatFromInt(m_hi));
                    var q_dot: f32 = 0.0;
                    var x_sum: f32 = 0.0;
                    for (0..32) |l| {
                        const gi = gi_lo + 32 + l;
                        if (gi >= k) break;
                        q_dot += x[gi] * @as(f32, @floatFromInt(qs[ql_off + l] >> 4));
                        x_sum += x[gi];
                    }
                    sum0 += d_sc * q_dot - dm_m * x_sum;
                }
            }
        }

        // Row 1 (if active)
        if (nr_active > 1) {
            const blk_addr = @intFromPtr(w) + (row_base + 1) * row_bytes + blk * q4_k_block_size;
            const d: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrFromInt(blk_addr)).*)));
            const dmin: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrFromInt(blk_addr + 2)).*)));
            const scales = @as([*]const u8, @ptrFromInt(blk_addr + 4));
            const qs = @as([*]const u8, @ptrFromInt(blk_addr + 16));

            for (0..4) |g| {
                const gi_lo = base_col + g * 64;
                if (gi_lo >= k) break;
                const ql_off = g * 32;
                var sc_lo: u8 = undefined;
                var m_lo: u8 = undefined;
                var sc_hi: u8 = undefined;
                var m_hi: u8 = undefined;
                getScaleMinK4(g * 2, scales, &sc_lo, &m_lo);
                getScaleMinK4(g * 2 + 1, scales, &sc_hi, &m_hi);
                {
                    const d_sc = d * @as(f32, @floatFromInt(sc_lo));
                    const dm_m = dmin * @as(f32, @floatFromInt(m_lo));
                    var q_dot: f32 = 0.0;
                    var x_sum: f32 = 0.0;
                    for (0..32) |l| {
                        const gi = gi_lo + l;
                        if (gi >= k) break;
                        q_dot += x[gi] * @as(f32, @floatFromInt(qs[ql_off + l] & 0x0F));
                        x_sum += x[gi];
                    }
                    sum1 += d_sc * q_dot - dm_m * x_sum;
                }
                {
                    const d_sc = d * @as(f32, @floatFromInt(sc_hi));
                    const dm_m = dmin * @as(f32, @floatFromInt(m_hi));
                    var q_dot: f32 = 0.0;
                    var x_sum: f32 = 0.0;
                    for (0..32) |l| {
                        const gi = gi_lo + 32 + l;
                        if (gi >= k) break;
                        q_dot += x[gi] * @as(f32, @floatFromInt(qs[ql_off + l] >> 4));
                        x_sum += x[gi];
                    }
                    sum1 += d_sc * q_dot - dm_m * x_sum;
                }
            }
        }
    }

    sum0 = cu.blockReduceAdd(sum0);
    if (tid == 0) y[row_base] = sum0;

    if (nr_active > 1) {
        cu.syncthreads();
        sum1 = cu.blockReduceAdd(sum1);
        if (tid == 0) y[row_base + 1] = sum1;
    }
}
