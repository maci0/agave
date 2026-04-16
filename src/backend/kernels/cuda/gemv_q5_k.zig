//! CUDA GEMV kernel for Q5_K quantization.
//! 256 values per super-block, 176 bytes: d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128].
//! NR=2: Launch with ceil(n/2) blocks, each block processes 2 output rows.
//! The x vector is shared across both rows for cache reuse.

const cu = @import("common.zig");

const f16tof32 = cu.f16tof32;
const getScaleMinK4 = cu.getScaleMinK4;

const bytes_per_block: usize = 176;
const values_per_block: usize = 256;
/// Number of output rows per CUDA block.
const nr: u32 = 2;

/// Q5_K high-bit contribution: the 5th bit adds 2^4 = 16 to the value.
const q5_k_high_bit_value: f32 = 16.0;

/// Compute one super-block's dot product for a single row.
inline fn q5kBlockDot(
    x: [*]const f32,
    bp: [*]const u8,
    k: u32,
    block_start: u32,
) f32 {
    const d = f16tof32(bp);
    const dmin = f16tof32(bp + 2);
    const scales = bp + 4;
    const qh = bp + 16;
    const qs = bp + 48;
    var sum: f32 = 0.0;

    var group: u32 = 0;
    while (group < 4) : (group += 1) {
        const j = group * 64;
        const is = group * 2;
        const shift: u3 = @intCast(group * 2);
        const umask1: u8 = @as(u8, 1) << shift;
        const umask2: u8 = @as(u8, 2) << shift;
        const ql_off = group * 32;

        var sc_a: u8 = undefined;
        var m_a: u8 = undefined;
        var sc_b: u8 = undefined;
        var m_b: u8 = undefined;
        getScaleMinK4(is + 0, scales, &sc_a, &m_a);
        getScaleMinK4(is + 1, scales, &sc_b, &m_b);

        const gi_base = block_start + j;
        if (gi_base >= k) break;

        const d_sc_a = d * @as(f32, @floatFromInt(sc_a));
        const dm_m_a = dmin * @as(f32, @floatFromInt(m_a));
        const d_sc_b = d * @as(f32, @floatFromInt(sc_b));
        const dm_m_b = dmin * @as(f32, @floatFromInt(m_b));

        // First half: low nibble + umask1
        for (0..32) |l| {
            const gi = gi_base + l;
            if (gi >= k) break;
            const lo: f32 = @floatFromInt(qs[ql_off + l] & 0x0F);
            const hi: f32 = if ((qh[l] & umask1) != 0) q5_k_high_bit_value else 0.0;
            const qv = lo + hi;
            sum += x[gi] * (d_sc_a * qv - dm_m_a);
        }

        // Second half: high nibble + umask2
        for (0..32) |l| {
            const gi = gi_base + 32 + l;
            if (gi >= k) break;
            const lo: f32 = @floatFromInt(qs[ql_off + l] >> 4);
            const hi: f32 = if ((qh[l] & umask2) != 0) q5_k_high_bit_value else 0.0;
            const qv = lo + hi;
            sum += x[gi] * (d_sc_b * qv - dm_m_b);
        }
    }
    return sum;
}

/// Q5_K GEMV kernel: NR=2 rows per block.
/// Each block processes rows [blockIdx*2] and [blockIdx*2+1].
export fn gemv_q5_k_kernel(
    x: [*]const f32,
    w: [*]const u8,
    y: [*]f32,
    n: u32,
    k: u32,
) callconv(.kernel) void {
    const row_base = cu.blockIdx() * nr;
    if (row_base >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const nr_active = @min(nr, n - row_base);

    const num_blocks = (k + values_per_block - 1) / values_per_block;

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;

    var b = tid;
    while (b < num_blocks) : (b += bdim) {
        const block_start = b * values_per_block;
        sum0 += q5kBlockDot(x, w + (row_base) * num_blocks * bytes_per_block + b * bytes_per_block, k, block_start);
        if (nr_active > 1)
            sum1 += q5kBlockDot(x, w + (row_base + 1) * num_blocks * bytes_per_block + b * bytes_per_block, k, block_start);
    }

    sum0 = cu.blockReduceAdd(sum0);
    if (tid == 0) y[row_base] = sum0;

    if (nr_active > 1) {
        cu.syncthreads();
        sum1 = cu.blockReduceAdd(sum1);
        if (tid == 0) y[row_base + 1] = sum1;
    }
}
