//! CUDA GEMV kernel for Q4_K quantization.
//! 256 values per super-block, 144 bytes: d(f16) + dmin(f16) + scales[12] + qs[128].
//! NR=2: Launch with ceil(n/2) blocks, each block processes 2 output rows.
//! The x vector is shared across both rows for cache reuse.

const cu = @import("common.zig");

const f16tof32 = cu.f16tof32;
const getScaleMinK4 = cu.getScaleMinK4;

const bytes_per_block: usize = 144;
const values_per_block: usize = 256;
/// Number of output rows per CUDA block.
const nr: u32 = 2;

/// Compute one super-block's dot product for a single row.
inline fn q4kBlockDot(
    x: [*]const f32,
    bp: [*]const u8,
    k: u32,
    block_start: u32,
) f32 {
    const d = f16tof32(bp);
    const dmin = f16tof32(bp + 2);
    const scales = bp + 4;
    const qs = bp + 16;
    var sum: f32 = 0.0;

    var g: u32 = 0;
    while (g < 4) : (g += 1) {
        const gi_lo = block_start + g * 64;
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
            sum += d_sc * q_dot - dm_m * x_sum;
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
            sum += d_sc * q_dot - dm_m * x_sum;
        }
    }
    return sum;
}

/// Q4_K GEMV kernel: NR=2 rows per block.
/// Each block processes rows [blockIdx*2] and [blockIdx*2+1].
export fn gemv_q4_k_kernel(
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
        sum0 += q4kBlockDot(x, w + (row_base) * num_blocks * bytes_per_block + b * bytes_per_block, k, block_start);
        if (nr_active > 1)
            sum1 += q4kBlockDot(x, w + (row_base + 1) * num_blocks * bytes_per_block + b * bytes_per_block, k, block_start);
    }

    sum0 = cu.blockReduceAdd(sum0);
    if (tid == 0) y[row_base] = sum0;

    if (nr_active > 1) {
        cu.syncthreads();
        sum1 = cu.blockReduceAdd(sum1);
        if (tid == 0) y[row_base + 1] = sum1;
    }
}
