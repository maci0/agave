//! CUDA GEMV kernel for Q5_K quantization.
//! 256 values per super-block, 176 bytes: d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128].
//! Launch with n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

const f16tof32 = cu.f16tof32;
const getScaleMinK4 = cu.getScaleMinK4;

const bytes_per_block: usize = 176;
const values_per_block: usize = 256;

/// Q5_K high-bit contribution: the 5th bit adds 2^4 = 16 to the value.
const q5_k_high_bit_value: f32 = 16.0;

/// Q5_K GEMV kernel: y[row] = dot(W[row,:], x)
/// Q5_K uses 5-bit quantization: 4-bit low nibble + 1-bit high bit stored separately.
/// 4 groups of 64 elements each, 2 sub-blocks per group.
export fn gemv_q5_k_kernel(
    x: [*]const f32,
    w: [*]const u8,
    y: [*]f32,
    n: u32,
    k: u32,
) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const num_blocks = (k + values_per_block - 1) / values_per_block;
    const row_ptr = w + row * num_blocks * bytes_per_block;

    var sum: f32 = 0.0;

    // Each thread processes a subset of super-blocks
    var b = tid;
    while (b < num_blocks) : (b += bdim) {
        const bp = row_ptr + b * bytes_per_block;
        const d = f16tof32(bp);
        const dmin = f16tof32(bp + 2);
        const scales = bp + 4;
        const qh = bp + 16; // High bits for 5-bit values
        const qs = bp + 48; // Low 4-bit nibbles
        const block_start = b * values_per_block;

        // 4 groups of 64 elements each
        var group: u32 = 0;
        while (group < 4) : (group += 1) {
            const j = group * 64;
            const is = group * 2; // Scale index base
            const shift: u3 = @intCast(group * 2);
            const umask1: u8 = @as(u8, 1) << shift;
            const umask2: u8 = @as(u8, 2) << shift;
            const ql_off = group * 32; // Byte offset in qs

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

            // First half: l=[0..31], low nibble, scale pair (sc_a, m_a)
            for (0..32) |l| {
                const gi = gi_base + l;
                if (gi >= k) break;
                const lo: f32 = @floatFromInt(qs[ql_off + l] & 0x0F);
                const hi: f32 = if ((qh[l] & umask1) != 0) q5_k_high_bit_value else 0.0;
                const qv = lo + hi;
                sum += x[gi] * (d_sc_a * qv - dm_m_a);
            }

            // Second half: l=[0..31], high nibble, scale pair (sc_b, m_b)
            for (0..32) |l| {
                const gi = gi_base + 32 + l;
                if (gi >= k) break;
                const lo: f32 = @floatFromInt(qs[ql_off + l] >> 4);
                const hi: f32 = if ((qh[l] & umask2) != 0) q5_k_high_bit_value else 0.0;
                const qv = lo + hi;
                sum += x[gi] * (d_sc_b * qv - dm_m_b);
            }
        }
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
