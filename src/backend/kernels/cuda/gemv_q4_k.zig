//! CUDA GEMV kernel for Q4_K quantization.
//! 256 values per super-block, 144 bytes: d(f16) + dmin(f16) + scales[12] + qs[128].
//! Launch with n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

const f16tof32 = cu.f16tof32;
const getScaleMinK4 = cu.getScaleMinK4;

const bytes_per_block: usize = 144;
const values_per_block: usize = 256;

/// Q4_K GEMV kernel: y[row] = dot(W[row,:], x)
/// Each thread processes a subset of blocks, accumulates per-sub-block,
/// then reduces via warp + shared memory to produce y[row].
export fn gemv_q4_k_kernel(
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
        const qs = bp + 16;
        const block_start = b * values_per_block;

        // Process 4 groups of 64 values: low nibbles (32 elems) then high nibbles (32 elems)
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

            // Low nibbles: elements gi_lo..gi_lo+31
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

            // High nibbles: elements gi_lo+32..gi_lo+63
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
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
