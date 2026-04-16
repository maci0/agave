//! CUDA GEMV kernel for Q6_K quantization.
//! 256 values per super-block, 210 bytes: ql[128] + qh[64] + scales[16] + d(f16).
//! Launch with n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

const bytes_per_block: usize = 210;
const values_per_block: usize = 256;

// Q6_K block layout offsets.
const q6_k_d_offset: usize = 208; // d(f16) at end of block
const q6_k_qh_offset: usize = 128; // qh starts after ql(128)
const q6_k_sc_offset: usize = 192; // scales start after ql(128) + qh(64)
const q6_k_ql_chunk_bytes: usize = 64;
const q6_k_qh_chunk_bytes: usize = 32;
const q6_k_sc_chunk_bytes: usize = 8;
/// Elements per half super-block chunk (256 / 2).
const chunk_elems = values_per_block / 2;

/// Q6_K dequant bias: 6-bit unsigned [0..63] centered to signed [-32..31].
const q6_k_dequant_bias: i32 = -32;
/// Mask for extracting 2-bit high-order field from qh byte.
const qh_2bit_mask: u8 = 3;

const f16tof32 = cu.f16tof32;

/// Q6_K GEMV kernel: y[row] = dot(W[row,:], x)
/// Q6_K uses 6-bit quantization: 4-bit low nibble + 2-bit high bits stored separately.
/// 2 chunks of 128 elements each, 4 scale groups of 32 elements per chunk.
export fn gemv_q6_k_kernel(
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
        const d = f16tof32(bp + q6_k_d_offset); // d is at the end
        const block_start = b * values_per_block;

        // 2 chunks of 128 elements each
        var chunk: u32 = 0;
        while (chunk < 2) : (chunk += 1) {
            const ql = bp + chunk * q6_k_ql_chunk_bytes; // Low 4 bits
            const qh = bp + q6_k_qh_offset + chunk * q6_k_qh_chunk_bytes; // High 2 bits
            const sc: [*]const i8 = @ptrCast(bp + q6_k_sc_offset + chunk * q6_k_sc_chunk_bytes); // Scales (i8)
            const base = block_start + chunk * chunk_elems;

            // 4 scale groups of 32 elements each
            for (0..32) |l| {
                const is: usize = l / 16; // Scale index (0-3, maps to 4 scale groups)
                const gi0 = base + l;
                const gi1 = base + l + 32;
                const gi2 = base + l + 64;
                const gi3 = base + l + 96;

                // Extract 6-bit values: low 4 bits from ql, high 2 bits from qh
                // q = (ql & 0xF) | ((qh >> shift) & 3) << 4, then subtract 32
                const q1: i8 = @as(i8, @intCast((ql[l] & 0x0F) | ((@as(u8, @truncate(qh[l] >> 0)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                const q2: i8 = @as(i8, @intCast((ql[l + 32] & 0x0F) | ((@as(u8, @truncate(qh[l] >> 2)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                const q3: i8 = @as(i8, @intCast((ql[l] >> 4) | ((@as(u8, @truncate(qh[l] >> 4)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                const q4: i8 = @as(i8, @intCast((ql[l + 32] >> 4) | ((@as(u8, @truncate(qh[l] >> 6)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;

                // Scale and accumulate
                const ds_q1 = d * @as(f32, @floatFromInt(sc[is + 0]));
                const ds_q2 = d * @as(f32, @floatFromInt(sc[is + 2]));
                const ds_q3 = d * @as(f32, @floatFromInt(sc[is + 4]));
                const ds_q4 = d * @as(f32, @floatFromInt(sc[is + 6]));

                if (gi0 < k) sum += x[gi0] * ds_q1 * @as(f32, @floatFromInt(q1));
                if (gi1 < k) sum += x[gi1] * ds_q2 * @as(f32, @floatFromInt(q2));
                if (gi2 < k) sum += x[gi2] * ds_q3 * @as(f32, @floatFromInt(q3));
                if (gi3 < k) sum += x[gi3] * ds_q4 * @as(f32, @floatFromInt(q4));
            }
        }
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
