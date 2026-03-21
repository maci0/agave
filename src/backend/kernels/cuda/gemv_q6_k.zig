//! CUDA GEMV kernel for Q6_K quantization.
//! 256 values per super-block, 210 bytes: ql[128] + qh[64] + scales[16] + d(f16).
//! Launch with n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

const bytes_per_block: usize = 210;
const values_per_block: usize = 256;

/// Helper: convert little-endian f16 (2 bytes) to f32
inline fn f16tof32(ptr: [*]const u8) f32 {
    const val = @as(u16, ptr[0]) | (@as(u16, ptr[1]) << 8);
    const sign: u32 = @as(u32, val >> 15) << 31;
    const exp_f16: u32 = (val >> 10) & 0x1F;
    const mant_f16: u32 = val & 0x3FF;

    if (exp_f16 == 0 and mant_f16 == 0) return @bitCast(sign);

    if (exp_f16 == 0) {
        const mant_f32 = mant_f16 << 13;
        const exp_f32: u32 = (127 - 15) << 23;
        return @bitCast(sign | exp_f32 | mant_f32);
    }

    if (exp_f16 == 0x1F) {
        const exp_f32: u32 = 0xFF << 23;
        const mant_f32: u32 = mant_f16 << 13;
        return @bitCast(sign | exp_f32 | mant_f32);
    }

    const exp_f32: u32 = (exp_f16 + (127 - 15)) << 23;
    const mant_f32: u32 = mant_f16 << 13;
    return @bitCast(sign | exp_f32 | mant_f32);
}

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
        const d = f16tof32(bp + 208); // d is at the end (bytes 208-209)
        const block_start = b * values_per_block;

        // 2 chunks of 128 elements each
        var chunk: u32 = 0;
        while (chunk < 2) : (chunk += 1) {
            const ql = bp + chunk * 64; // Low 4 bits
            const qh = bp + 128 + chunk * 32; // High 2 bits
            const sc: [*]const i8 = @ptrCast(bp + 192 + chunk * 8); // Scales (i8)
            const base = block_start + chunk * 128;

            // 4 scale groups of 32 elements each
            for (0..32) |l| {
                const is: usize = l / 16; // Scale index (0-3, maps to 4 scale groups)
                const gi0 = base + l;
                const gi1 = base + l + 32;
                const gi2 = base + l + 64;
                const gi3 = base + l + 96;

                // Extract 6-bit values: low 4 bits from ql, high 2 bits from qh
                // q = (ql & 0xF) | ((qh >> shift) & 3) << 4, then subtract 32
                const q1: i8 = @as(i8, @intCast((ql[l] & 0x0F) | ((@as(u8, @truncate(qh[l] >> 0)) & 3) << 4))) - 32;
                const q2: i8 = @as(i8, @intCast((ql[l + 32] & 0x0F) | ((@as(u8, @truncate(qh[l] >> 2)) & 3) << 4))) - 32;
                const q3: i8 = @as(i8, @intCast((ql[l] >> 4) | ((@as(u8, @truncate(qh[l] >> 4)) & 3) << 4))) - 32;
                const q4: i8 = @as(i8, @intCast((ql[l + 32] >> 4) | ((@as(u8, @truncate(qh[l] >> 6)) & 3) << 4))) - 32;

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

    // Warp reduction
    sum = cu.warpReduceAdd(sum);

    const lane = tid % 32;
    const warp_id = tid / 32;
    if (lane == 0) cu.sharedStore(warp_id, sum);
    cu.syncthreads();

    // Inter-warp reduction
    const n_warps = (bdim + 31) / 32;
    var result = if (tid < n_warps) cu.sharedLoad(tid) else 0.0;
    if (warp_id == 0) result = cu.warpReduceAdd(result);

    if (tid == 0) y[row] = result;
}
