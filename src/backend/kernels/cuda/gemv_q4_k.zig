//! CUDA GEMV kernel for Q4_K quantization.
//! 256 values per super-block, 144 bytes: d(f16) + dmin(f16) + scales[12] + qs[128].
//! Launch with n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

const bytes_per_block: usize = 144;
const values_per_block: usize = 256;

/// Helper: convert little-endian f16 (2 bytes) to f32
inline fn f16tof32(ptr: [*]const u8) f32 {
    const val = @as(u16, ptr[0]) | (@as(u16, ptr[1]) << 8);
    const sign: u32 = @as(u32, val >> 15) << 31;
    const exp_f16: u32 = (val >> 10) & 0x1F;
    const mant_f16: u32 = val & 0x3FF;

    // Zero
    if (exp_f16 == 0 and mant_f16 == 0) return @bitCast(sign);

    // Denormal (simplified: treat as tiny normal)
    if (exp_f16 == 0) {
        const mant_f32 = mant_f16 << 13;
        const exp_f32: u32 = (127 - 15) << 23;
        return @bitCast(sign | exp_f32 | mant_f32);
    }

    // Inf/NaN
    if (exp_f16 == 0x1F) {
        const exp_f32: u32 = 0xFF << 23;
        const mant_f32: u32 = mant_f16 << 13;
        return @bitCast(sign | exp_f32 | mant_f32);
    }

    // Normal: exp_f32 = exp_f16 + (127 - 15), mant_f32 = mant_f16 << 13
    const exp_f32: u32 = (exp_f16 + (127 - 15)) << 23;
    const mant_f32: u32 = mant_f16 << 13;
    return @bitCast(sign | exp_f32 | mant_f32);
}

/// Helper: extract packed scale and min for Q4_K sub-block.
/// Port of quant.getScaleMinK4() for GPU.
/// Scales are packed in 12 bytes for 8 sub-blocks (6 bits each).
inline fn getScaleMinK4(sb: u32, scales_ptr: [*]const u8, sc: *u8, m: *u8) void {
    if (sb < 4) {
        sc.* = scales_ptr[sb] & 63;
        m.* = scales_ptr[sb + 4] & 63;
    } else {
        sc.* = (scales_ptr[sb + 4] & 0xF) | ((scales_ptr[sb - 4] >> 6) << 4);
        m.* = (scales_ptr[sb + 4] >> 4) | ((scales_ptr[sb] >> 6) << 4);
    }
}

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

        // Process 8 sub-blocks of 32 values each
        var sb: u32 = 0;
        while (sb < 8) : (sb += 1) {
            const gi_base = block_start + sb * 32;
            if (gi_base >= k) break;

            var sc: u8 = undefined;
            var m: u8 = undefined;
            getScaleMinK4(sb, scales, &sc, &m);

            const d_sc = d * @as(f32, @floatFromInt(sc));
            const dm_m = dmin * @as(f32, @floatFromInt(m));

            // Accumulate dot product for sub-block using factored form:
            // dot(x, d*q - dm) = d*sc*dot(x, q) - dm*m*sum(x)
            var q_dot: f32 = 0.0;
            var x_sum: f32 = 0.0;
            const qi_base = sb * 16; // 32 values = 16 bytes (2 nibbles per byte)

            for (0..32) |l| {
                const gi = gi_base + l;
                if (gi >= k) break;
                const byte_idx = qi_base + l / 2;
                const nibble: f32 = @floatFromInt(
                    if (l % 2 == 0) qs[byte_idx] & 0x0F else qs[byte_idx] >> 4
                );
                q_dot += x[gi] * nibble;
                x_sum += x[gi];
            }

            // Apply scale and min correction
            sum += d_sc * q_dot - dm_m * x_sum;
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
