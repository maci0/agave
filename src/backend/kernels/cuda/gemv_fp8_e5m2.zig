//! CUDA GEMV kernel for FP8 E5M2 format.
//! 1:1 mapping (1 FP8 byte → 1 f32 value) with 256-entry LUT conversion.
//! Launch with n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

/// FP8 E5M2 denormal scale: 2^(-14) / 4 = 2^(-16).
const fp8_e5m2_denorm_scale: f32 = 1.0 / 65536.0;

/// Compute FP8 E5M2 → f32 conversion at comptime.
/// Bit layout: seeeeemm. Has infinities (e=31,m=0) and NaN (e=31,m!=0).
fn fp8e5m2Compute(val: u8) f32 {
    const sign: u32 = @as(u32, val >> 7) << 31;
    const exp: u32 = (val >> 2) & 0x1F;
    const mant: u32 = val & 0x03;

    if (exp == 0x1F) {
        if (mant == 0) return @bitCast(sign | 0x7F800000); // Infinity
        return @bitCast(sign | 0x7FC00000); // NaN
    }

    if (exp == 0) {
        if (mant == 0) return @bitCast(sign); // +/- 0
        const fmant: f32 = @floatFromInt(mant);
        const val_abs: f32 = fmant * fp8_e5m2_denorm_scale;
        return @bitCast(sign | @as(u32, @bitCast(val_abs)));
    }

    // Normal: value = (-1)^s * 2^(e-15) * (1 + m/4)
    const exp_f32: u32 = (exp + 112) << 23;
    const mant_f32: u32 = mant << 21;
    return @bitCast(sign | exp_f32 | mant_f32);
}

/// Precomputed FP8 E5M2 → f32 lookup table (256 entries, built at comptime).
const fp8e5m2_lut = blk: {
    var table: [256]f32 = undefined;
    for (0..256) |i| table[i] = fp8e5m2Compute(@intCast(i));
    break :blk table;
};

/// Convert FP8 E5M2 to f32 via lookup table.
inline fn fp8e5m2ToF32(val: u8) f32 {
    return fp8e5m2_lut[val];
}

/// FP8 E5M2 GEMV kernel: y[row] = dot(W[row,:], x)
/// Simple 1:1 element-wise conversion and accumulation.
export fn gemv_fp8_e5m2_kernel(
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

    const row_offset = row * k;

    var sum: f32 = 0.0;
    var j = tid;
    while (j < k) : (j += bdim) {
        const wval = fp8e5m2ToF32(w[row_offset + j]);
        sum += wval * x[j];
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
