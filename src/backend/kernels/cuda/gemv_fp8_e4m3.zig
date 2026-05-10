//! CUDA GEMV kernel for FP8 E4M3 format.
//! 1:1 mapping (1 FP8 byte → 1 f32 value) with 256-entry LUT conversion.
//! Launch with n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

/// FP8 E4M3 denormal scale: 2^(-6) / 8 = 2^(-9).
const fp8_e4m3_denorm_scale: f32 = 1.0 / 512.0;

/// Compute FP8 E4M3 → f32 conversion at comptime.
/// Bit layout: seeeemmm. No infinities; e=15,m=7 is NaN.
fn fp8e4m3Compute(val: u8) f32 {
    const sign: u32 = @as(u32, val >> 7) << 31;
    const exp: u32 = (val >> 3) & 0x0F;
    const mant: u32 = val & 0x07;

    if (exp == 0x0F and mant == 0x07) {
        return @bitCast(sign | 0x7FC00000); // NaN
    }

    if (exp == 0) {
        if (mant == 0) return @bitCast(sign); // +/- 0
        const fmant: f32 = @floatFromInt(mant);
        const val_abs: f32 = fmant * fp8_e4m3_denorm_scale;
        return @bitCast(sign | @as(u32, @bitCast(val_abs)));
    }

    // Normal: value = (-1)^s * 2^(e-7) * (1 + m/8)
    const exp_f32: u32 = (exp + 120) << 23;
    const mant_f32: u32 = mant << 20;
    return @bitCast(sign | exp_f32 | mant_f32);
}

/// Precomputed FP8 E4M3 → f32 lookup table (256 entries, built at comptime).
const fp8e4m3_lut = blk: {
    var table: [256]f32 = undefined;
    for (0..256) |i| table[i] = fp8e4m3Compute(@intCast(i));
    break :blk table;
};

/// Convert FP8 E4M3 to f32 via lookup table.
inline fn fp8e4m3ToF32(val: u8) f32 {
    return fp8e4m3_lut[val];
}

/// FP8 E4M3 GEMV kernel: y[row] = dot(W[row,:], x)
/// Simple 1:1 element-wise conversion and accumulation.
export fn gemv_fp8_e4m3_kernel(
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
        const wval = fp8e4m3ToF32(w[row_offset + j]);
        sum += wval * x[j];
    }

    // Block reduction (warp + inter-warp)
    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
