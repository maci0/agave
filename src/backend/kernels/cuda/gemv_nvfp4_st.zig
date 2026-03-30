//! NVFP4 SafeTensors GEMV: y[row] = dot(dequant(W[row,:]), x)
//! Weights: packed nibble pairs (4 bits each, 2 per byte), group_size=16.
//! Scales: FP8 E4M3 per group (1 byte each).
//! Dequant: float_val = mxfp4_lut[nibble] * fp8_scale.
//! Grid: n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

/// E2M1 FP4 → float lookup (OCP Microscaling Spec).
const e2m1_lut = [16]f32{
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

/// FP8 E4M3 denormal scale: 2^(-9).
const fp8_e4m3_denorm_scale: f32 = 1.0 / 512.0;

/// Convert FP8 E4M3 byte to f32.
fn fp8e4m3ToF32(val: u8) f32 {
    const sign: u32 = @as(u32, val >> 7) << 31;
    const exp: u32 = (val >> 3) & 0x0F;
    const mant: u32 = val & 0x07;

    if (exp == 0x0F and mant == 0x07) return @bitCast(sign | @as(u32, 0x7FC00000)); // NaN
    if (exp == 0) {
        if (mant == 0) return @bitCast(sign);
        const fmant: f32 = @floatFromInt(mant);
        const val_abs: f32 = fmant * fp8_e4m3_denorm_scale;
        return @bitCast(sign | @as(u32, @bitCast(val_abs)));
    }
    return @bitCast(sign | ((exp + 120) << 23) | (mant << 20));
}

export fn gemv_nvfp4_st_kernel(
    x: [*]const f32,
    w: [*]const u8,
    s: [*]const u8,
    y: [*]f32,
    n: u32,
    k: u32,
) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const bytes_per_row = k / 2;
    const groups_per_row = k / 16;

    var sum: f32 = 0.0;
    var g: u32 = tid;
    while (g < groups_per_row) : (g += bdim) {
        const scale = fp8e4m3ToF32(s[row * groups_per_row + g]);
        const w_base = row * bytes_per_row + g * 8;
        const x_base = g * 16;

        var gdot: f32 = 0.0;
        var j: u32 = 0;
        while (j < 8) : (j += 1) {
            const byte = w[w_base + j];
            const lo = e2m1_lut[byte & 0xF];
            const hi = e2m1_lut[byte >> 4];
            gdot += lo * x[x_base + 2 * j] + hi * x[x_base + 2 * j + 1];
        }
        sum += scale * gdot;
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
