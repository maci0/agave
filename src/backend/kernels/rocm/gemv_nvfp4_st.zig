//! NVFP4 SafeTensors GEMV kernel for ROCm.
//! FP4 E2M1 weights (2 per byte) with FP8 E4M3 per-16-element block scales.
//! Grid: n blocks of 256 threads (1 workgroup per output row).

const cu = @import("common.zig");

const mxfp4_table = [16]f32{
    0.0, 0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

fn fp8e4m3ToF32(val: u8) f32 {
    const sign: u32 = @as(u32, val >> 7) << 31;
    const exp: u32 = (val >> 3) & 0xF;
    const mant: u32 = val & 0x7;
    if (exp == 0) {
        if (mant == 0) return @bitCast(sign);
        const fmant: f32 = @floatFromInt(mant);
        const val_abs: f32 = fmant / 8.0 * @as(f32, @bitCast(@as(u32, (127 - 6) << 23)));
        return @bitCast(sign | @as(u32, @bitCast(val_abs)));
    }
    if (exp == 15) return 0.0;
    const exp_f32: u32 = (exp + 127 - 7) << 23;
    const mant_f32: u32 = mant << (23 - 3);
    return @bitCast(sign | exp_f32 | mant_f32);
}

export fn gemv_nvfp4_st_kernel(x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    const tid = cu.threadIdx();
    if (row >= n) return;

    const bytes_per_row = k / 2;
    const scales_per_row = k / 16;

    var sum: f32 = 0.0;
    var g: u32 = tid;
    while (g < scales_per_row) : (g += 256) {
        const sc = fp8e4m3ToF32(scale[row * scales_per_row + g]);
        const base = g * 16;
        const w_off = row * bytes_per_row + g * 8;
        for (0..8) |j| {
            const byte = weight[w_off + j];
            const v0 = mxfp4_table[byte & 0xF] * sc;
            const v1 = mxfp4_table[byte >> 4] * sc;
            sum += v0 * x[base + 2 * j] + v1 * x[base + 2 * j + 1];
        }
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
