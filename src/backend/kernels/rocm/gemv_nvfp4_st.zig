//! NVFP4 SafeTensors GEMV kernel for ROCm.
//! FP4 E2M1 weights (2 per byte) with FP8 E4M3 per-16-element block scales.
//! Grid: n blocks of 256 threads (1 workgroup per output row).

const cu = @import("common.zig");

const e2m1_lut = cu.e2m1_lut;

export fn gemv_nvfp4_st_kernel(x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    const tid = cu.threadIdx();
    if (row >= n) return;

    const bytes_per_row = k / 2;
    const scales_per_row = k / 16;

    var sum: f32 = 0.0;
    var g: u32 = tid;
    while (g < scales_per_row) : (g += cu.block_dim) {
        const sc = cu.fp8e4m3ToF32(scale[row * scales_per_row + g]);
        const base = g * 16;
        const w_off = row * bytes_per_row + g * 8;
        for (0..8) |j| {
            const byte = weight[w_off + j];
            const v0 = e2m1_lut[byte & 0xF] * sc;
            const v1 = e2m1_lut[byte >> 4] * sc;
            sum += v0 * x[base + 2 * j] + v1 * x[base + 2 * j + 1];
        }
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
