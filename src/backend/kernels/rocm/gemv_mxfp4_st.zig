//! MXFP4 SafeTensors GEMV kernel for ROCm.
//! FP4 E2M1 weights (2 per byte) with E8M0 per-32-element block scales.
//! Grid: n blocks of 256 threads (1 workgroup per output row).

const cu = @import("common.zig");

const mxfp4_table = [16]f32{
    0.0, 0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

fn e8m0ToF32(e: u8) f32 {
    if (e == 0) return 0.0;
    return @bitCast(@as(u32, e) << 23);
}

export fn gemv_mxfp4_st_kernel(x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    const tid = cu.threadIdx();
    if (row >= n) return;

    const blocks_per_row = k / 32;
    const bytes_per_row = k / 2;

    var sum: f32 = 0.0;
    var blk: u32 = tid;
    while (blk < blocks_per_row) : (blk += 256) {
        const sc = e8m0ToF32(scale[row * blocks_per_row + blk]);
        const base = blk * 32;
        const w_off = row * bytes_per_row + blk * 16;
        for (0..16) |j| {
            const byte = weight[w_off + j];
            const v0 = mxfp4_table[byte & 0xF] * sc;
            const v1 = mxfp4_table[byte >> 4] * sc;
            sum += v0 * x[base + 2 * j] + v1 * x[base + 2 * j + 1];
        }
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
