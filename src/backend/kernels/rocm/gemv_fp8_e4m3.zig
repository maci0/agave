//! GEMV FP8 E4M3 kernel: y[row] = dot(W_fp8[row,:], x)
//! FP8 E4M3: 1 byte per element.
//! Uses 256-entry comptime LUT for branch-free dequantization.

const cu = @import("common.zig");

export fn gemv_fp8_e4m3_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const row_offset = row * k;

    var sum: f32 = 0.0;

    var col = tid;
    while (col < k) : (col += bdim) {
        const wval = cu.fp8e4m3ToF32(w[row_offset + col]);
        sum += wval * x[col];
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
