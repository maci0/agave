//! CUDA GEMV kernel for FP8 E5M2 format.
//! 1:1 mapping (1 FP8 byte → 1 f32 value) with 256-entry LUT conversion.
//! Launch with n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

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
        const wval = cu.fp8e5m2ToF32(w[row_offset + j]);
        sum += wval * x[j];
    }

    // Block reduction (warp + inter-warp)
    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
