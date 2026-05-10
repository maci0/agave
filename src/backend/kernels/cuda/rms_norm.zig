//! RMS Norm kernel: out[i] = input[i] * weight[i] * rsqrt(mean(x^2) + eps)
//! Launch with 1 block of 256 threads per vector.

const cu = @import("common.zig");

export fn rms_norm_kernel(input: [*]const f32, weight: [*]const f32, output: [*]f32, n: u32, eps: f32) callconv(.kernel) void {
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    // Phase 1: partial sum of squares
    var sum_sq: f32 = 0.0;
    var i = tid;
    while (i < n) : (i += bdim) {
        const v = input[i];
        sum_sq += v * v;
    }

    // Phase 2: block reduction
    sum_sq = cu.blockReduceAdd(sum_sq);

    // Broadcast scale factor
    if (tid == 0) cu.sharedStore(0, cu.rsqrtf(sum_sq / @as(f32, @floatFromInt(n)) + eps));
    cu.syncthreads();
    const scale = cu.sharedLoad(0);

    // Phase 3: normalize
    i = tid;
    while (i < n) : (i += bdim) {
        output[i] = input[i] * weight[i] * scale;
    }
}
