//! L2 Norm kernel: x[i] = x[i] / sqrt(sum(x^2) + eps)
//! Launch with 1 block of 256 threads, in-place.

const cu = @import("common.zig");

export fn l2_norm_kernel(data: [*]f32, n: u32, eps: f32) callconv(.kernel) void {
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    // Phase 1: sum of squares
    var sum_sq: f32 = 0.0;
    var i = tid;
    while (i < n) : (i += bdim) {
        const v = data[i];
        sum_sq += v * v;
    }

    sum_sq = cu.blockReduceAdd(sum_sq);
    if (tid == 0) cu.sharedStore(0, cu.rsqrtf(sum_sq + eps));
    cu.syncthreads();
    const scale = cu.sharedLoad(0);

    // Phase 2: normalize
    i = tid;
    while (i < n) : (i += bdim) {
        data[i] *= scale;
    }
}
