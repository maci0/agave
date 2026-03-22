//! Softmax kernel: data[i] = exp(data[i] - max) / sum(exp(data - max))
//! Launch with 1 workgroup of 256 threads, in-place.

const cu = @import("common.zig");

export fn softmax_kernel(data: [*]f32, n: u32) callconv(.kernel) void {
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    // Phase 1: find max
    var max_val: f32 = cu.neg_f32_max;
    var i = tid;
    while (i < n) : (i += bdim) {
        max_val = @max(max_val, data[i]);
    }
    max_val = cu.blockReduceMax(max_val);
    if (tid == 0) cu.sharedStore(0, max_val);
    cu.syncthreads();
    max_val = cu.sharedLoad(0);

    // Phase 2: exp and sum
    var exp_sum: f32 = 0.0;
    i = tid;
    while (i < n) : (i += bdim) {
        const e = cu.expf(data[i] - max_val);
        data[i] = e;
        exp_sum += e;
    }
    cu.syncthreads();

    exp_sum = cu.blockReduceAdd(exp_sum);
    if (tid == 0) cu.sharedStore(0, cu.rcpf(exp_sum));
    cu.syncthreads();
    const inv_sum = cu.sharedLoad(0);

    // Phase 3: normalize
    i = tid;
    while (i < n) : (i += bdim) {
        data[i] *= inv_sum;
    }
}
