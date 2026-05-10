//! Batched RMS Norm: normalize each of n_tok rows independently.
//! Launch with n_tok blocks of 256 threads.
//! Each block normalizes one row of dim elements, sharing the same weight.

const cu = @import("common.zig");

export fn rms_norm_batched_kernel(input: [*]const f32, weight: [*]const f32, output: [*]f32, n_tok: u32, dim: u32, eps: f32) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n_tok) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const off = row * dim;

    // Sum of squares
    var sum_sq: f32 = 0.0;
    var i = tid;
    while (i < dim) : (i += bdim) {
        const v = input[off + i];
        sum_sq += v * v;
    }
    sum_sq = cu.blockReduceAdd(sum_sq);

    // Broadcast scale
    if (tid == 0) cu.sharedStore(0, cu.rsqrtf(sum_sq / @as(f32, @floatFromInt(dim)) + eps));
    cu.syncthreads();
    const scale = cu.sharedLoad(0);

    // Normalize
    i = tid;
    while (i < dim) : (i += bdim) {
        output[off + i] = input[off + i] * weight[i] * scale;
    }
}
