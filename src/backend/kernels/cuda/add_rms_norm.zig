//! Fused add + RMS norm kernel:
//!   a[i] = a[i] + b[i]
//!   output[i] = a[i] * weight[i] * rsqrt(mean(a^2) + eps)
//! Launch with 1 block of 256 threads.

const cu = @import("common.zig");

export fn add_rms_norm_kernel(
    a: [*]f32,
    b: [*]const f32,
    weight: [*]const f32,
    output: [*]f32,
    n: u32,
    eps: f32,
) callconv(.kernel) void {
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    // Phase 1: sum of squares of (a + b)
    var sum_sq: f32 = 0.0;
    var i = tid;
    while (i < n) : (i += bdim) {
        const v = a[i] + b[i];
        sum_sq += v * v;
    }

    // Block reduction
    sum_sq = cu.blockReduceAdd(sum_sq);

    // Broadcast scale factor
    if (tid == 0) cu.sharedStore(0, cu.rsqrtf(sum_sq / @as(f32, @floatFromInt(n)) + eps));
    cu.syncthreads();
    const scale = cu.sharedLoad(0);

    // Phase 2: write a = a+b and output = normalized
    i = tid;
    while (i < n) : (i += bdim) {
        const v = a[i] + b[i];
        a[i] = v;
        output[i] = v * weight[i] * scale;
    }
}
