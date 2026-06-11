//! Fused RMS norm + accumulate kernel:
//!   b[i] += a[i] * weight[i] * rsqrt(mean(a^2) + eps)
//! Launch with 1 block of 256 threads.

const cu = @import("common.zig");

export fn rms_norm_add_kernel(
    a: [*]const f32,
    weight: [*]const f32,
    b: [*]f32,
    n: u32,
    eps: f32,
) callconv(.kernel) void {
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    // Phase 1: sum of squares of a
    var sum_sq: f32 = 0.0;
    var i = tid;
    while (i < n) : (i += bdim) sum_sq += a[i] * a[i];

    // Block reduction
    sum_sq = cu.blockReduceAdd(sum_sq);

    if (tid == 0) cu.sharedStore(0, cu.rsqrtf(sum_sq / @as(f32, @floatFromInt(n)) + eps));
    cu.syncthreads();
    const scale = cu.sharedLoad(0);

    // Phase 2: accumulate normalized a into b
    i = tid;
    while (i < n) : (i += bdim) {
        b[i] += a[i] * weight[i] * scale;
    }
}

const std = @import("std");

test "constants valid" {
    _ = @sizeOf(u8);
}
