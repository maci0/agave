//! Multi-head RMS Norm kernel: per-head in-place normalization.
//! data[h*hd + i] = data[h*hd + i] * weight[i] * rsqrt(mean(x^2) + eps)
//! Launch with n_heads workgroups of 256 threads.
//! Weight is shared across all heads (same norm weights applied to each head).

const cu = @import("common.zig");

export fn rms_norm_multi_kernel(data: [*]f32, weight: [*]const f32, n_heads: u32, head_dim: u32, eps: f32) callconv(.kernel) void {
    const head = cu.blockIdx();
    if (head >= n_heads) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const offset = head * head_dim;

    // Phase 1: partial sum of squares
    var sum_sq: f32 = 0.0;
    var i = tid;
    while (i < head_dim) : (i += bdim) {
        const v = data[offset + i];
        sum_sq += v * v;
    }

    // Phase 2: block reduction
    sum_sq = cu.blockReduceAdd(sum_sq);

    // Broadcast scale factor via LDS
    if (tid == 0) cu.sharedStore(0, cu.rsqrtf(sum_sq / @as(f32, @floatFromInt(head_dim)) + eps));
    cu.syncthreads();
    const scale = cu.sharedLoad(0);

    // Phase 3: normalize in-place
    i = tid;
    while (i < head_dim) : (i += bdim) {
        data[offset + i] = data[offset + i] * weight[i] * scale;
    }
}
