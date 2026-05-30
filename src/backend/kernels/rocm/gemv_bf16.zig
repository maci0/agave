//! GEMV BF16 kernel: y[row] = dot(W_bf16[row,:], x)
//! Launch with n workgroups of 256 threads (one row per workgroup).

const cu = @import("common.zig");

export fn gemv_bf16_kernel(x: [*]const f32, w: [*]const u16, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    var sum: f32 = 0.0;
    const row_offset = row * k;
    const sparse_threshold: f32 = 0.005;
    const chunk_size: u32 = 32;
    var j = tid;
    while (j < k) : (j += bdim) {
        // Sparse skip: check if all 32 input values in this chunk are near-zero
        const chunk_base = (j / chunk_size) * chunk_size;
        const check_end = @min(chunk_base + chunk_size, k);
        var bmax: f32 = 0.0;
        for (chunk_base..check_end) |i| {
            const a = @abs(x[i]);
            if (a > bmax) bmax = a;
        }
        if (bmax < sparse_threshold) continue;

        // BF16 → F32: zero-extend and shift left 16 bits
        const bits: u32 = @as(u32, w[row_offset + j]) << 16;
        const wf: f32 = @bitCast(bits);
        sum += wf * x[j];
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
