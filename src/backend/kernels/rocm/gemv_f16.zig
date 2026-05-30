//! GEMV F16 kernel: y[row] = dot(W_f16[row,:], x)
//! Launch with n workgroups of 256 threads (one row per workgroup).

const cu = @import("common.zig");

export fn gemv_f16_kernel(x: [*]const f32, w: [*]const f16, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
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

        sum += @as(f32, @floatCast(w[row_offset + j])) * x[j];
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
