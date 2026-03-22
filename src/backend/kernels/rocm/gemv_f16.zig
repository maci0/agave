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
    var j = tid;
    while (j < k) : (j += bdim) {
        sum += @as(f32, @floatCast(w[row_offset + j])) * x[j];
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
