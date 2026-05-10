//! GEMV BF16 kernel: y[row] = dot(W_bf16[row,:], x)
//! Launch with n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

export fn gemv_bf16_kernel(x: [*]const f32, w: [*]const u16, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    var sum: f32 = 0.0;
    const row_offset = row * k;
    var j = tid;
    while (j < k) : (j += bdim) {
        // BF16 → F32: zero-extend and shift left 16 bits
        const bits: u32 = @as(u32, w[row_offset + j]) << 16;
        const wf: f32 = @bitCast(bits);
        sum += wf * x[j];
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
