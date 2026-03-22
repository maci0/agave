//! Batched GEMV Q4_0 kernel: compute up to 3 independent matrix-vector products
//! sharing the same input vector x. Used to fuse QKV projections (3 ops) and
//! Gate+Up projections (2 ops, n2=0) into a single kernel launch.
//!
//! Launch with (n0 + n1 + n2) blocks of 256 threads.

const cu = @import("common.zig");

/// Bytes per Q4_0 block (32 elements).
const q4_0_block_size: u32 = 18;
/// Elements per Q4_0 block.
const q4_0_group_size: u32 = 32;

export fn gemv_q4_0_batch_kernel(
    x: [*]const f32,
    w0: [*]const u8,
    y0: [*]f32,
    n0: u32,
    w1: [*]const u8,
    y1: [*]f32,
    n1: u32,
    w2: [*]const u8,
    y2: [*]f32,
    n2: u32,
    k: u32,
) callconv(.kernel) void {
    const global_row = cu.blockIdx();
    const total = n0 + n1 + n2;
    if (global_row >= total) return;

    // Select weight matrix and output based on block index
    var w: [*]const u8 = undefined;
    var y: [*]f32 = undefined;
    var row: u32 = undefined;

    if (global_row < n0) {
        w = w0;
        y = y0;
        row = global_row;
    } else if (global_row < n0 + n1) {
        w = w1;
        y = y1;
        row = global_row - n0;
    } else {
        w = w2;
        y = y2;
        row = global_row - n0 - n1;
    }

    // Standard Q4_0 dot product (identical to gemv_q4_0_kernel)
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const blocks_per_row = (k + q4_0_group_size - 1) / q4_0_group_size;
    const row_bytes = blocks_per_row * q4_0_block_size;

    var sum: f32 = 0.0;
    var blk = tid;
    while (blk < blocks_per_row) : (blk += bdim) {
        const block_ptr = w + row * row_bytes + blk * q4_0_block_size;

        const scale_bits = @as(u16, block_ptr[0]) | (@as(u16, block_ptr[1]) << 8);
        const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));

        const quants = block_ptr + 2;
        const base_col = blk * q4_0_group_size;

        var blk_sum: f32 = 0.0;
        for (0..16) |qi| {
            const byte = quants[qi];
            const lo = @as(i8, @intCast(@as(u4, @truncate(byte)))) - 8;
            const hi = @as(i8, @intCast(@as(u4, @truncate(byte >> 4)))) - 8;

            if (base_col + qi < k)
                blk_sum += @as(f32, @floatFromInt(lo)) * x[base_col + qi];
            if (base_col + qi + 16 < k)
                blk_sum += @as(f32, @floatFromInt(hi)) * x[base_col + qi + 16];
        }
        sum += scale * blk_sum;
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
