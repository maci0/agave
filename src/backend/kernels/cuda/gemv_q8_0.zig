//! GEMV Q8_0 kernel: y[row] = dot(W_q8[row,:], x)
//! Q8_0 block: 34 bytes = 2 bytes (f16 scale) + 32 bytes (32 × i8 quants).
//! Launch with n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

/// Bytes per Q8_0 block (32 elements).
const q8_0_block_size: u32 = 34;
/// Elements per Q8_0 block.
const q8_0_group_size: u32 = 32;

export fn gemv_q8_0_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const blocks_per_row = (k + q8_0_group_size - 1) / q8_0_group_size;
    const row_bytes = blocks_per_row * q8_0_block_size;

    var sum: f32 = 0.0;
    var blk = tid;
    while (blk < blocks_per_row) : (blk += bdim) {
        const block_ptr = w + row * row_bytes + blk * q8_0_block_size;

        // Scale: first 2 bytes are little-endian f16
        const scale_bits = @as(u16, block_ptr[0]) | (@as(u16, block_ptr[1]) << 8);
        const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));

        const quants = block_ptr + 2;
        const base_col = blk * q8_0_group_size;

        var blk_sum: f32 = 0.0;
        for (0..q8_0_group_size) |qi| {
            if (base_col + qi < k) {
                const q: i8 = @bitCast(quants[qi]);
                blk_sum += @as(f32, @floatFromInt(q)) * x[base_col + qi];
            }
        }
        sum += scale * blk_sum;
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
