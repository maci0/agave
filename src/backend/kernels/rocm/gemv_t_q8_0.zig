//! Transposed GEMV Q8_0 kernel: y[col] = sum_i(W[i, col] * x[i])
//! W is stored as [in_dim rows, out_dim cols] in Q8_0 blocks (row-major).
//! Each row has ceil(out_dim / 32) blocks of 34 bytes (f16 scale + 32 x i8).
//! One workgroup per output column. Threads stride over input rows,
//! dequantizing the single element at the target column and accumulating.

const cu = @import("common.zig");

/// Bytes per Q8_0 block (32 elements).
const q8_0_block_size: u32 = 34;
/// Elements per Q8_0 block.
const q8_0_group_size: u32 = 32;

export fn gemv_t_q8_0_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, out_dim: u32, in_dim: u32) callconv(.kernel) void {
    const col = cu.blockIdx();
    if (col >= out_dim) return;

    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const blocks_per_row = (out_dim + q8_0_group_size - 1) / q8_0_group_size;
    const row_bytes = blocks_per_row * q8_0_block_size;

    // Which block and offset within the block does column 'col' land in?
    const blk_idx = col / q8_0_group_size;
    const elem_in_blk = col % q8_0_group_size;

    // Byte offsets within a row to the target block's scale and quant
    const blk_byte_off = blk_idx * q8_0_block_size;
    const quant_offset = blk_byte_off + 2 + elem_in_blk;

    var sum: f32 = 0.0;

    // Each thread processes rows in stride
    var row = tid;
    while (row < in_dim) : (row += bdim) {
        const row_base = row * row_bytes;
        const block_ptr = w + row_base + blk_byte_off;

        // Read f16 scale from first 2 bytes of the block
        const scale: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(block_ptr)).*)));

        // Read single i8 quant value
        const qval: i8 = @bitCast((w + row_base + quant_offset)[0]);
        const dequant: f32 = scale * @as(f32, @floatFromInt(qval));

        sum += dequant * x[row];
    }

    // Block-level reduction (wave reduce + LDS inter-wave)
    sum = cu.blockReduceAdd(sum);

    if (tid == 0) y[col] = sum;
}
