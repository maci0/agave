//! Transposed Q8_0 GEMV: y[out_dim] = W^T @ x[in_dim]
//! W is stored as [in_dim rows, out_dim cols] in Q8_0 blocks.
//! Each row has ceil(out_dim/32) Q8_0 blocks (34 bytes each: f16 scale + 32 int8).
//! One block per output element — each block reduces across all in_dim rows.
//! Grid: out_dim blocks of 256 threads.

const cu = @import("common.zig");

/// Q8_0 block: 2-byte f16 scale + 32 int8 values = 34 bytes.
const q8_0_block_bytes: u32 = 34;
const q8_0_block_elems: u32 = 32;

export fn gemv_t_q8_0_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, out_dim: u32, in_dim: u32) callconv(.kernel) void {
    const col = cu.blockIdx(); // output element index
    if (col >= out_dim) return;

    const tid = cu.threadIdx();
    const bd = cu.blockDim();
    const blocks_per_row = (out_dim + q8_0_block_elems - 1) / q8_0_block_elems;
    const blk_col = col / q8_0_block_elems;
    const blk_off = col % q8_0_block_elems;

    var sum: f32 = 0.0;
    // Each thread strides over in_dim rows
    var j: u32 = tid;
    while (j < in_dim) : (j += bd) {
        const blk_ptr = w + (j * blocks_per_row + blk_col) * q8_0_block_bytes;
        // Read f16 scale (2 bytes, little-endian)
        const scale_bits: u16 = @as(u16, blk_ptr[0]) | (@as(u16, blk_ptr[1]) << 8);
        const scale: f32 = @as(f32, @as(f16, @bitCast(scale_bits)));
        // Read int8 value at block offset
        const val: i8 = @bitCast(blk_ptr[2 + blk_off]);
        sum += @as(f32, @floatFromInt(val)) * scale * x[j];
    }

    // Block reduction
    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[col] = sum;
}
