//! GEMV Q4_0 kernel: y[row] = dot(W_q4[row,:], x)
//! Q4_0 block: 18 bytes = 2 bytes (f16 scale) + 16 bytes (32 nibble-packed quants: byte[i] holds elements [i] and [i+16]).
//! NR=4: Launch with ceil(n/4) blocks, each block processes 4 output rows.
//! The x vector is shared across all rows for cache reuse.

const cu = @import("common.zig");

/// Bytes per Q4_0 block (32 elements).
const q4_0_block_size: u32 = 18;
/// Elements per Q4_0 block.
const q4_0_group_size: u32 = 32;
/// Q4_0 dequant bias: 4-bit unsigned [0..15] centered to signed [-8..7].
const q4_0_dequant_bias: i8 = -8;
/// Number of output rows per CUDA block.
const nr: u32 = 4;

/// Compute one Q4_0 block's dot product for a single row.
inline fn q4_0BlockDot(x: [*]const f32, block_ptr: [*]const u8, k: u32, base_col: u32) f32 {
    // Scale: first 2 bytes are little-endian f16
    const scale_bits = @as(u16, block_ptr[0]) | (@as(u16, block_ptr[1]) << 8);
    const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));

    const quants = block_ptr + 2;

    var blk_sum: f32 = 0.0;
    for (0..16) |qi| {
        const byte = quants[qi];
        // Low nibble: elements [0..15], high nibble: elements [16..31]
        const lo = @as(i8, @intCast(@as(u4, @truncate(byte)))) + q4_0_dequant_bias;
        const hi = @as(i8, @intCast(@as(u4, @truncate(byte >> 4)))) + q4_0_dequant_bias;

        if (base_col + qi < k)
            blk_sum += @as(f32, @floatFromInt(lo)) * x[base_col + qi];
        if (base_col + qi + 16 < k)
            blk_sum += @as(f32, @floatFromInt(hi)) * x[base_col + qi + 16];
    }
    return scale * blk_sum;
}

/// Q4_0 GEMV kernel: NR=4 rows per block.
/// Each block processes rows [blockIdx*4 .. blockIdx*4+3].
export fn gemv_q4_0_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row_base = cu.blockIdx() * nr;
    if (row_base >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const nr_active = @min(nr, n - row_base);

    const blocks_per_row = (k + q4_0_group_size - 1) / q4_0_group_size;
    const row_bytes = blocks_per_row * q4_0_block_size;

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    var sum3: f32 = 0.0;

    var blk = tid;
    while (blk < blocks_per_row) : (blk += bdim) {
        const base_col = blk * q4_0_group_size;

        sum0 += q4_0BlockDot(x, w + row_base * row_bytes + blk * q4_0_block_size, k, base_col);
        if (nr_active > 1)
            sum1 += q4_0BlockDot(x, w + (row_base + 1) * row_bytes + blk * q4_0_block_size, k, base_col);
        if (nr_active > 2)
            sum2 += q4_0BlockDot(x, w + (row_base + 2) * row_bytes + blk * q4_0_block_size, k, base_col);
        if (nr_active > 3)
            sum3 += q4_0BlockDot(x, w + (row_base + 3) * row_bytes + blk * q4_0_block_size, k, base_col);
    }

    sum0 = cu.blockReduceAdd(sum0);
    if (tid == 0) y[row_base] = sum0;

    if (nr_active > 1) {
        cu.syncthreads();
        sum1 = cu.blockReduceAdd(sum1);
        if (tid == 0) y[row_base + 1] = sum1;
    }
    if (nr_active > 2) {
        cu.syncthreads();
        sum2 = cu.blockReduceAdd(sum2);
        if (tid == 0) y[row_base + 2] = sum2;
    }
    if (nr_active > 3) {
        cu.syncthreads();
        sum3 = cu.blockReduceAdd(sum3);
        if (tid == 0) y[row_base + 3] = sum3;
    }
}
