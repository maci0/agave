//! GEMV Q4_0 kernel: y[row] = dot(W_q4[row,:], x)
//! Q4_0 block: 18 bytes = 2 bytes (f16 scale) + 16 bytes (32 nibble-packed quants: byte[i] holds elements [i] and [i+16]).
//! Launch with n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

/// Bytes per Q4_0 block (32 elements).
const q4_0_block_size: u32 = 18;
/// Elements per Q4_0 block.
const q4_0_group_size: u32 = 32;

export fn gemv_q4_0_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const blocks_per_row = (k + q4_0_group_size - 1) / q4_0_group_size;
    const row_bytes = blocks_per_row * q4_0_block_size;

    var sum: f32 = 0.0;
    var blk = tid;
    while (blk < blocks_per_row) : (blk += bdim) {
        const block_ptr = w + row * row_bytes + blk * q4_0_block_size;

        // Scale: first 2 bytes are little-endian f16
        const scale_bits = @as(u16, block_ptr[0]) | (@as(u16, block_ptr[1]) << 8);
        const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));

        const quants = block_ptr + 2;
        const base_col = blk * q4_0_group_size;

        var blk_sum: f32 = 0.0;
        for (0..16) |qi| {
            const byte = quants[qi];
            // Low nibble: elements [0..15], high nibble: elements [16..31]
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
