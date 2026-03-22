//! GEMV Q4_0 kernel: y[row] = dot(W_q4[row,:], x)
//! Q4_0 block: 18 bytes = 2 bytes (f16 scale) + 16 bytes (32 × 4-bit quants).
//! Launch with n workgroups of 256 threads (one row per workgroup).
//!
//! Optimized for RDNA3: uses dword (4-byte) global loads for quant data.

const cu = @import("common.zig");

/// Bytes per Q4_0 block (32 elements).
const q4_0_block_size: u32 = 18;
/// Elements per Q4_0 block.
const q4_0_group_size: u32 = 32;

/// Load 4 bytes from an arbitrary address as u32 (unaligned-safe).
/// Uses align(1) to avoid UB when Q4_0 block boundaries are not 4-byte aligned.
inline fn loadU32(addr: usize) u32 {
    return @as(*align(1) const u32, @ptrFromInt(addr)).*;
}

/// Process a packed dword of 4 quant bytes: accumulate into sum.
/// Low nibbles are elements [col_lo..col_lo+3],
/// high nibbles are elements [col_hi..col_hi+3].
inline fn accumDword(dw: u32, x: [*]const f32, col_lo: u32, col_hi: u32) f32 {
    var s: f32 = 0.0;
    inline for (0..4) |bi| {
        const shift: u5 = @intCast(bi * 8);
        const byte: u8 = @truncate(dw >> shift);
        const lo: f32 = @floatFromInt(@as(i8, @intCast(@as(u4, @truncate(byte)))) - 8);
        const hi: f32 = @floatFromInt(@as(i8, @intCast(@as(u4, @truncate(byte >> 4)))) - 8);
        s += lo * x[col_lo + bi] + hi * x[col_hi + bi];
    }
    return s;
}

export fn gemv_q4_0_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const blocks_per_row = (k + q4_0_group_size - 1) / q4_0_group_size;
    const row_bytes = blocks_per_row * q4_0_block_size;
    const row_start = @intFromPtr(w) + row * row_bytes;

    var sum: f32 = 0.0;
    var blk = tid;
    while (blk < blocks_per_row) : (blk += bdim) {
        const blk_addr = row_start + blk * q4_0_block_size;

        // Scale: f16 (2 bytes) at block start
        const scale: f32 = @floatCast(@as(f16, @bitCast(@as(
            *const u16,
            @ptrFromInt(blk_addr),
        ).*)));

        // Quants: 16 bytes at offset +2, load as 4 × dword
        const q_addr = blk_addr + 2;
        const q0 = loadU32(q_addr);
        const q1 = loadU32(q_addr + 4);
        const q2 = loadU32(q_addr + 8);
        const q3 = loadU32(q_addr + 12);

        const base_col = blk * q4_0_group_size;

        var blk_sum: f32 = 0.0;
        blk_sum += accumDword(q0, x, base_col, base_col + 16);
        blk_sum += accumDword(q1, x, base_col + 4, base_col + 20);
        blk_sum += accumDword(q2, x, base_col + 8, base_col + 24);
        blk_sum += accumDword(q3, x, base_col + 12, base_col + 28);

        sum += scale * blk_sum;
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
