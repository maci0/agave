//! GEMV Q4_0 kernel: y[row] = dot(W_q4[row,:], x)
//! Q4_0 block: 18 bytes = 2 bytes (f16 scale) + 16 bytes (32 x 4-bit quants).
//! NR=4: Launch with ceil(n/4) workgroups, each processes 4 output rows.
//! The x vector is shared across all rows for cache reuse.
//!
//! Optimized for RDNA3: uses dword (4-byte) global loads for quant data.

const cu = @import("common.zig");

/// Bytes per Q4_0 block (32 elements).
const q4_0_block_size: u32 = 18;
/// Elements per Q4_0 block.
const q4_0_group_size: u32 = 32;
/// Q4_0 dequant bias: 4-bit unsigned [0..15] centered to signed [-8..7].
const q4_0_dequant_bias: i8 = -8;
/// Number of output rows per workgroup.
const nr: u32 = 4;

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
        const lo: f32 = @floatFromInt(@as(i8, @intCast(@as(u4, @truncate(byte)))) + q4_0_dequant_bias);
        const hi: f32 = @floatFromInt(@as(i8, @intCast(@as(u4, @truncate(byte >> 4)))) + q4_0_dequant_bias);
        s += lo * x[col_lo + bi] + hi * x[col_hi + bi];
    }
    return s;
}

/// Compute one Q4_0 block's dot product for a single row using dword loads.
inline fn q4_0BlockDot(x: [*]const f32, row_start: usize, blk: u32, k: u32) f32 {
    const blk_addr = row_start + blk * q4_0_block_size;

    // Scale: f16 (2 bytes) at block start
    const scale: f32 = @floatCast(@as(f16, @bitCast(@as(
        *const u16,
        @ptrFromInt(blk_addr),
    ).*)));

    // Quants: 16 bytes at offset +2, load as 4 x dword
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

    _ = k;
    return scale * blk_sum;
}

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
        sum0 += q4_0BlockDot(x, @intFromPtr(w) + row_base * row_bytes, blk, k);
        if (nr_active > 1)
            sum1 += q4_0BlockDot(x, @intFromPtr(w) + (row_base + 1) * row_bytes, blk, k);
        if (nr_active > 2)
            sum2 += q4_0BlockDot(x, @intFromPtr(w) + (row_base + 2) * row_bytes, blk, k);
        if (nr_active > 3)
            sum3 += q4_0BlockDot(x, @intFromPtr(w) + (row_base + 3) * row_bytes, blk, k);
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
