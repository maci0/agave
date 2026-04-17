//! GEMV Q8_0 kernel: y[row] = dot(W_q8[row,:], x)
//! Q8_0 block: 34 bytes = 2 bytes (f16 scale) + 32 bytes (32 x i8 quants).
//! NR=4: Launch with ceil(n/4) workgroups, each processes 4 output rows.
//! The x vector is shared across all rows for cache reuse.

const cu = @import("common.zig");

/// Bytes per Q8_0 block (32 elements).
const q8_0_block_size: u32 = 34;
/// Elements per Q8_0 block.
const q8_0_group_size: u32 = 32;
/// Number of output rows per workgroup.
const nr: u32 = 4;

/// Load a u32 from an unaligned byte pointer.
inline fn loadDword(ptr: [*]const u8) u32 {
    return @as(*align(1) const u32, @ptrCast(ptr)).*;
}

/// Process 4 i8 quants packed in a dword.
inline fn accumDword(dw: u32, x: [*]const f32, base: u32) f32 {
    var s: f32 = 0.0;
    inline for (0..4) |bi| {
        const shift: u5 = @intCast(bi * 8);
        const q: i8 = @bitCast(@as(u8, @truncate(dw >> shift)));
        s += @as(f32, @floatFromInt(q)) * x[base + bi];
    }
    return s;
}

/// Compute one Q8_0 block's dot product for a single row using dword loads.
inline fn q8_0BlockDot(x: [*]const f32, block_ptr: [*]const u8, base_col: u32) f32 {
    // Scale: 2-byte f16
    const scale: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(block_ptr)).*)));

    const quants = block_ptr + 2;

    // Load 32 quant bytes as 8 x dword (8 loads instead of 32)
    var blk_sum: f32 = 0.0;
    inline for (0..8) |i| {
        blk_sum += accumDword(loadDword(quants + i * 4), x, base_col + @as(u32, i) * 4);
    }
    return scale * blk_sum;
}

export fn gemv_q8_0_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row_base = cu.blockIdx() * nr;
    if (row_base >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const nr_active = @min(nr, n - row_base);

    const blocks_per_row = (k + q8_0_group_size - 1) / q8_0_group_size;
    const row_bytes = blocks_per_row * q8_0_block_size;

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    var sum3: f32 = 0.0;

    var blk = tid;
    while (blk < blocks_per_row) : (blk += bdim) {
        const base_col = blk * q8_0_group_size;
        const row0_ptr = w + row_base * row_bytes + blk * q8_0_block_size;

        sum0 += q8_0BlockDot(x, row0_ptr, base_col);
        if (nr_active > 1)
            sum1 += q8_0BlockDot(x, w + (row_base + 1) * row_bytes + blk * q8_0_block_size, base_col);
        if (nr_active > 2)
            sum2 += q8_0BlockDot(x, w + (row_base + 2) * row_bytes + blk * q8_0_block_size, base_col);
        if (nr_active > 3)
            sum3 += q8_0BlockDot(x, w + (row_base + 3) * row_bytes + blk * q8_0_block_size, base_col);
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
