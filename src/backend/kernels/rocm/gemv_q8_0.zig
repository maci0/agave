//! GEMV Q8_0 kernel: y[row] = dot(W_q8[row,:], x)
//! Q8_0 block: 34 bytes = 2 bytes (f16 scale) + 32 bytes (32 × i8 quants).
//! Launch with n workgroups of 256 threads (one row per workgroup).

const cu = @import("common.zig");

/// Bytes per Q8_0 block (32 elements).
const q8_0_block_size: u32 = 34;
/// Elements per Q8_0 block.
const q8_0_group_size: u32 = 32;

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

export fn gemv_q8_0_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const blocks_per_row = (k + q8_0_group_size - 1) / q8_0_group_size;
    const row_bytes = blocks_per_row * q8_0_block_size;
    const row_start = w + row * row_bytes;

    var sum: f32 = 0.0;
    var blk = tid;
    while (blk < blocks_per_row) : (blk += bdim) {
        const block_ptr = row_start + blk * q8_0_block_size;

        // Scale: 2-byte f16
        const scale: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(block_ptr)).*)));

        const quants = block_ptr + 2;
        const base_col = blk * q8_0_group_size;

        // Load 32 quant bytes as 8 × dword (8 loads instead of 32)
        var blk_sum: f32 = 0.0;
        inline for (0..8) |i| {
            blk_sum += accumDword(loadDword(quants + i * 4), x, base_col + @as(u32, i) * 4);
        }
        sum += scale * blk_sum;
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
