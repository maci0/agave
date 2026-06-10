//! GEMV TQ1_0 kernel: y[row] = dot(W_tq1[row,:], x)
//! TQ1_0 block: 54 bytes = 2 bytes (f16 scale) + 48 bytes (5 trits/byte, 240 elems)
//!              + 4 bytes (4 trits/byte, 16 elems) = 256 elements per block.
//! Values are {-1, 0, +1}: decoded as (trit - 1) * scale.
//! NR=4: Launch with ceil(n/4) blocks, each block processes 4 output rows.

const cu = @import("common.zig");

/// Bytes per TQ1_0 block (256 elements).
const tq1_0_block_bytes: u32 = 54;
/// Elements per TQ1_0 block.
const tq1_0_block_elems: u32 = 256;
/// Packed bytes for 5-trit encoding (48 bytes = 240 elements).
const packed_bytes_5: u32 = 48;
/// Packed bytes for 4-trit encoding (4 bytes = 16 elements).
const packed_bytes_4: u32 = 4;
/// Scale field size in bytes.
const scale_bytes: u32 = 2;
/// Number of output rows per CUDA block.
const nr: u32 = 4;
/// Sparse skip threshold.
const sparse_threshold: f32 = 0.005;

/// Comptime LUT for 5-trit unpacking: 243 entries, each with 5 trit values {-1,0,+1}.
const trit5_table: [243][5]i8 = blk: {
    @setEvalBranchQuota(10000);
    var table: [243][5]i8 = undefined;
    for (0..243) |v| {
        var rem: usize = v;
        for (0..5) |i| {
            table[v][i] = @as(i8, @intCast(rem % 3)) - 1;
            rem /= 3;
        }
    }
    break :blk table;
};

/// Comptime LUT for 4-trit unpacking: 81 entries, each with 4 trit values {-1,0,+1}.
const trit4_table: [81][4]i8 = blk: {
    var table: [81][4]i8 = undefined;
    for (0..81) |v| {
        var rem: usize = v;
        for (0..4) |i| {
            table[v][i] = @as(i8, @intCast(rem % 3)) - 1;
            rem /= 3;
        }
    }
    break :blk table;
};

/// Compute one TQ1_0 block's dot product for a single row.
inline fn tq1_0BlockDot(x: [*]const f32, block_ptr: [*]const u8, k: u32, base_col: u32) f32 {
    // Scale: first 2 bytes are little-endian f16
    const scale = cu.f16tof32(block_ptr);
    const trit_data = block_ptr + scale_bytes;

    var blk_sum: f32 = 0.0;
    var elem: u32 = 0;

    // First 240 elements: 5 trits per byte, 48 bytes
    for (0..packed_bytes_5) |bi| {
        const byte_val = trit_data[bi];
        if (byte_val < 243) {
            const trits = trit5_table[byte_val];
            inline for (0..5) |ti| {
                if (base_col + elem < k) {
                    blk_sum += @as(f32, @floatFromInt(trits[ti])) * x[base_col + elem];
                }
                elem += 1;
            }
        } else {
            elem += 5;
        }
    }

    // Last 16 elements: 4 trits per byte, 4 bytes
    for (0..packed_bytes_4) |bi| {
        const byte_val = trit_data[packed_bytes_5 + bi];
        if (byte_val < 81) {
            const trits = trit4_table[byte_val];
            inline for (0..4) |ti| {
                if (base_col + elem < k) {
                    blk_sum += @as(f32, @floatFromInt(trits[ti])) * x[base_col + elem];
                }
                elem += 1;
            }
        } else {
            elem += 4;
        }
    }

    return scale * blk_sum;
}

/// TQ1_0 GEMV kernel: NR=4 rows per block.
/// Each block processes rows [blockIdx*4 .. blockIdx*4+3].
export fn gemv_tq1_0_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row_base = cu.blockIdx() * nr;
    if (row_base >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const nr_active = @min(nr, n - row_base);

    const blocks_per_row = (k + tq1_0_block_elems - 1) / tq1_0_block_elems;
    const row_bytes = blocks_per_row * tq1_0_block_bytes;

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    var sum3: f32 = 0.0;

    var blk = tid;
    while (blk < blocks_per_row) : (blk += bdim) {
        const base_col = blk * tq1_0_block_elems;

        // Sparse skip: check if all input values in this block are near-zero
        var bmax: f32 = 0.0;
        for (0..tq1_0_block_elems) |i| {
            if (base_col + i < k) {
                const a = @abs(x[base_col + i]);
                if (a > bmax) bmax = a;
            }
        }
        if (bmax < sparse_threshold) continue;

        sum0 += tq1_0BlockDot(x, w + row_base * row_bytes + blk * tq1_0_block_bytes, k, base_col);
        if (nr_active > 1)
            sum1 += tq1_0BlockDot(x, w + (row_base + 1) * row_bytes + blk * tq1_0_block_bytes, k, base_col);
        if (nr_active > 2)
            sum2 += tq1_0BlockDot(x, w + (row_base + 2) * row_bytes + blk * tq1_0_block_bytes, k, base_col);
        if (nr_active > 3)
            sum3 += tq1_0BlockDot(x, w + (row_base + 3) * row_bytes + blk * tq1_0_block_bytes, k, base_col);
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

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(tq1_0_block_bytes > 0);
    comptime std.debug.assert(tq1_0_block_elems > 0);
    comptime std.debug.assert(packed_bytes_5 > 0);
    comptime std.debug.assert(packed_bytes_4 > 0);
    comptime std.debug.assert(scale_bytes > 0);
    comptime std.debug.assert(nr > 0);
}

test "fuzz: gemv_tq1_0 functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = &tq1_0BlockDot;
            }
        }
    }.f, .{});
}
