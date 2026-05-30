//! CPU GEMV kernel for TQ1_0 (ternary 1.58-bit) quantization.
//! 256 values per block, 54 bytes: f16 scale + 52 bytes packed trits.
//! Packing: 5 trits per byte (3^5=243 < 256) for first 240 elements,
//! then 4 trits per byte for last 16 elements.
//! Values are {-1, 0, +1}, stored as base-3 digits (0=−1, 1=0, 2=+1).
//! Dequantization: value = (trit - 1) * scale

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const gemv_common = @import("gemv.zig");

const block_elems: usize = 256;
const block_bytes: usize = 54;
const scale_bytes: usize = 2;
const packed_bytes_5: usize = 48;
const packed_bytes_4: usize = 4;

/// Lookup tables for unpacking 5 trits from a byte (base-3: 3^5 = 243 values).
/// Each byte encodes 5 ternary digits d0..d4 where byte = d0 + 3*d1 + 9*d2 + 27*d3 + 81*d4.
/// Decoded as (digit - 1) to get {-1, 0, +1}.
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

/// Unpack 4 trits from a byte (base-3: 3^4 = 81 values, upper bits ignored).
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

pub fn gemvTQ1_0(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const nb = (k + block_elems - 1) / block_elems;
    const row_bytes = nb * block_bytes;

    var row: usize = 0;
    while (row < n) : (row += 1) {
        var sum: f32 = 0.0;
        const rp = w + row * row_bytes;

        for (0..nb) |b| {
            const bk = b * block_elems;
            if (gemv_common.isBlockSparse(x, bk, @min(block_elems, k - bk))) continue;

            const bp = rp + b * block_bytes;
            const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
            const trit_data = bp + scale_bytes;

            // First 240 elements: 5 trits per byte, 48 bytes
            var elem: usize = 0;
            for (0..packed_bytes_5) |bi| {
                const byte_val = trit_data[bi];
                if (byte_val < 243) {
                    const trits = trit5_table[byte_val];
                    inline for (0..5) |ti| {
                        if (bk + elem < k) {
                            sum += @as(f32, @floatFromInt(trits[ti])) * scale * x[bk + elem];
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
                        if (bk + elem < k) {
                            sum += @as(f32, @floatFromInt(trits[ti])) * scale * x[bk + elem];
                        }
                        elem += 1;
                    }
                } else {
                    elem += 4;
                }
            }
        }
        y[row] = sum;
    }
}
