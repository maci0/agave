//! CPU GEMV kernel for TQ1_0 (ternary 1.58-bit) quantization.
//! 256 values per block, 54 bytes: f16 scale + 52 bytes packed trits.
//! Packing: 5 trits per byte (3^5=243 < 256) for first 240 elements,
//! then 4 trits per byte for last 16 elements.
//! Values are {-1, 0, +1}, stored as base-3 digits (0=−1, 1=0, 2=+1).
//! Dequantization: value = (trit - 1) * scale

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const sparsity = @import("activation_sparsity.zig");

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

/// Entry point for TQ1_0 GEMV: y[row] = sum over blocks of (dequant(w) * x).
pub fn gemvTQ1_0(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const nb = (k + block_elems - 1) / block_elems;
    const row_bytes = nb * block_bytes;

    var row: usize = 0;
    while (row < n) : (row += 1) {
        var sum: f32 = 0.0;
        const rp = w + row * row_bytes;

        for (0..nb) |b| {
            const bk = b * block_elems;
            if (sparsity.isBlockSparse(x, bk, @min(block_elems, k - bk))) continue;

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Helper: fill a TQ1_0 block (54 bytes) with a constant trit-byte value
/// for the 5-packed region and a constant for the 4-packed region, plus an f16 scale.
fn fillBlock(buf: *[block_bytes]u8, scale: f16, trit5_val: u8, trit4_val: u8) void {
    const scale_bits = @as(u16, @bitCast(scale));
    std.mem.writeInt(u16, buf[0..2], scale_bits, .little);
    @memset(buf[scale_bytes .. scale_bytes + packed_bytes_5], trit5_val);
    @memset(buf[scale_bytes + packed_bytes_5 .. scale_bytes + packed_bytes_5 + packed_bytes_4], trit4_val);
}

test "TQ1_0 all-zero block (trit=1 → weight=0) produces zero output" {
    // Trit value 1 in every position → decoded weight = (1-1)*scale = 0.
    // 5-packed byte: 1 + 1*3 + 1*9 + 1*27 + 1*81 = 121.
    // 4-packed byte: 1 + 1*3 + 1*9 + 1*27 = 40.
    var block: [block_bytes]u8 = undefined;
    fillBlock(&block, @as(f16, 1.0), 121, 40);

    var x: [block_elems]f32 = undefined;
    @memset(&x, 1.0);

    var y: [1]f32 = .{999.0};
    gemvTQ1_0(&x, &block, &y, 1, block_elems);

    try testing.expectApproxEqAbs(@as(f32, 0.0), y[0], 0.01);
}

test "TQ1_0 all-positive block (trit=2 → weight=+1) sums to 256" {
    // Trit value 2 in every position → decoded weight = (2-1)*scale = +1*scale.
    // 5-packed byte: 2 + 2*3 + 2*9 + 2*27 + 2*81 = 242.
    // 4-packed byte: 2 + 2*3 + 2*9 + 2*27 = 80.
    var block: [block_bytes]u8 = undefined;
    fillBlock(&block, @as(f16, 1.0), 242, 80);

    var x: [block_elems]f32 = undefined;
    @memset(&x, 1.0);

    var y: [1]f32 = .{0.0};
    gemvTQ1_0(&x, &block, &y, 1, block_elems);

    // All 256 elements are +1 * 1.0 * 1.0 = 1.0, sum = 256.
    try testing.expectApproxEqAbs(@as(f32, 256.0), y[0], 0.01);
}

test "TQ1_0 scale factor applied correctly" {
    // Same as all-positive but scale=0.5 → sum should be 128.
    var block: [block_bytes]u8 = undefined;
    fillBlock(&block, @as(f16, 0.5), 242, 80);

    var x: [block_elems]f32 = undefined;
    @memset(&x, 1.0);

    var y: [1]f32 = .{0.0};
    gemvTQ1_0(&x, &block, &y, 1, block_elems);

    try testing.expectApproxEqAbs(@as(f32, 128.0), y[0], 0.5);
}

test "TQ1_0 all-negative block (trit=0 → weight=-1) sums to -256" {
    // Trit value 0 in every position → decoded weight = (0-1)*scale = -1.
    // 5-packed byte: 0. 4-packed byte: 0.
    var block: [block_bytes]u8 = undefined;
    fillBlock(&block, @as(f16, 1.0), 0, 0);

    var x: [block_elems]f32 = undefined;
    @memset(&x, 1.0);

    var y: [1]f32 = .{0.0};
    gemvTQ1_0(&x, &block, &y, 1, block_elems);

    try testing.expectApproxEqAbs(@as(f32, -256.0), y[0], 0.01);
}

test "TQ1_0 multi-row GEMV" {
    // Two rows: first all-positive, second all-zero trits.
    var blocks: [2 * block_bytes]u8 = undefined;
    fillBlock(blocks[0..block_bytes], @as(f16, 1.0), 242, 80);
    fillBlock(blocks[block_bytes..][0..block_bytes], @as(f16, 1.0), 121, 40);

    var x: [block_elems]f32 = undefined;
    @memset(&x, 1.0);

    var y: [2]f32 = .{ 0.0, 0.0 };
    gemvTQ1_0(&x, &blocks, &y, 2, block_elems);

    try testing.expectApproxEqAbs(@as(f32, 256.0), y[0], 0.01);
    try testing.expectApproxEqAbs(@as(f32, 0.0), y[1], 0.01);
}

test "fuzz: all gemv_tq1_0 functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // gemvTQ1_0: y[row] = dot(dequant(w), x) for each row
            // Use 1 row, 1 block (256 elements, 54 bytes weight data)
            const n_rows = 1;

            var x: [block_elems]f32 = undefined;
            for (&x) |*v| {
                v.* = @bitCast(smith.valueWithHash(u32, 0));
                if (!std.math.isFinite(v.*)) v.* = 0.0;
            }

            // Build a valid TQ1_0 block: 2-byte f16 scale + 48 bytes (5-packed) + 4 bytes (4-packed)
            var w_buf: [block_bytes]u8 align(2) = undefined;
            const scale_bits = smith.valueWithHash(u16, 1);
            std.mem.writeInt(u16, w_buf[0..2], scale_bits, .little);
            // 5-packed trit bytes: clamp to [0,242] so they index into trit5_table
            for (w_buf[scale_bytes .. scale_bytes + packed_bytes_5]) |*b| {
                b.* = smith.valueWithHash(u8, 2) % 243;
            }
            // 4-packed trit bytes: clamp to [0,80] so they index into trit4_table
            for (w_buf[scale_bytes + packed_bytes_5 .. scale_bytes + packed_bytes_5 + packed_bytes_4]) |*b| {
                b.* = smith.valueWithHash(u8, 3) % 81;
            }

            var y: [n_rows]f32 = .{0.0};
            gemvTQ1_0(&x, &w_buf, &y, n_rows, block_elems);

            // Scale may be NaN/Inf f16 -- only check finite when scale is finite
            const scale_f32: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));
            if (std.math.isFinite(scale_f32)) {
                if (!std.math.isFinite(y[0])) return error.TestUnexpectedResult;
            }
        }
    }.f, .{});
}
