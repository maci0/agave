//! CPU GEMV kernel for TQ2_0 (ternary 2-bit) quantization.
//! 256 values per block, 66 bytes: f16 scale (2) + qs[64] (packed 2-bit values).
//! Packing: 4 ternary values per byte, 2 bits each (bits 0-1, 2-3, 4-5, 6-7).
//! Values: 0→-1, 1→0, 2→+1 (3 undefined). Dequant: (q - 1) * scale.
//! struct block_tq2_0 { uint8_t qs[64]; ggml_half d; } // llama.cpp layout

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const sparsity = @import("activation_sparsity.zig");

/// Number of quantized elements (ternary values) packed into one TQ2_0 block.
pub const block_elems: usize = 256;
/// Byte size of one TQ2_0 block: 64 bytes packed qs + 2-byte f16 scale.
pub const block_bytes: usize = 66;
const scale_bytes: usize = 2;
const qs_bytes: usize = 64; // 256 × 2 bits / 8

const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);

inline fn f16ToF32(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

/// TQ2_0 GEMV: y[row] = sum_k dequant(w[row,k]) * x[k]
/// Row-major weight layout: each row is nb × 66 bytes.
pub fn gemvTQ2_0(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const nb = (k + block_elems - 1) / block_elems;
    const row_bytes = nb * block_bytes;

    // The activation vector is fixed for the whole GEMV, so its per-block
    // sparsity is computed once here instead of once per row group.
    const mask = sparsity.blockMask(x, nb, block_elems, k);

    var row: usize = 0;
    while (row < n) : (row += 1) {
        var sum: f32 = 0.0;
        const rp = w + row * row_bytes;

        for (0..nb) |b| {
            const bk = b * block_elems;
            const block_start = b * block_bytes;
            const elements_in_block = @min(block_elems, k - bk);

            if (mask.isSparse(b)) continue;

            // f16 scale at bytes [0..2], qs at bytes [2..66]
            const scale = f16ToF32(@as(*const u16, @ptrCast(@alignCast(rp + block_start))).*);
            const qs = rp + block_start + scale_bytes;

            // 4 values per byte, 2 bits each, ~4 SIMD iterations at 64 bytes
            var elem: usize = 0;
            while (elem + 4 <= elements_in_block) : (elem += 4) {
                const byte = qs[elem / 4];
                // Unroll 4 slots per byte
                const q0: f32 = @floatFromInt(byte & 0x3);
                const q1: f32 = @floatFromInt((byte >> 2) & 0x3);
                const q2: f32 = @floatFromInt((byte >> 4) & 0x3);
                const q3: f32 = @floatFromInt((byte >> 6) & 0x3);
                sum += (q0 - 1.0) * scale * x[bk + elem];
                sum += (q1 - 1.0) * scale * x[bk + elem + 1];
                sum += (q2 - 1.0) * scale * x[bk + elem + 2];
                sum += (q3 - 1.0) * scale * x[bk + elem + 3];
            }
            // Tail (only if k not divisible by 4)
            while (elem < elements_in_block) : (elem += 1) {
                const shift: u3 = @intCast((elem % 4) * 2);
                const q: f32 = @floatFromInt((qs[elem / 4] >> shift) & 0x3);
                sum += (q - 1.0) * scale * x[bk + elem];
            }
        }
        y[row] = sum;
    }
}

// ── Tests ────────────────────────────────────────────────────────

test "TQ2_0 block layout" {
    try std.testing.expectEqual(@as(usize, 256), block_elems);
    try std.testing.expectEqual(@as(usize, 66), block_bytes);
}

test "TQ2_0 value encoding" {
    // 0b00=0→-1, 0b01=1→0, 0b10=2→+1
    const scale: f32 = 2.0;
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), (0.0 - 1.0) * scale, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), (1.0 - 1.0) * scale, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), (2.0 - 1.0) * scale, 1e-6);
}

test "TQ2_0 gemv all positive (+1)" {
    // All qs bytes = 0xAA = 0b10101010 → all 4 values = 2 → +1
    var block: [block_bytes]u8 align(2) = undefined;
    @as(*f16, @ptrCast(@alignCast(&block[0]))).* = @as(f16, 1.0);
    @memset(block[2..], 0xAA);
    const x = [_]f32{1.0} ** 256;
    var y: [1]f32 = .{0.0};
    gemvTQ2_0(&x, &block, &y, 1, 256);
    try std.testing.expectApproxEqAbs(@as(f32, 256.0), y[0], 0.5);
}

test "TQ2_0 gemv all zero (0)" {
    // All qs bytes = 0x55 = 0b01010101 → all 4 values = 1 → 0
    var block: [block_bytes]u8 align(2) = undefined;
    @as(*f16, @ptrCast(@alignCast(&block[0]))).* = @as(f16, 1.0);
    @memset(block[2..], 0x55);
    const x = [_]f32{3.14} ** 256;
    var y: [1]f32 = .{99.0};
    gemvTQ2_0(&x, &block, &y, 1, 256);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[0], 1e-4);
}

test "TQ2_0 gemv all negative (-1)" {
    // All qs bytes = 0x00 → all values = 0 → -1, scale=0.5 → -128
    var block: [block_bytes]u8 align(2) = undefined;
    @as(*f16, @ptrCast(@alignCast(&block[0]))).* = @as(f16, 0.5);
    @memset(block[2..], 0x00);
    const x = [_]f32{1.0} ** 256;
    var y: [1]f32 = .{0.0};
    gemvTQ2_0(&x, &block, &y, 1, 256);
    try std.testing.expectApproxEqAbs(@as(f32, -128.0), y[0], 0.5);
}

test "TQ2_0 gemv multi-row" {
    // 2 rows: row0 all+1, row1 all-1
    var w: [2 * block_bytes]u8 align(2) = undefined;
    @as(*f16, @ptrCast(@alignCast(&w[0]))).* = @as(f16, 1.0);
    @memset(w[2..block_bytes], 0xAA); // row0: +1
    @as(*f16, @ptrCast(@alignCast(&w[block_bytes]))).* = @as(f16, 1.0);
    @memset(w[block_bytes + 2 ..], 0x00); // row1: -1
    const x = [_]f32{1.0} ** 256;
    var y: [2]f32 = .{ 0.0, 0.0 };
    gemvTQ2_0(&x, &w, &y, 2, 256);
    try std.testing.expectApproxEqAbs(@as(f32, 256.0), y[0], 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, -256.0), y[1], 0.5);
}

test "fuzz: all gemv_tq2_0 functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            var block: [block_bytes]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            // Clamp scale to finite f16 (avoid inf/nan)
            const raw: u16 = smith.valueWithHash(u16, 1);
            @as(*u16, @ptrCast(@alignCast(&block[0]))).* = raw & 0x7BFF;

            var x: [256]f32 = undefined;
            for (&x, 0..) |*v, i|
                v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 10)))) / 50.0;

            var y: [1]f32 = .{0.0};
            gemvTQ2_0(&x, &block, &y, 1, 256);
            try std.testing.expect(!std.math.isNan(y[0]));
        }
    }.f, .{});
}
