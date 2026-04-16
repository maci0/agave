//! CPU GEMV kernel for Q4_0 quantization.
//! 32 values per block, 18 bytes (f16 scale + 16 nibble-packed bytes).
//! Layout: byte j (j=0..15) contains element j in its low nibble and element j+16 in its high nibble.
//! 4-row batching with V8 SIMD for x-vector cache reuse.

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const V8 = @Vector(8, f32);
const V8u = @Vector(8, u8);
const V8i16 = @Vector(8, i16);
const v8zero: V8 = @splat(0.0);
const nib_mask: V8u = @splat(0x0F);
const shift4: @Vector(8, u3) = @splat(4);
/// Q4_0 dequant bias: 4-bit unsigned [0..15] centered to signed [-8..7].
const q4_0_dequant_bias: i8 = -8;
const q4_bias: V8i16 = @splat(q4_0_dequant_bias);

/// Q4_0 GEMV: y = W @ x. 4-row batched with V8 SIMD for x-vector cache reuse.
pub fn gemvQ4_0(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb = backend_mod.q4_0_block_bytes;
    const qk = backend_mod.quant_block_elems;
    const nb = (k + qk - 1) / qk;
    const row_bytes = nb * bpb;

    // Process 4 rows at a time for better x-vector cache reuse.
    var row: usize = 0;
    while (row + 4 <= n) : (row += 4) {
        var acc0: V8 = v8zero;
        var acc1: V8 = v8zero;
        var acc2: V8 = v8zero;
        var acc3: V8 = v8zero;
        const rp0 = w + row * row_bytes;
        const rp1 = w + (row + 1) * row_bytes;
        const rp2 = w + (row + 2) * row_bytes;
        const rp3 = w + (row + 3) * row_bytes;

        for (0..nb) |b| {
            const bp0 = rp0 + b * bpb;
            const bp1 = rp1 + b * bpb;
            const bp2 = rp2 + b * bpb;
            const bp3 = rp3 + b * bpb;
            const d0: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp0[0..2], .little))));
            const d1: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp1[0..2], .little))));
            const d2: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp2[0..2], .little))));
            const d3: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp3[0..2], .little))));
            const bk = b * qk;

            if (bk + qk - 1 < k) {
                const x_lo: V8 = x[bk..][0..8].*;
                const x_lo2: V8 = x[bk + 8 ..][0..8].*;
                const x_hi: V8 = x[bk + 16 ..][0..8].*;
                const x_hi2: V8 = x[bk + 24 ..][0..8].*;

                inline for (.{ .{ bp0, &acc0, d0 }, .{ bp1, &acc1, d1 }, .{ bp2, &acc2, d2 }, .{ bp3, &acc3, d3 } }) |pair| {
                    const bp = pair[0];
                    const acc = pair[1];
                    const d = pair[2];
                    // Vector byte extraction: load 16 bytes, split nibbles via AND/SHIFT,
                    // widen i8→i16→f32 via SIMD widening.
                    const raw0: V8u = bp[2..10].*;
                    const raw1: V8u = bp[10..18].*;
                    const lo: V8 = @floatFromInt(@as(V8i16, @intCast(raw0 & nib_mask)) + q4_bias);
                    const hi: V8 = @floatFromInt(@as(V8i16, @intCast(raw0 >> shift4)) + q4_bias);
                    const lo2: V8 = @floatFromInt(@as(V8i16, @intCast(raw1 & nib_mask)) + q4_bias);
                    const hi2: V8 = @floatFromInt(@as(V8i16, @intCast(raw1 >> shift4)) + q4_bias);
                    const block_v = x_lo * lo + x_lo2 * lo2 + x_hi * hi + x_hi2 * hi2;
                    acc.* += @as(V8, @splat(d)) * block_v;
                }
            } else {
                var bs0: f32 = 0.0;
                var bs1: f32 = 0.0;
                var bs2: f32 = 0.0;
                var bs3: f32 = 0.0;
                for (0..qk / 2) |j| {
                    const byte0 = bp0[2 + j];
                    const byte1 = bp1[2 + j];
                    const byte2 = bp2[2 + j];
                    const byte3 = bp3[2 + j];
                    const gi0 = bk + j;
                    const gi1 = bk + j + qk / 2;
                    if (gi0 < k) {
                        const xv = x[gi0];
                        bs0 += xv * @as(f32, @floatFromInt(@as(i8, @intCast(byte0 & 0x0F)) + q4_0_dequant_bias));
                        bs1 += xv * @as(f32, @floatFromInt(@as(i8, @intCast(byte1 & 0x0F)) + q4_0_dequant_bias));
                        bs2 += xv * @as(f32, @floatFromInt(@as(i8, @intCast(byte2 & 0x0F)) + q4_0_dequant_bias));
                        bs3 += xv * @as(f32, @floatFromInt(@as(i8, @intCast(byte3 & 0x0F)) + q4_0_dequant_bias));
                    }
                    if (gi1 < k) {
                        const xv = x[gi1];
                        bs0 += xv * @as(f32, @floatFromInt(@as(i8, @intCast(byte0 >> 4)) + q4_0_dequant_bias));
                        bs1 += xv * @as(f32, @floatFromInt(@as(i8, @intCast(byte1 >> 4)) + q4_0_dequant_bias));
                        bs2 += xv * @as(f32, @floatFromInt(@as(i8, @intCast(byte2 >> 4)) + q4_0_dequant_bias));
                        bs3 += xv * @as(f32, @floatFromInt(@as(i8, @intCast(byte3 >> 4)) + q4_0_dequant_bias));
                    }
                }
                acc0[0] += bs0 * d0;
                acc1[0] += bs1 * d1;
                acc2[0] += bs2 * d2;
                acc3[0] += bs3 * d3;
            }
        }
        y[row] = @reduce(.Add, acc0);
        y[row + 1] = @reduce(.Add, acc1);
        y[row + 2] = @reduce(.Add, acc2);
        y[row + 3] = @reduce(.Add, acc3);
    }

    while (row < n) : (row += 1) {
        var sum: f32 = 0.0;
        const rp = w + row * row_bytes;
        for (0..nb) |b| {
            const bp = rp + b * bpb;
            const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
            const bk = b * qk;
            var block_sum: f32 = 0.0;
            if (bk + qk - 1 < k) {
                const x_lo: V8 = x[bk..][0..8].*;
                const x_lo2: V8 = x[bk + 8 ..][0..8].*;
                const x_hi: V8 = x[bk + 16 ..][0..8].*;
                const x_hi2: V8 = x[bk + 24 ..][0..8].*;
                // Vectorized nibble extraction (same as 4-row batched path)
                const raw0: V8u = bp[2..10].*;
                const raw1: V8u = bp[10..18].*;
                const lo: V8 = @floatFromInt(@as(V8i16, @intCast(raw0 & nib_mask)) + q4_bias);
                const hi: V8 = @floatFromInt(@as(V8i16, @intCast(raw0 >> shift4)) + q4_bias);
                const lo2: V8 = @floatFromInt(@as(V8i16, @intCast(raw1 & nib_mask)) + q4_bias);
                const hi2: V8 = @floatFromInt(@as(V8i16, @intCast(raw1 >> shift4)) + q4_bias);
                block_sum = @reduce(.Add, x_lo * lo + x_lo2 * lo2 + x_hi * hi + x_hi2 * hi2);
            } else {
                for (0..qk / 2) |j| {
                    const byte = bp[2 + j];
                    const x0 = @as(i8, @intCast(byte & 0x0F)) + q4_0_dequant_bias;
                    const x1 = @as(i8, @intCast(byte >> 4)) + q4_0_dequant_bias;
                    const gi0 = bk + j;
                    const gi1 = bk + j + qk / 2;
                    if (gi0 < k) block_sum += x[gi0] * @as(f32, @floatFromInt(x0));
                    if (gi1 < k) block_sum += x[gi1] * @as(f32, @floatFromInt(x1));
                }
            }
            sum += block_sum * d;
        }
        y[row] = sum;
    }
}

test "gemvQ4_0 all zeros (nibble 8 bias cancels)" {
    // Q4_0: nibble value 8 → dequantized = 8 - 8 = 0.
    // 4 rows, k=32, scale=1.0, all nibbles=8 → all weights=0 → y = 0.
    const bpb = backend_mod.q4_0_block_bytes; // 18
    var w: [4 * bpb]u8 = undefined;
    for (0..4) |r| {
        const base = r * bpb;
        // f16(1.0) = 0x3C00 little-endian
        w[base] = 0x00;
        w[base + 1] = 0x3C;
        // Each byte: lo nibble=8, hi nibble=8 → 0x88
        for (0..16) |i| w[base + 2 + i] = 0x88;
    }
    var x: [32]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [4]f32 = undefined;
    gemvQ4_0(&x, &w, &y, 4, 32);
    for (0..4) |i| try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[i], 0.01);
}

test "gemvQ4_0 uniform positive weights" {
    // Q4_0: nibble value 9 → dequantized = 9 - 8 = 1.
    // 4 rows, k=32, scale=1.0, all nibbles=9 → all weights=1.
    // x = all 1.0 → each y[i] = 1.0 * 32 * 1 = 32.0
    const bpb = backend_mod.q4_0_block_bytes;
    var w: [4 * bpb]u8 = undefined;
    for (0..4) |r| {
        const base = r * bpb;
        w[base] = 0x00;
        w[base + 1] = 0x3C; // f16(1.0)
        // lo nibble=9, hi nibble=9 → 0x99
        for (0..16) |i| w[base + 2 + i] = 0x99;
    }
    var x: [32]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [4]f32 = undefined;
    gemvQ4_0(&x, &w, &y, 4, 32);
    for (0..4) |i| try std.testing.expectApproxEqAbs(@as(f32, 32.0), y[i], 0.01);
}

test "gemvQ4_0 negative weights" {
    // Q4_0: nibble value 5 → dequantized = 5 - 8 = -3.
    // 2 rows, k=32, scale=1.0, all nibbles=5 → all weights=-3.
    // x = all 1.0 → each y[i] = 1.0 * 32 * (-3) = -96.0
    const bpb = backend_mod.q4_0_block_bytes;
    var w: [2 * bpb]u8 = undefined;
    for (0..2) |r| {
        const base = r * bpb;
        w[base] = 0x00;
        w[base + 1] = 0x3C; // f16(1.0)
        // lo nibble=5, hi nibble=5 → 0x55
        for (0..16) |i| w[base + 2 + i] = 0x55;
    }
    var x: [32]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [2]f32 = undefined;
    gemvQ4_0(&x, &w, &y, 2, 32);
    for (0..2) |i| try std.testing.expectApproxEqAbs(@as(f32, -96.0), y[i], 0.01);
}

test "gemvQ4_0 single row scalar tail" {
    // n=1 exercises the single-row fallback path.
    // scale=0.5, all nibbles=10 → weight=2, x=all 1.0
    // y[0] = 0.5 * 32 * 2 = 32.0
    const bpb = backend_mod.q4_0_block_bytes;
    var w: [bpb]u8 = undefined;
    // f16(0.5) = 0x3800 little-endian
    w[0] = 0x00;
    w[1] = 0x38;
    // lo nibble=10, hi nibble=10 → 0xAA
    for (0..16) |i| w[2 + i] = 0xAA;
    var x: [32]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvQ4_0(&x, &w, &y, 1, 32);
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), y[0], 0.1);
}
