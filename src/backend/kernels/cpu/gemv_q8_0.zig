//! CPU GEMV kernel for Q8_0 quantization.
//! 32 values per block, 34 bytes (f16 scale + 32 signed bytes).
//! 4-row batching with V8 SIMD and vector byte widening.

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);

/// Widen 8 packed i8 bytes into a V8 float vector via i8→i16→f32 SIMD chain.
inline fn widenI8(raw: @Vector(8, u8)) V8 {
    return @floatFromInt(@as(@Vector(8, i16), @intCast(@as(@Vector(8, i8), @bitCast(raw)))));
}

/// Q8_0 GEMV: y = W @ x. 4-row batched with V8 fused multiply-accumulate.
/// Uses vector accumulators instead of per-chunk scalar reduction,
/// deferring @reduce to once per block.
pub fn gemvQ8_0(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb = backend_mod.q8_0_block_bytes;
    const qk = backend_mod.quant_block_elems;
    const nb = (k + qk - 1) / qk;
    const row_bytes = nb * bpb;

    var row: usize = 0;
    while (row + 4 <= n) : (row += 4) {
        var sum0: f32 = 0.0;
        var sum1: f32 = 0.0;
        var sum2: f32 = 0.0;
        var sum3: f32 = 0.0;
        const rp0 = w + row * row_bytes;
        const rp1 = w + (row + 1) * row_bytes;
        const rp2 = w + (row + 2) * row_bytes;
        const rp3 = w + (row + 3) * row_bytes;

        for (0..nb) |b| {
            const bp0 = rp0 + b * bpb;
            const bp1 = rp1 + b * bpb;
            const bp2 = rp2 + b * bpb;
            const bp3 = rp3 + b * bpb;
            const s0: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp0[0..2], .little))));
            const s1: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp1[0..2], .little))));
            const s2: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp2[0..2], .little))));
            const s3: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp3[0..2], .little))));
            const bk = b * qk;

            // V8 accumulators — defers @reduce to end of block.
            // @mulAdd maps to hardware FMA (single instruction vs mul+add chain).
            var acc0: V8 = v8zero;
            var acc1: V8 = v8zero;
            var acc2: V8 = v8zero;
            var acc3: V8 = v8zero;

            var i: usize = 0;
            while (i + 8 <= qk) : (i += 8) {
                const ki = bk + i;
                if (ki + 7 >= k) break;
                const xv: V8 = x[ki..][0..8].*;

                acc0 = @mulAdd(V8, xv, widenI8(bp0[2 + i ..][0..8].*), acc0);
                acc1 = @mulAdd(V8, xv, widenI8(bp1[2 + i ..][0..8].*), acc1);
                acc2 = @mulAdd(V8, xv, widenI8(bp2[2 + i ..][0..8].*), acc2);
                acc3 = @mulAdd(V8, xv, widenI8(bp3[2 + i ..][0..8].*), acc3);
            }

            // Scalar tail for partial blocks (k not multiple of 32)
            var tail0: f32 = 0.0;
            var tail1: f32 = 0.0;
            var tail2: f32 = 0.0;
            var tail3: f32 = 0.0;
            while (i < qk) : (i += 1) {
                const ki = bk + i;
                if (ki < k) {
                    const xv = x[ki];
                    tail0 += xv * @as(f32, @floatFromInt(@as(i8, @bitCast(bp0[2 + i]))));
                    tail1 += xv * @as(f32, @floatFromInt(@as(i8, @bitCast(bp1[2 + i]))));
                    tail2 += xv * @as(f32, @floatFromInt(@as(i8, @bitCast(bp2[2 + i]))));
                    tail3 += xv * @as(f32, @floatFromInt(@as(i8, @bitCast(bp3[2 + i]))));
                }
            }

            sum0 += (@reduce(.Add, acc0) + tail0) * s0;
            sum1 += (@reduce(.Add, acc1) + tail1) * s1;
            sum2 += (@reduce(.Add, acc2) + tail2) * s2;
            sum3 += (@reduce(.Add, acc3) + tail3) * s3;
        }
        y[row] = sum0;
        y[row + 1] = sum1;
        y[row + 2] = sum2;
        y[row + 3] = sum3;
    }

    // Single-row fallback for remaining rows (n % 4 != 0)
    while (row < n) : (row += 1) {
        var sum: f32 = 0.0;
        const rp = w + row * row_bytes;
        for (0..nb) |b| {
            const bp = rp + b * bpb;
            const s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
            const bk = b * qk;
            var acc: V8 = v8zero;
            var i: usize = 0;
            while (i + 8 <= qk) : (i += 8) {
                const ki = bk + i;
                if (ki + 7 >= k) break;
                const xv: V8 = x[ki..][0..8].*;
                acc = @mulAdd(V8, xv, widenI8(bp[2 + i ..][0..8].*), acc);
            }
            var tail: f32 = 0.0;
            while (i < qk) : (i += 1) {
                const ki = bk + i;
                if (ki < k) tail += x[ki] * @as(f32, @floatFromInt(@as(i8, @bitCast(bp[2 + i]))));
            }
            sum += (@reduce(.Add, acc) + tail) * s;
        }
        y[row] = sum;
    }
}

test "gemvQ8_0 uniform weights" {
    // 4 rows, k=32 (one block per row). scale=1.0, all quant values = 1.
    // x = all 1.0 → each y[i] = 1.0 * 32 * 1 = 32.0
    const bpb = backend_mod.q8_0_block_bytes; // 34
    var w: [4 * bpb]u8 = undefined;
    for (0..4) |r| {
        const base = r * bpb;
        // f16(1.0) = 0x3C00 little-endian
        w[base] = 0x00;
        w[base + 1] = 0x3C;
        for (0..32) |i| w[base + 2 + i] = 1; // i8(1) = u8(1)
    }
    var x: [32]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [4]f32 = undefined;
    gemvQ8_0(&x, &w, &y, 4, 32);
    for (0..4) |i| try std.testing.expectApproxEqAbs(@as(f32, 32.0), y[i], 0.01);
}

test "gemvQ8_0 scale factor" {
    // 1 row, k=32, scale=2.0, all quant values = 3.
    // x = all 1.0 → y[0] = 2.0 * 32 * 3 = 192.0
    const bpb = backend_mod.q8_0_block_bytes;
    var w: [bpb]u8 = undefined;
    // f16(2.0) = 0x4000 little-endian
    w[0] = 0x00;
    w[1] = 0x40;
    for (0..32) |i| w[2 + i] = 3; // i8(3)
    var x: [32]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvQ8_0(&x, &w, &y, 1, 32);
    try std.testing.expectApproxEqAbs(@as(f32, 192.0), y[0], 0.01);
}

test "gemvQ8_0 negative quant values" {
    // 1 row, k=32, scale=1.0, quant values alternate +1/-1.
    // x = all 1.0 → y[0] = 1.0 * (16*1 + 16*(-1)) = 0.0
    const bpb = backend_mod.q8_0_block_bytes;
    var w: [bpb]u8 = undefined;
    w[0] = 0x00;
    w[1] = 0x3C; // f16(1.0)
    for (0..32) |i| {
        // Alternating +1 / -1 as i8: 1 = 0x01, -1 = 0xFF
        w[2 + i] = if (i % 2 == 0) 1 else @bitCast(@as(i8, -1));
    }
    var x: [32]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvQ8_0(&x, &w, &y, 1, 32);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[0], 0.01);
}
