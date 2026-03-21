//! CPU GEMV kernel for Q8_0 quantization.
//! 32 values per block, 34 bytes (f16 scale + 32 signed bytes).
//! 4-row batching with V8 SIMD and vector byte widening.

const std = @import("std");
const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);

/// Widen 8 packed i8 bytes into a V8 float vector via i8→i16→f32 SIMD chain.
inline fn widenI8(raw: @Vector(8, u8)) V8 {
    return @floatFromInt(@as(@Vector(8, i16), @intCast(@as(@Vector(8, i8), @bitCast(raw)))));
}

/// Q8_0 GEMV: y = W @ x. 4-row batched with V8 fused multiply-accumulate.
/// Uses vector accumulators (maps to NEON fmla) instead of per-chunk scalar
/// reduction, deferring @reduce to once per block for ~3× fewer reductions.
pub fn gemvQ8_0(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb: usize = 34;
    const nb = (k + 31) / 32;
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
            const bk = b * 32;

            // V8 accumulators — defers @reduce to end of block.
            // @mulAdd maps to NEON fmla (1 instruction vs fmul+faddp chain).
            var acc0: V8 = v8zero;
            var acc1: V8 = v8zero;
            var acc2: V8 = v8zero;
            var acc3: V8 = v8zero;

            var i: usize = 0;
            while (i + 8 <= 32) : (i += 8) {
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
            while (i < 32) : (i += 1) {
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
            const bk = b * 32;
            var acc: V8 = v8zero;
            var i: usize = 0;
            while (i + 8 <= 32) : (i += 8) {
                const ki = bk + i;
                if (ki + 7 >= k) break;
                const xv: V8 = x[ki..][0..8].*;
                acc = @mulAdd(V8, xv, widenI8(bp[2 + i ..][0..8].*), acc);
            }
            var tail: f32 = 0.0;
            while (i < 32) : (i += 1) {
                const ki = bk + i;
                if (ki < k) tail += x[ki] * @as(f32, @floatFromInt(@as(i8, @bitCast(bp[2 + i]))));
            }
            sum += (@reduce(.Add, acc) + tail) * s;
        }
        y[row] = sum;
    }
}
