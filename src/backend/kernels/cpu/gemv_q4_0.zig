//! CPU GEMV kernel for Q4_0 quantization.
//! 32 values per block, 18 bytes (f16 scale + 16 nibble-packed bytes).
//! Layout: byte j (j=0..15) contains element j in its low nibble and element j+16 in its high nibble.
//! 4-row batching with V8 SIMD for x-vector cache reuse.

const std = @import("std");
const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);

/// Q4_0 GEMV: y = W @ x. 4-row batched with V8 SIMD for x-vector cache reuse.
pub fn gemvQ4_0(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb: usize = 18;
    const qk: usize = 32;
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

            if (bk + 31 < k) {
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
                    const V8u = @Vector(8, u8);
                    const V8i16 = @Vector(8, i16);
                    const nib_mask: V8u = @splat(0x0F);
                    const bias: V8i16 = @splat(-8);
                    const raw0: V8u = bp[2..10].*;
                    const raw1: V8u = bp[10..18].*;
                    const lo: V8 = @floatFromInt(@as(V8i16, @intCast(raw0 & nib_mask)) + bias);
                    const hi: V8 = @floatFromInt(@as(V8i16, @intCast(raw0 >> @splat(@as(u3, 4)))) + bias);
                    const lo2: V8 = @floatFromInt(@as(V8i16, @intCast(raw1 & nib_mask)) + bias);
                    const hi2: V8 = @floatFromInt(@as(V8i16, @intCast(raw1 >> @splat(@as(u3, 4)))) + bias);
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
                        bs0 += xv * @as(f32, @floatFromInt(@as(i8, @intCast(byte0 & 0x0F)) - 8));
                        bs1 += xv * @as(f32, @floatFromInt(@as(i8, @intCast(byte1 & 0x0F)) - 8));
                        bs2 += xv * @as(f32, @floatFromInt(@as(i8, @intCast(byte2 & 0x0F)) - 8));
                        bs3 += xv * @as(f32, @floatFromInt(@as(i8, @intCast(byte3 & 0x0F)) - 8));
                    }
                    if (gi1 < k) {
                        const xv = x[gi1];
                        bs0 += xv * @as(f32, @floatFromInt(@as(i8, @intCast(byte0 >> 4)) - 8));
                        bs1 += xv * @as(f32, @floatFromInt(@as(i8, @intCast(byte1 >> 4)) - 8));
                        bs2 += xv * @as(f32, @floatFromInt(@as(i8, @intCast(byte2 >> 4)) - 8));
                        bs3 += xv * @as(f32, @floatFromInt(@as(i8, @intCast(byte3 >> 4)) - 8));
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
            if (bk + 31 < k) {
                const x_lo: V8 = x[bk..][0..8].*;
                const x_lo2: V8 = x[bk + 8 ..][0..8].*;
                const x_hi: V8 = x[bk + 16 ..][0..8].*;
                const x_hi2: V8 = x[bk + 24 ..][0..8].*;
                var lo: V8 = undefined;
                var lo2: V8 = undefined;
                var hi: V8 = undefined;
                var hi2: V8 = undefined;
                inline for (0..8) |idx| {
                    const byte = bp[2 + idx];
                    lo[idx] = @floatFromInt(@as(i8, @intCast(byte & 0x0F)) - 8);
                    hi[idx] = @floatFromInt(@as(i8, @intCast(byte >> 4)) - 8);
                }
                inline for (0..8) |idx| {
                    const byte = bp[2 + 8 + idx];
                    lo2[idx] = @floatFromInt(@as(i8, @intCast(byte & 0x0F)) - 8);
                    hi2[idx] = @floatFromInt(@as(i8, @intCast(byte >> 4)) - 8);
                }
                block_sum = @reduce(.Add, x_lo * lo + x_lo2 * lo2 + x_hi * hi + x_hi2 * hi2);
            } else {
                for (0..qk / 2) |j| {
                    const byte = bp[2 + j];
                    const x0 = @as(i8, @intCast(byte & 0x0F)) - 8;
                    const x1 = @as(i8, @intCast(byte >> 4)) - 8;
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
