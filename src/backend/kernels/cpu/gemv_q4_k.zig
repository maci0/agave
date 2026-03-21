//! CPU GEMV kernel for Q4_K quantization.
//! 256 values per super-block, 144 bytes: d(f16) + dmin(f16) + scales[12] + qs[128].
//! 2-row batching with V8 SIMD.

const std = @import("std");
const quant = @import("../../../ops/quant.zig");
const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);

/// Extract 8 Q4_K nibbles from 4 consecutive bytes as a V8 float vector.
/// Bytes are packed as [lo0,hi0,lo1,hi1,...] — alternating low/high nibbles.
inline fn extractNibbles4(qs: [*]const u8, byte_off: usize) V8 {
    const b0 = qs[byte_off];
    const b1 = qs[byte_off + 1];
    const b2 = qs[byte_off + 2];
    const b3 = qs[byte_off + 3];
    return .{
        @floatFromInt(b0 & 0x0F), @floatFromInt(b0 >> 4),
        @floatFromInt(b1 & 0x0F), @floatFromInt(b1 >> 4),
        @floatFromInt(b2 & 0x0F), @floatFromInt(b2 >> 4),
        @floatFromInt(b3 & 0x0F), @floatFromInt(b3 >> 4),
    };
}

/// Q4_K GEMV: y = W @ x. 2-row batched with V8 fused multiply-accumulate.
/// Uses factored scale/min: dot(x, d*q - dm) = d*dot(x,q) - dm*sum(x),
/// accumulating q_dot and x_sum per sub-block, then applying scales once.
pub fn gemvQ4_K(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb: usize = 144;
    const bs: usize = 256;
    const nb = (k + bs - 1) / bs;
    const row_bytes = nb * bpb;

    // Process 2 rows at a time for x-vector cache reuse.
    var row: usize = 0;
    while (row + 2 <= n) : (row += 2) {
        var sum0: f32 = 0.0;
        var sum1: f32 = 0.0;
        const rp0 = w + row * row_bytes;
        const rp1 = w + (row + 1) * row_bytes;

        for (0..nb) |b| {
            const bp0 = rp0 + b * bpb;
            const bp1 = rp1 + b * bpb;
            const d_0: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp0[0..2], .little))));
            const d_1: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp1[0..2], .little))));
            const dmin_0: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp0[2..4], .little))));
            const dmin_1: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp1[2..4], .little))));
            const scales0 = bp0[4..16];
            const scales1 = bp1[4..16];
            const qs0 = bp0 + 16;
            const qs1 = bp1 + 16;
            const bk = b * bs;

            for (0..8) |sb| {
                const gi_base = bk + sb * 32;
                if (gi_base >= k) break;

                var sc0: u8 = undefined;
                var m0: u8 = undefined;
                var sc1: u8 = undefined;
                var m1: u8 = undefined;
                quant.getScaleMinK4(sb, scales0, &sc0, &m0);
                quant.getScaleMinK4(sb, scales1, &sc1, &m1);
                const qi_base = sb * 16; // byte offset = sb*32/2

                if (gi_base + 31 < k) {
                    // Factored: accumulate x·q and sum(x) per sub-block,
                    // then apply scale/min once. @mulAdd → NEON fmla.
                    var q_acc0: V8 = v8zero;
                    var q_acc1: V8 = v8zero;
                    var x_acc: V8 = v8zero;
                    inline for (0..4) |chunk| {
                        const off = chunk * 8;
                        const xv: V8 = x[gi_base + off ..][0..8].*;
                        q_acc0 = @mulAdd(V8, xv, extractNibbles4(qs0, qi_base + chunk * 4), q_acc0);
                        q_acc1 = @mulAdd(V8, xv, extractNibbles4(qs1, qi_base + chunk * 4), q_acc1);
                        x_acc += xv;
                    }
                    const d_sc0 = d_0 * @as(f32, @floatFromInt(sc0));
                    const d_sc1 = d_1 * @as(f32, @floatFromInt(sc1));
                    const dm_m0 = dmin_0 * @as(f32, @floatFromInt(m0));
                    const dm_m1 = dmin_1 * @as(f32, @floatFromInt(m1));
                    const x_sum = @reduce(.Add, x_acc);
                    sum0 += @reduce(.Add, q_acc0) * d_sc0 - x_sum * dm_m0;
                    sum1 += @reduce(.Add, q_acc1) * d_sc1 - x_sum * dm_m1;
                } else {
                    var s0: f32 = 0.0;
                    var s1: f32 = 0.0;
                    const d_sc0 = d_0 * @as(f32, @floatFromInt(sc0));
                    const d_sc1 = d_1 * @as(f32, @floatFromInt(sc1));
                    const dm_m0 = dmin_0 * @as(f32, @floatFromInt(m0));
                    const dm_m1 = dmin_1 * @as(f32, @floatFromInt(m1));
                    for (0..32) |l| {
                        const gi = gi_base + l;
                        if (gi >= k) break;
                        const byte_idx = qi_base + l / 2;
                        const qv0: f32 = @floatFromInt(if (l % 2 == 0) qs0[byte_idx] & 0x0F else qs0[byte_idx] >> 4);
                        const qv1: f32 = @floatFromInt(if (l % 2 == 0) qs1[byte_idx] & 0x0F else qs1[byte_idx] >> 4);
                        s0 += x[gi] * (d_sc0 * qv0 - dm_m0);
                        s1 += x[gi] * (d_sc1 * qv1 - dm_m1);
                    }
                    sum0 += s0;
                    sum1 += s1;
                }
            }
        }
        y[row] = sum0;
        y[row + 1] = sum1;
    }

    // Remainder: single row
    while (row < n) : (row += 1) {
        var sum: f32 = 0.0;
        const rp = w + row * row_bytes;
        for (0..nb) |b| {
            const bp = rp + b * bpb;
            const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
            const dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[2..4], .little))));
            const scales = bp[4..16];
            const qs = bp + 16;
            const bk = b * bs;

            for (0..8) |sb| {
                const gi_base = bk + sb * 32;
                if (gi_base >= k) break;
                var sc: u8 = undefined;
                var m: u8 = undefined;
                quant.getScaleMinK4(sb, scales, &sc, &m);
                const qi_base = sb * 16;

                if (gi_base + 31 < k) {
                    var q_acc: V8 = v8zero;
                    var x_acc: V8 = v8zero;
                    inline for (0..4) |chunk| {
                        const off = chunk * 8;
                        const xv: V8 = x[gi_base + off ..][0..8].*;
                        q_acc = @mulAdd(V8, xv, extractNibbles4(qs, qi_base + chunk * 4), q_acc);
                        x_acc += xv;
                    }
                    const d_sc = d * @as(f32, @floatFromInt(sc));
                    const dm_m = dmin * @as(f32, @floatFromInt(m));
                    sum += @reduce(.Add, q_acc) * d_sc - @reduce(.Add, x_acc) * dm_m;
                } else {
                    var s: f32 = 0.0;
                    const d_sc = d * @as(f32, @floatFromInt(sc));
                    const dm_m = dmin * @as(f32, @floatFromInt(m));
                    for (0..32) |l| {
                        const gi = gi_base + l;
                        if (gi >= k) break;
                        const byte_idx = qi_base + l / 2;
                        const q: f32 = @floatFromInt(if (l % 2 == 0) qs[byte_idx] & 0x0F else qs[byte_idx] >> 4);
                        s += x[gi] * (d_sc * q - dm_m);
                    }
                    sum += s;
                }
            }
        }
        y[row] = sum;
    }
}
