//! CPU GEMV kernel for Q5_K quantization.
//! 256 values per super-block, 176 bytes.
//! 2-row batching with V8 SIMD for full sub-blocks.

const std = @import("std");
const quant = @import("../../../ops/quant.zig");
const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);

/// Q5_K GEMV: y = W @ x. 2-row batched with V8 SIMD for full sub-blocks.
pub fn gemvQ5_K(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb: usize = 176;
    const bs: usize = 256;
    const nb = (k + bs - 1) / bs;
    const row_bytes = nb * bpb;

    // Process 2 rows at a time for x-vector cache reuse.
    var row: usize = 0;
    while (row + 2 <= n) : (row += 2) {
        var sum0: V8 = v8zero;
        var sum1: V8 = v8zero;
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
            const qh0 = bp0[16..48];
            const qh1 = bp1[16..48];
            const qs0 = bp0 + 48;
            const qs1 = bp1 + 48;
            const bk = b * bs;

            // 4 groups of 64 elements each (j=0,64,128,192)
            inline for (0..4) |group| {
                const j = group * 64;
                const is = group * 2;
                const umask1: u8 = @as(u8, 1) << (group * 2);
                const umask2: u8 = @as(u8, 2) << (group * 2);
                const ql_off = group * 32;

                var sc_a0: u8 = undefined;
                var m_a0: u8 = undefined;
                var sc_b0: u8 = undefined;
                var m_b0: u8 = undefined;
                var sc_a1: u8 = undefined;
                var m_a1: u8 = undefined;
                var sc_b1: u8 = undefined;
                var m_b1: u8 = undefined;
                quant.getScaleMinK4(is + 0, scales0, &sc_a0, &m_a0);
                quant.getScaleMinK4(is + 1, scales0, &sc_b0, &m_b0);
                quant.getScaleMinK4(is + 0, scales1, &sc_a1, &m_a1);
                quant.getScaleMinK4(is + 1, scales1, &sc_b1, &m_b1);

                const gi_base = bk + j;
                if (gi_base + 63 < k) {
                    // Full sub-block — vectorized (first 32: low nibble, next 32: high nibble)
                    const d_sc_a0: V8 = @splat(d_0 * @as(f32, @floatFromInt(sc_a0)));
                    const dm_m_a0: V8 = @splat(dmin_0 * @as(f32, @floatFromInt(m_a0)));
                    const d_sc_b0: V8 = @splat(d_0 * @as(f32, @floatFromInt(sc_b0)));
                    const dm_m_b0: V8 = @splat(dmin_0 * @as(f32, @floatFromInt(m_b0)));
                    const d_sc_a1: V8 = @splat(d_1 * @as(f32, @floatFromInt(sc_a1)));
                    const dm_m_a1: V8 = @splat(dmin_1 * @as(f32, @floatFromInt(m_a1)));
                    const d_sc_b1: V8 = @splat(d_1 * @as(f32, @floatFromInt(sc_b1)));
                    const dm_m_b1: V8 = @splat(dmin_1 * @as(f32, @floatFromInt(m_b1)));

                    // First half: l=[0..31], low nibble, scale pair (sc_a, m_a)
                    inline for (0..4) |chunk| {
                        const off = chunk * 8;
                        const xv: V8 = x[gi_base + off ..][0..8].*;
                        var qv0: V8 = undefined;
                        var qv1: V8 = undefined;
                        inline for (0..8) |idx| {
                            const l = off + idx;
                            const lo0: f32 = @floatFromInt(qs0[ql_off + l] & 0x0F);
                            const hi0: f32 = if ((qh0[l] & umask1) != 0) @as(f32, 16.0) else 0.0;
                            qv0[idx] = lo0 + hi0;
                            const lo1: f32 = @floatFromInt(qs1[ql_off + l] & 0x0F);
                            const hi1: f32 = if ((qh1[l] & umask1) != 0) @as(f32, 16.0) else 0.0;
                            qv1[idx] = lo1 + hi1;
                        }
                        sum0 += xv * (d_sc_a0 * qv0 - dm_m_a0);
                        sum1 += xv * (d_sc_a1 * qv1 - dm_m_a1);
                    }
                    // Second half: l=[0..31], high nibble, scale pair (sc_b, m_b)
                    inline for (0..4) |chunk| {
                        const off = chunk * 8;
                        const xv: V8 = x[gi_base + 32 + off ..][0..8].*;
                        var qv0: V8 = undefined;
                        var qv1: V8 = undefined;
                        inline for (0..8) |idx| {
                            const l = off + idx;
                            const lo0: f32 = @floatFromInt(qs0[ql_off + l] >> 4);
                            const hi0: f32 = if ((qh0[l] & umask2) != 0) @as(f32, 16.0) else 0.0;
                            qv0[idx] = lo0 + hi0;
                            const lo1: f32 = @floatFromInt(qs1[ql_off + l] >> 4);
                            const hi1: f32 = if ((qh1[l] & umask2) != 0) @as(f32, 16.0) else 0.0;
                            qv1[idx] = lo1 + hi1;
                        }
                        sum0 += xv * (d_sc_b0 * qv0 - dm_m_b0);
                        sum1 += xv * (d_sc_b1 * qv1 - dm_m_b1);
                    }
                } else {
                    // Partial sub-block — scalar fallback
                    var s0: f32 = 0.0;
                    var s1: f32 = 0.0;
                    const d1_0 = d_0 * @as(f32, @floatFromInt(sc_a0));
                    const dm1_0 = dmin_0 * @as(f32, @floatFromInt(m_a0));
                    const d2_0 = d_0 * @as(f32, @floatFromInt(sc_b0));
                    const dm2_0 = dmin_0 * @as(f32, @floatFromInt(m_b0));
                    const d1_1 = d_1 * @as(f32, @floatFromInt(sc_a1));
                    const dm1_1 = dmin_1 * @as(f32, @floatFromInt(m_a1));
                    const d2_1 = d_1 * @as(f32, @floatFromInt(sc_b1));
                    const dm2_1 = dmin_1 * @as(f32, @floatFromInt(m_b1));
                    for (0..32) |l| {
                        const gi = gi_base + l;
                        if (gi >= k) break;
                        const q0: f32 = @floatFromInt((qs0[ql_off + l] & 0x0F) + (if ((qh0[l] & umask1) != 0) @as(u8, 16) else 0));
                        const q1: f32 = @floatFromInt((qs1[ql_off + l] & 0x0F) + (if ((qh1[l] & umask1) != 0) @as(u8, 16) else 0));
                        s0 += x[gi] * (q0 * d1_0 - dm1_0);
                        s1 += x[gi] * (q1 * d1_1 - dm1_1);
                    }
                    for (0..32) |l| {
                        const gi = gi_base + 32 + l;
                        if (gi >= k) break;
                        const q0: f32 = @floatFromInt((qs0[ql_off + l] >> 4) + (if ((qh0[l] & umask2) != 0) @as(u8, 16) else 0));
                        const q1: f32 = @floatFromInt((qs1[ql_off + l] >> 4) + (if ((qh1[l] & umask2) != 0) @as(u8, 16) else 0));
                        s0 += x[gi] * (q0 * d2_0 - dm2_0);
                        s1 += x[gi] * (q1 * d2_1 - dm2_1);
                    }
                    sum0[0] += s0;
                    sum1[0] += s1;
                }
            }
        }
        y[row] = @reduce(.Add, sum0);
        y[row + 1] = @reduce(.Add, sum1);
    }

    // Remainder: single row with SIMD
    while (row < n) : (row += 1) {
        var sum: V8 = v8zero;
        const rp = w + row * row_bytes;
        for (0..nb) |b| {
            const bp = rp + b * bpb;
            const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
            const dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[2..4], .little))));
            const scales = bp[4..16];
            const qh = bp[16..48];
            const qs = bp + 48;
            const bk = b * bs;

            inline for (0..4) |group| {
                const j = group * 64;
                const is = group * 2;
                const umask1: u8 = @as(u8, 1) << (group * 2);
                const umask2: u8 = @as(u8, 2) << (group * 2);
                const ql_off = group * 32;

                var sc1: u8 = undefined;
                var m1: u8 = undefined;
                var sc2: u8 = undefined;
                var m2: u8 = undefined;
                quant.getScaleMinK4(is + 0, scales, &sc1, &m1);
                quant.getScaleMinK4(is + 1, scales, &sc2, &m2);

                const gi_base = bk + j;
                if (gi_base + 63 < k) {
                    const d_sc1: V8 = @splat(d * @as(f32, @floatFromInt(sc1)));
                    const dm_m1: V8 = @splat(dmin * @as(f32, @floatFromInt(m1)));
                    const d_sc2: V8 = @splat(d * @as(f32, @floatFromInt(sc2)));
                    const dm_m2: V8 = @splat(dmin * @as(f32, @floatFromInt(m2)));

                    inline for (0..4) |chunk| {
                        const off = chunk * 8;
                        const xv: V8 = x[gi_base + off ..][0..8].*;
                        var qv: V8 = undefined;
                        inline for (0..8) |idx| {
                            const l = off + idx;
                            const lo: f32 = @floatFromInt(qs[ql_off + l] & 0x0F);
                            const hi: f32 = if ((qh[l] & umask1) != 0) @as(f32, 16.0) else 0.0;
                            qv[idx] = lo + hi;
                        }
                        sum += xv * (d_sc1 * qv - dm_m1);
                    }
                    inline for (0..4) |chunk| {
                        const off = chunk * 8;
                        const xv: V8 = x[gi_base + 32 + off ..][0..8].*;
                        var qv: V8 = undefined;
                        inline for (0..8) |idx| {
                            const l = off + idx;
                            const lo: f32 = @floatFromInt(qs[ql_off + l] >> 4);
                            const hi: f32 = if ((qh[l] & umask2) != 0) @as(f32, 16.0) else 0.0;
                            qv[idx] = lo + hi;
                        }
                        sum += xv * (d_sc2 * qv - dm_m2);
                    }
                } else {
                    var s: f32 = 0.0;
                    const d1 = d * @as(f32, @floatFromInt(sc1));
                    const dm1 = dmin * @as(f32, @floatFromInt(m1));
                    const d2 = d * @as(f32, @floatFromInt(sc2));
                    const dm2 = dmin * @as(f32, @floatFromInt(m2));
                    for (0..32) |l| {
                        const gi = gi_base + l;
                        if (gi >= k) break;
                        s += x[gi] * (@as(f32, @floatFromInt((qs[ql_off + l] & 0x0F) + (if ((qh[l] & umask1) != 0) @as(u8, 16) else 0))) * d1 - dm1);
                    }
                    for (0..32) |l| {
                        const gi = gi_base + 32 + l;
                        if (gi >= k) break;
                        s += x[gi] * (@as(f32, @floatFromInt((qs[ql_off + l] >> 4) + (if ((qh[l] & umask2) != 0) @as(u8, 16) else 0))) * d2 - dm2);
                    }
                    sum[0] += s;
                }
            }
        }
        y[row] = @reduce(.Add, sum);
    }
}
