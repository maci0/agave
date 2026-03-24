//! CPU GEMV kernel for Q4_K quantization.
//! 256 values per super-block, 144 bytes: d(f16) + dmin(f16) + scales[12] + qs[128].
//! Each super-block has 4 groups of 64 elements, each group stored in 32 bytes:
//!   - Elements 0-31: LOW nibbles of bytes 0-31 (scale index 2g)
//!   - Elements 32-63: HIGH nibbles of bytes 0-31 (scale index 2g+1)
//! 2-row batching with V8 SIMD, using factored scale/min accumulation.

const std = @import("std");
const quant = @import("../../../ops/quant.zig");
const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);

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

            // 4 groups of 64 elements: low nibbles (32 elems) then high nibbles (32 elems)
            inline for (0..4) |g| {
                const ql_off = g * 32;
                const gi_lo = bk + g * 64; // first 32 elements of group
                const gi_hi = gi_lo + 32; // next 32 elements of group

                var sc_lo0: u8 = undefined;
                var m_lo0: u8 = undefined;
                var sc_hi0: u8 = undefined;
                var m_hi0: u8 = undefined;
                var sc_lo1: u8 = undefined;
                var m_lo1: u8 = undefined;
                var sc_hi1: u8 = undefined;
                var m_hi1: u8 = undefined;
                quant.getScaleMinK4(g * 2 + 0, scales0, &sc_lo0, &m_lo0);
                quant.getScaleMinK4(g * 2 + 1, scales0, &sc_hi0, &m_hi0);
                quant.getScaleMinK4(g * 2 + 0, scales1, &sc_lo1, &m_lo1);
                quant.getScaleMinK4(g * 2 + 1, scales1, &sc_hi1, &m_hi1);

                if (gi_lo + 63 < k) {
                    // Full group — vectorized
                    // Low nibbles: elements gi_lo..gi_lo+31
                    var q_lo0: V8 = v8zero;
                    var q_lo1: V8 = v8zero;
                    var x_lo_acc: V8 = v8zero;
                    inline for (0..4) |chunk| {
                        const off = chunk * 8;
                        const xv: V8 = x[gi_lo + off ..][0..8].*;
                        var qv0: V8 = undefined;
                        var qv1: V8 = undefined;
                        inline for (0..8) |idx| {
                            qv0[idx] = @floatFromInt(qs0[ql_off + off + idx] & 0x0F);
                            qv1[idx] = @floatFromInt(qs1[ql_off + off + idx] & 0x0F);
                        }
                        q_lo0 = @mulAdd(V8, xv, qv0, q_lo0);
                        q_lo1 = @mulAdd(V8, xv, qv1, q_lo1);
                        x_lo_acc += xv;
                    }
                    const d_sc_lo0 = d_0 * @as(f32, @floatFromInt(sc_lo0));
                    const d_sc_lo1 = d_1 * @as(f32, @floatFromInt(sc_lo1));
                    const dm_m_lo0 = dmin_0 * @as(f32, @floatFromInt(m_lo0));
                    const dm_m_lo1 = dmin_1 * @as(f32, @floatFromInt(m_lo1));
                    const x_lo_sum = @reduce(.Add, x_lo_acc);
                    sum0 += @reduce(.Add, q_lo0) * d_sc_lo0 - x_lo_sum * dm_m_lo0;
                    sum1 += @reduce(.Add, q_lo1) * d_sc_lo1 - x_lo_sum * dm_m_lo1;

                    // High nibbles: elements gi_hi..gi_hi+31
                    var q_hi0: V8 = v8zero;
                    var q_hi1: V8 = v8zero;
                    var x_hi_acc: V8 = v8zero;
                    inline for (0..4) |chunk| {
                        const off = chunk * 8;
                        const xv: V8 = x[gi_hi + off ..][0..8].*;
                        var qv0: V8 = undefined;
                        var qv1: V8 = undefined;
                        inline for (0..8) |idx| {
                            qv0[idx] = @floatFromInt(qs0[ql_off + off + idx] >> 4);
                            qv1[idx] = @floatFromInt(qs1[ql_off + off + idx] >> 4);
                        }
                        q_hi0 = @mulAdd(V8, xv, qv0, q_hi0);
                        q_hi1 = @mulAdd(V8, xv, qv1, q_hi1);
                        x_hi_acc += xv;
                    }
                    const d_sc_hi0 = d_0 * @as(f32, @floatFromInt(sc_hi0));
                    const d_sc_hi1 = d_1 * @as(f32, @floatFromInt(sc_hi1));
                    const dm_m_hi0 = dmin_0 * @as(f32, @floatFromInt(m_hi0));
                    const dm_m_hi1 = dmin_1 * @as(f32, @floatFromInt(m_hi1));
                    const x_hi_sum = @reduce(.Add, x_hi_acc);
                    sum0 += @reduce(.Add, q_hi0) * d_sc_hi0 - x_hi_sum * dm_m_hi0;
                    sum1 += @reduce(.Add, q_hi1) * d_sc_hi1 - x_hi_sum * dm_m_hi1;
                } else {
                    // Partial group — scalar fallback
                    var s0: f32 = 0.0;
                    var s1: f32 = 0.0;
                    const d1_0 = d_0 * @as(f32, @floatFromInt(sc_lo0));
                    const dm1_0 = dmin_0 * @as(f32, @floatFromInt(m_lo0));
                    const d2_0 = d_0 * @as(f32, @floatFromInt(sc_hi0));
                    const dm2_0 = dmin_0 * @as(f32, @floatFromInt(m_hi0));
                    const d1_1 = d_1 * @as(f32, @floatFromInt(sc_lo1));
                    const dm1_1 = dmin_1 * @as(f32, @floatFromInt(m_lo1));
                    const d2_1 = d_1 * @as(f32, @floatFromInt(sc_hi1));
                    const dm2_1 = dmin_1 * @as(f32, @floatFromInt(m_hi1));
                    for (0..32) |l| {
                        const gi = gi_lo + l;
                        if (gi >= k) break;
                        const q0: f32 = @floatFromInt(qs0[ql_off + l] & 0x0F);
                        const q1: f32 = @floatFromInt(qs1[ql_off + l] & 0x0F);
                        s0 += x[gi] * (d1_0 * q0 - dm1_0);
                        s1 += x[gi] * (d1_1 * q1 - dm1_1);
                    }
                    for (0..32) |l| {
                        const gi = gi_hi + l;
                        if (gi >= k) break;
                        const q0: f32 = @floatFromInt(qs0[ql_off + l] >> 4);
                        const q1: f32 = @floatFromInt(qs1[ql_off + l] >> 4);
                        s0 += x[gi] * (d2_0 * q0 - dm2_0);
                        s1 += x[gi] * (d2_1 * q1 - dm2_1);
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

            inline for (0..4) |g| {
                const ql_off = g * 32;
                const gi_lo = bk + g * 64;
                const gi_hi = gi_lo + 32;

                var sc_lo: u8 = undefined;
                var m_lo: u8 = undefined;
                var sc_hi: u8 = undefined;
                var m_hi: u8 = undefined;
                quant.getScaleMinK4(g * 2 + 0, scales, &sc_lo, &m_lo);
                quant.getScaleMinK4(g * 2 + 1, scales, &sc_hi, &m_hi);

                if (gi_lo + 63 < k) {
                    var q_lo: V8 = v8zero;
                    var x_lo_acc: V8 = v8zero;
                    inline for (0..4) |chunk| {
                        const off = chunk * 8;
                        const xv: V8 = x[gi_lo + off ..][0..8].*;
                        var qv: V8 = undefined;
                        inline for (0..8) |idx| {
                            qv[idx] = @floatFromInt(qs[ql_off + off + idx] & 0x0F);
                        }
                        q_lo = @mulAdd(V8, xv, qv, q_lo);
                        x_lo_acc += xv;
                    }
                    const d_sc_lo = d * @as(f32, @floatFromInt(sc_lo));
                    const dm_m_lo = dmin * @as(f32, @floatFromInt(m_lo));
                    sum += @reduce(.Add, q_lo) * d_sc_lo - @reduce(.Add, x_lo_acc) * dm_m_lo;

                    var q_hi: V8 = v8zero;
                    var x_hi_acc: V8 = v8zero;
                    inline for (0..4) |chunk| {
                        const off = chunk * 8;
                        const xv: V8 = x[gi_hi + off ..][0..8].*;
                        var qv: V8 = undefined;
                        inline for (0..8) |idx| {
                            qv[idx] = @floatFromInt(qs[ql_off + off + idx] >> 4);
                        }
                        q_hi = @mulAdd(V8, xv, qv, q_hi);
                        x_hi_acc += xv;
                    }
                    const d_sc_hi = d * @as(f32, @floatFromInt(sc_hi));
                    const dm_m_hi = dmin * @as(f32, @floatFromInt(m_hi));
                    sum += @reduce(.Add, q_hi) * d_sc_hi - @reduce(.Add, x_hi_acc) * dm_m_hi;
                } else {
                    var s: f32 = 0.0;
                    const d1 = d * @as(f32, @floatFromInt(sc_lo));
                    const dm1 = dmin * @as(f32, @floatFromInt(m_lo));
                    const d2 = d * @as(f32, @floatFromInt(sc_hi));
                    const dm2 = dmin * @as(f32, @floatFromInt(m_hi));
                    for (0..32) |l| {
                        const gi = gi_lo + l;
                        if (gi >= k) break;
                        s += x[gi] * (@as(f32, @floatFromInt(qs[ql_off + l] & 0x0F)) * d1 - dm1);
                    }
                    for (0..32) |l| {
                        const gi = gi_hi + l;
                        if (gi >= k) break;
                        s += x[gi] * (@as(f32, @floatFromInt(qs[ql_off + l] >> 4)) * d2 - dm2);
                    }
                    sum += s;
                }
            }
        }
        y[row] = sum;
    }
}
