//! CPU GEMV kernel for Q5_K quantization.
//! 256 values per super-block, 176 bytes.
//! 2-row batching with V8 SIMD for full sub-blocks.

const std = @import("std");
const quant = @import("../../../ops/quant.zig");
const backend_mod = @import("../../backend.zig");
const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);
const V8u = @Vector(8, u8);
const V8u16 = @Vector(8, u16);
/// Nibble extraction mask: low 4 bits of each byte.
const nib_mask: V8u = @splat(0x0F);
/// Shift amount for high nibble extraction.
const shift4: @Vector(8, u3) = @splat(4);

/// Elements per Q5_K group (super-block / 4 groups).
const group_elems = backend_mod.quant_super_block_elems / 4;
/// Quantized bytes per group (nibble-packed: 2 elements per byte).
const group_qs_bytes = group_elems / 2;
/// Q5_K high-bit contribution: the 5th bit adds 2^4 = 16 to the value.
const q5_k_high_bit_value: f32 = 16.0;
const q5_k_high_bit_int: u8 = 16;

/// Q5_K GEMV: y = W @ x. 2-row batched with V8 SIMD for full sub-blocks.
pub fn gemvQ5_K(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb = backend_mod.q5_k_block_bytes;
    const bs = backend_mod.quant_super_block_elems;
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
            const qh0: [*]const u8 = bp0 + 16;
            const qh1: [*]const u8 = bp1 + 16;
            const qs0 = bp0 + 48;
            const qs1 = bp1 + 48;
            const bk = b * bs;

            // 4 groups of 64 elements each (j=0,64,128,192)
            inline for (0..4) |group| {
                const j = group * group_elems;
                const is = group * 2;
                const umask1: u8 = @as(u8, 1) << (group * 2);
                const umask2: u8 = @as(u8, 2) << (group * 2);
                const ql_off = group * group_qs_bytes;

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
                if (gi_base + group_elems - 1 < k) {
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
                        // Vectorized nibble + high-bit extraction (matches Q4_K pattern)
                        const raw_lo0: V8u = qs0[ql_off + off ..][0..8].*;
                        const raw_lo1: V8u = qs1[ql_off + off ..][0..8].*;
                        const lo_v0: V8 = @floatFromInt(@as(V8u16, @intCast(raw_lo0 & nib_mask)));
                        const lo_v1: V8 = @floatFromInt(@as(V8u16, @intCast(raw_lo1 & nib_mask)));
                        const qh_c0: V8u = qh0[off..][0..8].*;
                        const qh_c1: V8u = qh1[off..][0..8].*;
                        const hi_shr: @Vector(8, u3) = @splat(group * 2);
                        const hi_b0: V8u = (qh_c0 >> hi_shr) & @as(V8u, @splat(1));
                        const hi_b1: V8u = (qh_c1 >> hi_shr) & @as(V8u, @splat(1));
                        const qv0 = lo_v0 + @as(V8, @splat(q5_k_high_bit_value)) * @as(V8, @floatFromInt(@as(V8u16, @intCast(hi_b0))));
                        const qv1 = lo_v1 + @as(V8, @splat(q5_k_high_bit_value)) * @as(V8, @floatFromInt(@as(V8u16, @intCast(hi_b1))));
                        sum0 += xv * (d_sc_a0 * qv0 - dm_m_a0);
                        sum1 += xv * (d_sc_a1 * qv1 - dm_m_a1);
                    }
                    // Second half: l=[0..31], high nibble, scale pair (sc_b, m_b)
                    inline for (0..4) |chunk| {
                        const off = chunk * 8;
                        const xv: V8 = x[gi_base + 32 + off ..][0..8].*;
                        const raw_hi0: V8u = qs0[ql_off + off ..][0..8].*;
                        const raw_hi1: V8u = qs1[ql_off + off ..][0..8].*;
                        const lo_v0: V8 = @floatFromInt(@as(V8u16, @intCast(raw_hi0 >> shift4)));
                        const lo_v1: V8 = @floatFromInt(@as(V8u16, @intCast(raw_hi1 >> shift4)));
                        const qh_c0: V8u = qh0[off..][0..8].*;
                        const qh_c1: V8u = qh1[off..][0..8].*;
                        const hi_shr: @Vector(8, u3) = @splat(group * 2 + 1);
                        const hi_b0: V8u = (qh_c0 >> hi_shr) & @as(V8u, @splat(1));
                        const hi_b1: V8u = (qh_c1 >> hi_shr) & @as(V8u, @splat(1));
                        const qv0 = lo_v0 + @as(V8, @splat(q5_k_high_bit_value)) * @as(V8, @floatFromInt(@as(V8u16, @intCast(hi_b0))));
                        const qv1 = lo_v1 + @as(V8, @splat(q5_k_high_bit_value)) * @as(V8, @floatFromInt(@as(V8u16, @intCast(hi_b1))));
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
                        const q0: f32 = @floatFromInt((qs0[ql_off + l] & 0x0F) + (if ((qh0[l] & umask1) != 0) q5_k_high_bit_int else 0));
                        const q1: f32 = @floatFromInt((qs1[ql_off + l] & 0x0F) + (if ((qh1[l] & umask1) != 0) q5_k_high_bit_int else 0));
                        s0 += x[gi] * (q0 * d1_0 - dm1_0);
                        s1 += x[gi] * (q1 * d1_1 - dm1_1);
                    }
                    for (0..32) |l| {
                        const gi = gi_base + 32 + l;
                        if (gi >= k) break;
                        const q0: f32 = @floatFromInt((qs0[ql_off + l] >> 4) + (if ((qh0[l] & umask2) != 0) q5_k_high_bit_int else 0));
                        const q1: f32 = @floatFromInt((qs1[ql_off + l] >> 4) + (if ((qh1[l] & umask2) != 0) q5_k_high_bit_int else 0));
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
            const qh: [*]const u8 = bp + 16;
            const qs = bp + 48;
            const bk = b * bs;

            inline for (0..4) |group| {
                const j = group * group_elems;
                const is = group * 2;
                const umask1: u8 = @as(u8, 1) << (group * 2);
                const umask2: u8 = @as(u8, 2) << (group * 2);
                const ql_off = group * group_qs_bytes;

                var sc1: u8 = undefined;
                var m1: u8 = undefined;
                var sc2: u8 = undefined;
                var m2: u8 = undefined;
                quant.getScaleMinK4(is + 0, scales, &sc1, &m1);
                quant.getScaleMinK4(is + 1, scales, &sc2, &m2);

                const gi_base = bk + j;
                if (gi_base + group_elems - 1 < k) {
                    const d_sc1: V8 = @splat(d * @as(f32, @floatFromInt(sc1)));
                    const dm_m1: V8 = @splat(dmin * @as(f32, @floatFromInt(m1)));
                    const d_sc2: V8 = @splat(d * @as(f32, @floatFromInt(sc2)));
                    const dm_m2: V8 = @splat(dmin * @as(f32, @floatFromInt(m2)));

                    inline for (0..4) |chunk| {
                        const off = chunk * 8;
                        const xv: V8 = x[gi_base + off ..][0..8].*;
                        const raw_lo: V8u = qs[ql_off + off ..][0..8].*;
                        const lo_v: V8 = @floatFromInt(@as(V8u16, @intCast(raw_lo & nib_mask)));
                        const qh_c: V8u = qh[off..][0..8].*;
                        const hi_shr: @Vector(8, u3) = @splat(group * 2);
                        const hi_b: V8u = (qh_c >> hi_shr) & @as(V8u, @splat(1));
                        const qv = lo_v + @as(V8, @splat(q5_k_high_bit_value)) * @as(V8, @floatFromInt(@as(V8u16, @intCast(hi_b))));
                        sum += xv * (d_sc1 * qv - dm_m1);
                    }
                    inline for (0..4) |chunk| {
                        const off = chunk * 8;
                        const xv: V8 = x[gi_base + 32 + off ..][0..8].*;
                        const raw_hi: V8u = qs[ql_off + off ..][0..8].*;
                        const lo_v: V8 = @floatFromInt(@as(V8u16, @intCast(raw_hi >> shift4)));
                        const qh_c: V8u = qh[off..][0..8].*;
                        const hi_shr: @Vector(8, u3) = @splat(group * 2 + 1);
                        const hi_b: V8u = (qh_c >> hi_shr) & @as(V8u, @splat(1));
                        const qv = lo_v + @as(V8, @splat(q5_k_high_bit_value)) * @as(V8, @floatFromInt(@as(V8u16, @intCast(hi_b))));
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
                        s += x[gi] * (@as(f32, @floatFromInt((qs[ql_off + l] & 0x0F) + (if ((qh[l] & umask1) != 0) q5_k_high_bit_int else 0))) * d1 - dm1);
                    }
                    for (0..32) |l| {
                        const gi = gi_base + 32 + l;
                        if (gi >= k) break;
                        s += x[gi] * (@as(f32, @floatFromInt((qs[ql_off + l] >> 4) + (if ((qh[l] & umask2) != 0) q5_k_high_bit_int else 0))) * d2 - dm2);
                    }
                    sum[0] += s;
                }
            }
        }
        y[row] = @reduce(.Add, sum);
    }
}

test "gemvQ5_K uniform weights" {
    // 2 rows, k=256. d=1.0, dmin=0.0, sc=1, m=0.
    // qs nibbles=1, qh=0 → 5-bit value = 1. x = all 1.0.
    // y = 1.0 * 1 * 256 = 256.0
    const bpb = backend_mod.q5_k_block_bytes; // 176
    var w: [2 * bpb]u8 = undefined;
    for (0..2) |r| {
        const base = r * bpb;
        w[base + 0] = 0x00;
        w[base + 1] = 0x3C; // d = f16(1.0)
        w[base + 2] = 0x00;
        w[base + 3] = 0x00; // dmin = 0
        for (4..16) |i| w[base + i] = 0;
        w[base + 4] = 1;
        w[base + 5] = 1;
        w[base + 6] = 1;
        w[base + 7] = 1;
        w[base + 12] = 1;
        w[base + 13] = 1;
        w[base + 14] = 1;
        w[base + 15] = 1;
        for (16..48) |i| w[base + i] = 0x00; // qh = 0
        for (48..176) |i| w[base + i] = 0x11; // qs: lo=1, hi=1
    }
    const bs = backend_mod.quant_super_block_elems;
    var x: [bs]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [2]f32 = undefined;
    gemvQ5_K(&x, &w, &y, 2, bs);
    for (0..2) |i| try std.testing.expectApproxEqAbs(@as(f32, 256.0), y[i], 0.01);
}

test "gemvQ5_K with high bits" {
    // 2 rows, k=256. d=1.0, dmin=0.0, sc=1, m=0.
    // qs nibbles=1, qh=0xFF → 5-bit value = 1 + 16 = 17.
    // y = 1.0 * 1 * 256 * 17 = 4352.0
    const bpb = backend_mod.q5_k_block_bytes;
    const bs = backend_mod.quant_super_block_elems;
    var w: [2 * bpb]u8 = undefined;
    for (0..2) |r| {
        const base = r * bpb;
        w[base + 0] = 0x00;
        w[base + 1] = 0x3C;
        w[base + 2] = 0x00;
        w[base + 3] = 0x00;
        for (4..16) |i| w[base + i] = 0;
        w[base + 4] = 1;
        w[base + 5] = 1;
        w[base + 6] = 1;
        w[base + 7] = 1;
        w[base + 12] = 1;
        w[base + 13] = 1;
        w[base + 14] = 1;
        w[base + 15] = 1;
        for (16..48) |i| w[base + i] = 0xFF; // qh = all set → +16
        for (48..176) |i| w[base + i] = 0x11;
    }
    var x: [bs]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [2]f32 = undefined;
    gemvQ5_K(&x, &w, &y, 2, bs);
    for (0..2) |i| try std.testing.expectApproxEqAbs(@as(f32, 4352.0), y[i], 0.1);
}

test "gemvQ5_K single row" {
    // n=1 exercises single-row fallback.
    // d=2.0, dmin=0, sc=1, m=0, qs nibbles=1, qh=0.
    // y = 2.0 * 1 * 256 = 512.0
    const bpb = backend_mod.q5_k_block_bytes;
    const bs = backend_mod.quant_super_block_elems;
    var w: [bpb]u8 = undefined;
    w[0] = 0x00;
    w[1] = 0x40; // d = f16(2.0)
    w[2] = 0x00;
    w[3] = 0x00;
    for (4..16) |i| w[i] = 0;
    w[4] = 1;
    w[5] = 1;
    w[6] = 1;
    w[7] = 1;
    w[12] = 1;
    w[13] = 1;
    w[14] = 1;
    w[15] = 1;
    for (16..48) |i| w[i] = 0x00;
    for (48..176) |i| w[i] = 0x11;
    var x: [bs]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvQ5_K(&x, &w, &y, 1, bs);
    try std.testing.expectApproxEqAbs(@as(f32, 512.0), y[0], 0.5);
}
