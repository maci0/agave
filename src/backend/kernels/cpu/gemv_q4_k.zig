//! CPU GEMV kernel for Q4_K quantization.
//! 256 values per super-block, 144 bytes: d(f16) + dmin(f16) + scales[12] + qs[128].
//! Each super-block has 4 groups of 64 elements, each group stored in 32 bytes:
//!   - Elements 0-31: LOW nibbles of bytes 0-31 (scale index 2g)
//!   - Elements 32-63: HIGH nibbles of bytes 0-31 (scale index 2g+1)
//! 2-row batching with V8 SIMD, using factored scale/min accumulation.

const std = @import("std");
const quant = @import("../../../ops/quant.zig");
const backend_mod = @import("../../backend.zig");
const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);
const V8u = @Vector(8, u8);
const V8u16 = @Vector(8, u16);
/// Elements per Q4_K group (super-block / 4 groups).
const group_elems = backend_mod.quant_super_block_elems / 4;
/// Quantized bytes per group (nibble-packed: 2 elements per byte).
const group_qs_bytes = group_elems / 2;
/// Nibble extraction mask: low 4 bits of each byte.
const nib_mask: V8u = @splat(0x0F);
/// Shift amount for high nibble extraction.
const shift4: @Vector(8, u3) = @splat(4);

/// Q4_K GEMV: y = W @ x. 2-row batched with V8 fused multiply-accumulate.
/// Uses factored scale/min: dot(x, d*q - dm) = d*dot(x,q) - dm*sum(x),
/// accumulating q_dot and x_sum per sub-block, then applying scales once.
pub fn gemvQ4_K(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb = backend_mod.q4_k_block_bytes;
    const bs = backend_mod.quant_super_block_elems;
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
                const ql_off = g * group_qs_bytes;
                const gi_lo = bk + g * group_elems; // first 32 elements of group
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

                if (gi_lo + group_elems - 1 < k) {
                    // Full group — vectorized
                    // Low nibbles: elements gi_lo..gi_lo+31
                    var q_lo0: V8 = v8zero;
                    var q_lo1: V8 = v8zero;
                    var x_lo_acc: V8 = v8zero;
                    inline for (0..4) |chunk| {
                        const off = chunk * 8;
                        const xv: V8 = x[gi_lo + off ..][0..8].*;
                        const raw_lo0: V8u = qs0[ql_off + off ..][0..8].*;
                        const raw_lo1: V8u = qs1[ql_off + off ..][0..8].*;
                        const qv0: V8 = @floatFromInt(@as(V8u16, @intCast(raw_lo0 & nib_mask)));
                        const qv1: V8 = @floatFromInt(@as(V8u16, @intCast(raw_lo1 & nib_mask)));
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
                        const raw_hi0: V8u = qs0[ql_off + off ..][0..8].*;
                        const raw_hi1: V8u = qs1[ql_off + off ..][0..8].*;
                        const qv0: V8 = @floatFromInt(@as(V8u16, @intCast(raw_hi0 >> shift4)));
                        const qv1: V8 = @floatFromInt(@as(V8u16, @intCast(raw_hi1 >> shift4)));
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

                if (gi_lo + group_elems - 1 < k) {
                    var q_lo: V8 = v8zero;
                    var x_lo_acc: V8 = v8zero;
                    inline for (0..4) |chunk| {
                        const off = chunk * 8;
                        const xv: V8 = x[gi_lo + off ..][0..8].*;
                        const raw_lo: V8u = qs[ql_off + off ..][0..8].*;
                        const qv: V8 = @floatFromInt(@as(V8u16, @intCast(raw_lo & nib_mask)));
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
                        const raw_hi: V8u = qs[ql_off + off ..][0..8].*;
                        const qv: V8 = @floatFromInt(@as(V8u16, @intCast(raw_hi >> shift4)));
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

test "gemvQ4_K uniform weights" {
    // 2 rows, k=256 (one super-block per row). d=1.0, dmin=0.0.
    // scales: sc=1 for all 8 group indices, m=0.
    // All qs nibbles = 1 → weight = 1 per element, x = all 1.0.
    // y = d * sc * sum(q) = 1.0 * 1 * 256 = 256.0
    const bpb = backend_mod.q4_k_block_bytes; // 144
    var w: [2 * bpb]u8 = undefined;
    for (0..2) |r| {
        const base = r * bpb;
        w[base + 0] = 0x00;
        w[base + 1] = 0x3C; // d = f16(1.0)
        w[base + 2] = 0x00;
        w[base + 3] = 0x00; // dmin = f16(0.0)
        // scales[12]: sc=1 for j<4, m=0 for j<4, sc low nibble=1 for j>=4
        for (4..16) |i| w[base + i] = 0;
        w[base + 4] = 1;
        w[base + 5] = 1;
        w[base + 6] = 1;
        w[base + 7] = 1;
        w[base + 12] = 1;
        w[base + 13] = 1;
        w[base + 14] = 1;
        w[base + 15] = 1;
        // qs[128]: lo=1, hi=1 → 0x11
        for (16..144) |i| w[base + i] = 0x11;
    }
    const bs = backend_mod.quant_super_block_elems;
    var x: [bs]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [2]f32 = undefined;
    gemvQ4_K(&x, &w, &y, 2, bs);
    for (0..2) |i| try std.testing.expectApproxEqAbs(@as(f32, 256.0), y[i], 0.01);
}

test "gemvQ4_K all zeros" {
    // 2 rows, k=256. All nibbles = 0, dmin = 0. y = 0.
    const bpb = backend_mod.q4_k_block_bytes;
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
        for (16..144) |i| w[base + i] = 0x00;
    }
    var x: [bs]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [2]f32 = undefined;
    gemvQ4_K(&x, &w, &y, 2, bs);
    for (0..2) |i| try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[i], 0.01);
}

test "gemvQ4_K dmin subtraction" {
    // 1 row, k=256. d=1.0, dmin=f16(0.5). sc=1, m=1 for all groups.
    // All qs nibbles=1 → raw sum = 256. dmin subtraction = 0.5 * 1 * 256 = 128.
    // y = d * sc * sum(q) - dmin * m * 256 = 1.0*1*256 - 0.5*1*256 = 128.0
    const bpb = backend_mod.q4_k_block_bytes;
    const bs = backend_mod.quant_super_block_elems;
    var w: [bpb]u8 = undefined;
    w[0] = 0x00;
    w[1] = 0x3C; // d = f16(1.0)
    w[2] = 0x00;
    w[3] = 0x38; // dmin = f16(0.5)
    // scales[12]: getScaleMinK4 encoding.
    // For j<4: sc = q[j] & 0x3F, m = q[j+4] & 0x3F → sc=1, m=1.
    // For j>=4: sc = (q[j+4]&0xF) | ((q[j-4]>>6)<<4), m = (q[j+4]>>4) | ((q[j]>>6)<<4).
    for (4..16) |i| w[i] = 0;
    // Groups 0-3 (j<4): q[0..3]=1 for sc=1, q[4..7]=1 for m=1
    w[4] = 1;
    w[5] = 1;
    w[6] = 1;
    w[7] = 1;
    w[8] = 1;
    w[9] = 1;
    w[10] = 1;
    w[11] = 1;
    // Groups 4-7 (j>=4): sc = (q[j+4]&0xF)|((q[j-4]>>6)<<4), m = (q[j+4]>>4)|((q[j]>>6)<<4)
    // For sc=1: q[j+4]&0xF = 1 → low nibble = 1
    // For m=1: q[j+4]>>4 = 1 → high nibble = 1, so q[j+4] = 0x11
    w[12] = 0x11;
    w[13] = 0x11;
    w[14] = 0x11;
    w[15] = 0x11;
    // qs[128]: lo=1, hi=1 → 0x11
    for (16..144) |i| w[i] = 0x11;
    var x: [bs]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvQ4_K(&x, &w, &y, 1, bs);
    try std.testing.expectApproxEqAbs(@as(f32, 128.0), y[0], 0.1);
}

test "gemvQ4_K single row" {
    // n=1 exercises the single-row fallback.
    // d=0.5, dmin=0, sc=1, m=0, all nibbles=2 → q=2.
    // y = 0.5 * 1 * 256 * 2 = 256.0
    const bpb = backend_mod.q4_k_block_bytes;
    const bs = backend_mod.quant_super_block_elems;
    var w: [bpb]u8 = undefined;
    w[0] = 0x00;
    w[1] = 0x38; // d = f16(0.5)
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
    for (16..144) |i| w[i] = 0x22; // lo=2, hi=2
    var x: [bs]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvQ4_K(&x, &w, &y, 1, bs);
    try std.testing.expectApproxEqAbs(@as(f32, 256.0), y[0], 0.1);
}
