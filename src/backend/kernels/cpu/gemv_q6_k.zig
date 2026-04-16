//! CPU GEMV kernel for Q6_K quantization.
//! 256 values per super-block, 210 bytes.
//! 2-row batching with V8 SIMD for full chunks.

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);

// Q6_K block layout offsets (total block = 210 bytes).
// [0..128): ql — low 4 bits per element (128 bytes for 256 elements)
// [128..192): qh — high 2 bits per element (64 bytes)
// [192..208): sc — scales (16 signed bytes)
// [208..210): d — super-block scale (f16)
const q6_k_ql_chunk_bytes: usize = 64;
const q6_k_qh_offset: usize = 128;
const q6_k_qh_chunk_bytes: usize = 32;
const q6_k_sc_offset: usize = 192;
const q6_k_sc_chunk_bytes: usize = 8;
const q6_k_d_offset: usize = 208;

/// Elements per half super-block chunk (256 / 2).
const chunk_elems = backend_mod.quant_super_block_elems / 2;
/// Q6_K dequant bias: 6-bit unsigned [0..63] centered to signed [-32..31].
const q6_k_dequant_bias: i8 = -32;
/// Mask for extracting 2-bit high-order field from qh byte.
const qh_2bit_mask: u8 = 3;

/// Q6_K GEMV: y = W @ x. 2-row batched with V8 SIMD for full chunks.
pub fn gemvQ6_K(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb = backend_mod.q6_k_block_bytes;
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
            const d_0: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp0[q6_k_d_offset..][0..2], .little))));
            const d_1: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp1[q6_k_d_offset..][0..2], .little))));
            const bk = b * bs;

            inline for (0..2) |chunk| {
                const ql0 = bp0 + chunk * q6_k_ql_chunk_bytes;
                const ql1 = bp1 + chunk * q6_k_ql_chunk_bytes;
                const qh0 = bp0 + q6_k_qh_offset + chunk * q6_k_qh_chunk_bytes;
                const qh1 = bp1 + q6_k_qh_offset + chunk * q6_k_qh_chunk_bytes;
                const sc0: [*]const i8 = @ptrCast(bp0 + q6_k_sc_offset + chunk * q6_k_sc_chunk_bytes);
                const sc1: [*]const i8 = @ptrCast(bp1 + q6_k_sc_offset + chunk * q6_k_sc_chunk_bytes);
                const base = bk + chunk * chunk_elems;

                if (base + chunk_elems - 1 < k) {
                    // Full chunk — process 8 l-values at a time with SIMD.
                    inline for (0..4) |lblock| {
                        const l_start = lblock * 8;
                        const is: usize = l_start / 16;

                        const ds0_q1: V8 = @splat(d_0 * @as(f32, @floatFromInt(sc0[is + 0])));
                        const ds0_q2: V8 = @splat(d_0 * @as(f32, @floatFromInt(sc0[is + 2])));
                        const ds0_q3: V8 = @splat(d_0 * @as(f32, @floatFromInt(sc0[is + 4])));
                        const ds0_q4: V8 = @splat(d_0 * @as(f32, @floatFromInt(sc0[is + 6])));
                        const ds1_q1: V8 = @splat(d_1 * @as(f32, @floatFromInt(sc1[is + 0])));
                        const ds1_q2: V8 = @splat(d_1 * @as(f32, @floatFromInt(sc1[is + 2])));
                        const ds1_q3: V8 = @splat(d_1 * @as(f32, @floatFromInt(sc1[is + 4])));
                        const ds1_q4: V8 = @splat(d_1 * @as(f32, @floatFromInt(sc1[is + 6])));

                        const xv0: V8 = x[base + l_start ..][0..8].*;
                        const xv1: V8 = x[base + l_start + 32 ..][0..8].*;
                        const xv2: V8 = x[base + l_start + 64 ..][0..8].*;
                        const xv3: V8 = x[base + l_start + 96 ..][0..8].*;

                        var q1v0: V8 = undefined;
                        var q2v0: V8 = undefined;
                        var q3v0: V8 = undefined;
                        var q4v0: V8 = undefined;
                        var q1v1: V8 = undefined;
                        var q2v1: V8 = undefined;
                        var q3v1: V8 = undefined;
                        var q4v1: V8 = undefined;

                        inline for (0..8) |idx| {
                            const l = l_start + idx;
                            q1v0[idx] = @floatFromInt(@as(i8, @intCast((ql0[l] & 0x0F) | ((@as(u8, @truncate(qh0[l] >> 0)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias);
                            q2v0[idx] = @floatFromInt(@as(i8, @intCast((ql0[l + 32] & 0x0F) | ((@as(u8, @truncate(qh0[l] >> 2)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias);
                            q3v0[idx] = @floatFromInt(@as(i8, @intCast((ql0[l] >> 4) | ((@as(u8, @truncate(qh0[l] >> 4)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias);
                            q4v0[idx] = @floatFromInt(@as(i8, @intCast((ql0[l + 32] >> 4) | ((@as(u8, @truncate(qh0[l] >> 6)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias);
                            q1v1[idx] = @floatFromInt(@as(i8, @intCast((ql1[l] & 0x0F) | ((@as(u8, @truncate(qh1[l] >> 0)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias);
                            q2v1[idx] = @floatFromInt(@as(i8, @intCast((ql1[l + 32] & 0x0F) | ((@as(u8, @truncate(qh1[l] >> 2)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias);
                            q3v1[idx] = @floatFromInt(@as(i8, @intCast((ql1[l] >> 4) | ((@as(u8, @truncate(qh1[l] >> 4)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias);
                            q4v1[idx] = @floatFromInt(@as(i8, @intCast((ql1[l + 32] >> 4) | ((@as(u8, @truncate(qh1[l] >> 6)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias);
                        }

                        sum0 += xv0 * ds0_q1 * q1v0 + xv1 * ds0_q2 * q2v0 + xv2 * ds0_q3 * q3v0 + xv3 * ds0_q4 * q4v0;
                        sum1 += xv0 * ds1_q1 * q1v1 + xv1 * ds1_q2 * q2v1 + xv2 * ds1_q3 * q3v1 + xv3 * ds1_q4 * q4v1;
                    }
                } else {
                    // Partial chunk — scalar fallback
                    var s0: f32 = 0.0;
                    var s1: f32 = 0.0;
                    for (0..32) |l| {
                        const is: usize = l / 16;
                        const gi0 = base + l;
                        const gi1 = base + l + 32;
                        const gi2 = base + l + 64;
                        const gi3 = base + l + 96;
                        const q10: i8 = @as(i8, @intCast((ql0[l] & 0x0F) | ((@as(u8, @truncate(qh0[l] >> 0)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                        const q20: i8 = @as(i8, @intCast((ql0[l + 32] & 0x0F) | ((@as(u8, @truncate(qh0[l] >> 2)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                        const q30: i8 = @as(i8, @intCast((ql0[l] >> 4) | ((@as(u8, @truncate(qh0[l] >> 4)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                        const q40: i8 = @as(i8, @intCast((ql0[l + 32] >> 4) | ((@as(u8, @truncate(qh0[l] >> 6)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                        const q11: i8 = @as(i8, @intCast((ql1[l] & 0x0F) | ((@as(u8, @truncate(qh1[l] >> 0)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                        const q21: i8 = @as(i8, @intCast((ql1[l + 32] & 0x0F) | ((@as(u8, @truncate(qh1[l] >> 2)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                        const q31: i8 = @as(i8, @intCast((ql1[l] >> 4) | ((@as(u8, @truncate(qh1[l] >> 4)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                        const q41: i8 = @as(i8, @intCast((ql1[l + 32] >> 4) | ((@as(u8, @truncate(qh1[l] >> 6)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                        if (gi0 < k) {
                            s0 += x[gi0] * d_0 * @as(f32, @floatFromInt(sc0[is + 0])) * @as(f32, @floatFromInt(q10));
                            s1 += x[gi0] * d_1 * @as(f32, @floatFromInt(sc1[is + 0])) * @as(f32, @floatFromInt(q11));
                        }
                        if (gi1 < k) {
                            s0 += x[gi1] * d_0 * @as(f32, @floatFromInt(sc0[is + 2])) * @as(f32, @floatFromInt(q20));
                            s1 += x[gi1] * d_1 * @as(f32, @floatFromInt(sc1[is + 2])) * @as(f32, @floatFromInt(q21));
                        }
                        if (gi2 < k) {
                            s0 += x[gi2] * d_0 * @as(f32, @floatFromInt(sc0[is + 4])) * @as(f32, @floatFromInt(q30));
                            s1 += x[gi2] * d_1 * @as(f32, @floatFromInt(sc1[is + 4])) * @as(f32, @floatFromInt(q31));
                        }
                        if (gi3 < k) {
                            s0 += x[gi3] * d_0 * @as(f32, @floatFromInt(sc0[is + 6])) * @as(f32, @floatFromInt(q40));
                            s1 += x[gi3] * d_1 * @as(f32, @floatFromInt(sc1[is + 6])) * @as(f32, @floatFromInt(q41));
                        }
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
            const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[q6_k_d_offset..][0..2], .little))));
            const bk = b * bs;

            inline for (0..2) |chunk| {
                const ql = bp + chunk * q6_k_ql_chunk_bytes;
                const qh = bp + q6_k_qh_offset + chunk * q6_k_qh_chunk_bytes;
                const sc: [*]const i8 = @ptrCast(bp + q6_k_sc_offset + chunk * q6_k_sc_chunk_bytes);
                const base = bk + chunk * chunk_elems;

                if (base + chunk_elems - 1 < k) {
                    inline for (0..4) |lblock| {
                        const l_start = lblock * 8;
                        const is: usize = l_start / 16;
                        const ds_q1: V8 = @splat(d * @as(f32, @floatFromInt(sc[is + 0])));
                        const ds_q2: V8 = @splat(d * @as(f32, @floatFromInt(sc[is + 2])));
                        const ds_q3: V8 = @splat(d * @as(f32, @floatFromInt(sc[is + 4])));
                        const ds_q4: V8 = @splat(d * @as(f32, @floatFromInt(sc[is + 6])));

                        const xv0: V8 = x[base + l_start ..][0..8].*;
                        const xv1: V8 = x[base + l_start + 32 ..][0..8].*;
                        const xv2: V8 = x[base + l_start + 64 ..][0..8].*;
                        const xv3: V8 = x[base + l_start + 96 ..][0..8].*;

                        var q1v: V8 = undefined;
                        var q2v: V8 = undefined;
                        var q3v: V8 = undefined;
                        var q4v: V8 = undefined;
                        inline for (0..8) |idx| {
                            const l = l_start + idx;
                            q1v[idx] = @floatFromInt(@as(i8, @intCast((ql[l] & 0x0F) | ((@as(u8, @truncate(qh[l] >> 0)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias);
                            q2v[idx] = @floatFromInt(@as(i8, @intCast((ql[l + 32] & 0x0F) | ((@as(u8, @truncate(qh[l] >> 2)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias);
                            q3v[idx] = @floatFromInt(@as(i8, @intCast((ql[l] >> 4) | ((@as(u8, @truncate(qh[l] >> 4)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias);
                            q4v[idx] = @floatFromInt(@as(i8, @intCast((ql[l + 32] >> 4) | ((@as(u8, @truncate(qh[l] >> 6)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias);
                        }
                        sum += xv0 * ds_q1 * q1v + xv1 * ds_q2 * q2v + xv2 * ds_q3 * q3v + xv3 * ds_q4 * q4v;
                    }
                } else {
                    var s: f32 = 0.0;
                    for (0..32) |l| {
                        const is: usize = l / 16;
                        const q1: i8 = @as(i8, @intCast((ql[l] & 0x0F) | ((@as(u8, @truncate(qh[l] >> 0)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                        const q2: i8 = @as(i8, @intCast((ql[l + 32] & 0x0F) | ((@as(u8, @truncate(qh[l] >> 2)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                        const q3: i8 = @as(i8, @intCast((ql[l] >> 4) | ((@as(u8, @truncate(qh[l] >> 4)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                        const q4: i8 = @as(i8, @intCast((ql[l + 32] >> 4) | ((@as(u8, @truncate(qh[l] >> 6)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                        const gi0 = base + l;
                        const gi1 = base + l + 32;
                        const gi2 = base + l + 64;
                        const gi3 = base + l + 96;
                        if (gi0 < k) s += x[gi0] * d * @as(f32, @floatFromInt(sc[is + 0])) * @as(f32, @floatFromInt(q1));
                        if (gi1 < k) s += x[gi1] * d * @as(f32, @floatFromInt(sc[is + 2])) * @as(f32, @floatFromInt(q2));
                        if (gi2 < k) s += x[gi2] * d * @as(f32, @floatFromInt(sc[is + 4])) * @as(f32, @floatFromInt(q3));
                        if (gi3 < k) s += x[gi3] * d * @as(f32, @floatFromInt(sc[is + 6])) * @as(f32, @floatFromInt(q4));
                    }
                    sum[0] += s;
                }
            }
        }
        y[row] = @reduce(.Add, sum);
    }
}

test "gemvQ6_K all zeros" {
    // 2 rows, k=256. 6-bit signed values biased -32.
    // ql=0x00, qh=0xAA → q = (0 | (2<<4)) - 32 = 32 - 32 = 0 for all positions.
    // sc=1 (i8), d=1.0. y = 0.
    const bpb = backend_mod.q6_k_block_bytes; // 210
    var w: [2 * bpb]u8 = undefined;
    for (0..2) |r| {
        const base = r * bpb;
        for (0..128) |i| w[base + i] = 0x00; // ql
        for (128..192) |i| w[base + i] = 0xAA; // qh: bits 10_10_10_10
        for (192..208) |i| w[base + i] = 1; // sc = i8(1)
        w[base + 208] = 0x00;
        w[base + 209] = 0x3C; // d = f16(1.0)
    }
    const bs = backend_mod.quant_super_block_elems;
    var x: [bs]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [2]f32 = undefined;
    gemvQ6_K(&x, &w, &y, 2, bs);
    for (0..2) |i| try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[i], 0.01);
}

test "gemvQ6_K uniform positive" {
    // 2 rows, k=256. ql=0x11, qh=0xAA → q = (1 | (2<<4)) - 32 = 1.
    // sc=1, d=1.0. y = 256 * 1.0 = 256.0.
    const bpb = backend_mod.q6_k_block_bytes;
    const bs = backend_mod.quant_super_block_elems;
    var w: [2 * bpb]u8 = undefined;
    for (0..2) |r| {
        const base = r * bpb;
        for (0..128) |i| w[base + i] = 0x11; // ql: lo=1, hi=1
        for (128..192) |i| w[base + i] = 0xAA; // qh
        for (192..208) |i| w[base + i] = 1; // sc = i8(1)
        w[base + 208] = 0x00;
        w[base + 209] = 0x3C;
    }
    var x: [bs]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [2]f32 = undefined;
    gemvQ6_K(&x, &w, &y, 2, bs);
    for (0..2) |i| try std.testing.expectApproxEqAbs(@as(f32, 256.0), y[i], 0.01);
}

test "gemvQ6_K single row" {
    // n=1 exercises single-row fallback. d=0.5, sc=1, q=1.
    // y = 0.5 * 1 * 256 = 128.0
    const bpb = backend_mod.q6_k_block_bytes;
    const bs = backend_mod.quant_super_block_elems;
    var w: [bpb]u8 = undefined;
    for (0..128) |i| w[i] = 0x11;
    for (128..192) |i| w[i] = 0xAA;
    for (192..208) |i| w[i] = 1;
    w[208] = 0x00;
    w[209] = 0x38; // d = f16(0.5)
    var x: [bs]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvQ6_K(&x, &w, &y, 1, bs);
    try std.testing.expectApproxEqAbs(@as(f32, 128.0), y[0], 1.0);
}
