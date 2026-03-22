//! CPU GEMV kernel for Q6_K quantization.
//! 256 values per super-block, 210 bytes.
//! 2-row batching with V8 SIMD for full chunks.

const std = @import("std");
const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);

/// Q6_K GEMV: y = W @ x. 2-row batched with V8 SIMD for full chunks.
pub fn gemvQ6_K(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb: usize = 210;
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
            const d_0: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp0[208..210], .little))));
            const d_1: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp1[208..210], .little))));
            const bk = b * bs;

            inline for (0..2) |chunk| {
                const ql0 = bp0 + chunk * 64;
                const ql1 = bp1 + chunk * 64;
                const qh0 = bp0 + 128 + chunk * 32;
                const qh1 = bp1 + 128 + chunk * 32;
                const sc0: [*]const i8 = @ptrCast(bp0 + 192 + chunk * 8);
                const sc1: [*]const i8 = @ptrCast(bp1 + 192 + chunk * 8);
                const base = bk + chunk * 128;

                if (base + 127 < k) {
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
                            q1v0[idx] = @floatFromInt(@as(i8, @intCast((ql0[l] & 0x0F) | ((@as(u8, @truncate(qh0[l] >> 0)) & 3) << 4))) - 32);
                            q2v0[idx] = @floatFromInt(@as(i8, @intCast((ql0[l + 32] & 0x0F) | ((@as(u8, @truncate(qh0[l] >> 2)) & 3) << 4))) - 32);
                            q3v0[idx] = @floatFromInt(@as(i8, @intCast((ql0[l] >> 4) | ((@as(u8, @truncate(qh0[l] >> 4)) & 3) << 4))) - 32);
                            q4v0[idx] = @floatFromInt(@as(i8, @intCast((ql0[l + 32] >> 4) | ((@as(u8, @truncate(qh0[l] >> 6)) & 3) << 4))) - 32);
                            q1v1[idx] = @floatFromInt(@as(i8, @intCast((ql1[l] & 0x0F) | ((@as(u8, @truncate(qh1[l] >> 0)) & 3) << 4))) - 32);
                            q2v1[idx] = @floatFromInt(@as(i8, @intCast((ql1[l + 32] & 0x0F) | ((@as(u8, @truncate(qh1[l] >> 2)) & 3) << 4))) - 32);
                            q3v1[idx] = @floatFromInt(@as(i8, @intCast((ql1[l] >> 4) | ((@as(u8, @truncate(qh1[l] >> 4)) & 3) << 4))) - 32);
                            q4v1[idx] = @floatFromInt(@as(i8, @intCast((ql1[l + 32] >> 4) | ((@as(u8, @truncate(qh1[l] >> 6)) & 3) << 4))) - 32);
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
                        const q10: i8 = @as(i8, @intCast((ql0[l] & 0x0F) | ((@as(u8, @truncate(qh0[l] >> 0)) & 3) << 4))) - 32;
                        const q20: i8 = @as(i8, @intCast((ql0[l + 32] & 0x0F) | ((@as(u8, @truncate(qh0[l] >> 2)) & 3) << 4))) - 32;
                        const q30: i8 = @as(i8, @intCast((ql0[l] >> 4) | ((@as(u8, @truncate(qh0[l] >> 4)) & 3) << 4))) - 32;
                        const q40: i8 = @as(i8, @intCast((ql0[l + 32] >> 4) | ((@as(u8, @truncate(qh0[l] >> 6)) & 3) << 4))) - 32;
                        const q11: i8 = @as(i8, @intCast((ql1[l] & 0x0F) | ((@as(u8, @truncate(qh1[l] >> 0)) & 3) << 4))) - 32;
                        const q21: i8 = @as(i8, @intCast((ql1[l + 32] & 0x0F) | ((@as(u8, @truncate(qh1[l] >> 2)) & 3) << 4))) - 32;
                        const q31: i8 = @as(i8, @intCast((ql1[l] >> 4) | ((@as(u8, @truncate(qh1[l] >> 4)) & 3) << 4))) - 32;
                        const q41: i8 = @as(i8, @intCast((ql1[l + 32] >> 4) | ((@as(u8, @truncate(qh1[l] >> 6)) & 3) << 4))) - 32;
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
            const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[208..210], .little))));
            const bk = b * bs;

            inline for (0..2) |chunk| {
                const ql = bp + chunk * 64;
                const qh = bp + 128 + chunk * 32;
                const sc: [*]const i8 = @ptrCast(bp + 192 + chunk * 8);
                const base = bk + chunk * 128;

                if (base + 127 < k) {
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
                            q1v[idx] = @floatFromInt(@as(i8, @intCast((ql[l] & 0x0F) | ((@as(u8, @truncate(qh[l] >> 0)) & 3) << 4))) - 32);
                            q2v[idx] = @floatFromInt(@as(i8, @intCast((ql[l + 32] & 0x0F) | ((@as(u8, @truncate(qh[l] >> 2)) & 3) << 4))) - 32);
                            q3v[idx] = @floatFromInt(@as(i8, @intCast((ql[l] >> 4) | ((@as(u8, @truncate(qh[l] >> 4)) & 3) << 4))) - 32);
                            q4v[idx] = @floatFromInt(@as(i8, @intCast((ql[l + 32] >> 4) | ((@as(u8, @truncate(qh[l] >> 6)) & 3) << 4))) - 32);
                        }
                        sum += xv0 * ds_q1 * q1v + xv1 * ds_q2 * q2v + xv2 * ds_q3 * q3v + xv3 * ds_q4 * q4v;
                    }
                } else {
                    var s: f32 = 0.0;
                    for (0..32) |l| {
                        const is: usize = l / 16;
                        const q1: i8 = @as(i8, @intCast((ql[l] & 0x0F) | ((@as(u8, @truncate(qh[l] >> 0)) & 3) << 4))) - 32;
                        const q2: i8 = @as(i8, @intCast((ql[l + 32] & 0x0F) | ((@as(u8, @truncate(qh[l] >> 2)) & 3) << 4))) - 32;
                        const q3: i8 = @as(i8, @intCast((ql[l] >> 4) | ((@as(u8, @truncate(qh[l] >> 4)) & 3) << 4))) - 32;
                        const q4: i8 = @as(i8, @intCast((ql[l + 32] >> 4) | ((@as(u8, @truncate(qh[l] >> 6)) & 3) << 4))) - 32;
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
