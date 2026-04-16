//! CPU GEMV kernels for less common quantization formats.
//! Q4_1, Q5_0, Q2_K, Q3_K — scalar implementations with 2-row batching.

const std = @import("std");
const backend_mod = @import("../../backend.zig");

/// Q5_0 dequant bias: 5-bit unsigned [0..31] centered to signed [-16..15].
const q5_0_dequant_bias: i8 = -16;
/// Q3_K dequant bias: 3-bit unsigned [0..7] centered to signed [-4..3].
const q3_k_dequant_bias: i8 = -4;
/// Q3_K scale bias: raw 4-bit scale [0..15] centered to signed [-8..7].
const q3_k_scale_bias: i8 = -8;
/// Q2_K 2-bit quantization mask.
const q2_k_bit_mask: u8 = 0x03;
/// Q3_K 2-bit quantization mask for low bits.
const q3_k_lo_mask: u8 = 0x03;

/// Q4_1: 32 values per block, 20 bytes (f16 scale + f16 min + 16 nibble-packed bytes)
/// 2-row batched to share x-vector cache reads.
pub fn gemvQ4_1(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb = backend_mod.q4_1_block_bytes;
    const qk = backend_mod.quant_block_elems;
    const nb = (k + qk - 1) / qk;
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
            const d0: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp0[0..2], .little))));
            const d1: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp1[0..2], .little))));
            const m0: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp0[2..4], .little))));
            const m1: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp1[2..4], .little))));
            const bk = b * qk;
            if (bk + qk - 1 < k) {
                for (0..qk / 2) |j| {
                    const byte0 = bp0[4 + j];
                    const byte1 = bp1[4 + j];
                    const xlo = x[bk + j];
                    const xhi = x[bk + j + qk / 2];
                    sum0 += xlo * (@as(f32, @floatFromInt(@as(u8, byte0 & 0x0F))) * d0 + m0) +
                        xhi * (@as(f32, @floatFromInt(@as(u8, byte0 >> 4))) * d0 + m0);
                    sum1 += xlo * (@as(f32, @floatFromInt(@as(u8, byte1 & 0x0F))) * d1 + m1) +
                        xhi * (@as(f32, @floatFromInt(@as(u8, byte1 >> 4))) * d1 + m1);
                }
            } else {
                for (0..qk / 2) |j| {
                    const byte0 = bp0[4 + j];
                    const byte1 = bp1[4 + j];
                    const gi0 = bk + j;
                    const gi1 = bk + j + qk / 2;
                    if (gi0 < k) {
                        const xv = x[gi0];
                        sum0 += xv * (@as(f32, @floatFromInt(@as(u8, byte0 & 0x0F))) * d0 + m0);
                        sum1 += xv * (@as(f32, @floatFromInt(@as(u8, byte1 & 0x0F))) * d1 + m1);
                    }
                    if (gi1 < k) {
                        const xv = x[gi1];
                        sum0 += xv * (@as(f32, @floatFromInt(@as(u8, byte0 >> 4))) * d0 + m0);
                        sum1 += xv * (@as(f32, @floatFromInt(@as(u8, byte1 >> 4))) * d1 + m1);
                    }
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
            const m: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[2..4], .little))));
            const bk = b * qk;
            if (bk + qk - 1 < k) {
                for (0..qk / 2) |j| {
                    const byte = bp[4 + j];
                    sum += x[bk + j] * (@as(f32, @floatFromInt(@as(u8, byte & 0x0F))) * d + m) +
                        x[bk + j + qk / 2] * (@as(f32, @floatFromInt(@as(u8, byte >> 4))) * d + m);
                }
            } else {
                for (0..qk / 2) |j| {
                    const byte = bp[4 + j];
                    const gi0 = bk + j;
                    const gi1 = bk + j + qk / 2;
                    if (gi0 < k) sum += x[gi0] * (@as(f32, @floatFromInt(@as(u8, byte & 0x0F))) * d + m);
                    if (gi1 < k) sum += x[gi1] * (@as(f32, @floatFromInt(@as(u8, byte >> 4))) * d + m);
                }
            }
        }
        y[row] = sum;
    }
}

/// Q5_0: 32 values per block, 22 bytes (f16 scale + 4 bytes qh + 16 nibble-packed bytes)
/// 2-row batched to share x-vector cache reads.
pub fn gemvQ5_0(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb = backend_mod.q5_0_block_bytes;
    const qk = backend_mod.quant_block_elems;
    const nb = (k + qk - 1) / qk;
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
            const d0: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp0[0..2], .little))));
            const d1: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp1[0..2], .little))));
            const qh0 = std.mem.readInt(u32, bp0[2..6], .little);
            const qh1 = std.mem.readInt(u32, bp1[2..6], .little);
            const bk = b * qk;
            var block_sum0: f32 = 0.0;
            var block_sum1: f32 = 0.0;
            if (bk + qk - 1 < k) {
                for (0..16) |j| {
                    const byte0 = bp0[6 + j];
                    const byte1 = bp1[6 + j];
                    const hb0_0: u8 = @truncate((qh0 >> @intCast(j)) & 1);
                    const hb1_0: u8 = @truncate((qh0 >> @intCast(j + 16)) & 1);
                    const hb0_1: u8 = @truncate((qh1 >> @intCast(j)) & 1);
                    const hb1_1: u8 = @truncate((qh1 >> @intCast(j + 16)) & 1);
                    const v0_0: i8 = @as(i8, @intCast((byte0 & 0x0F) | (hb0_0 << 4))) + q5_0_dequant_bias;
                    const v1_0: i8 = @as(i8, @intCast((byte0 >> 4) | (hb1_0 << 4))) + q5_0_dequant_bias;
                    const v0_1: i8 = @as(i8, @intCast((byte1 & 0x0F) | (hb0_1 << 4))) + q5_0_dequant_bias;
                    const v1_1: i8 = @as(i8, @intCast((byte1 >> 4) | (hb1_1 << 4))) + q5_0_dequant_bias;
                    const xlo = x[bk + j];
                    const xhi = x[bk + j + 16];
                    block_sum0 += xlo * @as(f32, @floatFromInt(v0_0)) +
                        xhi * @as(f32, @floatFromInt(v1_0));
                    block_sum1 += xlo * @as(f32, @floatFromInt(v0_1)) +
                        xhi * @as(f32, @floatFromInt(v1_1));
                }
            } else {
                for (0..16) |j| {
                    const byte0 = bp0[6 + j];
                    const byte1 = bp1[6 + j];
                    const hb0_0: u8 = @truncate((qh0 >> @intCast(j)) & 1);
                    const hb1_0: u8 = @truncate((qh0 >> @intCast(j + 16)) & 1);
                    const hb0_1: u8 = @truncate((qh1 >> @intCast(j)) & 1);
                    const hb1_1: u8 = @truncate((qh1 >> @intCast(j + 16)) & 1);
                    const v0_0: i8 = @as(i8, @intCast((byte0 & 0x0F) | (hb0_0 << 4))) + q5_0_dequant_bias;
                    const v1_0: i8 = @as(i8, @intCast((byte0 >> 4) | (hb1_0 << 4))) + q5_0_dequant_bias;
                    const v0_1: i8 = @as(i8, @intCast((byte1 & 0x0F) | (hb0_1 << 4))) + q5_0_dequant_bias;
                    const v1_1: i8 = @as(i8, @intCast((byte1 >> 4) | (hb1_1 << 4))) + q5_0_dequant_bias;
                    const gi0 = bk + j;
                    const gi1 = bk + j + 16;
                    if (gi0 < k) {
                        const xv = x[gi0];
                        block_sum0 += xv * @as(f32, @floatFromInt(v0_0));
                        block_sum1 += xv * @as(f32, @floatFromInt(v0_1));
                    }
                    if (gi1 < k) {
                        const xv = x[gi1];
                        block_sum0 += xv * @as(f32, @floatFromInt(v1_0));
                        block_sum1 += xv * @as(f32, @floatFromInt(v1_1));
                    }
                }
            }
            sum0 += block_sum0 * d0;
            sum1 += block_sum1 * d1;
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
            const qh = std.mem.readInt(u32, bp[2..6], .little);
            const bk = b * qk;
            var block_sum: f32 = 0.0;
            if (bk + qk - 1 < k) {
                for (0..16) |j| {
                    const byte = bp[6 + j];
                    const hb0: u8 = @truncate((qh >> @intCast(j)) & 1);
                    const hb1: u8 = @truncate((qh >> @intCast(j + 16)) & 1);
                    const v0: i8 = @as(i8, @intCast((byte & 0x0F) | (hb0 << 4))) + q5_0_dequant_bias;
                    const v1: i8 = @as(i8, @intCast((byte >> 4) | (hb1 << 4))) + q5_0_dequant_bias;
                    block_sum += x[bk + j] * @as(f32, @floatFromInt(v0)) +
                        x[bk + j + 16] * @as(f32, @floatFromInt(v1));
                }
            } else {
                for (0..16) |j| {
                    const byte = bp[6 + j];
                    const hb0: u8 = @truncate((qh >> @intCast(j)) & 1);
                    const hb1: u8 = @truncate((qh >> @intCast(j + 16)) & 1);
                    const v0: i8 = @as(i8, @intCast((byte & 0x0F) | (hb0 << 4))) + q5_0_dequant_bias;
                    const v1: i8 = @as(i8, @intCast((byte >> 4) | (hb1 << 4))) + q5_0_dequant_bias;
                    const gi0 = bk + j;
                    const gi1 = bk + j + 16;
                    if (gi0 < k) block_sum += x[gi0] * @as(f32, @floatFromInt(v0));
                    if (gi1 < k) block_sum += x[gi1] * @as(f32, @floatFromInt(v1));
                }
            }
            sum += block_sum * d;
        }
        y[row] = sum;
    }
}

/// Q2_K: 256 values per super-block, 84 bytes
/// Layout: scales[16] + qs[64] + d(f16) + dmin(f16)
/// 2-row batched to share x-vector cache reads.
pub fn gemvQ2_K(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb = backend_mod.q2_k_block_bytes;
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
            const scales0 = bp0[0..16];
            const scales1 = bp1[0..16];
            const qs0 = bp0 + 16;
            const qs1 = bp1 + 16;
            const d_0: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp0[80..82], .little))));
            const d_1: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp1[80..82], .little))));
            const dmin_0: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp0[82..84], .little))));
            const dmin_1: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp1[82..84], .little))));
            const bk = b * bs;
            if (bk + bs - 1 < k) {
                for (0..16) |sb| {
                    const sc0: f32 = @floatFromInt(scales0[sb] & 0x0F);
                    const m_val0: f32 = @floatFromInt(scales0[sb] >> 4);
                    const sc1: f32 = @floatFromInt(scales1[sb] & 0x0F);
                    const m_val1: f32 = @floatFromInt(scales1[sb] >> 4);
                    const d_sc0 = d_0 * sc0;
                    const dm_m0 = dmin_0 * m_val0;
                    const d_sc1 = d_1 * sc1;
                    const dm_m1 = dmin_1 * m_val1;
                    const sb_base = sb * 4;
                    for (0..16) |l| {
                        const shift: u3 = @intCast((l % 4) * 2);
                        const q0: f32 = @floatFromInt((qs0[sb_base + l / 4] >> shift) & q2_k_bit_mask);
                        const q1: f32 = @floatFromInt((qs1[sb_base + l / 4] >> shift) & q2_k_bit_mask);
                        const xv = x[bk + sb * 16 + l];
                        sum0 += xv * (d_sc0 * q0 - dm_m0);
                        sum1 += xv * (d_sc1 * q1 - dm_m1);
                    }
                }
            } else {
                for (0..16) |sb| {
                    const sc0: f32 = @floatFromInt(scales0[sb] & 0x0F);
                    const m_val0: f32 = @floatFromInt(scales0[sb] >> 4);
                    const sc1: f32 = @floatFromInt(scales1[sb] & 0x0F);
                    const m_val1: f32 = @floatFromInt(scales1[sb] >> 4);
                    const d_sc0 = d_0 * sc0;
                    const dm_m0 = dmin_0 * m_val0;
                    const d_sc1 = d_1 * sc1;
                    const dm_m1 = dmin_1 * m_val1;
                    for (0..16) |l| {
                        const gi = bk + sb * 16 + l;
                        if (gi >= k) break;
                        const qi = sb * 16 + l;
                        const byte_idx = qi / 4;
                        const shift: u3 = @intCast((qi % 4) * 2);
                        const q0: f32 = @floatFromInt((qs0[byte_idx] >> shift) & q2_k_bit_mask);
                        const q1: f32 = @floatFromInt((qs1[byte_idx] >> shift) & q2_k_bit_mask);
                        const xv = x[gi];
                        sum0 += xv * (d_sc0 * q0 - dm_m0);
                        sum1 += xv * (d_sc1 * q1 - dm_m1);
                    }
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
            const scales = bp[0..16];
            const qs = bp + 16;
            const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[80..82], .little))));
            const dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[82..84], .little))));
            const bk = b * bs;
            if (bk + bs - 1 < k) {
                for (0..16) |sb| {
                    const sc: f32 = @floatFromInt(scales[sb] & 0x0F);
                    const m_val: f32 = @floatFromInt(scales[sb] >> 4);
                    const d_sc = d * sc;
                    const dm_m = dmin * m_val;
                    const sb_base = sb * 4;
                    for (0..16) |l| {
                        const shift: u3 = @intCast((l % 4) * 2);
                        const q: f32 = @floatFromInt((qs[sb_base + l / 4] >> shift) & q2_k_bit_mask);
                        sum += x[bk + sb * 16 + l] * (d_sc * q - dm_m);
                    }
                }
            } else {
                for (0..16) |sb| {
                    const sc: f32 = @floatFromInt(scales[sb] & 0x0F);
                    const m_val: f32 = @floatFromInt(scales[sb] >> 4);
                    const d_sc = d * sc;
                    const dm_m = dmin * m_val;
                    for (0..16) |l| {
                        const gi = bk + sb * 16 + l;
                        if (gi >= k) break;
                        const qi = sb * 16 + l;
                        const byte_idx = qi / 4;
                        const shift: u3 = @intCast((qi % 4) * 2);
                        const q: f32 = @floatFromInt((qs[byte_idx] >> shift) & q2_k_bit_mask);
                        sum += x[gi] * (d_sc * q - dm_m);
                    }
                }
            }
        }
        y[row] = sum;
    }
}

/// Q3_K: 256 values per super-block, 110 bytes
/// Layout: hmask[32] + qs[64] + scales[12] + d(f16)
/// 2-row batched to share x-vector cache reads.
pub fn gemvQ3_K(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb = backend_mod.q3_k_block_bytes;
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
            const hmask0 = bp0[0..32];
            const hmask1 = bp1[0..32];
            const qs0 = bp0 + 32;
            const qs1 = bp1 + 32;
            const raw_scales0 = bp0[96..108];
            const raw_scales1 = bp1[96..108];
            const d_0: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp0[108..110], .little))));
            const d_1: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp1[108..110], .little))));
            const bk = b * bs;

            var scales0: [16]i8 = undefined;
            var scales1: [16]i8 = undefined;
            for (0..8) |j| {
                scales0[j] = @as(i8, @intCast(raw_scales0[j] & 0x0F)) + q3_k_scale_bias;
                scales1[j] = @as(i8, @intCast(raw_scales1[j] & 0x0F)) + q3_k_scale_bias;
            }
            for (0..8) |j| {
                scales0[8 + j] = @as(i8, @intCast(raw_scales0[j] >> 4)) + q3_k_scale_bias;
                scales1[8 + j] = @as(i8, @intCast(raw_scales1[j] >> 4)) + q3_k_scale_bias;
            }

            if (bk + bs - 1 < k) {
                for (0..16) |sb| {
                    const sc0: f32 = d_0 * @as(f32, @floatFromInt(scales0[sb]));
                    const sc1: f32 = d_1 * @as(f32, @floatFromInt(scales1[sb]));
                    const hm_bit: u3 = @intCast(sb / 2);
                    const hm_base: usize = (sb % 2) * 16;
                    const qs_base: usize = sb * 4;
                    var block_sum0: f32 = 0.0;
                    var block_sum1: f32 = 0.0;
                    for (0..16) |l| {
                        const shift: u3 = @intCast((l % 4) * 2);
                        const q_lo0: u8 = (qs0[qs_base + l / 4] >> shift) & q3_k_lo_mask;
                        const q_hi0: u8 = (hmask0[hm_base + l] >> hm_bit) & 1;
                        const q_lo1: u8 = (qs1[qs_base + l / 4] >> shift) & q3_k_lo_mask;
                        const q_hi1: u8 = (hmask1[hm_base + l] >> hm_bit) & 1;
                        const q3_0: i8 = @as(i8, @intCast(q_lo0 | (q_hi0 << 2))) + q3_k_dequant_bias;
                        const q3_1: i8 = @as(i8, @intCast(q_lo1 | (q_hi1 << 2))) + q3_k_dequant_bias;
                        const xv = x[bk + sb * 16 + l];
                        block_sum0 += xv * @as(f32, @floatFromInt(q3_0));
                        block_sum1 += xv * @as(f32, @floatFromInt(q3_1));
                    }
                    sum0 += block_sum0 * sc0;
                    sum1 += block_sum1 * sc1;
                }
            } else {
                for (0..16) |sb| {
                    const sub_bk = bk + sb * 16;
                    if (sub_bk >= k) break;
                    const sc0: f32 = d_0 * @as(f32, @floatFromInt(scales0[sb]));
                    const sc1: f32 = d_1 * @as(f32, @floatFromInt(scales1[sb]));
                    const hm_bit: u3 = @intCast(sb / 2);
                    const hm_base: usize = (sb % 2) * 16;
                    const qs_base: usize = sb * 4;
                    var block_sum0: f32 = 0.0;
                    var block_sum1: f32 = 0.0;
                    for (0..16) |l| {
                        if (sub_bk + l >= k) break;
                        const shift: u3 = @intCast((l % 4) * 2);
                        const q_lo0: u8 = (qs0[qs_base + l / 4] >> shift) & q3_k_lo_mask;
                        const q_hi0: u8 = (hmask0[hm_base + l] >> hm_bit) & 1;
                        const q_lo1: u8 = (qs1[qs_base + l / 4] >> shift) & q3_k_lo_mask;
                        const q_hi1: u8 = (hmask1[hm_base + l] >> hm_bit) & 1;
                        const q3_0: i8 = @as(i8, @intCast(q_lo0 | (q_hi0 << 2))) + q3_k_dequant_bias;
                        const q3_1: i8 = @as(i8, @intCast(q_lo1 | (q_hi1 << 2))) + q3_k_dequant_bias;
                        const xv = x[sub_bk + l];
                        block_sum0 += xv * @as(f32, @floatFromInt(q3_0));
                        block_sum1 += xv * @as(f32, @floatFromInt(q3_1));
                    }
                    sum0 += block_sum0 * sc0;
                    sum1 += block_sum1 * sc1;
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
            const hmask = bp[0..32];
            const qs = bp + 32;
            const raw_scales = bp[96..108];
            const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[108..110], .little))));
            const bk = b * bs;

            var scales: [16]i8 = undefined;
            for (0..8) |j| {
                scales[j] = @as(i8, @intCast(raw_scales[j] & 0x0F)) + q3_k_scale_bias;
            }
            for (0..8) |j| {
                scales[8 + j] = @as(i8, @intCast(raw_scales[j] >> 4)) + q3_k_scale_bias;
            }

            if (bk + bs - 1 < k) {
                for (0..16) |sb| {
                    const sc: f32 = d * @as(f32, @floatFromInt(scales[sb]));
                    const hm_bit: u3 = @intCast(sb / 2);
                    const hm_base: usize = (sb % 2) * 16;
                    const qs_base: usize = sb * 4;
                    var block_sum: f32 = 0.0;
                    for (0..16) |l| {
                        const shift: u3 = @intCast((l % 4) * 2);
                        const q_lo: u8 = (qs[qs_base + l / 4] >> shift) & q3_k_lo_mask;
                        const q_hi: u8 = (hmask[hm_base + l] >> hm_bit) & 1;
                        const q3: i8 = @as(i8, @intCast(q_lo | (q_hi << 2))) + q3_k_dequant_bias;
                        block_sum += x[bk + sb * 16 + l] * @as(f32, @floatFromInt(q3));
                    }
                    sum += block_sum * sc;
                }
            } else {
                for (0..16) |sb| {
                    const sub_bk = bk + sb * 16;
                    if (sub_bk >= k) break;
                    const sc: f32 = d * @as(f32, @floatFromInt(scales[sb]));
                    const hm_bit: u3 = @intCast(sb / 2);
                    const hm_base: usize = (sb % 2) * 16;
                    const qs_base: usize = sb * 4;
                    var block_sum: f32 = 0.0;
                    for (0..16) |l| {
                        if (sub_bk + l >= k) break;
                        const shift: u3 = @intCast((l % 4) * 2);
                        const q_lo: u8 = (qs[qs_base + l / 4] >> shift) & q3_k_lo_mask;
                        const q_hi: u8 = (hmask[hm_base + l] >> hm_bit) & 1;
                        const q3: i8 = @as(i8, @intCast(q_lo | (q_hi << 2))) + q3_k_dequant_bias;
                        block_sum += x[sub_bk + l] * @as(f32, @floatFromInt(q3));
                    }
                    sum += block_sum * sc;
                }
            }
        }
        y[row] = sum;
    }
}

// ── Tests ─────────────────────────────────────────────────────────

test "gemvQ4_1 all zeros" {
    // Q4_1: weight = nibble * d + m. With nibble=0, d=1.0, m=0.0 → weight=0.
    const bpb = backend_mod.q4_1_block_bytes; // 20
    var w: [2 * bpb]u8 = undefined;
    for (0..2) |r| {
        const base = r * bpb;
        // d = f16(1.0) = 0x3C00
        w[base] = 0x00;
        w[base + 1] = 0x3C;
        // m = f16(0.0) = 0x0000
        w[base + 2] = 0x00;
        w[base + 3] = 0x00;
        // All nibbles = 0
        for (0..16) |i| w[base + 4 + i] = 0x00;
    }
    var x: [32]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [2]f32 = undefined;
    gemvQ4_1(&x, &w, &y, 2, 32);
    for (0..2) |i| try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[i], 0.01);
}

test "gemvQ4_1 uniform with min offset" {
    // nibble=1, d=1.0, m=0.5 → weight = 1*1.0 + 0.5 = 1.5.
    // x=all 1.0 → y = 32 * 1.5 = 48.0
    const bpb = backend_mod.q4_1_block_bytes;
    var w: [bpb]u8 = undefined;
    // d = f16(1.0)
    w[0] = 0x00;
    w[1] = 0x3C;
    // m = f16(0.5) = 0x3800
    w[2] = 0x00;
    w[3] = 0x38;
    // All nibbles = 1 → byte = 0x11
    for (0..16) |i| w[4 + i] = 0x11;
    var x: [32]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvQ4_1(&x, &w, &y, 1, 32);
    try std.testing.expectApproxEqAbs(@as(f32, 48.0), y[0], 0.5);
}

test "gemvQ5_0 all zeros" {
    // Q5_0: value = (lo4 | (hi_bit << 4)) - 16. Value = 16 → 16-16=0.
    // lo4=0, hi_bit=1 → nibble=0, qh bit set → value = (0|16)-16 = 0.
    const bpb = backend_mod.q5_0_block_bytes; // 22
    var w: [2 * bpb]u8 = undefined;
    for (0..2) |r| {
        const base = r * bpb;
        // d = f16(1.0)
        w[base] = 0x00;
        w[base + 1] = 0x3C;
        // qh: all bits set for hi_bit=1 → 0xFFFFFFFF
        w[base + 2] = 0xFF;
        w[base + 3] = 0xFF;
        w[base + 4] = 0xFF;
        w[base + 5] = 0xFF;
        // All nibbles = 0 → bytes = 0x00
        for (0..16) |i| w[base + 6 + i] = 0x00;
    }
    var x: [32]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [2]f32 = undefined;
    gemvQ5_0(&x, &w, &y, 2, 32);
    for (0..2) |i| try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[i], 0.01);
}

test "gemvQ5_0 uniform positive" {
    // lo4=1, hi_bit=1 → value = (1|16)-16 = 1. d=1.0, x=all 1.0 → y=32.
    const bpb = backend_mod.q5_0_block_bytes;
    var w: [bpb]u8 = undefined;
    // d = f16(1.0)
    w[0] = 0x00;
    w[1] = 0x3C;
    // qh: all hi bits set
    w[2] = 0xFF;
    w[3] = 0xFF;
    w[4] = 0xFF;
    w[5] = 0xFF;
    // lo nibble=1, hi nibble=1 → byte = 0x11
    for (0..16) |i| w[6 + i] = 0x11;
    var x: [32]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvQ5_0(&x, &w, &y, 1, 32);
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), y[0], 0.5);
}

test "gemvQ2_K all zeros" {
    // Q2_K: weight = d * sc * q - dmin * m.
    // With q=0, dmin=0 → weight = 0.
    const bpb = backend_mod.q2_k_block_bytes; // 84
    const bs = backend_mod.quant_super_block_elems; // 256
    var w: [bpb]u8 = undefined;
    @memset(&w, 0);
    // d = f16(1.0) at offset 80
    w[80] = 0x00;
    w[81] = 0x3C;
    // dmin = f16(0.0) at offset 82, already 0
    // scales = 0 (sc=0, m=0), qs = 0 (q=0)
    var x: [bs]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvQ2_K(&x, &w, &y, 1, bs);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[0], 0.01);
}

test "gemvQ2_K uniform positive" {
    // q=1 (all 2-bit values = 1), sc=1 (lo nibble), m=0 (hi nibble), d=1.0, dmin=0.
    // weight = 1.0 * 1 * 1 - 0 = 1.0. y = 256 * 1.0 = 256.0
    const bpb = backend_mod.q2_k_block_bytes;
    const bs = backend_mod.quant_super_block_elems;
    var w: [bpb]u8 = undefined;
    @memset(&w, 0);
    // scales[0..16]: lo nibble=1 (sc), hi nibble=0 (m) → byte=0x01
    for (0..16) |i| w[i] = 0x01;
    // qs[16..80]: q=1 for all. 4 values per byte, each 2-bit = 01 → 0b01010101 = 0x55
    for (16..80) |i| w[i] = 0x55;
    // d = f16(1.0) at offset 80
    w[80] = 0x00;
    w[81] = 0x3C;
    // dmin = f16(0.0) at offset 82, already 0
    var x: [bs]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvQ2_K(&x, &w, &y, 1, bs);
    try std.testing.expectApproxEqAbs(@as(f32, 256.0), y[0], 1.0);
}

test "gemvQ3_K zero scale produces zero output" {
    // d=0.0 → all weights zero.
    const bpb = backend_mod.q3_k_block_bytes; // 110
    const bs = backend_mod.quant_super_block_elems;
    var w: [bpb]u8 = undefined;
    @memset(&w, 0);
    // d = f16(0.0) at offset 108, already 0
    // Fill hmask, qs, and scales with non-zero to verify d=0 zeroes everything
    for (0..32) |i| w[i] = 0xFF; // hmask
    for (32..96) |i| w[i] = 0xFF; // qs
    for (96..108) |i| w[i] = 0xFF; // raw_scales
    var x: [bs]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvQ3_K(&x, &w, &y, 1, bs);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[0], 0.01);
}

test "gemvQ3_K uniform positive" {
    // q_lo=1 (qs byte: 0b01010101=0x55), q_hi=1 (hmask all set).
    // q3 = (1 | (1<<2)) - 4 = 5 - 4 = 1.
    // raw_scales: lo nibble=9, hi nibble=9 → scales[j] = 9-8 = 1.
    // d=1.0 → weight = 1.0 * 1 * 1 = 1.0. y = 256 * 1.0 = 256.0
    const bpb = backend_mod.q3_k_block_bytes;
    const bs = backend_mod.quant_super_block_elems;
    var w: [bpb]u8 = undefined;
    @memset(&w, 0);
    // hmask[0..32]: all bits set → q_hi=1
    for (0..32) |i| w[i] = 0xFF;
    // qs[32..96]: q_lo=1 → each byte = 0b01_01_01_01 = 0x55
    for (32..96) |i| w[i] = 0x55;
    // raw_scales[96..108]: lo nibble=9, hi nibble=9 → 0x99
    // scales[j<8] = (raw[j] & 0xF) - 8 = 9-8 = 1
    // scales[j>=8] = (raw[j-8] >> 4) - 8 = 9-8 = 1
    for (96..108) |i| w[i] = 0x99;
    // d = f16(1.0) at offset 108
    w[108] = 0x00;
    w[109] = 0x3C;
    var x: [bs]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvQ3_K(&x, &w, &y, 1, bs);
    try std.testing.expectApproxEqAbs(@as(f32, 256.0), y[0], 2.0);
}
