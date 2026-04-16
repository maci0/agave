//! CPU GEMV kernels for IQ4 quantization formats.
//! IQ4_NL (non-linear lookup, 32-element blocks) and IQ4_XS (256-element super-blocks).
//! 2-row batched to share x-vector cache reads.

const std = @import("std");
const quant = @import("../../../ops/quant.zig");
const backend_mod = @import("../../backend.zig");

/// IQ4_XS scale bias: 6-bit unsigned [0..63] centered to signed [-32..31].
const iq4_xs_scale_bias: i32 = -32;
/// IQ4_XS high-scale 2-bit mask.
const iq4_xs_scale_hi_mask: u16 = 0x03;

/// IQ4_NL: 32 values per block, 18 bytes (f16 scale + 16 nibble-packed bytes)
/// Uses a non-linear lookup table instead of linear dequant.
/// 2-row batched to share x-vector cache reads.
pub fn gemvIQ4_NL(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb = backend_mod.iq4_nl_block_bytes;
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
            const bk = b * qk;
            var block_sum0: f32 = 0.0;
            var block_sum1: f32 = 0.0;
            if (bk + qk - 1 < k) {
                for (0..qk / 2) |j| {
                    const byte0 = bp0[2 + j];
                    const byte1 = bp1[2 + j];
                    const xlo = x[bk + j];
                    const xhi = x[bk + j + qk / 2];
                    block_sum0 += xlo * @as(f32, @floatFromInt(quant.iq4nl_table[byte0 & 0x0F])) +
                        xhi * @as(f32, @floatFromInt(quant.iq4nl_table[byte0 >> 4]));
                    block_sum1 += xlo * @as(f32, @floatFromInt(quant.iq4nl_table[byte1 & 0x0F])) +
                        xhi * @as(f32, @floatFromInt(quant.iq4nl_table[byte1 >> 4]));
                }
            } else {
                for (0..qk / 2) |j| {
                    const byte0 = bp0[2 + j];
                    const byte1 = bp1[2 + j];
                    const gi0 = bk + j;
                    const gi1 = bk + j + qk / 2;
                    if (gi0 < k) {
                        const xv = x[gi0];
                        block_sum0 += xv * @as(f32, @floatFromInt(quant.iq4nl_table[byte0 & 0x0F]));
                        block_sum1 += xv * @as(f32, @floatFromInt(quant.iq4nl_table[byte1 & 0x0F]));
                    }
                    if (gi1 < k) {
                        const xv = x[gi1];
                        block_sum0 += xv * @as(f32, @floatFromInt(quant.iq4nl_table[byte0 >> 4]));
                        block_sum1 += xv * @as(f32, @floatFromInt(quant.iq4nl_table[byte1 >> 4]));
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
            const bk = b * qk;
            var block_sum: f32 = 0.0;
            if (bk + qk - 1 < k) {
                for (0..qk / 2) |j| {
                    const byte = bp[2 + j];
                    block_sum += x[bk + j] * @as(f32, @floatFromInt(quant.iq4nl_table[byte & 0x0F])) +
                        x[bk + j + qk / 2] * @as(f32, @floatFromInt(quant.iq4nl_table[byte >> 4]));
                }
            } else {
                for (0..qk / 2) |j| {
                    const byte = bp[2 + j];
                    const gi0 = bk + j;
                    const gi1 = bk + j + qk / 2;
                    if (gi0 < k) block_sum += x[gi0] * @as(f32, @floatFromInt(quant.iq4nl_table[byte & 0x0F]));
                    if (gi1 < k) block_sum += x[gi1] * @as(f32, @floatFromInt(quant.iq4nl_table[byte >> 4]));
                }
            }
            sum += block_sum * d;
        }
        y[row] = sum;
    }
}

/// IQ4_XS: 256 values per super-block, 136 bytes
/// Layout: f16 d + u16 scales_h + u8 scales_l[4] + qs[128]
/// 8 sub-blocks of 32 elements; each uses iq4nl_table for dequant.
/// 2-row batched to share x-vector cache reads.
pub fn gemvIQ4_XS(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb = backend_mod.iq4_xs_block_bytes;
    const super_block_size = backend_mod.quant_super_block_elems;
    const nb = (k + super_block_size - 1) / super_block_size;
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
            const scales_h0 = std.mem.readInt(u16, bp0[2..4], .little);
            const scales_h1 = std.mem.readInt(u16, bp1[2..4], .little);
            const scales_l0 = bp0[4..8];
            const scales_l1 = bp1[4..8];
            const qs0 = bp0 + 8;
            const qs1 = bp1 + 8;
            const bk = b * super_block_size;

            for (0..8) |sb| {
                const lo4_0: u8 = if (sb % 2 == 0) scales_l0[sb / 2] & 0x0F else scales_l0[sb / 2] >> 4;
                const hi2_0: u8 = @truncate((scales_h0 >> @intCast(sb * 2)) & @as(u16, iq4_xs_scale_hi_mask));
                const scale_raw0: i32 = @as(i32, lo4_0 | (@as(u8, hi2_0) << 4)) + iq4_xs_scale_bias;
                const sub_scale0: f32 = d0 * @as(f32, @floatFromInt(scale_raw0));

                const lo4_1: u8 = if (sb % 2 == 0) scales_l1[sb / 2] & 0x0F else scales_l1[sb / 2] >> 4;
                const hi2_1: u8 = @truncate((scales_h1 >> @intCast(sb * 2)) & @as(u16, iq4_xs_scale_hi_mask));
                const scale_raw1: i32 = @as(i32, lo4_1 | (@as(u8, hi2_1) << 4)) + iq4_xs_scale_bias;
                const sub_scale1: f32 = d1 * @as(f32, @floatFromInt(scale_raw1));

                const sub_qs0 = qs0 + sb * 16;
                const sub_qs1 = qs1 + sb * 16;
                const sub_bk = bk + sb * 32;
                var block_sum0: f32 = 0.0;
                var block_sum1: f32 = 0.0;

                const sub_block_elems = backend_mod.quant_block_elems;
                const half_sub = sub_block_elems / 2;
                if (sub_bk + sub_block_elems - 1 < k) {
                    for (0..half_sub) |j| {
                        const byte0 = sub_qs0[j];
                        const byte1 = sub_qs1[j];
                        const xlo = x[sub_bk + j];
                        const xhi = x[sub_bk + j + half_sub];
                        block_sum0 += xlo * @as(f32, @floatFromInt(quant.iq4nl_table[byte0 & 0x0F])) +
                            xhi * @as(f32, @floatFromInt(quant.iq4nl_table[byte0 >> 4]));
                        block_sum1 += xlo * @as(f32, @floatFromInt(quant.iq4nl_table[byte1 & 0x0F])) +
                            xhi * @as(f32, @floatFromInt(quant.iq4nl_table[byte1 >> 4]));
                    }
                } else {
                    for (0..half_sub) |j| {
                        const byte0 = sub_qs0[j];
                        const byte1 = sub_qs1[j];
                        const gi0 = sub_bk + j;
                        const gi1 = sub_bk + j + half_sub;
                        if (gi0 < k) {
                            const xv = x[gi0];
                            block_sum0 += xv * @as(f32, @floatFromInt(quant.iq4nl_table[byte0 & 0x0F]));
                            block_sum1 += xv * @as(f32, @floatFromInt(quant.iq4nl_table[byte1 & 0x0F]));
                        }
                        if (gi1 < k) {
                            const xv = x[gi1];
                            block_sum0 += xv * @as(f32, @floatFromInt(quant.iq4nl_table[byte0 >> 4]));
                            block_sum1 += xv * @as(f32, @floatFromInt(quant.iq4nl_table[byte1 >> 4]));
                        }
                    }
                }
                sum0 += block_sum0 * sub_scale0;
                sum1 += block_sum1 * sub_scale1;
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
            const scales_h = std.mem.readInt(u16, bp[2..4], .little);
            const scales_l = bp[4..8];
            const qs = bp + 8;
            const bk = b * super_block_size;

            for (0..8) |sb| {
                const lo4: u8 = if (sb % 2 == 0) scales_l[sb / 2] & 0x0F else scales_l[sb / 2] >> 4;
                const hi2: u8 = @truncate((scales_h >> @intCast(sb * 2)) & @as(u16, iq4_xs_scale_hi_mask));
                const scale_raw: i32 = @as(i32, lo4 | (@as(u8, hi2) << 4)) + iq4_xs_scale_bias;
                const sub_scale: f32 = d * @as(f32, @floatFromInt(scale_raw));

                const sub_qs = qs + sb * 16;
                const sub_bk = bk + sb * 32;
                var block_sum: f32 = 0.0;

                const sub_block_elems = backend_mod.quant_block_elems;
                const half_sub = sub_block_elems / 2;
                if (sub_bk + sub_block_elems - 1 < k) {
                    for (0..half_sub) |j| {
                        const byte = sub_qs[j];
                        block_sum += x[sub_bk + j] * @as(f32, @floatFromInt(quant.iq4nl_table[byte & 0x0F])) +
                            x[sub_bk + j + half_sub] * @as(f32, @floatFromInt(quant.iq4nl_table[byte >> 4]));
                    }
                } else {
                    for (0..half_sub) |j| {
                        const byte = sub_qs[j];
                        const gi0 = sub_bk + j;
                        const gi1 = sub_bk + j + half_sub;
                        if (gi0 < k) block_sum += x[gi0] * @as(f32, @floatFromInt(quant.iq4nl_table[byte & 0x0F]));
                        if (gi1 < k) block_sum += x[gi1] * @as(f32, @floatFromInt(quant.iq4nl_table[byte >> 4]));
                    }
                }
                sum += block_sum * sub_scale;
            }
        }
        y[row] = sum;
    }
}

// ── Tests ─────────────────────────────────────────────────────────

test "gemvIQ4_NL zero scale produces zero output" {
    // d=0.0 → all weights zero regardless of nibble values.
    const bpb = backend_mod.iq4_nl_block_bytes; // 18
    var w: [2 * bpb]u8 = undefined;
    for (0..2) |r| {
        const base = r * bpb;
        // f16(0.0) = 0x0000
        w[base] = 0x00;
        w[base + 1] = 0x00;
        // Fill nibbles with non-zero values to ensure scale zeroes them
        for (0..16) |i| w[base + 2 + i] = 0xFF;
    }
    var x: [32]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [2]f32 = undefined;
    gemvIQ4_NL(&x, &w, &y, 2, 32);
    for (0..2) |i| try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[i], 0.01);
}

test "gemvIQ4_NL uniform nibble 8" {
    // iq4nl_table[8] = 1. All nibbles=8 → byte = 0x88.
    // d=1.0, x=all 1.0 → y = 1.0 * (32 * 1) = 32.0
    const bpb = backend_mod.iq4_nl_block_bytes;
    var w: [2 * bpb]u8 = undefined;
    for (0..2) |r| {
        const base = r * bpb;
        // f16(1.0) = 0x3C00 little-endian
        w[base] = 0x00;
        w[base + 1] = 0x3C;
        for (0..16) |i| w[base + 2 + i] = 0x88;
    }
    var x: [32]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [2]f32 = undefined;
    gemvIQ4_NL(&x, &w, &y, 2, 32);
    for (0..2) |i| try std.testing.expectApproxEqAbs(@as(f32, 32.0), y[i], 0.5);
}

test "gemvIQ4_NL single row" {
    // n=1 exercises single-row path. d=0.5, nibble=9 → table[9]=13.
    // y = 0.5 * (32 * 13) = 208.0
    const bpb = backend_mod.iq4_nl_block_bytes;
    var w: [bpb]u8 = undefined;
    // f16(0.5) = 0x3800
    w[0] = 0x00;
    w[1] = 0x38;
    for (0..16) |i| w[2 + i] = 0x99; // lo=9, hi=9
    var x: [32]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvIQ4_NL(&x, &w, &y, 1, 32);
    try std.testing.expectApproxEqAbs(@as(f32, 208.0), y[0], 1.0);
}

test "gemvIQ4_XS zero scale produces zero output" {
    // d=0.0 → all sub-block scales are 0 → output is 0.
    const bpb = backend_mod.iq4_xs_block_bytes; // 136
    const bs = backend_mod.quant_super_block_elems; // 256
    var w: [bpb]u8 = undefined;
    @memset(&w, 0);
    // d = f16(0.0) at offset 0..2, already 0
    // Fill nibbles with non-zero values (qs at offset 8..136)
    for (8..136) |i| w[i] = 0xFF;
    var x: [bs]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvIQ4_XS(&x, &w, &y, 1, bs);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[0], 0.01);
}

test "gemvIQ4_XS uniform nibble 8" {
    // iq4nl_table[8] = 1. d=1.0, all sub-scale_raw = 32+1 = 33 → scale = 33-32 = 1.
    // Each sub-block: 32 elements × 1 × 1.0 = 32. 8 sub-blocks → 256. d=1.0 → y=256.
    const bpb = backend_mod.iq4_xs_block_bytes;
    const bs = backend_mod.quant_super_block_elems;
    var w: [bpb]u8 = undefined;
    @memset(&w, 0);
    // d = f16(1.0) = 0x3C00
    w[0] = 0x00;
    w[1] = 0x3C;
    // scales_h (offset 2..4): hi2 bits. For scale_raw=33: lo4=1, hi2=2 → raw = 1|(2<<4) = 33.
    // scales_h: each sub-block needs 2 bits = 0b10. 8 sub-blocks → 16 bits = 0xAAAA.
    w[2] = 0xAA;
    w[3] = 0xAA;
    // scales_l (offset 4..8): packed lo4 nibbles. Even sub-blocks in lo nibble, odd in hi.
    // lo4 = 1 for all → byte = 0x11 for each pair.
    for (4..8) |i| w[i] = 0x11;
    // qs (offset 8..136): nibble=8 → byte = 0x88
    for (8..136) |i| w[i] = 0x88;
    var x: [bs]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvIQ4_XS(&x, &w, &y, 1, bs);
    try std.testing.expectApproxEqAbs(@as(f32, 256.0), y[0], 2.0);
}
