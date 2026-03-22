//! CPU GEMV kernels for IQ4 quantization formats.
//! IQ4_NL (non-linear lookup, 32-element blocks) and IQ4_XS (256-element super-blocks).

const std = @import("std");
const quant = @import("../../../ops/quant.zig");

/// IQ4_NL: 32 values per block, 18 bytes (f16 scale + 16 nibble-packed bytes)
/// Uses a non-linear lookup table instead of linear dequant.
pub fn gemvIQ4_NL(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb: usize = 18;
    const qk: usize = 32;
    const nb = (k + qk - 1) / qk;
    const row_bytes = nb * bpb;

    for (0..n) |row| {
        var sum: f32 = 0.0;
        const rp = w + row * row_bytes;
        for (0..nb) |b| {
            const bp = rp + b * bpb;
            const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
            const bk = b * qk;
            var block_sum: f32 = 0.0;
            for (0..qk / 2) |j| {
                const byte = bp[2 + j];
                const lo_nib: u8 = byte & 0x0F;
                const hi_nib: u8 = byte >> 4;
                const gi0 = bk + j;
                const gi1 = bk + j + qk / 2;
                if (gi0 < k) block_sum += x[gi0] * @as(f32, @floatFromInt(quant.iq4nl_table[lo_nib]));
                if (gi1 < k) block_sum += x[gi1] * @as(f32, @floatFromInt(quant.iq4nl_table[hi_nib]));
            }
            sum += block_sum * d;
        }
        y[row] = sum;
    }
}

/// IQ4_XS: 256 values per super-block, 138 bytes
pub fn gemvIQ4_XS(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb: usize = 138;
    const super_block_size: usize = 256;
    const nb = (k + super_block_size - 1) / super_block_size;
    const row_bytes = nb * bpb;

    for (0..n) |row| {
        var sum: f32 = 0.0;
        const rp = w + row * row_bytes;
        for (0..nb) |b| {
            const bp = rp + b * bpb;
            const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
            const scales_h = std.mem.readInt(u16, bp[2..4], .little);
            const scales_l = bp[4..12];
            const qs = bp + 12;
            const bk = b * super_block_size;

            for (0..8) |sb| {
                const lo4: u8 = if (sb % 2 == 0) scales_l[sb / 2] & 0x0F else scales_l[sb / 2] >> 4;
                const hi2: u8 = @truncate((scales_h >> @intCast(sb * 2)) & 0x03);
                const scale_raw: i32 = @as(i32, lo4 | (@as(u8, hi2) << 4)) - 32;
                const sub_scale: f32 = d * @as(f32, @floatFromInt(scale_raw));

                const sub_qs = qs + sb * 16;
                const sub_bk = bk + sb * 32;
                var block_sum: f32 = 0.0;

                for (0..16) |j| {
                    const byte = sub_qs[j];
                    const lo_nib: u8 = byte & 0x0F;
                    const hi_nib: u8 = byte >> 4;
                    const gi0 = sub_bk + j;
                    const gi1 = sub_bk + j + 16;
                    if (gi0 < k) block_sum += x[gi0] * @as(f32, @floatFromInt(quant.iq4nl_table[lo_nib]));
                    if (gi1 < k) block_sum += x[gi1] * @as(f32, @floatFromInt(quant.iq4nl_table[hi_nib]));
                }
                sum += block_sum * sub_scale;
            }
        }
        y[row] = sum;
    }
}
