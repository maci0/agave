//! CPU GEMV kernels for less common quantization formats.
//! Q4_1, Q5_0, Q2_K, Q3_K — scalar implementations.

const std = @import("std");
const quant = @import("../../../ops/quant.zig");

/// Q4_1: 32 values per block, 20 bytes (f16 scale + f16 min + 16 nibble-packed bytes)
pub fn gemvQ4_1(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb: usize = 20;
    const qk: usize = 32;
    const nb = (k + qk - 1) / qk;
    for (0..n) |row| {
        var sum: f32 = 0.0;
        const rp = w + row * nb * bpb;
        for (0..nb) |b| {
            const bp = rp + b * bpb;
            const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
            const m: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[2..4], .little))));
            const bk = b * qk;
            for (0..qk / 2) |j| {
                const byte = bp[4 + j];
                const gi0 = bk + j;
                const gi1 = bk + j + qk / 2;
                if (gi0 < k) sum += x[gi0] * (@as(f32, @floatFromInt(@as(u8, byte & 0x0F))) * d + m);
                if (gi1 < k) sum += x[gi1] * (@as(f32, @floatFromInt(@as(u8, byte >> 4))) * d + m);
            }
        }
        y[row] = sum;
    }
}

/// Q5_0: 32 values per block, 22 bytes (f16 scale + 4 bytes qh + 16 nibble-packed bytes)
pub fn gemvQ5_0(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb: usize = 22;
    const qk: usize = 32;
    const nb = (k + qk - 1) / qk;
    for (0..n) |row| {
        var sum: f32 = 0.0;
        const rp = w + row * nb * bpb;
        for (0..nb) |b| {
            const bp = rp + b * bpb;
            const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
            const qh = std.mem.readInt(u32, bp[2..6], .little);
            const bk = b * qk;
            var block_sum: f32 = 0.0;
            for (0..16) |j| {
                const byte = bp[6 + j];
                const lo_nib: u8 = byte & 0x0F;
                const hi_nib: u8 = byte >> 4;
                const hb0: u8 = @truncate((qh >> @intCast(j)) & 1);
                const hb1: u8 = @truncate((qh >> @intCast(j + 16)) & 1);
                const v0: i8 = @as(i8, @intCast(lo_nib | (hb0 << 4))) - 16;
                const v1: i8 = @as(i8, @intCast(hi_nib | (hb1 << 4))) - 16;
                const gi0 = bk + j;
                const gi1 = bk + j + 16;
                if (gi0 < k) block_sum += x[gi0] * @as(f32, @floatFromInt(v0));
                if (gi1 < k) block_sum += x[gi1] * @as(f32, @floatFromInt(v1));
            }
            sum += block_sum * d;
        }
        y[row] = sum;
    }
}

/// Q2_K: 256 values per super-block, 84 bytes
/// Layout: scales[16] + qs[64] + d(f16) + dmin(f16)
pub fn gemvQ2_K(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb: usize = 84;
    const bs: usize = 256;
    const nb = (k + bs - 1) / bs;
    for (0..n) |row| {
        var sum: f32 = 0.0;
        const rp = w + row * nb * bpb;
        for (0..nb) |b| {
            const bp = rp + b * bpb;
            const scales = bp[0..16];
            const qs = bp + 16;
            const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[80..82], .little))));
            const dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[82..84], .little))));
            const bk = b * bs;
            for (0..16) |sb| {
                const sc: f32 = @floatFromInt(scales[sb] & 0x0F);
                const m: f32 = @floatFromInt(scales[sb] >> 4);
                const d_sc = d * sc;
                const dm_m = dmin * m;
                for (0..16) |l| {
                    const gi = bk + sb * 16 + l;
                    if (gi >= k) break;
                    const qi = sb * 16 + l;
                    const byte_idx = qi / 4;
                    const shift: u3 = @intCast((qi % 4) * 2);
                    const q: f32 = @floatFromInt((qs[byte_idx] >> shift) & 0x03);
                    sum += x[gi] * (d_sc * q - dm_m);
                }
            }
        }
        y[row] = sum;
    }
}

/// Q3_K: 256 values per super-block, 110 bytes
/// Layout: hmask[32] + qs[64] + scales[12] + d(f16)
pub fn gemvQ3_K(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb: usize = 110;
    const bs: usize = 256;
    const nb = (k + bs - 1) / bs;
    for (0..n) |row| {
        var sum: f32 = 0.0;
        const rp = w + row * nb * bpb;
        for (0..nb) |b| {
            const bp = rp + b * bpb;
            const hmask = bp[0..32];
            const qs = bp + 32;
            const raw_scales = bp[96..108];
            const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[108..110], .little))));
            const bk = b * bs;

            var scales: [16]i8 = undefined;
            for (0..8) |j| {
                scales[j] = @as(i8, @intCast(raw_scales[j] & 0x0F)) - 8;
            }
            for (0..8) |j| {
                scales[8 + j] = @as(i8, @intCast(raw_scales[j] >> 4)) - 8;
            }

            for (0..256) |l| {
                const gi = bk + l;
                if (gi >= k) break;
                const byte_idx = l / 4;
                const shift: u3 = @intCast((l % 4) * 2);
                const q_lo: u8 = (qs[byte_idx] >> shift) & 0x03;
                const hm_byte = l % 32;
                const hm_bit: u3 = @intCast(l / 32);
                const q_hi: u8 = (hmask[hm_byte] >> hm_bit) & 1;
                const q3: i8 = @as(i8, @intCast(q_lo | (q_hi << 2))) - 4;
                const sb = l / 16;
                sum += x[gi] * d * @as(f32, @floatFromInt(scales[sb])) * @as(f32, @floatFromInt(q3));
            }
        }
        y[row] = sum;
    }
}
