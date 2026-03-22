//! CPU embedding dequantization kernels.
//! Each kernel looks up a token embedding row and converts to f32.

const std = @import("std");
const quant = @import("../../../ops/quant.zig");
const DType = @import("../../backend.zig").DType;

/// Dispatches embedding lookup to the appropriate dequant kernel.
pub fn embLookup(data: [*]const u8, dtype: DType, token_id: u32, output: [*]f32, dim: usize) void {
    switch (dtype) {
        .q4_0 => embQ4_0(data, token_id, output, dim),
        .q5_0 => embQ5_0(data, token_id, output, dim),
        .q8_0 => embQ8_0(data, token_id, output, dim),
        .q6_k => embQ6_K(data, token_id, output, dim),
        .mxfp4 => embMXFP4(data, token_id, output, dim),
        .bf16 => {
            const w: [*]const u16 = @ptrCast(@alignCast(data));
            for (0..dim) |i| output[i] = quant.bf16ToF32(w[token_id * dim + i]);
        },
        .f16 => {
            const w: [*]const f16 = @ptrCast(@alignCast(data));
            for (0..dim) |i| output[i] = @floatCast(w[token_id * dim + i]);
        },
        .f32 => {
            const w: [*]const f32 = @ptrCast(@alignCast(data));
            @memcpy(output[0..dim], w[token_id * dim ..][0..dim]);
        },
        .q4_k => embQ4_K(data, token_id, output, dim),
        .q5_k => embQ5_K(data, token_id, output, dim),
        else => @memset(output[0..dim], 0), // unsupported dtype — zero output
    }
}

/// Dequantizes a Q4_0 embedding row to f32.
pub fn embQ4_0(data: [*]const u8, tok: u32, out: [*]f32, dim: usize) void {
    const bpb: usize = 18;
    const qk: usize = 32;
    const nb = (dim + qk - 1) / qk;
    const rp = data + tok * nb * bpb;
    for (0..nb) |b| {
        const bp = rp + b * bpb;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
        for (0..qk / 2) |j| {
            const byte = bp[2 + j];
            const x0 = @as(i8, @intCast(byte & 0x0F)) - 8;
            const x1 = @as(i8, @intCast(byte >> 4)) - 8;
            const gi0 = b * qk + j;
            const gi1 = b * qk + j + qk / 2;
            if (gi0 < dim) out[gi0] = @as(f32, @floatFromInt(x0)) * d;
            if (gi1 < dim) out[gi1] = @as(f32, @floatFromInt(x1)) * d;
        }
    }
}

/// Dequantizes a Q8_0 embedding row to f32.
pub fn embQ8_0(data: [*]const u8, tok: u32, out: [*]f32, dim: usize) void {
    const bpb: usize = 34;
    const nb = (dim + 31) / 32;
    const rp = data + tok * nb * bpb;
    for (0..nb) |b| {
        const bp = rp + b * bpb;
        const s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
        for (0..32) |i| {
            const idx = b * 32 + i;
            if (idx < dim) {
                out[idx] = @as(f32, @floatFromInt(@as(i8, @bitCast(bp[2 + i])))) * s;
            }
        }
    }
}

/// Dequantizes a Q5_0 embedding row to f32.
pub fn embQ5_0(data: [*]const u8, tok: u32, out: [*]f32, dim: usize) void {
    const bpb: usize = 22;
    const qk: usize = 32;
    const nb = (dim + qk - 1) / qk;
    const rp = data + tok * nb * bpb;
    for (0..nb) |b| {
        const bp = rp + b * bpb;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
        const qh = std.mem.readInt(u32, bp[2..6], .little);
        for (0..16) |j| {
            const byte = bp[6 + j];
            const lo_nib: u8 = byte & 0x0F;
            const hi_nib: u8 = byte >> 4;
            const hb0: u8 = @truncate((qh >> @intCast(j)) & 1);
            const hb1: u8 = @truncate((qh >> @intCast(j + 16)) & 1);
            const v0: i8 = @as(i8, @intCast(lo_nib | (hb0 << 4))) - 16;
            const v1: i8 = @as(i8, @intCast(hi_nib | (hb1 << 4))) - 16;
            const gi0 = b * qk + j;
            const gi1 = b * qk + j + 16;
            if (gi0 < dim) out[gi0] = @as(f32, @floatFromInt(v0)) * d;
            if (gi1 < dim) out[gi1] = @as(f32, @floatFromInt(v1)) * d;
        }
    }
}

/// Dequantizes a Q6_K embedding row to f32.
pub fn embQ6_K(data: [*]const u8, tok: u32, out: [*]f32, dim: usize) void {
    const bpb: usize = 210;
    const bs: usize = 256;
    const nb = (dim + bs - 1) / bs;
    const rp = data + tok * nb * bpb;
    for (0..nb) |b| {
        const bp = rp + b * bpb;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[208..210], .little))));
        var chunk: usize = 0;
        while (chunk < 2) : (chunk += 1) {
            const ql = bp + chunk * 64;
            const qh = bp + 128 + chunk * 32;
            const sc: [*]const i8 = @ptrCast(bp + 192 + chunk * 8);
            const base = b * bs + chunk * 128;
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
                if (gi0 < dim) out[gi0] = d * @as(f32, @floatFromInt(sc[is + 0])) * @as(f32, @floatFromInt(q1));
                if (gi1 < dim) out[gi1] = d * @as(f32, @floatFromInt(sc[is + 2])) * @as(f32, @floatFromInt(q2));
                if (gi2 < dim) out[gi2] = d * @as(f32, @floatFromInt(sc[is + 4])) * @as(f32, @floatFromInt(q3));
                if (gi3 < dim) out[gi3] = d * @as(f32, @floatFromInt(sc[is + 6])) * @as(f32, @floatFromInt(q4));
            }
        }
    }
}

/// Dequantizes a Q4_K embedding row to f32.
/// 256 elements per super-block, 144 bytes: d(f16) + dmin(f16) + scales[12] + qs[128].
pub fn embQ4_K(data: [*]const u8, tok: u32, out: [*]f32, dim: usize) void {
    const bpb: usize = 144;
    const bs: usize = 256;
    const nb = (dim + bs - 1) / bs;
    const rp = data + @as(usize, tok) * nb * bpb;
    for (0..nb) |b| {
        const bp = rp + b * bpb;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
        const dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[2..4], .little))));
        const scales = bp[4..16];
        const qs = bp + 16;
        for (0..8) |sb| {
            const gi_base = b * bs + sb * 32;
            if (gi_base >= dim) break;
            var sc: u8 = undefined;
            var m: u8 = undefined;
            quant.getScaleMinK4(sb, scales, &sc, &m);
            const d_sc = d * @as(f32, @floatFromInt(sc));
            const dm_m = dmin * @as(f32, @floatFromInt(m));
            const qi_base = sb * 32;
            for (0..32) |l| {
                const gi = gi_base + l;
                if (gi >= dim) break;
                const qi = qi_base + l;
                const byte_idx = qi / 2;
                const qv: f32 = @floatFromInt(if (qi % 2 == 0) qs[byte_idx] & 0x0F else qs[byte_idx] >> 4);
                out[gi] = d_sc * qv - dm_m;
            }
        }
    }
}

/// Dequantizes a Q5_K embedding row to f32.
/// 256 elements per super-block, 176 bytes: d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128].
pub fn embQ5_K(data: [*]const u8, tok: u32, out: [*]f32, dim: usize) void {
    const bpb: usize = 176;
    const bs: usize = 256;
    const nb = (dim + bs - 1) / bs;
    const rp = data + @as(usize, tok) * nb * bpb;
    for (0..nb) |b| {
        const bp = rp + b * bpb;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
        const dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[2..4], .little))));
        const scales = bp[4..16];
        const qh = bp + 16;
        const qs = bp + 48;
        for (0..8) |sb| {
            const gi_base = b * bs + sb * 32;
            if (gi_base >= dim) break;
            var sc: u8 = undefined;
            var m: u8 = undefined;
            quant.getScaleMinK4(sb, scales, &sc, &m);
            const d_sc = d * @as(f32, @floatFromInt(sc));
            const dm_m = dmin * @as(f32, @floatFromInt(m));
            const qi_base = sb * 32;
            for (0..32) |l| {
                const gi = gi_base + l;
                if (gi >= dim) break;
                const qi = qi_base + l;
                const byte_idx = qi / 2;
                const lo: u8 = if (qi % 2 == 0) qs[byte_idx] & 0x0F else qs[byte_idx] >> 4;
                const hi: u8 = @truncate((qh[qi / 8] >> @intCast(qi % 8)) & 1);
                const qv: f32 = @floatFromInt(lo | (hi << 4));
                out[gi] = d_sc * qv - dm_m;
            }
        }
    }
}

/// Dequantizes an MXFP4 embedding row to f32.
pub fn embMXFP4(data: [*]const u8, tok: u32, out: [*]f32, dim: usize) void {
    const bpb: usize = 17;
    const qk: usize = 32;
    const nb = (dim + qk - 1) / qk;
    const rp = data + tok * nb * bpb;
    for (0..nb) |b| {
        const bp = rp + b * bpb;
        const d = quant.e8m0ToF32(bp[0]);
        for (0..qk / 2) |j| {
            const byte = bp[1 + j];
            const v0 = quant.mxfp4Lookup(byte & 0x0F);
            const v1 = quant.mxfp4Lookup(byte >> 4);
            const gi0 = b * qk + j;
            const gi1 = b * qk + j + qk / 2;
            if (gi0 < dim) out[gi0] = v0 * d;
            if (gi1 < dim) out[gi1] = v1 * d;
        }
    }
}
