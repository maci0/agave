//! CPU embedding dequantization kernels.
//! Each kernel looks up a token embedding row and converts to f32.

const std = @import("std");
const quant = @import("../../../ops/quant.zig");
const backend_mod = @import("../../backend.zig");
const DType = backend_mod.DType;

// Canonical block size constants (from backend.zig).
const quant_block_elems = backend_mod.quant_block_elems;
const quant_super_block_elems = backend_mod.quant_super_block_elems;
const q4_0_block_bytes = backend_mod.q4_0_block_bytes;
const q5_0_block_bytes = backend_mod.q5_0_block_bytes;
const q8_0_block_bytes = backend_mod.q8_0_block_bytes;
const q6_k_block_bytes = backend_mod.q6_k_block_bytes;
const q4_k_block_bytes = backend_mod.q4_k_block_bytes;
const q5_k_block_bytes = backend_mod.q5_k_block_bytes;
const mxfp4_block_bytes = backend_mod.mxfp4_block_bytes;

// Q6_K block layout constants (210-byte super-block).
const q6_k_ql_chunk_bytes: usize = 64;
const q6_k_qh_offset: usize = 128;
const q6_k_qh_chunk_bytes: usize = 32;
const q6_k_sc_offset: usize = 192;
const q6_k_sc_chunk_bytes: usize = 8;
const q6_k_d_offset: usize = 208;
const q6_k_chunk_elems = quant_super_block_elems / 2;
/// Q4_0 dequant bias: 4-bit unsigned [0..15] centered to signed [-8..7].
const q4_0_dequant_bias: i8 = -8;
/// Q5_0 dequant bias: 5-bit unsigned [0..31] centered to signed [-16..15].
const q5_0_dequant_bias: i8 = -16;
/// Q6_K dequant bias: 6-bit unsigned [0..63] centered to signed [-32..31].
const q6_k_dequant_bias: i8 = -32;
/// Q5_K high-bit contribution: the 5th bit adds 2^4 = 16 to the value.
const q5_k_high_bit_value: f32 = 16.0;
/// Elements per Q4_K/Q5_K group (super-block / 4 groups).
const group_elems = quant_super_block_elems / 4;
/// Mask for extracting 2-bit high-order field from qh byte.
const qh_2bit_mask: u8 = 3;

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
            const base = token_id * dim;
            const V8u16 = @Vector(8, u16);
            const V8u32 = @Vector(8, u32);
            const shift: V8u32 = @splat(16);
            var i: usize = 0;
            while (i + 8 <= dim) : (i += 8) {
                const raw: V8u32 = @intCast(@as(V8u16, w[base + i ..][0..8].*));
                output[i..][0..8].* = @bitCast(raw << shift);
            }
            while (i < dim) : (i += 1) output[i] = quant.bf16ToF32(w[base + i]);
        },
        .f16 => {
            const w: [*]const f16 = @ptrCast(@alignCast(data));
            const base = token_id * dim;
            const V8f16 = @Vector(8, f16);
            const V8f32 = @Vector(8, f32);
            var i: usize = 0;
            while (i + 8 <= dim) : (i += 8) {
                output[i..][0..8].* = @as(V8f32, @floatCast(@as(V8f16, w[base + i ..][0..8].*)));
            }
            while (i < dim) : (i += 1) output[i] = @floatCast(w[base + i]);
        },
        .f32 => {
            const w: [*]const f32 = @ptrCast(@alignCast(data));
            @memcpy(output[0..dim], w[token_id * dim ..][0..dim]);
        },
        .q4_k => embQ4_K(data, token_id, output, dim),
        .q5_k => embQ5_K(data, token_id, output, dim),
        else => @panic("embLookup: unsupported embedding dtype"),
    }
}

/// Dequantizes a Q4_0 embedding row to f32.
pub fn embQ4_0(data: [*]const u8, tok: u32, out: [*]f32, dim: usize) void {
    const nb = (dim + quant_block_elems - 1) / quant_block_elems;
    const rp = data + tok * nb * q4_0_block_bytes;
    for (0..nb) |b| {
        const bp = rp + b * q4_0_block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
        for (0..quant_block_elems / 2) |j| {
            const byte = bp[2 + j];
            const x0 = @as(i8, @intCast(byte & 0x0F)) + q4_0_dequant_bias;
            const x1 = @as(i8, @intCast(byte >> 4)) + q4_0_dequant_bias;
            const gi0 = b * quant_block_elems + j;
            const gi1 = b * quant_block_elems + j + quant_block_elems / 2;
            if (gi0 < dim) out[gi0] = @as(f32, @floatFromInt(x0)) * d;
            if (gi1 < dim) out[gi1] = @as(f32, @floatFromInt(x1)) * d;
        }
    }
}

/// Dequantizes a Q8_0 embedding row to f32.
pub fn embQ8_0(data: [*]const u8, tok: u32, out: [*]f32, dim: usize) void {
    const nb = (dim + quant_block_elems - 1) / quant_block_elems;
    const rp = data + tok * nb * q8_0_block_bytes;
    for (0..nb) |b| {
        const bp = rp + b * q8_0_block_bytes;
        const s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
        for (0..quant_block_elems) |i| {
            const idx = b * quant_block_elems + i;
            if (idx < dim) {
                out[idx] = @as(f32, @floatFromInt(@as(i8, @bitCast(bp[2 + i])))) * s;
            }
        }
    }
}

/// Dequantizes a Q5_0 embedding row to f32.
pub fn embQ5_0(data: [*]const u8, tok: u32, out: [*]f32, dim: usize) void {
    const nb = (dim + quant_block_elems - 1) / quant_block_elems;
    const rp = data + tok * nb * q5_0_block_bytes;
    for (0..nb) |b| {
        const bp = rp + b * q5_0_block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
        const qh = std.mem.readInt(u32, bp[2..6], .little);
        const half = quant_block_elems / 2;
        for (0..half) |j| {
            const byte = bp[6 + j];
            const lo_nib: u8 = byte & 0x0F;
            const hi_nib: u8 = byte >> 4;
            const hb0: u8 = @truncate((qh >> @intCast(j)) & 1);
            const hb1: u8 = @truncate((qh >> @intCast(j + half)) & 1);
            const v0: i8 = @as(i8, @intCast(lo_nib | (hb0 << 4))) + q5_0_dequant_bias;
            const v1: i8 = @as(i8, @intCast(hi_nib | (hb1 << 4))) + q5_0_dequant_bias;
            const gi0 = b * quant_block_elems + j;
            const gi1 = b * quant_block_elems + j + half;
            if (gi0 < dim) out[gi0] = @as(f32, @floatFromInt(v0)) * d;
            if (gi1 < dim) out[gi1] = @as(f32, @floatFromInt(v1)) * d;
        }
    }
}

/// Dequantizes a Q6_K embedding row to f32.
pub fn embQ6_K(data: [*]const u8, tok: u32, out: [*]f32, dim: usize) void {
    const nb = (dim + quant_super_block_elems - 1) / quant_super_block_elems;
    const rp = data + tok * nb * q6_k_block_bytes;
    for (0..nb) |b| {
        const bp = rp + b * q6_k_block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[q6_k_d_offset..][0..2], .little))));
        var chunk: usize = 0;
        while (chunk < 2) : (chunk += 1) {
            const ql = bp + chunk * q6_k_ql_chunk_bytes;
            const qh = bp + q6_k_qh_offset + chunk * q6_k_qh_chunk_bytes;
            const sc: [*]const i8 = @ptrCast(bp + q6_k_sc_offset + chunk * q6_k_sc_chunk_bytes);
            const base = b * quant_super_block_elems + chunk * q6_k_chunk_elems;
            for (0..quant_block_elems) |l| {
                const is: usize = l / 16;
                const q1: i8 = @as(i8, @intCast((ql[l] & 0x0F) | ((@as(u8, @truncate(qh[l] >> 0)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                const q2: i8 = @as(i8, @intCast((ql[l + 32] & 0x0F) | ((@as(u8, @truncate(qh[l] >> 2)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                const q3: i8 = @as(i8, @intCast((ql[l] >> 4) | ((@as(u8, @truncate(qh[l] >> 4)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
                const q4: i8 = @as(i8, @intCast((ql[l + 32] >> 4) | ((@as(u8, @truncate(qh[l] >> 6)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
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
/// Layout: 4 groups of 64 elements, each group in 32 bytes.
///   Elements 0-31: LOW nibbles of bytes 0-31 (scale index 2g)
///   Elements 32-63: HIGH nibbles of bytes 0-31 (scale index 2g+1)
pub fn embQ4_K(data: [*]const u8, tok: u32, out: [*]f32, dim: usize) void {
    const nb = (dim + quant_super_block_elems - 1) / quant_super_block_elems;
    const rp = data + @as(usize, tok) * nb * q4_k_block_bytes;
    for (0..nb) |b| {
        const bp = rp + b * q4_k_block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
        const dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[2..4], .little))));
        const scales = bp[4..16];
        const qs = bp + 16;
        for (0..4) |g| {
            const ql_off = g * quant_block_elems;
            var sc_lo: u8 = undefined;
            var m_lo: u8 = undefined;
            var sc_hi: u8 = undefined;
            var m_hi: u8 = undefined;
            quant.getScaleMinK4(g * 2, scales, &sc_lo, &m_lo);
            quant.getScaleMinK4(g * 2 + 1, scales, &sc_hi, &m_hi);
            const d_lo = d * @as(f32, @floatFromInt(sc_lo));
            const dm_lo = dmin * @as(f32, @floatFromInt(m_lo));
            const d_hi = d * @as(f32, @floatFromInt(sc_hi));
            const dm_hi = dmin * @as(f32, @floatFromInt(m_hi));
            // First 32 elements: low nibbles
            for (0..quant_block_elems) |l| {
                const gi = b * quant_super_block_elems + g * group_elems + l;
                if (gi >= dim) break;
                out[gi] = d_lo * @as(f32, @floatFromInt(qs[ql_off + l] & 0x0F)) - dm_lo;
            }
            // Next 32 elements: high nibbles
            for (0..quant_block_elems) |l| {
                const gi = b * quant_super_block_elems + g * group_elems + quant_block_elems + l;
                if (gi >= dim) break;
                out[gi] = d_hi * @as(f32, @floatFromInt(qs[ql_off + l] >> 4)) - dm_hi;
            }
        }
    }
}

/// Dequantizes a Q5_K embedding row to f32.
/// 256 elements per super-block, 176 bytes: d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128].
/// Layout: 4 groups of 64 elements, each group in 32 bytes of qs.
///   Elements 0-31: LOW nibbles + high bit (scale index 2g)
///   Elements 32-63: HIGH nibbles + high bit (scale index 2g+1)
/// High bits: qh[l] bit (2g) for low half, bit (2g+1) for high half.
pub fn embQ5_K(data: [*]const u8, tok: u32, out: [*]f32, dim: usize) void {
    const nb = (dim + quant_super_block_elems - 1) / quant_super_block_elems;
    const rp = data + @as(usize, tok) * nb * q5_k_block_bytes;
    for (0..nb) |b| {
        const bp = rp + b * q5_k_block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[0..2], .little))));
        const dmin: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bp[2..4], .little))));
        const scales = bp[4..16];
        const qh = bp + 16;
        const qs = bp + 48;
        for (0..4) |g| {
            const ql_off = g * quant_block_elems;
            const umask1: u8 = @as(u8, 1) << @intCast(g * 2);
            const umask2: u8 = @as(u8, 2) << @intCast(g * 2);
            var sc_lo: u8 = undefined;
            var m_lo: u8 = undefined;
            var sc_hi: u8 = undefined;
            var m_hi: u8 = undefined;
            quant.getScaleMinK4(g * 2, scales, &sc_lo, &m_lo);
            quant.getScaleMinK4(g * 2 + 1, scales, &sc_hi, &m_hi);
            const d_lo = d * @as(f32, @floatFromInt(sc_lo));
            const dm_lo = dmin * @as(f32, @floatFromInt(m_lo));
            const d_hi = d * @as(f32, @floatFromInt(sc_hi));
            const dm_hi = dmin * @as(f32, @floatFromInt(m_hi));
            // First 32 elements: low nibbles + high bit
            for (0..quant_block_elems) |l| {
                const gi = b * quant_super_block_elems + g * group_elems + l;
                if (gi >= dim) break;
                const lo: f32 = @floatFromInt(qs[ql_off + l] & 0x0F);
                const hi: f32 = if ((qh[l] & umask1) != 0) q5_k_high_bit_value else 0.0;
                out[gi] = d_lo * (lo + hi) - dm_lo;
            }
            // Next 32 elements: high nibbles + high bit
            for (0..quant_block_elems) |l| {
                const gi = b * quant_super_block_elems + g * group_elems + quant_block_elems + l;
                if (gi >= dim) break;
                const lo: f32 = @floatFromInt(qs[ql_off + l] >> 4);
                const hi: f32 = if ((qh[l] & umask2) != 0) q5_k_high_bit_value else 0.0;
                out[gi] = d_hi * (lo + hi) - dm_hi;
            }
        }
    }
}

/// Dequantizes an MXFP4 embedding row to f32.
pub fn embMXFP4(data: [*]const u8, tok: u32, out: [*]f32, dim: usize) void {
    const nb = (dim + quant_block_elems - 1) / quant_block_elems;
    const rp = data + tok * nb * mxfp4_block_bytes;
    for (0..nb) |b| {
        const bp = rp + b * mxfp4_block_bytes;
        const d = quant.e8m0ToF32(bp[0]);
        for (0..quant_block_elems / 2) |j| {
            const byte = bp[1 + j];
            const v0 = quant.mxfp4Lookup(byte & 0x0F);
            const v1 = quant.mxfp4Lookup(byte >> 4);
            const gi0 = b * quant_block_elems + j;
            const gi1 = b * quant_block_elems + j + quant_block_elems / 2;
            if (gi0 < dim) out[gi0] = v0 * d;
            if (gi1 < dim) out[gi1] = v1 * d;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

test "embLookup f32 passthrough" {
    // f32 embedding: embLookup should memcpy the correct row.
    const dim = 4;
    const vocab = 3;
    var weights: [vocab * dim]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    var out: [dim]f32 = undefined;

    // Token 0: verify all elements, not just first/last
    embLookup(@ptrCast(&weights), .f32, 0, &out, dim);
    for (0..dim) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i + 1)), out[i], 1e-6);
    }

    // Token 2 (last row): verify all elements
    embLookup(@ptrCast(&weights), .f32, 2, &out, dim);
    for (0..dim) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i + 9)), out[i], 1e-6);
    }
}

test "embLookup bf16 converts correctly" {
    // bf16 is upper 16 bits of f32. Construct bf16 for known values.
    const dim = 4;
    // bf16 for 1.0 = 0x3F80, 2.0 = 0x4000, -1.0 = 0xBF80, 0.5 = 0x3F00
    var weights = [_]u16{ 0x3F80, 0x4000, 0xBF80, 0x3F00 };
    var out: [dim]f32 = undefined;

    embLookup(@ptrCast(&weights), .bf16, 0, &out, dim);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[3], 1e-6);
}

test "embQ8_0 dequantizes single block" {
    // Q8_0 block: 2-byte f16 scale + 32 signed bytes.
    // Build a block with scale=1.0 (f16 0x3C00) and bytes [0, 1, -1, 2, ...].
    const dim = 32;
    var block: [q8_0_block_bytes]u8 = undefined;
    // f16 1.0 = 0x3C00 little-endian
    block[0] = 0x00;
    block[1] = 0x3C;
    // Fill quants: element i gets value i (as signed byte)
    for (0..32) |i| block[2 + i] = @bitCast(@as(i8, @intCast(i)));

    var out: [dim]f32 = undefined;
    embQ8_0(&block, 0, &out, dim);

    // out[i] = i8(i) * scale(1.0) = i
    for (0..32) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i)), out[i], 1e-4);
    }
}

test "embQ4_0 dequantizes single block" {
    // Q4_0 block: 2-byte f16 scale + 16 bytes of packed nibbles.
    // Each byte holds two 4-bit values (low nibble = first half, high nibble = second half).
    // Dequant: (nibble - 8) * scale.
    const dim = 32;
    var block: [q4_0_block_bytes]u8 = undefined;
    // f16 2.0 = 0x4000 little-endian
    block[0] = 0x00;
    block[1] = 0x40;
    // Set all nibbles to 8 → dequant value = (8-8)*2.0 = 0.0
    for (0..16) |i| block[2 + i] = 0x88;

    var out: [dim]f32 = undefined;
    embQ4_0(&block, 0, &out, dim);

    // All elements should be 0.0 since nibble=8 → (8-8)*scale = 0
    for (0..dim) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[i], 1e-6);
    }

    // Now set first byte nibbles to different values: low=0, high=15
    block[2] = 0xF0; // low nibble=0, high nibble=15
    embQ4_0(&block, 0, &out, dim);
    // Element 0 (low nibble of byte 0): (0 - 8) * 2.0 = -16.0
    try std.testing.expectApproxEqAbs(@as(f32, -16.0), out[0], 1e-4);
    // Element 16 (high nibble of byte 0): (15 - 8) * 2.0 = 14.0
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), out[16], 1e-4);
}

test "embLookup f16 converts correctly" {
    // f16 embedding: verify conversion to f32.
    const dim = 4;
    var weights = [_]f16{ 1.0, -0.5, 2.0, 0.25 };
    var out: [dim]f32 = undefined;

    embLookup(@ptrCast(&weights), .f16, 0, &out, dim);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), out[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[2], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), out[3], 1e-3);
}

test "embQ5_0 dequantizes single block" {
    // Q5_0 block: 2-byte f16 scale + 4-byte qh bits + 16 nibble-packed bytes.
    // Build a block with scale=1.0, all nibbles=0, all high bits=0.
    // Dequant: (nibble | (hb<<4)) - 16, so value = (0|0) - 16 = -16 per element.
    const dim = 32;
    var block: [q5_0_block_bytes]u8 = undefined;
    // f16(1.0) = 0x3C00 LE
    block[0] = 0x00;
    block[1] = 0x3C;
    // qh = 0 (no high bits set)
    block[2] = 0;
    block[3] = 0;
    block[4] = 0;
    block[5] = 0;
    // All nibbles = 0
    for (6..22) |i| block[i] = 0x00;

    var out: [dim]f32 = undefined;
    embQ5_0(&block, 0, &out, dim);
    // All elements = (0 - 16) * 1.0 = -16.0
    for (0..dim) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, -16.0), out[i], 1e-4);
    }

    // Now set nibble 0 low = 15, high bit for element 0 = 1.
    // value = (15 | (1<<4)) - 16 = 31 - 16 = 15
    block[6] = 0x0F; // low nibble=15, high nibble=0
    block[2] = 0x01; // qh bit 0 = 1 (for element 0)
    embQ5_0(&block, 0, &out, dim);
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), out[0], 1e-4);
}

test "embQ6_K dequantizes zero block" {
    // Q6_K: 210-byte super-block, 256 elements. d at offset 208.
    // Set scale d=1.0, all sub-scales sc=1, all quant bits=0.
    // Dequant: d * sc * (q6_val - 32). With q6_val=0 → d * sc * (-32).
    const dim = quant_super_block_elems;
    var block: [q6_k_block_bytes]u8 = @splat(0);
    // d = f16(1.0) at offset 208
    block[q6_k_d_offset] = 0x00;
    block[q6_k_d_offset + 1] = 0x3C;
    // Sub-scales at offset 192..208: set all to i8(1)
    for (q6_k_sc_offset..q6_k_d_offset) |i| block[i] = 1;

    var out: [dim]f32 = undefined;
    embQ6_K(&block, 0, &out, dim);
    // All quant values are 0, so dequant = 1.0 * 1 * (0 - 32) = -32.0
    for (0..dim) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, -32.0), out[i], 1e-4);
    }
}

test "embQ4_K dequantizes zero nibbles" {
    // Q4_K: 144-byte super-block, 256 elements.
    // d=1.0, dmin=0.0, sc=1 for all groups, m=0, all nibbles=0.
    // Dequant: d * sc * nibble - dmin * m = 1.0 * 1 * 0 - 0 = 0.0
    const dim = quant_super_block_elems;
    var block: [q4_k_block_bytes]u8 = @splat(0);
    // d = f16(1.0)
    block[0] = 0x00;
    block[1] = 0x3C;
    // dmin = f16(0.0) (already zero)
    // scales[12]: set sc=1, m=0 for all 8 scale indices
    // For j<4: scales[j] = sc_lo | (m_lo << 4) = 1 | 0 = 1
    block[4] = 1;
    block[5] = 1;
    block[6] = 1;
    block[7] = 1;
    // For j>=4: high bits via scales[8..12]
    block[12] = 1;
    block[13] = 1;
    block[14] = 1;
    block[15] = 1;
    // qs[128] all zeros (already)

    var out: [dim]f32 = undefined;
    embQ4_K(&block, 0, &out, dim);
    for (0..dim) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[i], 1e-4);
    }
}

test "embLookup f32 smoke" {
    const dim = 4;
    var dummy_data = [_]u8{0} ** 256;
    var out = [_]f32{ 99, 99, 99, 99 };
    embLookup(&dummy_data, .f32, 0, &out, dim);
    // All-zero f32 bytes dequantize to 0.0.
    for (&out) |v| try std.testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-6);
}
