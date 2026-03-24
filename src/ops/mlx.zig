//! MLX quantization kernels.
//! Provides dequantization for MLX's affine quantized format (4-bit, 6-bit, or 8-bit, group_size=32/64).
//! Used by models loaded from safetensors with MLX quantization.

const std = @import("std");
const quant = @import("quant.zig");

/// Default MLX quantization parameters.
/// 64 elements per quantization group.
pub const mlx_group_size: usize = 64;
/// Words per group for 6-bit (legacy default). Use wordsPerGroup(bits) for variable bit width.
pub const mlx_words_per_group: usize = 12;

/// Compute words (u32) per group for a given bit width.
pub fn wordsPerGroup(bits: u32) usize {
    return mlx_group_size * bits / 32;
}

/// Unpack a single 4-bit value from a packed u32 array.
pub fn unpackU4(w: [*]const u32, idx: usize) u4 {
    const wi = idx / 8;
    const bo: u5 = @intCast((idx % 8) * 4);
    return @truncate(w[wi] >> bo);
}

/// Unpack a single 6-bit value from a packed u32 array.
pub fn unpackU6(w: [*]const u32, idx: usize) u6 {
    const bp = idx * 6;
    const wi = bp / 32;
    const bo: u5 = @intCast(bp % 32);
    if (bo <= 26) return @truncate(w[wi] >> bo);
    const lo = w[wi] >> bo;
    const hi = w[wi + 1] << @intCast(32 - @as(u6, bo));
    return @truncate(lo | hi);
}

/// Unpack a single 8-bit value from a packed u32 array.
pub fn unpackU8(w: [*]const u32, idx: usize) u8 {
    const wi = idx / 4;
    const bo: u5 = @intCast((idx % 4) * 8);
    return @truncate(w[wi] >> bo);
}

/// MLX affine GEMV: y[row] = sum_j(dequant(W[row,j]) * x[j])
/// Dequant: float_val = scale * int_val + bias, per group of 64 elements.
pub fn mlxGemvRaw(
    x: [*]const f32,
    pw: [*]const u32,
    sc: [*]const u16,
    bi: [*]const u16,
    y: [*]f32,
    n: usize,
    k: usize,
    bits: u32,
) void {
    mlxGemvRows(x, pw, sc, bi, y, 0, n, k, bits);
}

/// Compute a range of rows [start_row, start_row + n_rows) for MLX affine GEMV.
/// Used by both the single-threaded and parallel paths.
pub fn mlxGemvRows(
    x: [*]const f32,
    pw: [*]const u32,
    sc: [*]const u16,
    bi: [*]const u16,
    y: [*]f32,
    start_row: usize,
    n_rows: usize,
    k: usize,
    bits: u32,
) void {
    std.debug.assert(bits == 4 or bits == 6 or bits == 8);
    const gs = mlx_group_size;
    const gpr = (k + gs - 1) / gs;
    const wpg = wordsPerGroup(bits);
    const wpr = gpr * wpg;

    if (bits == 4) {
        mlxGemvQ4Rows(x, pw, sc, bi, y, start_row, n_rows, k, gpr, wpr);
    } else if (bits == 6) {
        mlxGemvQ6Rows(x, pw, sc, bi, y, start_row, n_rows, k, gpr, wpg, wpr);
    } else {
        mlxGemvQ8Rows(x, pw, sc, bi, y, start_row, n_rows, k, gpr, wpr);
    }
}

/// SIMD-optimized 4-bit MLX GEMV for a range of rows.
/// Uses factored scale/bias: sum(x*(scale*q+bias)) = scale*dot(x,q) + bias*sum(x).
/// Accumulates q_dot and x_sum per group, applies scale/bias once.
/// @mulAdd maps to NEON fmla (1 instruction vs fmul+fadd chain).
/// 2-row batching reuses x vector loads across rows.
fn mlxGemvQ4Rows(
    x: [*]const f32,
    pw: [*]const u32,
    sc: [*]const u16,
    bi: [*]const u16,
    y: [*]f32,
    start_row: usize,
    n_rows: usize,
    k: usize,
    gpr: usize,
    wpr: usize,
) void {
    const gs = mlx_group_size;
    const V = 8; // 8 nibbles per u32 word
    const VecF32 = @Vector(V, f32);
    const VecU32 = @Vector(V, u32);
    const nibble_shifts: VecU32 = .{ 0, 4, 8, 12, 16, 20, 24, 28 };
    const mask4: VecU32 = @splat(0xF);
    const vzero: VecF32 = @splat(0.0);

    // 2-row batching for x-vector cache reuse
    var row = start_row;
    while (row + 2 <= start_row + n_rows) : (row += 2) {
        var sum0: f32 = 0.0;
        var sum1: f32 = 0.0;
        const wr0 = pw + row * wpr;
        const wr1 = pw + (row + 1) * wpr;
        const sr0 = sc + row * gpr;
        const sr1 = sc + (row + 1) * gpr;
        const br0 = bi + row * gpr;
        const br1 = bi + (row + 1) * gpr;

        for (0..gpr) |g| {
            const scale0 = quant.bf16ToF32(sr0[g]);
            const scale1 = quant.bf16ToF32(sr1[g]);
            const bias0 = quant.bf16ToF32(br0[g]);
            const bias1 = quant.bf16ToF32(br1[g]);
            const xo = g * gs;
            const wo = g * wordsPerGroup(4);
            const elems = @min(gs, k - xo);
            const full_words = elems / V;

            var q_acc0: VecF32 = vzero;
            var q_acc1: VecF32 = vzero;
            var x_acc: VecF32 = vzero;

            for (0..full_words) |wi| {
                const xv: VecF32 = (x + xo + wi * V)[0..V].*;
                const w0: VecU32 = @splat(wr0[wo + wi]);
                const vals0: VecF32 = @floatFromInt((w0 >> nibble_shifts) & mask4);
                q_acc0 = @mulAdd(VecF32, xv, vals0, q_acc0);
                const w1: VecU32 = @splat(wr1[wo + wi]);
                const vals1: VecF32 = @floatFromInt((w1 >> nibble_shifts) & mask4);
                q_acc1 = @mulAdd(VecF32, xv, vals1, q_acc1);
                x_acc += xv;
            }
            const x_sum = @reduce(.Add, x_acc);
            sum0 += scale0 * @reduce(.Add, q_acc0) + bias0 * x_sum;
            sum1 += scale1 * @reduce(.Add, q_acc1) + bias1 * x_sum;

            // Scalar tail
            const done = full_words * V;
            for (done..elems) |i| {
                const xval = x[xo + i];
                const val0: u32 = unpackU4(wr0 + wo, i);
                const val1: u32 = unpackU4(wr1 + wo, i);
                sum0 += xval * (scale0 * @as(f32, @floatFromInt(val0)) + bias0);
                sum1 += xval * (scale1 * @as(f32, @floatFromInt(val1)) + bias1);
            }
        }
        y[row] = sum0;
        y[row + 1] = sum1;
    }

    // Remainder: single row
    while (row < start_row + n_rows) : (row += 1) {
        var sum: f32 = 0.0;
        const wr = pw + row * wpr;
        const sr = sc + row * gpr;
        const br = bi + row * gpr;

        for (0..gpr) |g| {
            const scale = quant.bf16ToF32(sr[g]);
            const bias = quant.bf16ToF32(br[g]);
            const xo = g * gs;
            const wo = g * wordsPerGroup(4);
            const elems = @min(gs, k - xo);
            const full_words = elems / V;

            var q_acc: VecF32 = vzero;
            var x_acc: VecF32 = vzero;

            for (0..full_words) |wi| {
                const xv: VecF32 = (x + xo + wi * V)[0..V].*;
                const word: VecU32 = @splat(wr[wo + wi]);
                const vals: VecF32 = @floatFromInt((word >> nibble_shifts) & mask4);
                q_acc = @mulAdd(VecF32, xv, vals, q_acc);
                x_acc += xv;
            }
            sum += scale * @reduce(.Add, q_acc) + bias * @reduce(.Add, x_acc);

            const done = full_words * V;
            for (done..elems) |i| {
                const val: u32 = unpackU4(wr + wo, i);
                sum += x[xo + i] * (scale * @as(f32, @floatFromInt(val)) + bias);
            }
        }
        y[row] = sum;
    }
}

/// 6-bit MLX GEMV for a range of rows (scalar, cross-word bit spans make SIMD impractical).
fn mlxGemvQ6Rows(
    x: [*]const f32,
    pw: [*]const u32,
    sc: [*]const u16,
    bi: [*]const u16,
    y: [*]f32,
    start_row: usize,
    n_rows: usize,
    k: usize,
    gpr: usize,
    wpg: usize,
    wpr: usize,
) void {
    const gs = mlx_group_size;
    for (start_row..start_row + n_rows) |row| {
        var sum: f32 = 0.0;
        const wr = pw + row * wpr;
        const sr = sc + row * gpr;
        const br = bi + row * gpr;
        for (0..gpr) |g| {
            const scale = quant.bf16ToF32(sr[g]);
            const bias = quant.bf16ToF32(br[g]);
            const xo = g * gs;
            const wo = g * wpg;
            const elems = @min(gs, k - xo);
            for (0..elems) |i| {
                const val: u32 = unpackU6(wr + wo, i);
                sum += x[xo + i] * (scale * @as(f32, @floatFromInt(val)) + bias);
            }
        }
        y[row] = sum;
    }
}

/// SIMD-optimized 8-bit MLX GEMV for a range of rows.
/// 4 values per u32 word, same factored scale/bias pattern as Q4.
fn mlxGemvQ8Rows(
    x: [*]const f32,
    pw: [*]const u32,
    sc: [*]const u16,
    bi: [*]const u16,
    y: [*]f32,
    start_row: usize,
    n_rows: usize,
    k: usize,
    gpr: usize,
    wpr: usize,
) void {
    const gs = mlx_group_size;
    const V = 4; // 4 bytes per u32 word
    const VecF32 = @Vector(V, f32);
    const VecU32 = @Vector(V, u32);
    const byte_shifts: VecU32 = .{ 0, 8, 16, 24 };
    const mask8: VecU32 = @splat(0xFF);
    const vzero: VecF32 = @splat(0.0);

    for (start_row..start_row + n_rows) |row| {
        var sum: f32 = 0.0;
        const wr = pw + row * wpr;
        const sr = sc + row * gpr;
        const br = bi + row * gpr;
        for (0..gpr) |g| {
            const scale = quant.bf16ToF32(sr[g]);
            const bias = quant.bf16ToF32(br[g]);
            const xo = g * gs;
            const wo = g * wordsPerGroup(8);
            const elems = @min(gs, k - xo);
            const full_words = elems / V;

            var q_acc: VecF32 = vzero;
            var x_acc: VecF32 = vzero;

            for (0..full_words) |wi| {
                const xv: VecF32 = (x + xo + wi * V)[0..V].*;
                const word: VecU32 = @splat(wr[wo + wi]);
                const vals: VecF32 = @floatFromInt((word >> byte_shifts) & mask8);
                q_acc = @mulAdd(VecF32, xv, vals, q_acc);
                x_acc += xv;
            }
            sum += scale * @reduce(.Add, q_acc) + bias * @reduce(.Add, x_acc);

            // Scalar tail
            const done = full_words * V;
            for (done..elems) |i| {
                const val: u32 = unpackU8(wr + wo, i);
                sum += x[xo + i] * (scale * @as(f32, @floatFromInt(val)) + bias);
            }
        }
        y[row] = sum;
    }
}

/// GEMV for MLX MXFP4 SafeTensors layout.
/// Weight: U32-packed 4-bit nibbles (8 nibbles per word), group_size=32.
/// Scales: FP8 E4M3 per group (U8 array). No quantization bias.
/// Dequant: float_val = mxfp4_lookup(nibble) * fp8_scale.
pub fn mlxMxfp4Gemv(
    x: [*]const f32,
    pw: [*]const u32,
    scales_u8: [*]const u8,
    y: [*]f32,
    n: usize,
    k: usize,
) void {
    mlxMxfp4GemvRows(x, pw, scales_u8, y, 0, n, k);
}

/// Convert an E8M0 scale byte to f32: val = 2^(byte - 127).
/// E8M0 is a pure power-of-2 format used by OCP Microscaling (MX) spec.
inline fn e8m0ToF32(byte: u8) f32 {
    if (byte == 0) return 0.0; // Zero exponent → zero scale
    // Construct IEEE 754 float: exponent = byte - 127 + 127 = byte, mantissa = 0
    return @bitCast(@as(u32, byte) << 23);
}

/// Compute a range of rows for MLX MXFP4 GEMV.
/// Scales are E8M0 (pure power-of-2), NOT FP8 E4M3.
pub fn mlxMxfp4GemvRows(
    x: [*]const f32,
    pw: [*]const u32,
    scales_u8: [*]const u8,
    y: [*]f32,
    start_row: usize,
    n_rows: usize,
    k: usize,
) void {
    const mxfp4_gs: usize = 32;
    const gpr = (k + mxfp4_gs - 1) / mxfp4_gs;
    const wpg: usize = mxfp4_gs * 4 / 32; // 4 u32 words per group
    const wpr = gpr * wpg;

    const V8 = @Vector(8, f32);

    for (start_row..start_row + n_rows) |row| {
        var acc: V8 = @splat(0.0);
        const wr = pw + row * wpr;
        const sr = scales_u8 + row * gpr;

        for (0..gpr) |g| {
            const sv: V8 = @splat(e8m0ToF32(sr[g]));
            const xo = g * mxfp4_gs;
            const wo = g * wpg;
            const elems = @min(mxfp4_gs, k - xo);

            const full_words = elems / 8;
            for (0..full_words) |wi| {
                const word = wr[wo + wi];
                var vals: [8]f32 = undefined;
                inline for (0..8) |ni| {
                    const nibble: u8 = @truncate((word >> @as(u5, @intCast(ni * 4))) & 0xF);
                    vals[ni] = quant.mxfp4Lookup(nibble);
                }
                const v: V8 = vals;
                const xv: V8 = (x + xo + wi * 8)[0..8].*;
                acc += sv * v * xv;
            }

            // Scalar tail
            const done = full_words * 8;
            for (done..elems) |i| {
                const nibble = unpackU4(wr + wo, i);
                const val = quant.mxfp4Lookup(nibble);
                acc[0] += e8m0ToF32(sr[g]) * val * x[xo + i];
            }
        }
        y[row] = @reduce(.Add, acc);
    }
}

/// Dequantize a single row from an MLX-quantized embedding table into f32.
///
/// Parameters:
///   - out:   Output buffer [k] for the dequantized row.
///   - pw:    Packed u32 weight data for the full embedding table.
///   - sc:    BF16 scales for the full table.
///   - bi:    BF16 biases for the full table.
///   - row:   Row index (token ID).
///   - k:     Embedding dimension.
///   - bits:  Quantization bit width (4, 6, or 8).
pub fn mlxEmbLookup(
    out: [*]f32,
    pw: [*]const u32,
    sc: [*]const u16,
    bi: [*]const u16,
    row: usize,
    k: usize,
    bits: u32,
) void {
    const gs = mlx_group_size;
    const gpr = (k + gs - 1) / gs;
    const wpg = wordsPerGroup(bits);
    const wpr = gpr * wpg;
    const wr = pw + row * wpr;
    const sr = sc + row * gpr;
    const br = bi + row * gpr;
    for (0..gpr) |g| {
        const scale = quant.bf16ToF32(sr[g]);
        const bias = quant.bf16ToF32(br[g]);
        const xo = g * gs;
        const wo = g * wpg;
        const elems = @min(gs, k - xo);
        for (0..elems) |i| {
            const val: u32 = if (bits == 4) unpackU4(wr + wo, i) else if (bits == 8) unpackU8(wr + wo, i) else unpackU6(wr + wo, i);
            out[xo + i] = scale * @as(f32, @floatFromInt(val)) + bias;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

test "wordsPerGroup" {
    try std.testing.expectEqual(@as(usize, 8), wordsPerGroup(4)); // 64*4/32 = 8
    try std.testing.expectEqual(@as(usize, 12), wordsPerGroup(6)); // 64*6/32 = 12
}

test "unpackU4" {
    // Pack 0x76543210 = values 0,1,2,3,4,5,6,7 in 4-bit nibbles
    const data = [_]u32{0x76543210};
    try std.testing.expectEqual(@as(u4, 0), unpackU4(&data, 0));
    try std.testing.expectEqual(@as(u4, 1), unpackU4(&data, 1));
    try std.testing.expectEqual(@as(u4, 2), unpackU4(&data, 2));
    try std.testing.expectEqual(@as(u4, 7), unpackU4(&data, 7));
}

test "unpackU6" {
    // Test first element: bottom 6 bits of first word
    const data = [_]u32{ 0b00_111111, 0 };
    try std.testing.expectEqual(@as(u6, 63), unpackU6(&data, 0));

    // Test cross-word boundary (idx=5: bit position 30, spans bits 30-35 across two u32 words)
    // Word 0 bits [31:30] = low 2 bits, Word 1 bits [3:0] = high 4 bits
    const cross = [_]u32{ 0b11_000000_000000_000000_000000_000000, 0b0000_0000_0000_0000_0000_0000_0000_1010 };
    // 6-bit value = (word1[3:0] << 2) | (word0[31:30]) = (0b1010 << 2) | 0b11 = 0b101011 = 43
    try std.testing.expectEqual(@as(u6, 43), unpackU6(&cross, 5));
}
