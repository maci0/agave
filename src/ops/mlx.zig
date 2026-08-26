//! MLX quantization kernels.
//! Provides dequantization for MLX's affine quantized format (2-bit, 4-bit, 6-bit, or 8-bit, group_size=32/64).
//! Used by models loaded from safetensors with MLX quantization.

const std = @import("std");
const quant = @import("quant.zig");

/// Default MLX quantization parameters.
/// 64 elements per quantization group.
pub const mlx_group_size: usize = 64;
/// MXFP4 group size: NVIDIA MXFP4 spec uses 16-element blocks (not 32).
/// Each scale covers 16 weight elements. Group size 32 was wrong and caused
/// row-stride corruption: every row after row 0 read the wrong scale bytes.
pub const mxfp4_group_size: usize = 16;
/// MLX community MoE experts pack MXFP4 with 32-element groups and E8M0 scales.
pub const mxfp4_mlx_expert_group_size: usize = 32;

/// Bit-packing constants for u32-packed quantized weights.
const bits_per_u32: usize = 32;
const crumbs_per_u32: usize = 16; // 32 / 2
const bits_per_crumb: u32 = 2;
const nibbles_per_u32: usize = 8; // 32 / 4
const bits_per_nibble: u32 = 4;
const bytes_per_u32: usize = 4; // 32 / 8
const bits_per_byte: u32 = 8;
/// Maximum bit offset in a u32 where a 6-bit value fits without spanning two words.
const u6_max_single_word_offset: u5 = 26; // 32 - 6

/// Compute words (u32) per group for a given bit width and group size.
/// Do not use for MXFP4, which uses `mxfp4_group_size` (16) and computes words-per-group inline.
pub fn wordsPerGroup(bits: u32, gs: usize) usize {
    return gs * bits / bits_per_u32;
}

/// Unpack a single 2-bit value from a packed u32 array. 16 crumbs per word, LSB-first.
fn unpackU2(w: [*]const u32, idx: usize) u2 {
    const wi = idx / crumbs_per_u32;
    const bo: u5 = @intCast((idx % crumbs_per_u32) * bits_per_crumb);
    return @truncate(w[wi] >> bo);
}

/// Unpack a single 4-bit value from a packed u32 array.
fn unpackU4(w: [*]const u32, idx: usize) u4 {
    const wi = idx / nibbles_per_u32;
    const bo: u5 = @intCast((idx % nibbles_per_u32) * bits_per_nibble);
    return @truncate(w[wi] >> bo);
}

/// Unpack a single 6-bit value from a packed u32 array.
fn unpackU6(w: [*]const u32, idx: usize) u6 {
    const bp = idx * 6;
    const wi = bp / bits_per_u32;
    const bo: u5 = @intCast(bp % bits_per_u32);
    if (bo <= u6_max_single_word_offset) return @truncate(w[wi] >> bo);
    const lo = w[wi] >> bo;
    const hi = w[wi + 1] << @intCast(bits_per_u32 - @as(u6, bo));
    return @truncate(lo | hi);
}

/// Unpack a single 8-bit value from a packed u32 array.
fn unpackU8(w: [*]const u32, idx: usize) u8 {
    const wi = idx / bytes_per_u32;
    const bo: u5 = @intCast((idx % bytes_per_u32) * bits_per_byte);
    return @truncate(w[wi] >> bo);
}

/// MLX affine GEMV: y[row] = sum_j(dequant(W[row,j]) * x[j])
/// Dequant: float_val = scale * int_val + bias, per group of `gs` elements.
///
/// Parameters:
///   x   , input vector [k]
///   pw  , packed weight matrix (uint2/4/6/8 values stored in u32 words)
///   sc  , per-group scales (bf16, one per gs-element group)
///   bi  , per-group biases (bf16, one per gs-element group)
///   y   , output vector [n]
///   n   , number of output rows
///   k   , input dimension (columns per row)
///   bits, quantization width (2, 4, 6, or 8)
///   gs  , quantization group size (elements per scale/bias pair, e.g. 32 or 64)
pub fn mlxGemvRaw(
    x: [*]const f32,
    pw: [*]const u32,
    sc: [*]const u16,
    bi: [*]const u16,
    y: [*]f32,
    n: usize,
    k: usize,
    bits: u32,
    gs: usize,
) void {
    mlxGemvRows(x, pw, sc, bi, y, 0, n, k, bits, gs);
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
    gs: usize,
) void {
    if (bits != 2 and bits != 4 and bits != 6 and bits != 8) @panic("MLX GEMV: unsupported bits per weight (expected 2, 4, 6, or 8)");
    const gpr = (k + gs - 1) / gs;
    const wpg = wordsPerGroup(bits, gs);
    const wpr = gpr * wpg;

    if (bits == 2) {
        mlxGemvQ2Rows(x, pw, sc, bi, y, start_row, n_rows, k, gpr, wpg, wpr, gs);
    } else if (bits == 4) {
        mlxGemvQ4Rows(x, pw, sc, bi, y, start_row, n_rows, k, gpr, wpg, wpr, gs);
    } else if (bits == 6) {
        mlxGemvQ6Rows(x, pw, sc, bi, y, start_row, n_rows, k, gpr, wpg, wpr, gs);
    } else {
        mlxGemvQ8Rows(x, pw, sc, bi, y, start_row, n_rows, k, gpr, wpg, wpr, gs);
    }
}

/// SIMD-optimized 2-bit MLX GEMV for a range of rows.
/// 16 crumbs per u32 word, same factored scale/bias pattern as Q4.
/// @mulAdd maps to NEON fmla (1 instruction vs fmul+fadd chain).
/// 2-row batching reuses x vector loads across rows.
fn mlxGemvQ2Rows(
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
    gs: usize,
) void {
    const V = crumbs_per_u32;
    const VecF32 = @Vector(V, f32);
    const vzero: VecF32 = @splat(0.0);
    const VecU32 = @Vector(V, u32);
    const crumb_shifts: VecU32 = .{ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 };
    const mask2: VecU32 = @splat(0x3);

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
            const wo = g * wpg;
            const elems = @min(gs, k - xo);
            const full_words = elems / V;

            var q_acc0: VecF32 = vzero;
            var q_acc1: VecF32 = vzero;
            var x_acc: VecF32 = vzero;

            for (0..full_words) |wi| {
                const xv: VecF32 = (x + xo + wi * V)[0..V].*;
                const w0: VecU32 = @splat(wr0[wo + wi]);
                const vals0: VecF32 = @floatFromInt((w0 >> crumb_shifts) & mask2);
                q_acc0 = @mulAdd(VecF32, xv, vals0, q_acc0);
                const w1: VecU32 = @splat(wr1[wo + wi]);
                const vals1: VecF32 = @floatFromInt((w1 >> crumb_shifts) & mask2);
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
                const val0: u32 = unpackU2(wr0 + wo, i);
                const val1: u32 = unpackU2(wr1 + wo, i);
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
            const wo = g * wpg;
            const elems = @min(gs, k - xo);
            const full_words = elems / V;

            var q_acc: VecF32 = vzero;
            var x_acc: VecF32 = vzero;

            for (0..full_words) |wi| {
                const xv: VecF32 = (x + xo + wi * V)[0..V].*;
                const word: VecU32 = @splat(wr[wo + wi]);
                const vals: VecF32 = @floatFromInt((word >> crumb_shifts) & mask2);
                q_acc = @mulAdd(VecF32, xv, vals, q_acc);
                x_acc += xv;
            }
            sum += scale * @reduce(.Add, q_acc) + bias * @reduce(.Add, x_acc);

            const done = full_words * V;
            for (done..elems) |i| {
                const val: u32 = unpackU2(wr + wo, i);
                sum += x[xo + i] * (scale * @as(f32, @floatFromInt(val)) + bias);
            }
        }
        y[row] = sum;
    }
}

/// SIMD-optimized 4-bit MLX GEMV for a range of rows.
/// Uses factored scale/bias: sum(x*(scale*q+bias)) = scale*dot(x,q) + bias*sum(x).
/// Accumulates q_dot and x_sum per group, applies scale/bias once per group.
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
    wpg: usize,
    wpr: usize,
    gs: usize,
) void {
    const V = nibbles_per_u32;
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
            const wo = g * wpg;
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
            const wo = g * wpg;
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
    gs: usize,
) void {
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
    wpg: usize,
    wpr: usize,
    gs: usize,
) void {
    const V = bytes_per_u32;
    const VecF32 = @Vector(V, f32);
    const VecU32 = @Vector(V, u32);
    const byte_shifts: VecU32 = .{ 0, 8, 16, 24 };
    const vzero: VecF32 = @splat(0.0);
    const mask8: VecU32 = @splat(0xFF);

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

/// Scale format for MXFP4 GEMV.
pub const Mxfp4ScaleFormat = enum {
    /// FP8 E4M3: standard NVIDIA MXFP4 / GGUF format.
    fp8_e4m3,
    /// E8M0: pure power-of-2 exponent (OCP Microscaling spec, MLX community experts).
    e8m0,
};

/// MLX SafeTensors experts with group_size 32 store E8M0 scales, not FP8 E4M3.
pub fn mxfp4ScaleFormat(is_safetensors: bool, gs: usize) Mxfp4ScaleFormat {
    return if (is_safetensors and gs >= mxfp4_mlx_expert_group_size) .e8m0 else .fp8_e4m3;
}

/// Compute a range of rows for MLX MXFP4 GEMV.
/// Weights are E2M1 (4-bit) looked up via `mxfp4Lookup`, scaled by per-group U8 scales.
/// `gs` is the quantization group size (16 for standard MXFP4, 32 for MLX experts).
/// `scale_fmt` selects how U8 scale bytes are decoded:
///   - `.fp8_e4m3`: NVIDIA/GGUF MXFP4 format
///   - `.e8m0`: OCP Microscaling / MLX community MoE experts
pub fn mlxMxfp4GemvRows(
    x: [*]const f32,
    pw: [*]const u32,
    scales_u8: [*]const u8,
    y: [*]f32,
    start_row: usize,
    n_rows: usize,
    k: usize,
    gs: usize,
    scale_fmt: Mxfp4ScaleFormat,
) void {
    const gpr = (k + gs - 1) / gs;
    const wpg: usize = gs * bits_per_nibble / bits_per_u32;
    const wpr = gpr * wpg;

    const V8 = @Vector(nibbles_per_u32, f32);
    const vzero: V8 = @splat(0.0);

    // 2-row batching: reuse x vector loads across two output rows.
    var row = start_row;
    while (row + 2 <= start_row + n_rows) : (row += 2) {
        var acc0: V8 = vzero;
        var acc1: V8 = vzero;
        const wr0 = pw + row * wpr;
        const wr1 = pw + (row + 1) * wpr;
        const sr0 = scales_u8 + row * gpr;
        const sr1 = scales_u8 + (row + 1) * gpr;

        for (0..gpr) |g| {
            const scale0 = switch (scale_fmt) {
                .fp8_e4m3 => quant.fp8e4m3ToF32(sr0[g]),
                .e8m0 => quant.e8m0ToF32(sr0[g]),
            };
            const scale1 = switch (scale_fmt) {
                .fp8_e4m3 => quant.fp8e4m3ToF32(sr1[g]),
                .e8m0 => quant.e8m0ToF32(sr1[g]),
            };
            const sv0: V8 = @splat(scale0);
            const sv1: V8 = @splat(scale1);
            const xo = g * gs;
            const wo = g * wpg;
            const elems = @min(gs, k - xo);

            const full_words = elems / nibbles_per_u32;
            for (0..full_words) |wi| {
                const xv: V8 = (x + xo + wi * nibbles_per_u32)[0..nibbles_per_u32].*;
                // Row 0
                const w0 = wr0[wo + wi];
                var v0: [nibbles_per_u32]f32 = undefined;
                inline for (0..nibbles_per_u32) |ni| {
                    v0[ni] = quant.mxfp4Lookup(@truncate((w0 >> @as(u5, @intCast(ni * bits_per_nibble))) & 0xF));
                }
                acc0 = @mulAdd(V8, sv0 * @as(V8, v0), xv, acc0);
                // Row 1
                const w1 = wr1[wo + wi];
                var v1: [nibbles_per_u32]f32 = undefined;
                inline for (0..nibbles_per_u32) |ni| {
                    v1[ni] = quant.mxfp4Lookup(@truncate((w1 >> @as(u5, @intCast(ni * bits_per_nibble))) & 0xF));
                }
                acc1 = @mulAdd(V8, sv1 * @as(V8, v1), xv, acc1);
            }

            // Scalar tail
            const done = full_words * nibbles_per_u32;
            for (done..elems) |i| {
                const xval = x[xo + i];
                acc0[0] += scale0 * quant.mxfp4Lookup(unpackU4(wr0 + wo, i)) * xval;
                acc1[0] += scale1 * quant.mxfp4Lookup(unpackU4(wr1 + wo, i)) * xval;
            }
        }
        y[row] = @reduce(.Add, acc0);
        y[row + 1] = @reduce(.Add, acc1);
    }

    // Remainder: single row
    while (row < start_row + n_rows) : (row += 1) {
        var acc: V8 = vzero;
        const wr = pw + row * wpr;
        const sr = scales_u8 + row * gpr;

        for (0..gpr) |g| {
            const scale = switch (scale_fmt) {
                .fp8_e4m3 => quant.fp8e4m3ToF32(sr[g]),
                .e8m0 => quant.e8m0ToF32(sr[g]),
            };
            const sv: V8 = @splat(scale);
            const xo = g * gs;
            const wo = g * wpg;
            const elems = @min(gs, k - xo);

            const full_words = elems / nibbles_per_u32;
            for (0..full_words) |wi| {
                const word = wr[wo + wi];
                var vals: [nibbles_per_u32]f32 = undefined;
                inline for (0..nibbles_per_u32) |ni| {
                    vals[ni] = quant.mxfp4Lookup(@truncate((word >> @as(u5, @intCast(ni * bits_per_nibble))) & 0xF));
                }
                const v: V8 = vals;
                const xv: V8 = (x + xo + wi * nibbles_per_u32)[0..nibbles_per_u32].*;
                acc = @mulAdd(V8, sv * v, xv, acc);
            }

            const done = full_words * nibbles_per_u32;
            for (done..elems) |i| {
                acc[0] += scale * quant.mxfp4Lookup(unpackU4(wr + wo, i)) * x[xo + i];
            }
        }
        y[row] = @reduce(.Add, acc);
    }
}

/// Weight-stationary batched MLX-Q4 GEMM: y[n_tok, n_out] = x[n_tok, k] @ W[n_out, k]^T.
/// Reads each weight row ONCE and accumulates dot products for all n_tok input vectors.
/// This is N× less memory bandwidth than N sequential GEMVs.
/// Uses the same factored scale/bias as mlxGemvQ4Rows.
pub fn mlxGemmQ4(
    x: [*]const f32,
    pw: [*]const u32,
    sc: [*]const u16,
    bi: [*]const u16,
    y: [*]f32,
    n_tok: usize,
    n_out: usize,
    k: usize,
    gs: usize,
) void {
    const V = nibbles_per_u32;
    const VecF32 = @Vector(V, f32);
    const VecU32 = @Vector(V, u32);
    const nibble_shifts: VecU32 = .{ 0, 4, 8, 12, 16, 20, 24, 28 };
    const mask4: VecU32 = @splat(0xF);

    const gpr = (k + gs - 1) / gs;
    const wpg = wordsPerGroup(4, gs);
    const wpr = gpr * wpg;

    for (0..n_out) |row| {
        const wr = pw + row * wpr;
        const sr = sc + row * gpr;
        const br = bi + row * gpr;

        // Per-token accumulators (stack-allocated, max 128 tokens)
        var q_dots: [128]f32 = undefined;
        var x_sums: [128]f32 = undefined;
        for (0..n_tok) |t| {
            q_dots[t] = 0;
            x_sums[t] = 0;
        }

        for (0..gpr) |g| {
            const scale = quant.bf16ToF32(sr[g]);
            const bias = quant.bf16ToF32(br[g]);
            const xo = g * gs;
            const wo = g * wpg;
            const elems = @min(gs, k - xo);
            const full_words = elems / V;

            // For each word in this group, dequant weight ONCE, dot with all tokens
            for (0..full_words) |wi| {
                const w: VecU32 = @splat(wr[wo + wi]);
                const vals: VecF32 = @floatFromInt((w >> nibble_shifts) & mask4);
                for (0..n_tok) |t| {
                    const xv: VecF32 = (x + t * k + xo + wi * V)[0..V].*;
                    q_dots[t] += @reduce(.Add, xv * vals);
                    x_sums[t] += @reduce(.Add, xv);
                }
            }

            // Scalar tail
            const done = full_words * V;
            for (done..elems) |i| {
                const val: u32 = unpackU4(wr + wo, i);
                const fval = @as(f32, @floatFromInt(val));
                for (0..n_tok) |t| {
                    const xval = x[t * k + xo + i];
                    q_dots[t] += xval * fval;
                    x_sums[t] += xval;
                }
            }

            // Apply scale+bias per group
            for (0..n_tok) |t| {
                y[t * n_out + row] += scale * q_dots[t] + bias * x_sums[t];
                q_dots[t] = 0;
                x_sums[t] = 0;
            }
        }
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
///   - bits:  Quantization bit width (2, 4, 6, or 8).
pub fn mlxEmbLookup(
    out: [*]f32,
    pw: [*]const u32,
    sc: [*]const u16,
    bi: [*]const u16,
    row: usize,
    k: usize,
    bits: u32,
) void {
    if (bits != 2 and bits != 4 and bits != 6 and bits != 8) @panic("MLX GEMV: unsupported bits per weight (expected 2, 4, 6, or 8)");
    const gs = mlx_group_size;
    const gpr = (k + gs - 1) / gs;
    const wpg = wordsPerGroup(bits, gs);
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
            const val: u32 = if (bits == 2) unpackU2(wr + wo, i) else if (bits == 4) unpackU4(wr + wo, i) else if (bits == 8) unpackU8(wr + wo, i) else unpackU6(wr + wo, i);
            out[xo + i] = scale * @as(f32, @floatFromInt(val)) + bias;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

test "wordsPerGroup" {
    try std.testing.expectEqual(@as(usize, 4), wordsPerGroup(2, 64)); // 64*2/32 = 4
    try std.testing.expectEqual(@as(usize, 8), wordsPerGroup(4, 64)); // 64*4/32 = 8
    try std.testing.expectEqual(@as(usize, 12), wordsPerGroup(6, 64)); // 64*6/32 = 12
    try std.testing.expectEqual(@as(usize, 16), wordsPerGroup(8, 64)); // 64*8/32 = 16
    // Variable group sizes
    try std.testing.expectEqual(@as(usize, 4), wordsPerGroup(4, 32)); // 32*4/32 = 4
    try std.testing.expectEqual(@as(usize, 8), wordsPerGroup(8, 32)); // 32*8/32 = 8
}

test "unpackU2" {
    // Pack 16 crumbs into one u32: values 0,1,2,3,0,1,2,3,... (repeating)
    // Bit layout: 00_01_10_11_00_01_10_11_00_01_10_11_00_01_10_11
    // = 0b11_10_01_00_11_10_01_00_11_10_01_00_11_10_01_00 = 0xE4E4E4E4
    const data = [_]u32{0xE4E4E4E4};
    try std.testing.expectEqual(@as(u2, 0), unpackU2(&data, 0));
    try std.testing.expectEqual(@as(u2, 1), unpackU2(&data, 1));
    try std.testing.expectEqual(@as(u2, 2), unpackU2(&data, 2));
    try std.testing.expectEqual(@as(u2, 3), unpackU2(&data, 3));
    try std.testing.expectEqual(@as(u2, 0), unpackU2(&data, 4));
    try std.testing.expectEqual(@as(u2, 3), unpackU2(&data, 15));
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

test "mlxGemvRaw 4-bit basic" {
    // 1 output row, k=8, 4-bit quantization
    // Weights: nibbles 0..7 packed into first u32 word
    // scale=1.0 (bf16), bias=0.0 → dequant(val) = val
    // x = all ones → y[0] = sum(0..7) = 28
    var pw = [_]u32{ 0x76543210, 0, 0, 0, 0, 0, 0, 0 };
    const sc = [_]u16{0x3F80}; // bf16(1.0)
    const bi = [_]u16{0x0000}; // bf16(0.0)
    const x = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };
    var y = [_]f32{0};

    mlxGemvRaw(&x, &pw, &sc, &bi, &y, 1, 8, 4, mlx_group_size);

    try std.testing.expectApproxEqAbs(@as(f32, 28.0), y[0], 1e-3);
}

test "mlxGemvRaw 2-bit basic" {
    // 1 output row, k=16, 2-bit quantization (16 crumbs fit in 1 u32 word)
    // Weights: crumbs 0,1,2,3,0,1,2,3,... packed into first u32 word
    // 0xE4E4E4E4 = repeating pattern 0,1,2,3
    // scale=1.0 (bf16), bias=0.0 → dequant(val) = val
    // x = all ones → y[0] = 4*(0+1+2+3) = 24
    var pw: [4]u32 = .{0} ** 4; // 1 group × 4 words (only 1 word used for k=16)
    pw[0] = 0xE4E4E4E4;
    const sc = [_]u16{0x3F80}; // bf16(1.0)
    const bi = [_]u16{0x0000}; // bf16(0.0)
    const x = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    var y = [_]f32{0};

    mlxGemvRaw(&x, &pw, &sc, &bi, &y, 1, 16, 2, mlx_group_size);

    // sum = 4*(0+1+2+3) = 24.0
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), y[0], 1e-3);
}

test "mlxGemvRaw 2-bit with bias" {
    // Verify bias: all-zero weights + bias=1.0 → y = sum(x) * bias
    var pw: [4]u32 = .{0} ** 4;
    const sc = [_]u16{0x4000}; // bf16(2.0), scale doesn't matter, weights are 0
    const bi = [_]u16{0x3F80}; // bf16(1.0)
    const x = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    var y = [_]f32{0};

    mlxGemvRaw(&x, &pw, &sc, &bi, &y, 1, 16, 2, mlx_group_size);

    // Each element: x[j] * (scale*0 + bias) = 1.0 * 1.0 = 1.0, sum = 16.0
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), y[0], 1e-3);
}

test "mlxGemvRaw 2-bit two rows" {
    // 2 output rows to exercise the 2-row batching path
    // Row 0: all 3s (0xFF...), Row 1: all 1s (0x55...)
    // scale=2.0, bias=0.5, x = all ones, k=16
    // Row 0: sum(2.0*3 + 0.5) = 16 * 6.5 = 104.0
    // Row 1: sum(2.0*1 + 0.5) = 16 * 2.5 = 40.0
    var pw: [8]u32 = .{0} ** 8; // 2 rows × 4 words
    pw[0] = 0xFFFFFFFF; // row 0: all crumbs = 3
    pw[4] = 0x55555555; // row 1: all crumbs = 1
    const sc = [_]u16{ 0x4000, 0x4000 }; // bf16(2.0)
    const bi = [_]u16{ 0x3F00, 0x3F00 }; // bf16(0.5)
    const x = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    var y = [_]f32{ 0, 0 };

    mlxGemvRaw(&x, &pw, &sc, &bi, &y, 2, 16, 2, mlx_group_size);

    try std.testing.expectApproxEqAbs(@as(f32, 104.0), y[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 40.0), y[1], 1e-3);
}

test "mlxGemvRaw 4-bit with bias" {
    // Verify bias is applied: all-zero weights + bias=1.0 → y = sum(x) * bias
    var pw = [_]u32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const sc = [_]u16{0x4000}; // bf16(2.0), scale doesn't matter, weights are 0
    const bi = [_]u16{0x3F80}; // bf16(1.0)
    const x = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };
    var y = [_]f32{0};

    mlxGemvRaw(&x, &pw, &sc, &bi, &y, 1, 8, 4, mlx_group_size);

    // Each element: x[j] * (scale*0 + bias) = 1.0 * 1.0 = 1.0, sum = 8.0
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), y[0], 1e-3);
}

test "mlxEmbLookup 2-bit basic" {
    // 2 rows of k=16, look up row 1 (all 2s)
    // Row 0: crumbs 0,1,2,3 repeating, Row 1: all 2s (0xAA...)
    var pw: [8]u32 = .{0} ** 8; // 2 rows × 4 words per row
    pw[0] = 0xE4E4E4E4; // row 0: 0,1,2,3 repeating
    pw[4] = 0xAAAAAAAA; // row 1: all crumbs = 2 (0b10 = 2, 0xAA = 10_10_10_10)
    const sc = [_]u16{ 0x3F80, 0x3F80 }; // bf16(1.0) per row
    const bi = [_]u16{ 0x0000, 0x0000 }; // bf16(0.0)
    var out: [16]f32 = undefined;

    mlxEmbLookup(&out, &pw, &sc, &bi, 1, 16, 2);

    // Row 1: dequant = 1.0 * 2 + 0.0 = 2.0 for all elements
    for (0..16) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[i], 1e-6);
    }
}

test "mlxEmbLookup 4-bit basic" {
    // 2 rows of k=8, look up row 1 (all 5s)
    // Row 0: nibbles 0..7, Row 1: all 5s
    var pw: [16]u32 = .{0} ** 16; // 2 rows × 8 words per row
    pw[0] = 0x76543210; // row 0
    pw[8] = 0x55555555; // row 1: all nibbles = 5
    const sc = [_]u16{ 0x3F80, 0x3F80 }; // bf16(1.0) per row
    const bi = [_]u16{ 0x0000, 0x0000 }; // bf16(0.0)
    var out: [8]f32 = undefined;

    mlxEmbLookup(&out, &pw, &sc, &bi, 1, 8, 4);

    // Row 1: dequant = 1.0 * 5 + 0.0 = 5.0 for all elements
    for (0..8) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 5.0), out[i], 1e-6);
    }
}

test "fuzz: wordsPerGroup and mlxEmbLookup" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            var raw: [4]u8 = undefined;
            smith.bytesWithHash(&raw, 0);

            // wordsPerGroup: test with valid bit widths.
            const bits_choices = [_]u32{ 4, 6, 8 };
            const bits = bits_choices[raw[0] % 3];
            const wpg = wordsPerGroup(bits, mlx_group_size);
            // Invariant: words per group must be positive.
            try std.testing.expect(wpg > 0);
            // Invariant: wpg * 32 / bits == group_size (64).
            try std.testing.expectEqual(mlx_group_size, wpg * bits_per_u32 / bits);

            // mlxEmbLookup: small 4-bit test with random data.
            const k: usize = 8; // elements per row
            const n_rows: usize = 2;
            const words_per_row = k * 4 / bits_per_u32; // 4-bit: 1 word per 8 elems
            var pw: [n_rows * words_per_row]u32 = undefined;
            var pw_bytes: [n_rows * words_per_row * 4]u8 = undefined;
            smith.bytesWithHash(&pw_bytes, 1);
            pw = @bitCast(pw_bytes);

            // BF16 scales and biases: use known-good values.
            const sc = [n_rows]u16{ 0x3F80, 0x3F80 }; // bf16(1.0)
            const bi = [n_rows]u16{ 0x0000, 0x0000 }; // bf16(0.0)
            var out: [k]f32 = undefined;

            const row_idx = raw[1] % n_rows;
            mlxEmbLookup(&out, &pw, &sc, &bi, row_idx, k, 4);

            // Invariant: output must be finite (scale=1.0, bias=0.0, values 0..15).
            for (out) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "mxfp4ScaleFormat uses E8M0 for MLX expert group size" {
    try std.testing.expectEqual(Mxfp4ScaleFormat.fp8_e4m3, mxfp4ScaleFormat(false, mxfp4_mlx_expert_group_size));
    try std.testing.expectEqual(Mxfp4ScaleFormat.fp8_e4m3, mxfp4ScaleFormat(true, mxfp4_group_size));
    try std.testing.expectEqual(Mxfp4ScaleFormat.e8m0, mxfp4ScaleFormat(true, mxfp4_mlx_expert_group_size));
}
