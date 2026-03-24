//! Quantization operations and format conversions.
//! Shared dequantization building blocks used by CPU and GPU backends.
//! Dequantization is performed in-kernel via these helpers — CPU backends
//! dequantize during GEMV; GPU backends use native shader/PTX equivalents.

const std = @import("std");
const DType = @import("../backend/backend.zig").DType;

/// FP8 E4M3 denormal scale: 2^(-6) / 8 = 2^(-9).
const fp8_e4m3_denorm_scale: f32 = 1.0 / 512.0;
/// FP8 E5M2 denormal scale: 2^(-14) / 4 = 2^(-16).
const fp8_e5m2_denorm_scale: f32 = 1.0 / 65536.0;

/// Elements per Q8_0 / Q4_0 quantization block.
pub const quant_block_elems: usize = 32;
/// Bytes per Q8_0 block: 2-byte f16 scale + 32 int8 values.
pub const q8_0_block_bytes: usize = 34;
/// Bytes per Q4_0 block: 2-byte f16 scale + 16 nibble bytes.
const q4_0_block_bytes: usize = 18;

/// Convert a BF16 value (stored as u16) to f32.
/// BF16 shares f32's exponent range; conversion is a 16-bit left shift.
pub inline fn bf16ToF32(val: u16) f32 {
    return @bitCast(@as(u32, val) << 16);
}

/// E2M1 FP4 lookup table (OCP Microscaling Spec).
/// Maps a 4-bit nibble to its f32 value.
pub inline fn mxfp4Lookup(nibble: u8) f32 {
    const table = [16]f32{
        0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    };
    return table[nibble];
}

/// E8M0 scale: pure power-of-2 with bias 127.
/// Returns 2^(e - 127) for e > 0; for e = 0, returns 2^(-126) (smallest normal f32).
pub inline fn e8m0ToF32(e: u8) f32 {
    if (e == 0) return @bitCast(@as(u32, 0x00800000)); // 2^(-126) = smallest normal
    return @bitCast(@as(u32, @intCast(e)) << 23);
}

/// Extract scale and minimum from Q5_K / Q4_K packed scale byte array.
pub inline fn getScaleMinK4(j: usize, q: []const u8, sc: *u8, m: *u8) void {
    if (j < 4) {
        sc.* = q[j] & 63;
        m.* = q[j + 4] & 63;
    } else {
        sc.* = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m.* = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/// Dequantize an NVFP4 nibble using the E2M1 lookup (same values as MXFP4),
/// scaled by an FP8 E4M3 block scale.
pub inline fn nvfp4Dequant(nibble: u8, block_scale_fp8: u8) f32 {
    return mxfp4Lookup(nibble) * fp8e4m3ToF32(block_scale_fp8);
}

/// IQ4_NL non-linear lookup table: maps a 4-bit nibble to an i8 dequantized value.
/// Used by IQ4_NL and IQ4_XS formats.
pub const iq4nl_table: [16]i8 = .{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

/// Compute FP8 E4M3 → f32 conversion (used to build comptime LUT).
/// Bit layout: seeeemmm. No infinities; e=15,m=7 is NaN.
fn fp8e4m3Compute(val: u8) f32 {
    const sign: u32 = @as(u32, val >> 7) << 31;
    const exp: u32 = (val >> 3) & 0x0F;
    const mant: u32 = val & 0x07;

    if (exp == 0x0F and mant == 0x07) {
        return @bitCast(sign | 0x7FC00000); // NaN
    }

    if (exp == 0) {
        if (mant == 0) return @bitCast(sign); // +/- 0
        const fmant: f32 = @floatFromInt(mant);
        const val_abs: f32 = fmant * fp8_e4m3_denorm_scale;
        return @bitCast(sign | @as(u32, @bitCast(val_abs)));
    }

    // Normal: value = (-1)^s * 2^(e-7) * (1 + m/8)
    const exp_f32: u32 = (exp + 120) << 23;
    const mant_f32: u32 = mant << 20;
    return @bitCast(sign | exp_f32 | mant_f32);
}

/// Precomputed FP8 E4M3 → f32 lookup table (256 entries, built at comptime).
/// Eliminates all branches and arithmetic from the hot-path conversion.
const fp8e4m3_lut = blk: {
    var table: [256]f32 = undefined;
    for (0..256) |i| table[i] = fp8e4m3Compute(@intCast(i));
    break :blk table;
};

/// Convert an FP8 E4M3 value to f32 via comptime lookup table.
/// Single array index — no branches, no arithmetic at runtime.
pub inline fn fp8e4m3ToF32(val: u8) f32 {
    return fp8e4m3_lut[val];
}

/// Compute FP8 E5M2 → f32 conversion (used to build comptime LUT).
/// Bit layout: seeeeemm. Has infinities (e=31,m=0) and NaN (e=31,m!=0).
fn fp8e5m2Compute(val: u8) f32 {
    const sign: u32 = @as(u32, val >> 7) << 31;
    const exp: u32 = (val >> 2) & 0x1F;
    const mant: u32 = val & 0x03;

    if (exp == 0x1F) {
        if (mant == 0) return @bitCast(sign | 0x7F800000); // Infinity
        return @bitCast(sign | 0x7FC00000); // NaN
    }

    if (exp == 0) {
        if (mant == 0) return @bitCast(sign); // +/- 0
        const fmant: f32 = @floatFromInt(mant);
        const val_abs: f32 = fmant * fp8_e5m2_denorm_scale;
        return @bitCast(sign | @as(u32, @bitCast(val_abs)));
    }

    // Normal: value = (-1)^s * 2^(e-15) * (1 + m/4)
    const exp_f32: u32 = (exp + 112) << 23;
    const mant_f32: u32 = mant << 21;
    return @bitCast(sign | exp_f32 | mant_f32);
}

/// Precomputed FP8 E5M2 → f32 lookup table (256 entries, built at comptime).
const fp8e5m2_lut = blk: {
    var table: [256]f32 = undefined;
    for (0..256) |i| table[i] = fp8e5m2Compute(@intCast(i));
    break :blk table;
};

/// Convert an FP8 E5M2 value to f32 via comptime lookup table.
/// Single array index — no branches, no arithmetic at runtime.
pub inline fn fp8e5m2ToF32(val: u8) f32 {
    return fp8e5m2_lut[val];
}

/// Dequantize tensor data to f32.
/// Handles f32 (pass-through), bf16, f16, q8_0, and q4_0 formats.
/// Unsupported dtypes trigger a panic.
/// Used for "direct read" tensors (norms, biases, conv weights) that are
/// passed to CPU code expecting [*]const f32 but may be stored quantized.
///
/// Parameters:
///   - output: F32 destination buffer [n].
///   - data: Raw tensor data pointer.
///   - dtype: Data format of the tensor.
///   - n: Number of elements to dequantize.
pub fn dequantToF32(output: []f32, data: [*]const u8, dtype: DType, n: usize) void {
    switch (dtype) {
        .f32 => {
            const src: [*]const f32 = @ptrCast(@alignCast(data));
            @memcpy(output[0..n], src[0..n]);
        },
        .bf16 => {
            const src: [*]const u16 = @ptrCast(@alignCast(data));
            for (0..n) |i| output[i] = bf16ToF32(src[i]);
        },
        .f16 => {
            const src: [*]const f16 = @ptrCast(@alignCast(data));
            for (0..n) |i| output[i] = @floatCast(src[i]);
        },
        .q8_0 => {
            const n_blocks = (n + quant_block_elems - 1) / quant_block_elems;
            for (0..n_blocks) |b| {
                const blk = data[b * q8_0_block_bytes ..];
                const scale: f32 = @floatCast(@as(*const f16, @ptrCast(@alignCast(blk))).*);
                const count = @min(quant_block_elems, n - b * quant_block_elems);
                for (0..count) |i| {
                    output[b * quant_block_elems + i] = scale * @as(f32, @floatFromInt(@as(i8, @bitCast(blk[2 + i]))));
                }
            }
        },
        .q4_0 => {
            const n_blocks = (n + quant_block_elems - 1) / quant_block_elems;
            for (0..n_blocks) |b| {
                const blk = data[b * q4_0_block_bytes ..];
                const scale: f32 = @floatCast(@as(*const f16, @ptrCast(@alignCast(blk))).*);
                const nibbles = blk[2..];
                const count = @min(quant_block_elems, n - b * quant_block_elems);
                for (0..count) |i| {
                    const byte = nibbles[i / 2];
                    const nibble: i8 = if (i % 2 == 0)
                        @as(i8, @intCast(byte & 0xF)) - 8
                    else
                        @as(i8, @intCast(byte >> 4)) - 8;
                    output[b * quant_block_elems + i] = scale * @as(f32, @floatFromInt(nibble));
                }
            }
        },
        else => {
            @panic("dequantToF32: unsupported dtype");
        },
    }
}

/// GEMV for SafeTensors NVFP4: separate weight nibble array and FP8 E4M3 scale array.
///
/// Standard sequential nibble packing: byte j contains elements 2j (low nibble)
/// and 2j+1 (high nibble). Each group of 16 elements shares one FP8 E4M3 scale.
///
/// Parameters:
///   - x: Input vector [k].
///   - weight: Packed nibbles [n * k/2] bytes, row-major.
///   - scale: FP8 E4M3 block scales [n * k/16] bytes, row-major.
///   - y: Output vector [n].
///   - n: Number of output rows.
///   - k: Number of input columns (must be divisible by 16).
pub fn gemvNvfp4St(x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    std.debug.assert(k % 16 == 0);
    const bytes_per_row = k / 2;
    const scales_per_row = k / 16;
    const V8 = @Vector(8, f32);

    for (0..n) |row| {
        var acc: V8 = @splat(0.0);
        const w_row = weight + row * bytes_per_row;
        const s_row = scale + row * scales_per_row;
        for (0..scales_per_row) |g| {
            const sv: V8 = @splat(fp8e4m3ToF32(s_row[g]));
            const base = g * 16;

            // Unpack 8 bytes → 16 f32 values via LUT, then SIMD multiply-accumulate
            var vals: [16]f32 = undefined;
            inline for (0..8) |j| {
                const byte = w_row[g * 8 + j];
                vals[2 * j] = mxfp4Lookup(byte & 0x0F);
                vals[2 * j + 1] = mxfp4Lookup(byte >> 4);
            }

            const v0: V8 = vals[0..8].*;
            const x0: V8 = x[base..][0..8].*;
            acc += sv * v0 * x0;

            const v1: V8 = vals[8..16].*;
            const x1: V8 = x[base + 8 ..][0..8].*;
            acc += sv * v1 * x1;
        }
        y[row] = @reduce(.Add, acc);
    }
}

/// GEMV for SafeTensors NVFP4 with per-block scaling for numerical stability.
///
/// Instead of applying the FP8 scale to all 16 dequantized values before dot product,
/// this variant computes the unscaled dot product per block first, then multiplies by
/// the block scale. This prevents catastrophic error accumulation when router logits
/// would otherwise overflow (e.g., Nemotron Nano MoE router scores → -2.3e26).
///
/// Formula: y[row] = sum_blocks( scale[block] * dot(q_block, x_block) )
///
/// Parameters:
///   - x: Input vector [k].
///   - weight: Packed nibbles [n * k/2] bytes, row-major.
///   - scale: FP8 E4M3 block scales [n * k/16] bytes, row-major.
///   - y: Output vector [n].
///   - n: Number of output rows.
///   - k: Number of input columns (must be divisible by 16).
pub fn gemvNvfp4StPerBlock(x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    std.debug.assert(k % 16 == 0);
    const bytes_per_row = k / 2;
    const scales_per_row = k / 16;
    const V8 = @Vector(8, f32);

    for (0..n) |row| {
        var sum_scaled_blocks: f32 = 0.0;
        const w_row = weight + row * bytes_per_row;
        const s_row = scale + row * scales_per_row;

        for (0..scales_per_row) |g| {
            const base = g * 16;

            // Unpack 8 bytes → 16 unscaled f32 values via E2M1 LUT (MXFP4 baseline values)
            var vals: [16]f32 = undefined;
            inline for (0..8) |j| {
                const byte = w_row[g * 8 + j];
                vals[2 * j] = mxfp4Lookup(byte & 0x0F);
                vals[2 * j + 1] = mxfp4Lookup(byte >> 4);
            }

            // Compute unscaled dot product for this block (16 elements)
            const v0: V8 = vals[0..8].*;
            const x0: V8 = x[base..][0..8].*;
            var block_dot: V8 = v0 * x0;

            const v1: V8 = vals[8..16].*;
            const x1: V8 = x[base + 8 ..][0..8].*;
            block_dot += v1 * x1;

            const block_sum: f32 = @reduce(.Add, block_dot);

            // Apply FP8 E4M3 scale to the block sum (not to individual values)
            const block_scale: f32 = fp8e4m3ToF32(s_row[g]);
            sum_scaled_blocks += block_scale * block_sum;
        }

        y[row] = sum_scaled_blocks;
    }
}

// ── Tests ─────────────────────────────────────────────────────────

test "bf16ToF32" {
    // BF16 1.0 = 0x3F80
    try std.testing.expectEqual(@as(f32, 1.0), bf16ToF32(0x3F80));
    // BF16 -1.0 = 0xBF80
    try std.testing.expectEqual(@as(f32, -1.0), bf16ToF32(0xBF80));
    // BF16 0.0 = 0x0000
    try std.testing.expectEqual(@as(f32, 0.0), bf16ToF32(0x0000));
}

test "mxfp4Lookup" {
    // Positive side (0x0 - 0x7)
    try std.testing.expectEqual(@as(f32, 0.0), mxfp4Lookup(0));
    try std.testing.expectEqual(@as(f32, 0.5), mxfp4Lookup(1));
    try std.testing.expectEqual(@as(f32, 1.0), mxfp4Lookup(2));
    try std.testing.expectEqual(@as(f32, 6.0), mxfp4Lookup(7));
    // Negative side (0x8 - 0xF): sign bit set
    try std.testing.expectEqual(@as(f32, -0.0), mxfp4Lookup(8));
    try std.testing.expectEqual(@as(f32, -0.5), mxfp4Lookup(9));
    try std.testing.expectEqual(@as(f32, -1.0), mxfp4Lookup(10));
    try std.testing.expectEqual(@as(f32, -6.0), mxfp4Lookup(15));
}

test "e8m0ToF32" {
    // e=127 → 2^0 = 1.0
    try std.testing.expectEqual(@as(f32, 1.0), e8m0ToF32(127));
    // e=128 → 2^1 = 2.0
    try std.testing.expectEqual(@as(f32, 2.0), e8m0ToF32(128));
    // e=126 → 2^(-1) = 0.5
    try std.testing.expectEqual(@as(f32, 0.5), e8m0ToF32(126));
}

test "getScaleMinK4" {
    var scales = [_]u8{ 0x3F, 0x1F, 0x0F, 0x07, 0x3E, 0x1E, 0x0E, 0x06, 0x12, 0x34, 0x56, 0x78 };
    var sc: u8 = undefined;
    var m: u8 = undefined;
    // j < 4 path: sc = q[j] & 63, m = q[j+4] & 63
    getScaleMinK4(0, &scales, &sc, &m);
    try std.testing.expectEqual(@as(u8, 0x3F), sc);
    try std.testing.expectEqual(@as(u8, 0x3E), m);
    // j >= 4 path: uses bit shifts to extract high bits from q[j-4] and q[j]
    getScaleMinK4(4, &scales, &sc, &m);
    // sc = (q[8] & 0xF) | ((q[0] >> 6) << 4) = (0x12 & 0xF) | ((0x3F >> 6) << 4) = 0x02 | 0x00 = 0x02
    try std.testing.expectEqual(@as(u8, 0x02), sc);
    // m = (q[8] >> 4) | ((q[4] >> 6) << 4) = (0x12 >> 4) | ((0x3E >> 6) << 4) = 0x01 | 0x00 = 0x01
    try std.testing.expectEqual(@as(u8, 0x01), m);
}

test "iq4nl_table" {
    // Verify boundary values of the non-linear lookup table
    try std.testing.expectEqual(@as(i8, -127), iq4nl_table[0]);
    try std.testing.expectEqual(@as(i8, 1), iq4nl_table[8]);
    try std.testing.expectEqual(@as(i8, 113), iq4nl_table[15]);
    try std.testing.expectEqual(@as(i8, -10), iq4nl_table[7]);
}

test "nvfp4Dequant" {
    // nibble=2 (1.0 from MXFP4), scale=0x38 (FP8 E4M3 1.0) → 1.0 * 1.0 = 1.0
    try std.testing.expectEqual(@as(f32, 1.0), nvfp4Dequant(2, 0x38));
    // nibble=0 (0.0), any scale → 0.0
    try std.testing.expectEqual(@as(f32, 0.0), nvfp4Dequant(0, 0x38));
    // nibble=4 (2.0), scale=0x40 (FP8 E4M3 2.0) → 2.0 * 2.0 = 4.0
    try std.testing.expectEqual(@as(f32, 4.0), nvfp4Dequant(4, 0x40));
}

test "fp8e4m3ToF32" {
    // Zero: 0b00000000
    try std.testing.expectEqual(@as(f32, 0.0), fp8e4m3ToF32(0x00));
    // 1.0: e=7 (0b0111), m=0 → (-1)^0 * 2^(7-7) * (1+0) = 1.0
    // Encoding: 0_0111_000 = 0x38
    try std.testing.expectEqual(@as(f32, 1.0), fp8e4m3ToF32(0x38));
    // -1.0: 0b10111000 = 0xB8
    try std.testing.expectEqual(@as(f32, -1.0), fp8e4m3ToF32(0xB8));
    // 1.5: e=7, m=4 → 2^0 * (1 + 4/8) = 1.5
    // Encoding: 0_0111_100 = 0x3C
    try std.testing.expectEqual(@as(f32, 1.5), fp8e4m3ToF32(0x3C));
    // 2.0: e=8, m=0 → 2^1 = 2.0
    // Encoding: 0_1000_000 = 0x40
    try std.testing.expectEqual(@as(f32, 2.0), fp8e4m3ToF32(0x40));
    // Max normal: e=14, m=7 → 2^7 * (1+7/8) = 128 * 1.875 = 240
    // Encoding: 0_1110_111 = 0x77
    try std.testing.expectEqual(@as(f32, 240.0), fp8e4m3ToF32(0x77));
    // NaN: e=15, m=7 → 0_1111_111 = 0x7F
    try std.testing.expect(std.math.isNan(fp8e4m3ToF32(0x7F)));
    // Smallest denorm: e=0, m=1 → 2^(-6) * 1/8 = 2^(-9)
    // Encoding: 0_0000_001 = 0x01
    const smallest = fp8e4m3ToF32(0x01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 512.0), smallest, 1e-10);
}

test "fp8e5m2ToF32" {
    // Zero: 0b00000000
    try std.testing.expectEqual(@as(f32, 0.0), fp8e5m2ToF32(0x00));
    // 1.0: e=15 (0b01111), m=0 → 2^(15-15) * 1 = 1.0
    // Encoding: 0_01111_00 = 0x3C
    try std.testing.expectEqual(@as(f32, 1.0), fp8e5m2ToF32(0x3C));
    // -1.0: 0b10111100 = 0xBC
    try std.testing.expectEqual(@as(f32, -1.0), fp8e5m2ToF32(0xBC));
    // 1.5: e=15, m=2 → 2^0 * (1 + 2/4) = 1.5
    // Encoding: 0_01111_10 = 0x3E
    try std.testing.expectEqual(@as(f32, 1.5), fp8e5m2ToF32(0x3E));
    // 2.0: e=16, m=0 → 2^1 = 2.0
    // Encoding: 0_10000_00 = 0x40
    try std.testing.expectEqual(@as(f32, 2.0), fp8e5m2ToF32(0x40));
    // +Inf: e=31, m=0 → 0_11111_00 = 0x7C
    try std.testing.expect(std.math.isInf(fp8e5m2ToF32(0x7C)));
    // -Inf: 1_11111_00 = 0xFC
    try std.testing.expect(std.math.isInf(fp8e5m2ToF32(0xFC)));
    // NaN: e=31, m!=0 → 0_11111_01 = 0x7D
    try std.testing.expect(std.math.isNan(fp8e5m2ToF32(0x7D)));
    // Smallest denorm: e=0, m=1 → 2^(-14) * 1/4 = 2^(-16)
    // Encoding: 0_00000_01 = 0x01
    const smallest = fp8e5m2ToF32(0x01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 65536.0), smallest, 1e-12);
}

test "gemvNvfp4St basic" {
    // 1x16 GEMV: one output row, 16 input elements (one group).
    // x = [1.0, 0, 0, ..., 0]
    const x = [16]f32{ 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    // nibble 0x2 = mxfp4(2) = 1.0. byte[0] low nibble = elem[0] = 1.0.
    const weight = [8]u8{ 0x02, 0, 0, 0, 0, 0, 0, 0 };
    // scale = 0x38 = FP8 E4M3 for 1.0
    const scale = [1]u8{0x38};
    var y = [1]f32{99.0}; // trap value to verify function overwrites output
    gemvNvfp4St(&x, &weight, &scale, &y, 1, 16);
    // x[0]=1.0 * mxfp4(2)=1.0 * fp8(0x38)=1.0 = 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), y[0], 1e-6);
}

test "gemvNvfp4St multi-row" {
    // 2x16 GEMV: two output rows.
    const x = [16]f32{ 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    // Row 0: byte[0] = 0x42 → low=2 (1.0), high=4 (2.0)
    // Row 1: byte[0] = 0x24 → low=4 (2.0), high=2 (1.0)
    const weight = [16]u8{ 0x42, 0, 0, 0, 0, 0, 0, 0, 0x24, 0, 0, 0, 0, 0, 0, 0 };
    const scale = [2]u8{ 0x38, 0x38 }; // both rows scale = 1.0
    var y = [2]f32{ 0, 0 };
    gemvNvfp4St(&x, &weight, &scale, &y, 2, 16);
    // Row 0: 1.0*1.0 + 1.0*2.0 = 3.0
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), y[0], 1e-6);
    // Row 1: 1.0*2.0 + 1.0*1.0 = 3.0
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), y[1], 1e-6);
}
