//! KV cache quantization — store/load operations for compressed KV cache.
//!
//! Supported formats:
//!   - f32:       Full precision (4 bytes/element, baseline)
//!   - f16:       Half precision (2 bytes/element, lossless for inference)
//!   - q8_0:      Block-quantized INT8 with f16 scale per 32 elements (1.0625 B/elem)
//!   - int8:      Block-quantized INT8 with f32 scale per 32 elements (1.125 B/elem)
//!   - fp8_e4m3:  FP8 E4M3 format (1 byte/element, hardware-native on Hopper+)
//!   - nvfp4:     NVFP4 E2M1 with FP8 scale per 16 elements (0.5625 B/elem)

const std = @import("std");
const quant = @import("quant.zig");

/// Block size for Q8_0 and INT8 quantization.
const block_size: usize = 32;
/// Q8_0 block: f16 scale (2 bytes) + 32 i8 values = 34 bytes.
const q8_0_bytes: usize = 34;
/// INT8 block: f32 scale (4 bytes) + 32 i8 values = 36 bytes.
const int8_bytes: usize = 36;
/// NVFP4 block size: 16 elements.
const nvfp4_block: usize = 16;
/// NVFP4 block: fp8 scale (1 byte) + 8 packed nibble bytes = 9 bytes.
const nvfp4_bytes: usize = 9;
/// Maximum representable INT8 value (scale normalization factor for Q8_0/INT8).
const int8_max: f32 = 127.0;
/// Maximum representable E2M1 value (scale normalization factor for NVFP4).
const e2m1_max: f32 = 6.0;

/// Quantization type for KV cache storage.
pub const KvQuantType = enum {
    f32,
    f16,
    q8_0,
    int8,
    fp8_e4m3,
    nvfp4,

    /// Human-readable name for display.
    pub fn name(self: KvQuantType) []const u8 {
        return switch (self) {
            .f32 => "F32",
            .f16 => "F16",
            .q8_0 => "Q8_0",
            .int8 => "INT8",
            .fp8_e4m3 => "FP8",
            .nvfp4 => "NVFP4",
        };
    }

    /// Bits per element (approximate, includes scale overhead).
    pub fn bitsPerElement(self: KvQuantType) f32 {
        return switch (self) {
            .f32 => 32.0,
            .f16 => 16.0,
            .q8_0 => 8.5,
            .int8 => 9.0,
            .fp8_e4m3 => 8.0,
            .nvfp4 => 4.5,
        };
    }

    /// Parse from CLI string (case-insensitive).
    pub fn fromString(s: []const u8) ?KvQuantType {
        const eql = std.ascii.eqlIgnoreCase;
        if (eql(s, "f32")) return .f32;
        if (eql(s, "f16")) return .f16;
        if (eql(s, "q8_0") or eql(s, "q8")) return .q8_0;
        if (eql(s, "int8") or eql(s, "i8")) return .int8;
        if (eql(s, "fp8") or eql(s, "fp8_e4m3")) return .fp8_e4m3;
        if (eql(s, "nvfp4") or eql(s, "fp4")) return .nvfp4;
        return null;
    }
};

// ── Allocation sizing ────────────────────────────────────────────

/// Compute byte storage needed for `n` logical f32 elements.
pub fn kvSliceBytes(kv_type: KvQuantType, n: usize) usize {
    return switch (kv_type) {
        .f32 => n * 4,
        .f16 => n * 2,
        .q8_0 => ((n + block_size - 1) / block_size) * q8_0_bytes,
        .int8 => ((n + block_size - 1) / block_size) * int8_bytes,
        .fp8_e4m3 => n,
        .nvfp4 => ((n + nvfp4_block - 1) / nvfp4_block) * nvfp4_bytes,
    };
}

/// Byte offset for element index `i` (start of the block containing element `i`).
/// For element-wise formats (f32, f16, fp8), this is the exact byte offset.
/// For block formats, this is the start of the containing block.
pub fn kvByteOffset(kv_type: KvQuantType, i: usize) usize {
    return switch (kv_type) {
        .f32 => i * 4,
        .f16 => i * 2,
        .q8_0 => (i / block_size) * q8_0_bytes,
        .int8 => (i / block_size) * int8_bytes,
        .fp8_e4m3 => i,
        .nvfp4 => (i / nvfp4_block) * nvfp4_bytes,
    };
}

// ── Store (quantize f32 → format) ────────────────────────────────

/// Quantize `n` f32 values from `src` and write to `dst` in the given format.
pub fn kvStore(dst: [*]u8, src: [*]const f32, n: usize, kv_type: KvQuantType) void {
    switch (kv_type) {
        .f32 => storeF32(dst, src, n),
        .f16 => storeF16(dst, src, n),
        .q8_0 => storeQ8_0(dst, src, n),
        .int8 => storeInt8(dst, src, n),
        .fp8_e4m3 => storeFp8(dst, src, n),
        .nvfp4 => storeNvfp4(dst, src, n),
    }
}

fn storeF32(dst: [*]u8, src: [*]const f32, n: usize) void {
    @memcpy(dst[0 .. n * 4], @as([*]const u8, @ptrCast(src))[0 .. n * 4]);
}

fn storeF16(dst: [*]u8, src: [*]const f32, n: usize) void {
    const out: [*]u16 = @ptrCast(@alignCast(dst));
    for (0..n) |i| {
        out[i] = @bitCast(@as(f16, @floatCast(src[i])));
    }
}

fn storeQ8_0(dst: [*]u8, src: [*]const f32, n: usize) void {
    const nb = (n + block_size - 1) / block_size;
    for (0..nb) |b| {
        const base = b * block_size;
        const count = @min(block_size, n - base);
        // Find absmax
        var amax: f32 = 0;
        for (0..count) |i| amax = @max(amax, @abs(src[base + i]));
        const scale: f16 = if (amax > 0) @floatCast(amax / int8_max) else 0;
        const inv_scale: f32 = if (amax > 0) int8_max / amax else 0;
        // Write scale (f16)
        const bp = dst + b * q8_0_bytes;
        @as(*align(1) u16, @ptrCast(bp)).* = @bitCast(scale);
        // Write quantized values
        for (0..count) |i| {
            const v = src[base + i] * inv_scale;
            bp[2 + i] = @bitCast(@as(i8, @intFromFloat(std.math.clamp(std.math.round(v), -128, 127))));
        }
        // Zero-pad remainder
        for (count..block_size) |i| bp[2 + i] = 0;
    }
}

fn storeInt8(dst: [*]u8, src: [*]const f32, n: usize) void {
    const nb = (n + block_size - 1) / block_size;
    for (0..nb) |b| {
        const base = b * block_size;
        const count = @min(block_size, n - base);
        // Find absmax
        var amax: f32 = 0;
        for (0..count) |i| amax = @max(amax, @abs(src[base + i]));
        const scale: f32 = if (amax > 0) amax / int8_max else 0;
        const inv_scale: f32 = if (amax > 0) int8_max / amax else 0;
        // Write scale (f32)
        const bp = dst + b * int8_bytes;
        @as(*align(1) f32, @ptrCast(bp)).* = scale;
        // Write quantized values
        for (0..count) |i| {
            const v = src[base + i] * inv_scale;
            bp[4 + i] = @bitCast(@as(i8, @intFromFloat(std.math.clamp(std.math.round(v), -128, 127))));
        }
        for (count..block_size) |i| bp[4 + i] = 0;
    }
}

fn storeFp8(dst: [*]u8, src: [*]const f32, n: usize) void {
    for (0..n) |i| {
        dst[i] = f32ToFp8E4M3(src[i]);
    }
}

fn storeNvfp4(dst: [*]u8, src: [*]const f32, n: usize) void {
    const nb = (n + nvfp4_block - 1) / nvfp4_block;
    for (0..nb) |b| {
        const base = b * nvfp4_block;
        const count = @min(nvfp4_block, n - base);
        // Find absmax
        var amax: f32 = 0;
        for (0..count) |i| amax = @max(amax, @abs(src[base + i]));
        // Compute FP8 E4M3 scale: scale = amax / e2m1_max
        const scale_f32: f32 = if (amax > 0) amax / e2m1_max else 0;
        const scale_fp8 = f32ToFp8E4M3(scale_f32);
        const inv_scale: f32 = if (amax > 0) e2m1_max / amax else 0;

        const bp = dst + b * nvfp4_bytes;
        bp[0] = scale_fp8; // FP8 scale
        // Pack pairs of E2M1 nibbles
        for (0..8) |pair| {
            const idx0 = pair * 2;
            const idx1 = idx0 + 1;
            const v0: f32 = if (idx0 < count) src[base + idx0] * inv_scale else 0;
            const v1: f32 = if (idx1 < count) src[base + idx1] * inv_scale else 0;
            const n0 = f32ToE2M1(v0);
            const n1 = f32ToE2M1(v1);
            bp[1 + pair] = n0 | (n1 << 4);
        }
    }
}

// ── Dot product (query · quantized_kv) ───────────────────────────

/// Compute dot product between f32 query vector and quantized KV vector.
pub fn kvDot(q_vec: [*]const f32, kv_data: [*]const u8, n: usize, kv_type: KvQuantType) f32 {
    return switch (kv_type) {
        .f32 => dotF32(q_vec, kv_data, n),
        .f16 => dotF16(q_vec, kv_data, n),
        .q8_0 => dotQ8_0(q_vec, kv_data, n),
        .int8 => dotInt8(q_vec, kv_data, n),
        .fp8_e4m3 => dotFp8(q_vec, kv_data, n),
        .nvfp4 => dotNvfp4(q_vec, kv_data, n),
    };
}

fn dotF32(q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const kv: [*]const f32 = @ptrCast(@alignCast(kv_data));
    var sum: f32 = 0;
    for (0..n) |i| sum += q_vec[i] * kv[i];
    return sum;
}

fn dotF16(q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const kv: [*]const u16 = @ptrCast(@alignCast(kv_data));
    var sum: f32 = 0;
    for (0..n) |i| {
        sum += q_vec[i] * @as(f32, @floatCast(@as(f16, @bitCast(kv[i]))));
    }
    return sum;
}

fn dotQ8_0(q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const nb = (n + block_size - 1) / block_size;
    var sum: f32 = 0;
    for (0..nb) |b| {
        const bp = kv_data + b * q8_0_bytes;
        const scale: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp)).*)));
        const base = b * block_size;
        const count = @min(block_size, n - base);
        var block_sum: f32 = 0;
        for (0..count) |i| {
            const val: f32 = @floatFromInt(@as(i8, @bitCast(bp[2 + i])));
            block_sum += q_vec[base + i] * val;
        }
        sum += scale * block_sum;
    }
    return sum;
}

fn dotInt8(q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const nb = (n + block_size - 1) / block_size;
    var sum: f32 = 0;
    for (0..nb) |b| {
        const bp = kv_data + b * int8_bytes;
        const scale: f32 = @as(*align(1) const f32, @ptrCast(bp)).*;
        const base = b * block_size;
        const count = @min(block_size, n - base);
        var block_sum: f32 = 0;
        for (0..count) |i| {
            const val: f32 = @floatFromInt(@as(i8, @bitCast(bp[4 + i])));
            block_sum += q_vec[base + i] * val;
        }
        sum += scale * block_sum;
    }
    return sum;
}

fn dotFp8(q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    var sum: f32 = 0;
    for (0..n) |i| {
        sum += q_vec[i] * quant.fp8e4m3ToF32(kv_data[i]);
    }
    return sum;
}

fn dotNvfp4(q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const nb = (n + nvfp4_block - 1) / nvfp4_block;
    var sum: f32 = 0;
    for (0..nb) |b| {
        const bp = kv_data + b * nvfp4_bytes;
        const scale: f32 = quant.fp8e4m3ToF32(bp[0]);
        const base = b * nvfp4_block;
        const count = @min(nvfp4_block, n - base);
        var block_sum: f32 = 0;
        for (0..count) |i| {
            const byte = bp[1 + i / 2];
            const nibble: u8 = if (i % 2 == 0) byte & 0x0F else byte >> 4;
            block_sum += q_vec[base + i] * quant.mxfp4Lookup(nibble);
        }
        sum += scale * block_sum;
    }
    return sum;
}

// ── Weighted accumulation (acc += weight * dequant(kv)) ──────────

/// Accumulate: acc[0..n] += weight * dequant(kv_data[0..n]).
pub fn kvMulAccum(acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize, kv_type: KvQuantType) void {
    switch (kv_type) {
        .f32 => mulAccF32(acc, weight, kv_data, n),
        .f16 => mulAccF16(acc, weight, kv_data, n),
        .q8_0 => mulAccQ8_0(acc, weight, kv_data, n),
        .int8 => mulAccInt8(acc, weight, kv_data, n),
        .fp8_e4m3 => mulAccFp8(acc, weight, kv_data, n),
        .nvfp4 => mulAccNvfp4(acc, weight, kv_data, n),
    }
}

fn mulAccF32(acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const kv: [*]const f32 = @ptrCast(@alignCast(kv_data));
    for (0..n) |i| acc[i] += weight * kv[i];
}

fn mulAccF16(acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const kv: [*]const u16 = @ptrCast(@alignCast(kv_data));
    for (0..n) |i| {
        acc[i] += weight * @as(f32, @floatCast(@as(f16, @bitCast(kv[i]))));
    }
}

fn mulAccQ8_0(acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const nb = (n + block_size - 1) / block_size;
    for (0..nb) |b| {
        const bp = kv_data + b * q8_0_bytes;
        const scale: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp)).*)));
        const ws = weight * scale;
        const base = b * block_size;
        const count = @min(block_size, n - base);
        for (0..count) |i| {
            acc[base + i] += ws * @as(f32, @floatFromInt(@as(i8, @bitCast(bp[2 + i]))));
        }
    }
}

fn mulAccInt8(acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const nb = (n + block_size - 1) / block_size;
    for (0..nb) |b| {
        const bp = kv_data + b * int8_bytes;
        const scale: f32 = @as(*align(1) const f32, @ptrCast(bp)).*;
        const ws = weight * scale;
        const base = b * block_size;
        const count = @min(block_size, n - base);
        for (0..count) |i| {
            acc[base + i] += ws * @as(f32, @floatFromInt(@as(i8, @bitCast(bp[4 + i]))));
        }
    }
}

fn mulAccFp8(acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    for (0..n) |i| {
        acc[i] += weight * quant.fp8e4m3ToF32(kv_data[i]);
    }
}

fn mulAccNvfp4(acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const nb = (n + nvfp4_block - 1) / nvfp4_block;
    for (0..nb) |b| {
        const bp = kv_data + b * nvfp4_bytes;
        const scale: f32 = quant.fp8e4m3ToF32(bp[0]);
        const ws = weight * scale;
        const base = b * nvfp4_block;
        const count = @min(nvfp4_block, n - base);
        for (0..count) |i| {
            const byte = bp[1 + i / 2];
            const nibble: u8 = if (i % 2 == 0) byte & 0x0F else byte >> 4;
            acc[base + i] += ws * quant.mxfp4Lookup(nibble);
        }
    }
}

// ── FP8 E4M3 f32→u8 conversion ──────────────────────────────────

/// Convert f32 to FP8 E4M3 (clamp to representable range, round to nearest).
/// E4M3: 1 sign + 4 exponent (bias=7) + 3 mantissa. Max value = 448.0.
fn f32ToFp8E4M3(val: f32) u8 {
    const bits: u32 = @bitCast(val);
    const sign: u8 = @truncate(bits >> 31);
    const abs_val = @abs(val);

    if (abs_val == 0) return sign << 7;
    if (!std.math.isFinite(abs_val)) return (sign << 7) | 0x7E; // max finite

    // Clamp to max representable: 448.0
    const clamped = @min(abs_val, 448.0);

    // Convert via float manipulation
    // E4M3 bias = 7, f32 bias = 127, so e4m3_exp = f32_exp - 127 + 7 = f32_exp - 120
    const f32_bits: u32 = @bitCast(clamped);
    const f32_exp: i32 = @as(i32, @intCast((f32_bits >> 23) & 0xFF)) - 127;
    const f32_mant: u32 = f32_bits & 0x7FFFFF;

    const e4m3_exp = f32_exp + 7;

    if (e4m3_exp <= 0) {
        // Denormal in E4M3 (exp = 0, implied 0.mantissa)
        // value = 2^(-6) * mantissa / 8
        const shift: u5 = @intCast(@min(24, 1 - e4m3_exp));
        const mant_with_implicit = (1 << 23) | f32_mant;
        const shifted = mant_with_implicit >> shift;
        const mant3: u8 = @truncate((shifted + (1 << 19)) >> 20); // round
        return (sign << 7) | @as(u8, @min(mant3, 7));
    }

    if (e4m3_exp >= 15) {
        return (sign << 7) | 0x7E; // max finite (exp=15 is not used for inf in E4M3)
    }

    // Normal: round mantissa from 23 bits to 3 bits
    const mant3: u8 = @truncate((f32_mant + (1 << 19)) >> 20);
    if (mant3 >= 8) {
        // Mantissa overflow → increment exponent
        const new_exp: u8 = @intCast(e4m3_exp + 1);
        if (new_exp >= 15) return (sign << 7) | 0x7E;
        return (sign << 7) | (new_exp << 3);
    }
    return (sign << 7) | (@as(u8, @intCast(e4m3_exp)) << 3) | mant3;
}

// ── E2M1 f32→nibble conversion ───────────────────────────────────

/// Convert f32 (pre-scaled to [-6, 6]) to E2M1 4-bit nibble.
/// E2M1 representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (+ negatives).
fn f32ToE2M1(val: f32) u8 {
    const sign: u8 = if (val < 0) 8 else 0; // bit 3 = sign
    const abs_v = @abs(val);
    // Find nearest representable value
    const nibble: u8 = if (abs_v < 0.25) 0 // 0.0
        else if (abs_v < 0.75) 1 // 0.5
        else if (abs_v < 1.25) 2 // 1.0
        else if (abs_v < 1.75) 3 // 1.5
        else if (abs_v < 2.5) 4 // 2.0
        else if (abs_v < 3.5) 5 // 3.0
        else if (abs_v < 5.0) 6 // 4.0
        else 7; // 6.0
    return sign | nibble;
}

// ── Tests ─────────────────────────────────────────────────────────

test "kvSliceBytes" {
    // f32: 4 bytes per element
    try std.testing.expectEqual(@as(usize, 128), kvSliceBytes(.f32, 32));
    // f16: 2 bytes per element
    try std.testing.expectEqual(@as(usize, 64), kvSliceBytes(.f16, 32));
    // q8_0: 34 bytes per 32 elements
    try std.testing.expectEqual(@as(usize, 34), kvSliceBytes(.q8_0, 32));
    try std.testing.expectEqual(@as(usize, 68), kvSliceBytes(.q8_0, 64));
    // int8: 36 bytes per 32 elements
    try std.testing.expectEqual(@as(usize, 36), kvSliceBytes(.int8, 32));
    // fp8: 1 byte per element
    try std.testing.expectEqual(@as(usize, 32), kvSliceBytes(.fp8_e4m3, 32));
    // nvfp4: 9 bytes per 16 elements
    try std.testing.expectEqual(@as(usize, 18), kvSliceBytes(.nvfp4, 32));
    try std.testing.expectEqual(@as(usize, 9), kvSliceBytes(.nvfp4, 16));
}

test "f16 roundtrip" {
    const src = [_]f32{ 1.0, -0.5, 3.14, 0.0, -7.25, 100.0, 0.001, -0.001 };
    var buf: [16]u8 = undefined;
    kvStore(&buf, &src, 8, .f16);
    for (0..8) |i| {
        const expected: f32 = @floatCast(@as(f16, @floatCast(src[i])));
        var dot_q = [1]f32{0};
        var dot_v = [1]f32{1.0};
        _ = kvDot(&dot_v, buf[i * 2 ..].ptr, 1, .f16);
        // Verify via mulAccum
        kvMulAccum(&dot_q, 1.0, buf[i * 2 ..].ptr, 1, .f16);
        try std.testing.expectApproxEqAbs(expected, dot_q[0], 1e-6);
    }
}

test "q8_0 roundtrip accuracy" {
    // Values in a range where Q8_0 should preserve well
    const src = [_]f32{ 0.5, -0.3, 1.0, -1.0, 0.0, 0.7, -0.8, 0.1 };
    var buf: [q8_0_bytes]u8 = undefined;
    kvStore(&buf, &src, 8, .q8_0);

    // Dot with unit vector [1,0,0,...] should ≈ src[0]
    var q_unit = [_]f32{0} ** 8;
    q_unit[0] = 1.0;
    const dot = kvDot(&q_unit, &buf, 8, .q8_0);
    try std.testing.expectApproxEqAbs(src[0], dot, 0.02);

    // Dot with all-ones should ≈ sum(src)
    var q_ones = [_]f32{1.0} ** 8;
    const dot_sum = kvDot(&q_ones, &buf, 8, .q8_0);
    var expected_sum: f32 = 0;
    for (src) |v| expected_sum += v;
    try std.testing.expectApproxEqAbs(expected_sum, dot_sum, 0.1);
}

test "int8 roundtrip accuracy" {
    const src = [_]f32{ 0.5, -0.3, 1.0, -1.0, 0.0, 0.7, -0.8, 0.1 };
    var buf: [int8_bytes]u8 = undefined;
    kvStore(&buf, &src, 8, .int8);

    var q_ones = [_]f32{1.0} ** 8;
    const dot = kvDot(&q_ones, &buf, 8, .int8);
    var expected: f32 = 0;
    for (src) |v| expected += v;
    try std.testing.expectApproxEqAbs(expected, dot, 0.1);
}

test "fp8_e4m3 roundtrip" {
    const src = [_]f32{ 1.0, -1.0, 0.5, 2.0, 0.0, -0.5, 4.0, -4.0 };
    var buf: [8]u8 = undefined;
    kvStore(&buf, &src, 8, .fp8_e4m3);

    // FP8 E4M3 should preserve these simple values exactly or very closely
    var q_unit = [_]f32{0} ** 8;
    q_unit[0] = 1.0;
    const dot = kvDot(&q_unit, &buf, 8, .fp8_e4m3);
    try std.testing.expectApproxEqAbs(src[0], dot, 0.1);

    // Test mulAccum
    var acc = [_]f32{0} ** 8;
    kvMulAccum(&acc, 1.0, &buf, 8, .fp8_e4m3);
    for (0..8) |i| {
        try std.testing.expectApproxEqAbs(src[i], acc[i], 0.01);
    }
}

test "nvfp4 roundtrip" {
    // NVFP4 has very limited precision — test with representable values
    const src = [_]f32{ 1.0, -1.0, 0.5, 2.0, 0.0, -0.5, 3.0, -3.0 };
    var buf: [nvfp4_bytes]u8 = undefined;
    kvStore(&buf, &src, 8, .nvfp4);

    var q_ones = [_]f32{1.0} ** 8;
    const dot = kvDot(&q_ones, &buf, 8, .nvfp4);
    var expected: f32 = 0;
    for (src) |v| expected += v;
    // All test values are exactly representable after E2M1 quantization + FP8 scale
    try std.testing.expectApproxEqAbs(expected, dot, 0.1);
}

test "kvByteOffset consistency" {
    // For element-wise formats, byteOffset should match sliceBytes
    try std.testing.expectEqual(kvSliceBytes(.f32, 10), kvByteOffset(.f32, 10));
    try std.testing.expectEqual(kvSliceBytes(.f16, 10), kvByteOffset(.f16, 10));
    try std.testing.expectEqual(kvSliceBytes(.fp8_e4m3, 10), kvByteOffset(.fp8_e4m3, 10));
    // For block formats, byteOffset gives start of containing block
    try std.testing.expectEqual(@as(usize, 0), kvByteOffset(.q8_0, 0));
    try std.testing.expectEqual(@as(usize, 0), kvByteOffset(.q8_0, 31));
    try std.testing.expectEqual(@as(usize, 34), kvByteOffset(.q8_0, 32));
}

test "fromString" {
    try std.testing.expectEqual(KvQuantType.f32, KvQuantType.fromString("f32").?);
    try std.testing.expectEqual(KvQuantType.f16, KvQuantType.fromString("F16").?);
    try std.testing.expectEqual(KvQuantType.q8_0, KvQuantType.fromString("q8_0").?);
    try std.testing.expectEqual(KvQuantType.q8_0, KvQuantType.fromString("Q8").?);
    try std.testing.expectEqual(KvQuantType.int8, KvQuantType.fromString("int8").?);
    try std.testing.expectEqual(KvQuantType.fp8_e4m3, KvQuantType.fromString("fp8").?);
    try std.testing.expectEqual(KvQuantType.nvfp4, KvQuantType.fromString("nvfp4").?);
    try std.testing.expectEqual(KvQuantType.nvfp4, KvQuantType.fromString("fp4").?);
    try std.testing.expect(KvQuantType.fromString("invalid") == null);
}

test "f32ToFp8E4M3 basic values" {
    // 1.0 in E4M3: exp=7 (bias=7, so stored as 7), mant=000 → 0b_0_0111_000 = 0x38
    try std.testing.expectEqual(@as(u8, 0x38), f32ToFp8E4M3(1.0));
    // 0.0
    try std.testing.expectEqual(@as(u8, 0x00), f32ToFp8E4M3(0.0));
    // -1.0
    try std.testing.expectEqual(@as(u8, 0xB8), f32ToFp8E4M3(-1.0));
    // Roundtrip: 3.5 is exactly representable in E4M3 (exp=8, mant=0b110)
    const rt = quant.fp8e4m3ToF32(f32ToFp8E4M3(3.5));
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), rt, 1e-4);
}

test "f32ToE2M1 basic values" {
    try std.testing.expectEqual(@as(u8, 0), f32ToE2M1(0.0)); // 0
    try std.testing.expectEqual(@as(u8, 2), f32ToE2M1(1.0)); // 1.0
    try std.testing.expectEqual(@as(u8, 10), f32ToE2M1(-1.0)); // -1.0 = sign|2
    try std.testing.expectEqual(@as(u8, 7), f32ToE2M1(6.0)); // 6.0
    try std.testing.expectEqual(@as(u8, 7), f32ToE2M1(10.0)); // clamps to 6.0
}
