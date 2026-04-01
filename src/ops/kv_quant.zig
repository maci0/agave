//! KV cache quantization — store/load operations for compressed KV cache.
//! Separate from src/ops/quant.zig which handles weight dequantization.
//! KV cache formats are optimized for inference access patterns (loads during SDPA).
//!
//! Supported formats:
//!   - f32:       Full precision (4 bytes/element, baseline)
//!   - f16:       Half precision (2 bytes/element, lossless for inference)
//!   - q8_0:      Block-quantized INT8 with f16 scale per 32 elements (1.0625 B/elem)
//!   - int8:      Block-quantized INT8 with f32 scale per 32 elements (1.125 B/elem)
//!   - fp8_e4m3:  FP8 E4M3 format (1 byte/element, hardware-native on Hopper+)
//!   - nvfp4:     NVFP4 E2M1 with FP8 scale per 16 elements (0.5625 B/elem)
//!   - turbo2:    TurboQuant 2-bit — WHT + Lloyd-Max codebook (2.5 bits/elem)
//!   - turbo3:    TurboQuant 3-bit — WHT + Lloyd-Max codebook (3.5 bits/elem)
//!   - turbo4:    TurboQuant 4-bit — WHT + Lloyd-Max codebook (4.5 bits/elem)

const std = @import("std");
const quant = @import("quant.zig");

/// Block size for Q8_0 and INT8 quantization (shared with quant.zig).
const block_size: usize = quant.quant_block_elems;
/// Q8_0 block: f16 scale (2 bytes) + 32 i8 values = 34 bytes (shared with quant.zig).
const q8_0_bytes: usize = quant.q8_0_block_bytes;
/// INT8 block: f32 scale (4 bytes) + 32 i8 values = 36 bytes.
const int8_bytes: usize = 36;
/// NVFP4 block size: 16 elements.
const nvfp4_block: usize = 16;
/// NVFP4 block: fp8 scale (1 byte) + 8 packed nibble bytes = 9 bytes.
const nvfp4_bytes: usize = 9;
/// Maximum representable INT8 value (scale normalization factor for Q8_0/INT8).
const int8_max: f32 = 127.0;
/// Minimum representable INT8 value (lower clamp bound for quantized values).
const int8_min: f32 = -128.0;
/// Maximum representable E2M1 value (scale normalization factor for NVFP4).
const e2m1_max: f32 = 6.0;

// ── TurboQuant constants ─────────────────────────────────────────

/// TurboQuant block size: 32 elements (matches WHT-32 transform).
const turbo_block_size: usize = 32;
/// TurboQuant 2-bit block: f16 norm (2 bytes) + 64 packed bits = 10 bytes.
const turbo2_block_bytes: usize = 10;
/// TurboQuant 3-bit block: f16 norm (2 bytes) + 96 packed bits = 14 bytes.
const turbo3_block_bytes: usize = 14;
/// TurboQuant 4-bit block: f16 norm (2 bytes) + 128 packed bits = 18 bytes.
const turbo4_block_bytes: usize = 18;
/// WHT normalization factor: 1 / sqrt(32).
const wht_inv_sqrt: f32 = 1.0 / @sqrt(32.0);

/// Lloyd-Max optimal centroids for N(0,1) quantized to 2 bits (4 levels).
const lloyd_max_2bit = [4]f32{ -1.510, -0.453, 0.453, 1.510 };
/// Lloyd-Max optimal centroids for N(0,1) quantized to 3 bits (8 levels).
const lloyd_max_3bit = [8]f32{ -2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152 };
/// Lloyd-Max optimal centroids for N(0,1) quantized to 4 bits (16 levels).
const lloyd_max_4bit = [16]f32{ -2.733, -2.069, -1.618, -1.256, -0.942, -0.657, -0.388, -0.128, 0.128, 0.388, 0.657, 0.942, 1.256, 1.618, 2.069, 2.733 };

/// Return the Lloyd-Max centroid table for a given bit width.
inline fn lloydMaxCodebook(comptime bits: u3) []const f32 {
    return switch (bits) {
        2 => &lloyd_max_2bit,
        3 => &lloyd_max_3bit,
        4 => &lloyd_max_4bit,
        else => @compileError("TurboQuant only supports 2, 3, or 4 bits"),
    };
}

/// Return block byte size for a given TurboQuant bit width.
inline fn turboBlockBytes(comptime bits: u3) usize {
    return switch (bits) {
        2 => turbo2_block_bytes,
        3 => turbo3_block_bytes,
        4 => turbo4_block_bytes,
        else => @compileError("TurboQuant only supports 2, 3, or 4 bits"),
    };
}

/// In-place Walsh-Hadamard Transform of 32 elements (5-stage butterfly network).
///
/// WHT is its own inverse up to a scale factor of 32:
///   WHT(WHT(x)) = 32 * x
///
/// The transform decorrelates the input signal, making it more amenable to
/// scalar quantization (coefficients tend toward Gaussian distribution).
inline fn wht32(buf: *[32]f32) void {
    // 5 stages of butterfly operations: stride 1, 2, 4, 8, 16
    comptime var stride: usize = 1;
    inline while (stride <= 16) : (stride *= 2) {
        comptime var i: usize = 0;
        inline while (i < 32) : (i += stride * 2) {
            comptime var j: usize = 0;
            inline while (j < stride) : (j += 1) {
                const a = buf[i + j];
                const b = buf[i + j + stride];
                buf[i + j] = a + b;
                buf[i + j + stride] = a - b;
            }
        }
    }
}

/// Quantization type for KV cache storage.
pub const KvQuantType = enum {
    f32,
    f16,
    q8_0,
    int8,
    fp8_e4m3,
    nvfp4,
    turbo2,
    turbo3,
    turbo4,

    /// Human-readable name for display.
    pub fn name(self: KvQuantType) []const u8 {
        return switch (self) {
            .f32 => "F32",
            .f16 => "F16",
            .q8_0 => "Q8_0",
            .int8 => "INT8",
            .fp8_e4m3 => "FP8",
            .nvfp4 => "NVFP4",
            .turbo2 => "TQ2",
            .turbo3 => "TQ3",
            .turbo4 => "TQ4",
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
            .turbo2 => 2.5,
            .turbo3 => 3.5,
            .turbo4 => 4.5,
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
        if (eql(s, "turbo2") or eql(s, "tq2")) return .turbo2;
        if (eql(s, "turbo3") or eql(s, "tq3")) return .turbo3;
        if (eql(s, "turbo4") or eql(s, "tq4")) return .turbo4;
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
        .turbo2 => ((n + turbo_block_size - 1) / turbo_block_size) * turbo2_block_bytes,
        .turbo3 => ((n + turbo_block_size - 1) / turbo_block_size) * turbo3_block_bytes,
        .turbo4 => ((n + turbo_block_size - 1) / turbo_block_size) * turbo4_block_bytes,
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
        .turbo2 => (i / turbo_block_size) * turbo2_block_bytes,
        .turbo3 => (i / turbo_block_size) * turbo3_block_bytes,
        .turbo4 => (i / turbo_block_size) * turbo4_block_bytes,
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
        .turbo2 => turboStore(2, dst, src, n),
        .turbo3 => turboStore(3, dst, src, n),
        .turbo4 => turboStore(4, dst, src, n),
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
            bp[2 + i] = @bitCast(@as(i8, @intFromFloat(std.math.clamp(std.math.round(v), int8_min, int8_max))));
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
            bp[4 + i] = @bitCast(@as(i8, @intFromFloat(std.math.clamp(std.math.round(v), int8_min, int8_max))));
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
        .turbo2 => turboDot(2, q_vec, kv_data, n),
        .turbo3 => turboDot(3, q_vec, kv_data, n),
        .turbo4 => turboDot(4, q_vec, kv_data, n),
    };
}

fn dotF32(q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const kv: [*]const f32 = @ptrCast(@alignCast(kv_data));
    const V8 = @Vector(8, f32);
    var acc: V8 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const qv: V8 = q_vec[i..][0..8].*;
        const kv_v: V8 = kv[i..][0..8].*;
        acc = @mulAdd(V8, qv, kv_v, acc);
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < n) : (i += 1) sum += q_vec[i] * kv[i];
    return sum;
}

fn dotF16(q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const kv: [*]const u16 = @ptrCast(@alignCast(kv_data));
    const V8 = @Vector(8, f32);
    var acc: V8 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const qv: V8 = q_vec[i..][0..8].*;
        // Convert 8 f16 values to f32
        var kv_v: V8 = undefined;
        inline for (0..8) |j| {
            kv_v[j] = @as(f32, @floatCast(@as(f16, @bitCast(kv[i + j]))));
        }
        acc = @mulAdd(V8, qv, kv_v, acc);
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < n) : (i += 1) {
        sum += q_vec[i] * @as(f32, @floatCast(@as(f16, @bitCast(kv[i]))));
    }
    return sum;
}

fn dotQ8_0(q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const V8 = @Vector(8, f32);
    const nb = (n + block_size - 1) / block_size;
    var sum: f32 = 0;
    for (0..nb) |b| {
        const bp = kv_data + b * q8_0_bytes;
        const scale: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp)).*)));
        const base = b * block_size;
        const count = @min(block_size, n - base);
        var acc: V8 = @splat(0.0);
        var i: usize = 0;
        while (i + 8 <= count) : (i += 8) {
            const qv: V8 = q_vec[base + i ..][0..8].*;
            var val_v: V8 = undefined;
            inline for (0..8) |j| {
                val_v[j] = @floatFromInt(@as(i8, @bitCast(bp[2 + i + j])));
            }
            acc = @mulAdd(V8, qv, val_v, acc);
        }
        var block_sum: f32 = @reduce(.Add, acc);
        while (i < count) : (i += 1) {
            const val: f32 = @floatFromInt(@as(i8, @bitCast(bp[2 + i])));
            block_sum += q_vec[base + i] * val;
        }
        sum += scale * block_sum;
    }
    return sum;
}

fn dotInt8(q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const V8 = @Vector(8, f32);
    const nb = (n + block_size - 1) / block_size;
    var sum: f32 = 0;
    for (0..nb) |b| {
        const bp = kv_data + b * int8_bytes;
        const scale: f32 = @as(*align(1) const f32, @ptrCast(bp)).*;
        const base = b * block_size;
        const count = @min(block_size, n - base);
        var acc: V8 = @splat(0.0);
        var i: usize = 0;
        while (i + 8 <= count) : (i += 8) {
            const qv: V8 = q_vec[base + i ..][0..8].*;
            var val_v: V8 = undefined;
            inline for (0..8) |j| {
                val_v[j] = @floatFromInt(@as(i8, @bitCast(bp[4 + i + j])));
            }
            acc = @mulAdd(V8, qv, val_v, acc);
        }
        var block_sum: f32 = @reduce(.Add, acc);
        while (i < count) : (i += 1) {
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
        .turbo2 => turboMulAccum(2, acc, weight, kv_data, n),
        .turbo3 => turboMulAccum(3, acc, weight, kv_data, n),
        .turbo4 => turboMulAccum(4, acc, weight, kv_data, n),
    }
}

fn mulAccF32(acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const kv: [*]const f32 = @ptrCast(@alignCast(kv_data));
    const V8 = @Vector(8, f32);
    const wv: V8 = @splat(weight);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const cur: V8 = acc[i..][0..8].*;
        const kv_v: V8 = kv[i..][0..8].*;
        acc[i..][0..8].* = @mulAdd(V8, wv, kv_v, cur);
    }
    while (i < n) : (i += 1) acc[i] = @mulAdd(f32, weight, kv[i], acc[i]);
}

fn mulAccF16(acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const kv: [*]const u16 = @ptrCast(@alignCast(kv_data));
    const V8 = @Vector(8, f32);
    const wv: V8 = @splat(weight);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const cur: V8 = acc[i..][0..8].*;
        var kv_v: V8 = undefined;
        inline for (0..8) |j| {
            kv_v[j] = @as(f32, @floatCast(@as(f16, @bitCast(kv[i + j]))));
        }
        acc[i..][0..8].* = @mulAdd(V8, wv, kv_v, cur);
    }
    while (i < n) : (i += 1) {
        acc[i] = @mulAdd(f32, weight, @as(f32, @floatCast(@as(f16, @bitCast(kv[i])))), acc[i]);
    }
}

fn mulAccQ8_0(acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const V8 = @Vector(8, f32);
    const nb = (n + block_size - 1) / block_size;
    for (0..nb) |b| {
        const bp = kv_data + b * q8_0_bytes;
        const scale: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp)).*)));
        const ws_v: V8 = @splat(weight * scale);
        const base = b * block_size;
        const count = @min(block_size, n - base);
        var i: usize = 0;
        while (i + 8 <= count) : (i += 8) {
            const cur: V8 = acc[base + i ..][0..8].*;
            var val_v: V8 = undefined;
            inline for (0..8) |j| {
                val_v[j] = @floatFromInt(@as(i8, @bitCast(bp[2 + i + j])));
            }
            acc[base + i ..][0..8].* = @mulAdd(V8, ws_v, val_v, cur);
        }
        const ws = weight * scale;
        while (i < count) : (i += 1) {
            acc[base + i] = @mulAdd(f32, ws, @as(f32, @floatFromInt(@as(i8, @bitCast(bp[2 + i])))), acc[base + i]);
        }
    }
}

fn mulAccInt8(acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const V8 = @Vector(8, f32);
    const nb = (n + block_size - 1) / block_size;
    for (0..nb) |b| {
        const bp = kv_data + b * int8_bytes;
        const scale: f32 = @as(*align(1) const f32, @ptrCast(bp)).*;
        const ws_v: V8 = @splat(weight * scale);
        const base = b * block_size;
        const count = @min(block_size, n - base);
        var i: usize = 0;
        while (i + 8 <= count) : (i += 8) {
            const cur: V8 = acc[base + i ..][0..8].*;
            var val_v: V8 = undefined;
            inline for (0..8) |j| {
                val_v[j] = @floatFromInt(@as(i8, @bitCast(bp[4 + i + j])));
            }
            acc[base + i ..][0..8].* = @mulAdd(V8, ws_v, val_v, cur);
        }
        const ws = weight * scale;
        while (i < count) : (i += 1) {
            acc[base + i] = @mulAdd(f32, ws, @as(f32, @floatFromInt(@as(i8, @bitCast(bp[4 + i])))), acc[base + i]);
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

// ── TurboQuant store/dot/mulAccum ────────────────────────────────

/// Quantize `n` f32 values into TurboQuant format at the given bit width.
///
/// Per 32-element block:
///   1. Compute L2 norm and normalize to unit vector
///   2. Apply Walsh-Hadamard Transform (decorrelates coefficients)
///   3. Scale by 1/sqrt(32) (WHT normalization)
///   4. Quantize each coefficient to nearest Lloyd-Max centroid
///   5. Pack centroid indices and store f16 norm header
///
/// Block layout: [f16 norm (2 bytes)] [packed indices (bits*32/8 bytes)]
fn turboStore(comptime bits: u3, dst: [*]u8, src: [*]const f32, n: usize) void {
    const bb = comptime turboBlockBytes(bits);
    const codebook = comptime lloydMaxCodebook(bits);
    const nb = (n + turbo_block_size - 1) / turbo_block_size;

    for (0..nb) |blk| {
        const base = blk * turbo_block_size;
        const count = @min(turbo_block_size, n - base);

        // Load and compute L2 norm
        var buf: [32]f32 = undefined;
        var norm_sq: f32 = 0;
        for (0..count) |i| {
            buf[i] = src[base + i];
            norm_sq += buf[i] * buf[i];
        }
        // Zero-pad remainder
        for (count..32) |i| buf[i] = 0;

        const norm = @sqrt(norm_sq);
        const bp = dst + blk * bb;

        if (norm == 0) {
            // Zero vector: store zero norm and zero indices
            @as(*align(1) u16, @ptrCast(bp)).* = @bitCast(@as(f16, 0));
            @memset(bp[2 .. bb], 0);
            continue;
        }

        // Normalize to unit vector
        const inv_norm = 1.0 / norm;
        for (0..32) |i| buf[i] *= inv_norm;

        // Walsh-Hadamard Transform
        wht32(&buf);

        // Scale by 1/sqrt(32) (WHT normalization)
        for (0..32) |i| buf[i] *= wht_inv_sqrt;

        // Quantize to nearest Lloyd-Max centroid
        var indices: [32]u8 = undefined;
        for (0..32) |i| {
            indices[i] = nearestCentroid(bits, codebook, buf[i]);
        }

        // Store f16 norm header
        @as(*align(1) u16, @ptrCast(bp)).* = @bitCast(@as(f16, @floatCast(norm)));

        // Pack indices
        packIndices(bits, bp[2..bb], &indices);
    }
}

/// Find the nearest centroid index for a given value using binary search.
inline fn nearestCentroid(comptime bits: u3, codebook: []const f32, val: f32) u8 {
    const n_centroids = @as(usize, 1) << bits;
    // Decision boundaries are midpoints between adjacent centroids.
    // Binary search over the sorted codebook.
    var lo: usize = 0;
    var hi: usize = n_centroids;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        // Boundary between centroid[mid] and centroid[mid+1] is their midpoint
        if (mid + 1 < n_centroids) {
            const boundary = (codebook[mid] + codebook[mid + 1]) * 0.5;
            if (val > boundary) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        } else {
            // mid is the last centroid
            hi = mid;
        }
    }
    return @intCast(lo);
}

/// Pack 32 indices of `bits` width into a byte buffer (LSB-first).
inline fn packIndices(comptime bits: u3, dst: []u8, indices: *const [32]u8) void {
    @memset(dst, 0);
    if (bits == 2) {
        // 2-bit: 4 indices per byte, simple
        for (0..32) |i| {
            const byte_idx = i / 4;
            const bit_off: u3 = @intCast((i % 4) * 2);
            dst[byte_idx] |= @as(u8, indices[i] & 0x03) << bit_off;
        }
    } else if (bits == 4) {
        // 4-bit: 2 indices per byte, simple
        for (0..32) |i| {
            const byte_idx = i / 2;
            const bit_off: u3 = @intCast((i % 2) * 4);
            dst[byte_idx] |= @as(u8, indices[i] & 0x0F) << bit_off;
        }
    } else {
        // 3-bit: indices span byte boundaries, general bit-packing
        for (0..32) |i| {
            const bit_pos = i * bits;
            const byte_idx = bit_pos / 8;
            const bit_off: u3 = @intCast(bit_pos % 8);
            const mask = (@as(u8, 1) << bits) - 1;
            dst[byte_idx] |= @as(u8, indices[i] & mask) << bit_off;
            // Handle spanning into next byte
            if (bit_off + bits > 8) {
                const overflow: u3 = @intCast(bit_off + bits - 8);
                dst[byte_idx + 1] |= @as(u8, indices[i] & mask) >> @intCast(bits - overflow);
            }
        }
    }
}

/// Unpack 32 indices of `bits` width from a byte buffer (LSB-first).
inline fn unpackIndices(comptime bits: u3, src: []const u8, indices: *[32]u8) void {
    if (bits == 2) {
        for (0..32) |i| {
            const byte_idx = i / 4;
            const bit_off: u3 = @intCast((i % 4) * 2);
            indices[i] = (src[byte_idx] >> bit_off) & 0x03;
        }
    } else if (bits == 4) {
        for (0..32) |i| {
            const byte_idx = i / 2;
            const bit_off: u3 = @intCast((i % 2) * 4);
            indices[i] = (src[byte_idx] >> bit_off) & 0x0F;
        }
    } else {
        const mask: u8 = (@as(u8, 1) << bits) - 1;
        for (0..32) |i| {
            const bit_pos = i * bits;
            const byte_idx = bit_pos / 8;
            const bit_off: u3 = @intCast(bit_pos % 8);
            var val = src[byte_idx] >> bit_off;
            if (bit_off + bits > 8) {
                // Spans into next byte
                val |= src[byte_idx + 1] << @intCast(8 - bit_off);
            }
            indices[i] = val & mask;
        }
    }
}

/// Optimized asymmetric dot product: query · dequant(turbo_data).
///
/// Instead of inverse-WHT on the cached data, we forward-WHT the query block:
///   dot(q, dequant(data)) = norm / sqrt(32) * dot(WHT(q_block), codebook_values)
///
/// This avoids materializing the full dequantized vector.
fn turboDot(comptime bits: u3, q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const bb = comptime turboBlockBytes(bits);
    const codebook = comptime lloydMaxCodebook(bits);
    const nb = (n + turbo_block_size - 1) / turbo_block_size;
    const data_bytes = comptime bb - 2; // packed index bytes per block

    var sum: f32 = 0;

    for (0..nb) |blk| {
        const base = blk * turbo_block_size;
        const bp = kv_data + blk * bb;

        // Read f16 norm
        const norm: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp)).*)));
        if (norm == 0) continue;

        // Copy and WHT the query block
        var q_buf: [32]f32 = undefined;
        const count = @min(turbo_block_size, n - base);
        for (0..count) |i| q_buf[i] = q_vec[base + i];
        for (count..32) |i| q_buf[i] = 0;
        wht32(&q_buf);

        // Unpack indices
        var indices: [32]u8 = undefined;
        unpackIndices(bits, bp[2..][0..data_bytes], &indices);

        // Dot product in WHT domain: sum(q_wht[i] * codebook[idx[i]])
        var block_dot: f32 = 0;
        for (0..32) |i| {
            block_dot += q_buf[i] * codebook[indices[i]];
        }

        // Scale: norm / sqrt(32) accounts for WHT normalization in both store and inverse
        sum += norm * block_dot * wht_inv_sqrt;
    }

    return sum;
}

/// Full dequant accumulate: acc[0..n] += weight * dequant(turbo_data).
///
/// Per block: unpack → codebook lookup → inverse WHT → rescale by weight * norm / sqrt(32).
fn turboMulAccum(comptime bits: u3, acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const bb = comptime turboBlockBytes(bits);
    const codebook = comptime lloydMaxCodebook(bits);
    const nb = (n + turbo_block_size - 1) / turbo_block_size;
    const data_bytes = comptime bb - 2;

    for (0..nb) |blk| {
        const base = blk * turbo_block_size;
        const bp = kv_data + blk * bb;

        // Read f16 norm
        const norm: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp)).*)));
        if (norm == 0) continue;

        // Unpack indices and look up codebook values
        var indices: [32]u8 = undefined;
        unpackIndices(bits, bp[2..][0..data_bytes], &indices);

        var buf: [32]f32 = undefined;
        for (0..32) |i| {
            buf[i] = codebook[indices[i]];
        }

        // Inverse WHT: forward WHT followed by 1/sqrt(32) gives the orthonormal inverse,
        // matching the 1/sqrt(32) normalization applied during store.
        wht32(&buf);

        // Accumulate: rescale by weight * norm / sqrt(32) (orthonormal WHT inverse + denormalization)
        const scale = weight * norm * wht_inv_sqrt;
        const count = @min(turbo_block_size, n - base);
        for (0..count) |i| {
            acc[base + i] += buf[i] * scale;
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

/// Convert f32 to E2M1 4-bit nibble (clamps to [-6, 6] via threshold matching).
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
        const dot_result = kvDot(&dot_v, buf[i * 2 ..].ptr, 1, .f16);
        try std.testing.expectApproxEqAbs(expected, dot_result, 1e-6);
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
    try std.testing.expectApproxEqAbs(src[0], dot, 0.01);

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
    try std.testing.expectApproxEqAbs(expected, dot, 0.05);
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

// ── TurboQuant Tests ─────────────────────────────────────────────

test "wht32 self-inverse" {
    // WHT is self-inverse up to a scale factor of 32:
    // WHT(WHT(x)) = 32 * x
    var buf: [32]f32 = undefined;
    const original: [32]f32 = blk: {
        var arr: [32]f32 = undefined;
        for (0..32) |i| {
            arr[i] = @as(f32, @floatFromInt(i)) * 0.1 - 1.5;
        }
        break :blk arr;
    };
    buf = original;

    // Forward WHT
    wht32(&buf);
    // Second forward WHT = inverse (up to factor of 32)
    wht32(&buf);
    // Divide by 32 to recover original
    for (0..32) |i| buf[i] /= 32.0;

    for (0..32) |i| {
        try std.testing.expectApproxEqAbs(original[i], buf[i], 1e-5);
    }
}

test "turbo4 roundtrip accuracy" {
    // 32-element vector with varied values
    var src: [32]f32 = undefined;
    for (0..32) |i| {
        src[i] = @sin(@as(f32, @floatFromInt(i)) * 0.5) * 2.0;
    }

    var buf: [turbo4_block_bytes]u8 = undefined;
    kvStore(&buf, &src, 32, .turbo4);

    // Dot with all-ones should approximate sum(src)
    var q_ones = [_]f32{1.0} ** 32;
    const dot_sum = kvDot(&q_ones, &buf, 32, .turbo4);
    var expected_sum: f32 = 0;
    for (src) |v| expected_sum += v;
    try std.testing.expectApproxEqAbs(expected_sum, dot_sum, 1.0);

    // MulAccum MSE should be < 0.1
    var acc = [_]f32{0} ** 32;
    kvMulAccum(&acc, 1.0, &buf, 32, .turbo4);
    var mse: f32 = 0;
    for (0..32) |i| {
        const err = acc[i] - src[i];
        mse += err * err;
    }
    mse /= 32.0;
    try std.testing.expect(mse < 0.1);
}

test "turbo3 roundtrip accuracy" {
    var src: [32]f32 = undefined;
    for (0..32) |i| {
        src[i] = @sin(@as(f32, @floatFromInt(i)) * 0.5) * 2.0;
    }

    var buf: [turbo3_block_bytes]u8 = undefined;
    kvStore(&buf, &src, 32, .turbo3);

    // MulAccum MSE should be < 0.2
    var acc = [_]f32{0} ** 32;
    kvMulAccum(&acc, 1.0, &buf, 32, .turbo3);
    var mse: f32 = 0;
    for (0..32) |i| {
        const err = acc[i] - src[i];
        mse += err * err;
    }
    mse /= 32.0;
    try std.testing.expect(mse < 0.2);
}

test "turbo2 roundtrip accuracy" {
    var src: [32]f32 = undefined;
    for (0..32) |i| {
        src[i] = @sin(@as(f32, @floatFromInt(i)) * 0.5) * 2.0;
    }

    var buf: [turbo2_block_bytes]u8 = undefined;
    kvStore(&buf, &src, 32, .turbo2);

    // MulAccum MSE should be < 0.5
    var acc = [_]f32{0} ** 32;
    kvMulAccum(&acc, 1.0, &buf, 32, .turbo2);
    var mse: f32 = 0;
    for (0..32) |i| {
        const err = acc[i] - src[i];
        mse += err * err;
    }
    mse /= 32.0;
    try std.testing.expect(mse < 0.5);
}

test "turboDot matches naive dequant-then-dot" {
    // Store a vector, then verify that turboDot gives the same result
    // as manually dequanting via mulAccum and dotting.
    var src: [32]f32 = undefined;
    for (0..32) |i| {
        src[i] = @cos(@as(f32, @floatFromInt(i)) * 0.3) * 1.5;
    }

    var q_vec: [32]f32 = undefined;
    for (0..32) |i| {
        q_vec[i] = @sin(@as(f32, @floatFromInt(i)) * 0.7) * 0.8;
    }

    var buf: [turbo4_block_bytes]u8 = undefined;
    kvStore(&buf, &src, 32, .turbo4);

    // Optimized dot
    const dot_opt = kvDot(&q_vec, &buf, 32, .turbo4);

    // Naive: dequant then dot
    var dequant = [_]f32{0} ** 32;
    kvMulAccum(&dequant, 1.0, &buf, 32, .turbo4);
    var dot_naive: f32 = 0;
    for (0..32) |i| dot_naive += q_vec[i] * dequant[i];

    try std.testing.expectApproxEqAbs(dot_naive, dot_opt, 1e-4);
}

test "turbo kvSliceBytes" {
    // turbo2: 10 bytes per 32 elements
    try std.testing.expectEqual(@as(usize, 10), kvSliceBytes(.turbo2, 32));
    try std.testing.expectEqual(@as(usize, 20), kvSliceBytes(.turbo2, 64));
    try std.testing.expectEqual(@as(usize, 10), kvSliceBytes(.turbo2, 1)); // rounds up to 1 block
    // turbo3: 14 bytes per 32 elements
    try std.testing.expectEqual(@as(usize, 14), kvSliceBytes(.turbo3, 32));
    try std.testing.expectEqual(@as(usize, 28), kvSliceBytes(.turbo3, 64));
    // turbo4: 18 bytes per 32 elements
    try std.testing.expectEqual(@as(usize, 18), kvSliceBytes(.turbo4, 32));
    try std.testing.expectEqual(@as(usize, 36), kvSliceBytes(.turbo4, 64));
}

test "turbo fromString" {
    try std.testing.expectEqual(KvQuantType.turbo2, KvQuantType.fromString("turbo2").?);
    try std.testing.expectEqual(KvQuantType.turbo2, KvQuantType.fromString("tq2").?);
    try std.testing.expectEqual(KvQuantType.turbo2, KvQuantType.fromString("TQ2").?);
    try std.testing.expectEqual(KvQuantType.turbo3, KvQuantType.fromString("turbo3").?);
    try std.testing.expectEqual(KvQuantType.turbo3, KvQuantType.fromString("TQ3").?);
    try std.testing.expectEqual(KvQuantType.turbo4, KvQuantType.fromString("turbo4").?);
    try std.testing.expectEqual(KvQuantType.turbo4, KvQuantType.fromString("TURBO4").?);
}

test "turbo zero vector" {
    // Zero input should produce zero output
    const src = [_]f32{0} ** 32;

    // Test all bit widths
    inline for ([_]KvQuantType{ .turbo2, .turbo3, .turbo4 }) |kv_type| {
        const bb = kvSliceBytes(kv_type, 32);
        var buf: [18]u8 = undefined; // 18 is max (turbo4)
        kvStore(&buf, &src, 32, kv_type);

        // Dot with any vector should be 0
        var q_vec: [32]f32 = undefined;
        for (0..32) |i| q_vec[i] = @as(f32, @floatFromInt(i)) + 1.0;
        const dot = kvDot(&q_vec, &buf, 32, kv_type);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), dot, 1e-6);

        // MulAccum should leave accumulator unchanged
        var acc = [_]f32{1.0} ** 32;
        kvMulAccum(&acc, 2.5, buf[0..bb].ptr, 32, kv_type);
        for (0..32) |i| {
            try std.testing.expectApproxEqAbs(@as(f32, 1.0), acc[i], 1e-6);
        }
    }
}

test "pack/unpack indices roundtrip" {
    // Verify bit packing is lossless for all bit widths
    var indices: [32]u8 = undefined;

    // 2-bit: values 0-3
    for (0..32) |i| indices[i] = @intCast(i % 4);
    var buf2: [8]u8 = undefined;
    packIndices(2, &buf2, &indices);
    var out2: [32]u8 = undefined;
    unpackIndices(2, &buf2, &out2);
    for (0..32) |i| try std.testing.expectEqual(indices[i], out2[i]);

    // 3-bit: values 0-7
    for (0..32) |i| indices[i] = @intCast(i % 8);
    var buf3: [12]u8 = undefined;
    packIndices(3, &buf3, &indices);
    var out3: [32]u8 = undefined;
    unpackIndices(3, &buf3, &out3);
    for (0..32) |i| try std.testing.expectEqual(indices[i], out3[i]);

    // 4-bit: values 0-15
    for (0..32) |i| indices[i] = @intCast(i % 16);
    var buf4: [16]u8 = undefined;
    packIndices(4, &buf4, &indices);
    var out4: [32]u8 = undefined;
    unpackIndices(4, &buf4, &out4);
    for (0..32) |i| try std.testing.expectEqual(indices[i], out4[i]);
}
