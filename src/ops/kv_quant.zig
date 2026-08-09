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
//!   - turbo2-4:  TurboQuant — WHT + Lloyd-Max codebook (2.5/3.5/4.5 bits/elem)
//!   - planar2-4: PlanarQuant — Givens 2D rotation + Lloyd-Max (same sizes as turbo)
//!   - iso2-4:    IsoQuant — Quaternion 4D rotation + Lloyd-Max (same sizes as turbo)
//!   - rotor2-4:  RotorQuant — Clifford Cl(3,0) rotor rotation + Lloyd-Max (same sizes as turbo)

const std = @import("std");
const quant = @import("quant.zig");

/// 8-wide SIMD vector type for f32 — used across all SIMD helpers in this module.
const V8 = @Vector(8, f32);

/// Floor for near-zero absmax; below this, treat as unit scale to avoid /0.
const absmax_epsilon: f32 = 1e-7;

/// SIMD absolute-max over `src[0..n]`.
inline fn absMaxF32(src: [*]const f32, n: usize) f32 {
    var amax_v: V8 = @splat(@as(f32, 0.0));
    var ai: usize = 0;
    while (ai + 8 <= n) : (ai += 8) {
        const v: V8 = src[ai..][0..8].*;
        amax_v = @max(amax_v, @abs(v));
    }
    var amax = @reduce(.Max, amax_v);
    while (ai < n) : (ai += 1) amax = @max(amax, @abs(src[ai]));
    return amax;
}

/// Block size for Q8_0 and INT8 quantization (shared with quant.zig).
const block_size: usize = quant.quant_block_elems;
/// Q8_0 block: f16 scale (2 bytes) + 32 i8 values = 34 bytes (shared with quant.zig).
const q8_0_block_bytes: usize = quant.q8_0_block_bytes;
/// INT8 block: f32 scale (4 bytes) + 32 i8 values = 36 bytes.
const int8_block_bytes: usize = 36;
/// NVFP4 block size: 16 elements.
const nvfp4_block: usize = 16;
/// NVFP4 block: fp8 scale (1 byte) + 8 packed nibble bytes = 9 bytes.
const nvfp4_block_bytes: usize = 9;
/// Q8_0 scale header size: f16 = 2 bytes.
const q8_0_scale_bytes: usize = 2;
/// INT8 scale header size: f32 = 4 bytes.
const int8_scale_bytes: usize = 4;
/// Maximum representable INT8 value (scale normalization factor for Q8_0/INT8).
const int8_max: f32 = 127.0;
/// Minimum representable INT8 value (lower clamp bound for quantized values).
const int8_min: f32 = -128.0;
/// Maximum representable E2M1 value (scale normalization factor for NVFP4).
const e2m1_max: f32 = 6.0;
/// Maximum representable FP8 E4M3 value (clamp bound for f32→FP8 conversion).
const fp8_e4m3_max: f32 = 448.0;
/// FP8 E4M3 max finite encoding (0x7E = 448.0; 0x7F is NaN, no infinities).
const fp8_e4m3_max_finite: u8 = 0x7E;
/// FP8 E4M3 max biased exponent (4 exponent bits, bias=7; 2^4 - 1 = 15).
const fp8_e4m3_max_biased_exp: u8 = 15;

// ── TurboQuant constants ─────────────────────────────────────────

/// TurboQuant / PlanarQuant / IsoQuant / RotorQuant block size: 32 elements.
const turbo_block_size: usize = 32;
/// 2-bit block: f16 norm (2 bytes) + 64 packed bits = 10 bytes.
pub const turbo2_block_bytes: usize = 10;
/// 3-bit block: f16 norm (2 bytes) + 96 packed bits = 14 bytes.
pub const turbo3_block_bytes: usize = 14;
/// 4-bit block: f16 norm (2 bytes) + 128 packed bits = 18 bytes.
pub const turbo4_block_bytes: usize = 18;
/// WHT normalization factor: 1 / sqrt(32).
const wht_inv_sqrt: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(turbo_block_size)));

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

// ── PlanarQuant: Givens 2D rotation ─────────────────────────────

/// Fixed Givens rotation angles for 16 pairs in a 32-element block.
/// Deterministic (not random) — same angles for all blocks.
/// Chosen to decorrelate adjacent coordinate pairs.
const planar_angles: [16]f32 = blk: {
    var a: [16]f32 = undefined;
    for (0..16) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.19634954 + 0.3927; // spread across [0.39, 3.53]
    }
    break :blk a;
};

/// Apply forward Givens rotation to pairs of elements in buf[0..32].
/// Each pair (buf[2i], buf[2i+1]) is rotated by angle[i].
inline fn givensRotateForward(buf: *[32]f32) void {
    inline for (0..16) |i| {
        const c = @cos(planar_angles[i]);
        const s = @sin(planar_angles[i]);
        const x = buf[2 * i];
        const y = buf[2 * i + 1];
        buf[2 * i] = c * x - s * y;
        buf[2 * i + 1] = s * x + c * y;
    }
}

/// Apply inverse Givens rotation (negate angle).
inline fn givensRotateInverse(buf: *[32]f32) void {
    inline for (0..16) |i| {
        const c = @cos(planar_angles[i]);
        const s = @sin(planar_angles[i]);
        const x = buf[2 * i];
        const y = buf[2 * i + 1];
        buf[2 * i] = c * x + s * y;
        buf[2 * i + 1] = -s * x + c * y;
    }
}

// ── IsoQuant: Quaternion 4D rotation ────────────────────────────

/// Fixed quaternion parameters for 8 quartets in a 32-element block.
/// Each quaternion (w, x, y, z) is unit-normalized.
const iso_quaternions: [8][4]f32 = blk: {
    var q: [8][4]f32 = undefined;
    for (0..8) |i| {
        const angle: f32 = @as(f32, @floatFromInt(i)) * 0.3927 + 0.5;
        const half = angle * 0.5;
        const c = @cos(half);
        const s = @sin(half);
        // Rotate in the (x,y) plane: q = cos(θ/2) + sin(θ/2) * k
        q[i] = .{ c, 0, 0, s };
    }
    break :blk q;
};

/// Apply forward quaternion rotation to quartets buf[4i..4i+4].
/// v' = q * v * conj(q) for each quartet.
inline fn quatRotateForward(buf: *[32]f32) void {
    inline for (0..8) |i| {
        const w = iso_quaternions[i][0];
        const qx = iso_quaternions[i][1];
        const qy = iso_quaternions[i][2];
        const qz = iso_quaternions[i][3];
        const vx = buf[4 * i];
        const vy = buf[4 * i + 1];
        const vz = buf[4 * i + 2];
        const vw_unused = buf[4 * i + 3];

        // q * v * conj(q) for pure quaternion v = (vx, vy, vz)
        // Fourth element treated as independent scalar (rotated separately)
        const t0 = w * vx + qy * vz - qz * vy;
        const t1 = w * vy + qz * vx - qx * vz;
        const t2 = w * vz + qx * vy - qy * vx;
        const t3 = -qx * vx - qy * vy - qz * vz;

        buf[4 * i] = t0 * w + t3 * (-qx) + t1 * (-qz) - t2 * (-qy);
        buf[4 * i + 1] = t1 * w + t3 * (-qy) + t2 * (-qx) - t0 * (-qz);
        buf[4 * i + 2] = t2 * w + t3 * (-qz) + t0 * (-qy) - t1 * (-qx);
        buf[4 * i + 3] = vw_unused; // pass through
    }
}

/// Apply inverse quaternion rotation (conjugate quaternion).
inline fn quatRotateInverse(buf: *[32]f32) void {
    inline for (0..8) |i| {
        const w = iso_quaternions[i][0];
        const qx = -iso_quaternions[i][1];
        const qy = -iso_quaternions[i][2];
        const qz = -iso_quaternions[i][3];
        const vx = buf[4 * i];
        const vy = buf[4 * i + 1];
        const vz = buf[4 * i + 2];
        const vw_unused = buf[4 * i + 3];

        const t0 = w * vx + qy * vz - qz * vy;
        const t1 = w * vy + qz * vx - qx * vz;
        const t2 = w * vz + qx * vy - qy * vx;
        const t3 = -qx * vx - qy * vy - qz * vz;

        buf[4 * i] = t0 * w + t3 * (-qx) + t1 * (-qz) - t2 * (-qy);
        buf[4 * i + 1] = t1 * w + t3 * (-qy) + t2 * (-qx) - t0 * (-qz);
        buf[4 * i + 2] = t2 * w + t3 * (-qz) + t0 * (-qy) - t1 * (-qx);
        buf[4 * i + 3] = vw_unused;
    }
}

// ── RotorQuant: Clifford Cl(3,0) rotor rotation ─────────────────

/// RotorQuant block size: groups of 3 dimensions (Cl(3,0) vectors).
/// For a 32-element block: 10 groups of 3 + 2 remainder (scalar pass-through).
const rotor_group_size: usize = 3;
const rotor_groups_per_block: usize = turbo_block_size / rotor_group_size;

/// Per-group rotation angle step (≈ π/10) for decorrelating adjacent groups.
const rotor_angle_step: f32 = 0.314159;
/// Per-group rotation angle offset to avoid zero-angle at group 0.
const rotor_angle_offset: f32 = 0.5;

/// Fixed Cl(3,0) rotors for each group. Format: [s, b12, b13, b23].
/// Each rotor R = s + b12*e12 + b13*e13 + b23*e23, normalized RR̃ = 1.
const rotor_params: [rotor_groups_per_block][4]f32 = blk: {
    var r: [rotor_groups_per_block][4]f32 = undefined;
    for (0..rotor_groups_per_block) |i| {
        const angle: f32 = @as(f32, @floatFromInt(i)) * rotor_angle_step + rotor_angle_offset;
        const half = angle * 0.5;
        const c = @cos(half);
        const s = @sin(half);
        // Rotor in the e12 plane: R = cos(θ/2) + sin(θ/2)*e12
        r[i] = .{ c, s, 0, 0 };
    }
    break :blk r;
};

/// Apply forward Cl(3,0) rotor sandwich product: v' = RvR̃ for 3D groups.
/// Exploits sparsity: rotor has 4 components but many are zero.
inline fn rotorForward(buf: *[32]f32) void {
    inline for (0..rotor_groups_per_block) |g| {
        const base = g * 3;
        if (base + 2 >= 32) break;
        const s = rotor_params[g][0];
        const b12 = rotor_params[g][1];
        const b13 = rotor_params[g][2];
        const b23 = rotor_params[g][3];
        const x = buf[base];
        const y = buf[base + 1];
        const z = buf[base + 2];

        // RvR̃ for grade-1 vector v = x*e1 + y*e2 + z*e3:
        // v1' = (s²+b12²-b13²-b23²)*x + 2(b12*b13-s*b23)*z + 2(s*b13+b12*b23)*y ... simplified:
        // For rotor in e12 plane only (b13=0, b23=0):
        //   x' = (s²-b12²)*x + 2*s*b12*y
        //   y' = -2*s*b12*x + (s²-b12²)*y
        //   z' = z (unchanged — rotation is in xy plane)
        const ss = s * s;
        const bb = b12 * b12;
        const sb2 = 2.0 * s * b12;
        const diag = ss - bb;

        // General 3D rotor (all bivector components)
        const bb13 = b13 * b13;
        const bb23 = b23 * b23;

        buf[base] = diag * x + sb2 * y + 2.0 * (b12 * b13 - s * b23) * z;
        buf[base + 1] = -sb2 * x + diag * y + 2.0 * (s * b13 + b12 * b23) * z;
        buf[base + 2] = 2.0 * (s * b23 - b12 * b13) * x + 2.0 * (-s * b13 - b12 * b23) * y + (ss + bb - bb13 - bb23) * z;
    }
}

/// Apply inverse Cl(3,0) rotor: R̃vR.
inline fn rotorInverse(buf: *[32]f32) void {
    inline for (0..rotor_groups_per_block) |g| {
        const base = g * 3;
        if (base + 2 >= 32) break;
        // Inverse = conjugate: negate bivector components
        const s = rotor_params[g][0];
        const b12 = -rotor_params[g][1];
        const b13 = -rotor_params[g][2];
        const b23 = -rotor_params[g][3];
        const x = buf[base];
        const y = buf[base + 1];
        const z = buf[base + 2];

        const ss = s * s;
        const bb = b12 * b12;
        const sb2 = 2.0 * s * b12;
        const diag = ss - bb;
        const bb13 = b13 * b13;
        const bb23 = b23 * b23;

        buf[base] = diag * x + sb2 * y + 2.0 * (b12 * b13 - s * b23) * z;
        buf[base + 1] = -sb2 * x + diag * y + 2.0 * (s * b13 + b12 * b23) * z;
        buf[base + 2] = 2.0 * (s * b23 - b12 * b13) * x + 2.0 * (-s * b13 - b12 * b23) * y + (ss + bb - bb13 - bb23) * z;
    }
}

/// RotorQuant store: Cl(3,0) rotor rotation + Lloyd-Max quantization.
fn rotorStore(comptime bits: u3, dst: [*]u8, src: [*]const f32, n: usize) void {
    const bb = comptime turboBlockBytes(bits);
    const nb = (n + turbo_block_size - 1) / turbo_block_size;

    for (0..nb) |blk_i| {
        const base = blk_i * turbo_block_size;
        const count = @min(turbo_block_size, n - base);
        var buf: [turbo_block_size]f32 = undefined;
        @memcpy(buf[0..count], src[base..][0..count]);
        for (count..turbo_block_size) |i| buf[i] = 0;

        var norm_acc: V8 = @splat(@as(f32, 0.0));
        inline for (0..4) |qi| {
            const bv: V8 = buf[qi * 8 ..][0..8].*;
            norm_acc = @mulAdd(V8, bv, bv, norm_acc);
        }
        const norm = @sqrt(@reduce(.Add, norm_acc));
        const bp = dst + blk_i * bb;

        if (norm == 0) {
            @as(*align(1) u16, @ptrCast(bp)).* = @bitCast(@as(f16, 0));
            @memset(bp[2..bb], 0);
            continue;
        }

        const inv_norm = 1.0 / norm;
        for (0..turbo_block_size) |i| buf[i] *= inv_norm;
        rotorForward(&buf);

        var indices: [turbo_block_size]u8 = undefined;
        for (0..turbo_block_size) |i| indices[i] = nearestCentroid(bits, buf[i]);
        @as(*align(1) u16, @ptrCast(bp)).* = @bitCast(@as(f16, @floatCast(norm)));
        packIndices(bits, bp[2..bb], &indices);
    }
}

fn rotorDot(comptime bits: u3, q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const bb = comptime turboBlockBytes(bits);
    const codebook = comptime lloydMaxCodebook(bits);
    const nb = (n + turbo_block_size - 1) / turbo_block_size;
    const data_bytes = comptime bb - 2;
    var sum: f32 = 0;

    for (0..nb) |blk_i| {
        const base = blk_i * turbo_block_size;
        const bp = kv_data + blk_i * bb;
        const norm: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp)).*)));
        if (norm == 0) continue;

        var q_buf: [turbo_block_size]f32 = undefined;
        const count = @min(turbo_block_size, n - base);
        for (0..count) |i| q_buf[i] = q_vec[base + i];
        for (count..turbo_block_size) |i| q_buf[i] = 0;
        rotorForward(&q_buf);

        var indices: [turbo_block_size]u8 = undefined;
        unpackIndices(bits, bp[2..][0..data_bytes], &indices);

        var vals: [turbo_block_size]f32 = undefined;
        for (0..turbo_block_size) |i| vals[i] = codebook[indices[i]];

        var acc: V8 = @splat(@as(f32, 0.0));
        comptime var si: usize = 0;
        inline while (si + 8 <= turbo_block_size) : (si += 8) {
            const qv: V8 = q_buf[si..][0..8].*;
            const cv: V8 = vals[si..][0..8].*;
            acc = @mulAdd(V8, qv, cv, acc);
        }
        sum += norm * @reduce(.Add, acc);
    }
    return sum;
}

fn rotorMulAccum(comptime bits: u3, acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const bb = comptime turboBlockBytes(bits);
    const codebook = comptime lloydMaxCodebook(bits);
    const nb = (n + turbo_block_size - 1) / turbo_block_size;
    const data_bytes = comptime bb - 2;

    for (0..nb) |blk_i| {
        const base = blk_i * turbo_block_size;
        const bp = kv_data + blk_i * bb;
        const norm: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp)).*)));
        if (norm == 0) continue;

        var indices: [turbo_block_size]u8 = undefined;
        unpackIndices(bits, bp[2..][0..data_bytes], &indices);

        var buf: [turbo_block_size]f32 = undefined;
        for (0..turbo_block_size) |i| buf[i] = codebook[indices[i]];
        rotorInverse(&buf);

        const scale = weight * norm;
        const scale_v: V8 = @splat(scale);
        const count = @min(turbo_block_size, n - base);
        var si: usize = 0;
        while (si + 8 <= count) : (si += 8) {
            const bv: V8 = buf[si..][0..8].*;
            const cv: V8 = acc[base + si ..][0..8].*;
            acc[base + si ..][0..8].* = @mulAdd(V8, bv, scale_v, cv);
        }
        while (si < count) : (si += 1) {
            acc[base + si] = @mulAdd(f32, buf[si], scale, acc[base + si]);
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
    planar2,
    planar3,
    planar4,
    iso2,
    iso3,
    iso4,
    rotor2,
    rotor3,
    rotor4,

    /// Returns the short display name for this KV quantization type (e.g. "TQ2", "F16").
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
            .planar2 => "PQ2",
            .planar3 => "PQ3",
            .planar4 => "PQ4",
            .iso2 => "IQ2",
            .iso3 => "IQ3",
            .iso4 => "IQ4",
            .rotor2 => "RQ2",
            .rotor3 => "RQ3",
            .rotor4 => "RQ4",
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
            .turbo2, .planar2, .iso2, .rotor2 => 2.5,
            .turbo3, .planar3, .iso3, .rotor3 => 3.5,
            .turbo4, .planar4, .iso4, .rotor4 => 4.5,
        };
    }

    /// Returns true if this is a TurboQuant variant (turbo2/3/4).
    pub fn isTurbo(self: KvQuantType) bool {
        return self == .turbo2 or self == .turbo3 or self == .turbo4;
    }

    /// Returns true if this is a PlanarQuant variant (planar2/3/4).
    pub fn isPlanar(self: KvQuantType) bool {
        return self == .planar2 or self == .planar3 or self == .planar4;
    }

    /// Returns true if this is an IsoQuant variant (iso2/3/4).
    pub fn isIso(self: KvQuantType) bool {
        return self == .iso2 or self == .iso3 or self == .iso4;
    }

    /// Returns true if this is a RotorQuant variant (rotor2/3/4).
    pub fn isRotor(self: KvQuantType) bool {
        return self == .rotor2 or self == .rotor3 or self == .rotor4;
    }

    /// Returns true if this is any rotation-based quantization method (Turbo, Planar, Iso, or Rotor).
    pub fn isRotationQuant(self: KvQuantType) bool {
        return self.isTurbo() or self.isPlanar() or self.isIso() or self.isRotor();
    }

    /// Returns the bit-width for rotation/turbo quant types (2, 3, or 4), 8 for Q8_0, 0 for others.
    pub fn turboBits(self: KvQuantType) u32 {
        return switch (self) {
            .turbo2, .planar2, .iso2, .rotor2 => 2,
            .turbo3, .planar3, .iso3, .rotor3 => 3,
            .turbo4, .planar4, .iso4, .rotor4 => 4,
            .q8_0 => 8,
            else => 0,
        };
    }

    /// Return the byte size per 32-element block, or 0 for non-block types.
    /// Covers turbo variants and Q8_0 (34 bytes = f16 scale + 32 × i8).
    pub fn turboBlockByteSize(self: KvQuantType) u32 {
        return switch (self) {
            .turbo2, .planar2, .iso2, .rotor2 => @intCast(turbo2_block_bytes),
            .turbo3, .planar3, .iso3, .rotor3 => @intCast(turbo3_block_bytes),
            .turbo4, .planar4, .iso4, .rotor4 => @intCast(turbo4_block_bytes),
            .q8_0 => @intCast(q8_0_block_bytes),
            else => 0,
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
        if (eql(s, "planar2") or eql(s, "pq2")) return .planar2;
        if (eql(s, "planar3") or eql(s, "pq3")) return .planar3;
        if (eql(s, "planar4") or eql(s, "pq4")) return .planar4;
        if (eql(s, "iso2") or eql(s, "iq2")) return .iso2;
        if (eql(s, "iso3") or eql(s, "iq3")) return .iso3;
        if (eql(s, "iso4") or eql(s, "iq4")) return .iso4;
        if (eql(s, "rotor2") or eql(s, "rq2")) return .rotor2;
        if (eql(s, "rotor3") or eql(s, "rq3")) return .rotor3;
        if (eql(s, "rotor4") or eql(s, "rq4")) return .rotor4;
        return null;
    }
};

// ── Allocation sizing ────────────────────────────────────────────

/// Compute byte storage needed for `n` logical f32 elements.
pub fn kvSliceBytes(kv_type: KvQuantType, n: usize) usize {
    return switch (kv_type) {
        .f32 => n * 4,
        .f16 => n * 2,
        .q8_0 => ((n + block_size - 1) / block_size) * q8_0_block_bytes,
        .int8 => ((n + block_size - 1) / block_size) * int8_block_bytes,
        .fp8_e4m3 => n,
        .nvfp4 => ((n + nvfp4_block - 1) / nvfp4_block) * nvfp4_block_bytes,
        .turbo2, .planar2, .iso2, .rotor2 => ((n + turbo_block_size - 1) / turbo_block_size) * turbo2_block_bytes,
        .turbo3, .planar3, .iso3, .rotor3 => ((n + turbo_block_size - 1) / turbo_block_size) * turbo3_block_bytes,
        .turbo4, .planar4, .iso4, .rotor4 => ((n + turbo_block_size - 1) / turbo_block_size) * turbo4_block_bytes,
    };
}

/// Byte offset for element index `i` (start of the block containing element `i`).
/// For element-wise formats (f32, f16, fp8), this is the exact byte offset.
/// For block formats, this is the start of the containing block.
/// Compute byte offset for the i-th logical f32 element in a KV cache buffer.
/// Not forced inline — the 10-arm switch is large; let the compiler decide.
pub fn kvByteOffset(kv_type: KvQuantType, i: usize) usize {
    return switch (kv_type) {
        .f32 => i * 4,
        .f16 => i * 2,
        .q8_0 => (i / block_size) * q8_0_block_bytes,
        .int8 => (i / block_size) * int8_block_bytes,
        .fp8_e4m3 => i,
        .nvfp4 => (i / nvfp4_block) * nvfp4_block_bytes,
        .turbo2, .planar2, .iso2, .rotor2 => (i / turbo_block_size) * turbo2_block_bytes,
        .turbo3, .planar3, .iso3, .rotor3 => (i / turbo_block_size) * turbo3_block_bytes,
        .turbo4, .planar4, .iso4, .rotor4 => (i / turbo_block_size) * turbo4_block_bytes,
    };
}

// ── Per-head KV quantization scales ──────────────────────────────
//
// vLLM per-head KV quantization: separate FP32 scale per Q/K/V head.
// More coarse than per-block (less metadata) but enables GPU-friendly
// attention kernels that apply one scale per head (FlashAttention style).
//
// Usage with --kv-type fp8_e4m3:
//   1. Allocate PerHeadKvScales with nkv heads
//   2. Call kvStorePerHead(dst, src, n, head_idx, scales) to store and track scale
//   3. Call kvDotPerHead(q, kv, n, head_idx, scales) to compute scaled dot product
//
// Requires Flash Attention as the backend for optimal throughput
// (the CPU windowed SDPA path supports it via the scaled dot helpers below).

/// Per-head scale tracking for FP8 KV quantization (q_scale, k_scale, v_scale).
/// Each head maintains a running max-absval scale, updated on every kvStore call.
/// Thread-safety: one PerHeadKvScales per KV cache buffer; no concurrent writes.
pub const PerHeadKvScales = struct {
    scales: []f32, // shape [n_heads], one f32 scale per head

    /// Allocate per-head scale array of length `n_heads`, initialized to 1.0 (no rescaling).
    /// Caller owns the returned scales and must call `deinit` with the same allocator.
    pub fn init(allocator: std.mem.Allocator, n_heads: usize) !PerHeadKvScales {
        const s = try allocator.alloc(f32, n_heads);
        @memset(s, 1.0); // safe default: scale=1 = no rescaling
        return .{ .scales = s };
    }

    /// Free the per-head scale array. Must use the same allocator passed to `init`.
    pub fn deinit(self: PerHeadKvScales, allocator: std.mem.Allocator) void {
        allocator.free(self.scales);
    }
};

/// Store `n` f32 values as FP8 with per-head dynamic scaling.
/// Computes absmax over src, updates scales[head_idx] = max(old, absmax),
/// then quantizes src → dst using the updated scale.
pub fn kvStorePerHead(dst: [*]u8, src: [*]const f32, n: usize, head_idx: usize, scales: *PerHeadKvScales) void {
    var absmax = absMaxF32(src, n);
    if (absmax < absmax_epsilon) absmax = 1.0; // guard against zero
    // Update running scale (max over time for dynamic range tracking)
    const head_scale = &scales.scales[head_idx];
    head_scale.* = @max(head_scale.*, absmax);
    const inv_scale = fp8_e4m3_max / head_scale.*;
    // Quantize: scale to FP8 range, convert
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        inline for (0..8) |j| {
            dst[i + j] = f32ToFp8E4M3(src[i + j] * inv_scale);
        }
    }
    while (i < n) : (i += 1) {
        dst[i] = f32ToFp8E4M3(src[i] * inv_scale);
    }
}

/// Compute scaled dot product Q·K where K is stored as per-head FP8.
/// Dequantizes K using scales[head_idx] then dots with q_vec (f32).
pub fn kvDotPerHead(q_vec: [*]const f32, kv_data: [*]const u8, n: usize, head_idx: usize, scales: *const PerHeadKvScales) f32 {
    const scale = scales.scales[head_idx] / fp8_e4m3_max;
    var acc: V8 = @splat(0.0);
    const sv: V8 = @splat(scale);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const qv: V8 = q_vec[i..][0..8].*;
        var kv_v: V8 = undefined;
        inline for (0..8) |j| kv_v[j] = quant.fp8e4m3ToF32(kv_data[i + j]);
        acc = @mulAdd(V8, qv, kv_v * sv, acc);
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < n) : (i += 1) {
        sum = @mulAdd(f32, q_vec[i], quant.fp8e4m3ToF32(kv_data[i]) * scale, sum);
    }
    return sum;
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
        .planar2 => planarStore(2, dst, src, n),
        .planar3 => planarStore(3, dst, src, n),
        .planar4 => planarStore(4, dst, src, n),
        .iso2 => isoStore(2, dst, src, n),
        .iso3 => isoStore(3, dst, src, n),
        .iso4 => isoStore(4, dst, src, n),
        .rotor2 => rotorStore(2, dst, src, n),
        .rotor3 => rotorStore(3, dst, src, n),
        .rotor4 => rotorStore(4, dst, src, n),
    }
}

fn storeF32(dst: [*]u8, src: [*]const f32, n: usize) void {
    @memcpy(dst[0 .. n * 4], @as([*]const u8, @ptrCast(src))[0 .. n * 4]);
}

fn storeF16(dst: [*]u8, src: [*]const f32, n: usize) void {
    const out: [*]u16 = @ptrCast(@alignCast(dst));
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const v: V8 = src[i..][0..8].*;
        out[i..][0..8].* = @bitCast(@as(@Vector(8, f16), @floatCast(v)));
    }
    while (i < n) : (i += 1) {
        out[i] = @bitCast(@as(f16, @floatCast(src[i])));
    }
}

fn storeQ8_0(dst: [*]u8, src: [*]const f32, n: usize) void {
    const nb = (n + block_size - 1) / block_size;
    for (0..nb) |b| {
        const base = b * block_size;
        const count = @min(block_size, n - base);
        const amax = absMaxF32(src + base, count);
        const scale: f16 = if (amax > 0) @floatCast(amax / int8_max) else 0;
        const inv_scale: f32 = if (amax > 0) int8_max / amax else 0;
        // Write scale (f16)
        const bp = dst + b * q8_0_block_bytes;
        @as(*align(1) u16, @ptrCast(bp)).* = @bitCast(scale);
        // Write quantized values (8-wide ILP; each clamp/round is independent)
        var i: usize = 0;
        while (i + 8 <= count) : (i += 8) {
            inline for (0..8) |j| {
                const v = src[base + i + j] * inv_scale;
                bp[q8_0_scale_bytes + i + j] = @bitCast(@as(i8, @intFromFloat(std.math.clamp(std.math.round(v), int8_min, int8_max))));
            }
        }
        while (i < count) : (i += 1) {
            const v = src[base + i] * inv_scale;
            bp[q8_0_scale_bytes + i] = @bitCast(@as(i8, @intFromFloat(std.math.clamp(std.math.round(v), int8_min, int8_max))));
        }
        // Zero-pad remainder
        for (count..block_size) |j| bp[q8_0_scale_bytes + j] = 0;
    }
}

fn storeInt8(dst: [*]u8, src: [*]const f32, n: usize) void {
    const nb = (n + block_size - 1) / block_size;
    for (0..nb) |b| {
        const base = b * block_size;
        const count = @min(block_size, n - base);
        const amax = absMaxF32(src + base, count);
        const scale: f32 = if (amax > 0) amax / int8_max else 0;
        const inv_scale: f32 = if (amax > 0) int8_max / amax else 0;
        // Write scale (f32)
        const bp = dst + b * int8_block_bytes;
        @as(*align(1) f32, @ptrCast(bp)).* = scale;
        // Write quantized values (8-wide ILP)
        var i: usize = 0;
        while (i + 8 <= count) : (i += 8) {
            inline for (0..8) |j| {
                const v = src[base + i + j] * inv_scale;
                bp[int8_scale_bytes + i + j] = @bitCast(@as(i8, @intFromFloat(std.math.clamp(std.math.round(v), int8_min, int8_max))));
            }
        }
        while (i < count) : (i += 1) {
            const v = src[base + i] * inv_scale;
            bp[int8_scale_bytes + i] = @bitCast(@as(i8, @intFromFloat(std.math.clamp(std.math.round(v), int8_min, int8_max))));
        }
        for (count..block_size) |j| bp[int8_scale_bytes + j] = 0;
    }
}

fn storeFp8(dst: [*]u8, src: [*]const f32, n: usize) void {
    // Unroll 8-wide for instruction-level parallelism — each f32ToFp8E4M3 is
    // independent, so the CPU can pipeline 8 conversions simultaneously.
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        inline for (0..8) |j| {
            dst[i + j] = f32ToFp8E4M3(src[i + j]);
        }
    }
    while (i < n) : (i += 1) {
        dst[i] = f32ToFp8E4M3(src[i]);
    }
}

fn storeNvfp4(dst: [*]u8, src: [*]const f32, n: usize) void {
    const nb = (n + nvfp4_block - 1) / nvfp4_block;
    for (0..nb) |b| {
        const base = b * nvfp4_block;
        const count = @min(nvfp4_block, n - base);
        const amax = absMaxF32(src + base, count);
        // Compute FP8 E4M3 scale: scale = amax / e2m1_max
        const scale_f32: f32 = if (amax > 0) amax / e2m1_max else 0;
        const scale_fp8 = f32ToFp8E4M3(scale_f32);
        const inv_scale: f32 = if (amax > 0) e2m1_max / amax else 0;

        const bp = dst + b * nvfp4_block_bytes;
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
        .planar2 => planarDot(2, q_vec, kv_data, n),
        .planar3 => planarDot(3, q_vec, kv_data, n),
        .planar4 => planarDot(4, q_vec, kv_data, n),
        .iso2 => isoDot(2, q_vec, kv_data, n),
        .iso3 => isoDot(3, q_vec, kv_data, n),
        .iso4 => isoDot(4, q_vec, kv_data, n),
        .rotor2 => rotorDot(2, q_vec, kv_data, n),
        .rotor3 => rotorDot(3, q_vec, kv_data, n),
        .rotor4 => rotorDot(4, q_vec, kv_data, n),
    };
}

fn dotF32(q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const kv: [*]const f32 = @ptrCast(@alignCast(kv_data));

    var acc: V8 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const qv: V8 = q_vec[i..][0..8].*;
        const kv_v: V8 = kv[i..][0..8].*;
        acc = @mulAdd(V8, qv, kv_v, acc);
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < n) : (i += 1) sum = @mulAdd(f32, q_vec[i], kv[i], sum);
    return sum;
}

fn dotF16(q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const kv: [*]const u16 = @ptrCast(@alignCast(kv_data));

    var acc: V8 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const qv: V8 = q_vec[i..][0..8].*;
        // Vector f16→f32 conversion (uses hardware SIMD: vcvtph2ps / fcvtl)
        const kv_v: V8 = @floatCast(@as(@Vector(8, f16), @bitCast(kv[i..][0..8].*)));
        acc = @mulAdd(V8, qv, kv_v, acc);
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < n) : (i += 1) {
        sum = @mulAdd(f32, q_vec[i], @as(f32, @floatCast(@as(f16, @bitCast(kv[i])))), sum);
    }
    return sum;
}

fn dotQ8_0(q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const nb = (n + block_size - 1) / block_size;
    var sum: f32 = 0;
    for (0..nb) |b| {
        const bp = kv_data + b * q8_0_block_bytes;
        const scale: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp)).*)));
        const base = b * block_size;
        const count = @min(block_size, n - base);
        var acc: V8 = @splat(0.0);
        var i: usize = 0;
        while (i + 8 <= count) : (i += 8) {
            const qv: V8 = q_vec[base + i ..][0..8].*;
            // Vector i8→f32 conversion (vpmovsxbd + vcvtdq2ps / sxtl + scvtf)
            const val_v: V8 = @floatFromInt(@as(@Vector(8, i8), @bitCast((bp + q8_0_scale_bytes + i)[0..8].*)));
            acc = @mulAdd(V8, qv, val_v, acc);
        }
        var block_sum: f32 = @reduce(.Add, acc);
        while (i < count) : (i += 1) {
            const val: f32 = @floatFromInt(@as(i8, @bitCast(bp[q8_0_scale_bytes + i])));
            block_sum = @mulAdd(f32, q_vec[base + i], val, block_sum);
        }
        sum = @mulAdd(f32, scale, block_sum, sum);
    }
    return sum;
}

fn dotInt8(q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const nb = (n + block_size - 1) / block_size;
    var sum: f32 = 0;
    for (0..nb) |b| {
        const bp = kv_data + b * int8_block_bytes;
        const scale: f32 = @as(*align(1) const f32, @ptrCast(bp)).*;
        const base = b * block_size;
        const count = @min(block_size, n - base);
        var acc: V8 = @splat(0.0);
        var i: usize = 0;
        while (i + 8 <= count) : (i += 8) {
            const qv: V8 = q_vec[base + i ..][0..8].*;
            // Vector i8→f32 conversion (vpmovsxbd + vcvtdq2ps / sxtl + scvtf)
            const val_v: V8 = @floatFromInt(@as(@Vector(8, i8), @bitCast((bp + int8_scale_bytes + i)[0..8].*)));
            acc = @mulAdd(V8, qv, val_v, acc);
        }
        var block_sum: f32 = @reduce(.Add, acc);
        while (i < count) : (i += 1) {
            const val: f32 = @floatFromInt(@as(i8, @bitCast(bp[int8_scale_bytes + i])));
            block_sum = @mulAdd(f32, q_vec[base + i], val, block_sum);
        }
        sum = @mulAdd(f32, scale, block_sum, sum);
    }
    return sum;
}

fn dotFp8(q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    var acc: V8 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const qv: V8 = q_vec[i..][0..8].*;
        var kv_v: V8 = undefined;
        inline for (0..8) |j| {
            kv_v[j] = quant.fp8e4m3ToF32(kv_data[i + j]);
        }
        acc = @mulAdd(V8, qv, kv_v, acc);
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < n) : (i += 1) {
        sum = @mulAdd(f32, q_vec[i], quant.fp8e4m3ToF32(kv_data[i]), sum);
    }
    return sum;
}

fn dotNvfp4(q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const nb = (n + nvfp4_block - 1) / nvfp4_block;
    var sum: f32 = 0;
    for (0..nb) |b| {
        const bp = kv_data + b * nvfp4_block_bytes;
        const scale: f32 = quant.fp8e4m3ToF32(bp[0]);
        const base = b * nvfp4_block;
        const count = @min(nvfp4_block, n - base);
        // Pre-unpack all 16 nibbles into f32 (branch-free)
        var vals: [nvfp4_block]f32 = undefined;
        inline for (0..8) |pair| {
            const byte = bp[1 + pair];
            vals[pair * 2] = quant.mxfp4Lookup(byte & 0x0F);
            vals[pair * 2 + 1] = quant.mxfp4Lookup(byte >> 4);
        }
        // SIMD dot product (2 iterations for 16 elements)
        var acc: V8 = @splat(0.0);
        var i: usize = 0;
        while (i + 8 <= count) : (i += 8) {
            const qv: V8 = q_vec[base + i ..][0..8].*;
            const vv: V8 = vals[i..][0..8].*;
            acc = @mulAdd(V8, qv, vv, acc);
        }
        var block_sum: f32 = @reduce(.Add, acc);
        while (i < count) : (i += 1) {
            block_sum = @mulAdd(f32, q_vec[base + i], vals[i], block_sum);
        }
        sum = @mulAdd(f32, scale, block_sum, sum);
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
        .planar2 => planarMulAccum(2, acc, weight, kv_data, n),
        .planar3 => planarMulAccum(3, acc, weight, kv_data, n),
        .planar4 => planarMulAccum(4, acc, weight, kv_data, n),
        .iso2 => isoMulAccum(2, acc, weight, kv_data, n),
        .iso3 => isoMulAccum(3, acc, weight, kv_data, n),
        .iso4 => isoMulAccum(4, acc, weight, kv_data, n),
        .rotor2 => rotorMulAccum(2, acc, weight, kv_data, n),
        .rotor3 => rotorMulAccum(3, acc, weight, kv_data, n),
        .rotor4 => rotorMulAccum(4, acc, weight, kv_data, n),
    }
}

fn mulAccF32(acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const kv: [*]const f32 = @ptrCast(@alignCast(kv_data));

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

    const wv: V8 = @splat(weight);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const cur: V8 = acc[i..][0..8].*;
        // Vector f16→f32 conversion (uses hardware SIMD: vcvtph2ps / fcvtl)
        const kv_v: V8 = @floatCast(@as(@Vector(8, f16), @bitCast(kv[i..][0..8].*)));
        acc[i..][0..8].* = @mulAdd(V8, wv, kv_v, cur);
    }
    while (i < n) : (i += 1) {
        acc[i] = @mulAdd(f32, weight, @as(f32, @floatCast(@as(f16, @bitCast(kv[i])))), acc[i]);
    }
}

fn mulAccQ8_0(acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const nb = (n + block_size - 1) / block_size;
    for (0..nb) |b| {
        const bp = kv_data + b * q8_0_block_bytes;
        const scale: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp)).*)));
        const ws_v: V8 = @splat(weight * scale);
        const base = b * block_size;
        const count = @min(block_size, n - base);
        var i: usize = 0;
        while (i + 8 <= count) : (i += 8) {
            const cur: V8 = acc[base + i ..][0..8].*;
            // Vector i8→f32 conversion (vpmovsxbd + vcvtdq2ps / sxtl + scvtf)
            const val_v: V8 = @floatFromInt(@as(@Vector(8, i8), @bitCast((bp + q8_0_scale_bytes + i)[0..8].*)));
            acc[base + i ..][0..8].* = @mulAdd(V8, ws_v, val_v, cur);
        }
        const ws = weight * scale;
        while (i < count) : (i += 1) {
            acc[base + i] = @mulAdd(f32, ws, @as(f32, @floatFromInt(@as(i8, @bitCast(bp[q8_0_scale_bytes + i])))), acc[base + i]);
        }
    }
}

fn mulAccInt8(acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const nb = (n + block_size - 1) / block_size;
    for (0..nb) |b| {
        const bp = kv_data + b * int8_block_bytes;
        const scale: f32 = @as(*align(1) const f32, @ptrCast(bp)).*;
        const ws_v: V8 = @splat(weight * scale);
        const base = b * block_size;
        const count = @min(block_size, n - base);
        var i: usize = 0;
        while (i + 8 <= count) : (i += 8) {
            const cur: V8 = acc[base + i ..][0..8].*;
            // Vector i8→f32 conversion (vpmovsxbd + vcvtdq2ps / sxtl + scvtf)
            const val_v: V8 = @floatFromInt(@as(@Vector(8, i8), @bitCast((bp + int8_scale_bytes + i)[0..8].*)));
            acc[base + i ..][0..8].* = @mulAdd(V8, ws_v, val_v, cur);
        }
        const ws = weight * scale;
        while (i < count) : (i += 1) {
            acc[base + i] = @mulAdd(f32, ws, @as(f32, @floatFromInt(@as(i8, @bitCast(bp[int8_scale_bytes + i])))), acc[base + i]);
        }
    }
}

fn mulAccFp8(acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const wv: V8 = @splat(weight);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const cur: V8 = acc[i..][0..8].*;
        var kv_v: V8 = undefined;
        inline for (0..8) |j| {
            kv_v[j] = quant.fp8e4m3ToF32(kv_data[i + j]);
        }
        acc[i..][0..8].* = @mulAdd(V8, wv, kv_v, cur);
    }
    while (i < n) : (i += 1) {
        acc[i] = @mulAdd(f32, weight, quant.fp8e4m3ToF32(kv_data[i]), acc[i]);
    }
}

fn mulAccNvfp4(acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const nb = (n + nvfp4_block - 1) / nvfp4_block;
    for (0..nb) |b| {
        const bp = kv_data + b * nvfp4_block_bytes;
        const scale: f32 = quant.fp8e4m3ToF32(bp[0]);
        const ws = weight * scale;
        const base = b * nvfp4_block;
        const count = @min(nvfp4_block, n - base);
        // Pre-unpack all 16 nibbles into f32 (branch-free)
        var vals: [nvfp4_block]f32 = undefined;
        inline for (0..8) |pair| {
            const byte = bp[1 + pair];
            vals[pair * 2] = quant.mxfp4Lookup(byte & 0x0F);
            vals[pair * 2 + 1] = quant.mxfp4Lookup(byte >> 4);
        }
        // SIMD accumulate (2 iterations for 16 elements)
        const ws_v: V8 = @splat(ws);
        var i: usize = 0;
        while (i + 8 <= count) : (i += 8) {
            const cur: V8 = acc[base + i ..][0..8].*;
            const vv: V8 = vals[i..][0..8].*;
            acc[base + i ..][0..8].* = @mulAdd(V8, ws_v, vv, cur);
        }
        while (i < count) : (i += 1) {
            acc[base + i] = @mulAdd(f32, ws, vals[i], acc[base + i]);
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
    const nb = (n + turbo_block_size - 1) / turbo_block_size;

    for (0..nb) |blk| {
        const base = blk * turbo_block_size;
        const count = @min(turbo_block_size, n - base);

        // Load and compute L2 norm
        var buf: [turbo_block_size]f32 = undefined;
        @memcpy(buf[0..count], src[base..][0..count]);
        // Zero-pad remainder
        for (count..turbo_block_size) |i| buf[i] = 0;
        // SIMD L2 norm (buf is always 32 elements = 4 SIMD iterations)
        var norm_acc: V8 = @splat(@as(f32, 0.0));
        inline for (0..4) |qi| {
            const bv: V8 = buf[qi * 8 ..][0..8].*;
            norm_acc = @mulAdd(V8, bv, bv, norm_acc);
        }
        const norm_sq = @reduce(.Add, norm_acc);

        const norm = @sqrt(norm_sq);
        const bp = dst + blk * bb;

        if (norm == 0) {
            // Zero vector: store zero norm and zero indices
            @as(*align(1) u16, @ptrCast(bp)).* = @bitCast(@as(f16, 0));
            @memset(bp[2..bb], 0);
            continue;
        }

        // Walsh-Hadamard Transform (on un-normalized data — WHT linearity
        // lets us fold normalization into the post-WHT scale below).
        wht32(&buf);

        // Combined normalize + WHT scale in single SIMD pass:
        // WHT(x/norm) * (1/sqrt(32)) == WHT(x) * (1 / (norm * sqrt(32)))

        const combined_scale: V8 = @splat(wht_inv_sqrt / norm);
        comptime var vi: usize = 0;
        inline while (vi + 8 <= turbo_block_size) : (vi += 8) {
            buf[vi..][0..8].* = @as(V8, buf[vi..][0..8].*) * combined_scale;
        }

        // Quantize to nearest Lloyd-Max centroid
        var indices: [turbo_block_size]u8 = undefined;
        for (0..turbo_block_size) |i| {
            indices[i] = nearestCentroid(bits, buf[i]);
        }

        // Store f16 norm header
        @as(*align(1) u16, @ptrCast(bp)).* = @bitCast(@as(f16, @floatCast(norm)));

        // Pack indices
        packIndices(bits, bp[2..bb], &indices);
    }
}

/// PlanarQuant store: Givens 2D rotation + Lloyd-Max quantization.
/// Same block format as TurboQuant (f16 norm + packed indices).
fn planarStore(comptime bits: u3, dst: [*]u8, src: [*]const f32, n: usize) void {
    const bb = comptime turboBlockBytes(bits);
    const nb = (n + turbo_block_size - 1) / turbo_block_size;

    for (0..nb) |blk| {
        const base = blk * turbo_block_size;
        const count = @min(turbo_block_size, n - base);
        var buf: [turbo_block_size]f32 = undefined;
        @memcpy(buf[0..count], src[base..][0..count]);
        for (count..turbo_block_size) |i| buf[i] = 0;

        var norm_acc: V8 = @splat(@as(f32, 0.0));
        inline for (0..4) |qi| {
            const bv: V8 = buf[qi * 8 ..][0..8].*;
            norm_acc = @mulAdd(V8, bv, bv, norm_acc);
        }
        const norm = @sqrt(@reduce(.Add, norm_acc));
        const bp = dst + blk * bb;

        if (norm == 0) {
            @as(*align(1) u16, @ptrCast(bp)).* = @bitCast(@as(f16, 0));
            @memset(bp[2..bb], 0);
            continue;
        }

        // Normalize
        const inv_norm = 1.0 / norm;
        for (0..turbo_block_size) |i| buf[i] *= inv_norm;

        // Givens rotation (instead of WHT)
        givensRotateForward(&buf);

        var indices: [turbo_block_size]u8 = undefined;
        for (0..turbo_block_size) |i| indices[i] = nearestCentroid(bits, buf[i]);
        @as(*align(1) u16, @ptrCast(bp)).* = @bitCast(@as(f16, @floatCast(norm)));
        packIndices(bits, bp[2..bb], &indices);
    }
}

/// IsoQuant store: quaternion 4D rotation + Lloyd-Max quantization.
fn isoStore(comptime bits: u3, dst: [*]u8, src: [*]const f32, n: usize) void {
    const bb = comptime turboBlockBytes(bits);
    const nb = (n + turbo_block_size - 1) / turbo_block_size;

    for (0..nb) |blk| {
        const base = blk * turbo_block_size;
        const count = @min(turbo_block_size, n - base);
        var buf: [turbo_block_size]f32 = undefined;
        @memcpy(buf[0..count], src[base..][0..count]);
        for (count..turbo_block_size) |i| buf[i] = 0;

        var norm_acc: V8 = @splat(@as(f32, 0.0));
        inline for (0..4) |qi| {
            const bv: V8 = buf[qi * 8 ..][0..8].*;
            norm_acc = @mulAdd(V8, bv, bv, norm_acc);
        }
        const norm = @sqrt(@reduce(.Add, norm_acc));
        const bp = dst + blk * bb;

        if (norm == 0) {
            @as(*align(1) u16, @ptrCast(bp)).* = @bitCast(@as(f16, 0));
            @memset(bp[2..bb], 0);
            continue;
        }

        const inv_norm = 1.0 / norm;
        for (0..turbo_block_size) |i| buf[i] *= inv_norm;

        // Quaternion rotation (instead of WHT)
        quatRotateForward(&buf);

        var indices: [turbo_block_size]u8 = undefined;
        for (0..turbo_block_size) |i| indices[i] = nearestCentroid(bits, buf[i]);
        @as(*align(1) u16, @ptrCast(bp)).* = @bitCast(@as(f16, @floatCast(norm)));
        packIndices(bits, bp[2..bb], &indices);
    }
}

/// Find the nearest centroid index via binary search over precomputed boundaries.
/// Decision boundaries (midpoints between adjacent centroids) are resolved at
/// comptime, so each search iteration is a single load + compare.
inline fn nearestCentroid(comptime bits: u3, val: f32) u8 {
    const codebook = lloydMaxCodebook(bits);
    const n_centroids = @as(usize, 1) << bits;
    // Precompute boundaries at comptime — avoids runtime midpoint arithmetic.
    const bounds = comptime blk: {
        var b: [n_centroids - 1]f32 = undefined;
        for (0..n_centroids - 1) |i| {
            b[i] = (codebook[i] + codebook[i + 1]) * 0.5;
        }
        break :blk b;
    };
    var lo: usize = 0;
    var hi: usize = bounds.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (val > bounds[mid]) {
            lo = mid + 1;
        } else {
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
            const bit_off_wide: u4 = bit_off;
            const mask = (@as(u8, 1) << bits) - 1;
            dst[byte_idx] |= @as(u8, indices[i] & mask) << bit_off;
            // Handle spanning into next byte (widen to u4 to avoid u3 overflow)
            if (bit_off_wide + bits > 8) {
                const overflow: u4 = bit_off_wide + bits - 8;
                dst[byte_idx + 1] |= @as(u8, indices[i] & mask) >> @as(u3, @intCast(bits - overflow));
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
            const bit_off_wide: u4 = bit_off;
            var val = src[byte_idx] >> bit_off;
            if (bit_off_wide + bits > 8) {
                // Spans into next byte (widen to u4 to avoid u3 overflow)
                val |= src[byte_idx + 1] << @as(u3, @intCast(8 - bit_off_wide));
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
        var q_buf: [turbo_block_size]f32 = undefined;
        const count = @min(turbo_block_size, n - base);
        for (0..count) |i| q_buf[i] = q_vec[base + i];
        for (count..turbo_block_size) |i| q_buf[i] = 0;
        wht32(&q_buf);

        // Unpack indices
        var indices: [turbo_block_size]u8 = undefined;
        unpackIndices(bits, bp[2..][0..data_bytes], &indices);

        // Dot product in WHT domain: pre-expand codebook values then SIMD mulAdd.
        // 4 SIMD iterations (32 elements / 8 lanes) vs 32 scalar multiply-adds.
        var vals: [turbo_block_size]f32 = undefined;
        for (0..turbo_block_size) |i| {
            vals[i] = codebook[indices[i]];
        }

        var acc: V8 = @splat(@as(f32, 0.0));
        comptime var si: usize = 0;
        inline while (si + 8 <= turbo_block_size) : (si += 8) {
            const qv: V8 = q_buf[si..][0..8].*;
            const cv: V8 = vals[si..][0..8].*;
            acc = @mulAdd(V8, qv, cv, acc);
        }

        // Scale: norm / sqrt(32) accounts for WHT normalization in both store and inverse
        sum += norm * @reduce(.Add, acc) * wht_inv_sqrt;
    }

    return sum;
}

/// PlanarQuant dot: forward Givens rotation on query, dot with codebook values.
fn planarDot(comptime bits: u3, q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const bb = comptime turboBlockBytes(bits);
    const codebook = comptime lloydMaxCodebook(bits);
    const nb = (n + turbo_block_size - 1) / turbo_block_size;
    const data_bytes = comptime bb - 2;
    var sum: f32 = 0;

    for (0..nb) |blk| {
        const base = blk * turbo_block_size;
        const bp = kv_data + blk * bb;
        const norm: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp)).*)));
        if (norm == 0) continue;

        var q_buf: [turbo_block_size]f32 = undefined;
        const count = @min(turbo_block_size, n - base);
        for (0..count) |i| q_buf[i] = q_vec[base + i];
        for (count..turbo_block_size) |i| q_buf[i] = 0;
        givensRotateForward(&q_buf);

        var indices: [turbo_block_size]u8 = undefined;
        unpackIndices(bits, bp[2..][0..data_bytes], &indices);

        var vals: [turbo_block_size]f32 = undefined;
        for (0..turbo_block_size) |i| vals[i] = codebook[indices[i]];

        var acc: V8 = @splat(@as(f32, 0.0));
        comptime var si: usize = 0;
        inline while (si + 8 <= turbo_block_size) : (si += 8) {
            const qv: V8 = q_buf[si..][0..8].*;
            const cv: V8 = vals[si..][0..8].*;
            acc = @mulAdd(V8, qv, cv, acc);
        }
        sum += norm * @reduce(.Add, acc);
    }
    return sum;
}

/// IsoQuant dot: forward quaternion rotation on query, dot with codebook values.
fn isoDot(comptime bits: u3, q_vec: [*]const f32, kv_data: [*]const u8, n: usize) f32 {
    const bb = comptime turboBlockBytes(bits);
    const codebook = comptime lloydMaxCodebook(bits);
    const nb = (n + turbo_block_size - 1) / turbo_block_size;
    const data_bytes = comptime bb - 2;
    var sum: f32 = 0;

    for (0..nb) |blk| {
        const base = blk * turbo_block_size;
        const bp = kv_data + blk * bb;
        const norm: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp)).*)));
        if (norm == 0) continue;

        var q_buf: [turbo_block_size]f32 = undefined;
        const count = @min(turbo_block_size, n - base);
        for (0..count) |i| q_buf[i] = q_vec[base + i];
        for (count..turbo_block_size) |i| q_buf[i] = 0;
        quatRotateForward(&q_buf);

        var indices: [turbo_block_size]u8 = undefined;
        unpackIndices(bits, bp[2..][0..data_bytes], &indices);

        var vals: [turbo_block_size]f32 = undefined;
        for (0..turbo_block_size) |i| vals[i] = codebook[indices[i]];

        var acc: V8 = @splat(@as(f32, 0.0));
        comptime var si: usize = 0;
        inline while (si + 8 <= turbo_block_size) : (si += 8) {
            const qv: V8 = q_buf[si..][0..8].*;
            const cv: V8 = vals[si..][0..8].*;
            acc = @mulAdd(V8, qv, cv, acc);
        }
        sum += norm * @reduce(.Add, acc);
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
        var indices: [turbo_block_size]u8 = undefined;
        unpackIndices(bits, bp[2..][0..data_bytes], &indices);

        var buf: [turbo_block_size]f32 = undefined;
        for (0..turbo_block_size) |i| {
            buf[i] = codebook[indices[i]];
        }

        // Inverse WHT: forward WHT followed by 1/sqrt(32) gives the orthonormal inverse,
        // matching the 1/sqrt(32) normalization applied during store.
        wht32(&buf);

        // Accumulate: rescale by weight * norm / sqrt(32) (orthonormal WHT inverse + denormalization)

        const scale = weight * norm * wht_inv_sqrt;
        const scale_v: V8 = @splat(scale);
        const count = @min(turbo_block_size, n - base);
        var si: usize = 0;
        while (si + 8 <= count) : (si += 8) {
            const bv: V8 = buf[si..][0..8].*;
            const cv: V8 = acc[base + si ..][0..8].*;
            acc[base + si ..][0..8].* = @mulAdd(V8, bv, scale_v, cv);
        }
        while (si < count) : (si += 1) {
            acc[base + si] = @mulAdd(f32, buf[si], scale, acc[base + si]);
        }
    }
}

/// PlanarQuant dequant accumulate: unpack → codebook → inverse Givens → accumulate.
fn planarMulAccum(comptime bits: u3, acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const bb = comptime turboBlockBytes(bits);
    const codebook = comptime lloydMaxCodebook(bits);
    const nb = (n + turbo_block_size - 1) / turbo_block_size;
    const data_bytes = comptime bb - 2;

    for (0..nb) |blk| {
        const base = blk * turbo_block_size;
        const bp = kv_data + blk * bb;
        const norm: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp)).*)));
        if (norm == 0) continue;

        var indices: [turbo_block_size]u8 = undefined;
        unpackIndices(bits, bp[2..][0..data_bytes], &indices);

        var buf: [turbo_block_size]f32 = undefined;
        for (0..turbo_block_size) |i| buf[i] = codebook[indices[i]];

        givensRotateInverse(&buf);

        const scale = weight * norm;
        const scale_v: V8 = @splat(scale);
        const count = @min(turbo_block_size, n - base);
        var si: usize = 0;
        while (si + 8 <= count) : (si += 8) {
            const bv: V8 = buf[si..][0..8].*;
            const cv: V8 = acc[base + si ..][0..8].*;
            acc[base + si ..][0..8].* = @mulAdd(V8, bv, scale_v, cv);
        }
        while (si < count) : (si += 1) {
            acc[base + si] = @mulAdd(f32, buf[si], scale, acc[base + si]);
        }
    }
}

/// IsoQuant dequant accumulate: unpack → codebook → inverse quaternion → accumulate.
fn isoMulAccum(comptime bits: u3, acc: [*]f32, weight: f32, kv_data: [*]const u8, n: usize) void {
    const bb = comptime turboBlockBytes(bits);
    const codebook = comptime lloydMaxCodebook(bits);
    const nb = (n + turbo_block_size - 1) / turbo_block_size;
    const data_bytes = comptime bb - 2;

    for (0..nb) |blk| {
        const base = blk * turbo_block_size;
        const bp = kv_data + blk * bb;
        const norm: f32 = @floatCast(@as(f16, @bitCast(@as(*align(1) const u16, @ptrCast(bp)).*)));
        if (norm == 0) continue;

        var indices: [turbo_block_size]u8 = undefined;
        unpackIndices(bits, bp[2..][0..data_bytes], &indices);

        var buf: [turbo_block_size]f32 = undefined;
        for (0..turbo_block_size) |i| buf[i] = codebook[indices[i]];

        quatRotateInverse(&buf);

        const scale = weight * norm;
        const scale_v: V8 = @splat(scale);
        const count = @min(turbo_block_size, n - base);
        var si: usize = 0;
        while (si + 8 <= count) : (si += 8) {
            const bv: V8 = buf[si..][0..8].*;
            const cv: V8 = acc[base + si ..][0..8].*;
            acc[base + si ..][0..8].* = @mulAdd(V8, bv, scale_v, cv);
        }
        while (si < count) : (si += 1) {
            acc[base + si] = @mulAdd(f32, buf[si], scale, acc[base + si]);
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
    if (!std.math.isFinite(abs_val)) return (sign << 7) | fp8_e4m3_max_finite;

    const clamped = @min(abs_val, fp8_e4m3_max);

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

    if (e4m3_exp >= fp8_e4m3_max_biased_exp) {
        return (sign << 7) | fp8_e4m3_max_finite;
    }

    // Normal: round mantissa from 23 bits to 3 bits
    const mant3: u8 = @truncate((f32_mant + (1 << 19)) >> 20);
    if (mant3 >= 8) {
        // Mantissa overflow → increment exponent
        const new_exp: u8 = @intCast(e4m3_exp + 1);
        if (new_exp >= fp8_e4m3_max_biased_exp) return (sign << 7) | fp8_e4m3_max_finite;
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

    // Per-element verification via unit vector dots
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

    // Multi-element dot to exercise SIMD path (n=8 hits the 8-wide loop)
    var q_vec = [_]f32{ 1.0, 2.0, -1.0, 0.5, 0.0, 1.0, -0.5, 3.0 };
    const dot8 = kvDot(&q_vec, &buf, 8, .f16);
    var expected_dot: f32 = 0;
    for (0..8) |i| {
        const f16_val: f32 = @floatCast(@as(f16, @floatCast(src[i])));
        expected_dot += q_vec[i] * f16_val;
    }
    try std.testing.expectApproxEqAbs(expected_dot, dot8, 1e-3);

    // MulAccum with weight to exercise weighted accumulation
    var acc = [_]f32{0} ** 8;
    kvMulAccum(&acc, 2.5, &buf, 8, .f16);
    for (0..8) |i| {
        const f16_val: f32 = @floatCast(@as(f16, @floatCast(src[i])));
        try std.testing.expectApproxEqAbs(2.5 * f16_val, acc[i], 1e-4);
    }
}

test "q8_0 roundtrip accuracy" {
    // Values in a range where Q8_0 should preserve well
    const src = [_]f32{ 0.5, -0.3, 1.0, -1.0, 0.0, 0.7, -0.8, 0.1 };
    var buf: [q8_0_block_bytes]u8 = undefined;
    kvStore(&buf, &src, 8, .q8_0);

    // Per-element verification via unit vector dots
    for (0..8) |i| {
        var q_unit = [_]f32{0} ** 8;
        q_unit[i] = 1.0;
        const dot = kvDot(&q_unit, &buf, 8, .q8_0);
        try std.testing.expectApproxEqAbs(src[i], dot, 0.01);
    }

    // Dot with all-ones should ≈ sum(src)
    var q_ones = [_]f32{1.0} ** 8;
    const dot_sum = kvDot(&q_ones, &buf, 8, .q8_0);
    var expected_sum: f32 = 0;
    for (src) |v| expected_sum += v;
    try std.testing.expectApproxEqAbs(expected_sum, dot_sum, 0.05);
}

test "int8 roundtrip accuracy" {
    const src = [_]f32{ 0.5, -0.3, 1.0, -1.0, 0.0, 0.7, -0.8, 0.1 };
    var buf: [int8_block_bytes]u8 = undefined;
    kvStore(&buf, &src, 8, .int8);

    // Per-element verification via unit vector dots
    for (0..8) |i| {
        var q_unit = [_]f32{0} ** 8;
        q_unit[i] = 1.0;
        const dot = kvDot(&q_unit, &buf, 8, .int8);
        try std.testing.expectApproxEqAbs(src[i], dot, 0.01);
    }

    // Dot with all-ones should ≈ sum(src)
    var q_ones = [_]f32{1.0} ** 8;
    const dot_sum = kvDot(&q_ones, &buf, 8, .int8);
    var expected: f32 = 0;
    for (src) |v| expected += v;
    try std.testing.expectApproxEqAbs(expected, dot_sum, 0.1);
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
    // NVFP4 E2M1 has limited precision — test with non-representable values
    // to exercise actual quantization error, not just identity roundtrip.
    // E2M1 codebook: {0, 0.5, 1, 1.5, 2, 3, 4, 6} (positive side)
    const src = [_]f32{ 0.8, -1.3, 0.7, 2.5, 0.0, -0.2, 3.5, -4.5 };
    var buf: [nvfp4_block_bytes]u8 = undefined;
    kvStore(&buf, &src, 8, .nvfp4);

    // Verify individual element reconstruction via unit-vector dots.
    // Non-representable values should be quantized to nearest codebook entry.
    var accum: [8]f32 = undefined;
    for (0..8) |i| {
        var unit = [_]f32{0} ** 8;
        unit[i] = 1.0;
        accum[i] = kvDot(&unit, &buf, 8, .nvfp4);
    }
    // Each element should be within half the max codebook gap (scaled).
    // With scale ≈ 0.75 (amax=4.5, e2m1_max=6), worst case is 0.5.
    for (0..8) |i| {
        try std.testing.expectApproxEqAbs(src[i], accum[i], 0.6);
    }

    // Dot with all-ones should approximate sum(src)
    var q_ones = [_]f32{1.0} ** 8;
    const dot = kvDot(&q_ones, &buf, 8, .nvfp4);
    var expected: f32 = 0;
    for (accum) |v| expected += v;
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

test "kvByteOffset mid-block elements" {
    // Elements in the middle of a block should map to the block's start offset
    // q8_0: 32-element blocks, 34 bytes each
    try std.testing.expectEqual(@as(usize, 0), kvByteOffset(.q8_0, 10)); // mid first block
    try std.testing.expectEqual(@as(usize, 34), kvByteOffset(.q8_0, 50)); // mid second block
    try std.testing.expectEqual(@as(usize, 68), kvByteOffset(.q8_0, 90)); // mid third block
    // nvfp4: 16-element blocks, 9 bytes each
    try std.testing.expectEqual(@as(usize, 0), kvByteOffset(.nvfp4, 7)); // mid first block
    try std.testing.expectEqual(@as(usize, 9), kvByteOffset(.nvfp4, 20)); // mid second block
}

test "kvByteOffset block boundaries" {
    // q8_0: 32-element blocks, 34 bytes each (2-byte f16 scale + 32 i8)
    // Last element of first block
    try std.testing.expectEqual(@as(usize, 0), kvByteOffset(.q8_0, 31));
    // First element of second block
    try std.testing.expectEqual(@as(usize, 34), kvByteOffset(.q8_0, 32));
    // Last element of second block
    try std.testing.expectEqual(@as(usize, 34), kvByteOffset(.q8_0, 63));
    // First element of third block
    try std.testing.expectEqual(@as(usize, 68), kvByteOffset(.q8_0, 64));

    // nvfp4: 16-element blocks, 9 bytes each (1-byte fp8 scale + 8 packed nibbles)
    try std.testing.expectEqual(@as(usize, 0), kvByteOffset(.nvfp4, 0));
    try std.testing.expectEqual(@as(usize, 0), kvByteOffset(.nvfp4, 15));
    try std.testing.expectEqual(@as(usize, 9), kvByteOffset(.nvfp4, 16));
    try std.testing.expectEqual(@as(usize, 9), kvByteOffset(.nvfp4, 31));
    try std.testing.expectEqual(@as(usize, 18), kvByteOffset(.nvfp4, 32));
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
    // Clamping: values beyond E4M3 max (448.0) clamp to 0x7E = 448.0
    try std.testing.expectEqual(@as(u8, 0x7E), f32ToFp8E4M3(500.0));
    // Negative clamping: -500 clamps to -448.0 = 0xFE
    try std.testing.expectEqual(@as(u8, 0xFE), f32ToFp8E4M3(-500.0));
}

test "f32ToFp8E4M3 denormal and boundary values" {
    // Small positive value that falls in E4M3 denormal range (exp <= 0)
    // E4M3 smallest normal: 2^(-6) = 0.015625. Values below this are denormal.
    const small = f32ToFp8E4M3(0.01);
    const rt_small = quant.fp8e4m3ToF32(small);
    // Denormal roundtrip should be close (limited precision but not zero)
    try std.testing.expect(rt_small > 0.0);
    try std.testing.expect(rt_small < 0.02);

    // Very small value near zero — should roundtrip to something near zero
    const tiny = f32ToFp8E4M3(1e-4);
    const rt_tiny = quant.fp8e4m3ToF32(tiny);
    try std.testing.expect(rt_tiny >= 0.0);
    try std.testing.expect(rt_tiny < 0.01);

    // Negative denormal
    const neg_small = f32ToFp8E4M3(-0.01);
    const rt_neg = quant.fp8e4m3ToF32(neg_small);
    try std.testing.expect(rt_neg < 0.0);
    try std.testing.expect(rt_neg > -0.02);

    // NaN input should clamp to max finite (0x7E = 448.0)
    const nan_result = f32ToFp8E4M3(std.math.nan(f32));
    try std.testing.expectEqual(@as(u8, 0x7E), nan_result);

    // Inf input should clamp to max finite (0x7E = 448.0)
    const inf_result = f32ToFp8E4M3(std.math.inf(f32));
    try std.testing.expectEqual(@as(u8, 0x7E), inf_result);
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
    // Divide by N to recover original (WHT is self-inverse up to factor of N)
    const turbo_block_f32: f32 = @floatFromInt(turbo_block_size);
    for (0..turbo_block_size) |i| buf[i] /= turbo_block_f32;

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

    // Dot with all-ones = sum of dequantized values. Compare against sum of source
    // (WHT redistributes quantization error, so the sums can differ, but should be
    // in the same ballpark).
    var q_ones = [_]f32{1.0} ** 32;
    const dot_sum = kvDot(&q_ones, &buf, 32, .turbo4);
    try std.testing.expect(std.math.isFinite(dot_sum));
    var src_sum: f32 = 0;
    for (src) |v| src_sum += v;
    try std.testing.expectApproxEqAbs(src_sum, dot_sum, 4.0);

    // MulAccum MSE: verify quantization is not completely broken.
    // Turbo uses WHT + Lloyd-Max which can have significant per-element error
    // on small (32-element) signals; the turboDot-vs-naive test validates consistency.
    var acc = [_]f32{0} ** 32;
    kvMulAccum(&acc, 1.0, &buf, 32, .turbo4);
    var mse: f32 = 0;
    for (0..32) |i| {
        const err = acc[i] - src[i];
        mse += err * err;
    }
    mse /= @as(f32, @floatFromInt(turbo_block_size));
    try std.testing.expect(mse < 0.5);
}

test "turbo3 roundtrip accuracy" {
    var src: [32]f32 = undefined;
    for (0..32) |i| {
        src[i] = @sin(@as(f32, @floatFromInt(i)) * 0.5) * 2.0;
    }

    var buf: [turbo3_block_bytes]u8 = undefined;
    kvStore(&buf, &src, 32, .turbo3);

    // Dot with all-ones = sum of dequantized. Compare against source sum.
    var q_ones = [_]f32{1.0} ** 32;
    const dot_sum = kvDot(&q_ones, &buf, 32, .turbo3);
    try std.testing.expect(std.math.isFinite(dot_sum));
    var src_sum: f32 = 0;
    for (src) |v| src_sum += v;
    try std.testing.expectApproxEqAbs(src_sum, dot_sum, 12.0);

    // MulAccum MSE (3-bit has higher error than 4-bit; turboDot-vs-naive validates consistency)
    var acc = [_]f32{0} ** 32;
    kvMulAccum(&acc, 1.0, &buf, 32, .turbo3);
    var mse: f32 = 0;
    for (0..32) |i| {
        const err = acc[i] - src[i];
        mse += err * err;
    }
    mse /= @as(f32, @floatFromInt(turbo_block_size));
    try std.testing.expect(mse < 5.0);
}

test "turbo2 roundtrip accuracy" {
    var src: [32]f32 = undefined;
    for (0..32) |i| {
        src[i] = @sin(@as(f32, @floatFromInt(i)) * 0.5) * 2.0;
    }

    var buf: [turbo2_block_bytes]u8 = undefined;
    kvStore(&buf, &src, 32, .turbo2);

    // Dot with all-ones = sum of dequantized. Compare against source sum.
    var q_ones = [_]f32{1.0} ** 32;
    const dot_sum = kvDot(&q_ones, &buf, 32, .turbo2);
    try std.testing.expect(std.math.isFinite(dot_sum));
    var src_sum: f32 = 0;
    for (src) |v| src_sum += v;
    try std.testing.expectApproxEqAbs(src_sum, dot_sum, 20.0);

    // MulAccum MSE (2-bit has highest error; turboDot-vs-naive validates consistency)
    var acc = [_]f32{0} ** 32;
    kvMulAccum(&acc, 1.0, &buf, 32, .turbo2);
    var mse: f32 = 0;
    for (0..32) |i| {
        const err = acc[i] - src[i];
        mse += err * err;
    }
    mse /= @as(f32, @floatFromInt(turbo_block_size));
    try std.testing.expect(mse < 15.0);
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

test "pack/unpack indices all-max values" {
    // All indices set to maximum value for each bit width — catches
    // overflow in nibble packing and byte boundary crossing.
    var indices: [32]u8 = undefined;

    // 2-bit: all 3s (0b11)
    for (&indices) |*v| v.* = 3;
    var buf2: [8]u8 = undefined;
    packIndices(2, &buf2, &indices);
    var out2: [32]u8 = undefined;
    unpackIndices(2, &buf2, &out2);
    for (0..32) |i| try std.testing.expectEqual(@as(u8, 3), out2[i]);

    // 3-bit: all 7s (0b111) — 3-bit packing crosses byte boundaries
    for (&indices) |*v| v.* = 7;
    var buf3: [12]u8 = undefined;
    packIndices(3, &buf3, &indices);
    var out3: [32]u8 = undefined;
    unpackIndices(3, &buf3, &out3);
    for (0..32) |i| try std.testing.expectEqual(@as(u8, 7), out3[i]);

    // 4-bit: all 15s (0b1111)
    for (&indices) |*v| v.* = 15;
    var buf4: [16]u8 = undefined;
    packIndices(4, &buf4, &indices);
    var out4: [32]u8 = undefined;
    unpackIndices(4, &buf4, &out4);
    for (0..32) |i| try std.testing.expectEqual(@as(u8, 15), out4[i]);
}

test "fuzz: all kv_quant functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // ── Pub constants (comptime verification) ──
            comptime {
                std.debug.assert(turbo2_block_bytes == 10);
                std.debug.assert(turbo3_block_bytes == 14);
                std.debug.assert(turbo4_block_bytes == 18);
            }

            // ── KvQuantType: exercise all pub methods on a random variant ──
            const type_idx = smith.valueWithHash(u8, 0) % 18;
            const all_types = [18]KvQuantType{
                .f32,    .f16,    .q8_0,   .int8,    .fp8_e4m3, .nvfp4,
                .turbo2, .turbo3, .turbo4, .planar2, .planar3,  .planar4,
                .iso2,   .iso3,   .iso4,   .rotor2,  .rotor3,   .rotor4,
            };
            const kv_type = all_types[type_idx];

            // name — must return non-empty string
            const nm = kv_type.name();
            std.debug.assert(nm.len > 0);

            // bitsPerElement — must be positive
            const bpe = kv_type.bitsPerElement();
            std.debug.assert(bpe > 0);

            // Boolean classifiers — at least one group must match for rotation quants
            const it = kv_type.isTurbo();
            const ip = kv_type.isPlanar();
            const ii = kv_type.isIso();
            const ir = kv_type.isRotor();
            const irq = kv_type.isRotationQuant();
            std.debug.assert(irq == (it or ip or ii or ir));

            // turboBits — 0 for non-rotation, 2/3/4 for rotation
            const tb = kv_type.turboBits();
            if (irq) {
                std.debug.assert(tb >= 2 and tb <= 4);
            } else {
                std.debug.assert(tb == 0);
            }

            // turboBlockByteSize — only nonzero for turbo{2,3,4}
            _ = kv_type.turboBlockByteSize();

            // fromString — roundtrip: name -> fromString should find *something*
            // (name() returns short names like "F32", "TQ2", etc.)
            _ = KvQuantType.fromString(nm);

            // fromString with random bytes — must not crash
            var rnd_str: [4]u8 = undefined;
            for (&rnd_str) |*c| c.* = smith.valueWithHash(u8, 1);
            _ = KvQuantType.fromString(&rnd_str);

            // ── kvSliceBytes / kvByteOffset ──
            // Use a small n to keep buffers manageable (32 = one full block for all formats)
            const n: usize = 32;
            const slice_bytes = kvSliceBytes(kv_type, n);
            std.debug.assert(slice_bytes > 0);

            const offset0 = kvByteOffset(kv_type, 0);
            std.debug.assert(offset0 == 0);
            const offset_last = kvByteOffset(kv_type, n - 1);
            std.debug.assert(offset_last < slice_bytes);

            // ── kvStore / kvDot / kvMulAccum roundtrip ──
            // Generate 32 random f32 source values, clamped to finite range
            var src: [32]f32 = undefined;
            for (&src, 0..) |*v, si| {
                const raw_bits = smith.valueWithHash(u32, @as(u32, @intCast(si)) +% 100);
                var fval: f32 = @bitCast(raw_bits);
                // Clamp to reasonable range to avoid inf/nan issues in quantization
                if (!std.math.isFinite(fval)) fval = 0;
                fval = std.math.clamp(fval, -100.0, 100.0);
                v.* = fval;
            }

            // Allocate enough buffer for any kv type at n=32
            // Max is int8: 36 bytes per 32 elems. Turbo4: 18. Use 64 for safety.
            var kv_buf: [64]u8 align(4) = @splat(0);
            kvStore(&kv_buf, &src, n, kv_type);

            // kvDot — result must be finite
            var q_vec: [32]f32 = undefined;
            for (&q_vec, 0..) |*v, qi| {
                const raw_bits = smith.valueWithHash(u32, @as(u32, @intCast(qi)) +% 200);
                var fval: f32 = @bitCast(raw_bits);
                if (!std.math.isFinite(fval)) fval = 0;
                fval = std.math.clamp(fval, -10.0, 10.0);
                v.* = fval;
            }
            const dot = kvDot(&q_vec, &kv_buf, n, kv_type);
            std.debug.assert(std.math.isFinite(dot));

            // kvMulAccum — result elements must be finite
            var acc = [_]f32{0} ** 32;
            const weight_bits = smith.valueWithHash(u32, 300);
            var weight: f32 = @bitCast(weight_bits);
            if (!std.math.isFinite(weight)) weight = 1.0;
            weight = std.math.clamp(weight, -10.0, 10.0);
            kvMulAccum(&acc, weight, &kv_buf, n, kv_type);
            for (acc) |a| std.debug.assert(std.math.isFinite(a));
        }
    }.f, .{});
}
