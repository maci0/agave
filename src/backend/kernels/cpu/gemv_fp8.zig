//! CPU GEMV kernels for FP8 quantization formats.
//! FP8_E4M3 (4 exponent, 3 mantissa) and FP8_E5M2 (5 exponent, 2 mantissa).
//! 4-row batching with V8 SIMD, matching the F32/F16/BF16 kernel structure.

const quant = @import("../../../ops/quant.zig");
const gemv_common = @import("gemv.zig");
const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);
/// Sparse block-skip chunk size for element-level formats.
const sparse_chunk = 32;

/// FP8_E4M3: 1 byte per element (4 exponent, 3 mantissa, bias=7).
/// 4-row batched with V8 SIMD for instruction-level parallelism.
pub fn gemvFP8_E4M3(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    var row: usize = 0;
    while (row + 4 <= n) : (row += 4) {
        var acc0: V8 = v8zero;
        var acc1: V8 = v8zero;
        var acc2: V8 = v8zero;
        var acc3: V8 = v8zero;
        const r0 = row * k;
        const r1 = r0 + k;
        const r2 = r1 + k;
        const r3 = r2 + k;
        var i: usize = 0;
        while (i + 8 <= k) {
            if (i % sparse_chunk == 0 and i + sparse_chunk <= k and gemv_common.isBlockSparse(x, i, sparse_chunk)) {
                i += sparse_chunk;
                continue;
            }
            const xv: V8 = x[i..][0..8].*;
            var w0: V8 = undefined;
            var w1: V8 = undefined;
            var w2: V8 = undefined;
            var w3: V8 = undefined;
            inline for (0..8) |idx| {
                w0[idx] = quant.fp8e4m3ToF32(w[r0 + i + idx]);
                w1[idx] = quant.fp8e4m3ToF32(w[r1 + i + idx]);
                w2[idx] = quant.fp8e4m3ToF32(w[r2 + i + idx]);
                w3[idx] = quant.fp8e4m3ToF32(w[r3 + i + idx]);
            }
            acc0 = @mulAdd(V8, xv, w0, acc0);
            acc1 = @mulAdd(V8, xv, w1, acc1);
            acc2 = @mulAdd(V8, xv, w2, acc2);
            acc3 = @mulAdd(V8, xv, w3, acc3);
            i += 8;
        }
        var t0: f32 = 0.0;
        var t1: f32 = 0.0;
        var t2: f32 = 0.0;
        var t3: f32 = 0.0;
        while (i < k) : (i += 1) {
            const xv = x[i];
            t0 = @mulAdd(f32, xv, quant.fp8e4m3ToF32(w[r0 + i]), t0);
            t1 = @mulAdd(f32, xv, quant.fp8e4m3ToF32(w[r1 + i]), t1);
            t2 = @mulAdd(f32, xv, quant.fp8e4m3ToF32(w[r2 + i]), t2);
            t3 = @mulAdd(f32, xv, quant.fp8e4m3ToF32(w[r3 + i]), t3);
        }
        y[row] = @reduce(.Add, acc0) + t0;
        y[row + 1] = @reduce(.Add, acc1) + t1;
        y[row + 2] = @reduce(.Add, acc2) + t2;
        y[row + 3] = @reduce(.Add, acc3) + t3;
    }
    while (row < n) : (row += 1) {
        var acc: V8 = v8zero;
        var tail: f32 = 0.0;
        const roff = row * k;
        var i: usize = 0;
        while (i + 8 <= k) {
            if (i % sparse_chunk == 0 and i + sparse_chunk <= k and gemv_common.isBlockSparse(x, i, sparse_chunk)) {
                i += sparse_chunk;
                continue;
            }
            const xv: V8 = x[i..][0..8].*;
            var wv: V8 = undefined;
            inline for (0..8) |idx| {
                wv[idx] = quant.fp8e4m3ToF32(w[roff + i + idx]);
            }
            acc = @mulAdd(V8, xv, wv, acc);
            i += 8;
        }
        while (i < k) : (i += 1) tail = @mulAdd(f32, x[i], quant.fp8e4m3ToF32(w[roff + i]), tail);
        y[row] = @reduce(.Add, acc) + tail;
    }
}

/// FP8 E4M3 encoding for 1.0: sign=0, exp=0111=7, mant=000 → 2^(7-7) * 1.0 = 1.0.
const fp8_e4m3_one: u8 = 0x38;
/// FP8 E4M3 encoding for 2.0: sign=0, exp=1000=8, mant=000 → 2^(8-7) * 1.0 = 2.0.
const fp8_e4m3_two: u8 = 0x40;
/// FP8 E4M3 encoding for 0.5: sign=0, exp=0110=6, mant=000 → 2^(6-7) * 1.0 = 0.5.
const fp8_e4m3_half: u8 = 0x30;
/// FP8 E5M2 encoding for 1.0: sign=0, exp=01111=15, mant=00 → 2^(15-15) * 1.0 = 1.0.
const fp8_e5m2_one: u8 = 0x3C;
/// FP8 E5M2 encoding for 2.0: sign=0, exp=10000=16, mant=00 → 2^(16-15) * 1.0 = 2.0.
const fp8_e5m2_two: u8 = 0x40;
/// FP8 E5M2 encoding for 0.5: sign=0, exp=01110=14, mant=00 → 2^(14-15) * 1.0 = 0.5.
const fp8_e5m2_half: u8 = 0x38;

/// FP8_E5M2: 1 byte per element (5 exponent, 2 mantissa, bias=15).
/// 4-row batched with V8 SIMD for instruction-level parallelism.
pub fn gemvFP8_E5M2(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    var row: usize = 0;
    while (row + 4 <= n) : (row += 4) {
        var acc0: V8 = v8zero;
        var acc1: V8 = v8zero;
        var acc2: V8 = v8zero;
        var acc3: V8 = v8zero;
        const r0 = row * k;
        const r1 = r0 + k;
        const r2 = r1 + k;
        const r3 = r2 + k;
        var i: usize = 0;
        while (i + 8 <= k) {
            if (i % sparse_chunk == 0 and i + sparse_chunk <= k and gemv_common.isBlockSparse(x, i, sparse_chunk)) {
                i += sparse_chunk;
                continue;
            }
            const xv: V8 = x[i..][0..8].*;
            var w0: V8 = undefined;
            var w1: V8 = undefined;
            var w2: V8 = undefined;
            var w3: V8 = undefined;
            inline for (0..8) |idx| {
                w0[idx] = quant.fp8e5m2ToF32(w[r0 + i + idx]);
                w1[idx] = quant.fp8e5m2ToF32(w[r1 + i + idx]);
                w2[idx] = quant.fp8e5m2ToF32(w[r2 + i + idx]);
                w3[idx] = quant.fp8e5m2ToF32(w[r3 + i + idx]);
            }
            acc0 = @mulAdd(V8, xv, w0, acc0);
            acc1 = @mulAdd(V8, xv, w1, acc1);
            acc2 = @mulAdd(V8, xv, w2, acc2);
            acc3 = @mulAdd(V8, xv, w3, acc3);
            i += 8;
        }
        var t0: f32 = 0.0;
        var t1: f32 = 0.0;
        var t2: f32 = 0.0;
        var t3: f32 = 0.0;
        while (i < k) : (i += 1) {
            const xv = x[i];
            t0 = @mulAdd(f32, xv, quant.fp8e5m2ToF32(w[r0 + i]), t0);
            t1 = @mulAdd(f32, xv, quant.fp8e5m2ToF32(w[r1 + i]), t1);
            t2 = @mulAdd(f32, xv, quant.fp8e5m2ToF32(w[r2 + i]), t2);
            t3 = @mulAdd(f32, xv, quant.fp8e5m2ToF32(w[r3 + i]), t3);
        }
        y[row] = @reduce(.Add, acc0) + t0;
        y[row + 1] = @reduce(.Add, acc1) + t1;
        y[row + 2] = @reduce(.Add, acc2) + t2;
        y[row + 3] = @reduce(.Add, acc3) + t3;
    }
    while (row < n) : (row += 1) {
        var acc: V8 = v8zero;
        var tail: f32 = 0.0;
        const roff = row * k;
        var i: usize = 0;
        while (i + 8 <= k) {
            if (i % sparse_chunk == 0 and i + sparse_chunk <= k and gemv_common.isBlockSparse(x, i, sparse_chunk)) {
                i += sparse_chunk;
                continue;
            }
            const xv: V8 = x[i..][0..8].*;
            var wv: V8 = undefined;
            inline for (0..8) |idx| {
                wv[idx] = quant.fp8e5m2ToF32(w[roff + i + idx]);
            }
            acc = @mulAdd(V8, xv, wv, acc);
            i += 8;
        }
        while (i < k) : (i += 1) tail = @mulAdd(f32, x[i], quant.fp8e5m2ToF32(w[roff + i]), tail);
        y[row] = @reduce(.Add, acc) + tail;
    }
}

const std = @import("std");

// --- E4M3 tests ---

test "gemvFP8_E4M3 identity" {
    // n=1, k=8. x = [1,0,0,...,0], w = known FP8 values.
    // y[0] = dot(dequant(w[0,:]), x) = dequant(w[0]) since only x[0]=1.
    var w: [8]u8 = undefined;
    w[0] = fp8_e4m3_two; // 2.0
    for (1..8) |i| w[i] = fp8_e4m3_half; // 0.5 (won't contribute)
    var x = [_]f32{ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    var y: [1]f32 = undefined;
    gemvFP8_E4M3(&x, &w, &y, 1, 8);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), y[0], 1e-6);
}

test "gemvFP8_E4M3 dot product" {
    // n=2, k=16. x = all 1.0, w = all FP8(1.0).
    // y[row] = sum of 16 ones = 16.0.
    var w: [2 * 16]u8 = undefined;
    for (&w) |*b| b.* = fp8_e4m3_one;
    var x: [16]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [2]f32 = undefined;
    gemvFP8_E4M3(&x, &w, &y, 2, 16);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), y[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), y[1], 1e-5);
}

test "gemvFP8_E4M3 zero input" {
    // x = all zeros → y must be all zeros regardless of w.
    var w: [3 * 8]u8 = undefined;
    for (&w) |*b| b.* = fp8_e4m3_two; // Non-zero weights
    var x: [8]f32 = undefined;
    for (&x) |*v| v.* = 0.0;
    var y: [3]f32 = .{ 999.0, 999.0, 999.0 };
    gemvFP8_E4M3(&x, &w, &y, 3, 8);
    for (0..3) |i| try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[i], 1e-6);
}

test "gemvFP8_E4M3 non-aligned k" {
    // k=13 (not divisible by 8) exercises the scalar tail path.
    // n=1, x = all 1.0, w = all FP8(0.5).
    // y[0] = 13 * 0.5 = 6.5.
    var w: [13]u8 = undefined;
    for (&w) |*b| b.* = fp8_e4m3_half;
    var x: [13]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvFP8_E4M3(&x, &w, &y, 1, 13);
    try std.testing.expectApproxEqAbs(@as(f32, 6.5), y[0], 1e-5);
}

// --- E5M2 tests ---

test "gemvFP8_E5M2 identity" {
    // n=1, k=8. x = [1,0,0,...,0], w[0] = FP8_E5M2(2.0).
    // y[0] = 2.0.
    var w: [8]u8 = undefined;
    w[0] = fp8_e5m2_two; // 2.0
    for (1..8) |i| w[i] = fp8_e5m2_half; // 0.5 (won't contribute)
    var x = [_]f32{ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    var y: [1]f32 = undefined;
    gemvFP8_E5M2(&x, &w, &y, 1, 8);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), y[0], 1e-6);
}

test "gemvFP8_E5M2 dot product" {
    // n=2, k=16. x = all 1.0, w = all FP8_E5M2(1.0).
    // y[row] = 16.0.
    var w: [2 * 16]u8 = undefined;
    for (&w) |*b| b.* = fp8_e5m2_one;
    var x: [16]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [2]f32 = undefined;
    gemvFP8_E5M2(&x, &w, &y, 2, 16);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), y[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), y[1], 1e-5);
}

test "gemvFP8_E5M2 non-aligned k" {
    // k=13 (not divisible by 8) exercises the scalar tail path.
    // n=1, x = all 1.0, w = all FP8_E5M2(0.5).
    // y[0] = 13 * 0.5 = 6.5.
    var w: [13]u8 = undefined;
    for (&w) |*b| b.* = fp8_e5m2_half;
    var x: [13]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvFP8_E5M2(&x, &w, &y, 1, 13);
    try std.testing.expectApproxEqAbs(@as(f32, 6.5), y[0], 1e-5);
}

test "fuzz: all gemv_fp8 functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const k_choices = [_]usize{ 8, 9, 13, 16, 24, 32 };
            const n_raw = smith.valueWithHash(u8, 0);
            const k_idx = smith.valueWithHash(u8, 1);
            const n: usize = @as(usize, n_raw % 8) + 1;
            const k: usize = k_choices[k_idx % k_choices.len];

            var x_buf: [32]f32 = undefined;
            var w_buf: [8 * 32]u8 = undefined;
            var y_buf: [8]f32 = undefined;

            for (0..k) |i| {
                const bits = smith.valueWithHash(u32, @as(u32, @intCast(i)) +% 100);
                const raw: f32 = @bitCast(bits);
                x_buf[i] = if (std.math.isFinite(raw)) raw else 0.0;
            }
            for (0..n * k) |i| {
                w_buf[i] = smith.valueWithHash(u8, @as(u32, @intCast(i)) +% 1000);
            }

            // gemvFP8_E4M3
            @memset(y_buf[0..n], 0.0);
            gemvFP8_E4M3(&x_buf, &w_buf, &y_buf, n, k);
            for (0..n) |i| _ = y_buf[i];

            // gemvFP8_E5M2
            @memset(y_buf[0..n], 0.0);
            gemvFP8_E5M2(&x_buf, &w_buf, &y_buf, n, k);
            for (0..n) |i| _ = y_buf[i];
        }
    }.f, .{});
}
