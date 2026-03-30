//! CPU GEMV kernels for FP8 quantization formats.
//! FP8_E4M3 (4 exponent, 3 mantissa) and FP8_E5M2 (5 exponent, 2 mantissa).
//! 4-row batching with V8 SIMD, matching the F32/F16/BF16 kernel structure.

const quant = @import("../../../ops/quant.zig");
const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);

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
        while (i + 8 <= k) : (i += 8) {
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
        while (i + 8 <= k) : (i += 8) {
            const xv: V8 = x[i..][0..8].*;
            var wv: V8 = undefined;
            inline for (0..8) |idx| {
                wv[idx] = quant.fp8e4m3ToF32(w[roff + i + idx]);
            }
            acc = @mulAdd(V8, xv, wv, acc);
        }
        while (i < k) : (i += 1) tail = @mulAdd(f32, x[i], quant.fp8e4m3ToF32(w[roff + i]), tail);
        y[row] = @reduce(.Add, acc) + tail;
    }
}

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
        while (i + 8 <= k) : (i += 8) {
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
        while (i + 8 <= k) : (i += 8) {
            const xv: V8 = x[i..][0..8].*;
            var wv: V8 = undefined;
            inline for (0..8) |idx| {
                wv[idx] = quant.fp8e5m2ToF32(w[roff + i + idx]);
            }
            acc = @mulAdd(V8, xv, wv, acc);
        }
        while (i < k) : (i += 1) tail = @mulAdd(f32, x[i], quant.fp8e5m2ToF32(w[roff + i]), tail);
        y[row] = @reduce(.Add, acc) + tail;
    }
}
