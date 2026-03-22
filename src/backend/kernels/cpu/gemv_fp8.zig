//! CPU GEMV kernels for FP8 quantization formats.
//! FP8_E4M3 (4 exponent, 3 mantissa) and FP8_E5M2 (5 exponent, 2 mantissa).

const quant = @import("../../../ops/quant.zig");
const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);

/// FP8_E4M3: 1 byte per element (4 exponent, 3 mantissa, bias=7)
pub fn gemvFP8_E4M3(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    for (0..n) |row| {
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
            acc += xv * wv;
        }
        while (i < k) : (i += 1) tail += x[i] * quant.fp8e4m3ToF32(w[roff + i]);
        y[row] = @reduce(.Add, acc) + tail;
    }
}

/// FP8_E5M2: 1 byte per element (5 exponent, 2 mantissa, bias=15)
pub fn gemvFP8_E5M2(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    for (0..n) |row| {
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
            acc += xv * wv;
        }
        while (i < k) : (i += 1) tail += x[i] * quant.fp8e5m2ToF32(w[roff + i]);
        y[row] = @reduce(.Add, acc) + tail;
    }
}
