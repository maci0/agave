//! CPU activation function kernels: SiLU, SiLU+mul, GELU.

const V8 = @Vector(8, f32);
const math_ops = @import("../../../ops/math.zig");

const sqrt_2_over_pi = math_ops.sqrt_2_over_pi;
const gelu_coeff = math_ops.gelu_coeff;
const gelu_clamp_hi = math_ops.gelu_clamp_hi;
const gelu_clamp_lo = math_ops.gelu_clamp_lo;

/// Applies SiLU (Swish) activation: x * sigmoid(x).
/// Dual-vector processing hides exp() latency (~4-7 cycles) by overlapping two independent chains.
pub fn silu(input: [*]const f32, output: [*]f32, n: usize) void {
    const one: V8 = @splat(1.0);
    const neg: V8 = @splat(-1.0);
    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) {
        const x0: V8 = input[i..][0..8].*;
        const x1: V8 = input[i + 8 ..][0..8].*;
        output[i..][0..8].* = x0 / (one + @exp(neg * x0));
        output[i + 8 ..][0..8].* = x1 / (one + @exp(neg * x1));
    }
    while (i + 8 <= n) : (i += 8) {
        const x: V8 = input[i..][0..8].*;
        output[i..][0..8].* = x / (one + @exp(neg * x));
    }
    while (i < n) : (i += 1) {
        const x = input[i];
        output[i] = x / (1.0 + @exp(-x));
    }
}

/// Fused SiLU + multiply: out[i] = silu(a[i]) * b[i].
/// Dual-vector processing hides exp() latency by overlapping two independent chains.
pub fn siluMul(a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
    var i: usize = 0;
    const one: V8 = @splat(1.0);
    const neg: V8 = @splat(-1.0);
    while (i + 16 <= n) : (i += 16) {
        const x0: V8 = a[i..][0..8].*;
        const y0: V8 = b[i..][0..8].*;
        const x1: V8 = a[i + 8 ..][0..8].*;
        const y1: V8 = b[i + 8 ..][0..8].*;
        out[i..][0..8].* = (x0 / (one + @exp(neg * x0))) * y0;
        out[i + 8 ..][0..8].* = (x1 / (one + @exp(neg * x1))) * y1;
    }
    while (i + 8 <= n) : (i += 8) {
        const x: V8 = a[i..][0..8].*;
        const y: V8 = b[i..][0..8].*;
        out[i..][0..8].* = (x / (one + @exp(neg * x))) * y;
    }
    while (i < n) : (i += 1) {
        const x = a[i];
        out[i] = (x / (1.0 + @exp(-x))) * b[i];
    }
}

/// Applies GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))).
/// If input != output, copies input to output first (underlying impl is in-place).
pub fn gelu(input: [*]const f32, output: [*]f32, n: usize) void {
    // applyGelu is in-place — copy first if input and output differ
    if (input != output) @memcpy(output[0..n], input[0..n]);
    math_ops.applyGelu(output[0..n]);
}

/// Fused GELU + multiply: out[i] = gelu(a[i]) * b[i].
/// Single-pass SIMD avoids a second cache traversal compared to gelu() + mul().
/// Dual-vector processing hides exp() latency by overlapping two independent chains.
pub fn geluMul(a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
    const half: V8 = @splat(0.5);
    const one: V8 = @splat(1.0);
    const two: V8 = @splat(2.0);
    const coeff_v: V8 = @splat(gelu_coeff);
    const sqrt_v: V8 = @splat(sqrt_2_over_pi);
    const clamp_hi: V8 = @splat(gelu_clamp_hi);
    const clamp_lo: V8 = @splat(gelu_clamp_lo);

    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) {
        const x0: V8 = a[i..][0..8].*;
        const y0: V8 = b[i..][0..8].*;
        const x1: V8 = a[i + 8 ..][0..8].*;
        const y1: V8 = b[i + 8 ..][0..8].*;
        const inner0 = @mulAdd(V8, coeff_v, x0 * x0 * x0, x0);
        const inner1 = @mulAdd(V8, coeff_v, x1 * x1 * x1, x1);
        const t0 = @min(clamp_hi, @max(clamp_lo, sqrt_v * inner0));
        const t1 = @min(clamp_hi, @max(clamp_lo, sqrt_v * inner1));
        const e2t0 = @exp(two * t0);
        const e2t1 = @exp(two * t1);
        const tanh0 = (e2t0 - one) / (e2t0 + one);
        const tanh1 = (e2t1 - one) / (e2t1 + one);
        out[i..][0..8].* = half * x0 * (one + tanh0) * y0;
        out[i + 8 ..][0..8].* = half * x1 * (one + tanh1) * y1;
    }
    while (i + 8 <= n) : (i += 8) {
        const x: V8 = a[i..][0..8].*;
        const y: V8 = b[i..][0..8].*;
        const inner = @mulAdd(V8, coeff_v, x * x * x, x);
        const t = @min(clamp_hi, @max(clamp_lo, sqrt_v * inner));
        const e2t = @exp(two * t);
        const tanh_v = (e2t - one) / (e2t + one);
        out[i..][0..8].* = half * x * (one + tanh_v) * y;
    }
    // Scalar tail
    while (i < n) : (i += 1) {
        const x = a[i];
        const inner = @mulAdd(f32, gelu_coeff, x * x * x, x);
        const t = @min(gelu_clamp_hi, @max(gelu_clamp_lo, sqrt_2_over_pi * inner));
        const e2t = @exp(2.0 * t);
        const tanh_v = (e2t - 1.0) / (e2t + 1.0);
        out[i] = 0.5 * x * (1.0 + tanh_v) * b[i];
    }
}

const std = @import("std");

test "silu zero is zero" {
    var input = [_]f32{0.0};
    var output: [1]f32 = undefined;
    silu(&input, &output, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 1e-6);
}

test "silu positive" {
    var input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var output: [8]f32 = undefined;
    silu(&input, &output, 8);
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), output[0], 0.001);
    // Verify all outputs are positive and increasing (silu is monotonic for x > 0)
    for (1..8) |i| try std.testing.expect(output[i] > output[i - 1]);
    // silu(2) ≈ 1.762
    try std.testing.expectApproxEqAbs(@as(f32, 1.762), output[1], 0.001);
}

test "silu negative" {
    var input = [_]f32{ -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0 };
    var output: [8]f32 = undefined;
    silu(&input, &output, 8);
    // silu(-1) ≈ -0.269
    try std.testing.expectApproxEqAbs(@as(f32, -0.269), output[0], 0.001);
    // All negative outputs
    for (0..8) |i| try std.testing.expect(output[i] < 0);
}

test "siluMul" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var b = [_]f32{ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 };
    var out: [8]f32 = undefined;
    siluMul(&a, &b, &out, 8);
    // siluMul(x, 2) = silu(x) * 2
    try std.testing.expectApproxEqAbs(@as(f32, 1.4621), out[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.524), out[1], 0.001);
    // Verify relationship: siluMul(x, b) = silu(x) * b
    var silu_only: [8]f32 = undefined;
    silu(&a, &silu_only, 8);
    for (0..8) |i| try std.testing.expectApproxEqAbs(silu_only[i] * 2.0, out[i], 1e-5);
}

test "gelu" {
    var input = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0 };
    var output: [8]f32 = undefined;
    gelu(&input, &output, 8);
    // GELU(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 0.001);
    // GELU(1) ≈ 0.841
    try std.testing.expectApproxEqAbs(@as(f32, 0.841), output[1], 0.001);
    // GELU(-1) ≈ -0.159
    try std.testing.expectApproxEqAbs(@as(f32, -0.159), output[2], 0.001);
    // GELU(2) ≈ 1.955
    try std.testing.expectApproxEqAbs(@as(f32, 1.955), output[3], 0.001);
}

test "silu non-aligned size exercises scalar tail" {
    // n=3 (not a multiple of 8) exercises the scalar cleanup loop
    var input = [_]f32{ 0.0, 1.0, -1.0 };
    var output: [3]f32 = undefined;
    silu(&input, &output, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.2689), output[2], 0.001);
}

test "siluMul non-aligned size" {
    var a = [_]f32{ 1.0, 2.0, -1.0, 0.5, 3.0 };
    var b = [_]f32{ 2.0, 0.5, 1.0, 3.0, -1.0 };
    var out: [5]f32 = undefined;
    siluMul(&a, &b, &out, 5);
    // Verify against reference: siluMul(x,y) = silu(x) * y
    var silu_ref: [5]f32 = undefined;
    silu(&a, &silu_ref, 5);
    for (0..5) |i| try std.testing.expectApproxEqAbs(silu_ref[i] * b[i], out[i], 1e-5);
}

test "geluMul fused kernel matches gelu then mul" {
    // Verify the fused geluMul() kernel produces the same result as separate gelu() + mul().
    var a = [_]f32{ 1.0, 2.0, -1.0, 0.5, -2.0, 3.0, 0.0, -0.5 };
    var b = [_]f32{ 2.0, 0.5, 1.0, 3.0, -1.0, 2.0, 5.0, 0.1 };
    var fused_out: [8]f32 = undefined;
    geluMul(&a, &b, &fused_out, 8);
    // Compute reference via separate gelu() + mul()
    var ref_out: [8]f32 = undefined;
    gelu(&a, &ref_out, 8);
    const elementwise_kernel = @import("elementwise.zig");
    elementwise_kernel.mul(&ref_out, &b, &ref_out, 8);
    for (0..8) |i| try std.testing.expectApproxEqAbs(ref_out[i], fused_out[i], 1e-5);
}

test "gelu known values" {
    var a = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0 };
    gelu(&a, &a, 8);
    // Reference: torch.nn.functional.gelu(tensor)
    const expected = [_]f32{ 0.0, 0.8412, -0.1588, 1.9545, -0.0455, 0.3457, -0.1543, 2.9960 };
    for (0..8) |i| try std.testing.expectApproxEqAbs(expected[i], a[i], 0.001);
}

test "geluMul non-aligned size exercises scalar tail" {
    // n=5: exercises scalar tail path in fused geluMul (no 8-wide SIMD)
    var a = [_]f32{ 1.0, 2.0, -1.0, 0.5, 3.0 };
    var b = [_]f32{ 2.0, 0.5, 1.0, 3.0, -1.0 };
    var fused_out: [5]f32 = undefined;
    geluMul(&a, &b, &fused_out, 5);
    // Cross-reference against separate gelu() + mul()
    var ref_out: [5]f32 = undefined;
    gelu(&a, &ref_out, 5);
    const elementwise_kernel = @import("elementwise.zig");
    elementwise_kernel.mul(&ref_out, &b, &ref_out, 5);
    for (0..5) |i| try std.testing.expectApproxEqAbs(ref_out[i], fused_out[i], 1e-5);
}
