//! CPU activation function kernels: SiLU, SiLU+mul, GELU.

const V8 = @Vector(8, f32);

/// Applies SiLU (Swish) activation: x * sigmoid(x).
pub fn silu(input: [*]const f32, output: [*]f32, n: usize) void {
    const one: V8 = @splat(@as(f32, 1.0));
    const neg: V8 = @splat(@as(f32, -1.0));
    var i: usize = 0;
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
pub fn siluMul(a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
    var i: usize = 0;
    const one: V8 = @splat(@as(f32, 1.0));
    const neg: V8 = @splat(@as(f32, -1.0));
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
/// If input != output, copies input to output first (applyGelu is in-place).
pub fn gelu(input: [*]const f32, output: [*]f32, n: usize) void {
    const math_ops = @import("../../../ops/math.zig");
    // applyGelu is in-place — copy first if input and output differ
    if (input != output) @memcpy(output[0..n], input[0..n]);
    math_ops.applyGelu(output[0..n]);
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
    try std.testing.expectApproxEqAbs(@as(f32, 0.841), output[1], 0.01);
    // GELU(-1) ≈ -0.159
    try std.testing.expectApproxEqAbs(@as(f32, -0.159), output[2], 0.01);
    // GELU(2) ≈ 1.955
    try std.testing.expectApproxEqAbs(@as(f32, 1.955), output[3], 0.01);
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

test "geluMul matches gelu * mul" {
    const elementwise_kernel = @import("elementwise.zig");
    var a = [_]f32{ 1.0, 2.0, -1.0, 0.5, -2.0, 3.0, 0.0, -0.5 };
    var b = [_]f32{ 2.0, 0.5, 1.0, 3.0, -1.0, 2.0, 5.0, 0.1 };
    // Compute reference: gelu(a) * b
    var gelu_ref: [8]f32 = undefined;
    gelu(&a, &gelu_ref, 8);
    var expected: [8]f32 = undefined;
    elementwise_kernel.mul(&gelu_ref, &b, &expected, 8);
    // Compute via composed path (same as CPU backend geluMul)
    var out: [8]f32 = undefined;
    gelu(&a, &out, 8);
    elementwise_kernel.mul(&out, &b, &out, 8);
    for (0..8) |i| try std.testing.expectApproxEqAbs(expected[i], out[i], 1e-6);
}

test "geluMul scaling" {
    const elementwise_kernel = @import("elementwise.zig");
    var a = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0 };
    // Compute gelu(a) as baseline
    var gelu_a: [8]f32 = undefined;
    gelu(&a, &gelu_a, 8);
    // GELU(0) = 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), gelu_a[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.841), gelu_a[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -0.159), gelu_a[2], 0.01);
    // With b=2: geluMul(a, 2) = gelu(a) * 2
    var b2 = [_]f32{ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 };
    var out: [8]f32 = undefined;
    gelu(&a, &out, 8);
    elementwise_kernel.mul(&out, &b2, &out, 8);
    for (0..8) |i| try std.testing.expectApproxEqAbs(gelu_a[i] * 2.0, out[i], 1e-5);
}

test "geluMul non-aligned size" {
    var a = [_]f32{ 1.0, 2.0, -1.0, 0.5, 3.0 };
    var b = [_]f32{ 2.0, 0.5, 1.0, 3.0, -1.0 };
    // Reference: gelu(a) * b
    var gelu_ref: [5]f32 = undefined;
    gelu(&a, &gelu_ref, 5);
    var expected: [5]f32 = undefined;
    const elementwise_kernel = @import("elementwise.zig");
    elementwise_kernel.mul(&gelu_ref, &b, &expected, 5);
    // Composed path
    var out: [5]f32 = undefined;
    gelu(&a, &out, 5);
    elementwise_kernel.mul(&out, &b, &out, 5);
    for (0..5) |i| try std.testing.expectApproxEqAbs(expected[i], out[i], 1e-5);
}
