//! CPU GEMV kernel for F16 weights.
//! 4-row batching with V8 SIMD.

const V8 = @Vector(8, f32);
const V8f16 = @Vector(8, f16);
const v8zero: V8 = @splat(0.0);

const std = @import("std");

/// F16 GEMV: y = W @ x. 4-row batched with inline f16→f32 conversion.
pub fn gemvF16(x: [*]const f32, w: [*]const f16, y: [*]f32, n: usize, k: usize) void {
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
            const w0: V8 = @floatCast(@as(V8f16, w[r0 + i ..][0..8].*));
            const w1: V8 = @floatCast(@as(V8f16, w[r1 + i ..][0..8].*));
            const w2: V8 = @floatCast(@as(V8f16, w[r2 + i ..][0..8].*));
            const w3: V8 = @floatCast(@as(V8f16, w[r3 + i ..][0..8].*));
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
            t0 = @mulAdd(f32, xv, @as(f32, @floatCast(w[r0 + i])), t0);
            t1 = @mulAdd(f32, xv, @as(f32, @floatCast(w[r1 + i])), t1);
            t2 = @mulAdd(f32, xv, @as(f32, @floatCast(w[r2 + i])), t2);
            t3 = @mulAdd(f32, xv, @as(f32, @floatCast(w[r3 + i])), t3);
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
            const wv: V8 = @floatCast(@as(V8f16, w[roff + i ..][0..8].*));
            acc = @mulAdd(V8, @as(V8, x[i..][0..8].*), wv, acc);
        }
        while (i < k) : (i += 1) tail = @mulAdd(f32, x[i], @as(f32, @floatCast(w[roff + i])), tail);
        y[row] = @reduce(.Add, acc) + tail;
    }
}

test "gemvF16 identity" {
    var w: [16]f16 = .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var y: [4]f32 = undefined;
    gemvF16(&x, &w, &y, 4, 4);
    for (0..4) |i| try std.testing.expectApproxEqAbs(x[i], y[i], 1e-3);
}

test "gemvF16 dense matrix" {
    var w: [16]f16 = .{
        1.0,  0.5,  2.0,  0.25,
        3.0,  1.0,  0.5,  0.75,
        0.5,  2.0,  1.0,  1.5,
        0.25, 0.75, 1.25, 2.0,
    };
    var x = [_]f32{ 2.0, 3.0, 1.0, 4.0 };
    var y: [4]f32 = undefined;
    gemvF16(&x, &w, &y, 4, 4);
    // y[0] = 2×1 + 3×0.5 + 1×2 + 4×0.25 = 6.5
    // y[1] = 2×3 + 3×1 + 1×0.5 + 4×0.75 = 12.5
    // y[2] = 2×0.5 + 3×2 + 1×1 + 4×1.5 = 14.0
    // y[3] = 2×0.25 + 3×0.75 + 1×1.25 + 4×2.0 = 12.0
    try std.testing.expectApproxEqAbs(@as(f32, 6.5), y[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 12.5), y[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), y[2], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), y[3], 0.01);
}

test "gemvF16 SIMD path with k=16" {
    // k=16 exercises the V8 SIMD inner loop (2 iterations) + 4-row batch (n=4).
    // W is diagonal-ish: W[row][row*4] = 2.0, rest 0 → y[row] = 2.0 * x[row*4].
    var w: [4 * 16]f16 = .{0.0} ** (4 * 16);
    w[0 * 16 + 0] = 2.0;
    w[1 * 16 + 4] = 2.0;
    w[2 * 16 + 8] = 2.0;
    w[3 * 16 + 12] = 2.0;
    var x: [16]f32 = .{0.0} ** 16;
    x[0] = 1.0;
    x[4] = 3.0;
    x[8] = 5.0;
    x[12] = 7.0;
    var y: [4]f32 = undefined;
    gemvF16(&x, &w, &y, 4, 16);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), y[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), y[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), y[2], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), y[3], 0.01);
}

test "gemvF16 non-aligned k exercises scalar tail" {
    var w: [15]f16 = .{
        1.0, 2.0, 3.0, 4.0, 5.0,
        0.5, 0.5, 0.5, 0.5, 0.5,
        0.0, 0.0, 0.0, 0.0, 10.0,
    };
    var x = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0 };
    var y: [3]f32 = undefined;
    gemvF16(&x, &w, &y, 3, 5);
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), y[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), y[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), y[2], 0.01);
}
