//! CPU GEMV kernel for BF16 weights.
//! 4-row batching with V8 SIMD.

const quant = @import("../../../ops/quant.zig");
const V8 = @Vector(8, f32);
const V8u16 = @Vector(8, u16);
const V8u32 = @Vector(8, u32);
const bf16_shift: @Vector(8, u5) = @splat(16);
const v8zero: V8 = @splat(0.0);

/// BF16 GEMV: y = W @ x. 4-row batched with bf16→f32 conversion.
pub fn gemvBF16(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const w16: [*]const u16 = @ptrCast(@alignCast(w));
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
            const w0: V8 = @bitCast(@as(V8u32, @intCast(@as(V8u16, w16[r0 + i ..][0..8].*))) << bf16_shift);
            const w1: V8 = @bitCast(@as(V8u32, @intCast(@as(V8u16, w16[r1 + i ..][0..8].*))) << bf16_shift);
            const w2: V8 = @bitCast(@as(V8u32, @intCast(@as(V8u16, w16[r2 + i ..][0..8].*))) << bf16_shift);
            const w3: V8 = @bitCast(@as(V8u32, @intCast(@as(V8u16, w16[r3 + i ..][0..8].*))) << bf16_shift);
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
            t0 = @mulAdd(f32, xv, quant.bf16ToF32(w16[r0 + i]), t0);
            t1 = @mulAdd(f32, xv, quant.bf16ToF32(w16[r1 + i]), t1);
            t2 = @mulAdd(f32, xv, quant.bf16ToF32(w16[r2 + i]), t2);
            t3 = @mulAdd(f32, xv, quant.bf16ToF32(w16[r3 + i]), t3);
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
            const wv: V8 = @bitCast(@as(V8u32, @intCast(@as(V8u16, w16[roff + i ..][0..8].*))) << bf16_shift);
            acc = @mulAdd(V8, xv, wv, acc);
        }
        while (i < k) : (i += 1) tail = @mulAdd(f32, x[i], quant.bf16ToF32(w16[roff + i]), tail);
        y[row] = @reduce(.Add, acc) + tail;
    }
}

const std = @import("std");

/// Convert f32 to bf16 (truncate lower 16 bits).
fn f32ToBf16(val: f32) u16 {
    return @truncate(@as(u32, @bitCast(val)) >> 16);
}

test "gemvBF16 identity" {
    // W = I (4×4 identity in bf16), y = x.
    var w16 align(2) = [_]u16{
        f32ToBf16(1.0), f32ToBf16(0.0), f32ToBf16(0.0), f32ToBf16(0.0),
        f32ToBf16(0.0), f32ToBf16(1.0), f32ToBf16(0.0), f32ToBf16(0.0),
        f32ToBf16(0.0), f32ToBf16(0.0), f32ToBf16(1.0), f32ToBf16(0.0),
        f32ToBf16(0.0), f32ToBf16(0.0), f32ToBf16(0.0), f32ToBf16(1.0),
    };
    const w_bytes: [*]const u8 = @ptrCast(&w16);
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var y: [4]f32 = undefined;
    gemvBF16(&x, w_bytes, &y, 4, 4);
    for (0..4) |i| try std.testing.expectApproxEqAbs(x[i], y[i], 0.01);
}

test "gemvBF16 scaling" {
    // W = 2*I → y = 2*x.
    var w16 align(2) = [_]u16{
        f32ToBf16(2.0), f32ToBf16(0.0), f32ToBf16(0.0), f32ToBf16(0.0),
        f32ToBf16(0.0), f32ToBf16(2.0), f32ToBf16(0.0), f32ToBf16(0.0),
        f32ToBf16(0.0), f32ToBf16(0.0), f32ToBf16(2.0), f32ToBf16(0.0),
        f32ToBf16(0.0), f32ToBf16(0.0), f32ToBf16(0.0), f32ToBf16(2.0),
    };
    const w_bytes: [*]const u8 = @ptrCast(&w16);
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var y: [4]f32 = undefined;
    gemvBF16(&x, w_bytes, &y, 4, 4);
    const expected = [_]f32{ 2.0, 4.0, 6.0, 8.0 };
    for (0..4) |i| try std.testing.expectApproxEqAbs(expected[i], y[i], 0.02);
}

test "gemvBF16 SIMD path with k=16" {
    // k=16 exercises the V8 SIMD inner loop (2 iterations) + 4-row batch (n=4).
    // W[row][col] = row+1 for col=0, 0 elsewhere → y[row] = (row+1) * x[0].
    var w16: [4 * 16]u16 align(2) = undefined;
    for (0..4) |row| {
        for (0..16) |col| {
            w16[row * 16 + col] = if (col == 0) f32ToBf16(@floatFromInt(row + 1)) else f32ToBf16(0.0);
        }
    }
    const w_bytes: [*]const u8 = @ptrCast(&w16);
    var x: [16]f32 = undefined;
    for (0..16) |i| x[i] = if (i == 0) 3.0 else 0.0;
    var y: [4]f32 = undefined;
    gemvBF16(&x, w_bytes, &y, 4, 16);
    for (0..4) |i| {
        const expected: f32 = @as(f32, @floatFromInt(i + 1)) * 3.0;
        try std.testing.expectApproxEqAbs(expected, y[i], 0.05);
    }
}

test "gemvBF16 non-aligned k exercises scalar tail" {
    // k=5, n=3: exercises scalar tail (k not multiple of 8) and single-row tail (n not multiple of 4).
    var w16 align(2) = [_]u16{
        f32ToBf16(1.0), f32ToBf16(2.0), f32ToBf16(3.0), f32ToBf16(4.0), f32ToBf16(5.0),
        f32ToBf16(0.5), f32ToBf16(0.5), f32ToBf16(0.5), f32ToBf16(0.5), f32ToBf16(0.5),
        f32ToBf16(0.0), f32ToBf16(0.0), f32ToBf16(0.0), f32ToBf16(0.0), f32ToBf16(10.0),
    };
    const w_bytes: [*]const u8 = @ptrCast(&w16);
    var x = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0 };
    var y: [3]f32 = undefined;
    gemvBF16(&x, w_bytes, &y, 3, 5);
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), y[0], 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), y[1], 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), y[2], 0.1);
}
