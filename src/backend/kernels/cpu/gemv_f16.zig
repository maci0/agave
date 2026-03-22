//! CPU GEMV kernel for F16 weights.
//! 4-row batching with V8 SIMD.

const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);

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
            var w0: V8 = undefined;
            var w1: V8 = undefined;
            var w2: V8 = undefined;
            var w3: V8 = undefined;
            inline for (0..8) |idx| {
                w0[idx] = @floatCast(w[r0 + i + idx]);
                w1[idx] = @floatCast(w[r1 + i + idx]);
                w2[idx] = @floatCast(w[r2 + i + idx]);
                w3[idx] = @floatCast(w[r3 + i + idx]);
            }
            acc0 += xv * w0;
            acc1 += xv * w1;
            acc2 += xv * w2;
            acc3 += xv * w3;
        }
        var t0: f32 = 0.0;
        var t1: f32 = 0.0;
        var t2: f32 = 0.0;
        var t3: f32 = 0.0;
        while (i < k) : (i += 1) {
            const xv = x[i];
            t0 += xv * @as(f32, @floatCast(w[r0 + i]));
            t1 += xv * @as(f32, @floatCast(w[r1 + i]));
            t2 += xv * @as(f32, @floatCast(w[r2 + i]));
            t3 += xv * @as(f32, @floatCast(w[r3 + i]));
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
            var wv: V8 = undefined;
            inline for (0..8) |idx| {
                wv[idx] = @floatCast(w[roff + i + idx]);
            }
            acc += @as(V8, x[i..][0..8].*) * wv;
        }
        while (i < k) : (i += 1) tail += x[i] * @as(f32, @floatCast(w[roff + i]));
        y[row] = @reduce(.Add, acc) + tail;
    }
}
