//! High-precision FP64 SDPA reference implementation for dual-delta tests.
//! Shared between Metal and CUDA SDPA correctness tests.

const std = @import("std");

/// Maximum sequence length supported by the oracle's stack-allocated scores buffer.
/// Must be ≥ the largest seq_len passed by any SDPA test (currently 256 in CUDA test).
const max_oracle_seq_len: usize = 8192;

test "oracle self-test 2x2 hand-computed" {
    // 1 head, hd=2, seq_len=2, scale=1.0
    // Q = [1, 0], K = [[1, 0], [0, 1]], V = [[1, 0], [0, 1]]
    // Scores = [Q·K0, Q·K1] = [1, 0] → softmax = [e^1/(e^1+e^0), e^0/(e^1+e^0)]
    // = [e/(1+e), 1/(1+e)] ≈ [0.7311, 0.2689]
    // Output = w0*V0 + w1*V1 = [0.7311*1 + 0.2689*0, 0.7311*0 + 0.2689*1]
    // = [0.7311, 0.2689]
    const q = [_]f32{ 1.0, 0.0 };
    const keys = [_]f32{ 1.0, 0.0, 0.0, 1.0 }; // 2 positions × 1 KV head × 2 dims
    const values = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var output: [2]f64 = undefined;
    try computeOracleSdpa(&q, &keys, &values, &output, 1, 1, 2, 2, 1.0);

    const e = @exp(@as(f64, 1.0));
    const w0 = e / (1.0 + e);
    const w1 = 1.0 / (1.0 + e);
    try std.testing.expectApproxEqAbs(w0, output[0], 1e-10);
    try std.testing.expectApproxEqAbs(w1, output[1], 1e-10);
}

test "oracle single position returns value" {
    // With 1 position, softmax([score]) = [1.0], so output = V exactly.
    // Tolerance is 1e-7 (f32 precision): input values are f32, so @floatCast
    // to f64 introduces ~1 ULP of f32 error (e.g., 0.7f32 → 0.69999998...f64).
    const q = [_]f32{ 0.5, 0.3 };
    const keys = [_]f32{ 1.0, 0.0 };
    const values = [_]f32{ 0.7, 0.9 };
    var output: [2]f64 = undefined;
    try computeOracleSdpa(&q, &keys, &values, &output, 1, 1, 2, 1, 1.0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7), output[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9), output[1], 1e-7);
}

test "oracle GQA heads share KV correctly" {
    // 2 query heads, 1 KV head (GQA ratio 2:1), hd=2, seq_len=2, scale=1.0.
    // Both Q heads should attend over the same single KV head.
    const q = [_]f32{ 1.0, 0.0, 0.0, 1.0 }; // Q head 0 = [1,0], Q head 1 = [0,1]
    const keys = [_]f32{ 1.0, 0.0, 0.0, 1.0 }; // 2 positions × 1 KV head × 2 dims
    const values = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var output: [4]f64 = undefined; // 2 heads × 2 dims
    try computeOracleSdpa(&q, &keys, &values, &output, 2, 1, 2, 2, 1.0);

    const e = @exp(@as(f64, 1.0));
    const w0 = e / (1.0 + e);
    const w1 = 1.0 / (1.0 + e);

    // Head 0: Q=[1,0], scores=[1,0] → weights=[w0,w1] → out=[w0,w1]
    try std.testing.expectApproxEqAbs(w0, output[0], 1e-10);
    try std.testing.expectApproxEqAbs(w1, output[1], 1e-10);

    // Head 1: Q=[0,1], scores=[0,1] → weights=[w1,w0] → out=[w1,w0]
    try std.testing.expectApproxEqAbs(w1, output[2], 1e-10);
    try std.testing.expectApproxEqAbs(w0, output[3], 1e-10);
}

test "oracle GQA 4:1 ratio maps heads correctly" {
    // 4 query heads, 1 KV head (GQA 4:1), hd=2, seq_len=1, scale=1.0.
    // With single position, softmax is [1.0], so output = V for all heads.
    const q = [_]f32{ 1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -1.0, 0.0 }; // 4 heads × 2 dims
    const keys = [_]f32{ 1.0, 0.0 }; // 1 position × 1 KV head × 2 dims
    const values = [_]f32{ 0.3, 0.7 };
    var output: [8]f64 = undefined; // 4 heads × 2 dims
    try computeOracleSdpa(&q, &keys, &values, &output, 4, 1, 2, 1, 1.0);

    // All 4 heads should produce V=[0.3, 0.7] since there's only 1 position
    for (0..4) |h| {
        try std.testing.expectApproxEqAbs(@as(f64, 0.3), output[h * 2 + 0], 1e-7);
        try std.testing.expectApproxEqAbs(@as(f64, 0.7), output[h * 2 + 1], 1e-7);
    }
}

test "oracle no GQA (nh == nkv) each head independent" {
    // 2 query heads, 2 KV heads (no GQA), hd=2, seq_len=1, scale=1.0.
    // Each head attends to its own KV head with different V.
    const q = [_]f32{ 1.0, 0.0, 0.0, 1.0 }; // 2 Q heads × 2 dims
    const keys = [_]f32{ 1.0, 0.0, 0.0, 1.0 }; // 1 position × 2 KV heads × 2 dims
    const values = [_]f32{ 0.1, 0.2, 0.8, 0.9 }; // 1 position × 2 KV heads × 2 dims
    var output: [4]f64 = undefined;
    try computeOracleSdpa(&q, &keys, &values, &output, 2, 2, 2, 1, 1.0);

    // Head 0 uses KV head 0 → V=[0.1, 0.2]
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), output[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), output[1], 1e-7);
    // Head 1 uses KV head 1 → V=[0.8, 0.9]
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), output[2], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9), output[3], 1e-7);
}

test "oracle scale factor affects attention sharpness" {
    // All existing oracle tests use scale=1.0. GPU tests use scale=1/√hd.
    // Verify that scale sharpens attention when > 1.0 and softens when < 1.0.
    const q = [_]f32{ 1.0, 0.0 };
    const keys = [_]f32{ 1.0, 0.0, 0.0, 1.0 }; // 2 positions × 1 KV head × 2 dims
    const values = [_]f32{ 1.0, 0.0, 0.0, 1.0 };

    // Baseline: scale=1.0
    var out_base: [2]f64 = undefined;
    try computeOracleSdpa(&q, &keys, &values, &out_base, 1, 1, 2, 2, 1.0);

    // Sharp: scale=4.0 → scores=[4, 0] → softmax more peaked toward position 0
    var out_sharp: [2]f64 = undefined;
    try computeOracleSdpa(&q, &keys, &values, &out_sharp, 1, 1, 2, 2, 4.0);

    // Soft: scale=0.1 → scores=[0.1, 0] → softmax nearly uniform
    var out_soft: [2]f64 = undefined;
    try computeOracleSdpa(&q, &keys, &values, &out_soft, 1, 1, 2, 2, 0.1);

    // Sharp should weight V[0]=[1,0] more heavily → out[0] closer to 1.0
    try std.testing.expect(out_sharp[0] > out_base[0]);
    // Soft should distribute more evenly → out[0] closer to 0.5
    try std.testing.expect(out_soft[0] < out_base[0]);

    // Verify scale=0.1 gives near-uniform attention (both outputs ≈ 0.5)
    // With Q·K=[1,0], scores=[0.1,0], softmax is not exactly uniform:
    // w0=exp(0.1)/(exp(0.1)+1) ≈ 0.525, w1 ≈ 0.475
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), out_soft[0], 0.03);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), out_soft[1], 0.03);

    // Verify scale=4.0 gives peaked attention (out[0] ≈ 1.0)
    try std.testing.expect(out_sharp[0] > 0.98);
}

/// Compute scaled dot-product attention in FP64 for use as a ground-truth oracle.
pub fn computeOracleSdpa(
    q: []const f32,
    keys: []const f32,
    values: []const f32,
    output: []f64,
    nh: usize,
    nkv: usize,
    hd: usize,
    seq_len: usize,
    scale: f32,
) error{SeqLenExceedsOracle}!void {
    if (seq_len > max_oracle_seq_len) return error.SeqLenExceedsOracle;
    std.debug.assert(q.len >= nh * hd);
    std.debug.assert(keys.len >= seq_len * nkv * hd);
    std.debug.assert(values.len >= seq_len * nkv * hd);
    std.debug.assert(output.len >= nh * hd);
    std.debug.assert(nh % nkv == 0);
    const hpg = nh / nkv;
    for (0..nh) |h| {
        const kvh = h / hpg;

        // Compute attention scores once per head (independent of output dimension)
        var scores: [max_oracle_seq_len]f64 = undefined;
        var max_score: f64 = -std.math.inf(f64);
        for (0..seq_len) |t| {
            var dot: f64 = 0.0;
            for (0..hd) |k| {
                const q_val: f64 = @floatCast(q[h * hd + k]);
                const k_val: f64 = @floatCast(keys[t * nkv * hd + kvh * hd + k]);
                dot += q_val * k_val;
            }
            scores[t] = dot * @as(f64, @floatCast(scale));
            max_score = @max(max_score, scores[t]);
        }

        // Softmax
        var exp_sum: f64 = 0.0;
        for (0..seq_len) |t| {
            scores[t] = @exp(scores[t] - max_score);
            exp_sum += scores[t];
        }

        // Weighted sum over values for each output dimension
        for (0..hd) |d| {
            var sum: f64 = 0.0;
            for (0..seq_len) |t| {
                const weight = scores[t] / exp_sum;
                const v_val: f64 = @floatCast(values[t * nkv * hd + kvh * hd + d]);
                sum += weight * v_val;
            }
            output[h * hd + d] = sum;
        }
    }
}
