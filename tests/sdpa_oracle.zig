//! High-precision FP64 SDPA reference implementation for dual-delta tests.
//! Shared between Metal and CUDA SDPA correctness tests.

const std = @import("std");

/// Maximum sequence length supported by the oracle's stack-allocated scores buffer.
/// Must be ≥ the largest seq_len passed by any SDPA test (currently 256 in CUDA test).
const max_oracle_seq_len: usize = 8192;

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
) void {
    const hpg = nh / nkv;
    for (0..nh) |h| {
        const kvh = h / hpg;
        for (0..hd) |d| {
            var sum: f64 = 0.0;
            var max_score: f64 = -std.math.inf(f64);

            // Compute scores and find max
            var scores: [max_oracle_seq_len]f64 = undefined;
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

            // Softmax numerator and sum
            var exp_sum: f64 = 0.0;
            for (0..seq_len) |t| {
                scores[t] = @exp(scores[t] - max_score);
                exp_sum += scores[t];
            }

            // Weighted sum over values
            for (0..seq_len) |t| {
                const weight = scores[t] / exp_sum;
                const v_val: f64 = @floatCast(values[t * nkv * hd + kvh * hd + d]);
                sum += weight * v_val;
            }
            output[h * hd + d] = sum;
        }
    }
}
