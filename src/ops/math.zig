//! Shared mathematical utility functions for model implementations.
//! Provides common operations (argmax, softplus, sigmoid, GELU) used across
//! multiple model architectures, avoiding code duplication.

const std = @import("std");
const backend_mod = @import("../backend/backend.zig");
const Backend = backend_mod.Backend;
const TensorData = backend_mod.TensorData;

/// Constant for GELU tanh approximation: sqrt(2/pi).
const sqrt_2_over_pi: f32 = 0.7978845608028654;
/// Cubic coefficient in the GELU tanh approximation.
const gelu_coeff: f32 = 0.044715;
/// Softplus stability threshold: for x > this value, softplus(x) ≈ x.
const softplus_threshold: f32 = 20.0;
/// GELU tanh-argument clamp upper bound (prevents exp overflow).
const gelu_clamp_hi: f32 = 10.0;
/// GELU tanh-argument clamp lower bound (prevents exp overflow).
const gelu_clamp_lo: f32 = -10.0;
/// Maximum top-k value for stack-allocated selection buffer in sampleToken.
const max_top_k: usize = 1024;
/// Maximum candidates for top-p nucleus sampling buffer.
/// Caps the number of probabilities tracked during threshold computation.
const nucleus_max_candidates: usize = 1024;

/// Return the index of the maximum element in `buf`.
/// Single-pass scalar scan tracking both value and index.
///
/// Parameters:
///   - buf: Non-empty slice of f32 values to search.
///
/// Returns: Index of the maximum value as u32.
pub fn argmax(buf: []const f32) u32 {
    var best_idx: u32 = 0;
    var best_val: f32 = -std.math.inf(f32);
    for (buf, 0..) |v, i| {
        if (v > best_val) {
            best_val = v;
            best_idx = @intCast(i);
        }
    }
    return best_idx;
}

/// Select the top-k elements from `scores` by value.
/// Uses min-replacement: for each score, replaces the smallest current
/// top-k entry if the new score is larger. O(n*k), no heap allocation.
/// Output order is not sorted — callers that need sorted results must
/// sort the output arrays themselves.
///
/// Parameters:
///   - scores: Input scores to select from [n].
///   - k: Number of top elements to select.
///   - out_indices: Output buffer for selected indices (must have len >= k).
///   - out_scores: Output buffer for selected scores (must have len >= k).
pub fn topKExperts(
    scores: []const f32,
    k: usize,
    out_indices: []usize,
    out_scores: []f32,
) void {
    std.debug.assert(k > 0);
    std.debug.assert(out_indices.len >= k);
    std.debug.assert(out_scores.len >= k);
    for (0..k) |i| {
        out_scores[i] = -std.math.inf(f32);
        out_indices[i] = 0;
    }
    // Track min across iterations; rescan only after insertion.
    var min_idx: usize = 0;
    var min_val = out_scores[0];
    for (scores, 0..) |score, i| {
        if (score > min_val) {
            out_scores[min_idx] = score;
            out_indices[min_idx] = i;
            // Rescan for new min only when we insert
            min_idx = 0;
            min_val = out_scores[0];
            for (1..k) |j| {
                if (out_scores[j] < min_val) {
                    min_val = out_scores[j];
                    min_idx = j;
                }
            }
        }
    }
}

/// Numerically stable softplus: log(1 + exp(x)).
/// For large x (> 20), softplus(x) ≈ x (1 + exp(x) ≈ exp(x) in float precision).
///
/// Parameters:
///   - x: Input value.
///
/// Returns: softplus(x) = log(1 + exp(x)), clamped for numerical stability.
pub inline fn softplus(x: f32) f32 {
    return if (x > softplus_threshold) x else @log(1.0 + @exp(x));
}

/// Standard sigmoid activation: 1 / (1 + exp(-x)).
///
/// Parameters:
///   - x: Input value.
///
/// Returns: sigmoid(x) in range (0, 1).
pub inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x).
/// Scalar version for use in per-element loops (e.g., MoE expert paths).
pub inline fn silu(x: f32) f32 {
    return x * sigmoid(x);
}

/// Squared ReLU activation in-place: x[i] = max(0, x[i])².
/// SIMD-optimized with 8-wide vectors.
///
/// Parameters:
///   - x: Mutable slice of f32 values to transform in-place.
pub fn applyReluSquared(x: []f32) void {
    const V8 = @Vector(8, f32);
    const zero: V8 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) {
        const v: V8 = x[i..][0..8].*;
        const r = @max(v, zero);
        x[i..][0..8].* = r * r;
    }
    while (i < x.len) : (i += 1) {
        const v = @max(x[i], 0.0);
        x[i] = v * v;
    }
}

/// GELU (Gaussian Error Linear Unit) activation, applied in-place.
/// Uses the tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³))).
/// SIMD-optimized with 8-wide vectors; tanh computed via clamped exp to avoid overflow.
///
/// Parameters:
///   - x: Mutable slice of f32 values to transform in-place.
pub fn applyGelu(x: []f32) void {
    const V8 = @Vector(8, f32);
    const half: V8 = @splat(0.5);
    const one: V8 = @splat(1.0);
    const two: V8 = @splat(2.0);
    const coeff_v: V8 = @splat(gelu_coeff);
    const s2p_v: V8 = @splat(sqrt_2_over_pi);
    const clamp_hi: V8 = @splat(gelu_clamp_hi);
    const clamp_lo: V8 = @splat(gelu_clamp_lo);

    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) {
        const a: V8 = x[i..][0..8].*;
        const inner = s2p_v * @mulAdd(V8, coeff_v * a * a, a, a);
        // tanh via (exp(2x)-1)/(exp(2x)+1); x pre-clamped to [-10,10] to prevent exp overflow
        const clamped = @min(clamp_hi, @max(clamp_lo, inner));
        const e2 = @exp(two * clamped);
        x[i..][0..8].* = half * a * (one + (e2 - one) / (e2 + one));
    }
    while (i < x.len) : (i += 1) {
        const a = x[i];
        const inner = sqrt_2_over_pi * @mulAdd(f32, gelu_coeff * a * a, a, a);
        const clamped = @min(gelu_clamp_hi, @max(gelu_clamp_lo, inner));
        const e2 = @exp(2.0 * clamped);
        x[i] = 0.5 * a * (1.0 + (e2 - 1.0) / (e2 + 1.0));
    }
}

/// Compute final logits: RMS-norm the hidden state, project to vocab via LM head,
/// sync GPU, and return the argmax token ID.
///
/// Parameters:
///   - hidden: Hidden state buffer [n_embd], modified in-place by rmsNorm.
///   - norm_weight: RMS norm weight tensor data pointer.
///   - lm_head: LM head weight tensor (data + dtype).
///   - logits: Scratch buffer for logit output [vocab_size].
///   - vocab_size: Vocabulary size.
///   - n_embd: Embedding dimension.
///   - rms_eps: RMS norm epsilon.
///   - be: Backend for rmsNorm and gemv dispatch.
///
/// Returns: argmax token ID.
pub fn finalLogits(
    hidden: [*]f32,
    norm_weight: [*]const u8,
    lm_head: TensorData,
    logits: []f32,
    vocab_size: usize,
    n_embd: usize,
    rms_eps: f32,
    be: Backend,
) u32 {
    be.rmsNorm(hidden, @ptrCast(@alignCast(norm_weight)), hidden, n_embd, rms_eps);
    be.gemv(hidden, lm_head, logits.ptr, vocab_size, n_embd);
    be.sync(); // GPU gemv wrote logits — sync before CPU argmax
    return argmax(logits);
}

/// Sample a token from logits using temperature, top-k, and top-p (nucleus) filtering.
///
/// When temperature == 0, returns argmax (greedy). Otherwise:
///   1. Scale logits by 1/temperature.
///   2. If top_k > 0, keep only the top_k highest logits (rest set to -inf).
///   3. Softmax over remaining candidates.
///   4. If top_p < 1.0, keep smallest set of tokens with cumulative probability >= top_p.
///   5. Sample from the filtered distribution.
///
/// Modifies the logits buffer in-place.
pub fn sampleToken(logits: []f32, temperature: f32, top_k: u32, top_p: f32, rng: std.Random) u32 {
    if (temperature == 0) return argmax(logits);

    const n = logits.len;
    const neg_inf = -std.math.inf(f32);

    // 1. Temperature scaling
    const inv_temp = 1.0 / temperature;
    for (logits) |*v| v.* *= inv_temp;

    // 2. Top-k: find k-th largest value via min-replacement scan, mask the rest.
    // Tracks min index to avoid O(k) rescan per element → O(n) total instead of O(n*k).
    if (top_k > 0 and top_k < n) {
        const k: usize = top_k;
        var top_buf: [max_top_k]f32 = undefined;
        const buf_k = @min(k, max_top_k);
        for (0..buf_k) |i| top_buf[i] = neg_inf;
        // Track current min position in top_buf
        var mi: usize = 0;

        for (logits) |v| {
            if (v > top_buf[mi]) {
                top_buf[mi] = v;
                // Rescan for new min only when we insert
                mi = 0;
                for (1..buf_k) |j| {
                    if (top_buf[j] < top_buf[mi]) mi = j;
                }
            }
        }
        const top_min = top_buf[mi];
        for (logits) |*v| {
            if (v.* < top_min) v.* = neg_inf;
        }
    }

    // 3. Softmax
    var max_val: f32 = neg_inf;
    for (logits) |v| if (v > max_val) {
        max_val = v;
    };
    var sum: f32 = 0;
    for (logits) |*v| {
        v.* = if (v.* == neg_inf) 0 else @exp(v.* - max_val);
        sum += v.*;
    }
    const inv_sum = 1.0 / sum;
    for (logits) |*v| v.* *= inv_sum;

    // 4. Top-p (nucleus): threshold-based filtering — O(n) instead of O(n*k).
    // Collects top candidates via single scan, sorts them, finds the probability
    // threshold where cumulative probability >= top_p, then zeroes out below it.
    if (top_p < 1.0 and top_p > 0.0) {
        var top_vals: [nucleus_max_candidates]f32 = undefined;
        var n_top: usize = 0;
        var mi2: usize = 0;

        // Collect top-N probabilities via min-replacement scan (O(n))
        for (logits) |v| {
            if (v <= 0) continue;
            if (n_top < nucleus_max_candidates) {
                top_vals[n_top] = v;
                n_top += 1;
                if (n_top == nucleus_max_candidates) {
                    // Buffer just filled — find initial minimum
                    for (1..nucleus_max_candidates) |j| {
                        if (top_vals[j] < top_vals[mi2]) mi2 = j;
                    }
                }
            } else if (v > top_vals[mi2]) {
                top_vals[mi2] = v;
                mi2 = 0;
                for (1..nucleus_max_candidates) |j| {
                    if (top_vals[j] < top_vals[mi2]) mi2 = j;
                }
            }
        }

        // Sort candidates descending (insertion sort — small N, cache-friendly)
        var ci: usize = 1;
        while (ci < n_top) : (ci += 1) {
            const val = top_vals[ci];
            var j = ci;
            while (j > 0 and top_vals[j - 1] < val) {
                top_vals[j] = top_vals[j - 1];
                j -= 1;
            }
            top_vals[j] = val;
        }

        // Cumsum scan to find probability threshold
        var cumsum: f32 = 0;
        var threshold: f32 = 0;
        for (top_vals[0..n_top]) |v| {
            cumsum += v;
            if (cumsum >= top_p) {
                threshold = v;
                break;
            }
        }

        // Apply threshold: zero out tokens below cutoff, recompute sum
        sum = 0;
        for (logits) |*v| {
            if (v.* < threshold) {
                v.* = 0;
            } else {
                sum += v.*;
            }
        }
        if (sum > 0) {
            const inv = 1.0 / sum;
            for (logits) |*v| v.* *= inv;
        }
    }

    // 5. Weighted random sampling
    const r = rng.float(f32);
    var cumulative: f32 = 0;
    for (logits, 0..) |p, i| {
        cumulative += p;
        if (r < cumulative) return @intCast(i);
    }
    return @intCast(n - 1);
}

// ── Tests ─────────────────────────────────────────────────────────

test "argmax basic" {
    const buf = [_]f32{ 1.0, 3.0, 2.0, 0.5 };
    try std.testing.expectEqual(@as(u32, 1), argmax(&buf));
}

test "argmax single element" {
    const buf = [_]f32{42.0};
    try std.testing.expectEqual(@as(u32, 0), argmax(&buf));
}

test "topKExperts basic" {
    const scores = [_]f32{ 0.1, 0.9, 0.5, 0.3, 0.7, 0.2 };
    var indices: [8]usize = undefined;
    var values: [8]f32 = undefined;
    topKExperts(&scores, 3, indices[0..3], values[0..3]);

    // Top 3 should be indices 1 (0.9), 4 (0.7), 2 (0.5)
    var found = [_]bool{false} ** 6;
    for (0..3) |i| {
        found[indices[i]] = true;
        // Verify score values match the original scores
        try std.testing.expectApproxEqAbs(scores[indices[i]], values[i], 1e-6);
    }
    try std.testing.expect(found[1]);
    try std.testing.expect(found[4]);
    try std.testing.expect(found[2]);
    // Verify non-top indices were NOT selected
    try std.testing.expect(!found[0]); // 0.1
    try std.testing.expect(!found[3]); // 0.3
    try std.testing.expect(!found[5]); // 0.2
}

test "topKExperts single" {
    const scores = [_]f32{ 0.3, 0.1, 0.7 };
    var indices: [1]usize = undefined;
    var values: [1]f32 = undefined;
    topKExperts(&scores, 1, &indices, &values);
    try std.testing.expectEqual(@as(usize, 2), indices[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), values[0], 0.001);
}

test "softplus" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.6931), softplus(0.0), 0.001);
    // softplus(x) → x for large x
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), softplus(10.0), 0.001);
    // Large values should not overflow
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), softplus(100.0), 0.001);
    // At threshold boundary: softplus(20) ≈ 20.0 (linear regime)
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), softplus(softplus_threshold), 1e-4);
    // Just below threshold: still computed via log(1+exp(x))
    try std.testing.expectApproxEqAbs(@as(f32, 19.0), softplus(19.0), 0.01);
    // Negative value
    try std.testing.expectApproxEqAbs(@as(f32, 0.3133), softplus(-1.0), 0.001);
}

test "sigmoid" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid(0.0), 1e-6);
    // sigmoid(10) = 1/(1+exp(-10)) ≈ 0.9999546
    try std.testing.expectApproxEqAbs(@as(f32, 0.9999546), sigmoid(10.0), 1e-5);
    // sigmoid(-10) ≈ 0.0000454
    try std.testing.expectApproxEqAbs(@as(f32, 4.5397868e-5), sigmoid(-10.0), 1e-5);
}

test "applyReluSquared" {
    var buf = [_]f32{ -2.0, 0.0, 3.0, -1.0, 0.5, 4.0, -0.1, 2.0, 1.0 };
    applyReluSquared(&buf);
    // Negative values → 0
    try std.testing.expectEqual(@as(f32, 0.0), buf[0]);
    try std.testing.expectEqual(@as(f32, 0.0), buf[3]);
    try std.testing.expectEqual(@as(f32, 0.0), buf[6]);
    // Zero stays zero
    try std.testing.expectEqual(@as(f32, 0.0), buf[1]);
    // Positive values squared
    try std.testing.expectEqual(@as(f32, 9.0), buf[2]);
    try std.testing.expectEqual(@as(f32, 0.25), buf[4]);
    try std.testing.expectEqual(@as(f32, 16.0), buf[5]);
}

test "applyGelu" {
    // 10 elements: 8 via SIMD loop + 2 via scalar tail
    var buf = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0, 1.5 };
    applyGelu(&buf);
    // GELU(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), buf[0], 0.001);
    // GELU(1) ≈ 0.841
    try std.testing.expectApproxEqAbs(@as(f32, 0.841), buf[1], 0.01);
    // GELU(-1) ≈ -0.159
    try std.testing.expectApproxEqAbs(@as(f32, -0.159), buf[2], 0.01);
    // GELU(2) ≈ 1.955
    try std.testing.expectApproxEqAbs(@as(f32, 1.955), buf[3], 0.01);
    // Scalar tail: GELU(-3) ≈ -0.004 (near zero)
    try std.testing.expect(buf[8] < 0 and buf[8] > -0.01);
    // Scalar tail: GELU(1.5) ≈ 1.399
    try std.testing.expectApproxEqAbs(@as(f32, 1.399), buf[9], 0.01);
}

test "sampleToken greedy" {
    var logits = [_]f32{ 1.0, 5.0, 2.0, 0.5 };
    var prng = std.Random.DefaultPrng.init(42);
    // temperature=0 should return argmax regardless of RNG
    try std.testing.expectEqual(@as(u32, 1), sampleToken(&logits, 0, 0, 1.0, prng.random()));
}

test "sampleToken deterministic with seed" {
    // Same seed should produce same result
    var logits1 = [_]f32{ 1.0, 2.0, 3.0, 2.0 };
    var logits2 = [_]f32{ 1.0, 2.0, 3.0, 2.0 };
    var prng1 = std.Random.DefaultPrng.init(123);
    var prng2 = std.Random.DefaultPrng.init(123);
    try std.testing.expectEqual(
        sampleToken(&logits1, 1.0, 0, 1.0, prng1.random()),
        sampleToken(&logits2, 1.0, 0, 1.0, prng2.random()),
    );
}

test "sampleToken top_k filters" {
    // top_k=2 keeps only the two highest logits (indices 1=3.0 and 3=2.5)
    // Close values + temp=1.0 ensure both get sampled across many seeds
    var seen = [_]bool{false} ** 4;
    for (0..100) |seed| {
        var l = [_]f32{ 0.1, 3.0, 0.2, 2.5 };
        var p = std.Random.DefaultPrng.init(seed);
        seen[sampleToken(&l, 1.0, 2, 1.0, p.random())] = true;
    }
    try std.testing.expect(!seen[0]); // index 0 (0.1) filtered out
    try std.testing.expect(seen[1]); // index 1 (3.0) kept
    try std.testing.expect(!seen[2]); // index 2 (0.2) filtered out
    try std.testing.expect(seen[3]); // index 3 (2.5) kept
}
