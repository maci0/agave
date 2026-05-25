//! Math primitives (argmax, softplus, sigmoid, GELU), sampling strategies
//! (top-k/top-p/min-p/XTC/Mirostat), repetition penalties (frequency/presence/DRY),
//! and log-probability extraction — shared across model architectures and the HTTP server.

const std = @import("std");

/// Constant for GELU tanh approximation: sqrt(2/pi).
pub const sqrt_2_over_pi: f32 = 0.7978845608028654;
/// Cubic coefficient in the GELU tanh approximation.
pub const gelu_coeff: f32 = 0.044715;
/// Softplus stability threshold: for x > this value, softplus(x) ≈ x.
const softplus_threshold: f32 = 20.0;
/// GELU tanh-argument clamp upper bound (prevents exp overflow).
pub const gelu_clamp_hi: f32 = 10.0;
/// GELU tanh-argument clamp lower bound (prevents exp overflow).
pub const gelu_clamp_lo: f32 = -10.0;
/// Maximum top-k value for stack-allocated selection buffer in sampleToken.
pub const max_top_k: usize = 1024;
/// Maximum candidates for top-p nucleus sampling buffer.
/// Caps the number of probabilities tracked during threshold computation.
const nucleus_max_candidates: usize = 1024;
/// Maximum top-N for stack-allocated logprob selection buffer in topLogProbs.
pub const max_top_logprobs: usize = 20;
/// Minimum probability floor for log-probability computation (prevents -inf from log(0)).
const log_prob_epsilon: f32 = 1e-10;
/// 8-wide SIMD vector type for f32 — used across all SIMD helpers in this module.
const V8 = @Vector(8, f32);

/// SIMD max-reduce over f32 slice. Used by argmax, softmax, log-sum-exp, and sampling.
inline fn simdMaxF32(buf: []const f32) f32 {
    var max_v: V8 = @splat(-std.math.inf(f32));
    var i: usize = 0;
    while (i + 8 <= buf.len) : (i += 8) {
        max_v = @max(max_v, @as(V8, buf[i..][0..8].*));
    }
    var m = @reduce(.Max, max_v);
    while (i < buf.len) : (i += 1) m = @max(m, buf[i]);
    return m;
}

/// Return index of maximum element (first occurrence on ties).
pub fn argmax(buf: []const f32) u32 {
    const best_val = simdMaxF32(buf);
    for (buf, 0..) |v, idx| {
        if (v >= best_val) return @intCast(idx);
    }
    return 0;
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

/// SIMD dot product of two f32 slices.
pub inline fn simdDotF32(a: [*]const f32, b: [*]const f32, n: usize) f32 {
    var acc: V8 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        acc = @mulAdd(V8, @as(V8, a[i..][0..8].*), @as(V8, b[i..][0..8].*), acc);
    }
    var s = @reduce(.Add, acc);
    while (i < n) : (i += 1) s += a[i] * b[i];
    return s;
}

/// SIMD in-place uniform scale: buf[i] *= scale.
pub inline fn simdScaleF32(buf: [*]f32, scale: f32, n: usize) void {
    const sv: V8 = @splat(scale);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        buf[i..][0..8].* = @as(V8, buf[i..][0..8].*) * sv;
    }
    while (i < n) : (i += 1) buf[i] *= scale;
}

/// Numerically stable softplus: log(1 + exp(x)).
/// For large x (> 20), softplus(x) ≈ x since 1 + exp(x) ≈ exp(x) in float precision.
pub inline fn softplus(x: f32) f32 {
    return if (x > softplus_threshold) x else @log(1.0 + @exp(x));
}

/// Standard sigmoid activation: 1 / (1 + exp(-x)).
pub inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x).
/// Scalar version for use in per-element loops (e.g., MoE expert paths).
pub inline fn silu(x: f32) f32 {
    return x * sigmoid(x);
}

/// Squared ReLU activation in-place: x[i] = max(0, x[i])². SIMD-optimized.
pub fn applyReluSquared(x: []f32) void {
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

/// GELU activation in-place (tanh approximation), SIMD-optimized.
/// Tanh computed via clamped exp to avoid overflow.
pub fn applyGelu(x: []f32) void {
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
        const clamped = @min(clamp_hi, @max(clamp_lo, inner));
        const e2 = @exp(two * clamped);
        x[i..][0..8].* = a - a / (e2 + one);
    }
    while (i < x.len) : (i += 1) {
        const a = x[i];
        const inner = sqrt_2_over_pi * @mulAdd(f32, gelu_coeff * a * a, a, a);
        const clamped = @min(gelu_clamp_hi, @max(gelu_clamp_lo, inner));
        const e2 = @exp(2.0 * clamped);
        x[i] = a - a / (e2 + 1.0);
    }
}

/// Apply repetition penalty: divide positive logits, multiply negative by `penalty`.
/// Standard repeat penalty (Keskar et al. 2019). penalty > 1.0 = more suppression.
pub fn applyRepeatPenalty(logits: []f32, recent_ids: []const u32, penalty: f32) void {
    std.debug.assert(penalty > 0);
    for (recent_ids) |tok_id| {
        if (tok_id < logits.len) {
            if (logits[tok_id] > 0) {
                logits[tok_id] /= penalty;
            } else {
                logits[tok_id] *= penalty;
            }
        }
    }
}

/// Apply logit bias: add bias values to specific token logits.
/// OpenAI API `logit_bias` parameter: {"token_id": bias, ...}.
pub fn applyLogitBias(logits: []f32, ids: []const u32, biases: []const f32, count: u32) void {
    for (0..count) |i| {
        if (ids[i] < logits.len) logits[ids[i]] += biases[i];
    }
}

/// DRY (Don't Repeat Yourself) sampling: penalize tokens that would continue
/// a repeated n-gram sequence. For each candidate token, check if appending it
/// would create an n-gram that already appeared in the recent output.
/// `multiplier` scales the penalty; `allowed_length` sets minimum repeat length.
pub fn applyDry(logits: []f32, recent_ids: []const u32, multiplier: f32, allowed_length: u32) void {
    if (multiplier <= 0 or recent_ids.len < allowed_length + 1) return;
    const n = recent_ids.len;
    const al: usize = allowed_length;

    // Scan recent_ids to find which token would continue an existing n-gram.
    // For each position, check if the suffix ending there matches the tail of
    // recent_ids. If so, the token at (position + match_len) would extend the
    // repetition — penalize it. O(seq^2) instead of O(vocab * seq).
    for (0..n - al) |search_pos| {
        var match_len: usize = 0;
        var j: usize = 0;
        while (j < al and search_pos + j < n) : (j += 1) {
            const tail_idx = n - al + j;
            if (recent_ids[search_pos + j] != recent_ids[tail_idx]) break;
            match_len += 1;
        }
        if (match_len < al) continue;

        // The token at search_pos + match_len would continue the repeat
        const continuation_pos = search_pos + match_len;
        if (continuation_pos >= n) continue;
        const tid = recent_ids[continuation_pos];
        if (tid < logits.len and logits[tid] != -std.math.inf(f32)) {
            const penalty = multiplier * @as(f32, @floatFromInt(match_len));
            logits[tid] -= penalty;
        }
    }
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
/// Apply OpenAI-compatible frequency and presence penalties to logits.
/// frequency_penalty: penalize by count(token_in_output) * penalty
/// presence_penalty: penalize by 1 * penalty if token appeared at all
pub fn applyPenalties(logits: []f32, gen_tokens: []const u32, frequency_penalty: f32, presence_penalty: f32) void {
    if (frequency_penalty == 0 and presence_penalty == 0) return;
    if (gen_tokens.len == 0) return;

    // Fast path: frequency-only (skip 32KB hash set initialization)
    if (presence_penalty == 0) {
        for (gen_tokens) |tid| {
            if (tid < logits.len) logits[tid] -= frequency_penalty;
        }
        return;
    }

    // Single pass: frequency penalty per occurrence + presence penalty per unique token.
    // Open-addressing hash set (power-of-2 table) for O(1) amortized uniqueness check.
    // Right-size table to ~2× input to keep load factor <50% and minimize init cost.
    const empty_slot = std.math.maxInt(u32);
    const min_bits = 6;
    const max_bits = 13;
    const set_bits = blk: {
        var bits: u5 = min_bits;
        while (bits < max_bits) : (bits += 1) {
            if ((@as(usize, 1) << bits) >= gen_tokens.len * 2) break;
        }
        break :blk bits;
    };
    const set_size = @as(usize, 1) << set_bits;
    const set_mask: u32 = @intCast(set_size - 1);
    var set_buf: [1 << max_bits]u32 = undefined;
    const set: []u32 = set_buf[0..set_size];
    @memset(set, empty_slot);

    for (gen_tokens) |tid| {
        if (tid >= logits.len) continue;
        if (frequency_penalty != 0) logits[tid] -= frequency_penalty;
        // Probe hash set for uniqueness
        var slot = tid & set_mask;
        var is_new = true;
        while (set[slot] != empty_slot) {
            if (set[slot] == tid) { is_new = false; break; }
            slot = (slot +% 1) & set_mask;
        }
        if (is_new) {
            set[slot] = tid;
            logits[tid] -= presence_penalty;
        }
    }
}

/// Compute log probability of a specific token from raw logits.
/// Returns log(softmax(logits)[token_id]).
pub fn tokenLogProb(logits: []const f32, token_id: u32) f32 {
    if (token_id >= logits.len) return -std.math.inf(f32);

    const n = logits.len;
    const max_val = simdMaxF32(logits);
    // SIMD exp-sum
    const log_sum: f32 = blk: {
        const max_v: V8 = @splat(max_val);
        var sum_v: V8 = @splat(@as(f32, 0.0));
        var si: usize = 0;
        while (si + 8 <= n) : (si += 8) {
            sum_v += @exp(@as(V8, logits[si..][0..8].*) - max_v);
        }
        var s = @reduce(.Add, sum_v);
        while (si < n) : (si += 1) s += @exp(logits[si] - max_val);
        break :blk s;
    };
    return (logits[token_id] - max_val) - @log(log_sum);
}

/// Compute top-N tokens by logit value and their log probabilities.
/// Writes to provided output slices. Returns actual count written (<= n).
pub fn topLogProbs(logits: []const f32, n: u32, out_ids: []u32, out_logprobs: []f32) u32 {
    const limit = @min(n, @min(@as(u32, max_top_logprobs), @as(u32, @intCast(out_ids.len))));
    if (limit == 0) return 0;
    const len = logits.len;

    const max_val = simdMaxF32(logits);
    // SIMD exp-sum
    const log_norm: f32 = blk: {
        const max_v: V8 = @splat(max_val);
        var sum_v: V8 = @splat(@as(f32, 0.0));
        var si: usize = 0;
        while (si + 8 <= len) : (si += 8) {
            sum_v += @exp(@as(V8, logits[si..][0..8].*) - max_v);
        }
        var s = @reduce(.Add, sum_v);
        while (si < len) : (si += 1) s += @exp(logits[si] - max_val);
        break :blk @log(s);
    };

    // Find top-N by min-replacement scan
    var top_vals: [max_top_logprobs]f32 = .{-std.math.inf(f32)} ** max_top_logprobs;
    var top_ids: [max_top_logprobs]u32 = .{0} ** max_top_logprobs;
    var mi: usize = 0;

    for (logits, 0..) |v, i| {
        if (v > top_vals[mi]) {
            top_vals[mi] = v;
            top_ids[mi] = @intCast(i);
            mi = 0;
            for (1..limit) |j| {
                if (top_vals[j] < top_vals[mi]) mi = j;
            }
        }
    }

    for (0..limit) |i| {
        out_ids[i] = top_ids[i];
        out_logprobs[i] = (top_vals[i] - max_val) - log_norm;
    }
    return limit;
}

/// Apply min_p filtering: zero out tokens with probability < min_p * max_probability.
/// Must be called AFTER temperature scaling (logits are still pre-softmax).
/// Converts to probabilities, finds max, masks below threshold, restores to logits.
pub fn applyMinP(logits: []f32, min_p: f32) void {
    if (min_p <= 0 or min_p >= 1.0) return;

    const n = logits.len;
    const neg_inf = -std.math.inf(f32);

    const max_val = simdMaxF32(logits);
    const log_threshold = max_val + @log(min_p);

    // SIMD threshold masking
    const thresh_v: V8 = @splat(log_threshold);
    const neg_inf_v: V8 = @splat(neg_inf);
    var si: usize = 0;
    while (si + 8 <= n) : (si += 8) {
        const chunk: V8 = logits[si..][0..8].*;
        logits[si..][0..8].* = @select(f32, chunk < thresh_v, neg_inf_v, chunk);
    }
    while (si < n) : (si += 1) {
        if (logits[si] < log_threshold) logits[si] = neg_inf;
    }
}

/// XTC sampling: with probability `xtc_probability`, exclude top tokens that
/// exceed `xtc_threshold` probability. Increases diversity by preventing mode collapse.
/// Must be called AFTER temperature scaling. Operates on pre-softmax logits.
pub fn applyXtc(logits: []f32, xtc_probability: f32, xtc_threshold: f32, rng: std.Random) void {
    if (xtc_probability <= 0 or xtc_threshold <= 0) return;
    if (rng.float(f32) > xtc_probability) return;

    const n = logits.len;
    const neg_inf = -std.math.inf(f32);

    const max_val = simdMaxF32(logits);
    const log_threshold = max_val + @log(xtc_threshold);

    // Single pass: mask all above-threshold tokens except the last one.
    // When a new above-threshold token is found, mask the previous one.
    var prev_above: ?usize = null;
    for (0..n) |i| {
        if (logits[i] >= log_threshold) {
            if (prev_above) |prev| logits[prev] = neg_inf;
            prev_above = i;
        }
    }
}

/// Mirostat 2.0 sampling: maintain target surprise (entropy) during generation.
/// `mu` tracks the running surprise estimate, adjusted by learning rate `eta`.
/// Returns the sampled token ID. `mu` is updated in place for the next step.
/// When Mirostat is active, top-k/top-p are bypassed — Mirostat controls its own truncation.
pub fn sampleMirostat(logits: []f32, tau: f32, eta: f32, mu: *f32, temperature: f32, rng: std.Random) u32 {
    const n = logits.len;
    if (n == 0) return 0;

    // Temperature scaling (needed for max computation)
    if (temperature > 0 and temperature != 1.0) {
        const inv_temp = 1.0 / temperature;
        const inv_v: V8 = @splat(inv_temp);
        var si: usize = 0;
        while (si + 8 <= n) : (si += 8) {
            logits[si..][0..8].* = @as(V8, logits[si..][0..8].*) * inv_v;
        }
        while (si < n) : (si += 1) logits[si] *= inv_temp;
    }

    // Fused softmax + threshold + normalize (3 passes → 2).
    // Pass 1: exp and sum. Pass 2: threshold, zero, and normalize.
    const max_val = simdMaxF32(logits);
    var sum_exp: f32 = 0;
    {
        const max_v: V8 = @splat(max_val);
        var sum_v: V8 = @splat(@as(f32, 0.0));
        var si: usize = 0;
        while (si + 8 <= n) : (si += 8) {
            const exp_v = @exp(@as(V8, logits[si..][0..8].*) - max_v);
            logits[si..][0..8].* = exp_v;
            sum_v += exp_v;
        }
        sum_exp = @reduce(.Add, sum_v);
        while (si < n) : (si += 1) {
            logits[si] = @exp(logits[si] - max_val);
            sum_exp += logits[si];
        }
    }

    // Fused threshold + normalize: apply Mirostat truncation and normalize in one pass.
    // Tokens with p < exp(-mu) are zeroed; survivors are divided by their sum.
    const min_prob = @exp(-mu.*);
    var new_sum: f32 = 0;
    if (sum_exp > 0) {
        const inv_sum = 1.0 / sum_exp;
        const min_unnorm = min_prob * sum_exp;
        const min_v: V8 = @splat(min_unnorm);
        const zero_v: V8 = @splat(@as(f32, 0.0));
        const inv_v: V8 = @splat(inv_sum);
        var new_sum_v: V8 = zero_v;
        var si: usize = 0;
        while (si + 8 <= n) : (si += 8) {
            const chunk: V8 = logits[si..][0..8].*;
            const kept = @select(f32, chunk < min_v, zero_v, chunk * inv_v);
            logits[si..][0..8].* = kept;
            new_sum_v += kept;
        }
        new_sum = @reduce(.Add, new_sum_v);
        while (si < n) : (si += 1) {
            const p = logits[si] * inv_sum;
            if (logits[si] < min_unnorm) {
                logits[si] = 0;
            } else {
                logits[si] = p;
                new_sum += p;
            }
        }
    }

    // Sample from filtered distribution — scale threshold by sum
    // instead of renormalizing the entire array (saves one SIMD pass over vocab).
    var r = rng.float(f32) * new_sum;
    var chosen: u32 = 0;
    for (0..n) |i| {
        r -= logits[i];
        if (r <= 0) {
            chosen = @intCast(i);
            break;
        }
    }

    // Update mu: normalize only the chosen token's probability
    const chosen_p = if (new_sum > 0) logits[chosen] / new_sum else log_prob_epsilon;
    const chosen_surprise = -@log(if (chosen_p > log_prob_epsilon) chosen_p else log_prob_epsilon);
    mu.* -= eta * (chosen_surprise - tau);

    return chosen;
}

/// Modifies the logits buffer in-place.
pub fn sampleToken(logits: []f32, temperature: f32, top_k: u32, top_p: f32, rng: std.Random) u32 {
    if (temperature == 0) return argmax(logits);

    const n = logits.len;
    const neg_inf = -std.math.inf(f32);

    // 1. Temperature scaling (SIMD) — skip identity scaling
    if (temperature != 1.0) {
        const inv_temp = 1.0 / temperature;
        const inv_v: V8 = @splat(inv_temp);
        var si: usize = 0;
        while (si + 8 <= n) : (si += 8) {
            logits[si..][0..8].* = @as(V8, logits[si..][0..8].*) * inv_v;
        }
        while (si < n) : (si += 1) logits[si] *= inv_temp;
    }

    // 2+3. Top-k + softmax fused: when top-k active, derive max from
    // top_buf O(k) and fuse mask+exp+sum into single vocab pass,
    // saving 2 full SIMD sweeps vs. separate mask, max, exp+sum.
    var sum: f32 = 0;
    if (top_k > 0 and top_k < n) {
        const k: usize = top_k;
        var top_buf: [max_top_k]f32 = undefined;
        const buf_k = @min(k, max_top_k);
        for (0..buf_k) |i| top_buf[i] = neg_inf;
        var mi: usize = 0;

        for (logits) |v| {
            if (v > top_buf[mi]) {
                top_buf[mi] = v;
                mi = 0;
                for (1..buf_k) |j| {
                    if (top_buf[j] < top_buf[mi]) mi = j;
                }
            }
        }
        const top_min = top_buf[mi];
        var max_val: f32 = top_buf[0];
        for (1..buf_k) |j| max_val = @max(max_val, top_buf[j]);

        const min_v: V8 = @splat(top_min);
        const neg_inf_v: V8 = @splat(neg_inf);
        const max_v: V8 = @splat(max_val);
        var sum_v: V8 = @splat(@as(f32, 0.0));
        var si: usize = 0;
        while (si + 8 <= n) : (si += 8) {
            const chunk: V8 = logits[si..][0..8].*;
            const masked = @select(f32, chunk < min_v, neg_inf_v, chunk);
            const exp_vals = @exp(masked - max_v);
            logits[si..][0..8].* = exp_vals;
            sum_v += exp_vals;
        }
        sum = @reduce(.Add, sum_v);
        while (si < n) : (si += 1) {
            if (logits[si] < top_min) logits[si] = neg_inf;
            logits[si] = @exp(logits[si] - max_val);
            sum += logits[si];
        }
    } else {
        const max_val = simdMaxF32(logits);
        const max_v: V8 = @splat(max_val);
        var sum_v: V8 = @splat(@as(f32, 0.0));
        var si: usize = 0;
        while (si + 8 <= n) : (si += 8) {
            const exp_vals = @exp(@as(V8, logits[si..][0..8].*) - max_v);
            logits[si..][0..8].* = exp_vals;
            sum_v += exp_vals;
        }
        sum = @reduce(.Add, sum_v);
        while (si < n) : (si += 1) {
            logits[si] = @exp(logits[si] - max_val);
            sum += logits[si];
        }
    }

    // 4. Top-p (nucleus): threshold-based filtering.
    // Collects top candidates via min-replacement scan (O(n + insertions*k)),
    // sorts them, finds the probability threshold where cumulative
    // probability >= top_p, then zeroes out tokens below the threshold.
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

        // Sort candidates descending
        std.mem.sort(f32, top_vals[0..n_top], {}, std.sort.desc(f32));

        // Cumsum scan to find probability threshold
        var cumsum: f32 = 0;
        var threshold: f32 = 0;
        for (top_vals[0..n_top]) |v| {
            cumsum += v;
            if (cumsum >= top_p * sum) {
                threshold = v;
                break;
            }
        }

        // Apply threshold: zero out tokens below cutoff, recompute sum
        sum = 0;
        {
            const thresh_v: V8 = @splat(threshold);
            const zero_v: V8 = @splat(0.0);
            var sum_v: V8 = zero_v;
            var si: usize = 0;
            while (si + 8 <= n) : (si += 8) {
                const chunk: V8 = logits[si..][0..8].*;
                const keep = chunk >= thresh_v;
                const result = @select(f32, keep, chunk, zero_v);
                logits[si..][0..8].* = result;
                sum_v += result;
            }
            sum = @reduce(.Add, sum_v);
            while (si < n) : (si += 1) {
                if (logits[si] < threshold) {
                    logits[si] = 0;
                } else {
                    sum += logits[si];
                }
            }
        }
    }

    // 5. Weighted random sampling (unnormalized — scale threshold by sum)
    var cumulative: f32 = 0;
    const sample_threshold = rng.float(f32) * sum;
    for (logits, 0..) |p, i| {
        cumulative += p;
        if (sample_threshold < cumulative) return @intCast(i);
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
    // Just below threshold: still computed via log(1+exp(x)), result ≈ 19.0
    try std.testing.expectApproxEqAbs(@as(f32, 19.0), softplus(19.0), 1e-4);
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
    // GELU(0) = 0 exactly
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), buf[0], 1e-6);
    // GELU(1) ≈ 0.8412
    try std.testing.expectApproxEqAbs(@as(f32, 0.8412), buf[1], 0.001);
    // GELU(-1) ≈ -0.1588
    try std.testing.expectApproxEqAbs(@as(f32, -0.1588), buf[2], 0.001);
    // GELU(2) ≈ 1.9546
    try std.testing.expectApproxEqAbs(@as(f32, 1.9546), buf[3], 0.001);
    // GELU(-2) ≈ -0.0454
    try std.testing.expectApproxEqAbs(@as(f32, -0.0454), buf[4], 0.001);
    // Scalar tail: GELU(-3) ≈ -0.00436
    try std.testing.expectApproxEqAbs(@as(f32, -0.00436), buf[8], 0.001);
    // Scalar tail: GELU(1.5) ≈ 1.3990
    try std.testing.expectApproxEqAbs(@as(f32, 1.3990), buf[9], 0.001);
}

test "sampleToken greedy" {
    var logits = [_]f32{ 1.0, 5.0, 2.0, 0.5 };
    var prng = std.Random.DefaultPrng.init(42);
    // temperature=0 should return argmax regardless of RNG
    try std.testing.expectEqual(@as(u32, 1), sampleToken(&logits, 0, 0, 1.0, prng.random()));
}

test "sampleToken deterministic with seed" {
    // Same seed should produce same result, and result should be valid
    var logits1 = [_]f32{ 1.0, 2.0, 3.0, 2.0 };
    var logits2 = [_]f32{ 1.0, 2.0, 3.0, 2.0 };
    var prng1 = std.Random.DefaultPrng.init(123);
    var prng2 = std.Random.DefaultPrng.init(123);
    const result1 = sampleToken(&logits1, 1.0, 0, 1.0, prng1.random());
    const result2 = sampleToken(&logits2, 1.0, 0, 1.0, prng2.random());
    try std.testing.expectEqual(result1, result2);
    // Result must be a valid token index
    try std.testing.expect(result1 < 4);
}

test "sampleToken top_k filters" {
    // top_k=2 keeps only the two highest logits (indices 1=3.0 and 3=2.5)
    // Close values + temp=1.0 ensure both get sampled across many seeds
    var seen = [_]bool{false} ** 4;
    for (0..500) |seed| {
        var l = [_]f32{ 0.1, 3.0, 0.2, 2.5 };
        var p = std.Random.DefaultPrng.init(seed);
        seen[sampleToken(&l, 1.0, 2, 1.0, p.random())] = true;
    }
    try std.testing.expect(!seen[0]); // index 0 (0.1) filtered out
    try std.testing.expect(seen[1]); // index 1 (3.0) kept
    try std.testing.expect(!seen[2]); // index 2 (0.2) filtered out
    try std.testing.expect(seen[3]); // index 3 (2.5) kept
}

test "sampleToken top_p nucleus sampling" {
    // top_p=0.5 should keep only the highest-probability token(s) until
    // cumulative probability >= 0.5. With logits [0.1, 5.0, 0.2, 0.3],
    // index 1 dominates after softmax and should be the only token sampled.
    var seen = [_]bool{false} ** 4;
    for (0..500) |seed| {
        var l = [_]f32{ 0.1, 5.0, 0.2, 0.3 };
        var p = std.Random.DefaultPrng.init(seed);
        seen[sampleToken(&l, 1.0, 0, 0.5, p.random())] = true;
    }
    // Index 1 should always be selected (softmax(5.0) >> 0.5 cumulative)
    try std.testing.expect(seen[1]);
    // Other indices should be filtered out by nucleus
    try std.testing.expect(!seen[0]);
    try std.testing.expect(!seen[2]);
    try std.testing.expect(!seen[3]);
}

test "sampleToken top_p allows multiple tokens" {
    // With close logits and top_p=0.9, multiple tokens should be sampled.
    // logits [2.0, 2.1, 2.0, 2.1] are close → softmax near uniform.
    var seen = [_]bool{false} ** 4;
    for (0..500) |seed| {
        var l = [_]f32{ 2.0, 2.1, 2.0, 2.1 };
        var p = std.Random.DefaultPrng.init(seed);
        seen[sampleToken(&l, 1.0, 0, 0.9, p.random())] = true;
    }
    // With near-uniform distribution and top_p=0.9, at least 3 tokens should appear
    var count: usize = 0;
    for (seen) |s| if (s) {
        count += 1;
    };
    try std.testing.expect(count >= 3);
}

test "applyRepeatPenalty positive logits divided" {
    var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const recent = [_]u32{ 1, 3 };
    applyRepeatPenalty(&logits, &recent, 2.0);
    // Unpenalized tokens unchanged
    try std.testing.expectEqual(@as(f32, 1.0), logits[0]);
    try std.testing.expectEqual(@as(f32, 3.0), logits[2]);
    // Positive logits divided by penalty
    try std.testing.expectEqual(@as(f32, 1.0), logits[1]); // 2.0 / 2.0
    try std.testing.expectEqual(@as(f32, 2.0), logits[3]); // 4.0 / 2.0
}

test "applyRepeatPenalty negative logits multiplied" {
    var logits = [_]f32{ -1.0, 2.0, -3.0 };
    const recent = [_]u32{ 0, 2 };
    applyRepeatPenalty(&logits, &recent, 1.5);
    // Negative logits multiplied by penalty (made more negative)
    try std.testing.expectApproxEqAbs(@as(f32, -1.5), logits[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -4.5), logits[2], 1e-6);
    // Unpenalized positive logit unchanged
    try std.testing.expectEqual(@as(f32, 2.0), logits[1]);
}

test "applyRepeatPenalty no-op at 1.0" {
    var logits = [_]f32{ 1.0, -2.0, 3.0 };
    const original = [_]f32{ 1.0, -2.0, 3.0 };
    const recent = [_]u32{ 0, 1, 2 };
    applyRepeatPenalty(&logits, &recent, 1.0);
    // penalty=1.0 should not change any logits
    for (0..3) |i| {
        try std.testing.expectEqual(original[i], logits[i]);
    }
}

test "applyRepeatPenalty out-of-range token ignored" {
    var logits = [_]f32{ 1.0, 2.0 };
    const recent = [_]u32{ 0, 999 }; // 999 is out of range
    applyRepeatPenalty(&logits, &recent, 2.0);
    try std.testing.expectEqual(@as(f32, 0.5), logits[0]); // 1.0 / 2.0
    try std.testing.expectEqual(@as(f32, 2.0), logits[1]); // unchanged
}

test "topKExperts bias-corrected selection vs raw weighting" {
    // Verify the Nemotron-Nano MoE routing pattern:
    // Use bias to shift expert SELECTION, but weight with raw sigmoid scores.
    const raw_sigmoid = [_]f32{ 0.3, 0.7, 0.1, 0.6, 0.2 };
    // Bias boosts expert 2 (raw=0.1) to top of selection
    const bias = [_]f32{ 0.0, 0.0, 0.9, 0.0, 0.0 };

    // Add bias for selection
    var biased: [5]f32 = undefined;
    for (0..5) |i| biased[i] = raw_sigmoid[i] + bias[i];

    // Select top-2 using biased scores
    var top_idx: [2]usize = undefined;
    var top_unused: [2]f32 = undefined;
    topKExperts(&biased, 2, &top_idx, &top_unused);

    // Expert 2 (biased=1.0) and expert 1 (biased=0.7) should be selected
    var selected = [_]bool{false} ** 5;
    for (0..2) |i| selected[top_idx[i]] = true;
    try std.testing.expect(selected[2]); // boosted by bias
    try std.testing.expect(selected[1]); // naturally high

    // Gather RAW sigmoid scores for weighting (NOT biased)
    var raw_weights: [2]f32 = undefined;
    for (0..2) |i| raw_weights[i] = raw_sigmoid[top_idx[i]];

    // Expert 2's weight should be 0.1 (raw sigmoid), NOT 1.0 (biased)
    for (0..2) |i| {
        if (top_idx[i] == 2)
            try std.testing.expectApproxEqAbs(@as(f32, 0.1), raw_weights[i], 1e-6);
        if (top_idx[i] == 1)
            try std.testing.expectApproxEqAbs(@as(f32, 0.7), raw_weights[i], 1e-6);
    }

    // Normalized weights should sum to 1.0
    var sum: f32 = 0;
    for (0..2) |i| sum += raw_weights[i];
    for (0..2) |i| raw_weights[i] /= sum;
    var weight_sum: f32 = 0;
    for (0..2) |i| weight_sum += raw_weights[i];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), weight_sum, 1e-5);
}

test "topKExperts duplicate scores tie breaking" {
    const scores = [_]f32{ 0.5, 0.5, 0.1 };
    var indices: [1]usize = undefined;
    var values: [1]f32 = undefined;
    topKExperts(&scores, 1, &indices, &values);
    // First 0.5 (index 0) wins — ties broken by position
    try std.testing.expectEqual(@as(usize, 0), indices[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), values[0], 1e-6);
}

test "topKExperts k equals n" {
    const scores = [_]f32{ 0.3, 0.1, 0.7 };
    var indices: [3]usize = undefined;
    var values: [3]f32 = undefined;
    topKExperts(&scores, 3, &indices, &values);
    var found = [_]bool{false} ** 3;
    for (0..3) |i| {
        found[indices[i]] = true;
        try std.testing.expectApproxEqAbs(scores[indices[i]], values[i], 1e-6);
    }
    try std.testing.expect(found[0]);
    try std.testing.expect(found[1]);
    try std.testing.expect(found[2]);
}

test "topKExperts negative scores" {
    const scores = [_]f32{ -0.5, -0.1, -0.9, -0.3 };
    var indices: [2]usize = undefined;
    var values: [2]f32 = undefined;
    topKExperts(&scores, 2, &indices, &values);
    var found = [_]bool{false} ** 4;
    for (0..2) |i| found[indices[i]] = true;
    try std.testing.expect(found[1]); // -0.1 (highest)
    try std.testing.expect(found[3]); // -0.3 (second highest)
}

test "argmax all equal returns first" {
    const buf = [_]f32{ 5.0, 5.0, 5.0, 5.0 };
    // Ties broken by first occurrence (> not >=)
    try std.testing.expectEqual(@as(u32, 0), argmax(&buf));
}

test "argmax negative values" {
    const buf = [_]f32{ -10.0, -5.0, -20.0, -1.0 };
    try std.testing.expectEqual(@as(u32, 3), argmax(&buf));
}

test "argmax with inf" {
    const buf = [_]f32{ 1.0, std.math.inf(f32), 2.0, 0.5 };
    try std.testing.expectEqual(@as(u32, 1), argmax(&buf));
}

test "argmax with negative inf" {
    const buf = [_]f32{ -std.math.inf(f32), -1.0, -std.math.inf(f32) };
    try std.testing.expectEqual(@as(u32, 1), argmax(&buf));
}

test "argmax exercises SIMD path" {
    // 16 elements: the 8-wide SIMD reduction loop executes twice.
    // Max at index 13 ensures SIMD finds it in the second chunk.
    const buf = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 9.9, 1.5, 1.6 };
    try std.testing.expectEqual(@as(u32, 13), argmax(&buf));
}

test "sigmoid symmetry" {
    // sigmoid(-x) = 1 - sigmoid(x)
    const x: f32 = 3.7;
    const pos = sigmoid(x);
    const neg = sigmoid(-x);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), pos + neg, 1e-6);
}

test "applyGelu extreme values clamped" {
    // Values beyond clamp range should not produce NaN/Inf
    var buf = [_]f32{ 100.0, -100.0 };
    applyGelu(&buf);
    // GELU(large positive) ≈ x (linear regime)
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), buf[0], 0.01);
    // GELU(large negative) ≈ 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), buf[1], 0.01);
}

test "applyPenalties frequency" {
    var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const tokens = [_]u32{ 1, 1, 2 }; // token 1 appears 2x, token 2 appears 1x
    applyPenalties(&logits, &tokens, 0.5, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), logits[0], 0.001); // unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), logits[1], 0.001); // 2.0 - 2*0.5
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), logits[2], 0.001); // 3.0 - 1*0.5
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), logits[3], 0.001); // unchanged
}

test "applyPenalties presence" {
    var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const tokens = [_]u32{ 1, 1, 2 };
    applyPenalties(&logits, &tokens, 0, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), logits[0], 0.001); // unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), logits[1], 0.001); // 2.0 - 1.0 (once)
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), logits[2], 0.001); // 3.0 - 1.0 (once)
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), logits[3], 0.001); // unchanged
}

test "applyRepeatPenalty positive logit" {
    var logits = [_]f32{ 1.0, 2.0, 3.0 };
    const tokens = [_]u32{1};
    applyRepeatPenalty(&logits, &tokens, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), logits[0], 0.001); // unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), logits[1], 0.001); // 2.0 / 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), logits[2], 0.001); // unchanged
}

test "applyRepeatPenalty negative logit" {
    var logits = [_]f32{ -2.0, 1.0 };
    const tokens = [_]u32{0};
    applyRepeatPenalty(&logits, &tokens, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, -4.0), logits[0], 0.001); // -2.0 * 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), logits[1], 0.001); // unchanged
}

test "applyMinP filters low probability tokens" {
    // min_p=0.01, max=10, threshold = 10 + ln(0.01) = 5.395
    // logit=10 ≥ 5.395 → kept, logit=5 < 5.395 → masked
    var logits = [_]f32{ 10, 5, 0, -5 };
    applyMinP(&logits, 0.01);
    try std.testing.expectEqual(@as(f32, 10), logits[0]);
    try std.testing.expectEqual(-std.math.inf(f32), logits[1]);
    try std.testing.expectEqual(-std.math.inf(f32), logits[2]);
    try std.testing.expectEqual(-std.math.inf(f32), logits[3]);
}

test "applyMinP keeps multiple tokens" {
    var logits = [_]f32{ 10, 9, 8, 0 };
    applyMinP(&logits, 0.1); // threshold = 10 + ln(0.1) = 10 - 2.303 = 7.697
    try std.testing.expectEqual(@as(f32, 10), logits[0]); // kept
    try std.testing.expectEqual(@as(f32, 9), logits[1]); // kept
    try std.testing.expectEqual(@as(f32, 8), logits[2]); // kept
    try std.testing.expectEqual(-std.math.inf(f32), logits[3]); // masked
}

test "tokenLogProb" {
    // logits [0, 0, 0] → uniform → each has prob 1/3 → logprob = ln(1/3) ≈ -1.0986
    const logits = [_]f32{ 0, 0, 0 };
    const lp = tokenLogProb(&logits, 0);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0986), lp, 0.01);
}

test "tokenLogProb dominant" {
    // logits [10, 0, 0] → token 0 dominates → logprob ≈ 0 (must be ≤ 0)
    const logits = [_]f32{ 10, 0, 0 };
    const lp = tokenLogProb(&logits, 0);
    try std.testing.expect(lp > -0.001);
    try std.testing.expect(lp <= 0);
}

test "topLogProbs returns correct ids" {
    const logits = [_]f32{ 1.0, 5.0, 3.0, 2.0 };
    var ids: [2]u32 = undefined;
    var probs: [2]f32 = undefined;
    const n = topLogProbs(&logits, 2, &ids, &probs);
    try std.testing.expectEqual(@as(u32, 2), n);
    // Top 2 should be indices 1 (5.0) and 2 (3.0)
    var has_1 = false;
    var has_2 = false;
    for (ids[0..n]) |id| {
        if (id == 1) has_1 = true;
        if (id == 2) has_2 = true;
    }
    try std.testing.expect(has_1);
    try std.testing.expect(has_2);
    // Log probabilities must be finite and non-positive
    for (probs[0..n]) |p| {
        try std.testing.expect(std.math.isFinite(p));
        try std.testing.expect(p <= 0);
    }
}

test "applyXtc excludes top tokens" {
    var logits = [_]f32{ 10.0, 9.0, 1.0, 0.5 };
    var prng = std.Random.Xoshiro256.init(42);
    // Force XTC to trigger (probability=1.0, threshold=0.01)
    applyXtc(&logits, 1.0, 0.01, prng.random());
    // At least one of the top tokens should be -inf
    var n_neg_inf: u32 = 0;
    for (logits) |v| {
        if (v == -std.math.inf(f32)) n_neg_inf += 1;
    }
    try std.testing.expect(n_neg_inf >= 1);
    // At least one token must survive (XTC always keeps one)
    var n_alive: u32 = 0;
    for (logits) |v| {
        if (v != -std.math.inf(f32)) n_alive += 1;
    }
    try std.testing.expect(n_alive >= 1);
}

test "applyXtc no-op at probability 0" {
    var logits = [_]f32{ 10.0, 9.0, 1.0 };
    const original = logits;
    var prng = std.Random.Xoshiro256.init(42);
    applyXtc(&logits, 0.0, 0.1, prng.random());
    try std.testing.expectEqualSlices(f32, &original, &logits);
}

test "applyDry penalizes repeated sequence" {
    var logits = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    // History: [1, 2, 3, 1, 2] — token 3 would continue the repeat
    const history = [_]u32{ 1, 2, 3, 1, 2 };
    const hist_slice: []const u32 = &history;
    applyDry(&logits, hist_slice, 1.0, 2);
    // Token 3 should be penalized (repeating "1 2 3")
    try std.testing.expect(logits[3] < 0);
    // Non-continuation tokens must remain unpenalized
    try std.testing.expectEqual(@as(f32, 0.0), logits[0]);
    try std.testing.expectEqual(@as(f32, 0.0), logits[4]);
}

test "applyDry no-op with no repeats" {
    var logits = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const history = [_]u32{ 1, 2, 3 };
    const hist_slice: []const u32 = &history;
    applyDry(&logits, hist_slice, 1.0, 2);
    // No repeated bigrams → no penalty
    for (logits) |v| try std.testing.expectEqual(@as(f32, 0.0), v);
}

test "sampleMirostat returns valid token" {
    var logits = [_]f32{ 1.0, 2.0, 3.0, 0.5 };
    var mu: f32 = 10.0;
    var prng = std.Random.Xoshiro256.init(42);
    const token = sampleMirostat(&logits, 5.0, 0.1, &mu, 1.0, prng.random());
    try std.testing.expect(token < 4);
    // mu must be updated, remain finite, and stay positive (entropy target)
    try std.testing.expect(mu != 10.0);
    try std.testing.expect(std.math.isFinite(mu));
    try std.testing.expect(mu > 0);
    // Second call should also produce valid results with updated mu
    var logits2 = [_]f32{ 1.0, 2.0, 3.0, 0.5 };
    const mu_before = mu;
    const token2 = sampleMirostat(&logits2, 5.0, 0.1, &mu, 1.0, prng.random());
    try std.testing.expect(token2 < 4);
    try std.testing.expect(mu != mu_before);
    try std.testing.expect(std.math.isFinite(mu));
}

test "applyLogitBias basic" {
    var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const ids = [_]u32{ 0, 2 };
    const biases = [_]f32{ 5.0, -1.0 };
    applyLogitBias(&logits, &ids, &biases, 2);
    try std.testing.expectEqual(@as(f32, 6.0), logits[0]); // 1.0 + 5.0
    try std.testing.expectEqual(@as(f32, 2.0), logits[1]); // unchanged
    try std.testing.expectEqual(@as(f32, 2.0), logits[2]); // 3.0 + (-1.0)
    try std.testing.expectEqual(@as(f32, 4.0), logits[3]); // unchanged
}

test "applyLogitBias out-of-range id ignored" {
    var logits = [_]f32{ 1.0, 2.0 };
    const ids = [_]u32{ 0, 999 };
    const biases = [_]f32{ 1.0, 10.0 };
    applyLogitBias(&logits, &ids, &biases, 2);
    try std.testing.expectEqual(@as(f32, 2.0), logits[0]); // 1.0 + 1.0
    try std.testing.expectEqual(@as(f32, 2.0), logits[1]); // unchanged
}

test "applyLogitBias zero count is no-op" {
    var logits = [_]f32{ 1.0, 2.0 };
    const ids = [_]u32{0};
    const biases = [_]f32{99.0};
    applyLogitBias(&logits, &ids, &biases, 0);
    try std.testing.expectEqual(@as(f32, 1.0), logits[0]);
    try std.testing.expectEqual(@as(f32, 2.0), logits[1]);
}
