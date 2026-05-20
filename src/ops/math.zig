//! Shared mathematical utility functions for model implementations.
//! Provides common operations (argmax, softplus, sigmoid, GELU) used across
//! multiple model architectures, avoiding code duplication.

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
const max_top_k: usize = 1024;
/// Maximum candidates for top-p nucleus sampling buffer.
/// Caps the number of probabilities tracked during threshold computation.
const nucleus_max_candidates: usize = 1024;

/// Return the index of the maximum element in `buf`.
/// Two-phase SIMD: branchless max-reduce, then early-exit index scan.
/// Faster than scalar for large vocabularies (32K+) because the max-find
/// phase has no loop-carried data dependency.
///
/// Parameters:
///   - buf: Non-empty slice of f32 values to search.
///
/// Returns: Index of the maximum value as u32 (first occurrence on ties).
pub fn argmax(buf: []const f32) u32 {
    const V8 = @Vector(8, f32);
    // Phase 1: SIMD reduction to find maximum value (branchless)
    var max_v: V8 = @splat(-std.math.inf(f32));
    var i: usize = 0;
    while (i + 8 <= buf.len) : (i += 8) {
        const chunk: V8 = buf[i..][0..8].*;
        max_v = @max(max_v, chunk);
    }
    var best_val = @reduce(.Max, max_v);
    while (i < buf.len) : (i += 1) {
        best_val = @max(best_val, buf[i]);
    }
    // Phase 2: find first occurrence (early-exit scan)
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

/// Apply repetition penalty to logits for recently generated tokens.
/// For each token ID in `recent_ids`, divides its logit by `penalty` if positive,
/// multiplies if negative. This discourages the model from repeating tokens.
/// Standard repeat penalty from the Transformer paper (Keskar et al. 2019).
///
/// Parameters:
///   - logits: Mutable logit buffer [vocab_size], modified in-place.
///   - recent_ids: Slice of recently generated token IDs to penalize.
///   - penalty: Repetition penalty factor (> 1.0 = more penalty, 1.0 = no penalty).
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

/// DRY (Don't Repeat Yourself) sampling: penalize tokens that would continue
/// a repeated n-gram sequence. For each candidate token, check if appending it
/// would create an n-gram that already appeared in the recent output.
/// `multiplier` scales the penalty; `allowed_length` sets minimum repeat length.
/// Inspired by llama.cpp's DRY sampler.
pub fn applyDry(logits: []f32, recent_ids: []const u32, multiplier: f32, allowed_length: u32) void {
    if (multiplier <= 0 or recent_ids.len < allowed_length + 1) return;
    const n = recent_ids.len;

    // For each candidate token, check: if we appended it to recent_ids,
    // would the last (allowed_length+1) tokens match an earlier subsequence?
    for (0..logits.len) |token_id| {
        if (logits[token_id] == -std.math.inf(f32)) continue;

        // Check all possible match lengths starting from allowed_length
        var max_match: u32 = 0;
        const tid: u32 = @intCast(token_id);

        // The pattern to match: recent_ids[n-allowed_length..n] ++ tid
        // Search for this pattern earlier in recent_ids
        var search_pos: usize = 0;
        while (search_pos + allowed_length < n) : (search_pos += 1) {
            // Check if recent_ids[search_pos..] matches the tail
            if (recent_ids[search_pos + allowed_length] != tid) continue;

            var match_len: u32 = 0;
            var j: usize = 0;
            while (j <= allowed_length and search_pos + j < n) : (j += 1) {
                const tail_idx = n - allowed_length + j;
                if (tail_idx >= n) break;
                if (search_pos + j >= n) break;
                if (recent_ids[search_pos + j] != recent_ids[tail_idx]) break;
                match_len += 1;
            }
            if (match_len > max_match) max_match = match_len;
        }

        if (max_match >= allowed_length) {
            const penalty = multiplier * @as(f32, @floatFromInt(max_match));
            logits[token_id] -= penalty;
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

    // Frequency penalty: subtract once per occurrence in history
    if (frequency_penalty != 0) {
        for (gen_tokens) |tid| {
            if (tid < logits.len) logits[tid] -= frequency_penalty;
        }
    }
    // Presence penalty: subtract once per unique token in history
    if (presence_penalty != 0) {
        var seen: [256]u32 = undefined;
        var n_seen: usize = 0;
        for (gen_tokens) |tid| {
            if (tid >= logits.len) continue;
            var found = false;
            for (seen[0..n_seen]) |s| {
                if (s == tid) { found = true; break; }
            }
            if (!found and n_seen < 256) {
                seen[n_seen] = tid;
                n_seen += 1;
                logits[tid] -= presence_penalty;
            }
        }
    }
}

/// Compute log probability of a specific token from raw logits.
/// Returns log(softmax(logits)[token_id]).
pub fn tokenLogProb(logits: []const f32, token_id: u32) f32 {
    if (token_id >= logits.len) return -std.math.inf(f32);
    var max_val: f32 = -std.math.inf(f32);
    for (logits) |v| max_val = @max(max_val, v);
    var log_sum: f32 = 0;
    for (logits) |v| log_sum += @exp(v - max_val);
    return (logits[token_id] - max_val) - @log(log_sum);
}

/// Compute top-N tokens by logit value and their log probabilities.
/// Writes to provided output slices. Returns actual count written (<= n).
pub fn topLogProbs(logits: []const f32, n: u32, out_ids: []u32, out_logprobs: []f32) u32 {
    const limit = @min(n, @as(u32, @intCast(out_ids.len)));
    if (limit == 0) return 0;

    // Find max for log-sum-exp
    var max_val: f32 = -std.math.inf(f32);
    for (logits) |v| max_val = @max(max_val, v);
    var log_sum: f32 = 0;
    for (logits) |v| log_sum += @exp(v - max_val);
    const log_norm = @log(log_sum);

    // Find top-N by min-replacement scan
    var top_vals: [20]f32 = .{-std.math.inf(f32)} ** 20;
    var top_ids: [20]u32 = .{0} ** 20;
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

    // Find max logit — exp(max - max) = 1.0, so threshold in logit space is max + ln(min_p)
    var max_val: f32 = neg_inf;
    for (logits) |v| max_val = @max(max_val, v);
    const log_threshold = max_val + @log(min_p);

    for (0..n) |i| {
        if (logits[i] < log_threshold) logits[i] = neg_inf;
    }
}

/// XTC sampling: with probability `xtc_probability`, exclude top tokens that
/// exceed `xtc_threshold` probability. Increases diversity by preventing mode collapse.
/// Must be called AFTER temperature scaling. Operates on pre-softmax logits.
pub fn applyXtc(logits: []f32, xtc_probability: f32, xtc_threshold: f32, rng: std.Random) void {
    if (xtc_probability <= 0 or xtc_threshold <= 0) return;
    if (rng.float(f32) > xtc_probability) return;

    var max_val: f32 = -std.math.inf(f32);
    for (logits) |v| max_val = @max(max_val, v);
    const log_threshold = max_val + @log(xtc_threshold);
    const neg_inf = -std.math.inf(f32);

    // Count tokens above threshold — keep at least one
    var n_above: usize = 0;
    for (logits) |v| {
        if (v >= log_threshold) n_above += 1;
    }
    if (n_above <= 1) return;

    // Exclude all but the last above-threshold token (keep one for valid sampling)
    var kept: usize = 0;
    for (0..logits.len) |i| {
        if (logits[i] >= log_threshold) {
            kept += 1;
            if (kept < n_above) logits[i] = neg_inf;
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

    // Temperature scaling
    if (temperature > 0 and temperature != 1.0) {
        const inv_temp = 1.0 / temperature;
        for (0..n) |i| logits[i] *= inv_temp;
    }

    // Softmax
    var max_val: f32 = -std.math.inf(f32);
    for (logits) |v| max_val = @max(max_val, v);
    var sum_exp: f32 = 0;
    for (0..n) |i| {
        logits[i] = @exp(logits[i] - max_val);
        sum_exp += logits[i];
    }
    const inv_sum = if (sum_exp > 0) 1.0 / sum_exp else 0;
    for (0..n) |i| logits[i] *= inv_sum;

    // Sort indices by probability descending (partial — only need until we exceed mu threshold)
    // Mirostat 2.0: keep tokens where -ln(p) <= mu (surprise <= target)
    const surprise_threshold = mu.*;
    var cum_prob: f32 = 0;
    // Simple approach: scan in descending probability order
    // Since we need sorted order, find max repeatedly (vocab is large but this runs once per token)
    var mask: [32768]bool = undefined;
    const mask_n = @min(n, 32768);
    @memset(mask[0..mask_n], false);

    for (0..mask_n) |_| {
        // Find highest unmasked probability
        var best_idx: usize = 0;
        var best_val: f32 = -1;
        for (0..mask_n) |j| {
            if (!mask[j] and logits[j] > best_val) {
                best_val = logits[j];
                best_idx = j;
            }
        }
        if (best_val <= 0) break;
        mask[best_idx] = true;

        const surprise = -@log(best_val + 1e-10);
        if (surprise > surprise_threshold and cum_prob > 0) break;
        cum_prob += best_val;
    }

    // Zero out tokens with surprise > mu
    for (0..mask_n) |i| {
        const p = logits[i];
        if (p > 0) {
            const surprise = -@log(p + 1e-10);
            if (surprise > surprise_threshold) logits[i] = 0;
        }
    }

    // Re-normalize
    var new_sum: f32 = 0;
    for (0..n) |i| new_sum += logits[i];
    if (new_sum > 0) {
        const inv = 1.0 / new_sum;
        for (0..n) |i| logits[i] *= inv;
    }

    // Sample from filtered distribution
    var r = rng.float(f32);
    var chosen: u32 = 0;
    for (0..n) |i| {
        r -= logits[i];
        if (r <= 0) {
            chosen = @intCast(i);
            break;
        }
    }

    // Update mu: mu = mu - eta * (surprise(chosen) - tau)
    const chosen_p = logits[chosen];
    const chosen_surprise = -@log(if (chosen_p > 1e-10) chosen_p else 1e-10);
    mu.* -= eta * (chosen_surprise - tau);

    return chosen;
}

/// Modifies the logits buffer in-place.
pub fn sampleToken(logits: []f32, temperature: f32, top_k: u32, top_p: f32, rng: std.Random) u32 {
    if (temperature == 0) return argmax(logits);

    const V8 = @Vector(8, f32);
    const n = logits.len;
    const neg_inf = -std.math.inf(f32);

    // 1. Temperature scaling (SIMD)
    {
        const inv_temp = 1.0 / temperature;
        const inv_v: V8 = @splat(inv_temp);
        var si: usize = 0;
        while (si + 8 <= n) : (si += 8) {
            logits[si..][0..8].* = @as(V8, logits[si..][0..8].*) * inv_v;
        }
        while (si < n) : (si += 1) logits[si] *= inv_temp;
    }

    // 2. Top-k: find k-th largest value via min-replacement scan, mask the rest.
    // Tracks min index; rescans k-buffer only on insertion (not every element).
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
        const min_v: V8 = @splat(top_min);
        const neg_inf_v: V8 = @splat(neg_inf);
        {
            var si: usize = 0;
            while (si + 8 <= n) : (si += 8) {
                const chunk: V8 = logits[si..][0..8].*;
                logits[si..][0..8].* = @select(f32, chunk < min_v, neg_inf_v, chunk);
            }
            while (si < n) : (si += 1) {
                if (logits[si] < top_min) logits[si] = neg_inf;
            }
        }
    }

    // 3. Softmax (unnormalized — normalization deferred to sampling)
    const max_val: f32 = blk: {
        var max_v: V8 = @splat(neg_inf);
        var si: usize = 0;
        while (si + 8 <= n) : (si += 8) {
            max_v = @max(max_v, @as(V8, logits[si..][0..8].*));
        }
        var m = @reduce(.Max, max_v);
        while (si < n) : (si += 1) m = @max(m, logits[si]);
        break :blk m;
    };
    var sum: f32 = 0;
    {
        const max_v: V8 = @splat(max_val);
        const neg_inf_v: V8 = @splat(neg_inf);
        const zero_v: V8 = @splat(0.0);
        var sum_v: V8 = zero_v;
        var si: usize = 0;
        while (si + 8 <= n) : (si += 8) {
            const chunk: V8 = logits[si..][0..8].*;
            const is_neginf = chunk == neg_inf_v;
            const exp_vals = @exp(chunk - max_v);
            const result = @select(f32, is_neginf, zero_v, exp_vals);
            logits[si..][0..8].* = result;
            sum_v += result;
        }
        sum = @reduce(.Add, sum_v);
        while (si < n) : (si += 1) {
            logits[si] = if (logits[si] == neg_inf) 0 else @exp(logits[si] - max_val);
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

test "sampleToken top_p nucleus sampling" {
    // top_p=0.5 should keep only the highest-probability token(s) until
    // cumulative probability >= 0.5. With logits [0.1, 5.0, 0.2, 0.3],
    // index 1 dominates after softmax and should be the only token sampled.
    var seen = [_]bool{false} ** 4;
    for (0..50) |seed| {
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
    for (0..200) |seed| {
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
    // logits [10, 0, 0] → token 0 dominates → logprob ≈ 0
    const logits = [_]f32{ 10, 0, 0 };
    const lp = tokenLogProb(&logits, 0);
    try std.testing.expect(lp > -0.001);
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
    // mu should have been updated
    try std.testing.expect(mu != 10.0);
}
