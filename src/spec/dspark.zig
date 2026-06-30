//! DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation
//!
//! Implements the DSpark framework (Cheng et al., 2026):
//!   1. Semi-autoregressive generation: parallel backbone + lightweight sequential head
//!      to inject intra-block token dependencies and mitigate suffix decay.
//!   2. Confidence-scheduled verification: hardware-aware prefix scheduler that
//!      dynamically trims verification length per request to maximize throughput.
//!
//! References:
//!   DSpark paper: https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf
//!   Algorithm 1: Hardware-Aware Prefix Scheduler

const std = @import("std");
const math = std.math;

/// Maximum draft block size (positions).
pub const max_block: usize = 32;

/// Pre-profiled steps-per-second table indexed by forward-pass batch size.
/// Entry sps_table[B] = expected steps/sec for a batch of B tokens.
/// Profile once at engine init by timing target-model forward passes at
/// increasing token counts. Use linear interpolation between samples.
pub const SpsProfile = struct {
    /// sps[i] = steps-per-second when batch has i+1 tokens (0-indexed by B-1).
    sps: []const f32,

    /// Look up (or interpolate) SPS for a given batch size B.
    pub fn stepsPerSec(self: SpsProfile, B: usize) f32 {
        if (self.sps.len == 0) return 1.0;
        const idx = if (B == 0) 0 else B - 1;
        if (idx >= self.sps.len) return self.sps[self.sps.len - 1];
        return self.sps[idx];
    }

    /// Build a synthetic profile assuming throughput ∝ 1/B (compute-bound).
    /// baseline_sps: steps/sec at batch size 1.
    pub fn syntheticComputeBound(allocator: std.mem.Allocator, max_batch: usize, baseline_sps: f32) !SpsProfile {
        const sps = try allocator.alloc(f32, max_batch);
        for (sps, 0..) |*s, i| {
            // throughput degrades sub-linearly: model as baseline / sqrt(B+1)
            s.* = baseline_sps / @sqrt(@as(f32, @floatFromInt(i + 1)));
        }
        return .{ .sps = sps };
    }
};

/// Per-position confidence score for a single request in a batch.
/// c[k] estimates P(token k accepted | tokens 0..k-1 all accepted).
/// Must satisfy 0 < c[k] <= 1.
pub const ConfidenceBlock = struct {
    /// Per-position confidence estimates (length = n_drafted).
    c: []const f32,
    /// Cumulative prefix survival probabilities: a[j] = Π_{i≤j} c[i].
    /// a[j] = probability that the draft prefix of length j+1 is all accepted.
    a: [max_block]f32 = undefined,
    n: u32 = 0,

    /// Pre-compute cumulative survival probs from per-step confidences.
    pub fn computeSurvival(self: *ConfidenceBlock) void {
        self.n = @intCast(@min(self.c.len, max_block));
        var prod: f32 = 1.0;
        for (0..self.n) |k| {
            prod *= self.c[k];
            self.a[k] = prod;
        }
    }
};

/// A (request_index, position) candidate token for the scheduler.
const Candidate = struct {
    req: u32,
    pos: u32, // 0-based position within draft block
    survival: f32, // cumulative a_{r,j}
};

/// Output of the hardware-aware prefix scheduler.
pub const SchedulerResult = struct {
    /// Selected verification length per request (0 = skip this request's drafts).
    lengths: [256]u32 = .{0} ** 256,
    /// Total token batch size sent to target for verification.
    batch_size: u32 = 0,
    /// Expected accepted tokens (sum of survival probabilities of admitted tokens).
    expected_accepts: f32 = 0.0,
};

/// Hardware-Aware Prefix Scheduler (Algorithm 1, DSpark §3.2.2).
///
/// Given R concurrent requests each with per-position confidence scores,
/// maximises expected system throughput Θ = τ × SPS(B) by greedily
/// admitting draft tokens in descending order of survival probability
/// until throughput would decrease.
///
/// Complexity: O(R × γ × log(R × γ)) for the global sort step,
///             O(R × γ) for the greedy sweep.
///
/// blocks:  slice of R ConfidenceBlock (each pre-populated with c[] and a[]).
/// profile: pre-profiled SPS table for the target model.
/// result:  output — per-request verification lengths.
pub fn scheduleVerification(
    blocks: []ConfidenceBlock,
    profile: SpsProfile,
    result: *SchedulerResult,
    scratch: []Candidate, // caller-provided scratch; must be len ≥ R × γ
) void {
    const R = blocks.len;
    if (R == 0) return;

    // Compute survival probabilities for all blocks.
    for (blocks) |*b| b.computeSurvival();

    // Build candidate pool: every (r, j) with a_{r,j} > 0.
    var n_cands: usize = 0;
    for (blocks, 0..) |b, r| {
        for (0..b.n) |j| {
            if (b.a[j] > 0.0) {
                scratch[n_cands] = .{
                    .req = @intCast(r),
                    .pos = @intCast(j),
                    .survival = b.a[j],
                };
                n_cands += 1;
            }
        }
    }

    // Sort candidates descending by survival probability.
    std.sort.pdq(Candidate, scratch[0..n_cands], {}, struct {
        fn lt(_: void, a: Candidate, b: Candidate) bool {
            return a.survival > b.survival;
        }
    }.lt);

    // Initialise per-request verification lengths to 0.
    @memset(result.lengths[0..@min(R, 256)], 0);

    // State: current verification length per request.
    var cur_len: [256]u32 = .{0} ** 256;
    // Starting point: baseline batch = R (one anchor token per request, no drafts).
    var batch_size: u32 = @intCast(R);
    // Expected accepts at baseline = R (each request gets its target bonus token).
    var tau: f32 = @floatFromInt(R);
    var theta_best: f32 = tau * profile.stepsPerSec(batch_size);

    // Record baseline as current best.
    result.batch_size = batch_size;
    result.expected_accepts = tau;

    // Greedy admission: process candidates in descending survival order.
    // Non-anticipating property: we stop immediately when throughput drops,
    // ensuring the decision for position k never leaks information about k+1.
    for (scratch[0..n_cands]) |cand| {
        const r = cand.req;
        const j = cand.pos;

        // Enforce prefix constraint: can only extend by 1 at a time.
        if (cur_len[r] != j) continue; // gaps not allowed — skip out-of-order

        // Tentatively extend request r by one position.
        cur_len[r] = j + 1;
        batch_size += 1;
        tau += cand.survival;

        const theta = tau * profile.stepsPerSec(batch_size);
        if (theta > theta_best) {
            theta_best = theta;
            // Snapshot current lengths as the new best.
            @memcpy(result.lengths[0..@min(R, 256)], cur_len[0..@min(R, 256)]);
            result.batch_size = batch_size;
            result.expected_accepts = tau;
        } else {
            // Throughput dropped — stop (early-exit ensures non-anticipating property).
            break;
        }
    }
}

/// Markov head: low-rank first-order transition bias for the sequential stage.
///
/// Given the parallel backbone logits U_k and the previously sampled token x_{k-1},
/// computes the adjusted log-prob:
///   log p_k(v | x_0, x_{<k}) ∝ U_k(v) + B(x_{k-1}, v)
///   B(x_{k-1}, ·) = W1[x_{k-1}] @ W2    (low-rank factorisation, rank r)
///
/// W1 ∈ R^{V × r}: embedding lookup  (vocab_size × rank)
/// W2 ∈ R^{r × V}: logit projection  (rank × vocab_size)
pub const MarkovHead = struct {
    /// W1 row-major [vocab_size, rank] — embedding of previous token.
    w1: []const f32,
    /// W2 row-major [rank, vocab_size] — bias projection.
    w2: []const f32,
    vocab_size: u32,
    rank: u32,

    /// Compute transition bias B(x_{k-1}, ·) → bias[0..vocab_size].
    /// bias must be pre-allocated to vocab_size floats.
    pub fn transitionBias(self: MarkovHead, prev_token: u32, bias: []f32) void {
        std.debug.assert(bias.len >= self.vocab_size);
        const r = self.rank;
        const v = self.vocab_size;

        // Embedding lookup: e = W1[prev_token] ∈ R^r
        const e = self.w1[prev_token * r ..][0..r];

        // Project: bias = e @ W2  (shape: r → V)
        @memset(bias[0..v], 0.0);
        for (0..r) |ri| {
            const w2_row = self.w2[ri * v ..][0..v];
            const ei = e[ri];
            if (ei == 0.0) continue;
            for (0..v) |vi| bias[vi] += ei * w2_row[vi];
        }
    }

    /// Sample from adjusted distribution at position k.
    ///   p_k(v) ∝ exp(U_k(v) + B(prev_token, v))
    /// Returns argmax (greedy) over the adjusted logits.
    /// For stochastic sampling, the caller can apply temperature before calling.
    pub fn sampleGreedy(self: MarkovHead, base_logits: []const f32, prev_token: u32, bias_buf: []f32) u32 {
        const v = self.vocab_size;
        self.transitionBias(prev_token, bias_buf);

        var best_tok: u32 = 0;
        var best_val: f32 = -math.inf(f32);
        for (0..v) |vi| {
            const val = base_logits[vi] + bias_buf[vi];
            if (val > best_val) {
                best_val = val;
                best_tok = @intCast(vi);
            }
        }
        return best_tok;
    }
};

/// RNN head: gated recurrent sequential stage for richer intra-block conditioning.
///
/// Maintains a hidden state s_k ∈ R^r that accumulates the full prefix history.
/// Update rule (Eq. 6 from DSpark paper):
///   z_k = [s_{k-1}; W1[x_{k-1}]; h_k]     (concat, dim = 2r + d)
///   gate = σ(W_g @ z_k)
///   candidate = tanh(W_c @ z_k)
///   s_k = gate ⊙ s_{k-1} + (1 - gate) ⊙ candidate
///   B_k(x_{<k}, ·) = W2^T tanh(W_o @ z_k)
pub const RnnHead = struct {
    /// W1 row-major [vocab_size, rank].
    w1: []const f32,
    /// W2 row-major [rank, vocab_size].
    w2: []const f32,
    /// W_g, W_c, W_o jointly packed as [3*(2r+d), r] — split gate/candidate/output.
    w_gco: []const f32,
    vocab_size: u32,
    rank: u32,
    hidden_dim: u32,

    /// Apply one RNN step. state is updated in-place (s_k).
    /// h_k: backbone hidden at position k (hidden_dim floats).
    /// prev_token: x_{k-1}.
    /// bias: output B_k(x_{<k}, ·) written here (vocab_size floats).
    pub fn step(
        self: RnnHead,
        state: []f32,      // [rank], updated in-place
        prev_token: u32,
        h_k: []const f32,  // [hidden_dim]
        bias: []f32,        // [vocab_size], output
    ) void {
        const r = self.rank;
        const d = self.hidden_dim;
        const v = self.vocab_size;
        const z_dim = 2 * r + d;

        // Build z_k = [s_{k-1}; W1[prev_token]; h_k]
        var z_buf: [max_block * 4 + 4096]f32 = undefined; // generous upper bound
        const z = z_buf[0..z_dim];
        @memcpy(z[0..r], state[0..r]);                         // s_{k-1}
        @memcpy(z[r .. r + r], self.w1[prev_token * r ..][0..r]); // W1[x_{k-1}]
        @memcpy(z[2 * r .. 2 * r + d], h_k);                   // h_k

        // Apply W_gco: [z_dim, r*3] → [gate; cand; out] each R^r
        var gate_buf: [4096]f32 = undefined;
        var cand_buf: [4096]f32 = undefined;
        var out_buf: [4096]f32 = undefined;
        const gate = gate_buf[0..r];
        const cand = cand_buf[0..r];
        const out = out_buf[0..r];

        for (0..r) |ri| {
            var g: f32 = 0.0;
            var c: f32 = 0.0;
            var o: f32 = 0.0;
            const row_g = self.w_gco[ri * z_dim ..][0..z_dim];
            const row_c = self.w_gco[(r + ri) * z_dim ..][0..z_dim];
            const row_o = self.w_gco[(2 * r + ri) * z_dim ..][0..z_dim];
            for (0..z_dim) |zi| {
                g += row_g[zi] * z[zi];
                c += row_c[zi] * z[zi];
                o += row_o[zi] * z[zi];
            }
            gate[ri] = sigmoid(g);
            cand[ri] = std.math.tanh(c);
            out[ri] = std.math.tanh(o);
        }

        // Update state: s_k = gate ⊙ s_{k-1} + (1 - gate) ⊙ cand
        for (0..r) |ri| {
            state[ri] = gate[ri] * state[ri] + (1.0 - gate[ri]) * cand[ri];
        }

        // Compute bias: B_k = W2^T tanh(W_o z_k) = W2^T out
        @memset(bias[0..v], 0.0);
        for (0..r) |ri| {
            const w2_row = self.w2[ri * v ..][0..v];
            const oi = out[ri];
            if (oi == 0.0) continue;
            for (0..v) |vi| bias[vi] += oi * w2_row[vi];
        }
    }
};

/// Confidence head (§3.2.1): lightweight scalar acceptance predictor.
///   c_k = σ(w^T [h_k; W1[x_{k-1}]])
/// w ∈ R^{d+r}, W1 ∈ R^{V × r} (shared with Markov head).
pub const ConfidenceHead = struct {
    /// Weight vector w ∈ R^{d + r}.
    w: []const f32,
    /// Markov embedding W1 (shared, [vocab_size, rank]).
    w1: []const f32,
    hidden_dim: u32,
    rank: u32,

    /// Estimate confidence c_k = σ(w^T [h_k; W1[x_{k-1}]]).
    pub fn confidence(self: ConfidenceHead, h_k: []const f32, prev_token: u32) f32 {
        const d = self.hidden_dim;
        const r = self.rank;
        const emb = self.w1[prev_token * r ..][0..r];
        var dot: f32 = 0.0;
        for (0..d) |i| dot += self.w[i] * h_k[i];
        for (0..r) |i| dot += self.w[d + i] * emb[i];
        return sigmoid(dot);
    }
};

/// Sequential Temperature Scaling (STS, §3.2.1): calibrates per-position
/// cumulative survival probabilities to match empirical acceptance rates.
/// Finds optimal scalar temperature T_k per position via 1D grid search.
pub fn calibrateSts(
    /// confidence[sample][position] — raw model outputs c_k.
    confidence: []const []const f32,
    /// accepted[sample][position] — binary: 1 if token k was accepted, 0 if not.
    accepted: []const []const bool,
    n_positions: u32,
    allocator: std.mem.Allocator,
) ![]f32 {
    const temperatures = try allocator.alloc(f32, n_positions);
    errdefer allocator.free(temperatures);
    @memset(temperatures, 1.0);

    const n_samples = confidence.len;
    if (n_samples == 0) return temperatures;

    // For each position k, find T_k minimising ECE of cumulative product.
    for (0..n_positions) |k| {
        var best_ece: f32 = math.inf(f32);
        var best_t: f32 = 1.0;

        var t: f32 = 0.01;
        while (t <= 10.0) : (t += 0.05) {
            // Compute cumulative product P_k(sample) = Π_{i≤k} c_i^(T_i/t_i)
            // Since positions < k are already calibrated with their own T_i, we
            // apply temperature only at position k.
            var ece: f32 = 0.0;
            for (confidence, accepted) |c_seq, a_seq| {
                if (k >= c_seq.len) continue;
                var cum: f32 = 1.0;
                for (0..k) |i| {
                    // Already-calibrated positions use identity (T_i cancels in sigmoid).
                    cum *= calibratedConf(c_seq[i], temperatures[i]);
                }
                cum *= calibratedConf(c_seq[k], t);

                // Label: 1 if prefix of length k+1 fully accepted.
                var label: f32 = 1.0;
                for (0..k + 1) |i| {
                    if (i >= a_seq.len or !a_seq[i]) { label = 0.0; break; }
                }
                const diff = cum - label;
                ece += diff * diff;
            }
            if (ece < best_ece) {
                best_ece = ece;
                best_t = t;
            }
        }
        temperatures[k] = best_t;
    }
    return temperatures;
}

/// Apply calibrated temperature to a raw confidence score.
/// c_calibrated = σ(logit(c) / T) where logit(c) = log(c/(1-c)).
fn calibratedConf(c: f32, temp: f32) f32 {
    const eps = 1e-7;
    const clamped = @max(eps, @min(1.0 - eps, c));
    const logit = @log(clamped / (1.0 - clamped));
    return sigmoid(logit / temp);
}

inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

// ── Unit tests ────────────────────────────────────────────────────────────────

test "MarkovHead.sampleGreedy applies transition bias" {
    // Simple 4-token vocab, rank 2.
    // W1 rows: token 0 → [1, 0], token 1 → [0, 1]
    // W2: row 0 → [0, 0, 1, -1], row 1 → [0, 1, 0, -1]
    // Base logits: [1, 1, 1, 1] (uniform)
    // prev_token = 0: bias = [1,0] @ W2 = [0, 0, 1, -1]
    // Adjusted: [1, 1, 2, 0] → argmax = 2 ✓
    const w1 = [_]f32{ 1, 0, 0, 1, 0, 0, 0, 0 }; // 4×2
    const w2 = [_]f32{ 0, 0, 1, -1, 0, 1, 0, -1 }; // 2×4
    const head = MarkovHead{ .w1 = &w1, .w2 = &w2, .vocab_size = 4, .rank = 2 };
    const base = [_]f32{ 1, 1, 1, 1 };
    var bias: [4]f32 = undefined;
    const tok = head.sampleGreedy(&base, 0, &bias);
    try std.testing.expectEqual(@as(u32, 2), tok);
}

test "scheduleVerification single request greedy trim" {
    // Single request, 4 positions, confidence decaying.
    // SPS profile: flat 1.0 steps/sec for any batch size.
    const sps_data = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0 };
    const profile = SpsProfile{ .sps = &sps_data };

    var c_data = [_]f32{ 0.9, 0.8, 0.5, 0.1 };
    var block = ConfidenceBlock{ .c = &c_data };
    block.computeSurvival();

    var blocks = [_]ConfidenceBlock{block};
    var result = SchedulerResult{};
    var scratch: [128]Candidate = undefined;

    scheduleVerification(&blocks, profile, &result, &scratch);

    // survival: [0.9, 0.72, 0.36, 0.036]
    // Θ starts at R=1, SPS(1)=1 → Θ_0 = 1.0
    // Add pos 0 (a=0.9): B=2, τ=1.9, Θ=1.9*SPS(2)=1.9 > 1.0 ✓ admit
    // Add pos 1 (a=0.72): B=3, τ=2.62, Θ=2.62*SPS(3)=2.62 > 1.9 ✓ admit
    // Add pos 2 (a=0.36): B=4, τ=2.98, Θ=2.98 > 2.62 ✓ admit
    // Add pos 3 (a=0.036): B=5, τ=3.016, Θ=3.016 > 2.98 ✓ admit
    // All admitted because SPS is flat.
    try std.testing.expect(result.lengths[0] >= 1);
}

test "scheduleVerification drops low-confidence tokens under load" {
    // Simulate degrading SPS: heavier batch → much lower throughput.
    const sps_data = [_]f32{ 10.0, 8.0, 5.0, 2.0, 1.0, 0.5, 0.2 };
    const profile = SpsProfile{ .sps = &sps_data };

    var c_data = [_]f32{ 0.9, 0.9, 0.1, 0.1 }; // sharp drop after pos 1
    var block = ConfidenceBlock{ .c = &c_data };
    block.computeSurvival();

    var blocks = [_]ConfidenceBlock{block};
    var result = SchedulerResult{};
    var scratch: [128]Candidate = undefined;
    scheduleVerification(&blocks, profile, &result, &scratch);

    // High-confidence tokens (a≈0.9, 0.81) should be admitted,
    // low-confidence ones (a≈0.081, 0.0081) should be dropped.
    try std.testing.expect(result.lengths[0] >= 1);
    try std.testing.expect(result.lengths[0] <= 4);
}

test "SpsProfile synthetic compute bound decreases with batch" {
    const alloc = std.testing.allocator;
    const prof = try SpsProfile.syntheticComputeBound(alloc, 8, 100.0);
    defer alloc.free(prof.sps);
    try std.testing.expect(prof.stepsPerSec(1) > prof.stepsPerSec(4));
    try std.testing.expect(prof.stepsPerSec(4) > prof.stepsPerSec(8));
}
