//! KV cache eviction — norm-based scoring, victim selection, and cache compaction.
//! Implements an attention-sink-aware eviction policy: always preserves the first
//! `sink_size` positions (attention sinks) and the last `recent_window` positions
//! (working context). From the middle region, positions with the highest K-vector
//! L2 norms are retained up to a given budget.

const std = @import("std");

/// Number of initial positions always preserved (attention sinks).
pub const default_sink_size: usize = 4;
/// Number of trailing positions always preserved (recent working context).
pub const default_recent_window: usize = 128;

/// SIMD vector width for f32 accumulation.
const simd_width: usize = 8;
/// SIMD vector type for f32 dot-product accumulation.
const V8 = @Vector(simd_width, f32);

/// Compute per-position L2 norms from a flat K cache.
///
/// For each position `p` in `[0, seq_len)`, computes the L2 norm (Euclidean length)
/// of the K vector at `k_cache[p * kv_dim .. (p+1) * kv_dim]` and writes it to
/// `scores[p]`. Uses 8-wide SIMD accumulation for the dot product.
///
/// Parameters:
///   - k_cache: Flat K cache buffer, layout `[seq_len * kv_dim]`.
///   - scores:  Output buffer for L2 norms, length `>= seq_len`.
///   - seq_len: Number of cached positions.
///   - kv_dim:  Dimensionality of each K vector.
pub fn scorePositions(k_cache: [*]const f32, scores: [*]f32, seq_len: usize, kv_dim: usize) void {
    for (0..seq_len) |pos| {
        const row = k_cache + pos * kv_dim;
        var acc: V8 = @splat(0.0);
        var i: usize = 0;

        // SIMD accumulation of squared elements
        while (i + simd_width <= kv_dim) : (i += simd_width) {
            const v: V8 = row[i..][0..simd_width].*;
            acc = @mulAdd(V8, v, v, acc);
        }

        // Scalar tail
        var tail_sum: f32 = @reduce(.Add, acc);
        while (i < kv_dim) : (i += 1) {
            tail_sum = @mulAdd(f32, row[i], row[i], tail_sum);
        }

        scores[pos] = @sqrt(tail_sum);
    }
}

/// Select which positions to keep after eviction.
///
/// The policy preserves three regions unconditionally:
///   1. **Sinks** — the first `sink_size` positions (attention sinks).
///   2. **Recent** — the last `recent_window` positions (working context).
///   3. **Middle** — from the remaining positions, the highest-scoring ones up to
///      `budget - sink_size - recent_window`.
///
/// A binary search over score thresholds determines the cutoff that retains
/// exactly `middle_budget` positions from the middle region.
///
/// Parameters:
///   - scores:        Per-position L2 norms (from `scorePositions`), length `>= seq_len`.
///   - keep:          Output boolean mask, length `>= seq_len`. `true` = keep.
///   - seq_len:       Total cached positions.
///   - budget:        Target number of positions to retain (total across all regions).
///   - sink_size:     Number of leading positions to always keep.
///   - recent_window: Number of trailing positions to always keep.
///
/// Returns: The actual number of positions marked as kept.
pub fn selectVictims(
    scores: [*]const f32,
    keep: [*]bool,
    seq_len: usize,
    budget: usize,
    sink_size: usize,
    recent_window: usize,
) usize {
    // If budget covers everything, keep all
    if (budget >= seq_len) {
        for (0..seq_len) |i| keep[i] = true;
        return seq_len;
    }

    const effective_sink = @min(sink_size, budget / 2);
    // Recent window shrinks to fit budget: sinks + recent must leave room for eviction
    const max_recent = if (budget > effective_sink) budget - effective_sink else 0;
    const effective_recent = @min(@min(recent_window, max_recent), seq_len -| effective_sink);
    const protected = effective_sink + effective_recent;

    // Middle region bounds
    const middle_start = effective_sink;
    const middle_end = seq_len -| effective_recent;
    const middle_len = middle_end -| middle_start;
    const middle_budget = budget -| protected;

    // Mark sinks and recent as kept, everything else initially evicted
    for (0..seq_len) |i| {
        if (i < effective_sink or i >= seq_len - effective_recent) {
            keep[i] = true;
        } else {
            keep[i] = false;
        }
    }

    // If no middle positions to keep, we're done
    if (middle_len == 0 or middle_budget == 0) {
        return protected;
    }

    // If budget covers all middle positions, keep them all
    if (middle_budget >= middle_len) {
        for (middle_start..middle_end) |i| keep[i] = true;
        return protected + middle_len;
    }

    // Find min/max scores in the middle region for binary search bounds
    var lo: f32 = std.math.inf(f32);
    var hi: f32 = -std.math.inf(f32);
    for (middle_start..middle_end) |i| {
        lo = @min(lo, scores[i]);
        hi = @max(hi, scores[i]);
    }

    // Binary search for threshold that keeps exactly middle_budget positions.
    // Positions with score >= threshold are kept.
    const max_iters: usize = 64;
    var threshold: f32 = lo;
    for (0..max_iters) |_| {
        const mid = (lo + hi) * 0.5;
        var count: usize = 0;
        for (middle_start..middle_end) |i| {
            if (scores[i] >= mid) count += 1;
        }
        if (count > middle_budget) {
            lo = mid;
        } else {
            hi = mid;
            threshold = mid;
        }
        if (hi - lo < 1e-10) break;
    }

    // Apply threshold: keep positions with score >= threshold, up to middle_budget
    var kept: usize = 0;
    for (middle_start..middle_end) |i| {
        if (scores[i] >= threshold and kept < middle_budget) {
            keep[i] = true;
            kept += 1;
        }
    }

    return protected + kept;
}

/// Compact the cache by removing evicted positions.
///
/// Copies kept positions (where `keep[i] == true`) contiguously toward the front
/// of `cache`. Each position occupies `dim` contiguous f32 elements.
///
/// Parameters:
///   - cache:   Flat cache buffer, layout `[seq_len * dim]`. Modified in place.
///   - keep:    Boolean mask from `selectVictims`, length `>= seq_len`.
///   - seq_len: Current number of cached positions.
///   - dim:     Dimensionality per position (number of f32 elements per row).
///
/// Returns: The new sequence length (number of kept positions).
pub fn compactCache(cache: [*]f32, keep: [*]const bool, seq_len: usize, dim: usize) usize {
    var write_pos: usize = 0;
    for (0..seq_len) |read_pos| {
        if (keep[read_pos]) {
            if (write_pos != read_pos) {
                const src = cache + read_pos * dim;
                const dst = cache + write_pos * dim;
                @memcpy(dst[0..dim], src[0..dim]);
            }
            write_pos += 1;
        }
    }
    return write_pos;
}

// ── TriAttention Phase 2: Trigonometric frequency-domain scoring ───

/// Precomputed per-head calibration statistics for TriAttention scoring.
/// Generated offline from representative data (50K-960K tokens).
/// Shape: [n_freq_bands] for each array, where n_freq_bands = head_dim / 2.
pub const TriCalibration = struct {
    /// ||E[q_f]||: magnitude of mean Q vector per frequency band.
    q_center_norm: []const f32,
    /// arg(E[q_f]): phase angle of mean Q vector per frequency band.
    q_center_phase: []const f32,
    /// E[||q_f||]: expected Q magnitude per frequency band.
    q_expected_norm: []const f32,
    /// R_f: Mean Resultant Length (concentration) per frequency band. Range [0, 1].
    concentration: []const f32,
    /// RoPE frequencies: ω_f = 1/θ^(2f/d) per frequency band.
    rope_freqs: []const f32,
    /// Number of frequency bands (= head_dim / 2).
    n_bands: usize,
};

/// Eviction compression interval: trigger eviction every this many tokens.
pub const compression_interval: usize = 128;

/// Number of future offset distances for averaging the trigonometric score.
/// Uses geometric spacing: {1, 2, 4, 8, ..., 2^15}.
const n_offsets: usize = 16;

/// Compute trigonometric importance score for a single K position.
///
/// Combines two signals:
///   S_trig = Σ_f ||E[q_f]|| · ||k_f|| · cos(ω_f·Δ + φ_f)  (position-aware)
///   S_norm = Σ_f (1 - R_f) · E[||q_f||] · ||k_f||          (concentration-weighted)
///
/// The trigonometric score captures how well a key aligns with the expected
/// query distribution at each RoPE frequency, while the norm score provides
/// a fallback for bands with weak concentration (R_f → 0).
///
/// Averaged over geometric future offsets {1, 2, 4, ..., 2^15}.
pub fn triScore(
    k_vec: [*]const f32,
    head_dim: usize,
    query_pos: usize,
    key_pos: usize,
    cal: TriCalibration,
) f32 {
    const n_bands = cal.n_bands;
    std.debug.assert(n_bands == head_dim / 2);

    // Compute per-band K norms: ||k_f|| = sqrt(k_2f² + k_{2f+1}²)
    // and K phases: arg(k_f) = atan2(k_{2f+1}, k_{2f})
    var s_norm: f32 = 0;
    var s_trig_sum: f32 = 0;

    for (0..n_bands) |f| {
        const k_re = k_vec[2 * f];
        const k_im = k_vec[2 * f + 1];
        const k_norm = @sqrt(k_re * k_re + k_im * k_im);
        const k_phase = std.math.atan2(k_im, k_re);

        // Norm component: (1 - R_f) · E[||q_f||] · ||k_f||
        s_norm += (1.0 - cal.concentration[f]) * cal.q_expected_norm[f] * k_norm;

        // Trigonometric component: averaged over future offsets
        const phi = cal.q_center_phase[f] - k_phase;
        const q_cn = cal.q_center_norm[f];
        const omega = cal.rope_freqs[f];

        var trig_acc: f32 = 0;
        const base_delta: f32 = @floatFromInt(query_pos -| key_pos);
        var offset: usize = 1;
        for (0..n_offsets) |_| {
            const delta = base_delta + @as(f32, @floatFromInt(offset));
            trig_acc += q_cn * k_norm * @cos(omega * delta + phi);
            offset *= 2;
        }
        s_trig_sum += trig_acc;
    }

    return s_trig_sum / @as(f32, @floatFromInt(n_offsets)) + s_norm;
}

/// Score all positions using TriAttention trigonometric analysis.
/// For GQA: scores are computed per query head, z-normalized, then max-aggregated.
///
/// Parameters:
///   k_cache: flat [seq_len * kv_dim] f32 (pre-RoPE K vectors)
///   scores: output [seq_len] f32
///   seq_len: cached positions
///   head_dim: dimension per KV head
///   n_kv_heads: number of KV heads
///   n_q_heads: number of query heads (for GQA aggregation)
///   query_pos: current generation position
///   calibrations: [n_q_heads] calibration data (one per query head)
pub fn scorePositionsTri(
    k_cache: [*]const f32,
    scores: [*]f32,
    seq_len: usize,
    head_dim: usize,
    n_kv_heads: usize,
    n_q_heads: usize,
    query_pos: usize,
    calibrations: []const TriCalibration,
    scratch: []f32, // [n_q_heads * seq_len] for per-head scores
) void {
    const kv_dim = n_kv_heads * head_dim;
    const heads_per_group = n_q_heads / n_kv_heads;

    // Score each position from each query head's perspective
    for (0..n_q_heads) |qh| {
        const kvh = qh / heads_per_group;
        const cal = calibrations[qh];
        const head_scores = scratch[qh * seq_len ..][0..seq_len];

        for (0..seq_len) |pos| {
            const k_off = pos * kv_dim + kvh * head_dim;
            head_scores[pos] = triScore(k_cache + k_off, head_dim, query_pos, pos, cal);
        }

        // Z-score normalize this head's scores
        var mean: f32 = 0;
        for (0..seq_len) |i| mean += head_scores[i];
        mean /= @as(f32, @floatFromInt(seq_len));

        var variance: f32 = 0;
        for (0..seq_len) |i| {
            const d = head_scores[i] - mean;
            variance += d * d;
        }
        const std_dev = @sqrt(variance / @as(f32, @floatFromInt(seq_len)) + 1e-8);
        const inv_std = 1.0 / std_dev;
        for (0..seq_len) |i| head_scores[i] = (head_scores[i] - mean) * inv_std;
    }

    // Aggregate: max across query heads (a key is important if ANY head needs it)
    for (0..seq_len) |pos| {
        var max_score: f32 = scratch[pos]; // head 0
        for (1..n_q_heads) |qh| {
            max_score = @max(max_score, scratch[qh * seq_len + pos]);
        }
        scores[pos] = max_score;
    }
}

/// Generate RoPE frequency table for calibration.
/// ω_f = 1 / θ^(2f/d) for f = 0..d/2-1.
pub fn ropeFrequencies(allocator: std.mem.Allocator, head_dim: usize, theta: f32) ![]f32 {
    const n_bands = head_dim / 2;
    const freqs = try allocator.alloc(f32, n_bands);
    const inv_dim: f32 = 1.0 / @as(f32, @floatFromInt(head_dim));
    for (0..n_bands) |f| {
        const exp = @as(f32, @floatFromInt(2 * f)) * inv_dim;
        freqs[f] = 1.0 / std.math.pow(f32, theta, exp);
    }
    return freqs;
}

// ── Tests ──────────────────────────────────────────────────────────

test "scorePositions computes L2 norms" {
    // 4 positions, kv_dim = 3
    // Row 0: [1, 0, 0] → norm = 1.0
    // Row 1: [0, 3, 4] → norm = 5.0
    // Row 2: [1, 1, 1] → norm = sqrt(3) ≈ 1.7320508
    // Row 3: [2, 0, 0] → norm = 2.0
    const k_cache = [_]f32{
        1, 0, 0,
        0, 3, 4,
        1, 1, 1,
        2, 0, 0,
    };
    var scores: [4]f32 = undefined;
    scorePositions(&k_cache, &scores, 4, 3);

    const tolerance: f32 = 1e-5;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scores[0], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), scores[1], tolerance);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f32, 3.0)), scores[2], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), scores[3], tolerance);
}

test "selectVictims preserves sinks and recent window" {
    // 10 positions, budget 6, sink_size 2, recent_window 3
    // Sinks: 0, 1 (always kept)
    // Recent: 7, 8, 9 (always kept)
    // Middle: 2, 3, 4, 5, 6 — need to pick best 1 (budget 6 - 5 protected = 1)
    // Scores designed so position 4 has the highest middle score
    const scores = [_]f32{ 10, 10, 1, 2, 9, 3, 1, 10, 10, 10 };
    var keep: [10]bool = undefined;

    const kept = selectVictims(&scores, &keep, 10, 6, 2, 3);
    try std.testing.expectEqual(@as(usize, 6), kept);

    // Sinks must be kept
    try std.testing.expect(keep[0]);
    try std.testing.expect(keep[1]);

    // Recent window must be kept
    try std.testing.expect(keep[7]);
    try std.testing.expect(keep[8]);
    try std.testing.expect(keep[9]);

    // Position 4 has highest middle score (9), must be kept
    try std.testing.expect(keep[4]);

    // Other middle positions should be evicted
    try std.testing.expect(!keep[2]);
    try std.testing.expect(!keep[3]);
    try std.testing.expect(!keep[5]);
    try std.testing.expect(!keep[6]);
}

test "compactCache removes evicted positions" {
    // 4 positions, dim = 2
    // Keep positions 0 and 2, evict 1 and 3
    var cache = [_]f32{
        1, 2, // pos 0 (keep)
        3, 4, // pos 1 (evict)
        5, 6, // pos 2 (keep)
        7, 8, // pos 3 (evict)
    };
    const keep = [_]bool{ true, false, true, false };

    const new_len = compactCache(&cache, &keep, 4, 2);
    try std.testing.expectEqual(@as(usize, 2), new_len);

    // Compacted cache should have pos 0 then pos 2
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cache[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), cache[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), cache[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), cache[3], 1e-6);
}

test "triScore returns finite value with mock calibration" {
    // head_dim=4 → 2 frequency bands
    const q_cn = [_]f32{ 1.0, 0.5 };
    const q_cp = [_]f32{ 0.0, 0.0 };
    const q_en = [_]f32{ 1.0, 1.0 };
    const conc = [_]f32{ 0.8, 0.2 };
    const freqs = [_]f32{ 1.0, 0.01 };
    const cal = TriCalibration{
        .q_center_norm = &q_cn,
        .q_center_phase = &q_cp,
        .q_expected_norm = &q_en,
        .concentration = &conc,
        .rope_freqs = &freqs,
        .n_bands = 2,
    };
    const k = [_]f32{ 1.0, 0.0, 0.5, 0.5 };
    const score = triScore(&k, 4, 100, 50, cal);
    try std.testing.expect(std.math.isFinite(score));
    try std.testing.expect(score != 0.0);
}

test "ropeFrequencies produces decreasing values" {
    const freqs = try ropeFrequencies(std.testing.allocator, 8, 10000.0);
    defer std.testing.allocator.free(freqs);
    try std.testing.expectEqual(@as(usize, 4), freqs.len);
    // ω_0 = 1/θ^0 = 1.0 (highest frequency)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), freqs[0], 1e-5);
    // Each subsequent frequency should be smaller
    for (1..freqs.len) |i| {
        try std.testing.expect(freqs[i] < freqs[i - 1]);
    }
}
