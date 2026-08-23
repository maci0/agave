//! DFlash2 algorithm kernels: grouped dynamic depthwise convolution and the
//! candidate path selector (Inco AI, "DFlash 2: Keep Drafting Parallel", 2026).
//!
//! Pure, backend-independent math operating on caller-owned f32 buffers so it
//! can be unit-tested without a model or backend. The drafter model
//! (`src/models/dflash2.zig`) drives these functions with backend-dispatched
//! GEMV results.
//!
//! Reference semantics follow z-lab/dflash `model_mlx.py`:
//!   Conv_k(x)_t = Σ_{j<k} (base[j] + dyn[t,j,g(c)]) ⊙ x[t-j]
//!   edge(p→c)   = U_t(c) + ⟨A[p] ⊙ H(h_t), B[c]⟩

const std = @import("std");

/// Grouped dynamic depthwise convolution over row-major [L][E] buffers.
///
/// For row t, channel c (group g = c / group_size):
///   out[t,c] = Σ_j (base[j*E + c] + dyn[(t*K + j)*groups + g]) * h[(t-j)*E + c]
/// with out-of-range taps (t-j < 0) contributing zero.
///
/// Parameters:
///   h          - [L*E] input rows
///   dyn        - [L*K*groups] dynamic per-position corrections
///   base       - [K*E] learned base kernel for one tap stage
///   out        - [L*E] output buffer; must not alias `h`
pub fn groupedDynamicConv(
    h: []const f32,
    dyn: []const f32,
    base: []const f32,
    out: []f32,
    L: usize,
    E: usize,
    kernel: usize,
    group_size: usize,
) void {
    const groups = E / group_size;
    for (0..L) |t| {
        const out_row = out[t * E ..][0..E];
        var first = true;
        for (0..kernel) |j| {
            if (t < j) break; // zero padding before the block start
            // Source row for tap j is row t-j of the ORIGINAL input.
            const in_row = h[(t - j) * E ..][0..E];
            const base_row = base[j * E ..][0..E];
            const dyn_row = dyn[(t * kernel + j) * groups ..][0..groups];
            if (first) {
                for (0..E) |c| {
                    out_row[c] = (base_row[c] + dyn_row[c / group_size]) * in_row[c];
                }
                first = false;
            } else {
                for (0..E) |c| {
                    out_row[c] += (base_row[c] + dyn_row[c / group_size]) * in_row[c];
                }
            }
        }
        if (first) @memset(out_row, 0); // t=0 with kernel=0 guard; unreachable for kernel>=1
    }
}

/// Top-K selection over a vocab-sized score row.
///
/// Writes the K largest entries (ids and scores), ordered by descending score;
/// ties broken by lower token id. Deterministic. `ids`/`vals` must have length
/// >= top_k. Returns how many entries were written (== top_k when n >= top_k).
pub fn topK(
    scores: []const f32,
    ids: []u32,
    vals: []f32,
) usize {
    const want = @min(ids.len, vals.len);
    if (want == 0 or scores.len == 0) return 0;
    // Bounded insertion into a descending-sorted array of size `want`.
    var n: usize = 0;
    for (scores, 0..) |s, i| {
        if (n == want) {
            if (!(s > vals[n - 1])) continue;
            // Drop smallest, shift right from the insertion point.
            var pos = n - 1;
            while (pos > 0 and vals[pos - 1] < s) : (pos -= 1) {
                vals[pos] = vals[pos - 1];
                ids[pos] = ids[pos - 1];
            }
            vals[pos] = s;
            ids[pos] = @intCast(i);
        } else {
            var pos = n;
            while (pos > 0 and vals[pos - 1] < s) : (pos -= 1) {
                vals[pos] = vals[pos - 1];
                ids[pos] = ids[pos - 1];
            }
            vals[pos] = s;
            ids[pos] = @intCast(i);
            n += 1;
        }
    }
    return n;
}

pub const SelectResult = struct {
    /// Number of proposal slots resolved.
    slots: usize,
};

/// Walk the candidate lattice selecting one token per slot.
///
/// For slot t with predecessor token p and candidate c:
///   score(t, c) = unary[t*K + ci] + Σ_r A[p*R + r] * hid[t*R + r] * B[c*R + r]
/// where A/B are the predecessor/successor codebooks and hid is the projected
/// selector hidden H(h_t).
///
/// temperature == 0: greedy argmax walk.
/// temperature > 0: sample via inverse-CDF over softmax(score/T); the full
/// candidate distribution q_t is written to `q_out[t][ci]` (length K per slot)
/// and the log-probability of the chosen candidate to `chosen_logq[t]`, which
/// the lossless rejection-sampling verifier consumes.
pub fn selectPath(
    unary: []const f32,
    cand_ids: []const u32,
    hid: []const f32,
    a_codebook: []const f32,
    b_codebook: []const f32,
    anchor: u32,
    slots: usize,
    k: usize,
    rank: usize,
    temperature: f32,
    rng: std.Random,
    edge_scratch: []f32,
    path_out: []u32,
    q_out: ?[]f32,
    chosen_logq: ?[]f32,
) SelectResult {
    var pred = anchor;
    for (0..slots) |t| {
        const a_row = a_codebook[@as(usize, pred) * rank ..][0..rank];
        const hid_row = hid[t * rank ..][0..rank];
        // Precompute gated predecessor embedding: g[r] = A[p,r] * H[h,r]
        for (0..k) |ci| {
            const b_row = b_codebook[@as(usize, cand_ids[t * k + ci]) * rank ..][0..rank];
            var dot: f32 = 0;
            for (0..rank) |r| {
                dot += b_row[r] * (a_row[r] * hid_row[r]);
            }
            edge_scratch[ci] = unary[t * k + ci] + dot;
        }
        if (temperature <= 0) {
            var best: usize = 0;
            var best_val = edge_scratch[0];
            for (1..k) |ci| {
                if (edge_scratch[ci] > best_val) {
                    best_val = edge_scratch[ci];
                    best = ci;
                }
            }
            pred = cand_ids[t * k + best];
            path_out[t] = pred;
        } else {
            // In-place softmax over the K edge scores at temperature T.
            var max_v = edge_scratch[0];
            for (edge_scratch[1..k]) |s| max_v = @max(max_v, s);
            var sum: f32 = 0;
            for (0..k) |ci| {
                const e = @exp((edge_scratch[ci] - max_v) / temperature);
                edge_scratch[ci] = e;
                sum += e;
            }
            const inv_sum = if (sum > 0) 1.0 / sum else 0;
            if (q_out) |q| {
                const q_row = q[t * k ..][0..k];
                for (0..k) |ci| q_row[ci] = edge_scratch[ci] * inv_sum;
            }
            // Inverse-CDF draw.
            const r = rng.float(f32);
            var chosen: usize = k - 1;
            var cdf: f32 = 0;
            for (0..k) |ci| {
                cdf += edge_scratch[ci] * inv_sum;
                if (r < cdf) {
                    chosen = ci;
                    break;
                }
            }
            pred = cand_ids[t * k + chosen];
            path_out[t] = pred;
            if (chosen_logq) |lq| {
                lq[t] = @log(@max(edge_scratch[chosen] * inv_sum, 1e-30));
            }
        }
    }
    return .{ .slots = slots };
}

// ── Tests ────────────────────────────────────────────────────────────────────

test "groupedDynamicConv matches hand computation" {
    // E=4 channels, group_size=2 (2 groups), kernel=2, L=3.
    const E = 4;
    const L = 3;
    const kernel = 2;
    const gs = 2;
    const groups = E / gs;
    const h = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    // base: tap0 scales by 1, tap1 scales by 10 (per channel identity-ish).
    const base = [_]f32{ 1, 1, 1, 1, 10, 10, 10, 10 };
    // dynamic corrections: tap0 adds 0.5 to group 0 only; tap1 adds 1 to group 1 only.
    var dyn = [_]f32{0} ** (L * kernel * groups);
    for (0..L) |t| {
        dyn[(t * kernel + 0) * groups + 0] = 0.5;
        dyn[(t * kernel + 1) * groups + 1] = 1.0;
    }
    var out: [L * E]f32 = undefined;
    groupedDynamicConv(&h, &dyn, &base, &out, L, E, kernel, gs);

    // Row 0: only tap0 (tap1 padded): c0,c1 group0 coeff 1.5; c2,c3 group1 coeff 1.
    try std.testing.expectEqualSlices(f32, &.{ 1.5, 3.0, 3.0, 4.0 }, out[0..4]);
    // Row 1: tap0 coeff row + tap1 coeff (10 + 1 on group1) times row0.
    // c0: 1.5*5 + 10*1 = 17.5 ; c1: 1.5*6 + 10*2 = 29 ; c2: 1*7 + 11*3 = 40 ; c3: 1*8 + 11*4 = 52
    try std.testing.expectEqualSlices(f32, &.{ 17.5, 29.0, 40.0, 52.0 }, out[4..8]);
    // Row 2: tap0*row2 + tap1 coeff*row1.
    // c0: 1.5*9 + 10*5 = 63.5 ; c1: 1.5*10 + 10*6 = 75 ;
    // c2: 1*11 + 11*7 = 88 ; c3: 1*12 + 11*8 = 100
    try std.testing.expectEqualSlices(f32, &.{ 63.5, 75.0, 88.0, 100.0 }, out[8..12]);
}

test "topK selects largest entries descending" {
    const scores = [_]f32{ 0.1, 5.0, -3.0, 4.0, 4.5, 5.0 };
    var ids: [3]u32 = undefined;
    var vals: [3]f32 = undefined;
    const n = topK(&scores, &ids, &vals);
    try std.testing.expectEqual(@as(usize, 3), n);
    try std.testing.expectEqual(@as(u32, 1), ids[0]); // 5.0 (lower id wins tie)
    try std.testing.expectEqual(@as(u32, 5), ids[1]); // 5.0
    try std.testing.expectEqual(@as(u32, 4), ids[2]); // 4.5
    try std.testing.expectEqual(vals[0], vals[1]);
    try std.testing.expect(vals[1] > vals[2]);
}

test "topK handles short inputs" {
    const scores = [_]f32{ 2.0 };
    var ids: [4]u32 = undefined;
    var vals: [4]f32 = undefined;
    const n = topK(&scores, &ids, &vals);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(@as(u32, 0), ids[0]);
}

test "selectPath greedy follows strongest edges" {
    // 2 slots, K=2 candidates each, rank=2 codebooks.
    const k = 2;
    const rank = 2;
    const unary = [_]f32{
        10.0, 0.0, // slot 0: candidate 0 strongly preferred by logits
        0.0,  20.0, // slot 1: candidate 1 preferred by logits
    };
    const cand_ids = [_]u32{ 1, 3, 5, 7 };
    // Hidden projection all zeros → edges vanish → pure unary walk.
    const hid = [_]f32{ 0, 0, 0, 0 };
    var a_codebook: [8 * rank]f32 = undefined;
    @memset(&a_codebook, 1);
    var b_codebook: [8 * rank]f32 = undefined;
    @memset(&b_codebook, 1);
    var path: [2]u32 = undefined;
    var edge: [2]f32 = undefined;
    var prng = std.Random.Xoshiro256.init(1);
    _ = selectPath(&unary, &cand_ids, &hid, &a_codebook, &b_codebook, 3, 2, k, rank, 0, prng.random(), &edge, &path, null, null);
    try std.testing.expectEqual(@as(u32, 1), path[0]);
    try std.testing.expectEqual(@as(u32, 7), path[1]);
}

test "selectPath edge term overrides weak unary preference" {
    // Slot 0: unary prefers cand id 1 (val 12 vs 10), but the bilinear edge
    // strongly favors cand id 0. Edge must win.
    const k = 2;
    const rank = 2;
    const unary = [_]f32{ 10.0, 12.0 };
    const cand_ids = [_]u32{ 0, 1 };
    // gate vector g = A[anchor]⊙H = [2, 2]; B[cand0]=[3,3] → edge 12; B[cand1]=[-3,-3] → −12.
    const hid = [_]f32{ 2.0, 2.0 };
    var a_codebook: [4 * rank]f32 = undefined;
    @memset(&a_codebook, 1);
    var b_codebook: [2 * rank]f32 = undefined;
    b_codebook[0] = 3;
    b_codebook[1] = 3;
    b_codebook[2] = -3;
    b_codebook[3] = -3;
    var path: [1]u32 = undefined;
    var edge: [2]f32 = undefined;
    var prng = std.Random.Xoshiro256.init(1);
    _ = selectPath(&unary, &cand_ids, &hid, &a_codebook, &b_codebook, 0, 1, k, rank, 0, prng.random(), &edge, &path, null, null);
    try std.testing.expectEqual(@as(u32, 0), path[0]);
}

test "selectPath sampling produces valid distribution" {
    const k = 4;
    const rank = 2;
    const unary = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const cand_ids = [_]u32{ 0, 1, 2, 3 };
    const hid = [_]f32{ 0, 0 };
    var a_codebook: [4 * rank]f32 = undefined;
    @memset(&a_codebook, 0);
    var b_codebook: [4 * rank]f32 = undefined;
    @memset(&b_codebook, 0);
    var path: [1]u32 = undefined;
    var edge: [4]f32 = undefined;
    var q: [4]f32 = undefined;
    var lq: [1]f32 = undefined;
    var counts = [_]f32{ 0, 0, 0, 0 };
    const trials = 2000;
    for (0..trials) |seed| {
        var prng = std.Random.Xoshiro256.init(seed);
        _ = selectPath(&unary, &cand_ids, &hid, &a_codebook, &b_codebook, 0, 1, k, rank, 1.0, prng.random(), &edge, &path, &q, &lq);
        var qsum: f32 = 0;
        for (q) |v| {
            try std.testing.expect(v >= 0);
            qsum += v;
        }
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), qsum, 1e-4);
        counts[path[0]] += 1;
        try std.testing.expect(lq[0] <= 0);
    }
    // Higher-scored candidates must be drawn more often.
    try std.testing.expect(counts[3] > counts[0]);
}

test "selectPath multi-slot chains through predecessors" {
    // Slot 1's edge depends on slot 0's pick: A rows differ per token so the
    // second-slot decision flips based on which token slot 0 chose.
    const k = 2;
    const rank = 1;
    const unary = [_]f32{
        5.0, 5.0, // slot 0 tied on unary
        0.0, 1.0, // slot 1: slight unary preference for cand 1
    };
    const cand_ids = [_]u32{ 0, 1, 0, 1 };
    // A[token0]=[10], A[token1]=[-10]; H=[1].
    // B[cand0(slot1)]=[1] → edge(A0)=10, edge(A1)=−10
    // B[cand1(slot1)]=[−1] → edge(A0)=−10, edge(A1)=10
    const hid = [_]f32{ 1.0, 1.0 };
    var a_codebook: [2]f32 = .{ 10.0, -10.0 };
    var b_codebook: [2 * 2]f32 = .{ 1.0, -1.0, 1.0, -1.0 };
    var path: [2]u32 = undefined;
    var edge: [2]f32 = undefined;
    var prng = std.Random.Xoshiro256.init(7);
    _ = selectPath(&unary, &cand_ids, &hid, &a_codebook, &b_codebook, 0, 2, k, rank, 0, prng.random(), &edge, &path, null, null);
    // Slot 0 ties at 5 → first index wins → token 0. Slot 1 then sees
    // edge(token0→cand0)=10 vs edge(token0→cand1)=−10 → cand0 wins despite
    // its weaker unary score.
    try std.testing.expectEqual(@as(u32, 0), path[0]);
    try std.testing.expectEqual(@as(u32, 0), path[1]);
}

test "fuzz conv against naive reference" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const E = 4;
            const gs = 2;
            const kernel: usize = 1 + smith.valueWithHash(u2, 0) % 3; // 1..3
            const L: usize = 1 + smith.valueWithHash(u3, 1) % 4; // 1..4
            var h: [16]f32 = undefined;
            var base: [12]f32 = undefined;
            var dyn: [48]f32 = undefined;
            var out: [16]f32 = undefined;
            for (&h) |*v| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, 2))) / 8.0;
            for (&base) |*v| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, 3))) / 8.0;
            for (&dyn) |*v| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, 4))) / 8.0;
            groupedDynamicConv(&h, &dyn, &base, &out, L, E, kernel, gs);
            // Naive reference.
            for (0..L) |t| {
                for (0..E) |c| {
                    var expect: f32 = 0;
                    for (0..kernel) |j| {
                        if (t < j) continue;
                        const coeff = base[j * E + c] + dyn[(t * kernel + j) * (E / gs) + c / gs];
                        expect += coeff * h[(t - j) * E + c];
                    }
                    try std.testing.expectApproxEqAbs(expect, out[t * E + c], 1e-4);
                }
            }
        }
    }.f, .{});
}
