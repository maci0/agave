//! Speculative decoding orchestrator.
//!
//! Verification modes (selected via --spec-mode):
//! - Standard greedy: single-path draft → sequential argmax verification
//! - Rejection sampling: stochastic acceptance with temperature (Leviathan et al. 2023)
//! - DDTree: tree-structured draft → greedy tree walk (Ringel & Romano, 2026)
//! - Self-speculative: layer-skip self-drafting (no separate draft model)
//! - N-gram: history-based n-gram prediction (no draft model)
//! - MTP: multi-token prediction heads

const std = @import("std");
const Model = @import("../models/model.zig").Model;
const math_ops = @import("../ops/math.zig");
const ddtree = @import("ddtree.zig");

pub const max_draft_tokens: usize = 32;
const log_softmax_eps: f32 = 1e-10;
/// Minimum verification rounds before adaptive K profiling engages.
const adaptive_k_min_rounds: u64 = 10;
/// Minimum samples per draft length before including it in optimal K selection.
const adaptive_k_min_samples: u32 = 3;

/// Pre-allocated state for speculative decoding.
pub const SpecState = struct {
    draft_tokens: [max_draft_tokens]u32 = undefined,
    /// Log-softmax distributions from draft model at each depth.
    draft_log_probs: []f32 = &.{},
    /// Slices into draft_log_probs for DDTree presort.
    depth_slices: [max_draft_tokens][]const f32 = undefined,
    /// Pre-allocated buffer for target probabilities (rejection sampling).
    sampling_buf: []f32 = &.{},
    n_draft: u32 = 0,
    k: u32,
    vocab_size: u32,

    total_accepted: u64 = 0,
    total_drafted: u64 = 0,
    total_rounds: u64 = 0,

    /// Profile-guided adaptive K: adjust draft length based on acceptance history.
    /// Tracks per-K acceptance rates to find optimal draft length.
    k_accept_counts: [max_draft_tokens]u32 = .{0} ** max_draft_tokens,
    k_total_counts: [max_draft_tokens]u32 = .{0} ** max_draft_tokens,
    adaptive_k_enabled: bool = false,

    /// FR-Spec token mask: if non-null, a boolean array of vocab_size where mask[id]=true
    /// means the token is in the high-frequency set. Draft logits for mask[id]=false tokens
    /// are set to -inf before argmax, restricting proposals to frequent tokens only.
    /// Improves acceptance rate by biasing draft toward tokens the target model prefers.
    /// Built from --spec-token-map <file> (one token ID per line) via buildTokenMask().
    /// Owned by caller (allocator.free(mask) in deinit).
    token_mask: ?[]bool = null,

    pub fn init(allocator: std.mem.Allocator, k: u32, vocab_size: u32) !SpecState {
        const draft_log_probs = try allocator.alloc(f32, max_draft_tokens * vocab_size);
        errdefer allocator.free(draft_log_probs);
        const sampling_buf = try allocator.alloc(f32, vocab_size);
        return .{
            .k = k,
            .vocab_size = vocab_size,
            .draft_log_probs = draft_log_probs,
            .sampling_buf = sampling_buf,
        };
    }

    pub fn deinit(self: *SpecState, allocator: std.mem.Allocator) void {
        if (self.draft_log_probs.len > 0) allocator.free(self.draft_log_probs);
        if (self.sampling_buf.len > 0) allocator.free(self.sampling_buf);
    }

    pub fn acceptanceRate(self: SpecState) f32 {
        if (self.total_drafted == 0) return 0;
        return @as(f32, @floatFromInt(self.total_accepted)) / @as(f32, @floatFromInt(self.total_drafted));
    }

    pub fn meanAccepted(self: SpecState) f32 {
        if (self.total_rounds == 0) return 0;
        return @as(f32, @floatFromInt(self.total_accepted)) / @as(f32, @floatFromInt(self.total_rounds));
    }

    pub fn recordRound(self: *SpecState, accepted: u32) void {
        self.total_accepted += accepted;
        self.total_drafted += self.n_draft;
        self.total_rounds += 1;

        // Profile-guided: record per-K acceptance for adaptive tuning
        if (self.adaptive_k_enabled and self.n_draft > 0 and self.n_draft <= max_draft_tokens) {
            const ki = self.n_draft - 1;
            self.k_total_counts[ki] += 1;
            self.k_accept_counts[ki] += accepted;
        }
    }

    /// Compute optimal K based on acceptance history.
    /// Expected value: E[tokens] = k × accept_rate(k) + 1 (bonus token).
    /// Cost model: verify cost ≈ 1 forward pass regardless of k (tree verify).
    /// Optimal k maximizes E[tokens] / cost = k × accept_rate(k) + 1.
    /// Returns the configured k if insufficient data for profiling.
    pub fn optimalK(self: *const SpecState) u32 {
        if (!self.adaptive_k_enabled or self.total_rounds < adaptive_k_min_rounds) return self.k;

        var best_k: u32 = self.k;
        var best_ev: f32 = 0;

        for (0..@min(self.k, max_draft_tokens)) |ki| {
            if (self.k_total_counts[ki] < adaptive_k_min_samples) continue;
            const accept_rate = @as(f32, @floatFromInt(self.k_accept_counts[ki])) /
                @as(f32, @floatFromInt(self.k_total_counts[ki] * (@as(u32, @intCast(ki)) + 1)));
            const k_val: f32 = @floatFromInt(ki + 1);
            const ev = k_val * accept_rate + 1.0;
            if (ev > best_ev) {
                best_ev = ev;
                best_k = @intCast(ki + 1);
            }
        }
        return best_k;
    }
};

pub const SpecResult = struct {
    accepted: u32,
    next_token: u32,
};

/// Generate draft tokens using MTP (Multi-Token Prediction) heads.
/// Each depth produces one draft token from a lightweight single-layer forward pass.
pub fn draftMtp(state: *SpecState, model: *Model, last_token: u32) u32 {
    const max_depth = model.getMtpDepth();
    if (max_depth == 0) return 0;
    var n: u32 = 0;
    const effective_k = @min(state.k, max_depth);
    while (n < effective_k and n < max_draft_tokens) {
        const tok = model.mtpForward(last_token, n) catch break;
        state.draft_tokens[n] = tok;
        n += 1;
    }
    state.n_draft = n;
    return n;
}

/// Generate K draft tokens without saving logits (fastest for self-draft).
pub fn draft(state: *SpecState, draft_model: *Model, last_token: u32) u32 {
    var tok = last_token;
    var n: u32 = 0;
    while (n < state.k and n < max_draft_tokens) {
        tok = draft_model.forward(tok) catch break;
        state.draft_tokens[n] = tok;
        n += 1;
    }
    state.n_draft = n;
    return n;
}

/// Generate K draft tokens, saving logit distributions at each step.
/// FR-Spec mask (correct implementation): restrict logits to token_mask.
/// MUST be called right after @memcpy(dst, model_logits) while dst has original values.
/// Sets all positions where token_mask[i]=false to -inf. O(vocab_size) per call.
fn applyFrSpecMask(dst: []f32, token_mask: []const bool) void {
    const n = @min(dst.len, token_mask.len);
    for (dst[0..n], token_mask[0..n]) |*v, in_map| {
        if (!in_map) v.* = -std.math.inf(f32);
    }
    // Any logits beyond token_mask.len are also masked out
    for (dst[n..]) |*v| v.* = -std.math.inf(f32);
}

/// Build a boolean token mask from a whitespace/newline-separated list of token IDs.
/// Returns an allocated []bool of size vocab_size (caller owns and must free).
pub fn buildTokenMask(allocator: std.mem.Allocator, token_map_path: []const u8, vocab_size: u32) ![]bool {
    const posix = std.posix;
    const mask = try allocator.alloc(bool, vocab_size);
    errdefer allocator.free(mask);
    @memset(mask, false);

    // Read file via POSIX
    const fd = try posix.openat(posix.AT.FDCWD, token_map_path, .{}, 0);
    defer _ = std.c.close(fd);
    // Get file size via std.c.fstat
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return error.FileNotFound;
    const file_size: usize = @intCast(st.size);
    const content = try allocator.alloc(u8, file_size);
    defer allocator.free(content);
    var off: usize = 0;
    while (off < file_size) {
        const result = std.c.pread(fd, content[off..].ptr, content[off..].len, @intCast(off));
        const n: isize = @bitCast(result);
        if (n <= 0) break;
        off += @intCast(n);
    }

    var it = std.mem.tokenizeAny(u8, content[0..off], " \t\r\n,");
    var count: usize = 0;
    while (it.next()) |tok| {
        const id = std.fmt.parseInt(u32, std.mem.trim(u8, tok, " \t\r\n"), 10) catch continue;
        if (id < vocab_size) {
            mask[id] = true;
            count += 1;
        }
    }
    std.log.info("FR-Spec: loaded {d} tokens from {s} (vocab={d})", .{ count, token_map_path, vocab_size });
    return mask;
}

pub fn draftWithLogits(state: *SpecState, draft_model: *Model, last_token: u32) u32 {
    var tok = last_token;
    var n: u32 = 0;
    const vs = state.vocab_size;
    while (n < state.k and n < max_draft_tokens) {
        _ = draft_model.forward(tok) catch break;
        const logits = draft_model.getLogits();
        const offset = @as(usize, n) * vs;
        const dst = state.draft_log_probs[offset..][0..vs];
        @memcpy(dst, logits);
        // FR-Spec: restrict draft to high-frequency tokens only
        if (state.token_mask) |tm| applyFrSpecMask(dst, tm);
        logSoftmax(dst);
        tok = math_ops.argmax(dst);
        state.draft_tokens[n] = tok;
        state.depth_slices[n] = dst;
        n += 1;
    }
    state.n_draft = n;
    return n;
}

/// Standard greedy verification: verify single draft path sequentially.
pub fn verifySequential(
    state: *SpecState,
    target_model: *Model,
    draft_model: *Model,
    last_accepted_token: u32,
    pre_draft_pos: usize,
) SpecResult {
    if (state.n_draft == 0) return .{ .accepted = 0, .next_token = last_accepted_token };
    target_model.setKvSeqLen(pre_draft_pos);
    var accepted: u32 = 0;

    for (0..state.n_draft) |i| {
        const input = if (i == 0) last_accepted_token else state.draft_tokens[i - 1];
        const target_next = target_model.forward(input) catch |err| {
            std.log.warn("spec verify: target forward failed at draft {d}/{d}: {s}", .{ i, state.n_draft, @errorName(err) });
            break;
        };

        if (target_next == state.draft_tokens[i]) {
            accepted += 1;
        } else {
            return finishRound(state, target_model, draft_model, accepted, pre_draft_pos, target_next);
        }
    }

    // All accepted — bonus token from target
    const last_draft = state.draft_tokens[state.n_draft - 1];
    const bonus = target_model.forward(last_draft) catch |err| {
        std.log.warn("spec verify: bonus forward failed: {s}", .{@errorName(err)});
        return finishRound(state, target_model, draft_model, accepted, pre_draft_pos, last_draft);
    };
    return finishRound(state, target_model, draft_model, accepted, pre_draft_pos, bonus);
}

/// Rejection sampling verification (Leviathan et al. 2023).
/// Requires draftWithLogits() to have been called first.
pub fn verifySampling(
    state: *SpecState,
    target_model: *Model,
    draft_model: *Model,
    last_accepted_token: u32,
    pre_draft_pos: usize,
    temperature: f32,
    rng: std.Random,
) SpecResult {
    if (state.n_draft == 0) return .{ .accepted = 0, .next_token = last_accepted_token };
    target_model.setKvSeqLen(pre_draft_pos);
    const vs = state.vocab_size;
    var accepted: u32 = 0;

    for (0..state.n_draft) |i| {
        const input = if (i == 0) last_accepted_token else state.draft_tokens[i - 1];
        _ = target_model.forward(input) catch |err| {
            std.log.warn("spec sampling: target forward failed at draft {d}/{d}: {s}", .{ i, state.n_draft, @errorName(err) });
            break;
        };

        const target_logits = target_model.getLogits();
        const draft_lp = state.draft_log_probs[i * vs ..][0..vs];
        const tp = state.sampling_buf[0..vs];
        softmaxWithTemp(target_logits, tp, temperature);

        const draft_tok = state.draft_tokens[i];
        const q_tok = @exp(draft_lp[draft_tok]);
        const p_tok = tp[draft_tok];

        // Accept with probability min(1, p/q)
        if (q_tok > 0 and rng.float(f32) < @min(1.0, p_tok / q_tok)) {
            accepted += 1;
        } else {
            const correction = sampleResidual(tp, draft_lp, vs, rng, state.sampling_buf);
            return finishRound(state, target_model, draft_model, accepted, pre_draft_pos, correction);
        }
    }

    // All accepted — sample bonus from target distribution
    const last_draft = state.draft_tokens[state.n_draft - 1];
    _ = target_model.forward(last_draft) catch |err| {
        std.log.warn("spec sampling: bonus forward failed: {s}", .{@errorName(err)});
        return finishRound(state, target_model, draft_model, accepted, pre_draft_pos, last_draft);
    };
    const bonus = math_ops.sampleToken(target_model.getLogits(), temperature, 0, 1.0, rng);
    return finishRound(state, target_model, draft_model, accepted, pre_draft_pos, bonus);
}

/// DDTree verification: build tree, verify via batch forwardTree or sequential fallback.
pub fn verifyDDTree(
    state: *SpecState,
    target_model: *Model,
    draft_model: *Model,
    last_accepted_token: u32,
    budget: u32,
    pre_draft_pos: usize,
) SpecResult {
    if (state.n_draft == 0) return .{ .accepted = 0, .next_token = last_accepted_token };

    target_model.setKvSeqLen(pre_draft_pos);

    // Build DDTree
    var builder = ddtree.DDTreeBuilder{};
    builder.budget = @min(budget, ddtree.max_budget);
    builder.presort(state.depth_slices[0..state.n_draft]);
    builder.buildTree();

    if (builder.n_nodes == 0) return .{ .accepted = 0, .next_token = last_accepted_token };

    const tree = builder.compile(@intCast(pre_draft_pos));

    batch: {
        var aug_ids: [ddtree.max_budget + 1]u32 = undefined;
        var aug_pos: [ddtree.max_budget + 1]u32 = undefined;
        var aug_masks: [ddtree.max_budget + 1][8]u64 = undefined;
        const total: u32 = tree.n_nodes + 1;

        aug_ids[0] = last_accepted_token;
        aug_pos[0] = @intCast(pre_draft_pos);
        @memset(&aug_masks[0], 0);
        aug_masks[0][0] = 1;

        for (0..tree.n_nodes) |i| {
            aug_ids[i + 1] = tree.input_ids[i];
            aug_pos[i + 1] = tree.position_ids[i] + 1;
            @memset(&aug_masks[i + 1], 0);
            aug_masks[i + 1][0] = 1;
            for (0..tree.n_nodes) |j| {
                if (ddtree.CompiledTree.isAncestor(tree.ancestor_masks[i], j)) {
                    const shifted = j + 1;
                    aug_masks[i + 1][shifted / 64] |= @as(u64, 1) << @intCast(@as(u6, @truncate(shifted)));
                }
            }
        }

        target_model.forwardTree(aug_ids[0..total], aug_pos[0..total], &aug_masks, total) catch break :batch;

        var accepted: u32 = 0;
        var current_parent: i32 = -1;
        const first_target = target_model.treeLogits(0);

        if (tree.findChild(-1, first_target)) |first_child| {
            state.draft_tokens[0] = first_target;
            accepted = 1;
            current_parent = @intCast(first_child);
            var cur_child: u32 = first_child;

            while (accepted < builder.n_nodes) {
                const next = target_model.treeLogits(cur_child + 1);
                if (tree.findChild(current_parent, next)) |next_child| {
                    state.draft_tokens[accepted] = next;
                    accepted += 1;
                    current_parent = @intCast(next_child);
                    cur_child = next_child;
                } else {
                    return finishRound(state, target_model, draft_model, accepted, pre_draft_pos, next);
                }
            }
        } else {
            // Commit root token to KV cache (forwardTree didn't modify cache)
            target_model.setKvSeqLen(pre_draft_pos);
            _ = target_model.forward(last_accepted_token) catch |err| {
                std.log.warn("spec verify: target forward failed: {s}", .{@errorName(err)});
                return .{ .accepted = 0, .next_token = last_accepted_token };
            };
            return finishRound(state, target_model, draft_model, 0, pre_draft_pos, first_target);
        }

        // Commit accepted tokens to KV cache (forwardTree didn't modify cache)
        target_model.setKvSeqLen(pre_draft_pos);
        var commit_tok = last_accepted_token;
        for (0..accepted) |i| {
            _ = target_model.forward(commit_tok) catch |err| {
                std.log.warn("spec commit: target forward failed at token {d}/{d}: {s}", .{ i, accepted, @errorName(err) });
                return .{ .accepted = @intCast(i), .next_token = commit_tok };
            };
            commit_tok = state.draft_tokens[i];
        }
        const bonus = target_model.forward(commit_tok) catch |err| {
            std.log.warn("spec commit: bonus forward failed: {s}", .{@errorName(err)});
            return .{ .accepted = accepted, .next_token = commit_tok };
        };
        return finishRound(state, target_model, draft_model, accepted, pre_draft_pos, bonus);
    }
    return verifyDDTreeSequential(state, target_model, draft_model, last_accepted_token, &builder, &tree, pre_draft_pos);
}

fn verifyDDTreeSequential(
    state: *SpecState,
    target_model: *Model,
    draft_model: *Model,
    last_accepted_token: u32,
    builder: *const ddtree.DDTreeBuilder,
    tree: *const ddtree.CompiledTree,
    pre_draft_pos: usize,
) SpecResult {
    var accepted: u32 = 0;
    var current_parent: i32 = -1;
    var input_tok = last_accepted_token;

    while (true) {
        const target_next = target_model.forward(input_tok) catch break;

        if (tree.findChild(current_parent, target_next)) |child_idx| {
            state.draft_tokens[accepted] = target_next;
            accepted += 1;
            current_parent = @intCast(child_idx);
            input_tok = target_next;
        } else {
            return finishRound(state, target_model, draft_model, accepted, pre_draft_pos, target_next);
        }

        if (accepted >= builder.n_nodes) break;
    }

    const bonus = target_model.forward(input_tok) catch input_tok;
    return finishRound(state, target_model, draft_model, accepted, pre_draft_pos, bonus);
}

/// Shared exit path: record stats, sync draft KV cache, return result.
fn finishRound(state: *SpecState, target_model: *Model, draft_model: *Model, accepted: u32, pre_draft_pos: usize, next_token: u32) SpecResult {
    state.recordRound(accepted);
    if (target_model.ptr != draft_model.ptr)
        draft_model.setKvSeqLen(pre_draft_pos + accepted);
    return .{ .accepted = accepted, .next_token = next_token };
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn softmaxWithTemp(logits: []const f32, out: []f32, temperature: f32) void {
    if (logits.len == 0) return;
    const V8 = @Vector(8, f32);
    const n = logits.len;
    const safe_temp = if (temperature > 0) temperature else 1.0;
    const inv_t = 1.0 / safe_temp;

    const inv_tv: V8 = @splat(inv_t);
    var max_v: V8 = @splat(-std.math.inf(f32));
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        max_v = @max(max_v, @as(V8, logits[i..][0..8].*) * inv_tv);
    }
    var max_val = @reduce(.Max, max_v);
    while (i < n) : (i += 1) max_val = @max(max_val, logits[i] * inv_t);

    const mv: V8 = @splat(max_val);
    var sum_v: V8 = @splat(@as(f32, 0.0));
    i = 0;
    while (i + 8 <= n) : (i += 8) {
        const exp_v = @exp(@as(V8, logits[i..][0..8].*) * inv_tv - mv);
        out[i..][0..8].* = exp_v;
        sum_v += exp_v;
    }
    var sum = @reduce(.Add, sum_v);
    while (i < n) : (i += 1) {
        out[i] = @exp(logits[i] * inv_t - max_val);
        sum += out[i];
    }

    if (sum > 0) {
        const inv_sum = 1.0 / sum;
        const isv: V8 = @splat(inv_sum);
        i = 0;
        while (i + 8 <= n) : (i += 8) {
            out[i..][0..8].* = @as(V8, out[i..][0..8].*) * isv;
        }
        while (i < n) : (i += 1) out[i] *= inv_sum;
    }
}

/// Sample from norm(max(0, p_target - p_draft)).
fn sampleResidual(target_probs: []const f32, draft_log_probs: []const f32, vs: u32, rng: std.Random, buf: []f32) u32 {
    const V8 = @Vector(8, f32);
    const zero: V8 = @splat(@as(f32, 0.0));
    const n: usize = vs;
    var sum_v: V8 = zero;
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const p: V8 = target_probs[i..][0..8].*;
        const q = @exp(@as(V8, draft_log_probs[i..][0..8].*));
        const diff = @max(zero, p - q);
        buf[i..][0..8].* = diff;
        sum_v += diff;
    }
    var sum = @reduce(.Add, sum_v);
    while (i < n) : (i += 1) {
        const diff = @max(0.0, target_probs[i] - @exp(draft_log_probs[i]));
        buf[i] = diff;
        sum += diff;
    }
    if (sum <= 0) return 0;
    var r = rng.float(f32) * sum;
    for (0..n) |j| {
        r -= buf[j];
        if (r <= 0) return @intCast(j);
    }
    return vs - 1;
}

/// Log-softmax: v_i = v_i - max - log(sum(exp(v - max))).
/// SIMD-optimized (8-wide) — called per draft token on vocab-sized arrays.
fn logSoftmax(logits: []f32) void {
    if (logits.len == 0) return;
    const V8 = @Vector(8, f32);
    const n = logits.len;

    var max_v: V8 = @splat(-std.math.inf(f32));
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        max_v = @max(max_v, @as(V8, logits[i..][0..8].*));
    }
    var max_val = @reduce(.Max, max_v);
    while (i < n) : (i += 1) max_val = @max(max_val, logits[i]);

    const mv: V8 = @splat(max_val);
    var sum_v: V8 = @splat(@as(f32, 0.0));
    i = 0;
    while (i + 8 <= n) : (i += 8) {
        sum_v += @exp(@as(V8, logits[i..][0..8].*) - mv);
    }
    var sum_exp = @reduce(.Add, sum_v);
    while (i < n) : (i += 1) sum_exp += @exp(logits[i] - max_val);

    const offset = max_val + @log(sum_exp + log_softmax_eps);
    const ov: V8 = @splat(offset);
    i = 0;
    while (i + 8 <= n) : (i += 8) {
        logits[i..][0..8].* = @as(V8, logits[i..][0..8].*) - ov;
    }
    while (i < n) : (i += 1) logits[i] -= offset;
}

// ── Tests ────────────────────────────────────────────────────────────────────

test "SpecState init and stats" {
    var s = try SpecState.init(std.testing.allocator, 5, 100);
    defer s.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 5), s.k);
    try std.testing.expectEqual(@as(f32, 0), s.acceptanceRate());

    s.total_accepted = 8;
    s.total_drafted = 10;
    s.total_rounds = 2;
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), s.acceptanceRate(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), s.meanAccepted(), 0.01);
}

test "SpecState recordRound updates stats" {
    var s = try SpecState.init(std.testing.allocator, 5, 100);
    defer s.deinit(std.testing.allocator);

    s.n_draft = 5;
    s.recordRound(4);
    try std.testing.expectEqual(@as(u64, 4), s.total_accepted);
    try std.testing.expectEqual(@as(u64, 5), s.total_drafted);
    try std.testing.expectEqual(@as(u64, 1), s.total_rounds);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), s.acceptanceRate(), 0.01);

    s.n_draft = 3;
    s.recordRound(3);
    try std.testing.expectEqual(@as(u64, 7), s.total_accepted);
    try std.testing.expectEqual(@as(u64, 8), s.total_drafted);
    try std.testing.expectEqual(@as(u64, 2), s.total_rounds);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), s.meanAccepted(), 0.01);
}

test "optimalK returns configured k without sufficient data" {
    var s = try SpecState.init(std.testing.allocator, 5, 100);
    defer s.deinit(std.testing.allocator);

    // Without adaptive enabled, always returns configured k
    try std.testing.expectEqual(@as(u32, 5), s.optimalK());

    // With adaptive enabled but insufficient rounds, still returns configured k
    s.adaptive_k_enabled = true;
    s.total_rounds = 3;
    try std.testing.expectEqual(@as(u32, 5), s.optimalK());

    // After enough rounds with per-K profiling data, picks best k
    s.total_rounds = adaptive_k_min_rounds;
    // k=3 (index 2): 90% accept rate → EV = 3*0.9 + 1 = 3.7
    s.k_total_counts[2] = adaptive_k_min_samples;
    s.k_accept_counts[2] = adaptive_k_min_samples * 3 * 9 / 10;
    // k=5 (index 4): 40% accept rate → EV = 5*0.4 + 1 = 3.0
    s.k_total_counts[4] = adaptive_k_min_samples;
    s.k_accept_counts[4] = adaptive_k_min_samples * 5 * 4 / 10;
    const optimal = s.optimalK();
    // k=3 has highest EV (3.7 vs 3.0 for k=5), so optimalK must pick 3
    try std.testing.expectEqual(@as(u32, 3), optimal);
}

test "logSoftmax produces valid log-probabilities" {
    var logits = [_]f32{ 1.0, 2.0, 3.0 };
    logSoftmax(&logits);

    // All values should be <= 0 (log of probability)
    for (logits) |v| try std.testing.expect(v <= 0);

    // exp(log_probs) should sum to ~1.0
    var sum: f32 = 0;
    for (logits) |v| sum += @exp(v);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);

    // Largest input should have largest log-prob
    try std.testing.expect(logits[2] > logits[1]);
    try std.testing.expect(logits[1] > logits[0]);

    // log_softmax([1,2,3]) = x - log(e^1 + e^2 + e^3), verify against known values
    const log_sum_exp = @log(@as(f32, @exp(1.0) + @exp(2.0) + @exp(3.0)));
    try std.testing.expectApproxEqAbs(1.0 - log_sum_exp, logits[0], 1e-5);
    try std.testing.expectApproxEqAbs(2.0 - log_sum_exp, logits[1], 1e-5);
    try std.testing.expectApproxEqAbs(3.0 - log_sum_exp, logits[2], 1e-5);
}

test "softmaxWithTemp concentrates on max at low temperature" {
    const logits = [_]f32{ 1.0, 2.0, 5.0 };
    var out: [3]f32 = undefined;

    softmaxWithTemp(&logits, &out, 0.1);

    // At very low temperature, nearly all mass on the maximum (index 2)
    try std.testing.expect(out[2] > 0.99);
    // Max logit must produce highest probability; others must be ordered
    try std.testing.expect(out[2] > out[1]);
    try std.testing.expect(out[1] > out[0]);

    // Sum should be 1.0
    var sum: f32 = 0;
    for (out) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "SpecState recordRound with adaptive K profiling" {
    var s = try SpecState.init(std.testing.allocator, 5, 100);
    defer s.deinit(std.testing.allocator);

    s.adaptive_k_enabled = true;

    // Record rounds with different draft lengths
    s.n_draft = 3;
    s.recordRound(2);
    try std.testing.expectEqual(@as(u32, 1), s.k_total_counts[2]); // index 2 = k=3
    try std.testing.expectEqual(@as(u32, 2), s.k_accept_counts[2]);

    s.n_draft = 5;
    s.recordRound(4);
    try std.testing.expectEqual(@as(u32, 1), s.k_total_counts[4]); // index 4 = k=5
    try std.testing.expectEqual(@as(u32, 4), s.k_accept_counts[4]);

    // Cumulative stats should be correct
    try std.testing.expectEqual(@as(u64, 6), s.total_accepted);
    try std.testing.expectEqual(@as(u64, 8), s.total_drafted);
    try std.testing.expectEqual(@as(u64, 2), s.total_rounds);
}

test "SpecState zero draft round" {
    var s = try SpecState.init(std.testing.allocator, 5, 100);
    defer s.deinit(std.testing.allocator);

    s.n_draft = 0;
    s.recordRound(0);
    try std.testing.expectEqual(@as(u64, 0), s.total_drafted);
    try std.testing.expectEqual(@as(u64, 1), s.total_rounds);
    try std.testing.expectEqual(@as(f32, 0), s.acceptanceRate());
    try std.testing.expectEqual(@as(f32, 0), s.meanAccepted());
}

test "logSoftmax single element" {
    var logits = [_]f32{5.0};
    logSoftmax(&logits);
    // Single element: log(1.0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), logits[0], 1e-4);
}

test "logSoftmax large input" {
    // Test with larger input to exercise SIMD path (>8 elements)
    var logits: [16]f32 = undefined;
    for (0..16) |i| logits[i] = @floatFromInt(i);

    logSoftmax(&logits);

    // All should be <= 0
    for (logits) |v| try std.testing.expect(v <= 0);

    // exp(log_probs) should sum to ~1.0
    var sum: f32 = 0;
    for (logits) |v| sum += @exp(v);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-4);

    // Last element should have the largest log-prob
    try std.testing.expect(logits[15] > logits[14]);
    try std.testing.expect(logits[14] > logits[0]);
}

test "logSoftmax empty is no-op" {
    var empty: [0]f32 = .{};
    logSoftmax(&empty); // Should not crash
}

test "softmaxWithTemp temperature=1 is standard softmax" {
    const logits = [_]f32{ 1.0, 2.0, 3.0 };
    var out: [3]f32 = undefined;

    softmaxWithTemp(&logits, &out, 1.0);

    // Standard softmax
    var sum: f32 = 0;
    for (out) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);

    // p(3) > p(2) > p(1)
    try std.testing.expect(out[2] > out[1]);
    try std.testing.expect(out[1] > out[0]);
}

test "softmaxWithTemp zero temperature uses temp=1 fallback" {
    const logits = [_]f32{ 1.0, 2.0, 3.0 };
    var out: [3]f32 = undefined;

    // Zero temperature should fallback to 1.0 (safe_temp guard)
    softmaxWithTemp(&logits, &out, 0.0);

    var sum: f32 = 0;
    for (out) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "softmaxWithTemp SIMD path with >8 elements" {
    var logits: [16]f32 = undefined;
    var out: [16]f32 = undefined;
    for (0..16) |i| logits[i] = @as(f32, @floatFromInt(i)) * 0.5;

    softmaxWithTemp(&logits, &out, 0.5);

    var sum: f32 = 0;
    for (out) |v| {
        try std.testing.expect(v >= 0);
        sum += v;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-4);
}

test "sampleResidual all mass on draft returns 0 fallback" {
    // When target <= draft everywhere, residual is all zeros
    const target = [_]f32{ 0.1, 0.3, 0.6 };
    const draft_lp = [_]f32{ @log(@as(f32, 0.2)), @log(@as(f32, 0.4)), @log(@as(f32, 0.7)) };
    var buf: [3]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(42);
    const tok = sampleResidual(&target, &draft_lp, 3, prng.random(), &buf);
    // Sum of residual is 0, function returns 0
    try std.testing.expectEqual(@as(u32, 0), tok);
}

test "sampleResidual returns valid token" {
    const target = [_]f32{ 0.1, 0.3, 0.6 };
    const draft_lp = [_]f32{ @log(@as(f32, 0.5)), @log(@as(f32, 0.3)), @log(@as(f32, 0.2)) };
    var buf: [3]f32 = undefined;
    // Residual mass: target - draft = {-0.4, 0, 0.4} → clamp negatives → {0, 0, 0.4} → normalized
    // Token 2 should be strongly favored (all residual mass)
    var counts = [_]u32{ 0, 0, 0 };
    for (0..100) |seed| {
        var prng = std.Random.DefaultPrng.init(seed);
        const tok = sampleResidual(&target, &draft_lp, 3, prng.random(), &buf);
        try std.testing.expect(tok < 3);
        counts[tok] += 1;
    }
    // Token 2 must dominate (has all positive residual mass)
    try std.testing.expect(counts[2] > 50);
}

test "fuzz: all spec_decode functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const vocab_size: u32 = 16;

            // -- SpecState.init / deinit --
            const k_raw: u32 = smith.valueWithHash(u8, 0);
            const k: u32 = k_raw % @as(u32, max_draft_tokens) + 1;
            var s = try SpecState.init(std.testing.allocator, k, vocab_size);
            defer s.deinit(std.testing.allocator);

            // -- SpecState.acceptanceRate (zero state) --
            const rate0 = s.acceptanceRate();
            try std.testing.expect(rate0 == 0);

            // -- SpecState.meanAccepted (zero state) --
            const mean0 = s.meanAccepted();
            try std.testing.expect(mean0 == 0);

            // -- SpecState.recordRound --
            const n_draft_raw: u32 = smith.valueWithHash(u8, 1);
            s.n_draft = n_draft_raw % @as(u32, max_draft_tokens);
            s.adaptive_k_enabled = (smith.valueWithHash(u8, 2) & 1) != 0;
            const accepted_val: u32 = @as(u32, smith.valueWithHash(u8, 3)) % (s.n_draft + 1);
            s.recordRound(accepted_val);
            try std.testing.expect(s.total_rounds == 1);
            try std.testing.expect(s.total_accepted == accepted_val);

            // -- SpecState.acceptanceRate (after round) --
            const rate1 = s.acceptanceRate();
            try std.testing.expect(rate1 >= 0 and rate1 <= 1.0);

            // -- SpecState.meanAccepted (after round) --
            const mean1 = s.meanAccepted();
            try std.testing.expect(mean1 >= 0);

            // -- SpecState.optimalK --
            // Populate enough rounds for adaptive path
            s.total_rounds = 20;
            for (0..@min(@as(usize, k), max_draft_tokens)) |ki| {
                s.k_total_counts[ki] = smith.valueWithHash(u8, @intCast(ki + 10)) / 2 + 1;
                s.k_accept_counts[ki] = smith.valueWithHash(u8, @intCast(ki + 42)) % (s.k_total_counts[ki] * (@as(u32, @intCast(ki)) + 1) + 1);
            }
            const ok = s.optimalK();
            try std.testing.expect(ok >= 1 and ok <= @as(u32, max_draft_tokens));

            // -- logSoftmax (private but accessible in-file tests) --
            var logits: [vocab_size]f32 = undefined;
            for (0..vocab_size) |i| {
                const raw = smith.valueWithHash(i16, @intCast(i + 100));
                logits[i] = @as(f32, @floatFromInt(raw)) * 0.01;
            }
            logSoftmax(&logits);
            var sum_exp: f32 = 0;
            for (logits) |v| {
                try std.testing.expect(v <= 0.001); // log-probs are <= 0 (with eps tolerance)
                sum_exp += @exp(v);
            }
            try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum_exp, 1e-3);

            // -- softmaxWithTemp --
            var inp: [vocab_size]f32 = undefined;
            for (0..vocab_size) |i| {
                const raw = smith.valueWithHash(i16, @intCast(i + 200));
                inp[i] = @as(f32, @floatFromInt(raw)) * 0.01;
            }
            var out: [vocab_size]f32 = undefined;
            const temp_raw = smith.valueWithHash(u8, 50);
            const temp: f32 = @as(f32, @floatFromInt(temp_raw)) * 0.02 + 0.01;
            softmaxWithTemp(&inp, &out, temp);
            var prob_sum: f32 = 0;
            for (out) |v| {
                try std.testing.expect(v >= 0);
                prob_sum += v;
            }
            try std.testing.expectApproxEqAbs(@as(f32, 1.0), prob_sum, 1e-3);

            // -- sampleResidual --
            var target_probs: [vocab_size]f32 = undefined;
            var draft_lp: [vocab_size]f32 = undefined;
            var sample_buf: [vocab_size]f32 = undefined;
            softmaxWithTemp(&inp, &target_probs, 1.0);
            for (0..vocab_size) |i| {
                const raw = smith.valueWithHash(i16, @intCast(i + 300));
                draft_lp[i] = @as(f32, @floatFromInt(raw)) * 0.01;
            }
            logSoftmax(&draft_lp);
            var prng = std.Random.DefaultPrng.init(smith.valueWithHash(u64, 99));
            const sampled = sampleResidual(&target_probs, &draft_lp, vocab_size, prng.random(), &sample_buf);
            try std.testing.expect(sampled < vocab_size);

            // -- max_draft_tokens constant --
            try std.testing.expect(max_draft_tokens > 0);

            // -- SpecResult type --
            const sr = SpecResult{ .accepted = accepted_val, .next_token = 0 };
            try std.testing.expect(sr.accepted <= max_draft_tokens);

            // -- Model-dependent pub fns: comptime-verify reachability --
            // draftMtp, draft, draftWithLogits, verifySequential,
            // verifySampling, verifyDDTree all require *Model (vtable runtime).
            // Verify they exist and are callable at comptime.
            comptime {
                _ = &draftMtp;
                _ = &draft;
                _ = &draftWithLogits;
                _ = &verifySequential;
                _ = &verifySampling;
                _ = &verifyDDTree;
            }
        }
    }.f, .{});
}
