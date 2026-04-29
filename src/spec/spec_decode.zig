//! Speculative decoding orchestrator.
//!
//! Supports three verification modes:
//! - Standard greedy: single-path draft → sequential argmax verification
//! - Rejection sampling: stochastic acceptance with temperature (Leviathan et al. 2023)
//! - DDTree: tree-structured draft → greedy tree walk (Ringel & Romano, 2026)

const std = @import("std");
const Model = @import("../models/model.zig").Model;
const math_ops = @import("../ops/math.zig");
const ddtree = @import("ddtree.zig");

pub const max_draft_tokens: usize = 32;
const log_softmax_eps: f32 = 1e-10;

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

    pub fn init(allocator: std.mem.Allocator, k: u32, vocab_size: u32) !SpecState {
        return .{
            .k = k,
            .vocab_size = vocab_size,
            .draft_log_probs = try allocator.alloc(f32, max_draft_tokens * vocab_size),
            .sampling_buf = try allocator.alloc(f32, vocab_size),
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
    }
};

pub const SpecResult = struct {
    accepted: u32,
    next_token: u32,
};

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
pub fn draftWithLogits(state: *SpecState, draft_model: *Model, last_token: u32) u32 {
    var tok = last_token;
    var n: u32 = 0;
    const vs = state.vocab_size;
    while (n < state.k and n < max_draft_tokens) {
        tok = draft_model.forward(tok) catch break;
        const logits = draft_model.getLogits();
        const offset = @as(usize, n) * vs;
        const dst = state.draft_log_probs[offset..][0..vs];
        @memcpy(dst, logits);
        logSoftmax(dst);
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
        const target_next = target_model.forward(input) catch break;

        if (target_next == state.draft_tokens[i]) {
            accepted += 1;
        } else {
            return finishRound(state, target_model, draft_model, accepted, pre_draft_pos, target_next);
        }
    }

    // All accepted — bonus token from target
    const last_draft = state.draft_tokens[state.n_draft - 1];
    const bonus = target_model.forward(last_draft) catch last_draft;
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
        _ = target_model.forward(input) catch break;

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
    _ = target_model.forward(last_draft) catch {
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
            _ = target_model.forward(last_accepted_token) catch {};
            return finishRound(state, target_model, draft_model, 0, pre_draft_pos, first_target);
        }

        // Commit accepted tokens to KV cache (forwardTree didn't modify cache)
        target_model.setKvSeqLen(pre_draft_pos);
        var commit_tok = last_accepted_token;
        for (0..accepted) |i| {
            _ = target_model.forward(commit_tok) catch break;
            commit_tok = state.draft_tokens[i];
        }
        const bonus = target_model.forward(commit_tok) catch commit_tok;
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
    const inv_t = 1.0 / temperature;
    var max_val: f32 = logits[0] * inv_t;
    for (logits[1..]) |v| {
        const scaled = v * inv_t;
        if (scaled > max_val) max_val = scaled;
    }
    var sum: f32 = 0;
    for (logits, 0..) |v, i| {
        out[i] = @exp(v * inv_t - max_val);
        sum += out[i];
    }
    const inv_sum = 1.0 / sum;
    for (out[0..logits.len]) |*v| v.* *= inv_sum;
}

/// Sample from norm(max(0, p_target - p_draft)).
fn sampleResidual(target_probs: []const f32, draft_log_probs: []const f32, vs: u32, rng: std.Random, buf: []f32) u32 {
    var sum: f32 = 0;
    for (0..vs) |i| {
        const p = target_probs[i];
        const q = @exp(draft_log_probs[i]);
        buf[i] = @max(0.0, p - q);
        sum += buf[i];
    }
    if (sum <= 0) return 0;
    var r = rng.float(f32) * sum;
    for (0..vs) |i| {
        r -= buf[i];
        if (r <= 0) return @intCast(i);
    }
    return vs - 1;
}

/// Two-pass log-softmax: v_i = v_i - max - log(sum(exp(v - max))).
fn logSoftmax(logits: []f32) void {
    var max_val: f32 = logits[0];
    for (logits[1..]) |v| if (v > max_val) {
        max_val = v;
    };
    var sum_exp: f32 = 0;
    for (logits) |*v| {
        v.* -= max_val;
        const e = @exp(v.*);
        sum_exp += e;
    }
    const log_z = @log(sum_exp + log_softmax_eps);
    for (logits) |*v| v.* -= log_z;
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
