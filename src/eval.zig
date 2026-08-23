//! Token-by-token negative log-likelihood (NLL) evaluation.
//!
//! Given a model and a set of official continuations (prompt + expected tokens),
//! measures how much probability the model assigns to each ground-truth token.
//! This provides a quantitative quality metric without requiring a full benchmark
//! suite — the score directly reflects how closely a local GGUF matches the
//! reference model that generated the continuations.
//!
//! Metric: mean negative log probability across all continuation tokens.
//! Lower = better (model assigns higher probability to the correct tokens).
//!
//! Library API (no `--eval` CLI yet). Score one prompt + continuation:
//!   const result = eval.scoreCase(model, prompt_ids, continuation_ids) orelse return error.EvalFailed;
//!
//! Collect reference continuations with `tools/quality-testing/collect_continuations.py`
//! (JSONL: prompt plus `continuation` text and/or pre-tokenized `tokens`).
//!
//! Based on the quality testing approach from antirez/ds4.

const std = @import("std");
const Allocator = std.mem.Allocator;
const math_ops = @import("ops/math.zig");

/// Result of evaluating one continuation.
pub const CaseResult = struct {
    /// Mean negative log probability across continuation tokens.
    mean_nll: f32,
    /// Number of continuation tokens scored.
    n_tokens: u32,
    /// Number of tokens where the model's argmax matched the ground truth.
    n_correct_argmax: u32,
    /// Total negative log probability (sum, not mean).
    total_nll: f64,
};

/// Aggregate result across all continuations.
pub const EvalResult = struct {
    /// Per-case results.
    cases: []CaseResult,
    /// Mean NLL across all cases (macro-average: each case weighted equally).
    mean_nll: f32,
    /// Total continuation tokens scored.
    total_tokens: u32,
    /// Total tokens where argmax matched ground truth.
    total_correct: u32,
    /// Argmax accuracy (total_correct / total_tokens).
    accuracy: f32,
    /// Number of cases that failed (forward error, missing tokens, etc.).
    n_failed: u32,

    /// Prints a summary line with case count, token count, mean NLL, argmax accuracy, and failure count.
    pub fn print(self: *const EvalResult) void {
        std.log.info("eval: {d} cases, {d} tokens, mean_nll={d:.4}, argmax_acc={d:.2}%, failed={d}", .{
            self.cases.len,
            self.total_tokens,
            self.mean_nll,
            self.accuracy * 100.0,
            self.n_failed,
        });
    }

    /// Frees the per-case result slice.
    pub fn deinit(self: *EvalResult, allocator: Allocator) void {
        allocator.free(self.cases);
    }
};

/// Score a single continuation: run the model forward on prompt + continuation
/// tokens, collecting the log probability assigned to each ground-truth token.
///
/// Returns null if the forward pass fails.
pub fn scoreCase(
    model: anytype,
    prompt_ids: []const u32,
    continuation_ids: []const u32,
) ?CaseResult {
    if (continuation_ids.len == 0) return CaseResult{
        .mean_nll = 0,
        .n_tokens = 0,
        .n_correct_argmax = 0,
        .total_nll = 0,
    };

    // Reset KV cache for a clean evaluation
    model.resetCache();

    // Prefill prompt
    if (prompt_ids.len > 0) {
        _ = model.prefill(prompt_ids) catch return null;
    }

    var total_nll: f64 = 0;
    var n_correct: u32 = 0;
    var n_scored: u32 = 0;

    // Score each continuation token.
    // After prefill, logits already predict the first continuation token
    // (position prompt_len). For subsequent tokens, forward the previous
    // continuation token to advance the model and get fresh logits.
    for (continuation_ids, 0..) |gt_token, ci| {
        if (ci > 0 or prompt_ids.len == 0) {
            // Forward the previous token to get logits for this position.
            // Skip for ci==0 when prompt was prefilled — logits are already set.
            const prev = if (ci > 0) continuation_ids[ci - 1] else 0;
            _ = model.forward(prev) catch return null;
        }

        const logits = model.getLogits();
        if (gt_token >= logits.len) {
            // OOV token — skip without counting toward scored total,
            // to avoid diluting the mean NLL.
            continue;
        }

        // Log-softmax probability of the ground-truth token and greedy argmax,
        // both via the SIMD helpers shared with the sampler (single fused pass
        // over the vocab instead of scalar max/sum-exp/argmax rescans).
        const log_prob = math_ops.tokenLogProb(logits, gt_token);
        total_nll -= @as(f64, log_prob);

        if (math_ops.argmax(logits) == gt_token) n_correct += 1;

        n_scored += 1;
    }

    if (n_scored == 0) return CaseResult{
        .mean_nll = 0,
        .n_tokens = 0,
        .n_correct_argmax = 0,
        .total_nll = 0,
    };

    return CaseResult{
        .mean_nll = @floatCast(total_nll / @as(f64, @floatFromInt(n_scored))),
        .n_tokens = n_scored,
        .n_correct_argmax = n_correct,
        .total_nll = total_nll,
    };
}

// ── Tests ────────────────────────────────────────────────────────

test "CaseResult zero tokens" {
    const r = CaseResult{
        .mean_nll = 0,
        .n_tokens = 0,
        .n_correct_argmax = 0,
        .total_nll = 0,
    };
    try std.testing.expectEqual(@as(u32, 0), r.n_tokens);
}

test "scoreCase empty continuation skips model" {
    // Early return before any model method; stubs exist only for anytype typecheck.
    const Dummy = struct {
        fn resetCache(_: @This()) void {}
        fn prefill(_: @This(), _: []const u32) error{Unused}!void {
            return error.Unused;
        }
        fn forward(_: @This(), _: u32) error{Unused}!void {
            return error.Unused;
        }
        fn getLogits(_: @This()) []const f32 {
            return &.{};
        }
    };
    const r = scoreCase(Dummy{}, &.{ 1, 2, 3 }, &.{});
    try std.testing.expect(r != null);
    try std.testing.expectEqual(@as(u32, 0), r.?.n_tokens);
    try std.testing.expectEqual(@as(u32, 0), r.?.n_correct_argmax);
    try std.testing.expectEqual(@as(f32, 0), r.?.mean_nll);
    try std.testing.expectEqual(@as(f64, 0), r.?.total_nll);
}

test "EvalResult print does not crash" {
    var cases = [_]CaseResult{.{
        .mean_nll = 1.5,
        .n_tokens = 10,
        .n_correct_argmax = 7,
        .total_nll = 15.0,
    }};
    const result = EvalResult{
        .cases = &cases,
        .mean_nll = 1.5,
        .total_tokens = 10,
        .total_correct = 7,
        .accuracy = 0.7,
        .n_failed = 0,
    };
    try std.testing.expectEqual(@as(usize, 1), result.cases.len);
    try std.testing.expectEqual(@as(u32, 10), result.total_tokens);
    try std.testing.expectEqual(@as(u32, 7), result.total_correct);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), result.accuracy, 1e-6);
    try std.testing.expectEqual(@as(u32, 0), result.n_failed);
    result.print();
}

test "scoreCase scores continuation tokens correctly" {
    // Dummy model that returns fixed logits making token 2 the argmax.
    const Model = struct {
        cache_reset: bool = false,
        forward_count: u32 = 0,

        fn resetCache(self: *@This()) void {
            self.cache_reset = true;
        }
        fn prefill(self: *@This(), _: []const u32) !void {
            _ = self;
        }
        fn forward(self: *@This(), _: u32) !void {
            self.forward_count += 1;
        }
        fn getLogits(_: *@This()) []const f32 {
            // 4-token vocab; token 2 has the highest logit
            return &[_]f32{ 0.0, 1.0, 5.0, 0.5 };
        }
    };
    var model = Model{};

    // Continuation is [2] — model's argmax matches, so n_correct_argmax = 1.
    const r = scoreCase(&model, &.{10}, &.{2});
    try std.testing.expect(r != null);
    try std.testing.expectEqual(@as(u32, 1), r.?.n_tokens);
    try std.testing.expectEqual(@as(u32, 1), r.?.n_correct_argmax);
    // NLL should be finite and positive (- log_prob of correct token)
    try std.testing.expect(r.?.mean_nll > 0);
    try std.testing.expect(std.math.isFinite(r.?.mean_nll));
    // Cache must have been reset
    try std.testing.expect(model.cache_reset);
}

test "scoreCase returns null on forward failure" {
    const Model = struct {
        fn resetCache(_: *@This()) void {}
        fn prefill(_: *@This(), _: []const u32) !void {}
        fn forward(_: *@This(), _: u32) !void {
            return error.ForwardFailed;
        }
        fn getLogits(_: *@This()) []const f32 {
            return &[_]f32{ 0.0, 1.0 };
        }
    };
    var model = Model{};
    // ci=0 with prompt: logits already set, forward not called.
    // ci=1: forward is called and fails → returns null.
    const r = scoreCase(&model, &.{1}, &.{ 0, 1 });
    try std.testing.expect(r == null);
}
