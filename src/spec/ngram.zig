//! N-gram speculative decoding: propose continuation tokens from output history.
//!
//! Searches the generated token history for n-gram matches of the most recent
//! tokens. When a match is found, the tokens that followed that match in history
//! are proposed as draft tokens for tree verification.
//!
//! Zero overhead — no draft model, no extra forward passes for drafting.
//! Works best for repetitive text: code, lists, structured output, templates.
//!
//! Inspired by vLLM's n-gram speculative decoding (v0.10.1).

const std = @import("std");

const history_capacity: usize = 2048;
const min_ngram: usize = 3;
const max_ngram: usize = 10;

/// N-gram proposal state. Maintains a ring buffer of generated tokens.
pub const NgramState = struct {
    history: [history_capacity]u32 = undefined,
    len: usize = 0,

    /// Record a generated token.
    pub fn push(self: *NgramState, token: u32) void {
        if (self.len < history_capacity) {
            self.history[self.len] = token;
            self.len += 1;
        } else {
            // Shift left by half to make room (amortized)
            const keep = history_capacity / 2;
            std.mem.copyForwards(u32, self.history[0..keep], self.history[history_capacity - keep ..]);
            self.len = keep;
            self.history[self.len] = token;
            self.len += 1;
        }
    }

    /// Propose up to `max_draft` continuation tokens based on n-gram matching.
    /// Returns the number of proposed tokens written to `out`.
    pub fn propose(self: *const NgramState, max_draft: usize, out: []u32) usize {
        if (self.len < min_ngram + 1 or max_draft == 0) return 0;

        const hist = self.history[0..self.len];
        var best_match_pos: usize = 0;
        var best_match_len: usize = 0;

        // Try longest n-gram first (greedy — longer match = better prediction)
        const max_n = @min(max_ngram, self.len - 1);
        var n: usize = max_n;
        while (n >= min_ngram) : (n -= 1) {
            // Pattern = last n tokens
            const pattern = hist[self.len - n ..];

            // Search for this pattern earlier in history
            if (self.len < n + 1) continue;
            const search_end = self.len - n;

            var pos: usize = 0;
            while (pos + n <= search_end) : (pos += 1) {
                if (std.mem.eql(u32, hist[pos .. pos + n], pattern[0..n])) {
                    best_match_pos = pos + n;
                    best_match_len = n;
                    break;
                }
            }
            if (best_match_len > 0) break;
        }

        if (best_match_len == 0) return 0;

        // Copy continuation tokens after the match
        const avail = self.len - best_match_pos;
        const n_propose = @min(@min(avail, max_draft), out.len);
        @memcpy(out[0..n_propose], hist[best_match_pos..][0..n_propose]);
        return n_propose;
    }
};

/// Shared n-gram pool for server mode: all concurrent requests contribute to
/// and draw from a single global token history.  When a slot generates a token
/// it calls SharedNgramPool.push(); when drafting it searches both its own
/// NgramState AND the shared pool, taking whichever gives a longer match.
///
/// Inspired by llama.cpp ngram-mod (PR #19164): a shared pool means concurrent
/// requests on similar content benefit from each other's history for free.
///
/// Thread-safety: guarded by a plain Mutex; critical sections are short
/// (ring-buffer push or linear scan), so contention is negligible.
pub const SharedNgramPool = struct {
    const pool_capacity: usize = 8192; // ~32 KB — larger than per-request 2 KB

    history: [pool_capacity]u32 = undefined,
    len: usize = 0,
    mu: std.atomic.Mutex = .unlocked,

    fn lock(self: *SharedNgramPool) void {
        while (!self.mu.tryLock()) std.atomic.spinLoopHint();
    }
    fn unlock(self: *SharedNgramPool) void {
        self.mu.unlock();
    }

    /// Record a generated token into the shared pool (called by every server slot).
    pub fn push(self: *SharedNgramPool, token: u32) void {
        self.lock();
        defer self.unlock();
        if (self.len < pool_capacity) {
            self.history[self.len] = token;
            self.len += 1;
        } else {
            const keep = pool_capacity / 2;
            std.mem.copyForwards(u32, self.history[0..keep], self.history[pool_capacity - keep ..]);
            self.len = keep;
            self.history[self.len] = token;
            self.len += 1;
        }
    }

    /// Propose continuation tokens from shared history given the current tail.
    /// `tail` is the most recent tokens (the n-gram query); writes into `out`.
    /// Returns number of tokens proposed (0 if no match).
    pub fn propose(self: *SharedNgramPool, tail: []const u32, max_draft: usize, out: []u32) usize {
        if (tail.len < min_ngram or max_draft == 0) return 0;
        self.lock();
        defer self.unlock();
        const hist = self.history[0..self.len];
        if (hist.len < min_ngram + 1) return 0;

        var best_pos: usize = 0;
        var best_len: usize = 0;
        const max_n = @min(max_ngram, @min(tail.len, hist.len - 1));
        var n: usize = max_n;
        while (n >= min_ngram) : (n -= 1) {
            const pat = tail[tail.len - n ..];
            const end = hist.len - n;
            var pos: usize = 0;
            while (pos + n <= end) : (pos += 1) {
                if (std.mem.eql(u32, hist[pos .. pos + n], pat)) {
                    best_pos = pos + n;
                    best_len = n;
                    break;
                }
            }
            if (best_len > 0) break;
        }
        if (best_len == 0) return 0;
        const avail = hist.len - best_pos;
        const n_out = @min(@min(avail, max_draft), out.len);
        @memcpy(out[0..n_out], hist[best_pos..][0..n_out]);
        return n_out;
    }
};

/// Global singleton for server mode.  Created once at server start; null in CLI mode.
pub var global_pool: ?SharedNgramPool = null;

/// Suffix Decoding: exact suffix matching with dynamic speculation depth.
///
/// vLLM-style suffix decoding (https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/):
/// - Maintains a large cross-request token cache (default: 10k tokens)
/// - Finds the LONGEST suffix of the current context that exists earlier in the cache
/// - Longer matches → deeper speculation (up to max_tree_depth)
/// - No draft model required; zero overhead beyond cache lookups
///
/// Dynamic depth: match_len == min_suffix → k=1; match_len >= max_suffix → k=max_k.
pub const SuffixState = struct {
    const cache_capacity: usize = 10_000;
    const min_suffix: usize = 2; // minimum suffix length to attempt
    const max_suffix: usize = 32; // maximum suffix length to search
    const default_max_k: usize = 24; // vLLM default max tree depth

    history: []u32,
    len: usize = 0,
    max_k: usize = default_max_k,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !SuffixState {
        return SuffixState{
            .history = try allocator.alloc(u32, cache_capacity),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SuffixState) void {
        self.allocator.free(self.history);
    }

    /// Add a generated token to the suffix cache.
    pub fn push(self: *SuffixState, token: u32) void {
        if (self.len < cache_capacity) {
            self.history[self.len] = token;
            self.len += 1;
        } else {
            // Compact: keep the second half
            const keep = cache_capacity / 2;
            std.mem.copyForwards(u32, self.history[0..keep], self.history[cache_capacity - keep ..]);
            self.len = keep;
            self.history[self.len] = token;
            self.len += 1;
        }
    }

    /// Propose up to `max_draft` tokens using suffix matching.
    /// Returns both the number proposed AND the effective match length
    /// (used to compute dynamic k: longer match → more draft tokens).
    pub fn proposeWithDepth(self: *const SuffixState, max_draft: usize, out: []u32) struct { n: usize, match_len: usize } {
        if (self.len < min_suffix + 1 or max_draft == 0) return .{ .n = 0, .match_len = 0 };

        const hist = self.history[0..self.len];

        // Try longest suffix first (dynamic: longer match → more tokens proposed)
        const max_n = @min(max_suffix, self.len - 1);
        var n: usize = max_n;
        while (n >= min_suffix) : (n -= 1) {
            const suffix = hist[self.len - n ..];
            const search_end = self.len - n;

            var pos: usize = 0;
            while (pos + n <= search_end) : (pos += 1) {
                if (std.mem.eql(u32, hist[pos .. pos + n], suffix)) {
                    // Found match at pos: propose continuation
                    const avail = self.len - (pos + n);
                    const n_out = @min(@min(avail, max_draft), out.len);
                    @memcpy(out[0..n_out], hist[pos + n ..][0..n_out]);
                    return .{ .n = n_out, .match_len = n };
                }
            }
        }
        return .{ .n = 0, .match_len = 0 };
    }

    /// Propose tokens and compute dynamic speculation depth.
    /// Depth scales with match quality: 1 token for minimum match, max_k for maximum.
    pub fn propose(self: *const SuffixState, out: []u32) usize {
        const result = self.proposeWithDepth(self.max_k, out);
        if (result.n == 0) return 0;
        // Dynamic depth: scale proposed count by match quality
        const quality = @as(f32, @floatFromInt(result.match_len - min_suffix)) /
            @as(f32, @floatFromInt(max_suffix - min_suffix));
        const dynamic_k = @as(usize, @intFromFloat(@as(f32, @floatFromInt(self.max_k)) * @min(1.0, quality + 0.2)));
        return @min(result.n, @max(1, dynamic_k));
    }
};

test "suffix propose basic" {
    var s = try SuffixState.init(std.testing.allocator);
    defer s.deinit();

    const tokens = [_]u32{ 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8 };
    for (tokens) |t| s.push(t);

    var draft: [8]u32 = undefined;
    const n = s.propose(&draft);
    // Suffix "1 2 3 4 5" (last 5) matches hist[0..5] → propose 4 5 6 7 8
    try std.testing.expect(n > 0);
    try std.testing.expect(n <= s.max_k);
}

test "suffix no match" {
    var s = try SuffixState.init(std.testing.allocator);
    defer s.deinit();

    // History with no suffix match for current tail
    for ([_]u32{ 1, 2, 3, 4 }) |t| s.push(t);
    // Current tail (from history last 2 = "3 4") — no match earlier
    var draft: [4]u32 = undefined;
    const n = s.propose(&draft);
    try std.testing.expect(n == 0);
}

test "fuzz: SuffixState" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            var s = try SuffixState.init(std.testing.allocator);
            defer s.deinit();

            const n_push = smith.valueWithHash(u8, 0);
            for (0..n_push) |i| {
                s.push(smith.valueWithHash(u32, @as(u32, @truncate(i)) +% 100));
            }
            var draft: [32]u32 = undefined;
            const n = s.propose(&draft);
            try std.testing.expect(n <= s.max_k);
        }
    }.f, .{});
}

test "ngram basic proposal" {
    var state = NgramState{};

    // Push a repeating pattern: 1 2 3 4 5 1 2 3 4 5 1 2 3
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3 };
    for (tokens) |t| state.push(t);

    var draft: [8]u32 = undefined;
    const n = state.propose(5, &draft);

    // Last 8 tokens "1 2 3 4 5 1 2 3" match hist[0..8] → propose hist[8..13] = "4 5 1 2 3"
    try std.testing.expectEqual(@as(usize, 5), n);
    try std.testing.expectEqual(@as(u32, 4), draft[0]);
    try std.testing.expectEqual(@as(u32, 5), draft[1]);
    try std.testing.expectEqual(@as(u32, 1), draft[2]);
    try std.testing.expectEqual(@as(u32, 2), draft[3]);
    try std.testing.expectEqual(@as(u32, 3), draft[4]);
}

test "ngram no match" {
    var state = NgramState{};
    const tokens = [_]u32{ 1, 2, 3, 4, 5 };
    for (tokens) |t| state.push(t);

    var draft: [8]u32 = undefined;
    const n = state.propose(5, &draft);
    try std.testing.expectEqual(@as(usize, 0), n);
}

test "ngram short history" {
    var state = NgramState{};
    state.push(1);
    state.push(2);

    var draft: [8]u32 = undefined;
    const n = state.propose(5, &draft);
    try std.testing.expectEqual(@as(usize, 0), n);
}

test "fuzz: all ngram functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            var state = NgramState{};

            // push: fill state with random tokens
            const num_pushes = smith.valueWithHash(u8, 0) | 4; // at least 4
            for (0..num_pushes) |i| {
                const token = smith.valueWithHash(u32, @truncate(i));
                state.push(token);
                // Invariant: len never exceeds capacity
                try std.testing.expect(state.len <= history_capacity);
            }

            // push: verify len is correct (capped by capacity)
            try std.testing.expect(state.len > 0);

            // propose: call with random max_draft
            var draft: [64]u32 = undefined;
            const max_draft = smith.valueWithHash(u8, 100) % 64;
            const n = state.propose(max_draft, &draft);

            // Invariant: proposed count never exceeds max_draft or output buffer
            try std.testing.expect(n <= max_draft);
            try std.testing.expect(n <= draft.len);

            // propose: with zero max_draft always returns 0
            const n_zero = state.propose(0, &draft);
            try std.testing.expect(n_zero == 0);

            // push: force ring buffer wrap by pushing beyond capacity
            var state2 = NgramState{};
            for (0..history_capacity + 10) |i| {
                state2.push(smith.valueWithHash(u32, @as(u32, @truncate(i)) +% 0xBEEF));
                try std.testing.expect(state2.len <= history_capacity);
            }

            // propose: on wrapped state still returns bounded result
            const n2 = state2.propose(8, draft[0..8]);
            try std.testing.expect(n2 <= 8);
        }
    }.f, .{});
}

/// Lookahead decoding (Jacobi/lookahead method, Fu et al. 2024).
///
/// Maintains a lookahead window of W branches, each of length N.
/// At each step, all W×N draft tokens are proposed simultaneously and verified
/// in one target forward pass. Unlike n-gram, lookahead generates NOVEL tokens
/// via parallel sampling rather than replaying history.
///
/// Algorithm (simplified):
///   Window W = 5 branches of N = 7 tokens each (W×N = 35 candidates)
///   1. Each branch advances: branch[i] = sample(target(branch[i][-1])) × N times
///   2. Check if any n-gram in the window matches the current context suffix
///   3. If match: propose that branch as draft tokens (up to N tokens)
///   4. If no match: fall back to single-token decode
///
/// In practice, lookahead gives lower acceptance than EAGLE but higher than n-gram,
/// since branches explore the likely continuation space rather than pure repetition.
pub const LookaheadState = struct {
    pub const max_branches: usize = 7;
    pub const max_window: usize = 16;
    const lookahead_min_match: usize = 2;

    /// Each branch is a sequence of candidate tokens generated by lookahead.
    branches: [max_branches][max_window]u32 = undefined,
    branch_len: [max_branches]usize = .{0} ** max_branches,
    n_branches: usize = 5,
    window: usize = 7,

    /// Seed branches with continuations from an initial token set.
    /// Called after prefill; branches start from the last `n_branches` distinct tokens.
    pub fn seed(self: *LookaheadState, history: []const u32) void {
        const n = @min(self.n_branches, history.len);
        for (0..n) |i| {
            self.branches[i][0] = history[history.len - n + i];
            self.branch_len[i] = 1;
        }
    }

    /// Advance all branches by one token (caller provides next tokens per branch).
    pub fn advance(self: *LookaheadState, next_tokens: []const u32) void {
        const n = @min(next_tokens.len, self.n_branches);
        for (0..n) |i| {
            const bl = self.branch_len[i];
            if (bl < max_window) {
                self.branches[i][bl] = next_tokens[i];
                self.branch_len[i] = bl + 1;
            } else {
                // Shift branch left, append new token
                std.mem.copyForwards(u32, &self.branches[i], self.branches[i][1..max_window]);
                self.branches[i][max_window - 1] = next_tokens[i];
            }
        }
    }

    /// Try to find an n-gram match between any branch and the current context tail.
    /// Returns the matched branch index and match length, or null if no match.
    pub fn findMatch(self: *const LookaheadState, context_tail: []const u32) ?struct { branch: usize, match_len: usize } {
        if (context_tail.len < lookahead_min_match) return null;
        var best_branch: usize = 0;
        var best_len: usize = 0;
        for (0..self.n_branches) |b| {
            const br = self.branches[b][0..self.branch_len[b]];
            if (br.len < lookahead_min_match) continue;
            // Try longest suffix match
            var n: usize = @min(lookahead_min_match + 3, @min(context_tail.len, br.len));
            while (n >= lookahead_min_match) : (n -= 1) {
                const tail = context_tail[context_tail.len - n ..];
                if (std.mem.eql(u32, br[0..n], tail)) {
                    if (n > best_len) {
                        best_len = n;
                        best_branch = b;
                    }
                    break;
                }
            }
        }
        if (best_len < lookahead_min_match) return null;
        return .{ .branch = best_branch, .match_len = best_len };
    }

    /// Copy continuation tokens from a matched branch into the draft buffer.
    /// Returns number of tokens copied (tokens AFTER the matched prefix).
    pub fn proposeContinuation(self: *const LookaheadState, branch: usize, match_len: usize, max_draft: usize, out: []u32) usize {
        const br = self.branches[branch][0..self.branch_len[branch]];
        if (match_len >= br.len) return 0;
        const start = match_len;
        const avail = br.len - start;
        const n = @min(@min(avail, max_draft), out.len);
        @memcpy(out[0..n], br[start..][0..n]);
        return n;
    }
};

test "lookahead seed and match" {
    var ls = LookaheadState{};
    ls.seed(&[_]u32{ 1, 2, 3, 4, 5 });
    // Branch 0 starts with token 1, branch 1 with 2, etc.
    try std.testing.expectEqual(@as(u32, 1), ls.branches[0][0]);
    try std.testing.expectEqual(@as(u32, 3), ls.branches[2][0]);

    // Advance branches
    ls.advance(&[_]u32{ 10, 20, 30, 40, 50 });
    try std.testing.expectEqual(@as(usize, 2), ls.branch_len[0]);
    try std.testing.expectEqual(@as(u32, 10), ls.branches[0][1]);
}
