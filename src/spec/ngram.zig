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
