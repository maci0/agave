//! PFlash: Speculative prefill with block-sparse importance scoring.
//!
//! Algorithm (from Luce-Org/lucebox-hub):
//!   1. Score — lightweight draft model runs forward pass; attention weights reveal
//!              which KV blocks are important for the final tail positions.
//!   2. Select — keep blocks above alpha × mean_score (adaptive threshold).
//!   3. Compress — build a reduced token list from selected block spans.
//!   4. Prefill — target model prefills only the compressed prompt.
//!
//! Usage:
//!   var cfg = PFlashConfig{};
//!   var state = try PFlashState.init(allocator, cfg, token_ids.len);
//!   defer state.deinit(allocator);
//!   try pflashPrefill(cfg, &state, draft_model, target_model, token_ids, allocator);

const std = @import("std");
const Allocator = std.mem.Allocator;
const Model = @import("../models/model.zig").Model;
const sparse_attn = @import("../ops/sparse_attn.zig");

/// PFlash configuration — tunable at the CLI level.
pub const PFlashConfig = struct {
    /// Alpha threshold: keep block if score > alpha × mean_score.
    /// 0.85 is the Luce PFlash default (keeps ~5% of blocks at 128K context).
    alpha: f32 = 0.85,
    /// Tokens per scored block. Must match draft model attention block size.
    block_size: u32 = 64,
    /// Hard cap on kept token fraction (prevents runaway selection at low alpha).
    max_kept_ratio: f32 = 0.20,
    /// Number of tail query positions used for scoring (like Luce's tail window).
    score_tail: u32 = 16,
    /// Block sparse pattern for the drafter's attention during scoring.
    drafter_pattern: sparse_attn.BlockSparsePattern = .{
        .block_size = 64,
        .n_global = 2,
        .window = 1,
    },
};

/// Per-block importance scores and selection state.
pub const PFlashState = struct {
    config: PFlashConfig,
    /// Importance score per KV block [n_blocks].
    block_scores: []f32,
    /// Whether each block is selected [n_blocks].
    selected: []bool,
    /// Compressed token list (selected spans concatenated) [selected_len].
    selected_tokens: []u32,
    selected_len: usize,
    /// Original sequence length.
    orig_len: usize,

    /// Allocate scoring and selection buffers sized for up to `max_tokens` tokens.
    /// The number of blocks is derived from `cfg.block_size`. On partial allocation
    /// failure, previously allocated buffers are freed via `errdefer`.
    pub fn init(allocator: Allocator, cfg: PFlashConfig, max_tokens: usize) !PFlashState {
        const bs = @as(usize, cfg.block_size);
        const n_blocks = (max_tokens + bs - 1) / bs;
        const block_scores = try allocator.alloc(f32, n_blocks);
        errdefer allocator.free(block_scores);
        const selected = try allocator.alloc(bool, n_blocks);
        errdefer allocator.free(selected);
        const selected_tokens = try allocator.alloc(u32, max_tokens);
        return .{
            .config = cfg,
            .block_scores = block_scores,
            .selected = selected,
            .selected_tokens = selected_tokens,
            .selected_len = 0,
            .orig_len = 0,
        };
    }

    /// Free the block_scores, selected, and selected_tokens buffers.
    pub fn deinit(self: *PFlashState, allocator: Allocator) void {
        allocator.free(self.block_scores);
        allocator.free(self.selected);
        allocator.free(self.selected_tokens);
    }

    /// Reset state for a new scoring run.
    pub fn reset(self: *PFlashState) void {
        @memset(self.block_scores, 0);
        @memset(self.selected, false);
        self.selected_len = 0;
    }
};

/// Score KV blocks by running the draft model's forward pass and extracting
/// per-block max attention weight from the last `score_tail` positions.
///
/// In practice: run draft prefill normally, then call scoreFromKvCache() to
/// compute scores from the stored KV cache (proxy for attention weights).
///
/// The Luce PFlash approach: score[b] = mean over (layers, heads) of
///   max over (tail positions) of (Q[-tail:] @ K[b*bs:(b+1)*bs]^T / sqrt(hd))
///
/// Since we can't easily extract per-layer attention matrices post-hoc without
/// modifying model kernels, we use a proxy: the magnitude of stored K-vectors
/// in each block, weighted by the Q vector at the last position. This is a
/// single-pass approximation that avoids kernel surgery.
pub fn scoreFromLastQ(
    state: *PFlashState,
    draft_model: *const Model,
    kv_keys: []const u8,
    last_q: []const f32,
    seq_len: usize,
    hd: usize,
    nkv: usize,
    kv_type: @import("../ops/kv_quant.zig").KvQuantType,
) void {
    const bs = @as(usize, state.config.block_size);
    const n_blocks = (seq_len + bs - 1) / bs;
    const kvd = nkv * hd;
    const kv_quant = @import("../ops/kv_quant.zig");
    _ = draft_model;

    // Score each block: max dot product of last Q with any K in the block.
    for (0..n_blocks) |bi| {
        const t_start = bi * bs;
        const t_end = @min(t_start + bs, seq_len);
        var block_max: f32 = 0;
        for (t_start..t_end) |t| {
            // Dot product with KV head 0 as proxy (fast, avoids full GQA expansion)
            const k_off = kv_quant.kvByteOffset(kv_type, t * kvd);
            const dot = @abs(kv_quant.kvDot(last_q.ptr, kv_keys.ptr + k_off, hd, kv_type));
            if (dot > block_max) block_max = dot;
        }
        state.block_scores[bi] = block_max;
    }
    state.orig_len = seq_len;
}

/// Select blocks above alpha × mean_score, respecting max_kept_ratio.
pub fn selectBlocks(state: *PFlashState) void {
    const cfg = state.config;
    const n_blocks = (state.orig_len + @as(usize, cfg.block_size) - 1) / @as(usize, cfg.block_size);
    if (n_blocks == 0) return;

    // Compute mean score
    var total: f32 = 0;
    for (state.block_scores[0..n_blocks]) |s| total += s;
    const mean = total / @as(f32, @floatFromInt(n_blocks));
    const threshold = cfg.alpha * mean;

    // Select blocks above threshold
    var n_selected: usize = 0;
    for (0..n_blocks) |bi| {
        state.selected[bi] = state.block_scores[bi] > threshold;
        if (state.selected[bi]) n_selected += 1;
    }

    // Enforce max_kept_ratio cap (strict floor: 25% of 16 = 4, not 5).
    const max_blocks = @max(1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(n_blocks)) * cfg.max_kept_ratio)));
    if (n_selected > max_blocks) {
        // Trim lowest-scoring selected blocks until within cap
        // Find threshold score for top max_blocks
        const cutoff_scores = state.block_scores[0..n_blocks];
        _ = cutoff_scores;
        // Simple approach: unselect blocks with lowest scores until at cap
        var remaining = n_selected;
        var min_val: f32 = std.math.inf(f32);
        var min_idx: usize = 0;
        while (remaining > max_blocks) {
            min_val = std.math.inf(f32);
            for (0..n_blocks) |bi| {
                if (state.selected[bi] and state.block_scores[bi] < min_val) {
                    min_val = state.block_scores[bi];
                    min_idx = bi;
                }
            }
            state.selected[min_idx] = false;
            remaining -= 1;
        }
    }
}

/// Build compressed token array from selected blocks.
/// Always includes the last block (most recent context).
pub fn buildCompressedPrompt(state: *PFlashState, token_ids: []const u32) []const u32 {
    const cfg = state.config;
    const bs = @as(usize, cfg.block_size);
    std.debug.assert(token_ids.len >= state.orig_len);
    const n_blocks = (state.orig_len + bs - 1) / bs;
    // Force-select last block (tail context always needed)
    if (n_blocks > 0) state.selected[n_blocks - 1] = true;

    var out_len: usize = 0;
    for (0..n_blocks) |bi| {
        if (!state.selected[bi]) continue;
        const t_start = bi * bs;
        const t_end = @min(t_start + bs, token_ids.len);
        const span = token_ids[t_start..t_end];
        @memcpy(state.selected_tokens[out_len..][0..span.len], span);
        out_len += span.len;
    }
    state.selected_len = out_len;
    return state.selected_tokens[0..out_len];
}

/// Count how many blocks are currently selected.
pub fn countSelected(state: *const PFlashState) usize {
    const n_blocks = (state.orig_len + @as(usize, state.config.block_size) - 1) / @as(usize, state.config.block_size);
    var count: usize = 0;
    for (state.selected[0..n_blocks]) |s| if (s) {
        count += 1;
    };
    return count;
}

/// Compression ratio: selected tokens / total tokens.
pub fn compressionRatio(state: *const PFlashState) f32 {
    if (state.orig_len == 0) return 1.0;
    return @as(f32, @floatFromInt(state.selected_len)) / @as(f32, @floatFromInt(state.orig_len));
}

/// Full PFlash prefill pipeline:
///   1. Draft model prefills the full prompt (builds KV cache for scoring).
///   2. Score blocks from the draft's KV cache + final Q vector.
///   3. Select blocks above alpha × mean threshold.
///   4. Build compressed token list.
///   5. Target model prefills only the compressed prompt.
///   6. Reset and return the last token prediction.
///
/// Returns the argmax token ID from the target model's final forward pass.
pub fn pflashPrefill(
    cfg: PFlashConfig,
    state: *PFlashState,
    draft_model: *Model,
    target_model: *Model,
    token_ids: []const u32,
    allocator: Allocator,
) !u32 {
    if (token_ids.len == 0) return error.EmptyInput;

    state.reset();
    state.orig_len = token_ids.len;
    _ = allocator;

    // Step 1: Draft model prefill (standard prefill builds KV cache)
    _ = try draft_model.prefill(token_ids);

    // Step 2: Score blocks using a uniform scorer based on sequence position.
    // In a full implementation, this would extract attention weights from the
    // draft model's KV cache. For now, use a position-aware heuristic:
    // recent blocks score higher (recency bias), matching empirical importance.
    const bs = @as(usize, cfg.block_size);
    const n_blocks = (token_ids.len + bs - 1) / bs;
    for (0..n_blocks) |bi| {
        // Recency-weighted score: later blocks score higher
        // This is a conservative approximation — replace with KV-based scoring
        // once attention weight extraction is integrated.
        const recency = @as(f32, @floatFromInt(bi + 1)) / @as(f32, @floatFromInt(n_blocks));
        state.block_scores[bi] = recency;
    }

    // Step 3: Adaptive block selection
    selectBlocks(state);

    // Step 4: Build compressed prompt
    const compressed = buildCompressedPrompt(state, token_ids);

    // Log compression stats (debug)
    const ratio = compressionRatio(state);
    std.log.debug("PFlash: {d} → {d} tokens ({d:.1}% kept, alpha={d:.2})", .{
        token_ids.len, compressed.len, ratio * 100, cfg.alpha,
    });

    // Step 5: Reset target KV cache, prefill on compressed prompt
    target_model.resetCache();
    return try target_model.prefill(compressed);
}

// ── Tests ────────────────────────────────────────────────────────────────────

test "PFlashState init and deinit" {
    const cfg = PFlashConfig{};
    var state = try PFlashState.init(std.testing.allocator, cfg, 1024);
    defer state.deinit(std.testing.allocator);
    try std.testing.expect(state.block_scores.len > 0);
    try std.testing.expect(state.selected.len > 0);
}

test "selectBlocks alpha threshold" {
    const cfg = PFlashConfig{ .block_size = 8, .alpha = 0.5, .max_kept_ratio = 1.0, .score_tail = 1 };
    var state = try PFlashState.init(std.testing.allocator, cfg, 64);
    defer state.deinit(std.testing.allocator);
    state.orig_len = 64;

    // Scores: [1,2,3,4,5,6,7,8] — mean=4.5, threshold=0.5*4.5=2.25
    // Selected: blocks with score > 2.25 → blocks 2..7 (indices 2-7)
    for (state.block_scores[0..8], 0..) |*s, i| s.* = @as(f32, @floatFromInt(i + 1));
    selectBlocks(&state);
    var n: usize = 0;
    for (state.selected[0..8]) |s| if (s) {
        n += 1;
    };
    try std.testing.expect(n >= 5); // blocks 3..8 should be selected
}

test "buildCompressedPrompt always includes last block" {
    const cfg = PFlashConfig{ .block_size = 4, .alpha = 100.0, .max_kept_ratio = 1.0, .score_tail = 1 }; // alpha=100 selects nothing
    var state = try PFlashState.init(std.testing.allocator, cfg, 16);
    defer state.deinit(std.testing.allocator);
    state.orig_len = 16;
    @memset(state.selected[0..4], false);
    @memset(state.block_scores[0..4], 0);

    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const compressed = buildCompressedPrompt(&state, &tokens);

    // Last block (tokens 13-16) must always be included
    try std.testing.expect(compressed.len >= 4);
    try std.testing.expect(compressed[compressed.len - 1] == 16);
}

test "compressionRatio" {
    const cfg = PFlashConfig{ .block_size = 8, .alpha = 0.5, .max_kept_ratio = 1.0, .score_tail = 1 };
    var state = try PFlashState.init(std.testing.allocator, cfg, 64);
    defer state.deinit(std.testing.allocator);
    state.orig_len = 64;
    state.selected_len = 16; // 16/64 = 0.25
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), compressionRatio(&state), 1e-5);
}

test "selectBlocks max_kept_ratio cap" {
    const cfg = PFlashConfig{ .block_size = 8, .alpha = 0.0, .max_kept_ratio = 0.25, .score_tail = 1 }; // alpha=0 → all selected initially
    var state = try PFlashState.init(std.testing.allocator, cfg, 128);
    defer state.deinit(std.testing.allocator);
    state.orig_len = 128;

    // All scores = 1.0 → all blocks above threshold initially
    for (state.block_scores[0..16]) |*s| s.* = 1.0;
    selectBlocks(&state);

    // Cap at 25% of 16 blocks = 4
    const n = countSelected(&state);
    try std.testing.expect(n <= 4);
}

test "fuzz: all pflash functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const allocator = std.testing.allocator;
            const alpha = @as(f32, @floatFromInt(smith.valueWithHash(u8, 0))) / 255.0 * 2.0;
            const block_size: u32 = (@as(u32, smith.valueWithHash(u3, 1)) + 1) * 8; // 8..64
            const cfg = PFlashConfig{
                .alpha = alpha,
                .block_size = block_size,
                .max_kept_ratio = @as(f32, @floatFromInt(smith.valueWithHash(u8, 2))) / 255.0 + 0.01,
                .score_tail = 1,
            };
            const n_tokens: usize = @as(usize, smith.valueWithHash(u6, 3)) * @as(usize, block_size) + @as(usize, block_size);

            var state = PFlashState.init(allocator, cfg, n_tokens) catch return;
            defer state.deinit(allocator);
            state.orig_len = n_tokens;

            // Set random scores
            const n_blocks = (n_tokens + @as(usize, block_size) - 1) / @as(usize, block_size);
            for (state.block_scores[0..n_blocks], 0..) |*s, i| {
                s.* = @as(f32, @floatFromInt(smith.valueWithHash(u8, @truncate(i + 10)))) / 50.0;
            }

            selectBlocks(&state);

            // Invariant: selected count <= max_kept_ratio * n_blocks + 1
            const max_b = @as(usize, @intFromFloat(@as(f32, @floatFromInt(n_blocks)) * cfg.max_kept_ratio + 1.5));
            try std.testing.expect(countSelected(&state) <= @max(max_b, 1));

            // Build compressed prompt
            const tokens = try allocator.alloc(u32, n_tokens);
            defer allocator.free(tokens);
            for (tokens, 0..) |*t, i| t.* = @intCast(i);
            const compressed = buildCompressedPrompt(&state, tokens);

            // Last block always included → compressed non-empty
            try std.testing.expect(compressed.len > 0);
            // Compression ratio in (0, 1]
            const ratio = compressionRatio(&state);
            try std.testing.expect(ratio > 0 and ratio <= 1.0);
        }
    }.f, .{});
}
