//! Continuous batching scheduler for multi-tenant LLM serving.
//!
//! Implements vLLM-style iteration-level continuous batching: maintains a waiting
//! queue and running list, processes one decode step across all active requests,
//! ejects finished/cancelled requests, and fills batch from waiting queue (FIFO).

const std = @import("std");
const Model = @import("../models/model.zig").Model;
const Allocator = std.mem.Allocator;
const RadixTree = @import("../kvcache/manager.zig").RadixTree;
const Metrics = @import("metrics.zig").Metrics;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;
const Prefetcher = @import("../kvcache/prefetch.zig").Prefetcher;

/// Scheduler loop poll interval (nanoseconds).
const scheduler_poll_ns: u64 = 1_000_000; // 1ms

/// Cache-aware priority coefficient (α in the priority formula).
/// Higher values give more weight to cached prefix length.
/// Configurable via future CLI flag.
const cache_priority_alpha: f32 = 0.5;

/// Per-request state for continuous batching.
pub const Request = struct {
    id: u64,
    tokens: std.ArrayList(u32),
    last_token_id: u32,
    is_finished: bool,
    is_cancelled: std.atomic.Value(bool),
    enqueued_at: i64,
    prompt_tokens: u32,
    cached_prefix_len: u32 = 0,
    cached_blocks: []const u32 = &[_]u32{},
    prompt_tokens_slice: []const u32 = &[_]u32{},
    block_table: []u32 = &[_]u32{}, // Physical block IDs for this request (placeholder for PagedAttention integration)
    /// Current position in prompt prefill. When < prompt_tokens, the scheduler
    /// feeds prompt tokens to the model. When == prompt_tokens, decode begins.
    prefill_pos: u32 = 0,
    allocator: Allocator,

    /// Append a token to the output sequence.
    /// Sets is_finished if the token matches any EOG (end-of-generation) ID.
    pub fn appendToken(self: *Request, token: u32, eog_ids: []const u32) void {
        self.tokens.append(self.allocator, token) catch return; // Best-effort in hot path
        self.last_token_id = token;

        for (eog_ids) |eog_id| {
            if (token == eog_id) {
                self.is_finished = true;
                return;
            }
        }
    }

    /// Calculate elapsed time since request was enqueued (in seconds).
    pub fn elapsedSeconds(self: *const Request, now: i64) u32 {
        const elapsed_ms = now - self.enqueued_at;
        return @intFromFloat(@as(f64, @floatFromInt(elapsed_ms)) / 1000.0);
    }

    /// Clean up allocated resources.
    pub fn deinit(self: *Request) void {
        self.tokens.deinit(self.allocator);
    }
};

/// Calculate cache-aware priority for a request.
/// SGLang-style cache-aware scheduling: longer cached prefixes get priority boost.
/// Formula: priority = -1 × (deadline + α × cached_prefix_length)
/// Higher priority = better (should be scheduled sooner).
fn requestPriority(req: *const Request, now: i64) i64 {
    const elapsed_ms = now - req.enqueued_at;
    const deadline_penalty = @as(i64, @intCast(elapsed_ms));
    const cache_bonus = @as(i64, @intCast(req.cached_prefix_len)) * @as(i64, @intFromFloat(cache_priority_alpha * 1000.0));
    return -1 * (deadline_penalty - cache_bonus); // Higher = better
}

/// Scheduler statistics for monitoring.
pub const SchedulerStats = struct {
    waiting_count: u32,
    running_count: u32,
    completed_total: u32,
    cancelled_total: u32,
};

/// Request manager with continuous batching scheduler.
/// Thread-safe: uses mutex to protect queue manipulation.
pub const RequestManager = struct {
    waiting: std.ArrayList(*Request),
    running: std.ArrayList(*Request),
    radix_tree: RadixTree,
    metrics: *Metrics,
    max_batch_size: usize,
    timeout_sec: u32,
    allocator: Allocator,
    mutex: std.Thread.Mutex,
    next_id: std.atomic.Value(u64),
    completed_total: u32 = 0,
    cancelled_total: u32 = 0,

    /// Optional tiered KV cache (from Plan 02).
    tiered_cache: ?*TieredKvCache = null,
    /// Optional prefetch worker (Plan 03).
    prefetcher: ?Prefetcher = null,

    /// Initialize request manager.
    ///
    /// If tiered_cache is provided, Prefetcher is initialized and started.
    /// Otherwise, prefetcher remains null.
    pub fn init(allocator: Allocator, metrics: *Metrics, max_batch_size: usize, timeout_sec: u32, tiered_cache: ?*TieredKvCache) !RequestManager {
        var mgr = RequestManager{
            .waiting = .{},
            .running = .{},
            .radix_tree = try RadixTree.init(allocator),
            .metrics = metrics,
            .max_batch_size = max_batch_size,
            .timeout_sec = timeout_sec,
            .allocator = allocator,
            .mutex = .{},
            .next_id = std.atomic.Value(u64).init(1),
            .tiered_cache = tiered_cache,
            .prefetcher = null,
        };

        // Initialize and start prefetcher if tiered cache available
        if (tiered_cache) |cache| {
            var prefetcher = try Prefetcher.init(allocator, cache);
            errdefer prefetcher.deinit();
            try prefetcher.start();
            mgr.prefetcher = prefetcher;
        }

        return mgr;
    }

    /// Clean up allocated resources.
    /// Note: does NOT free individual requests — caller owns them.
    pub fn deinit(self: *RequestManager) void {
        // Stop prefetcher before deinit (shutdown worker thread)
        if (self.prefetcher) |*prefetcher| {
            prefetcher.deinit();
        }

        self.radix_tree.deinit();
        self.waiting.deinit(self.allocator);
        self.running.deinit(self.allocator);
    }

    /// Enqueue a new request into the waiting queue.
    /// Returns pointer to the request (caller keeps reference for polling).
    /// Queries RadixTree for cached prefix match before allocating new blocks.
    pub fn enqueue(self: *RequestManager, prompt_tokens_slice: []const u32) !*Request {
        const req = try self.allocator.create(Request);
        errdefer self.allocator.destroy(req);

        const now = std.time.milliTimestamp();
        const id = self.next_id.fetchAdd(1, .monotonic);

        // Query RadixTree for longest matching prefix
        const prefix_match = self.radix_tree.matchPrefix(prompt_tokens_slice);

        // Record cache hit or miss in metrics
        if (prefix_match.matched > 0) {
            self.metrics.recordCacheHit(@intCast(prefix_match.matched));
        } else {
            self.metrics.recordCacheMiss(@intCast(prompt_tokens_slice.len));
        }

        req.* = .{
            .id = id,
            .tokens = .{},
            .last_token_id = 0,
            .is_finished = false,
            .is_cancelled = std.atomic.Value(bool).init(false),
            .enqueued_at = now,
            .prompt_tokens = @intCast(prompt_tokens_slice.len),
            .cached_prefix_len = @intCast(prefix_match.matched),
            .cached_blocks = prefix_match.blocks,
            .prompt_tokens_slice = prompt_tokens_slice,
            .allocator = self.allocator,
        };

        self.mutex.lock();
        defer self.mutex.unlock();

        try self.waiting.append(self.allocator, req);
        return req;
    }

    /// Execute one scheduler iteration.
    /// 1. Remove finished/cancelled requests from running
    /// 2. Check timeout on all running requests
    /// 3. Fill batch from waiting queue (FIFO, up to max_batch_size)
    /// 4. Call model.forward() for each running request
    pub fn step(self: *RequestManager, model: *Model, eog_ids: []const u32) !void {
        const now = std.time.milliTimestamp();

        // Lock during queue manipulation
        self.mutex.lock();

        // 1. Remove finished/cancelled from running
        var i: usize = 0;
        while (i < self.running.items.len) {
            const req = self.running.items[i];
            if (req.is_finished) {
                _ = self.running.swapRemove(i);
                self.completed_total += 1;
            } else if (req.is_cancelled.load(.monotonic)) {
                _ = self.running.swapRemove(i);
                self.cancelled_total += 1;
            } else {
                i += 1;
            }
        }

        // 2. Check timeout on running requests
        for (self.running.items) |req| {
            if (req.elapsedSeconds(now) > self.timeout_sec) {
                req.is_cancelled.store(true, .monotonic);
            }
        }

        // 3. Sort waiting queue by cache-aware priority before filling batch
        if (self.waiting.items.len > 1) {
            const SortCtx = struct { now: i64 };
            std.mem.sort(*Request, self.waiting.items, SortCtx{ .now = now }, struct {
                fn lessThan(ctx: SortCtx, a: *Request, b: *Request) bool {
                    return requestPriority(a, ctx.now) < requestPriority(b, ctx.now);
                }
            }.lessThan);
        }

        // 4. Fill batch from waiting queue (highest priority first after sort)
        while (self.running.items.len < self.max_batch_size and self.waiting.items.len > 0) {
            const req = self.waiting.orderedRemove(self.waiting.items.len - 1); // Take from end (highest priority)
            try self.running.append(self.allocator, req);
        }

        self.mutex.unlock();

        // 5. Promote all blocks in running requests' block tables to VRAM (if tiered cache enabled)
        // 6. Prefetch next N blocks during attention compute (if prefetcher enabled)
        if (self.tiered_cache) |cache| {
            for (self.running.items) |req| {
                // Promote all blocks in this request's block table to VRAM
                for (req.block_table) |block_id| {
                    if (cache.needsPromotion(block_id)) {
                        cache.promoteToVram(block_id) catch |err| {
                            std.log.warn("Failed to promote block {d} for request {d}: {}", .{ block_id, req.id, err });
                        };
                    }
                }

                // Prefetch next blocks asynchronously (per D-07: next 2 blocks)
                // This overlaps SSD I/O with GPU attention compute to hide latency
                if (self.prefetcher) |*prefetcher| {
                    const current_block_idx = @divFloor(req.tokens.items.len, cache.block_size);
                    prefetcher.prefetchNext(req.block_table, current_block_idx) catch |err| {
                        std.log.warn("Prefetch failed for request {d}: {}", .{ req.id, err });
                    };
                }
            }
        }

        // 7. Execute forward for all running requests (unlocked — model owns its concurrency)
        for (self.running.items) |req| {
            if (req.is_cancelled.load(.monotonic)) continue;

            // Prefill phase: feed prompt tokens one at a time.
            // Note: continuous batching currently processes one request at a time
            // (no batched prefill). Each step() call processes one token per request.
            if (req.prefill_pos < req.prompt_tokens) {
                const prompt_tid = req.prompt_tokens_slice[req.prefill_pos];
                const next_token = model.forward(prompt_tid) catch |err| {
                    std.log.err("Request {d} prefill failed: {}", .{ req.id, err });
                    req.is_cancelled.store(true, .monotonic);
                    continue;
                };
                req.prefill_pos += 1;

                // Last prefill token produces the first generated token
                if (req.prefill_pos == req.prompt_tokens) {
                    req.last_token_id = next_token;
                    req.appendToken(next_token, eog_ids);
                }
                continue;
            }

            // Decode phase: generate tokens
            const next_token = model.forward(req.last_token_id) catch |err| {
                std.log.err("Request {d} forward failed: {}", .{ req.id, err });
                req.is_cancelled.store(true, .monotonic);
                continue;
            };

            req.appendToken(next_token, eog_ids);

            // On completion, insert full sequence into RadixTree for future reuse
            if (req.is_finished and req.tokens.items.len > 0) {
                const block_ids = model.getBlockTable();
                self.radix_tree.insert(req.tokens.items, block_ids) catch |err| {
                    std.log.warn("Failed to insert sequence into RadixTree: {}", .{err});
                };
            }
        }
    }

    /// Get current scheduler statistics.
    pub fn getStats(self: *RequestManager) SchedulerStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        return .{
            .waiting_count = @intCast(self.waiting.items.len),
            .running_count = @intCast(self.running.items.len),
            .completed_total = self.completed_total,
            .cancelled_total = self.cancelled_total,
        };
    }
};

/// Background scheduler loop.
/// Continuously calls step() until shutdown flag is set.
/// NOT auto-started — server controls lifecycle.
pub fn runSchedulerLoop(
    manager: *RequestManager,
    model: *Model,
    eog_ids: []const u32,
    shutdown: *std.atomic.Value(bool),
) void {
    while (!shutdown.load(.acquire)) {
        manager.step(model, eog_ids) catch |err| {
            std.log.err("Scheduler step failed: {}", .{err});
        };
        std.Thread.sleep(scheduler_poll_ns); // 1ms between iterations
    }
}

// Unit tests
test "enqueue increments waiting count" {
    const allocator = std.testing.allocator;
    var metrics = Metrics{};
    var manager = try RequestManager.init(allocator, &metrics, 4, 30, null);
    defer manager.deinit();

    const req1 = try manager.enqueue(10);
    defer {
        req1.deinit();
        allocator.destroy(req1);
    }
    const req2 = try manager.enqueue(20);
    defer {
        req2.deinit();
        allocator.destroy(req2);
    }

    const stats = manager.getStats();
    try std.testing.expectEqual(@as(u32, 2), stats.waiting_count);
    try std.testing.expectEqual(@as(u32, 0), stats.running_count);

    try std.testing.expectEqual(@as(u64, 1), req1.id);
}

test "step fills batch from waiting queue" {
    const allocator = std.testing.allocator;
    var metrics = Metrics{};
    var manager = try RequestManager.init(allocator, &metrics, 2, 30, null);
    defer manager.deinit();

    // Enqueue 3 requests
    const req1 = try manager.enqueue(10);
    defer {
        req1.deinit();
        allocator.destroy(req1);
    }
    const req2 = try manager.enqueue(10);
    defer {
        req2.deinit();
        allocator.destroy(req2);
    }
    const req3 = try manager.enqueue(10);
    defer {
        req3.deinit();
        allocator.destroy(req3);
    }

    // Create mock model
    var mock_model = MockModel{};
    var model = Model.from(MockModel, &mock_model);

    // Step 1: should fill batch with 2 requests
    try manager.step(&model, &[_]u32{});

    var stats = manager.getStats();
    try std.testing.expectEqual(@as(u32, 1), stats.waiting_count);
    try std.testing.expectEqual(@as(u32, 2), stats.running_count);

    // Step 2: should keep 2 running (not finished)
    try manager.step(&model, &[_]u32{});

    stats = manager.getStats();
    try std.testing.expectEqual(@as(u32, 1), stats.waiting_count);
    try std.testing.expectEqual(@as(u32, 2), stats.running_count);
}

test "step removes finished requests" {
    const allocator = std.testing.allocator;
    var metrics = Metrics{};
    var manager = try RequestManager.init(allocator, &metrics, 2, 30, null);
    defer manager.deinit();

    const req1 = try manager.enqueue(10);
    defer {
        req1.deinit();
        allocator.destroy(req1);
    }
    const req2 = try manager.enqueue(10);
    defer {
        req2.deinit();
        allocator.destroy(req2);
    }

    // Create mock model that returns non-EOS
    var mock_model = MockModel{};
    var model = Model.from(MockModel, &mock_model);

    // Step to move both to running
    try manager.step(&model, &[_]u32{});

    var stats = manager.getStats();
    try std.testing.expectEqual(@as(u32, 2), stats.running_count);

    // Mark first request as finished
    req1.is_finished = true;

    // Step again — should remove only the finished request
    try manager.step(&model, &[_]u32{1}); // Pass EOS token ID

    stats = manager.getStats();
    try std.testing.expectEqual(@as(u32, 1), stats.running_count); // One removed, one still running
}

test "step cancels timed-out requests" {
    const allocator = std.testing.allocator;
    var metrics = Metrics{};
    var manager = try RequestManager.init(allocator, &metrics, 2, 1, null); // 1 second timeout
    defer manager.deinit();

    const req = try manager.enqueue(10);
    defer {
        req.deinit();
        allocator.destroy(req);
    }

    // Simulate request enqueued 2 seconds ago
    req.enqueued_at = std.time.milliTimestamp() - 2000;

    var mock_model = MockModel{};
    var model = Model.from(MockModel, &mock_model);

    // Step 1: move to running
    try manager.step(&model, &[_]u32{});

    // Step 2: check timeout on running requests
    try manager.step(&model, &[_]u32{});

    // Should be cancelled due to timeout
    try std.testing.expect(req.is_cancelled.load(.monotonic));
}

// Mock model for testing
const MockModel = struct {
    eos_token_id: u32 = 1,
    vocab_size: u32 = 1000,
    n_layers: u32 = 12,
    n_embd: u32 = 768,
    n_head: u32 = 12,
    n_head_kv: u32 = 12,
    logits_buf: []f32 = &.{},

    pub fn forward(_: *MockModel, _: u32) !u32 {
        return 42; // Return dummy token
    }

    pub fn resetCache(_: *MockModel) void {}

    pub fn cancel(_: *MockModel) void {}

    pub fn getBlockTable(_: *MockModel) []const u32 {
        return &[_]u32{};
    }
};

// Mock model that returns EOS token
const MockEosModel = struct {
    eos_token_id: u32 = 1,
    vocab_size: u32 = 1000,
    n_layers: u32 = 12,
    n_embd: u32 = 768,
    n_head: u32 = 12,
    n_head_kv: u32 = 12,
    logits_buf: []f32 = &.{},

    pub fn forward(_: *MockEosModel, _: u32) !u32 {
        return 1; // Return EOS token
    }

    pub fn resetCache(_: *MockEosModel) void {}

    pub fn cancel(_: *MockEosModel) void {}

    pub fn getBlockTable(_: *MockEosModel) []const u32 {
        return &[_]u32{};
    }
};
