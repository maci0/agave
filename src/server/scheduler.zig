//! Continuous batching scheduler for multi-tenant LLM serving.
//!
//! Implements vLLM-style iteration-level continuous batching: maintains a waiting
//! queue and running list, processes one decode step across all active requests,
//! ejects finished/cancelled requests, and fills batch from waiting queue (cache-aware priority).

const std = @import("std");
const Io = std.Io;
const Mutex = Io.Mutex;

/// Millisecond timestamp via raw C clock_gettime (Zig 0.16 idiom).
fn milliTimestamp() i64 {
    var ts: std.posix.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &ts);
    return @as(i64, ts.sec) * 1000 + @divTrunc(@as(i64, ts.nsec), 1_000_000);
}

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
const cache_priority_alpha: f64 = 0.5;

/// Milliseconds per second — converts alpha from per-second to per-millisecond units.
const ms_per_second: f64 = 1000.0;

/// Maximum number of requests allowed in the waiting queue.
/// Prevents unbounded memory growth under sustained load.
const max_waiting_queue_size: usize = 1024;

/// Initial token output buffer capacity per request.
/// Avoids repeated reallocation during the decode phase.
const initial_token_capacity: usize = 256;

/// Per-request state for continuous batching.
pub const Request = struct {
    id: u64,
    tokens: std.ArrayList(u32),
    last_token_id: u32,
    is_finished: std.atomic.Value(bool),
    is_cancelled: std.atomic.Value(bool),
    enqueued_at: i64,
    prompt_tokens: u32,
    cached_prefix_len: u32 = 0,
    cached_blocks: []const u32 = &[_]u32{},
    prompt_tokens_slice: []const u32 = &[_]u32{},
    block_table: []u32 = &[_]u32{}, // Physical block IDs for cache-aware scheduling and prefetching
    /// Current position in prompt prefill. When < prompt_tokens, the scheduler
    /// feeds prompt tokens to the model. When == prompt_tokens, decode begins.
    prefill_pos: u32 = 0,
    /// Timestamp (milliTimestamp) when prefill completed and decode began.
    /// Zero until prefill finishes. Used by the server to record TTFT metrics.
    prefill_done_at: i64 = 0,
    allocator: Allocator,

    /// Append a token to the output sequence.
    /// If the token matches any EOG (end-of-generation) ID, sets is_finished
    /// without appending the token — EOG tokens are stop signals, not output.
    pub fn appendToken(self: *Request, token: u32, eog_ids: []const u32) void {
        for (eog_ids) |eog_id| {
            if (token == eog_id) {
                self.is_finished.store(true, .release);
                return;
            }
        }

        self.tokens.append(self.allocator, token) catch |err| {
            std.log.err("req={d} token append failed ({s}), cancelling request", .{ self.id, @errorName(err) });
            self.is_cancelled.store(true, .release);
            return;
        };
        self.last_token_id = token;
    }

    /// Calculate elapsed time since request was enqueued (in seconds).
    /// Clamps to zero if the clock moved backwards (e.g. NTP adjustment).
    pub fn elapsedSeconds(self: *const Request, now: i64) u32 {
        if (now <= self.enqueued_at) return 0;
        const elapsed_ms: u64 = @intCast(now - self.enqueued_at);
        return std.math.cast(u32, elapsed_ms / 1000) orelse std.math.maxInt(u32);
    }

    /// Clean up allocated resources.
    pub fn deinit(self: *Request) void {
        self.tokens.deinit(self.allocator);
    }
};

/// Calculate cache-aware priority for a request.
/// SGLang-style cache-aware scheduling: longer cached prefixes get priority boost.
/// Formula: priority = α × cached_prefix_length − elapsed_ms
/// Higher priority = better (should be scheduled sooner).
fn requestPriority(req: *const Request, now: i64) i64 {
    const elapsed_ms = now - req.enqueued_at;
    const deadline_penalty = @as(i64, @intCast(elapsed_ms));
    const cache_bonus = @as(i64, @intCast(req.cached_prefix_len)) * @as(i64, @intFromFloat(cache_priority_alpha * ms_per_second));
    return cache_bonus - deadline_penalty;
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
    mutex: Mutex,
    io: Io,
    next_id: std.atomic.Value(u64),
    completed_total: u32 = 0,
    cancelled_total: u32 = 0,
    /// Dirty flag: set when enqueue adds a new request, cleared after sort.
    /// Avoids re-sorting an already-sorted waiting queue every 1ms step.
    queue_dirty: bool = false,

    /// Optional tiered KV cache (from Plan 02).
    tiered_cache: ?*TieredKvCache = null,
    /// Optional prefetch worker (Plan 03).
    prefetcher: ?Prefetcher = null,

    /// Initialize request manager.
    ///
    /// If tiered_cache is provided, Prefetcher is initialized and started.
    /// Otherwise, prefetcher remains null.
    pub fn init(allocator: Allocator, metrics: *Metrics, max_batch_size: usize, timeout_sec: u32, tiered_cache: ?*TieredKvCache, io: Io) !RequestManager {
        var waiting: std.ArrayList(*Request) = .empty;
        try waiting.ensureTotalCapacity(allocator, max_waiting_queue_size);
        errdefer waiting.deinit(allocator);

        var running: std.ArrayList(*Request) = .empty;
        try running.ensureTotalCapacity(allocator, max_batch_size);
        errdefer running.deinit(allocator);

        var mgr = RequestManager{
            .waiting = waiting,
            .running = running,
            .radix_tree = try RadixTree.init(allocator),
            .metrics = metrics,
            .max_batch_size = max_batch_size,
            .timeout_sec = timeout_sec,
            .allocator = allocator,
            .mutex = .init,
            .io = io,
            .next_id = std.atomic.Value(u64).init(1),
            .tiered_cache = tiered_cache,
            .prefetcher = null,
        };

        // Initialize and start prefetcher if tiered cache available
        if (tiered_cache) |cache| {
            var prefetcher = Prefetcher.init(cache);
            errdefer prefetcher.deinit();
            try prefetcher.start(io);
            mgr.prefetcher = prefetcher;
        }

        return mgr;
    }

    /// Clean up allocated resources, including any requests still in queues.
    pub fn deinit(self: *RequestManager) void {
        // Stop prefetcher before deinit (shutdown worker thread)
        if (self.prefetcher) |*prefetcher| {
            prefetcher.deinit();
        }

        // Free any requests still in queues (e.g. on shutdown with pending work)
        for (self.waiting.items) |req| {
            req.deinit();
            self.allocator.destroy(req);
        }
        for (self.running.items) |req| {
            req.deinit();
            self.allocator.destroy(req);
        }

        self.radix_tree.deinit();
        self.waiting.deinit(self.allocator);
        self.running.deinit(self.allocator);
    }

    /// Enqueue a new request into the waiting queue.
    /// Returns pointer to the request (caller keeps reference for polling).
    /// Queries RadixTree for cached prefix match before allocating new blocks.
    /// Returns error.Overflow if the waiting queue is full.
    pub fn enqueue(self: *RequestManager, prompt_tokens_slice: []const u32) !*Request {
        const req = try self.allocator.create(Request);
        errdefer self.allocator.destroy(req);

        const now = milliTimestamp();
        const id = self.next_id.fetchAdd(1, .monotonic);

        // Lock mutex for both RadixTree access and queue append (atomic check-and-insert)
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);

        if (self.waiting.items.len >= max_waiting_queue_size) {
            return error.Overflow;
        }

        // Query RadixTree for longest matching prefix (under mutex to prevent
        // concurrent insert from corrupting tree traversal)
        const prefix_match = self.radix_tree.matchPrefix(prompt_tokens_slice);

        // Record cache hit or miss in metrics
        if (prefix_match.matched > 0) {
            self.metrics.recordCacheHit(@intCast(prefix_match.matched), @intCast(prompt_tokens_slice.len));
        } else {
            self.metrics.recordCacheMiss(@intCast(prompt_tokens_slice.len));
        }

        req.* = .{
            .id = id,
            .tokens = .empty,
            .last_token_id = 0,
            .is_finished = std.atomic.Value(bool).init(false),
            .is_cancelled = std.atomic.Value(bool).init(false),
            .enqueued_at = now,
            .prompt_tokens = @intCast(prompt_tokens_slice.len),
            .cached_prefix_len = @intCast(prefix_match.matched),
            .cached_blocks = prefix_match.blocks,
            .prompt_tokens_slice = prompt_tokens_slice,
            .allocator = self.allocator,
        };

        try req.tokens.ensureTotalCapacity(self.allocator, initial_token_capacity);
        errdefer req.tokens.deinit(self.allocator);

        try self.waiting.append(self.allocator, req);
        self.queue_dirty = true;
        return req;
    }

    /// Execute one scheduler iteration.
    /// 1. Remove finished/cancelled requests from running
    /// 2. Check timeout on all running requests
    /// 3. Fill batch from waiting queue (cache-aware priority, up to max_batch_size)
    /// 4. Call model.forward() for each running request
    pub fn step(self: *RequestManager, model: *Model, eog_ids: []const u32) !void {
        const now = milliTimestamp();

        // Lock during queue manipulation (scoped to ensure unlock on all paths)
        {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);

            // 1. Remove finished/cancelled from running list.
            // Do NOT free the request here — the HTTP handler thread holds a
            // pointer to it (via rm.enqueue()) and frees it in its defer block.
            // Freeing here would cause a double-free race with the handler.
            var i: usize = 0;
            while (i < self.running.items.len) {
                const req = self.running.items[i];
                if (req.is_finished.load(.acquire)) {
                    _ = self.running.swapRemove(i);
                    self.completed_total += 1;
                } else if (req.is_cancelled.load(.acquire)) {
                    _ = self.running.swapRemove(i);
                    self.cancelled_total += 1;
                } else {
                    i += 1;
                }
            }

            // 2. Check timeout on running requests
            for (self.running.items) |req| {
                if (req.elapsedSeconds(now) > self.timeout_sec) {
                    std.log.warn("req={d} timed out after {d}s (limit {d}s), cancelling", .{ req.id, req.elapsedSeconds(now), self.timeout_sec });
                    req.is_cancelled.store(true, .release);
                    self.metrics.recordTimeout();
                }
            }

            // 3. Sort waiting queue by cache-aware priority before filling batch.
            // Skip sort when no new requests arrived — relative ordering is stable
            // because priority = α×cache_prefix − elapsed_ms and elapsed_ms changes
            // uniformly for all waiting requests between steps.
            if (self.queue_dirty and self.waiting.items.len > 1) {
                const SortCtx = struct { now: i64 };
                std.mem.sort(*Request, self.waiting.items, SortCtx{ .now = now }, struct {
                    fn lessThan(ctx: SortCtx, a: *Request, b: *Request) bool {
                        return requestPriority(a, ctx.now) < requestPriority(b, ctx.now);
                    }
                }.lessThan);
                self.queue_dirty = false;
            }

            // 4. Fill batch from waiting queue (ascending sort, pop takes highest priority)
            while (self.running.items.len < self.max_batch_size and self.waiting.items.len > 0) {
                const req = self.waiting.pop().?;
                self.running.append(self.allocator, req) catch |err| {
                    // Re-queue to prevent request loss on allocation failure.
                    // If re-queue also fails (OOM), cancel the request so the
                    // handler thread unblocks instead of spinning forever.
                    self.waiting.append(self.allocator, req) catch {
                        req.is_cancelled.store(true, .release);
                    };
                    return err;
                };
                // Record how long this request waited in the queue
                const queue_ms: u64 = @intCast(@max(now - req.enqueued_at, 0));
                self.metrics.recordQueueTime(queue_ms);
            }

            // Update Prometheus gauges
            self.metrics.updateQueueDepth(@intCast(self.waiting.items.len));
            self.metrics.updateActiveRequests(@intCast(self.running.items.len));
        }

        // Update KV cache block metrics from tiered cache (outside mutex —
        // block counts are stable during step since forward() hasn't run yet)
        if (self.tiered_cache) |cache| {
            const total: u32 = @intCast(cache.vram_block_count + cache.ram_block_count + cache.ssd_block_count);
            const free: u32 = @intCast(cache.vram_free_list.items.len + cache.ram_free_list.items.len + cache.ssd_free_list.items.len);
            self.metrics.updateKvBlocks(total - free, total);
        }

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
                    prefetcher.prefetchNext(req.block_table, current_block_idx);
                }
            }
        }

        // 7. Execute forward for all running requests (unlocked — model owns its concurrency)
        for (self.running.items) |req| {
            if (req.is_cancelled.load(.acquire)) continue;
            if (req.is_finished.load(.acquire)) continue;

            // Prefill phase: feed prompt tokens one at a time.
            // Note: continuous batching currently processes one request at a time
            // (no batched prefill). Each step() call processes one token per request.
            if (req.prefill_pos < req.prompt_tokens) {
                const prompt_tid = req.prompt_tokens_slice[req.prefill_pos];
                const next_token = model.forward(prompt_tid) catch |err| {
                    std.log.err("Request {d} prefill failed: {}", .{ req.id, err });
                    req.is_cancelled.store(true, .release);
                    continue;
                };
                req.prefill_pos += 1;

                // Last prefill token produces the first generated token
                if (req.prefill_pos == req.prompt_tokens) {
                    req.prefill_done_at = milliTimestamp();
                    req.last_token_id = next_token;
                    req.appendToken(next_token, eog_ids);
                }
                continue;
            }

            // Decode phase: generate tokens
            const next_token = model.forward(req.last_token_id) catch |err| {
                std.log.err("Request {d} forward failed: {}", .{ req.id, err });
                req.is_cancelled.store(true, .release);
                continue;
            };

            req.appendToken(next_token, eog_ids);

            // On completion, insert full sequence into RadixTree for future reuse.
            // Must hold mutex — enqueue() calls matchPrefix() under the same lock
            // from HTTP handler threads, so concurrent insert would be a data race.
            if (req.is_finished.load(.acquire) and req.tokens.items.len > 0) {
                const block_ids = model.getBlockTable();
                self.mutex.lockUncancelable(self.io);
                defer self.mutex.unlock(self.io);
                self.radix_tree.insert(req.tokens.items, block_ids) catch |err| {
                    std.log.warn("req={d} failed to insert sequence into RadixTree: {}", .{ req.id, err });
                };
            }
        }
    }

    /// Get current scheduler statistics.
    pub fn getStats(self: *RequestManager) SchedulerStats {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);

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
            std.log.err("Scheduler step failed: {} (waiting={d}, running={d})", .{ err, manager.waiting.items.len, manager.running.items.len });
            manager.metrics.recordSchedulerError();
        };
        {
            // Sleep 1ms between iterations via C nanosleep (std.Thread.sleep removed in Zig 0.16)
            const ts = std.posix.timespec{
                .sec = @intCast(scheduler_poll_ns / std.time.ns_per_s),
                .nsec = @intCast(scheduler_poll_ns % std.time.ns_per_s),
            };
            _ = std.c.nanosleep(&ts, null);
        }
    }
}

/// Create a test Io instance for unit tests.
fn testIo() Io {
    var threaded = Io.Threaded.init(std.testing.allocator, .{});
    return threaded.io();
}

// Unit tests
test "enqueue increments waiting count" {
    const allocator = std.testing.allocator;
    var metrics = Metrics{};
    var manager = try RequestManager.init(allocator, &metrics, 4, 30, null, testIo());
    defer manager.deinit();

    const tokens_a = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const req1 = try manager.enqueue(&tokens_a);
    // Requests in waiting queue are freed by manager.deinit() — no defer needed.
    const tokens_b = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    _ = try manager.enqueue(&tokens_b);

    const stats = manager.getStats();
    try std.testing.expectEqual(@as(u32, 2), stats.waiting_count);
    try std.testing.expectEqual(@as(u32, 0), stats.running_count);

    try std.testing.expectEqual(@as(u64, 1), req1.id);
}

test "step fills batch from waiting queue" {
    const allocator = std.testing.allocator;
    var metrics = Metrics{};
    var manager = try RequestManager.init(allocator, &metrics, 2, 30, null, testIo());
    defer manager.deinit();

    // Enqueue 3 requests — all freed by manager.deinit() (still in queues at test end)
    const dummy_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    _ = try manager.enqueue(&dummy_tokens);
    _ = try manager.enqueue(&dummy_tokens);
    _ = try manager.enqueue(&dummy_tokens);

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
    var manager = try RequestManager.init(allocator, &metrics, 2, 30, null, testIo());
    defer manager.deinit();

    const dummy_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const req1 = try manager.enqueue(&dummy_tokens);
    // req1 is removed from running by step() when finished — handler (test) owns cleanup.
    defer {
        req1.deinit();
        allocator.destroy(req1);
    }
    // req2 stays in running — freed by manager.deinit()
    _ = try manager.enqueue(&dummy_tokens);

    // Create mock model that returns non-EOS
    var mock_model = MockModel{};
    var model = Model.from(MockModel, &mock_model);

    // Step to move both to running
    try manager.step(&model, &[_]u32{});

    var stats = manager.getStats();
    try std.testing.expectEqual(@as(u32, 2), stats.running_count);

    // Mark first request as finished
    req1.is_finished.store(true, .release);

    // Step again — should remove the finished request from running
    try manager.step(&model, &[_]u32{1}); // Pass EOS token ID

    stats = manager.getStats();
    try std.testing.expectEqual(@as(u32, 1), stats.running_count); // One removed, one still running
}

test "step cancels timed-out requests" {
    const allocator = std.testing.allocator;
    var metrics = Metrics{};
    var manager = try RequestManager.init(allocator, &metrics, 2, 1, null, testIo()); // 1 second timeout
    defer manager.deinit();

    const dummy_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const req = try manager.enqueue(&dummy_tokens);
    // req stays in running (cancelled but not removed until next step) — freed by manager.deinit()

    // Simulate request enqueued 2 seconds ago
    req.enqueued_at = milliTimestamp() - 2000;

    var mock_model = MockModel{};
    var model = Model.from(MockModel, &mock_model);

    // Step 1: move to running
    try manager.step(&model, &[_]u32{});

    // Step 2: check timeout on running requests
    try manager.step(&model, &[_]u32{});

    // Should be cancelled due to timeout
    try std.testing.expect(req.is_cancelled.load(.acquire));
}

// Mock model for testing
const MockModel = struct {
    eos_token_id: u32 = 1,
    vocab_size: u32 = 1000,
    n_layers: u32 = 12,
    n_embd: u32 = 768,
    n_head: u32 = 12,
    n_head_kv: u32 = 12,
    kv_seq_len: usize = 0,
    logits_buf: []f32 = &.{},

    pub fn forward(_: *MockModel, _: u32) !u32 {
        return 42; // Return dummy token
    }

    pub fn prefill(self: *MockModel, token_ids: []const u32) !u32 {
        var last: u32 = 0;
        for (token_ids) |tid| last = self.forward(tid) catch return error.Cancelled;
        return last;
    }

    pub fn resetCache(_: *MockModel) void {}

    pub fn cancel(_: *MockModel) void {}

    pub fn getBlockTable(_: *MockModel) []const u32 {
        return &[_]u32{};
    }
};

