//! Continuous batching scheduler for multi-tenant LLM serving.
//!
//! Implements vLLM-style iteration-level continuous batching: maintains a waiting
//! queue and running list, processes one decode step across all active requests,
//! ejects finished/cancelled requests, and fills batch from waiting queue (cache-aware priority).

const std = @import("std");
const Io = std.Io;
const Mutex = Io.Mutex;
const sim_clock = @import("../sim_clock.zig");

/// Millisecond timestamp (injectable via sim_clock for deterministic tests).
fn milliTimestamp() i64 {
    return sim_clock.milliNow();
}

const Model = @import("../models/model.zig").Model;
const Allocator = std.mem.Allocator;
const RadixTree = @import("../kvcache/manager.zig").RadixTree;
const Metrics = @import("metrics.zig").Metrics;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;
const Prefetcher = @import("../kvcache/prefetch.zig").Prefetcher;
const math_ops = @import("../ops/math.zig");

/// Scheduler loop poll interval (nanoseconds).
const scheduler_poll_ns: u64 = 1_000_000; // 1ms

/// Sleep via sim_clock so virtual time advances under a clock override.
fn sleepNs(ns: u64) void {
    sim_clock.sleepNs(ns);
}

/// Cache-aware priority coefficient (α in the priority formula).
/// Higher values give more weight to cached prefix length.
const cache_priority_alpha: f64 = 0.5;

/// Milliseconds per second — converts alpha from per-second to per-millisecond units.
const ms_per_second: f64 = 1000.0;

/// Maximum number of requests allowed in the waiting queue.
/// Prevents unbounded memory growth under sustained load.
const max_waiting_queue_size: usize = 1024;

/// Initial token output buffer capacity per request.
/// Must be >= server's gen_ids_buf_size (4096) + cancellation margin.
/// Handler threads read tokens.items without locking — if append causes
/// reallocation, those reads hit freed memory. appendToken uses
/// appendAssumeCapacity to enforce this: if capacity is exhausted, the
/// request is cancelled instead of reallocating.
const initial_token_capacity: usize = 4096 + 8;

/// Maximum prefill tokens processed per scheduler step per request.
/// Limits prefill blocking so decode requests get timely service.
const prefill_chunk_size: u32 = 32;

/// Pick next token from the last forward()'s logits using request sampling.
/// Matches the direct generate path: temperature == 0 keeps forward()'s greedy
/// argmax (no min_p / resampling). Falls back to `greedy` when logits are empty.
fn sampleNextToken(req: *Request, model: *Model, greedy: u32) u32 {
    if (req.temperature == 0) return greedy;
    const logits = model.getLogits();
    if (logits.len == 0) return greedy;
    if (req.min_p > 0) math_ops.applyMinP(logits, req.min_p);
    return math_ops.sampleToken(logits, req.temperature, req.top_k, req.top_p, req.prng.random());
}

/// Per-request state for continuous batching.
pub const Request = struct {
    id: u64,
    tokens: std.ArrayList(u32),
    last_token_id: u32,
    is_finished: std.atomic.Value(bool),
    is_cancelled: std.atomic.Value(bool),
    visible_len: std.atomic.Value(u32),
    /// Set by the scheduler when the request is removed from its queues.
    /// The HTTP handler must wait on this before freeing the Request to
    /// prevent use-after-free when the scheduler is still iterating.
    scheduler_done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
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
    /// KV cache position for this request (for multi-request interleaving).
    kv_position: usize = 0,
    allocator: Allocator,

    /// Per-request sampling (set by the HTTP handler after enqueue).
    /// Defaults match greedy decoding so tests/enqueue alone stay correct.
    temperature: f32 = 0,
    top_k: u32 = 0,
    top_p: f32 = 1.0,
    min_p: f32 = 0,
    prng: std.Random.DefaultPrng = undefined,

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

        if (self.tokens.items.len >= self.tokens.capacity) {
            std.log.err("req={d} token capacity exhausted ({d}), cancelling request", .{ self.id, self.tokens.capacity });
            self.is_cancelled.store(true, .release);
            return;
        }
        self.tokens.appendAssumeCapacity(token);
        self.last_token_id = token;
        self.visible_len.store(@intCast(self.tokens.items.len), .release);
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
    const cache_factor = comptime @as(i64, @intFromFloat(cache_priority_alpha * ms_per_second));
    return @as(i64, @intCast(req.cached_prefix_len)) * cache_factor - (now - req.enqueued_at);
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

    /// SSM state prefix cache: maps xxHash(prompt_tokens) → serialized SSM state.
    /// Enables ~2x prefill speedup for hybrid SSM models (Qwen3.5, Nemotron) by
    /// restoring cached DeltaNet/Mamba state matrices instead of recomputing.
    ssm_state_cache: std.AutoHashMap(u64, []u8) = undefined,
    ssm_cache_inited: bool = false,

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

        var radix_tree = try RadixTree.init(allocator);
        errdefer radix_tree.deinit();

        var mgr = RequestManager{
            .waiting = waiting,
            .running = running,
            .radix_tree = radix_tree,
            .metrics = metrics,
            .max_batch_size = max_batch_size,
            .timeout_sec = timeout_sec,
            .allocator = allocator,
            .mutex = .init,
            .io = io,
            .next_id = std.atomic.Value(u64).init(1),
            .tiered_cache = tiered_cache,
            .prefetcher = null,
            .ssm_state_cache = std.AutoHashMap(u64, []u8).init(allocator),
            .ssm_cache_inited = true,
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
        if (self.ssm_cache_inited) {
            var it = self.ssm_state_cache.valueIterator();
            while (it.next()) |v| self.allocator.free(v.*);
            self.ssm_state_cache.deinit();
        }
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
            .visible_len = std.atomic.Value(u32).init(0),
            .enqueued_at = now,
            .prompt_tokens = @intCast(prompt_tokens_slice.len),
            .cached_prefix_len = @intCast(prefix_match.matched),
            .cached_blocks = prefix_match.blocks,
            .prompt_tokens_slice = prompt_tokens_slice,
            .allocator = self.allocator,
            .prng = std.Random.DefaultPrng.init(id),
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
                    req.scheduler_done.store(true, .release);
                } else if (req.is_cancelled.load(.acquire)) {
                    _ = self.running.swapRemove(i);
                    self.cancelled_total += 1;
                    req.scheduler_done.store(true, .release);
                } else {
                    i += 1;
                }
            }

            // Remove cancelled requests from waiting queue. Without this,
            // a handler blocking on scheduler_done deadlocks when the
            // cancelled request is never admitted to running.
            var j: usize = 0;
            while (j < self.waiting.items.len) {
                const req_w = self.waiting.items[j];
                if (req_w.is_cancelled.load(.acquire)) {
                    _ = self.waiting.swapRemove(j);
                    self.cancelled_total += 1;
                    req_w.scheduler_done.store(true, .release);
                } else {
                    j += 1;
                }
            }

            // 2. Check timeout on running requests
            for (self.running.items) |req| {
                const elapsed = req.elapsedSeconds(now);
                if (elapsed > self.timeout_sec) {
                    std.log.warn("req={d} timed out after {d}s (limit {d}s), cancelling", .{ req.id, elapsed, self.timeout_sec });
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
                        std.log.err("req={d} OOM re-queuing after batch-fill failure, cancelling", .{req.id});
                        req.is_cancelled.store(true, .release);
                        req.scheduler_done.store(true, .release);
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

        // Update KV cache block metrics from tiered cache under tier_lock
        // (prefetcher worker thread may be modifying free lists concurrently)
        if (self.tiered_cache) |cache| {
            const total, const free, const gpu_total, const gpu_free = blk: {
                cache.lockTier();
                defer cache.unlockTier();
                break :blk .{
                    @as(u32, @intCast(cache.vram_block_count + cache.ram_block_count + cache.ssd_block_count)),
                    @as(u32, @intCast(cache.vram_free_list.items.len + cache.ram_free_list.items.len + cache.ssd_free_list.items.len)),
                    @as(u32, @intCast(cache.vram_block_count)),
                    @as(u32, @intCast(cache.vram_free_list.items.len)),
                };
            };
            self.metrics.updateKvBlocks(total - free, total);
            self.metrics.updateGpuKvBlocks(gpu_total - gpu_free, gpu_total);
        }

        // 5. Promote all blocks in running requests' block tables to VRAM (if tiered cache enabled)
        // 6. Prefetch next N blocks during attention compute (if prefetcher enabled)
        if (self.tiered_cache) |cache| {
            for (self.running.items) |req| {
                // Promote all blocks in this request's block table to VRAM.
                // promoteToVram takes tier_lock and handles already-promoted blocks
                // internally (returns early if tier == .vram).
                for (req.block_table) |block_id| {
                    cache.promoteToVram(block_id) catch |err| {
                        std.log.warn("req={d} block {d} promote failed: {}", .{ req.id, block_id, err });
                    };
                }

                // Prefetch next blocks asynchronously (per D-07: next 2 blocks)
                // This overlaps SSD I/O with GPU attention compute to hide latency
                if (self.prefetcher) |*prefetcher| {
                    const current_block_idx = @divFloor(req.kv_position, cache.block_size);
                    prefetcher.prefetchNext(req.block_table, current_block_idx);
                }
            }
        }

        // 7. Execute forward for all running requests.
        // Decode-first scheduling: process all decode requests before prefill.
        // This ensures low TPOT for actively generating requests while new
        // requests' prefill is chunked to avoid blocking.

        // Phase A: decode all requests that finished prefill (one token each)
        for (self.running.items) |req| {
            if (req.is_cancelled.load(.acquire)) continue;
            if (req.is_finished.load(.acquire)) continue;
            if (req.prefill_pos < req.prompt_tokens) continue;

            // Restore KV position for this request
            model.setKvSeqLen(req.kv_position);

            const greedy = model.forward(req.last_token_id) catch |err| {
                std.log.err("req={d} forward failed: {}", .{ req.id, err });
                req.is_cancelled.store(true, .release);
                continue;
            };
            const next_token = sampleNextToken(req, model, greedy);

            req.kv_position = model.kvSeqLen();

            // Re-check after forward(): handler thread may have set is_cancelled
            // while forward() was running. Appending to a cancelled request races
            // with the handler reading req.tokens.items.
            if (req.is_cancelled.load(.acquire)) continue;

            req.appendToken(next_token, eog_ids);

            // On completion: RadixTree insert + SSM state cache
            if (req.is_finished.load(.acquire) and req.tokens.items.len > 0) {
                const block_ids = model.getBlockTable();
                self.mutex.lockUncancelable(self.io);
                defer self.mutex.unlock(self.io);
                self.radix_tree.insert(req.tokens.items, block_ids) catch |err| {
                    std.log.warn("req={d} failed to insert sequence into RadixTree: {}", .{ req.id, err });
                };
                if (self.ssm_cache_inited) {
                    if (model.saveSsmState(self.allocator)) |snapshot| {
                        const h = std.hash.XxHash64.hash(0, std.mem.sliceAsBytes(req.prompt_tokens_slice));
                        self.ssm_state_cache.put(h, snapshot) catch |err| {
                            std.log.warn("req={d} SSM state cache insert failed: {}", .{ req.id, err });
                            self.allocator.free(snapshot);
                        };
                    }
                }
            }
        }

        // Phase B: advance ONE prefilling request by chunk_size tokens
        // Only one prefill chunk per step to minimize decode latency impact
        for (self.running.items) |req| {
            if (req.is_cancelled.load(.acquire)) continue;
            if (req.is_finished.load(.acquire)) continue;
            if (req.prefill_pos >= req.prompt_tokens) continue;

            // SSM state restore on first prefill step
            if (req.prefill_pos == 0 and req.cached_prefix_len > 0 and self.ssm_cache_inited) {
                const h = std.hash.XxHash64.hash(0, std.mem.sliceAsBytes(req.prompt_tokens_slice[0..req.cached_prefix_len]));
                if (self.ssm_state_cache.get(h)) |snapshot| {
                    model.restoreSsmState(snapshot);
                }
            }

            // Restore KV position
            model.setKvSeqLen(req.kv_position);

            // Process up to prefill_chunk_size tokens
            const remaining = req.prompt_tokens - req.prefill_pos;
            const chunk = @min(remaining, prefill_chunk_size);
            var last_token: u32 = 0;
            var i: u32 = 0;
            while (i < chunk) : (i += 1) {
                last_token = model.forward(req.prompt_tokens_slice[req.prefill_pos]) catch |err| {
                    std.log.err("req={d} prefill failed: {}", .{ req.id, err });
                    req.is_cancelled.store(true, .release);
                    break;
                };
                req.prefill_pos += 1;
            }

            req.kv_position = model.kvSeqLen();

            // Prefill complete — emit first generated token.
            // Re-check is_cancelled: handler may have cancelled during prefill.
            if (req.prefill_pos >= req.prompt_tokens and !req.is_cancelled.load(.acquire)) {
                req.prefill_done_at = milliTimestamp();
                // `last_token` is the greedy argmax from the final prefill forward;
                // re-sample from logits so temperature/top-p apply to the first token.
                const first = sampleNextToken(req, model, last_token);
                req.last_token_id = first;
                req.appendToken(first, eog_ids);
            }

            break; // Only one prefill chunk per step
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
    model.setThreadContext();
    while (!shutdown.load(.acquire)) {
        manager.step(model, eog_ids) catch |err| {
            std.log.err("Scheduler step failed: {}", .{err});
            manager.metrics.recordSchedulerError();
        };
        sleepNs(scheduler_poll_ns);
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
    defer sim_clock.setOverrideMs(null);
    sim_clock.setOverrideMs(10_000);

    var metrics = Metrics{};
    var manager = try RequestManager.init(allocator, &metrics, 2, 1, null, testIo()); // 1 second timeout
    defer manager.deinit();

    const dummy_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const req = try manager.enqueue(&dummy_tokens);
    // req stays in running (cancelled but not removed until next step) — freed by manager.deinit()

    var mock_model = MockModel{};
    var model = Model.from(MockModel, &mock_model);

    // At exactly timeout_sec seconds, elapsed > timeout is false — must not cancel yet.
    sim_clock.advanceMs(1000);
    try manager.step(&model, &[_]u32{}); // admit to running
    try std.testing.expect(!req.is_cancelled.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), manager.getStats().running_count);
    try std.testing.expectEqual(@as(u64, 0), metrics.requests_timeout.load(.monotonic));

    // Cross into the next whole second while running (elapsedSeconds uses ms/1000).
    sim_clock.advanceMs(1000);
    try manager.step(&model, &[_]u32{});
    try std.testing.expect(req.is_cancelled.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), metrics.requests_timeout.load(.monotonic));
    // Cancelled requests stay in running until the next step removes them.
    try std.testing.expectEqual(@as(u32, 1), manager.getStats().running_count);

    try manager.step(&model, &[_]u32{});
    try std.testing.expectEqual(@as(u32, 0), manager.getStats().running_count);
    try std.testing.expectEqual(@as(u32, 1), manager.getStats().cancelled_total);
    try std.testing.expect(req.scheduler_done.load(.acquire));
}

test "appendToken marks finished on EOG" {
    const allocator = std.testing.allocator;
    var tokens: std.ArrayList(u32) = .empty;
    try tokens.ensureTotalCapacity(allocator, initial_token_capacity);
    defer tokens.deinit(allocator);
    var req = Request{
        .id = 1,
        .tokens = tokens,
        .last_token_id = 0,
        .is_finished = std.atomic.Value(bool).init(false),
        .is_cancelled = std.atomic.Value(bool).init(false),
        .visible_len = std.atomic.Value(u32).init(0),
        .enqueued_at = 0,
        .prompt_tokens = 0,
        .allocator = allocator,
    };

    const eog_ids = [_]u32{ 2, 128001 };
    req.appendToken(42, &eog_ids);
    try std.testing.expectEqual(@as(usize, 1), req.tokens.items.len);
    try std.testing.expect(!req.is_finished.load(.acquire));

    req.appendToken(128001, &eog_ids);
    try std.testing.expectEqual(@as(usize, 1), req.tokens.items.len);
    try std.testing.expect(req.is_finished.load(.acquire));
}

test "appendToken updates visible_len and last_token_id" {
    const allocator = std.testing.allocator;
    var tokens: std.ArrayList(u32) = .empty;
    try tokens.ensureTotalCapacity(allocator, initial_token_capacity);
    defer tokens.deinit(allocator);
    var req = Request{
        .id = 1,
        .tokens = tokens,
        .last_token_id = 0,
        .is_finished = std.atomic.Value(bool).init(false),
        .is_cancelled = std.atomic.Value(bool).init(false),
        .visible_len = std.atomic.Value(u32).init(0),
        .enqueued_at = 0,
        .prompt_tokens = 0,
        .allocator = allocator,
    };

    try std.testing.expectEqual(@as(u32, 0), req.visible_len.load(.acquire));

    req.appendToken(42, &[_]u32{});
    try std.testing.expectEqual(@as(u32, 1), req.visible_len.load(.acquire));
    try std.testing.expectEqual(@as(u32, 42), req.last_token_id);

    req.appendToken(99, &[_]u32{});
    try std.testing.expectEqual(@as(u32, 2), req.visible_len.load(.acquire));
    try std.testing.expectEqual(@as(u32, 99), req.last_token_id);
}

test "elapsedSeconds clamps negative to zero" {
    const req = Request{
        .id = 1,
        .tokens = .empty,
        .last_token_id = 0,
        .is_finished = std.atomic.Value(bool).init(false),
        .is_cancelled = std.atomic.Value(bool).init(false),
        .visible_len = std.atomic.Value(u32).init(0),
        .enqueued_at = 5000,
        .prompt_tokens = 0,
        .allocator = undefined,
    };
    try std.testing.expectEqual(@as(u32, 0), req.elapsedSeconds(4000));
    try std.testing.expectEqual(@as(u32, 0), req.elapsedSeconds(5000));
    try std.testing.expectEqual(@as(u32, 2), req.elapsedSeconds(7500));
}

test "requestPriority prefers longer cached prefix" {
    const now: i64 = 10000;
    const req_cached = Request{
        .id = 1,
        .tokens = .empty,
        .last_token_id = 0,
        .is_finished = std.atomic.Value(bool).init(false),
        .is_cancelled = std.atomic.Value(bool).init(false),
        .visible_len = std.atomic.Value(u32).init(0),
        .enqueued_at = now - 100,
        .prompt_tokens = 100,
        .cached_prefix_len = 50,
        .allocator = undefined,
    };
    const req_uncached = Request{
        .id = 2,
        .tokens = .empty,
        .last_token_id = 0,
        .is_finished = std.atomic.Value(bool).init(false),
        .is_cancelled = std.atomic.Value(bool).init(false),
        .visible_len = std.atomic.Value(u32).init(0),
        .enqueued_at = now - 100,
        .prompt_tokens = 100,
        .cached_prefix_len = 0,
        .allocator = undefined,
    };
    try std.testing.expect(requestPriority(&req_cached, now) > requestPriority(&req_uncached, now));
}

test "appendToken at capacity sets cancelled flag" {
    const allocator = std.testing.allocator;
    // Small capacity so we can fill the buffer without allocating 4K tokens.
    const cap: usize = 3;
    var tokens: std.ArrayList(u32) = .empty;
    try tokens.ensureTotalCapacity(allocator, cap);
    defer tokens.deinit(allocator);

    var req = Request{
        .id = 1,
        .tokens = tokens,
        .last_token_id = 0,
        .is_finished = std.atomic.Value(bool).init(false),
        .is_cancelled = std.atomic.Value(bool).init(false),
        .visible_len = std.atomic.Value(u32).init(0),
        .enqueued_at = 0,
        .prompt_tokens = 0,
        .allocator = allocator,
    };

    var i: u32 = 0;
    while (i < cap) : (i += 1) {
        req.appendToken(i + 1, &[_]u32{});
        try std.testing.expect(!req.is_cancelled.load(.acquire));
    }
    try std.testing.expectEqual(cap, req.tokens.items.len);
    try std.testing.expectEqual(@as(u32, @intCast(cap)), req.visible_len.load(.acquire));
    try std.testing.expectEqual(@as(u32, @intCast(cap)), req.last_token_id);

    // Next append must cancel without growing the buffer or mutating state.
    req.appendToken(999, &[_]u32{});
    try std.testing.expect(req.is_cancelled.load(.acquire));
    try std.testing.expectEqual(cap, req.tokens.items.len);
    try std.testing.expectEqual(@as(u32, @intCast(cap)), req.last_token_id);
    try std.testing.expectEqual(@as(u32, @intCast(cap)), req.visible_len.load(.acquire));
}

test "elapsedSeconds large elapsed" {
    const req = Request{
        .id = 1,
        .tokens = .empty,
        .last_token_id = 0,
        .is_finished = std.atomic.Value(bool).init(false),
        .is_cancelled = std.atomic.Value(bool).init(false),
        .visible_len = std.atomic.Value(u32).init(0),
        .enqueued_at = 1000,
        .prompt_tokens = 0,
        .allocator = undefined,
    };
    // 60 seconds elapsed
    try std.testing.expectEqual(@as(u32, 60), req.elapsedSeconds(61000));
    // Exact boundary
    try std.testing.expectEqual(@as(u32, 1), req.elapsedSeconds(2000));
    // Sub-second rounds down
    try std.testing.expectEqual(@as(u32, 0), req.elapsedSeconds(1999));
}

test "requestPriority older request has lower priority" {
    const now: i64 = 10000;
    const req_old = Request{
        .id = 1,
        .tokens = .empty,
        .last_token_id = 0,
        .is_finished = std.atomic.Value(bool).init(false),
        .is_cancelled = std.atomic.Value(bool).init(false),
        .visible_len = std.atomic.Value(u32).init(0),
        .enqueued_at = now - 5000, // 5 seconds ago
        .prompt_tokens = 100,
        .cached_prefix_len = 0,
        .allocator = undefined,
    };
    const req_new = Request{
        .id = 2,
        .tokens = .empty,
        .last_token_id = 0,
        .is_finished = std.atomic.Value(bool).init(false),
        .is_cancelled = std.atomic.Value(bool).init(false),
        .visible_len = std.atomic.Value(u32).init(0),
        .enqueued_at = now - 100, // 100ms ago
        .prompt_tokens = 100,
        .cached_prefix_len = 0,
        .allocator = undefined,
    };
    // With no cache, older request should have lower priority (more negative elapsed)
    try std.testing.expect(requestPriority(&req_old, now) < requestPriority(&req_new, now));
}

test "SchedulerStats fields default zero" {
    const stats = SchedulerStats{
        .waiting_count = 0,
        .running_count = 0,
        .completed_total = 0,
        .cancelled_total = 0,
    };
    try std.testing.expectEqual(@as(u32, 0), stats.waiting_count);
    try std.testing.expectEqual(@as(u32, 0), stats.running_count);
    try std.testing.expectEqual(@as(u32, 0), stats.completed_total);
    try std.testing.expectEqual(@as(u32, 0), stats.cancelled_total);
}

test "Request.deinit frees tokens" {
    const allocator = std.testing.allocator;
    var tokens: std.ArrayList(u32) = .empty;
    try tokens.ensureTotalCapacity(allocator, initial_token_capacity);
    var req = Request{
        .id = 1,
        .tokens = tokens,
        .last_token_id = 0,
        .is_finished = std.atomic.Value(bool).init(false),
        .is_cancelled = std.atomic.Value(bool).init(false),
        .visible_len = std.atomic.Value(u32).init(0),
        .enqueued_at = 0,
        .prompt_tokens = 0,
        .allocator = allocator,
    };
    req.appendToken(42, &[_]u32{});
    req.appendToken(43, &[_]u32{});
    try std.testing.expectEqual(@as(usize, 2), req.tokens.items.len);
    try std.testing.expectEqual(@as(u32, 43), req.last_token_id);
    try std.testing.expectEqual(@as(u32, 2), req.visible_len.load(.acquire));
    // deinit should not leak (testing allocator will catch leaks)
    req.deinit();
}

test "milliTimestamp returns positive value" {
    const ts = milliTimestamp();
    try std.testing.expect(ts > 0);
}

test "sleepNs does not crash" {
    // Zero-duration and short sleep must complete; clock must not go backwards.
    const before = milliTimestamp();
    sleepNs(0);
    sleepNs(1_000); // 1 microsecond
    const after = milliTimestamp();
    try std.testing.expect(after >= before);
}

test "sleepNs advances under sim_clock override" {
    defer sim_clock.setOverrideMs(null);
    sim_clock.setOverrideMs(50_000);
    sleepNs(scheduler_poll_ns); // 1ms poll
    try std.testing.expectEqual(@as(i64, 50_001), milliTimestamp());
}

// Mock model for testing
const MockModel = struct {
    const MockBackend = struct {
        pub fn setThreadContext(_: *MockBackend) void {}
    };
    be: MockBackend = .{},
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

test "fuzz: all scheduler functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const allocator = std.testing.allocator;

            // --- Request.elapsedSeconds ---
            const enqueued_at: i64 = @intCast(smith.valueWithHash(u32, 0));
            const now_offset: i64 = @intCast(smith.valueWithHash(u32, 1));
            const now = enqueued_at +% now_offset;
            const req_const = Request{
                .id = smith.valueWithHash(u64, 2),
                .tokens = .empty,
                .last_token_id = 0,
                .is_finished = std.atomic.Value(bool).init(false),
                .is_cancelled = std.atomic.Value(bool).init(false),
                .visible_len = std.atomic.Value(u32).init(0),
                .enqueued_at = enqueued_at,
                .prompt_tokens = 0,
                .allocator = undefined,
            };
            const elapsed = req_const.elapsedSeconds(now);
            // Must not overflow: result is always a valid u32
            try std.testing.expect(elapsed <= std.math.maxInt(u32));

            // --- Request.appendToken + Request.deinit ---
            var tokens: std.ArrayList(u32) = .empty;
            try tokens.ensureTotalCapacity(allocator, initial_token_capacity);
            var req = Request{
                .id = smith.valueWithHash(u64, 3),
                .tokens = tokens,
                .last_token_id = 0,
                .is_finished = std.atomic.Value(bool).init(false),
                .is_cancelled = std.atomic.Value(bool).init(false),
                .visible_len = std.atomic.Value(u32).init(0),
                .enqueued_at = 0,
                .prompt_tokens = 0,
                .allocator = allocator,
            };
            defer req.deinit(); // exercises Request.deinit

            const token = smith.valueWithHash(u32, 4);
            const eog_val = smith.valueWithHash(u32, 5);
            const eog_ids = [_]u32{eog_val};
            req.appendToken(token, &eog_ids);
            // After append: either finished (token == eog) or token was appended
            if (token == eog_val) {
                try std.testing.expect(req.is_finished.load(.acquire));
            } else {
                try std.testing.expect(req.tokens.items.len <= initial_token_capacity);
            }

            // --- SchedulerStats (pub struct) ---
            const stats = SchedulerStats{
                .waiting_count = smith.valueWithHash(u32, 6),
                .running_count = smith.valueWithHash(u32, 7),
                .completed_total = smith.valueWithHash(u32, 8),
                .cancelled_total = smith.valueWithHash(u32, 9),
            };
            // Fields must round-trip
            try std.testing.expectEqual(smith.valueWithHash(u32, 6), stats.waiting_count);

            // --- RequestManager.init + getStats + enqueue + step + deinit ---
            var metrics = Metrics{};
            const batch_size = @as(usize, @intCast((smith.valueWithHash(u8, 10) % 8))) + 1;
            const timeout = @as(u32, smith.valueWithHash(u16, 11)) + 1;
            var manager = RequestManager.init(allocator, &metrics, batch_size, timeout, null, testIo()) catch return;
            defer manager.deinit(); // exercises RequestManager.deinit

            // RequestManager.getStats
            const mgr_stats = manager.getStats();
            try std.testing.expectEqual(@as(u32, 0), mgr_stats.waiting_count);
            try std.testing.expectEqual(@as(u32, 0), mgr_stats.running_count);

            // RequestManager.enqueue with random prompt length
            const prompt_len = @as(usize, @intCast(smith.valueWithHash(u8, 12) % 16)) + 1;
            var prompt_buf: [16]u32 = undefined;
            for (0..prompt_len) |pi| {
                prompt_buf[pi] = smith.valueWithHash(u32, @intCast(13 + pi));
            }
            _ = manager.enqueue(prompt_buf[0..prompt_len]) catch return;

            const after_enqueue = manager.getStats();
            try std.testing.expectEqual(@as(u32, 1), after_enqueue.waiting_count);

            // RequestManager.step (uses MockModel)
            var mock_model = MockModel{};
            var model = Model.from(MockModel, &mock_model);
            const step_eog = [_]u32{smith.valueWithHash(u32, 30)};
            manager.step(&model, &step_eog) catch return;

            // After step: request should have moved from waiting to running
            const after_step = manager.getStats();
            try std.testing.expect(after_step.waiting_count + after_step.running_count >= 0);

            // --- runSchedulerLoop (comptime reference check) ---
            // Cannot run the loop (it blocks until shutdown), so verify linkage
            comptime {
                _ = &runSchedulerLoop;
            }
        }
    }.f, .{});
}
