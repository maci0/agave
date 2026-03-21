//! Continuous batching scheduler for multi-tenant LLM serving.
//!
//! Implements vLLM-style iteration-level continuous batching: maintains a waiting
//! queue and running list, processes one decode step across all active requests,
//! ejects finished/cancelled requests, and fills batch from waiting queue (FIFO).

const std = @import("std");
const Model = @import("models/model.zig").Model;
const Allocator = std.mem.Allocator;

/// Per-request state for continuous batching.
pub const Request = struct {
    id: u64,
    tokens: std.ArrayList(u32),
    last_token_id: u32,
    is_finished: bool,
    is_cancelled: std.atomic.Value(bool),
    enqueued_at: i64,
    prompt_tokens: u32,
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
    max_batch_size: usize,
    timeout_sec: u32,
    allocator: Allocator,
    mutex: std.Thread.Mutex,
    next_id: std.atomic.Value(u64),

    /// Initialize request manager.
    pub fn init(allocator: Allocator, max_batch_size: usize, timeout_sec: u32) RequestManager {
        return .{
            .waiting = .{},
            .running = .{},
            .max_batch_size = max_batch_size,
            .timeout_sec = timeout_sec,
            .allocator = allocator,
            .mutex = .{},
            .next_id = std.atomic.Value(u64).init(1),
        };
    }

    /// Clean up allocated resources.
    /// Note: does NOT free individual requests — caller owns them.
    pub fn deinit(self: *RequestManager) void {
        self.waiting.deinit(self.allocator);
        self.running.deinit(self.allocator);
    }

    /// Enqueue a new request into the waiting queue.
    /// Returns pointer to the request (caller keeps reference for polling).
    pub fn enqueue(self: *RequestManager, prompt_tokens: u32) !*Request {
        const req = try self.allocator.create(Request);
        errdefer self.allocator.destroy(req);

        const now = std.time.milliTimestamp();
        const id = self.next_id.fetchAdd(1, .monotonic);

        req.* = .{
            .id = id,
            .tokens = .{},
            .last_token_id = 0,
            .is_finished = false,
            .is_cancelled = std.atomic.Value(bool).init(false),
            .enqueued_at = now,
            .prompt_tokens = prompt_tokens,
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
            if (req.is_finished or req.is_cancelled.load(.monotonic)) {
                _ = self.running.swapRemove(i);
                // Note: caller owns request memory, we just remove from list
                // Don't advance i — swapRemove puts last element at current index
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

        // 3. Fill batch from waiting queue (FIFO)
        while (self.running.items.len < self.max_batch_size and self.waiting.items.len > 0) {
            const req = self.waiting.orderedRemove(0);
            try self.running.append(self.allocator, req);
        }

        self.mutex.unlock();

        // 4. Execute forward for all running requests (unlocked — model owns its concurrency)
        for (self.running.items) |req| {
            if (req.is_cancelled.load(.monotonic)) continue;

            const next_token = model.forward(req.last_token_id) catch |err| {
                std.log.err("Request {d} forward failed: {}", .{ req.id, err });
                req.is_cancelled.store(true, .monotonic);
                continue;
            };

            req.appendToken(next_token, eog_ids);
        }
    }

    /// Get current scheduler statistics.
    pub fn getStats(self: *RequestManager) SchedulerStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        return .{
            .waiting_count = @intCast(self.waiting.items.len),
            .running_count = @intCast(self.running.items.len),
            .completed_total = 0, // TODO: track in real implementation
            .cancelled_total = 0, // TODO: track in real implementation
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
        std.time.sleep(1_000_000); // 1ms between iterations
    }
}

// Unit tests
test "enqueue increments waiting count" {
    const allocator = std.testing.allocator;
    var manager = RequestManager.init(allocator, 4, 30);
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
    var manager = RequestManager.init(allocator, 2, 30);
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
    var manager = RequestManager.init(allocator, 2, 30);
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
    var manager = RequestManager.init(allocator, 2, 1); // 1 second timeout
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
};
