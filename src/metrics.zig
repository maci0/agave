//! Prometheus metrics collector for server observability.
//!
//! Tracks request throughput, latency histogram, queue depth, and KV cache usage.
//! All operations are lock-free using atomic operations.

const std = @import("std");

/// Prometheus metrics collector with atomic counters and gauges.
pub const Metrics = struct {
    // Counters (monotonically increasing)
    requests_total: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    requests_completed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    requests_cancelled: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    tokens_generated_total: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

    // Gauges (current value)
    queue_depth: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    active_requests: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    kv_blocks_used: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    kv_blocks_total: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),

    // Histogram buckets for latency (milliseconds)
    latency_10ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_50ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_100ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_500ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_1s: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_5s: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_10s: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_30s: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_inf: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_sum: std.atomic.Value(u64) = std.atomic.Value(u64).init(0), // sum of all latencies (for avg)

    // Prefix cache metrics
    kv_cache_hits: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    kv_cache_misses: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    prefix_tokens_reused: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    prefix_tokens_total: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

    /// Increment total request counter.
    pub fn recordRequest(self: *Metrics) void {
        _ = self.requests_total.fetchAdd(1, .monotonic);
    }

    /// Increment completed request counter.
    pub fn recordCompletion(self: *Metrics) void {
        _ = self.requests_completed.fetchAdd(1, .monotonic);
    }

    /// Increment cancelled request counter.
    pub fn recordCancellation(self: *Metrics) void {
        _ = self.requests_cancelled.fetchAdd(1, .monotonic);
    }

    /// Increment tokens generated counter.
    pub fn recordTokens(self: *Metrics, count: u32) void {
        _ = self.tokens_generated_total.fetchAdd(count, .monotonic);
    }

    /// Record request latency and update histogram buckets.
    pub fn recordLatency(self: *Metrics, duration_ms: u64) void {
        _ = self.latency_sum.fetchAdd(duration_ms, .monotonic);

        if (duration_ms <= 10) {
            _ = self.latency_10ms.fetchAdd(1, .monotonic);
        } else if (duration_ms <= 50) {
            _ = self.latency_50ms.fetchAdd(1, .monotonic);
        } else if (duration_ms <= 100) {
            _ = self.latency_100ms.fetchAdd(1, .monotonic);
        } else if (duration_ms <= 500) {
            _ = self.latency_500ms.fetchAdd(1, .monotonic);
        } else if (duration_ms <= 1000) {
            _ = self.latency_1s.fetchAdd(1, .monotonic);
        } else if (duration_ms <= 5000) {
            _ = self.latency_5s.fetchAdd(1, .monotonic);
        } else if (duration_ms <= 10000) {
            _ = self.latency_10s.fetchAdd(1, .monotonic);
        } else if (duration_ms <= 30000) {
            _ = self.latency_30s.fetchAdd(1, .monotonic);
        } else {
            _ = self.latency_inf.fetchAdd(1, .monotonic);
        }
    }

    /// Update current queue depth gauge.
    pub fn updateQueueDepth(self: *Metrics, depth: u32) void {
        self.queue_depth.store(depth, .monotonic);
    }

    /// Update active requests gauge.
    pub fn updateActiveRequests(self: *Metrics, count: u32) void {
        self.active_requests.store(count, .monotonic);
    }

    /// Update KV cache block usage.
    pub fn updateKvBlocks(self: *Metrics, used: u32, total: u32) void {
        self.kv_blocks_used.store(used, .monotonic);
        self.kv_blocks_total.store(total, .monotonic);
    }

    /// Record a cache hit (prefix match found in RadixTree).
    pub fn recordCacheHit(self: *Metrics, prefix_len: u32) void {
        _ = self.kv_cache_hits.fetchAdd(1, .monotonic);
        _ = self.prefix_tokens_reused.fetchAdd(prefix_len, .monotonic);
    }

    /// Record a cache miss (no prefix match found).
    pub fn recordCacheMiss(self: *Metrics, total_len: u32) void {
        _ = self.kv_cache_misses.fetchAdd(1, .monotonic);
        _ = self.prefix_tokens_total.fetchAdd(total_len, .monotonic);
    }

    /// Render metrics in Prometheus text format.
    pub fn renderPrometheus(self: *const Metrics, writer: anytype) !void {
        // Counters
        try writer.writeAll("# HELP agave_requests_total Total HTTP requests received\n");
        try writer.writeAll("# TYPE agave_requests_total counter\n");
        try writer.print("agave_requests_total {d}\n", .{self.requests_total.load(.monotonic)});

        try writer.writeAll("# HELP agave_requests_completed Total requests completed successfully\n");
        try writer.writeAll("# TYPE agave_requests_completed counter\n");
        try writer.print("agave_requests_completed {d}\n", .{self.requests_completed.load(.monotonic)});

        try writer.writeAll("# HELP agave_requests_cancelled Total requests cancelled by timeout\n");
        try writer.writeAll("# TYPE agave_requests_cancelled counter\n");
        try writer.print("agave_requests_cancelled {d}\n", .{self.requests_cancelled.load(.monotonic)});

        try writer.writeAll("# HELP agave_tokens_generated_total Total tokens generated\n");
        try writer.writeAll("# TYPE agave_tokens_generated_total counter\n");
        try writer.print("agave_tokens_generated_total {d}\n", .{self.tokens_generated_total.load(.monotonic)});

        // Gauges
        try writer.writeAll("# HELP agave_queue_depth Current request queue depth\n");
        try writer.writeAll("# TYPE agave_queue_depth gauge\n");
        try writer.print("agave_queue_depth {d}\n", .{self.queue_depth.load(.monotonic)});

        try writer.writeAll("# HELP agave_active_requests Currently running requests\n");
        try writer.writeAll("# TYPE agave_active_requests gauge\n");
        try writer.print("agave_active_requests {d}\n", .{self.active_requests.load(.monotonic)});

        try writer.writeAll("# HELP agave_kv_blocks_used KV cache blocks in use\n");
        try writer.writeAll("# TYPE agave_kv_blocks_used gauge\n");
        try writer.print("agave_kv_blocks_used {d}\n", .{self.kv_blocks_used.load(.monotonic)});

        try writer.writeAll("# HELP agave_kv_blocks_total Total KV cache blocks available\n");
        try writer.writeAll("# TYPE agave_kv_blocks_total gauge\n");
        try writer.print("agave_kv_blocks_total {d}\n", .{self.kv_blocks_total.load(.monotonic)});

        // Histogram
        try writer.writeAll("# HELP agave_request_duration_seconds Request latency histogram\n");
        try writer.writeAll("# TYPE agave_request_duration_seconds histogram\n");
        try writer.print("agave_request_duration_seconds_bucket{{le=\"0.01\"}} {d}\n", .{self.latency_10ms.load(.monotonic)});
        try writer.print("agave_request_duration_seconds_bucket{{le=\"0.05\"}} {d}\n", .{self.latency_50ms.load(.monotonic)});
        try writer.print("agave_request_duration_seconds_bucket{{le=\"0.1\"}} {d}\n", .{self.latency_100ms.load(.monotonic)});
        try writer.print("agave_request_duration_seconds_bucket{{le=\"0.5\"}} {d}\n", .{self.latency_500ms.load(.monotonic)});
        try writer.print("agave_request_duration_seconds_bucket{{le=\"1\"}} {d}\n", .{self.latency_1s.load(.monotonic)});
        try writer.print("agave_request_duration_seconds_bucket{{le=\"5\"}} {d}\n", .{self.latency_5s.load(.monotonic)});
        try writer.print("agave_request_duration_seconds_bucket{{le=\"10\"}} {d}\n", .{self.latency_10s.load(.monotonic)});
        try writer.print("agave_request_duration_seconds_bucket{{le=\"30\"}} {d}\n", .{self.latency_30s.load(.monotonic)});
        try writer.print("agave_request_duration_seconds_bucket{{le=\"+Inf\"}} {d}\n", .{self.latency_inf.load(.monotonic)});
        try writer.print("agave_request_duration_seconds_sum {d}\n", .{self.latency_sum.load(.monotonic)});
        try writer.print("agave_request_duration_seconds_count {d}\n", .{self.requests_completed.load(.monotonic)});

        // Prefix cache metrics
        const hits = self.kv_cache_hits.load(.monotonic);
        const misses = self.kv_cache_misses.load(.monotonic);
        const total = hits + misses;
        const hit_rate: f64 = if (total > 0) @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(total)) else 0.0;

        try writer.writeAll("# HELP agave_kv_cache_hit_rate Ratio of cache hits to total requests\n");
        try writer.writeAll("# TYPE agave_kv_cache_hit_rate gauge\n");
        try writer.print("agave_kv_cache_hit_rate {d:.4}\n", .{hit_rate});

        const reused = self.prefix_tokens_reused.load(.monotonic);
        const total_tokens = self.prefix_tokens_total.load(.monotonic);
        const reuse_ratio: f64 = if (total_tokens > 0) @as(f64, @floatFromInt(reused)) / @as(f64, @floatFromInt(total_tokens)) else 0.0;

        try writer.writeAll("# HELP agave_prefix_reuse_ratio Fraction of tokens served from cache\n");
        try writer.writeAll("# TYPE agave_prefix_reuse_ratio gauge\n");
        try writer.print("agave_prefix_reuse_ratio {d:.4}\n", .{reuse_ratio});
    }
};

const test_render_buf_size: usize = 8192;

// Tests
test "Metrics: recordRequest increments counter" {
    var metrics = Metrics{};

    try std.testing.expectEqual(@as(u64, 0), metrics.requests_total.load(.monotonic));

    metrics.recordRequest();
    try std.testing.expectEqual(@as(u64, 1), metrics.requests_total.load(.monotonic));

    metrics.recordRequest();
    metrics.recordRequest();
    try std.testing.expectEqual(@as(u64, 3), metrics.requests_total.load(.monotonic));
}

test "Metrics: recordLatency increments correct bucket" {
    var metrics = Metrics{};

    // Test 250ms → should go into latency_500ms bucket
    metrics.recordLatency(250);
    try std.testing.expectEqual(@as(u64, 0), metrics.latency_100ms.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 1), metrics.latency_500ms.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 250), metrics.latency_sum.load(.monotonic));

    // Test 10ms → should go into latency_10ms bucket
    metrics.recordLatency(10);
    try std.testing.expectEqual(@as(u64, 1), metrics.latency_10ms.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 260), metrics.latency_sum.load(.monotonic));

    // Test 5000ms → should go into latency_5s bucket
    metrics.recordLatency(5000);
    try std.testing.expectEqual(@as(u64, 1), metrics.latency_5s.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 5260), metrics.latency_sum.load(.monotonic));
}

test "Metrics: renderPrometheus outputs valid format" {
    var metrics = Metrics{};

    // Record some data
    metrics.recordRequest();
    metrics.recordRequest();
    metrics.recordCompletion();
    metrics.recordTokens(42);
    metrics.recordLatency(250);
    metrics.updateQueueDepth(5);

    // Render to buffer
    var buf: [test_render_buf_size]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    const writer = fbs.writer();

    try metrics.renderPrometheus(writer);

    const output = fbs.getWritten();

    // Verify format
    try std.testing.expect(std.mem.indexOf(u8, output, "# HELP agave_requests_total") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "# TYPE agave_requests_total counter") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "agave_requests_total 2") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "agave_tokens_generated_total 42") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "agave_queue_depth 5") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "agave_request_duration_seconds_bucket{le=\"0.5\"} 1") != null);
}
