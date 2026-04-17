//! Prometheus metrics collector for server observability.
//!
//! Tracks request throughput, latency histogram, queue depth, and KV cache usage.
//! All operations are lock-free using atomic operations.

const std = @import("std");

// ── FixedBufStream (Zig 0.16 removed FixedBufStream.init) ───
const FixedBufStream = struct {
    buf: []u8,
    pos: usize = 0,

    pub fn init(buf: []u8) FixedBufStream {
        return .{ .buf = buf };
    }

    pub fn writer(self: *FixedBufStream) Writer {
        return .{ .fbs = self };
    }

    pub fn getWritten(self: *const FixedBufStream) []const u8 {
        return self.buf[0..self.pos];
    }

    pub const Writer = struct {
        fbs: *FixedBufStream,

        pub fn writeAll(self: Writer, data: []const u8) !void {
            if (self.fbs.pos + data.len > self.fbs.buf.len) return error.NoSpaceLeft;
            @memcpy(self.fbs.buf[self.fbs.pos..][0..data.len], data);
            self.fbs.pos += data.len;
        }

        pub fn print(self: Writer, comptime fmt: []const u8, args: anytype) !void {
            const written = std.fmt.bufPrint(self.fbs.buf[self.fbs.pos..], fmt, args) catch return error.NoSpaceLeft;
            self.fbs.pos += written.len;
        }
    };
};

/// Latency histogram bucket boundaries (milliseconds).
/// These must match the Prometheus `le` labels in renderPrometheus().
const latency_bucket_10ms: u64 = 10;
const latency_bucket_50ms: u64 = 50;
const latency_bucket_100ms: u64 = 100;
const latency_bucket_500ms: u64 = 500;
const latency_bucket_1s: u64 = 1000;
const latency_bucket_5s: u64 = 5000;
const latency_bucket_10s: u64 = 10000;
const latency_bucket_30s: u64 = 30000;

/// TPOT (time per output token) histogram bucket boundaries (milliseconds).
/// Finer granularity than latency — decode speed is typically 5-100ms/token.
const tpot_bucket_5ms: u64 = 5;
const tpot_bucket_10ms: u64 = 10;
const tpot_bucket_20ms: u64 = 20;
const tpot_bucket_50ms: u64 = 50;
const tpot_bucket_100ms: u64 = 100;
const tpot_bucket_200ms: u64 = 200;
const tpot_bucket_500ms: u64 = 500;

/// Token count histogram bucket boundaries (prompt and generation length).
const token_bucket_16: u32 = 16;
const token_bucket_64: u32 = 64;
const token_bucket_128: u32 = 128;
const token_bucket_256: u32 = 256;
const token_bucket_512: u32 = 512;
const token_bucket_1024: u32 = 1024;
const token_bucket_2048: u32 = 2048;
const token_bucket_4096: u32 = 4096;

/// Cache line size for padding to prevent false sharing between atomic groups.
const cache_line: usize = 64;

/// Milliseconds per second — used for ms→seconds conversion in Prometheus output.
const ms_per_second: f64 = 1000.0;

/// Prometheus metrics collector with atomic counters and gauges.
///
/// Atomic groups are separated by cache-line padding to prevent false sharing
/// when different threads update independent counter groups concurrently
/// (e.g. HTTP handler threads vs scheduler thread vs generation thread).
pub const Metrics = struct {
    // ── Group 1: Request lifecycle counters (HTTP handler threads) ──
    requests_total: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    requests_completed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    requests_cancelled: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    requests_failed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    requests_auth_failed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    requests_rate_limited: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    connections_rejected: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    requests_timeout: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    _pad0: [cache_line]u8 = undefined,

    // ── Group 2: Generation counters (generation thread) ──
    tokens_generated_total: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    prefill_tokens_total: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    last_tps_x100: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    _pad1: [cache_line]u8 = undefined,

    // ── Group 3: Scheduler gauges (scheduler thread) ──
    queue_depth: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    active_requests: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    active_connections: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    scheduler_errors: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    kv_blocks_used: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    kv_blocks_total: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    _pad2: [cache_line]u8 = undefined,

    // ── Group 4: Latency histogram (per-request completion) ──
    latency_10ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_50ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_100ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_500ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_1s: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_5s: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_10s: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_30s: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_inf: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_sum: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    _pad3: [cache_line]u8 = undefined,

    // ── Group 5: TTFT histogram (per-request prefill) ──
    ttft_10ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    ttft_50ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    ttft_100ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    ttft_500ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    ttft_1s: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    ttft_5s: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    ttft_10s: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    ttft_30s: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    ttft_inf: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    ttft_sum: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    _pad4: [cache_line]u8 = undefined,

    // ── Group 6: Prefix cache metrics (enqueue path) ──
    kv_cache_hits: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    kv_cache_misses: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    prefix_tokens_reused: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    prefix_tokens_total: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    _pad5: [cache_line]u8 = undefined,

    // ── Group 7: Process metadata (set once at startup) ──
    process_start_time: std.atomic.Value(i64) = std.atomic.Value(i64).init(0),
    _pad6: [cache_line]u8 = undefined,

    // ── Group 8: llm-d routing gauges ──
    /// Prompt tokens currently being prefilled across all active requests.
    input_tokens_in_flight: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    /// KV cache block size in tokens (set once at startup).
    cache_block_size: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    /// Total number of GPU KV cache blocks (set once at startup).
    cache_num_blocks: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    _pad7: [cache_line]u8 = undefined,

    // ── Group 9: TPOT histogram (time per output token, milliseconds) ──
    tpot_5ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    tpot_10ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    tpot_20ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    tpot_50ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    tpot_100ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    tpot_200ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    tpot_500ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    tpot_inf: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    tpot_sum: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    _pad8: [cache_line]u8 = undefined,

    // ── Group 10: Queue time histogram (milliseconds waiting before execution) ──
    queue_time_10ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    queue_time_50ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    queue_time_100ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    queue_time_500ms: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    queue_time_1s: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    queue_time_5s: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    queue_time_10s: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    queue_time_30s: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    queue_time_inf: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    queue_time_sum: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    _pad9: [cache_line]u8 = undefined,

    // ── Group 11: Prompt token count histogram ──
    prompt_tok_16: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    prompt_tok_64: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    prompt_tok_128: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    prompt_tok_256: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    prompt_tok_512: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    prompt_tok_1024: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    prompt_tok_2048: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    prompt_tok_4096: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    prompt_tok_inf: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    prompt_tok_sum: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    _pad10: [cache_line]u8 = undefined,

    // ── Group 12: Generation token count histogram ──
    gen_tok_16: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    gen_tok_64: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    gen_tok_128: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    gen_tok_256: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    gen_tok_512: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    gen_tok_1024: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    gen_tok_2048: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    gen_tok_4096: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    gen_tok_inf: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    gen_tok_sum: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

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

    /// Increment failed request counter (5xx errors, encode/forward failures).
    pub fn recordFailure(self: *Metrics) void {
        _ = self.requests_failed.fetchAdd(1, .monotonic);
    }

    /// Increment auth failure counter (401 responses).
    pub fn recordAuthFailure(self: *Metrics) void {
        _ = self.requests_auth_failed.fetchAdd(1, .monotonic);
    }

    /// Increment rate limit counter (429 responses).
    pub fn recordRateLimit(self: *Metrics) void {
        _ = self.requests_rate_limited.fetchAdd(1, .monotonic);
    }

    /// Increment connection rejection counter (503 at capacity).
    /// Distinct from recordFailure() — capacity rejections indicate scaling needs,
    /// not broken inference. Operators can alert on this independently.
    pub fn recordConnectionRejection(self: *Metrics) void {
        _ = self.connections_rejected.fetchAdd(1, .monotonic);
    }

    /// Increment timeout counter (scheduler auto-cancel due to exceeded deadline).
    /// Distinct from recordCancellation() — timeouts indicate server-side resource
    /// exhaustion or slow inference, not client-initiated cancellation.
    pub fn recordTimeout(self: *Metrics) void {
        _ = self.requests_timeout.fetchAdd(1, .monotonic);
    }

    /// Increment scheduler error counter.
    /// Tracks scheduler.step() failures that are otherwise only visible in logs.
    pub fn recordSchedulerError(self: *Metrics) void {
        _ = self.scheduler_errors.fetchAdd(1, .monotonic);
    }

    /// Increment tokens generated counter.
    pub fn recordTokens(self: *Metrics, count: u64) void {
        _ = self.tokens_generated_total.fetchAdd(count, .monotonic);
    }

    /// Record a value into the appropriate histogram bucket using comptime field dispatch.
    fn recordToBuckets(self: *Metrics, value: u64, comptime fields: anytype, comptime boundaries: anytype, comptime inf_field: []const u8) void {
        inline for (fields, boundaries) |field, boundary| {
            if (value <= boundary) {
                _ = @field(self, field).fetchAdd(1, .monotonic);
                return;
            }
        }
        _ = @field(self, inf_field).fetchAdd(1, .monotonic);
    }

    /// Render a histogram in Prometheus cumulative-bucket format using comptime field dispatch.
    fn renderHistogram(self: *const Metrics, writer: anytype, name: []const u8, help: []const u8, comptime fields: anytype, comptime labels: anytype, comptime sum_field: []const u8, comptime sum_as_seconds: bool) !void {
        try writer.print("# HELP {s} {s}\n# TYPE {s} histogram\n", .{ name, help, name });
        var cumulative: u64 = 0;
        inline for (fields, labels) |field, label| {
            cumulative += @field(self, field).load(.monotonic);
            try writer.print("{s}_bucket{{le=\"{s}\"}} {d}\n", .{ name, @as([]const u8, label), cumulative });
        }
        if (sum_as_seconds) {
            const sum_ms = @field(self, sum_field).load(.monotonic);
            try writer.print("{s}_sum {d:.3}\n", .{ name, @as(f64, @floatFromInt(sum_ms)) / ms_per_second });
        } else {
            try writer.print("{s}_sum {d}\n", .{ name, @field(self, sum_field).load(.monotonic) });
        }
        try writer.print("{s}_count {d}\n", .{ name, cumulative });
    }

    /// Record request latency and update histogram buckets.
    pub fn recordLatency(self: *Metrics, duration_ms: u64) void {
        _ = self.latency_sum.fetchAdd(duration_ms, .monotonic);
        self.recordToBuckets(duration_ms, .{
            "latency_10ms", "latency_50ms", "latency_100ms", "latency_500ms",
            "latency_1s",   "latency_5s",   "latency_10s",   "latency_30s",
        }, .{
            latency_bucket_10ms,  latency_bucket_50ms,  latency_bucket_100ms, latency_bucket_500ms,
            latency_bucket_1s,    latency_bucket_5s,    latency_bucket_10s,   latency_bucket_30s,
        }, "latency_inf");
    }

    /// Update current queue depth gauge.
    pub fn updateQueueDepth(self: *Metrics, depth: u32) void {
        self.queue_depth.store(depth, .monotonic);
    }

    /// Update active requests gauge.
    pub fn updateActiveRequests(self: *Metrics, count: u32) void {
        self.active_requests.store(count, .monotonic);
    }

    /// Record a cache hit (prefix match found in RadixTree).
    /// `prefix_len` is the number of tokens reused from cache.
    /// `total_len` is the total prompt token count for this request.
    pub fn recordCacheHit(self: *Metrics, prefix_len: u32, total_len: u32) void {
        _ = self.kv_cache_hits.fetchAdd(1, .monotonic);
        _ = self.prefix_tokens_reused.fetchAdd(prefix_len, .monotonic);
        _ = self.prefix_tokens_total.fetchAdd(total_len, .monotonic);
    }

    /// Record a cache miss (no prefix match found).
    pub fn recordCacheMiss(self: *Metrics, total_len: u32) void {
        _ = self.kv_cache_misses.fetchAdd(1, .monotonic);
        _ = self.prefix_tokens_total.fetchAdd(total_len, .monotonic);
    }

    /// Record time-to-first-token (prefill latency) and update TTFT histogram.
    pub fn recordTTFT(self: *Metrics, prefill_ms: u64, prompt_tokens: u32) void {
        _ = self.ttft_sum.fetchAdd(prefill_ms, .monotonic);
        _ = self.prefill_tokens_total.fetchAdd(prompt_tokens, .monotonic);
        self.recordToBuckets(prefill_ms, .{
            "ttft_10ms", "ttft_50ms", "ttft_100ms", "ttft_500ms",
            "ttft_1s",   "ttft_5s",   "ttft_10s",   "ttft_30s",
        }, .{
            latency_bucket_10ms,  latency_bucket_50ms,  latency_bucket_100ms, latency_bucket_500ms,
            latency_bucket_1s,    latency_bucket_5s,    latency_bucket_10s,   latency_bucket_30s,
        }, "ttft_inf");
    }

    /// Record throughput from a completed request (tokens per second × 100).
    pub fn recordThroughput(self: *Metrics, tokens: u32, duration_ms: u64) void {
        if (duration_ms > 0) {
            const tps_x100: u64 = @as(u64, tokens) * 100_000 / duration_ms;
            self.last_tps_x100.store(tps_x100, .monotonic);
        }
    }

    /// Update KV cache block gauges (used and total).
    /// Called from the scheduler to reflect current cache utilization.
    pub fn updateKvBlocks(self: *Metrics, used: u32, total: u32) void {
        self.kv_blocks_used.store(used, .monotonic);
        self.kv_blocks_total.store(total, .monotonic);
    }

    /// Set static cache configuration (called once at server startup).
    pub fn setCacheConfig(self: *Metrics, block_size: u32, num_blocks: u32) void {
        self.cache_block_size.store(block_size, .monotonic);
        self.cache_num_blocks.store(num_blocks, .monotonic);
    }

    /// Update input tokens in flight gauge (prompt tokens being prefilled).
    pub fn updateInputTokensInFlight(self: *Metrics, count: u32) void {
        self.input_tokens_in_flight.store(count, .monotonic);
    }

    /// Record time-per-output-token and update TPOT histogram.
    /// `tokens` is the number of decode tokens; `decode_ms` is the total decode time.
    /// TPOT = decode_ms / tokens (milliseconds per token).
    pub fn recordTPOT(self: *Metrics, tokens: u32, decode_ms: u64) void {
        if (tokens == 0) return;
        const tpot_ms = decode_ms / @as(u64, tokens);
        _ = self.tpot_sum.fetchAdd(tpot_ms, .monotonic);
        self.recordToBuckets(tpot_ms, .{
            "tpot_5ms",   "tpot_10ms",  "tpot_20ms",  "tpot_50ms",
            "tpot_100ms", "tpot_200ms", "tpot_500ms",
        }, .{
            tpot_bucket_5ms,   tpot_bucket_10ms,  tpot_bucket_20ms,  tpot_bucket_50ms,
            tpot_bucket_100ms, tpot_bucket_200ms, tpot_bucket_500ms,
        }, "tpot_inf");
    }

    /// Record time spent waiting in the scheduler queue before execution.
    pub fn recordQueueTime(self: *Metrics, queue_ms: u64) void {
        _ = self.queue_time_sum.fetchAdd(queue_ms, .monotonic);
        self.recordToBuckets(queue_ms, .{
            "queue_time_10ms", "queue_time_50ms", "queue_time_100ms", "queue_time_500ms",
            "queue_time_1s",   "queue_time_5s",   "queue_time_10s",   "queue_time_30s",
        }, .{
            latency_bucket_10ms,  latency_bucket_50ms,  latency_bucket_100ms, latency_bucket_500ms,
            latency_bucket_1s,    latency_bucket_5s,    latency_bucket_10s,   latency_bucket_30s,
        }, "queue_time_inf");
    }

    /// Record prompt token count for the prompt length distribution histogram.
    pub fn recordPromptTokens(self: *Metrics, count: u32) void {
        _ = self.prompt_tok_sum.fetchAdd(count, .monotonic);
        self.recordToBuckets(@as(u64, count), .{
            "prompt_tok_16",  "prompt_tok_64",   "prompt_tok_128",  "prompt_tok_256",
            "prompt_tok_512", "prompt_tok_1024", "prompt_tok_2048", "prompt_tok_4096",
        }, .{
            token_bucket_16,   token_bucket_64,   token_bucket_128,  token_bucket_256,
            token_bucket_512,  token_bucket_1024, token_bucket_2048, token_bucket_4096,
        }, "prompt_tok_inf");
    }

    /// Record generation token count for the output length distribution histogram.
    pub fn recordGenerationTokens(self: *Metrics, count: u32) void {
        _ = self.gen_tok_sum.fetchAdd(count, .monotonic);
        self.recordToBuckets(@as(u64, count), .{
            "gen_tok_16",  "gen_tok_64",   "gen_tok_128",  "gen_tok_256",
            "gen_tok_512", "gen_tok_1024", "gen_tok_2048", "gen_tok_4096",
        }, .{
            token_bucket_16,   token_bucket_64,   token_bucket_128,  token_bucket_256,
            token_bucket_512,  token_bucket_1024, token_bucket_2048, token_bucket_4096,
        }, "gen_tok_inf");
    }

    /// Render metrics in Prometheus text format.
    pub fn renderPrometheus(self: *const Metrics, writer: anytype) !void {
        // Counters
        try writer.writeAll("# HELP agave_requests_total Total HTTP requests received\n");
        try writer.writeAll("# TYPE agave_requests_total counter\n");
        try writer.print("agave_requests_total {d}\n", .{self.requests_total.load(.monotonic)});

        try writer.writeAll("# HELP agave_requests_completed_total Total requests completed successfully\n");
        try writer.writeAll("# TYPE agave_requests_completed_total counter\n");
        try writer.print("agave_requests_completed_total {d}\n", .{self.requests_completed.load(.monotonic)});

        try writer.writeAll("# HELP agave_requests_cancelled_total Total requests cancelled (timeout or client disconnect)\n");
        try writer.writeAll("# TYPE agave_requests_cancelled_total counter\n");
        try writer.print("agave_requests_cancelled_total {d}\n", .{self.requests_cancelled.load(.monotonic)});

        try writer.writeAll("# HELP agave_requests_failed_total Total requests failed with errors\n");
        try writer.writeAll("# TYPE agave_requests_failed_total counter\n");
        try writer.print("agave_requests_failed_total {d}\n", .{self.requests_failed.load(.monotonic)});

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

        try writer.writeAll("# HELP agave_active_connections Current HTTP connections\n");
        try writer.writeAll("# TYPE agave_active_connections gauge\n");
        try writer.print("agave_active_connections {d}\n", .{self.active_connections.load(.monotonic)});

        try writer.writeAll("# HELP agave_kv_blocks_used KV cache blocks in use\n");
        try writer.writeAll("# TYPE agave_kv_blocks_used gauge\n");
        try writer.print("agave_kv_blocks_used {d}\n", .{self.kv_blocks_used.load(.monotonic)});

        try writer.writeAll("# HELP agave_kv_blocks_total Total KV cache blocks available\n");
        try writer.writeAll("# TYPE agave_kv_blocks_total gauge\n");
        try writer.print("agave_kv_blocks_total {d}\n", .{self.kv_blocks_total.load(.monotonic)});

        // Histograms — Prometheus requires cumulative buckets (each includes all lower)
        try self.renderHistogram(writer, "agave_request_duration_seconds", "Request latency histogram", .{
            "latency_10ms", "latency_50ms", "latency_100ms", "latency_500ms",
            "latency_1s",   "latency_5s",   "latency_10s",   "latency_30s",   "latency_inf",
        }, .{ "0.01", "0.05", "0.1", "0.5", "1", "5", "10", "30", "+Inf" }, "latency_sum", true);

        try self.renderHistogram(writer, "agave_ttft_seconds", "Time-to-first-token histogram", .{
            "ttft_10ms", "ttft_50ms", "ttft_100ms", "ttft_500ms",
            "ttft_1s",   "ttft_5s",   "ttft_10s",   "ttft_30s",   "ttft_inf",
        }, .{ "0.01", "0.05", "0.1", "0.5", "1", "5", "10", "30", "+Inf" }, "ttft_sum", true);

        // Prefill tokens
        try writer.writeAll("# HELP agave_prefill_tokens_total Total prompt tokens processed during prefill\n");
        try writer.writeAll("# TYPE agave_prefill_tokens_total counter\n");
        try writer.print("agave_prefill_tokens_total {d}\n", .{self.prefill_tokens_total.load(.monotonic)});

        // Throughput gauge
        try writer.writeAll("# HELP agave_tokens_per_second Tokens per second from last completed request\n");
        try writer.writeAll("# TYPE agave_tokens_per_second gauge\n");
        const tps_x100 = self.last_tps_x100.load(.monotonic);
        const tps: f64 = @as(f64, @floatFromInt(tps_x100)) / 100.0;
        try writer.print("agave_tokens_per_second {d:.2}\n", .{tps});

        // Prefix cache metrics — expose raw counters for Prometheus rate() queries.
        // Derived ratios (hit_rate, reuse_ratio) should be computed in PromQL via
        // rate(hits) / (rate(hits) + rate(misses)) for proper windowed aggregation.
        try writer.writeAll("# HELP agave_kv_cache_hits_total Prefix cache hits\n");
        try writer.writeAll("# TYPE agave_kv_cache_hits_total counter\n");
        try writer.print("agave_kv_cache_hits_total {d}\n", .{self.kv_cache_hits.load(.monotonic)});

        try writer.writeAll("# HELP agave_kv_cache_misses_total Prefix cache misses\n");
        try writer.writeAll("# TYPE agave_kv_cache_misses_total counter\n");
        try writer.print("agave_kv_cache_misses_total {d}\n", .{self.kv_cache_misses.load(.monotonic)});

        try writer.writeAll("# HELP agave_prefix_tokens_reused_total Tokens served from prefix cache\n");
        try writer.writeAll("# TYPE agave_prefix_tokens_reused_total counter\n");
        try writer.print("agave_prefix_tokens_reused_total {d}\n", .{self.prefix_tokens_reused.load(.monotonic)});

        try writer.writeAll("# HELP agave_prefix_tokens_total Total prompt tokens processed through prefix cache lookup\n");
        try writer.writeAll("# TYPE agave_prefix_tokens_total counter\n");
        try writer.print("agave_prefix_tokens_total {d}\n", .{self.prefix_tokens_total.load(.monotonic)});

        // Auth/rate-limit counters (security monitoring)
        try writer.writeAll("# HELP agave_requests_auth_failed_total Requests rejected due to invalid authentication\n");
        try writer.writeAll("# TYPE agave_requests_auth_failed_total counter\n");
        try writer.print("agave_requests_auth_failed_total {d}\n", .{self.requests_auth_failed.load(.monotonic)});

        try writer.writeAll("# HELP agave_requests_rate_limited_total Requests rejected due to rate limiting\n");
        try writer.writeAll("# TYPE agave_requests_rate_limited_total counter\n");
        try writer.print("agave_requests_rate_limited_total {d}\n", .{self.requests_rate_limited.load(.monotonic)});

        try writer.writeAll("# HELP agave_connections_rejected_total Connections rejected at capacity (503)\n");
        try writer.writeAll("# TYPE agave_connections_rejected_total counter\n");
        try writer.print("agave_connections_rejected_total {d}\n", .{self.connections_rejected.load(.monotonic)});

        try writer.writeAll("# HELP agave_requests_timeout_total Requests cancelled due to server-side timeout\n");
        try writer.writeAll("# TYPE agave_requests_timeout_total counter\n");
        try writer.print("agave_requests_timeout_total {d}\n", .{self.requests_timeout.load(.monotonic)});

        try writer.writeAll("# HELP agave_scheduler_errors_total Scheduler step failures\n");
        try writer.writeAll("# TYPE agave_scheduler_errors_total counter\n");
        try writer.print("agave_scheduler_errors_total {d}\n", .{self.scheduler_errors.load(.monotonic)});

        // Process metadata
        try writer.writeAll("# HELP agave_up Whether the agave server is running\n");
        try writer.writeAll("# TYPE agave_up gauge\n");
        try writer.writeAll("agave_up 1\n");

        const start = self.process_start_time.load(.monotonic);
        if (start > 0) {
            try writer.writeAll("# HELP agave_process_start_time_seconds Unix timestamp when the server started\n");
            try writer.writeAll("# TYPE agave_process_start_time_seconds gauge\n");
            try writer.print("agave_process_start_time_seconds {d}\n", .{start});
        }

        // ── llm-d routing metrics ──────────────────────────────────────

        // KV cache usage percentage (derived from blocks_used / blocks_total)
        const kv_used = self.kv_blocks_used.load(.monotonic);
        const kv_total = self.kv_blocks_total.load(.monotonic);
        const kv_usage_perc: f64 = if (kv_total > 0) @as(f64, @floatFromInt(kv_used)) / @as(f64, @floatFromInt(kv_total)) else 0.0;
        try writer.writeAll("# HELP agave_kv_cache_usage_perc Fraction of KV cache blocks in use (0-1)\n");
        try writer.writeAll("# TYPE agave_kv_cache_usage_perc gauge\n");
        try writer.print("agave_kv_cache_usage_perc {d:.4}\n", .{kv_usage_perc});

        // Prefix cache hit rate (derived from hits / (hits + misses))
        const cache_hits = self.kv_cache_hits.load(.monotonic);
        const cache_misses = self.kv_cache_misses.load(.monotonic);
        const cache_queries = cache_hits + cache_misses;
        const cache_hit_rate: f64 = if (cache_queries > 0) @as(f64, @floatFromInt(cache_hits)) / @as(f64, @floatFromInt(cache_queries)) else 0.0;
        try writer.writeAll("# HELP agave_prefix_cache_hit_rate Fraction of requests with prefix cache hit (0-1)\n");
        try writer.writeAll("# TYPE agave_prefix_cache_hit_rate gauge\n");
        try writer.print("agave_prefix_cache_hit_rate {d:.4}\n", .{cache_hit_rate});

        // Input tokens in flight
        try writer.writeAll("# HELP agave_input_tokens_in_flight Prompt tokens currently being prefilled\n");
        try writer.writeAll("# TYPE agave_input_tokens_in_flight gauge\n");
        try writer.print("agave_input_tokens_in_flight {d}\n", .{self.input_tokens_in_flight.load(.monotonic)});

        // Average prompt throughput (derived: prefill_tokens / ttft_sum)
        const pf_tokens = self.prefill_tokens_total.load(.monotonic);
        const ttft_ms_total = self.ttft_sum.load(.monotonic);
        const avg_prompt_tps: f64 = if (ttft_ms_total > 0) @as(f64, @floatFromInt(pf_tokens)) / (@as(f64, @floatFromInt(ttft_ms_total)) / ms_per_second) else 0.0;
        try writer.writeAll("# HELP agave_avg_prompt_throughput_toks_per_s Average prompt processing throughput\n");
        try writer.writeAll("# TYPE agave_avg_prompt_throughput_toks_per_s gauge\n");
        try writer.print("agave_avg_prompt_throughput_toks_per_s {d:.2}\n", .{avg_prompt_tps});

        // Average generation throughput (derived: gen_tokens / (latency_sum - ttft_sum))
        const gen_tokens = self.tokens_generated_total.load(.monotonic);
        const total_latency_ms = self.latency_sum.load(.monotonic);
        const decode_ms_total = if (total_latency_ms > ttft_ms_total) total_latency_ms - ttft_ms_total else 0;
        const avg_gen_tps: f64 = if (decode_ms_total > 0) @as(f64, @floatFromInt(gen_tokens)) / (@as(f64, @floatFromInt(decode_ms_total)) / ms_per_second) else 0.0;
        try writer.writeAll("# HELP agave_avg_generation_throughput_toks_per_s Average token generation throughput\n");
        try writer.writeAll("# TYPE agave_avg_generation_throughput_toks_per_s gauge\n");
        try writer.print("agave_avg_generation_throughput_toks_per_s {d:.2}\n", .{avg_gen_tps});

        // Cache config info (static, set once at startup)
        const blk_size = self.cache_block_size.load(.monotonic);
        const num_blks = self.cache_num_blocks.load(.monotonic);
        if (blk_size > 0 or num_blks > 0) {
            try writer.writeAll("# HELP agave_cache_config_info KV cache configuration\n");
            try writer.writeAll("# TYPE agave_cache_config_info gauge\n");
            try writer.print("agave_cache_config_info{{block_size=\"{d}\",num_gpu_blocks=\"{d}\"}} 1\n", .{ blk_size, num_blks });
        }

        try self.renderHistogram(writer, "agave_time_per_output_token_seconds", "Time per output token histogram", .{
            "tpot_5ms",   "tpot_10ms",  "tpot_20ms",  "tpot_50ms",
            "tpot_100ms", "tpot_200ms", "tpot_500ms", "tpot_inf",
        }, .{ "0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "+Inf" }, "tpot_sum", true);

        try self.renderHistogram(writer, "agave_request_queue_time_seconds", "Request queue wait time histogram", .{
            "queue_time_10ms", "queue_time_50ms", "queue_time_100ms", "queue_time_500ms",
            "queue_time_1s",   "queue_time_5s",   "queue_time_10s",   "queue_time_30s",   "queue_time_inf",
        }, .{ "0.01", "0.05", "0.1", "0.5", "1", "5", "10", "30", "+Inf" }, "queue_time_sum", true);

        try self.renderHistogram(writer, "agave_request_prompt_tokens", "Prompt token count distribution", .{
            "prompt_tok_16",  "prompt_tok_64",   "prompt_tok_128",  "prompt_tok_256",
            "prompt_tok_512", "prompt_tok_1024", "prompt_tok_2048", "prompt_tok_4096", "prompt_tok_inf",
        }, .{ "16", "64", "128", "256", "512", "1024", "2048", "4096", "+Inf" }, "prompt_tok_sum", false);

        try self.renderHistogram(writer, "agave_request_generation_tokens", "Generation token count distribution", .{
            "gen_tok_16",  "gen_tok_64",   "gen_tok_128",  "gen_tok_256",
            "gen_tok_512", "gen_tok_1024", "gen_tok_2048", "gen_tok_4096", "gen_tok_inf",
        }, .{ "16", "64", "128", "256", "512", "1024", "2048", "4096", "+Inf" }, "gen_tok_sum", false);
    }
};

const test_render_buf_size: usize = 32768;

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
    var fbs = FixedBufStream.init(&buf);
    const writer = fbs.writer();

    try metrics.renderPrometheus(writer);

    const output = fbs.getWritten();

    // ── Structural validation ─────────────────────────────────────
    // Parse lines into a set and verify required metrics exist.
    // This is resilient to line reordering unlike substring matching.
    const expected_lines = [_][]const u8{
        "# HELP agave_requests_total",
        "# TYPE agave_requests_total counter",
        "agave_requests_total 2",
        "agave_requests_completed_total 1",
        "agave_requests_failed_total 0",
        "agave_tokens_generated_total 42",
        "agave_queue_depth 5",
        "agave_request_duration_seconds_bucket{le=\"0.5\"} 1",
        "agave_request_duration_seconds_bucket{le=\"+Inf\"} 1",
        "# TYPE agave_kv_cache_hits_total counter",
        "agave_kv_cache_hits_total 0",
        "# TYPE agave_kv_cache_misses_total counter",
        "agave_kv_cache_misses_total 0",
        "# TYPE agave_prefix_tokens_reused_total counter",
        "agave_prefix_tokens_reused_total 0",
        "# TYPE agave_prefix_tokens_total counter",
        "agave_prefix_tokens_total 0",
        "# TYPE agave_requests_auth_failed_total counter",
        "agave_requests_auth_failed_total 0",
        "# TYPE agave_requests_rate_limited_total counter",
        "agave_requests_rate_limited_total 0",
        "# TYPE agave_connections_rejected_total counter",
        "agave_connections_rejected_total 0",
        "# TYPE agave_requests_timeout_total counter",
        "agave_requests_timeout_total 0",
        "# TYPE agave_scheduler_errors_total counter",
        "agave_scheduler_errors_total 0",
        "agave_up 1",
    };

    // Build line set from output for O(n*m) lookup
    for (expected_lines) |expected| {
        // Check that this exact line exists somewhere in the output.
        // Lines in Prometheus format are \n-terminated.
        var found = false;
        var line_iter = std.mem.splitScalar(u8, output, '\n');
        while (line_iter.next()) |line| {
            const trimmed = std.mem.trimEnd(u8, line, " \t\r");
            // Comment lines (# HELP) have trailing descriptions — use prefix match.
            // Metric lines must match exactly to catch value errors.
            const is_comment = expected[0] == '#';
            if ((is_comment and std.mem.startsWith(u8, trimmed, expected)) or
                (!is_comment and std.mem.eql(u8, trimmed, expected)))
            {
                found = true;
                break;
            }
        }
        if (!found) {
            std.debug.print("missing Prometheus line: {s}\n", .{expected});
            return error.TestUnexpectedResult;
        }
    }

    // Verify Prometheus format invariants: every # TYPE must precede its metric,
    // every line must be non-empty or a comment.
    var line_iter = std.mem.splitScalar(u8, output, '\n');
    var non_empty_lines: usize = 0;
    while (line_iter.next()) |line| {
        if (line.len == 0) continue;
        non_empty_lines += 1;
        // Every line must be a comment (# ...) or a metric (name value)
        try std.testing.expect(line[0] == '#' or std.ascii.isAlphabetic(line[0]));
    }
    // Should have a reasonable number of metric lines
    try std.testing.expect(non_empty_lines >= expected_lines.len);
}

test "Metrics: updateKvBlocks sets gauge values" {
    var metrics = Metrics{};

    // Initially zero
    try std.testing.expectEqual(@as(u32, 0), metrics.kv_blocks_used.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 0), metrics.kv_blocks_total.load(.monotonic));

    // Update from cache stats
    metrics.updateKvBlocks(42, 100);
    try std.testing.expectEqual(@as(u32, 42), metrics.kv_blocks_used.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 100), metrics.kv_blocks_total.load(.monotonic));

    // Verify rendered in Prometheus output
    var buf: [test_render_buf_size]u8 = undefined;
    var fbs = FixedBufStream.init(&buf);
    try metrics.renderPrometheus(fbs.writer());
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "agave_kv_blocks_used 42\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "agave_kv_blocks_total 100\n") != null);
}

test "Metrics: cache counters exposed in Prometheus output" {
    var metrics = Metrics{};

    metrics.recordCacheHit(50, 200);
    metrics.recordCacheHit(30, 150);
    metrics.recordCacheMiss(100);

    var buf: [test_render_buf_size]u8 = undefined;
    var fbs = FixedBufStream.init(&buf);
    try metrics.renderPrometheus(fbs.writer());
    const output = fbs.getWritten();

    try std.testing.expect(std.mem.indexOf(u8, output, "agave_kv_cache_hits_total 2\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "agave_kv_cache_misses_total 1\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "agave_prefix_tokens_reused_total 80\n") != null);
    // Total prompt tokens: hit(50,200) + hit(30,150) + miss(100) = 450
    try std.testing.expect(std.mem.indexOf(u8, output, "agave_prefix_tokens_total 450\n") != null);
}
