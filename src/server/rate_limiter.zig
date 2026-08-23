//! Token bucket rate limiter for per-API-key request and token limits.
//!
//! Implements the token bucket algorithm with dual buckets (requests/min, tokens/min).
//! Tokens refill continuously based on elapsed time, clamped to capacity.

const std = @import("std");
const Io = std.Io;
const Mutex = Io.Mutex;
const sim_clock = @import("../sim_clock.zig");

/// Monotonic interval clock: bucket refill must not burst on an NTP step
/// forward nor stall on a step backward, which REALTIME reads would cause.
fn milliTimestamp() i64 {
    return sim_clock.monoMilli();
}

const ms_per_second: f64 = 1000.0;
const seconds_per_minute: f64 = 60.0;

/// Single token bucket for rate limiting.
/// Refills tokens at a constant rate, with maximum burst capacity.
pub const TokenBucket = struct {
    capacity: f64,
    tokens: f64,
    refill_rate: f64,
    last_refill: i64,

    /// Refill tokens based on elapsed time since last refill.
    /// Accepts a pre-fetched timestamp so callers can refill multiple buckets
    /// with a consistent `now` value under a single lock.
    fn refill(self: *TokenBucket, now: i64) void {
        if (now <= self.last_refill) {
            self.last_refill = now;
            return;
        }
        const elapsed_sec = @as(f64, @floatFromInt(now - self.last_refill)) / ms_per_second;
        self.tokens = @min(self.capacity, self.tokens + elapsed_sec * self.refill_rate);
        self.last_refill = now;
    }

    /// Maximum Retry-After value (1 hour) to avoid absurd HTTP headers
    /// when refill rate is near-zero.
    const max_retry_after: u32 = 3600;

    /// Calculate how many seconds until the given amount becomes available.
    /// Used for HTTP Retry-After header. Capped to max_retry_after (1 hour).
    pub fn retryAfterSeconds(self: *const TokenBucket, amount: f64) u32 {
        const deficit = amount - self.tokens;
        if (deficit <= 0) return 0;
        if (self.refill_rate <= 0) return max_retry_after;
        const raw = @ceil(deficit / self.refill_rate);
        return if (raw >= @as(f64, @floatFromInt(max_retry_after))) max_retry_after else @intFromFloat(raw);
    }
};

/// Global rate limiter with dual limits (requests/min and tokens/min).
/// A single instance is shared across all requests regardless of API key.
/// Thread-safe: guards bucket state with a mutex since multiple HTTP handler
/// threads call tryConsumeOrRetryAfter() concurrently.
pub const RateLimiter = struct {
    request_bucket: TokenBucket,
    token_bucket: TokenBucket,
    mutex: Mutex = .init,
    io: Io,

    /// Initialize rate limiter with per-minute limits.
    /// Both buckets start at full capacity.
    pub fn init(req_per_min: u32, tokens_per_min: u32, io: Io) RateLimiter {
        const now = milliTimestamp();
        const req_capacity = @as(f64, @floatFromInt(req_per_min));
        const token_capacity = @as(f64, @floatFromInt(tokens_per_min));

        return .{
            .request_bucket = .{
                .capacity = req_capacity,
                .tokens = req_capacity,
                .refill_rate = req_capacity / seconds_per_minute,
                .last_refill = now,
            },
            .token_bucket = .{
                .capacity = token_capacity,
                .tokens = token_capacity,
                .refill_rate = token_capacity / seconds_per_minute,
                .last_refill = now,
            },
            .io = io,
        };
    }

    /// Refill both buckets based on elapsed time since last refill.
    /// Must be called under mutex. Caller should obtain `now` via
    /// `milliTimestamp()` (monotonic) *before* acquiring the lock to keep
    /// the syscall outside the critical section.
    fn refillBuckets(self: *RateLimiter, now: i64) void {
        self.request_bucket.refill(now);
        self.token_bucket.refill(now);
    }

    /// Try to consume one request and the given number of tokens.
    /// Returns true if both buckets had sufficient tokens.
    /// Checks both buckets before consuming either to avoid wasting
    /// capacity when one bucket is exhausted.
    /// Thread-safe: acquires mutex to protect bucket state.
    pub fn tryConsumeRequest(self: *RateLimiter, token_count: u32) bool {
        const now = milliTimestamp();
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);

        const tokens_f64 = @as(f64, @floatFromInt(token_count));
        self.refillBuckets(now);

        // Check both before consuming either
        if (self.request_bucket.tokens >= 1.0 and self.token_bucket.tokens >= tokens_f64) {
            self.request_bucket.tokens -= 1.0;
            self.token_bucket.tokens -= tokens_f64;
            return true;
        }
        return false;
    }

    /// Try to consume one request and the given number of tokens.
    /// Returns null on success (tokens consumed), or retry-after seconds on failure.
    /// Single lock acquisition — avoids the TOCTOU gap and double-lock overhead
    /// of calling tryConsumeRequest() then retryAfter() separately.
    pub fn tryConsumeOrRetryAfter(self: *RateLimiter, token_count: u32) ?u32 {
        const now = milliTimestamp();
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);

        const tokens_f64 = @as(f64, @floatFromInt(token_count));
        self.refillBuckets(now);

        // Check both before consuming either
        if (self.request_bucket.tokens >= 1.0 and self.token_bucket.tokens >= tokens_f64) {
            self.request_bucket.tokens -= 1.0;
            self.token_bucket.tokens -= tokens_f64;
            return null; // Success
        }

        // Rate limited — return retry-after under the same lock
        return @max(
            self.request_bucket.retryAfterSeconds(1.0),
            self.token_bucket.retryAfterSeconds(tokens_f64),
        );
    }

    /// Calculate retry-after delay in seconds.
    /// Returns the maximum of the two bucket retry times.
    /// Thread-safe: acquires mutex to read consistent bucket state.
    pub fn retryAfter(self: *RateLimiter, token_count: u32) u32 {
        const now = milliTimestamp();
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);

        self.refillBuckets(now);
        const tokens_f64 = @as(f64, @floatFromInt(token_count));
        return @max(
            self.request_bucket.retryAfterSeconds(1.0),
            self.token_bucket.retryAfterSeconds(tokens_f64),
        );
    }
};

/// Create a test Io instance for unit tests.
fn testIo() Io {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    return threaded.io();
}

// Unit tests
test "consume full capacity then fail" {
    var limiter = RateLimiter.init(10, 100, testIo());

    // Consume all 10 requests
    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        try std.testing.expect(limiter.tryConsumeRequest(1));
    }

    // Next request should fail
    try std.testing.expect(!limiter.tryConsumeRequest(1));
}

test "refill after 1 second" {
    defer sim_clock.setOverrideMs(null);
    sim_clock.setOverrideMs(1_000_000);
    var limiter = RateLimiter.init(60, 600, testIo());

    // Consume one request
    try std.testing.expect(limiter.tryConsumeRequest(10));

    // Advance simulated clock by 1 second (refill driven by injectable clock)
    sim_clock.advanceMs(1000);

    // Should be able to consume again (refilled 1 request and 10 tokens)
    try std.testing.expect(limiter.tryConsumeRequest(10));
}

test "long idle clamps to capacity" {
    defer sim_clock.setOverrideMs(null);
    sim_clock.setOverrideMs(1_000_000);
    var limiter = RateLimiter.init(10, 100, testIo());

    // Consume 5 requests
    var i: u32 = 0;
    while (i < 5) : (i += 1) {
        try std.testing.expect(limiter.tryConsumeRequest(1));
    }

    // Advance simulated clock by 10 minutes (would refill 100 requests without clamping)
    sim_clock.advanceMs(600_000);

    // Should have exactly 10 requests available (clamped to capacity)
    i = 0;
    while (i < 10) : (i += 1) {
        try std.testing.expect(limiter.tryConsumeRequest(1));
    }
    try std.testing.expect(!limiter.tryConsumeRequest(1));
}

test "retry after matches calculation" {
    var limiter = RateLimiter.init(60, 600, testIo());

    // Consume all requests (60)
    var i: u32 = 0;
    while (i < 60) : (i += 1) {
        try std.testing.expect(limiter.tryConsumeRequest(1));
    }

    // Need 1 more request, refill rate is 1/sec, so retry = 1 second
    const retry = limiter.retryAfter(1);
    try std.testing.expectEqual(@as(u32, 1), retry);
}

test "token bucket exhaustion blocks even with requests available" {
    // 100 requests/min but only 5 tokens/min — token bucket should be the bottleneck.
    // Verifies the dual-bucket check: both must have capacity.
    var limiter = RateLimiter.init(100, 5, testIo());

    // Consume 5 requests with 1 token each — exhausts token bucket
    var i: u32 = 0;
    while (i < 5) : (i += 1) {
        try std.testing.expect(limiter.tryConsumeRequest(1));
    }

    // Request bucket has 95 remaining, but token bucket is empty → should fail
    try std.testing.expect(!limiter.tryConsumeRequest(1));

    // Retry-after should reflect token bucket deficit, not request bucket
    // Token rate = 5/min = 1 per 12s, so retryAfter(1 token) = 12s
    const retry = limiter.retryAfter(1);
    try std.testing.expectEqual(@as(u32, 12), retry);
}

test "tryConsumeOrRetryAfter combines check and retry" {
    var limiter = RateLimiter.init(10, 100, testIo());

    // Should succeed (returns null) when capacity is available
    try std.testing.expectEqual(@as(?u32, null), limiter.tryConsumeOrRetryAfter(1));

    // Exhaust remaining requests
    var i: u32 = 0;
    while (i < 9) : (i += 1) {
        try std.testing.expectEqual(@as(?u32, null), limiter.tryConsumeOrRetryAfter(1));
    }

    // Next call should return retry-after seconds (non-null)
    // Request rate = 10/min = 1 per 6s, so retryAfter(1 request) = 6s
    const retry = limiter.tryConsumeOrRetryAfter(1);
    try std.testing.expectEqual(@as(?u32, 6), retry);
}

test "fuzz: TokenBucket and RateLimiter" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            var raw: [16]u8 = undefined;
            smith.bytesWithHash(&raw, 0);

            // TokenBucket.retryAfterSeconds with random state.
            const capacity: f64 = @floatFromInt(@as(u32, std.mem.readInt(u16, raw[0..2], .little)) + 1);
            const tokens: f64 = @floatFromInt(@as(u32, std.mem.readInt(u16, raw[2..4], .little)));
            const refill_rate = capacity / seconds_per_minute;
            var bucket = TokenBucket{
                .capacity = capacity,
                .tokens = @min(tokens, capacity),
                .refill_rate = refill_rate,
                .last_refill = 0,
            };

            const amount: f64 = @floatFromInt(@as(u32, std.mem.readInt(u16, raw[4..6], .little)));
            const retry = bucket.retryAfterSeconds(amount);
            // Invariant: retry must be bounded.
            try std.testing.expect(retry <= TokenBucket.max_retry_after);

            // TokenBucket.refill with forward time.
            const time_delta: i64 = @intCast(@as(u32, std.mem.readInt(u16, raw[6..8], .little)));
            const old_tokens = bucket.tokens;
            bucket.refill(time_delta);
            // Invariant: tokens must be clamped to capacity.
            try std.testing.expect(bucket.tokens <= bucket.capacity);
            // Invariant: refill must not decrease tokens (time goes forward).
            try std.testing.expect(bucket.tokens >= old_tokens or time_delta <= bucket.last_refill);

            // RateLimiter: consume some requests.
            const io = testIo();
            const req_pm = @as(u32, std.mem.readInt(u16, raw[8..10], .little) % 1000) + 1;
            const tok_pm = @as(u32, std.mem.readInt(u16, raw[10..12], .little) % 10000) + 1;
            var limiter = RateLimiter.init(req_pm, tok_pm, io);
            const consume_count = raw[12] % 5;
            for (0..consume_count) |_| {
                _ = limiter.tryConsumeRequest(1);
            }
            // retryAfter must not crash.
            const r = limiter.retryAfter(1);
            try std.testing.expect(r <= TokenBucket.max_retry_after);

            // tryConsumeOrRetryAfter must return consistent results.
            const result = limiter.tryConsumeOrRetryAfter(1);
            if (result) |retry_secs| {
                try std.testing.expect(retry_secs <= TokenBucket.max_retry_after);
            }
        }
    }.f, .{});
}
