//! Token bucket rate limiter for per-API-key request and token limits.
//!
//! Implements the token bucket algorithm with dual buckets (requests/min, tokens/min).
//! Tokens refill continuously based on elapsed time, clamped to capacity.

const std = @import("std");

/// Milliseconds per second — used for elapsed-time calculations.
const ms_per_second: f64 = 1000.0;
/// Seconds per minute — used for per-minute refill rate calculations.
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
        const elapsed_sec = @as(f64, @floatFromInt(now - self.last_refill)) / ms_per_second;
        self.tokens = @min(self.capacity, self.tokens + elapsed_sec * self.refill_rate);
        self.last_refill = now;
    }

    /// Calculate how many seconds until the given amount becomes available.
    /// Used for HTTP Retry-After header.
    pub fn retryAfterSeconds(self: *const TokenBucket, amount: f64) u32 {
        const deficit = amount - self.tokens;
        if (deficit <= 0) return 0;
        if (self.refill_rate <= 0) return std.math.maxInt(u32);
        return @intFromFloat(@ceil(deficit / self.refill_rate));
    }
};

/// Global rate limiter with dual limits (requests/min and tokens/min).
/// A single instance is shared across all requests regardless of API key.
/// Thread-safe: guards bucket state with a mutex since multiple HTTP handler
/// threads call tryConsumeOrRetryAfter() concurrently.
pub const RateLimiter = struct {
    request_bucket: TokenBucket,
    token_bucket: TokenBucket,
    mutex: std.Thread.Mutex = .{},

    /// Initialize rate limiter with per-minute limits.
    /// Both buckets start at full capacity.
    pub fn init(req_per_min: u32, tokens_per_min: u32) RateLimiter {
        const now = std.time.milliTimestamp();
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
        };
    }

    /// Refill both buckets based on elapsed time since last refill.
    /// Must be called under mutex. Caller should obtain `now` via
    /// `std.time.milliTimestamp()` *before* acquiring the lock to keep
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
        const now = std.time.milliTimestamp();
        self.mutex.lock();
        defer self.mutex.unlock();

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
        const now = std.time.milliTimestamp();
        self.mutex.lock();
        defer self.mutex.unlock();

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
        const now = std.time.milliTimestamp();
        self.mutex.lock();
        defer self.mutex.unlock();

        self.refillBuckets(now);
        const tokens_f64 = @as(f64, @floatFromInt(token_count));
        return @max(
            self.request_bucket.retryAfterSeconds(1.0),
            self.token_bucket.retryAfterSeconds(tokens_f64),
        );
    }
};

// Unit tests
test "consume full capacity then fail" {
    var limiter = RateLimiter.init(10, 100);

    // Consume all 10 requests
    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        try std.testing.expect(limiter.tryConsumeRequest(1));
    }

    // Next request should fail
    try std.testing.expect(!limiter.tryConsumeRequest(1));
}

test "refill after 1 second" {
    var limiter = RateLimiter.init(60, 600);

    // Consume one request
    try std.testing.expect(limiter.tryConsumeRequest(10));

    // Manually advance time by simulating 1 second elapsed
    limiter.request_bucket.last_refill -= 1000;
    limiter.token_bucket.last_refill -= 1000;

    // Should be able to consume again (refilled 1 request and 10 tokens)
    try std.testing.expect(limiter.tryConsumeRequest(10));
}

test "long idle clamps to capacity" {
    var limiter = RateLimiter.init(10, 100);

    // Consume 5 requests
    var i: u32 = 0;
    while (i < 5) : (i += 1) {
        try std.testing.expect(limiter.tryConsumeRequest(1));
    }

    // Simulate 10 minutes idle (would refill 100 requests without clamping)
    limiter.request_bucket.last_refill -= 600_000;
    limiter.token_bucket.last_refill -= 600_000;

    // Should have exactly 10 requests available (clamped to capacity)
    i = 0;
    while (i < 10) : (i += 1) {
        try std.testing.expect(limiter.tryConsumeRequest(1));
    }
    try std.testing.expect(!limiter.tryConsumeRequest(1));
}

test "retry after matches calculation" {
    var limiter = RateLimiter.init(60, 600);

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
    var limiter = RateLimiter.init(100, 5);

    // Consume 5 requests with 1 token each — exhausts token bucket
    var i: u32 = 0;
    while (i < 5) : (i += 1) {
        try std.testing.expect(limiter.tryConsumeRequest(1));
    }

    // Request bucket has 95 remaining, but token bucket is empty → should fail
    try std.testing.expect(!limiter.tryConsumeRequest(1));

    // Retry-after should reflect token bucket deficit, not request bucket
    const retry = limiter.retryAfter(1);
    try std.testing.expect(retry >= 1); // Need at least 1 second to refill 1 token
}

test "tryConsumeOrRetryAfter combines check and retry" {
    var limiter = RateLimiter.init(10, 100);

    // Should succeed (returns null) when capacity is available
    try std.testing.expectEqual(@as(?u32, null), limiter.tryConsumeOrRetryAfter(1));

    // Exhaust remaining requests
    var i: u32 = 0;
    while (i < 9) : (i += 1) {
        try std.testing.expectEqual(@as(?u32, null), limiter.tryConsumeOrRetryAfter(1));
    }

    // Next call should return retry-after seconds (non-null)
    const retry = limiter.tryConsumeOrRetryAfter(1);
    try std.testing.expect(retry != null);
    try std.testing.expect(retry.? >= 1);
}
