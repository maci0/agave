//! Token bucket rate limiter for per-API-key request and token limits.
//!
//! Implements the token bucket algorithm with dual buckets (requests/min, tokens/min).
//! Tokens refill continuously based on elapsed time, clamped to capacity.

const std = @import("std");

/// Single token bucket for rate limiting.
/// Refills tokens at a constant rate, with maximum burst capacity.
pub const TokenBucket = struct {
    capacity: f64,
    tokens: f64,
    refill_rate: f64,
    last_refill: i64,

    /// Try to consume the given amount of tokens.
    /// Refills based on elapsed time before checking availability.
    /// Returns true if tokens were available and consumed, false otherwise.
    pub fn tryConsume(self: *TokenBucket, amount: f64) bool {
        const now = std.time.milliTimestamp();
        const elapsed_sec = @as(f64, @floatFromInt(now - self.last_refill)) / 1000.0;
        self.tokens = @min(self.capacity, self.tokens + elapsed_sec * self.refill_rate);
        self.last_refill = now;

        if (self.tokens >= amount) {
            self.tokens -= amount;
            return true;
        }
        return false;
    }

    /// Calculate how many seconds until the given amount becomes available.
    /// Used for HTTP Retry-After header.
    pub fn retryAfterSeconds(self: *const TokenBucket, amount: f64) u32 {
        const deficit = amount - self.tokens;
        if (deficit <= 0) return 0;
        return @intFromFloat(@ceil(deficit / self.refill_rate));
    }
};

/// Per-API-key rate limiter with dual limits (requests/min and tokens/min).
pub const RateLimiter = struct {
    request_bucket: TokenBucket,
    token_bucket: TokenBucket,

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
                .refill_rate = req_capacity / 60.0,
                .last_refill = now,
            },
            .token_bucket = .{
                .capacity = token_capacity,
                .tokens = token_capacity,
                .refill_rate = token_capacity / 60.0,
                .last_refill = now,
            },
        };
    }

    /// Try to consume one request and the given number of tokens.
    /// Returns true if both buckets had sufficient tokens.
    pub fn tryConsumeRequest(self: *RateLimiter, token_count: u32) bool {
        const tokens_f64 = @as(f64, @floatFromInt(token_count));
        return self.request_bucket.tryConsume(1.0) and self.token_bucket.tryConsume(tokens_f64);
    }

    /// Calculate retry-after delay in seconds.
    /// Returns the maximum of the two bucket retry times.
    pub fn retryAfter(self: *const RateLimiter, token_count: u32) u32 {
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
