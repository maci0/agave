//! Bounded replay ledger for the mutating conversation routes.
//!
//! A retried `POST /v1/chat` otherwise appends a second user turn and persists
//! a second assistant reply; a retried `POST /v1/chat/regenerate` pops a second
//! assistant message. A client that retries after a lost response ends up with
//! a different conversation than one that never retried, so the repeat has to
//! collapse onto the first execution instead of redoing it.
//!
//! Callers key on the sanitized `X-Request-Id` header. A request without one
//! has nothing to deduplicate against and runs normally.
//!
//! Storage is a fixed ring of slots, so memory is bounded by construction, and
//! every slot expires on a monotonic deadline. A claim left behind by a
//! dropped connection or a crash expires on the in-flight deadline instead of
//! blocking retries forever.
//!
//! Not thread-safe on its own: the server touches it under `Server.mutex`.

const std = @import("std");

/// Longest key accepted, matching the server's `X-Request-Id` sanitize limit.
pub const max_key_len: usize = 64;
/// Ring size. Only the most recent `capacity` keys stay replayable.
pub const capacity: usize = 64;
/// Largest response body kept for replay. A longer one is still recorded as
/// completed, so a retry collapses instead of re-running, but the retry is
/// answered with a conflict rather than the original bytes.
pub const max_body_len: usize = 64 * 1024;
/// A claim older than this is assumed abandoned and the key frees up.
pub const in_flight_ttl_ms: i64 = 5 * 60 * 1000;
/// How long a completed key stays replayable, comfortably past any client
/// retry budget. A key older than this is treated as a fresh operation.
pub const retention_ms: i64 = 60 * 60 * 1000;

const State = enum { free, in_flight, done };

const Slot = struct {
    state: State = .free,
    key_len: u8 = 0,
    key: [max_key_len]u8 = @splat(0),
    /// In-flight claim start, or completion time, depending on `state`.
    deadline_ms: i64 = 0,
    status_line: []const u8 = "",
    content_type: []const u8 = "text/html; charset=utf-8",
    /// Owned, or empty when the response was too large to cache.
    body: []u8 = &.{},

    fn matches(self: *const Slot, key: []const u8) bool {
        if (self.state == .free) return false;
        if (self.key_len != key.len) return false;
        return std.mem.eql(u8, self.key[0..key.len], key);
    }

    fn live(self: *const Slot, now_ms: i64) bool {
        return switch (self.state) {
            .in_flight => now_ms - self.deadline_ms < in_flight_ttl_ms,
            .done => now_ms - self.deadline_ms < retention_ms,
            .free => false,
        };
    }
};

pub const Replay = struct {
    /// Borrowed from the ledger; valid until the next `claim` or `complete`.
    status_line: []const u8,
    content_type: []const u8,
    body: []const u8,
};

pub const Claim = union(enum) {
    /// Caller owns the key and must `complete` or `release` it.
    fresh,
    /// A live request already holds the key.
    duplicate,
    /// The key completed within the retention window.
    replay: Replay,
};

pub const Ledger = struct {
    allocator: std.mem.Allocator,
    slots: [capacity]Slot = @splat(.{}),
    next: usize = 0,

    pub fn init(allocator: std.mem.Allocator) Ledger {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Ledger) void {
        for (&self.slots) |*s| self.discard(s);
    }

    /// Take ownership of `key` for the calling request, or report that another
    /// request already owns it. `now_ms` is monotonic milliseconds.
    pub fn claim(self: *Ledger, key: []const u8, now_ms: i64) Claim {
        if (key.len == 0 or key.len > max_key_len) return .fresh;

        var free_slot: ?*Slot = null;
        for (&self.slots) |*s| {
            if (s.state == .free) {
                if (free_slot == null) free_slot = s;
                continue;
            }
            if (!s.matches(key)) continue;
            if (s.live(now_ms)) {
                return if (s.state == .done)
                    .{ .replay = .{ .status_line = s.status_line, .content_type = s.content_type, .body = s.body } }
                else
                    .duplicate;
            }
            // Expired: reuse this slot rather than leaving its body resident.
            self.discard(s);
            arm(s, key, now_ms);
            return .fresh;
        }

        const slot = free_slot orelse blk: {
            const s = &self.slots[self.next];
            self.next = (self.next + 1) % capacity;
            break :blk s;
        };
        self.discard(slot);
        arm(slot, key, now_ms);
        return .fresh;
    }

    /// Record the outcome of a claimed key so a retry replays it.
    /// `content_type` and `body` are stored as given; slices must outlive the
    /// ledger. No-op when the key is absent or no longer owned by this request.
    pub fn complete(self: *Ledger, key: []const u8, now_ms: i64, status_line: []const u8, content_type: []const u8, body: []const u8) void {
        if (key.len == 0 or key.len > max_key_len) return;
        for (&self.slots) |*s| {
            if (!s.matches(key)) continue;
            self.freeBody(s);
            s.status_line = status_line;
            s.content_type = content_type;
            if (body.len <= max_body_len) {
                s.body = self.allocator.dupe(u8, body) catch {
                    s.status_line = "409 Conflict";
                    s.state = .done;
                    s.deadline_ms = now_ms;
                    return;
                };
            }
            s.state = .done;
            s.deadline_ms = now_ms;
            return;
        }
    }

    /// Drop a claim without recording a result, so a failed request does not
    /// block its own retries for the in-flight window.
    pub fn release(self: *Ledger, key: []const u8) void {
        if (key.len == 0 or key.len > max_key_len) return;
        for (&self.slots) |*s| {
            if (s.matches(key) and s.state == .in_flight) {
                self.discard(s);
                return;
            }
        }
    }

    fn arm(slot: *Slot, key: []const u8, now_ms: i64) void {
        @memcpy(slot.key[0..key.len], key);
        slot.key_len = @intCast(key.len);
        slot.state = .in_flight;
        slot.deadline_ms = now_ms;
        slot.status_line = "";
    }

    fn freeBody(self: *Ledger, slot: *Slot) void {
        if (slot.body.len == 0) return;
        @memset(slot.body, 0);
        self.allocator.free(slot.body);
        slot.body = &.{};
    }

    fn discard(self: *Ledger, slot: *Slot) void {
        self.freeBody(slot);
        slot.state = .free;
        slot.key_len = 0;
        slot.status_line = "";
        slot.deadline_ms = 0;
    }
};

const testing = std.testing;

test "claim then complete then claim replays" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    try testing.expectEqual(Claim{ .fresh = {} }, l.claim("a", 0));
    l.complete("a", 10, "200 OK", "text/html", "hello");
    const second = l.claim("a", 20);
    try testing.expect(second == .replay);
    try testing.expectEqualStrings("200 OK", second.replay.status_line);
    try testing.expectEqualStrings("hello", second.replay.body);
}

test "duplicate key while in flight is rejected" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    try testing.expectEqual(Claim{ .fresh = {} }, l.claim("a", 0));
    try testing.expectEqual(Claim{ .duplicate = {} }, l.claim("a", 1));
}

test "abandoned claim expires and frees the key" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    _ = l.claim("a", 0);
    try testing.expectEqual(Claim{ .fresh = {} }, l.claim("a", in_flight_ttl_ms));
}

test "completed key expires past the retention window" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    l.complete("a", 0, "200 OK", "text/html", "hello");
    try testing.expectEqual(Claim{ .fresh = {} }, l.claim("a", retention_ms));
}

test "release frees a claim without a result" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    _ = l.claim("a", 0);
    l.release("a");
    try testing.expectEqual(Claim{ .fresh = {} }, l.claim("a", 1));
}

test "release leaves a completed key alone" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    l.complete("a", 0, "200 OK", "text/html", "hello");
    l.release("a");
    try testing.expect(l.claim("a", 1) == .replay);
}

test "empty and oversized keys are always fresh" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    try testing.expectEqual(Claim{ .fresh = {} }, l.claim("", 0));
    const long = "x" ** (max_key_len + 1);
    try testing.expectEqual(Claim{ .fresh = {} }, l.claim(long, 0));
}

test "ring eviction keeps a bounded replay set" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    for (0..capacity * 2) |i| {
        var buf: [16]u8 = undefined;
        const k = try std.fmt.bufPrint(&buf, "k{d}", .{i});
        try testing.expectEqual(Claim{ .fresh = {} }, l.claim(k, 0));
        l.complete(k, 1, "200 OK", "text/html", "body");
    }
    // The first `capacity` keys are gone; the newest `capacity` replay.
    for (0..capacity) |i| {
        var buf: [16]u8 = undefined;
        const k = try std.fmt.bufPrint(&buf, "k{d}", .{i});
        try testing.expect(l.claim(k, 2) == .fresh);
    }
    for (capacity..capacity * 2) |i| {
        var buf: [16]u8 = undefined;
        const k = try std.fmt.bufPrint(&buf, "k{d}", .{i});
        try testing.expect(l.claim(k, 2) == .replay);
    }
}

test "oversized body still records completion" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    const big = "y" ** (max_body_len + 1);
    l.complete("a", 0, "200 OK", "text/html", big);
    const c = l.claim("a", 1);
    try testing.expect(c == .replay);
    try testing.expectEqualStrings("", c.replay.body);
}

test "slot reuse does not leak the previous body" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    for (0..capacity + 4) |i| {
        var buf: [16]u8 = undefined;
        const k = try std.fmt.bufPrint(&buf, "k{d}", .{i});
        _ = l.claim(k, 0);
        l.complete(k, 0, "200 OK", "text/html", "secret-body");
    }
    try testing.expectEqual(@as(usize, 0), countStoredBodies(&l));
}

fn countStoredBodies(l: *const Ledger) usize {
    var n: usize = 0;
    for (&l.slots) |*s| {
        if (s.body.len != 0) n += 1;
    }
    return n;
}
