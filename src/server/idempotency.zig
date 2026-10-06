//! Bounded replay ledger for the mutating conversation routes.
//!
//! A retried `POST /v1/chat` otherwise appends a second user turn and persists
//! a second assistant reply; a retried `POST /v1/chat/regenerate` pops a second
//! assistant message. A client that retries after a lost response ends up with
//! a different conversation than one that never retried, so the repeat has to
//! collapse onto the first execution instead of redoing it.
//!
//! Callers key on the sanitized `X-Request-Id` header, scoped to the route that
//! claimed it *and* to the principal that claimed it: the id alone is
//! client-chosen and says nothing about which operation it names or who sent
//! it, so a client that reuses one id across two routes (or two clients that
//! pick the same id) would otherwise replay the first route's response to the
//! second, and a second principal could read the first one's stored reply or
//! suppress its retry by presenting the same key. `owner` is a tag the caller
//! derives from whatever it already authenticated with, never a credential. A
//! route longer than `max_route_len` is not claimable, so a truncated route
//! can never alias another one. A request without an id has nothing to
//! deduplicate against and runs normally.
//!
//! Storage is a fixed ring of slots, so memory is bounded by construction, and
//! every slot expires on a monotonic deadline. A claim left behind by a
//! dropped connection or a crash expires on the in-flight deadline instead of
//! blocking retries forever.
//!
//! Not thread-safe on its own: the server touches it under `Server.mutex`.
//!
//! `complete` versus `release` is the caller's contract. Release a claim only
//! while the request left nothing behind: a rolled-back append, a rejected
//! rate limit, a failed allocation. Once the operation has changed stored state
//! and persisted it, a failure response completes the key instead, so the retry
//! collapses onto what the first execution left rather than applying it twice.
//!
//! Both take the token from `claim`. A request that runs past
//! `in_flight_ttl_ms` can have its slot re-armed for a retry; without the token
//! its late `complete` would answer the retry, and its late `release` would
//! hand the key to a third request while the retry is still executing.
//!
//! A full ring reclaims the oldest completed slot, never one still `in_flight`:
//! freeing a live claim would hand its key to a concurrent retry, which would
//! then run the same mutation a second time while the first is still running.
//! With every slot in flight the new key runs unclaimed (a zero `fresh`
//! token), which costs that one operation its replay window and nothing else.

const std = @import("std");

/// Longest key accepted, matching the server's `X-Request-Id` sanitize limit.
pub const max_key_len: usize = 64;
/// Longest route a claim can be scoped to. The longest mutating route is
/// `/v1/chat/regenerate`; anything longer is left unclaimed rather than
/// truncated, since a truncated route would match a different one.
pub const max_route_len: usize = 32;
/// Ring size. Only the most recent `capacity` keys stay replayable. A slot
/// holding a live `in_flight` claim is never evicted (see `claim`), so the
/// effective ceiling on concurrently deduplicated requests is this many.
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

/// A claim is only ever visible to the principal that took it. Zero is the
/// anonymous scope, which a caller with no credential of its own uses; every
/// such caller shares it, so it must compare equal to itself. A caller that
/// can distinguish principals passes a tag derived from what it already
/// authenticated with, never the credential itself.
pub const anonymous_owner: Owner = 0;

/// 128-bit principal tag. Wide enough that a digest collision is not a
/// practical concern; the value is a hash, never a secret.
pub const Owner = u128;

const State = enum { free, in_flight, done };

const Slot = struct {
    state: State = .free,
    key_len: u8 = 0,
    key: [max_key_len]u8 = @splat(0),
    /// Route that claimed the key. Part of `matches`, so one `X-Request-Id`
    /// can only ever replay the route it was first used on.
    route_len: u8 = 0,
    route: [max_route_len]u8 = @splat(0),
    /// Identifies the claim that armed this slot. `complete` and `release`
    /// match on it so a request whose claim expired and was re-armed for
    /// another caller cannot overwrite or drop the new claim.
    token: u64 = 0,
    /// Arm order, monotonic across the ledger's life. Breaks eviction ties
    /// between two `done` slots whose `deadline_ms` are equal, which is the
    /// normal case when many requests complete in one millisecond: without it
    /// the oldest-by-index slot is evicted over and over and the ring stops
    /// holding the most recent `capacity` keys.
    seq: u64 = 0,
    /// Principal that took the claim. Part of `matches`, so one key is only
    /// ever replayed back to the requester that created it.
    owner: Owner = anonymous_owner,
    /// In-flight claim start, or completion time, depending on `state`.
    deadline_ms: i64 = 0,
    status_line: []const u8 = "",
    content_type: []const u8 = "text/html; charset=utf-8",
    /// Owned, or empty when the response was too large to cache.
    body: []u8 = &.{},

    fn matches(self: *const Slot, key: []const u8, route: []const u8, owner: Owner) bool {
        if (self.state == .free) return false;
        if (self.key_len != key.len) return false;
        if (self.route_len != route.len) return false;
        if (self.owner != owner) return false;
        if (!std.mem.eql(u8, self.key[0..key.len], key)) return false;
        return std.mem.eql(u8, self.route[0..route.len], route);
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
    status_line: []const u8,
    content_type: []const u8,
    /// Owned copy of the recorded body. Free it with `deinit` once the bytes
    /// are on the wire: the ledger's own copy can be freed by a later `claim`
    /// or `complete` while the response is still being written.
    body: []u8,

    /// No-op when the body was empty, which is how a streamed or oversized
    /// response is recorded.
    pub fn deinit(self: Replay, allocator: std.mem.Allocator) void {
        if (self.body.len == 0) return;
        allocator.free(self.body);
    }
};

pub const Claim = union(enum) {
    /// Caller owns the key under `token` and must pass it to `complete` or
    /// `release`. A zero token means the key was not claimable (empty,
    /// oversized, or every slot held a live in-flight claim) and there is
    /// nothing to complete.
    fresh: u64,
    /// A live request already holds the key.
    duplicate,
    /// The key completed within the retention window.
    replay: Replay,
};

pub const Ledger = struct {
    allocator: std.mem.Allocator,
    slots: [capacity]Slot = @splat(.{}),
    next_token: u64 = 1,
    next_seq: u64 = 1,

    pub fn init(allocator: std.mem.Allocator) Ledger {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Ledger) void {
        for (&self.slots) |*s| self.discard(s);
    }

    /// Take ownership of `key` for the calling request, or report that another
    /// request already owns it. `now_ms` is monotonic milliseconds. The claim
    /// is scoped to `route` and `owner`, so either difference makes a different
    /// key. A fresh claim carries a token that scopes its `complete` and
    /// `release`.
    ///
    /// A `fresh` token of 0 means the ring held no slot this key could take
    /// without stealing one from a request still running. Evicting that one
    /// would let its owner's retry re-enter as `fresh` and run the same
    /// mutation twice at once, so instead the new key runs unclaimed: it has
    /// no duplicate yet, and an unclaimed run repeats no side effect. Only
    /// the deduplication window is lost, not the run.
    pub fn claim(self: *Ledger, key: []const u8, route: []const u8, owner: Owner, now_ms: i64) Claim {
        if (key.len == 0 or key.len > max_key_len) return .{ .fresh = 0 };
        if (route.len == 0 or route.len > max_route_len) return .{ .fresh = 0 };

        var free_slot: ?*Slot = null;
        // Oldest live `done` slot. Its replay is a convenience for a retry,
        // not the guard against a second execution, so reclaiming it only
        // costs that one retry its cached bytes. `in_flight` slots are never
        // candidates: reclaiming one re-opens its key to a concurrent retry.
        var oldest_done: ?*Slot = null;
        var oldest_seq: u64 = 0;
        for (&self.slots) |*s| {
            if (s.state == .free) {
                if (free_slot == null) free_slot = s;
                continue;
            }
            if (s.matches(key, route, owner)) {
                if (s.live(now_ms)) {
                    if (s.state == .in_flight) return .duplicate;
                    return .{
                        .replay = .{
                            .status_line = s.status_line,
                            .content_type = s.content_type,
                            // Copied out under the ledger lock: the slot body is freed
                            // as soon as any other request claims, evicts, or completes.
                            .body = blk: {
                                if (s.body.len == 0) break :blk @as([]u8, @constCast(""));
                                break :blk self.allocator.dupe(u8, s.body) catch @as([]u8, @constCast(""));
                            },
                        },
                    };
                }
                // Expired: reuse this slot rather than leaving its body resident.
                self.discard(s);
                return .{ .fresh = self.arm(s, key, route, owner, now_ms) };
            }
            // Not this key. Track the cheapest reclaimable slot for the miss.
            if (s.state == .done and s.live(now_ms) and
                (oldest_done == null or s.seq < oldest_seq))
            {
                oldest_done = s;
                oldest_seq = s.seq;
            }
        }

        const slot = free_slot orelse oldest_done orelse return .{ .fresh = 0 };
        self.discard(slot);
        return .{ .fresh = self.arm(slot, key, route, owner, now_ms) };
    }

    /// Record the outcome of a claimed key so a retry replays it.
    /// `content_type` and `body` are stored as given; slices must outlive the
    /// ledger. No-op when the key is absent, or when the slot has moved on to a
    /// later claim, so a request that ran past the in-flight TTL cannot
    /// complete the claim that replaced it.
    pub fn complete(self: *Ledger, key: []const u8, route: []const u8, owner: Owner, token: u64, now_ms: i64, status_line: []const u8, content_type: []const u8, body: []const u8) void {
        if (token == 0 or key.len == 0 or key.len > max_key_len) return;
        for (&self.slots) |*s| {
            if (!s.matches(key, route, owner) or s.token != token) continue;
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
    /// block its own retries for the in-flight window. A stale `token` is a
    /// no-op: the key now belongs to a later claim.
    pub fn release(self: *Ledger, key: []const u8, route: []const u8, owner: Owner, token: u64) void {
        if (token == 0 or key.len == 0 or key.len > max_key_len) return;
        for (&self.slots) |*s| {
            if (s.matches(key, route, owner) and s.state == .in_flight and s.token == token) {
                self.discard(s);
                return;
            }
        }
    }

    fn arm(self: *Ledger, slot: *Slot, key: []const u8, route: []const u8, owner: Owner, now_ms: i64) u64 {
        @memcpy(slot.key[0..key.len], key);
        slot.key_len = @intCast(key.len);
        @memcpy(slot.route[0..route.len], route);
        slot.route_len = @intCast(route.len);
        const token = self.next_token;
        self.next_token +%= 1;
        slot.token = token;
        slot.seq = self.next_seq;
        self.next_seq +%= 1;
        slot.owner = owner;
        slot.state = .in_flight;
        slot.deadline_ms = now_ms;
        slot.status_line = "";
        return token;
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
        slot.route_len = 0;
        slot.seq = 0;
        slot.owner = anonymous_owner;
        slot.status_line = "";
        slot.deadline_ms = 0;
    }
};

const testing = std.testing;
/// Route every test claims on, unless it is exercising route scoping.
const test_route = "/v1/chat/regenerate";
/// Principal every test claims as, unless it is exercising owner scoping.
const test_owner: Owner = anonymous_owner;
/// A second principal, distinct from `test_owner`.
const other_owner: Owner = 0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c835;

/// Claim `key` and return its token, failing the test if the claim was refused.
fn claimKey(l: *Ledger, key: []const u8, now_ms: i64) !u64 {
    const c = l.claim(key, test_route, test_owner, now_ms);
    try testing.expect(c == .fresh);
    return c.fresh;
}

test "claim then complete then claim replays" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    const t = try claimKey(&l, "a", 0);
    l.complete("a", test_route, test_owner, t, 10, "200 OK", "text/html", "hello");
    const second = l.claim("a", test_route, test_owner, 20);
    try testing.expect(second == .replay);
    defer second.replay.deinit(testing.allocator);
    try testing.expectEqualStrings("200 OK", second.replay.status_line);
    try testing.expectEqualStrings("hello", second.replay.body);
}

test "duplicate key while in flight is rejected" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    _ = try claimKey(&l, "a", 0);
    try testing.expectEqual(Claim{ .duplicate = {} }, l.claim("a", test_route, test_owner, 1));
}

test "abandoned claim expires and frees the key" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    _ = try claimKey(&l, "a", 0);
    try testing.expect(l.claim("a", test_route, test_owner, in_flight_ttl_ms) == .fresh);
}

test "completed key expires past the retention window" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    const t = try claimKey(&l, "a", 0);
    l.complete("a", test_route, test_owner, t, 0, "200 OK", "text/html", "hello");
    try testing.expect(l.claim("a", test_route, test_owner, retention_ms) == .fresh);
}

test "release frees a claim without a result" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    const t = try claimKey(&l, "a", 0);
    l.release("a", test_route, test_owner, t);
    try testing.expect(l.claim("a", test_route, test_owner, 1) == .fresh);
}

test "release leaves a completed key alone" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    const t = try claimKey(&l, "a", 0);
    l.complete("a", test_route, test_owner, t, 0, "200 OK", "text/html", "hello");
    l.release("a", test_route, test_owner, t);
    const c = l.claim("a", test_route, test_owner, 1);
    try testing.expect(c == .replay);
    c.replay.deinit(testing.allocator);
}

test "a stale claim cannot complete the request that replaced it" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    const stale = try claimKey(&l, "a", 0);
    // The first request runs past the in-flight window, so the key is
    // re-armed for a second request.
    const current = try claimKey(&l, "a", in_flight_ttl_ms);
    try testing.expect(current != stale);

    l.complete("a", test_route, test_owner, stale, in_flight_ttl_ms, "200 OK", "text/html", "first");
    const c = l.claim("a", test_route, test_owner, in_flight_ttl_ms);
    try testing.expectEqual(Claim{ .duplicate = {} }, c);
}

test "a stale claim cannot release the request that replaced it" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    const stale = try claimKey(&l, "a", 0);
    _ = try claimKey(&l, "a", in_flight_ttl_ms);
    l.release("a", test_route, test_owner, stale);
    try testing.expectEqual(Claim{ .duplicate = {} }, l.claim("a", test_route, test_owner, in_flight_ttl_ms));
}

test "empty and oversized keys are always fresh" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    const empty = l.claim("", test_route, test_owner, 0);
    try testing.expectEqual(@as(u64, 0), empty.fresh);
    const long = &@as([max_key_len + 1]u8, @splat(0x78));
    try testing.expectEqual(@as(u64, 0), l.claim(long, test_route, test_owner, 0).fresh);
}

test "the same key on another route is a different operation" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    const t = try claimKey(&l, "a", 0);
    l.complete("a", test_route, test_owner, t, 1, "200 OK", "text/html", "chat reply");

    // A second client reusing the id on another route must run, not replay.
    const other = l.claim("a", "/v1/conversations", test_owner, 2);
    try testing.expect(other == .fresh);
    try testing.expect(other.fresh != 0);

    // The original route still replays.
    const c = l.claim("a", test_route, test_owner, 3);
    try testing.expect(c == .replay);
    defer c.replay.deinit(testing.allocator);
    try testing.expectEqualStrings("chat reply", c.replay.body);
}

test "an in-flight claim does not block the same id on another route" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    _ = try claimKey(&l, "a", 0);
    try testing.expectEqual(Claim{ .duplicate = {} }, l.claim("a", test_route, test_owner, 1));
    try testing.expect(l.claim("a", "/v1/chat", test_owner, 1) == .fresh);
}

test "empty and oversized routes are never claimed" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    try testing.expectEqual(@as(u64, 0), l.claim("a", "", test_owner, 0).fresh);
    const long = &@as([max_route_len + 1]u8, @splat(0x72));
    try testing.expectEqual(@as(u64, 0), l.claim("a", long, test_owner, 0).fresh);
}

test "ring eviction keeps a bounded replay set" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    for (0..capacity * 2) |i| {
        var buf: [16]u8 = undefined;
        const k = try std.fmt.bufPrint(&buf, "k{d}", .{i});
        const t = try claimKey(&l, k, 0);
        l.complete(k, test_route, test_owner, t, 1, "200 OK", "text/html", "body");
    }
    // The first `capacity` keys are gone; the newest `capacity` replay.
    // Claiming a live key leaves the ring untouched, so the replays are
    // asserted first: the fresh claims below evict live slots.
    for (capacity..capacity * 2) |i| {
        var buf: [16]u8 = undefined;
        const k = try std.fmt.bufPrint(&buf, "k{d}", .{i});
        const c = l.claim(k, test_route, test_owner, 2);
        try testing.expect(c == .replay);
        c.replay.deinit(testing.allocator);
    }
    for (0..capacity) |i| {
        var buf: [16]u8 = undefined;
        const k = try std.fmt.bufPrint(&buf, "k{d}", .{i});
        try testing.expect(l.claim(k, test_route, test_owner, 2) == .fresh);
    }
}

test "oversized body still records completion" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    const big = &@as([max_body_len + 1]u8, @splat(0x79));
    const t = try claimKey(&l, "a", 0);
    l.complete("a", test_route, test_owner, t, 0, "200 OK", "text/html", big);
    const c = l.claim("a", test_route, test_owner, 1);
    try testing.expect(c == .replay);
    defer c.replay.deinit(testing.allocator);
    try testing.expectEqualStrings("", c.replay.body);
}

test "a full ring never steals a live in-flight claim" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    // Fill every slot with a request that has not finished yet.
    var labels: [capacity][16]u8 = undefined;
    var label_lens: [capacity]usize = undefined;
    var tokens: [capacity]u64 = undefined;
    for (0..capacity) |i| {
        label_lens[i] = (try std.fmt.bufPrint(&labels[i], "k{d}", .{i})).len;
        tokens[i] = try claimKey(&l, labels[i][0..label_lens[i]], 0);
        try testing.expect(tokens[i] != 0);
    }

    // A key that needs a slot cannot take a live one, so it runs unclaimed
    // rather than evicting a request that is still executing. Reclaiming one
    // would let its owner's retry re-enter as fresh and run the same mutation
    // a second time concurrently.
    const overflow = l.claim("overflow", test_route, test_owner, 1);
    try testing.expectEqual(@as(u64, 0), overflow.fresh);

    // Every in-flight label still reports the duplicate, not a fresh claim.
    for (0..capacity) |i| {
        try testing.expectEqual(Claim{ .duplicate = {} }, l.claim(labels[i][0..label_lens[i]], test_route, test_owner, 1));
    }

    // The request the overflow was not allowed to evict can still complete,
    // and its owner still holds its claim.
    const first = labels[0][0..label_lens[0]];
    l.complete(first, test_route, test_owner, tokens[0], 2, "200 OK", "text/html", "first");
    const replayed = l.claim(first, test_route, test_owner, 3);
    try testing.expect(replayed == .replay);
    replayed.replay.deinit(testing.allocator);

    // Once slots expire, the ring takes them again.
    const reclaimed = l.claim("overflow", test_route, test_owner, in_flight_ttl_ms);
    try testing.expect(reclaimed == .fresh);
    try testing.expect(reclaimed.fresh != 0);
}

test "slot reuse does not leak the previous body" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    for (0..capacity + 4) |i| {
        var buf: [16]u8 = undefined;
        const k = try std.fmt.bufPrint(&buf, "k{d}", .{i});
        const t = try claimKey(&l, k, 0);
        l.complete(k, test_route, test_owner, t, 0, "200 OK", "text/html", "secret-body");
    }
    // One body per live slot, never one per completed operation.
    try testing.expectEqual(capacity, countStoredBodies(&l));
}

fn countStoredBodies(l: *const Ledger) usize {
    var n: usize = 0;
    for (&l.slots) |*s| {
        if (s.body.len != 0) n += 1;
    }
    return n;
}

test "the same key from another principal does not replay this one's response" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    const t = try claimKey(&l, "a", 0);
    l.complete("a", test_route, test_owner, t, 1, "200 OK", "text/html", "secret reply");

    // A second holder presenting the same id must run its own operation and
    // must never be handed the first one's stored body.
    const other = l.claim("a", test_route, other_owner, 2);
    try testing.expect(other == .fresh);
    try testing.expect(other.fresh != 0);

    // The original principal still replays its own response.
    const c = l.claim("a", test_route, test_owner, 3);
    try testing.expect(c == .replay);
    defer c.replay.deinit(testing.allocator);
    try testing.expectEqualStrings("secret reply", c.replay.body);
}

test "another principal cannot suppress an in-flight claim or drop it" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    const t = try claimKey(&l, "a", 0);

    // Present the same key from a second principal while the first is running:
    // it gets its own fresh claim rather than a `duplicate` refusal, and its
    // token differs, so neither can complete or release the other's slot.
    const other = l.claim("a", test_route, other_owner, 1);
    try testing.expect(other == .fresh);
    try testing.expect(other.fresh != t);

    // Releasing the other principal's claim leaves the first one in flight.
    l.release("a", test_route, other_owner, other.fresh);
    l.release("a", test_route, other_owner, t);
    try testing.expectEqual(Claim{ .duplicate = {} }, l.claim("a", test_route, test_owner, 1));

    // Completing under the wrong owner stores nothing: the first principal's
    // slot keeps its own in-flight state and is still released by its own token.
    l.complete("a", test_route, other_owner, t, 1, "500 Internal Server Error", "text/html", "poisoned");
    l.release("a", test_route, test_owner, t);
    try testing.expect(l.claim("a", test_route, test_owner, 1) == .fresh);
}

test "a recycled slot stamps the new owner, not the stale one" {
    var l = Ledger.init(testing.allocator);
    defer l.deinit();

    const t = try claimKey(&l, "a", 0);
    l.complete("a", test_route, test_owner, t, 0, "200 OK", "text/html", "body");
    l.release("a", test_route, test_owner, t);

    // The slot is free again; reusing the same id under a different principal
    // takes the slot with the new owner stamped in, not the stale one.
    const other = l.claim("a", test_route, other_owner, 1);
    try testing.expect(other == .fresh);
    l.complete("a", test_route, other_owner, other.fresh, 1, "200 OK", "text/html", "second body");
    const c = l.claim("a", test_route, other_owner, 2);
    try testing.expect(c == .replay);
    defer c.replay.deinit(testing.allocator);
    try testing.expectEqualStrings("second body", c.replay.body);
}
