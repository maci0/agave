//! Zero-config peer discovery via UDP broadcast.
//!
//! Distinct from `devices/discovery.zig` (local GPU/CPU enumeration for
//! `--list-devices`). This module finds remote ranks on the LAN when `--peers`
//! is omitted.
//!
//! When --peers is not specified, rank 0 broadcasts a beacon on the LAN
//! and waits for rank 1+ to respond. Peers respond with their IP address.
//! Eliminates manual --peers configuration for same-network setups.
//!
//! Protocol:
//!   Rank 0: broadcast "AGAVE-DISCOVER:<port>:<world_size>" every 500ms
//!   Rank 1+: listen for beacon, respond with "AGAVE-JOIN:<rank>" via unicast
//!   Rank 0: collect responses until world_size peers joined

const std = @import("std");
const c = std.c;
const posix = std.posix;
const sim_clock = @import("../sim_clock.zig");

const discovery_port: u16 = 49460;
const beacon_interval_ms: u32 = 500;
const discovery_timeout_ms: u32 = 30000;
const beacon_prefix = "AGAVE-DISCOVER:";
const join_prefix = "AGAVE-JOIN:";
const max_msg_len: usize = 64;
const usec_per_ms: u32 = 1000;

/// Monotonic milliseconds for the discovery deadline. Counting assumed
/// `beacon_interval_ms` after a 1s `SO_RCVTIMEO` made a 30s timeout take ~60s.
/// Routes through sim_clock so a clock override can expire the 30s window
/// without waiting on wall-clock time (and without hanging a replay).
fn monoMilli() u64 {
    return @intCast(@max(@as(i64, 0), sim_clock.monoMilli()));
}

/// Split a millisecond interval into `timeval` seconds + microseconds.
fn msToTimeval(interval_ms: u32) posix.system.timeval {
    return .{
        .sec = @intCast(interval_ms / std.time.ms_per_s),
        .usec = @intCast((interval_ms % std.time.ms_per_s) * usec_per_ms),
    };
}

/// A peer node discovered via UDP broadcast, identified by IPv4 address and rank.
pub const DiscoveredPeer = struct {
    addr: [4]u8,
    rank: u32,
};

/// Discover peers via UDP broadcast. Returns the peer's IP address.
/// Rank 0 broadcasts; rank 1+ listens and responds.
/// Returns null on timeout or failure.
pub fn discoverPeer(rank: u32, world_size: u32, port: u16) ?[4]u8 {
    if (world_size < 2) return null;
    // Under a clock override there is no network model: expire the deadline
    // in virtual time and return immediately so SO_RCVTIMEO cannot block a
    // simulated run for 30s of wall-clock time.
    if (sim_clock.isOverridden()) {
        sim_clock.advanceMs(discovery_timeout_ms);
        return null;
    }

    const sock = c.socket(posix.AF.INET, posix.SOCK.DGRAM, 0);
    if (sock < 0) return null;
    defer _ = c.close(sock);

    // Enable broadcast
    var one: c_int = 1;
    _ = c.setsockopt(sock, posix.SOL.SOCKET, posix.SO.BROADCAST, @ptrCast(&one), @sizeOf(c_int));
    _ = c.setsockopt(sock, posix.SOL.SOCKET, posix.SO.REUSEADDR, @ptrCast(&one), @sizeOf(c_int));

    if (rank == 0) {
        return discoverAsRank0(sock, world_size, port);
    } else {
        return discoverAsWorker(sock, rank, port);
    }
}

fn discoverAsRank0(sock: c_int, world_size: u32, port: u16) ?[4]u8 {
    // Bind to discovery port to receive responses
    var bind_addr: posix.sockaddr.in = .{
        .port = std.mem.nativeToBig(u16, discovery_port),
        .addr = 0,
    };
    if (c.bind(sock, @ptrCast(&bind_addr), @sizeOf(@TypeOf(bind_addr))) != 0) return null;

    // Recv timeout matches the beacon interval so each empty poll waits ~500ms,
    // not 1s (which previously doubled the advertised 30s discovery window).
    const tv = msToTimeval(beacon_interval_ms);
    _ = c.setsockopt(sock, posix.SOL.SOCKET, posix.SO.RCVTIMEO, @ptrCast(&tv), @sizeOf(@TypeOf(tv)));

    // Broadcast address
    var bcast_addr: posix.sockaddr.in = .{
        .port = std.mem.nativeToBig(u16, discovery_port + 1),
        .addr = 0xFFFFFFFF,
    };

    // Format beacon message
    var beacon: [max_msg_len]u8 = undefined;
    const beacon_msg = std.fmt.bufPrint(&beacon, "{s}{d}:{d}", .{ beacon_prefix, port, world_size }) catch return null;

    std.log.info("discovery: broadcasting on UDP port {d}...", .{discovery_port});

    const start_ms = monoMilli();
    while (monoMilli() - start_ms < discovery_timeout_ms) {
        // Broadcast beacon
        _ = c.sendto(sock, beacon_msg.ptr, beacon_msg.len, 0, @ptrCast(&bcast_addr), @sizeOf(@TypeOf(bcast_addr)));

        // Listen for responses
        var resp: [max_msg_len]u8 = undefined;
        var from_addr: posix.sockaddr.in = undefined;
        var from_len: c.socklen_t = @sizeOf(@TypeOf(from_addr));
        const n = c.recvfrom(sock, &resp, max_msg_len, 0, @ptrCast(&from_addr), &from_len);
        if (n > 0) {
            const msg = resp[0..@intCast(n)];
            if (std.mem.startsWith(u8, msg, join_prefix)) {
                const peer_ip = @as([4]u8, @bitCast(from_addr.addr));
                std.log.info("discovery: peer joined from {d}.{d}.{d}.{d}", .{ peer_ip[0], peer_ip[1], peer_ip[2], peer_ip[3] });
                return peer_ip;
            }
        }
    }

    std.log.warn("discovery: timeout after {d}ms, no peers found", .{discovery_timeout_ms});
    return null;
}

// ── Tests ───────────────────────────────────────────────────────────

test "DiscoveredPeer, struct layout" {
    const peer = DiscoveredPeer{ .addr = .{ 192, 168, 1, 42 }, .rank = 1 };
    try @import("std").testing.expectEqual(@as(u8, 192), peer.addr[0]);
    try @import("std").testing.expectEqual(@as(u8, 168), peer.addr[1]);
    try @import("std").testing.expectEqual(@as(u8, 1), peer.addr[2]);
    try @import("std").testing.expectEqual(@as(u8, 42), peer.addr[3]);
    try @import("std").testing.expectEqual(@as(u32, 1), peer.rank);
}

test "discovery, protocol constants" {
    try @import("std").testing.expectEqual(@as(u16, 49460), discovery_port);
    try @import("std").testing.expectEqual(@as(u32, 500), beacon_interval_ms);
    try @import("std").testing.expectEqual(@as(u32, 30000), discovery_timeout_ms);
    try @import("std").testing.expectEqualStrings("AGAVE-DISCOVER:", beacon_prefix);
    try @import("std").testing.expectEqualStrings("AGAVE-JOIN:", join_prefix);
    try @import("std").testing.expectEqual(@as(usize, 64), max_msg_len);
}

test "msToTimeval splits seconds and microseconds" {
    const half = msToTimeval(beacon_interval_ms);
    try std.testing.expectEqual(@as(@TypeOf(half.sec), 0), half.sec);
    try std.testing.expectEqual(@as(@TypeOf(half.usec), 500_000), half.usec);
    const thirty = msToTimeval(discovery_timeout_ms);
    try std.testing.expectEqual(@as(@TypeOf(thirty.sec), 30), thirty.sec);
    try std.testing.expectEqual(@as(@TypeOf(thirty.usec), 0), thirty.usec);
}

test "discovery, world_size < 2 returns null" {
    // Single-node (world_size=1) should return null immediately.
    const result = discoverPeer(0, 1, 8080);
    try @import("std").testing.expectEqual(@as(?[4]u8, null), result);
}

test "discovery times out in virtual time under sim_clock override" {
    defer sim_clock.setOverrideMs(null);
    sim_clock.setOverrideMs(1_000);
    // world_size >= 2 would otherwise bind UDP and block on SO_RCVTIMEO.
    const result = discoverPeer(0, 2, 8080);
    try std.testing.expectEqual(@as(?[4]u8, null), result);
    try std.testing.expectEqual(@as(i64, 1_000 + discovery_timeout_ms), sim_clock.milliNow());
    try std.testing.expectEqual(@as(?[4]u8, null), discoverPeer(1, 2, 8080));
}

test "discovery, beacon message format" {
    // Verify the beacon message format matches the protocol spec.
    var beacon: [max_msg_len]u8 = undefined;
    const msg = std.fmt.bufPrint(&beacon, "{s}{d}:{d}", .{ beacon_prefix, @as(u16, 8080), @as(u32, 2) }) catch "";
    try @import("std").testing.expectEqualStrings("AGAVE-DISCOVER:8080:2", msg);
}

test "discovery, join message format" {
    var join_msg: [max_msg_len]u8 = undefined;
    const msg = std.fmt.bufPrint(&join_msg, "{s}{d}", .{ join_prefix, @as(u32, 1) }) catch "";
    try @import("std").testing.expectEqualStrings("AGAVE-JOIN:1", msg);
}

test "discovery, function signatures exist" {
    // world_size < 2 must short-circuit without opening sockets.
    try std.testing.expectEqual(@as(?[4]u8, null), discoverPeer(0, 0, 8080));
    try std.testing.expectEqual(@as(?[4]u8, null), discoverPeer(0, 1, 8080));
    try std.testing.expectEqual(@as(?[4]u8, null), discoverPeer(1, 1, 8080));
}

test "discovery, beacon prefix detection" {
    // Verify that std.mem.startsWith correctly identifies beacon vs join messages.
    const valid_beacon = "AGAVE-DISCOVER:8080:2";
    const valid_join = "AGAVE-JOIN:1";
    const garbage = "HTTP/1.1 200 OK";

    try std.testing.expect(std.mem.startsWith(u8, valid_beacon, beacon_prefix));
    try std.testing.expect(!std.mem.startsWith(u8, valid_beacon, join_prefix));

    try std.testing.expect(std.mem.startsWith(u8, valid_join, join_prefix));
    try std.testing.expect(!std.mem.startsWith(u8, valid_join, beacon_prefix));

    try std.testing.expect(!std.mem.startsWith(u8, garbage, beacon_prefix));
    try std.testing.expect(!std.mem.startsWith(u8, garbage, join_prefix));
}

test "discovery, beacon parses port and world_size" {
    // After stripping the beacon prefix, the remaining payload is "<port>:<world_size>".
    var beacon: [max_msg_len]u8 = undefined;
    const msg = std.fmt.bufPrint(&beacon, "{s}{d}:{d}", .{ beacon_prefix, @as(u16, 12345), @as(u32, 4) }) catch unreachable;

    // Strip prefix
    const payload = msg[beacon_prefix.len..];
    var it = std.mem.splitScalar(u8, payload, ':');
    const port_str = it.first();
    const ws_str = it.next() orelse "";

    const port_val = std.fmt.parseInt(u16, port_str, 10) catch 0;
    const ws_val = std.fmt.parseInt(u32, ws_str, 10) catch 0;

    try std.testing.expectEqual(@as(u16, 12345), port_val);
    try std.testing.expectEqual(@as(u32, 4), ws_val);
}

test "discovery, DiscoveredPeer edge addresses" {
    // Loopback
    const lo = DiscoveredPeer{ .addr = .{ 127, 0, 0, 1 }, .rank = 0 };
    try std.testing.expectEqual(@as(u8, 127), lo.addr[0]);
    try std.testing.expectEqual(@as(u32, 0), lo.rank);

    // Broadcast
    const bcast = DiscoveredPeer{ .addr = .{ 255, 255, 255, 255 }, .rank = 7 };
    try std.testing.expectEqual(@as(u8, 255), bcast.addr[3]);
    try std.testing.expectEqual(@as(u32, 7), bcast.rank);

    // Zero address
    const zero = DiscoveredPeer{ .addr = .{ 0, 0, 0, 0 }, .rank = 0 };
    try std.testing.expectEqual(@as(u8, 0), zero.addr[0]);
}

test "discovery, private function signatures exist" {
    // Join payload after prefix must parse as a rank integer.
    const join_msg = "AGAVE-JOIN:3";
    try std.testing.expect(std.mem.startsWith(u8, join_msg, join_prefix));
    const rank_str = join_msg[join_prefix.len..];
    const rank_val = try std.fmt.parseInt(u32, rank_str, 10);
    try std.testing.expectEqual(@as(u32, 3), rank_val);

    // Malformed join payload must fail parse (worker path ignores bad ranks).
    try std.testing.expectError(error.InvalidCharacter, std.fmt.parseInt(u32, "not-a-rank", 10));

    comptime {
        _ = @TypeOf(discoverAsRank0);
        _ = @TypeOf(discoverAsWorker);
    }
}

test "fuzz: all discovery functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // ── DiscoveredPeer (pub struct) ──
            const addr = [4]u8{
                smith.valueWithHash(u8, 0),
                smith.valueWithHash(u8, 1),
                smith.valueWithHash(u8, 2),
                smith.valueWithHash(u8, 3),
            };
            const rank_val = smith.valueWithHash(u32, 4);
            const peer = DiscoveredPeer{ .addr = addr, .rank = rank_val };
            try std.testing.expectEqual(addr, peer.addr);
            try std.testing.expectEqual(rank_val, peer.rank);

            // ── discoverPeer (pub fn) ──
            // Clock override expires the deadline in virtual time so fuzz
            // inputs with world_size >= 2 cannot block on SO_RCVTIMEO.
            defer sim_clock.setOverrideMs(null);
            sim_clock.setOverrideMs(1_000);
            const fuzz_rank = smith.valueWithHash(u32, 5);
            const fuzz_world = smith.valueWithHash(u32, 6);
            const fuzz_port = smith.valueWithHash(u16, 7);
            const result = discoverPeer(fuzz_rank, fuzz_world, fuzz_port);
            try std.testing.expectEqual(@as(?[4]u8, null), result);
        }
    }.f, .{});
}

fn discoverAsWorker(sock: c_int, rank: u32, port: u16) ?[4]u8 {
    _ = port;
    // Bind to beacon listen port
    var bind_addr: posix.sockaddr.in = .{
        .port = std.mem.nativeToBig(u16, discovery_port + 1),
        .addr = 0,
    };
    if (c.bind(sock, @ptrCast(&bind_addr), @sizeOf(@TypeOf(bind_addr))) != 0) return null;

    const tv = msToTimeval(discovery_timeout_ms);
    _ = c.setsockopt(sock, posix.SOL.SOCKET, posix.SO.RCVTIMEO, @ptrCast(&tv), @sizeOf(@TypeOf(tv)));

    std.log.info("discovery: listening for rank 0 beacon on UDP port {d}...", .{discovery_port + 1});

    var beacon: [max_msg_len]u8 = undefined;
    var from_addr: posix.sockaddr.in = undefined;
    var from_len: c.socklen_t = @sizeOf(@TypeOf(from_addr));
    const n = c.recvfrom(sock, &beacon, max_msg_len, 0, @ptrCast(&from_addr), &from_len);
    if (n <= 0) {
        std.log.warn("discovery: no beacon received", .{});
        return null;
    }

    const msg = beacon[0..@intCast(n)];
    if (!std.mem.startsWith(u8, msg, beacon_prefix)) return null;

    const peer_ip = @as([4]u8, @bitCast(from_addr.addr));
    std.log.info("discovery: found rank 0 at {d}.{d}.{d}.{d}", .{ peer_ip[0], peer_ip[1], peer_ip[2], peer_ip[3] });

    // Send join response back to rank 0
    var join_msg: [max_msg_len]u8 = undefined;
    const join = std.fmt.bufPrint(&join_msg, "{s}{d}", .{ join_prefix, rank }) catch return null;
    var reply_addr: posix.sockaddr.in = .{
        .port = std.mem.nativeToBig(u16, discovery_port),
        .addr = from_addr.addr,
    };
    _ = c.sendto(sock, join.ptr, join.len, 0, @ptrCast(&reply_addr), @sizeOf(@TypeOf(reply_addr)));

    return peer_ip;
}
