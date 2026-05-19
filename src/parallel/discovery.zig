//! Zero-config peer discovery via UDP broadcast.
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

const discovery_port: u16 = 49460;
const beacon_interval_ms: u32 = 500;
const discovery_timeout_ms: u32 = 30000;
const beacon_prefix = "AGAVE-DISCOVER:";
const join_prefix = "AGAVE-JOIN:";
const max_msg_len: usize = 64;

pub const DiscoveredPeer = struct {
    addr: [4]u8,
    rank: u32,
};

/// Discover peers via UDP broadcast. Returns the peer's IP address.
/// Rank 0 broadcasts; rank 1+ listens and responds.
/// Returns null on timeout or failure.
pub fn discoverPeer(rank: u32, world_size: u32, port: u16) ?[4]u8 {
    if (world_size < 2) return null;

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

    // Set receive timeout
    const tv = posix.system.timeval{ .sec = 1, .usec = 0 };
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

    var elapsed: u32 = 0;
    while (elapsed < discovery_timeout_ms) {
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

        elapsed += beacon_interval_ms;
    }

    std.log.warn("discovery: timeout after {d}ms, no peers found", .{discovery_timeout_ms});
    return null;
}

fn discoverAsWorker(sock: c_int, rank: u32, port: u16) ?[4]u8 {
    _ = port;
    // Bind to beacon listen port
    var bind_addr: posix.sockaddr.in = .{
        .port = std.mem.nativeToBig(u16, discovery_port + 1),
        .addr = 0,
    };
    if (c.bind(sock, @ptrCast(&bind_addr), @sizeOf(@TypeOf(bind_addr))) != 0) return null;

    // Set receive timeout
    const tv = posix.system.timeval{ .sec = @intCast(discovery_timeout_ms / 1000), .usec = 0 };
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
