//! Rank-to-rank peer link setup for distributed inference.
//!
//! Sits between the CLI (which picks a transport) and `transport.zig` (which
//! moves the payload). This module owns the address syntax of `--peers` and
//! the TCP handshake that runs before any bulk transfer: connect, capability
//! exchange, RTT probe, and NCCL interop wiring.
//!
//! `peer_discovery.zig` finds the peer address when `--peers` is omitted;
//! `transport.zig` is the channel itself.

const std = @import("std");
const build_options = @import("build_options");
const backend_mod = @import("../backend/backend.zig");
const sim_clock = @import("../sim_clock.zig");
const TransportMod = @import("transport.zig");

/// Bound on the rank-0/rank-1 capability and RTT handshake. A peer that
/// accepts the connection and then stalls must not hang the run forever.
const peer_handshake_timeout_ms: i32 = 5000;

pub const PeerAddr = struct { host: [4]u8, port: u16 };

/// Parse a dotted-quad IPv4 literal into `out`. Rejects empty input, leading
/// or trailing dots, empty octets, out-of-range octets, and overflow.
pub fn parseIpv4(s: []const u8, out: *[4]u8) bool {
    if (s.len == 0) return false;
    if (s[0] == '.' or s[s.len - 1] == '.') return false;
    if (std.mem.indexOf(u8, s, "..") != null) return false;
    var parts: [4]u8 = .{ 0, 0, 0, 0 };
    var part_idx: usize = 0;
    var acc: u32 = 0;
    for (s) |c| {
        if (c == '.') {
            if (acc > 255 or part_idx >= 4) return false;
            parts[part_idx] = @intCast(acc);
            part_idx += 1;
            acc = 0;
        } else if (c >= '0' and c <= '9') {
            acc = std.math.mul(u32, acc, 10) catch return false;
            acc = std.math.add(u32, acc, c - '0') catch return false;
        } else {
            return false;
        }
    }
    if (acc > 255 or part_idx != 3) return false;
    parts[3] = @intCast(acc);
    out.* = parts;
    return true;
}

/// Parse "host:port" or "host" peer address string. Returns null on invalid input.
pub fn parsePeerAddr(peers_str: []const u8, fallback_port: u16) ?PeerAddr {
    var result = PeerAddr{ .host = .{ 0, 0, 0, 0 }, .port = fallback_port };
    if (std.mem.indexOfScalar(u8, peers_str, ':')) |colon| {
        result.port = std.fmt.parseInt(u16, peers_str[colon + 1 ..], 10) catch return null;
        if (!parseIpv4(peers_str[0..colon], &result.host)) return null;
    } else {
        if (!parseIpv4(peers_str, &result.host)) return null;
    }
    return result;
}

/// Initializes and connects the distributed-inference transport for the given
/// rank and peer. `kind` is already resolved from the CLI choice. Shared memory
/// short-circuits to `setupShm`; every other kind connects over TCP first
/// (NCCL needs that connection for its unique-ID exchange). Returns null when
/// the link cannot be brought up; the caller owns the returned transport.
pub fn setup(allocator: std.mem.Allocator, peers_str: []const u8, rank: u32, world_size: u32, kind: TransportMod.TransportKind, port_base: u16, be_union: anytype) ?*TransportMod.Transport {
    const t = allocator.create(TransportMod.Transport) catch return null;
    var transport_ok = false;
    defer if (!transport_ok) allocator.destroy(t);
    t.* = TransportMod.Transport.init(allocator, kind, rank, world_size) catch return null;

    var effective_kind = kind;
    if (effective_kind == .shm) {
        t.setupShm() catch {
            std.log.warn("shm setup failed, falling back to tcp", .{});
            t.kind = .tcp;
            effective_kind = .tcp;
        };
    }

    if (effective_kind == .shm) {
        transport_ok = true;
        return t;
    }

    // NCCL: establish TCP first (for unique ID exchange), then init NCCL
    const want_nccl = (effective_kind == .nccl);

    // TCP path
    const peer = parsePeerAddr(peers_str, port_base) orelse {
        std.log.err("invalid peer address: {s}", .{peers_str});
        return null;
    };
    const host = peer.host;
    const port = peer.port;
    if (rank == 0) {
        var la: std.posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = 0 };
        const ls = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
        if (ls < 0) {
            std.log.err("could not create the rank 0 listening socket: {s}", .{@tagName(std.c.errno(ls))});
            return null;
        }
        defer _ = std.c.close(ls);
        var one: c_int = 1;
        _ = std.c.setsockopt(ls, std.posix.SOL.SOCKET, std.posix.SO.REUSEADDR, @ptrCast(&one), @sizeOf(c_int));
        // A bind failure is the most common way this setup fails, so name the
        // port and the errno rather than returning null with no output.
        const bind_rc = std.c.bind(ls, @ptrCast(&la), @sizeOf(@TypeOf(la)));
        if (bind_rc != 0) {
            std.log.err("could not bind the rank 0 listener to port {d}: {s} (already in use by another rank?)", .{
                port,
                @tagName(std.c.errno(bind_rc)),
            });
            return null;
        }
        const listen_rc = std.c.listen(ls, 1);
        if (listen_rc != 0) {
            std.log.err("could not listen on port {d}: {s}", .{ port, @tagName(std.c.errno(listen_rc)) });
            return null;
        }
        std.log.info("waiting for rank 1 on port {d}...", .{port});
        t.acceptPeer(ls) catch |err| {
            std.log.err("rank 1 never connected on port {d}: {s}", .{ port, @errorName(err) });
            return null;
        };
        std.log.info("rank 1 connected", .{});
    } else {
        std.log.info("connecting to rank 0 at {d}.{d}.{d}.{d}:{d}...", .{ host[0], host[1], host[2], host[3], port });
        t.connectPeer(host, port) catch |err| {
            std.log.err("could not reach rank 0 at {d}.{d}.{d}.{d}:{d}: {s}", .{ host[0], host[1], host[2], host[3], port, @errorName(err) });
            return null;
        };
        std.log.info("connected to rank 0", .{});
    }
    // Bound only the handshake exchanges; bulk transfers on this descriptor
    // must stay untimed so a large KV or allreduce payload is never cut short.
    setPeerSocketTimeout(t.tcp_fds[0], peer_handshake_timeout_ms);

    // Measure peer RTT via TCP ping-pong (4-byte round-trip)
    const rtt_us = measurePeerRtt(t, rank);
    if (rtt_us > 0) std.log.info("peer RTT: {d} µs", .{rtt_us});

    // Exchange device capabilities for topology-aware partitioning
    const local_mem = backend_mod.detectSystemMem();
    const peer_mem = exchangeDeviceCaps(t, rank, local_mem);
    setPeerSocketTimeout(t.tcp_fds[0], 0);
    if (peer_mem > 0) {
        // Store for topology-aware PP layer assignment
        t.peer_mem = peer_mem;
        t.local_mem = local_mem;
    }

    // NCCL: wire CUDA interop BEFORE init so ncclCommInitRank has a valid context
    if (want_nccl) {
        switch (be_union) {
            .cuda => |cuda_be| {
                if (comptime build_options.enable_cuda) {
                    t.cuda_sync = cuda_be.cuCtxSynchronize;
                    t.cuda_ctx = cuda_be.context;
                    t.cuda_ctx_set = if (cuda_be.cuCtxSetCurrent) |f| f else null;
                    t.cuda_backend = @ptrCast(cuda_be);
                    t.cuda_get_dev_ptr = backend_mod.CudaBackend.getDevicePtrOpaque;
                    t.cuda_mem_alloc = cuda_be.cuMemAlloc;
                    t.cuda_mem_free = cuda_be.cuMemFree;
                    t.cuda_memcpy_htod = cuda_be.cuMemcpyHtoD;
                    t.cuda_memcpy_dtoh = cuda_be.cuMemcpyDtoH;
                }
            },
            else => {},
        }
        t.setupNccl() catch |err| {
            std.log.warn("NCCL init failed ({s}), using TCP", .{@errorName(err)});
            t.kind = .tcp;
            transport_ok = true;
            return t;
        };
        // Both ranks are synchronized after setupNccl (TCP ID exchange).
        // Init comm NOW while both are at the same point.
        t.ensureNcclComm();
    }
    transport_ok = true;
    return t;
}

/// Set (or clear, with `timeout_ms` of 0) the send/receive timeout on a peer
/// socket. Only the fixed-size handshake exchanges are wrapped in this; bulk
/// transfers on the same descriptor run with no timeout, so a large KV or
/// allreduce payload is never cut short.
fn setPeerSocketTimeout(fd: std.posix.fd_t, timeout_ms: i32) void {
    const tv: std.c.timeval = .{
        .sec = @intCast(@divTrunc(timeout_ms, 1000)),
        .usec = @intCast(@mod(timeout_ms, 1000) * std.time.us_per_ms),
    };
    _ = std.c.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, @ptrCast(&tv), @sizeOf(@TypeOf(tv)));
    _ = std.c.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.SNDTIMEO, @ptrCast(&tv), @sizeOf(@TypeOf(tv)));
}

/// Send exactly `len` bytes on `fd`, reporting a short or failed write.
/// The rank-0/rank-1 handshake is a fixed-size exchange, so a partial write
/// means the peer is not what it claims and the run cannot continue.
fn sendExact(fd: std.posix.fd_t, bytes: []const u8) !void {
    var sent: usize = 0;
    while (sent < bytes.len) {
        const n = std.posix.system.send(fd, bytes.ptr + sent, bytes.len - sent, 0);
        if (n <= 0) return error.PeerSendFailed;
        sent += @intCast(n);
    }
}

/// Receive exactly `len` bytes into `bytes`, reporting a short or failed read.
fn recvExact(fd: std.posix.fd_t, bytes: []u8) !void {
    var got: usize = 0;
    while (got < bytes.len) {
        const n = std.posix.system.recv(fd, bytes.ptr + got, bytes.len - got, 0);
        if (n <= 0) return error.PeerRecvFailed;
        got += @intCast(n);
    }
}

/// Exchange device capabilities with peer for topology-aware partitioning.
/// Returns peer's available memory in bytes, or 0 on failure.
fn exchangeDeviceCaps(t: *TransportMod.Transport, rank: u32, local_mem: usize) usize {
    if (t.tcp_connected == 0) return 0;
    const fd = t.tcp_fds[0];
    var local_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &local_bytes, @intCast(local_mem), .little);
    // Zeroed, not undefined: a failed transfer must read as "unknown" (0)
    // rather than as stack garbage that then drives layer partitioning.
    var remote_bytes: [8]u8 = @splat(0);

    const exchanged = if (rank == 0) blk: {
        sendExact(fd, &local_bytes) catch break :blk false;
        recvExact(fd, &remote_bytes) catch break :blk false;
        break :blk true;
    } else blk: {
        recvExact(fd, &remote_bytes) catch break :blk false;
        sendExact(fd, &local_bytes) catch break :blk false;
        break :blk true;
    };
    if (!exchanged) {
        std.log.warn("device capability exchange with peer failed: transport is up but the peer did not complete the 8-byte handshake within {d}ms; assuming unknown peer memory", .{peer_handshake_timeout_ms});
        return 0;
    }
    const peer_mem = std.mem.readInt(u64, &remote_bytes, .little);
    if (peer_mem > 0) {
        std.log.info("topology: local {d} MB, peer {d} MB", .{
            local_mem / (1024 * 1024), peer_mem / (1024 * 1024),
        });
    }
    return @intCast(peer_mem);
}

/// Measure round-trip time to peer via TCP ping-pong. Returns µs, or 0 on failure.
fn measurePeerRtt(t: *TransportMod.Transport, rank: u32) u64 {
    if (t.tcp_connected == 0) return 0;
    const fd = t.tcp_fds[0];
    const ping: [4]u8 = .{ 'P', 'I', 'N', 'G' };
    var pong: [4]u8 = @splat(0);
    const t0 = sim_clock.monoNano();
    const answered = if (rank == 0) blk: {
        sendExact(fd, &ping) catch break :blk false;
        recvExact(fd, &pong) catch break :blk false;
        break :blk true;
    } else blk: {
        recvExact(fd, &pong) catch break :blk false;
        sendExact(fd, &ping) catch break :blk false;
        break :blk true;
    };
    if (!answered) {
        std.log.warn("peer RTT probe failed: no answer within {d}ms, continuing without a timing estimate", .{peer_handshake_timeout_ms});
        return 0;
    }
    if (!std.mem.eql(u8, &ping, &pong)) {
        std.log.warn("peer RTT probe got an unexpected reply, ignoring the timing estimate", .{});
        return 0;
    }
    const delta_us = @divTrunc(sim_clock.monoNano() - t0, 1000);
    return if (delta_us > 0) @intCast(delta_us) else 0;
}

test "parseIpv4 valid addresses" {
    var out: [4]u8 = undefined;
    try std.testing.expect(parseIpv4("192.168.1.1", &out));
    try std.testing.expectEqual([4]u8{ 192, 168, 1, 1 }, out);
    try std.testing.expect(parseIpv4("0.0.0.0", &out));
    try std.testing.expectEqual([4]u8{ 0, 0, 0, 0 }, out);
    try std.testing.expect(parseIpv4("255.255.255.255", &out));
    try std.testing.expectEqual([4]u8{ 255, 255, 255, 255 }, out);
    try std.testing.expect(parseIpv4("10.0.0.1", &out));
    try std.testing.expectEqual([4]u8{ 10, 0, 0, 1 }, out);
}

test "parseIpv4 invalid addresses" {
    var out: [4]u8 = undefined;
    try std.testing.expect(!parseIpv4("", &out));
    try std.testing.expect(!parseIpv4("256.0.0.1", &out));
    try std.testing.expect(!parseIpv4("1.2.3", &out));
    try std.testing.expect(!parseIpv4("1.2.3.4.5", &out));
    try std.testing.expect(!parseIpv4(".1.2.3.4", &out));
    try std.testing.expect(!parseIpv4("1.2.3.4.", &out));
    try std.testing.expect(!parseIpv4("1..2.3.4", &out));
    try std.testing.expect(!parseIpv4("abc", &out));
    try std.testing.expect(!parseIpv4("1.2.x.4", &out));
}

test "parsePeerAddr host only" {
    const pa = parsePeerAddr("10.0.0.1", 8080);
    try std.testing.expect(pa != null);
    try std.testing.expectEqual([4]u8{ 10, 0, 0, 1 }, pa.?.host);
    try std.testing.expectEqual(@as(u16, 8080), pa.?.port);
}

test "parsePeerAddr host:port" {
    const pa = parsePeerAddr("10.0.0.2:9000", 8080);
    try std.testing.expect(pa != null);
    try std.testing.expectEqual([4]u8{ 10, 0, 0, 2 }, pa.?.host);
    try std.testing.expectEqual(@as(u16, 9000), pa.?.port);
}

test "parsePeerAddr invalid" {
    try std.testing.expect(parsePeerAddr("notanip", 8080) == null);
    try std.testing.expect(parsePeerAddr("1.2.3.4:99999", 8080) == null);
    try std.testing.expect(parsePeerAddr("", 8080) == null);
}
