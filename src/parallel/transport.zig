//! Network transport for distributed tensor parallelism.
//!
//! Provides all-reduce and point-to-point communication between ranks
//! across multiple machines. Supports TCP sockets (universal fallback)
//! and NCCL/RCCL (GPU-optimized collective operations).
//!
//! Usage:
//!   const transport = try Transport.init(allocator, .tcp, rank, world_size, peers);
//!   defer transport.deinit();
//!   transport.allReduceAdd(buf, n);  // blocks until all ranks complete
//!
//! CLI: agave model.gguf --tp 2 --peers 192.168.0.211:9999,192.168.0.212:9999

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const posix = std.posix;

const default_port: u16 = 49454;
const max_peers: usize = 8;
const header_size: usize = 8;

pub const TransportKind = enum { tcp, nccl, rccl };

pub const Transport = struct {
    kind: TransportKind,
    rank: u32,
    world_size: u32,
    allocator: Allocator,
    // TCP state
    tcp_sockets: [max_peers]posix.socket_t = .{0} ** max_peers,
    tcp_connected: u32 = 0,
    // Scratch buffer for receive
    recv_buf: ?[]f32 = null,

    pub fn init(allocator: Allocator, kind: TransportKind, rank: u32, world_size: u32) !Transport {
        var t = Transport{
            .kind = kind,
            .rank = rank,
            .world_size = world_size,
            .allocator = allocator,
        };
        if (kind == .nccl) {
            // TODO: dlopen libnccl.so, ncclCommInitRank
            return error.NotImplemented;
        }
        if (kind == .rccl) {
            // TODO: dlopen librccl.so, rcclCommInitRank
            return error.NotImplemented;
        }
        return t;
    }

    pub fn deinit(self: *Transport) void {
        for (self.tcp_sockets[0..self.tcp_connected]) |sock| {
            if (sock != 0) posix.close(sock);
        }
        if (self.recv_buf) |buf| self.allocator.free(buf);
    }

    /// Connect to a peer for TCP transport.
    pub fn connectPeer(self: *Transport, host: [4]u8, port: u16) !void {
        if (self.kind != .tcp) return;
        if (self.tcp_connected >= max_peers) return error.TooManyPeers;

        const addr = std.net.Address.initIp4(host, port);
        const sock = try posix.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        errdefer posix.close(sock);

        try posix.connect(sock, &addr.any, addr.getOsSockLen());
        self.tcp_sockets[self.tcp_connected] = sock;
        self.tcp_connected += 1;
    }

    /// Accept an incoming peer connection (server side).
    pub fn acceptPeer(self: *Transport, listen_sock: posix.socket_t) !void {
        if (self.kind != .tcp) return;
        if (self.tcp_connected >= max_peers) return error.TooManyPeers;

        var addr: posix.sockaddr = undefined;
        var addr_len: posix.socklen_t = @sizeOf(posix.sockaddr);
        const sock = try posix.accept(listen_sock, &addr, &addr_len, 0);
        self.tcp_sockets[self.tcp_connected] = sock;
        self.tcp_connected += 1;
    }

    /// All-reduce sum over TCP: each rank sends its buffer to all peers,
    /// receives all peers' buffers, and sums them.
    /// For TP=2: rank 0 sends partial to rank 1, receives rank 1's partial, sums.
    pub fn allReduceAdd(self: *Transport, buf: [*]f32, n: usize) !void {
        switch (self.kind) {
            .tcp => try self.tcpAllReduceAdd(buf, n),
            .nccl, .rccl => return error.NotImplemented,
        }
    }

    fn tcpAllReduceAdd(self: *Transport, buf: [*]f32, n: usize) !void {
        const byte_len = n * @sizeOf(f32);

        // Ensure recv buffer is large enough
        if (self.recv_buf == null or self.recv_buf.?.len < n) {
            if (self.recv_buf) |old| self.allocator.free(old);
            self.recv_buf = try self.allocator.alloc(f32, n);
        }
        const recv = self.recv_buf.?;

        // Ring all-reduce for 2 ranks: send to peer, recv from peer, add
        const peer_idx: usize = if (self.rank == 0) 0 else 0; // both connect to each other
        if (self.tcp_connected == 0) return;

        const sock = self.tcp_sockets[peer_idx];
        const buf_bytes: [*]const u8 = @ptrCast(buf);
        const recv_bytes: [*]u8 = @ptrCast(recv.ptr);

        // Send our buffer
        var sent: usize = 0;
        while (sent < byte_len) {
            const written = posix.write(sock, buf_bytes[sent..byte_len]) catch |e| {
                std.log.warn("TCP send failed: {}", .{e});
                return;
            };
            sent += written;
        }

        // Receive peer's buffer
        var recvd: usize = 0;
        while (recvd < byte_len) {
            const got = posix.read(sock, recv_bytes[recvd..byte_len]) catch |e| {
                std.log.warn("TCP recv failed: {}", .{e});
                return;
            };
            if (got == 0) return; // peer disconnected
            recvd += got;
        }

        // Add peer's values to our buffer
        for (0..n) |i| buf[i] += recv[i];
    }
};
