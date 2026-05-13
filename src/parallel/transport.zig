//! Network transport for distributed tensor parallelism.
//! TCP all-reduce between ranks. NCCL/RCCL stubs for future GPU-optimized collectives.

const std = @import("std");
const Allocator = std.mem.Allocator;
const c = std.c;

const max_peers: usize = 8;

pub const TransportKind = enum { tcp, nccl, rccl };

pub const Transport = struct {
    kind: TransportKind,
    rank: u32,
    world_size: u32,
    allocator: Allocator,
    tcp_fds: [max_peers]c_int = .{-1} ** max_peers,
    tcp_connected: u32 = 0,
    recv_buf: ?[]f32 = null,

    pub fn init(allocator: Allocator, kind: TransportKind, rank: u32, world_size: u32) !Transport {
        if (kind != .tcp) return error.NotImplemented;
        return .{ .kind = kind, .rank = rank, .world_size = world_size, .allocator = allocator };
    }

    pub fn deinit(self: *Transport) void {
        for (self.tcp_fds[0..self.tcp_connected]) |fd| {
            if (fd >= 0) _ = c.close(fd);
        }
        if (self.recv_buf) |buf| self.allocator.free(buf);
    }

    pub fn connectPeer(self: *Transport, host: [4]u8, port: u16) !void {
        if (self.tcp_connected >= max_peers) return error.TooManyPeers;
        const fd = c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
        if (fd < 0) return error.SocketFailed;

        var addr: std.posix.sockaddr.in = .{
            .port = std.mem.nativeToBig(u16, port),
            .addr = @as(u32, host[0]) | (@as(u32, host[1]) << 8) | (@as(u32, host[2]) << 16) | (@as(u32, host[3]) << 24),
        };
        if (c.connect(fd, @ptrCast(&addr), @sizeOf(@TypeOf(addr))) != 0) {
            _ = c.close(fd);
            return error.ConnectFailed;
        }
        self.tcp_fds[self.tcp_connected] = fd;
        self.tcp_connected += 1;
    }

    pub fn acceptPeer(self: *Transport, listen_fd: c_int) !void {
        if (self.tcp_connected >= max_peers) return error.TooManyPeers;
        var addr: std.posix.sockaddr.in = undefined;
        var addr_len: c.socklen_t = @sizeOf(@TypeOf(addr));
        const fd = c.accept(listen_fd, @ptrCast(&addr), &addr_len);
        if (fd < 0) return error.AcceptFailed;
        self.tcp_fds[self.tcp_connected] = fd;
        self.tcp_connected += 1;
    }

    pub fn allReduceAdd(self: *Transport, buf: [*]f32, n: usize) !void {
        if (self.kind != .tcp or self.tcp_connected == 0) return;
        const byte_len = n * @sizeOf(f32);

        if (self.recv_buf == null or self.recv_buf.?.len < n) {
            if (self.recv_buf) |old| self.allocator.free(old);
            self.recv_buf = try self.allocator.alloc(f32, n);
        }
        const recv = self.recv_buf.?;
        const fd = self.tcp_fds[0];
        const buf_u8: [*]const u8 = @ptrCast(buf);
        const recv_u8: [*]u8 = @ptrCast(recv.ptr);

        // Send
        var sent: usize = 0;
        while (sent < byte_len) {
            const rc = c.send(fd, buf_u8 + sent, byte_len - sent, 0);
            if (rc <= 0) return;
            sent += @intCast(rc);
        }
        // Recv
        var got: usize = 0;
        while (got < byte_len) {
            const rc = c.recv(fd, recv_u8 + got, byte_len - got, 0);
            if (rc <= 0) return;
            got += @intCast(rc);
        }
        // Sum
        for (0..n) |i| buf[i] += recv[i];
    }
};
