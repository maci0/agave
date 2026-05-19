//! Network transport for distributed tensor/pipeline parallelism.
//! Supports TCP (cross-node), POSIX shared memory (same-node, zero-copy),
//! and NCCL (GPU-optimized collectives over RoCE RDMA / TCP sockets).

const std = @import("std");
const Allocator = std.mem.Allocator;
const c = std.c;
const posix = std.posix;

const max_peers: usize = 8;
const shm_buf_size: usize = 16 * 1024 * 1024; // 16MB max per transfer
const shm_region_size: usize = shm_buf_size + 64; // data + header

const builtin = @import("builtin");
const shm_O_CREAT: c_int = if (builtin.os.tag == .macos) 0x200 else 0o100;
const shm_O_RDWR: c_int = 0o2;
const shm_PROT_RW: c_int = 0x1 | 0x2;
const shm_MAP_SHARED: c_int = 0x01;

pub const TransportKind = enum { tcp, shm, nccl, rccl };

fn getenv(name: []const u8) ?[]const u8 {
    var buf: [128:0]u8 = undefined;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    const val = c.getenv(&buf);
    if (val == null) return null;
    const ptr: [*:0]const u8 = @ptrCast(val.?);
    return std.mem.sliceTo(ptr, 0);
}

// NCCL types and constants
const NcclComm = ?*anyopaque;
const NcclUniqueId = [128]u8;
const NcclResult = c_int;
const ncclSuccess: NcclResult = 0;
const ncclFloat: c_int = 7;
const ncclSum: c_int = 0;
const FnNcclGetUniqueId = *const fn (*NcclUniqueId) callconv(.c) NcclResult;
const FnNcclCommInitRank = *const fn (*NcclComm, c_int, *const NcclUniqueId, c_int) callconv(.c) NcclResult;
const FnNcclAllReduce = *const fn (*const anyopaque, *anyopaque, usize, c_int, c_int, NcclComm, ?*anyopaque) callconv(.c) NcclResult;
const FnNcclSend = *const fn (*const anyopaque, usize, c_int, c_int, NcclComm, ?*anyopaque) callconv(.c) NcclResult;
const FnNcclRecv = *const fn (*anyopaque, usize, c_int, c_int, NcclComm, ?*anyopaque) callconv(.c) NcclResult;
const FnNcclCommDestroy = *const fn (NcclComm) callconv(.c) NcclResult;
const FnNcclGroupStart = *const fn () callconv(.c) NcclResult;
const FnNcclGroupEnd = *const fn () callconv(.c) NcclResult;

/// Shared memory region header (64-byte aligned).
const ShmHeader = extern struct {
    ready: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    size: u32 = 0,
    _pad: [56]u8 = [_]u8{0} ** 56,
};

pub const Transport = struct {
    kind: TransportKind,
    rank: u32,
    world_size: u32,
    allocator: Allocator,
    tcp_fds: [max_peers]c_int = .{-1} ** max_peers,
    tcp_connected: u32 = 0,
    recv_buf: ?[]f32 = null,
    // Shared memory regions: send_region for outgoing, recv_region for incoming
    shm_send: ?[*]align(64) u8 = null,
    shm_recv: ?[*]align(64) u8 = null,
    shm_send_fd: c_int = -1,
    shm_recv_fd: c_int = -1,
    shm_name_send: [32:0]u8 = [_:0]u8{0} ** 32,
    shm_name_recv: [32:0]u8 = [_:0]u8{0} ** 32,
    // NCCL communicator + function pointers
    nccl_comm: NcclComm = null,
    nccl_lib: ?std.DynLib = null,
    nccl_allreduce: ?FnNcclAllReduce = null,
    nccl_send: ?FnNcclSend = null,
    nccl_recv: ?FnNcclRecv = null,
    nccl_destroy: ?FnNcclCommDestroy = null,
    nccl_group_start: ?FnNcclGroupStart = null,
    nccl_group_end: ?FnNcclGroupEnd = null,
    nccl_get_unique_id: ?FnNcclGetUniqueId = null,
    nccl_comm_init_rank: ?FnNcclCommInitRank = null,
    nccl_unique_id: NcclUniqueId = undefined,
    cuda_sync: ?*const fn () callconv(.c) c_int = null,
    cuda_ctx: ?*anyopaque = null,
    cuda_ctx_set: ?*const fn (?*anyopaque) callconv(.c) c_int = null,
    cuda_host_register: ?*const fn (*const anyopaque, usize, c_uint) callconv(.c) c_int = null,
    cuda_mem_alloc: ?*const fn (*u64, usize) callconv(.c) c_int = null,
    cuda_memcpy_htod: ?*const fn (u64, *const anyopaque, usize) callconv(.c) c_int = null,
    cuda_memcpy_dtoh: ?*const fn (*anyopaque, u64, usize) callconv(.c) c_int = null,
    nccl_dev_buf: u64 = 0,
    nccl_dev_buf_size: usize = 0,
    /// Device capabilities for topology-aware partitioning.
    local_mem: usize = 0,
    peer_mem: usize = 0,

    /// Opaque backend pointer for device pointer lookup.
    cuda_backend: ?*anyopaque = null,
    /// Function to get device pointer: fn(backend, host_ptr) -> device_ptr.
    cuda_get_dev_ptr: ?*const fn (*anyopaque, [*]const f32) u64 = null,

    pub fn init(allocator: Allocator, kind: TransportKind, rank: u32, world_size: u32) !Transport {
        if (kind != .tcp and kind != .shm and kind != .nccl) return error.NotImplemented;
        return .{ .kind = kind, .rank = rank, .world_size = world_size, .allocator = allocator };
    }

    pub fn deinit(self: *Transport) void {
        if (self.nccl_comm != null) {
            if (self.nccl_destroy) |destroy| _ = destroy(self.nccl_comm);
        }
        if (self.nccl_lib) |*lib| lib.close();
        for (self.tcp_fds[0..self.tcp_connected]) |fd| {
            if (fd >= 0) _ = c.close(fd);
        }
        if (self.recv_buf) |buf| self.allocator.free(buf);
        if (self.shm_send) |ptr| _ = posix.system.munmap(@ptrCast(ptr), shm_region_size);
        if (self.shm_recv) |ptr| _ = posix.system.munmap(@ptrCast(ptr), shm_region_size);
        if (self.shm_send_fd >= 0) {
            _ = std.c.close(self.shm_send_fd);
            _ = std.c.shm_unlink(&self.shm_name_send);
        }
        if (self.shm_recv_fd >= 0) {
            _ = std.c.close(self.shm_recv_fd);
        }
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

    /// Set up POSIX shared memory regions for same-node IPC.
    /// Creates two shm regions: one for sending (this rank writes), one for receiving (peer writes).
    /// Rank 0 creates send=agave_0to1, opens recv=agave_1to0 (and vice versa for rank 1).
    pub fn setupShm(self: *Transport) !void {
        const send_name = if (self.rank == 0) "/agave_0to1" else "/agave_1to0";
        const recv_name = if (self.rank == 0) "/agave_1to0" else "/agave_0to1";
        @memset(&self.shm_name_send, 0);
        @memset(&self.shm_name_recv, 0);
        @memcpy(self.shm_name_send[0..send_name.len], send_name);
        @memcpy(self.shm_name_recv[0..recv_name.len], recv_name);

        // Create send region
        self.shm_send_fd = std.c.shm_open(&self.shm_name_send, shm_O_CREAT | shm_O_RDWR, @as(c.mode_t, 0o666));
        if (self.shm_send_fd < 0) return error.ShmOpenFailed;
        _ = std.c.ftruncate(self.shm_send_fd, @intCast(shm_region_size));
        const send_ptr = posix.system.mmap(null, shm_region_size, @bitCast(shm_PROT_RW), @bitCast(shm_MAP_SHARED), self.shm_send_fd, 0);
        if (send_ptr == posix.system.MAP_FAILED) return error.ShmMmapFailed;
        self.shm_send = @ptrCast(@alignCast(send_ptr));
        // Zero the header
        const send_hdr: *ShmHeader = @ptrCast(@alignCast(self.shm_send.?));
        send_hdr.ready.store(0, .release);
        send_hdr.size = 0;

        // Wait for peer to create their send region (our recv), then open it
        var retry: u32 = 0;
        while (retry < 5000) : (retry += 1) {
            self.shm_recv_fd = std.c.shm_open(&self.shm_name_recv, shm_O_RDWR, @as(c.mode_t, 0o666));
            if (self.shm_recv_fd >= 0) break;
            var ts = posix.system.timespec{ .sec = 0, .nsec = 1_000_000 };
            _ = posix.system.nanosleep(&ts, null);
        }
        if (self.shm_recv_fd < 0) return error.ShmPeerTimeout;
        const recv_ptr = posix.system.mmap(null, shm_region_size, @bitCast(shm_PROT_RW), @bitCast(shm_MAP_SHARED), self.shm_recv_fd, 0);
        if (recv_ptr == posix.system.MAP_FAILED) return error.ShmMmapFailed;
        self.shm_recv = @ptrCast(@alignCast(recv_ptr));
        std.log.info("shm: connected ({s} → {s})", .{ send_name, recv_name });
    }

    fn shmSend(self: *Transport, data: [*]const u8, byte_len: usize) void {
        const send = self.shm_send orelse return;
        const hdr: *ShmHeader = @ptrCast(@alignCast(send));
        // Spin until receiver consumed previous message
        while (hdr.ready.load(.acquire) != 0) std.atomic.spinLoopHint();
        const payload = send + 64;
        @memcpy(payload[0..byte_len], data[0..byte_len]);
        hdr.size = @intCast(byte_len);
        hdr.ready.store(1, .release);
    }

    fn shmRecv(self: *Transport, data: [*]u8, byte_len: usize) void {
        const recv = self.shm_recv orelse return;
        const hdr: *ShmHeader = @ptrCast(@alignCast(recv));
        // Spin until sender has data ready
        while (hdr.ready.load(.acquire) == 0) std.atomic.spinLoopHint();
        const payload = recv + 64;
        @memcpy(data[0..byte_len], payload[0..byte_len]);
        hdr.ready.store(0, .release);
    }

    /// Initialize NCCL communicator. Requires TCP connection for unique ID exchange.
    /// Call after connectPeer/acceptPeer establishes TCP link.
    pub fn setupNccl(self: *Transport) !void {
        var lib = std.DynLib.open("libnccl.so.2") catch
            std.DynLib.open("libnccl.so") catch
            std.DynLib.open("/usr/lib/aarch64-linux-gnu/libnccl.so.2") catch
            std.DynLib.open("/usr/lib/x86_64-linux-gnu/libnccl.so.2") catch
            return error.NcclNotAvailable;

        const getUniqueId = lib.lookup(FnNcclGetUniqueId, "ncclGetUniqueId") orelse return error.NcclNotAvailable;
        const commInitRank = lib.lookup(FnNcclCommInitRank, "ncclCommInitRank") orelse return error.NcclNotAvailable;
        self.nccl_allreduce = lib.lookup(FnNcclAllReduce, "ncclAllReduce");
        self.nccl_send = lib.lookup(FnNcclSend, "ncclSend");
        self.nccl_recv = lib.lookup(FnNcclRecv, "ncclRecv");
        self.nccl_destroy = lib.lookup(FnNcclCommDestroy, "ncclCommDestroy");
        self.nccl_group_start = lib.lookup(FnNcclGroupStart, "ncclGroupStart");
        self.nccl_group_end = lib.lookup(FnNcclGroupEnd, "ncclGroupEnd");

        if (self.nccl_allreduce == null) return error.NcclNotAvailable;
        self.nccl_get_unique_id = getUniqueId;
        self.nccl_comm_init_rank = commInitRank;

        // Exchange unique ID over TCP (both ranks must do this synchronously)
        if (self.rank == 0) {
            if (getUniqueId(&self.nccl_unique_id) != ncclSuccess) return error.NcclInitFailed;
            const id_bytes: [*]const u8 = @ptrCast(&self.nccl_unique_id);
            var sent: usize = 0;
            while (sent < 128) {
                const rc = c.send(self.tcp_fds[0], id_bytes + sent, 128 - sent, 0);
                if (rc <= 0) return error.NcclInitFailed;
                sent += @intCast(rc);
            }
        } else {
            const id_bytes: [*]u8 = @ptrCast(&self.nccl_unique_id);
            var got: usize = 0;
            while (got < 128) {
                const rc = c.recv(self.tcp_fds[0], id_bytes + got, 128 - got, 0);
                if (rc <= 0) return error.NcclInitFailed;
                got += @intCast(rc);
            }
        }

        self.nccl_lib = lib;

        // Query NCCL version for display
        const FnNcclGetVersion = *const fn (*c_int) callconv(.c) NcclResult;
        var nccl_ver: c_int = 0;
        if (lib.lookup(FnNcclGetVersion, "ncclGetVersion")) |getVer| _ = getVer(&nccl_ver);
        const major = @as(u32, @intCast(nccl_ver)) / 10000;
        const minor = (@as(u32, @intCast(nccl_ver)) % 10000) / 100;
        const patch = @as(u32, @intCast(nccl_ver)) % 100;
        std.log.info("NCCL: rank {d}/{d} ready — v{d}.{d}.{d}, deferred init, ID exchanged over TCP fd={d}", .{
            self.rank, self.world_size, major, minor, patch, self.tcp_fds[0],
        });

        // Log NCCL environment variables for debugging transport selection
        const nccl_env_vars = [_][]const u8{
            "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX",
            "NCCL_NET_GDR_LEVEL", "NCCL_IB_AR_THRESHOLD", "NCCL_IB_PCI_RELAXED_ORDERING",
            "NCCL_IB_TIMEOUT", "NCCL_IB_RETRY_CNT", "NCCL_DEBUG",
            "NCCL_P2P_LEVEL", "NCCL_SHM_DISABLE", "NCCL_ALGO", "NCCL_PROTO",
        };
        for (nccl_env_vars) |name| {
            if (getenv(name)) |val| {
                std.log.info("NCCL env: {s}={s}", .{ name, val });
            }
        }
    }

    /// Lazily initialize NCCL communicator. Called on first NCCL operation.
    pub fn ensureNcclComm(self: *Transport) void {
        if (self.nccl_comm != null) return;
        // TCP barrier: both ranks must reach this point before ncclCommInitRank
        // (which is a blocking collective requiring all ranks to participate)
        if (self.tcp_connected > 0) {
            var sync_byte: [1]u8 = .{0x42};
            var recv_byte: [1]u8 = undefined;
            const fd = self.tcp_fds[0];
            _ = c.send(fd, &sync_byte, 1, 0);
            _ = c.recv(fd, &recv_byte, 1, 0);
            std.log.info("NCCL: rank {d} barrier passed, initializing comm", .{self.rank});
        }
        if (self.nccl_comm_init_rank) |initRank| {
            const rc = initRank(&self.nccl_comm, @intCast(self.world_size), &self.nccl_unique_id, @intCast(self.rank));
            if (rc != ncclSuccess) {
                std.log.warn("NCCL ncclCommInitRank failed: rc={d} (cuda_ctx={}, cuda_mem_alloc={})", .{
                    rc, self.cuda_ctx != null, self.cuda_mem_alloc != null,
                });
                self.kind = .tcp;
                return;
            }
            if (self.cuda_ctx_set) |setCtx| _ = setCtx(self.cuda_ctx);
            std.log.info("NCCL: rank {d}/{d} communicator ready (group_ops={}, dev_buf={})", .{
                self.rank, self.world_size,
                self.nccl_group_start != null,
                self.nccl_dev_buf != 0,
            });
        }
    }

    pub fn allReduceAdd(self: *Transport, buf: [*]f32, n: usize) !void {
        if (self.kind == .nccl) {
            // Lazy init: create communicator on first use (after CUDA kernels have run)
            self.ensureNcclComm();
            if (self.nccl_comm == null) {
                return self.allReduceAdd(buf, n);
            }
            // Get CUDA device pointer — if buf is dirty on device, use it directly.
            // If stale (CPU fallback wrote to host), fall back to TCP for this call.
            const dptr: u64 = if (self.cuda_get_dev_ptr) |getPtr|
                if (self.cuda_backend) |be| getPtr(be, buf) else 0
            else
                0;
            if (dptr != 0) {
                // GPU path: allReduce directly on device memory (fastest)
                if (self.nccl_allreduce) |allreduce|
                    _ = allreduce(@ptrFromInt(dptr), @ptrFromInt(dptr), n, ncclFloat, ncclSum, self.nccl_comm, null);
            } else {
                // CPU fallback wrote to host — upload to device staging, NCCL allReduce, download
                if (self.cuda_sync) |sync| _ = sync();
                const byte_len = n * @sizeOf(f32);
                if (self.nccl_dev_buf_size < byte_len) {
                    if (self.cuda_mem_alloc) |alloc| _ = alloc(&self.nccl_dev_buf, byte_len);
                    self.nccl_dev_buf_size = byte_len;
                }
                if (self.nccl_dev_buf != 0) {
                    if (self.cuda_memcpy_htod) |htod| _ = htod(self.nccl_dev_buf, @ptrCast(buf), byte_len);
                    if (self.nccl_allreduce) |allreduce|
                        _ = allreduce(@ptrFromInt(self.nccl_dev_buf), @ptrFromInt(self.nccl_dev_buf), n, ncclFloat, ncclSum, self.nccl_comm, null);
                    if (self.cuda_sync) |sync| _ = sync();
                    if (self.cuda_memcpy_dtoh) |dtoh| _ = dtoh(@ptrCast(buf), self.nccl_dev_buf, byte_len);
                }
            }
            return;
        }
        if (self.kind == .shm) {
            const byte_len = n * @sizeOf(f32);
            if (byte_len > shm_buf_size) return error.BufferTooLarge;
            if (self.recv_buf == null or self.recv_buf.?.len < n) {
                if (self.recv_buf) |old| self.allocator.free(old);
                self.recv_buf = try self.allocator.alloc(f32, n);
            }
            const recv = self.recv_buf.?;
            self.shmSend(@ptrCast(buf), byte_len);
            self.shmRecv(@ptrCast(recv.ptr), byte_len);
            for (0..n) |i| buf[i] += recv[i];
            return;
        }
        if (self.tcp_connected > 0) self.tcpAllReduce(buf, n);
    }

    fn tcpAllReduce(self: *Transport, buf: [*]f32, n: usize) void {
        if (self.tcp_connected == 0) return;
        const byte_len = n * @sizeOf(f32);

        if (self.recv_buf == null or self.recv_buf.?.len < n) {
            if (self.recv_buf) |old| self.allocator.free(old);
            self.recv_buf = self.allocator.alloc(f32, n) catch return;
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

    /// Point-to-point send: send buffer to peer.
    pub fn sendBuf(self: *Transport, buf: [*]const f32, n: usize) void {
        const byte_len = n * @sizeOf(f32);
        if (self.kind == .nccl and self.nccl_send != null) {
            self.ensureNcclComm();
            if (self.nccl_comm == null) { self.tcpSend(buf, byte_len); return; }
            const peer: c_int = if (self.rank == 0) 1 else 0;
            // Try device pointer first (avoids host→device copy)
            const dptr: u64 = if (self.cuda_get_dev_ptr) |getPtr|
                if (self.cuda_backend) |be| getPtr(be, buf) else 0
            else
                0;
            if (dptr != 0) {
                _ = self.nccl_send.?(@ptrFromInt(dptr), n, ncclFloat, peer, self.nccl_comm, null);
            } else {
                // Host staging: upload then send
                if (self.nccl_dev_buf_size < byte_len) {
                    if (self.cuda_mem_alloc) |alloc| _ = alloc(&self.nccl_dev_buf, byte_len);
                    self.nccl_dev_buf_size = byte_len;
                }
                if (self.nccl_dev_buf != 0) {
                    if (self.cuda_memcpy_htod) |htod| _ = htod(self.nccl_dev_buf, @ptrCast(buf), byte_len);
                    _ = self.nccl_send.?(@ptrFromInt(self.nccl_dev_buf), n, ncclFloat, peer, self.nccl_comm, null);
                }
            }
            // Single sync after all sends (not per-send)
            if (self.cuda_sync) |sync| _ = sync();
            return;
        }
        if (self.kind == .shm) { self.shmSend(@ptrCast(buf), byte_len); return; }
        if (self.tcp_connected == 0) return;
        const fd = self.tcp_fds[0];
        const data: [*]const u8 = @ptrCast(buf);
        var sent: usize = 0;
        while (sent < byte_len) {
            const rc = c.send(fd, data + sent, byte_len - sent, 0);
            if (rc <= 0) return;
            sent += @intCast(rc);
        }
    }

    /// Point-to-point recv: receive buffer from peer.
    /// Batched send: send multiple buffers sequentially.
    /// Each upload + ncclSend uses the shared staging buffer.
    pub fn sendBufs(self: *Transport, bufs: []const [*]const f32, lens: []const usize) void {
        for (bufs, lens) |buf, n| self.sendBuf(buf, n);
    }

    /// Batched recv: receive multiple buffers sequentially.
    /// Cannot group NCCL recvs with a single staging buffer — each needs its own.
    pub fn recvBufs(self: *Transport, bufs: []const [*]f32, lens: []const usize) void {
        for (bufs, lens) |buf, n| self.recvBuf(buf, n);
    }

    fn tcpSend(self: *Transport, buf: [*]const f32, byte_len: usize) void {
        if (self.tcp_connected == 0) return;
        const fd = self.tcp_fds[0];
        const data: [*]const u8 = @ptrCast(buf);
        var sent: usize = 0;
        while (sent < byte_len) {
            const rc = c.send(fd, data + sent, byte_len - sent, 0);
            if (rc <= 0) return;
            sent += @intCast(rc);
        }
    }

    pub fn recvBuf(self: *Transport, buf: [*]f32, n: usize) void {
        const byte_len = n * @sizeOf(f32);
        if (self.kind == .nccl and self.nccl_recv != null) {
            self.ensureNcclComm();
            if (self.nccl_comm == null) { self.tcpRecv(buf, byte_len); return; }
            const peer: c_int = if (self.rank == 0) 1 else 0;
            if (self.nccl_dev_buf_size < byte_len) {
                if (self.cuda_mem_alloc) |alloc| _ = alloc(&self.nccl_dev_buf, byte_len);
                self.nccl_dev_buf_size = byte_len;
            }
            if (self.nccl_dev_buf != 0) {
                _ = self.nccl_recv.?(@ptrFromInt(self.nccl_dev_buf), n, ncclFloat, peer, self.nccl_comm, null);
                if (self.cuda_sync) |sync| _ = sync();
                if (self.cuda_memcpy_dtoh) |dtoh| _ = dtoh(@ptrCast(buf), self.nccl_dev_buf, byte_len);
            }
            return;
        }
        if (self.kind == .shm) { self.shmRecv(@ptrCast(buf), byte_len); return; }
        self.tcpRecv(buf, byte_len);
    }

    fn tcpRecv(self: *Transport, buf: [*]f32, byte_len: usize) void {
        if (self.tcp_connected == 0) return;
        const fd = self.tcp_fds[0];
        const data: [*]u8 = @ptrCast(buf);
        var got: usize = 0;
        while (got < byte_len) {
            const rc = c.recv(fd, data + got, byte_len - got, 0);
            if (rc <= 0) return;
            got += @intCast(rc);
        }
    }
};
