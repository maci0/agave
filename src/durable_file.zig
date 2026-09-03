//! Crash-safe file replace for operator-facing artifacts.
//!
//! Write a sibling `*.tmp`, fsync, rename over the live path, then fsync the
//! parent directory. A crash mid-write leaves the previous live file intact
//! (or no file on first write). Used by calibration output, conversation
//! store, Vulkan pipeline cache, expert profiles, and Hub download publish.
//!
//! Not a hot-path helper: callers are one-shot CLI/server I/O.

const std = @import("std");
const builtin = @import("builtin");

/// fsync is a no-op on targets without a real POSIX file descriptor.
const posix_sync = builtin.os.tag != .wasi and builtin.os.tag != .freestanding;

/// Flush file contents and metadata for `fd`. No-op on WASI/freestanding.
pub fn syncFd(fd: std.posix.fd_t) !void {
    if (comptime !posix_sync) return;
    while (true) {
        const rc = std.c.fsync(fd);
        if (rc == 0) return;
        switch (std.c.errno(rc)) {
            .INTR => continue,
            else => return error.FileSyncFailed,
        }
    }
}

/// Best-effort fsync of the directory that contains `path`, so a rename is
/// durable. Failures are ignored: the rename itself already succeeded.
pub fn syncParent(path: []const u8) void {
    if (comptime !posix_sync) return;
    const parent = std.fs.path.dirname(path) orelse ".";
    const fd = std.posix.openat(std.posix.AT.FDCWD, parent, .{
        .ACCMODE = .RDONLY,
        .DIRECTORY = true,
    }, 0) catch return;
    defer closeFd(fd);
    _ = std.c.fsync(fd);
}

/// Rename `old_path` over `new_path`. Both must be on the same filesystem.
pub fn renameOver(old_path: []const u8, new_path: []const u8) !void {
    var old_z: [std.fs.max_path_bytes]u8 = undefined;
    var new_z: [std.fs.max_path_bytes]u8 = undefined;
    if (old_path.len >= old_z.len or new_path.len >= new_z.len) return error.NameTooLong;
    @memcpy(old_z[0..old_path.len], old_path);
    old_z[old_path.len] = 0;
    @memcpy(new_z[0..new_path.len], new_path);
    new_z[new_path.len] = 0;
    const rc = std.c.rename(@ptrCast(old_z[0..old_path.len :0]), @ptrCast(new_z[0..new_path.len :0]));
    if (rc != 0) return error.RenameFailed;
}

/// Write `data` over `path` via `{path}.tmp` so a crash cannot truncate the live file.
pub fn replace(path: []const u8, data: []const u8) !void {
    var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmpPath(&tmp_buf, path);

    const fd = try std.posix.openat(std.posix.AT.FDCWD, tmp_path, .{
        .ACCMODE = .WRONLY,
        .CREAT = true,
        .TRUNC = true,
    }, 0o644);
    var fd_open = true;
    errdefer {
        if (fd_open) closeFd(fd);
        deletePath(tmp_path);
    }

    var off: usize = 0;
    while (off < data.len) {
        const n = std.posix.system.write(fd, data[off..].ptr, data.len - off);
        if (n <= 0) return error.WriteFailed;
        off += @intCast(n);
    }

    try syncFd(fd);
    closeFd(fd);
    fd_open = false;
    try renameOver(tmp_path, path);
    syncParent(path);
}

fn tmpPath(buf: []u8, path: []const u8) ![]u8 {
    return std.fmt.bufPrint(buf, "{s}.tmp", .{path}) catch error.NameTooLong;
}

fn closeFd(fd: std.posix.fd_t) void {
    if (comptime builtin.os.tag == .linux) {
        _ = std.posix.system.close(fd);
    } else {
        _ = std.c.close(fd);
    }
}

fn deletePath(path: []const u8) void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len >= buf.len) return;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    _ = std.c.unlink(@ptrCast(buf[0..path.len :0]));
}

fn readPath(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const fd = try std.posix.openat(std.posix.AT.FDCWD, path, .{}, 0);
    defer closeFd(fd);
    const size: usize = blk: {
        if (comptime builtin.os.tag == .linux) {
            var st: std.os.linux.Statx = undefined;
            const rc = std.os.linux.statx(fd, @ptrCast(""), std.os.linux.AT.EMPTY_PATH, std.os.linux.STATX{ .SIZE = true }, &st);
            if (rc != 0) return error.StatFailed;
            break :blk @intCast(st.size);
        } else {
            var st: std.c.Stat = undefined;
            if (std.c.fstat(fd, &st) != 0) return error.StatFailed;
            if (st.size <= 0) return error.StatFailed;
            break :blk @intCast(st.size);
        }
    };
    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);
    var got: usize = 0;
    while (got < size) {
        const n = std.posix.read(fd, buf[got..]) catch return error.ReadFailed;
        if (n == 0) break;
        got += n;
    }
    if (got != size) return error.ReadFailed;
    return buf;
}

test "replace round-trips bytes and removes tmp" {
    if (comptime !posix_sync) return;
    const path = "test_durable_file.bin";
    const payload = "agave-durable-replace";
    try replace(path, payload);
    defer deletePath(path);

    const got = try readPath(std.testing.allocator, path);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings(payload, got);

    // Sibling tmp must not remain after a successful replace.
    const tmp_fd = std.posix.openat(std.posix.AT.FDCWD, path ++ ".tmp", .{}, 0) catch |err| {
        try std.testing.expect(err == error.FileNotFound);
        return;
    };
    closeFd(tmp_fd);
    deletePath(path ++ ".tmp");
    return error.TmpLeftBehind;
}

test "replace overwrites previous contents atomically" {
    if (comptime !posix_sync) return;
    const path = "test_durable_file_overwrite.bin";
    try replace(path, "v1");
    defer deletePath(path);
    try replace(path, "v2-longer");
    const got = try readPath(std.testing.allocator, path);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings("v2-longer", got);
}
