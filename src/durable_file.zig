//! Crash-safe file replace for operator-facing artifacts.
//!
//! Write a sibling `*.tmp.<pid>`, fsync, rename over the live path, then fsync the
//! parent directory. A crash mid-write leaves the previous live file intact
//! (or no file on first write). Used by calibration output, conversation
//! store, Vulkan pipeline cache, expert profiles, and Hub download publish.
//!
//! Not a hot-path helper: callers are one-shot CLI/server I/O.
//!
//! The publish sequence is fault-injectable (see `armFault`) so a deterministic
//! simulator can reproduce a crash or an I/O failure at any step from a single
//! (step, mode) seed, then restart from the on-disk state and assert the live
//! file was never torn.

const std = @import("std");
const builtin = @import("builtin");

/// fsync is a no-op on targets without a real POSIX file descriptor.
const posix_sync = builtin.os.tag != .wasi and builtin.os.tag != .freestanding;

// ── Deterministic fault injection ────────────────────────────────────────
//
// `replace` is the single chokepoint every crash-safe durable write goes
// through, so it is where a simulator arms a reproducible fault. Faults are
// keyed to a named step in the publish sequence, never random, so a failing
// scenario replays from (step, mode) alone. The fault state is process-global
// and disarmed by default: with nothing armed, `replace` executes exactly the
// production sequence below.
//
// `fail` models a recoverable I/O error the running process observes and
// cleans up (the `errdefer` still removes the tmp). `crash` models process
// death: no cleanup runs, the partially-written tmp is left on disk exactly as
// a real crash would leave it, and the live path is untouched unless the
// rename already landed. A simulator uses `crash` to tear down in-memory state
// and restart from disk, which is only meaningful because the live file and
// the tmp are kept distinct.

/// A point in the `replace` publish sequence where a fault can be armed.
pub const Step = enum {
    /// Before the tmp file is created.
    open,
    /// After the tmp is created/truncated, before any bytes are written.
    write,
    /// Partway through writing the bytes, before the tmp is complete.
    write_partial,
    /// After the bytes are written, before the tmp is fsynced.
    file_sync,
    /// After the tmp is fsynced, before its descriptor is closed.
    file_close,
    /// After the tmp is renamed over the live path, before the dir is fsynced.
    rename,
    /// After the parent directory is fsynced.
    dir_sync,
};

/// How an armed fault manifests.
pub const Mode = enum {
    /// Recoverable: the caller sees an error and normal cleanup runs.
    fail,
    /// Unrecoverable: models process death, so cleanup is skipped.
    crash,
};

/// 0 = disarmed. Otherwise `1 + step * 2 + mode`, matching `encodeFault`.
const fault_disarmed: u8 = 0;
var fault_armed = std.atomic.Value(u8).init(fault_disarmed);

fn encodeFault(step: Step, mode: Mode) u8 {
    return 1 + @as(u8, @intFromEnum(step)) * 2 + @as(u8, @intFromEnum(mode));
}

/// Arm `mode` at `step` for the next `replace` that reaches it. One-shot:
/// the fault fires at most once, then disarms, so a retry after a recoverable
/// fault makes progress instead of failing forever.
pub fn armFault(step: Step, mode: Mode) void {
    fault_armed.store(encodeFault(step, mode), .release);
}

/// Disarm any pending fault. Tests defer this so a fault never leaks into a
/// later test in the same process.
pub fn clearFault() void {
    fault_armed.store(fault_disarmed, .release);
}

/// Return the armed mode if `step` is the armed one, else null.
fn injectIfArmed(step: Step) ?Mode {
    const want = encodeFault(step, .fail); // low bit 0 => fail
    const crash = encodeFault(step, .crash);
    const cur = fault_armed.load(.acquire);
    if (cur != want and cur != crash) return null;
    // Consume the one-shot so a resumed sequence advances past the fault.
    _ = fault_armed.cmpxchgStrong(cur, fault_disarmed, .acq_rel, .acquire);
    return if (cur == crash) .crash else .fail;
}

/// The error a faulted step returns. `crash` is distinct so `replace`'s
/// `errdefer` can skip cleanup the way process death would.
fn faultError(mode: Mode) error{ InjectedFault, InjectedCrash }!void {
    return switch (mode) {
        .fail => error.InjectedFault,
        .crash => error.InjectedCrash,
    };
}

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

/// Mode for artifacts that are not personal data and are meant to be readable
/// by other users of the host. `mode_t` is the platform's own width (u32 on
/// Linux, u16 on macOS), which is what `std.posix.openat` takes.
pub const shared_file_mode: std.posix.mode_t = 0o644;
/// Mode for artifacts holding user data (the conversation store): owner only.
pub const private_file_mode: std.posix.mode_t = 0o600;

/// Write `data` over `path` via a sibling tmp so a crash cannot truncate the
/// live file and a second process cannot truncate this write.
pub fn replace(path: []const u8, data: []const u8) !void {
    return replaceWithMode(path, data, shared_file_mode);
}

/// `replace` for content only the owning user should read. The tmp is
/// created with `private_file_mode`, and a umask that is not more restrictive
/// cannot widen the file, so the rename publishes it owner-only.
pub fn replacePrivate(path: []const u8, data: []const u8) !void {
    return replaceWithMode(path, data, private_file_mode);
}

fn replaceWithMode(path: []const u8, data: []const u8, file_mode: std.posix.mode_t) !void {
    var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmpPath(&tmp_buf, path);

    if (injectIfArmed(.open)) |mode| return faultError(mode);
    const fd = try std.posix.openat(std.posix.AT.FDCWD, tmp_path, .{
        .ACCMODE = .WRONLY,
        .CREAT = true,
        .TRUNC = true,
    }, file_mode);
    var fd_open = true;
    errdefer |e| {
        // Whether or not the process is faulting, the descriptor is released;
        // only a recoverable fault removes the tmp. A `crash` leaves the
        // partially-written tmp on disk, as process death would.
        if (fd_open) closeFd(fd);
        if (e != error.InjectedCrash) deletePath(tmp_path);
    }

    if (injectIfArmed(.write)) |mode| return faultError(mode);
    // A full disk stops the write partway, so the bytes already on disk stay
    // there: a `crash` leaves a torn tmp, `fail` removes it.
    if (injectIfArmed(.write_partial)) |mode| {
        try writeAll(fd, data[0 .. data.len / 2]);
        return faultError(mode);
    }
    try writeAll(fd, data);

    if (injectIfArmed(.file_sync)) |mode| return faultError(mode);
    try syncFd(fd);
    // A failed close on a write path means the data may never reach disk, so
    // the rename must not publish the file. The descriptor is released even on
    // a close error, so clear `fd_open` first and let the errdefer drop the tmp.
    if (injectIfArmed(.file_close)) |mode| return faultError(mode);
    fd_open = false;
    try closeFdChecked(fd);
    try renameOver(tmp_path, path);
    if (injectIfArmed(.rename)) |mode| return faultError(mode);
    syncParent(path);
    if (injectIfArmed(.dir_sync)) |mode| return faultError(mode);
}

/// Write every byte, tolerating a short write from the kernel.
fn writeAll(fd: std.posix.fd_t, data: []const u8) !void {
    var off: usize = 0;
    while (off < data.len) {
        const n = std.posix.system.write(fd, data[off..].ptr, data.len - off);
        if (n <= 0) return error.WriteFailed;
        off += @intCast(n);
    }
}

/// Sibling tmp path for `path`, qualified with the writing process id so two
/// processes replacing the same file cannot truncate or rename each other's
/// partial write. Callers in one process must still serialize writes to the
/// same path (the server does so under its mutex).
fn tmpPath(buf: []u8, path: []const u8) ![]u8 {
    if (comptime !posix_sync) return std.fmt.bufPrint(buf, "{s}.tmp", .{path}) catch error.NameTooLong;
    return std.fmt.bufPrint(buf, "{s}.tmp.{d}", .{ path, std.c.getpid() }) catch error.NameTooLong;
}

fn closeFd(fd: std.posix.fd_t) void {
    if (comptime builtin.os.tag == .linux) {
        _ = std.posix.system.close(fd);
    } else {
        _ = std.c.close(fd);
    }
}

/// Close a descriptor and report failure. Not retried on EINTR: POSIX leaves
/// the descriptor released when close reports an error, so a retry could close
/// a descriptor another thread opened in the meantime.
fn closeFdChecked(fd: std.posix.fd_t) !void {
    const rc: isize = if (comptime builtin.os.tag == .linux)
        @intCast(std.posix.system.close(fd))
    else
        @intCast(std.c.close(fd));
    if (rc != 0) return error.CloseFailed;
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

/// Pid-unique test path. `zig build test` runs several test binaries in
/// parallel in one working directory, and every binary that links
/// `backend.zig` compiles these tests, so a shared live path lets one
/// overwrite the other's file mid-test.
fn testPath(buf: []u8, name: []const u8) []u8 {
    return std.fmt.bufPrint(buf, "test_durable_file_{d}_{s}", .{ std.c.getpid(), name }) catch unreachable;
}

test "replace round-trips bytes and removes tmp" {
    if (comptime !posix_sync) return;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "roundtrip.bin");
    const payload = "agave-durable-replace";
    try replace(path, payload);
    defer deletePath(path);

    const got = try readPath(std.testing.allocator, path);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings(payload, got);

    // Sibling tmp must not remain after a successful replace.
    var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = tmpPath(&tmp_buf, path) catch unreachable;
    const tmp_fd = std.posix.openat(std.posix.AT.FDCWD, tmp_path, .{}, 0) catch |err| {
        try std.testing.expect(err == error.FileNotFound);
        return;
    };
    closeFd(tmp_fd);
    deletePath(tmp_path);
    return error.TmpLeftBehind;
}

test "replacePrivate writes owner-only, replace writes shared" {
    if (comptime !posix_sync) return;
    var shared_buf: [std.fs.max_path_bytes]u8 = undefined;
    var private_buf: [std.fs.max_path_bytes]u8 = undefined;
    const shared_path = testPath(&shared_buf, "modeshared.bin");
    const private_path = testPath(&private_buf, "modeprivate.bin");
    try replace(shared_path, "shared");
    defer deletePath(shared_path);
    try replacePrivate(private_path, "private");
    defer deletePath(private_path);

    const prev_umask = readUmask();
    defer _ = std.c.umask(prev_umask);
    try std.testing.expectEqual(@as(u32, shared_file_mode) & ~@as(u32, prev_umask), fileMode(shared_path));
    try std.testing.expectEqual(@as(u32, private_file_mode) & ~@as(u32, prev_umask), fileMode(private_path));
}

/// Permission bits of `path`, or 0 if it cannot be stat'ed. Per-platform:
/// `std.c.stat` is not declared for arm64 darwin in this Zig release, and
/// `std.c.fstatat` is declared for darwin but empty on Linux.
fn fileMode(path: []const u8) u32 {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len >= path_buf.len) return 0;
    @memcpy(path_buf[0..path.len], path);
    path_buf[path.len] = 0;
    if (comptime builtin.os.tag == .linux) {
        var st: std.os.linux.Statx = undefined;
        const rc = std.os.linux.statx(
            std.posix.AT.FDCWD,
            @ptrCast(&path_buf),
            std.os.linux.AT.EMPTY_PATH,
            std.os.linux.STATX{ .MODE = true },
            &st,
        );
        if (rc != 0) return 0;
        return st.mode & 0o777;
    }
    var st: std.posix.Stat = undefined;
    if (std.c.fstatat(std.posix.AT.FDCWD, @ptrCast(&path_buf), &st, 0) != 0) return 0;
    return @intCast(st.mode & 0o777);
}

/// Read the process umask without changing it, so a test can assert the mode
/// the caller asked for rather than the narrower one a restrictive umask
/// leaves behind.
fn readUmask() std.c.mode_t {
    const probe: std.c.mode_t = 0o022;
    const prev = std.c.umask(probe);
    _ = std.c.umask(prev);
    return prev;
}

test "replace overwrites previous contents atomically" {
    if (comptime !posix_sync) return;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "overwrite.bin");
    try replace(path, "v1");
    defer deletePath(path);
    try replace(path, "v2-longer");
    const got = try readPath(std.testing.allocator, path);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings("v2-longer", got);
}

// ── Crash-consistency tests driven by the fault seam ─────────────────────
//
// The invariant every step must uphold: a crash anywhere in the sequence
// leaves the live file holding either the complete old contents or the
// complete new contents, never a partial write. The tmp/live split is what
// makes that true, so each step is faulted in turn and the live file re-read.

// Steps at or before the rename has not yet touched the live path.
const pre_rename_steps = [_]Step{ .open, .write, .file_sync, .file_close };
// Steps at or after the rename has already published the new contents.
const post_rename_steps = [_]Step{ .rename, .dir_sync };

test "injected crash before rename leaves the live file intact" {
    if (comptime !posix_sync) return;
    for (pre_rename_steps) |step| {
        var name_buf: [std.fs.max_path_bytes]u8 = undefined;
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "crash_pre_{s}.bin", .{@tagName(step)});
        const path = testPath(&path_buf, name);
        try replace(path, "old");
        defer deletePath(path);
        var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
        const tmp_path = try tmpPath(&tmp_buf, path);
        defer deletePath(tmp_path);

        armFault(step, .crash);
        defer clearFault();
        try std.testing.expectError(error.InjectedCrash, replace(path, "new"));
        clearFault();

        // A pre-rename crash must not disturb the live file.
        const got = try readPath(std.testing.allocator, path);
        defer std.testing.allocator.free(got);
        try std.testing.expectEqualStrings("old", got);
    }
}

test "injected crash after rename publishes the new contents" {
    if (comptime !posix_sync) return;
    for (post_rename_steps) |step| {
        var name_buf: [std.fs.max_path_bytes]u8 = undefined;
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "crash_post_{s}.bin", .{@tagName(step)});
        const path = testPath(&path_buf, name);
        try replace(path, "old");
        defer deletePath(path);

        armFault(step, .crash);
        defer clearFault();
        try std.testing.expectError(error.InjectedCrash, replace(path, "new"));
        clearFault();

        // The rename is atomic, so a post-rename crash shows the whole new
        // file, never a mixture of old and new.
        const got = try readPath(std.testing.allocator, path);
        defer std.testing.allocator.free(got);
        try std.testing.expectEqualStrings("new", got);
    }
}

test "injected crash is a one-shot that a retry clears" {
    if (comptime !posix_sync) return;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "crash_retry.bin");
    try replace(path, "old");
    defer deletePath(path);
    var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmpPath(&tmp_buf, path);
    defer deletePath(tmp_path);

    armFault(.file_sync, .crash);
    defer clearFault();
    try std.testing.expectError(error.InjectedCrash, replace(path, "new"));
    // The fault was consumed, so a restart-from-disk retry makes progress.
    clearFault();
    try replace(path, "new");
    const got = try readPath(std.testing.allocator, path);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings("new", got);
}

test "injected recoverable fault returns error and cleans the tmp" {
    if (comptime !posix_sync) return;
    for (pre_rename_steps) |step| {
        var name_buf: [std.fs.max_path_bytes]u8 = undefined;
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "fail_{s}.bin", .{@tagName(step)});
        const path = testPath(&path_buf, name);
        try replace(path, "old");
        defer deletePath(path);
        var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
        const tmp_path = try tmpPath(&tmp_buf, path);
        defer deletePath(tmp_path);

        armFault(step, .fail);
        defer clearFault();
        try std.testing.expectError(error.InjectedFault, replace(path, "new"));
        clearFault();

        // A recoverable fault keeps the live file and removes the tmp.
        const got = try readPath(std.testing.allocator, path);
        defer std.testing.allocator.free(got);
        try std.testing.expectEqualStrings("old", got);
        const leftover = std.posix.openat(std.posix.AT.FDCWD, tmp_path, .{}, 0);
        if (leftover) |fd| {
            closeFd(fd);
            return error.TmpLeftBehind;
        } else |_| {}
    }
}

test "torn tmp from a crashed partial write never reaches the live path" {
    if (comptime !posix_sync) return;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "torn_write.bin");
    try replace(path, "old");
    defer deletePath(path);
    var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try tmpPath(&tmp_buf, path);
    defer deletePath(tmp_path);

    armFault(.write_partial, .crash);
    defer clearFault();
    try std.testing.expectError(error.InjectedCrash, replace(path, "new contents"));
    clearFault();

    // A crash mid-write leaves a half-written tmp on disk, exactly as a full
    // disk would, and the live file still holds the old contents.
    const got = try readPath(std.testing.allocator, path);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings("old", got);
    const torn = try readPath(std.testing.allocator, tmp_path);
    defer std.testing.allocator.free(torn);
    try std.testing.expectEqualStrings("new co", torn);

    // A restart from disk succeeds, so the one-shot fault is recoverable.
    try replace(path, "new contents");
    const after = try readPath(std.testing.allocator, path);
    defer std.testing.allocator.free(after);
    try std.testing.expectEqualStrings("new contents", after);
}

test "fault is disarmed by default and after clearFault" {
    if (comptime !posix_sync) return;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "fault_disarmed.bin");
    defer deletePath(path);
    // Arming then clearing must restore the plain production sequence.
    armFault(.write, .crash);
    clearFault();
    try replace(path, "clean");
    const got = try readPath(std.testing.allocator, path);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings("clean", got);
}
