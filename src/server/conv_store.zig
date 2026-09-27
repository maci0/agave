//! On-disk conversation store for the HTTP server web UI.
//!
//! JSON envelope (version 1):
//!   {"version":1,"active_id":N,"next_id":N,"conversations":[...]}
//!
//! Written with `durable_file.replace` so a crash cannot truncate the live
//! file. Load is best-effort: missing file starts empty; a corrupt file is
//! quarantined to `{path}.corrupt` so the next save cannot overwrite the
//! only remaining copy.

const std = @import("std");
const Allocator = std.mem.Allocator;
const durable = @import("../durable_file.zig");
const json = @import("json.zig");
const Message = @import("../chat_template.zig").Message;
const Role = @import("../chat_template.zig").Role;

/// Current on-disk schema. Bump when the envelope is no longer readable.
pub const format_version: u32 = 1;
/// Refuse to load a store larger than this (protects against a huge corrupt file).
const max_store_bytes: usize = 64 * 1024 * 1024;
const max_conversations: usize = 100;
const max_messages_per_conv: usize = 1000;
const max_title_len: usize = 48;

/// One conversation as loaded from disk. Contents are owned by `Snapshot`.
pub const LoadedConv = struct {
    id: u32,
    title: []u8,
    messages: []Message,
};

/// Owned snapshot of the conversation list.
pub const Snapshot = struct {
    allocator: Allocator,
    active_id: u32,
    next_id: u32,
    conversations: []LoadedConv,

    /// Free titles, message contents, and the conversation slice.
    pub fn deinit(self: *Snapshot) void {
        for (self.conversations) |*conv| {
            self.allocator.free(conv.title);
            for (conv.messages) |msg| {
                const content = @constCast(msg.content);
                @memset(content, 0);
                self.allocator.free(content);
                if (msg.tool_call_id) |tcid| {
                    const t = @constCast(tcid);
                    @memset(t, 0);
                    self.allocator.free(t);
                }
            }
            self.allocator.free(conv.messages);
        }
        self.allocator.free(self.conversations);
        self.conversations = &.{};
    }
};

/// View of one in-memory conversation for `save`.
pub const ConvView = struct {
    id: u32,
    title: []const u8,
    messages: []const Message,
};

/// Default path: `$XDG_CACHE_HOME/agave/conversations.json`, else
/// `$HOME/.cache/agave/conversations.json`. Null if neither env var is set.
pub fn defaultPath(buf: []u8) ?[]u8 {
    const xdg = if (std.c.getenv("XDG_CACHE_HOME")) |c| std.mem.trim(u8, std.mem.span(c), " \t\r\n") else null;
    const home = if (std.c.getenv("HOME")) |c| std.mem.trim(u8, std.mem.span(c), " \t\r\n") else null;
    return formatDefaultPath(buf, xdg, home);
}

fn formatDefaultPath(buf: []u8, xdg: ?[]const u8, home: ?[]const u8) ?[]u8 {
    if (xdg) |dir| {
        if (dir.len > 0)
            return std.fmt.bufPrint(buf, "{s}/agave/conversations.json", .{dir}) catch null;
    }
    const h = home orelse return null;
    if (h.len == 0) return null;
    return std.fmt.bufPrint(buf, "{s}/.cache/agave/conversations.json", .{h}) catch null;
}

/// Create parent directories of `path` (mkdir -p). Best-effort.
fn ensureParent(path: []const u8) void {
    const parent = std.fs.path.dirname(path) orelse return;
    mkdirP(parent);
}

/// Serialize `convs` and atomically replace `path`.
pub fn save(
    allocator: Allocator,
    path: []const u8,
    active_id: u32,
    next_id: u32,
    convs: []const ConvView,
) !void {
    ensureParent(path);
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try buf.appendSlice(allocator, "{\"version\":");
    try buf.print(allocator, "{d},\"active_id\":{d},\"next_id\":{d},\"conversations\":[", .{
        format_version, active_id, next_id,
    });

    for (convs, 0..) |conv, ci| {
        if (ci > 0) try buf.append(allocator, ',');
        const title_esc = try json.jsonEscape(allocator, conv.title);
        defer if (title_esc.ptr != conv.title.ptr) allocator.free(title_esc);
        try buf.print(allocator, "{{\"id\":{d},\"title\":\"{s}\",\"messages\":[", .{ conv.id, title_esc });
        for (conv.messages, 0..) |msg, mi| {
            if (mi > 0) try buf.append(allocator, ',');
            const role_str: []const u8 = switch (msg.role) {
                .user => "user",
                .assistant => "assistant",
                .tool => "tool",
            };
            const content_esc = try json.jsonEscape(allocator, msg.content);
            defer if (content_esc.ptr != msg.content.ptr) allocator.free(content_esc);
            try buf.print(allocator, "{{\"role\":\"{s}\",\"content\":\"{s}\"", .{ role_str, content_esc });
            if (msg.tool_call_id) |tcid| {
                const tcid_esc = try json.jsonEscape(allocator, tcid);
                defer if (tcid_esc.ptr != tcid.ptr) allocator.free(tcid_esc);
                try buf.print(allocator, ",\"tool_call_id\":\"{s}\"", .{tcid_esc});
            }
            try buf.appendSlice(allocator, "}");
        }
        try buf.appendSlice(allocator, "]}");
    }
    try buf.appendSlice(allocator, "]}");

    try durable.replace(path, buf.items);
}

/// Load a store from `path`. FileNotFound if missing. Quarantines a corrupt
/// file to `{path}.corrupt` and returns error.CorruptStore. OutOfMemory and
/// I/O errors leave the live file in place (error.QuarantineFailed if a
/// corrupt file could not be preserved).
pub fn load(allocator: Allocator, path: []const u8) !Snapshot {
    const data = readFile(allocator, path) catch |err| {
        if (err == error.FileNotFound) return error.FileNotFound;
        return err;
    };
    defer allocator.free(data);

    const snap = parse(allocator, data) catch |err| {
        // OOM is not corruption: quarantining would rename a valid store away
        // and the next save would replace it with an empty one.
        if (err == error.OutOfMemory) return err;
        quarantine(path, data) catch |qerr| {
            std.log.err("conversation store: failed to preserve {s} ({}, original {})", .{
                path, qerr, err,
            });
            return error.QuarantineFailed;
        };
        return err;
    };
    return snap;
}

/// Narrow a decoded JSON integer to the u32 the store keeps ids in.
/// `json.extractIntField` yields any `usize`, so a value past `maxInt(u32)`
/// means the file is corrupt: returning error.CorruptStore lets `load`
/// quarantine it, where an unchecked `@intCast` would panic on every start.
fn castId(raw: usize) !u32 {
    return std.math.cast(u32, raw) orelse error.CorruptStore;
}

/// Scan the `{...}` object whose opening brace is at `arr[idx.*]` and return its
/// body (between the braces), advancing `idx` past the closing brace. Braces
/// inside strings do not count. An object that runs to the end of the array
/// without closing is corruption: a truncated store must not load as if the
/// lost tail were complete, because the next save would then rewrite the file
/// without it.
fn scanObject(arr: []const u8, idx: *usize) ![]const u8 {
    std.debug.assert(idx.* < arr.len and arr[idx.*] == '{');
    const start = idx.* + 1;
    var depth: usize = 1;
    var i = start;
    while (i < arr.len and depth > 0) : (i += 1) {
        switch (arr[i]) {
            '{' => depth += 1,
            '}' => depth -= 1,
            '"' => {
                i += 1;
                while (i < arr.len and arr[i] != '"') : (i += 1) {
                    if (arr[i] == '\\' and i + 1 < arr.len) i += 1;
                }
            },
            else => {},
        }
    }
    if (depth != 0) return error.CorruptStore;
    idx.* = i;
    return arr[start .. i - 1];
}

fn parse(allocator: Allocator, data: []const u8) !Snapshot {
    const version = json.extractIntField(data, "version") orelse return error.CorruptStore;
    if (version != format_version) return error.UnsupportedVersion;
    // extractIntField rejects negatives but accepts values above u32 max; a
    // hand-edited or truncated store must not truncate into a colliding id.
    const active_id: u32 = try castId(json.extractIntField(data, "active_id") orelse 0);
    const next_id_raw = json.extractIntField(data, "next_id") orelse 1;
    const next_id: u32 = try castId(@max(next_id_raw, 1));

    const arr = json.extractObjectField(data, "conversations") orelse return error.CorruptStore;
    if (arr.len < 2 or arr[0] != '[') return error.CorruptStore;

    var convs: std.ArrayList(LoadedConv) = .empty;
    errdefer {
        for (convs.items) |*conv| {
            allocator.free(conv.title);
            for (conv.messages) |msg| {
                allocator.free(@constCast(msg.content));
                if (msg.tool_call_id) |tcid| allocator.free(@constCast(tcid));
            }
            allocator.free(conv.messages);
        }
        convs.deinit(allocator);
    }

    var i: usize = 1;
    while (i < arr.len) {
        // The next save writes back only what loaded here, so hitting the cap
        // deletes the overflow with no record. Name it instead of dropping it.
        if (convs.items.len == max_conversations) {
            std.log.warn("conversation store: more than {d} conversations; the rest are dropped on the next save", .{max_conversations});
            break;
        }
        while (i < arr.len and (arr[i] == ' ' or arr[i] == '\n' or arr[i] == '\r' or arr[i] == '\t' or arr[i] == ',')) : (i += 1) {}
        if (i >= arr.len or arr[i] == ']') break;
        if (arr[i] != '{') return error.CorruptStore;

        const obj = try scanObject(arr, &i);

        const id_raw = json.extractIntField(obj, "id") orelse return error.CorruptStore;
        const id: u32 = try castId(id_raw);
        const title_raw = json.extractField(obj, "title") orelse "";
        const title_un = try json.jsonUnescapeOwned(allocator, title_raw);
        const title_len = @min(title_un.len, max_title_len);
        const title = allocator.dupe(u8, title_un[0..title_len]) catch |err| {
            allocator.free(title_un);
            return err;
        };
        allocator.free(title_un);

        var messages: []Message = &.{};
        if (json.extractObjectField(obj, "messages")) |msgs_arr| {
            messages = parseMessages(allocator, msgs_arr) catch |err| {
                allocator.free(title);
                return err;
            };
        }

        convs.append(allocator, .{
            .id = id,
            .title = title,
            .messages = messages,
        }) catch |err| {
            allocator.free(title);
            for (messages) |msg| {
                allocator.free(@constCast(msg.content));
                if (msg.tool_call_id) |tcid| allocator.free(@constCast(tcid));
            }
            allocator.free(messages);
            return err;
        };
    }

    return Snapshot{
        .allocator = allocator,
        .active_id = active_id,
        .next_id = next_id,
        .conversations = try convs.toOwnedSlice(allocator),
    };
}

fn parseMessages(allocator: Allocator, arr: []const u8) ![]Message {
    if (arr.len < 2 or arr[0] != '[') return error.CorruptStore;
    var list: std.ArrayList(Message) = .empty;
    errdefer {
        for (list.items) |msg| {
            allocator.free(msg.content);
            if (msg.tool_call_id) |tcid| allocator.free(tcid);
        }
        list.deinit(allocator);
    }

    var i: usize = 1;
    while (i < arr.len) {
        if (list.items.len == max_messages_per_conv) {
            std.log.warn("conversation store: more than {d} messages in one conversation; the rest are dropped on the next save", .{max_messages_per_conv});
            break;
        }
        while (i < arr.len and (arr[i] == ' ' or arr[i] == '\n' or arr[i] == '\r' or arr[i] == '\t' or arr[i] == ',')) : (i += 1) {}
        if (i >= arr.len or arr[i] == ']') break;
        if (arr[i] != '{') return error.CorruptStore;

        const obj = try scanObject(arr, &i);

        const role_str = json.extractField(obj, "role") orelse return error.CorruptStore;
        const role: Role = if (std.mem.eql(u8, role_str, "user"))
            .user
        else if (std.mem.eql(u8, role_str, "assistant"))
            .assistant
        else if (std.mem.eql(u8, role_str, "tool"))
            .tool
        else
            return error.CorruptStore;

        const content_raw = json.extractField(obj, "content") orelse "";
        const content = try json.jsonUnescapeOwned(allocator, content_raw);

        var tool_call_id: ?[]const u8 = null;
        if (json.extractField(obj, "tool_call_id")) |tcid_raw| {
            tool_call_id = json.jsonUnescapeOwned(allocator, tcid_raw) catch |err| {
                allocator.free(content);
                return err;
            };
        }

        list.append(allocator, .{
            .role = role,
            .content = content,
            .tool_call_id = tool_call_id,
        }) catch |err| {
            allocator.free(content);
            if (tool_call_id) |tcid| allocator.free(@constCast(tcid));
            return err;
        };
    }
    return try list.toOwnedSlice(allocator);
}

fn readFile(allocator: Allocator, path: []const u8) ![]u8 {
    const fd = std.posix.openat(std.posix.AT.FDCWD, path, .{}, 0) catch |err| {
        if (err == error.FileNotFound) return error.FileNotFound;
        return err;
    };
    defer _ = std.posix.system.close(fd);

    const fsize: usize = blk: {
        if (comptime @import("builtin").os.tag == .linux) {
            var st: std.os.linux.Statx = undefined;
            const rc = std.os.linux.statx(fd, @ptrCast(""), std.os.linux.AT.EMPTY_PATH, std.os.linux.STATX{ .SIZE = true }, &st);
            if (rc != 0) return error.StatFailed;
            if (st.size < 0) return error.StatFailed;
            break :blk @intCast(st.size);
        } else {
            var st: std.c.Stat = undefined;
            if (std.c.fstat(fd, &st) != 0) return error.StatFailed;
            if (st.size < 0) return error.StatFailed;
            break :blk @intCast(st.size);
        }
    };
    if (fsize == 0) return error.CorruptStore;
    if (fsize > max_store_bytes) return error.StoreTooLarge;

    const buf = try allocator.alloc(u8, fsize);
    errdefer allocator.free(buf);
    var got: usize = 0;
    while (got < fsize) {
        const n = std.posix.read(fd, buf[got..]) catch return error.ReadFailed;
        if (n == 0) break;
        got += n;
    }
    if (got != fsize) return error.ReadFailed;
    return buf;
}

fn quarantine(path: []const u8, data: []const u8) !void {
    var dest_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dest = std.fmt.bufPrint(&dest_buf, "{s}.corrupt", .{path}) catch return error.NameTooLong;
    durable.renameOver(path, dest) catch |err| {
        // Rename is preferred so the live path is vacated. If it fails, copy
        // the already-read bytes so the next save cannot destroy the only copy.
        std.log.warn("conversation store: rename {s} -> {s} failed ({}): writing copy", .{
            path, dest, err,
        });
        try durable.replace(dest, data);
    };
    std.log.warn("conversation store: quarantined corrupt file to {s}", .{dest});
}

fn mkdirP(path: []const u8) void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len == 0 or path.len >= buf.len) return;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    // Walk components and mkdir each.
    var i: usize = if (path[0] == '/') 1 else 0;
    while (i <= path.len) : (i += 1) {
        if (i != path.len and path[i] != '/') continue;
        buf[i] = 0;
        _ = std.c.mkdir(@ptrCast(buf[0..i :0]), 0o755);
        if (i < path.len) buf[i] = '/';
    }
}

/// Pid-unique test path: test binaries run in parallel in one working
/// directory, so a shared name lets one truncate or quarantine another's
/// store mid-test.
fn testPath(buf: []u8, name: []const u8) []u8 {
    return std.fmt.bufPrint(buf, "test_conv_store_{d}_{s}", .{ std.c.getpid(), name }) catch unreachable;
}

/// Suffix `durable.replace` appends to the live path for its sibling tmp.
fn tmpSuffix() []const u8 {
    return std.fmt.bufPrint(&tmp_suffix_buf, ".tmp.{d}", .{std.c.getpid()}) catch unreachable;
}

var tmp_suffix_buf: [32]u8 = undefined;

fn testPathSuffix(buf: []u8, path: []const u8, suffix: []const u8) []u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ path, suffix }) catch unreachable;
}

fn deleteTestPath(path: []const u8) void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len >= buf.len) return;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    _ = std.c.unlink(@ptrCast(buf[0..path.len :0]));
}

test "defaultPath prefers XDG_CACHE_HOME then HOME/.cache" {
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings(
        "/custom/cache/agave/conversations.json",
        formatDefaultPath(&buf, "/custom/cache", "/home/user").?,
    );
    try std.testing.expectEqualStrings(
        "/home/user/.cache/agave/conversations.json",
        formatDefaultPath(&buf, null, "/home/user").?,
    );
    try std.testing.expectEqualStrings(
        "/home/user/.cache/agave/conversations.json",
        formatDefaultPath(&buf, "", "/home/user").?,
    );
    try std.testing.expect(formatDefaultPath(&buf, null, null) == null);
    try std.testing.expect(formatDefaultPath(&buf, "", "") == null);
}

test "load caps conversations at the save cap instead of dropping silently" {
    const allocator = std.testing.allocator;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "capped.json");
    var suf_buf: [std.fs.max_path_bytes]u8 = undefined;
    defer deleteTestPath(path);
    defer deleteTestPath(testPathSuffix(&suf_buf, path, ".corrupt"));

    // One more conversation than the server will ever write, as a hand-edited
    // or externally written store can carry.
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    try buf.appendSlice(allocator, "{\"version\":1,\"active_id\":0,\"next_id\":1,\"conversations\":[");
    for (0..max_conversations + 1) |n| {
        if (n > 0) try buf.append(allocator, ',');
        try buf.print(allocator, "{{\"id\":{d},\"title\":\"c{d}\",\"messages\":[]}}", .{ n, n });
    }
    try buf.appendSlice(allocator, "]}");
    try durable.replace(path, buf.items);

    var snap = try load(allocator, path);
    defer snap.deinit();
    // Capped, not rejected: the file is still usable and the overflow is
    // logged by parse.
    try std.testing.expectEqual(max_conversations, snap.conversations.len);
}

test "save/load round-trips conversations and tool ids" {
    const allocator = std.testing.allocator;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "roundtrip.json");
    var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
    var corrupt_buf: [std.fs.max_path_bytes]u8 = undefined;
    defer deleteTestPath(path);
    defer deleteTestPath(testPathSuffix(&tmp_buf, path, tmpSuffix()));
    defer deleteTestPath(testPathSuffix(&corrupt_buf, path, ".corrupt"));

    const msgs = [_]Message{
        .{ .role = .user, .content = "hello \"world\"" },
        .{ .role = .assistant, .content = "hi\nthere" },
        .{ .role = .tool, .content = "ok", .tool_call_id = "call_1" },
    };
    const convs = [_]ConvView{
        .{ .id = 3, .title = "Chat 3", .messages = &msgs },
    };
    try save(allocator, path, 3, 4, &convs);

    var snap = try load(allocator, path);
    defer snap.deinit();
    try std.testing.expectEqual(@as(u32, 3), snap.active_id);
    try std.testing.expectEqual(@as(u32, 4), snap.next_id);
    try std.testing.expectEqual(@as(usize, 1), snap.conversations.len);
    try std.testing.expectEqual(@as(u32, 3), snap.conversations[0].id);
    try std.testing.expectEqualStrings("Chat 3", snap.conversations[0].title);
    try std.testing.expectEqual(@as(usize, 3), snap.conversations[0].messages.len);
    try std.testing.expectEqualStrings("hello \"world\"", snap.conversations[0].messages[0].content);
    try std.testing.expectEqualStrings("hi\nthere", snap.conversations[0].messages[1].content);
    try std.testing.expectEqualStrings("ok", snap.conversations[0].messages[2].content);
    try std.testing.expectEqualStrings("call_1", snap.conversations[0].messages[2].tool_call_id.?);
}

test "load quarantines corrupt store" {
    const allocator = std.testing.allocator;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "corrupt.json");
    var suf_buf: [std.fs.max_path_bytes]u8 = undefined;
    defer deleteTestPath(path);
    defer deleteTestPath(testPathSuffix(&suf_buf, path, ".corrupt"));

    try durable.replace(path, "{\"not\": \"a store\"}");
    try std.testing.expectError(error.CorruptStore, load(allocator, path));

    // Original should have been renamed away.
    _ = std.posix.openat(std.posix.AT.FDCWD, path, .{}, 0) catch |err| {
        try std.testing.expect(err == error.FileNotFound);
        return;
    };
    return error.CorruptNotQuarantined;
}

test "load quarantines a store whose ids overflow u32" {
    const allocator = std.testing.allocator;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "wideid.json");
    var suf_buf: [std.fs.max_path_bytes]u8 = undefined;
    defer deleteTestPath(path);
    defer deleteTestPath(testPathSuffix(&suf_buf, path, ".corrupt"));

    // extractIntField yields any usize, so a hand-edited store decodes this
    // cleanly. It must be treated as corruption, not truncated or @intCast.
    try durable.replace(path,
        \\{"version": 1, "active_id": 0, "next_id": 2, "conversations": [{"id": 4294967296, "title": "x", "messages": []}]}
    );
    try std.testing.expectError(error.CorruptStore, load(allocator, path));
}

test "load quarantines a store truncated mid-object" {
    const allocator = std.testing.allocator;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "truncated.json");
    var suf_buf: [std.fs.max_path_bytes]u8 = undefined;
    defer deleteTestPath(path);
    defer deleteTestPath(testPathSuffix(&suf_buf, path, ".corrupt"));

    // The array closes but the conversation object before it never does:
    // loading that as a whole conversation would let the next save rewrite the
    // file without the lost tail.
    try durable.replace(path,
        \\{"version": 1, "active_id": 0, "next_id": 2, "conversations": [{"id": 1, "title": "x", "messages": []
    );
    try std.testing.expectError(error.CorruptStore, load(allocator, path));
}

test "load missing file is FileNotFound" {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    try std.testing.expectError(error.FileNotFound, load(std.testing.allocator, testPath(&path_buf, "missing.json")));
}

test "load OOM does not quarantine a valid store" {
    const allocator = std.testing.allocator;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "oom.json");
    var suf_buf: [std.fs.max_path_bytes]u8 = undefined;
    defer deleteTestPath(path);
    defer deleteTestPath(testPathSuffix(&suf_buf, path, ".corrupt"));

    const msgs = [_]Message{
        .{ .role = .user, .content = "keep me" },
    };
    const convs = [_]ConvView{
        .{ .id = 1, .title = "Keep", .messages = &msgs },
    };
    try save(allocator, path, 1, 2, &convs);

    // One allocation is the file buffer in readFile; the next (parse) must fail.
    var fail = FailAfterN{ .parent = allocator, .remaining = 1 };
    try std.testing.expectError(error.OutOfMemory, load(fail.allocator(), path));

    const fd = std.posix.openat(std.posix.AT.FDCWD, path, .{}, 0) catch return error.StoreDeletedOnOom;
    _ = std.posix.system.close(fd);

    _ = std.posix.openat(std.posix.AT.FDCWD, testPathSuffix(&suf_buf, path, ".corrupt"), .{}, 0) catch |err| {
        try std.testing.expect(err == error.FileNotFound);
        return;
    };
    return error.QuarantinedOnOom;
}

const FailAfterN = struct {
    parent: Allocator,
    remaining: usize,

    fn allocator(self: *FailAfterN) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *FailAfterN = @ptrCast(@alignCast(ctx));
        if (self.remaining == 0) return null;
        self.remaining -= 1;
        return self.parent.rawAlloc(len, alignment, ret_addr);
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *FailAfterN = @ptrCast(@alignCast(ctx));
        return self.parent.rawResize(memory, alignment, new_len, ret_addr);
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *FailAfterN = @ptrCast(@alignCast(ctx));
        return self.parent.rawRemap(memory, alignment, new_len, ret_addr);
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *FailAfterN = @ptrCast(@alignCast(ctx));
        self.parent.rawFree(memory, alignment, ret_addr);
    }
};
