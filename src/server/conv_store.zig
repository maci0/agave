//! On-disk conversation store for the HTTP server web UI.
//!
//! JSON envelope (version 1):
//!   {"version":1,"active_id":N,"next_id":N,"conversations":[...]}
//!
//! Message content is user data, so every write here goes through
//! `durable_file.replacePrivate` (owner-only mode 0600): the store, the
//! quarantine copy, and the overflow sidecar never become readable to other
//! users of the host.
//!
//! Written with `durable_file.replace` so a crash cannot truncate the live
//! file. Load is best-effort: missing file starts empty; a corrupt file is
//! quarantined to `{path}.corrupt` so the next save cannot overwrite the
//! only remaining copy. A store larger than the load caps keeps its full
//! bytes at `{path}.overflow`, because the next save writes back only what
//! loaded. Both sidecars are the only copy of state the server does not hold,
//! so a second quarantine or overflow takes the next free `.corrupt.N` /
//! `.overflow.N` name instead of overwriting the first.
//!
//! Deleting a conversation or clearing the active one is the other way state
//! leaves the live path: the delete persists a store without those messages, so
//! the live file alone stops being a recovery path for them and only a backup
//! taken earlier still holds them. Every destructive mutation therefore copies
//! the store aside first, at `{path}.deleted[.N]` (`snapshotBeforeDelete`),
//! which the backup tier carries like the other sidecars.

const std = @import("std");
const Allocator = std.mem.Allocator;
const durable = @import("../durable_file.zig");
const json = @import("json.zig");
const term = @import("../term.zig");
const Message = @import("../chat_template.zig").Message;
const Role = @import("../chat_template.zig").Role;
const config = @import("../config.zig");

/// Current on-disk schema. Bump when the envelope is no longer readable.
pub const format_version: u32 = 1;
/// Refuse to load a store larger than this (protects against a huge corrupt file).
const max_store_bytes: usize = 64 * 1024 * 1024;
/// Conversation-store limits. Owned here because persistence is what drops
/// the overflow; `server.zig` enforces the same caps in memory so a request
/// cannot admit more than the next save would keep.
pub const max_conversations: usize = 100;
pub const max_messages_per_conv: usize = 1000;
/// Cap on one conversation title, in UTF-8 bytes. Clipping is on a character
/// boundary (`term.utf8BytePrefix`), so a multi-byte title yields fewer visible
/// characters than this bound.
pub const max_title_len: usize = 48;

/// One conversation as loaded from disk. Contents are owned by `Snapshot`.
pub const LoadedConv = struct {
    id: u32,
    title: []u8,
    messages: []Message,
};

/// Free one loaded message: content and optional `tool_call_id`, both wiped
/// first because they hold user text. Every owner of a `[]Message` frees
/// through this so a path added later cannot skip the wipe or leave a field
/// behind.
pub fn freeMessages(allocator: Allocator, messages: []const Message) void {
    for (messages) |msg| {
        freeMessage(allocator, msg);
    }
}

/// Wipe and free one loaded message. See `freeMessages`.
pub fn freeMessage(allocator: Allocator, msg: Message) void {
    const content = @constCast(msg.content);
    @memset(content, 0);
    allocator.free(content);
    if (msg.tool_call_id) |tcid| {
        const owned = @constCast(tcid);
        @memset(owned, 0);
        allocator.free(owned);
    }
}

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
            freeMessages(self.allocator, conv.messages);
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
    const xdg = config.getenv("XDG_CACHE_HOME");
    const home = config.getenv("HOME");
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
    const bytes = try encode(allocator, active_id, next_id, convs);
    defer allocator.free(bytes);
    try durable.replacePrivate(path, bytes);
}

/// Serialize a store envelope. The result is owned by the caller; every slice
/// of `convs` is only read, so a `Snapshot` can be written back without
/// copying it into a `ConvView` first.
pub fn encode(
    allocator: Allocator,
    active_id: u32,
    next_id: u32,
    convs: []const ConvView,
) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);

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

    return buf.toOwnedSlice(allocator);
}

/// Load a store from `path`. FileNotFound if missing. Quarantines a corrupt
/// file to `{path}.corrupt` and returns error.CorruptStore. A store whose
/// version this build does not read returns error.UnsupportedVersion with the
/// live file left in place. OutOfMemory and I/O errors leave the live file in
/// place (error.QuarantineFailed if a corrupt file could not be preserved).
pub fn load(allocator: Allocator, path: []const u8) !Snapshot {
    restrictToOwner(path);
    const data = readFile(allocator, path) catch |err| {
        if (err == error.FileNotFound) return error.FileNotFound;
        return err;
    };
    defer allocator.free(data);

    const result = parse(allocator, data) catch |err| {
        // OOM is not corruption: quarantining would rename a valid store away
        // and the next save would replace it with an empty one.
        if (err == error.OutOfMemory) return err;
        // A store written by a newer agave is intact, just unreadable here.
        // Quarantining it would move the only copy aside and let the next save
        // write an empty store over the live path, so a downgrade destroys the
        // history the newer build wrote. Leave it and let the caller keep
        // persisting nowhere until a build that understands it runs.
        if (err == error.UnsupportedVersion) return err;
        quarantine(path, data) catch |qerr| {
            std.log.err("conversation store: failed to preserve {s} ({}, original {})", .{
                path, qerr, err,
            });
            return error.QuarantineFailed;
        };
        return err;
    };
    // The caps in parse dropped part of a file that parsed cleanly, and the
    // next save writes back only what loaded, so the dropped tail is destroyed
    // on the first save after this load. Keep the original bytes beside the
    // live path until then.
    if (result.truncated) preserveOverflow(path, data);
    return result.snap;
}

/// Drop any group or other bits a store written by an older agave still
/// carries. Best-effort: a mode that stays wide only widens who can read
/// user content that is already on disk, and the next save replaces the
/// inode at 0600 anyway. A filesystem without modes, or a path that is gone
/// by now, is not an error.
fn restrictToOwner(path: []const u8) void {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len >= path_buf.len) return;
    @memcpy(path_buf[0..path.len], path);
    path_buf[path.len] = 0;
    _ = std.c.chmod(@ptrCast(path_buf[0..path.len :0]), @as(std.c.mode_t, durable.private_file_mode));
}

/// Write the full bytes of a store that `parse` capped to `{path}.overflow`.
/// Best-effort: the copy is a second chance, and the live file is still intact
/// at this point, so a failure here costs the sidecar, not the store.
fn preserveOverflow(path: []const u8, data: []const u8) void {
    var dest_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dest = freeSidecarPath(&dest_buf, path, ".overflow") orelse {
        std.log.err("conversation store: store at {s} exceeds the load caps and no free {s}.overflow[.n] name is left ({d} kept); the part past the caps is lost on the next save", .{ path, path, max_sidecar_copies });
        return;
    };
    durable.replacePrivate(dest, data) catch |err| {
        std.log.err("conversation store: failed to preserve {s} ({}); the part past the load caps is lost on the next save", .{ dest, err });
        return;
    };
    std.log.warn("conversation store: {s} exceeds the load caps; the full store is kept at {s} until the next save rewrites the live path", .{ path, dest });
}

/// Preserve the live store before a destructive mutation: deleting one
/// conversation (`POST /v1/conversations` with `action=delete`), or clearing
/// the active one (`/clear`, `/reset`). The mutation persists a store that no
/// longer holds those messages, so every in-process copy is wiped by it and the
/// next scheduled backup is the only thing that still has them. This keeps the
/// store as it is *now* beside the live path, where `backup` copies it with
/// the other sidecars and the runbook already knows how to install one:
/// `restore` takes any file that verifies, not only a dated copy.
///
/// Best-effort and never fatal: the deletion happens whether or not the
/// snapshot could be written. A missing live store has nothing to preserve, and
/// a failure is reported so an operator knows the recovery window is the
/// backup tier alone.
///
/// Same slot rule as a quarantine or an overflow (`freeSidecarPath`): each
/// deletion holds different history, so a second one takes `{path}.deleted.1`
/// instead of overwriting the first, and a full set is reported rather than
/// dropping one of them.
pub fn snapshotBeforeDelete(allocator: std.mem.Allocator, path: []const u8) void {
    const data = readFile(allocator, path) catch |err| switch (err) {
        // No store yet: there is nothing to preserve.
        error.FileNotFound => return,
        // Over the load cap: the server would not read it back anyway, and a
        // copy the server refuses to load is not a recovery path. The bytes
        // past the cap are already preserved at `{path}.overflow`.
        error.StoreTooLarge => {
            std.log.warn("conversation store: {s} is over the load cap; this deletion is recoverable only from the backup tier", .{path});
            return;
        },
        else => {
            std.log.err("conversation store: could not read {s} to snapshot before deleting ({}); this deletion is recoverable only from the backup tier", .{ path, err });
            return;
        },
    };
    defer allocator.free(data);

    var dest_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dest = freeSidecarPath(&dest_buf, path, ".deleted") orelse {
        std.log.err("conversation store: no free {s}.deleted[.n] name is left ({d} kept); this deletion is recoverable only from the backup tier", .{ path, max_sidecar_copies });
        return;
    };
    durable.replacePrivate(dest, data) catch |err| {
        std.log.err("conversation store: failed to preserve {s} ({}); this deletion is recoverable only from the backup tier", .{ dest, err });
        return;
    };
    std.log.warn("conversation store: kept the store as it was before the deletion at {s}; delete it once the deletion is confirmed", .{dest});
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

/// A parsed store plus whether a cap dropped part of the file. The dropped
/// part is not recoverable from the Snapshot, so `load` preserves the original
/// bytes when this is set.
const ParseResult = struct {
    snap: Snapshot,
    truncated: bool,
};

fn parse(allocator: Allocator, data: []const u8) !ParseResult {
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
    var truncated = false;
    // `id` is the store's primary key: the server looks a conversation up by
    // id, and a second record with the same id is unreachable behind the first
    // (a delete or select would only ever hit the first). Bound by
    // `max_conversations`, so a fixed table needs no allocation.
    var seen_ids: [max_conversations]u32 = undefined;
    var seen_len: usize = 0;
    errdefer {
        for (convs.items) |*conv| {
            allocator.free(conv.title);
            freeMessages(allocator, conv.messages);
            allocator.free(conv.messages);
        }
        convs.deinit(allocator);
    }

    var i: usize = 1;
    convs_loop: while (i < arr.len) {
        // The next save writes back only what loaded here, so hitting the cap
        // deletes the overflow with no record. Name it instead of dropping it.
        if (convs.items.len == max_conversations) {
            truncated = true;
            std.log.warn("conversation store: more than {d} conversations; the rest are dropped on the next save", .{max_conversations});
            break;
        }
        while (i < arr.len and (arr[i] == ' ' or arr[i] == '\n' or arr[i] == '\r' or arr[i] == '\t' or arr[i] == ',')) : (i += 1) {}
        if (i >= arr.len or arr[i] == ']') break;
        if (arr[i] != '{') return error.CorruptStore;

        const obj = try scanObject(arr, &i);

        const id_raw = json.extractIntField(obj, "id") orelse return error.CorruptStore;
        const id: u32 = try castId(id_raw);
        for (seen_ids[0..seen_len]) |seen| {
            if (seen != id) continue;
            // Same handling as a cap overflow: the record cannot be kept under
            // a key the first one already owns, and dropping it silently would
            // destroy it on the next save, so preserve the original bytes.
            truncated = true;
            std.log.warn("conversation store: conversation id {d} appears more than once; the later record is dropped on the next save", .{id});
            continue :convs_loop;
        }
        seen_ids[seen_len] = id;
        seen_len += 1;
        const title_raw = json.extractField(obj, "title") orelse "";
        const title_un = try json.jsonUnescapeOwned(allocator, title_raw);
        // The writer clips titles on a character boundary, but this file is
        // user-editable, so clip again on load: a hand-edited title over the
        // cap must not reach the UI as a half-encoded character.
        const title_prefix = term.utf8BytePrefix(title_un, max_title_len);
        const title = allocator.dupe(u8, title_prefix) catch |err| {
            allocator.free(title_un);
            return err;
        };
        allocator.free(title_un);

        var messages: []Message = &.{};
        if (json.extractObjectField(obj, "messages")) |msgs_arr| {
            messages = parseMessages(allocator, msgs_arr, &truncated) catch |err| {
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
            freeMessages(allocator, messages);
            allocator.free(messages);
            return err;
        };
    }

    return ParseResult{
        .snap = .{
            .allocator = allocator,
            .active_id = active_id,
            .next_id = next_id,
            .conversations = try convs.toOwnedSlice(allocator),
        },
        .truncated = truncated,
    };
}

fn parseMessages(allocator: Allocator, arr: []const u8, truncated: *bool) ![]Message {
    if (arr.len < 2 or arr[0] != '[') return error.CorruptStore;
    var list: std.ArrayList(Message) = .empty;
    errdefer {
        freeMessages(allocator, list.items);
        list.deinit(allocator);
    }

    var i: usize = 1;
    while (i < arr.len) {
        if (list.items.len == max_messages_per_conv) {
            truncated.* = true;
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
                // No tool_call_id to free yet, so only the content needs the wipe.
                const owned = @constCast(content);
                @memset(owned, 0);
                allocator.free(owned);
                return err;
            };
        }

        list.append(allocator, .{
            .role = role,
            .content = content,
            .tool_call_id = tool_call_id,
        }) catch |err| {
            // Not yet in `list`, so the errdefer above does not see it.
            freeMessage(allocator, .{
                .role = role,
                .content = content,
                .tool_call_id = tool_call_id,
            });
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

/// Distinct sidecar copies kept beside the live store. A quarantine or an
/// overflow preserves state the server does not hold, so a second event writes
/// a different file: a fixed name drops the first copy, which is the only copy
/// of that state. Bounded, and a full set is reported rather than overwritten.
const max_sidecar_copies: usize = 8;

fn sidecarExists(path: []const u8) bool {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len >= buf.len) return false;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    return std.c.access(@ptrCast(buf[0..path.len :0]), 0) == 0;
}

/// First free `{path}{suffix}`, then the first free `{path}{suffix}.{n}` up to
/// `max_sidecar_copies`. Null when the name does not fit or every slot is taken.
fn freeSidecarPath(buf: []u8, path: []const u8, suffix: []const u8) ?[]u8 {
    const first = std.fmt.bufPrint(buf, "{s}{s}", .{ path, suffix }) catch return null;
    if (!sidecarExists(first)) return first;
    var numbered: [std.fs.max_path_bytes]u8 = undefined;
    for (1..max_sidecar_copies) |n| {
        const candidate = std.fmt.bufPrint(&numbered, "{s}{s}.{d}", .{ path, suffix, n }) catch return null;
        if (sidecarExists(candidate)) continue;
        return std.fmt.bufPrint(buf, "{s}", .{candidate}) catch null;
    }
    return null;
}

/// Name of sidecar slot `n`: the bare suffix for the first, `.N` after it.
fn sidecarIndexSuffix(n: usize, buf: []u8) ![]const u8 {
    if (n == 0) return buf[0..0];
    return std.fmt.bufPrint(buf, ".{d}", .{n});
}

fn quarantine(path: []const u8, data: []const u8) !void {
    var dest_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dest = freeSidecarPath(&dest_buf, path, ".corrupt") orelse return error.TooManyQuarantines;
    durable.renameOver(path, dest) catch |err| {
        // Rename is preferred so the live path is vacated. If it fails, copy
        // the already-read bytes so the next save cannot destroy the only copy.
        std.log.warn("conversation store: rename {s} -> {s} failed ({}): writing copy", .{
            path, dest, err,
        });
        try durable.replacePrivate(dest, data);
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

/// Suffix `durable_file.replace` appends to the live path for its sibling tmp.
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

/// Permission bits of `path`, or `null` when it cannot be stat'ed.
fn testPathMode(path: []const u8) ?u32 {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len >= buf.len) return null;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    if (comptime @import("builtin").os.tag == .linux) {
        var st: std.os.linux.Statx = undefined;
        const rc = std.os.linux.statx(
            std.posix.AT.FDCWD,
            @ptrCast(&buf),
            std.os.linux.AT.EMPTY_PATH,
            std.os.linux.STATX{ .MODE = true },
            &st,
        );
        if (rc != 0) return null;
        return st.mode & 0o777;
    }
    // `std.c.stat` is not declared for arm64 darwin in this Zig release;
    // `std.c.fstatat` is declared there (and empty on Linux, hence the branch).
    var st: std.posix.Stat = undefined;
    if (std.c.fstatat(std.posix.AT.FDCWD, @ptrCast(&buf), &st, 0) != 0) return null;
    return @intCast(st.mode & 0o777);
}

test "a saved store is owner-only, and a store an older agave left wide is tightened on load" {
    const allocator = std.testing.allocator;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "mode.json");
    defer deleteTestPath(path);
    var suf_buf: [std.fs.max_path_bytes]u8 = undefined;
    defer deleteTestPath(testPathSuffix(&suf_buf, path, ".corrupt"));

    const msgs = [_]Message{.{ .role = .user, .content = "my address is 1 Main St" }};
    const convs = [_]ConvView{.{ .id = 1, .title = "t", .messages = &msgs }};
    try save(allocator, path, 1, 2, &convs);
    // Message content is user data: no other user of the host may read it,
    // whatever umask the build or the operator runs with.
    try std.testing.expectEqual(@as(u32, 0), testPathMode(path).? & 0o077);

    // A store written before the mode existed loads, and is narrowed before
    // any other user can open it.
    try durable.replace(path, "{\"version\":1,\"active_id\":0,\"next_id\":1,\"conversations\":[]}");
    var snap = try load(allocator, path);
    snap.deinit();
    try std.testing.expectEqual(@as(u32, 0), testPathMode(path).? & 0o077);
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
    defer deleteTestPath(testPathSuffix(&suf_buf, path, ".overflow"));

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
    try durable.replacePrivate(path, buf.items);

    var snap = try load(allocator, path);
    defer snap.deinit();
    // Capped, not rejected: the file is still usable and the overflow is
    // logged by parse.
    try std.testing.expectEqual(max_conversations, snap.conversations.len);

    // The next save writes back only these 100, so the conversation past the
    // cap is destroyed then. It has to be recoverable before that happens.
    const kept = try readFile(allocator, testPathSuffix(&suf_buf, path, ".overflow"));
    defer allocator.free(kept);
    try std.testing.expectEqualStrings(buf.items, kept);
}

test "load keeps conversation ids unique and preserves the dropped record" {
    const allocator = std.testing.allocator;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "dupid.json");
    var suf_buf: [std.fs.max_path_bytes]u8 = undefined;
    defer deleteTestPath(path);
    defer deleteTestPath(testPathSuffix(&suf_buf, path, ".corrupt"));
    defer deleteTestPath(testPathSuffix(&suf_buf, path, ".overflow"));

    // A hand-edited store carrying the same id twice: the second record is
    // unreachable behind the first, so it cannot be kept.
    const raw =
        \\{"version":1,"active_id":2,"next_id":3,"conversations":[
        \\{"id":2,"title":"first","messages":[]},
        \\{"id":2,"title":"second","messages":[]},
        \\{"id":9,"title":"third","messages":[]}]}
    ;
    try durable.replacePrivate(path, raw);

    var snap = try load(allocator, path);
    defer snap.deinit();
    try std.testing.expectEqual(@as(usize, 2), snap.conversations.len);
    try std.testing.expectEqualStrings("first", snap.conversations[0].title);
    try std.testing.expectEqualStrings("third", snap.conversations[1].title);

    // The dropped record has to survive the next save somewhere.
    const kept = try readFile(allocator, testPathSuffix(&suf_buf, path, ".overflow"));
    defer allocator.free(kept);
    try std.testing.expectEqualStrings(raw, kept);
}

test "load clips an over-long non-ASCII title on a character boundary" {
    const allocator = std.testing.allocator;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "utf8_title.json");
    var suf_buf: [std.fs.max_path_bytes]u8 = undefined;
    defer deleteTestPath(path);
    defer deleteTestPath(testPathSuffix(&suf_buf, path, ".corrupt"));

    // 47 ASCII bytes then a 3-byte "世": the cap of 48 lands inside it.
    const raw = "{\"version\":1,\"active_id\":0,\"next_id\":1,\"conversations\":" ++
        "[{\"id\":0,\"title\":\"" ++ (&@as([47]u8, @splat(0x61))) ++ "\\u4e16 extra\",\"messages\":[]}]}";
    try durable.replacePrivate(path, raw);

    var snap = try load(allocator, path);
    defer snap.deinit();
    try std.testing.expectEqualStrings(&@as([47]u8, @splat(0x61)), snap.conversations[0].title);
    try std.testing.expect(std.unicode.utf8ValidateSlice(snap.conversations[0].title));
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

    try durable.replacePrivate(path, "{\"not\": \"a store\"}");
    try std.testing.expectError(error.CorruptStore, load(allocator, path));

    // Original should have been renamed away.
    _ = std.posix.openat(std.posix.AT.FDCWD, path, .{}, 0) catch |err| {
        try std.testing.expect(err == error.FileNotFound);
        return;
    };
    return error.CorruptNotQuarantined;
}

test "a second corrupt store does not overwrite the first quarantine" {
    const allocator = std.testing.allocator;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "recorrupt.json");
    var first_buf: [std.fs.max_path_bytes]u8 = undefined;
    var second_buf: [std.fs.max_path_bytes]u8 = undefined;
    const first_copy = std.fmt.bufPrint(&first_buf, "{s}.corrupt", .{path}) catch unreachable;
    const second_copy = std.fmt.bufPrint(&second_buf, "{s}.corrupt.1", .{path}) catch unreachable;
    defer deleteTestPath(path);
    defer deleteTestPath(first_copy);
    defer deleteTestPath(second_copy);

    const first_raw = "{\"first\": 1}";
    try durable.replacePrivate(path, first_raw);
    try std.testing.expectError(error.CorruptStore, load(allocator, path));
    const first_kept = try readFile(allocator, first_copy);
    defer allocator.free(first_kept);
    try std.testing.expectEqualStrings(first_raw, first_kept);

    // A later corruption is a different file holding different history, so it
    // takes the next name rather than replacing the only copy of the first.
    const second_raw = "{\"second\": 2}";
    try durable.replacePrivate(path, second_raw);
    try std.testing.expectError(error.CorruptStore, load(allocator, path));
    const second_kept = try readFile(allocator, second_copy);
    defer allocator.free(second_kept);
    try std.testing.expectEqualStrings(second_raw, second_kept);
    const first_again = try readFile(allocator, first_copy);
    defer allocator.free(first_again);
    try std.testing.expectEqualStrings(first_raw, first_again);
}

test "sidecar slots are bounded and never overwritten" {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "slots");
    var name_buf: [std.fs.max_path_bytes]u8 = undefined;
    var index_buf: [16]u8 = undefined;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    // Fill every slot: `.corrupt` plus `.corrupt.1` through `.corrupt.7`.
    for (0..max_sidecar_copies) |n| {
        const index = try sidecarIndexSuffix(n, &index_buf);
        const name = std.fmt.bufPrint(&name_buf, "{s}.corrupt{s}", .{ path, index }) catch unreachable;
        try durable.replacePrivate(name, "x");
    }
    // Every slot holds a distinct copy, so the next event has nowhere to put
    // its own and is reported instead of dropping one of them.
    try std.testing.expect(freeSidecarPath(&buf, path, ".corrupt") == null);
    for (0..max_sidecar_copies) |n| {
        const index = try sidecarIndexSuffix(n, &index_buf);
        const name = std.fmt.bufPrint(&name_buf, "{s}.corrupt{s}", .{ path, index }) catch unreachable;
        deleteTestPath(name);
    }
}

test "snapshotBeforeDelete keeps the store a deletion is about to destroy" {
    const allocator = std.testing.allocator;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "deleted.json");
    var first_buf: [std.fs.max_path_bytes]u8 = undefined;
    var second_buf: [std.fs.max_path_bytes]u8 = undefined;
    const first_copy = std.fmt.bufPrint(&first_buf, "{s}.deleted", .{path}) catch unreachable;
    const second_copy = std.fmt.bufPrint(&second_buf, "{s}.deleted.1", .{path}) catch unreachable;
    defer deleteTestPath(path);
    defer deleteTestPath(first_copy);
    defer deleteTestPath(second_copy);

    const raw = "{\"version\":1,\"active_id\":1,\"next_id\":2,\"conversations\":[{\"id\":1,\"title\":\"doomed\",\"messages\":[{\"role\":\"user\",\"content\":\"keep me\"}]}]}";
    try durable.replacePrivate(path, raw);

    snapshotBeforeDelete(allocator, path);
    // Owner-only, like every other copy of user text.
    try std.testing.expectEqual(@as(?u32, 0o600), testPathMode(first_copy));
    const kept = try readFile(allocator, first_copy);
    defer allocator.free(kept);
    try std.testing.expectEqualStrings(raw, kept);

    // A second deletion holds different history, so it takes the next slot
    // rather than replacing the only copy of the first.
    try durable.replacePrivate(path, "{\"version\":1,\"active_id\":2,\"next_id\":3,\"conversations\":[]}");
    snapshotBeforeDelete(allocator, path);
    const kept_again = try readFile(allocator, first_copy);
    defer allocator.free(kept_again);
    try std.testing.expectEqualStrings(raw, kept_again);
    const second = try readFile(allocator, second_copy);
    defer allocator.free(second);
    try std.testing.expectEqualStrings("{\"version\":1,\"active_id\":2,\"next_id\":3,\"conversations\":[]}", second);
}

test "snapshotBeforeDelete on a missing store writes nothing and does not fail" {
    const allocator = std.testing.allocator;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "nosnapshot.json");
    var copy_buf: [std.fs.max_path_bytes]u8 = undefined;
    const copy = std.fmt.bufPrint(&copy_buf, "{s}.deleted", .{path}) catch unreachable;
    defer deleteTestPath(copy);

    // Nothing to preserve, so this is a no-op rather than a failure the
    // delete path has to handle.
    snapshotBeforeDelete(allocator, path);
    try std.testing.expectEqual(@as(?u32, null), testPathMode(copy));
}

test "a deleted conversation survives in the pre-deletion snapshot" {
    // The delete path end to end: snapshot, then persist a store without the
    // conversation. The sidecar is the only remaining copy of those messages
    // until the next backup, so it has to be a loadable store after the
    // mutation, not a copy of one.
    const allocator = std.testing.allocator;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "afterdelete.json");
    var copy_buf: [std.fs.max_path_bytes]u8 = undefined;
    const copy = std.fmt.bufPrint(&copy_buf, "{s}.deleted", .{path}) catch unreachable;
    defer deleteTestPath(path);
    defer deleteTestPath(copy);

    try save(allocator, path, 1, 2, &.{
        .{ .id = 1, .title = "gone", .messages = &.{.{ .role = .user, .content = "why" }} },
    });
    snapshotBeforeDelete(allocator, path);
    try save(allocator, path, 1, 2, &.{});

    var after = try load(allocator, path);
    defer after.deinit();
    try std.testing.expectEqual(@as(usize, 0), after.conversations.len);

    var snap = try load(allocator, copy);
    defer snap.deinit();
    try std.testing.expectEqual(@as(usize, 1), snap.conversations.len);
    try std.testing.expectEqualStrings("why", snap.conversations[0].messages[0].content);
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
    try durable.replacePrivate(path,
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
    try durable.replacePrivate(path,
        \\{"version": 1, "active_id": 0, "next_id": 2, "conversations": [{"id": 1, "title": "x", "messages": []
    );
    try std.testing.expectError(error.CorruptStore, load(allocator, path));
}

test "load missing file is FileNotFound" {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    try std.testing.expectError(error.FileNotFound, load(std.testing.allocator, testPath(&path_buf, "missing.json")));
}

test "load leaves a newer store in place instead of quarantining it" {
    const allocator = std.testing.allocator;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = testPath(&path_buf, "future.json");
    var suf_buf: [std.fs.max_path_bytes]u8 = undefined;
    defer deleteTestPath(path);
    defer deleteTestPath(testPathSuffix(&suf_buf, path, ".corrupt"));

    const future = "{\"version\": 99, \"active_id\": 0, \"next_id\": 2, \"conversations\": []}";
    try durable.replace(path, future);
    try std.testing.expectError(error.UnsupportedVersion, load(allocator, path));

    // The file a newer agave wrote is intact, not renamed aside: the next save
    // must not be able to replace it with an empty store.
    const kept = try std.posix.openat(std.posix.AT.FDCWD, path, .{}, 0);
    defer _ = std.posix.system.close(kept);
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

test "fuzz: store parse caps, and a load/save round trip preserves what loaded" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const allocator = std.testing.allocator;

            // A body spliced between the envelope fields so the parser sees
            // structurally plausible stores, not only unrelated bytes.
            var body: [256]u8 = undefined;
            smith.bytesWithHash(&body, 0);
            const body_len = smith.indexWithHash(body.len + 1, 1);
            var envelope: [512]u8 = undefined;
            const data = std.fmt.bufPrint(&envelope, "{{\"version\":1,\"active_id\":{d},\"next_id\":{d},\"conversations\":{s}", .{
                smith.indexWithHash(8, 2),
                @as(u32, @intCast(smith.indexWithHash(8, 3))) + 1,
                body[0..body_len],
            }) catch return;

            const result = parse(allocator, data) catch return;
            var snap = result.snap;
            defer snap.deinit();

            try std.testing.expect(snap.conversations.len <= max_conversations);
            try std.testing.expect(snap.next_id >= 1);
            for (snap.conversations) |conv| {
                try std.testing.expect(conv.title.len <= max_title_len);
                try std.testing.expect(conv.messages.len <= max_messages_per_conv);
            }

            // Pair assertion across the persistence boundary: what load
            // accepted must come back byte-identical through the writer, or
            // the next save silently rewrites the user's conversations.
            const views = try allocator.alloc(ConvView, snap.conversations.len);
            defer allocator.free(views);
            for (snap.conversations, views) |conv, *view| {
                view.* = .{ .id = conv.id, .title = conv.title, .messages = conv.messages };
            }
            const encoded = try encode(allocator, snap.active_id, snap.next_id, views);
            defer allocator.free(encoded);

            const reloaded = parse(allocator, encoded) catch |err| {
                // The writer emits what the reader accepts, so a store it
                // cannot read back is a writer bug, not bad input.
                std.debug.print("store the writer emitted did not parse: {s}\n{any}\n", .{ @errorName(err), encoded });
                return err;
            };
            var again = reloaded.snap;
            defer again.deinit();

            try std.testing.expectEqual(snap.active_id, again.active_id);
            try std.testing.expectEqual(snap.next_id, again.next_id);
            try std.testing.expectEqual(snap.conversations.len, again.conversations.len);
            for (snap.conversations, again.conversations) |a, b| {
                try std.testing.expectEqual(a.id, b.id);
                try std.testing.expectEqualStrings(a.title, b.title);
                try std.testing.expectEqual(a.messages.len, b.messages.len);
                for (a.messages, b.messages) |ma, mb| {
                    try std.testing.expectEqual(ma.role, mb.role);
                    try std.testing.expectEqualStrings(ma.content, mb.content);
                    if (ma.tool_call_id) |tcid| {
                        try std.testing.expectEqualStrings(tcid, mb.tool_call_id.?);
                    } else {
                        try std.testing.expect(mb.tool_call_id == null);
                    }
                }
            }
        }
    }.f, .{});
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
