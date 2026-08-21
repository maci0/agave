//! Process-level tool registry with unregister (revertible registration).
//!
//! Request JSON tools stay per-request. This registry is the in-process
//! surface: `register` returns a handle whose `dispose` drops the slot.
//! Prompt emission (system-prompt text) stays outside this boundary.

const std = @import("std");

/// Maximum tools the process-level registry holds.
pub const max_tools: usize = 16;

/// Tool schema used for prompt injection. Strings are borrowed.
pub const Tool = struct {
    name: []const u8,
    description: []const u8,
    parameters_json: []const u8,
};

/// Disposer for one `register` call.
pub const Handle = struct {
    registry: *Registry,
    slot: u32,

    /// Unregister this tool. Idempotent.
    pub fn dispose(self: Handle) void {
        self.registry.unregister(self.slot);
    }
};

/// Dense-with-holes registry. Unregister leaves a free slot.
pub const Registry = struct {
    slots: [max_tools]?Tool = .{null} ** max_tools,

    /// Insert `tool`. Fails if the name is already registered or the table is full.
    pub fn register(self: *Registry, tool: Tool) error{ RegistryFull, DuplicateName }!Handle {
        for (self.slots) |s| {
            if (s) |t| {
                if (std.mem.eql(u8, t.name, tool.name)) return error.DuplicateName;
            }
        }
        for (&self.slots, 0..) |*s, i| {
            if (s.* == null) {
                s.* = tool;
                return .{ .registry = self, .slot = @intCast(i) };
            }
        }
        return error.RegistryFull;
    }

    /// Drop slot `id` if it is occupied.
    pub fn unregister(self: *Registry, id: u32) void {
        if (id < max_tools) self.slots[id] = null;
    }

    /// Occupied slot count.
    pub fn count(self: *const Registry) u32 {
        var n: u32 = 0;
        for (self.slots) |s| {
            if (s != null) n += 1;
        }
        return n;
    }

    /// Write occupied tools into `out` (registry first). Returns number written.
    /// Request tools with the same name replace the registry entry.
    pub fn mergeInto(self: *const Registry, request: []const ?Tool, out: []?Tool) u32 {
        var n: u32 = 0;
        for (self.slots) |s| {
            const t = s orelse continue;
            if (n >= out.len) break;
            out[n] = t;
            n += 1;
        }
        for (request) |maybe| {
            const t = maybe orelse continue;
            var replaced = false;
            for (out[0..n]) |*slot| {
                if (slot.*) |existing| {
                    if (std.mem.eql(u8, existing.name, t.name)) {
                        slot.* = t;
                        replaced = true;
                        break;
                    }
                }
            }
            if (!replaced) {
                if (n >= out.len) break;
                out[n] = t;
                n += 1;
            }
        }
        return n;
    }
};

test "register then dispose drops the tool" {
    var reg = Registry{};
    const h = try reg.register(.{
        .name = "greet",
        .description = "Say hello",
        .parameters_json = "{}",
    });
    try std.testing.expectEqual(@as(u32, 1), reg.count());
    h.dispose();
    try std.testing.expectEqual(@as(u32, 0), reg.count());
}

test "duplicate name is rejected" {
    var reg = Registry{};
    _ = try reg.register(.{ .name = "a", .description = "", .parameters_json = "{}" });
    try std.testing.expectError(error.DuplicateName, reg.register(.{
        .name = "a",
        .description = "other",
        .parameters_json = "{}",
    }));
}

test "dispose is idempotent" {
    var reg = Registry{};
    const h = try reg.register(.{ .name = "x", .description = "", .parameters_json = "{}" });
    h.dispose();
    h.dispose();
    try std.testing.expectEqual(@as(u32, 0), reg.count());
}

test "request tool wins on name clash" {
    var reg = Registry{};
    _ = try reg.register(.{
        .name = "search",
        .description = "registry",
        .parameters_json = "{}",
    });
    const req = [_]?Tool{
        .{ .name = "search", .description = "request", .parameters_json = "{\"q\":true}" },
    };
    var out: [max_tools]?Tool = .{null} ** max_tools;
    const n = reg.mergeInto(&req, &out);
    try std.testing.expectEqual(@as(u32, 1), n);
    try std.testing.expectEqualStrings("request", out[0].?.description);
}

test "merge appends distinct request tools" {
    var reg = Registry{};
    _ = try reg.register(.{ .name = "a", .description = "", .parameters_json = "{}" });
    const req = [_]?Tool{
        .{ .name = "b", .description = "", .parameters_json = "{}" },
    };
    var out: [max_tools]?Tool = .{null} ** max_tools;
    const n = reg.mergeInto(&req, &out);
    try std.testing.expectEqual(@as(u32, 2), n);
}
