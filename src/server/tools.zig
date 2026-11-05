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
    slots: [max_tools]?Tool = @splat(null),

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

    /// True when `name` is registered.
    pub fn hasName(self: *const Registry, name: []const u8) bool {
        for (self.slots) |s| {
            if (s) |t| {
                if (std.mem.eql(u8, t.name, name)) return true;
            }
        }
        return false;
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
    var out: [max_tools]?Tool = @splat(null);
    const n = reg.mergeInto(&req, &out);
    try std.testing.expectEqual(@as(u32, 1), n);
    try std.testing.expectEqualStrings("request", out[0].?.description);
}

test "hasName matches registered tools only" {
    var reg = Registry{};
    try std.testing.expect(!reg.hasName("greet"));
    _ = try reg.register(.{ .name = "greet", .description = "", .parameters_json = "{}" });
    try std.testing.expect(reg.hasName("greet"));
    try std.testing.expect(!reg.hasName("other"));
}

test "merge appends distinct request tools" {
    var reg = Registry{};
    _ = try reg.register(.{ .name = "a", .description = "", .parameters_json = "{}" });
    const req = [_]?Tool{
        .{ .name = "b", .description = "", .parameters_json = "{}" },
    };
    var out: [max_tools]?Tool = @splat(null);
    const n = reg.mergeInto(&req, &out);
    try std.testing.expectEqual(@as(u32, 2), n);
}

test "fuzz: registry register/dispose/unregister/merge state machine" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const names = [_][]const u8{
                "alpha",  "beta", "gamma", "delta",
                "search", "code", "x",     "",
            };
            const n_ops = smith.indexWithHash(24, 0) + 1;

            var reg = Registry{};
            var handles: [max_tools]?Handle = @splat(null);
            var next_handle: usize = 0;

            for (0..n_ops) |op| {
                const kind = smith.indexWithHash(4, @truncate(op + 1));
                switch (kind) {
                    0 => {
                        // Register a (possibly duplicate) name.
                        const name = names[smith.indexWithHash(names.len, @truncate(op + 2))];
                        const had = reg.hasName(name);
                        const before = reg.count();
                        const res = reg.register(.{
                            .name = name,
                            .description = "fuzz",
                            .parameters_json = "{}",
                        });
                        if (res) |h| {
                            // Success implies the name was absent and a slot freed.
                            try std.testing.expect(!had);
                            try std.testing.expectEqual(before + 1, reg.count());
                            try std.testing.expect(reg.hasName(name));
                            try std.testing.expect(h.slot < max_tools);
                            if (next_handle < handles.len) {
                                handles[next_handle % handles.len] = h;
                                next_handle += 1;
                            }
                        } else |err| {
                            // Failure leaves the registry untouched.
                            try std.testing.expectEqual(before, reg.count());
                            if (err == error.DuplicateName) try std.testing.expect(had);
                            if (err == error.RegistryFull) try std.testing.expectEqual(@as(u32, max_tools), before);
                        }
                    },
                    1 => {
                        // Dispose a handle (idempotent, may be stale).
                        if (next_handle > 0) {
                            const idx = smith.indexWithHash(next_handle, @truncate(op + 3)) % handles.len;
                            if (handles[idx]) |h| h.dispose();
                        }
                        // Unregister a raw slot id, in or out of range.
                        const slot = smith.valueWithHash(u32, @truncate(op + 4));
                        reg.unregister(slot);
                    },
                    2 => {
                        // hasName / count agree with the slots.
                        const name = names[smith.indexWithHash(names.len, @truncate(op + 5))];
                        var expected = false;
                        var occupied: u32 = 0;
                        for (reg.slots) |s| {
                            if (s) |t| {
                                occupied += 1;
                                if (std.mem.eql(u8, t.name, name)) expected = true;
                            }
                        }
                        try std.testing.expectEqual(occupied, reg.count());
                        try std.testing.expectEqual(expected, reg.hasName(name));
                        try std.testing.expect(reg.count() <= max_tools);
                    },
                    else => {
                        // mergeInto: output names are unique, count within
                        // bounds, and request tools win on a name clash.
                        var req_buf: [4]?Tool = @splat(null);
                        const req_len = smith.indexWithHash(req_buf.len + 1, @truncate(op + 6));
                        for (0..req_len) |ri| {
                            if (smith.boolWeightedWithHash(1, 1, 1)) {
                                req_buf[ri] = .{
                                    .name = names[smith.indexWithHash(names.len, @truncate(op + ri + 7))],
                                    .description = "request",
                                    .parameters_json = "{\"r\":true}",
                                };
                            }
                        }
                        var out: [max_tools + 4]?Tool = @splat(null);
                        const n = reg.mergeInto(&req_buf, &out);
                        try std.testing.expect(n <= out.len);
                        try std.testing.expect(n <= max_tools + req_len);
                        // Uniqueness of output names.
                        for (out[0..n], 0..) |a, ai| {
                            const ta = a orelse continue;
                            for (out[ai + 1 .. n]) |b| {
                                if (b) |tb| try std.testing.expect(!std.mem.eql(u8, ta.name, tb.name));
                            }
                        }
                        // A request tool that clamped into `out` carries the
                        // request description, not the registry one.
                        for (out[0..n]) |entry| {
                            if (entry) |t| {
                                if (std.mem.eql(u8, t.description, "request")) {
                                    var in_req = false;
                                    for (req_buf) |r| {
                                        if (r) |rt| {
                                            if (std.mem.eql(u8, rt.name, t.name)) in_req = true;
                                        }
                                    }
                                    try std.testing.expect(in_req);
                                }
                            }
                        }
                    },
                }
            }
        }
    }.f, .{});
}
