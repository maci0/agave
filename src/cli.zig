//! Self-contained CLI argument parser for Agave.
//! Replaces external `clap` dependency — zero external dependencies.
//!
//! Supports:
//!   - `--flag` boolean flags
//!   - `--option value` or `--option=value` string options
//!   - `-f` short flags (mapped via ArgSpec)
//!   - `-f value` short options (mapped via ArgSpec)
//!   - `--` stops option parsing (everything after is positional)
//!   - Bare arguments (no `-` prefix) are positional

const std = @import("std");

/// Specification for a single CLI argument.
pub const ArgSpec = struct {
    /// Long option name without `--` prefix (e.g. "max-tokens").
    long: []const u8,
    /// Optional single-character short alias (e.g. 'n').
    short: ?u8 = null,
    /// Whether this argument is a boolean flag or takes a value.
    kind: enum { flag, option } = .flag,
    /// Help text (used for documentation, not printed by this parser).
    help: []const u8 = "",
};

/// Result of parsing CLI arguments.
pub const ParseResult = struct {
    flags: std.StringHashMap(void),
    options: std.StringHashMap([]const u8),
    positionals: std.ArrayList([]const u8),
    allocator: std.mem.Allocator,

    /// Release all memory owned by this result.
    pub fn deinit(self: *ParseResult) void {
        self.flags.deinit();
        self.options.deinit();
        self.positionals.deinit(self.allocator);
    }

    /// Returns true if the named flag was present on the command line.
    pub fn flag(self: *const ParseResult, name: []const u8) bool {
        return self.flags.contains(name);
    }

    /// Returns the string value of a named option, or null if not provided.
    pub fn option(self: *const ParseResult, name: []const u8) ?[]const u8 {
        return self.options.get(name);
    }

    /// Returns the positional argument at the given index, or null.
    pub fn positional(self: *const ParseResult, index: usize) ?[]const u8 {
        if (index >= self.positionals.items.len) return null;
        return self.positionals.items[index];
    }

    /// Parse a named option as u16, returning null if absent or invalid.
    pub fn optionU16(self: *const ParseResult, name: []const u8) ?u16 {
        const s = self.options.get(name) orelse return null;
        return std.fmt.parseInt(u16, s, 10) catch null;
    }

    /// Parse a named option as u32, returning null if absent or invalid.
    pub fn optionU32(self: *const ParseResult, name: []const u8) ?u32 {
        const s = self.options.get(name) orelse return null;
        return std.fmt.parseInt(u32, s, 10) catch null;
    }

    /// Parse a named option as u64, returning null if absent or invalid.
    pub fn optionU64(self: *const ParseResult, name: []const u8) ?u64 {
        const s = self.options.get(name) orelse return null;
        return std.fmt.parseInt(u64, s, 10) catch null;
    }
};

/// Look up an ArgSpec by its long name. Returns null if not found.
fn findByLong(specs: []const ArgSpec, name: []const u8) ?*const ArgSpec {
    for (specs) |*spec| {
        if (std.mem.eql(u8, spec.long, name)) return spec;
    }
    return null;
}

/// Look up an ArgSpec by its short character. Returns null if not found.
fn findByShort(specs: []const ArgSpec, ch: u8) ?*const ArgSpec {
    for (specs) |*spec| {
        if (spec.short) |s| {
            if (s == ch) return spec;
        }
    }
    return null;
}

/// Parse command-line arguments against the given specs.
///
/// Skips argv[0] (program name). After `--`, all remaining arguments
/// are treated as positionals. Unrecognized flags/options are silently
/// treated as positionals to avoid breaking on model paths that start
/// with `-` (rare but possible).
pub fn parse(allocator: std.mem.Allocator, args: std.process.Args, specs: []const ArgSpec) ParseResult {
    var result = ParseResult{
        .flags = std.StringHashMap(void).init(allocator),
        .options = std.StringHashMap([]const u8).init(allocator),
        .positionals = .empty,
        .allocator = allocator,
    };

    var iter = args.iterate();
    _ = iter.skip(); // skip argv[0]

    var past_double_dash = false;

    while (iter.next()) |arg| {
        if (past_double_dash) {
            result.positionals.append(allocator, arg) catch {};
            continue;
        }

        if (std.mem.eql(u8, arg, "--")) {
            past_double_dash = true;
            continue;
        }

        // Long option: --name or --name=value
        if (arg.len > 2 and arg[0] == '-' and arg[1] == '-') {
            const rest = arg[2..];

            // Check for --name=value form
            if (std.mem.indexOfScalar(u8, rest, '=')) |eq_pos| {
                const name = rest[0..eq_pos];
                const value = rest[eq_pos + 1 ..];
                if (findByLong(specs, name)) |spec| {
                    if (spec.kind == .option) {
                        result.options.put(name, value) catch {};
                    } else {
                        // --flag=value is unusual but store as flag
                        result.flags.put(name, {}) catch {};
                    }
                } else {
                    // Unknown --name=value: store as option anyway
                    result.options.put(name, value) catch {};
                }
                continue;
            }

            // --name (no =)
            const name = rest;
            if (findByLong(specs, name)) |spec| {
                if (spec.kind == .flag) {
                    result.flags.put(name, {}) catch {};
                } else {
                    // Option: consume next arg as value
                    if (iter.next()) |val| {
                        result.options.put(name, val) catch {};
                    }
                }
            } else {
                // Unknown long option: treat as flag (common for --help-like unknowns)
                result.flags.put(name, {}) catch {};
            }
            continue;
        }

        // Short option: -X
        if (arg.len == 2 and arg[0] == '-' and arg[1] != '-') {
            const ch = arg[1];
            if (findByShort(specs, ch)) |spec| {
                if (spec.kind == .flag) {
                    result.flags.put(spec.long, {}) catch {};
                } else {
                    // Option: consume next arg as value
                    if (iter.next()) |val| {
                        result.options.put(spec.long, val) catch {};
                    }
                }
            } else {
                // Unknown short: treat as positional
                result.positionals.append(allocator, arg) catch {};
            }
            continue;
        }

        // Positional argument
        result.positionals.append(allocator, arg) catch {};
    }

    return result;
}

// ── Tests ───────────────────────────────────────────────────────────

test "flag parsing" {
    const specs = [_]ArgSpec{
        .{ .long = "help", .short = 'h' },
        .{ .long = "verbose", .short = 'V' },
    };

    // Simulate args: ["prog", "--help", "-V"]
    // We can't easily construct std.process.Args in tests, so we test
    // the lookup helpers instead.
    const h = findByLong(&specs, "help");
    try std.testing.expect(h != null);
    try std.testing.expectEqual(@as(?u8, 'h'), h.?.short);

    const v = findByShort(&specs, 'V');
    try std.testing.expect(v != null);
    try std.testing.expectEqualStrings("verbose", v.?.long);

    const missing = findByLong(&specs, "nonexistent");
    try std.testing.expect(missing == null);
}

test "option spec lookup" {
    const specs = [_]ArgSpec{
        .{ .long = "max-tokens", .short = 'n', .kind = .option },
        .{ .long = "backend", .kind = .option },
    };

    const mt = findByLong(&specs, "max-tokens");
    try std.testing.expect(mt != null);
    try std.testing.expectEqual(.option, mt.?.kind);
    try std.testing.expectEqual(@as(?u8, 'n'), mt.?.short);

    const be = findByShort(&specs, 'n');
    try std.testing.expect(be != null);
    try std.testing.expectEqualStrings("max-tokens", be.?.long);
}

test "ParseResult typed accessors" {
    var r = ParseResult{
        .flags = std.StringHashMap(void).init(std.testing.allocator),
        .options = std.StringHashMap([]const u8).init(std.testing.allocator),
        .positionals = .empty,
        .allocator = std.testing.allocator,
    };
    defer r.deinit();

    r.flags.put("help", {}) catch unreachable;
    r.options.put("port", "8080") catch unreachable;
    r.options.put("seed", "42") catch unreachable;
    r.options.put("bad", "notanumber") catch unreachable;
    r.positionals.append(std.testing.allocator, "model.gguf") catch unreachable;

    try std.testing.expect(r.flag("help"));
    try std.testing.expect(!r.flag("version"));

    try std.testing.expectEqualStrings("8080", r.option("port").?);
    try std.testing.expect(r.option("missing") == null);

    try std.testing.expectEqual(@as(?u16, 8080), r.optionU16("port"));
    try std.testing.expectEqual(@as(?u32, 42), r.optionU32("seed"));
    try std.testing.expectEqual(@as(?u64, 42), r.optionU64("seed"));
    try std.testing.expect(r.optionU32("bad") == null);
    try std.testing.expect(r.optionU32("missing") == null);

    try std.testing.expectEqualStrings("model.gguf", r.positional(0).?);
    try std.testing.expect(r.positional(1) == null);
}
