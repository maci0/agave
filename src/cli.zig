//! Self-contained CLI argument parser for Agave.
//!
//! Supports:
//!   - `--flag` boolean flags
//!   - `--option value` or `--option=value` string options
//!   - `-f` short flags (mapped via ArgSpec)
//!   - `-abc` short flag clusters (boolean flags only; option chars must be last)
//!   - `-f value` or `-fVALUE` short options (mapped via ArgSpec)
//!   - `--` stops option parsing (everything after is positional)
//!   - Bare arguments (no `-` prefix) are positional
//!
//! `--flag=value` on a boolean flag is recorded in `options` (not `flags`) so
//! callers can reject it; do not treat that as the flag being set.

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
    /// Set when a known option at end of args had no value to consume.
    missing_value: ?[]const u8 = null,

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

    /// Parse a named option as an unsigned integer, returning null if absent or invalid.
    pub fn optionInt(self: *const ParseResult, comptime T: type, name: []const u8) ?T {
        const s = self.options.get(name) orelse return null;
        return std.fmt.parseInt(T, s, 10) catch null;
    }

    pub fn optionU16(self: *const ParseResult, name: []const u8) ?u16 {
        return self.optionInt(u16, name);
    }
    pub fn optionU32(self: *const ParseResult, name: []const u8) ?u32 {
        return self.optionInt(u32, name);
    }
    pub fn optionU64(self: *const ParseResult, name: []const u8) ?u64 {
        return self.optionInt(u64, name);
    }

    /// Parse a named option as f32, returning null if absent or invalid.
    pub fn optionF32(self: *const ParseResult, name: []const u8) ?f32 {
        const s = self.options.get(name) orelse return null;
        const val = std.fmt.parseFloat(f32, s) catch return null;
        if (!std.math.isFinite(val)) return null;
        return val;
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
/// are treated as positionals. Unrecognized long options are stored as
/// flags (or as options when using `--name=value` form). Unrecognized
/// short options are treated as positionals to avoid breaking on model
/// paths that start with `-` (rare but possible). Callers should reject
/// letter-only shorts that look like typos (see rejectUnknownShortPositionals).
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
            result.positionals.append(allocator, arg) catch @panic("out of memory");
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
                // Always store in options (not flags) so callers can detect and
                // reject boolean --flag=value forms like --help=1.
                result.options.put(name, value) catch @panic("out of memory");
                continue;
            }

            // --name (no =)
            const name = rest;
            if (findByLong(specs, name)) |spec| {
                if (spec.kind == .flag) {
                    result.flags.put(name, {}) catch @panic("out of memory");
                } else {
                    // Option: consume next arg as value
                    if (iter.next()) |val| {
                        result.options.put(name, val) catch @panic("out of memory");
                    } else {
                        result.missing_value = name;
                    }
                }
            } else {
                // Unknown long option: treat as flag (common for --help-like unknowns)
                result.flags.put(name, {}) catch @panic("out of memory");
            }
            continue;
        }

        // Short option: -X, -abc (flag cluster), or -n512 / -n 512 (option)
        if (arg.len >= 2 and arg[0] == '-' and arg[1] != '-') {
            // Validate the whole cluster first. Applying flags then falling back
            // to positional on a mid-cluster typo (e.g. -qZ) would both set the
            // flag and treat "-qZ" as a model path.
            var vi: usize = 1;
            var cluster_ok = true;
            while (vi < arg.len) {
                const ch = arg[vi];
                if (findByShort(specs, ch)) |spec| {
                    if (spec.kind == .flag) {
                        vi += 1;
                    } else {
                        break; // option consumes the rest (or next argv)
                    }
                } else {
                    cluster_ok = false;
                    break;
                }
            }
            if (!cluster_ok) {
                result.positionals.append(allocator, arg) catch @panic("out of memory");
                continue;
            }
            var i: usize = 1;
            while (i < arg.len) {
                const ch = arg[i];
                const spec = findByShort(specs, ch).?;
                if (spec.kind == .flag) {
                    result.flags.put(spec.long, {}) catch @panic("out of memory");
                    i += 1;
                } else {
                    // Option must be last in the cluster; attached rest or next argv.
                    const rest = arg[i + 1 ..];
                    if (rest.len > 0) {
                        result.options.put(spec.long, rest) catch @panic("out of memory");
                    } else if (iter.next()) |val| {
                        result.options.put(spec.long, val) catch @panic("out of memory");
                    } else {
                        result.missing_value = spec.long;
                    }
                    break;
                }
            }
            continue;
        }

        // Positional argument
        result.positionals.append(allocator, arg) catch @panic("out of memory");
    }

    return result;
}

// ── Tests ───────────────────────────────────────────────────────────

test "flag parsing" {
    const specs = [_]ArgSpec{
        .{ .long = "help", .short = 'h' },
        .{ .long = "verbose", .short = 'V' },
    };

    // POSIX Args.Vector is []const [*:0]const u8 (same construction as fuzz tests).
    const argv = [_][*:0]const u8{ "agave", "--help", "-V" };
    const args = std.process.Args{ .vector = &argv };
    var r = parse(std.testing.allocator, args, &specs);
    defer r.deinit();

    try std.testing.expect(r.flag("help"));
    try std.testing.expect(r.flag("verbose"));
    try std.testing.expect(!r.flag("serve"));
    try std.testing.expect(r.positional(0) == null);
}

test "findByLong and findByShort lookup" {
    const specs = [_]ArgSpec{
        .{ .long = "help", .short = 'h' },
        .{ .long = "verbose", .short = 'V' },
    };

    const h = findByLong(&specs, "help");
    try std.testing.expect(h != null);
    try std.testing.expectEqual(@as(?u8, 'h'), h.?.short);

    const v = findByShort(&specs, 'V');
    try std.testing.expect(v != null);
    try std.testing.expectEqualStrings("verbose", v.?.long);

    const missing = findByLong(&specs, "nonexistent");
    try std.testing.expect(missing == null);
}

test "parse options equals form and double dash" {
    const specs = [_]ArgSpec{
        .{ .long = "backend", .short = 'b', .kind = .option },
        .{ .long = "max-tokens", .short = 'n', .kind = .option },
        .{ .long = "verbose", .short = 'V' },
    };

    const argv = [_][*:0]const u8{ "agave", "--backend=metal", "-n", "128", "--", "--verbose", "model.gguf" };
    const args = std.process.Args{ .vector = &argv };
    var r = parse(std.testing.allocator, args, &specs);
    defer r.deinit();

    try std.testing.expectEqualStrings("metal", r.option("backend").?);
    try std.testing.expectEqual(@as(?u32, 128), r.optionU32("max-tokens"));
    // After `--`, `--verbose` is positional, not a flag.
    try std.testing.expect(!r.flag("verbose"));
    try std.testing.expectEqualStrings("--verbose", r.positional(0).?);
    try std.testing.expectEqualStrings("model.gguf", r.positional(1).?);
}

test "parse missing option value sets missing_value" {
    const specs = [_]ArgSpec{
        .{ .long = "backend", .kind = .option },
    };

    const argv = [_][*:0]const u8{ "agave", "--backend" };
    const args = std.process.Args{ .vector = &argv };
    var r = parse(std.testing.allocator, args, &specs);
    defer r.deinit();

    try std.testing.expectEqualStrings("backend", r.missing_value.?);
    try std.testing.expect(r.option("backend") == null);
}

test "boolean flag with equals goes to options not flags" {
    const specs = [_]ArgSpec{
        .{ .long = "quiet", .short = 'q' },
        .{ .long = "help", .short = 'h' },
    };

    const argv = [_][*:0]const u8{ "agave", "--quiet=true", "--help=1" };
    const args = std.process.Args{ .vector = &argv };
    var r = parse(std.testing.allocator, args, &specs);
    defer r.deinit();

    try std.testing.expect(!r.flag("quiet"));
    try std.testing.expect(!r.flag("help"));
    try std.testing.expectEqualStrings("true", r.option("quiet").?);
    try std.testing.expectEqualStrings("1", r.option("help").?);
}

test "short flag cluster and attached option value" {
    const specs = [_]ArgSpec{
        .{ .long = "quiet", .short = 'q' },
        .{ .long = "verbose", .short = 'V' },
        .{ .long = "max-tokens", .short = 'n', .kind = .option },
    };

    const argv = [_][*:0]const u8{ "agave", "-qV", "-n128", "model.gguf" };
    const args = std.process.Args{ .vector = &argv };
    var r = parse(std.testing.allocator, args, &specs);
    defer r.deinit();

    try std.testing.expect(r.flag("quiet"));
    try std.testing.expect(r.flag("verbose"));
    try std.testing.expectEqual(@as(?u32, 128), r.optionU32("max-tokens"));
    try std.testing.expectEqualStrings("model.gguf", r.positional(0).?);
}

test "parse unknown short treated as positional" {
    const specs = [_]ArgSpec{
        .{ .long = "help", .short = 'h' },
    };

    const argv = [_][*:0]const u8{ "agave", "-x", "model.gguf" };
    const args = std.process.Args{ .vector = &argv };
    var r = parse(std.testing.allocator, args, &specs);
    defer r.deinit();

    try std.testing.expectEqualStrings("-x", r.positional(0).?);
    try std.testing.expectEqualStrings("model.gguf", r.positional(1).?);
}

test "unknown char mid-cluster does not partially set flags" {
    const specs = [_]ArgSpec{
        .{ .long = "quiet", .short = 'q' },
        .{ .long = "verbose", .short = 'V' },
    };

    const argv = [_][*:0]const u8{ "agave", "-qZ", "model.gguf" };
    const args = std.process.Args{ .vector = &argv };
    var r = parse(std.testing.allocator, args, &specs);
    defer r.deinit();

    try std.testing.expect(!r.flag("quiet"));
    try std.testing.expect(!r.flag("verbose"));
    try std.testing.expectEqualStrings("-qZ", r.positional(0).?);
    try std.testing.expectEqualStrings("model.gguf", r.positional(1).?);
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

    try r.flags.put("help", {});
    try r.options.put("port", "8080");
    try r.options.put("seed", "42");
    try r.options.put("bad", "notanumber");
    try r.options.put("temp", "0.7");
    try r.options.put("inf", "inf");
    try r.positionals.append(std.testing.allocator, "model.gguf");

    try std.testing.expect(r.flag("help"));
    try std.testing.expect(!r.flag("version"));

    try std.testing.expectEqualStrings("8080", r.option("port").?);
    try std.testing.expect(r.option("missing") == null);

    try std.testing.expectEqual(@as(?u16, 8080), r.optionU16("port"));
    try std.testing.expectEqual(@as(?u32, 42), r.optionU32("seed"));
    try std.testing.expectEqual(@as(?u64, 42), r.optionU64("seed"));
    try std.testing.expect(r.optionU32("bad") == null);
    try std.testing.expect(r.optionU32("missing") == null);

    try std.testing.expectEqual(@as(?f32, 0.7), r.optionF32("temp"));
    try std.testing.expect(r.optionF32("bad") == null);
    try std.testing.expect(r.optionF32("inf") == null);
    try std.testing.expect(r.optionF32("missing") == null);

    try std.testing.expectEqualStrings("model.gguf", r.positional(0).?);
    try std.testing.expect(r.positional(1) == null);
}

test "findByLong empty specs" {
    const specs = [_]ArgSpec{};
    try std.testing.expect(findByLong(&specs, "anything") == null);
}

test "findByShort empty specs" {
    const specs = [_]ArgSpec{};
    try std.testing.expect(findByShort(&specs, 'x') == null);
}

test "findByShort no match when short is null" {
    const specs = [_]ArgSpec{
        .{ .long = "verbose" }, // no short alias
    };
    try std.testing.expect(findByShort(&specs, 'v') == null);
}

test "optionInt negative value" {
    var r = ParseResult{
        .flags = std.StringHashMap(void).init(std.testing.allocator),
        .options = std.StringHashMap([]const u8).init(std.testing.allocator),
        .positionals = .empty,
        .allocator = std.testing.allocator,
    };
    defer r.deinit();

    try r.options.put("val", "-5");
    // i32 should parse negative
    try std.testing.expectEqual(@as(?i32, -5), r.optionInt(i32, "val"));
    // u32 should fail on negative
    try std.testing.expect(r.optionInt(u32, "val") == null);
}

test "optionF32 negative and zero" {
    var r = ParseResult{
        .flags = std.StringHashMap(void).init(std.testing.allocator),
        .options = std.StringHashMap([]const u8).init(std.testing.allocator),
        .positionals = .empty,
        .allocator = std.testing.allocator,
    };
    defer r.deinit();

    try r.options.put("neg", "-0.5");
    try r.options.put("zero", "0.0");
    try r.options.put("nan", "nan");

    try std.testing.expectEqual(@as(?f32, -0.5), r.optionF32("neg"));
    try std.testing.expectEqual(@as(?f32, 0.0), r.optionF32("zero"));
    // NaN is not finite, should return null
    try std.testing.expect(r.optionF32("nan") == null);
}

test "ParseResult.flag and option" {
    var r = ParseResult{
        .flags = std.StringHashMap(void).init(std.testing.allocator),
        .options = std.StringHashMap([]const u8).init(std.testing.allocator),
        .positionals = .empty,
        .allocator = std.testing.allocator,
    };
    defer r.deinit();

    // Empty result
    try std.testing.expect(!r.flag("verbose"));
    try std.testing.expect(r.option("backend") == null);
    try std.testing.expect(r.positional(0) == null);
}

test "ArgSpec default values" {
    const spec = ArgSpec{ .long = "test" };
    try std.testing.expectEqual(@as(?u8, null), spec.short);
    try std.testing.expectEqual(.flag, spec.kind);
    try std.testing.expectEqualStrings("", spec.help);
}

test "fuzz: all cli functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const allocator = std.testing.allocator;

            var r = ParseResult{
                .flags = std.StringHashMap(void).init(allocator),
                .options = std.StringHashMap([]const u8).init(allocator),
                .positionals = .empty,
                .allocator = allocator,
            };
            defer r.deinit();

            const key_idx = smith.valueWithHash(u8, 0) % 4;
            const keys = [_][]const u8{ "help", "port", "temp", "seed" };
            const key = keys[key_idx];
            r.flags.put(key, {}) catch return;

            const val_idx = smith.valueWithHash(u8, 1) % 4;
            const vals = [_][]const u8{ "8080", "-3", "0.7", "nan" };
            r.options.put(key, vals[val_idx]) catch return;

            const pos_idx = smith.valueWithHash(u8, 2) % 3;
            const positional_vals = [_][]const u8{ "model.gguf", "", "arg2" };
            r.positionals.append(allocator, positional_vals[pos_idx]) catch return;

            _ = r.flag(key);
            _ = r.flag("nonexistent");

            _ = r.option(key);
            _ = r.option("missing");

            const pos_query = smith.valueWithHash(usize, 3);
            _ = r.positional(pos_query);
            _ = r.positional(0);

            _ = r.optionInt(u8, key);
            _ = r.optionInt(i32, key);
            _ = r.optionInt(u64, key);
            _ = r.optionInt(u32, "missing");

            _ = r.optionU16(key);
            _ = r.optionU16("missing");

            _ = r.optionU32(key);
            _ = r.optionU32("missing");

            _ = r.optionU64(key);
            _ = r.optionU64("missing");

            const f32_result = r.optionF32(key);
            if (f32_result) |v| {
                std.debug.assert(std.math.isFinite(v));
            }
            _ = r.optionF32("missing");

            // parse() takes std.process.Args (OS-specific); verify reference at comptime
            comptime {
                _ = &parse;
            }

            const short_val = smith.valueWithHash(u8, 4);
            const spec = ArgSpec{
                .long = key,
                .short = if (short_val > 128) short_val else null,
                .kind = if (smith.valueWithHash(u1, 5) == 0) .flag else .option,
                .help = "fuzz help",
            };
            _ = spec.long;
            _ = spec.short;
            _ = spec.kind;
            _ = spec.help;
        }
    }.f, .{});
}
