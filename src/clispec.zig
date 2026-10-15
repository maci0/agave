//! CLI Spec 0.3 document for the agave binary.
//! `schema` and `version` run before a model is loaded.

const std = @import("std");
const display = @import("display.zig");

const Io = std.Io;
const stdout_file = Io.File.stdout();
const stderr_file = Io.File.stderr();

const output_values = [_][]const u8{ "auto", "text", "json" };
const bad_output_msg = "agave: --output requires auto, text, or json";

const Command = struct {
    name: []const u8,
    description: []const u8,
    effects: []const u8,
    /// JSON fields after `effects`, without a leading comma.
    tail: []const u8,
};

const commands = [_]Command{
    .{
        .name = "schema",
        .description = "Print this CLI spec document. Needs no model, no config, and no network.",
        .effects = "read_only",
        .tail = "\"cardinality\":\"single\",\"output_fields\":[{\"name\":\"clispec\",\"type\":\"string\"},{\"name\":\"name\",\"type\":\"string\"},{\"name\":\"version\",\"type\":\"string\"},{\"name\":\"commands\",\"type\":\"array\",\"items\":{\"type\":\"object\"}}]",
    },
    .{
        .name = "version",
        .description = "Print the product version.",
        .effects = "read_only",
        .tail = "\"cardinality\":\"single\",\"output_fields\":[{\"name\":\"version\",\"type\":\"string\"}]",
    },
    .{
        .name = "pull",
        .description = "Download a model from Hugging Face. A retry is not safe.",
        .effects = "non_idempotent",
        .tail = "\"output_kind\":\"opaque\",\"media_type\":\"text/plain\"",
    },
    .{
        .name = "calibrate",
        .description = "Write TriAttention calibration data for a model. A retry is not safe.",
        .effects = "non_idempotent",
        .tail = "\"output_kind\":\"opaque\",\"media_type\":\"text/plain\"",
    },
    .{
        .name = "update",
        .description = "Replace this binary with a published release. A retry is not safe.",
        .effects = "non_idempotent",
        .tail = "\"output_kind\":\"opaque\",\"media_type\":\"text/plain\"",
    },
    .{
        .name = "run",
        .description = "Generate, serve, print model info, or benchmark. Invoked as agave [options] <model> [prompt], without the word run. A retry is not safe.",
        .effects = "non_idempotent",
        .tail = "\"cardinality\":\"single\",\"output_fields\":[{\"name\":\"model\",\"type\":\"string\"},{\"name\":\"arch\",\"type\":\"string\"},{\"name\":\"quant\",\"type\":\"string\"},{\"name\":\"backend\",\"type\":\"string\"},{\"name\":\"version\",\"type\":\"string\"},{\"name\":\"output\",\"type\":\"string\"},{\"name\":\"tokens\",\"type\":\"integer\"},{\"name\":\"tok_per_sec\",\"type\":\"number\"},{\"name\":\"prefill_tokens\",\"type\":\"integer\"},{\"name\":\"prefill_ms\",\"type\":\"integer\"},{\"name\":\"prefill_tok_per_sec\",\"type\":\"number\"},{\"name\":\"gen_ms\",\"type\":\"integer\"}]",
    },
};

fn writeStdout(text: []const u8) void {
    _ = std.posix.system.write(stdout_file.handle, text.ptr, text.len);
}

fn writeStderr(text: []const u8) void {
    _ = std.posix.system.write(stderr_file.handle, text.ptr, text.len);
}

fn parseMode(value: []const u8) ?[]const u8 {
    for (output_values) |allowed| {
        if (std.mem.eql(u8, value, allowed)) return allowed;
    }
    return null;
}

const Scan = struct {
    word: ?[]const u8 = null,
    bad: ?[]const u8 = null,
    json: bool = false,
};

fn scan(args: std.process.Args) Scan {
    var iter = args.iterate();
    _ = iter.next();
    var out = Scan{};
    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--")) return out;
        if (std.mem.eql(u8, arg, "--json")) {
            out.json = true;
            continue;
        }
        const value: ?[]const u8 = if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output"))
            iter.next()
        else if (std.mem.startsWith(u8, arg, "--output="))
            arg["--output=".len..]
        else if (arg.len > 2 and arg[0] == '-' and arg[1] == 'o')
            arg[2..]
        else
            null;
        if (value != null or std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output") or std.mem.startsWith(u8, arg, "--output=") or (arg.len > 2 and arg[0] == '-' and arg[1] == 'o')) {
            const got = value orelse {
                out.bad = bad_output_msg;
                return out;
            };
            const mode = parseMode(got) orelse {
                out.bad = bad_output_msg;
                return out;
            };
            out.json = std.mem.eql(u8, mode, "json");
            continue;
        }
        out.word = arg;
        return out;
    }
    return out;
}

pub fn badOutput(args: std.process.Args) ?[]const u8 {
    return scan(args).bad;
}

pub fn commandWord(args: std.process.Args) ?[]const u8 {
    return scan(args).word;
}

pub fn wantsJson(args: std.process.Args) bool {
    var iter = args.iterate();
    _ = iter.next();
    var json = false;
    var word: ?[]const u8 = null;
    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--")) break;
        if (std.mem.eql(u8, arg, "--json")) {
            json = true;
            continue;
        }
        const short_o = arg.len > 2 and arg[0] == '-' and arg[1] == 'o';
        const is_output = std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output") or std.mem.startsWith(u8, arg, "--output=") or short_o;
        if (is_output) {
            const value: ?[]const u8 = if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output"))
                iter.next()
            else if (std.mem.startsWith(u8, arg, "--output="))
                arg["--output=".len..]
            else
                arg[2..];
            // `agave calibrate --output` is the .cal path, not the format.
            if (word != null and std.mem.eql(u8, word.?, "calibrate")) continue;
            if (value) |got| {
                if (parseMode(got)) |mode| json = std.mem.eql(u8, mode, "json");
            }
            continue;
        }
        if (word == null and arg.len > 0 and arg[0] != '-') word = arg;
    }
    return json;
}

pub fn writeError(args: std.process.Args, kind: []const u8, message: []const u8) void {
    if (!wantsJson(args)) return;
    var buf: [192]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, "{{\"error\":{{\"kind\":\"{s}\",\"message\":\"{s}\"}}}}\n", .{ kind, message }) catch return;
    writeStderr(line);
}

pub fn printVersionCommand(args: std.process.Args) void {
    if (wantsJson(args)) {
        var buf: [80]u8 = undefined;
        const line = std.fmt.bufPrint(&buf, "{{\"version\":\"{s}\"}}\n", .{display.version}) catch return;
        writeStdout(line);
        return;
    }
    display.printVersion();
}

fn matches(name: []const u8, path: []const []const u8) bool {
    if (path.len == 0) return true;
    var rest = name;
    for (path, 0..) |seg, index| {
        if (!std.mem.startsWith(u8, rest, seg)) return false;
        rest = rest[seg.len..];
        if (index + 1 == path.len) return rest.len == 0 or rest[0] == ' ';
        if (rest.len == 0 or rest[0] != ' ') return false;
        rest = rest[1..];
    }
    return true;
}

fn append(buf: []u8, used: *usize, text: []const u8) !void {
    if (used.* + text.len > buf.len) return error.NoSpace;
    @memcpy(buf[used.*..][0..text.len], text);
    used.* += text.len;
}

pub fn renderSchema(buf: []u8, version: []const u8, path: []const []const u8) ![]const u8 {
    var matched: usize = 0;
    for (commands) |cmd| {
        if (matches(cmd.name, path)) matched += 1;
    }
    if (matched == 0) return error.Unknown;

    var used: usize = 0;
    try append(buf, &used, "{\"clispec\":\"0.3\",\"name\":\"agave\",\"version\":\"");
    try append(buf, &used, version);
    try append(buf, &used, "\",\"description\":\"Zig LLM inference engine.\",\"output\":{\"tty\":\"text\",\"piped\":\"text\"},\"global_args\":[{\"name\":\"--output\",\"short\":\"-o\",\"type\":\"string\",\"enum\":[\"auto\",\"text\",\"json\"],\"default\":\"auto\",\"description\":\"Before the command. auto stays text when piped.\"},{\"name\":\"--json\",\"type\":\"boolean\",\"description\":\"Same as --output json.\"}],\"commands\":[");
    var wrote: usize = 0;
    for (commands) |cmd| {
        if (!matches(cmd.name, path)) continue;
        if (wrote != 0) try append(buf, &used, ",");
        try append(buf, &used, "{\"name\":\"");
        try append(buf, &used, cmd.name);
        try append(buf, &used, "\",\"description\":\"");
        try append(buf, &used, cmd.description);
        try append(buf, &used, "\",\"effects\":\"");
        try append(buf, &used, cmd.effects);
        try append(buf, &used, "\",");
        try append(buf, &used, cmd.tail);
        try append(buf, &used, "}");
        wrote += 1;
    }
    try append(buf, &used, "],\"errors\":[{\"kind\":\"usage\",\"exit_code\":2,\"retryable\":false,\"description\":\"Unknown command, flag, or a rejected flag value.\"},{\"kind\":\"failure\",\"exit_code\":1,\"retryable\":false,\"description\":\"The command was valid and did not finish.\"}]}");
    try append(buf, &used, "\n");
    return buf[0..used];
}

fn schemaPath(args: std.process.Args, storage: [][]const u8) ![]const []const u8 {
    var iter = args.iterate();
    _ = iter.next();
    var seen_schema = false;
    var count: usize = 0;
    while (iter.next()) |arg| {
        if (!seen_schema) {
            if (std.mem.eql(u8, arg, "schema")) seen_schema = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "help")) {
            return error.Help;
        }
        if (arg.len > 0 and arg[0] == '-') return error.BadFlag;
        if (count >= storage.len) return error.BadFlag;
        storage[count] = arg;
        count += 1;
    }
    return storage[0..count];
}

pub fn schemaMain(args: std.process.Args) u8 {
    var storage: [8][]const u8 = undefined;
    const path = schemaPath(args, &storage) catch |err| switch (err) {
        error.Help => {
            writeStdout("usage: agave schema [command]\n");
            return 0;
        },
        error.BadFlag => {
            writeStderr("usage: agave schema [command]\n");
            return 2;
        },
    };
    var buf: [8192]u8 = undefined;
    const doc = renderSchema(&buf, display.version, path) catch |err| switch (err) {
        error.Unknown => {
            writeStderr("unknown command path\n");
            return 2;
        },
        error.NoSpace => {
            writeStderr("Error: schema document does not fit\n");
            return 1;
        },
    };
    writeStdout(doc);
    return 0;
}

/// Skip argv[0] and every token up to and including `word`.
pub fn skipThrough(iter: *std.process.Args.Iterator, word: []const u8) void {
    _ = iter.skip();
    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, word)) return;
    }
}

test "schema document is clispec 0.3" {
    var buf: [8192]u8 = undefined;
    const doc = try renderSchema(&buf, "0.13.0", &.{});
    try std.testing.expect(std.mem.startsWith(u8, doc, "{\"clispec\":\"0.3\""));
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"name\":\"agave\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"version\":\"0.13.0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"piped\":\"text\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"name\":\"--output\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"kind\":\"usage\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"name\":\"version\"") != null);
    try std.testing.expect(std.mem.endsWith(u8, doc, "\n"));

    const filtered = try renderSchema(&buf, "0.13.0", &.{"pull"});
    try std.testing.expect(std.mem.indexOf(u8, filtered, "\"name\":\"pull\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, filtered, "\"name\":\"version\"") == null);
    try std.testing.expectError(error.Unknown, renderSchema(&buf, "0.13.0", &.{"nope"}));
}
