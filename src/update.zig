//! `agave update`: compare this build with the latest GitHub release and,
//! when asked to install, replace the running executable only after its bytes
//! match the `.sha256` sidecar the release publishes.
//!
//! The decision (repo shape, exact version, asset name, checksum, trusted
//! URL) is pure. `run` is the only function that talks to GitHub or
//! names the running executable, and tests never execute `run` directly.

const std = @import("std");
const builtin = @import("builtin");
const config = @import("config.zig");
const display_mod = @import("display.zig");
const test_stdout = @import("test_stdout.zig");

const version = @import("build_options").version;

pub const default_repo = "maci0/agave";
pub const tool_name = "agave";

const stdout_file = std.Io.File.stdout();
const stderr_file = std.Io.File.stderr();

const exec_mode: std.Io.File.Permissions = @fromBackingInt(@intCast(@as(std.posix.mode_t, 0o755)));

const max_api_bytes: usize = 10 * 1024 * 1024;
const max_sidecar_bytes: usize = 64 * 1024;
const max_asset_bytes: usize = 256 * 1024 * 1024;

pub const Verdict = enum {
    current,
    missing_asset,
    untrusted_url,
    missing_sidecar,
    checksum_mismatch,
    replaced,
};

pub const Inputs = struct {
    running: []const u8,
    tag: []const u8,
    asset_url: ?[]const u8 = null,
    asset: ?[]const u8 = null,
    sidecar_url: ?[]const u8 = null,
    sidecar: ?[]const u8 = null,
    basename: []const u8 = "",
};

pub const ListedAsset = struct {
    name: []const u8,
    url: []const u8,
};

pub const Release = struct {
    tag: []const u8,
    page: []const u8,
    assets: []const ListedAsset,
};

/// One leading `v` on the tag, then exact equality. `v0.7.0` is not `0.7.0.1`.
pub fn sameRelease(running: []const u8, tag: []const u8) bool {
    const bare = if (std.mem.startsWith(u8, tag, "v")) tag[1..] else tag;
    return std.mem.eql(u8, running, bare);
}

/// The release matrix names macOS `aarch64-macos` and `x86_64-macos` (no abi)
/// and Linux `arch-linux-musl` or `arch-linux-gnu`. Zig's abi tag for those macOS
/// targets is `none`; appending it asks for an asset the release does not publish.
pub fn targetTriple(buf: []u8, arch: []const u8, os_name: []const u8, abi: []const u8) []const u8 {
    if (std.mem.eql(u8, abi, "none")) {
        return std.fmt.bufPrint(buf, "{s}-{s}", .{ arch, os_name }) catch buf[0..0];
    }
    return std.fmt.bufPrint(buf, "{s}-{s}-{s}", .{ arch, os_name, abi }) catch buf[0..0];
}

pub fn thisTarget(buf: []u8) []const u8 {
    return targetTriple(buf, @tagName(builtin.cpu.arch), @tagName(builtin.os.tag), @tagName(builtin.abi));
}

pub fn writeAssetName(buf: []u8, tag: []const u8, target: []const u8) error{NameTooLong}![]const u8 {
    return std.fmt.bufPrint(buf, "agave-{s}-{s}", .{ tag, target }) catch return error.NameTooLong;
}

pub fn writeSidecarName(buf: []u8, asset_name: []const u8) error{NameTooLong}![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.sha256", .{asset_name}) catch return error.NameTooLong;
}

fn repoPartOk(part: []const u8) bool {
    if (part.len == 0 or part.len > 100) return false;
    if (std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) return false;
    for (part) |c| {
        if (!(std.ascii.isAlphanumeric(c) or c == '_' or c == '.' or c == '-')) return false;
    }
    return true;
}

/// `owner/name` only. A URL, a second slash, or an empty side is not a repo.
pub fn validRepo(text: []const u8) bool {
    if (std.mem.indexOf(u8, text, "://") != null) return false;
    const slash = std.mem.findScalar(u8, text, '/') orelse return false;
    const owner = text[0..slash];
    const name = text[slash + 1 ..];
    if (std.mem.findScalar(u8, name, '/') != null) return false;
    return repoPartOk(owner) and repoPartOk(name);
}

/// The release API URL. A repo that is not `owner/name` fails here, before
/// any bytes are requested.
pub fn releaseApiUrl(buf: []u8, repo: []const u8) error{ BadRepo, NameTooLong }![]const u8 {
    if (!validRepo(repo)) return error.BadRepo;
    return std.fmt.bufPrint(buf, "https://api.github.com/repos/{s}/releases/latest", .{repo}) catch
        return error.NameTooLong;
}

fn hostTrusted(host: []const u8) bool {
    var lower: [253]u8 = undefined;
    if (host.len == 0 or host.len > lower.len) return false;
    for (host, 0..) |c, i| lower[i] = std.ascii.toLower(c);
    const h = lower[0..host.len];
    if (std.mem.eql(u8, h, "github.com")) return true;
    if (std.mem.endsWith(u8, h, ".github.com")) return true;
    if (std.mem.endsWith(u8, h, ".githubusercontent.com")) return true;
    return false;
}

/// https, and the host is `github.com`, `*.github.com`, or `*.githubusercontent.com`.
/// Userinfo and lookalikes such as `github.com.evil.com` are refused.
pub fn trustedGithubUrl(url: []const u8) bool {
    const prefix = "https://";
    if (url.len < prefix.len) return false;
    for (prefix, 0..) |c, i| {
        if (std.ascii.toLower(url[i]) != c) return false;
    }
    const rest = url[prefix.len..];
    if (std.mem.indexOfAny(u8, rest, "@\\ \t\r\n") != null) return false;
    const slash = std.mem.findScalar(u8, rest, '/') orelse rest.len;
    var host = rest[0..slash];
    if (std.mem.findScalar(u8, host, ':')) |colon| {
        const port = host[colon + 1 ..];
        if (port.len == 0) return false;
        for (port) |c| if (!std.ascii.isDigit(c)) return false;
        host = host[0..colon];
    }
    return hostTrusted(host);
}

/// Stdout of `--check` is this URL, or error when the page is not a trusted GitHub URL.
pub fn releasePageLine(url: []const u8) error{UntrustedUrl}![]const u8 {
    if (!trustedGithubUrl(url)) return error.UntrustedUrl;
    return url;
}

/// `--check` never downloads an asset. An equal version never does either.
pub fn fetchesAsset(check_only: bool, running: []const u8, tag: []const u8) bool {
    if (check_only) return false;
    return !sameRelease(running, tag);
}

/// Sidecar line from `scripts/release-checksum.sh`: `<hex>  <basename>`.
pub fn checksumMatches(asset: []const u8, sidecar: []const u8, basename: []const u8) bool {
    const line_end = std.mem.findScalar(u8, sidecar, '\n') orelse sidecar.len;
    var line = sidecar[0..line_end];
    if (line.len > 0 and line[line.len - 1] == '\r') line = line[0 .. line.len - 1];
    if (line.len < 66) return false;
    const hex = line[0..64];
    if (!std.mem.eql(u8, line[64..66], "  ")) return false;
    if (!std.mem.eql(u8, line[66..], basename)) return false;
    for (hex) |c| if (!std.ascii.isHex(c)) return false;
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(asset, &digest, .{});
    const got = std.fmt.bytesToHex(digest, .lower);
    for (hex, 0..) |c, i| {
        if (std.ascii.toLower(c) != got[i]) return false;
    }
    return true;
}

pub fn decide(in: Inputs) Verdict {
    if (sameRelease(in.running, in.tag)) return .current;
    const url = in.asset_url orelse return .missing_asset;
    if (!trustedGithubUrl(url)) return .untrusted_url;
    const side_url = in.sidecar_url orelse return .missing_sidecar;
    if (!trustedGithubUrl(side_url)) return .untrusted_url;
    const bytes = in.asset orelse return .missing_asset;
    const side = in.sidecar orelse return .missing_sidecar;
    if (side.len == 0) return .missing_sidecar;
    if (!checksumMatches(bytes, side, in.basename)) return .checksum_mismatch;
    return .replaced;
}

/// Writes `asset` over `dest_name` only when the verdict is `replaced`.
/// Follows symlinks to ensure the real target binary is replaced rather
/// than breaking the link. Permissions are set to executable (0755).
pub fn replaceVerified(
    io: std.Io,
    dir: std.Io.Dir,
    dest_name: []const u8,
    decision: Verdict,
    asset: []const u8,
) !void {
    if (decision != .replaced) return error.Refused;
    var link_buf: [4096]u8 = undefined;
    var joined_buf: [4096]u8 = undefined;
    const target: []const u8 = if (dir.readLink(io, dest_name, &link_buf)) |n| blk: {
        const link = link_buf[0..n];
        if (link.len > 0 and link[0] == '/') break :blk link;
        const dir_end = std.mem.findScalarLast(u8, dest_name, '/') orelse break :blk link;
        break :blk std.fmt.bufPrint(&joined_buf, "{s}/{s}", .{ dest_name[0..dir_end], link }) catch break :blk link;
    } else |err| switch (err) {
        error.NotLink, error.FileNotFound => dest_name,
        else => return err,
    };

    var af = try dir.createFileAtomic(io, target, .{ .replace = true, .make_path = true, .permissions = exec_mode });
    defer af.deinit(io);
    try af.file.writeStreamingAll(io, asset);
    try af.replace(io);
}

pub fn formatCurrent(buf: []u8, tool: []const u8, running: []const u8, tag: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s} {s} is current (latest release: {s})", .{ tool, running, tag });
}

pub fn formatNewRelease(buf: []u8, tag: []const u8, running: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "New release: {s} (running {s})", .{ tag, running });
}

pub fn formatInstalled(buf: []u8, tag: []const u8, path: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "Installed {s} to {s}", .{ tag, path });
}

pub fn parseRelease(arena: std.mem.Allocator, body: []const u8) !Release {
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{}) catch return error.MalformedRelease;
    const obj = switch (parsed) {
        .object => |o| o,
        else => return error.MalformedRelease,
    };
    const tag = switch (obj.get("tag_name") orelse return error.MalformedRelease) {
        .string => |s| s,
        else => return error.MalformedRelease,
    };
    const page = switch (obj.get("html_url") orelse return error.MalformedRelease) {
        .string => |s| s,
        else => return error.MalformedRelease,
    };
    const arr = switch (obj.get("assets") orelse return error.MalformedRelease) {
        .array => |a| a,
        else => return error.MalformedRelease,
    };
    var list: std.ArrayList(ListedAsset) = .empty;
    for (arr.items) |item| {
        const asset_obj = switch (item) {
            .object => |o| o,
            else => continue,
        };
        const name = switch (asset_obj.get("name") orelse continue) {
            .string => |s| s,
            else => continue,
        };
        const url = switch (asset_obj.get("browser_download_url") orelse continue) {
            .string => |s| s,
            else => continue,
        };
        try list.append(arena, .{ .name = name, .url = url });
    }
    return .{
        .tag = tag,
        .page = page,
        .assets = try list.toOwnedSlice(arena),
    };
}

pub fn assetUrl(rel: Release, name: []const u8) ?[]const u8 {
    for (rel.assets) |asset| {
        if (std.mem.eql(u8, asset.name, name)) return asset.url;
    }
    return null;
}

fn writeErr(bytes: []const u8) void {
    _ = std.posix.system.write(stderr_file.handle, bytes.ptr, bytes.len);
}

fn writeOut(bytes: []const u8) void {
    _ = std.posix.system.write(stdout_file.handle, bytes.ptr, bytes.len);
}

fn fail(comptime fmt: []const u8, args: anytype) u8 {
    var buf: [512]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, "Error: " ++ fmt ++ "\n", args) catch "Error: update failed\n";
    writeErr(line);
    return 1;
}

/// Bound on the gap between two reads of a GitHub API response, in seconds.
/// Without it a proxy or middlebox that accepts the connection and then stops
/// responding leaves `agave update` blocked forever with no output. Matches the
/// download stall bound in `pull.zig`; it never fires while bytes keep arriving.
const fetch_stall_timeout_sec: i64 = 60;

/// Apply SO_RCVTIMEO to the socket backing `req` so the body read cannot block
/// forever. Advisory: a socket that cannot take the option is left untimed.
fn setFetchSocketTimeout(req: std.http.Client.Request, seconds: i64) void {
    const conn = req.connection orelse return;
    const timeout = std.posix.timeval{ .sec = seconds, .usec = 0 };
    std.posix.setsockopt(
        conn.stream_reader.stream.socket.handle,
        std.posix.SOL.SOCKET,
        std.posix.SO.RCVTIMEO,
        std.mem.asBytes(&timeout),
    ) catch {};
}

fn fetchBody(
    io: std.Io,
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    url: []const u8,
    bearer: ?[]const u8,
    max_size: usize,
) ![]const u8 {
    var client: std.http.Client = .{ .allocator = gpa, .io = io };
    defer client.deinit();

    var priv_headers_buf: [1]std.http.Header = undefined;
    const priv_headers: []const std.http.Header = if (bearer) |b| blk: {
        priv_headers_buf[0] = .{ .name = "Authorization", .value = b };
        break :blk priv_headers_buf[0..1];
    } else &.{};

    const headers: std.http.Client.Request.Headers = .{
        .user_agent = .{ .override = "agave/" ++ version },
    };

    // Goes through `request` rather than `fetch` so the socket is reachable
    // for the read timeout; `fetch` hides it and would leave this call able to
    // hang indefinitely.
    const uri = std.Uri.parse(url) catch |err| return err;
    var req = client.request(.GET, uri, .{
        .headers = headers,
        .privileged_headers = priv_headers,
    }) catch |err| return err;
    defer req.deinit();
    try req.sendBodiless();
    setFetchSocketTimeout(req, fetch_stall_timeout_sec);

    var redirect_buf: [8 * 1024]u8 = undefined;
    var response = req.receiveHead(&redirect_buf) catch |err| return err;

    const status_code = @backingInt(response.head.status);
    if (status_code >= 400) return error.HttpStatus;

    var transfer_buf: [64 * 1024]u8 = undefined;
    const reader = response.reader(&transfer_buf);
    var writer = try std.Io.Writer.Allocating.initCapacity(gpa, @min(max_size, 64 * 1024));
    defer writer.deinit();
    _ = reader.streamRemaining(&writer.writer) catch |err| return err;

    const data = writer.written();
    if (data.len > max_size) return error.PayloadTooLarge;

    return try arena.dupe(u8, data);
}

fn replaceExecutable(io: std.Io, gpa: std.mem.Allocator, asset: []const u8) ![]const u8 {
    const exe = try std.process.executablePathAlloc(io, gpa);
    const base = std.fs.path.basename(exe);
    if (std.fs.path.dirname(exe)) |dir_path| {
        var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{});
        defer dir.close(io);
        try replaceVerified(io, dir, base, .replaced, asset);
    } else {
        try replaceVerified(io, std.Io.Dir.cwd(), base, .replaced, asset);
    }
    return exe;
}

pub fn printUsage() void {
    const usage =
        \\Usage: agave update [OPTIONS]
        \\
        \\Compare this build with the latest GitHub release and, unless --check is
        \\passed, replace the running executable after verifying the SHA-256 sidecar.
        \\
        \\OPTIONS:
        \\  -h, --help           Show this help message and exit
        \\  -v, --version        Print version and exit
        \\  -c, --check          Report latest version without downloading or replacing
        \\      --repo <OWNER/R> GitHub repository to check [default: maci0/agave]
        \\
        \\ENVIRONMENT:
        \\  GITHUB_TOKEN         GitHub personal access token (avoids API rate limits)
        \\
    ;
    _ = std.posix.system.write(stdout_file.handle, usage.ptr, usage.len);
}

/// Subcommand entry point called by main.zig.
/// Returns exit code (0 = success/current, 2 = usage error, 1 = operational failure).
pub fn run(allocator: std.mem.Allocator, process_args: std.process.Args, io: std.Io) u8 {
    var args_iter = process_args.iterate();
    _ = args_iter.skip(); // argv[0]
    _ = args_iter.skip(); // "update"

    var check_only = false;
    var repo_arg: ?[]const u8 = null;

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "help")) {
            printUsage();
            return 0;
        } else if (std.mem.eql(u8, arg, "--version") or std.mem.eql(u8, arg, "-v")) {
            display_mod.printVersion();
            return 0;
        } else if (std.mem.eql(u8, arg, "--check") or std.mem.eql(u8, arg, "-c")) {
            check_only = true;
        } else if (std.mem.eql(u8, arg, "--repo")) {
            const val = args_iter.next() orelse {
                writeErr("Error: --repo requires a value (e.g. maci0/agave)\n");
                writeErr("Run 'agave update --help' for more information.\n");
                return 2;
            };
            repo_arg = val;
        } else if (std.mem.startsWith(u8, arg, "--repo=")) {
            repo_arg = arg["--repo=".len..];
        } else if (arg.len > 0 and arg[0] == '-') {
            var buf: [160]u8 = undefined;
            const line = std.fmt.bufPrint(&buf, "Error: unknown option '{s}'\n", .{arg}) catch "Error: unknown option\n";
            writeErr(line);
            writeErr("Run 'agave update --help' for more information.\n");
            return 2;
        } else {
            var buf: [160]u8 = undefined;
            const line = std.fmt.bufPrint(&buf, "Error: unexpected argument '{s}'\n", .{arg}) catch "Error: unexpected argument\n";
            writeErr(line);
            writeErr("Run 'agave update --help' for more information.\n");
            return 2;
        }
    }

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const repo = repo_arg orelse default_repo;
    var api_buf: [240]u8 = undefined;
    const api = releaseApiUrl(&api_buf, repo) catch {
        const shown = repo[0..@min(repo.len, 80)];
        var msg: [160]u8 = undefined;
        const line = std.fmt.bufPrint(&msg, "Error: want owner/repo, not a URL (got '{s}')\n", .{shown}) catch
            "Error: want owner/repo, not a URL\n";
        writeErr(line);
        return 2;
    };

    const token = config.getenv("GITHUB_TOKEN");
    var bearer_buf: [256]u8 = undefined;
    const bearer: ?[]const u8 = if (token) |t|
        std.fmt.bufPrint(&bearer_buf, "Bearer {s}", .{t}) catch null
    else
        null;

    const body = fetchBody(io, allocator, arena, api, bearer, max_api_bytes) catch |err| {
        return fail("could not reach GitHub ({s})", .{@errorName(err)});
    };
    const rel = parseRelease(arena, body) catch return fail("the latest release could not be read", .{});
    const page = releasePageLine(rel.page) catch return fail("refusing to install unverified binary", .{});

    var line_buf: [256]u8 = undefined;
    if (sameRelease(version, rel.tag)) {
        const line = formatCurrent(&line_buf, tool_name, version, rel.tag) catch
            return fail("could not format the version comparison", .{});
        writeErr(line);
        writeErr("\n");
    } else {
        const line = formatNewRelease(&line_buf, rel.tag, version) catch
            return fail("could not format the version comparison", .{});
        writeErr(line);
        writeErr("\n");
    }

    if (!fetchesAsset(check_only, version, rel.tag)) {
        if (check_only) {
            writeOut(page);
            writeOut("\n");
        }
        return 0;
    }

    var target_buf: [64]u8 = undefined;
    const target = thisTarget(&target_buf);
    var name_buf: [192]u8 = undefined;
    const asset_name = writeAssetName(&name_buf, rel.tag, target) catch
        return fail("release asset name does not fit", .{});
    var side_name_buf: [208]u8 = undefined;
    const side_name = writeSidecarName(&side_name_buf, asset_name) catch
        return fail("release asset name does not fit", .{});

    const a_url = assetUrl(rel, asset_name) orelse
        return fail("missing release asset {s}; the binary was not replaced", .{asset_name});
    const s_url = assetUrl(rel, side_name) orelse
        return fail("missing checksum sidecar; the binary was not replaced", .{});
    if (!trustedGithubUrl(a_url) or !trustedGithubUrl(s_url)) {
        return fail("refusing to install unverified binary", .{});
    }

    const asset = fetchBody(io, allocator, arena, a_url, bearer, max_asset_bytes) catch |err| {
        return fail("could not download {s} ({s}); the binary was not replaced", .{ asset_name, @errorName(err) });
    };
    const sidecar = fetchBody(io, allocator, arena, s_url, bearer, max_sidecar_bytes) catch |err| {
        return fail("could not download the checksum sidecar ({s}); the binary was not replaced", .{@errorName(err)});
    };

    const decision = decide(.{
        .running = version,
        .tag = rel.tag,
        .asset_url = a_url,
        .asset = asset,
        .sidecar_url = s_url,
        .sidecar = sidecar,
        .basename = asset_name,
    });
    switch (decision) {
        .replaced => {},
        .checksum_mismatch => return fail("checksum mismatch; refusing to install unverified binary", .{}),
        .missing_sidecar => return fail("missing checksum sidecar; the binary was not replaced", .{}),
        .missing_asset => return fail("missing release asset; the binary was not replaced", .{}),
        .untrusted_url => return fail("refusing to install unverified binary", .{}),
        .current => return 0,
    }

    const exe = replaceExecutable(io, allocator, asset) catch |err| {
        return fail("could not replace the binary ({s})", .{@errorName(err)});
    };
    const installed = formatInstalled(&line_buf, rel.tag, exe) catch
        return fail("could not format the install line", .{});
    writeOut(installed);
    writeOut("\n");
    return 0;
}

// ── Tests ───────────────────────────────────────────────────────────────────

const abc_sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
const asset_base = "agave-v0.7.0-x86_64-linux-gnu";

fn copyOf(io: std.Io, dir: std.Io.Dir) ![]u8 {
    return dir.readFileAlloc(io, "agave", std.testing.allocator, .limited(64));
}

test "update: a v-prefixed tag equals the running version exactly" {
    try std.testing.expect(sameRelease("0.7.0", "v0.7.0"));
    try std.testing.expect(sameRelease("0.7.0", "0.7.0"));
    try std.testing.expect(!sameRelease("0.7.0", "v0.7.0.1"));
    try std.testing.expect(!sameRelease("0.7.1", "v0.7.10"));
    try std.testing.expect(!sameRelease("0.7.10", "v0.7.1"));
    try std.testing.expect(sameRelease("0.7.10", "v0.7.10"));
    try std.testing.expect(!sameRelease("v0.7.0", "v0.7.0"));
    try std.testing.expect(!sameRelease("0.7.0", "vv0.7.0"));
}

test "update: asset name is agave-tag-target" {
    var buf: [80]u8 = undefined;
    const name = try writeAssetName(&buf, "v0.7.0", "x86_64-linux-gnu");
    try std.testing.expectEqualStrings("agave-v0.7.0-x86_64-linux-gnu", name);
    var side: [96]u8 = undefined;
    try std.testing.expectEqualStrings("agave-v0.7.0-x86_64-linux-gnu.sha256", try writeSidecarName(&side, name));
}

test "update: this target is the name the release matrix publishes" {
    const rows = .{
        .{ "x86_64", "linux", "gnu", "x86_64-linux-gnu" },
        .{ "aarch64", "linux", "gnu", "aarch64-linux-gnu" },
        .{ "x86_64", "linux", "musl", "x86_64-linux-musl" },
        .{ "aarch64", "linux", "musl", "aarch64-linux-musl" },
        .{ "aarch64", "macos", "none", "aarch64-macos" },
        .{ "x86_64", "macos", "none", "x86_64-macos" },
    };
    inline for (rows) |row| {
        var triple_buf: [64]u8 = undefined;
        const triple = targetTriple(&triple_buf, row[0], row[1], row[2]);
        try std.testing.expectEqualStrings(row[3], triple);
        var name_buf: [96]u8 = undefined;
        const asset = try writeAssetName(&name_buf, "v0.7.0", triple);
        try std.testing.expectEqualStrings("agave-v0.7.0-" ++ row[3], asset);
    }
    var live_buf: [64]u8 = undefined;
    var via_buf: [64]u8 = undefined;
    const want = targetTriple(&via_buf, @tagName(builtin.cpu.arch), @tagName(builtin.os.tag), @tagName(builtin.abi));
    const got = thisTarget(&live_buf);
    try std.testing.expectEqualStrings(want, got);
    try std.testing.expect(!std.mem.endsWith(u8, got, "-none"));
}

test "update: a repo that is not owner/name is refused before a release url exists" {
    var buf: [160]u8 = undefined;
    try std.testing.expectError(error.BadRepo, releaseApiUrl(&buf, "https://github.com/maci0/agave"));
    try std.testing.expectError(error.BadRepo, releaseApiUrl(&buf, "maci0/agave/extra"));
    try std.testing.expectError(error.BadRepo, releaseApiUrl(&buf, "maci0"));
    try std.testing.expectError(error.BadRepo, releaseApiUrl(&buf, "/agave"));
    try std.testing.expect(!validRepo("maci0/agave/"));
    const url = try releaseApiUrl(&buf, default_repo);
    try std.testing.expectEqualStrings("https://api.github.com/repos/maci0/agave/releases/latest", url);
}

test "update: a release page that is not https on a GitHub host is not printed" {
    try std.testing.expectError(error.UntrustedUrl, releasePageLine("http://github.com/maci0/agave/releases/tag/v0.7.0"));
    try std.testing.expectError(error.UntrustedUrl, releasePageLine("https://github.com.evil.com/maci0/agave"));
    try std.testing.expectError(error.UntrustedUrl, releasePageLine("https://user@github.com/maci0/agave"));
    try std.testing.expectError(error.UntrustedUrl, releasePageLine("https://example.com/agave"));
    const page = "https://github.com/maci0/agave/releases/tag/v0.7.0";
    try std.testing.expectEqualStrings(page, try releasePageLine(page));
    try std.testing.expect(trustedGithubUrl("https://api.github.com/repos/maci0/agave/releases/latest"));
    try std.testing.expect(trustedGithubUrl("https://release-assets.githubusercontent.com/agave"));
    try std.testing.expect(!trustedGithubUrl("https://objects.githubusercontent.com.evil.com/x"));
}

test "update: --check and an equal version do not fetch an asset" {
    try std.testing.expect(!fetchesAsset(true, "0.7.0", "v0.8.0"));
    try std.testing.expect(!fetchesAsset(false, "0.7.0", "v0.7.0"));
    try std.testing.expect(fetchesAsset(false, "0.7.0", "v0.8.0"));
}

test "update: checksum line is the published hex, two spaces, and the basename" {
    const sidecar = abc_sha ++ "  " ++ asset_base ++ "\n";
    try std.testing.expect(checksumMatches("abc", sidecar, asset_base));
    try std.testing.expect(!checksumMatches("abd", sidecar, asset_base));
    try std.testing.expect(!checksumMatches("abc", sidecar, "other"));
    const one_space = abc_sha ++ " " ++ asset_base;
    try std.testing.expect(!checksumMatches("abc", one_space, asset_base));
}

test "update: fixture release picks the named asset" {
    const alloc = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const body =
        \\{"tag_name":"v0.7.0","html_url":"https://github.com/maci0/agave/releases/tag/v0.7.0","assets":[
        \\{"name":"agave-v0.7.0-aarch64-macos-none","browser_download_url":"https://example.com/nope"},
        \\{"name":"agave-v0.7.0-x86_64-linux-gnu","browser_download_url":"https://github.com/maci0/agave/releases/download/v0.7.0/agave-v0.7.0-x86_64-linux-gnu"},
        \\{"name":"agave-v0.7.0-x86_64-linux-gnu.sha256","browser_download_url":"https://github.com/maci0/agave/releases/download/v0.7.0/agave-v0.7.0-x86_64-linux-gnu.sha256"}
        \\]}
    ;
    const rel = try parseRelease(arena_state.allocator(), body);
    try std.testing.expectEqualStrings("v0.7.0", rel.tag);
    var name_buf: [80]u8 = undefined;
    const name = try writeAssetName(&name_buf, rel.tag, "x86_64-linux-gnu");
    const url = assetUrl(rel, name) orelse return error.TestUnexpectedResult;
    try std.testing.expect(trustedGithubUrl(url));
    try std.testing.expect(assetUrl(rel, "agave-v0.7.0-no-such") == null);
    try std.testing.expect(!trustedGithubUrl(assetUrl(rel, "agave-v0.7.0-aarch64-macos-none").?));
}

test "update: comparison and install lines use the release wording" {
    var buf: [128]u8 = undefined;
    try std.testing.expectEqualStrings(
        "agave 0.7.0 is current (latest release: v0.7.0)",
        try formatCurrent(&buf, "agave", "0.7.0", "v0.7.0"),
    );
    try std.testing.expectEqualStrings(
        "New release: v0.7.1 (running 0.7.0)",
        try formatNewRelease(&buf, "v0.7.1", "0.7.0"),
    );
    try std.testing.expectEqualStrings(
        "Installed v0.7.1 to /usr/local/bin/agave",
        try formatInstalled(&buf, "v0.7.1", "/usr/local/bin/agave"),
    );
}

test "update: checksum match replaces a copy; mismatch, missing sidecar, and a bad url do not" {
    const alloc = std.testing.allocator;
    var threaded = std.Io.Threaded.init(alloc, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const good_url = "https://github.com/maci0/agave/releases/download/v0.7.0/" ++ asset_base;
    const good_side_url = good_url ++ ".sha256";
    const good_side = abc_sha ++ "  " ++ asset_base ++ "\n";
    const bad_side = "0000000000000000000000000000000000000000000000000000000000000000  " ++ asset_base ++ "\n";

    try tmp.dir.writeFile(io, .{ .sub_path = "agave", .data = "old-binary" });

    const current = decide(.{
        .running = "0.7.0",
        .tag = "v0.7.0",
        .asset_url = good_url,
        .asset = "abc",
        .sidecar_url = good_side_url,
        .sidecar = good_side,
        .basename = asset_base,
    });
    try std.testing.expectEqual(Verdict.current, current);
    try std.testing.expectError(error.Refused, replaceVerified(io, tmp.dir, "agave", current, "abc"));

    const mismatch = decide(.{
        .running = "0.7.0",
        .tag = "v0.7.1",
        .asset_url = good_url,
        .asset = "abc",
        .sidecar_url = good_side_url,
        .sidecar = bad_side,
        .basename = asset_base,
    });
    try std.testing.expectEqual(Verdict.checksum_mismatch, mismatch);
    try std.testing.expectError(error.Refused, replaceVerified(io, tmp.dir, "agave", mismatch, "abc"));

    const missing = decide(.{
        .running = "0.7.0",
        .tag = "v0.7.1",
        .asset_url = good_url,
        .asset = "abc",
        .sidecar_url = null,
        .basename = asset_base,
    });
    try std.testing.expectEqual(Verdict.missing_sidecar, missing);
    try std.testing.expectError(error.Refused, replaceVerified(io, tmp.dir, "agave", missing, "abc"));

    const untrusted = decide(.{
        .running = "0.7.0",
        .tag = "v0.7.1",
        .asset_url = "http://github.com/maci0/agave/releases/download/v0.7.1/" ++ asset_base,
        .asset = "abc",
        .sidecar_url = good_side_url,
        .sidecar = good_side,
        .basename = asset_base,
    });
    try std.testing.expectEqual(Verdict.untrusted_url, untrusted);
    try std.testing.expectError(error.Refused, replaceVerified(io, tmp.dir, "agave", untrusted, "abc"));

    const off_host = decide(.{
        .running = "0.7.0",
        .tag = "v0.7.1",
        .asset_url = "https://example.com/agave",
        .asset = "abc",
        .sidecar_url = good_side_url,
        .sidecar = good_side,
        .basename = asset_base,
    });
    try std.testing.expectEqual(Verdict.untrusted_url, off_host);
    try std.testing.expectError(error.Refused, replaceVerified(io, tmp.dir, "agave", off_host, "abc"));

    {
        const got = try copyOf(io, tmp.dir);
        defer alloc.free(got);
        try std.testing.expectEqualStrings("old-binary", got);
    }

    const replaced = decide(.{
        .running = "0.7.0",
        .tag = "v0.7.1",
        .asset_url = good_url,
        .asset = "abc",
        .sidecar_url = good_side_url,
        .sidecar = good_side,
        .basename = asset_base,
    });
    try std.testing.expectEqual(Verdict.replaced, replaced);
    try replaceVerified(io, tmp.dir, "agave", replaced, "abc");
    const got = try copyOf(io, tmp.dir);
    defer alloc.free(got);
    try std.testing.expectEqualStrings("abc", got);
}

test "update: replaceVerified follows a symlinked destination" {
    const alloc = std.testing.allocator;
    var threaded = std.Io.Threaded.init(alloc, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(io, .{ .sub_path = "real_bin", .data = "old-content" });
    try tmp.dir.symLink(io, "real_bin", "agave", .{});

    try replaceVerified(io, tmp.dir, "agave", .replaced, "new-content");

    // Symlink remains intact pointing to real_bin
    var link_buf: [256]u8 = undefined;
    const n = try tmp.dir.readLink(io, "agave", &link_buf);
    try std.testing.expectEqualStrings("real_bin", link_buf[0..n]);

    // real_bin was updated
    const got = try tmp.dir.readFileAlloc(io, "real_bin", alloc, .limited(64));
    defer alloc.free(got);
    try std.testing.expectEqualStrings("new-content", got);
}

test "update: printUsage writes to stdout with Silencer" {
    const s = try test_stdout.Silencer.init();
    defer s.release();
    printUsage();
}
