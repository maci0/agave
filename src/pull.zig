//! Download GGUF models from HuggingFace Hub.
//!
//! Usage: agave pull <org/repo> [--quant Q4_K_M] [--list]
//!
//! Fetches the HuggingFace model repository listing via the API, selects the
//! best GGUF file based on quantization preference, and downloads it into the
//! standard HuggingFace cache layout with an agave convenience symlink.

const std = @import("std");
const Allocator = std.mem.Allocator;
const display_mod = @import("display.zig");
const version = display_mod.version;

// ── Named constants ──────────────────────────────────────────────────────────

/// Buffer size for stderr formatting output.
const print_buf_size: usize = 4096;
/// Streaming download read chunk size (256 KB).
const download_buf_size: usize = 256 * 1024;
/// Minimum interval between progress bar updates (500 ms).
const progress_interval_ns: u64 = 500 * std.time.ns_per_ms;
/// Maximum number of download retry attempts.
const max_retries: u32 = 3;
/// Base delay between retry attempts (doubles per attempt: 1s, 2s, 4s).
const retry_base_delay_ns: u64 = 1 * std.time.ns_per_s;
const hf_api_base = "https://huggingface.co";
/// Maximum API metadata response size (10 MB — prevents OOM from malicious server).
const max_api_response_size: usize = 10 * 1024 * 1024;
const progress_bar_width: usize = 30;
const bytes_per_mb: f64 = 1024.0 * 1024.0;
const bytes_per_gb: f64 = 1024.0 * 1024.0 * 1024.0;

const Io = std.Io;
const stderr_file = Io.File.stderr();

/// Module-level Io instance, set by run() from caller.
var mod_io: Io = undefined;

/// Nanosecond timestamp via raw C call.
fn nanoTimestamp() i128 {
    var ts: std.posix.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

/// Validate that a filename from the API has no path traversal components
/// and no URL-special characters that could inject query/fragment into
/// download URLs (CWE-74).
/// Rejects embedded '..', '/', '\', null bytes, and URL metacharacters.
fn isSafeFilename(name: []const u8) bool {
    if (name.len == 0 or name.len > 255) return false;
    if (std.mem.indexOf(u8, name, "..") != null) return false;
    for (name) |c| {
        switch (c) {
            '/', '\\', '?', '#', '@' => return false,
            0...31, 127 => return false, // control characters
            else => {},
        }
    }
    return true;
}

/// Validate that a repository name contains only safe characters.
/// Rejects URL-special characters (?, #, @, etc.) that could cause
/// query/fragment injection in HuggingFace API URLs.
fn isValidRepoName(name: []const u8) bool {
    if (name.len == 0) return false;
    // Reject leading/trailing slash and consecutive slashes (empty segments).
    if (name[0] == '/' or name[name.len - 1] == '/') return false;
    var prev_slash = false;
    for (name) |c| {
        switch (c) {
            'a'...'z', 'A'...'Z', '0'...'9', '-', '_', '.' => {
                prev_slash = false;
            },
            '/' => {
                if (prev_slash) return false; // consecutive slashes
                prev_slash = true;
            },
            else => return false,
        }
    }
    return true;
}

/// Validate that a commit SHA contains only hex characters (a-f, 0-9).
fn isValidHexSha(s: []const u8) bool {
    if (s.len == 0 or s.len > 64) return false;
    for (s) |c| {
        if (!std.ascii.isHex(c)) return false;
    }
    return true;
}

/// Quantization preference order: most preferred first.
const quant_preference = [_][]const u8{
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q6_K", "Q8_0",
    "Q3_K_M", "Q2_K",   "f16",    "f32",
};

// ── Types ────────────────────────────────────────────────────────────────────

/// Parsed command-line arguments for the `pull` sub-command.
pub const PullArgs = struct {
    /// Repository identifier in `org/repo` format.
    repo: []const u8,
    /// Optional quantization filter string (e.g. "Q4_K_M").
    quant: ?[]const u8 = null,
    /// If true, list available GGUF files and exit without downloading.
    list_only: bool = false,
    /// Optional HuggingFace API token for private repositories.
    token: ?[]const u8 = null,
};

/// A GGUF file available in a HuggingFace repository.
pub const GgufFile = struct {
    /// Filename within the repository (e.g. "model-Q4_K_M.gguf").
    filename: []const u8,
    /// File size in bytes.
    size: u64,
};

/// Result from listing GGUF files in a repository.
pub const ListResult = struct {
    /// Available GGUF files.
    files: []GgufFile,
    /// Git commit SHA for the repository HEAD.
    commit_sha: []const u8,
    /// Arena allocator that owns all returned memory. Caller must call
    /// `deinit()` to free.
    arena: std.heap.ArenaAllocator,

    pub fn deinit(self: *ListResult) void {
        self.arena.deinit();
    }
};

/// Errors that can occur during the pull operation.
pub const PullError = error{
    /// Invalid command-line argument (missing value, unknown flag, etc.).
    InvalidArgument,
    /// Repository identifier is not in `org/repo` format.
    InvalidRepoFormat,
    /// No GGUF files found in the repository.
    NoGgufFiles,
    /// The requested quantization was not found among available files.
    QuantNotFound,
    /// The repository was not found (HTTP 404).
    RepoNotFound,
    /// Authentication failed (HTTP 401/403).
    AuthenticationFailed,
    /// Failed to parse API response JSON.
    ApiResponseInvalid,
    /// Download failed after all retry attempts.
    DownloadFailed,
    /// HOME environment variable not set.
    HomeNotSet,
    /// HTTP request failed.
    HttpRequestFailed,
    /// Downloaded file failed integrity check (e.g. invalid GGUF magic bytes).
    IntegrityCheckFailed,
};

// ── Stderr helpers ───────────────────────────────────────────────────────────

/// Print a formatted message to stderr.
fn eprint(comptime fmt: []const u8, args: anytype) void {
    var buf: [print_buf_size]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, fmt, args) catch return;
    _ = std.c.write(stderr_file.handle, text.ptr, text.len);
}

/// Write bytes to a file handle via C write.
fn fileWrite(file: Io.File, bytes: []const u8) void {
    _ = std.c.write(file.handle, bytes.ptr, bytes.len);
}

/// Get an environment variable (Zig 0.16 idiom via C getenv).
fn getenv(name: []const u8) ?[]const u8 {
    var buf: [256]u8 = undefined;
    if (name.len >= buf.len) return null;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    const result = std.c.getenv(@ptrCast(buf[0..name.len :0])) orelse return null;
    return std.mem.span(result);
}

// ── Argument parsing ─────────────────────────────────────────────────────────

/// Print usage information to stdout (pipeable: agave pull --help | less).
pub fn printUsage() void {
    const usage =
        \\agave pull — Download GGUF models from HuggingFace Hub
        \\
        \\USAGE:
        \\  agave pull [OPTIONS] <org/repo>
        \\  agave pull [OPTIONS] -- <org/repo>
        \\
        \\ARGUMENTS:
        \\  <org/repo>           Repository in org/repo format (e.g. Qwen/Qwen3.5-27B-GGUF)
        \\
        \\GENERAL:
        \\  -h, --help           Show this help message
        \\  -v, --version        Print version
        \\
        \\OPTIONS:
        \\      --quant <QUANT>  Select quantization (e.g. Q4_K_M, Q8_0)
        \\  -l, --list           List available GGUF files and exit
        \\
        \\ENVIRONMENT:
        \\  HF_TOKEN             HuggingFace API token for private repos
        \\
        \\EXAMPLES:
        \\  agave pull Qwen/Qwen3.5-27B-GGUF
        \\  agave pull Qwen/Qwen3.5-27B-GGUF --quant Q4_K_M
        \\  agave pull Qwen/Qwen3.5-27B-GGUF --list
        \\
        \\SCRIPTING:
        \\  MODEL=$(agave pull org/repo 2>/dev/null)
        \\  agave "$MODEL" "prompt"
        \\
    ;
    fileWrite(Io.File.stdout(), usage);
}

/// Parse command-line arguments for the `pull` sub-command.
///
/// Expects `args_iter` to be positioned after the "pull" token (i.e. the
/// program name and "pull" have already been consumed). Reads HF_TOKEN from
/// the environment.
///
/// Returns `null` if `--help` was requested (caller should exit cleanly).
pub fn parseArgs(args_iter: *std.process.Args.Iterator) PullError!?PullArgs {
    var result = PullArgs{
        .repo = "",
    };
    var have_repo = false;

    result.token = getenv("HF_TOKEN");

    var past_options = false;

    while (args_iter.next()) |arg| {
        // After `--`, treat all remaining arguments as positional.
        if (past_options) {
            if (have_repo) {
                eprint("Error: unexpected argument '{s}'\n", .{arg});
                eprint("Run 'agave pull --help' for more information.\n", .{});
                return PullError.InvalidArgument;
            }
            result.repo = arg;
            have_repo = true;
            continue;
        }

        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "help")) {
            printUsage();
            return null;
        } else if (std.mem.eql(u8, arg, "--version") or std.mem.eql(u8, arg, "-v")) {
            display_mod.printVersion();
            return null;
        } else if (std.mem.eql(u8, arg, "--list") or std.mem.eql(u8, arg, "-l")) {
            result.list_only = true;
        } else if (std.mem.eql(u8, arg, "--quant")) {
            const val = args_iter.next() orelse {
                eprint("Error: --quant requires a value (e.g. Q4_K_M)\n", .{});
                eprint("Run 'agave pull --help' for more information.\n", .{});
                return PullError.InvalidArgument;
            };
            if (val.len > 0 and val[0] == '-') {
                eprint("Error: --quant requires a value, got '{s}' (looks like a flag)\n", .{val});
                eprint("  Example: agave pull org/repo --quant Q4_K_M\n", .{});
                eprint("Run 'agave pull --help' for more information.\n", .{});
                return PullError.InvalidArgument;
            }
            result.quant = val;
        } else if (std.mem.startsWith(u8, arg, "--quant=")) {
            const val = arg["--quant=".len..];
            if (val.len == 0) {
                eprint("Error: --quant requires a value (e.g. Q4_K_M)\n", .{});
                eprint("Run 'agave pull --help' for more information.\n", .{});
                return PullError.InvalidArgument;
            }
            result.quant = val;
        } else if (std.mem.eql(u8, arg, "--")) {
            past_options = true;
        } else if (arg.len > 0 and arg[0] == '-') {
            eprint("Error: unknown option '{s}'\n", .{arg});
            eprint("Run 'agave pull --help' for more information.\n", .{});
            return PullError.InvalidArgument;
        } else {
            // Positional argument: repo name.
            if (have_repo) {
                eprint("Error: unexpected argument '{s}'\n", .{arg});
                eprint("Run 'agave pull --help' for more information.\n", .{});
                return PullError.InvalidArgument;
            }
            result.repo = arg;
            have_repo = true;
        }
    }

    if (!have_repo) {
        eprint("Error: repository name required (e.g. Qwen/Qwen3.5-27B-GGUF)\n", .{});
        eprint("Run 'agave pull --help' for more information.\n", .{});
        return PullError.InvalidArgument;
    }

    // Validate org/repo format: exactly one slash, no path traversal.
    const slash_pos = std.mem.indexOfScalar(u8, result.repo, '/') orelse {
        eprint("Error: repository must be in 'org/repo' format, got '{s}'\n", .{result.repo});
        eprint("  Example: agave pull Qwen/Qwen3.5-27B-GGUF\n", .{});
        return PullError.InvalidRepoFormat;
    };
    if (std.mem.indexOfScalarPos(u8, result.repo, slash_pos + 1, '/') != null or
        slash_pos == 0 or slash_pos == result.repo.len - 1 or
        std.mem.indexOf(u8, result.repo, "..") != null or
        !isValidRepoName(result.repo))
    {
        eprint("Error: repository must be in 'org/repo' format, got '{s}'\n", .{result.repo});
        eprint("  Example: agave pull Qwen/Qwen3.5-27B-GGUF\n", .{});
        return PullError.InvalidRepoFormat;
    }

    return result;
}

// ── HuggingFace API client ───────────────────────────────────────────────────

/// Fetch the list of GGUF files available in a HuggingFace repository.
///
/// Makes a GET request to the HuggingFace API and parses the JSON response
/// to extract filenames, sizes, and the commit SHA. All returned memory is
/// owned by the `ListResult.arena`; call `ListResult.deinit()` when done.
pub fn listGgufFiles(allocator: Allocator, repo: []const u8, token: ?[]const u8) (PullError || Allocator.Error)!ListResult {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_alloc = arena.allocator();

    // Build API URL: https://huggingface.co/api/models/{repo}
    const url = std.fmt.allocPrint(arena_alloc, "{s}/api/models/{s}", .{ hf_api_base, repo }) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
    };

    // Perform HTTP GET request.
    const body = httpGet(allocator, url, token) catch |err| {
        switch (err) {
            PullError.RepoNotFound => {
                eprint("Error: repository '{s}' not found\n", .{repo});
                return PullError.RepoNotFound;
            },
            PullError.AuthenticationFailed => {
                eprint("Error: repository '{s}' not found or is private\n", .{repo});
                if (token == null) {
                    eprint("  Check the name, or set HF_TOKEN for private repos.\n", .{});
                } else {
                    eprint("  Check that HF_TOKEN is valid and has access.\n", .{});
                }
                return PullError.AuthenticationFailed;
            },
            else => {
                eprint("Error: failed to fetch repository info for '{s}'\n", .{repo});
                return PullError.HttpRequestFailed;
            },
        }
    };
    defer allocator.free(body);

    // Parse JSON response.
    const parsed = std.json.parseFromSlice(std.json.Value, arena_alloc, body, .{}) catch {
        const preview_len = @min(body.len, 200);
        eprint("Error: failed to parse API response (first {d} bytes: {s})\n", .{ preview_len, body[0..preview_len] });
        return PullError.ApiResponseInvalid;
    };
    const root = parsed.value;

    // Extract commit SHA from "sha" field.
    const sha_val = if (root == .object) root.object.get("sha") else null;
    const commit_sha: []const u8 = if (sha_val) |sv| switch (sv) {
        .string => |s| if (isValidHexSha(s)) s else "unknown",
        else => "unknown",
    } else "unknown";

    // Extract siblings array and filter for .gguf files.
    const siblings_val = if (root == .object) root.object.get("siblings") else null;
    const siblings = if (siblings_val) |sv| switch (sv) {
        .array => sv.array.items,
        else => &[_]std.json.Value{},
    } else &[_]std.json.Value{};

    // First pass: count GGUF files.
    var gguf_count: usize = 0;
    for (siblings) |sibling| {
        if (sibling != .object) continue;
        const rfilename = sibling.object.get("rfilename") orelse continue;
        if (rfilename != .string) continue;
        if (std.mem.endsWith(u8, rfilename.string, ".gguf") and isSafeFilename(rfilename.string)) {
            gguf_count += 1;
        }
    }

    if (gguf_count == 0) {
        eprint("Error: no GGUF files found in '{s}'\n", .{repo});
        return PullError.NoGgufFiles;
    }

    // Second pass: collect GGUF file info.
    var files = try arena_alloc.alloc(GgufFile, gguf_count);
    var idx: usize = 0;
    for (siblings) |sibling| {
        if (sibling != .object) continue;
        const rfilename = sibling.object.get("rfilename") orelse continue;
        if (rfilename != .string) continue;
        if (!std.mem.endsWith(u8, rfilename.string, ".gguf") or !isSafeFilename(rfilename.string)) continue;

        // Extract size from the "size" field if available.
        const size_val = sibling.object.get("size");
        const size: u64 = if (size_val) |sv| switch (sv) {
            .integer => |i| if (i >= 0) @intCast(i) else 0,
            else => 0,
        } else 0;

        files[idx] = .{
            .filename = rfilename.string,
            .size = size,
        };
        idx += 1;
    }

    return ListResult{
        .files = files[0..idx],
        .commit_sha = commit_sha,
        .arena = arena,
    };
}

/// Select the best GGUF file from the list based on quantization preference.
///
/// If `quant` is provided, returns the first file whose name contains that
/// string (case-insensitive). Otherwise, ranks files by the built-in quant
/// preference order and returns the best match.
pub fn selectFile(files: []const GgufFile, quant: ?[]const u8) PullError!GgufFile {
    if (files.len == 0) return PullError.NoGgufFiles;

    if (quant) |q| {
        // User-specified quantization: find first file containing the string.
        for (files) |f| {
            if (std.ascii.indexOfIgnoreCase(f.filename, q) != null) {
                return f;
            }
        }
        eprint("Error: no GGUF file found matching quantization '{s}'\n", .{q});
        eprint("Available files:\n", .{});
        for (files) |f| {
            if (f.size > 0) {
                const size_gb = @as(f64, @floatFromInt(f.size)) / bytes_per_gb;
                eprint("  {s}  ({d:.1} GB)\n", .{ f.filename, size_gb });
            } else {
                eprint("  {s}\n", .{f.filename});
            }
        }
        return PullError.QuantNotFound;
    }

    // Auto-select by preference order.
    for (&quant_preference) |pref| {
        for (files) |f| {
            if (std.ascii.indexOfIgnoreCase(f.filename, pref) != null) {
                return f;
            }
        }
    }

    // Fallback: first file.
    return files[0];
}

/// Print a formatted list of available GGUF files with sizes to stdout (pipeable).
pub fn printFileList(files: []const GgufFile) void {
    const stdout = Io.File.stdout();
    for (files) |f| {
        const size_gb = @as(f64, @floatFromInt(f.size)) / bytes_per_gb;
        var buf: [print_buf_size]u8 = undefined;
        if (f.size > 0) {
            fileWrite(stdout, std.fmt.bufPrint(&buf, "{s}  ({d:.1} GB)\n", .{ f.filename, size_gb }) catch continue);
        } else {
            fileWrite(stdout, std.fmt.bufPrint(&buf, "{s}\n", .{f.filename}) catch continue);
        }
    }
}

// ── HTTP helpers ─────────────────────────────────────────────────────────────

/// Perform an HTTP GET request and return the response body as an owned slice.
///
/// Handles HTTP status codes: 404 -> RepoNotFound, 401/403 -> AuthenticationFailed.
/// The caller owns the returned slice and must free it with `allocator`.
fn httpGet(allocator: Allocator, url: []const u8, token: ?[]const u8) (PullError || Allocator.Error)![]u8 {
    var client: std.http.Client = .{ .allocator = allocator, .io = mod_io };
    defer client.deinit();

    // Build extra headers for auth token. Formatted header buffer is zeroed
    // on exit to reduce credential exposure in stack memory.
    var auth_buf: [256]u8 = undefined;
    defer @memset(&auth_buf, 0);
    const auth_value = if (token) |t|
        std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{t}) catch return PullError.HttpRequestFailed
    else
        null;

    // Use privileged_headers for auth — stripped on redirect to prevent token leaking to CDN.
    var priv_headers_buf: [1]std.http.Header = undefined;
    const priv_headers: []const std.http.Header = if (auth_value) |av| blk: {
        priv_headers_buf[0] = .{ .name = "Authorization", .value = av };
        break :blk priv_headers_buf[0..1];
    } else &.{};

    // Use allocating writer for response body.
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();

    const result = client.fetch(.{
        .location = .{ .url = url },
        .privileged_headers = priv_headers,
        .response_writer = &aw.writer,
    }) catch |err| {
        eprint("Error: HTTP request failed: {}\n", .{err});
        return PullError.HttpRequestFailed;
    };

    switch (result.status) {
        .ok => {},
        .not_found => return PullError.RepoNotFound,
        .unauthorized, .forbidden => return PullError.AuthenticationFailed,
        else => {
            eprint("Error: HTTP {d}\n", .{@intFromEnum(result.status)});
            return PullError.HttpRequestFailed;
        },
    }

    const body = aw.toOwnedSlice() catch return error.OutOfMemory;
    if (body.len > max_api_response_size) {
        allocator.free(body);
        return PullError.HttpRequestFailed;
    }
    return body;
}

// ── Cache layout & file system helpers ───────────────────────────────────────

/// Compute the HuggingFace cache directory path for a repository.
///
/// Respects HF ecosystem env vars in standard precedence order:
///   1. HF_HOME → $HF_HOME/hub/models--{org}--{repo}
///   2. XDG_CACHE_HOME → $XDG_CACHE_HOME/huggingface/hub/models--{org}--{repo}
///   3. HOME → $HOME/.cache/huggingface/hub/models--{org}--{repo}
///
/// Slashes in the repo name are replaced with `--` per HF convention.
pub fn hfCacheDir(allocator: Allocator, repo: []const u8) (PullError || Allocator.Error)![]u8 {
    // Replace '/' with '--' in repo name.
    const repo_escaped = replaceSlashes(allocator, repo) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
    };
    defer allocator.free(repo_escaped);

    // HF_HOME takes highest precedence (e.g. /data/huggingface)
    if (getenv("HF_HOME")) |hf_home| {
        return std.fmt.allocPrint(allocator, "{s}/hub/models--{s}", .{ hf_home, repo_escaped }) catch
            return error.OutOfMemory;
    }

    // XDG_CACHE_HOME overrides default cache location
    if (getenv("XDG_CACHE_HOME")) |xdg| {
        return std.fmt.allocPrint(allocator, "{s}/huggingface/hub/models--{s}", .{ xdg, repo_escaped }) catch
            return error.OutOfMemory;
    }

    // Default: $HOME/.cache/huggingface/hub/
    const home = getenv("HOME") orelse {
        eprint("Error: HOME environment variable not set\n", .{});
        return PullError.HomeNotSet;
    };

    return std.fmt.allocPrint(allocator, "{s}/.cache/huggingface/hub/models--{s}", .{ home, repo_escaped }) catch
        return error.OutOfMemory;
}

/// Replace all occurrences of '/' with '--' in a string.
fn replaceSlashes(allocator: Allocator, input: []const u8) Allocator.Error![]u8 {
    // Count slashes to determine output length.
    var slash_count: usize = 0;
    for (input) |c| {
        if (c == '/') slash_count += 1;
    }

    const out_len = std.math.add(usize, input.len, slash_count) catch return error.OutOfMemory; // each '/' becomes '--' (1 extra char)
    var result = try allocator.alloc(u8, out_len);
    var out_idx: usize = 0;
    for (input) |c| {
        if (c == '/') {
            result[out_idx] = '-';
            result[out_idx + 1] = '-';
            out_idx += 2;
        } else {
            result[out_idx] = c;
            out_idx += 1;
        }
    }
    return result;
}

/// Create a directory path, ignoring if it already exists.
fn ensureDir(path: []const u8) PullError!void {
    Io.Dir.cwd().createDirPath(mod_io, path) catch |err| {
        eprint("Error: could not create directory '{s}': {}\n", .{ path, err });
        return PullError.DownloadFailed;
    };
}

/// Create a symbolic link using the C library (std.posix.symlink removed in Zig 0.16).
fn createSymlink(allocator: Allocator, target: []const u8, link_path: []const u8) !void {
    const target_z = try allocator.dupeZ(u8, target);
    defer allocator.free(target_z);
    const link_z = try allocator.dupeZ(u8, link_path);
    defer allocator.free(link_z);
    const ret = std.c.symlink(target_z, link_z);
    const e = std.c.errno(ret);
    if (e != .SUCCESS) {
        return std.posix.unexpectedErrno(e);
    }
}

/// Create the agave convenience symlink.
///
/// Creates `$HOME/.cache/agave/models/{org}/{repo}` pointing to the
/// snapshot directory containing the downloaded model.
fn createAgaveSymlink(allocator: Allocator, repo: []const u8, snapshot_dir: []const u8) void {
    const home = getenv("HOME") orelse return;

    // Split repo into org and name.
    const slash_idx = std.mem.indexOfScalar(u8, repo, '/') orelse return;
    const org = repo[0..slash_idx];
    const name = repo[slash_idx + 1 ..];

    // Create $HOME/.cache/agave/models/{org}/
    const agave_dir = std.fmt.allocPrint(allocator, "{s}/.cache/agave/models/{s}", .{ home, org }) catch return;
    defer allocator.free(agave_dir);
    ensureDir(agave_dir) catch return;

    // Create symlink: $HOME/.cache/agave/models/{org}/{repo_name} -> snapshot_dir
    const link_path = std.fmt.allocPrint(allocator, "{s}/{s}", .{ agave_dir, name }) catch return;
    defer allocator.free(link_path);

    // Atomic symlink replacement: create at temp path with random suffix to
    // prevent TOCTOU races (CWE-367), then rename over target.
    var rand_buf: [8]u8 = undefined;
    mod_io.random(&rand_buf);
    const tmp_path = std.fmt.allocPrint(allocator, "{s}.tmp.{x}", .{
        link_path, std.mem.readInt(u64, &rand_buf, .little),
    }) catch return;
    defer allocator.free(tmp_path);

    createSymlink(allocator, snapshot_dir, tmp_path) catch |err| {
        eprint("Warning: could not create agave symlink: {}\n", .{err});
        return;
    };
    Io.Dir.rename(Io.Dir.cwd(), tmp_path, Io.Dir.cwd(), link_path, mod_io) catch |err| {
        eprint("Warning: could not finalize agave symlink: {}\n", .{err});
        Io.Dir.cwd().deleteFile(mod_io,tmp_path) catch {};
    };
}

// ── Download with progress ───────────────────────────────────────────────────

/// Build a progress bar string for a given percentage.
///
/// Returns a slice like `[===============>               ]` representing
/// the current download progress.
fn progressBar(buf: *[progress_bar_width + 2]u8, pct: u8) []const u8 {
    const clamped = @min(pct, 100);
    const filled: usize = @intCast((@as(u32, clamped) * progress_bar_width) / 100);

    buf[0] = '[';
    var i: usize = 0;
    while (i < progress_bar_width) : (i += 1) {
        if (i < filled) {
            buf[i + 1] = '=';
        } else if (i == filled and clamped < 100) {
            buf[i + 1] = '>';
        } else {
            buf[i + 1] = ' ';
        }
    }
    buf[progress_bar_width + 1] = ']';
    return buf[0 .. progress_bar_width + 2];
}

/// Download a single file from a HuggingFace repository with resume support
/// and a progress bar.
///
/// The file is downloaded to `blob_path`. If the file already partially
/// exists, the download resumes from where it left off using HTTP Range
/// headers. Progress is reported to stderr every 500ms.
fn downloadFile(
    allocator: Allocator,
    repo: []const u8,
    filename: []const u8,
    blob_path: []const u8,
    token: ?[]const u8,
) PullError!void {
    // Build download URL.
    const url = std.fmt.allocPrint(allocator, "{s}/{s}/resolve/main/{s}", .{ hf_api_base, repo, filename }) catch
        return PullError.DownloadFailed;
    defer allocator.free(url);

    // Detect TTY once for all attempts — progress bars use \r which
    // produces garbled output when stderr is redirected to a file.
    const is_tty = std.c.isatty(stderr_file.handle) != 0;

    var attempt: u32 = 0;
    while (attempt < max_retries) : (attempt += 1) {
        if (attempt > 0) {
            if (is_tty) {
                eprint("\rRetrying download (attempt {d}/{d})...\n", .{ attempt + 1, max_retries });
            } else {
                eprint("Retrying download (attempt {d}/{d})...\n", .{ attempt + 1, max_retries });
            }
            mod_io.sleep(Io.Duration.fromNanoseconds(@intCast(retry_base_delay_ns << @intCast(attempt))), .awake) catch {};
        }

        downloadFileOnce(allocator, url, blob_path, token, is_tty) catch |err| {
            // Don't retry non-transient errors
            switch (err) {
                PullError.RepoNotFound, PullError.AuthenticationFailed => return err,
                else => {},
            }
            if (attempt + 1 < max_retries) {
                eprint("  Attempt {d} failed: {}\n", .{ attempt + 1, err });
                continue;
            }
            eprint("\nError: download failed after {d} attempts: {}\n", .{ max_retries, err });
            return PullError.DownloadFailed;
        };
        return; // Success.
    }
    return PullError.DownloadFailed;
}

/// Single download attempt (used by `downloadFile` retry loop).
fn downloadFileOnce(
    allocator: Allocator,
    url: []const u8,
    blob_path: []const u8,
    token: ?[]const u8,
    is_tty: bool,
) PullError!void {
    // Check for existing partial download.
    var existing_size: u64 = 0;
    if (Io.Dir.cwd().statFile(mod_io, blob_path, .{})) |stat| {
        existing_size = stat.size;
    } else |_| {}

    var client: std.http.Client = .{ .allocator = allocator, .io = mod_io };
    defer client.deinit();

    const uri = std.Uri.parse(url) catch return PullError.DownloadFailed;

    // Build headers. Buffer zeroed on exit to avoid leaving credentials in stack memory.
    var auth_buf: [256]u8 = undefined;
    defer @memset(&auth_buf, 0);
    const auth_value: ?[]const u8 = if (token) |t|
        std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{t}) catch return PullError.DownloadFailed
    else
        null;

    var range_buf: [64]u8 = undefined;
    const range_value: ?[]const u8 = if (existing_size > 0)
        std.fmt.bufPrint(&range_buf, "bytes={d}-", .{existing_size}) catch return PullError.DownloadFailed
    else
        null;

    // Auth in privileged_headers (stripped on redirect to prevent token leaking to CDN).
    var priv_storage: [1]std.http.Header = undefined;
    var priv_idx: usize = 0;
    if (auth_value) |av| {
        priv_storage[priv_idx] = .{ .name = "Authorization", .value = av };
        priv_idx += 1;
    }
    // Range stays in extra_headers (needed on CDN redirect).
    var extra_storage: [1]std.http.Header = undefined;
    var ext_idx: usize = 0;
    if (range_value) |rv| {
        extra_storage[ext_idx] = .{ .name = "Range", .value = rv };
        ext_idx += 1;
    }

    var req = client.request(.GET, uri, .{
        .extra_headers = extra_storage[0..ext_idx],
        .privileged_headers = priv_storage[0..priv_idx],
        .keep_alive = false,
    }) catch |err| {
        eprint("Error: download connection failed: {}\n", .{err});
        return PullError.HttpRequestFailed;
    };
    defer req.deinit();

    req.sendBodiless() catch |err| {
        eprint("Error: download request failed: {}\n", .{err});
        return PullError.HttpRequestFailed;
    };

    var redirect_buf: [8192]u8 = undefined;
    var response = req.receiveHead(&redirect_buf) catch |err| {
        eprint("Error: download response failed: {}\n", .{err});
        return PullError.HttpRequestFailed;
    };

    const status = response.head.status;
    switch (status) {
        .ok, .partial_content => {},
        .not_found => return PullError.RepoNotFound,
        .unauthorized, .forbidden => return PullError.AuthenticationFailed,
        .range_not_satisfiable => {
            // File already fully downloaded.
            return;
        },
        else => {
            eprint("Error: HTTP {d}\n", .{@intFromEnum(status)});
            return PullError.HttpRequestFailed;
        },
    }

    // Determine total size (checked arithmetic prevents overflow from crafted Content-Length).
    const content_length = response.head.content_length orelse 0;
    const total_size: u64 = if (status == .partial_content)
        std.math.add(u64, existing_size, content_length) catch return PullError.DownloadFailed
    else
        content_length;

    // If server returned 200 (not 206), we're starting from scratch.
    const start_offset: u64 = if (status == .partial_content) existing_size else 0;

    // Open file for writing with O_NOFOLLOW to atomically reject symlinks (CWE-59).
    // This prevents symlink-following attacks where a compromised cache directory
    // redirects writes elsewhere. Unlike the previous readLink-then-open pattern,
    // the NOFOLLOW flag is checked atomically by the kernel during open, eliminating
    // the TOCTOU race window between a separate symlink check and the open call.
    var os_flags: std.posix.O = .{ .ACCMODE = .WRONLY };
    os_flags.NOFOLLOW = true;
    if (@hasField(std.posix.O, "CLOEXEC")) os_flags.CLOEXEC = true;
    if (start_offset == 0) {
        os_flags.CREAT = true;
        os_flags.TRUNC = true;
    }
    const fd = std.posix.openat(Io.Dir.cwd().handle, blob_path, os_flags, 0o644) catch |err| {
        if (err == error.SymLinkLoop) {
            eprint("Error: blob path is a symlink — refusing to write (possible symlink attack)\n", .{});
        }
        return PullError.DownloadFailed;
    };
    const file: Io.File = .{ .handle = fd, .flags = .{ .nonblocking = false } };
    defer _ = std.c.close(file.handle);

    // Seek to end for append.
    if (start_offset > 0) {
        _ = std.c.lseek(file.handle, @intCast(start_offset), std.c.SEEK.SET);
    }

    // Get a reader from the response body.
    var transfer_buf: [download_buf_size]u8 = undefined;
    var body_reader = response.reader(&transfer_buf);

    // Download with progress reporting.
    var downloaded: u64 = start_offset;
    var last_progress_time: i128 = nanoTimestamp();
    var last_progress_bytes: u64 = downloaded;

    if (start_offset > 0) {
        eprint("Resuming download from {d:.1} MB\n", .{@as(f64, @floatFromInt(start_offset)) / bytes_per_mb});
    }

    // Note: std.http.Client does not support read timeouts. A stalled TCP
    // connection will block readSliceShort indefinitely. Users can Ctrl+C
    // and re-run to resume. A proper fix would require async I/O or a
    // separate watchdog thread, which is not worth the complexity here.
    var read_buf: [download_buf_size]u8 = undefined;
    while (true) {
        const bytes_read = body_reader.readSliceShort(&read_buf) catch {
            eprint("\nError: network read failed during download\n", .{});
            return PullError.DownloadFailed;
        };
        if (bytes_read == 0) break;

        file.writePositionalAll(mod_io, read_buf[0..bytes_read], downloaded) catch |err| {
            eprint("\nError: failed to write to disk: {}\n", .{err});
            if (err == error.NoSpaceLeft or err == error.DiskQuota)
                eprint("  Disk is full. Free space and re-run to resume.\n", .{});
            return PullError.DownloadFailed;
        };
        downloaded += bytes_read;

        // Update progress bar periodically (TTY only — \r produces garbled
        // output when stderr is redirected to a file or pipe).
        if (is_tty) {
            const now = nanoTimestamp();
            const elapsed_ns: u64 = @intCast(@max(now - last_progress_time, 0));
            if (elapsed_ns >= progress_interval_ns or downloaded == total_size) {
                const elapsed_secs = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s));
                const bytes_since = downloaded - last_progress_bytes;
                const speed_mbps = if (elapsed_secs > 0.0)
                    @as(f64, @floatFromInt(bytes_since)) / bytes_per_mb / elapsed_secs
                else
                    0.0;

                const pct: u8 = if (total_size > 0)
                    @intCast(@min((@as(u64, 100) * downloaded) / total_size, 100))
                else
                    0;

                var bar_buf: [progress_bar_width + 2]u8 = undefined;
                const bar = progressBar(&bar_buf, pct);

                // Estimate time remaining.
                const remaining_bytes = if (total_size > downloaded) total_size - downloaded else 0;
                const eta_secs: u64 = if (speed_mbps > 0.0)
                    @intFromFloat(@as(f64, @floatFromInt(remaining_bytes)) / (speed_mbps * bytes_per_mb))
                else
                    0;

                const downloaded_mb = @as(f64, @floatFromInt(downloaded)) / bytes_per_mb;
                const total_mb = @as(f64, @floatFromInt(total_size)) / bytes_per_mb;

                if (eta_secs > 0) {
                    eprint("\r{s} {d:>3}%  {d:.1}/{d:.1} MB  {d:.1} MB/s  ETA {d}s  ", .{
                        bar, pct, downloaded_mb, total_mb, speed_mbps, eta_secs,
                    });
                } else {
                    eprint("\r{s} {d:>3}%  {d:.1}/{d:.1} MB  {d:.1} MB/s  ", .{
                        bar, pct, downloaded_mb, total_mb, speed_mbps,
                    });
                }

                last_progress_time = now;
                last_progress_bytes = downloaded;
            }
        }
    }

    if (is_tty) eprint("\n", .{}); // Newline after progress bar.

    // Verify downloaded size matches expected size (catches silent truncation).
    if (total_size > 0 and downloaded != total_size) {
        eprint("Error: downloaded {d} bytes but expected {d} — file may be truncated\n", .{ downloaded, total_size });
        return PullError.DownloadFailed;
    }
}

// ── Orchestrator ─────────────────────────────────────────────────────────────

/// Execute the full model pull workflow.
///
///  1. List GGUF files in the repository
///  2. If --list, print file list to stdout and return
///  3. Select the best file matching --quant filter
///  4. Build HuggingFace cache directory structure
///  5. Check if already downloaded (skip download if complete)
///  6. Download if needed
///  7. Verify GGUF magic bytes (integrity check)
///  8. Create snapshot symlink (relative path)
///  9. Write refs/main with commit SHA
/// 10. Create agave convenience symlink
/// 11. Print the final model path
pub fn pullModel(allocator: Allocator, args: PullArgs) (PullError || Allocator.Error)!void {
    // Step 1: List available files.
    eprint("Fetching model info for '{s}'...\n", .{args.repo});
    var list_result = try listGgufFiles(allocator, args.repo, args.token);
    defer list_result.deinit();

    // Step 2: If --list, print file list to stdout and return.
    if (args.list_only) {
        eprint("Available GGUF files in '{s}':\n", .{args.repo});
        printFileList(list_result.files);
        return;
    }

    // Step 3: Select file.
    const selected = try selectFile(list_result.files, args.quant);
    const size_gb = @as(f64, @floatFromInt(selected.size)) / bytes_per_gb;
    eprint("Selected: {s} ({d:.1} GB)\n", .{ selected.filename, size_gb });

    // Step 4: Build cache paths (arena-allocated — freed together at function exit).
    var path_arena = std.heap.ArenaAllocator.init(allocator);
    defer path_arena.deinit();
    const pa = path_arena.allocator();

    const cache_dir = try hfCacheDir(pa, args.repo);
    const blobs_dir = std.fmt.allocPrint(pa, "{s}/blobs", .{cache_dir}) catch return error.OutOfMemory;
    const snapshots_dir = std.fmt.allocPrint(pa, "{s}/snapshots/{s}", .{ cache_dir, list_result.commit_sha }) catch return error.OutOfMemory;
    const refs_dir = std.fmt.allocPrint(pa, "{s}/refs", .{cache_dir}) catch return error.OutOfMemory;

    try ensureDir(blobs_dir);
    try ensureDir(snapshots_dir);
    try ensureDir(refs_dir);

    // Step 5: Check if already downloaded.
    const blob_path = std.fmt.allocPrint(pa, "{s}/{s}", .{ blobs_dir, selected.filename }) catch return error.OutOfMemory;

    var already_complete = false;
    if (Io.Dir.cwd().statFile(mod_io, blob_path, .{})) |stat| {
        if (selected.size > 0 and stat.size == selected.size) {
            eprint("File already downloaded: {s}\n", .{blob_path});
            already_complete = true;
        } else if (selected.size == 0 and stat.size > 0) {
            // Size unknown from API but file exists — assume complete.
            eprint("File already exists: {s}\n", .{blob_path});
            already_complete = true;
        }
    } else |_| {}

    // Step 6: Download if needed.
    if (!already_complete) {
        try downloadFile(allocator, args.repo, selected.filename, blob_path, args.token);
        eprint("Download complete.\n", .{});
    }

    // Step 6b: Verify GGUF magic bytes (catches truncation and corruption).
    if (std.mem.endsWith(u8, selected.filename, ".gguf")) {
        if (Io.Dir.cwd().openFile(mod_io, blob_path, .{})) |f| {
            defer f.close(mod_io);
            var magic: [4]u8 = undefined;
            const n = f.readPositionalAll(mod_io, &magic, 0) catch 0;
            if (n < 4 or !std.mem.eql(u8, &magic, "GGUF")) {
                eprint("Error: downloaded file does not have valid GGUF header — corrupt or truncated\n", .{});
                eprint("  Delete and re-download: rm {s}\n", .{blob_path});
                return PullError.IntegrityCheckFailed;
            }
        } else |_| {}
    }

    // Step 7: Create snapshot symlink (relative path).
    const snapshot_link = std.fmt.allocPrint(pa, "{s}/{s}", .{ snapshots_dir, selected.filename }) catch
        return error.OutOfMemory;

    const relative_blob = std.fmt.allocPrint(pa, "../../blobs/{s}", .{selected.filename}) catch
        return error.OutOfMemory;

    // Atomic symlink replacement: create at temp path with random suffix to
    // prevent TOCTOU races (CWE-367), then rename over target.
    var snap_rand_buf: [8]u8 = undefined;
    mod_io.random(&snap_rand_buf);
    const tmp_snapshot_link = std.fmt.allocPrint(pa, "{s}.tmp.{x}", .{
        snapshot_link, std.mem.readInt(u64, &snap_rand_buf, .little),
    }) catch return error.OutOfMemory;
    createSymlink(pa, relative_blob, tmp_snapshot_link) catch |err| {
        eprint("Warning: could not create snapshot symlink: {}\n", .{err});
        return;
    };
    Io.Dir.rename(Io.Dir.cwd(), tmp_snapshot_link, Io.Dir.cwd(), snapshot_link, mod_io) catch |err| {
        eprint("Warning: could not finalize snapshot symlink: {}\n", .{err});
        Io.Dir.cwd().deleteFile(mod_io,tmp_snapshot_link) catch {};
    };

    // Step 8: Write refs/main with commit SHA.
    const refs_main = std.fmt.allocPrint(pa, "{s}/main", .{refs_dir}) catch
        return error.OutOfMemory;

    if (Io.Dir.cwd().createFile(mod_io, refs_main, .{})) |f| {
        defer f.close(mod_io);
        f.writePositionalAll(mod_io, list_result.commit_sha, 0) catch |err| {
            eprint("Warning: could not write refs/main: {}\n", .{err});
        };
    } else |_| {}

    // Step 9: Create agave convenience symlink.
    createAgaveSymlink(allocator, args.repo, snapshots_dir);

    // Step 10: Print final model path.
    // Write path to stdout (scriptable: MODEL=$(agave pull repo 2>/dev/null))
    fileWrite(Io.File.stdout(), snapshot_link);
    fileWrite(Io.File.stdout(), "\n");

    // Human-friendly summary on stderr
    eprint("\nModel ready at:\n  {s}\n", .{snapshot_link});
    eprint("Run:\n  agave {s} \"your prompt\"\n", .{snapshot_link});
}

// ── Entry point ──────────────────────────────────────────────────────────────

/// Main entry point for the `agave pull` sub-command.
///
/// Parses arguments, runs the pull workflow, and reports errors to stderr.
pub fn run(allocator: Allocator, process_args: std.process.Args, io: Io) u8 {
    mod_io = io;
    var args_iter = process_args.iterate();
    _ = args_iter.skip(); // Skip program name (argv[0]).
    _ = args_iter.skip(); // Skip "pull" subcommand (already verified by main.zig).

    const maybe_args = parseArgs(&args_iter) catch {
        return 1;
    };

    const args = maybe_args orelse return 0; // --help was shown.

    pullModel(allocator, args) catch |err| {
        switch (err) {
            // These errors are already reported with context by inner functions
            PullError.NoGgufFiles,
            PullError.QuantNotFound,
            PullError.RepoNotFound,
            PullError.AuthenticationFailed,
            PullError.DownloadFailed,
            PullError.HomeNotSet,
            PullError.HttpRequestFailed,
            PullError.ApiResponseInvalid,
            PullError.IntegrityCheckFailed,
            => {},
            error.OutOfMemory => eprint("Error: out of memory\n", .{}),
            else => eprint("Error: {}\n", .{err}),
        }
        return 1;
    };

    return 0;
}

// ── Tests ────────────────────────────────────────────────────────────────────

test "replaceSlashes basic" {
    const allocator = std.testing.allocator;
    const result = try replaceSlashes(allocator, "org/repo");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("org--repo", result);
}

test "replaceSlashes no slashes" {
    const allocator = std.testing.allocator;
    const result = try replaceSlashes(allocator, "noslash");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("noslash", result);
}

test "replaceSlashes multiple slashes" {
    const allocator = std.testing.allocator;
    const result = try replaceSlashes(allocator, "a/b/c");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("a--b--c", result);
}

test "progressBar 0 percent" {
    var buf: [progress_bar_width + 2]u8 = undefined;
    const bar = progressBar(&buf, 0);
    try std.testing.expect(bar[0] == '[');
    try std.testing.expect(bar[bar.len - 1] == ']');
    try std.testing.expect(bar[1] == '>');
    // Remaining chars must be spaces (no fill)
    for (bar[2 .. bar.len - 1]) |c| {
        try std.testing.expect(c == ' ');
    }
}

test "progressBar 100 percent" {
    var buf: [progress_bar_width + 2]u8 = undefined;
    const bar = progressBar(&buf, 100);
    try std.testing.expect(bar[0] == '[');
    try std.testing.expect(bar[bar.len - 1] == ']');
    // All should be '='
    for (bar[1 .. bar.len - 1]) |c| {
        try std.testing.expect(c == '=');
    }
}

test "progressBar 50 percent" {
    var buf: [progress_bar_width + 2]u8 = undefined;
    const bar = progressBar(&buf, 50);
    try std.testing.expect(bar[0] == '[');
    try std.testing.expect(bar[bar.len - 1] == ']');
    // Halfway should have '>' at position 15 (0-indexed within bar content)
    const mid = progress_bar_width / 2;
    try std.testing.expect(bar[mid + 1] == '>');
    // Chars before cursor must be '=', chars after must be ' '
    for (bar[1..mid + 1]) |c| {
        try std.testing.expect(c == '=');
    }
    for (bar[mid + 2 .. bar.len - 1]) |c| {
        try std.testing.expect(c == ' ');
    }
}

test "selectFile with explicit quant" {
    const files = [_]GgufFile{
        .{ .filename = "model-Q8_0.gguf", .size = 1000 },
        .{ .filename = "model-Q4_K_M.gguf", .size = 500 },
        .{ .filename = "model-f16.gguf", .size = 2000 },
    };
    const result = try selectFile(&files, "Q4_K_M");
    try std.testing.expectEqualStrings("model-Q4_K_M.gguf", result.filename);
}

test "selectFile auto preference" {
    const files = [_]GgufFile{
        .{ .filename = "model-Q8_0.gguf", .size = 1000 },
        .{ .filename = "model-Q4_K_M.gguf", .size = 500 },
        .{ .filename = "model-f16.gguf", .size = 2000 },
    };
    const result = try selectFile(&files, null);
    // Q4_K_M is highest preference.
    try std.testing.expectEqualStrings("model-Q4_K_M.gguf", result.filename);
}

test "selectFile fallback to first" {
    const files = [_]GgufFile{
        .{ .filename = "model-weird.gguf", .size = 1000 },
    };
    const result = try selectFile(&files, null);
    try std.testing.expectEqualStrings("model-weird.gguf", result.filename);
}

test "selectFile quant not found" {
    const files = [_]GgufFile{
        .{ .filename = "model-Q8_0.gguf", .size = 1000 },
    };
    const result = selectFile(&files, "NONEXISTENT");
    try std.testing.expectError(PullError.QuantNotFound, result);
}

test "selectFile empty files" {
    const files = [_]GgufFile{};
    const result = selectFile(&files, null);
    try std.testing.expectError(PullError.NoGgufFiles, result);
}

test "selectFile case insensitive match" {
    const files = [_]GgufFile{
        .{ .filename = "model-Q4_K_M.gguf", .size = 500 },
    };
    const result = try selectFile(&files, "q4_k_m");
    try std.testing.expectEqualStrings("model-Q4_K_M.gguf", result.filename);
}

test "selectFile preference order" {
    // Q4_K_S should be picked over Q6_K (higher in quant_preference)
    const files = [_]GgufFile{
        .{ .filename = "model-Q6_K.gguf", .size = 1000 },
        .{ .filename = "model-Q4_K_S.gguf", .size = 600 },
    };
    const result = try selectFile(&files, null);
    try std.testing.expectEqualStrings("model-Q4_K_S.gguf", result.filename);
}

test "replaceSlashes empty string" {
    const allocator = std.testing.allocator;
    const result = try replaceSlashes(allocator, "");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("", result);
}

test "progressBar clamps above 100" {
    var buf1: [progress_bar_width + 2]u8 = undefined;
    const bar_100 = progressBar(&buf1, 100);
    var buf2: [progress_bar_width + 2]u8 = undefined;
    const bar_200 = progressBar(&buf2, 200);
    try std.testing.expectEqualStrings(bar_100, bar_200);
}

test "isSafeFilename rejects traversal" {
    try std.testing.expect(!isSafeFilename(""));
    try std.testing.expect(!isSafeFilename("../etc/passwd"));
    try std.testing.expect(!isSafeFilename("foo/../bar"));
    try std.testing.expect(!isSafeFilename("sub/file.bin"));
    try std.testing.expect(!isSafeFilename("back\\slash"));
    try std.testing.expect(!isSafeFilename("null\x00byte"));
    try std.testing.expect(!isSafeFilename(".."));
    try std.testing.expect(!isSafeFilename("model.gguf?q=1"));
    try std.testing.expect(!isSafeFilename("model.gguf#frag"));
    try std.testing.expect(!isSafeFilename("user@host"));
    try std.testing.expect(isSafeFilename("model-Q4_K_M.gguf"));
    try std.testing.expect(isSafeFilename("weights.safetensors"));
}

test "isValidRepoName accepts safe names" {
    try std.testing.expect(isValidRepoName("meta-llama/Llama-3.1-8B"));
    try std.testing.expect(isValidRepoName("google/gemma-3-4b-it"));
    try std.testing.expect(isValidRepoName("org_name/model.v2"));
    try std.testing.expect(!isValidRepoName("org/model?rev=main"));
    try std.testing.expect(!isValidRepoName("org/model#fragment"));
    try std.testing.expect(!isValidRepoName("org/model@latest"));
    try std.testing.expect(!isValidRepoName("org/model name"));
    try std.testing.expect(!isValidRepoName(""));
}

test "isValidHexSha accepts valid hashes" {
    try std.testing.expect(isValidHexSha("abc123"));
    try std.testing.expect(isValidHexSha("deadbeef0123456789abcdef"));
    try std.testing.expect(!isValidHexSha(""));
    try std.testing.expect(!isValidHexSha("ghijk")); // non-hex
    try std.testing.expect(!isValidHexSha("ABCXYZ"));
    // Too long (>64 chars)
    try std.testing.expect(!isValidHexSha("a" ** 65));
    // Exactly 64 chars (valid)
    try std.testing.expect(isValidHexSha("a" ** 64));
}
