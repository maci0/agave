//! Download GGUF and SafeTensors models from HuggingFace Hub.
//!
//! Usage: agave pull <org/repo> [--quant Q4_K_M] [--list]
//!
//! Fetches the HuggingFace model repository listing via the API, selects the
//! best model file(s) based on quantization preference, and downloads them into
//! the standard HuggingFace cache layout with an agave convenience symlink.
//!
//! Supports two model formats:
//!   - **GGUF**: Single-file models (e.g. `model-Q4_K_M.gguf`). Selected by
//!     quantization preference.
//!   - **SafeTensors**: Multi-file models (shards + config.json + tokenizer.json).
//!     Downloaded as a complete directory. Use `--quant safetensors` to prefer
//!     SafeTensors when both formats are available.

const std = @import("std");
const Allocator = std.mem.Allocator;
const display_mod = @import("display.zig");
const durable = @import("durable_file.zig");
const sim_clock = @import("sim_clock.zig");
const config = @import("config.zig");
const version = display_mod.version;

// ── Named constants ──────────────────────────────────────────────────────────

const print_buf_size: usize = 4096;
const download_buf_size: usize = 256 * 1024;
const progress_interval_ns: u64 = 500 * std.time.ns_per_ms;
const max_retries: u32 = 3;
/// Doubles per attempt: 1s, 2s, 4s.
const retry_base_delay_ns: u64 = 1 * std.time.ns_per_s;
/// Public Hugging Face API base, used when `HF_ENDPOINT` is unset.
const hf_api_base_default = "https://huggingface.co";
/// Prevents OOM from malicious server.
const max_api_response_size: usize = 10 * 1024 * 1024;
const progress_bar_width: usize = 30;
const bytes_per_mb: f64 = 1024.0 * 1024.0;
const bytes_per_gb: f64 = 1024.0 * 1024.0 * 1024.0;

const Io = std.Io;
const stderr_file = Io.File.stderr();

/// Module-level Io instance, set by run() from caller.
var mod_io: Io = undefined;

/// Counter behind `tempSuffix` under a clock override. u32 so 32-bit atomics
/// cover every target.
var temp_seq = std.atomic.Value(u32).init(0);

/// Unique suffix for the temp path of an atomic symlink publish.
///
/// Production draws OS entropy: the suffix only has to keep two concurrent
/// `pull` processes off each other's temp path, and unpredictability costs
/// nothing there. Under a clock override the entropy is replaced by a call
/// counter, because that temp path is printed in diagnostics and a replay
/// must reproduce the same run byte-for-byte. Simulation mode drives one
/// process, where the counter is unique per publish.
fn tempSuffix() u64 {
    if (!sim_clock.isOverridden()) {
        var buf: [8]u8 = undefined;
        mod_io.random(&buf);
        return std.mem.readInt(u64, &buf, .little);
    }
    return temp_seq.fetchAdd(1, .monotonic);
}

/// Nanosecond timestamp via sim_clock's MONOTONIC timeline (progress-interval
/// deltas). REALTIME can jump under NTP and make the progress bar stutter
/// or stall. Under override, progress intervals follow virtual time so a
/// replay does not inherit host download speed into the bar.
fn nanoTimestamp() i128 {
    return sim_clock.monoNano();
}

/// Backoff before download retry `attempt`, 0-based: the first retry (attempt
/// 0) waits `retry_base_delay_ns`, then it doubles per attempt.
/// Routes through sim_clock so a clock override advances virtual time
/// instead of blocking wall-clock time between attempts.
fn sleepRetry(attempt: u32) void {
    sim_clock.sleepNs(retry_base_delay_ns << @intCast(attempt));
}

/// Validate that a filename from the API has no path traversal components
/// and no URL-special characters that could inject query/fragment into
/// download URLs (CWE-74).
/// Allowlist: ASCII alphanumerics, `-`, `_`, `.`. Rejects `..`, `%` (encoding
/// bypass), spaces, quotes, and other URL/JSON metacharacters.
fn isSafeFilename(name: []const u8) bool {
    if (name.len == 0 or name.len > 255) return false;
    if (std.mem.indexOf(u8, name, "..") != null) return false;
    for (name) |c| {
        switch (c) {
            'a'...'z', 'A'...'Z', '0'...'9', '-', '_', '.' => {},
            else => return false,
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

/// Model format discriminator for download strategy.
pub const ModelFormat = enum {
    gguf,
    safetensors,
};

/// Parsed command-line arguments for the `pull` sub-command.
pub const PullArgs = struct {
    /// Repository identifier in `org/repo` format.
    repo: []const u8,
    /// Optional quantization filter string (e.g. "Q4_K_M", "safetensors").
    quant: ?[]const u8 = null,
    /// If true, list available model files and exit without downloading.
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

/// SafeTensors model metadata: shard files and auxiliary files present in the repo.
pub const SafeTensorsModel = struct {
    /// Shard filenames (e.g. "model-00001-of-00005.safetensors" or "model.safetensors").
    shards: []const []const u8,
    /// Per-shard file sizes in bytes (parallel to `shards`).
    shard_sizes: []const u64,
    /// Total size of all shards combined.
    total_size: u64,
    /// Whether `model.safetensors.index.json` exists (multi-shard model).
    has_index: bool,
    /// Size of `model.safetensors.index.json` in bytes (0 if missing or unknown).
    index_size: u64 = 0,
    /// Whether `config.json` exists in the repo.
    has_config: bool,
    /// Size of `config.json` in bytes (0 if missing or unknown).
    config_size: u64 = 0,
    /// Whether `tokenizer.json` exists in the repo.
    has_tokenizer: bool,
    /// Size of `tokenizer.json` in bytes (0 if missing or unknown).
    tokenizer_size: u64 = 0,
    /// Whether `tokenizer_config.json` exists in the repo.
    has_tokenizer_config: bool,
    /// Size of `tokenizer_config.json` in bytes (0 if missing or unknown).
    tokenizer_config_size: u64 = 0,
};

/// Auxiliary files that should be downloaded alongside SafeTensors shards.
const safetensors_aux_files = [_][]const u8{
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
};

/// Result from listing model files in a repository.
pub const ListResult = struct {
    /// Available GGUF files (empty slice if none).
    files: []GgufFile,
    /// SafeTensors model info (null if no SafeTensors files found).
    safetensors: ?SafeTensorsModel,
    /// Git commit SHA for the repository HEAD.
    commit_sha: []const u8,
    /// Arena allocator that owns all returned memory. Caller must call
    /// `deinit()` to free.
    arena: std.heap.ArenaAllocator,

    /// Free all memory owned by the list result (files, metadata, strings).
    pub fn deinit(self: *ListResult) void {
        self.arena.deinit();
    }

    /// Returns true if the repository has any downloadable model files.
    pub fn hasAnyFiles(self: *const ListResult) bool {
        return self.files.len > 0 or self.safetensors != null;
    }
};

/// Errors that can occur during the pull operation.
pub const PullError = error{
    /// Invalid command-line argument (missing value, unknown flag, etc.).
    InvalidArgument,
    /// Repository identifier is not in `org/repo` format.
    InvalidRepoFormat,
    /// No model files (GGUF or SafeTensors) found in the repository.
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
    /// The corrupt blob is removed so the next run re-downloads instead of
    /// accepting it via the size-based "already downloaded" check.
    IntegrityCheckFailed,
    /// A snapshot symlink could not be created or renamed into place, so the
    /// published model path does not exist.
    SymlinkFailed,
    /// Local blob size differs from the repository's current file (stale
    /// leftover from an older revision); the stale copy was removed so the
    /// next attempt downloads fresh.
    LocalSizeMismatch,
};

// ── Stderr helpers ───────────────────────────────────────────────────────────

/// Print a formatted message to stderr.
fn eprint(comptime fmt: []const u8, args: anytype) void {
    var buf: [print_buf_size]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, fmt, args) catch return;
    _ = std.posix.system.write(stderr_file.handle, text.ptr, text.len);
}

/// Write bytes to a file handle via posix write.
fn fileWrite(file: Io.File, bytes: []const u8) void {
    _ = std.posix.system.write(file.handle, bytes.ptr, bytes.len);
}

// ── Argument parsing ─────────────────────────────────────────────────────────

/// Print usage information to stdout (pipeable: agave pull --help | less).
pub fn printUsage() void {
    const usage =
        \\agave pull: Download models from HuggingFace Hub
        \\
        \\USAGE:
        \\  agave pull [OPTIONS] <org/repo>
        \\  agave pull [OPTIONS] -- <org/repo>
        \\
        \\ARGUMENTS:
        \\  <org/repo>           Repository in org/repo format
        \\
        \\GENERAL:
        \\  -h, --help           Show this help message and exit
        \\  -v, --version        Print version and exit
        \\
        \\OPTIONS:
        \\      --quant <QUANT>  Select quantization (e.g. Q4_K_M, Q8_0, safetensors)
        \\  -l, --list           List available model files and exit
        \\
        \\ENVIRONMENT:
        \\  HF_TOKEN             HuggingFace API token for private repos
        \\                         Empty/whitespace is unset
        \\  HF_ENDPOINT          HuggingFace API base URL, for a mirror or air-gapped
        \\                         gateway [default: https://huggingface.co]
        \\                         Must start with http:// or https://; a trailing / is
        \\                         trimmed, anything else is rejected before any request
        \\  HF_HOME              Custom HuggingFace cache directory
        \\  XDG_CACHE_HOME       XDG cache base (fallback: ~/.cache)
        \\
        \\EXAMPLES:
        \\  agave pull Qwen/Qwen3.5-27B-GGUF
        \\  agave pull Qwen/Qwen3.5-27B-GGUF --quant Q4_K_M
        \\  agave pull RedHatAI/Qwen3.6-35B-A3B-NVFP4
        \\  agave pull RedHatAI/Qwen3.6-35B-A3B-NVFP4 --quant safetensors
        \\  agave pull Qwen/Qwen3.5-27B-GGUF --list
        \\
        \\FORMATS:
        \\  GGUF           Single-file quantized models (auto-selected by quant preference)
        \\  SafeTensors    Multi-file models (shards + config + tokenizer)
        \\                 Use --quant safetensors to prefer when both formats exist
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

    // Empty/whitespace HF_TOKEN would send `Authorization: Bearer ` and fail
    // auth with a confusing 401; getenv treats those as unset.
    result.token = config.getenv("HF_TOKEN");

    // Reject a malformed HF_ENDPOINT before any request, not on the first
    // network call, so a mirror typo names itself in the error.
    _ = hfApiBase() catch |err| return err;

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
            eprint("  Valid options: --help, --version, --list, --quant\n", .{});
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

/// File size from a HuggingFace `siblings[]` entry. Missing or non-integer
/// `size` is 0 (unknown), which keeps the legacy "any non-empty local file
/// is complete" rule for that blob.
fn siblingSize(sibling: std.json.Value) u64 {
    if (sibling != .object) return 0;
    const size_val = sibling.object.get("size") orelse return 0;
    return switch (size_val) {
        .integer => |i| if (i >= 0) @intCast(i) else 0,
        else => 0,
    };
}

/// Resolve the Hugging Face API base URL: `HF_ENDPOINT` when set, otherwise
/// the public endpoint. A trailing `/` is trimmed so callers appending
/// `/api/...` cannot build a double slash. A value that is not an http(s) URL
/// is rejected here, at startup, instead of producing an unparseable URL at
/// the first request. `HF_ENDPOINT` is the name the `huggingface_hub` tooling
/// uses, so a mirror already configured for it works unchanged.
pub fn hfApiBase() PullError![]const u8 {
    return resolveHfApiBase(config.getenv("HF_ENDPOINT"));
}

fn resolveHfApiBase(val: ?[]const u8) PullError![]const u8 {
    const v = val orelse return hf_api_base_default;
    const trimmed = std.mem.trimEnd(u8, v, "/");
    if (!std.mem.startsWith(u8, trimmed, "https://") and !std.mem.startsWith(u8, trimmed, "http://")) {
        eprint("Error: HF_ENDPOINT must be an http(s) URL, got '{s}'\n", .{v});
        return PullError.InvalidArgument;
    }
    return trimmed;
}

/// Fetch the list of model files available in a HuggingFace repository.
///
/// Makes a GET request to the HuggingFace API and parses the JSON response
/// to extract GGUF filenames, SafeTensors shards, sizes, and the commit SHA.
/// All returned memory is owned by the `ListResult.arena`; call
/// `ListResult.deinit()` when done.
pub fn listModelFiles(allocator: Allocator, repo: []const u8, token: ?[]const u8) (PullError || Allocator.Error)!ListResult {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_alloc = arena.allocator();

    // Build API URL: {endpoint}/api/models/{repo}
    const url = std.fmt.allocPrint(arena_alloc, "{s}/api/models/{s}", .{ try hfApiBase(), repo }) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
    };

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
    defer {
        @memset(body, 0);
        allocator.free(body);
    }

    // Parse JSON response. Do not echo the body: Hub model-card JSON includes
    // publisher identity fields (author/username) and may include account data.
    return parseModelListing(allocator, body) catch |err| switch (err) {
        error.ApiResponseInvalid => {
            eprint("Error: failed to parse API response ({d} bytes)\n", .{body.len});
            return PullError.ApiResponseInvalid;
        },
        error.NoGgufFiles => {
            eprint("Error: no model files (GGUF or SafeTensors) found in '{s}'\n", .{repo});
            return PullError.NoGgufFiles;
        },
        else => |e| return e,
    };
}

/// Parse a HuggingFace `api/models/{repo}` response body into a `ListResult`.
///
/// Split out of `listModelFiles` so the network and the parser can be tested
/// apart: this is the only consumer of remote-controlled bytes, and it is what
/// the fuzz target drives. Silent by contract; diagnostics belong to the
/// caller, which knows the repository name and the body size.
fn parseModelListing(allocator: Allocator, body: []const u8) (PullError || Allocator.Error)!ListResult {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_alloc = arena.allocator();

    const parsed = std.json.parseFromSlice(std.json.Value, arena_alloc, body, .{}) catch
        return PullError.ApiResponseInvalid;
    const root = parsed.value;

    // Extract commit SHA from "sha" field.
    const sha_val = if (root == .object) root.object.get("sha") else null;
    const commit_sha: []const u8 = if (sha_val) |sv| switch (sv) {
        .string => |s| if (isValidHexSha(s)) s else "unknown",
        else => "unknown",
    } else "unknown";

    // Extract siblings array.
    const siblings_val = if (root == .object) root.object.get("siblings") else null;
    const siblings = if (siblings_val) |sv| switch (sv) {
        .array => sv.array.items,
        else => &[_]std.json.Value{},
    } else &[_]std.json.Value{};

    // First pass: count GGUF and SafeTensors files, capture sidecar sizes.
    var gguf_count: usize = 0;
    var st_count: usize = 0;
    var has_st_index = false;
    var has_config = false;
    var has_tokenizer = false;
    var has_tokenizer_config = false;
    var index_size: u64 = 0;
    var config_size: u64 = 0;
    var tokenizer_size: u64 = 0;
    var tokenizer_config_size: u64 = 0;
    for (siblings) |sibling| {
        if (sibling != .object) continue;
        const rfilename = sibling.object.get("rfilename") orelse continue;
        if (rfilename != .string) continue;
        const name = rfilename.string;
        if (std.mem.endsWith(u8, name, ".gguf") and isSafeFilename(name)) {
            gguf_count += 1;
        } else if (std.mem.endsWith(u8, name, ".safetensors") and isSafeFilename(name)) {
            st_count += 1;
        } else if (std.mem.eql(u8, name, "model.safetensors.index.json")) {
            has_st_index = true;
            index_size = siblingSize(sibling);
        } else if (std.mem.eql(u8, name, "config.json")) {
            has_config = true;
            config_size = siblingSize(sibling);
        } else if (std.mem.eql(u8, name, "tokenizer.json")) {
            has_tokenizer = true;
            tokenizer_size = siblingSize(sibling);
        } else if (std.mem.eql(u8, name, "tokenizer_config.json")) {
            has_tokenizer_config = true;
            tokenizer_config_size = siblingSize(sibling);
        }
    }

    if (gguf_count == 0 and st_count == 0) return PullError.NoGgufFiles;

    // Second pass: collect GGUF file info.
    var gguf_files: []GgufFile = &.{};
    if (gguf_count > 0) {
        gguf_files = try arena_alloc.alloc(GgufFile, gguf_count);
        var idx: usize = 0;
        for (siblings) |sibling| {
            if (sibling != .object) continue;
            const rfilename = sibling.object.get("rfilename") orelse continue;
            if (rfilename != .string) continue;
            if (!std.mem.endsWith(u8, rfilename.string, ".gguf") or !isSafeFilename(rfilename.string)) continue;

            gguf_files[idx] = .{
                .filename = rfilename.string,
                .size = siblingSize(sibling),
            };
            idx += 1;
        }
        gguf_files = gguf_files[0..idx];
    }

    // Collect SafeTensors shard info.
    var st_model: ?SafeTensorsModel = null;
    if (st_count > 0) {
        const shards = try arena_alloc.alloc([]const u8, st_count);
        const shard_sizes = try arena_alloc.alloc(u64, st_count);
        var st_idx: usize = 0;
        var total_size: u64 = 0;
        for (siblings) |sibling| {
            if (sibling != .object) continue;
            const rfilename = sibling.object.get("rfilename") orelse continue;
            if (rfilename != .string) continue;
            if (!std.mem.endsWith(u8, rfilename.string, ".safetensors") or !isSafeFilename(rfilename.string)) continue;

            const size = siblingSize(sibling);
            shards[st_idx] = rfilename.string;
            shard_sizes[st_idx] = size;
            // Sizes come from a remote listing, so a sum can exceed u64.
            // Saturate: keeping the pre-overflow total would advertise a
            // download size smaller than the sum of its own shards.
            total_size = std.math.add(u64, total_size, size) catch std.math.maxInt(u64);
            st_idx += 1;
        }

        st_model = .{
            .shards = shards[0..st_idx],
            .shard_sizes = shard_sizes[0..st_idx],
            .total_size = total_size,
            .has_index = has_st_index,
            .index_size = index_size,
            .has_config = has_config,
            .config_size = config_size,
            .has_tokenizer = has_tokenizer,
            .tokenizer_size = tokenizer_size,
            .has_tokenizer_config = has_tokenizer_config,
            .tokenizer_config_size = tokenizer_config_size,
        };
    }

    return ListResult{
        .files = gguf_files,
        .safetensors = st_model,
        .commit_sha = commit_sha,
        .arena = arena,
    };
}

/// Discriminated selection result: either a single GGUF file or the SafeTensors model.
pub const SelectedModel = union(ModelFormat) {
    gguf: GgufFile,
    safetensors: SafeTensorsModel,
};

/// Select the best model file(s) from the listing based on quantization preference.
///
/// If `quant` is "safetensors" (case-insensitive), selects the SafeTensors
/// model if available. If `quant` is provided as a quantization string,
/// searches GGUF files for a match. Otherwise, selects the best GGUF file
/// by built-in quant preference, falling back to SafeTensors if no GGUF
/// files exist.
pub fn selectModel(result: *const ListResult, quant: ?[]const u8) PullError!SelectedModel {
    if (!result.hasAnyFiles()) return PullError.NoGgufFiles;

    if (quant) |q| {
        // Explicit SafeTensors selection.
        if (std.ascii.eqlIgnoreCase(q, "safetensors")) {
            if (result.safetensors) |st| {
                return .{ .safetensors = st };
            }
            eprint("Error: no SafeTensors files found in this repository\n", .{});
            return PullError.QuantNotFound;
        }

        // User-specified quantization: search GGUF files.
        for (result.files) |f| {
            if (std.ascii.indexOfIgnoreCase(f.filename, q) != null) {
                return .{ .gguf = f };
            }
        }
        eprint("Error: no file found matching '{s}'\n", .{q});
        eprint("Available files:\n", .{});
        for (result.files) |f| {
            if (f.size > 0) {
                const size_gb = @as(f64, @floatFromInt(f.size)) / bytes_per_gb;
                eprint("  {s}  ({d:.1} GB)\n", .{ f.filename, size_gb });
            } else {
                eprint("  {s}\n", .{f.filename});
            }
        }
        if (result.safetensors != null) {
            eprint("  (SafeTensors also available, use --quant safetensors)\n", .{});
        }
        return PullError.QuantNotFound;
    }

    // Auto-select: try GGUF by preference order first.
    if (result.files.len > 0) {
        for (&quant_preference) |pref| {
            for (result.files) |f| {
                if (std.ascii.indexOfIgnoreCase(f.filename, pref) != null) {
                    return .{ .gguf = f };
                }
            }
        }
        // Fallback: first GGUF file.
        return .{ .gguf = result.files[0] };
    }

    // No GGUF files, use SafeTensors.
    if (result.safetensors) |st| {
        return .{ .safetensors = st };
    }

    return PullError.NoGgufFiles;
}

/// Select the best GGUF file from the list based on quantization preference.
/// GGUF-only subset of `selectModel` (which also handles SafeTensors).
pub fn selectFile(files: []const GgufFile, quant: ?[]const u8) PullError!GgufFile {
    if (files.len == 0) return PullError.NoGgufFiles;

    if (quant) |q| {
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

    for (&quant_preference) |pref| {
        for (files) |f| {
            if (std.ascii.indexOfIgnoreCase(f.filename, pref) != null) {
                return f;
            }
        }
    }

    return files[0];
}

/// Print a formatted list of available model files with sizes to stdout (pipeable).
pub fn printFileList(files: []const GgufFile, st_model: ?SafeTensorsModel) void {
    const stdout = Io.File.stdout();

    if (files.len > 0) {
        fileWrite(stdout, "GGUF:\n");
        for (files) |f| {
            const size_gb = @as(f64, @floatFromInt(f.size)) / bytes_per_gb;
            var buf: [print_buf_size]u8 = undefined;
            if (f.size > 0) {
                fileWrite(stdout, std.fmt.bufPrint(&buf, "  {s}  ({d:.1} GB)\n", .{ f.filename, size_gb }) catch continue);
            } else {
                fileWrite(stdout, std.fmt.bufPrint(&buf, "  {s}\n", .{f.filename}) catch continue);
            }
        }
    }

    if (st_model) |st| {
        if (files.len > 0) fileWrite(stdout, "\n");
        var buf: [print_buf_size]u8 = undefined;
        const total_gb = @as(f64, @floatFromInt(st.total_size)) / bytes_per_gb;
        fileWrite(stdout, std.fmt.bufPrint(&buf, "SafeTensors:  {d} shard(s)  ({d:.1} GB total)\n", .{ st.shards.len, total_gb }) catch "SafeTensors:\n");
        for (st.shards, st.shard_sizes) |shard, size| {
            if (size > 0) {
                const shard_gb = @as(f64, @floatFromInt(size)) / bytes_per_gb;
                fileWrite(stdout, std.fmt.bufPrint(&buf, "  {s}  ({d:.1} GB)\n", .{ shard, shard_gb }) catch continue);
            } else {
                fileWrite(stdout, std.fmt.bufPrint(&buf, "  {s}\n", .{shard}) catch continue);
            }
        }
        // Show auxiliary files.
        if (st.has_config) fileWrite(stdout, "  config.json\n");
        if (st.has_tokenizer) fileWrite(stdout, "  tokenizer.json\n");
        if (st.has_tokenizer_config) fileWrite(stdout, "  tokenizer_config.json\n");
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

    // Use privileged_headers for auth, stripped on redirect to prevent token leaking to CDN.
    var priv_headers_buf: [1]std.http.Header = undefined;
    const priv_headers: []const std.http.Header = if (auth_value) |av| blk: {
        priv_headers_buf[0] = .{ .name = "Authorization", .value = av };
        break :blk priv_headers_buf[0..1];
    } else &.{};

    const uri = std.Uri.parse(url) catch return PullError.HttpRequestFailed;
    // Issue the request through `client.request` rather than `client.fetch`
    // so the underlying socket is reachable for `api_stall_timeout_sec`.
    // Without it, a server that accepts the connection and then stops
    // responding blocks this call forever and `agave pull` never returns or
    // retries. Same bound and rationale as the download path below.
    var req = client.request(.GET, uri, .{ .privileged_headers = priv_headers }) catch |err| {
        eprint("Error: HTTP request failed: {}\n", .{err});
        return PullError.HttpRequestFailed;
    };
    defer req.deinit();
    req.sendBodiless() catch |err| {
        eprint("Error: HTTP request failed: {}\n", .{err});
        return PullError.HttpRequestFailed;
    };
    setSocketReadTimeout(req, api_stall_timeout_sec);

    // Cap the listing body while reading. Checking length after
    // `toOwnedSlice` still allocated the full payload (CWE-400).
    const cap = max_api_response_size + 1;
    const buf = try allocator.alloc(u8, cap);
    defer allocator.free(buf);

    var redirect_buf: [8 * 1024]u8 = undefined;
    var response = req.receiveHead(&redirect_buf) catch |err| {
        eprint("Error: HTTP request failed: {}\n", .{err});
        return PullError.HttpRequestFailed;
    };

    var transfer_buf: [64 * 1024]u8 = undefined;
    const reader = response.reader(&transfer_buf);
    var writer: std.Io.Writer = .fixed(buf);
    _ = reader.streamRemaining(&writer) catch |err| {
        if (err == error.WriteFailed) {
            eprint("Error: API response exceeds {d} bytes\n", .{max_api_response_size});
        } else {
            eprint("Error: HTTP request failed after {d}s without completing: {}\n", .{ api_stall_timeout_sec, err });
        }
        return PullError.HttpRequestFailed;
    };

    switch (response.head.status) {
        .ok => {},
        .not_found => return PullError.RepoNotFound,
        .unauthorized, .forbidden => return PullError.AuthenticationFailed,
        else => {
            eprint("Error: HTTP {d}\n", .{@intFromEnum(response.head.status)});
            return PullError.HttpRequestFailed;
        },
    }

    if (writer.end > max_api_response_size) {
        eprint("Error: API response exceeds {d} bytes\n", .{max_api_response_size});
        return PullError.HttpRequestFailed;
    }

    return allocator.dupe(u8, buf[0..writer.end]);
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
    if (config.getenv("HF_HOME")) |hf_home| {
        return std.fmt.allocPrint(allocator, "{s}/hub/models--{s}", .{ hf_home, repo_escaped }) catch
            return error.OutOfMemory;
    }

    // XDG_CACHE_HOME overrides default cache location
    if (config.getenv("XDG_CACHE_HOME")) |xdg| {
        return std.fmt.allocPrint(allocator, "{s}/huggingface/hub/models--{s}", .{ xdg, repo_escaped }) catch
            return error.OutOfMemory;
    }

    // Default: $HOME/.cache/huggingface/hub/
    const home = config.getenv("HOME") orelse {
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

/// Hugging Face cache layout for a repo/commit, with all three directories created.
const CachePaths = struct {
    blobs: []const u8,
    snapshots: []const u8,
    refs: []const u8,
};

fn cachePaths(pa: Allocator, repo: []const u8, commit_sha: []const u8) (PullError || Allocator.Error)!CachePaths {
    const cache_dir = try hfCacheDir(pa, repo);
    const paths = CachePaths{
        .blobs = try std.fmt.allocPrint(pa, "{s}/blobs", .{cache_dir}),
        .snapshots = try std.fmt.allocPrint(pa, "{s}/snapshots/{s}", .{ cache_dir, commit_sha }),
        .refs = try std.fmt.allocPrint(pa, "{s}/refs", .{cache_dir}),
    };
    try ensureDir(paths.blobs);
    try ensureDir(paths.snapshots);
    try ensureDir(paths.refs);
    return paths;
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
    const home = config.getenv("HOME") orelse {
        eprint("Warning: HOME not set, skipping agave model symlink\n", .{});
        return;
    };

    // Split repo into org and name.
    const slash_idx = std.mem.indexOfScalar(u8, repo, '/') orelse {
        eprint("Warning: repo '{s}' has no org/name separator, skipping agave symlink\n", .{repo});
        return;
    };
    const org = repo[0..slash_idx];
    const name = repo[slash_idx + 1 ..];

    // Create $HOME/.cache/agave/models/{org}/
    const agave_dir = std.fmt.allocPrint(allocator, "{s}/.cache/agave/models/{s}", .{ home, org }) catch {
        eprint("Warning: OOM creating agave cache path for {s}/{s}\n", .{ org, name });
        return;
    };
    defer allocator.free(agave_dir);
    ensureDir(agave_dir) catch |err| {
        eprint("Warning: could not create agave cache dir '{s}': {}\n", .{ agave_dir, err });
        return;
    };

    // Create symlink: $HOME/.cache/agave/models/{org}/{repo_name} -> snapshot_dir
    const link_path = std.fmt.allocPrint(allocator, "{s}/{s}", .{ agave_dir, name }) catch {
        eprint("Warning: OOM creating agave symlink path for {s}/{s}\n", .{ org, name });
        return;
    };
    defer allocator.free(link_path);

    // Atomic symlink replacement: create at temp path with a unique suffix to
    // prevent TOCTOU races (CWE-367), then rename over target.
    const tmp_path = std.fmt.allocPrint(allocator, "{s}.tmp.{x}", .{
        link_path, tempSuffix(),
    }) catch {
        eprint("Warning: OOM creating temp path for agave symlink {s}/{s}\n", .{ org, name });
        return;
    };
    defer allocator.free(tmp_path);

    createSymlink(allocator, snapshot_dir, tmp_path) catch |err| {
        eprint("Warning: could not create agave symlink: {}\n", .{err});
        return;
    };
    Io.Dir.rename(Io.Dir.cwd(), tmp_path, Io.Dir.cwd(), link_path, mod_io) catch |err| {
        eprint("Warning: could not finalize agave symlink: {}\n", .{err});
        Io.Dir.cwd().deleteFile(mod_io, tmp_path) catch |del_err| {
            eprint("Warning: could not clean up temp file '{s}': {}\n", .{ tmp_path, del_err });
        };
    };
}

// ── Download with progress ───────────────────────────────────────────────────

/// Outcome of receiving HTTP 416 (Range Not Satisfiable) during a resume.
const RangeNotSatisfiable = enum {
    /// Local blob already holds exactly the expected content length.
    complete,
    /// Local blob length differs from the repository's current file: stale
    /// leftover from an older revision or a corrupted oversized file.
    stale_local_file,
};

/// Decide whether a 416 response means "already fully downloaded" or "the
/// local file no longer matches what the repository serves".
///
/// A 416 only certifies completion when the local size equals the size the
/// API reported for this revision (`expected_size`; 0 = unknown size, which
/// keeps the historical trust-416 behavior). Without this check, re-running
/// `agave pull` against a repository whose file shrank (or was replaced)
/// accepts the stale local copy as valid and publishes a corrupt model.
fn classifyRangeNotSatisfiable(existing_size: u64, expected_size: u64) RangeNotSatisfiable {
    if (expected_size > 0 and existing_size != expected_size) return .stale_local_file;
    return .complete;
}

/// True when a local blob is already the repository's current file.
///
/// Known `expected_size` must match exactly: a truncated leftover from a
/// failed attempt, or a leftover from an older (larger or smaller) revision,
/// is not complete. Unknown size (`0`) keeps the legacy rule that any
/// non-empty local file is treated as complete.
fn localSizeIsComplete(local_size: u64, expected_size: u64) bool {
    if (expected_size > 0) return local_size == expected_size;
    return local_size > 0;
}

/// Stat `path` and apply `localSizeIsComplete`. Missing files are incomplete.
fn isLocalBlobComplete(path: []const u8, expected_size: u64) bool {
    const stat = Io.Dir.cwd().statFile(mod_io, path, .{}) catch return false;
    return localSizeIsComplete(stat.size, expected_size);
}

/// True when the HTTP-advertised total matches the repository listing.
///
/// `expected_size` 0 (listing omitted the size) cannot contradict the
/// response. `content_length_present` false means the server omitted
/// Content-Length; the byte-count check after the body is the remaining
/// guard. Used before opening the local file so a 200 of the wrong length
/// does not truncate a valid resume prefix.
fn advertisedSizeAgrees(total_size: u64, expected_size: u64, content_length_present: bool) bool {
    if (expected_size == 0 or !content_length_present) return true;
    return total_size == expected_size;
}

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
///
/// `expected_size` is the file size reported by the API listing for the
/// current revision (0 = unknown). It guards the resume path: a local blob
/// whose size no longer matches the repository is removed and re-downloaded
/// instead of being accepted as complete, so repeated runs converge on the
/// repository's actual content.
fn downloadFile(
    allocator: Allocator,
    repo: []const u8,
    filename: []const u8,
    blob_path: []const u8,
    token: ?[]const u8,
    expected_size: u64,
) PullError!void {
    // Build download URL.
    const url = std.fmt.allocPrint(allocator, "{s}/{s}/resolve/main/{s}", .{ try hfApiBase(), repo, filename }) catch
        return PullError.DownloadFailed;
    defer allocator.free(url);

    // Detect TTY once for all attempts, progress bars use \r which
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
            sleepRetry(attempt);
        }

        downloadFileOnce(allocator, url, blob_path, token, is_tty, expected_size) catch |err| {
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
///
/// Bound on the gap between two reads of a download body, in seconds. A
/// stalled connection then fails into the normal retry path instead of hanging
/// with a partial blob on disk. Generous enough that a slow mirror that keeps
/// sending bytes never trips it.
const download_stall_timeout_sec: i64 = 60;

/// Bound on the gap between two reads of an API listing response, in seconds.
/// Same reasoning as the download bound above: it never fires while bytes keep
/// arriving, and a connection that accepts the request and then goes quiet
/// fails fast instead of hanging `agave pull` with no output forever.
const api_stall_timeout_sec: i64 = 60;

/// Set SO_RCVTIMEO on the socket backing `req`, so the body read loop cannot
/// block forever on a connection that stopped delivering. Advisory: a socket
/// that cannot take the option (an already-released connection) warns and
/// leaves the read untimed rather than failing the request.
fn setSocketReadTimeout(req: std.http.Client.Request, seconds: i64) void {
    const conn = req.connection orelse {
        eprint("Warning: connection already released, no read timeout set\n", .{});
        return;
    };
    const timeout = std.posix.timeval{ .sec = seconds, .usec = 0 };
    std.posix.setsockopt(
        conn.stream_reader.stream.socket.handle,
        std.posix.SOL.SOCKET,
        std.posix.SO.RCVTIMEO,
        std.mem.asBytes(&timeout),
    ) catch |err| {
        eprint("Warning: could not set the {d}s download read timeout ({}); a stalled connection must be interrupted with Ctrl+C\n", .{ seconds, err });
    };
}

/// `expected_size` is the repository's current file size per the API listing
/// (0 = unknown); see `downloadFile`.
fn downloadFileOnce(
    allocator: Allocator,
    url: []const u8,
    blob_path: []const u8,
    token: ?[]const u8,
    is_tty: bool,
    expected_size: u64,
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
    // Accept-Encoding: identity prevents server from gzipping JSON files.
    var extra_storage: [2]std.http.Header = undefined;
    var ext_idx: usize = 0;
    if (range_value) |rv| {
        extra_storage[ext_idx] = .{ .name = "Range", .value = rv };
        ext_idx += 1;
    }
    extra_storage[ext_idx] = .{ .name = "Accept-Encoding", .value = "identity" };
    ext_idx += 1;

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
            switch (classifyRangeNotSatisfiable(existing_size, expected_size)) {
                .complete => return,
                .stale_local_file => {
                    // Local blob size differs from the repository's current
                    // file: leftover from an older revision or corrupted.
                    // Remove it so the retry starts from a clean state.
                    eprint("Error: local file size ({d}) does not match repository file ({d}), removing stale copy\n", .{ existing_size, expected_size });
                    Io.Dir.cwd().deleteFile(mod_io, blob_path) catch |del_err| {
                        eprint("Warning: could not remove stale file '{s}': {}\n", .{ blob_path, del_err });
                    };
                    return PullError.LocalSizeMismatch;
                },
            }
        },
        else => {
            eprint("Error: HTTP {d}\n", .{@intFromEnum(status)});
            return PullError.HttpRequestFailed;
        },
    }

    // Determine total size (checked arithmetic prevents overflow from crafted Content-Length).
    const maybe_len = response.head.content_length;
    const content_length_present = maybe_len != null;
    const content_length = maybe_len orelse blk: {
        eprint("Warning: server omitted Content-Length; size-based integrity check disabled\n", .{});
        break :blk @as(u64, 0);
    };
    const total_size: u64 = if (status == .partial_content)
        std.math.add(u64, existing_size, content_length) catch return PullError.DownloadFailed
    else
        content_length;

    // Refuse a body whose advertised length disagrees with the listing
    // *before* opening the file: a 200 of the wrong size must not truncate
    // a valid resume prefix. The retry then Range-resumes (or starts over
    // after a 416 stale-delete) instead of appending to a short 200 body.
    if (!advertisedSizeAgrees(total_size, expected_size, content_length_present)) {
        eprint("Error: download size {d} does not match repository file {d}\n", .{ total_size, expected_size });
        return PullError.DownloadFailed;
    }

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
            eprint("Error: blob path is a symlink, refusing to write (possible symlink attack)\n", .{});
        }
        return PullError.DownloadFailed;
    };
    const file: Io.File = .{ .handle = fd, .flags = .{ .nonblocking = false } };
    // Success path closes explicitly below and checks the result: a failed
    // close on a write handle means buffered data was lost (e.g. ENOSPC at
    // flush). Error paths close best-effort via errdefer.
    errdefer _ = std.c.close(file.handle);

    // Seek to resume offset for append.
    if (start_offset > 0) {
        if (std.c.lseek(file.handle, @intCast(start_offset), std.c.SEEK.SET) == -1) {
            eprint("Error: failed to seek to resume offset {d}\n", .{start_offset});
            return PullError.DownloadFailed;
        }
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

    // A stalled TCP connection would otherwise block readSliceShort forever,
    // with the partial blob on disk and no message. SO_RCVTIMEO bounds the gap
    // between two reads: it never fires while bytes keep arriving, so a slow
    // mirror is unaffected, and a dead one fails into the normal retry path
    // that Range-resumes. Failure to set it is advisory, not fatal.
    setSocketReadTimeout(req, download_stall_timeout_sec);
    var read_buf: [download_buf_size]u8 = undefined;
    while (true) {
        const bytes_read = body_reader.readSliceShort(&read_buf) catch |err| {
            eprint("\nError: network read failed during download: {}\n", .{err});
            eprint("  The connection stalled for over {d}s, or the network dropped. Re-run to resume from {d:.1} MB.\n", .{ download_stall_timeout_sec, @as(f64, @floatFromInt(downloaded)) / bytes_per_mb });
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

        // Update progress bar periodically (TTY only, \r produces garbled
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
                    @intFromFloat(@floor(@as(f64, @floatFromInt(remaining_bytes)) / (speed_mbps * bytes_per_mb)))
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
    // Skip the Content-Length comparison when the header was omitted: a 206
    // with no length would otherwise treat the resume offset as the total
    // and fail a successful remainder download.
    if (content_length_present and total_size > 0 and downloaded != total_size) {
        eprint("Error: downloaded {d} bytes but expected {d}, file may be truncated\n", .{ downloaded, total_size });
        return PullError.DownloadFailed;
    }
    if (expected_size > 0 and downloaded != expected_size) {
        eprint("Error: downloaded {d} bytes but repository lists {d}\n", .{ downloaded, expected_size });
        return PullError.DownloadFailed;
    }

    // fsync before advertising the blob as complete: close() is not durable,
    // and the snapshot symlink is created immediately after this returns.
    durable.syncFd(file.handle) catch |err| {
        eprint("Error: fsync failed for '{s}': {}\n", .{ blob_path, err });
        return PullError.DownloadFailed;
    };

    if (std.c.close(file.handle) != 0) {
        eprint("Error: failed to close '{s}' after download (flush may have failed)\n", .{blob_path});
        return PullError.DownloadFailed;
    }
}

// ── Orchestrator ─────────────────────────────────────────────────────────────

/// Execute the full model pull workflow.
///
///  1. List model files in the repository (GGUF + SafeTensors)
///  2. If --list, print file list to stdout and return
///  3. Select the best model based on --quant filter and format preference
///  4. Build HuggingFace cache directory structure
///  5. Download model files (single GGUF or multi-file SafeTensors)
///  6. Verify integrity (GGUF magic bytes, SafeTensors shard count)
///  7. Create snapshot symlinks
///  8. Write refs/main with commit SHA
///  9. Create agave convenience symlink
/// 10. Print the final model path
pub fn pullModel(allocator: Allocator, args: PullArgs) (PullError || Allocator.Error)!void {
    eprint("Fetching model info for '{s}'...\n", .{args.repo});
    var list_result = try listModelFiles(allocator, args.repo, args.token);
    defer list_result.deinit();

    if (args.list_only) {
        eprint("Available model files in '{s}':\n", .{args.repo});
        printFileList(list_result.files, list_result.safetensors);
        return;
    }

    const selected = try selectModel(&list_result, args.quant);

    switch (selected) {
        .gguf => |gguf| try pullGgufModel(allocator, args, &list_result, gguf),
        .safetensors => |st| try pullSafeTensorsModel(allocator, args, &list_result, st),
    }
}

/// Return the shard count if filename matches split-GGUF pattern "…-00001-of-NNNNN.gguf".
/// Returns 0 if not a split file (single shard).
fn detectGgufShardCount(filename: []const u8) u32 {
    // Pattern: basename ends with -NNNNN-of-MMMMM.gguf
    const gguf_sfx = ".gguf";
    if (!std.mem.endsWith(u8, filename, gguf_sfx)) return 0;
    const stem = filename[0 .. filename.len - gguf_sfx.len];
    // Find the last two hyphen-separated digit groups: "…-NNNNN-of-MMMMM"
    const of_pos = std.mem.lastIndexOf(u8, stem, "-of-") orelse return 0;
    const total_str = stem[of_pos + 4 ..];
    const total = std.fmt.parseInt(u32, total_str, 10) catch return 0;
    if (total < 2) return 0;
    // Verify the preceding group is also digits
    const pre = stem[0..of_pos];
    const dash_pos = std.mem.lastIndexOfScalar(u8, pre, '-') orelse return 0;
    const idx_str = pre[dash_pos + 1 ..];
    _ = std.fmt.parseInt(u32, idx_str, 10) catch return 0;
    return total;
}

/// Build the shard filename for a given index (1-based) given the first shard filename.
/// Preserves zero-padding width of the original filename.
fn buildShardFilename(allocator: Allocator, shard1: []const u8, idx: u32, total: u32) ![]u8 {
    const gguf_sfx = ".gguf";
    // A remote filename shorter than the suffix underflows the stem slice, so
    // check the suffix rather than trusting the caller's shard count.
    if (!std.mem.endsWith(u8, shard1, gguf_sfx)) return error.InvalidShardName;
    const stem = shard1[0 .. shard1.len - gguf_sfx.len];
    const of_pos = std.mem.lastIndexOf(u8, stem, "-of-") orelse return error.InvalidShardName;
    const pre_dash = std.mem.lastIndexOfScalar(u8, stem[0..of_pos], '-') orelse return error.InvalidShardName;
    const base = stem[0 .. pre_dash + 1]; // includes trailing dash
    const idx1_str = stem[pre_dash + 1 .. of_pos];
    const width = idx1_str.len; // e.g. 5 for "00001"
    // Format idx and total with zero-padding to match original width
    var idx_buf: [16]u8 = undefined;
    var tot_buf: [16]u8 = undefined;
    const idx_raw = std.fmt.bufPrint(&idx_buf, "{d}", .{idx}) catch return error.InvalidShardName;
    const tot_raw = std.fmt.bufPrint(&tot_buf, "{d}", .{total}) catch return error.InvalidShardName;
    // Guard against malformed filenames where digit width is too narrow for the values.
    if (idx_raw.len > width or tot_raw.len > width) return error.InvalidShardName;
    // Pad with leading zeros
    var idx_padded: [16]u8 = [_]u8{'0'} ** 16;
    @memcpy(idx_padded[width - idx_raw.len .. width], idx_raw);
    var tot_padded: [16]u8 = [_]u8{'0'} ** 16;
    @memcpy(tot_padded[width - tot_raw.len .. width], tot_raw);
    return std.fmt.allocPrint(allocator, "{s}{s}-of-{s}{s}", .{
        base, idx_padded[0..width], tot_padded[0..width], gguf_sfx,
    });
}

/// Verify the GGUF magic bytes of a downloaded blob.
///
/// A blob with an invalid header (wrong magic, fewer than 4 readable bytes)
/// is removed before returning `false`: its size already matches the
/// repository listing, so leaving it in place would make every rerun take
/// the "already downloaded" path and fail this same check forever. Removing
/// it keeps repeated `agave pull` invocations convergent. Returns `true`
/// when the header is valid or the file cannot be opened for checking
/// (nothing observed to act on).
fn verifyGgufBlob(io: Io, blob_path: []const u8) bool {
    const f = Io.Dir.cwd().openFile(io, blob_path, .{}) catch |err| {
        eprint("Warning: could not open file for integrity check: {}\n", .{err});
        return true;
    };
    defer f.close(io);
    var magic: [4]u8 = undefined;
    // A read error is not evidence of corruption: reporting it as a short read
    // would unlink a multi-gigabyte completed download over a transient EIO.
    const n = f.readPositionalAll(io, &magic, 0) catch |err| {
        eprint("Warning: could not read '{s}' for integrity check: {}\n", .{ blob_path, err });
        return true;
    };
    if (n >= 4 and std.mem.eql(u8, &magic, "GGUF")) return true;
    Io.Dir.cwd().deleteFile(io, blob_path) catch |del_err| {
        eprint("Warning: could not remove corrupt file '{s}': {}\n", .{ blob_path, del_err });
    };
    eprint("Error: downloaded file does not have valid GGUF header, corrupt file removed\n", .{});
    eprint("  Re-run 'agave pull' to download a fresh copy\n", .{});
    return false;
}

/// Download a single GGUF model file with cache layout and integrity check.
/// Automatically downloads all shards when the selected file is part of a split GGUF.
fn pullGgufModel(
    allocator: Allocator,
    args: PullArgs,
    list_result: *const ListResult,
    selected: GgufFile,
) (PullError || Allocator.Error)!void {
    const size_gb = @as(f64, @floatFromInt(selected.size)) / bytes_per_gb;
    eprint("Selected: {s} ({d:.1} GB)\n", .{ selected.filename, size_gb });

    // Build cache paths (arena-allocated, freed together at function exit).
    var path_arena = std.heap.ArenaAllocator.init(allocator);
    defer path_arena.deinit();
    const pa = path_arena.allocator();

    const cache = try cachePaths(pa, args.repo, list_result.commit_sha);
    const blobs_dir = cache.blobs;
    const snapshots_dir = cache.snapshots;
    const refs_dir = cache.refs;

    // Check if already downloaded.
    const blob_path = std.fmt.allocPrint(pa, "{s}/{s}", .{ blobs_dir, selected.filename }) catch return error.OutOfMemory;

    const already_complete = isLocalBlobComplete(blob_path, selected.size);
    if (already_complete) {
        eprint("File already downloaded: {s}\n", .{blob_path});
    }

    // Download shard 1 (or the only shard) if needed.
    if (!already_complete) {
        try downloadFile(allocator, args.repo, selected.filename, blob_path, args.token, selected.size);
        eprint("Download complete.\n", .{});
    }

    // Detect split-GGUF and download remaining shards (shard 2..N).
    const total_shards = detectGgufShardCount(selected.filename);
    if (total_shards > 1) {
        eprint("Split GGUF: {d} shards total, downloading remaining shards...\n", .{total_shards});
        for (2..total_shards + 1) |shard_idx| {
            // Fatal, not a skip: a published model missing one shard fails at
            // open time with no indication that `pull` said anything was wrong.
            const shard_name = buildShardFilename(pa, selected.filename, @intCast(shard_idx), total_shards) catch |err| {
                eprint("Error: could not build shard {d}/{d} filename: {}\n", .{ shard_idx, total_shards, err });
                return error.IntegrityCheckFailed;
            };
            defer pa.free(shard_name);

            // Find matching GgufFile entry for size info.
            var shard_size: u64 = 0;
            for (list_result.files) |f| {
                if (std.mem.eql(u8, f.filename, shard_name)) {
                    shard_size = f.size;
                    break;
                }
            }
            const shard_gb = @as(f64, @floatFromInt(shard_size)) / bytes_per_gb;
            const shard_blob = std.fmt.allocPrint(pa, "{s}/{s}", .{ blobs_dir, shard_name }) catch return error.OutOfMemory;
            const shard_link = std.fmt.allocPrint(pa, "{s}/{s}", .{ snapshots_dir, shard_name }) catch return error.OutOfMemory;

            const shard_done = isLocalBlobComplete(shard_blob, shard_size);
            if (shard_done) {
                eprint("  shard {d}/{d} already downloaded: {s}\n", .{ shard_idx, total_shards, shard_name });
            }

            if (!shard_done) {
                eprint("  shard {d}/{d}: {s} ({d:.1} GB)\n", .{ shard_idx, total_shards, shard_name, shard_gb });
                try downloadFile(allocator, args.repo, shard_name, shard_blob, args.token, shard_size);
                eprint("  shard {d}/{d} complete.\n", .{ shard_idx, total_shards });
            }
            try atomicSymlink(pa, std.fmt.allocPrint(pa, "../../blobs/{s}", .{shard_name}) catch return error.OutOfMemory, shard_link);
        }
    }

    // Verify GGUF magic bytes (catches truncation and corruption).
    if (std.mem.endsWith(u8, selected.filename, ".gguf")) {
        if (!verifyGgufBlob(mod_io, blob_path)) return PullError.IntegrityCheckFailed;
    }

    // Create snapshot symlink (relative path).
    const snapshot_link = std.fmt.allocPrint(pa, "{s}/{s}", .{ snapshots_dir, selected.filename }) catch
        return error.OutOfMemory;

    const relative_blob = std.fmt.allocPrint(pa, "../../blobs/{s}", .{selected.filename}) catch
        return error.OutOfMemory;

    try atomicSymlink(pa, relative_blob, snapshot_link);

    // Write refs/main and create convenience symlink.
    writeRefsMain(pa, refs_dir, list_result.commit_sha);
    createAgaveSymlink(allocator, args.repo, snapshots_dir);

    // Print final model path.
    fileWrite(Io.File.stdout(), snapshot_link);
    fileWrite(Io.File.stdout(), "\n");
    eprint("\nModel ready at:\n  {s}\n", .{snapshot_link});
    eprint("Run:\n  agave {s} \"your prompt\"\n", .{snapshot_link});
}

/// Download a SafeTensors model: all shards + config.json + tokenizer files.
fn pullSafeTensorsModel(
    allocator: Allocator,
    args: PullArgs,
    list_result: *const ListResult,
    st: SafeTensorsModel,
) (PullError || Allocator.Error)!void {
    const total_gb = @as(f64, @floatFromInt(st.total_size)) / bytes_per_gb;
    eprint("Selected: SafeTensors model ({d} shards, {d:.1} GB total)\n", .{ st.shards.len, total_gb });

    // Build cache paths.
    var path_arena = std.heap.ArenaAllocator.init(allocator);
    defer path_arena.deinit();
    const pa = path_arena.allocator();

    const cache = try cachePaths(pa, args.repo, list_result.commit_sha);
    const blobs_dir = cache.blobs;
    const snapshots_dir = cache.snapshots;
    const refs_dir = cache.refs;

    // Download each shard.
    for (st.shards, st.shard_sizes, 0..) |shard, expected_size, i| {
        const blob_path = std.fmt.allocPrint(pa, "{s}/{s}", .{ blobs_dir, shard }) catch return error.OutOfMemory;

        const already_complete = isLocalBlobComplete(blob_path, expected_size);
        if (already_complete) {
            eprint("[{d}/{d}] Already downloaded: {s}\n", .{ i + 1, st.shards.len, shard });
        }

        if (!already_complete) {
            eprint("[{d}/{d}] Downloading {s}...\n", .{ i + 1, st.shards.len, shard });
            try downloadFile(allocator, args.repo, shard, blob_path, args.token, expected_size);
        }

        // Create snapshot symlink for this shard.
        const snapshot_link = std.fmt.allocPrint(pa, "{s}/{s}", .{ snapshots_dir, shard }) catch {
            eprint("Warning: OOM creating symlink for shard {s}\n", .{shard});
            continue;
        };
        const relative_blob = std.fmt.allocPrint(pa, "../../blobs/{s}", .{shard}) catch {
            eprint("Warning: OOM creating symlink for shard {s}\n", .{shard});
            continue;
        };
        try atomicSymlink(pa, relative_blob, snapshot_link);
    }

    // Download index file if present.
    if (st.has_index) {
        try pullSidecarFile(allocator, pa, args, blobs_dir, snapshots_dir, "model.safetensors.index.json", st.index_size);
    }

    // Download auxiliary files (config.json, tokenizer.json, tokenizer_config.json).
    const aux_flags = [_]bool{ st.has_config, st.has_tokenizer, st.has_tokenizer_config };
    const aux_sizes = [_]u64{ st.config_size, st.tokenizer_size, st.tokenizer_config_size };
    for (&safetensors_aux_files, aux_flags, aux_sizes) |aux_name, has_file, aux_size| {
        if (!has_file) continue;
        try pullSidecarFile(allocator, pa, args, blobs_dir, snapshots_dir, aux_name, aux_size);
    }

    eprint("Download complete.\n", .{});

    // Write refs/main and create convenience symlink.
    writeRefsMain(pa, refs_dir, list_result.commit_sha);
    createAgaveSymlink(allocator, args.repo, snapshots_dir);

    // Print final model path (the snapshot directory for SafeTensors).
    fileWrite(Io.File.stdout(), snapshots_dir);
    fileWrite(Io.File.stdout(), "\n");
    eprint("\nModel ready at:\n  {s}\n", .{snapshots_dir});
    eprint("Run:\n  agave {s} \"your prompt\"\n", .{snapshots_dir});
}

/// Download one SafeTensors sidecar (index or tokenizer/config) with the same
/// size-based skip used for shards. A leftover from a failed attempt whose
/// length does not match the listing is re-fetched instead of being treated
/// as complete, so a second `agave pull` converges on the repository file.
fn pullSidecarFile(
    allocator: Allocator,
    pa: Allocator,
    args: PullArgs,
    blobs_dir: []const u8,
    snapshots_dir: []const u8,
    filename: []const u8,
    expected_size: u64,
) (PullError || Allocator.Error)!void {
    const blob_path = std.fmt.allocPrint(pa, "{s}/{s}", .{ blobs_dir, filename }) catch
        return error.OutOfMemory;
    if (isLocalBlobComplete(blob_path, expected_size)) {
        eprint("Already downloaded: {s}\n", .{filename});
    } else {
        eprint("Downloading {s}...\n", .{filename});
        // Propagate: a half-written sidecar that is symlinked and reported as
        // complete is accepted by every later run's size check.
        try downloadFile(allocator, args.repo, filename, blob_path, args.token, expected_size);
    }
    const snapshot_link = std.fmt.allocPrint(pa, "{s}/{s}", .{ snapshots_dir, filename }) catch
        return error.OutOfMemory;
    const relative_blob = std.fmt.allocPrint(pa, "../../blobs/{s}", .{filename}) catch
        return error.OutOfMemory;
    try atomicSymlink(pa, relative_blob, snapshot_link);
}

/// Create an atomic symlink replacement (temp + rename) to prevent TOCTOU races.
/// Errors are returned: a snapshot link that is not published makes the whole
/// pull unusable, and the path is printed to stdout for scripting.
fn atomicSymlink(pa: Allocator, target: []const u8, link_path: []const u8) !void {
    const tmp_link = try std.fmt.allocPrint(pa, "{s}.tmp.{x}", .{
        link_path, tempSuffix(),
    });
    defer pa.free(tmp_link);
    createSymlink(pa, target, tmp_link) catch |err| {
        eprint("Error: could not create symlink '{s}': {}\n", .{ link_path, err });
        return error.SymlinkFailed;
    };
    Io.Dir.rename(Io.Dir.cwd(), tmp_link, Io.Dir.cwd(), link_path, mod_io) catch |err| {
        eprint("Error: could not finalize symlink '{s}': {}\n", .{ link_path, err });
        Io.Dir.cwd().deleteFile(mod_io, tmp_link) catch |del_err| {
            eprint("Warning: could not clean up temp file '{s}': {}\n", .{ tmp_link, del_err });
        };
        return error.SymlinkFailed;
    };
}

/// Write the commit SHA to refs/main.
fn writeRefsMain(pa: Allocator, refs_dir: []const u8, commit_sha: []const u8) void {
    const refs_main = std.fmt.allocPrint(pa, "{s}/main", .{refs_dir}) catch |err| {
        eprint("Warning: could not allocate refs/main path: {}\n", .{err});
        return;
    };
    durable.replace(refs_main, commit_sha) catch |err| {
        eprint("Warning: could not write refs/main: {}\n", .{err});
    };
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

    // Both parse-time failures are usage errors: the message is already on
    // stderr, and scripts distinguish a bad invocation from a failed download.
    const maybe_args = parseArgs(&args_iter) catch |err| switch (err) {
        PullError.InvalidArgument, PullError.InvalidRepoFormat => return 2,
        else => {
            eprint("Error: {}\n", .{err});
            return 1;
        },
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
            PullError.LocalSizeMismatch,
            => {},
            error.OutOfMemory => eprint("Error: out of memory\n", .{}),
            else => eprint("Error: {}\n", .{err}),
        }
        return 1;
    };

    return 0;
}

// ── Tests ────────────────────────────────────────────────────────────────────

test "nanoTimestamp follows sim_clock override" {
    defer sim_clock.setOverrideMs(null);
    sim_clock.setOverrideMs(2_000);
    try std.testing.expectEqual(@as(i128, 2_000) * 1_000_000, nanoTimestamp());
    sim_clock.advanceMs(5);
    try std.testing.expectEqual(@as(i128, 2_005) * 1_000_000, nanoTimestamp());
}

test "sleepRetry advances virtual clock" {
    defer sim_clock.setOverrideMs(null);
    sim_clock.setOverrideMs(0);
    sleepRetry(1); // 1s << 1 = 2s
    try std.testing.expectEqual(@as(i64, 2_000), sim_clock.milliNow());
}

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
    for (bar[1 .. mid + 1]) |c| {
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
    try std.testing.expect(!isSafeFilename("evil%2e%2e%2fpasswd.gguf"));
    try std.testing.expect(!isSafeFilename("model\".gguf"));
    try std.testing.expect(!isSafeFilename("model name.gguf"));
    try std.testing.expect(!isSafeFilename("model&x=1.gguf"));
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

// ── Rerun safety (416 resume handling) ──────────────────────────────────────

test "classifyRangeNotSatisfiable matching size is complete" {
    try std.testing.expectEqual(RangeNotSatisfiable.complete, classifyRangeNotSatisfiable(5000, 5000));
}

test "classifyRangeNotSatisfiable stale oversized local file is rejected" {
    // Second run after the repository replaced its file with a smaller one:
    // the stale local blob must not be accepted as fully downloaded.
    try std.testing.expectEqual(RangeNotSatisfiable.stale_local_file, classifyRangeNotSatisfiable(7000, 5000));
}

test "classifyRangeNotSatisfiable stale undersized local file is rejected" {
    try std.testing.expectEqual(RangeNotSatisfiable.stale_local_file, classifyRangeNotSatisfiable(3000, 5000));
}

test "classifyRangeNotSatisfiable unknown expected size keeps legacy behavior" {
    // API listing without size metadata cannot contradict the local file.
    try std.testing.expectEqual(RangeNotSatisfiable.complete, classifyRangeNotSatisfiable(7000, 0));
}

test "localSizeIsComplete matching size is complete" {
    try std.testing.expect(localSizeIsComplete(5000, 5000));
}

test "localSizeIsComplete truncated leftover is not complete" {
    // Second `agave pull` after a failed sidecar/shard download: the leftover
    // must not be skipped as "already downloaded".
    try std.testing.expect(!localSizeIsComplete(12, 5000));
    try std.testing.expect(!localSizeIsComplete(0, 5000));
}

test "localSizeIsComplete oversized leftover is not complete" {
    try std.testing.expect(!localSizeIsComplete(7000, 5000));
}

test "localSizeIsComplete unknown size treats non-empty as complete" {
    try std.testing.expect(localSizeIsComplete(1, 0));
    try std.testing.expect(localSizeIsComplete(7000, 0));
    try std.testing.expect(!localSizeIsComplete(0, 0));
}

test "advertisedSizeAgrees rejects listing mismatch when Content-Length is present" {
    try std.testing.expect(advertisedSizeAgrees(5000, 5000, true));
    try std.testing.expect(!advertisedSizeAgrees(3000, 5000, true));
    try std.testing.expect(!advertisedSizeAgrees(7000, 5000, true));
}

test "advertisedSizeAgrees cannot contradict when size or Content-Length is unknown" {
    try std.testing.expect(advertisedSizeAgrees(3000, 0, true));
    try std.testing.expect(advertisedSizeAgrees(3000, 5000, false));
    try std.testing.expect(advertisedSizeAgrees(0, 0, false));
}

test "siblingSize reads integer size and treats missing as unknown" {
    const allocator = std.testing.allocator;
    {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{\"size\":42}", .{});
        defer parsed.deinit();
        try std.testing.expectEqual(@as(u64, 42), siblingSize(parsed.value));
    }
    {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{\"rfilename\":\"x\"}", .{});
        defer parsed.deinit();
        try std.testing.expectEqual(@as(u64, 0), siblingSize(parsed.value));
    }
    {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{\"size\":-1}", .{});
        defer parsed.deinit();
        try std.testing.expectEqual(@as(u64, 0), siblingSize(parsed.value));
    }
}

test "isLocalBlobComplete rejects truncated sidecar so rerun re-downloads" {
    const io = std.testing.io;
    mod_io = io;
    var path_buf: [80]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "test_pull_sidecar_{d}.json", .{std.c.getpid()});
    defer Io.Dir.cwd().deleteFile(io, path) catch {};

    {
        var f = try Io.Dir.cwd().createFile(io, path, .{ .read = true });
        defer f.close(io);
        try f.writePositionalAll(io, "{", 0);
    }
    try std.testing.expect(!isLocalBlobComplete(path, 20));
    try std.testing.expect(isLocalBlobComplete(path, 1));
    try std.testing.expect(isLocalBlobComplete(path, 0));
    try std.testing.expect(!isLocalBlobComplete("test_pull_no_such_sidecar.json", 20));
}

// ── Rerun safety (corrupt blob removal) ─────────────────────────────────────

// A corrupt-at-full-size blob must not survive a failed integrity check:
// the next run's size-based "already downloaded" check would accept it and
// fail the same magic-byte test forever. Removal is what makes reruns of
// `agave pull` converge.
test "verifyGgufBlob removes corrupt blob so rerun re-downloads" {
    const io = std.testing.io;
    var path_buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "test_pull_{d}.gguf", .{std.c.getpid()});
    defer Io.Dir.cwd().deleteFile(io, path) catch {};

    // Corrupt content at full expected size: right size, wrong magic.
    {
        var f = try Io.Dir.cwd().createFile(io, path, .{ .read = true });
        defer f.close(io);
        try f.writePositionalAll(io, "NOPE-not-a-gguf", 0);
    }
    try std.testing.expect(!verifyGgufBlob(io, path));
    // The rerun must not find the stale blob: it was removed.
    if (Io.Dir.cwd().statFile(io, path, .{})) |_| return error.TestFailed else |_| {}

    // Convergence: a fresh valid download then passes the same check.
    const valid_blob = "GGUF" ++ "\x00\x00\x00";
    {
        var f = try Io.Dir.cwd().createFile(io, path, .{ .read = true });
        defer f.close(io);
        try f.writePositionalAll(io, valid_blob, 0);
    }
    try std.testing.expect(verifyGgufBlob(io, path));
    try std.testing.expect((try Io.Dir.cwd().statFile(io, path, .{})).size == valid_blob.len);
}

test "verifyGgufBlob accepts valid header and short-magic file is removed" {
    const io = std.testing.io;
    var path_buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "test_pull_short_{d}.gguf", .{std.c.getpid()});
    defer Io.Dir.cwd().deleteFile(io, path) catch {};

    // Fewer than 4 readable bytes: truncated header counts as corrupt.
    {
        var f = try Io.Dir.cwd().createFile(io, path, .{ .read = true });
        defer f.close(io);
        try f.writePositionalAll(io, "GG", 0);
    }
    try std.testing.expect(!verifyGgufBlob(io, path));
    if (Io.Dir.cwd().statFile(io, path, .{})) |_| return error.TestFailed else |_| {}
}

test "verifyGgufBlob missing file passes without acting" {
    // Cannot open for checking: nothing observed, pull proceeds with warning.
    try std.testing.expect(verifyGgufBlob(std.testing.io, "test_pull_no_such_blob.gguf"));
}

// ── selectModel tests ───────────────────────────────────────────────────────

test "selectModel prefers GGUF by quant preference" {
    const gguf_files = [_]GgufFile{
        .{ .filename = "model-Q8_0.gguf", .size = 1000 },
        .{ .filename = "model-Q4_K_M.gguf", .size = 500 },
    };
    const result = ListResult{
        .files = @constCast(&gguf_files),
        .safetensors = null,
        .commit_sha = "abc123",
        .arena = undefined,
    };
    const selected = try selectModel(&result, null);
    try std.testing.expect(selected == .gguf);
    try std.testing.expectEqualStrings("model-Q4_K_M.gguf", selected.gguf.filename);
}

test "selectModel falls back to safetensors when no GGUF" {
    const shards = [_][]const u8{"model.safetensors"};
    const shard_sizes = [_]u64{2000};
    const st = SafeTensorsModel{
        .shards = &shards,
        .shard_sizes = &shard_sizes,
        .total_size = 2000,
        .has_index = false,
        .has_config = true,
        .has_tokenizer = true,
        .has_tokenizer_config = false,
    };
    const result = ListResult{
        .files = &.{},
        .safetensors = st,
        .commit_sha = "abc123",
        .arena = undefined,
    };
    const selected = try selectModel(&result, null);
    try std.testing.expect(selected == .safetensors);
    try std.testing.expectEqual(@as(usize, 1), selected.safetensors.shards.len);
}

test "selectModel explicit safetensors quant" {
    const gguf_files = [_]GgufFile{
        .{ .filename = "model-Q4_K_M.gguf", .size = 500 },
    };
    const shards = [_][]const u8{ "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" };
    const shard_sizes = [_]u64{ 1000, 1000 };
    const st = SafeTensorsModel{
        .shards = &shards,
        .shard_sizes = &shard_sizes,
        .total_size = 2000,
        .has_index = true,
        .has_config = true,
        .has_tokenizer = true,
        .has_tokenizer_config = true,
    };
    const result = ListResult{
        .files = @constCast(&gguf_files),
        .safetensors = st,
        .commit_sha = "abc123",
        .arena = undefined,
    };
    // Explicit --quant safetensors selects SafeTensors even when GGUF is available.
    const selected = try selectModel(&result, "safetensors");
    try std.testing.expect(selected == .safetensors);
    try std.testing.expectEqual(@as(usize, 2), selected.safetensors.shards.len);
}

test "selectModel safetensors quant case insensitive" {
    const shards = [_][]const u8{"model.safetensors"};
    const shard_sizes = [_]u64{1000};
    const st = SafeTensorsModel{
        .shards = &shards,
        .shard_sizes = &shard_sizes,
        .total_size = 1000,
        .has_index = false,
        .has_config = true,
        .has_tokenizer = false,
        .has_tokenizer_config = false,
    };
    const result = ListResult{
        .files = &.{},
        .safetensors = st,
        .commit_sha = "abc123",
        .arena = undefined,
    };
    const selected = try selectModel(&result, "SafeTensors");
    try std.testing.expect(selected == .safetensors);
}

test "selectModel safetensors quant fails when no safetensors" {
    const gguf_files = [_]GgufFile{
        .{ .filename = "model-Q4_K_M.gguf", .size = 500 },
    };
    const result = ListResult{
        .files = @constCast(&gguf_files),
        .safetensors = null,
        .commit_sha = "abc123",
        .arena = undefined,
    };
    const selected = selectModel(&result, "safetensors");
    try std.testing.expectError(PullError.QuantNotFound, selected);
}

test "selectModel no files returns error" {
    const result = ListResult{
        .files = &.{},
        .safetensors = null,
        .commit_sha = "abc123",
        .arena = undefined,
    };
    const selected = selectModel(&result, null);
    try std.testing.expectError(PullError.NoGgufFiles, selected);
}

test "selectModel hasAnyFiles" {
    const empty = ListResult{
        .files = &.{},
        .safetensors = null,
        .commit_sha = "abc123",
        .arena = undefined,
    };
    try std.testing.expect(!empty.hasAnyFiles());

    const gguf_files = [_]GgufFile{
        .{ .filename = "model.gguf", .size = 100 },
    };
    const with_gguf = ListResult{
        .files = @constCast(&gguf_files),
        .safetensors = null,
        .commit_sha = "abc123",
        .arena = undefined,
    };
    try std.testing.expect(with_gguf.hasAnyFiles());

    const shards = [_][]const u8{"model.safetensors"};
    const shard_sizes = [_]u64{100};
    const with_st = ListResult{
        .files = &.{},
        .safetensors = .{
            .shards = &shards,
            .shard_sizes = &shard_sizes,
            .total_size = 100,
            .has_index = false,
            .has_config = false,
            .has_tokenizer = false,
            .has_tokenizer_config = false,
        },
        .commit_sha = "abc123",
        .arena = undefined,
    };
    try std.testing.expect(with_st.hasAnyFiles());
}

test "fuzz: pull helper functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            var buf: [256]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = buf[0];
            const input = buf[1..@min(@as(usize, len) + 1, buf.len)];

            _ = config.nonemptyEnv(if (input.len == 0) null else input);
            _ = config.envFlagIsOne(if (input.len == 0) null else input);

            // isSafeFilename: must not crash on any input.
            const safe = isSafeFilename(input);
            // If safe, only allowlisted characters and no "..".
            if (safe) {
                try std.testing.expect(std.mem.indexOf(u8, input, "..") == null);
                try std.testing.expect(input.len <= 255);
                for (input) |c| {
                    const ok = (c >= 'a' and c <= 'z') or
                        (c >= 'A' and c <= 'Z') or
                        (c >= '0' and c <= '9') or
                        c == '-' or c == '_' or c == '.';
                    try std.testing.expect(ok);
                }
            }

            // isValidRepoName: must not crash on any input.
            const valid_repo = isValidRepoName(input);
            // If valid, must not be empty or start/end with '/'.
            if (valid_repo) {
                try std.testing.expect(input.len > 0);
                try std.testing.expect(input[0] != '/');
                try std.testing.expect(input[input.len - 1] != '/');
            }

            // isValidHexSha: must not crash on any input.
            const valid_sha = isValidHexSha(input);
            // If valid, must only contain hex chars and be 1..64 length.
            if (valid_sha) {
                try std.testing.expect(input.len >= 1 and input.len <= 64);
                for (input) |c| try std.testing.expect(std.ascii.isHex(c));
            }

            // replaceSlashes: must produce output with no '/'.
            if (input.len <= 128) {
                const result = replaceSlashes(std.testing.allocator, input) catch return;
                defer std.testing.allocator.free(result);
                for (result) |c| try std.testing.expect(c != '/');
            }

            // selectFile: empty slice returns error.
            const empty: []const GgufFile = &.{};
            try std.testing.expectError(PullError.NoGgufFiles, selectFile(empty, null));

            // progressBar: any pct value must not crash.
            var pbar_buf: [progress_bar_width + 2]u8 = undefined;
            const pct = buf[1] % 101; // 0..100
            const bar = progressBar(&pbar_buf, pct);
            try std.testing.expect(bar.len > 0);
        }
    }.f, .{});
}

test "HF_ENDPOINT overrides the API base and rejects a non-URL" {
    try std.testing.expectEqualStrings(hf_api_base_default, try resolveHfApiBase(null));
    try std.testing.expectEqualStrings("https://hf-mirror.internal", try resolveHfApiBase("https://hf-mirror.internal"));
    try std.testing.expectEqualStrings("https://hf-mirror.internal", try resolveHfApiBase("https://hf-mirror.internal/"));
    try std.testing.expectEqualStrings("http://localhost:8080", try resolveHfApiBase("http://localhost:8080/"));
    try std.testing.expectError(PullError.InvalidArgument, resolveHfApiBase("hf-mirror.internal"));
    try std.testing.expectError(PullError.InvalidArgument, resolveHfApiBase("ftp://huggingface.co"));
    try std.testing.expectError(PullError.InvalidArgument, resolveHfApiBase("/"));
}

test "fuzz: HuggingFace API listing parser" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const allocator = std.testing.allocator;
            var buf: [512]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            var body = buf[0..len];
            // Owned only when the seeded listing below replaces `body`; freed
            // after the parse so the slices never outlive their storage.
            var body_owned: ?[]u8 = null;
            defer if (body_owned) |owned| allocator.free(owned);

            // Half the time, plant a well-formed listing so the parser is
            // driven past the JSON check into the siblings walk, the shard
            // collection, and the saturating size sum.
            if (smith.valueWithHash(u8, 2) & 1 == 0) {
                const n_gguf: usize = smith.indexWithHash(3, 3);
                const n_st: usize = smith.indexWithHash(3, 4);
                const bogus_size = smith.valueWithHash(u64, 5);
                var w = try std.Io.Writer.Allocating.initCapacity(allocator, 1024);
                defer w.deinit();
                w.writer.print("{{\"id\":\"o/r\",\"sha\":\"{s}\",\"siblings\":[{{\"rfilename\":\"config.json\",\"size\":{d}}},{{\"rfilename\":\"tokenizer.json\",\"size\":-5}},{{\"rfilename\":\"../escape.gguf\",\"size\":1}}", .{
                    if (smith.valueWithHash(u8, 6) & 1 == 0) "0123456789abcdef" else "NOTHEX",
                    bogus_size,
                }) catch return;
                for (0..n_gguf) |i| {
                    w.writer.print(
                        "{{\"rfilename\":\"model-Q4_K_M-{d}.gguf\",\"size\":{d}}}",
                        .{ i, @as(u64, 1) << @intCast(smith.valueWithHash(u8, 7) % 64) },
                    ) catch return;
                }
                for (0..n_st) |i| {
                    w.writer.print(
                        "{{\"rfilename\":\"model-{d}.safetensors\",\"size\":{d}}}",
                        .{ i, bogus_size },
                    ) catch return;
                }
                w.writer.writeAll("]}") catch return;
                body_owned = try w.toOwnedSlice();
                body = body_owned.?;
            }

            var result = parseModelListing(allocator, body) catch |err| switch (err) {
                // A body that is not JSON, or a listing with no model files,
                // is the only rejected shape.
                error.ApiResponseInvalid, error.NoGgufFiles => return,
                else => |e| return e,
            };
            defer result.deinit();

            // A successful parse always reports at least one downloadable file.
            try std.testing.expect(result.hasAnyFiles());

            // Every filename handed to the download path is a safe basename
            // with the extension it was selected for, and every size slice is
            // parallel to its filename slice.
            for (result.files) |gf| {
                try std.testing.expect(isSafeFilename(gf.filename));
                try std.testing.expect(std.mem.endsWith(u8, gf.filename, ".gguf"));
            }
            if (result.safetensors) |st| {
                try std.testing.expectEqual(st.shards.len, st.shard_sizes.len);
                for (st.shards, st.shard_sizes) |name, size| {
                    try std.testing.expect(isSafeFilename(name));
                    try std.testing.expect(std.mem.endsWith(u8, name, ".safetensors"));
                    _ = size;
                }
                // The advertised total never under-reports the sum of its own
                // shards: a sum that overflowed u64 saturates at maxInt.
                var sum: u64 = 0;
                for (st.shard_sizes) |size| {
                    sum = std.math.add(u64, sum, size) catch std.math.maxInt(u64);
                }
                try std.testing.expect(st.total_size >= @min(sum, std.math.maxInt(u64)));
                if (sum == std.math.maxInt(u64)) {
                    try std.testing.expectEqual(std.math.maxInt(u64), st.total_size);
                }
            }

            // The commit SHA is either the literal fallback or a hex string.
            if (!std.mem.eql(u8, result.commit_sha, "unknown")) {
                try std.testing.expect(isValidHexSha(result.commit_sha));
            }
        }
    }.f, .{});
}

test "fuzz: GGUF shard filename parse and rebuild" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const allocator = std.testing.allocator;
            var buf: [64]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            const name = buf[0..len];

            const total = detectGgufShardCount(name);
            // A non-zero count means the name matched `-NNNNN-of-MMMMM.gguf`
            // with a total of at least 2, and the count is recoverable.
            if (total != 0) {
                try std.testing.expect(total >= 2);
                const rebuilt1 = try buildShardFilename(allocator, name, 1, total);
                defer allocator.free(rebuilt1);
                // Rebuilding index 1 of N must round-trip to the same shard
                // count, which is what the download loop trusts.
                try std.testing.expectEqual(total, detectGgufShardCount(rebuilt1));

                // Every shard in 1..=total parses back to the same count and
                // ends at the same digit width, so no shard is skipped.
                for (2..total + 1) |idx| {
                    const rebuilt = buildShardFilename(allocator, name, @intCast(idx), total) catch continue;
                    defer allocator.free(rebuilt);
                    try std.testing.expectEqual(total, detectGgufShardCount(rebuilt));
                    try std.testing.expect(std.mem.endsWith(u8, rebuilt, ".gguf"));
                    try std.testing.expectEqual(rebuilt.len, rebuilt1.len);
                }
            }

            // buildShardFilename is only ever called with a name that already
            // parsed; anything else must be refused rather than truncated.
            if (!std.mem.endsWith(u8, name, ".gguf") or
                std.mem.indexOf(u8, name, "-of-") == null)
            {
                try std.testing.expectError(error.InvalidShardName, buildShardFilename(allocator, name, 1, 2));
            }
        }
    }.f, .{});
}
