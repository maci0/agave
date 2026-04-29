//! CLI entry point for the Agave LLM inference engine.
//! Parses command-line arguments, loads a GGUF model or SafeTensors directory,
//! auto-detects the architecture, and runs interactive generation, one-shot prompts,
//! or an HTTP server.

const std = @import("std");
const Io = std.Io;
const cli_mod = @import("cli.zig");
const backend_mod = @import("backend/backend.zig");
const format_mod = @import("format/format.zig");
const model_mod = @import("models/model.zig");
const spec_decode = @import("spec/spec_decode.zig");
const tok_mod = @import("tokenizer/tokenizer.zig");
const server = @import("server/server.zig");
const display_mod = @import("display.zig");
const Display = display_mod.Display;
const chat_tmpl_mod = @import("chat_template.zig");
const ChatTemplate = chat_tmpl_mod.ChatTemplate;
const Message = chat_tmpl_mod.Message;
const arch_mod = @import("arch.zig");
const Arch = arch_mod.Arch;
const TokenizerKind = tok_mod.TokenizerKind;
const Recipe = @import("recipe.zig").Recipe;

const Backend = backend_mod.Backend;
const BackendState = backend_mod.BackendState;
const BackendChoice = backend_mod.BackendChoice;
const ThreadPool = @import("thread_pool.zig").ThreadPool;
const Format = format_mod.Format;
const GGUFFile = format_mod.GGUFFile;
const SafeTensorsDir = format_mod.SafeTensorsDir;
const Model = model_mod.Model;
const ModelStorage = model_mod.ModelStorage;
const BpeTokenizer = tok_mod.BpeTokenizer;
const LineEditor = @import("readline.zig").LineEditor;
const KvQuantType = @import("ops/kv_quant.zig").KvQuantType;
const math_ops = @import("ops/math.zig");
const TieredKvCache = @import("kvcache/tiered.zig").TieredKvCache;
const pull = @import("pull.zig");
const image = @import("image.zig");

/// Standard I/O file handles via std.Io.File (Zig 0.16 idiom).
const stdout_file = Io.File.stdout();
const stderr_file = Io.File.stderr();
const stdin_file = Io.File.stdin();

/// Millisecond timestamp from the Io clock (Zig 0.16 idiom).
fn milliTimestamp(io: Io) i64 {
    return Io.Clock.real.now(io).toMilliseconds();
}

/// Nanosecond timestamp from the Io clock (Zig 0.16 idiom).
fn nanoTimestamp(io: Io) i96 {
    return Io.Clock.real.now(io).toNanoseconds();
}

/// Read all piped stdin into an allocated buffer.
fn readStdinAll(allocator: std.mem.Allocator, max_size: usize) ?[]const u8 {
    var buf = std.ArrayList(u8).empty;
    var read_buf: [4096]u8 = undefined;
    while (true) {
        const n = std.posix.read(stdin_file.handle, &read_buf) catch break;
        if (n == 0) break;
        if (buf.items.len + n > max_size) {
            eprint("Error: piped input exceeds 1 MB limit\n", .{});
            buf.deinit(allocator);
            std.process.exit(1);
        }
        buf.appendSlice(allocator, read_buf[0..n]) catch {
            buf.deinit(allocator);
            return null;
        };
    }
    if (buf.items.len == 0) {
        buf.deinit(allocator);
        return null;
    }
    return buf.toOwnedSlice(allocator) catch {
        buf.deinit(allocator);
        return null;
    };
}

const version = display_mod.version;

// ── Generation constants ────────────────────────────────────────

/// Buffer for formatting print/eprint output.
const print_buf_size: usize = 8192;
/// Maximum token IDs buffered during generation.
const gen_ids_buf_size: usize = 4096;
/// Number of consecutive identical tokens before halting generation.
const repeat_halt_threshold: u32 = 6;
/// Batch size for TTY streaming (smaller = more responsive).
const tty_batch_size: u32 = 4;
/// Batch size for piped/file output (larger = fewer decode+write calls).
const pipe_batch_size: u32 = 32;
/// Maximum bytes to read from piped stdin as a prompt.
const max_stdin_prompt_size: usize = 1024 * 1024;
/// Default HTTP server port.
const default_port: u16 = 49453;
/// Default maximum tokens to generate per request.
const default_max_tokens: u32 = 512;
/// Default KV cache context size when user/recipe doesn't specify.
/// 4096 balances memory usage with practical conversation length.
const default_ctx_size: u32 = 4096;
/// Default prefill chunk size (tokens per batch).
const default_chunk_size: u32 = 512;
/// Minimum prompt tokens before showing prefill progress indicator.
const prefill_progress_threshold: usize = 50;
/// Default free RAM estimate when platform detection is not implemented (16 GB).
const default_free_ram: usize = 16 * 1024 * 1024 * 1024;
/// Minimum pages between progress reports during model preloading.
const min_report_pages: usize = 256;
/// Default tiered KV cache SSD budget when unspecified (GB).
const default_ssd_budget_gb: usize = 10;
/// Bytes per GiB (2^30) for memory budget calculations.
const gib_bytes: usize = 1024 * 1024 * 1024;
/// Block size for tiered KV cache block allocation.
const tiered_kv_block_size: u16 = 16;
/// Number of KV tensors per position (key + value).
const kv_tensors_per_position: usize = 2;
/// Fraction of free RAM to allocate for KV cache (N=2 means 1/2 = 50%).
const ram_budget_divisor: usize = 2;
/// Buffer size for warmup progress bar formatting.
const warmup_buf_size: usize = 256;
/// Fallback n_layers for tiered KV cache sizing when metadata is missing.
const tiered_fallback_n_layers: u32 = 32;
/// Fallback n_embd for tiered KV cache sizing when metadata is missing.
const tiered_fallback_n_embd: u32 = 2048;
/// Fallback n_kv_heads for tiered KV cache sizing when metadata is missing.
const tiered_fallback_n_kv_heads: u32 = 8;
/// Fallback n_heads for tiered KV cache sizing when metadata is missing.
const tiered_fallback_n_heads: u32 = 32;
const max_eog_ids = arch_mod.max_eog_ids;
/// Valid KV cache quantization type names (shared across all --kv-type* validation).
const kv_valid_types = "f32, f16, q8_0/q8, int8/i8, fp8/fp8_e4m3, nvfp4/fp4, turbo2/tq2, turbo3/tq3, turbo4/tq4, turbo (preset: K=q8_0 V=turbo4)";

// ── Output control ──────────────────────────────────────────────

var g_color: bool = true;
var g_quiet: bool = false;
var g_tty: bool = true;
var g_debug: bool = false;
var g_verbose: bool = false;
/// Global Io instance, set once from std.process.Init in main().
var g_io: Io = undefined;
/// Global args, set once from std.process.Init in main().
var init_args: std.process.Args = undefined;
/// Global environment map, set once from std.process.Init in main().
var g_environ: *std.process.Environ.Map = undefined;

fn print(comptime fmt: []const u8, args: anytype) void {
    var buf: [print_buf_size]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, fmt, args) catch return;
    _ = std.c.write(stdout_file.handle, text.ptr, text.len);
}

fn eprint(comptime fmt: []const u8, args: anytype) void {
    var buf: [print_buf_size]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, fmt, args) catch return;
    _ = std.c.write(stderr_file.handle, text.ptr, text.len);
}

/// Debug output. Only printed when --debug is active.
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (!g_debug) return;
    eprint("[dbg] " ++ fmt ++ "\n", args);
}

/// Parse a KV quantization type from an optional per-component override and
/// a shared --kv-type fallback. Exits on unrecognized values.
fn kvTypeOrExit(s: []const u8, flag_name: []const u8) KvQuantType {
    return KvQuantType.fromString(s) orelse {
        eprint("Error: unknown {s} value '{s}'\n", .{ flag_name, s });
        eprint("  Valid options: " ++ kv_valid_types ++ "\n", .{});
        std.process.exit(1);
    };
}

/// Detect free system RAM in bytes.
/// Uses platform-specific detection (sysctl on macOS, /proc/meminfo on Linux).
/// Falls back to default_free_ram on unsupported platforms.
fn detectFreeRam() usize {
    const avail = backend_mod.detectAvailMem();
    return if (avail > 0) avail else default_free_ram;
}

// ── Preload (fault-in mmap'd pages) ─────────────────────────────

/// Touch every page of a mmap'd region to fault it into RAM.
/// This eliminates page-fault stalls during inference by paying the I/O
/// cost upfront during model load. Uses madvise(SEQUENTIAL) to hint
/// kernel readahead, then switches to RANDOM after pages are resident.
fn preloadRegion(data: []align(std.heap.page_size_min) const u8) void {
    const MADV = std.posix.MADV;
    // Hint the kernel to read ahead sequentially
    std.posix.madvise(@alignCast(@constCast(data.ptr)), data.len, MADV.SEQUENTIAL) catch {};

    // Touch one byte per page to force all pages into RAM
    const page_size = std.heap.page_size_min;
    var offset: usize = 0;
    while (offset < data.len) : (offset += page_size) {
        _ = @as(*const volatile u8, @ptrCast(&data[offset])).*;
    }

    // Switch to random access hint now that all pages are resident
    std.posix.madvise(@alignCast(@constCast(data.ptr)), data.len, MADV.RANDOM) catch {};
}

/// Progress bar width for warmup display.
const warmup_bar_width: u32 = 30;

/// Preload all mmap'd model data into RAM with progress bar.
fn preloadModel(gguf: ?*GGUFFile, st: ?*SafeTensorsDir, quiet: bool, tty: bool, total_bytes: usize) u64 {
    const start = milliTimestamp(g_io);
    if (quiet or (gguf == null and st == null)) {
        // Still preload, just don't show progress
        if (gguf) |g| preloadRegion(g.mapped_data);
        if (st) |s| for (s.shard_data) |shard| preloadRegion(shard.data);
        return elapsedMs(start);
    }

    const fsize = display_mod.formatSize(total_bytes);

    if (tty) {
        // TTY: progress bar with percentage
        var loaded: usize = 0;
        if (gguf) |g| {
            preloadRegionProgress(g.mapped_data, &loaded, total_bytes, fsize);
        }
        if (st) |s| {
            for (s.shard_data) |shard| {
                preloadRegionProgress(shard.data, &loaded, total_bytes, fsize);
            }
        }
        eprint("\r\x1b[K", .{}); // clear progress line
    } else {
        // Non-TTY: simple start/done message
        eprint("loading {d:.1} {s}...", .{ fsize.val, fsize.unit });
        if (gguf) |g| preloadRegion(g.mapped_data);
        if (st) |s| for (s.shard_data) |shard| preloadRegion(shard.data);
        eprint(" done ({d}ms)\n", .{elapsedMs(start)});
    }
    return elapsedMs(start);
}

/// Touch every page with progress reporting. Updates `loaded` bytes counter
/// and prints a progress bar to stderr at ~1% intervals (at least min_report_pages apart).
fn preloadRegionProgress(data: []align(std.heap.page_size_min) const u8, loaded: *usize, total_bytes: usize, fsize: display_mod.FormattedSize) void {
    const MADV = std.posix.MADV;
    std.posix.madvise(@alignCast(@constCast(data.ptr)), data.len, MADV.SEQUENTIAL) catch {};

    const page_size = std.heap.page_size_min;
    // Report progress every ~1% of total, but no more frequently than min_report_pages
    const report_interval = @max(total_bytes / 100, page_size * min_report_pages);
    var last_report: usize = loaded.*;
    var offset: usize = 0;
    while (offset < data.len) : (offset += page_size) {
        _ = @as(*const volatile u8, @ptrCast(&data[offset])).*;
        loaded.* += page_size;

        if (loaded.* - last_report >= report_interval or offset + page_size >= data.len) {
            last_report = loaded.*;
            const pct: u32 = if (total_bytes > 0) @intCast(@min(loaded.* * 100 / total_bytes, 100)) else 100;
            const filled: u32 = @intCast(@min(@as(u64, pct) * warmup_bar_width / 100, warmup_bar_width));
            var buf: [warmup_buf_size]u8 = undefined;
            var pos: usize = 0;
            const append = struct {
                fn f(b: []u8, p: *usize, s: []const u8) void {
                    const n = @min(s.len, b.len - p.*);
                    @memcpy(b[p.*..][0..n], s[0..n]);
                    p.* += n;
                }
            }.f;
            if (g_color) {
                append(&buf, &pos, "\r\x1b[2m\xe2\x96\x90"); // CR + dim + ▐
            } else {
                append(&buf, &pos, "\r\xe2\x96\x90"); // CR + ▐
            }
            for (0..warmup_bar_width) |i| {
                if (i < filled) {
                    append(&buf, &pos, "\xe2\x96\x88"); // █
                } else {
                    append(&buf, &pos, "\xe2\x96\x91"); // ░
                }
            }
            append(&buf, &pos, "\xe2\x96\x8c "); // ▌ + space
            const text = if (g_color)
                std.fmt.bufPrint(buf[pos..], "loading {d:.1} {s} ({d}%)\x1b[0m", .{ fsize.val, fsize.unit, pct }) catch ""
            else
                std.fmt.bufPrint(buf[pos..], "loading {d:.1} {s} ({d}%)", .{ fsize.val, fsize.unit, pct }) catch "";
            pos += text.len;
            _ = std.c.write(stderr_file.handle, buf[0..pos].ptr, pos);
        }
    }

    std.posix.madvise(@alignCast(@constCast(data.ptr)), data.len, MADV.RANDOM) catch {};
}

// ── REPL help (shared between --help and /help) ─────────────────

const repl_help =
    \\  /clear, /reset      Clear conversation and KV cache (stay in chat)
    \\  /context, /ctx      Show context window usage (tokens used / max)
    \\  /system <text>      Set system prompt (clears conversation)
    \\  /system             Show current system prompt
    \\  /stats              Toggle generation stats
    \\  /verbose            Toggle technical details
    \\  /debug              Toggle debug logging
    \\  /model              Show model info
    \\  /help               Show this help
    \\  /quit, /exit, /q    Exit interactive mode
    \\  Ctrl+C              Cancel input (double-tap to quit)
    \\  Ctrl+D              Quit (on empty line)
    \\  Ctrl+L              Clear screen
    \\  Ctrl+R              Reverse search history
    \\  Up/Down             Navigate history
    \\
;

// ── CLI definition ───────────────────────────────────────────────

const cli_specs = [_]cli_mod.ArgSpec{
    // General
    .{ .long = "help", .short = 'h', .help = "Show this help message and exit." },
    .{ .long = "version", .short = 'v', .help = "Print version and exit." },
    .{ .long = "quiet", .short = 'q', .help = "Suppress banner and stats (raw output only)." },
    .{ .long = "color", .kind = .option, .help = "Color mode: auto, always, never [default: auto]." },
    .{ .long = "no-color", .help = "Disable colored output (same as --color=never)." },
    // Generation
    .{ .long = "max-tokens", .short = 'n', .kind = .option, .help = "Maximum tokens to generate [default: 512]." },
    .{ .long = "temperature", .short = 't', .kind = .option, .help = "Sampling temperature, 0 = greedy [default: 0]." },
    .{ .long = "top-p", .kind = .option, .help = "Nucleus sampling threshold [default: 1.0]." },
    .{ .long = "top-k", .kind = .option, .help = "Top-k sampling, 0 = disabled [default: 0]." },
    .{ .long = "repeat-penalty", .kind = .option, .help = "Repetition penalty [default: 1.0]." },
    .{ .long = "seed", .kind = .option, .help = "Random seed for sampling [default: random]." },
    .{ .long = "system", .kind = .option, .help = "System prompt for chat formatting." },
    // Backend & model
    .{ .long = "backend", .kind = .option, .help = "Compute backend: auto, cpu, metal, vulkan, cuda, rocm, webgpu [default: auto]." },
    .{ .long = "ctx-size", .kind = .option, .help = "Context window size; 0 = full model context [default: min(model, 4096)]." },
    .{ .long = "allow-cpu-fallback", .help = "Allow GPU backends to fall back to CPU for unsupported ops." },
    .{ .long = "mmap", .help = "Use lazy mmap instead of eagerly paging weights into RAM." },
    .{ .long = "prefill-batch-size", .kind = .option, .help = "Prefill chunk size in tokens [default: 512]." },
    // KV cache
    .{ .long = "kv-type", .kind = .option, .help = "KV cache quantization [default: f16]." },
    .{ .long = "kv-type-k", .kind = .option, .help = "KV cache key quantization (overrides --kv-type for keys)." },
    .{ .long = "kv-type-v", .kind = .option, .help = "KV cache value quantization (overrides --kv-type for values)." },
    .{ .long = "cache-type-k", .kind = .option, .help = "Alias for --kv-type-k." },
    .{ .long = "cache-type-v", .kind = .option, .help = "Alias for --kv-type-v." },
    .{ .long = "kv-tiers", .kind = .option, .help = "Enable tiered KV cache: vram+ram, vram+ram+ssd [default: off]." },
    .{ .long = "kv-ram-budget", .kind = .option, .help = "RAM tier budget, integer GB, requires --kv-tiers [default: 50% of free RAM]." },
    .{ .long = "kv-ssd-path", .kind = .option, .help = "SSD tier file path, requires --kv-tiers with ssd." },
    .{ .long = "kv-ssd-budget", .kind = .option, .help = "SSD tier budget, integer GB, requires --kv-tiers with ssd [default: 10]." },
    .{ .long = "kv-eviction", .kind = .option, .help = "KV eviction policy: none, norm, tri [default: none]." },
    .{ .long = "kv-budget", .kind = .option, .help = "Max KV positions to keep during eviction [default: 80% of ctx-size]." },
    // Server
    .{ .long = "serve", .short = 's', .help = "Start HTTP server (OpenAI + Anthropic API)." },
    .{ .long = "port", .short = 'p', .kind = .option, .help = "Server port [default: 49453]." },
    .{ .long = "host", .kind = .option, .help = "Server bind address [default: 127.0.0.1]." },
    .{ .long = "api-key", .kind = .option, .help = "API key for server auth (or AGAVE_API_KEY env)." },
    // Multimodal
    .{ .long = "mmproj", .kind = .option, .help = "Path to vision projector GGUF (mmproj file)." },
    .{ .long = "image", .kind = .option, .help = "Path to image file for multimodal inference (PNG or PPM P6)." },
    // Speculative decoding
    .{ .long = "draft-model", .kind = .option, .help = "Path to draft model for speculative decoding." },
    .{ .long = "spec-tokens", .short = 'K', .kind = .option, .help = "Draft tokens per speculation round [default: 5]." },
    .{ .long = "tree-budget", .kind = .option, .help = "DDTree node budget [default: 64]." },
    .{ .long = "spec-mode", .kind = .option, .help = "Speculative mode: standard, ddtree, self [default: ddtree]." },
    .{ .long = "draft-layers", .kind = .option, .help = "Layers for self-speculative draft [default: auto]." },
    // Diagnostics
    .{ .long = "verbose", .short = 'V', .help = "Show technical details (params, load times, EOG)." },
    .{ .long = "debug", .short = 'd', .help = "Enable debug logging (token IDs, layer timing); implies --verbose." },
    .{ .long = "json", .help = "Output results as JSON (implies --quiet)." },
    .{ .long = "model-info", .help = "Print model metadata and exit (supports --json)." },
    .{ .long = "megakernel", .help = "Use fused megakernel (single GPU dispatch per token)." },
    .{ .long = "profile", .help = "Profile per-op timing (halves throughput)." },
};

const SpecMode = enum { none, standard, ddtree, self_spec };

const CliArgs = struct {
    model_path: []const u8,
    prompt: ?[]const u8,
    serve: bool,
    port: u16,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    repeat_penalty: f32,
    system_prompt: ?[]const u8,
    backend_choice: BackendChoice,
    ctx_size: u32,
    kv_type_k: KvQuantType,
    kv_type_v: KvQuantType,
    /// Number of boundary layers (first N + last N) that use f16 for V cache
    /// instead of the configured kv_type_v. Protects attention quality at
    /// layer boundaries where compression is most harmful. 0 = disabled.
    kv_boundary_v: u32 = 0,
    seed: u64,
    // Tiered KV cache CLI options
    kv_tiers: ?[]const u8 = null,
    kv_ram_budget: ?u32 = null,
    kv_ssd_path: ?[]const u8 = null,
    kv_ssd_budget: ?u32 = null,
    kv_eviction: bool = false,
    kv_budget: u32 = 0,
    host: [4]u8 = .{ 127, 0, 0, 1 },
    api_key: ?[]const u8 = null,
    allow_cpu_fallback: bool,
    debug: bool,
    json: bool,
    model_info: bool,
    profile: bool,
    use_mmap: bool,
    prefill_batch_size: u32,
    /// Path to vision projector GGUF (mmproj file) for multimodal inference.
    mmproj: ?[]const u8 = null,
    /// Path to image file (PNG or PPM P6) for multimodal inference.
    image: ?[]const u8 = null,
    /// Enable fused megakernel for single-dispatch forward pass.
    megakernel: bool = false,
    // Speculative decoding
    draft_model_path: ?[]const u8 = null,
    spec_tokens: u32 = 5,
    tree_budget: u32 = 64,
    spec_mode: SpecMode = .none,
    draft_layers: ?u32 = null,
    /// Tracks which CLI args the user explicitly set (so recipes don't override them).
    user_set: Recipe.Overrides = .{},
};

/// Check if the first positional arg is a subcommand (e.g. "pull").
/// Returns true if a subcommand was handled (caller should return).
fn checkSubcommand(allocator: std.mem.Allocator) bool {
    var args_iter = init_args.iterate();
    _ = args_iter.next(); // skip argv[0]

    const first = args_iter.next() orelse return false;
    if (std.mem.eql(u8, first, "pull")) {
        const exit_code = pull.run(allocator, init_args, g_io);
        if (exit_code != 0) std.process.exit(exit_code);
        return true;
    }
    if (std.mem.eql(u8, first, "calibrate")) {
        const calibrate = @import("calibrate.zig");
        const exit_code = calibrate.run(allocator, g_io, init_args);
        if (exit_code != 0) std.process.exit(exit_code);
        return true;
    }
    // Support `agave help [subcommand]` (git convention)
    if (std.mem.eql(u8, first, "help")) {
        const sub = args_iter.next() orelse {
            printUsage();
            return true;
        };
        if (std.mem.eql(u8, sub, "pull")) {
            pull.printUsage();
            return true;
        }
        if (std.mem.eql(u8, sub, "calibrate")) {
            const calibrate = @import("calibrate.zig");
            calibrate.printUsage();
            return true;
        }
        // Handle flags and self-referential "help help" gracefully
        if (std.mem.eql(u8, sub, "help") or
            std.mem.eql(u8, sub, "--help") or
            std.mem.eql(u8, sub, "-h"))
        {
            printUsage();
            return true;
        }
        if (std.mem.eql(u8, sub, "--version") or std.mem.eql(u8, sub, "-v")) {
            display_mod.printVersion();
            return true;
        }
        eprint("Error: no help available for '{s}'\n", .{sub});
        eprint("Available help topics: pull, calibrate\n", .{});
        eprint("Run 'agave --help' for more information.\n", .{});
        std.process.exit(1);
    }
    return false;
}

fn parseCli(allocator: std.mem.Allocator) ?CliArgs {
    var res = cli_mod.parse(allocator, init_args, &cli_specs);

    if (res.flag("help")) {
        printUsage();
        res.deinit();
        return null;
    }

    if (res.flag("version")) {
        display_mod.printVersion();
        res.deinit();
        return null;
    }

    // Auto-detect TTY: disable color when stdout is not a terminal
    g_tty = stdout_file.isTty(g_io) catch false;
    g_color = blk: {
        // --color=always|never|auto takes precedence
        if (res.option("color")) |cm| {
            if (std.mem.eql(u8, cm, "always")) break :blk true;
            if (std.mem.eql(u8, cm, "never")) break :blk false;
            if (!std.mem.eql(u8, cm, "auto")) {
                eprint("Error: unknown --color value '{s}'\n", .{cm});
                eprint("  Valid options: auto, always, never\n", .{});
                std.process.exit(1);
            }
        }
        // --no-color flag
        if (res.flag("no-color")) break :blk false;
        // NO_COLOR env var (https://no-color.org)
        if (g_environ.get("NO_COLOR") != null) break :blk false;
        // Auto: color only on TTY
        break :blk g_tty;
    };
    g_quiet = res.flag("quiet");
    g_debug = res.flag("debug");
    g_verbose = res.flag("verbose") or g_debug;

    const json_mode = res.flag("json");
    if (json_mode) {
        g_quiet = true;
    }

    const n_positionals = res.positionals.items.len;
    if (n_positionals == 0) {
        eprint("Error: missing model path\n", .{});
        eprint("Usage: agave <model.gguf|model-dir/> [prompt]\n", .{});
        eprint("Run 'agave --help' for more information.\n", .{});
        std.process.exit(1);
    }

    const backend_choice: BackendChoice = blk: {
        const be_str = res.option("backend") orelse "auto";
        break :blk std.meta.stringToEnum(BackendChoice, be_str) orelse {
            eprint("Error: unknown backend '{s}'\n", .{be_str});
            eprint("  Valid options: auto, cpu, metal, vulkan, cuda, rocm, webgpu\n", .{});
            std.process.exit(1);
        };
    };

    const temperature = parseF32(res.option("temperature"), "temperature") orelse 0.0;
    const top_p = parseF32(res.option("top-p"), "top-p") orelse 1.0;
    const repeat_penalty = parseF32(res.option("repeat-penalty"), "repeat-penalty") orelse 1.0;

    // Validate sampling parameter ranges
    if (temperature < 0) {
        eprint("Error: --temperature must be >= 0 (got {d:.2})\n", .{temperature});
        std.process.exit(1);
    }
    if (top_p <= 0 or top_p > 1.0) {
        eprint("Error: --top-p must be in (0, 1.0] (got {d:.2})\n", .{top_p});
        std.process.exit(1);
    }
    if (repeat_penalty <= 0) {
        eprint("Error: --repeat-penalty must be > 0 (got {d:.2})\n", .{repeat_penalty});
        std.process.exit(1);
    }

    // Validate --kv-tiers value (mutable copy needed for "off" → null conversion)
    var kv_tiers_val = res.option("kv-tiers");
    if (kv_tiers_val) |tiers_str| {
        if (std.mem.eql(u8, tiers_str, "off")) {
            // "off" is the documented default — treat as if the flag was not passed
            kv_tiers_val = null;
        } else if (!std.mem.eql(u8, tiers_str, "vram+ram") and !std.mem.eql(u8, tiers_str, "vram+ram+ssd")) {
            eprint("Error: unknown --kv-tiers value '{s}'\n", .{tiers_str});
            eprint("  Valid options: off, vram+ram, vram+ram+ssd\n", .{});
            std.process.exit(1);
        }
    }

    // Warn about KV tier flags that have no effect without --kv-tiers
    if (kv_tiers_val == null) {
        if (res.option("kv-ram-budget") != null)
            eprint("Warning: --kv-ram-budget has no effect without --kv-tiers\n", .{});
        if (res.option("kv-ssd-budget") != null)
            eprint("Warning: --kv-ssd-budget has no effect without --kv-tiers\n", .{});
        if (res.option("kv-ssd-path") != null)
            eprint("Warning: --kv-ssd-path has no effect without --kv-tiers\n", .{});
    } else if (kv_tiers_val) |tiers_str| {
        // Warn about SSD flags when --kv-tiers doesn't include ssd
        if (std.mem.indexOf(u8, tiers_str, "ssd") == null) {
            if (res.option("kv-ssd-path") != null)
                eprint("Warning: --kv-ssd-path has no effect without ssd in --kv-tiers\n", .{});
            if (res.option("kv-ssd-budget") != null)
                eprint("Warning: --kv-ssd-budget has no effect without ssd in --kv-tiers\n", .{});
        }
    }

    // Validate --kv-eviction value
    if (res.option("kv-eviction")) |ev_str| {
        if (!std.mem.eql(u8, ev_str, "none") and !std.mem.eql(u8, ev_str, "norm") and !std.mem.eql(u8, ev_str, "tri")) {
            eprint("Error: unknown --kv-eviction value '{s}'\n", .{ev_str});
            eprint("  Valid options: none, norm, tri\n", .{});
            std.process.exit(1);
        }
    }

    // Warn about --kv-budget without --kv-eviction
    if (res.option("kv-budget") != null) {
        const has_eviction = if (res.option("kv-eviction")) |e| (!std.mem.eql(u8, e, "none")) else false;
        if (!has_eviction)
            eprint("Warning: --kv-budget has no effect without --kv-eviction\n", .{});
    }

    // Warn about --kv-type having no effect when both per-component types are set
    if (res.option("kv-type") != null) {
        const has_k = res.option("kv-type-k") != null or res.option("cache-type-k") != null;
        const has_v = res.option("kv-type-v") != null or res.option("cache-type-v") != null;
        if (has_k and has_v)
            eprint("Warning: --kv-type has no effect when both --kv-type-k and --kv-type-v are set\n", .{});
    }

    // Validate max-tokens
    if (res.optionU32("max-tokens")) |mt| {
        if (mt == 0) {
            eprint("Error: --max-tokens must be >= 1\n", .{});
            std.process.exit(1);
        }
    }

    // Validate prefill batch size
    if (res.optionU32("prefill-batch-size")) |pbs| {
        if (pbs == 0) {
            eprint("Error: --prefill-batch-size must be >= 1\n", .{});
            std.process.exit(1);
        }
    }

    // Warn about extra positional arguments (e.g. unquoted multi-word prompt)
    if (n_positionals > 2) {
        eprint("Warning: extra arguments after prompt ignored (did you forget to quote it?)\n", .{});
        eprint("  Usage: agave model.gguf \"multi word prompt\"\n", .{});
    }

    // Warn about server-only flags that have no effect without --serve
    if (!res.flag("serve")) {
        if (res.option("port") != null)
            eprint("Warning: --port has no effect without --serve\n", .{});
        if (res.option("host") != null)
            eprint("Warning: --host has no effect without --serve\n", .{});
        if (res.option("api-key") != null)
            eprint("Warning: --api-key has no effect without --serve\n", .{});
    } else {
        // Warn about flags ignored in server mode (early, before model loading)
        if (n_positionals > 1)
            eprint("Warning: prompt ignored in server mode (--serve)\n", .{});
        if (res.option("system") != null)
            eprint("Warning: --system ignored in server mode (system prompt comes from API request)\n", .{});
        if (res.option("image") != null)
            eprint("Warning: --image ignored in server mode (images come from API request)\n", .{});
    }

    // Warn about --allow-cpu-fallback with CPU backend (already on CPU, nothing to fall back to)
    if (res.flag("allow-cpu-fallback") and backend_choice == .cpu)
        eprint("Warning: --allow-cpu-fallback has no effect with --backend cpu\n", .{});

    // JSON mode + interactive REPL would corrupt the JSON output stream
    if (json_mode and !res.flag("model-info") and !res.flag("serve") and n_positionals < 2) {
        if ((stdin_file.isTty(g_io) catch false)) {
            eprint("Error: --json requires a prompt or --model-info\n", .{});
            eprint("  Usage: agave model.gguf --json \"prompt\"\n", .{});
            eprint("  Or: echo \"prompt\" | agave model.gguf --json\n", .{});
            std.process.exit(1);
        }
    }

    return .{
        .model_path = res.positional(0).?,
        .prompt = res.positional(1),
        .serve = res.flag("serve"),
        .port = res.optionU16("port") orelse default_port,
        .max_tokens = res.optionU32("max-tokens") orelse default_max_tokens,
        .temperature = temperature,
        .top_p = top_p,
        .top_k = res.optionU32("top-k") orelse 0,
        .repeat_penalty = repeat_penalty,
        .system_prompt = res.option("system"),
        .backend_choice = backend_choice,
        .ctx_size = res.optionU32("ctx-size") orelse 0,
        .seed = res.optionU64("seed") orelse @as(u64, @truncate(@as(u96, @bitCast(nanoTimestamp(g_io))))),
        .kv_type_k = blk: {
            if (res.option("kv-type-k")) |s| break :blk kvTypeOrExit(s, "--kv-type-k");
            if (res.option("cache-type-k")) |s| break :blk kvTypeOrExit(s, "--cache-type-k");
            const kv_str = res.option("kv-type") orelse break :blk KvQuantType.f16;
            // "turbo" preset: asymmetric K=q8_0 V=turbo4 (K precision protects attention routing)
            if (std.mem.eql(u8, kv_str, "turbo")) break :blk KvQuantType.q8_0;
            break :blk kvTypeOrExit(kv_str, "--kv-type");
        },
        .kv_type_v = blk: {
            if (res.option("kv-type-v")) |s| break :blk kvTypeOrExit(s, "--kv-type-v");
            if (res.option("cache-type-v")) |s| break :blk kvTypeOrExit(s, "--cache-type-v");
            const kv_str = res.option("kv-type") orelse break :blk KvQuantType.f16;
            // "turbo" preset: asymmetric K=q8_0 V=turbo4 (V compression is nearly free)
            if (std.mem.eql(u8, kv_str, "turbo")) break :blk KvQuantType.turbo4;
            break :blk kvTypeOrExit(kv_str, "--kv-type");
        },
        // Turbo preset enables boundary V protection (first/last 2 layers at f16-V)
        .kv_boundary_v = if (res.option("kv-type")) |kv| (if (std.mem.eql(u8, kv, "turbo")) @as(u32, 2) else 0) else 0,
        .kv_tiers = kv_tiers_val,
        .kv_ram_budget = res.optionU32("kv-ram-budget"),
        .kv_ssd_path = res.option("kv-ssd-path"),
        .kv_ssd_budget = res.optionU32("kv-ssd-budget"),
        .kv_eviction = if (res.option("kv-eviction")) |e| (!std.mem.eql(u8, e, "none")) else false,
        .kv_budget = res.optionU32("kv-budget") orelse 0,
        .host = blk: {
            const host_str = res.option("host") orelse break :blk [4]u8{ 127, 0, 0, 1 };
            if (std.mem.eql(u8, host_str, "0.0.0.0")) break :blk [4]u8{ 0, 0, 0, 0 };
            if (std.mem.eql(u8, host_str, "127.0.0.1") or std.mem.eql(u8, host_str, "localhost")) break :blk [4]u8{ 127, 0, 0, 1 };
            // Parse dotted-quad IPv4
            var parts: [4]u8 = .{ 0, 0, 0, 0 };
            var iter = std.mem.splitScalar(u8, host_str, '.');
            var pi: usize = 0;
            while (iter.next()) |part| {
                if (pi >= 4) {
                    eprint("Error: invalid host address '{s}' (expected IPv4 address or 'localhost')\n", .{host_str});
                    std.process.exit(1);
                }
                parts[pi] = std.fmt.parseInt(u8, part, 10) catch {
                    eprint("Error: invalid host address '{s}' (expected IPv4 address or 'localhost')\n", .{host_str});
                    std.process.exit(1);
                };
                pi += 1;
            }
            if (pi != 4) {
                eprint("Error: invalid host address '{s}' (expected IPv4 address or 'localhost')\n", .{host_str});
                std.process.exit(1);
            }
            break :blk parts;
        },
        .api_key = res.option("api-key") orelse g_environ.get("AGAVE_API_KEY"),
        .allow_cpu_fallback = res.flag("allow-cpu-fallback"),
        .debug = res.flag("debug"),
        .json = json_mode,
        .model_info = res.flag("model-info"),
        .profile = res.flag("profile"),
        .megakernel = res.flag("megakernel"),
        .use_mmap = res.flag("mmap"),
        .prefill_batch_size = res.optionU32("prefill-batch-size") orelse default_chunk_size,
        .mmproj = res.option("mmproj"),
        .image = res.option("image"),
        .draft_model_path = res.option("draft-model"),
        .spec_tokens = res.optionU32("spec-tokens") orelse 5,
        .tree_budget = res.optionU32("tree-budget") orelse 64,
        .spec_mode = blk: {
            const dm = res.option("draft-model");
            const sm = res.option("spec-mode");
            if (sm) |s| {
                if (std.mem.eql(u8, s, "standard")) break :blk SpecMode.standard;
                if (std.mem.eql(u8, s, "ddtree")) break :blk SpecMode.ddtree;
                if (std.mem.eql(u8, s, "self")) break :blk SpecMode.self_spec;
                eprint("Error: unknown --spec-mode '{s}' (expected: standard, ddtree, self)\n", .{s});
                std.process.exit(1);
            }
            break :blk if (dm != null) SpecMode.ddtree else SpecMode.none;
        },
        .draft_layers = res.optionU32("draft-layers"),
        .user_set = .{
            .temperature = res.option("temperature") != null,
            .top_p = res.option("top-p") != null,
            .top_k = res.option("top-k") != null,
            .repeat_penalty = res.option("repeat-penalty") != null,
            .max_tokens = res.option("max-tokens") != null,
            .ctx_size = res.option("ctx-size") != null,
        },
    };
}

fn parseF32(s: ?[]const u8, comptime flag: []const u8) ?f32 {
    const str = s orelse return null;
    const val = std.fmt.parseFloat(f32, str) catch {
        eprint("Error: invalid value for --" ++ flag ++ ": '{s}' is not a valid number\n", .{str});
        std.process.exit(1);
    };
    if (!std.math.isFinite(val)) {
        eprint("Error: --" ++ flag ++ " must be a finite number, got '{s}'\n", .{str});
        std.process.exit(1);
    }
    return val;
}

fn printUsage() void {
    const usage =
        \\agave — Zig LLM inference engine
        \\
        \\USAGE:
        \\  agave [OPTIONS] <model.gguf|model-dir/> [prompt]
        \\  echo "prompt" | agave model.gguf
        \\
        \\ARGUMENTS:
        \\  <model.gguf|model-dir/>  Path to GGUF model file or SafeTensors directory
        \\  [prompt]                 Text prompt (omit for interactive REPL)
        \\
        \\GENERAL:
        \\  -h, --help             Show this help message
        \\  -v, --version          Print version
        \\  -q, --quiet            Suppress banner and stats (raw output only)
        \\      --color <MODE>     Color mode: auto, always, never [default: auto]
        \\      --no-color         Disable colored output (same as --color=never, respects NO_COLOR env)
        \\
        \\GENERATION:
        \\  -n, --max-tokens <N>      Maximum tokens to generate [default: 512]
        \\  -t, --temperature <T>     Sampling temperature, 0 = greedy [default: 0]
        \\      --top-p <P>           Nucleus sampling threshold [default: 1.0]
        \\      --top-k <K>           Top-k sampling, 0 = disabled [default: 0]
        \\      --repeat-penalty <R>  Repetition penalty [default: 1.0]
        \\      --seed <N>            Random seed for sampling [default: random]
        \\      --system <TEXT>       System prompt for chat formatting
        \\
        \\BACKEND & MODEL:
        \\      --backend <BE>        Compute backend: auto, cpu, metal, vulkan, cuda, rocm, webgpu [default: auto]
        \\      --ctx-size <N>        Context window size; 0 = full model context [default: min(model, 4096)]
        \\      --allow-cpu-fallback  Allow GPU backends to fall back to CPU for unsupported ops
        \\      --mmap                Use lazy mmap instead of eagerly paging weights into RAM
        \\      --prefill-batch-size <N>  Prefill chunk size in tokens [default: 512]
        \\
        \\KV CACHE:
        \\      --kv-type <TYPE>      KV cache quantization [default: f16]
        \\                            Types: f32, f16, q8_0/q8, int8/i8, fp8/fp8_e4m3, nvfp4/fp4,
        \\                                   turbo2/tq2, turbo3/tq3, turbo4/tq4
        \\      --kv-type-k <TYPE>    KV key quantization (overrides --kv-type, alias: --cache-type-k)
        \\      --kv-type-v <TYPE>    KV value quantization (overrides --kv-type, alias: --cache-type-v)
        \\      --kv-tiers <TIERS>    Tiered KV cache: vram+ram, vram+ram+ssd [default: off]
        \\      --kv-ram-budget <GB>  RAM tier budget, integer GB (requires --kv-tiers) [default: 50% of free RAM]
        \\      --kv-ssd-path <PATH>  SSD tier file path (requires --kv-tiers with ssd)
        \\      --kv-ssd-budget <GB>  SSD tier budget, integer GB (requires --kv-tiers with ssd) [default: 10]
        \\      --kv-eviction <POL>   KV eviction policy: none, norm [default: none]
        \\      --kv-budget <N>       Max KV positions to keep during eviction [default: 80% of ctx-size]
        \\
        \\SERVER:
        \\  -s, --serve            Start HTTP server (OpenAI + Anthropic API)
        \\  -p, --port <PORT>      Server port [default: 49453]
        \\      --host <ADDR>      Bind address: IPv4, localhost, or 0.0.0.0 [default: 127.0.0.1]
        \\      --api-key <KEY>    API key for server auth (or AGAVE_API_KEY env)
        \\
        \\MULTIMODAL:
        \\      --mmproj <PATH>    Path to vision projector GGUF (mmproj file)
        \\      --image <PATH>     Path to image file (PNG or PPM P6)
        \\
        \\DIAGNOSTICS:
        \\  -V, --verbose          Show technical details (params, load times, EOG)
        \\  -d, --debug            Enable debug logging (token IDs, layer timing); implies --verbose
        \\      --json             Output results as JSON (implies --quiet)
        \\      --model-info       Print model metadata and exit (supports --json)
        \\      --profile          Profile per-op timing (halves throughput)
        \\
        \\ENVIRONMENT:
        \\  NO_COLOR             Disable colored output when set (https://no-color.org)
        \\  AGAVE_API_KEY        API key for server auth (alternative to --api-key)
        \\  HF_TOKEN             HuggingFace API token for private repos (used by pull)
        \\
        \\EXAMPLES:
        \\  agave model.gguf                          Interactive REPL
        \\  agave model.gguf "What is 2+2?"           Single prompt
        \\  agave model.gguf -q "Hello" > out.txt     Pipe output (no banner)
        \\  agave model.gguf --serve --port 3000      HTTP server
        \\  agave model.gguf -t 0.7 --top-p 0.9 "Tell me a joke"
        \\  agave model.gguf --backend cpu "Hello"    Force CPU backend
        \\  agave ./glm-4-9b/ "Hello"                 Load SafeTensors directory
        \\  echo "Explain TCP" | agave model.gguf     Pipe prompt from stdin
        \\  agave model.gguf --json "Hello"           JSON output with stats
        \\  agave model.gguf --json --model-info      Model metadata as JSON
        \\  agave model.gguf --kv-type tq4 "Hello"   TurboQuant KV cache (saves VRAM)
        \\  agave model.gguf --ctx-size 0 "Hello"    Use full model context window
        \\  agave model.gguf --image pic.png "What's this?"  Vision (auto-detects mmproj)
        \\
        \\SUBCOMMANDS:
        \\  agave pull <org/repo>                    Download GGUF model from HuggingFace
        \\  agave pull <org/repo> --quant Q4_K_M     Download specific quantization
        \\  agave pull <org/repo> --list             List available GGUF files
        \\  agave calibrate <model.gguf>             Generate TriAttention calibration data
        \\  agave help <topic>                       Show help for a subcommand (e.g. pull, calibrate)
        \\
        \\SUPPORTED ARCHITECTURES:
        \\  gemma3, gemma4, qwen35, gpt-oss, nemotron-h, nemotron-nano, glm4
        \\
        \\REPL COMMANDS:
    ++ repl_help;
    _ = std.c.write(stdout_file.handle, usage.ptr, usage.len);
}

// ── Formatting helpers ───────────────────────────────────────────

fn elapsedMs(start: i64) u64 {
    return @intCast(@max(milliTimestamp(g_io) - start, 0));
}

const EogTokens = struct { ids: [max_eog_ids]u32, len: usize };

/// Collect additional EOS/EOG token IDs from GGUF metadata.
fn getEogTokens(fmt_iface: Format, primary_eos: u32) EogTokens {
    var result: EogTokens = .{ .ids = undefined, .len = 0 };
    result.ids[0] = primary_eos;
    result.len = 1;
    // Check for EOG token arrays or single-value EOG IDs from GGUF metadata
    const array_keys = [_][]const u8{
        "tokenizer.ggml.eog_token_id",
        "tokenizer.ggml.eot_token_id",
    };
    for (array_keys) |key| {
        if (fmt_iface.getMetaU32Array(key)) |ids| {
            for (ids) |id| {
                if (id != primary_eos and result.len < result.ids.len) {
                    result.ids[result.len] = id;
                    result.len += 1;
                }
            }
        } else if (fmt_iface.getMetaU32(key)) |id| {
            if (id != primary_eos and result.len < result.ids.len) {
                result.ids[result.len] = id;
                result.len += 1;
            }
        }
    }
    return result;
}

fn isEogToken(token: u32, eog: anytype) bool {
    for (eog.ids[0..eog.len]) |id| {
        if (token == id) return true;
    }
    return false;
}

// ── Main ─────────────────────────────────────────────────────────

pub fn main(init: std.process.Init) !void {
    g_io = init.io;
    init_args = init.minimal.args;
    g_environ = init.environ_map;
    const allocator = init.gpa;

    // Check for subcommands before CLI parsing
    if (checkSubcommand(allocator)) return;

    var cli = parseCli(allocator) orelse return;

    // ── Load model format ────────────────────────────────────────
    const load_start = milliTimestamp(g_io);

    // Detect format: directory → SafeTensors, else → GGUF
    const is_dir = blk: {
        const dir = Io.Dir.cwd().openDir(g_io, cli.model_path, .{}) catch break :blk false;
        dir.close(g_io);
        break :blk true;
    };

    var gguf_file: ?GGUFFile = null;
    var st_dir: ?SafeTensorsDir = null;
    defer {
        if (gguf_file) |*g| g.deinit();
        if (st_dir) |*s| s.deinit();
    }

    var fmt: Format = undefined;
    if (is_dir) {
        st_dir = SafeTensorsDir.open(allocator, cli.model_path) catch |e| {
            eprint("Error: failed to open safetensors dir '{s}': {}\n", .{ cli.model_path, e });
            if (e == error.FileNotFound or e == error.NotDir)
                eprint("  Directory does not exist or is not a valid SafeTensors directory.\n", .{})
            else if (e == error.OutOfMemory)
                eprint("  Not enough memory to load model metadata.\n", .{});
            std.process.exit(1);
        };
        fmt = st_dir.?.format();
    } else {
        gguf_file = GGUFFile.open(allocator, cli.model_path) catch |e| {
            eprint("Error: failed to open '{s}': {}\n", .{ cli.model_path, e });
            if (e == error.FileNotFound)
                eprint("  File does not exist. Check the path and try again.\n", .{})
            else if (e == error.InvalidMagic)
                eprint("  Not a valid GGUF file. Expected GGUF magic bytes.\n", .{})
            else if (e == error.UnsupportedVersion)
                eprint("  GGUF version not supported. Agave supports v2 and v3.\n", .{})
            else if (e == error.FileTooSmall)
                eprint("  File is too small to be a valid GGUF model.\n", .{});
            std.process.exit(1);
        };
        fmt = gguf_file.?.format();
    }
    const load_ms = elapsedMs(load_start);

    const arch_str = fmt.getMetaStr("general.architecture") orelse
        fmt.getMetaStr("model_type") orelse "unknown";
    const name = fmt.getMetaStr("general.name") orelse
        fmt.getMetaStr("model_type") orelse "agave";
    const quant = Format.getQuantName(fmt);

    var arch = Arch.detect(arch_str) orelse {
        eprint("Error: unsupported architecture '{s}'\n", .{arch_str});
        eprint("  Supported: gemma3, gemma4, qwen35, gpt-oss, nemotron-h, nemotron-nano, glm4\n", .{});
        std.process.exit(1);
    };

    // SafeTensors Nemotron Nano variant: detected by backbone.embeddings.weight tensor
    if (arch == .nemotron_h and fmt.getTensor("backbone.embeddings.weight") != null) {
        arch = .nemotron_nano;
    }

    if (!arch.isEnabled()) {
        eprint("Error: {s} model support disabled at compile time\n", .{arch.displayName()});
        eprint("  Rebuild with -Denable-{s}=true to enable.\n", .{arch.buildFlag()});
        std.process.exit(1);
    }

    // ── Backend selection ─────────────────────────────────────────
    var bs = BackendState{};
    bs.init(allocator, cli.backend_choice, g_io);
    defer if (bs.pool) |*p| p.deinit();
    const be = bs.be;
    const be_name = bs.name;

    // ── Display setup ──────────────────────────────────────────────
    const output_mode: display_mod.OutputMode = if (cli.json)
        .json
    else if (g_tty and g_color)
        .tty
    else
        .plain;
    var display = Display.init(output_mode, g_verbose);

    // ── Compute file size (needed for banner and progress) ────────
    const file_size_bytes: usize = if (gguf_file) |g| g.file_size else if (st_dir) |s| s.totalBytes() else 0;

    // ── Banner (printed before loading so user sees info immediately) ─
    const meta_n_embed = fmt.getArchU32(arch_str, "embedding_length") orelse fmt.getMetaU32("hidden_size") orelse 0;
    const meta_n_heads = fmt.getArchU32(arch_str, "attention.head_count") orelse fmt.getMetaU32("num_attention_heads") orelse 0;
    var disp_info = display_mod.ModelInfo{
        .name = name,
        .arch_name = arch.displayName(),
        .quant = quant,
        .be_name = be_name,
        .n_layers = fmt.getArchU32(arch_str, "block_count") orelse fmt.getMetaU32("num_hidden_layers") orelse 0,
        .n_embed = meta_n_embed,
        .n_heads = meta_n_heads,
        .n_kv_heads = fmt.getArchU32(arch_str, "attention.head_count_kv") orelse
            fmt.getArchArrayFirstU32(arch_str, "attention.head_count_kv") orelse
            fmt.getArchU32(arch_str, "attention.head_count_kv_global") orelse
            fmt.getMetaU32("num_key_value_heads") orelse 0,
        .head_dim = fmt.getArchU32(arch_str, "attention.key_length") orelse
            fmt.getMetaU32("head_dim") orelse
            if (meta_n_embed > 0 and meta_n_heads > 0) meta_n_embed / meta_n_heads else 0,
        .ff_dim = fmt.getArchU32(arch_str, "feed_forward_length") orelse
            fmt.getArchArrayFirstU32(arch_str, "feed_forward_length") orelse
            fmt.getMetaU32("intermediate_size") orelse 0,
        .vocab_size = if (fmt.getVocab()) |v| @intCast(v.len) else fmt.getArchU32(arch_str, "vocab_size") orelse 0,
        .ctx_size = fmt.getArchU32(arch_str, "context_length") orelse fmt.getMetaU32("max_position_embeddings") orelse 0,
        .rope_theta = fmt.getArchF32(arch_str, "rope.freq_base") orelse fmt.getMetaF32("rope_theta") orelse 0,
        .n_params = if (gguf_file != null) gguf_file.?.totalParams() else if (st_dir != null) st_dir.?.totalParams() else 0,
        .n_experts = fmt.getArchU32(arch_str, "expert_count") orelse fmt.getMetaU32("num_local_experts") orelse 0,
        .n_experts_used = fmt.getArchU32(arch_str, "expert_used_count") orelse fmt.getMetaU32("num_experts_per_tok") orelse 0,
        .file_size_bytes = file_size_bytes,
        .load_ms = load_ms,
        .warmup_ms = 0, // updated after preload
    };

    // ── Recipe defaults ─────────────────────────────────────────
    const recipe = Recipe.match(arch_str, be_name, quant) orelse Recipe.default;
    const applied = recipe.applyDefaults(
        cli.temperature,
        cli.top_p,
        cli.top_k,
        cli.repeat_penalty,
        cli.max_tokens,
        cli.ctx_size,
        cli.user_set,
    );
    cli.temperature = applied.temperature;
    cli.top_p = applied.top_p;
    cli.top_k = applied.top_k;
    cli.repeat_penalty = applied.repeat_penalty;
    cli.max_tokens = applied.max_tokens;
    cli.ctx_size = applied.ctx_size;

    // ── Context size defaults ────────────────────────────────────
    // When ctx_size is 0, check whether the user explicitly passed --ctx-size 0
    // (meaning "use full context") or no one set it (apply smart default cap).
    // This avoids massive KV cache allocations for models with very large
    // context lengths (e.g. 128K). --ctx-size 0 = use model's full context.
    const model_native_ctx = disp_info.ctx_size; // from model metadata
    if (cli.ctx_size == 0) {
        if (cli.user_set.ctx_size) {
            // User explicitly passed --ctx-size 0 → use model's full context
            cli.ctx_size = if (model_native_ctx > 0) model_native_ctx else default_ctx_size;
        } else {
            // No user or recipe override → apply smart default cap
            cli.ctx_size = if (model_native_ctx > 0)
                @min(model_native_ctx, default_ctx_size)
            else
                default_ctx_size;
        }
    }
    // Update banner info to show effective context size
    disp_info.ctx_size = cli.ctx_size;
    if (cli.kv_type_k == cli.kv_type_v) {
        disp_info.kv_type_name = cli.kv_type_k.name();
        disp_info.kv_bpe = cli.kv_type_k.bitsPerElement();
    } else {
        // Asymmetric: show "K-type / V-type" and average bpe
        const kv_label_buf = &disp_info.kv_asym_name_buf;
        disp_info.kv_asym_name_len = if (std.fmt.bufPrint(kv_label_buf, "{s}-K / {s}-V", .{ cli.kv_type_k.name(), cli.kv_type_v.name() })) |s| s.len else |_| 0;
        disp_info.kv_type_name = kv_label_buf[0..disp_info.kv_asym_name_len];
        disp_info.kv_bpe = (cli.kv_type_k.bitsPerElement() + cli.kv_type_v.bitsPerElement()) / 2.0;
    }

    if (!g_quiet) {
        display.printBanner(disp_info);
        var be_info = bs.be.backendInfo();
        be_info.n_threads = @intCast(std.Thread.getCpuCount() catch 1);
        if (be_info.system_mem == 0) be_info.system_mem = backend_mod.detectSystemMem();
        if (be_info.system_avail == 0) be_info.system_avail = backend_mod.detectAvailMem();
        if (be_info.l2_cache == 0) {
            const caches = backend_mod.detectCacheSizes();
            be_info.l1_cache = caches.l1;
            be_info.l2_cache = caches.l2;
            be_info.l3_cache = caches.l3;
        }
        if (be_info.os_version.len == 0) be_info.os_version = backend_mod.detectOsVersion();
        display.printSystemInfo(be_info);
    }

    // ── Preload weights into RAM (after banner so user sees info first) ──
    const warmup_ms: u64 = if (!cli.use_mmap and !cli.model_info)
        preloadModel(
            if (gguf_file != null) &gguf_file.? else null,
            if (st_dir != null) &st_dir.? else null,
            g_quiet,
            g_tty,
            file_size_bytes,
        )
    else
        0;
    disp_info.warmup_ms = warmup_ms;

    if (recipe.name.len > 0 and !std.mem.eql(u8, recipe.name, "default") and !cli.model_info and !g_quiet) {
        eprint("recipe: {s}\n", .{recipe.name});
    }

    // ── Model info early exit ─────────────────────────────────────
    if (cli.model_info) {
        if (cli.json) {
            display.printJsonModelInfo(disp_info);
        } else {
            display.printModelInfo(disp_info);
        }
        return;
    }

    // ── Tokenizer ────────────────────────────────────────────────
    var tok = BpeTokenizer.init(allocator);
    defer tok.deinit();

    const vocab = fmt.getVocab();
    const merges = fmt.getMerges();
    // Gemma uses SentencePiece tokenization even when merges are present in tokenizer.json
    const tok_kind: TokenizerKind = if (arch == .gemma3 or arch == .gemma4) .spm_no_dummy else if (merges != null) .bpe else .spm;
    dbg("tokenizer: vocab={s}, merges={s}, kind={s}", .{
        if (vocab != null) @as([]const u8, "yes") else @as([]const u8, "null"),
        if (merges != null) @as([]const u8, "yes") else @as([]const u8, "null"),
        @as([]const u8, @tagName(tok_kind)),
    });
    const eos_id = fmt.getMetaU32("tokenizer.ggml.eos_token_id") orelse
        fmt.getMetaU32("eos_token_id") orelse
        arch.defaultEos();
    const bos_id: u32 = blk: {
        // GLM-4: template includes [gMASK]<sop> — don't also prepend metadata BOS
        if (arch == .glm4) break :blk 0;
        if (fmt.getMetaU32("tokenizer.ggml.bos_token_id")) |id| break :blk id;
        if (fmt.getMetaU32("bos_token_id")) |id| break :blk id;
        // GPT-2 based tokenizers (Qwen, etc.) don't use BOS by default.
        if (fmt.getMetaStr("tokenizer.ggml.model")) |m| {
            if (std.mem.eql(u8, m, "gpt2")) break :blk 0;
        }
        break :blk arch.defaultBos() orelse 0;
    };
    var eog = getEogTokens(fmt, eos_id);

    if (vocab) |v| {
        switch (tok_kind) {
            .spm, .spm_no_dummy => tok.loadFromGGUFSpm(v, eos_id) catch |e| {
                eprint("Error: failed to load {s} tokenizer: {}\n", .{ @tagName(tok_kind), e });
                std.process.exit(1);
            },
            .bpe => tok.loadFromGGUF(v, merges.?, eos_id) catch |e| {
                eprint("Error: failed to load {s} tokenizer: {}\n", .{ @tagName(tok_kind), e });
                std.process.exit(1);
            },
        }
        tok.bos_token_id = bos_id;
        tok.tok_kind = tok_kind;

        // Add EOG tokens defined by the chat template
        const tmpl = arch.chatTemplate();
        for (tmpl.eog_tokens) |eog_name| {
            if (tok.special_tokens.get(eog_name)) |id| {
                if (!isEogToken(id, eog) and eog.len < eog.ids.len) {
                    eog.ids[eog.len] = id;
                    eog.len += 1;
                }
            }
        }
    } else {
        eprint("Error: no embedded tokenizer found (expected vocab in GGUF or tokenizer.json)\n", .{});
        std.process.exit(1);
    }

    if (!g_quiet and !cli.json and !cli.model_info) {
        if (model_native_ctx > 0 and cli.ctx_size < model_native_ctx) {
            eprint("context: {d} (model supports {d}, use --ctx-size to increase)\n", .{ cli.ctx_size, model_native_ctx });
        } else if (cli.user_set.ctx_size) {
            eprint("context: {d}\n", .{cli.ctx_size});
        }
    }

    // ── Piped stdin → single prompt ──────────────────────────────
    var piped_prompt: ?[]const u8 = null;
    defer if (piped_prompt) |p| allocator.free(p);
    if (cli.prompt == null and !cli.serve) {
        if (!(stdin_file.isTty(g_io) catch false)) {
            piped_prompt = readStdinAll(allocator, max_stdin_prompt_size);
        }
    }

    // ── Init model ───────────────────────────────────────────────
    const effective_prompt = cli.prompt orelse if (piped_prompt) |p|
        std.mem.trim(u8, p, " \t\r\n")
    else
        null;

    // Warn about piped stdin in server mode (positional prompt and --system
    // are already warned in parseCli before model loading).
    // Note: piped_prompt is only read when !cli.serve, so check isatty directly.
    if (cli.serve and cli.prompt == null and !(stdin_file.isTty(g_io) catch false)) {
        eprint("Warning: piped stdin ignored in server mode (--serve)\n", .{});
    }

    // ── Construct load info ────────────────────────────────────────
    const n_tensors: u64 = if (gguf_file) |g| g.tensor_count else if (st_dir) |s| s.tensorCount() else 0;
    const format_name: []const u8 = if (gguf_file) |g|
        (if (g.version == 3) "GGUF v3" else if (g.version == 2) "GGUF v2" else "GGUF")
    else if (st_dir != null)
        "SafeTensors"
    else
        "";
    disp_info.format_name = format_name;

    const load_info = display_mod.LoadInfo{
        .n_tensors = n_tensors,
        .tok_kind = @tagName(tok_kind),
        .vocab_size = disp_info.vocab_size,
        .eos_id = eos_id,
        .bos_id = bos_id,
        .n_eog = eog.len,
        .template_name = arch.templateName(),
        .format_name = format_name,
        .init_ms = 0, // filled by initAndRun
    };

    if (!initAndRun(arch, allocator, fmt, be, &tok, &cli, tok_kind, eog, effective_prompt, disp_info, display, if (bs.pool) |*p| p else null, load_info))
        std.process.exit(1);
}

/// Load an image file (PNG, PPM P6, or JPEG) and resize to
/// target_size x target_size. Returns [target_size * target_size * 3]u8
/// RGB pixels in row-major, channel-last order.
///
/// Format is auto-detected from magic bytes:
///   - PNG (0x89 P N G): full decode via image.decodePng
///   - PPM P6 ("P6"): raw RGB parse
///   - JPEG (0xFF 0xD8): returns error with suggestion to convert
///
/// Resize uses bilinear interpolation via image.resize.
fn loadImage(allocator: std.mem.Allocator, path: []const u8, target_size: u32) ![]u8 {
    const file = try Io.Dir.cwd().openFile(g_io, path, .{});
    defer file.close(g_io);

    // Read entire file into memory (vision images are small, typically < 10MB)
    const max_image_file_size: usize = 64 * 1024 * 1024; // 64 MB limit
    const file_stat = try file.stat(g_io);
    if (file_stat.size > max_image_file_size) return error.FileTooBig;
    const file_data = try allocator.alloc(u8, @intCast(file_stat.size));
    errdefer allocator.free(file_data);
    _ = try file.readPositionalAll(g_io, file_data, 0);
    defer allocator.free(file_data);

    const format = image.detectFormat(file_data);
    switch (format) {
        .png => {
            var png = try image.decodePng(allocator, file_data);
            defer png.deinit();
            return image.resize(allocator, png.pixels, png.width, png.height, target_size, target_size);
        },
        .ppm => {
            return loadPpmData(allocator, file_data, target_size);
        },
        .jpeg => {
            eprint("Error: JPEG images are not supported. Please convert to PNG:\n", .{});
            eprint("  convert input.jpg input.png   (ImageMagick)\n", .{});
            eprint("  ffmpeg -i input.jpg input.png  (ffmpeg)\n", .{});
            return error.InvalidImageFormat;
        },
        .unknown => {
            eprint("Error: unrecognized image format. Supported: PNG, PPM (P6)\n", .{});
            return error.InvalidImageFormat;
        },
    }
}

/// Parse PPM P6 data from a buffer and resize to target dimensions.
///
/// PPM P6 format:
///   - Magic "P6", optional comments starting with '#'
///   - "<width> <height>\n", "<maxval>\n" (usually 255)
///   - width * height * 3 raw bytes (RGB)
fn loadPpmData(allocator: std.mem.Allocator, file_data: []const u8, target_size: u32) ![]u8 {
    var pos: usize = 0;

    // Validate magic "P6"
    if (file_data.len < 3) return error.InvalidImageFormat;
    if (file_data[0] != 'P' or file_data[1] != '6') return error.InvalidImageFormat;
    pos = 2;
    // Skip whitespace after magic
    while (pos < file_data.len and (file_data[pos] == '\n' or file_data[pos] == '\r' or file_data[pos] == ' ')) : (pos += 1) {}

    // Helper: skip comments and whitespace, read next non-comment token
    const readToken = struct {
        fn call(data: []const u8, start: *usize) ?[]const u8 {
            var p = start.*;
            while (p < data.len) {
                while (p < data.len and (data[p] == ' ' or data[p] == '\t' or data[p] == '\n' or data[p] == '\r')) : (p += 1) {}
                if (p >= data.len) return null;
                if (data[p] == '#') {
                    while (p < data.len and data[p] != '\n') : (p += 1) {}
                    continue;
                }
                const tok_start = p;
                while (p < data.len and data[p] != ' ' and data[p] != '\t' and data[p] != '\n' and data[p] != '\r') : (p += 1) {}
                start.* = p;
                return data[tok_start..p];
            }
            return null;
        }
    }.call;

    const w_str = readToken(file_data, &pos) orelse return error.InvalidImageFormat;
    const width = std.fmt.parseInt(u32, w_str, 10) catch return error.InvalidImageFormat;
    const h_str = readToken(file_data, &pos) orelse return error.InvalidImageFormat;
    const height = std.fmt.parseInt(u32, h_str, 10) catch return error.InvalidImageFormat;
    const max_str = readToken(file_data, &pos) orelse return error.InvalidImageFormat;
    _ = std.fmt.parseInt(u32, max_str, 10) catch return error.InvalidImageFormat;

    if (width == 0 or height == 0) return error.InvalidImageFormat;

    // Skip exactly one whitespace character after maxval (part of PPM spec)
    if (pos < file_data.len) pos += 1;

    // Remaining data is raw pixels (checked arithmetic to prevent overflow with crafted dimensions)
    const src_pixels: usize = std.math.mul(usize, std.math.mul(usize, @as(usize, width), height) catch
        return error.InvalidImageSize, 3) catch return error.InvalidImageSize;
    if (pos + src_pixels > file_data.len) return error.InvalidImageSize;
    const src_data = file_data[pos..][0..src_pixels];

    // Resize using bilinear interpolation
    return image.resize(allocator, src_data, width, height, target_size, target_size);
}

/// Initialize the model and run inference/server/REPL. Returns false on failure.
fn initAndRun(
    arch: Arch,
    allocator: std.mem.Allocator,
    fmt: Format,
    be: Backend,
    tok: *BpeTokenizer,
    cli: *CliArgs,
    tok_kind: TokenizerKind,
    eog: anytype,
    effective_prompt: ?[]const u8,
    minfo: display_mod.ModelInfo,
    display: Display,
    pool: ?*ThreadPool,
    load_info_in: display_mod.LoadInfo,
) bool {
    // Initialize optional tiered KV cache from CLI flags.
    // This is model-independent — only reads format metadata.
    var tiered_cache_storage: ?TieredKvCache = null;
    defer if (tiered_cache_storage) |*tc| tc.deinit();

    if (cli.kv_tiers) |tiers_str| {
        const has_ram = std.mem.indexOf(u8, tiers_str, "ram") != null;
        const has_ssd = std.mem.indexOf(u8, tiers_str, "ssd") != null;

        const ram_gb: usize = if (cli.kv_ram_budget) |b|
            @as(usize, b)
        else
            detectFreeRam() / (ram_budget_divisor * gib_bytes);

        const ssd_gb: usize = if (cli.kv_ssd_budget) |b|
            @as(usize, b)
        else
            default_ssd_budget_gb;

        // Read model metadata for cache dimension calculations.
        const n_layers = fmt.getMetaU32("llama.block_count") orelse
            fmt.getMetaU32("num_hidden_layers") orelse tiered_fallback_n_layers;
        const n_embd = fmt.getMetaU32("llama.embedding_length") orelse
            fmt.getMetaU32("hidden_size") orelse tiered_fallback_n_embd;
        const n_kv_heads = fmt.getMetaU32("llama.attention.head_count_kv") orelse
            fmt.getMetaU32("num_key_value_heads") orelse tiered_fallback_n_kv_heads;
        const n_heads = fmt.getMetaU32("llama.attention.head_count") orelse
            fmt.getMetaU32("num_attention_heads") orelse tiered_fallback_n_heads;
        const head_dim = fmt.getMetaU32("llama.attention.key_length") orelse
            fmt.getMetaU32("head_dim") orelse (n_embd / n_heads);
        const kv_dim: usize = std.math.mul(usize, @as(usize, n_kv_heads), head_dim) catch {
            eprint("Error: KV dimensions overflow (n_kv_heads={d}, head_dim={d})\n", .{ n_kv_heads, head_dim });
            return false;
        };

        const block_size = tiered_kv_block_size;
        const bytes_per_block = std.math.mul(usize, @as(usize, block_size) * (@sizeOf(f32) * kv_tensors_per_position), kv_dim) catch {
            eprint("Error: KV cache block size overflow (block_size={d}, kv_dim={d})\n", .{ block_size, kv_dim });
            return false;
        };
        const ctx = if (cli.ctx_size > 0) cli.ctx_size else default_ctx_size;
        const vram_blocks: usize = (@as(usize, ctx) + block_size - 1) / block_size;
        const ram_blocks: usize = if (has_ram and bytes_per_block > 0) (ram_gb * gib_bytes) / bytes_per_block else 0;
        const ssd_blocks: usize = if (has_ssd and bytes_per_block > 0) (ssd_gb * gib_bytes) / bytes_per_block else 0;

        tiered_cache_storage = TieredKvCache.init(
            allocator,
            n_layers,
            kv_dim,
            vram_blocks,
            ram_blocks,
            ssd_blocks,
            block_size,
            if (has_ssd) cli.kv_ssd_path else null,
        ) catch |e| {
            eprint("Error: failed to initialize tiered KV cache: {}\n", .{e});
            return false;
        };

        if (!g_quiet) {
            eprint("  Tiered KV cache: {d} VRAM + {d} RAM + {d} SSD blocks\n", .{ vram_blocks, ram_blocks, ssd_blocks });
        }
    }

    const tiered_ptr: ?*TieredKvCache = if (tiered_cache_storage != null) &tiered_cache_storage.? else null;

    // Use ModelStorage to initialize the model without exposing concrete types.
    const init_start = milliTimestamp(g_io);
    const eviction_budget: u32 = if (cli.kv_eviction)
        (if (cli.kv_budget > 0) cli.kv_budget else @as(u32, @intCast(cli.ctx_size * 4 / 5)))
    else
        0;
    var mdl = ModelStorage.initFromArch(arch, allocator, fmt, be, cli.ctx_size, cli.kv_type_k, cli.kv_type_v, cli.kv_boundary_v, eviction_budget, tiered_ptr) catch |e| {
        eprint("Error: failed to initialize {s}: {}\n", .{ arch.displayName(), e });
        if (e == error.OutOfMemory)
            eprint("  Not enough memory. Try a smaller quantization or model.\n", .{})
        else if (e == error.TensorNotFound)
            eprint("  Required tensor missing. The model file may be corrupted or incomplete.\n", .{});
        return false;
    };
    defer mdl.deinit();
    mdl.setPool(pool);
    mdl.fixBlockAllocator();
    mdl.setChunkSize(cli.prefill_batch_size);

    // Megakernel mode: validate support and enable
    if (cli.megakernel) {
        const supported = switch (be) {
            .metal => switch (arch) {
                .qwen35, .gemma4, .gemma3, .glm4 => true,
                else => false,
            },
            .cuda => switch (arch) {
                .qwen35 => true,
                else => false,
            },
            else => false,
        };
        if (!supported) {
            eprint("Error: --megakernel not supported for {s} on this backend.\n" ++
                "Supported: qwen35/gemma4/gemma3 on Metal, qwen35 on CUDA. See docs/MEGAKERNEL.md\n", .{@tagName(arch)});
            return false;
        }
        mdl.setMegakernel(true);
    }

    const init_ms = elapsedMs(init_start);
    if (!g_quiet) {
        var li = load_info_in;
        li.init_ms = init_ms;
        display.printLoadInfo(li);
    }
    if (cli.profile) mdl.enableProfiling();

    var model_if = mdl.model();

    // ── Vision encoder (multimodal) ──────────────────────────────
    const VisionEncoder = model_mod.VisionEncoder;
    var mmproj_gguf: ?GGUFFile = null;
    defer if (mmproj_gguf) |*mf| mf.deinit();
    var vision_enc: ?VisionEncoder = null;
    defer if (vision_enc) |*ve| ve.deinit();

    // Auto-detect mmproj file if user didn't specify one.
    // For GGUF files, check the containing directory; for SafeTensors dirs, check
    // the directory itself. Uses dirname() which returns null for bare filenames.
    var auto_mmproj_buf: [Io.Dir.max_path_bytes]u8 = undefined;
    var mmproj_path: ?[]const u8 = cli.mmproj;
    if (mmproj_path == null and (cli.image != null or cli.serve)) {
        const model_dir: []const u8 = blk: {
            // Check if model_path is a directory (SafeTensors)
            const probe_dir = Io.Dir.cwd().openDir(g_io, cli.model_path, .{}) catch break :blk Io.Dir.path.dirname(cli.model_path) orelse ".";
            probe_dir.close(g_io);
            break :blk cli.model_path;
        };
        // Use Io.Dir.Reader for directory iteration (Zig 0.16 idiom)
        const scan_dir = Io.Dir.cwd().openDir(g_io, model_dir, .{ .iterate = true }) catch Io.Dir.cwd();
        var dir_buf: [Io.Dir.Reader.min_buffer_len]u8 align(@alignOf(usize)) = undefined;
        var reader = Io.Dir.Reader.init(scan_dir, &dir_buf);
        var best: ?[]const u8 = null;
        while (true) {
            var entries: [1]Io.Dir.Entry = undefined;
            const n = reader.read(g_io, &entries) catch break;
            if (n == 0) break;
            const name = entries[0].name;
            if (std.mem.startsWith(u8, name, "mmproj") and
                std.mem.endsWith(u8, name, ".gguf"))
            {
                if (best == null or std.mem.lessThan(u8, name, best.?)) {
                    const full = std.fmt.bufPrint(&auto_mmproj_buf, "{s}/{s}", .{ model_dir, name }) catch continue;
                    best = full;
                }
            }
        }
        if (scan_dir.handle != Io.Dir.cwd().handle) scan_dir.close(g_io);
        if (best) |b| {
            mmproj_path = b;
            if (!g_quiet) eprint("vision: auto-detected {s}\n", .{Io.Dir.path.basename(b)});
        }
    }

    if (mmproj_path) |mpath| {
        mmproj_gguf = GGUFFile.open(allocator, mpath) catch |err| {
            eprint("Error: failed to load mmproj '{s}': {}\n", .{ mpath, err });
            return false;
        };
        const mmproj_fmt = mmproj_gguf.?.format();
        vision_enc = VisionEncoder.init(allocator, mmproj_fmt, be, pool) catch |err| {
            eprint("Error: failed to init vision encoder: {}\n", .{err});
            return false;
        };
        {
            const ve = &vision_enc.?;
            std.debug.assert(ve.patch_size > 0);
            std.debug.assert(ve.projection_dim > 0);
        }
        if (!g_quiet) {
            const ve = &vision_enc.?;
            eprint("vision: {d} layers, {d}x{d} patches -> {d}D\n", .{
                ve.n_blocks,
                ve.image_size / ve.patch_size,
                ve.image_size / ve.patch_size,
                ve.projection_dim,
            });
        }
    }

    // Resolve image pad token ID for multimodal prompt injection.
    const img_tokens = arch.imageTokens();
    var n_visual_tokens: u32 = 0;

    if (vision_enc) |*ve| {
        if (cli.image) |image_path| {
            // For Qwen VL, use the original image dimensions aligned to
            // patch_size * 2 (spatial_merge_size) instead of upscaling to image_size.
            // This matches llama.cpp which processes at native resolution.
            const target_size: u32 = if (ve.use_native_resolution) blk: {
                const grid: u32 = ve.patch_size * 2; // patch_size * spatial_merge
                const orig = image.getImageDimensions(allocator, g_io, image_path) catch break :blk ve.image_size;
                const side = @max(orig.width, orig.height);
                break :blk ((side + grid - 1) / grid) * grid;
            } else ve.image_size;
            const img_pixels = loadImage(allocator, image_path, target_size) catch |err| {
                eprint("Error: failed to load image '{s}': {}\n", .{ image_path, err });
                return false;
            };
            defer allocator.free(img_pixels);

            // Update n_patches for the actual processing resolution
            ve.n_patches = (target_size / ve.patch_size) * (target_size / ve.patch_size);
            ve.n_output_patches = ve.n_patches / 4; // Qwen 4× merge
            ve.image_size = target_size;

            const visual_tokens = ve.encode(img_pixels) catch |err| {
                eprint("Error: vision encode failed: {}\n", .{err});
                return false;
            };
            n_visual_tokens = @intCast(visual_tokens.len / ve.projection_dim);
            const pad_id: u32 = if (img_tokens) |it| it.pad else 0;
            model_if.setImageEmbeddings(visual_tokens, n_visual_tokens, pad_id);
            if (!g_quiet) eprint("vision: encoded {d} visual tokens\n", .{n_visual_tokens});
        }
    } else if (cli.image != null) {
        eprint("Warning: --image ignored (no vision projector found — use --mmproj <path> to specify)\n", .{});
    }

    // ── Draft model loading (speculative decoding) ──────────────
    var draft_gguf: ?GGUFFile = null;
    var draft_st: ?SafeTensorsDir = null;
    var draft_mdl_storage: ?ModelStorage = null;
    defer {
        if (draft_mdl_storage) |*dm| dm.deinit();
        if (draft_gguf) |*g| g.deinit();
        if (draft_st) |*s| s.deinit();
    }

    var draft_ptr: ?*Model = null;
    var draft_model_if: Model = undefined;

    if (cli.draft_model_path) |draft_path| {
        const draft_is_dir = blk: {
            const d = Io.Dir.cwd().openDir(g_io, draft_path, .{}) catch break :blk false;
            d.close(g_io);
            break :blk true;
        };
        var draft_fmt: Format = undefined;
        if (draft_is_dir) {
            draft_st = SafeTensorsDir.open(allocator, draft_path) catch |e| {
                eprint("Error: failed to open draft model '{s}': {}\n", .{ draft_path, e });
                return false;
            };
            draft_fmt = draft_st.?.format();
        } else {
            draft_gguf = GGUFFile.open(allocator, draft_path) catch |e| {
                eprint("Error: failed to open draft model '{s}': {}\n", .{ draft_path, e });
                return false;
            };
            draft_fmt = draft_gguf.?.format();
        }
        const draft_arch_str = draft_fmt.getMetaStr("general.architecture") orelse
            draft_fmt.getMetaStr("model_type") orelse "unknown";
        var draft_arch = Arch.detect(draft_arch_str) orelse {
            eprint("Error: unsupported draft model architecture '{s}'\n", .{draft_arch_str});
            return false;
        };
        if (draft_arch == .nemotron_h and draft_fmt.getTensor("backbone.embeddings.weight") != null)
            draft_arch = .nemotron_nano;
        if (!draft_arch.isEnabled()) {
            eprint("Error: draft model arch {s} disabled at compile time\n", .{draft_arch.displayName()});
            return false;
        }
        draft_mdl_storage = ModelStorage.initFromArch(draft_arch, allocator, draft_fmt, be, cli.ctx_size, .f16, .f16, 0, 0, null) catch |e| {
            eprint("Error: failed to init draft model: {}\n", .{e});
            return false;
        };
        draft_mdl_storage.?.setPool(pool);
        draft_mdl_storage.?.fixBlockAllocator();
        draft_model_if = draft_mdl_storage.?.model();
        draft_ptr = &draft_model_if;
        eprint("draft: {s} · {s}\n", .{ draft_arch.displayName(), Format.getQuantName(draft_fmt) });
    } else if (cli.spec_mode != .none) {
        draft_ptr = &model_if;
    }

    if (cli.serve) {
        var tok_if = tok.tokenizer();
        const ve_ptr: ?*VisionEncoder = if (vision_enc != null) &vision_enc.? else null;
        const srv_pad_id: u32 = if (img_tokens) |it| it.pad else 0;
        const srv_start_id: u32 = if (img_tokens) |it| it.start else 0;
        const srv_end_id: u32 = if (img_tokens) |it| it.end else 0;
        server.run(.{
            .allocator = allocator,
            .model = &model_if,
            .tokenizer = &tok_if,
            .chat_template = arch.chatTemplate(),
            .model_name = minfo.name,
            .backend_name = minfo.be_name,
            .port = cli.port,
            .bos_token_id = tok.bos_token_id,
            .eog_ids = eog.ids,
            .eog_len = eog.len,
            .tiered_cache = tiered_ptr,
            .api_key = cli.api_key,
            .host = cli.host,
            .ctx_size = cli.ctx_size,
            .vision_encoder = ve_ptr,
            .image_pad_token_id = srv_pad_id,
            .image_start_token_id = srv_start_id,
            .image_end_token_id = srv_end_id,
            .io = g_io,
            .draft_model = draft_ptr,
            .spec_tokens = cli.spec_tokens,
            .tree_budget = cli.tree_budget,
        }) catch |e| {
            eprint("Error: server failed: {}\n", .{e});
            return false;
        };
    } else if (effective_prompt) |prompt| {
        generateAndPrint(allocator, &model_if, tok, cli, tok_kind, eog, arch, prompt, !g_quiet, minfo, display, img_tokens, n_visual_tokens, draft_ptr);
    } else {
        runRepl(allocator, &model_if, tok, cli, tok_kind, eog, arch, minfo, display, img_tokens, n_visual_tokens, if (vision_enc != null) &vision_enc.? else null);
    }
    mdl.reportPerf();
    return true;
}

// ── Interactive REPL ─────────────────────────────────────────────

fn runRepl(
    allocator: std.mem.Allocator,
    mdl: *Model,
    tok: *BpeTokenizer,
    cli: *CliArgs,
    tok_kind: TokenizerKind,
    eog: anytype,
    arch: Arch,
    minfo: display_mod.ModelInfo,
    display_in: Display,
    img_tokens: ?arch_mod.ImageTokens,
    n_visual_tokens_init: u32,
    vision_enc: ?*model_mod.VisionEncoder,
) void {
    var n_visual_tokens: u32 = n_visual_tokens_init;
    _ = vision_enc;
    var display = display_in;
    print("Type a message, /help for commands, Ctrl+D to quit.\n", .{});

    var editor = LineEditor.init(allocator);
    defer editor.deinit();

    const repl_prompt = if (g_color) "\x1b[1;32m> \x1b[0m" else "> ";
    var show_stats: bool = !g_quiet;

    // Track REPL-owned system prompt (from /system command)
    var system_prompt_owned: ?[]const u8 = null;
    defer if (system_prompt_owned) |sp| allocator.free(sp);

    // Conversation history for multi-turn support
    var history: std.ArrayList(Message) = .empty;
    defer {
        for (history.items) |msg| allocator.free(@constCast(msg.content));
        history.deinit(allocator);
    }

    const template = arch.chatTemplate();

    while (true) {
        print("\n", .{});
        const line_owned = editor.readline(repl_prompt) orelse {
            print("\n", .{});
            return;
        };
        defer allocator.free(line_owned);

        const trimmed = std.mem.trim(u8, line_owned, " \t\r\n");
        if (trimmed.len == 0) continue;

        editor.addHistory(trimmed);

        // REPL commands
        if (trimmed[0] == '/') {
            if (std.mem.eql(u8, trimmed, "/quit") or std.mem.eql(u8, trimmed, "/exit") or std.mem.eql(u8, trimmed, "/q")) {
                return;
            } else if (std.mem.eql(u8, trimmed, "/clear") or std.mem.eql(u8, trimmed, "/reset")) {
                mdl.resetCache();
                for (history.items) |msg| allocator.free(@constCast(msg.content));
                history.clearRetainingCapacity();
                print("Conversation and KV cache cleared.\n", .{});
                continue;
            } else if (std.mem.eql(u8, trimmed, "/context") or std.mem.eql(u8, trimmed, "/ctx")) {
                const used = mdl.kvSeqLen();
                const max_ctx = cli.ctx_size;
                const pct: f32 = if (max_ctx > 0) @as(f32, @floatFromInt(used)) / @as(f32, @floatFromInt(max_ctx)) * 100.0 else 0.0;
                print("Context: {d} / {d} tokens ({d:.1}% used)\n", .{ used, max_ctx, pct });
                continue;
            } else if (std.mem.startsWith(u8, trimmed, "/system ")) {
                const new_system = std.mem.trim(u8, trimmed[8..], " \t");
                if (new_system.len == 0) {
                    print("Usage: /system <prompt text>\n", .{});
                    continue;
                }
                // Free old system prompt if we own it
                if (system_prompt_owned) |old| allocator.free(old);
                const duped = allocator.dupe(u8, new_system) catch {
                    eprint("Error: out of memory\n", .{});
                    continue;
                };
                system_prompt_owned = duped;
                cli.system_prompt = duped;
                // Clear conversation since system prompt is baked into first turn
                mdl.resetCache();
                for (history.items) |msg| allocator.free(@constCast(msg.content));
                history.clearRetainingCapacity();
                print("System prompt set. Conversation cleared.\n", .{});
                continue;
            } else if (std.mem.eql(u8, trimmed, "/system")) {
                if (cli.system_prompt) |sp| {
                    print("System prompt: {s}\n", .{sp});
                } else {
                    print("No system prompt set. Usage: /system <prompt text>\n", .{});
                }
                continue;
            } else if (std.mem.eql(u8, trimmed, "/stats")) {
                show_stats = !show_stats;
                print("Stats {s}.\n", .{if (show_stats) "on" else "off"});
                continue;
            } else if (std.mem.eql(u8, trimmed, "/verbose")) {
                g_verbose = !g_verbose;
                display.verbose = g_verbose;
                print("Verbose {s}.\n", .{if (g_verbose) "on" else "off"});
                continue;
            } else if (std.mem.eql(u8, trimmed, "/debug")) {
                g_debug = !g_debug;
                // debug implies verbose — turning debug on enables verbose,
                // but turning debug off leaves verbose unchanged (user may
                // have enabled it independently via /verbose).
                if (g_debug) {
                    g_verbose = true;
                    display.verbose = true;
                    print("Debug on (verbose enabled).\n", .{});
                } else {
                    print("Debug off.\n", .{});
                }
                continue;
            } else if (std.mem.eql(u8, trimmed, "/model")) {
                display.printModelInfo(minfo);
                continue;
            } else if (std.mem.eql(u8, trimmed, "/help")) {
                _ = std.c.write(stdout_file.handle, repl_help.ptr, repl_help.len);
                continue;
            } else {
                print("Unknown command: {s} (try /help)\n", .{trimmed});
                continue;
            }
        }

        // Add user message to history
        const user_content = allocator.dupe(u8, trimmed) catch continue;
        history.append(allocator, .{ .role = .user, .content = user_content }) catch {
            allocator.free(user_content);
            continue;
        };

        const is_first_turn = history.items.len == 1;

        // First turn: format full conversation (system prompt + user message).
        // Subsequent turns: format only the continuation (assistant_suffix +
        // user_prefix + new message + assistant_prefix) and reuse the KV cache.
        const formatted = if (is_first_turn)
            template.formatConversation(allocator, cli.system_prompt, history.items) catch {
                eprint("Error: failed to format conversation\n", .{});
                continue;
            }
        else
            template.formatContinuation(allocator, trimmed) catch {
                eprint("Error: failed to format continuation\n", .{});
                continue;
            };
        defer allocator.free(formatted);

        if (is_first_turn) mdl.resetCache();
        // Image tokens only on first turn (from --image CLI flag). After first
        // turn, reset to 0 so continuation turns don't re-inject image tokens.
        const turn_n_vis = if (is_first_turn) n_visual_tokens else @as(u32, 0);
        const response = generateAndPrintInner(allocator, mdl, tok, cli, tok_kind, eog, template, formatted, false, !is_first_turn, show_stats, minfo, display, true, img_tokens, turn_n_vis);
        if (is_first_turn and n_visual_tokens > 0) n_visual_tokens = 0;

        // Add assistant response to history
        if (response) |text| {
            // Trim trailing whitespace from response for clean history
            const trimmed_resp = std.mem.trimEnd(u8, text, " \t\r\n");
            if (trimmed_resp.len > 0) {
                const resp_content = allocator.dupe(u8, trimmed_resp) catch {
                    allocator.free(text);
                    continue;
                };
                history.append(allocator, .{ .role = .assistant, .content = resp_content }) catch {
                    allocator.free(resp_content);
                };
            }
            allocator.free(text);
        }
    }
}

// ── Shared generation logic ──────────────────────────────────────

fn generateAndPrint(
    allocator: std.mem.Allocator,
    mdl: *Model,
    tok: *BpeTokenizer,
    cli: *const CliArgs,
    tok_kind: TokenizerKind,
    eog: anytype,
    arch: Arch,
    prompt: []const u8,
    show_stats: bool,
    minfo: display_mod.ModelInfo,
    display: Display,
    img_tokens: ?arch_mod.ImageTokens,
    n_visual_tokens: u32,
    draft_model: ?*Model,
) void {
    if (draft_model) |dm| {
        generateSpeculative(allocator, mdl, dm, tok, cli, tok_kind, eog, arch, prompt, show_stats);
    } else {
        const response = generateAndPrintInner(allocator, mdl, tok, cli, tok_kind, eog, arch.chatTemplate(), prompt, true, false, show_stats, minfo, display, false, img_tokens, n_visual_tokens);
        if (response) |r| allocator.free(r);
    }
}

fn generateSpeculative(
    allocator: std.mem.Allocator,
    target: *Model,
    draft_model: *Model,
    tok: *BpeTokenizer,
    cli: *const CliArgs,
    tok_kind: TokenizerKind,
    eog: anytype,
    arch: Arch,
    prompt: []const u8,
    show_stats: bool,
) void {
    const template = arch.chatTemplate();
    const formatted = template.format(allocator, cli.system_prompt, prompt) catch @as([]const u8, prompt);
    defer if (formatted.ptr != prompt.ptr) allocator.free(formatted);

    const token_ids = switch (tok_kind) {
        .spm => tok.encodeSpm(formatted),
        .spm_no_dummy => tok.encodeSpmNoDummy(formatted),
        .bpe => tok.encode(formatted),
    } catch {
        eprint("Error: tokenization failed\n", .{});
        return;
    };
    defer allocator.free(token_ids);
    if (token_ids.len == 0) {
        eprint("Error: empty token sequence\n", .{});
        return;
    }

    // Prepend BOS if needed (same as generateAndPrintInner)
    var prefill_buf: ?[]u32 = null;
    defer if (prefill_buf) |ids| allocator.free(ids);
    const prefill_toks: []const u32 = blk: {
        if (tok.bos_token_id > 0 and token_ids.len > 0) {
            var all = allocator.alloc(u32, token_ids.len + 1) catch break :blk token_ids;
            all[0] = tok.bos_token_id;
            @memcpy(all[1..], token_ids);
            prefill_buf = all;
            break :blk all;
        }
        break :blk token_ids;
    };

    // Prefill both models with the prompt
    const prefill_start = milliTimestamp(g_io);
    var first_target = target.prefill(prefill_toks) catch |e| {
        eprint("Error: target prefill failed: {}\n", .{e});
        return;
    };
    // Only prefill draft model separately when it's a different model
    if (target.ptr != draft_model.ptr) {
        _ = draft_model.prefill(prefill_toks) catch |e| {
            eprint("Error: draft prefill failed: {}\n", .{e});
            return;
        };
    }
    const prefill_ms = milliTimestamp(g_io) - prefill_start;

    // Sampling setup
    const use_sampling = cli.temperature > 0;
    var prng = std.Random.Xoshiro256.init(cli.seed);
    if (use_sampling) {
        first_target = math_ops.sampleToken(target.getLogits(), cli.temperature, cli.top_k, cli.top_p, prng.random());
    }

    // Speculative generation loop
    var spec_state = spec_decode.SpecState.init(allocator, cli.spec_tokens, target.vocabSize()) catch {
        eprint("Error: failed to allocate speculative state\n", .{});
        return;
    };
    defer spec_state.deinit(allocator);

    const gen_start = milliTimestamp(g_io);
    var last = first_target;
    var token_count: u32 = 0;
    var gen_ids_buf: [gen_ids_buf_size]u32 = undefined;
    var batch_start: u32 = 0;
    var started_output = false;
    const batch_size: u32 = if (g_tty) tty_batch_size else pipe_batch_size;

    if (!isEogToken(first_target, eog)) {
        gen_ids_buf[0] = first_target;
        token_count = 1;
    }

    const use_ddtree = (cli.spec_mode == .ddtree);
    const self_spec = (cli.spec_mode == .self_spec);

    // Self-speculative: auto-detect layer skip range (skip middle 50%)
    const self_spec_skip_divisor = 4; // skip starts at 25% of layers
    const self_spec_default_skip_fraction = 2; // skip 50% of layers by default
    const skip_start: u32 = if (self_spec) target.nLayers() / self_spec_skip_divisor else 0;
    const skip_end: u32 = if (self_spec) blk: {
        const skip_count = cli.draft_layers orelse (target.nLayers() / self_spec_default_skip_fraction);
        break :blk skip_start + skip_count;
    } else 0;

    while (token_count < cli.max_tokens and !isEogToken(last, eog)) {
        const pre_draft_pos = target.kvSeqLen();

        // Draft phase
        if (self_spec) target.setLayerSkip(skip_start, skip_end);
        const is_self_draft = (target.ptr == draft_model.ptr and !self_spec);
        const n_drafted = if (is_self_draft and !use_sampling)
            spec_decode.draft(&spec_state, draft_model, last)
        else
            spec_decode.draftWithLogits(&spec_state, draft_model, last);
        if (self_spec) target.setLayerSkip(0, 0);
        if (n_drafted == 0) break;

        // Verify phase
        const result = if (is_self_draft) blk: {
            // Self-draft: draft == target, 100% acceptance. Get bonus token.
            spec_state.recordRound(spec_state.n_draft);
            const last_draft = spec_state.draft_tokens[spec_state.n_draft - 1];
            const bonus = target.forward(last_draft) catch last_draft;
            break :blk spec_decode.SpecResult{ .accepted = spec_state.n_draft, .next_token = bonus };
        } else if (use_ddtree or self_spec)
            spec_decode.verifyDDTree(&spec_state, target, draft_model, last, cli.tree_budget, pre_draft_pos)
        else if (use_sampling)
            spec_decode.verifySampling(&spec_state, target, draft_model, last, pre_draft_pos, cli.temperature, prng.random())
        else
            spec_decode.verifySequential(&spec_state, target, draft_model, last, pre_draft_pos);

        // Emit accepted draft tokens
        var hit_eog = false;
        for (0..result.accepted) |i| {
            const accepted_tok = spec_state.draft_tokens[i];
            if (token_count >= gen_ids_buf.len) break;
            if (isEogToken(accepted_tok, eog)) {
                hit_eog = true;
                break;
            }
            gen_ids_buf[token_count] = accepted_tok;
            token_count += 1;
        }

        // Emit correction/bonus token
        if (!hit_eog and token_count < gen_ids_buf.len) {
            if (isEogToken(result.next_token, eog)) {
                hit_eog = true;
            } else {
                gen_ids_buf[token_count] = result.next_token;
                token_count += 1;
            }
        }
        last = if (hit_eog) target.eosId() else result.next_token;

        // Stream
        if (token_count - batch_start >= batch_size) {
            flushTokenBatch(tok, tok_kind, allocator, gen_ids_buf[batch_start..@min(token_count, gen_ids_buf.len)], &started_output);
            batch_start = token_count;
        }
    }

    // Flush remaining
    if (token_count > batch_start and token_count <= gen_ids_buf.len) {
        flushTokenBatch(tok, tok_kind, allocator, gen_ids_buf[batch_start..token_count], &started_output);
    }
    if (!g_tty and started_output) {
        _ = std.c.write(stdout_file.handle, "\n", 1);
    }

    const gen_ms = milliTimestamp(g_io) - gen_start;
    if (show_stats) {
        const gen_toks = if (token_count > 0) token_count else 1;
        const tok_per_sec = if (gen_ms > 0) @as(f32, @floatFromInt(gen_toks)) / @as(f32, @floatFromInt(gen_ms)) * 1000.0 else 0;
        eprint("\n{d} tok · {d:.1} tok/s · {d}ms prefill · spec: {d:.0}% accept ({d:.1} mean)\n", .{
            gen_toks,
            tok_per_sec,
            prefill_ms,
            spec_state.acceptanceRate() * 100,
            spec_state.meanAccepted(),
        });
    }
}

/// Core generation: formats (or uses pre-formatted) prompt, prefills, generates, streams output.
/// When `skip_bos` is true, the BOS token is not sent (for continuation turns with KV cache reuse).
/// When `need_response` is false, skips the full-sequence decode (avoids allocating
/// response text the caller will discard) unless JSON output mode is active.
/// Returns the generated response text (caller-owned) or null on error.
fn generateAndPrintInner(
    allocator: std.mem.Allocator,
    mdl: *Model,
    tok: *BpeTokenizer,
    cli: *const CliArgs,
    tok_kind: TokenizerKind,
    eog: anytype,
    template: ChatTemplate,
    prompt: []const u8,
    format_prompt: bool,
    skip_bos: bool,
    show_stats: bool,
    minfo: display_mod.ModelInfo,
    display: Display,
    need_response: bool,
    img_tokens: ?arch_mod.ImageTokens,
    n_visual_tokens: u32,
) ?[]u8 {
    const formatted = if (format_prompt)
        template.format(allocator, cli.system_prompt, prompt) catch @as([]const u8, prompt)
    else
        prompt;
    defer if (format_prompt and formatted.ptr != prompt.ptr) allocator.free(formatted);
    if (g_debug) dbg("formatted prompt ({d} bytes): [{s}]", .{ formatted.len, formatted });

    const text_token_ids = switch (tok_kind) {
        .spm => tok.encodeSpm(formatted),
        .spm_no_dummy => tok.encodeSpmNoDummy(formatted),
        .bpe => tok.encode(formatted),
    } catch {
        eprint("Error: failed to encode prompt (tokenizer may not support this input)\n", .{});
        return null;
    };
    defer allocator.free(text_token_ids);
    dbg("encoded {d} tokens, tok_kind={s}", .{ text_token_ids.len, @tagName(tok_kind) });

    // Inject image placeholder token IDs into the token array when an image
    // is attached. The image tokens (start + pad*N + end) are spliced in
    // after the user_prefix tokens in the formatted prompt. The model's
    // forward() detects these pad tokens and replaces their embeddings with
    // visual embeddings from the vision encoder.
    var injected_token_ids: ?[]u32 = null;
    defer if (injected_token_ids) |ids| allocator.free(ids);

    const token_ids: []const u32 = if (n_visual_tokens > 0 and img_tokens != null) blk: {
        // Find insertion point: right after the user_prefix tokens.
        const prefix_tokens = switch (tok_kind) {
            .spm => tok.encodeSpm(template.user_prefix),
            .spm_no_dummy => tok.encodeSpmNoDummy(template.user_prefix),
            .bpe => tok.encode(template.user_prefix),
        } catch break :blk text_token_ids;
        defer allocator.free(prefix_tokens);

        const insert_pos: usize = chat_tmpl_mod.findImageInsertPos(text_token_ids, prefix_tokens);

        // Use injectImageTokens which handles architecture-specific wrapping:
        // Gemma 4 (start=end=pad): just pad×N
        // Qwen 3.5 (distinct start/end): [start, pad×N, end]
        const result = chat_tmpl_mod.injectImageTokens(
            allocator,
            text_token_ids,
            insert_pos,
            img_tokens.?,
            n_visual_tokens,
        ) catch break :blk text_token_ids;

        injected_token_ids = result;
        dbg("injected {d} image tokens at pos {d}, total {d}", .{ n_visual_tokens, insert_pos, result.len });
        if (insert_pos >= 3) dbg("  before: [{d},{d},{d}]", .{ result[insert_pos - 3], result[insert_pos - 2], result[insert_pos - 1] });
        break :blk injected_token_ids.?;
    } else text_token_ids;

    // Build prefill array: BOS (if needed) + prompt tokens
    const prefill_start = milliTimestamp(g_io);
    if (!g_quiet and token_ids.len > prefill_progress_threshold) {
        display.showPrefillStart(token_ids.len);
    }

    var prefill_buf: ?[]u32 = null;
    defer if (prefill_buf) |ids| allocator.free(ids);

    const prefill_toks: []const u32 = blk: {
        if (tok.bos_token_id > 0 and !skip_bos and token_ids.len > 0) {
            var all = allocator.alloc(u32, token_ids.len + 1) catch {
                eprint("Error: out of memory for prefill buffer\n", .{});
                break :blk token_ids;
            };
            all[0] = tok.bos_token_id;
            @memcpy(all[1..], token_ids);
            prefill_buf = all;
            break :blk all;
        } else if (tok.bos_token_id > 0 and !skip_bos) {
            _ = mdl.forward(tok.bos_token_id) catch |e| {
                eprint("Error: BOS token forward failed: {}\n", .{e});
                return null;
            };
            break :blk token_ids;
        } else {
            break :blk token_ids;
        }
    };

    var first_gen_token: u32 = 0;
    if (prefill_toks.len > 0) {
        dbg("entering batched prefill, {d} tokens", .{prefill_toks.len});
        first_gen_token = mdl.prefill(prefill_toks) catch |e| {
            eprint("Error: prefill failed: {}\n", .{e});
            return null;
        };
        dbg("prefill done in {d}ms", .{elapsedMs(prefill_start)});
    }
    const prefill_ms = elapsedMs(prefill_start);
    if (!g_quiet and prefill_toks.len > prefill_progress_threshold) {
        display.clearPrefillProgress();
    }

    // Apply sampling to the first generated token (from prefill's last forward call)
    const use_sampling = cli.temperature > 0;
    const use_repeat_penalty = cli.repeat_penalty != 1.0;
    var prng = std.Random.Xoshiro256.init(cli.seed);
    if (use_sampling and token_ids.len > 0) {
        // No recent tokens yet for the first generated token — repeat penalty
        // will be applied starting from the generation loop below.
        first_gen_token = math_ops.sampleToken(mdl.getLogits(), cli.temperature, cli.top_k, cli.top_p, prng.random());
    }

    // Generate — stream tokens to stdout immediately.
    // Decode in small batches to balance responsiveness vs alloc count.
    // Stop early if the model enters a repetitive loop (same token 6+ times).
    const gen_start = milliTimestamp(g_io);
    var last = first_gen_token;
    var token_count: u32 = 0;
    var gen_ids_buf: [gen_ids_buf_size]u32 = undefined;
    var batch_start: u32 = 0;
    var repeat_count: u32 = 0;
    var prev_token: u32 = 0;
    var started_output = false;
    const batch_size: u32 = if (g_tty) tty_batch_size else pipe_batch_size;

    // Handle first generated token (from prefill's last forward call)
    const first_is_eog = token_ids.len > 0 and isEogToken(first_gen_token, eog);
    var hit_eog = first_is_eog;
    if (!first_is_eog and token_ids.len > 0) {
        gen_ids_buf[0] = first_gen_token;
        token_count = 1;
        prev_token = first_gen_token;
        repeat_count = 1;
    }

    for (0..cli.max_tokens -| 1) |gi| {
        if (first_is_eog or token_ids.len == 0) break;
        var next = mdl.forward(last) catch |e| {
            eprint("Error: generation failed at token {d}: {}\n", .{ gi + 1, e });
            break;
        };
        // Apply repeat penalty to logits for recently generated tokens
        const logits = mdl.getLogits();
        if (use_repeat_penalty and token_count > 0) {
            math_ops.applyRepeatPenalty(logits, gen_ids_buf[0..token_count], cli.repeat_penalty);
        }
        if (use_sampling) {
            next = math_ops.sampleToken(logits, cli.temperature, cli.top_k, cli.top_p, prng.random());
        } else if (use_repeat_penalty and token_count > 0) {
            // Greedy decoding with repeat penalty: re-argmax after penalty
            next = math_ops.argmax(logits);
        }
        dbg("gen step {d}: token={d}", .{ gi, next });
        if (isEogToken(next, eog)) {
            hit_eog = true;
            break;
        }
        if (token_count >= gen_ids_buf.len) break;
        gen_ids_buf[token_count] = next;
        last = next;
        token_count += 1;

        // Repetition detection — stop if same token repeats 6+ times
        if (next == prev_token) {
            repeat_count += 1;
            if (repeat_count >= repeat_halt_threshold) break;
        } else {
            repeat_count = 1;
            prev_token = next;
        }

        // Stream batches — small batches for TTY (responsive), larger for pipes (efficient)
        if (token_count - batch_start >= batch_size) {
            if (display.mode != .json) {
                flushTokenBatch(tok, tok_kind, allocator, gen_ids_buf[batch_start..@min(token_count, gen_ids_buf.len)], &started_output);
                batch_start = token_count;
            }
        }
    }
    // Flush remaining tokens
    if (display.mode != .json and token_count > batch_start and token_count <= gen_ids_buf.len) {
        flushTokenBatch(tok, tok_kind, allocator, gen_ids_buf[batch_start..token_count], &started_output);
    }
    // Ensure a trailing newline for piped output (not TTY, not JSON)
    if (!g_tty and display.mode != .json and started_output) {
        _ = std.c.write(stdout_file.handle, "\n", 1);
    }
    if (hit_eog and g_verbose) print("\n[EOG]\n", .{});
    const gen_ms = elapsedMs(gen_start);

    // Decode full response text for return value (skip if caller doesn't need it)
    const response_text: ?[]u8 = if (token_count > 0 and (need_response or display.mode == .json))
        switch (tok_kind) {
            .spm, .spm_no_dummy => tok.decodeSpm(gen_ids_buf[0..token_count]) catch null,
            .bpe => tok.decode(gen_ids_buf[0..token_count]) catch null,
        }
    else
        null;

    // JSON output — decode all tokens at once and print structured result
    if (display.mode == .json) {
        const stats = display_mod.GenStats{
            .token_count = token_count,
            .gen_ms = gen_ms,
            .prefill_token_count = @intCast(token_ids.len),
            .prefill_ms = prefill_ms,
        };
        display.printJsonPrompt(minfo, response_text orelse "", stats);
        return response_text; // Don't print stats separately
    }

    // Stats
    if (show_stats and token_count > 0) {
        const stats = display_mod.GenStats{
            .token_count = token_count,
            .gen_ms = gen_ms,
            .prefill_token_count = @intCast(token_ids.len),
            .prefill_ms = prefill_ms,
        };
        print("\n", .{});
        display.printStats(stats);
    }
    return response_text;
}

/// Decode a batch of token IDs to text and write to stdout.
/// Skips a single leading newline on the first batch (common model artifact).
fn flushTokenBatch(tok: *BpeTokenizer, tok_kind: TokenizerKind, allocator: std.mem.Allocator, batch: []const u32, started: *bool) void {
    const decoded = switch (tok_kind) {
        .spm, .spm_no_dummy => tok.decodeSpm(batch) catch return,
        .bpe => tok.decode(batch) catch return,
    };
    defer allocator.free(decoded);
    var text: []const u8 = decoded;
    if (!started.* and text.len > 0 and text[0] == '\n') {
        text = text[1..];
    }
    if (text.len > 0) started.* = true;
    _ = std.c.write(stdout_file.handle, text.ptr, text.len);
}

test {
    // Force test discovery for all modules with test blocks.
    // Zig 0.15 uses lazy test discovery — files imported at the top level
    // but not referenced by any test block are silently excluded.
    _ = @import("cli.zig");
    _ = @import("display.zig");
    _ = @import("ops/split_attention.zig");
    _ = @import("arch.zig");
    _ = @import("perf.zig");
    _ = @import("recipe.zig");
    _ = @import("chat_template.zig");
    _ = @import("pull.zig");
    _ = @import("calibrate.zig");
    _ = @import("image.zig");
    _ = @import("thread_pool.zig");
    _ = @import("ops/kv_quant.zig");
    _ = @import("ops/quant.zig");
    _ = @import("ops/math.zig");
    _ = @import("ops/attention.zig");
    _ = @import("ops/kv_evict.zig");
    _ = @import("ops/ssm.zig");
    _ = @import("ops/mlx.zig");
    _ = @import("format/format.zig");
    _ = @import("format/gguf.zig");
    _ = @import("format/safetensors.zig");
    _ = @import("tokenizer/bpe.zig");
    _ = @import("tokenizer/tokenizer.zig");
    _ = @import("server/server.zig");
    _ = @import("server/json.zig");
    _ = @import("server/rate_limiter.zig");
    _ = @import("server/metrics.zig");
    _ = @import("server/scheduler.zig");
    _ = @import("kvcache/block_allocator.zig");
    _ = @import("kvcache/manager.zig");
    _ = @import("kvcache/tiered.zig");
    _ = @import("models/model.zig");
    _ = @import("models/gemma4.zig");
    _ = @import("models/nemotron_nano.zig");
    _ = @import("models/nemotron_h.zig");
    _ = @import("models/vision.zig");
    _ = @import("backend/cpu.zig");
    _ = @import("backend/metal.zig");
    _ = @import("backend/vulkan.zig");
    _ = @import("backend/kernels/cpu/activation.zig");
    _ = @import("backend/kernels/cpu/elementwise.zig");
    _ = @import("backend/kernels/cpu/embedding.zig");
    _ = @import("backend/kernels/cpu/norm.zig");
    _ = @import("backend/kernels/cpu/rope.zig");
    _ = @import("backend/kernels/cpu/sdpa.zig");
    _ = @import("backend/kernels/cpu/softmax.zig");
    _ = @import("backend/kernels/cpu/gemv_bf16.zig");
    _ = @import("backend/kernels/cpu/gemv_f16.zig");
    _ = @import("backend/kernels/cpu/gemv_f32.zig");
    _ = @import("backend/kernels/cpu/gemv_fp4.zig");
    _ = @import("backend/kernels/cpu/gemv_iq4.zig");
    _ = @import("backend/kernels/cpu/gemv_q_small.zig");
    _ = @import("backend/kernels/cpu/gemv_q4_0.zig");
    _ = @import("backend/kernels/cpu/gemv_q4_k.zig");
    _ = @import("backend/kernels/cpu/gemv_q5_k.zig");
    _ = @import("backend/kernels/cpu/gemv_q6_k.zig");
    _ = @import("backend/kernels/cpu/gemv_q8_0.zig");
    _ = @import("spec/spec_decode.zig");
    _ = @import("spec/ddtree.zig");
    _ = @import("backend/kernels/cpu/sdpa_tree.zig");
}

test "cpu backend rms_norm via tagged union dispatch" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    var bs = BackendState{};
    bs.init(std.testing.allocator, .cpu, threaded.io());
    defer if (bs.pool) |*p| p.deinit();
    const be = bs.be;
    var input = [_]f32{ 1, 2, 3, 4 };
    var weight = [_]f32{ 1, 1, 1, 1 };
    var output_buf: [4]f32 = undefined;
    be.rmsNorm(&input, &weight, &output_buf, 4, 1e-6);
    // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    const rms = @sqrt(@as(f32, 7.5));
    for (0..4) |i| {
        const expected = @as(f32, @floatFromInt(i + 1)) / rms;
        try std.testing.expectApproxEqAbs(expected, output_buf[i], 1e-4);
    }
}

test "cpu backend softmax via tagged union dispatch" {
    var threaded2 = std.Io.Threaded.init(std.testing.allocator, .{});
    var bs = BackendState{};
    bs.init(std.testing.allocator, .cpu, threaded2.io());
    defer if (bs.pool) |*p| p.deinit();
    const be = bs.be;
    var data = [_]f32{ 1.0, 2.0, 3.0 };
    be.softmax(&data, 3);
    // softmax should sum to 1.0
    const sum = data[0] + data[1] + data[2];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);
    // Largest input should have largest probability
    try std.testing.expect(data[2] > data[1]);
    try std.testing.expect(data[1] > data[0]);
    // Verify approximate expected values: softmax([1,2,3])
    // exp(1-3)=exp(-2), exp(2-3)=exp(-1), exp(3-3)=exp(0)=1
    // Z = exp(-2) + exp(-1) + 1 ≈ 0.1353 + 0.3679 + 1.0 = 1.5032
    try std.testing.expectApproxEqAbs(@as(f32, 0.0900), data[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2447), data[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6652), data[2], 1e-4);
}

test "cpu backend silu via tagged union dispatch" {
    var threaded3 = std.Io.Threaded.init(std.testing.allocator, .{});
    var bs = BackendState{};
    bs.init(std.testing.allocator, .cpu, threaded3.io());
    defer if (bs.pool) |*p| p.deinit();
    const be = bs.be;
    var input = [_]f32{ 0.0, 1.0, -1.0 };
    var output: [3]f32 = undefined;
    be.silu(&input, &output, 3);
    // SiLU(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 0.001);
    // SiLU(1) = 1 * sigmoid(1) ≈ 0.731
    try std.testing.expectApproxEqAbs(@as(f32, 0.731), output[1], 0.01);
    // SiLU(-1) = -1 * sigmoid(-1) ≈ -0.269
    try std.testing.expectApproxEqAbs(@as(f32, -0.269), output[2], 0.01);
}
