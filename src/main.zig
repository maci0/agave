//! CLI entry point for the Agave LLM inference engine.
//! Parses command-line arguments, loads a GGUF model or SafeTensors directory,
//! auto-detects the architecture, and runs interactive generation, one-shot prompts,
//! or an HTTP server.

const std = @import("std");
const clap = @import("clap");
const build_options = @import("build_options");

const backend_mod = @import("backend/backend.zig");
const format_mod = @import("format/format.zig");
const model_mod = @import("models/model.zig");
const tok_mod = @import("tokenizer/tokenizer.zig");
const server = @import("server.zig");
const display_mod = @import("display.zig");
const Display = display_mod.Display;

const Backend = backend_mod.Backend;
const CpuBackend = backend_mod.CpuBackend;
const ThreadPool = @import("thread_pool.zig").ThreadPool;
const Format = format_mod.Format;
const GGUFFile = format_mod.GGUFFile;
const SafeTensorsDir = format_mod.SafeTensorsDir;
const Model = model_mod.Model;
const Gemma3Model = model_mod.Gemma3Model;
const Qwen35Model = model_mod.Qwen35Model;
const GptOssModel = model_mod.GptOssModel;
const NemotronHModel = model_mod.NemotronHModel;
const Glm4Model = model_mod.Glm4Model;
const NemotronNanoModel = model_mod.NemotronNanoModel;
const BpeTokenizer = tok_mod.BpeTokenizer;
const LineEditor = @import("readline.zig").LineEditor;
const KvQuantType = @import("ops/kv_quant.zig").KvQuantType;
const math_ops = @import("ops/math.zig");
const TieredKvCache = @import("kvcache/tiered.zig").TieredKvCache;

const stdout = std.fs.File.stdout();
const stderr = std.fs.File.stderr();

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
/// Matches GPU SDPA kernel limit (threadgroup memory on Metal).
const default_ctx_size: u32 = 4096;
/// Minimum prompt tokens before showing prefill progress indicator.
const prefill_progress_threshold: usize = 50;
/// Default free RAM estimate when platform detection is not implemented (16 GB).
const default_free_ram: usize = 16 * 1024 * 1024 * 1024;
/// Default tiered KV cache RAM budget when unspecified (GB).
const default_ram_budget_gb: usize = 16;
/// Default tiered KV cache SSD budget when unspecified (GB).
const default_ssd_budget_gb: usize = 10;
const max_eog_ids = arch_mod.max_eog_ids;
const gemma_fallback_eos = arch_mod.gemma_fallback_eos;
const default_fallback_eos = arch_mod.default_fallback_eos;
const default_bos_id = arch_mod.default_bos_id;

// ── Output control ──────────────────────────────────────────────

var g_color: bool = true;
var g_quiet: bool = false;
var g_tty: bool = true;
var g_debug: bool = false;
var g_verbose: bool = false;

fn print(comptime fmt: []const u8, args: anytype) void {
    var buf: [print_buf_size]u8 = undefined;
    stdout.writeAll(std.fmt.bufPrint(&buf, fmt, args) catch return) catch {};
}

fn eprint(comptime fmt: []const u8, args: anytype) void {
    var buf: [print_buf_size]u8 = undefined;
    stderr.writeAll(std.fmt.bufPrint(&buf, fmt, args) catch return) catch {};
}

/// Debug output. Only printed when --debug is active.
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (!g_debug) return;
    eprint("[dbg] " ++ fmt ++ "\n", args);
}

/// Print a fatal error message to stderr and exit with status 1.
fn fatalExit(comptime msg: []const u8) noreturn {
    eprint("Fatal: " ++ msg ++ "\n", .{});
    std.process.exit(1);
}

/// Detect free system RAM in bytes.
/// Platform-specific: sysctl on macOS, /proc/meminfo on Linux.
/// Placeholder: returns 16GB default for now.
/// Future: implement platform-specific detection via sysctl.vm.stats.vm.v_free_count
/// on macOS and MemAvailable from /proc/meminfo on Linux.
fn detectFreeRam() usize {
    // TODO: Platform-specific implementation
    // macOS: sysctl vm.stats.vm.v_free_count + vm_page_size
    // Linux: read /proc/meminfo, extract MemAvailable
    return default_free_ram;
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

/// Progress bar width for warming display.
const warmup_bar_width: u32 = 30;

/// Preload all mmap'd model data into RAM with progress bar.
fn preloadModel(gguf: ?*GGUFFile, st: ?*SafeTensorsDir, quiet: bool, tty: bool, total_bytes: usize) u64 {
    const start = std.time.milliTimestamp();
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
/// and prints a progress bar to stderr on each 1% increment.
fn preloadRegionProgress(data: []align(std.heap.page_size_min) const u8, loaded: *usize, total_bytes: usize, fsize: display_mod.FormattedSize) void {
    const MADV = std.posix.MADV;
    std.posix.madvise(@alignCast(@constCast(data.ptr)), data.len, MADV.SEQUENTIAL) catch {};

    const page_size = std.heap.page_size_min;
    // Report progress every ~1% of total or every 256 pages, whichever is smaller
    const report_interval = @max(total_bytes / 100, page_size * 256);
    var last_report: usize = loaded.*;
    var offset: usize = 0;
    while (offset < data.len) : (offset += page_size) {
        _ = @as(*const volatile u8, @ptrCast(&data[offset])).*;
        loaded.* += page_size;

        if (loaded.* - last_report >= report_interval or offset + page_size >= data.len) {
            last_report = loaded.*;
            const pct: u32 = if (total_bytes > 0) @intCast(@min(loaded.* * 100 / total_bytes, 100)) else 100;
            const filled: u32 = @intCast(@min(@as(u64, pct) * warmup_bar_width / 100, warmup_bar_width));
            var buf: [256]u8 = undefined;
            var pos: usize = 0;
            const append = struct {
                fn f(b: []u8, p: *usize, s: []const u8) void {
                    const n = @min(s.len, b.len - p.*);
                    @memcpy(b[p.*..][0..n], s[0..n]);
                    p.* += n;
                }
            }.f;
            append(&buf, &pos, "\r\x1b[2m\xe2\x96\x90"); // CR + dim + ▐
            for (0..warmup_bar_width) |i| {
                if (i < filled) {
                    append(&buf, &pos, "\xe2\x96\x88"); // █
                } else {
                    append(&buf, &pos, "\xe2\x96\x91"); // ░
                }
            }
            append(&buf, &pos, "\xe2\x96\x8c "); // ▌ + space
            const text = std.fmt.bufPrint(buf[pos..], "loading {d:.1} {s} ({d}%)\x1b[0m", .{ fsize.val, fsize.unit, pct }) catch "";
            pos += text.len;
            stderr.writeAll(buf[0..pos]) catch {};
        }
    }

    std.posix.madvise(@alignCast(@constCast(data.ptr)), data.len, MADV.RANDOM) catch {};
}

// ── Architecture detection ───────────────────────────────────────

const ChatTemplate = @import("chat_template.zig").ChatTemplate;
const Recipe = @import("recipe.zig").Recipe;
const arch_mod = @import("arch.zig");
const Arch = arch_mod.Arch;
const TokenizerKind = arch_mod.TokenizerKind;

// ── CLI definition ───────────────────────────────────────────────

const cli_params = clap.parseParamsComptime(
    \\-h, --help                 Show this help message and exit.
    \\-v, --version              Print version and exit.
    \\-s, --serve                Start HTTP server instead of interactive mode.
    \\-q, --quiet                Suppress banner and stats (output only).
    \\-p, --port <u16>           Server port (default: 49453).
    \\-n, --max-tokens <u32>     Maximum tokens to generate (default: 512).
    \\-t, --temperature <str>    Sampling temperature, 0 = greedy (default: 0).
    \\    --top-p <str>          Nucleus sampling threshold (default: 1.0).
    \\    --top-k <u32>          Top-k sampling, 0 = disabled (default: 0).
    \\    --repeat-penalty <str> Repetition penalty (default: 1.0).
    \\    --system <str>         System prompt for chat formatting.
    \\    --backend <str>        Compute backend: auto, cpu, metal, vulkan, cuda, rocm (default: auto).
    \\    --ctx-size <u32>       Context window size (default: 4096, 0 = model max).
    \\    --seed <u64>           Random seed for sampling (default: random).
    \\    --kv-type <str>        KV cache quantization: f32, f16, q8_0, int8, fp8, nvfp4 (default: f16).
    \\    --kv-tiers <str>       Enable tiered KV cache: vram, vram+ram, vram+ram+ssd (default: vram).
    \\    --kv-ram-budget <str>  RAM tier budget in GB (default: 50% of free RAM).
    \\    --kv-ssd-path <str>    SSD tier file path (enables SSD tier if set).
    \\    --kv-ssd-budget <str>  SSD tier budget in GB (default: 10).
    \\    --no-color             Disable colored output.
    \\    --verbose              Show technical details (params, load times, EOG).
    \\    --allow-cpu-fallback   Allow GPU backends to fall back to CPU for unsupported ops.
    \\    --debug                Enable debug logging (token IDs, layer timing).
    \\    --json                 Output results as JSON (implies --quiet).
    \\    --model-info           Print model metadata and exit (combine with --json).
    \\    --profile              Profile per-op timing (halves throughput).
    \\    --mmap                 Use lazy mmap instead of eagerly paging weights into RAM.
    \\<str>...
    \\
);

/// Requested compute backend from CLI.
const BackendChoice = enum { auto, cpu, metal, vulkan, cuda, rocm };

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
    kv_type: KvQuantType,
    seed: u64,
    // Tiered KV cache CLI options
    kv_tiers: ?[]const u8 = null,
    kv_ram_budget: ?[]const u8 = null,
    kv_ssd_path: ?[]const u8 = null,
    kv_ssd_budget: ?[]const u8 = null,
    allow_cpu_fallback: bool,
    debug: bool,
    json: bool,
    model_info: bool,
    profile: bool,
    use_mmap: bool,
    /// Tracks which sampling/generation args the user explicitly set on the CLI.
    user_set: Recipe.Overrides = .{},
};

fn parseCli(allocator: std.mem.Allocator) ?CliArgs {
    var diag = clap.Diagnostic{};
    var res = clap.parse(clap.Help, &cli_params, clap.parsers.default, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |e| {
        diag.reportToFile(stderr, e) catch {};
        eprint("Run 'agave --help' for usage information.\n", .{});
        std.process.exit(1);
    };

    if (res.args.help != 0) {
        printUsage();
        res.deinit();
        return null;
    }

    if (res.args.version != 0) {
        if (std.posix.isatty(std.fs.File.stdout().handle)) {
            print("\xf0\x9f\x8c\xb5 agave {s}\n", .{version});
        } else {
            print("agave {s}\n", .{version});
        }
        res.deinit();
        return null;
    }

    // Auto-detect TTY: disable color when stdout is not a terminal
    g_tty = std.posix.isatty(stdout.handle);
    g_color = res.args.@"no-color" == 0 and g_tty;
    g_quiet = res.args.quiet != 0;
    g_debug = res.args.debug != 0;
    g_verbose = res.args.verbose != 0 or g_debug;

    const json_mode = res.args.json != 0;
    if (json_mode) {
        g_quiet = true;
    }

    const positionals = res.positionals[0];
    if (positionals.len == 0) {
        eprint("Error: missing model path\n", .{});
        eprint("Usage: agave <model.gguf|model-dir/> [prompt]\n", .{});
        eprint("Run 'agave --help' for more information.\n", .{});
        std.process.exit(1);
    }

    const backend_choice: BackendChoice = blk: {
        const be_str = res.args.backend orelse "auto";
        if (std.mem.eql(u8, be_str, "cpu")) break :blk .cpu;
        if (std.mem.eql(u8, be_str, "metal")) break :blk .metal;
        if (std.mem.eql(u8, be_str, "vulkan")) break :blk .vulkan;
        if (std.mem.eql(u8, be_str, "cuda")) break :blk .cuda;
        if (std.mem.eql(u8, be_str, "rocm")) break :blk .rocm;
        if (std.mem.eql(u8, be_str, "auto")) break :blk .auto;
        eprint("Error: unknown backend '{s}'\n", .{be_str});
        eprint("  Valid options: auto, cpu, metal, vulkan, cuda, rocm\n", .{});
        std.process.exit(1);
    };

    return .{
        .model_path = if (positionals.len > 0) positionals[0] else "",
        .prompt = if (positionals.len > 1) positionals[1] else null,
        .serve = res.args.serve != 0,
        .port = res.args.port orelse default_port,
        .max_tokens = res.args.@"max-tokens" orelse default_max_tokens,
        .temperature = parseF32(res.args.temperature, "temperature") orelse 0.0,
        .top_p = parseF32(res.args.@"top-p", "top-p") orelse 1.0,
        .top_k = res.args.@"top-k" orelse 0,
        .repeat_penalty = parseF32(res.args.@"repeat-penalty", "repeat-penalty") orelse 1.0,
        .system_prompt = res.args.system,
        .backend_choice = backend_choice,
        .ctx_size = res.args.@"ctx-size" orelse 0,
        .seed = res.args.seed orelse @as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())))),
        .kv_type = blk: {
            const kv_str = res.args.@"kv-type" orelse break :blk .f16;
            break :blk KvQuantType.fromString(kv_str) orelse {
                eprint("Error: unknown KV type '{s}'\n", .{kv_str});
                eprint("  Valid options: f32, f16, q8_0, int8, fp8, nvfp4\n", .{});
                std.process.exit(1);
            };
        },
        .kv_tiers = res.args.@"kv-tiers",
        .kv_ram_budget = res.args.@"kv-ram-budget",
        .kv_ssd_path = res.args.@"kv-ssd-path",
        .kv_ssd_budget = res.args.@"kv-ssd-budget",
        .allow_cpu_fallback = res.args.@"allow-cpu-fallback" != 0,
        .debug = res.args.debug != 0,
        .json = json_mode,
        .model_info = res.args.@"model-info" != 0,
        .profile = res.args.profile != 0,
        .use_mmap = res.args.mmap != 0,
        .user_set = .{
            .temperature = res.args.temperature != null,
            .top_p = res.args.@"top-p" != null,
            .top_k = res.args.@"top-k" != null,
            .repeat_penalty = res.args.@"repeat-penalty" != null,
            .max_tokens = res.args.@"max-tokens" != null,
            .ctx_size = res.args.@"ctx-size" != null,
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
        \\OPTIONS:
        \\  -h, --help             Show this help message
        \\  -v, --version          Print version
        \\  -q, --quiet            Suppress banner and stats (raw output only)
        \\  -s, --serve            Start HTTP server (OpenAI-compatible API)
        \\  -p, --port <PORT>      Server port [default: 49453]
        \\  -n, --max-tokens <N>   Maximum tokens to generate [default: 512]
        \\  -t, --temperature <T>  Sampling temperature, 0 = greedy [default: 0]
        \\      --top-p <P>        Nucleus sampling threshold [default: 1.0]
        \\      --top-k <K>        Top-k sampling, 0 = disabled [default: 0]
        \\      --repeat-penalty <R>  Repetition penalty [default: 1.0]
        \\      --system <TEXT>     System prompt for chat formatting
        \\      --backend <BE>     Compute backend: auto, cpu, metal, vulkan, cuda, rocm [default: auto]
        \\      --ctx-size <N>     Context window size [default: 4096, 0 = model max]
        \\      --kv-type <TYPE>   KV cache quantization: f32, f16, q8_0, int8, fp8, nvfp4 [default: f16]
        \\      --verbose          Show technical details (params, load times, EOG)
        \\      --no-color         Disable colored output
        \\      --allow-cpu-fallback  Allow GPU backends to fall back to CPU for unsupported ops
        \\      --debug            Enable debug logging (token IDs, layer timing)
        \\      --json             Output results as JSON (implies --quiet)
        \\      --model-info       Print model metadata and exit (combine with --json)
        \\      --profile          Profile per-op timing (halves throughput)
        \\      --mmap             Use lazy mmap instead of eagerly paging weights into RAM
        \\
        \\EXAMPLES:
        \\  agave model.gguf                          Interactive REPL
        \\  agave model.gguf "What is 2+2?"           Single prompt
        \\  agave model.gguf -q "Hello" > out.txt     Pipe output (no banner)
        \\  agave model.gguf --serve --port 3000       HTTP server
        \\  agave model.gguf -t 0.7 --top-p 0.9 "Tell me a joke"
        \\  agave model.gguf --backend cpu "Hello"     Force CPU backend
        \\  agave ./glm-4-9b/ "Hello"                 Load SafeTensors directory
        \\  echo "Explain TCP" | agave model.gguf      Pipe prompt from stdin
        \\  agave model.gguf --json "Hello"            JSON output with stats
        \\  agave model.gguf --json --model-info       Model metadata as JSON
        \\
        \\SUPPORTED ARCHITECTURES:
        \\  gemma3, qwen35, gpt-oss, nemotron-h, nemotron-nano, glm4
        \\
        \\REPL COMMANDS:
        \\  /clear                 Clear conversation and KV cache (stay in chat)
        \\  /stats                 Toggle generation stats
        \\  /verbose               Toggle technical details
        \\  /model                 Show model info
        \\  /quit, /exit, /q       Exit interactive mode
        \\  /help                  Show available commands
        \\  Ctrl+C                 Cancel input (double-tap to quit)
        \\  Ctrl+D                 Quit (on empty line)
        \\  Ctrl+L                 Clear screen
        \\  Ctrl+R                 Reverse search history
        \\  Up/Down                Navigate history
        \\
    ;
    stdout.writeAll(usage) catch {};
}

// ── Formatting helpers ───────────────────────────────────────────

fn elapsedMs(start: i64) u64 {
    return @intCast(@max(std.time.milliTimestamp() - start, 0));
}

const getQuantName = Format.getQuantName;

const EogTokens = struct { ids: [max_eog_ids]u32, len: usize };

/// Collect additional EOS/EOG token IDs from GGUF metadata.
fn getEogTokens(fmt_iface: Format, primary_eos: u32) EogTokens {
    var result: EogTokens = .{ .ids = undefined, .len = 0 };
    result.ids[0] = primary_eos;
    result.len = 1;
    // Check for EOG token arrays (e.g., Gemma 3 stores multiple EOG IDs)
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

// ── Backend initialization ───────────────────────────────────────

/// Holds the mutable backend storage alongside the tagged-union interface.
/// The backend variables must outlive the `Backend` union (which holds pointers
/// to them), so they are bundled together and kept on the caller's stack.
/// No backend `deinit()` is called — process exit releases all resources.
/// The thread pool is the sole exception (cleaned up via defer in `main()`).
const BackendState = struct {
    const builtin = @import("builtin");

    cpu_be: CpuBackend = .{},
    metal_be: if (builtin.os.tag == .macos) backend_mod.MetalBackend else void = undefined,
    vulkan_be: backend_mod.VulkanBackend = undefined,
    cuda_be: backend_mod.CudaBackend = undefined,
    rocm_be: backend_mod.RocmBackend = undefined,
    pool: ?ThreadPool = null,
    be: Backend = undefined,
    name: []const u8 = "CPU",

    /// Initialize the requested compute backend, with automatic fallback.
    /// Must be called on a stack-allocated `BackendState` — the `be` field
    /// stores pointers into the struct's own backend fields.
    fn init(self: *BackendState, allocator: std.mem.Allocator, backend_choice: BackendChoice) void {
        // Create thread pool for CPU GEMV parallelism.
        // n_workers = CPU count - 1 (main thread also participates).
        const cpu_count = std.Thread.getCpuCount() catch 1;
        const n_workers = if (cpu_count > 1) cpu_count - 1 else 0;
        self.pool = ThreadPool.init(n_workers);
        self.pool.?.spawn();
        self.cpu_be.pool = &self.pool.?;
        self.be = blk: {
            switch (backend_choice) {
                .cpu => {
                    if (comptime build_options.enable_cpu) {
                        break :blk .{ .cpu = &self.cpu_be };
                    } else {
                        fatalExit("CPU backend disabled at build time");
                    }
                },
                .metal => {
                    if (comptime build_options.enable_metal and builtin.os.tag == .macos) {
                        self.metal_be = backend_mod.MetalBackend.init(allocator) catch |err| {
                            if (comptime build_options.enable_cpu) {
                                eprint("Warning: Metal unavailable ({s}), falling back to CPU\n", .{@errorName(err)});
                                break :blk .{ .cpu = &self.cpu_be };
                            } else {
                                fatalExit("Metal unavailable and CPU backend disabled");
                            }
                        };
                        self.metal_be.pool = &self.pool.?;
                        self.name = "Metal";
                        break :blk .{ .metal = &self.metal_be };
                    } else {
                        if (comptime build_options.enable_cpu) {
                            eprint("Warning: Metal not available on this platform\n", .{});
                            break :blk .{ .cpu = &self.cpu_be };
                        } else {
                            fatalExit("Metal not available and CPU backend disabled");
                        }
                    }
                },
                .vulkan => {
                    if (comptime build_options.enable_vulkan) {
                        self.vulkan_be = backend_mod.VulkanBackend.init(allocator) catch {
                            if (comptime build_options.enable_cpu) {
                                eprint("Warning: Vulkan unavailable, falling back to CPU\n", .{});
                                break :blk .{ .cpu = &self.cpu_be };
                            } else {
                                fatalExit("Vulkan unavailable and CPU backend disabled");
                            }
                        };
                        self.name = "Vulkan";
                        break :blk .{ .vulkan = &self.vulkan_be };
                    } else {
                        fatalExit("Vulkan backend disabled at build time");
                    }
                },
                .cuda => {
                    if (comptime build_options.enable_cuda) {
                        self.cuda_be = backend_mod.CudaBackend.init(allocator) catch {
                            if (comptime build_options.enable_cpu) {
                                eprint("Warning: CUDA unavailable, falling back to CPU\n", .{});
                                break :blk .{ .cpu = &self.cpu_be };
                            } else {
                                fatalExit("CUDA unavailable and CPU backend disabled");
                            }
                        };
                        self.name = "CUDA";
                        break :blk .{ .cuda = &self.cuda_be };
                    } else {
                        fatalExit("CUDA backend disabled at build time");
                    }
                },
                .rocm => {
                    if (comptime build_options.enable_rocm) {
                        self.rocm_be = backend_mod.RocmBackend.init(allocator) catch {
                            if (comptime build_options.enable_cpu) {
                                eprint("Warning: ROCm unavailable, falling back to CPU\n", .{});
                                break :blk .{ .cpu = &self.cpu_be };
                            } else {
                                fatalExit("ROCm unavailable and CPU backend disabled");
                            }
                        };
                        self.name = "ROCm";
                        break :blk .{ .rocm = &self.rocm_be };
                    } else {
                        fatalExit("ROCm backend disabled at build time");
                    }
                },
                .auto => {
                    // Try Metal (macOS only)
                    if (comptime build_options.enable_metal and builtin.os.tag == .macos) {
                        self.metal_be = backend_mod.MetalBackend.init(allocator) catch {
                            if (comptime build_options.enable_cuda) {
                                self.cuda_be = backend_mod.CudaBackend.init(allocator) catch {
                                    if (comptime build_options.enable_vulkan) {
                                        self.vulkan_be = backend_mod.VulkanBackend.init(allocator) catch {
                                            if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else fatalExit("All GPU backends failed and CPU disabled");
                                        };
                                        self.name = "Vulkan";
                                        break :blk .{ .vulkan = &self.vulkan_be };
                                    }
                                    if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else fatalExit("CUDA failed, no other backends enabled");
                                };
                                self.name = "CUDA";
                                break :blk .{ .cuda = &self.cuda_be };
                            } else if (comptime build_options.enable_vulkan) {
                                self.vulkan_be = backend_mod.VulkanBackend.init(allocator) catch {
                                    if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else fatalExit("Vulkan failed and CPU disabled");
                                };
                                self.name = "Vulkan";
                                break :blk .{ .vulkan = &self.vulkan_be };
                            } else if (comptime build_options.enable_cpu) {
                                break :blk .{ .cpu = &self.cpu_be };
                            } else {
                                fatalExit("No backends enabled");
                            }
                        };
                        self.name = "Metal";
                        break :blk .{ .metal = &self.metal_be };
                    }
                    // Try CUDA
                    if (comptime build_options.enable_cuda) {
                        self.cuda_be = backend_mod.CudaBackend.init(allocator) catch {
                            if (comptime build_options.enable_rocm) {
                                self.rocm_be = backend_mod.RocmBackend.init(allocator) catch {
                                    if (comptime build_options.enable_vulkan) {
                                        self.vulkan_be = backend_mod.VulkanBackend.init(allocator) catch {
                                            if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else fatalExit("All GPU backends failed and CPU disabled");
                                        };
                                        self.name = "Vulkan";
                                        break :blk .{ .vulkan = &self.vulkan_be };
                                    }
                                    if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else fatalExit("All GPU backends failed and CPU disabled");
                                };
                                self.name = "ROCm";
                                break :blk .{ .rocm = &self.rocm_be };
                            }
                            if (comptime build_options.enable_vulkan) {
                                self.vulkan_be = backend_mod.VulkanBackend.init(allocator) catch {
                                    if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else fatalExit("All GPU backends failed and CPU disabled");
                                };
                                self.name = "Vulkan";
                                break :blk .{ .vulkan = &self.vulkan_be };
                            }
                            if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else fatalExit("CUDA failed, no other backends enabled");
                        };
                        self.name = "CUDA";
                        break :blk .{ .cuda = &self.cuda_be };
                    }
                    // Try ROCm
                    if (comptime build_options.enable_rocm) {
                        self.rocm_be = backend_mod.RocmBackend.init(allocator) catch {
                            if (comptime build_options.enable_vulkan) {
                                self.vulkan_be = backend_mod.VulkanBackend.init(allocator) catch {
                                    if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else fatalExit("All GPU backends failed and CPU disabled");
                                };
                                self.name = "Vulkan";
                                break :blk .{ .vulkan = &self.vulkan_be };
                            }
                            if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else fatalExit("ROCm failed, no other backends enabled");
                        };
                        self.name = "ROCm";
                        break :blk .{ .rocm = &self.rocm_be };
                    }
                    // Try Vulkan
                    if (comptime build_options.enable_vulkan) {
                        self.vulkan_be = backend_mod.VulkanBackend.init(allocator) catch {
                            if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else fatalExit("Vulkan failed and CPU disabled");
                        };
                        self.name = "Vulkan";
                        break :blk .{ .vulkan = &self.vulkan_be };
                    }
                    // CPU fallback
                    if (comptime build_options.enable_cpu) {
                        break :blk .{ .cpu = &self.cpu_be };
                    }
                    fatalExit("No backends enabled");
                },
            }
        };
    }
};

// ── Main ─────────────────────────────────────────────────────────

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cli = parseCli(allocator) orelse return;

    // ── Load model format ────────────────────────────────────────
    const load_start = std.time.milliTimestamp();

    // Detect format: directory → SafeTensors, else → GGUF
    const is_dir = blk: {
        var dir = std.fs.cwd().openDir(cli.model_path, .{}) catch break :blk false;
        dir.close();
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
    const quant = getQuantName(fmt);

    var arch = Arch.detect(arch_str) orelse {
        eprint("Error: unsupported architecture '{s}'\n", .{arch_str});
        eprint("  Supported: gemma3, qwen35, gpt-oss, nemotron-h, nemotron-nano, glm4\n", .{});
        std.process.exit(1);
    };

    // SafeTensors Nemotron Nano variant: uses backbone.layers.* tensor naming + NVFP4
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
    bs.init(allocator, cli.backend_choice);
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
    const file_size_bytes: usize = if (gguf_file) |g| g.file_size else if (st_dir) |s| blk: {
        var total: u64 = 0;
        for (s.shard_data) |shard| total += shard.data.len;
        break :blk total;
    } else 0;

    // ── Banner (printed before loading so user sees info immediately) ─
    var disp_info = display_mod.ModelInfo{
        .name = name,
        .arch_name = arch.displayName(),
        .quant = quant,
        .be_name = be_name,
        .n_layers = fmt.getArchU32(arch_str, "block_count") orelse fmt.getMetaU32("num_hidden_layers") orelse 0,
        .n_embed = fmt.getArchU32(arch_str, "embedding_length") orelse fmt.getMetaU32("hidden_size") orelse 0,
        .n_heads = fmt.getArchU32(arch_str, "attention.head_count") orelse fmt.getMetaU32("num_attention_heads") orelse 0,
        .n_kv_heads = fmt.getArchU32(arch_str, "attention.head_count_kv") orelse fmt.getMetaU32("num_key_value_heads") orelse 0,
        .head_dim = fmt.getArchU32(arch_str, "attention.key_length") orelse
            fmt.getMetaU32("head_dim") orelse
            // Compute from n_embed / n_heads when metadata is missing
            if ((fmt.getArchU32(arch_str, "embedding_length") orelse fmt.getMetaU32("hidden_size") orelse 0) > 0 and
                (fmt.getArchU32(arch_str, "attention.head_count") orelse fmt.getMetaU32("num_attention_heads") orelse 0) > 0)
                (fmt.getArchU32(arch_str, "embedding_length") orelse fmt.getMetaU32("hidden_size") orelse 0) /
                    (fmt.getArchU32(arch_str, "attention.head_count") orelse fmt.getMetaU32("num_attention_heads") orelse 1)
            else
                0,
        .ff_dim = fmt.getArchU32(arch_str, "feed_forward_length") orelse fmt.getMetaU32("intermediate_size") orelse 0,
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
    // When neither user nor recipe set ctx_size, cap to default_ctx_size
    // to avoid massive KV cache allocations for models that report very
    // large context lengths (e.g. 128K). --ctx-size 0 = use model's full context.
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
    disp_info.kv_type_name = cli.kv_type.name();
    disp_info.kv_bpe = cli.kv_type.bitsPerElement();

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
        display.printSystemInfo(be_info);
    }

    // ── Preload weights into RAM (after banner so user sees info first) ──
    const warmup_ms: u64 = if (!cli.use_mmap)
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
    const tok_kind: TokenizerKind = if (arch == .gemma3) .spm_no_dummy else if (merges != null) .bpe else .spm;
    dbg("tokenizer: vocab={s}, merges={s}, kind={s}", .{
        if (vocab != null) @as([]const u8, "yes") else @as([]const u8, "null"),
        if (merges != null) @as([]const u8, "yes") else @as([]const u8, "null"),
        @as([]const u8, @tagName(tok_kind)),
    });
    const eos_id = fmt.getMetaU32("tokenizer.ggml.eos_token_id") orelse
        fmt.getMetaU32("eos_token_id") orelse
        if (arch == .gemma3) gemma_fallback_eos else default_fallback_eos;
    const bos_id: u32 = blk: {
        // Explicit BOS token ID from metadata takes priority.
        if (fmt.getMetaU32("tokenizer.ggml.bos_token_id")) |id| break :blk id;
        if (fmt.getMetaU32("bos_token_id")) |id| break :blk id;
        // GPT-2 based tokenizers (Qwen, etc.) don't use BOS by default.
        if (fmt.getMetaStr("tokenizer.ggml.model")) |m| {
            if (std.mem.eql(u8, m, "gpt2")) break :blk 0;
        }
        break :blk default_bos_id;
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
        const stdin_fd = std.fs.File.stdin();
        if (!std.posix.isatty(stdin_fd.handle)) {
            piped_prompt = stdin_fd.readToEndAlloc(allocator, max_stdin_prompt_size) catch |e| blk: {
                if (e == error.StreamTooLong)
                    eprint("Error: piped input exceeds {d} bytes limit\n", .{max_stdin_prompt_size});
                break :blk null;
            };
        }
    }

    // ── Init model ───────────────────────────────────────────────
    const effective_prompt = cli.prompt orelse if (piped_prompt) |p|
        std.mem.trim(u8, p, " \t\r\n")
    else
        null;

    if (cli.serve and effective_prompt != null) {
        eprint("Warning: prompt ignored in server mode (--serve)\n", .{});
    }

    // ── Construct load info ────────────────────────────────────────
    const n_tensors: u64 = if (gguf_file) |g| g.tensor_count else if (st_dir) |s| s.tensorCount() else 0;
    const format_name: []const u8 = if (gguf_file) |g|
        (if (g.version == 3) "GGUF v3" else if (g.version == 2) "GGUF v2" else "GGUF")
    else if (st_dir != null)
        "SafeTensors"
    else
        "";
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

/// Initialize the model and run inference/server/REPL. Returns false on failure.
fn initAndRun(
    arch: Arch,
    allocator: std.mem.Allocator,
    fmt: Format,
    be: Backend,
    tok: *BpeTokenizer,
    cli: *const CliArgs,
    tok_kind: TokenizerKind,
    eog: anytype,
    effective_prompt: ?[]const u8,
    minfo: display_mod.ModelInfo,
    display: Display,
    pool: ?*ThreadPool,
    load_info_in: display_mod.LoadInfo,
) bool {
    switch (arch) {
        inline .gemma3, .qwen35, .gpt_oss, .nemotron_h, .nemotron_nano, .glm4 => |a| {
            if (comptime !a.isEnabled()) unreachable; // Validated in main()
            const ModelType = switch (a) {
                .gemma3 => Gemma3Model,
                .qwen35 => Qwen35Model,
                .gpt_oss => GptOssModel,
                .nemotron_h => NemotronHModel,
                .nemotron_nano => NemotronNanoModel,
                .glm4 => Glm4Model,
            };
            // Initialize optional tiered KV cache from CLI flags.
            var tiered_cache_storage: ?TieredKvCache = null;
            defer if (tiered_cache_storage) |*tc| tc.deinit();

            if (cli.kv_tiers) |tiers_str| {
                const has_ram = std.mem.indexOf(u8, tiers_str, "ram") != null;
                const has_ssd = std.mem.indexOf(u8, tiers_str, "ssd") != null;

                const gib: usize = 1024 * 1024 * 1024;
                const ram_gb: usize = if (cli.kv_ram_budget) |b|
                    std.fmt.parseInt(usize, b, 10) catch default_ram_budget_gb
                else
                    detectFreeRam() / (2 * gib); // 50% of free RAM in GB

                const ssd_gb: usize = if (cli.kv_ssd_budget) |b|
                    std.fmt.parseInt(usize, b, 10) catch default_ssd_budget_gb
                else
                    default_ssd_budget_gb;

                // Read model metadata for cache dimension calculations.
                const n_layers = fmt.getMetaU32("llama.block_count") orelse
                    fmt.getMetaU32("num_hidden_layers") orelse 32;
                const n_embd = fmt.getMetaU32("llama.embedding_length") orelse
                    fmt.getMetaU32("hidden_size") orelse 2048;
                const n_kv_heads = fmt.getMetaU32("llama.attention.head_count_kv") orelse
                    fmt.getMetaU32("num_key_value_heads") orelse 8;
                const n_heads = fmt.getMetaU32("llama.attention.head_count") orelse
                    fmt.getMetaU32("num_attention_heads") orelse 32;
                const head_dim = fmt.getMetaU32("llama.attention.key_length") orelse
                    fmt.getMetaU32("head_dim") orelse (n_embd / n_heads);
                const kv_dim: usize = @as(usize, n_kv_heads) * head_dim;

                const block_size: u16 = 16;
                const bytes_per_block = @as(usize, block_size) * kv_dim * @sizeOf(f32) * 2;
                const ctx = if (cli.ctx_size > 0) cli.ctx_size else default_ctx_size;
                const vram_blocks: usize = (@as(usize, ctx) + block_size - 1) / block_size;
                const ram_blocks: usize = if (has_ram and bytes_per_block > 0) (ram_gb * gib) / bytes_per_block else 0;
                const ssd_blocks: usize = if (has_ssd and bytes_per_block > 0) (ssd_gb * gib) / bytes_per_block else 0;

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

            const init_start = std.time.milliTimestamp();
            var mdl = ModelType.init(allocator, fmt, be, cli.ctx_size, cli.kv_type, tiered_ptr) catch |e| {
                eprint("Error: failed to initialize {s}: {}\n", .{ arch.displayName(), e });
                if (e == error.OutOfMemory)
                    eprint("  Not enough memory. Try a smaller quantization or model.\n", .{})
                else if (e == error.TensorNotFound)
                    eprint("  Required tensor missing. The model file may be corrupted or incomplete.\n", .{});
                return false;
            };
            defer mdl.deinit();
            if (comptime @hasField(ModelType, "pool")) {
                mdl.pool = pool;
            }
            const init_ms = elapsedMs(init_start);
            if (!g_quiet) {
                var li = load_info_in;
                li.init_ms = init_ms;
                display.printLoadInfo(li);
            }
            if (cli.profile) {
                if (comptime @hasField(ModelType, "perf")) {
                    mdl.perf.enabled = true;
                }
            }

            var model_if = mdl.model();
            if (cli.serve) {
                var tok_if = tok.tokenizer();
                server.run(allocator, &model_if, &tok_if, arch.chatTemplate(), minfo.name, minfo.be_name, cli.port, tok.bos_token_id, eog.ids, eog.len, tiered_ptr) catch |e| {
                    eprint("Error: server failed: {}\n", .{e});
                    return false;
                };
            } else if (effective_prompt) |prompt| {
                generateAndPrint(allocator, &model_if, tok, cli, tok_kind, eog, a, prompt, !g_quiet, minfo, display);
            } else {
                runRepl(allocator, &model_if, tok, cli, tok_kind, eog, a, minfo, display);
            }
            if (comptime @hasField(ModelType, "perf")) {
                mdl.perf.report();
            }
            return true;
        },
    }
}

// ── Interactive REPL ─────────────────────────────────────────────

fn runRepl(
    allocator: std.mem.Allocator,
    mdl: *Model,
    tok: *BpeTokenizer,
    cli: *const CliArgs,
    tok_kind: TokenizerKind,
    eog: anytype,
    arch: Arch,
    minfo: display_mod.ModelInfo,
    display_in: Display,
) void {
    var display = display_in;
    print("Type a message, /help for commands, Ctrl+D to quit.\n", .{});

    var editor = LineEditor.init(allocator);
    defer editor.deinit();

    const repl_prompt = if (g_color) "\x1b[1;32m> \x1b[0m" else "> ";
    var show_stats: bool = !g_quiet;

    const Message = @import("chat_template.zig").Message;

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
            } else if (std.mem.eql(u8, trimmed, "/clear")) {
                mdl.resetCache();
                for (history.items) |msg| allocator.free(@constCast(msg.content));
                history.clearRetainingCapacity();
                print("Conversation and KV cache cleared.\n", .{});
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
            } else if (std.mem.eql(u8, trimmed, "/model")) {
                display.printModelInfo(minfo);
                continue;
            } else if (std.mem.eql(u8, trimmed, "/help")) {
                print(
                    \\  /clear              Clear conversation and KV cache (stay in chat)
                    \\  /stats              Toggle generation stats
                    \\  /verbose            Toggle technical details
                    \\  /model              Show model info
                    \\  /help               Show this help
                    \\  /quit, /exit, /q    Exit interactive mode
                    \\  Ctrl+C              Cancel input (double-tap to quit)
                    \\  Ctrl+D              Quit (on empty line)
                    \\  Ctrl+L              Clear screen
                    \\  Ctrl+R              Reverse search history
                    \\  Up/Down             Navigate history
                    \\
                , .{});
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
        const response = generateAndPrintInner(allocator, mdl, tok, cli, tok_kind, eog, template, formatted, false, !is_first_turn, show_stats, minfo, display, true);

        // Add assistant response to history
        if (response) |text| {
            // Trim trailing whitespace from response for clean history
            const trimmed_resp = std.mem.trimRight(u8, text, " \t\r\n");
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
) void {
    // One-shot mode: pass need_response=false to skip redundant full-sequence decode
    const response = generateAndPrintInner(allocator, mdl, tok, cli, tok_kind, eog, arch.chatTemplate(), prompt, true, false, show_stats, minfo, display, false);
    if (response) |r| allocator.free(r);
}

/// Core generation: formats (or uses pre-formatted) prompt, prefills, generates, streams output.
/// When `skip_bos` is true, the BOS token is not sent (for continuation turns with KV cache reuse).
/// When `need_response` is false, skips the full-sequence decode (avoids allocating
/// response text the caller will discard).
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
) ?[]u8 {
    const formatted = if (format_prompt)
        template.format(allocator, cli.system_prompt, prompt) catch @as([]const u8, prompt)
    else
        prompt;
    defer if (format_prompt and formatted.ptr != prompt.ptr) allocator.free(formatted);
    if (g_debug) dbg("formatted prompt ({d} bytes): [{s}]", .{ formatted.len, formatted });

    const token_ids = switch (tok_kind) {
        .spm => tok.encodeSpm(formatted) catch {
            eprint("Error: failed to encode prompt (tokenizer may not support this input)\n", .{});
            return null;
        },
        .spm_no_dummy => tok.encodeSpmNoDummy(formatted) catch {
            eprint("Error: failed to encode prompt (tokenizer may not support this input)\n", .{});
            return null;
        },
        .bpe => tok.encode(formatted) catch {
            eprint("Error: failed to encode prompt (tokenizer may not support this input)\n", .{});
            return null;
        },
    };
    defer allocator.free(token_ids);
    dbg("encoded {d} tokens, tok_kind={s}", .{ token_ids.len, @tagName(tok_kind) });

    // BOS token — sent on first turn for models that require it (Gemma, etc.)
    if (tok.bos_token_id > 0 and !skip_bos) {
        dbg("sending BOS token (id={d}) before prefill...", .{tok.bos_token_id});
        const bos_t0 = std.time.milliTimestamp();
        _ = mdl.forward(tok.bos_token_id) catch return null;
        dbg("BOS forward done in {d}ms", .{elapsedMs(bos_t0)});
    }

    // Prefill with progress
    const prefill_start = std.time.milliTimestamp();
    if (!g_quiet and token_ids.len > prefill_progress_threshold) {
        display.showPrefillStart(token_ids.len);
    }
    dbg("entering prefill loop, {d} tokens", .{token_ids.len});
    // Prefill all prompt tokens. The last forward() call's result is the
    // first generated token — capture it to avoid re-processing the last
    // prompt token (which would duplicate it in the KV/SSM state).
    var first_gen_token: u32 = 0;
    for (token_ids, 0..) |tid, pi| {
        dbg("prefill {d}/{d} token={d}", .{ pi + 1, token_ids.len, tid });
        const pfill_t0 = std.time.milliTimestamp();
        first_gen_token = mdl.forward(tid) catch |e| {
            eprint("Error: forward failed on token {d}: {}\n", .{ tid, e });
            return null;
        };
        dbg("  done in {d}ms", .{elapsedMs(pfill_t0)});
    }
    const prefill_ms = elapsedMs(prefill_start);
    if (!g_quiet and token_ids.len > prefill_progress_threshold) {
        display.clearPrefillProgress();
    }

    // Apply sampling to the first generated token (from prefill's last forward call)
    const use_sampling = cli.temperature > 0;
    var prng = std.Random.Xoshiro256.init(cli.seed);
    if (use_sampling and token_ids.len > 0) {
        first_gen_token = math_ops.sampleToken(mdl.getLogits(), cli.temperature, cli.top_k, cli.top_p, prng.random());
    }

    // Generate — stream tokens to stdout immediately.
    // Decode in small batches to balance responsiveness vs alloc count.
    // Stop early if the model enters a repetitive loop (same token 6+ times).
    const gen_start = std.time.milliTimestamp();
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
            dbg("gen forward failed at step {d}: {}", .{ gi, e });
            break;
        };
        if (use_sampling) {
            next = math_ops.sampleToken(mdl.getLogits(), cli.temperature, cli.top_k, cli.top_p, prng.random());
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
        stdout.writeAll("\n") catch {};
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
    if (display.mode == .json and token_count > 0) {
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
    // Skip a single leading newline (common model artifact)
    if (!started.* and text.len > 0 and text[0] == '\n') {
        text = text[1..];
    }
    if (text.len > 0) started.* = true;
    stdout.writeAll(text) catch {};
}

test {
    _ = @import("display.zig");
}

test "cpu backend rms_norm via tagged union dispatch" {
    var cpu = CpuBackend{};
    const be: Backend = .{ .cpu = &cpu };
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
    var cpu = CpuBackend{};
    const be: Backend = .{ .cpu = &cpu };
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
    var cpu = CpuBackend{};
    const be: Backend = .{ .cpu = &cpu };
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
