//! CLI entry point for the Agave LLM inference engine.
//! Parses command-line arguments, loads a GGUF model or SafeTensors directory,
//! auto-detects the architecture, and runs interactive generation, one-shot prompts,
//! or an HTTP server.

const std = @import("std");
const Io = std.Io;
const build_options = @import("build_options");
const cli_mod = @import("cli.zig");
const backend_mod = @import("backend/backend.zig");
const format_mod = @import("format/format.zig");
const model_mod = @import("models/model.zig");
const spec_decode = @import("spec/spec_decode.zig");
const spec_caps = @import("spec/caps.zig");
const ngram_mod = @import("spec/ngram.zig");
const tok_mod = @import("tokenizer/tokenizer.zig");
const server = @import("server/server.zig");
const lora_mod = @import("lora.zig");
const display_mod = @import("display.zig");
const Display = display_mod.Display;
const chat_tmpl_mod = @import("chat_template.zig");
const ChatTemplate = chat_tmpl_mod.ChatTemplate;
const Message = chat_tmpl_mod.Message;
const arch_mod = @import("arch.zig");
const Arch = arch_mod.Arch;
const TokenizerKind = tok_mod.TokenizerKind;
const Recipe = @import("recipe.zig").Recipe;
const DirectionalSteering = @import("steering.zig").DirectionalSteering;

const Backend = backend_mod.Backend;
const BackendState = backend_mod.BackendState;
const BackendChoice = backend_mod.BackendChoice;
const TransportChoice = enum { auto, tcp, shm, nccl, rdma, udp, grpc };
const ThreadPool = @import("thread_pool.zig").ThreadPool;
const Format = format_mod.Format;
const GGUFFile = format_mod.GGUFFile;
const SafeTensorsDir = format_mod.SafeTensorsDir;
const Model = model_mod.Model;
const ModelStorage = model_mod.ModelStorage;
const DFlash2Model = @import("models/dflash2.zig").DFlash2Model;
const BpeTokenizer = tok_mod.BpeTokenizer;
const LineEditor = @import("readline.zig").LineEditor;
const KvQuantType = @import("ops/kv_quant.zig").KvQuantType;
const math_ops = @import("ops/math.zig");
const grammar_mod = @import("grammar.zig");
const TieredKvCache = @import("kvcache/tiered.zig").TieredKvCache;
const pull = @import("pull.zig");
const image = @import("image.zig");
const sim_clock = @import("sim_clock.zig");

const stdout_file = Io.File.stdout();
const stderr_file = Io.File.stderr();
const stdin_file = Io.File.stdin();

/// Base directory for scratch files (overridable via TMPDIR).
const default_tmp_base = "/tmp";
/// Fixed-name fallback when the unique temp path does not fit its buffer.
const video_tmp_fallback = "/tmp/agave_video";
/// Buffer size for composing the video frame temp directory path.
const tmp_path_buf_size = 256;

fn milliTimestamp(io: Io) i64 {
    _ = io;
    return sim_clock.milliNow();
}

fn nanoTimestamp(io: Io) i96 {
    _ = io;
    return sim_clock.nanoNow();
}

/// Read all piped stdin into an allocated buffer.
fn readStdinAll(allocator: std.mem.Allocator, max_size: usize) ?[]const u8 {
    var buf = std.ArrayList(u8).empty;
    var read_buf: [4096]u8 = undefined;
    while (true) {
        const n = std.posix.read(stdin_file.handle, &read_buf) catch break;
        if (n == 0) break;
        if (buf.items.len + n > max_size) {
            eprint("Error: piped input exceeds {d} MB limit\n", .{max_size / (1024 * 1024)});
            buf.deinit(allocator);
            std.process.exit(1);
        }
        buf.appendSlice(allocator, read_buf[0..n]) catch {
            eprint("Error: out of memory reading piped input ({d} bytes read)\n", .{buf.items.len});
            buf.deinit(allocator);
            return null;
        };
    }
    if (buf.items.len == 0) {
        buf.deinit(allocator);
        return null;
    }
    return buf.toOwnedSlice(allocator) catch {
        eprint("Error: out of memory finalizing piped input ({d} bytes)\n", .{buf.items.len});
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
/// Default UDP port for tensor-parallel peer discovery.
const tp_discovery_port: u16 = 49454;
/// Default UDP port for pipeline-parallel peer discovery.
const pp_discovery_port: u16 = 49455;
/// Default TCP port for disaggregated prefill/decode transport.
const disagg_default_port: u16 = 49456;
/// Default maximum tokens to generate per request.
const default_max_tokens: u32 = 512;
/// Default KV cache context size when user/recipe doesn't specify.
/// 4096 balances memory usage with practical conversation length.
const default_ctx_size: u32 = 4096;
/// Default prefill chunk size (tokens per batch).
const default_chunk_size: u32 = 512;
/// Milliseconds per second — used for tok/s calculations.
const ms_per_second: f32 = 1000.0;
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
const kv_valid_types = "f32, f16, q8_0/q8, int8/i8, fp8/fp8_e4m3, nvfp4/fp4, nvfp4_ds_mla, turbo2-4/tq2-4, planar2-4/pq2-4, iso2-4/iq2-4, rotor2-4/rq2-4, turbo (preset: K=q8_0, V=turbo4)";

// ── Output control ──────────────────────────────────────────────

var g_color: bool = true;
var g_quiet: bool = false;
var g_tty: bool = true;
var g_debug: bool = false;
var g_verbose: bool = false;
var g_io: Io = undefined;
var init_args: std.process.Args = undefined;
var g_environ: *std.process.Environ.Map = undefined;

fn print(comptime fmt: []const u8, args: anytype) void {
    var buf: [print_buf_size]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, fmt, args) catch return;
    _ = std.posix.system.write(stdout_file.handle, text.ptr, text.len);
}

fn eprint(comptime fmt: []const u8, args: anytype) void {
    var buf: [print_buf_size]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, fmt, args) catch return;
    _ = std.posix.system.write(stderr_file.handle, text.ptr, text.len);
}

/// Rank 1+ of a TP/PP pair still runs the decode loop (NCCL lockstep) but must
/// not write tokens; rank 0 is the only stdout owner.
fn emitGeneratedTokens(cli: *const CliArgs) bool {
    return cli.tp_rank == 0 or (cli.tp_degree <= 1 and cli.pp_degree <= 1);
}

/// True when two ranks share one generate loop over the pair transport.
/// Draft/sample RNG would desync the pair; greedy argmax keeps lockstep.
fn distributedLockstep(cli: *const CliArgs) bool {
    return cli.tp_degree > 1 or cli.pp_degree > 1;
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
        std.process.exit(2);
    };
}

/// Detect free system RAM in bytes.
/// Delegates to backend_mod.detectAvailMem(); falls back to default_free_ram.
fn detectFreeRam() usize {
    const avail = backend_mod.detectAvailMem();
    return if (avail > 0) avail else default_free_ram;
}

/// Estimate bytes per single MoE expert (gate + up + down projections).
fn estimateExpertBytes(fmt: Format, n_experts: u32) usize {
    // Use layer 3 (first learned-routed layer) as representative
    const gate = fmt.getTensor("blk.3.ffn_gate_exps.weight") orelse return 0;
    const gate_total = gate.dataByteLen();
    const per_expert_gate = gate_total / @as(usize, n_experts);

    const up = fmt.getTensor("blk.3.ffn_up_exps.weight") orelse return per_expert_gate;
    const per_expert_up = up.dataByteLen() / @as(usize, n_experts);

    const down = fmt.getTensor("blk.3.ffn_down_exps.weight") orelse return per_expert_gate + per_expert_up;
    const per_expert_down = down.dataByteLen() / @as(usize, n_experts);

    return per_expert_gate + per_expert_up + per_expert_down;
}

// ── Preload (fault-in mmap'd pages) ─────────────────────────────

/// Touch every page of a mmap'd region to fault it into RAM.
/// This eliminates page-fault stalls during inference by paying the I/O
/// cost upfront during model load. Uses madvise(SEQUENTIAL) to hint
/// kernel readahead, then switches to RANDOM after pages are resident.
fn preloadRegion(data: []align(std.heap.page_size_min) const u8) void {
    const MADV = std.posix.MADV;
    // Best-effort OS hint — failure is harmless
    std.posix.madvise(@alignCast(@constCast(data.ptr)), data.len, MADV.SEQUENTIAL) catch {};

    // Touch one byte per page to force all pages into RAM
    const page_size = std.heap.page_size_min;
    var offset: usize = 0;
    while (offset < data.len) : (offset += page_size) {
        _ = @as(*const volatile u8, @ptrCast(&data[offset])).*;
    }

    // Best-effort OS hint — failure is harmless
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
    // Best-effort OS hint — failure is harmless
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
            _ = std.posix.system.write(stderr_file.handle, buf[0..pos].ptr, pos);
        }
    }

    // Best-effort OS hint — failure is harmless
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
    .{ .long = "min-p", .kind = .option, .help = "Min-p sampling threshold [default: 0]." },
    .{ .long = "repeat-penalty", .kind = .option, .help = "Repetition penalty [default: 1.0]." },
    .{ .long = "dry-multiplier", .kind = .option, .help = "DRY n-gram repetition penalty multiplier [default: 0]." },
    .{ .long = "dry-length", .kind = .option, .help = "DRY minimum n-gram length to penalize [default: 2]." },
    .{ .long = "xtc-probability", .kind = .option, .help = "XTC exclude-top-choices probability [default: 0]." },
    .{ .long = "xtc-threshold", .kind = .option, .help = "XTC probability threshold for exclusion [default: 0.1]." },
    .{ .long = "mirostat-mode", .kind = .option, .help = "Mirostat mode: 0=disabled, 2=Mirostat 2.0 [default: 0]." },
    .{ .long = "mirostat-tau", .kind = .option, .help = "Mirostat target entropy [default: 5.0]." },
    .{ .long = "mirostat-eta", .kind = .option, .help = "Mirostat learning rate [default: 0.1]." },
    .{ .long = "seed", .kind = .option, .help = "Random seed for sampling [default: random]." },
    .{ .long = "grammar", .kind = .option, .help = "GBNF grammar file for constrained decoding." },
    .{ .long = "grammar-string", .kind = .option, .help = "Inline GBNF grammar string." },
    .{ .long = "json-output", .help = "Constrain generation to valid JSON via grammar (not output format; see --json)." },
    .{ .long = "json-schema", .kind = .option, .help = "JSON schema for structured output (converts to GBNF grammar)." },
    .{ .long = "system", .kind = .option, .help = "System prompt for chat formatting." },
    // Backend & model
    .{ .long = "backend", .kind = .option, .help = "Compute backend: auto, cpu, metal, vulkan, cuda, rocm, webgpu [default: auto]." },
    .{ .long = "device", .kind = .option, .help = "GPU device index for CUDA/ROCm/Vulkan [default: 0]. Use --list-devices to see available." },
    .{ .long = "list-devices", .help = "List available compute devices and exit." },
    .{ .long = "disagg", .help = "Disaggregated inference: rank 0 prefills, sends KV to rank 1 for decode." },
    .{ .long = "tp", .kind = .option, .help = "Tensor parallelism degree [default: 1]." },
    .{ .long = "pp", .kind = .option, .help = "Pipeline parallelism stages [default: 1]." },
    .{ .long = "peers", .kind = .option, .help = "TP peer addresses for distributed inference (e.g. 192.168.0.212:9999)." },
    .{ .long = "rank", .kind = .option, .help = "This node's rank for TP/PP/disagg [default: 0]." },
    .{ .long = "transport", .kind = .option, .help = "IPC transport: auto, tcp, shm, nccl [default: auto]. rdma/udp/grpc are rejected until implemented." },
    .{ .long = "ctx-size", .kind = .option, .help = "Context window size; 0 = full, auto = fit to memory [default: 4096 or model limit, whichever is smaller]." },
    .{ .long = "allow-cpu-fallback", .help = "Allow GPU backends to fall back to CPU for unsupported ops." },
    .{ .long = "mmap", .help = "Use lazy mmap instead of eagerly paging weights into RAM." },
    .{ .long = "prefill-batch-size", .kind = .option, .help = "Prefill chunk size in tokens [default: 512]." },
    // KV cache
    .{ .long = "no-kv-cache", .help = "Disable KV cache allocation (prefill-only / embedding use cases). Prevents any decode-phase caching." },
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
    .{ .long = "port", .short = 'p', .kind = .option, .help = "Server port [default: 49453]. Falls back to AGAVE_PORT." },
    .{ .long = "host", .kind = .option, .help = "Server bind address: IPv4, localhost, 0.0.0.0, or 0 [default: 127.0.0.1]. Falls back to AGAVE_HOST." },
    .{ .long = "api-key", .kind = .option, .help = "API key for server auth. Prefer AGAVE_API_KEY (avoids process-list exposure; env wins if both set)." },
    .{ .long = "sleep-after", .kind = .option, .help = "Enter sleep mode after N seconds of server inactivity (0 = disabled). Signals /health sleeping:true; wakes on next request." },
    .{ .long = "max-batch-size", .kind = .option, .help = "Max concurrent requests to batch per scheduler cycle [default: 8]. Higher values increase throughput at the cost of latency per request." },
    .{ .long = "rate-limit-rpm", .kind = .option, .help = "Server max requests per minute (0 = unlimited). Enables token-bucket rate limiting when set with or without --rate-limit-tpm." },
    .{ .long = "rate-limit-tpm", .kind = .option, .help = "Server max prompt tokens per minute (0 = unlimited). Enables token-bucket rate limiting when set with or without --rate-limit-rpm." },
    // LoRA
    .{ .long = "lora", .kind = .option, .help = "Path to LoRA adapter GGUF file. Merged at load time into the base model weights." },
    // Multimodal
    .{ .long = "mmproj", .kind = .option, .help = "Path to vision projector GGUF (mmproj file)." },
    .{ .long = "image", .kind = .option, .help = "Path to image file for multimodal inference (PNG or PPM P6)." },
    .{ .long = "video", .kind = .option, .help = "Path to video file for multimodal inference. Extracts frames via ffmpeg and feeds them to the vision encoder. Use --video-fps to control sampling rate." },
    .{ .long = "video-fps", .kind = .option, .help = "Frames per second to sample from video (default: 1). Higher FPS = more visual tokens." },
    // Speculative decoding
    .{ .long = "draft-model", .kind = .option, .help = "Path to draft model for speculative decoding." },
    .{ .long = "mtp-model", .kind = .option, .help = "Path to MTP weight file (safetensors) for multi-token prediction speculative decoding." },
    .{ .long = "spec-tokens", .short = 'K', .kind = .option, .help = "Draft tokens per speculation round [default: 5]." },
    .{ .long = "tree-budget", .kind = .option, .help = "DDTree node budget [default: 64]." },
    .{ .long = "spec-mode", .kind = .option, .help = "Speculative mode: auto, standard, ddtree, self, ngram, suffix, lookahead, mtp, medusa, eagle, eagle3, mlp, pflash, dspark, dflash2 [default: ddtree with --draft-model]." },
    .{ .long = "spec-token-map", .kind = .option, .help = "FR-Spec token frequency map file (one token ID per line). Restricts draft to high-frequency tokens for improved acceptance rate." },
    .{ .long = "draft-layers", .kind = .option, .help = "Layers for self-speculative draft [default: auto]." },
    .{ .long = "pflash-alpha", .kind = .option, .help = "PFlash block selection threshold (0.0-2.0) [default: 0.85]." },
    .{ .long = "pflash-block-size", .kind = .option, .help = "PFlash scoring block size in tokens [default: 64]." },
    .{ .long = "pflash-scorer", .kind = .option, .help = "Separate model for PFlash block importance scoring (defaults to --draft-model)." },
    // Diffusion generation
    .{ .long = "diffusion-steps", .kind = .option, .help = "Max denoising iterations for DiffusionGemma [default: 16]." },
    .{ .long = "diffusion-canvas", .kind = .option, .help = "Canvas size (tokens per generation block) for DiffusionGemma [default: 256]." },
    .{ .long = "diffusion-confidence", .kind = .option, .help = "Token acceptance confidence threshold for diffusion (0.0-1.0) [default: 0.5]." },
    // Diagnostics
    .{ .long = "verbose", .short = 'V', .help = "Show technical details (params, load times, EOG)." },
    .{ .long = "debug", .short = 'd', .help = "Enable debug logging (token IDs, layer timing); implies --verbose." },
    .{ .long = "json", .help = "Output results as JSON (implies --quiet)." },
    .{ .long = "model-info", .help = "Print model metadata and exit (supports --json)." },
    .{ .long = "megakernel", .help = "Enable fused FFN megakernels (3→1 dispatch per layer)." },
    .{ .long = "profile", .help = "Profile per-op timing (halves throughput)." },
    .{ .long = "dir-steering-file", .kind = .option, .help = "Directional steering f32 vector file (n_layers × n_embd floats)." },
    .{ .long = "dir-steering-ffn", .kind = .option, .help = "Steering scale for FFN outputs [default: 1.0 when file provided]." },
    .{ .long = "dir-steering-attn", .kind = .option, .help = "Steering scale for attention outputs [default: 0]." },
    .{ .long = "benchmark", .help = "Run decode benchmark: prefill + decode, print stats (supports --json)." },
    // SSD expert streaming (MoE models)
    .{ .long = "ssd-streaming", .help = "Enable SSD expert streaming for large MoE models that don't fit in RAM/VRAM. Uses demand-paged LRU expert cache." },
    .{ .long = "ssd-cache-slots", .kind = .option, .help = "Number of expert slots to keep resident in the SSD expert cache [default: 256]. Higher = fewer SSD reads, more RAM." },
    .{ .long = "expert-profile-out", .kind = .option, .help = "Write expert activation profile JSON to this path after inference (for hotlist pre-pinning on future runs)." },
    .{ .long = "expert-profile-in", .kind = .option, .help = "Load expert activation profile JSON and pre-pin top experts into the SSD cache before inference starts." },
    // Power throttling
    .{ .long = "power", .kind = .option, .help = "Target GPU utilisation percent (1-100). Inserts inter-layer sleeps to reduce heat and fan noise without changing outputs [default: 100 = no throttle]." },
    // Frontier benchmarking
    .{ .long = "frontier-bench", .help = "Frontier benchmark: snapshot KV at each context length in --frontier-ctx, report prefill+generation t/s per frontier." },
    .{ .long = "frontier-ctx", .kind = .option, .help = "Comma-separated context lengths for frontier benchmark, e.g. 1024,4096,16384 [default: 512,2048,8192]." },
};

/// Speculative decoding strategy. CLI aliases (e.g. `medusa` → `mtp`) normalize at
/// parse time so call sites never branch on synonym variants.
const SpecMode = spec_caps.SpecMode;

const CliArgs = struct {
    model_path: []const u8,
    prompt: ?[]const u8,
    serve: bool,
    port: u16,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    min_p: f32,
    repeat_penalty: f32,
    dry_multiplier: f32 = 0,
    dry_length: u32 = 2,
    xtc_probability: f32 = 0,
    xtc_threshold: f32 = 0.1,
    mirostat_mode: u32 = 0,
    mirostat_tau: f32 = 5.0,
    mirostat_eta: f32 = 0.1,
    grammar_path: ?[]const u8,
    grammar_string: ?[]const u8,
    json_schema: ?[]const u8,
    json_output: bool,
    system_prompt: ?[]const u8,
    backend_choice: BackendChoice,
    device_id: u32,
    ctx_size: u32,
    /// --no-kv-cache: disable KV cache allocation (prefill-only / embedding mode).
    no_kv_cache: bool = false,
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
    kv_eviction: enum { none, norm, tri } = .none,
    kv_budget: u32 = 0,
    host: [4]u8 = .{ 127, 0, 0, 1 },
    api_key: ?[]const u8 = null,
    allow_cpu_fallback: bool,
    debug: bool,
    json: bool,
    model_info: bool,
    benchmark: bool = false,
    profile: bool,
    use_mmap: bool,
    prefill_batch_size: u32,
    /// Path to LoRA adapter GGUF. Applied at load time (merged into base weights as F32).
    lora_path: ?[]const u8 = null,
    /// Path to vision projector GGUF (mmproj file) for multimodal inference.
    mmproj: ?[]const u8 = null,
    /// Path to image file (PNG or PPM P6) for multimodal inference.
    image: ?[]const u8 = null,
    /// Path to video file for multimodal inference (requires ffmpeg).
    video: ?[]const u8 = null,
    /// Frames per second to sample from video (default: 1).
    video_fps: f32 = 1.0,
    /// Enable fused megakernel for single-dispatch forward pass.
    megakernel: bool = false,
    /// Tensor parallelism degree (split weights across ranks).
    tp_degree: u32 = 1,
    tp_rank: u32 = 0,
    tp_peers: ?[]const u8 = null,
    transport: TransportChoice = .auto,
    pp_degree: u32 = 1,
    disagg: bool = false,
    /// Sleep mode: enter sleep after N seconds of server inactivity (0 = disabled).
    sleep_after_s: u32 = 0,
    /// Maximum concurrent requests to batch together per scheduler cycle (default 8).
    max_batch_size: u32 = 8,
    /// Server rate limit: max requests per minute (0 = unlimited / disabled).
    rate_limit_rpm: u32 = 0,
    /// Server rate limit: max prompt tokens per minute (0 = unlimited / disabled).
    rate_limit_tpm: u32 = 0,
    // Speculative decoding
    draft_model_path: ?[]const u8 = null,
    /// Path to MTP weight file (safetensors) for multi-token prediction.
    mtp_model_path: ?[]const u8 = null,
    spec_tokens: u32 = 5,
    tree_budget: u32 = 64,
    spec_mode: SpecMode = .none,
    draft_layers: ?u32 = null,
    spec_token_map: ?[]const u8 = null,
    // PFlash speculative prefill
    pflash_alpha: f32 = 0.85,
    pflash_block_size: u32 = 64,
    /// Separate model for PFlash block scoring (optional; defaults to --draft-model).
    pflash_scorer_path: ?[]const u8 = null,
    // Directional steering
    dir_steering_file: ?[]const u8 = null,
    dir_steering_ffn: f32 = 0,
    dir_steering_attn: f32 = 0,
    // Diffusion generation (DiffusionGemma)
    /// Maximum denoising steps for block diffusion (default 16).
    diffusion_steps: u32 = 16,
    /// Number of tokens in the generation canvas per diffusion block (default 256).
    diffusion_canvas: u32 = 256,
    /// Confidence threshold: tokens above this probability are accepted (default 0.5).
    diffusion_confidence: f32 = 0.5,
    // SSD expert streaming
    /// Demand-paged LRU expert cache for large MoE models.
    ssd_streaming: bool = false,
    /// Number of expert slots to keep resident (default 256).
    ssd_cache_slots: u32 = 256,
    /// Write expert activation profile JSON after inference.
    expert_profile_out: ?[]const u8 = null,
    /// Load expert profile JSON and pre-pin top experts before inference.
    expert_profile_in: ?[]const u8 = null,
    // Power throttling
    /// Target GPU utilisation percent (1-100). 100 = no throttle.
    power_pct: u32 = 100,
    // Frontier benchmarking
    /// Run frontier benchmark.
    frontier_bench: bool = false,
    /// Comma-separated context lengths for frontier benchmark.
    frontier_ctx: []const u8 = "512,2048,8192",
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
        const exit_code = calibrate.run(allocator, init_args, g_io);
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
        const topics = [_][]const u8{ "pull", "calibrate" };
        for (topics) |topic| {
            if (closeMatch(sub, topic)) {
                eprint("  Did you mean 'agave help {s}'?\n", .{topic});
                break;
            }
        }
        eprint("Available help topics: pull, calibrate\n", .{});
        eprint("Run 'agave --help' for more information.\n", .{});
        std.process.exit(2);
    }
    return false;
}

/// Parses command-line arguments into a `CliArgs` struct, returning `null` on `--help`/`--version` or invalid input.
fn parseCli(allocator: std.mem.Allocator) ?CliArgs {
    var res = cli_mod.parse(allocator, init_args, &cli_specs);
    defer res.deinit();

    if (res.flag("help")) {
        printUsage();
        return null;
    }

    if (res.flag("version")) {
        display_mod.printVersion();
        return null;
    }

    // Error on options that appeared at end of args without a value
    if (res.missing_value) |name| {
        eprint("Error: --{s} requires a value\n", .{name});
        eprint("Run 'agave --help' for more information.\n", .{});
        std.process.exit(2);
    }

    // Reject --flag=value on boolean flags (parser stores them in options).
    rejectEqualsOnFlag(&res);

    // Reject unknown flags (catches typos like --temeprature); matches pull/calibrate.
    rejectUnknownOptions(&res);

    // Reject when a known flag was consumed as another option's value.
    // Example: `--system --serve` sets system prompt to "--serve" and loses --serve.
    rejectFlagAsValue(&res);

    // Reject unknown short options that the parser treated as positionals (e.g. -z, -qv).
    // Letter-only forms only so numeric prompts like "-5" still work. Use -- for odd paths.
    rejectUnknownShortPositionals(&res);

    // Auto-detect TTY: disable color when stdout is not a terminal
    g_tty = stdout_file.isTty(g_io) catch false;
    g_color = blk: {
        // Conflicting pair: require a single explicit choice.
        if (res.option("color") != null and res.flag("no-color")) {
            eprint("Error: conflicting --color and --no-color; use only one\n", .{});
            eprint("Run 'agave --help' for more information.\n", .{});
            std.process.exit(2);
        }
        // --color=always|never|auto takes precedence
        if (res.option("color")) |cm| {
            if (std.mem.eql(u8, cm, "always")) break :blk true;
            if (std.mem.eql(u8, cm, "never")) break :blk false;
            if (!std.mem.eql(u8, cm, "auto")) {
                eprint("Error: unknown --color value '{s}'\n", .{cm});
                eprint("  Valid options: auto, always, never\n", .{});
                std.process.exit(2);
            }
        }
        // --no-color flag
        if (res.flag("no-color")) break :blk false;
        // NO_COLOR env var (https://no-color.org): present and non-empty
        if (noColorRequested(g_environ.get("NO_COLOR"))) break :blk false;
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

    // --list-devices: enumerate and print available compute devices
    if (res.flag("list-devices")) {
        const discovery = @import("devices/discovery.zig");
        const device_list = discovery.enumerate();
        discovery.printDeviceTable(&device_list);
        return null;
    }

    const n_positionals = res.positionals.items.len;

    const backend_choice: BackendChoice = blk: {
        const be_str = res.option("backend") orelse "auto";
        break :blk std.meta.stringToEnum(BackendChoice, be_str) orelse {
            eprint("Error: unknown backend '{s}'\n", .{be_str});
            eprint("  Valid options: auto, cpu, metal, vulkan, cuda, rocm, webgpu\n", .{});
            std.process.exit(2);
        };
    };
    const device_id: u32 = if (res.option("device")) |d| std.fmt.parseInt(u32, d, 10) catch {
        eprint("Error: invalid value for --device: '{s}' is not a valid integer\n", .{d});
        std.process.exit(2);
    } else 0;

    const temperature = parseF32(res.option("temperature"), "temperature") orelse 0.0;
    const top_p = parseF32(res.option("top-p"), "top-p") orelse 1.0;
    const repeat_penalty = parseF32(res.option("repeat-penalty"), "repeat-penalty") orelse 1.0;
    const grammar_path = res.option("grammar");
    const grammar_string = res.option("grammar-string");
    const json_schema = res.option("json-schema");

    // Validate sampling parameter ranges
    if (temperature < 0) {
        eprint("Error: --temperature must be >= 0 (got {d:.2})\n", .{temperature});
        std.process.exit(2);
    }
    if (top_p <= 0 or top_p > 1.0) {
        eprint("Error: --top-p must be in (0, 1.0] (got {d:.2})\n", .{top_p});
        std.process.exit(2);
    }
    if (repeat_penalty <= 0) {
        eprint("Error: --repeat-penalty must be > 0 (got {d:.2})\n", .{repeat_penalty});
        std.process.exit(2);
    }
    const min_p = parseF32(res.option("min-p"), "min-p") orelse 0.0;
    if (min_p < 0 or min_p > 1.0) {
        eprint("Error: --min-p must be in [0, 1.0] (got {d:.2})\n", .{min_p});
        std.process.exit(2);
    }
    const dry_multiplier = parseF32(res.option("dry-multiplier"), "dry-multiplier") orelse 0;
    if (dry_multiplier < 0) {
        eprint("Error: --dry-multiplier must be >= 0 (got {d:.2})\n", .{dry_multiplier});
        std.process.exit(2);
    }
    const xtc_probability = parseF32(res.option("xtc-probability"), "xtc-probability") orelse 0;
    if (xtc_probability < 0 or xtc_probability > 1.0) {
        eprint("Error: --xtc-probability must be in [0, 1.0] (got {d:.2})\n", .{xtc_probability});
        std.process.exit(2);
    }
    const xtc_threshold = parseF32(res.option("xtc-threshold"), "xtc-threshold") orelse 0.1;
    if (xtc_threshold < 0 or xtc_threshold > 1.0) {
        eprint("Error: --xtc-threshold must be in [0, 1.0] (got {d:.2})\n", .{xtc_threshold});
        std.process.exit(2);
    }
    const mirostat_tau = parseF32(res.option("mirostat-tau"), "mirostat-tau") orelse 5.0;
    if (mirostat_tau <= 0) {
        eprint("Error: --mirostat-tau must be > 0 (got {d:.2})\n", .{mirostat_tau});
        std.process.exit(2);
    }
    const mirostat_eta = parseF32(res.option("mirostat-eta"), "mirostat-eta") orelse 0.1;
    if (mirostat_eta <= 0) {
        eprint("Error: --mirostat-eta must be > 0 (got {d:.2})\n", .{mirostat_eta});
        std.process.exit(2);
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
            std.process.exit(2);
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
            std.process.exit(2);
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
    if (parseU32(res.option("max-tokens"), "max-tokens")) |mt| {
        if (mt == 0) {
            eprint("Error: --max-tokens must be >= 1\n", .{});
            std.process.exit(2);
        }
    }

    // Validate prefill batch size
    if (parseU32(res.option("prefill-batch-size"), "prefill-batch-size")) |pbs| {
        if (pbs == 0) {
            eprint("Error: --prefill-batch-size must be >= 1\n", .{});
            std.process.exit(2);
        }
    }

    // Validate mirostat-mode (only 0 and 2 are supported)
    if (parseU32(res.option("mirostat-mode"), "mirostat-mode")) |mm| {
        if (mm != 0 and mm != 2) {
            eprint("Error: --mirostat-mode must be 0 (disabled) or 2 (Mirostat 2.0), got {d}\n", .{mm});
            std.process.exit(2);
        }
    }

    // Validate DRY minimum n-gram length (0 would match everything)
    if (parseU32(res.option("dry-length"), "dry-length")) |dl| {
        if (dl == 0) {
            eprint("Error: --dry-length must be >= 1\n", .{});
            std.process.exit(2);
        }
    }

    // Validate parallelism degrees (must be >= 1)
    if (parseU32(res.option("tp"), "tp")) |tp| {
        if (tp == 0) {
            eprint("Error: --tp must be >= 1\n", .{});
            std.process.exit(2);
        }
        // Transport is a fixed rank-0 ↔ rank-1 pair (NCCL unique-id exchange,
        // shm names, tcp_fds[0]). Same cap as --pp.
        if (tp > 2) {
            eprint("Error: --tp > 2 is not supported yet (transport is a 2-rank pair only)\n", .{});
            eprint("  Use --tp 2 with --transport nccl (CUDA) or tcp/shm, or run a single rank.\n", .{});
            std.process.exit(2);
        }
    }
    if (parseU32(res.option("pp"), "pp")) |pp| {
        if (pp == 0) {
            eprint("Error: --pp must be >= 1\n", .{});
            std.process.exit(2);
        }
        // Transport is a fixed rank-0 ↔ rank-1 pair (shm names, tcp_fds[0],
        // sendBuf peer). Multi-stage rings are not implemented yet.
        if (pp > 2) {
            eprint("Error: --pp > 2 is not supported yet (transport is a 2-rank pair only)\n", .{});
            eprint("  Use --pp 2, or run a single stage (--pp 1).\n", .{});
            std.process.exit(2);
        }
    }
    {
        const tp = parseU32(res.option("tp"), "tp") orelse 1;
        const pp = parseU32(res.option("pp"), "pp") orelse 1;
        if ((tp > 1 or pp > 1) and temperature > 0) {
            eprint("Warning: --temperature ignored with --tp/--pp (greedy lockstep)\n", .{});
        }
    }
    if (parseU32(res.option("rank"), "rank")) |rank| {
        const pp = parseU32(res.option("pp"), "pp") orelse 1;
        const tp = parseU32(res.option("tp"), "tp") orelse 1;
        const world = @max(pp, tp);
        if (world > 1 and rank >= world) {
            eprint("Error: --rank {d} is out of range for world size {d} (valid: 0..{d})\n", .{ rank, world, world - 1 });
            std.process.exit(2);
        }
    }

    // Validate speculative decoding parameters
    if (parseU32(res.option("spec-tokens"), "spec-tokens")) |st| {
        if (st == 0) {
            eprint("Error: --spec-tokens must be >= 1\n", .{});
            std.process.exit(2);
        }
    }
    if (parseU32(res.option("tree-budget"), "tree-budget")) |tb| {
        if (tb == 0) {
            eprint("Error: --tree-budget must be >= 1\n", .{});
            std.process.exit(2);
        }
    }

    // Validate port range (1-65535, u16 parse already enforces upper bound).
    // CLI --port wins over AGAVE_PORT; both are validated the same way.
    const port_raw = res.option("port") orelse g_environ.get("AGAVE_PORT");
    if (parseU16(port_raw, "port")) |p| {
        if (p == 0) {
            eprint("Error: --port / AGAVE_PORT must be in range 1-65535\n", .{});
            std.process.exit(2);
        }
    }

    // Validate max-batch-size (0 would silently fall back inside the server)
    if (parseU32(res.option("max-batch-size"), "max-batch-size")) |mbs| {
        if (mbs == 0) {
            eprint("Error: --max-batch-size must be >= 1\n", .{});
            std.process.exit(2);
        }
    }

    // Validate --video-fps early so bad values are reported before "missing model path"
    if (parseF32(res.option("video-fps"), "video-fps")) |v| {
        if (v <= 0) {
            eprint("Error: --video-fps must be > 0 (got {d:.2})\n", .{v});
            std.process.exit(2);
        }
    }

    // Require model after option values are validated so typos like
    // --temperature=abc report the bad value instead of "missing model path".
    if (n_positionals == 0) {
        eprint("Error: missing model path\n", .{});
        eprint("Usage: agave <model.gguf|model-dir/> [prompt]\n", .{});
        eprint("Run 'agave --help' for more information.\n", .{});
        std.process.exit(2);
    }

    // Warn about extra positional arguments (e.g. unquoted multi-word prompt)
    if (n_positionals > 2) {
        eprint("Warning: extra arguments after prompt ignored (did you forget to quote it?)\n", .{});
        eprint("  Usage: agave model.gguf \"multi word prompt\"\n", .{});
    }

    // Warn about server-only flags/env that have no effect without --serve
    if (!res.flag("serve")) {
        if (res.option("port") != null) {
            eprint("Warning: --port has no effect without --serve\n", .{});
        } else if (g_environ.get("AGAVE_PORT") != null) {
            eprint("Warning: AGAVE_PORT has no effect without --serve\n", .{});
        }
        if (res.option("host") != null) {
            eprint("Warning: --host has no effect without --serve\n", .{});
        } else if (g_environ.get("AGAVE_HOST") != null) {
            eprint("Warning: AGAVE_HOST has no effect without --serve\n", .{});
        }
        if (res.option("api-key") != null) {
            eprint("Warning: --api-key has no effect without --serve\n", .{});
        } else if (g_environ.get("AGAVE_API_KEY") != null) {
            eprint("Warning: AGAVE_API_KEY has no effect without --serve\n", .{});
        }
        if (res.option("rate-limit-rpm") != null or res.option("rate-limit-tpm") != null) {
            eprint("Warning: --rate-limit-rpm/--rate-limit-tpm have no effect without --serve\n", .{});
        }
    } else {
        // Warn about flags ignored in server mode (early, before model loading)
        if (n_positionals > 1)
            eprint("Warning: prompt ignored in server mode (--serve)\n", .{});
        if (res.option("system") != null)
            eprint("Warning: --system ignored in server mode (system prompt comes from API request)\n", .{});
        if (res.option("image") != null)
            eprint("Warning: --image ignored in server mode (images come from API request)\n", .{});
        if (res.flag("benchmark"))
            eprint("Warning: --benchmark exits before server starts; remove --serve or --benchmark\n", .{});
        if (res.flag("model-info"))
            eprint("Warning: --model-info exits before server starts; remove --serve or --model-info\n", .{});
        // --api-key appears in `ps`/`/proc/*/cmdline`; AGAVE_API_KEY wins when both are set.
        if (res.option("api-key") != null and g_environ.get("AGAVE_API_KEY") != null) {
            eprint("Warning: both --api-key and AGAVE_API_KEY set; using AGAVE_API_KEY (CLI value ignored)\n", .{});
        } else if (res.option("api-key") != null) {
            eprint("Warning: --api-key is visible in process listings; prefer AGAVE_API_KEY\n", .{});
        }
    }

    // Warn about conflicting exit-early flags (--model-info runs first, --benchmark skipped)
    if (res.flag("model-info") and res.flag("benchmark"))
        eprint("Warning: --model-info and --benchmark both exit early; only --model-info will run\n", .{});

    // Warn about --profile with --benchmark (profile halves throughput, skewing benchmark results)
    if (res.flag("profile") and res.flag("benchmark"))
        eprint("Warning: --profile halves throughput; benchmark results will be misleading\n", .{});

    // Warn about --mmap with --benchmark (lazy mmap means page faults during benchmark)
    if (res.flag("mmap") and res.flag("benchmark"))
        eprint("Warning: --mmap with --benchmark includes page fault overhead in timing\n", .{});

    // Warn about prompt with --benchmark (benchmark uses a fixed prompt, user prompt is ignored)
    if (res.flag("benchmark") and n_positionals > 1 and !res.flag("serve"))
        eprint("Warning: --benchmark uses a fixed prompt; your prompt will be ignored\n", .{});

    // Warn about --allow-cpu-fallback: not wired into GPU backends yet (they fail closed).
    if (res.flag("allow-cpu-fallback"))
        eprint("Warning: --allow-cpu-fallback is not implemented; GPU backends fail closed on missing kernels\n", .{});
    if (res.flag("allow-cpu-fallback") and backend_choice == .cpu)
        eprint("Warning: --allow-cpu-fallback has no effect with --backend cpu\n", .{});

    // Warn about --device with CPU backend (CPU ignores device index)
    if (res.option("device") != null and backend_choice == .cpu)
        eprint("Warning: --device has no effect with --backend cpu\n", .{});

    // Warn about --disagg without --peers (disagg requires a peer to send/receive KV)
    if (res.flag("disagg") and res.option("peers") == null)
        eprint("Warning: --disagg has no effect without --peers\n", .{});

    // Warn about --tp > 1 without --peers (need peers for tensor parallelism)
    if ((parseU32(res.option("tp"), "tp") orelse 1) > 1 and res.option("peers") == null)
        eprint("Warning: --tp > 1 has no effect without --peers (peer discovery only works for --pp)\n", .{});

    // Warn about speculative decoding flags that have no effect without a spec mode
    {
        const has_spec = res.option("draft-model") != null or res.option("spec-mode") != null;
        if (!has_spec) {
            if (res.option("spec-tokens") != null)
                eprint("Warning: --spec-tokens has no effect without --draft-model or --spec-mode\n", .{});
            if (res.option("tree-budget") != null)
                eprint("Warning: --tree-budget has no effect without --draft-model or --spec-mode\n", .{});
        }
        if (res.option("draft-layers") != null) {
            const sm = res.option("spec-mode");
            if (sm == null or !std.mem.eql(u8, sm.?, "self"))
                eprint("Warning: --draft-layers only applies to --spec-mode self\n", .{});
        }
        // Warn about --spec-mode standard/ddtree without a draft model (self-draft still works).
        if (res.option("spec-mode")) |sm| {
            if (res.option("draft-model") == null and
                (std.mem.eql(u8, sm, "standard") or std.mem.eql(u8, sm, "ddtree")))
                eprint("Warning: --spec-mode {s} has no --draft-model (using self-draft)\n", .{sm});
        }
    }

    // Warn about conflicting constrained-decoding flags (only one takes effect)
    {
        const json_out = res.flag("json-output");
        const has_schema = json_schema != null;
        const has_grammar_s = grammar_string != null;
        const has_grammar_f = grammar_path != null;
        const constraint_count = @as(u32, @intFromBool(json_out)) + @as(u32, @intFromBool(has_schema)) +
            @as(u32, @intFromBool(has_grammar_s)) + @as(u32, @intFromBool(has_grammar_f));
        if (constraint_count > 1) {
            if (json_out)
                eprint("Warning: --json-output active; --grammar/--grammar-string/--json-schema ignored\n", .{})
            else if (has_schema)
                eprint("Warning: --json-schema active; --grammar/--grammar-string ignored\n", .{})
            else
                eprint("Warning: both --grammar and --grammar-string given; only --grammar-string takes effect\n", .{});
        }
    }

    // Warn when draft model is the same as target model (likely copy-paste mistake)
    if (res.option("draft-model")) |dm| {
        if (n_positionals > 0 and std.mem.eql(u8, dm, res.positional(0).?))
            eprint("Warning: --draft-model is the same file as the target model\n", .{});
    }

    // Warn about --mmproj without image/video in non-server mode (loads but never uses vision)
    if (res.option("mmproj") != null and res.option("image") == null and res.option("video") == null and !res.flag("serve"))
        eprint("Warning: --mmproj has no effect without --image, --video, or --serve\n", .{});

    // Warn about --video-fps without --video
    if (res.option("video-fps") != null and res.option("video") == null)
        eprint("Warning: --video-fps has no effect without --video\n", .{});

    // Warn about directional steering scales without a vector file
    if (res.option("dir-steering-file") == null) {
        if (res.option("dir-steering-ffn") != null)
            eprint("Warning: --dir-steering-ffn has no effect without --dir-steering-file\n", .{});
        if (res.option("dir-steering-attn") != null)
            eprint("Warning: --dir-steering-attn has no effect without --dir-steering-file\n", .{});
    }

    // Early file existence checks — fail fast before slow model loading
    if (grammar_path) |p| validateFileExists(p, "--grammar");
    if (res.option("image")) |p| validateFileExists(p, "--image");
    if (res.option("video")) |p| validateFileExists(p, "--video");
    if (res.option("lora")) |p| validateFileExists(p, "--lora");
    if (res.option("mmproj")) |p| validateFileExists(p, "--mmproj");
    if (res.option("draft-model")) |p| validateFileExists(p, "--draft-model");
    if (res.option("pflash-scorer")) |p| validateFileExists(p, "--pflash-scorer");
    if (res.option("spec-token-map")) |p| validateFileExists(p, "--spec-token-map");
    if (res.option("dir-steering-file")) |p| validateFileExists(p, "--dir-steering-file");
    if (res.option("expert-profile-in")) |p| validateFileExists(p, "--expert-profile-in");

    // JSON mode + interactive REPL would corrupt the JSON output stream
    if (json_mode and !res.flag("model-info") and !res.flag("serve") and n_positionals < 2) {
        if ((stdin_file.isTty(g_io) catch false)) {
            eprint("Error: --json requires a prompt or --model-info\n", .{});
            eprint("  Usage: agave model.gguf --json \"prompt\"\n", .{});
            eprint("  Or: echo \"prompt\" | agave model.gguf --json\n", .{});
            std.process.exit(2);
        }
    }

    // Resolve bind address before auth checks so loopback uses the parsed octets
    // (entire 127.0.0.0/8), not only the string forms "127.0.0.1"/"localhost".
    // CLI --host wins over AGAVE_HOST.
    const bind_host: [4]u8 = blk: {
        const host_str = res.option("host") orelse g_environ.get("AGAVE_HOST") orelse break :blk [4]u8{ 127, 0, 0, 1 };
        if (std.mem.eql(u8, host_str, "0.0.0.0") or std.mem.eql(u8, host_str, "0")) break :blk [4]u8{ 0, 0, 0, 0 };
        if (std.mem.eql(u8, host_str, "127.0.0.1") or std.mem.eql(u8, host_str, "localhost")) break :blk [4]u8{ 127, 0, 0, 1 };
        var parts: [4]u8 = undefined;
        if (!parseIpv4(host_str, &parts)) {
            eprint("Error: invalid host address '{s}' (expected IPv4, 'localhost', '0.0.0.0', or '0')\n", .{host_str});
            std.process.exit(2);
        }
        break :blk parts;
    };
    const api_key: ?[]const u8 = blk: {
        // Prefer AGAVE_API_KEY over --api-key so the active secret is not taken from
        // process listings when both are present (CLI value is still visible in ps).
        const key = g_environ.get("AGAVE_API_KEY") orelse res.option("api-key");
        // Non-loopback bind without a key exposes the full inference API.
        const is_loopback = bind_host[0] == 127;
        if (res.flag("serve") and !is_loopback and key == null) {
            const host_str = res.option("host") orelse g_environ.get("AGAVE_HOST") orelse "127.0.0.1";
            eprint("Error: --host {s} requires --api-key (or AGAVE_API_KEY) for non-loopback binds\n", .{host_str});
            eprint("  Use --host 127.0.0.1 for local-only access without auth.\n", .{});
            std.process.exit(2);
        }
        // Empty/whitespace key would satisfy "key present" checks while accepting any empty header.
        if (key) |k| {
            const trimmed = std.mem.trim(u8, k, " \t\r\n");
            if (trimmed.len == 0) {
                eprint("Error: --api-key (or AGAVE_API_KEY) must be non-empty\n", .{});
                std.process.exit(2);
            }
            break :blk trimmed;
        }
        break :blk null;
    };

    const parsed_cli: CliArgs = .{
        .model_path = res.positional(0).?,
        .prompt = res.positional(1),
        .serve = res.flag("serve"),
        .port = parseU16(port_raw, "port") orelse default_port,
        .max_tokens = parseU32(res.option("max-tokens"), "max-tokens") orelse default_max_tokens,
        .temperature = temperature,
        .top_p = top_p,
        .top_k = parseU32(res.option("top-k"), "top-k") orelse 0,
        .min_p = min_p,
        .repeat_penalty = repeat_penalty,
        .dry_multiplier = dry_multiplier,
        .dry_length = parseU32(res.option("dry-length"), "dry-length") orelse 2,
        .xtc_probability = xtc_probability,
        .xtc_threshold = xtc_threshold,
        .mirostat_mode = parseU32(res.option("mirostat-mode"), "mirostat-mode") orelse 0,
        .mirostat_tau = mirostat_tau,
        .mirostat_eta = mirostat_eta,
        .grammar_path = grammar_path,
        .grammar_string = grammar_string,
        .json_schema = json_schema,
        .json_output = res.flag("json-output"),
        .system_prompt = res.option("system"),
        .backend_choice = backend_choice,
        .device_id = device_id,
        .ctx_size = blk: {
            // --no-kv-cache: set ctx_size=0 → model init skips KV allocation entirely.
            // Suitable for prefill-only (embedding extraction, scoring) workloads.
            if (res.flag("no-kv-cache")) break :blk 0;
            const raw = res.option("ctx-size") orelse break :blk 0;
            if (std.mem.eql(u8, raw, "auto")) break :blk std.math.maxInt(u32);
            break :blk std.fmt.parseInt(u32, raw, 10) catch {
                eprint("Error: --ctx-size must be a non-negative integer or 'auto', got '{s}'\n", .{raw});
                std.process.exit(2);
            };
        },
        .no_kv_cache = res.flag("no-kv-cache"),
        .seed = parseU64(res.option("seed"), "seed") orelse @as(u64, @truncate(@as(u96, @bitCast(nanoTimestamp(g_io))))),
        .kv_type_k = blk: {
            if (res.option("kv-type-k")) |s| break :blk kvTypeOrExit(s, "--kv-type-k");
            if (res.option("cache-type-k")) |s| break :blk kvTypeOrExit(s, "--cache-type-k");
            const kv_str = res.option("kv-type") orelse break :blk KvQuantType.f16;
            // "turbo" preset: asymmetric K=q8_0 V=turbo4 (K needs precision for QK score accuracy)
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
        .kv_ram_budget = parseU32(res.option("kv-ram-budget"), "kv-ram-budget"),
        .kv_ssd_path = res.option("kv-ssd-path"),
        .kv_ssd_budget = parseU32(res.option("kv-ssd-budget"), "kv-ssd-budget"),
        .kv_eviction = if (res.option("kv-eviction")) |e| blk: {
            if (std.mem.eql(u8, e, "tri")) break :blk .tri;
            if (std.mem.eql(u8, e, "norm")) break :blk .norm;
            break :blk .none;
        } else .none,
        .kv_budget = parseU32(res.option("kv-budget"), "kv-budget") orelse 0,
        .host = bind_host,
        .api_key = api_key,
        .allow_cpu_fallback = res.flag("allow-cpu-fallback"),
        .debug = res.flag("debug"),
        .json = json_mode,
        .model_info = res.flag("model-info"),
        .benchmark = res.flag("benchmark"),
        .profile = res.flag("profile"),
        .megakernel = res.flag("megakernel"),
        .tp_degree = parseU32(res.option("tp"), "tp") orelse 1,
        .tp_rank = parseU32(res.option("rank"), "rank") orelse 0,
        .tp_peers = res.option("peers"),
        .transport = if (res.option("transport")) |t| blk: {
            const choice = std.meta.stringToEnum(TransportChoice, t) orelse {
                eprint("Error: unknown transport '{s}'\n", .{t});
                eprint("  Valid options: auto, tcp, shm, nccl\n", .{});
                std.process.exit(2);
            };
            if (choice == .rdma or choice == .udp or choice == .grpc) {
                eprint("Error: transport '{s}' is not implemented (use auto, tcp, shm, or nccl)\n", .{t});
                std.process.exit(2);
            }
            break :blk choice;
        } else .auto,
        .pp_degree = parseU32(res.option("pp"), "pp") orelse 1,
        .disagg = res.flag("disagg"),
        .use_mmap = res.flag("mmap"),
        .prefill_batch_size = parseU32(res.option("prefill-batch-size"), "prefill-batch-size") orelse default_chunk_size,
        .lora_path = res.option("lora"),
        .mmproj = res.option("mmproj"),
        .image = res.option("image"),
        .draft_model_path = res.option("draft-model"),
        .mtp_model_path = res.option("mtp-model"),
        .sleep_after_s = parseU32(res.option("sleep-after"), "sleep-after") orelse 0,
        .max_batch_size = parseU32(res.option("max-batch-size"), "max-batch-size") orelse 8,
        .rate_limit_rpm = parseU32(res.option("rate-limit-rpm"), "rate-limit-rpm") orelse 0,
        .rate_limit_tpm = parseU32(res.option("rate-limit-tpm"), "rate-limit-tpm") orelse 0,
        .video = res.option("video"),
        .video_fps = blk: {
            // Range already validated above; re-parse for the struct field.
            if (parseF32(res.option("video-fps"), "video-fps")) |v| break :blk v;
            break :blk 1.0;
        },
        .spec_tokens = parseU32(res.option("spec-tokens"), "spec-tokens") orelse 5,
        .tree_budget = parseU32(res.option("tree-budget"), "tree-budget") orelse 64,
        .spec_mode = blk: {
            const dm = res.option("draft-model");
            const sm = res.option("spec-mode");
            if (sm) |s| {
                if (std.mem.eql(u8, s, "standard")) break :blk SpecMode.standard;
                if (std.mem.eql(u8, s, "ddtree")) break :blk SpecMode.ddtree;
                if (std.mem.eql(u8, s, "self")) break :blk SpecMode.self_spec;
                if (std.mem.eql(u8, s, "ngram")) break :blk SpecMode.ngram;
                if (std.mem.eql(u8, s, "suffix")) break :blk SpecMode.suffix;
                if (std.mem.eql(u8, s, "mtp")) break :blk SpecMode.mtp;
                // Medusa heads share the MTP inference path; normalize at the CLI boundary.
                if (std.mem.eql(u8, s, "medusa")) break :blk SpecMode.mtp;
                if (std.mem.eql(u8, s, "eagle")) break :blk SpecMode.eagle;
                if (std.mem.eql(u8, s, "eagle3")) break :blk SpecMode.eagle3;
                if (std.mem.eql(u8, s, "mlp")) break :blk SpecMode.mlp;
                if (std.mem.eql(u8, s, "lookahead")) break :blk SpecMode.lookahead;
                if (std.mem.eql(u8, s, "pflash")) break :blk SpecMode.pflash;
                if (std.mem.eql(u8, s, "dspark")) break :blk SpecMode.dspark;
                if (std.mem.eql(u8, s, "dflash2")) break :blk SpecMode.dflash2;
                // DFlash CLI alias: the published checkpoints are DFlash2.
                if (std.mem.eql(u8, s, "dflash")) break :blk SpecMode.dflash2;
                if (std.mem.eql(u8, s, "auto")) break :blk if (dm != null) SpecMode.ddtree else SpecMode.ngram;
                eprint("Error: unknown --spec-mode '{s}' (expected: auto, standard, ddtree, self, ngram, suffix, lookahead, mtp, medusa, eagle, eagle3, mlp, pflash, dspark, dflash2)\n", .{s});
                std.process.exit(2);
            }
            break :blk if (dm != null) SpecMode.ddtree else SpecMode.none;
        },
        .draft_layers = parseU32(res.option("draft-layers"), "draft-layers"),
        .pflash_alpha = blk: {
            if (res.option("pflash-alpha")) |s| {
                const v = std.fmt.parseFloat(f32, s) catch {
                    eprint("Error: --pflash-alpha must be a number\n", .{});
                    std.process.exit(2);
                };
                if (v < 0.0 or v > 2.0) {
                    eprint("Error: --pflash-alpha must be in [0.0, 2.0] (got {d:.2})\n", .{v});
                    std.process.exit(2);
                }
                break :blk v;
            }
            break :blk 0.85;
        },
        .pflash_block_size = blk: {
            const v = parseU32(res.option("pflash-block-size"), "pflash-block-size") orelse 64;
            if (v == 0) {
                eprint("Warning: --pflash-block-size must be > 0, using default 64\n", .{});
                break :blk 64;
            }
            break :blk v;
        },
        .pflash_scorer_path = res.option("pflash-scorer"),
        .spec_token_map = res.option("spec-token-map"),
        .dir_steering_file = res.option("dir-steering-file"),
        .dir_steering_ffn = blk: {
            if (res.option("dir-steering-ffn")) |s| {
                break :blk std.fmt.parseFloat(f32, s) catch {
                    eprint("Error: --dir-steering-ffn must be a number\n", .{});
                    std.process.exit(2);
                };
            }
            // Default: 1.0 when a steering file is provided, 0 otherwise
            break :blk if (res.option("dir-steering-file") != null) @as(f32, 1.0) else @as(f32, 0);
        },
        .dir_steering_attn = blk: {
            if (res.option("dir-steering-attn")) |s| {
                break :blk std.fmt.parseFloat(f32, s) catch {
                    eprint("Error: --dir-steering-attn must be a number\n", .{});
                    std.process.exit(2);
                };
            }
            break :blk 0;
        },
        .diffusion_steps = @max(1, parseU32(res.option("diffusion-steps"), "diffusion-steps") orelse 16),
        .diffusion_canvas = @max(1, parseU32(res.option("diffusion-canvas"), "diffusion-canvas") orelse 256),
        .diffusion_confidence = blk: {
            if (res.option("diffusion-confidence")) |s| {
                const v = std.fmt.parseFloat(f32, s) catch {
                    eprint("Error: --diffusion-confidence must be a number\n", .{});
                    std.process.exit(2);
                };
                if (v < 0.0 or v > 1.0) {
                    eprint("Error: --diffusion-confidence must be in [0.0, 1.0] (got {d:.2})\n", .{v});
                    std.process.exit(2);
                }
                break :blk v;
            }
            break :blk 0.5;
        },
        .ssd_streaming = res.flag("ssd-streaming"),
        .ssd_cache_slots = blk: {
            const v = parseU32(res.option("ssd-cache-slots"), "ssd-cache-slots") orelse 256;
            if (v == 0) {
                eprint("Warning: --ssd-cache-slots is 0; no experts will be cached\n", .{});
            }
            break :blk v;
        },
        .expert_profile_out = res.option("expert-profile-out"),
        .expert_profile_in = res.option("expert-profile-in"),
        .power_pct = blk: {
            const v = parseU32(res.option("power"), "power") orelse 100;
            if (v == 0 or v > 100) {
                eprint("Error: --power must be 1-100 (got {d})\n", .{v});
                std.process.exit(2);
            }
            break :blk v;
        },
        .frontier_bench = res.flag("frontier-bench"),
        .frontier_ctx = res.option("frontier-ctx") orelse "512,2048,8192",
        .user_set = .{
            .temperature = res.option("temperature") != null,
            .top_p = res.option("top-p") != null,
            .top_k = res.option("top-k") != null,
            .repeat_penalty = res.option("repeat-penalty") != null,
            .max_tokens = res.option("max-tokens") != null,
            .ctx_size = res.option("ctx-size") != null or res.flag("no-kv-cache"),
        },
    };

    if (spec_caps.unsatisfied(parsed_cli.spec_mode, .{ .draft = parsed_cli.draft_model_path != null })) |p| {
        if (p == .draft) {
            eprint("Error: --spec-mode {s} waiting for {s}\n", .{ @tagName(parsed_cli.spec_mode), spec_caps.providerName(p) });
            eprint("  {s}\n", .{spec_caps.howToProvide(p)});
            std.process.exit(2);
        }
    }
    return parsed_cli;
}

fn parseIpv4(s: []const u8, out: *[4]u8) bool {
    if (s.len == 0) return false;
    if (s[0] == '.' or s[s.len - 1] == '.') return false;
    if (std.mem.indexOf(u8, s, "..") != null) return false;
    var parts: [4]u8 = .{ 0, 0, 0, 0 };
    var part_idx: usize = 0;
    var acc: u32 = 0;
    for (s) |c| {
        if (c == '.') {
            if (acc > 255 or part_idx >= 4) return false;
            parts[part_idx] = @intCast(acc);
            part_idx += 1;
            acc = 0;
        } else if (c >= '0' and c <= '9') {
            acc = std.math.mul(u32, acc, 10) catch return false;
            acc = std.math.add(u32, acc, c - '0') catch return false;
        } else {
            return false;
        }
    }
    if (acc > 255 or part_idx != 3) return false;
    parts[3] = @intCast(acc);
    out.* = parts;
    return true;
}

const PeerAddr = struct { host: [4]u8, port: u16 };

/// Parse "host:port" or "host" peer address string. Returns null on invalid input.
fn parsePeerAddr(peers_str: []const u8, fallback_port: u16) ?PeerAddr {
    var result = PeerAddr{ .host = .{ 0, 0, 0, 0 }, .port = fallback_port };
    if (std.mem.indexOfScalar(u8, peers_str, ':')) |colon| {
        result.port = std.fmt.parseInt(u16, peers_str[colon + 1 ..], 10) catch return null;
        if (!parseIpv4(peers_str[0..colon], &result.host)) return null;
    } else {
        if (!parseIpv4(peers_str, &result.host)) return null;
    }
    return result;
}

const TransportMod = @import("parallel/transport.zig");

fn resolveTransportKind(choice: TransportChoice, peers_str: []const u8) error{TransportNotImplemented}!TransportMod.TransportKind {
    return switch (choice) {
        .tcp => .tcp,
        .shm => .shm,
        .nccl => .nccl,
        .rdma, .udp, .grpc => error.TransportNotImplemented,
        .auto => {
            const is_local = std.mem.eql(u8, peers_str, "localhost") or std.mem.eql(u8, peers_str, "127.0.0.1");
            const resolved: TransportMod.TransportKind = if (is_local) .shm else .tcp;
            std.log.info("transport: auto → {s}", .{@tagName(resolved)});
            return resolved;
        },
    };
}

/// Initializes and connects the distributed-inference transport (TCP, shared-memory, or NCCL) for the given rank and peers.
fn setupTransport(allocator: std.mem.Allocator, peers_str: []const u8, rank: u32, world_size: u32, choice: TransportChoice, port_base: u16, be_union: anytype) ?*TransportMod.Transport {
    const t = allocator.create(TransportMod.Transport) catch return null;
    var transport_ok = false;
    defer if (!transport_ok) allocator.destroy(t);
    var kind = resolveTransportKind(choice, peers_str) catch {
        std.log.err("transport '{s}' is not implemented (use auto, tcp, shm, or nccl)", .{@tagName(choice)});
        return null;
    };
    t.* = TransportMod.Transport.init(allocator, kind, rank, world_size) catch return null;

    if (kind == .shm) {
        t.setupShm() catch {
            std.log.warn("shm setup failed, falling back to tcp", .{});
            t.kind = .tcp;
            kind = .tcp;
        };
    }

    if (kind == .shm) {
        transport_ok = true;
        return t;
    }

    // NCCL: establish TCP first (for unique ID exchange), then init NCCL
    const want_nccl = (kind == .nccl);

    // TCP path
    const peer = parsePeerAddr(peers_str, port_base) orelse {
        std.log.err("invalid peer address: {s}", .{peers_str});
        return null;
    };
    const host = peer.host;
    const port = peer.port;
    if (rank == 0) {
        var la: std.posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = 0 };
        const ls = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
        if (ls < 0) return null;
        defer _ = std.c.close(ls);
        var one: c_int = 1;
        _ = std.c.setsockopt(ls, std.posix.SOL.SOCKET, std.posix.SO.REUSEADDR, @ptrCast(&one), @sizeOf(c_int));
        if (std.c.bind(ls, @ptrCast(&la), @sizeOf(@TypeOf(la))) != 0) return null;
        if (std.c.listen(ls, 1) != 0) return null;
        std.log.info("waiting for rank 1 on port {d}...", .{port});
        t.acceptPeer(ls) catch return null;
        std.log.info("rank 1 connected", .{});
    } else {
        std.log.info("connecting to rank 0 at {d}.{d}.{d}.{d}:{d}...", .{ host[0], host[1], host[2], host[3], port });
        t.connectPeer(host, port) catch return null;
        std.log.info("connected to rank 0", .{});
    }

    // Measure peer RTT via TCP ping-pong (4-byte round-trip)
    const rtt_us = measurePeerRtt(t, rank);
    if (rtt_us > 0) std.log.info("peer RTT: {d} µs", .{rtt_us});

    // Exchange device capabilities for topology-aware partitioning
    const local_mem = backend_mod.detectSystemMem();
    const peer_mem = exchangeDeviceCaps(t, rank, local_mem);
    if (peer_mem > 0) {
        // Store for topology-aware PP layer assignment
        t.peer_mem = peer_mem;
        t.local_mem = local_mem;
    }

    // NCCL: wire CUDA interop BEFORE init so ncclCommInitRank has a valid context
    if (want_nccl) {
        switch (be_union) {
            .cuda => |cuda_be| {
                if (comptime build_options.enable_cuda) {
                    t.cuda_sync = cuda_be.cuCtxSynchronize;
                    t.cuda_ctx = cuda_be.context;
                    t.cuda_ctx_set = if (cuda_be.cuCtxSetCurrent) |f| f else null;
                    t.cuda_backend = @ptrCast(cuda_be);
                    t.cuda_get_dev_ptr = backend_mod.CudaBackend.getDevicePtrOpaque;
                    t.cuda_mem_alloc = cuda_be.cuMemAlloc;
                    t.cuda_mem_free = cuda_be.cuMemFree;
                    t.cuda_memcpy_htod = cuda_be.cuMemcpyHtoD;
                    t.cuda_memcpy_dtoh = cuda_be.cuMemcpyDtoH;
                }
            },
            else => {},
        }
        t.setupNccl() catch |err| {
            std.log.warn("NCCL init failed ({s}), using TCP", .{@errorName(err)});
            t.kind = .tcp;
            transport_ok = true;
            return t;
        };
        // Both ranks are synchronized after setupNccl (TCP ID exchange).
        // Init comm NOW while both are at the same point.
        t.ensureNcclComm();
    }
    transport_ok = true;
    return t;
}

/// Exchange device capabilities with peer for topology-aware partitioning.
/// Returns peer's available memory in bytes, or 0 on failure.
fn exchangeDeviceCaps(t: *TransportMod.Transport, rank: u32, local_mem: usize) usize {
    if (t.tcp_connected == 0) return 0;
    const fd = t.tcp_fds[0];
    var local_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &local_bytes, @intCast(local_mem), .little);
    var remote_bytes: [8]u8 = undefined;

    if (rank == 0) {
        _ = std.posix.system.send(fd, &local_bytes, 8, 0);
        _ = std.posix.system.recv(fd, &remote_bytes, 8, 0);
    } else {
        _ = std.posix.system.recv(fd, &remote_bytes, 8, 0);
        _ = std.posix.system.send(fd, &local_bytes, 8, 0);
    }
    const peer_mem = std.mem.readInt(u64, &remote_bytes, .little);
    if (peer_mem > 0) {
        std.log.info("topology: local {d} MB, peer {d} MB", .{
            local_mem / (1024 * 1024), peer_mem / (1024 * 1024),
        });
    }
    return @intCast(peer_mem);
}

/// Measure round-trip time to peer via TCP ping-pong. Returns µs, or 0 on failure.
fn measurePeerRtt(t: *TransportMod.Transport, rank: u32) u64 {
    if (t.tcp_connected == 0) return 0;
    const fd = t.tcp_fds[0];
    var ping: [4]u8 = .{ 'P', 'I', 'N', 'G' };
    var pong: [4]u8 = undefined;
    var ts_start: std.posix.system.timespec = undefined;
    var ts_end: std.posix.system.timespec = undefined;
    if (rank == 0) {
        _ = std.posix.system.clock_gettime(.MONOTONIC, &ts_start);
        _ = std.posix.system.send(fd, &ping, 4, 0);
        _ = std.posix.system.recv(fd, &pong, 4, 0);
        _ = std.posix.system.clock_gettime(.MONOTONIC, &ts_end);
    } else {
        _ = std.posix.system.clock_gettime(.MONOTONIC, &ts_start);
        _ = std.posix.system.recv(fd, &pong, 4, 0);
        _ = std.posix.system.send(fd, &ping, 4, 0);
        _ = std.posix.system.clock_gettime(.MONOTONIC, &ts_end);
    }
    const start_us: u64 = @intCast(ts_start.sec * 1_000_000 + @divTrunc(ts_start.nsec, 1000));
    const end_us: u64 = @intCast(ts_end.sec * 1_000_000 + @divTrunc(ts_end.nsec, 1000));
    return end_us -| start_us;
}

fn parseUint(comptime T: type, s: ?[]const u8, comptime flag: []const u8) ?T {
    const str = s orelse return null;
    return std.fmt.parseInt(T, str, 10) catch {
        eprint("Error: invalid value for --" ++ flag ++ ": '{s}' is not a valid integer\n", .{str});
        eprint("Run 'agave --help' for more information.\n", .{});
        std.process.exit(2);
    };
}

fn parseU32(s: ?[]const u8, comptime flag: []const u8) ?u32 {
    return parseUint(u32, s, flag);
}
fn parseU64(s: ?[]const u8, comptime flag: []const u8) ?u64 {
    return parseUint(u64, s, flag);
}

/// True when the NO_COLOR env var disables color output.
/// Per https://no-color.org the variable must be present and non-empty;
/// `NO_COLOR=` (empty) keeps auto behavior.
fn noColorRequested(val: ?[]const u8) bool {
    const v = val orelse return false;
    return v.len > 0;
}
fn parseU16(s: ?[]const u8, comptime flag: []const u8) ?u16 {
    return parseUint(u16, s, flag);
}

/// Check if a long option name matches any known CLI spec.
fn isKnownSpec(name: []const u8) bool {
    for (cli_specs) |spec| {
        if (std.mem.eql(u8, spec.long, name)) return true;
    }
    return false;
}

/// Find the closest matching spec for a typo suggestion (edit distance ≤ 2).
fn suggestSpec(name: []const u8) ?[]const u8 {
    for (cli_specs) |spec| {
        if (closeMatch(name, spec.long)) return spec.long;
    }
    return null;
}

/// Check if two strings differ by at most 2 substitutions, or 1 insertion/deletion.
fn closeMatch(a: []const u8, b: []const u8) bool {
    if (a.len == b.len) {
        var diffs: usize = 0;
        for (a, b) |ca, cb| {
            if (ca != cb) diffs += 1;
            if (diffs > 2) return false;
        }
        return diffs > 0 and diffs <= 2;
    }
    if (a.len + 1 == b.len) return insertionMatch(a, b);
    if (b.len + 1 == a.len) return insertionMatch(b, a);
    return false;
}

/// Check if `shorter` matches `longer` with exactly one character inserted.
fn insertionMatch(shorter: []const u8, longer: []const u8) bool {
    var si: usize = 0;
    var li: usize = 0;
    var skips: usize = 0;
    while (si < shorter.len and li < longer.len) {
        if (shorter[si] == longer[li]) {
            si += 1;
            li += 1;
        } else {
            skips += 1;
            if (skips > 1) return false;
            li += 1;
        }
    }
    return true;
}

/// Reject `--flag=value` on boolean flags (parser stores them in options, not flags).
fn rejectEqualsOnFlag(res: *const cli_mod.ParseResult) void {
    for (cli_specs) |spec| {
        if (spec.kind != .flag) continue;
        if (res.option(spec.long)) |val| {
            eprint("Error: --{s} does not take a value (got '--{s}={s}')\n", .{ spec.long, spec.long, val });
            eprint("  Use --{s} alone, without '=...'\n", .{spec.long});
            eprint("Run 'agave --help' for more information.\n", .{});
            std.process.exit(2);
        }
    }
}

/// Reject flags or options not recognized by cli_specs (exit 2).
/// Catches typos like --temeprature that would otherwise silently use defaults.
fn rejectUnknownOptions(res: *const cli_mod.ParseResult) void {
    var found = false;
    var flag_it = res.flags.iterator();
    while (flag_it.next()) |entry| {
        if (!isKnownSpec(entry.key_ptr.*)) {
            eprint("Error: unknown option '--{s}'", .{entry.key_ptr.*});
            if (suggestSpec(entry.key_ptr.*)) |s|
                eprint(" (did you mean '--{s}'?)", .{s});
            eprint("\n", .{});
            found = true;
        }
    }
    var opt_it = res.options.iterator();
    while (opt_it.next()) |entry| {
        if (!isKnownSpec(entry.key_ptr.*)) {
            eprint("Error: unknown option '--{s}'", .{entry.key_ptr.*});
            if (suggestSpec(entry.key_ptr.*)) |s|
                eprint(" (did you mean '--{s}'?)", .{s});
            eprint("\n", .{});
            found = true;
        }
    }
    if (found) {
        eprint("Run 'agave --help' for more information.\n", .{});
        std.process.exit(2);
    }
}

/// Reject when an option's value looks like a known flag that was accidentally consumed.
/// Catches `--system --serve` (system prompt becomes "--serve", --serve flag lost)
/// and `--system -s` (short flag consumed as value).
fn rejectFlagAsValue(res: *const cli_mod.ParseResult) void {
    var opt_it = res.options.iterator();
    while (opt_it.next()) |entry| {
        const val = entry.value_ptr.*;
        if (val.len > 2 and val[0] == '-' and val[1] == '-' and isKnownSpec(val[2..])) {
            eprint("Error: --{s} has value '{s}' which looks like a flag (missing value for --{s}?)\n", .{ entry.key_ptr.*, val, entry.key_ptr.* });
            eprint("Run 'agave --help' for more information.\n", .{});
            std.process.exit(2);
        } else if (val.len == 2 and val[0] == '-' and val[1] != '-') {
            if (isKnownShort(val[1])) {
                eprint("Error: --{s} has value '{s}' which looks like a flag (missing value for --{s}?)\n", .{ entry.key_ptr.*, val, entry.key_ptr.* });
                eprint("Run 'agave --help' for more information.\n", .{});
                std.process.exit(2);
            }
        }
    }
}

/// True if a positional looks like an unknown short option (-z) or unknown cluster (-xy).
/// Letter-only so prompts like "-5" are not rejected. Paths like "-n" need `./-n` or `--`.
/// Known clusters (e.g. -qV) are parsed in cli.zig and never land here.
fn looksLikeUnknownShortOpt(pos: []const u8) bool {
    if (pos.len < 2 or pos[0] != '-' or pos[1] == '-') return false;
    for (pos[1..]) |c| {
        if (!std.ascii.isAlphabetic(c)) return false;
    }
    return true;
}

/// Reject unknown short options that landed in positionals (parser treats them as args).
fn rejectUnknownShortPositionals(res: *const cli_mod.ParseResult) void {
    for (res.positionals.items) |pos| {
        if (looksLikeUnknownShortOpt(pos)) {
            eprint("Error: unknown option '{s}'\n", .{pos});
            eprint("Run 'agave --help' for more information.\n", .{});
            std.process.exit(2);
        }
    }
}

/// Check if a short character matches any known CLI spec.
fn isKnownShort(ch: u8) bool {
    for (cli_specs) |spec| {
        if (spec.short) |s| {
            if (s == ch) return true;
        }
    }
    return false;
}

fn parseF32(s: ?[]const u8, comptime flag: []const u8) ?f32 {
    const str = s orelse return null;
    const val = std.fmt.parseFloat(f32, str) catch {
        eprint("Error: invalid value for --" ++ flag ++ ": '{s}' is not a valid number\n", .{str});
        eprint("Run 'agave --help' for more information.\n", .{});
        std.process.exit(2);
    };
    if (!std.math.isFinite(val)) {
        eprint("Error: --" ++ flag ++ " must be a finite number, got '{s}'\n", .{str});
        eprint("Run 'agave --help' for more information.\n", .{});
        std.process.exit(2);
    }
    return val;
}

/// Check that a file path exists before expensive model loading.
fn validateFileExists(path: []const u8, comptime flag: []const u8) void {
    const file = Io.Dir.cwd().openFile(g_io, path, .{}) catch {
        eprint("Error: " ++ flag ++ " file not found: '{s}'\n", .{path});
        std.process.exit(2);
    };
    file.close(g_io);
}

/// Built-in benchmark: prefill a short prompt, decode N tokens, report stats.
fn runBenchmark(model: *Model, tok_state: anytype, allocator: std.mem.Allocator, cli: anytype, eog: anytype) void {
    const bench_prompt = "The quick brown fox jumps over the lazy dog. Once upon a time";
    var tok_if = tok_state.*.tokenizer();
    const token_ids = tok_if.encode(bench_prompt) catch {
        eprint("Benchmark: tokenizer encode failed\n", .{});
        return;
    };
    defer allocator.free(token_ids);
    const n_prompt = token_ids.len;
    const n_gen = cli.max_tokens;

    // Prefill (batched when model supports it, sequential fallback)
    var ts_start: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts_start);
    _ = model.prefill(token_ids) catch {
        eprint("Benchmark: prefill failed\n", .{});
        return;
    };
    var ts_prefill: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts_prefill);

    // Decode
    var last: u32 = math_ops.argmax(model.getLogits());
    var gen_count: u32 = 0;
    while (gen_count < n_gen) {
        if (isEogToken(last, eog)) break;
        last = model.forward(last) catch |err| {
            eprint("benchmark: decode forward failed: {}\n", .{err});
            break;
        };
        last = math_ops.argmax(model.getLogits());
        gen_count += 1;
    }
    var ts_end: std.posix.system.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts_end);

    const prefill_us = (@as(i64, ts_prefill.sec) - @as(i64, ts_start.sec)) * 1_000_000 + @divTrunc(@as(i64, ts_prefill.nsec) - @as(i64, ts_start.nsec), 1000);
    const decode_us = (@as(i64, ts_end.sec) - @as(i64, ts_prefill.sec)) * 1_000_000 + @divTrunc(@as(i64, ts_end.nsec) - @as(i64, ts_prefill.nsec), 1000);
    const prefill_tps: f64 = if (prefill_us > 0) @as(f64, @floatFromInt(n_prompt)) / (@as(f64, @floatFromInt(prefill_us)) / 1e6) else 0;
    const decode_tps: f64 = if (decode_us > 0) @as(f64, @floatFromInt(gen_count)) / (@as(f64, @floatFromInt(decode_us)) / 1e6) else 0;
    const prefill_ms = @as(f64, @floatFromInt(prefill_us)) / 1000.0;
    const decode_ms = @as(f64, @floatFromInt(decode_us)) / 1000.0;

    var buf: [1024]u8 = undefined;
    const msg = if (cli.json)
        std.fmt.bufPrint(&buf,
            \\{{"prefill_tokens":{d},"prefill_ms":{d:.1},"prefill_tps":{d:.1},"decode_tokens":{d},"decode_ms":{d:.1},"decode_tps":{d:.1},"ttft_ms":{d:.1}}}
            \\
        , .{ n_prompt, prefill_ms, prefill_tps, gen_count, decode_ms, decode_tps, prefill_ms })
    else
        std.fmt.bufPrint(&buf,
            \\
            \\Benchmark Results:
            \\  Prefill: {d} tokens in {d:.1} ms ({d:.1} tok/s)
            \\  Decode:  {d} tokens in {d:.1} ms ({d:.1} tok/s)
            \\  TTFT:    {d:.1} ms
            \\
        , .{ n_prompt, prefill_ms, prefill_tps, gen_count, decode_ms, decode_tps, prefill_ms });
    if (msg) |m| _ = std.posix.system.write(stdout_file.handle, m.ptr, m.len) else |_| {}
}

/// Frontier benchmark (ds4-bench style): snapshot KV at each context frontier,
/// measure prefill and generation throughput separately per length.
/// Mirrors ds4's approach: greedy probe at each frontier, restore state, continue.
fn runFrontierBench(model: *Model, tok_state: anytype, allocator: std.mem.Allocator, cli: anytype, eog: anytype) void {
    // Parse comma-separated context lengths from --frontier-ctx
    const frontier_str = cli.frontier_ctx;
    var frontiers_buf: [16]u32 = undefined;
    var n_frontiers: usize = 0;
    {
        var it = std.mem.splitScalar(u8, frontier_str, ',');
        while (it.next()) |s| {
            if (n_frontiers >= frontiers_buf.len) break;
            const v = std.fmt.parseInt(u32, std.mem.trim(u8, s, " "), 10) catch continue;
            frontiers_buf[n_frontiers] = v;
            n_frontiers += 1;
        }
    }
    if (n_frontiers == 0) {
        eprint("frontier-bench: no valid context lengths in --frontier-ctx\n", .{});
        return;
    }

    // Build a prompt long enough to cover the largest frontier.
    const max_ctx = std.mem.max(u32, frontiers_buf[0..n_frontiers]);
    var tok_if = tok_state.*.tokenizer();

    // Repeat a filler sentence to fill max_ctx tokens.
    const filler = "The quick brown fox jumps over the lazy dog. ";
    const filler_ids = tok_if.encode(filler) catch {
        eprint("frontier-bench: encode failed\n", .{});
        return;
    };
    defer allocator.free(filler_ids);
    if (filler_ids.len == 0) {
        eprint("frontier-bench: empty filler token list\n", .{});
        return;
    }
    var full_prompt = std.ArrayList(u32).empty;
    defer full_prompt.deinit(allocator);
    while (full_prompt.items.len < max_ctx) {
        full_prompt.appendSlice(allocator, filler_ids) catch break;
    }
    const prompt = full_prompt.items[0..@min(full_prompt.items.len, max_ctx)];

    const probe_tokens: u32 = 16; // greedy probe length at each frontier
    var cursor: usize = 0; // tokens prefilled so far

    if (!cli.json) eprint("\nFrontier Benchmark ({d} frontiers, {d} probe tokens each):\n", .{ n_frontiers, probe_tokens });
    if (cli.json) _ = std.posix.system.write(stdout_file.handle, "[", 1);

    for (frontiers_buf[0..n_frontiers], 0..) |ctx_len, fi| {
        // Prefill from cursor to ctx_len
        const slice = if (ctx_len <= prompt.len) prompt[cursor..ctx_len] else prompt[cursor..];
        if (slice.len == 0) continue;

        var ts0: std.posix.system.timespec = undefined;
        _ = std.posix.system.clock_gettime(.MONOTONIC, &ts0);
        _ = model.prefill(slice) catch {
            eprint("frontier-bench: prefill failed at ctx={d}\n", .{ctx_len});
            break;
        };
        var ts1: std.posix.system.timespec = undefined;
        _ = std.posix.system.clock_gettime(.MONOTONIC, &ts1);
        cursor = @min(ctx_len, prompt.len);

        // Export KV snapshot before probe (64 MB should cover most models at frontier sizes).
        const kv_export_cap: usize = 64 * 1024 * 1024;
        const kv_snapshot_buf = allocator.alloc(u8, kv_export_cap) catch null;
        defer if (kv_snapshot_buf) |s| allocator.free(s);
        const kv_snap_len: usize = if (kv_snapshot_buf) |buf| model.exportKvPrefix(buf, cursor) else 0;

        // Greedy probe starting from the last prefill logits.
        var last = math_ops.argmax(model.getLogits());
        var gen: u32 = 0;
        var ts_dec_start: std.posix.system.timespec = undefined;
        _ = std.posix.system.clock_gettime(.MONOTONIC, &ts_dec_start);
        while (gen < probe_tokens) : (gen += 1) {
            if (isEogToken(last, eog)) break;
            last = model.forward(last) catch |err| {
                eprint("benchmark: decode forward failed: {}\n", .{err});
                break;
            };
            last = math_ops.argmax(model.getLogits());
        }
        var ts2: std.posix.system.timespec = undefined;
        _ = std.posix.system.clock_gettime(.MONOTONIC, &ts2);

        // Restore KV state to the snapshot so the next frontier continues cleanly.
        if (kv_snapshot_buf) |s| {
            if (kv_snap_len > 0) _ = model.importKvPrefix(s[0..kv_snap_len], cursor);
        }

        const pf_us: i64 = (@as(i64, ts1.sec) - @as(i64, ts0.sec)) * 1_000_000 + @divTrunc(@as(i64, ts1.nsec) - @as(i64, ts0.nsec), 1000);
        const dec_us: i64 = (@as(i64, ts2.sec) - @as(i64, ts_dec_start.sec)) * 1_000_000 + @divTrunc(@as(i64, ts2.nsec) - @as(i64, ts_dec_start.nsec), 1000);
        const pf_tps: f64 = if (pf_us > 0) @as(f64, @floatFromInt(slice.len)) / (@as(f64, @floatFromInt(pf_us)) / 1e6) else 0;
        const dec_tps: f64 = if (dec_us > 0 and gen > 0) @as(f64, @floatFromInt(gen)) / (@as(f64, @floatFromInt(dec_us)) / 1e6) else 0;

        var buf: [256]u8 = undefined;
        if (cli.json) {
            const comma: []const u8 = if (fi + 1 < n_frontiers) "," else "";
            const msg = std.fmt.bufPrint(&buf, "{{\"ctx\":{d},\"prefill_tokens\":{d},\"prefill_tps\":{d:.1},\"decode_tps\":{d:.1}}}{s}", .{ ctx_len, slice.len, pf_tps, dec_tps, comma }) catch continue;
            _ = std.posix.system.write(stdout_file.handle, msg.ptr, msg.len);
        } else {
            const msg = std.fmt.bufPrint(&buf, "  ctx={d:6}: prefill {d:.1} t/s  decode {d:.1} t/s  (prefill {d} tok)\n", .{ ctx_len, pf_tps, dec_tps, slice.len }) catch continue;
            _ = std.posix.system.write(stdout_file.handle, msg.ptr, msg.len);
        }
    }
    if (cli.json) _ = std.posix.system.write(stdout_file.handle, "]\n", 2);
}

fn printUsage() void {
    const usage =
        \\agave: Zig LLM inference engine
        \\
        \\USAGE:
        \\  agave [OPTIONS] <model.gguf|model-dir/> [prompt]
        \\  agave [OPTIONS] -- <model.gguf|model-dir/> [prompt]
        \\  echo "prompt" | agave model.gguf
        \\
        \\ARGUMENTS:
        \\  <model.gguf|model-dir/>  Path to GGUF model file or SafeTensors directory
        \\  [prompt]                 Text prompt (omit for interactive REPL)
        \\
        \\GENERAL:
        \\  -h, --help             Show this help message and exit
        \\  -v, --version          Print version and exit
        \\  -q, --quiet            Suppress banner and stats (raw output only)
        \\                         Short boolean flags may be clustered (e.g. -qV)
        \\      --color <MODE>     Color mode: auto, always, never [default: auto]
        \\      --no-color         Disable colored output (same as --color=never; do not combine with --color)
        \\
        \\GENERATION:
        \\  -n, --max-tokens <N>      Maximum tokens to generate [default: 512]
        \\  -t, --temperature <T>     Sampling temperature, 0 = greedy [default: 0]
        \\      --top-p <P>           Nucleus sampling threshold [default: 1.0]
        \\      --top-k <K>           Top-k sampling, 0 = disabled [default: 0]
        \\      --repeat-penalty <R>  Repetition penalty [default: 1.0]
        \\      --min-p <P>           Min-p sampling: keep tokens with prob >= P * max_prob [default: 0]
        \\      --dry-multiplier <M>  DRY n-gram repetition penalty [default: 0 = disabled]
        \\      --dry-length <N>      DRY minimum n-gram length [default: 2]
        \\      --xtc-probability <P> XTC exclude-top-choices probability [default: 0]
        \\      --xtc-threshold <T>   XTC probability threshold [default: 0.1]
        \\      --mirostat-mode <N>   Mirostat sampling: 0=off, 2=Mirostat 2.0 [default: 0]
        \\      --mirostat-tau <T>    Mirostat target entropy [default: 5.0]
        \\      --mirostat-eta <E>    Mirostat learning rate [default: 0.1]
        \\      --seed <N>            Random seed for sampling [default: random]
        \\      --system <TEXT>       System prompt for chat formatting
        \\      --grammar <FILE>      GBNF grammar file for constrained decoding
        \\      --grammar-string <G>  Inline GBNF grammar string
        \\      --json-output         Constrain generation to valid JSON via grammar (not output format; see --json)
        \\      --json-schema <JSON>  JSON schema for structured output
        \\
        \\BACKEND & MODEL:
        \\      --backend <BE>        Compute backend: auto, cpu, metal, vulkan, cuda, rocm, webgpu [default: auto]
        \\      --device <N>          GPU device index for CUDA/ROCm/Vulkan [default: 0]
        \\      --list-devices        List available compute devices and exit
        \\      --ctx-size <N|auto>   Context window size; 0 = full, auto = fit to memory [default: 4096 or model limit]
        \\      --allow-cpu-fallback  Allow GPU backends to fall back to CPU for unsupported ops
        \\      --mmap                Use lazy mmap instead of eagerly paging weights into RAM
        \\      --prefill-batch-size <N>  Prefill chunk size in tokens [default: 512]
        \\
        \\KV CACHE:
        \\      --kv-type <TYPE>      KV cache quantization [default: f16]
        \\                            Types: f32, f16, q8_0/q8, int8/i8, fp8/fp8_e4m3, nvfp4/fp4, nvfp4_ds_mla,
        \\                                   turbo2-4/tq2-4, planar2-4/pq2-4, iso2-4/iq2-4, rotor2-4/rq2-4
        \\                            Preset: turbo (K=q8_0, V=turbo4)
        \\      --kv-type-k <TYPE>    KV key quantization (overrides --kv-type, alias: --cache-type-k)
        \\      --kv-type-v <TYPE>    KV value quantization (overrides --kv-type, alias: --cache-type-v)
        \\      --kv-tiers <TIERS>    Tiered KV cache: vram+ram, vram+ram+ssd [default: off]
        \\      --kv-ram-budget <GB>  RAM tier budget, integer GB (requires --kv-tiers) [default: 50% of free RAM]
        \\      --kv-ssd-path <PATH>  SSD tier file path (requires --kv-tiers with ssd)
        \\      --kv-ssd-budget <GB>  SSD tier budget, integer GB (requires --kv-tiers with ssd) [default: 10]
        \\      --kv-eviction <POL>   KV eviction policy: none, norm, tri [default: none]
        \\      --kv-budget <N>       Max KV positions to keep during eviction [default: 80% of ctx-size]
        \\
        \\SERVER:
        \\  -s, --serve            Start HTTP server (OpenAI + Anthropic API)
        \\  -p, --port <PORT>      Server port [default: 49453] (falls back to AGAVE_PORT)
        \\      --host <ADDR>      Bind address: IPv4, localhost, 0.0.0.0, or 0 [default: 127.0.0.1]
        \\                         Non-loopback binds require --api-key (or AGAVE_API_KEY)
        \\                         Falls back to AGAVE_HOST when --host is omitted
        \\      --api-key <KEY>    API key for server auth (prefer AGAVE_API_KEY; CLI arg is visible in ps)
        \\                         When both are set, AGAVE_API_KEY wins
        \\      --sleep-after <N>  Enter sleep mode after N seconds idle (0 = disabled)
        \\      --max-batch-size <N>  Max concurrent batched requests [default: 8]
        \\      --rate-limit-rpm <N>  Max requests/min (0 = unlimited; enables rate limiting)
        \\      --rate-limit-tpm <N>  Max prompt tokens/min (0 = unlimited; enables rate limiting)
        \\      --no-kv-cache      Prefill-only / embedding server (no decode KV)
        \\
        \\PARALLELISM:
        \\      --tp <N>              Tensor parallelism degree [default: 1; 1 or 2 (2-rank pair)]
        \\      --pp <N>              Pipeline parallelism stages [default: 1; 1 or 2]
        \\      --peers <ADDR>        Peer address (e.g. 192.168.0.2 or localhost for same-node)
        \\      --rank <N>            This node's rank for TP/PP/disagg [default: 0]
        \\      --transport <TYPE>    IPC transport: auto, tcp, shm, nccl [default: auto]
        \\      --disagg              Disaggregated prefill/decode (rank 0 prefills, rank 1 decodes)
        \\
        \\SPECULATIVE DECODING:
        \\      --draft-model <PATH>  Draft model GGUF for speculative decoding
        \\      --spec-mode <MODE>    Speculative mode: auto, standard, ddtree, self, ngram, suffix, lookahead, mtp, medusa, eagle, eagle3, mlp, pflash, dspark
        \\  -K, --spec-tokens <N>     Draft tokens per speculation round [default: 5]
        \\      --tree-budget <N>     DDTree node budget [default: 64]
        \\      --draft-layers <N>    Layers for self-speculative draft [default: auto]
        \\      --spec-token-map <F>  FR-Spec token frequency map for vocab truncation
        \\      --pflash-alpha <F>    PFlash block selection threshold (0.0-2.0) [default: 0.85]
        \\      --pflash-block-size <N>  PFlash scoring block size in tokens [default: 64]
        \\      --pflash-scorer <PATH>  Separate model for PFlash block scoring (defaults to --draft-model)
        \\
        \\ADAPTERS & DIFFUSION:
        \\      --lora <PATH>         Merge LoRA adapter GGUF at load time into base weights
        \\      --diffusion-steps <N> DiffusionGemma denoising steps [default: 16]
        \\      --diffusion-canvas <N> DiffusionGemma canvas length [default: 256]
        \\      --diffusion-confidence <F>  Diffusion acceptance confidence (0.0-1.0) [default: 0.5]
        \\      --dir-steering-file <PATH>  Directional steering f32 vector (n_layers × n_embd floats)
        \\      --dir-steering-ffn <F>      Steering scale for FFN outputs [default: 1.0 with file]
        \\      --dir-steering-attn <F>     Steering scale for attention outputs [default: 0]
        \\
        \\OPTIMIZATION:
        \\      --megakernel          Enable fused FFN megakernels (3→1 dispatch per layer)
        \\      --power <N>                  Target GPU utilisation percent (1-100)
        \\
        \\EXPERT STREAMING:
        \\      --ssd-streaming              Stream MoE experts from SSD
        \\      --ssd-cache-slots <N>        LRU expert cache size [default: 256]
        \\      --expert-profile-out <FILE>  Save expert activation profile
        \\      --expert-profile-in <FILE>   Load expert activation profile for cache warming
        \\
        \\MULTIMODAL:
        \\      --mmproj <PATH>    Path to vision projector GGUF (mmproj file)
        \\      --image <PATH>     Path to image file (PNG or PPM P6)
        \\      --video <PATH>     Path to video file (frames extracted via ffmpeg)
        \\      --video-fps <N>    Video frame sampling rate (default: 1 fps)
        \\
        \\DIAGNOSTICS:
        \\  -V, --verbose          Show technical details (params, load times, EOG)
        \\  -d, --debug            Enable debug logging (token IDs, layer timing); implies --verbose
        \\      --json             Output results as JSON (implies --quiet)
        \\      --model-info       Print model metadata and exit (supports --json)
        \\      --profile          Profile per-op timing (halves throughput)
        \\      --benchmark        Run decode benchmark: prefill + decode, print stats (supports --json)
        \\      --frontier-bench             Frontier benchmark (snapshot KV at each context)
        \\      --frontier-ctx <LIST>        Comma-separated context lengths for frontier bench
        \\
        \\ENVIRONMENT:
        \\  NO_COLOR             Disable colored output when set (https://no-color.org)
        \\  AGAVE_API_KEY        API key for server auth (preferred over --api-key; wins if both set)
        \\  AGAVE_HOST           Server bind address when --host is omitted [default: 127.0.0.1]
        \\  AGAVE_PORT           Server port when --port is omitted [default: 49453]
        \\  AGAVE_VISION_DEBUG   Dump vision encoder intermediate buffers when set to 1
        \\  HF_TOKEN             HuggingFace API token for private repos (used by pull)
        \\  HF_HOME              Custom HuggingFace cache directory (used by pull)
        \\  XDG_CACHE_HOME       XDG cache base for pull (fallback: ~/.cache)
        \\
        \\EXAMPLES:
        \\  agave model.gguf                          Interactive REPL
        \\  agave model.gguf "What is 2+2?"           Single prompt
        \\  agave model.gguf -q "Hello" > out.txt     Pipe output (no banner)
        \\  agave model.gguf --serve --port 3000      HTTP server on port 3000
        \\  agave model.gguf --serve --host 0          HTTP server on all interfaces
        \\  agave model.gguf -t 0.7 --top-p 0.9 "Tell me a joke"
        \\  agave model.gguf --backend cpu "Hello"    Force CPU backend
        \\  agave ./glm-4-9b/ "Hello"                 Load SafeTensors directory
        \\  echo "Explain TCP" | agave model.gguf     Pipe prompt from stdin
        \\  agave model.gguf --json "Hello"           JSON output with stats
        \\  agave model.gguf --json --model-info      Model metadata as JSON
        \\  agave model.gguf --kv-type tq4 "Hello"   TurboQuant KV cache (saves VRAM)
        \\  agave model.gguf --ctx-size 0 "Hello"    Use full model context window
        \\  agave model.gguf --ctx-size auto "Hello"  Auto-fit context to available memory
        \\  agave model.gguf --image pic.png "What's this?"  Vision (auto-detects mmproj)
        \\  agave model.gguf --json-output "Generate a user profile"  Force JSON output
        \\  agave model.gguf --grammar-string 'root ::= "yes" | "no"' "Is sky blue?"
        \\  agave model.gguf --json-schema '{"type":"object","properties":{"name":{"type":"string"}}}' "User info"
        \\  agave target.gguf --draft-model draft.gguf "Hello"    Speculative decoding (DDTree)
        \\  agave model.gguf --spec-mode self --draft-layers 9 "Hello"  Self-speculative
        \\  agave model.gguf --megakernel "Hello"                 Fused FFN megakernel
        \\  agave model.gguf --benchmark --json                   Benchmark with JSON output
        \\
        \\SUBCOMMANDS:
        \\  agave pull <org/repo>                    Download model from HuggingFace
        \\  agave pull <org/repo> --quant Q4_K_M     Download specific quantization
        \\  agave pull <org/repo> --list             List available model files
        \\  agave calibrate <model.gguf|model-dir/>   Generate TriAttention calibration data
        \\  agave help <topic>                       Show help for a subcommand (e.g. pull, calibrate)
        \\
        \\SUPPORTED ARCHITECTURES:
        \\  gemma3, gemma4, diffusion-gemma, qwen35, gpt-oss, nemotron-h, nemotron-nano, glm4, deepseek4, llama4
        \\
        \\REPL COMMANDS:
    ++ repl_help;
    _ = std.posix.system.write(stdout_file.handle, usage.ptr, usage.len);
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
    var lora_handle: lora_mod.Handle = .{ .allocator = allocator };
    defer {
        lora_handle.dispose();
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
        if (cli.lora_path != null) eprint("warning: --lora is only supported for GGUF models; ignored for SafeTensors\n", .{});
    } else {
        gguf_file = GGUFFile.open(allocator, cli.model_path) catch |e| {
            eprint("Error: failed to open '{s}': {}\n", .{ cli.model_path, e });
            if (e == error.FileNotFound) {
                eprint("  File does not exist. Check the path and try again.\n", .{});
                if (std.mem.indexOfScalar(u8, cli.model_path, '/') == null and
                    std.mem.indexOfScalar(u8, cli.model_path, '.') == null)
                {
                    const subs = [_][]const u8{ "pull", "calibrate", "help" };
                    for (subs) |sub| {
                        if (std.mem.eql(u8, cli.model_path, sub) or closeMatch(cli.model_path, sub)) {
                            eprint("  Did you mean 'agave {s}'?\n", .{sub});
                            break;
                        }
                    }
                }
            } else if (e == error.InvalidMagic)
                eprint("  Not a valid GGUF file. Expected GGUF magic bytes.\n", .{})
            else if (e == error.UnsupportedVersion)
                eprint("  GGUF version not supported. Agave supports v2 and v3.\n", .{})
            else if (e == error.FileTooSmall)
                eprint("  File is too small to be a valid GGUF model.\n", .{});
            std.process.exit(1);
        };
        // Apply LoRA adapter (load-time merge into base weights as F32).
        if (cli.lora_path) |lp| {
            lora_handle = lora_mod.applyLoraGguf(allocator, &gguf_file.?, lp) catch |e| {
                eprint("Error: failed to apply LoRA '{s}': {}\n", .{ lp, e });
                std.process.exit(1);
            };
            eprint("lora: merged adapter '{s}' ({d} tensors overridden)\n", .{ lp, gguf_file.?.lora_overrides.count() });
        }
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
        eprint("  Supported: gemma3, gemma4, diffusion-gemma, qwen35, gpt-oss, nemotron-h, nemotron-nano, glm4, deepseek4, llama4\n", .{});
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
    bs.init(allocator, cli.backend_choice, g_io, cli.device_id);
    defer if (bs.pool) |*p| p.deinit();
    const be = bs.be;
    const be_name = bs.name;

    // Register mmap'd weight regions for UMA zero-copy GPU access
    if (gguf_file) |g| {
        if (g.mapped_data.len > 0) be.registerHostRegion(g.mapped_data.ptr, g.mapped_data.len);
    }
    if (st_dir) |s| {
        for (s.shard_data) |shard| {
            if (shard.data.len > 0) be.registerHostRegion(shard.data.ptr, shard.data.len);
        }
    }

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
        .n_experts = fmt.getArchU32(arch_str, "expert_count") orelse fmt.getMetaU32("num_local_experts") orelse fmt.getMetaU32("n_routed_experts") orelse 0,
        .n_experts_used = fmt.getArchU32(arch_str, "expert_used_count") orelse fmt.getMetaU32("num_experts_per_tok") orelse 0,
        .file_size_bytes = file_size_bytes,
        .load_ms = load_ms,
        .warmup_ms = 0, // updated after preload
        .system_mem = backend_mod.detectSystemMem(),
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
    const ctx_auto_sentinel = std.math.maxInt(u32);
    if (cli.ctx_size == ctx_auto_sentinel) {
        // --ctx-size auto: probe available memory and fit largest safe context
        const avail_mem = backend_mod.detectSystemMem();
        const n_kv = disp_info.n_kv_heads;
        const hd = disp_info.head_dim;
        const nl = disp_info.n_layers;
        if (n_kv > 0 and hd > 0 and nl > 0 and avail_mem > 0) {
            // Use float arithmetic for per-token KV bytes — bitsPerElement() returns
            // fractional values (e.g. Q8_0 = 8.5). @intFromFloat on a non-integer
            // is UB in ReleaseFast, so compute in float and round up.
            const kv_bpe = cli.kv_type_k.bitsPerElement() + cli.kv_type_v.bitsPerElement();
            const per_token_bytes: usize = @intFromFloat(@ceil(@as(f64, @floatFromInt(@as(usize, nl) * @as(usize, n_kv) * @as(usize, hd))) * @as(f64, kv_bpe) / 8.0));
            const model_bytes = disp_info.file_size_bytes;
            const usable = if (avail_mem > model_bytes * 2) avail_mem - model_bytes * 2 else avail_mem / 4;
            const fit_ctx = if (per_token_bytes > 0) usable * 8 / (per_token_bytes * 10) else default_ctx_size;
            const max_ctx = if (model_native_ctx > 0) @as(usize, model_native_ctx) else 131072;
            cli.ctx_size = @intCast(@max(128, @min(fit_ctx, max_ctx)));
            std.log.info("ctx-size: auto → {d} ({d} MB available, {d} B/token KV)", .{
                cli.ctx_size,
                avail_mem / (1024 * 1024),
                per_token_bytes,
            });
        } else {
            cli.ctx_size = default_ctx_size;
            std.log.info("ctx-size: auto → {d} (insufficient metadata for auto-fit)", .{cli.ctx_size});
        }
    } else if (cli.ctx_size == 0 and !cli.no_kv_cache) {
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
    // Skip preload when SSD streaming is enabled — the whole point is demand-paged
    // access. Preloading 90-155GB through a 48GB page cache just thrashes.
    const warmup_ms: u64 = if (!cli.use_mmap and !cli.model_info and !cli.ssd_streaming)
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
    disp_info.mtp_depth = fmt.getArchU32(arch_str, "nextn_predict_layers") orelse
        if (fmt.layerTensor(disp_info.n_layers, "nextn.eh_proj") != null) @as(u32, 1) else @as(u32, 0);
    disp_info.has_vision = arch.imageTokens() != null;

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
    const tok_kind: TokenizerKind = if (arch == .gemma3 or arch == .gemma4 or arch == .diffusion_gemma) .spm_no_dummy else if (merges != null) .bpe else .spm;
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

        // Add EOG tokens defined by the chat template (use n_layers for gemma4 variant selection)
        const tmpl = arch.chatTemplateForLayers(disp_info.n_layers);
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

/// Load a PNG or PPM P6 image and resize to target_size x target_size.
/// Returns [target_size * target_size * 3]u8 RGB pixels (row-major, channel-last).
///
/// Format is auto-detected from magic bytes:
///   - PNG (0x89 P N G): full decode via image.decodePng
///   - PPM P6 ("P6"): raw RGB parse
///   - JPEG (0xFF 0xD8): rejected with a convert-to-PNG hint (`error.InvalidImageFormat`)
///
/// Resize uses bilinear interpolation via image.resize.
fn loadImage(allocator: std.mem.Allocator, path: []const u8, target_size: u32) ![]u8 {
    const file = try Io.Dir.cwd().openFile(g_io, path, .{});
    defer file.close(g_io);

    const file_stat = try file.stat(g_io);
    if (file_stat.size > image.max_file_size) return error.FileTooBig;
    const file_data = try allocator.alloc(u8, @intCast(file_stat.size));
    defer allocator.free(file_data);
    _ = try file.readPositionalAll(g_io, file_data, 0);

    const format = image.detectFormat(file_data);
    switch (format) {
        .png => {
            var png = try image.decodePng(allocator, file_data);
            defer png.deinit();
            return image.resize(allocator, png.pixels, png.width, png.height, target_size, target_size);
        },
        .ppm => {
            const ppm = try image.decodePpm(file_data);
            return image.resize(allocator, ppm.pixels, ppm.width, ppm.height, target_size, target_size);
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
        // Warn: split-attention tiered SDPA is only fully implemented for Gemma 3.
        // Other architectures store KV in tiered blocks but compute SDPA against the
        // first block only, giving wrong results for long sequences.
        if (arch != .gemma3) {
            eprint("Warning: --kv-tiers is only fully implemented for Gemma 3. Other models may produce incorrect output.\n", .{});
        }
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
            fmt.getMetaU32("head_dim") orelse if (n_heads > 0) (n_embd / n_heads) else {
            eprint("Error: n_heads=0 and no head_dim in model metadata\n", .{});
            return false;
        };
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
    const eviction_budget: u32 = if (cli.kv_eviction != .none)
        (if (cli.kv_budget > 0) cli.kv_budget else @as(u32, @intCast(cli.ctx_size * 4 / 5)))
    else
        0;
    var mdl = ModelStorage.initFromArch(arch, allocator, fmt, be, cli.ctx_size, cli.kv_type_k, cli.kv_type_v, cli.kv_boundary_v, eviction_budget, tiered_ptr, cli.tp_rank, cli.tp_degree) catch |e| {
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

    // Directional steering: load direction vectors and attach to model
    var dir_steering: ?DirectionalSteering = null;
    if (cli.dir_steering_file) |steer_path| {
        const n_layers = mdl.model().nLayers();
        const n_embd = mdl.model().nEmbd();
        dir_steering = DirectionalSteering.init(
            allocator,
            g_io,
            steer_path,
            n_layers,
            n_embd,
            cli.dir_steering_ffn,
            cli.dir_steering_attn,
        ) catch |err| blk: {
            eprint("Error: failed to load steering file '{s}': {}\n", .{ steer_path, err });
            break :blk null;
        };
        if (dir_steering) |*steer| {
            mdl.setSteering(steer);
            eprint("steering: {s} (ffn={d:.1}, attn={d:.1})\n", .{ steer_path, cli.dir_steering_ffn, cli.dir_steering_attn });
        }
    }
    defer if (dir_steering) |*steer| steer.deinit(allocator);

    // Peer discovery buffers — must outlive both TP and PP blocks because
    // cli.tp_peers may borrow into them and be read by the PP setup path.
    var tp_ip_buf: [16]u8 = undefined;
    var pp_ip_buf: [16]u8 = undefined;

    // TP: allocate row-shard scratch buffer for weight column extraction
    if (cli.tp_degree > 1) {
        const n_embd = mdl.model().nEmbd();
        const a_str = fmt.getMetaStr("general.architecture") orelse "unknown";
        const n_ff = fmt.getArchU32(a_str, "feed_forward_length") orelse 0;
        if (n_embd > 0 and n_ff > 0) {
            const local_ff = n_ff / cli.tp_degree;
            // Use f32 row bytes (largest possible) to cover any quant format
            const row_bytes = @max(
                backend_mod.gemvRowBytes(.f32, local_ff),
                @max(backend_mod.gemvRowBytes(.q8_0, local_ff), backend_mod.gemvRowBytes(.q4_k, local_ff)),
            );
            const shard_size = n_embd * row_bytes;
            if (shard_size > 0) {
                const shard_buf = allocator.alloc(u8, shard_size) catch |err| blk: {
                    std.log.err("TP shard buffer alloc failed ({d} bytes): {}", .{ shard_size, err });
                    break :blk null;
                };
                if (shard_buf) |buf| {
                    mdl.setTpRowShardBuf(buf);
                }
            }
        }
        std.log.info("TP={d} rank={d} active", .{ cli.tp_degree, cli.tp_rank });

        // Auto-discover peers via UDP broadcast if --peers not specified.
        if (cli.tp_peers == null and cli.tp_degree > 1) {
            const peer_discovery = @import("parallel/peer_discovery.zig");
            if (peer_discovery.discoverPeer(cli.tp_rank, cli.tp_degree, tp_discovery_port)) |ip| {
                const ip_str = std.fmt.bufPrint(&tp_ip_buf, "{d}.{d}.{d}.{d}", .{ ip[0], ip[1], ip[2], ip[3] }) catch "";
                if (ip_str.len > 0) cli.tp_peers = ip_str;
            }
        }

        // Distributed TP: connect to peers
        if (cli.pp_degree <= 1) if (cli.tp_peers) |peers_str| {
            if (setupTransport(allocator, peers_str, cli.tp_rank, cli.tp_degree, cli.transport, tp_discovery_port, be)) |tr| {
                mdl.setTpTransport(tr);
            }
        };
    }

    // PP: pipeline parallelism setup (uses --pp, --rank, --peers)
    if (cli.pp_degree > 1) {
        std.log.info("PP={d} rank={d}", .{ cli.pp_degree, cli.tp_rank });
        // Auto-discover peers for PP if --peers not specified.
        if (cli.tp_peers == null) {
            const peer_discovery = @import("parallel/peer_discovery.zig");
            if (peer_discovery.discoverPeer(cli.tp_rank, cli.pp_degree, pp_discovery_port)) |ip| {
                const ip_str = std.fmt.bufPrint(&pp_ip_buf, "{d}.{d}.{d}.{d}", .{ ip[0], ip[1], ip[2], ip[3] }) catch "";
                if (ip_str.len > 0) cli.tp_peers = ip_str;
            }
        }
        if (cli.tp_peers) |peers_str| {
            if (setupTransport(allocator, peers_str, cli.tp_rank, cli.pp_degree, cli.transport, pp_discovery_port, be)) |t| {
                mdl.setPpConfig(cli.tp_rank, cli.pp_degree, t);
            }
        }
    }

    // TriAttention: load calibration data when --kv-eviction tri
    if (cli.kv_eviction == .tri) {
        // cal_buf must outlive the blk: block because cal_path borrows into it.
        var cal_buf: [1024]u8 = undefined;
        const cal_path = blk: {
            // Auto-detect .cal file next to model: model.gguf → model.cal
            if (std.mem.endsWith(u8, cli.model_path, ".gguf")) {
                const stem = cli.model_path[0 .. cli.model_path.len - 5];
                const cal = std.fmt.bufPrint(&cal_buf, "{s}.cal", .{stem}) catch break :blk @as(?[]const u8, null);
                break :blk cal;
            }
            break :blk @as(?[]const u8, null);
        };
        if (cal_path) |cp| {
            const calibrate = @import("calibrate.zig");
            if (calibrate.readCalFile(allocator, g_io, cp)) |cals| {
                mdl.setTriCalibration(cals);
                if (!g_quiet) eprint("tri-attention: loaded {d} calibrations from {s}\n", .{ cals.len, cp });
            } else |_| {
                eprint("Warning: --kv-eviction tri but no .cal file found ({s})\n", .{cp});
                eprint("  Generate with: agave calibrate {s}\n", .{cli.model_path});
            }
        }
    }

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
            eprint("Error: --megakernel not supported for {s} on this backend\n", .{@tagName(arch)});
            eprint("  Supported: qwen35/gemma4/gemma3/glm4 on Metal, qwen35 on CUDA.\n", .{});
            eprint("  See docs/MEGAKERNEL.md for details.\n", .{});
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

    // ── SSD expert streaming (ds4-style demand-paged MoE expert cache) ──
    const ExpertCache = @import("expert_cache.zig").ExpertCache;
    const ExpertProfile = @import("expert_profile.zig").ExpertProfile;
    var expert_cache_opt: ?ExpertCache = null;
    var expert_profile_opt: ?ExpertProfile = null;
    defer if (expert_cache_opt) |*ec| ec.deinit(allocator);
    defer if (expert_profile_opt) |*ep| ep.deinit(allocator);

    const n_exp = minfo.n_experts;
    const n_lay = minfo.n_layers;

    if (cli.ssd_streaming and n_exp > 0) {
        const cache_slots: u32 = if (cli.ssd_cache_slots != 256)
            cli.ssd_cache_slots // user override
        else blk: {
            // Auto-size based on how many unique experts fit in the page cache.
            // The ExpertCache tracks metadata (~20 bytes/slot), not expert data —
            // expert weights live in the mmap'd GGUF. We estimate how many unique
            // (layer, expert) pairs the OS can keep warm in physical RAM.
            const expert_bytes = estimateExpertBytes(fmt, n_exp);
            if (expert_bytes > 0) {
                // Use total physical RAM (not free — mmap pages are reclaimable)
                const total_ram = @import("backend/backend.zig").detectSystemMem();
                // Budget: total RAM minus ~8GB for OS/KV/scratch, rest for page cache
                const overhead: usize = 8 * 1024 * 1024 * 1024;
                const page_cache = if (total_ram > overhead) total_ram - overhead else total_ram / 2;
                const auto_slots = page_cache / expert_bytes;
                const capped = @min(auto_slots, ExpertCache.max_cache_slots);
                const slots: u32 = @intCast(@max(capped, 256));
                eprint("ssd-streaming: auto-sized cache to {d} slots ({d} MB page cache est, {d} KB/expert)\n", .{
                    slots, page_cache / (1024 * 1024), expert_bytes / 1024,
                });
                break :blk slots;
            }
            break :blk @as(u32, 1024); // fallback: generous default
        };
        expert_cache_opt = ExpertCache.init(allocator, n_lay, n_exp, cache_slots) catch |e| blk: {
            eprint("Warning: failed to init expert cache ({s}), SSD streaming disabled\n", .{@errorName(e)});
            break :blk null;
        };
        if (expert_cache_opt != null) {
            eprint("ssd-streaming: expert cache {d} slots, {d} experts × {d} layers\n", .{ cache_slots, n_exp, n_lay });
        }
        // Enable volatile weights for Metal safety.
        // Heapification handles expert + non-expert weights,
        // but Metal SDPA/wo_a still has issues — needs investigation.
        be.setVolatileWeights(true); // OFF: causes buf_cache invalidation race // disabled: buf_cache flush causes 0.002% drift
    }

    // Pre-pin hot experts from a prior profile run before first token.
    if (cli.expert_profile_in) |prof_path| {
        if (expert_cache_opt) |*ec| {
            var prof = ExpertProfile.loadJson(allocator, prof_path) catch |e| blk: {
                eprint("Warning: failed to load expert profile '{s}': {s}\n", .{ prof_path, @errorName(e) });
                break :blk null;
            };
            if (prof) |*p| {
                defer p.deinit(allocator);
                // Pre-pin top-8 experts per layer into the LRU cache.
                var top_ids: [8]u32 = undefined;
                for (0..@min(n_lay, p.n_layers)) |li| {
                    const k = ec.admit_prepin(@intCast(li), &top_ids, p.topExperts(@intCast(li), 8, &top_ids));
                    _ = k;
                }
                eprint("ssd-streaming: pre-pinned hot experts from '{s}'\n", .{prof_path});

                // mlock the hot experts' weight ranges so the OS cannot evict them.
                if (cli.ssd_streaming) {
                    var pin_count: u32 = 0;
                    var pin_buf: [128]u8 = undefined;
                    for (0..@min(n_lay, p.n_layers)) |li| {
                        const k_pinned = p.topExperts(@intCast(li), 6, &top_ids);
                        for (0..k_pinned) |j| {
                            const eid = top_ids[j];
                            for ([_][]const u8{ "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight" }) |suffix| {
                                const name = std.fmt.bufPrint(&pin_buf, "blk.{d}.{s}", .{ li, suffix }) catch continue;
                                if (fmt.getTensor(name)) |t| {
                                    const stride = t.dataByteLen() / @as(usize, n_exp);
                                    if (ec.pinExpert(t.data_ptr + eid * stride, stride))
                                        pin_count += 1;
                                }
                            }
                        }
                    }
                    if (pin_count > 0) {
                        eprint("ssd-streaming: mlocked {d} expert weight ranges ({d} MB)\n", .{
                            pin_count, ec.total_pinned_bytes / (1024 * 1024),
                        });
                    }
                }
            }
        } else {
            eprint("Warning: --expert-profile-in requires --ssd-streaming, ignored\n", .{});
        }
    }

    // Auto-pin: when --ssd-streaming is active but no --expert-profile-in,
    // mlock the non-routed weights that are accessed every token (shared
    // expert weights and router gate weights). Disabled by default on
    // memory-constrained systems (≤64GB) — benchmarks show mlock reduces
    // available page cache space and hurts overall throughput when model >> RAM.
    const total_mem = @import("backend/backend.zig").detectSystemMem();
    if (cli.expert_profile_in == null and cli.ssd_streaming and expert_cache_opt != null and total_mem > 64 * 1024 * 1024 * 1024) {
        var ec = &expert_cache_opt.?;
        var auto_pin: u32 = 0;
        for (0..n_lay) |li| {
            // Pin shared expert weights (always activated, every layer)
            for ([_][]const u8{ "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight" }) |suffix| {
                var buf: [128]u8 = undefined;
                const name = std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ li, suffix }) catch continue;
                if (fmt.getTensor(name)) |t| {
                    if (ec.pinExpert(t.data_ptr, t.dataByteLen())) auto_pin += 1;
                }
            }
            // Pin router gate (small, always read for expert routing)
            {
                var buf: [128]u8 = undefined;
                const name = std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate_inp.weight", .{li}) catch continue;
                if (fmt.getTensor(name)) |t| {
                    if (ec.pinExpert(t.data_ptr, t.dataByteLen())) auto_pin += 1;
                }
            }
        }
        if (auto_pin > 0) {
            eprint("ssd-streaming: auto-pinned {d} non-routed weight ranges ({d} MB)\n", .{
                auto_pin, ec.total_pinned_bytes / (1024 * 1024),
            });
        }
    }

    // Start profiling if requested (even without SSD streaming).
    if (cli.expert_profile_out != null and n_exp > 0) {
        expert_profile_opt = ExpertProfile.init(allocator, n_lay, n_exp) catch |e| blk: {
            eprint("Warning: failed to init expert profiler ({s})\n", .{@errorName(e)});
            break :blk null;
        };
    }

    // Wire expert cache + profiler into the model so ffnLayer() can use them.
    if (expert_cache_opt) |*ec| {
        const prof_ptr = if (expert_profile_opt) |*ep| ep else null;
        mdl.setExpertCache(ec, prof_ptr);
    }

    // ── MTP weight loading ──────────────────────────────────────
    const MtpWeights = @import("models/ds4_mtp.zig").MtpWeights;
    var mtp_weights: ?MtpWeights = null;
    defer if (mtp_weights) |*mw| mw.deinit(allocator);

    if (cli.mtp_model_path) |mtp_path| {
        var mtp = MtpWeights.init(allocator);
        mtp.load(allocator, mtp_path) catch |e| {
            eprint("Error: failed to load MTP weights from '{s}': {s}\n", .{ mtp_path, @errorName(e) });
        };
        if (mtp.n_depths > 0) {
            mtp_weights = mtp;
            mdl.setMtpWeights(&mtp_weights.?);
            eprint("MTP: {d} draft depths available\n", .{mtp.n_depths});
        }
    }

    var model_if = mdl.model();

    if (spec_caps.unsatisfied(cli.spec_mode, .{
        .draft = cli.draft_model_path != null,
        .mtp = model_if.getMtpDepth() > 0,
    })) |p| {
        eprint("Error: --spec-mode {s} waiting for {s}\n", .{ @tagName(cli.spec_mode), spec_caps.providerName(p) });
        eprint("  {s}\n", .{spec_caps.howToProvide(p)});
        return false;
    }

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
    if (mmproj_path == null and (cli.image != null or cli.serve or cli.video != null)) {
        const model_dir: []const u8 = blk: {
            // Check if model_path is a directory (SafeTensors)
            const probe_dir = Io.Dir.cwd().openDir(g_io, cli.model_path, .{}) catch break :blk Io.Dir.path.dirname(cli.model_path) orelse ".";
            probe_dir.close(g_io);
            break :blk cli.model_path;
        };
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
            if (ve.patch_size == 0 or ve.projection_dim == 0) {
                eprint("Error: vision encoder has invalid patch_size or projection_dim\n", .{});
                return false;
            }
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
    } else if (cli.image != null or cli.serve or cli.video != null) {
        // Qwen3.8 ships the ViT in the same checkpoint (model.visual.* / v.blk.*).
        if (fmt.getTensor("v.blk.0.attn_qkv.weight") != null) {
            vision_enc = VisionEncoder.init(allocator, fmt, be, pool) catch |err| {
                eprint("Error: failed to init in-checkpoint vision encoder: {}\n", .{err});
                return false;
            };
            {
                const ve = &vision_enc.?;
                if (ve.patch_size == 0 or ve.projection_dim == 0) {
                    eprint("Error: vision encoder has invalid patch_size or projection_dim\n", .{});
                    return false;
                }
            }
            if (!g_quiet) {
                const ve = &vision_enc.?;
                eprint("vision: in-checkpoint {d} layers, {d}x{d} patches -> {d}D\n", .{
                    ve.n_blocks,
                    ve.image_size / ve.patch_size,
                    ve.image_size / ve.patch_size,
                    ve.projection_dim,
                });
            }
        }
    }

    // Resolve image pad token ID for multimodal prompt injection.
    const img_tokens = arch.imageTokens();
    var n_visual_tokens: u32 = 0;

    if (vision_enc) |*ve| {
        if (cli.image) |image_path| {
            // Qwen VL: use original image dims rounded up to patch_size * 2.
            // Matches llama.cpp which processes at native resolution.
            const target_size: u32 = if (ve.use_native_resolution) blk: {
                const grid: u32 = ve.patch_size * 2;
                const orig = image.getImageDimensions(allocator, g_io, image_path) catch break :blk ve.image_size;
                const side = @max(orig.width, orig.height);
                const rounded = ((side + grid - 1) / grid) * grid;
                if (rounded > ve.image_size) {
                    std.log.warn("Qwen VL: native resolution {d}×{d} exceeds allocated buffer size {d}, capping to {d}", .{ rounded, rounded, ve.image_size, ve.image_size });
                    break :blk ve.image_size;
                }
                break :blk rounded;
            } else ve.image_size;
            const img_pixels = loadImage(allocator, image_path, target_size) catch |err| {
                eprint("Error: failed to load image '{s}': {}\n", .{ image_path, err });
                return false;
            };
            defer allocator.free(img_pixels);

            // Update n_patches for the actual processing resolution (Qwen VL native resolution)
            if (ve.use_native_resolution) {
                ve.n_patches = (target_size / ve.patch_size) * (target_size / ve.patch_size);
                ve.n_output_patches = ve.n_patches / 4; // Qwen 4× merge
                ve.image_size = target_size;
            }

            const visual_tokens = ve.encode(img_pixels) catch |err| {
                eprint("Error: vision encode failed: {}\n", .{err});
                return false;
            };
            if (ve.projection_dim == 0) {
                eprint("Error: vision encoder projection_dim is 0\n", .{});
                return false;
            }
            n_visual_tokens = @intCast(visual_tokens.len / ve.projection_dim);
            const pad_id: u32 = if (img_tokens) |it| it.pad else 0;
            model_if.setImageEmbeddings(visual_tokens, n_visual_tokens, pad_id);
            if (!g_quiet) eprint("vision: encoded {d} visual tokens\n", .{n_visual_tokens});
        }
    } else if (cli.image != null) {
        eprint("Warning: --image ignored (no vision projector found — use --mmproj <path> to specify)\n", .{});
    }

    // ── Video input: extract frames via ffmpeg and encode with vision encoder ──
    if (cli.video) |video_path| {
        if (vision_enc == null) {
            eprint("Warning: --video ignored (no vision projector found — use --mmproj <path> to specify)\n", .{});
        } else {
            const ve = &vision_enc.?;
            // Create temp directory for extracted frames under $TMPDIR (default /tmp)
            const tmp_base = pull.getenv("TMPDIR") orelse default_tmp_base;
            var tmp_buf: [tmp_path_buf_size]u8 = undefined;
            const tmp_dir_slice = std.fmt.bufPrint(&tmp_buf, "{s}/agave_video_{d}", .{ tmp_base, milliTimestamp(g_io) }) catch video_tmp_fallback;
            Io.Dir.cwd().createDir(g_io, tmp_dir_slice, .default_dir) catch {};
            defer Io.Dir.cwd().deleteTree(g_io, tmp_dir_slice) catch {};

            // Extract frames using ffmpeg: one PNG per second (or custom fps)
            var fps_buf: [32]u8 = undefined;
            const fps_str = std.fmt.bufPrint(&fps_buf, "fps={d:.2}", .{cli.video_fps}) catch "fps=1";
            var fp_buf: [400]u8 = undefined;
            const frame_pattern = std.fmt.bufPrint(&fp_buf, "{s}/frame_%04d.png", .{tmp_dir_slice}) catch "";
            {
                const ffmpeg_argv = [_][]const u8{ "ffmpeg", "-i", video_path, "-vf", fps_str, frame_pattern, "-y", "-loglevel", "quiet" };
                if (std.process.spawn(g_io, .{ .argv = &ffmpeg_argv })) |ffmpeg_proc_val| {
                    var ffmpeg_proc = ffmpeg_proc_val;
                    _ = ffmpeg_proc.wait(g_io) catch null;
                } else |_| {
                    eprint("Warning: ffmpeg not found (is ffmpeg installed?)\n", .{});
                }
            }

            // Collect extracted frames (sorted by name = temporal order)
            // Open temp dir and iterate frame PNGs via POSIX readdir
            var all_visual_tokens: std.ArrayList(f32) = .empty;
            defer all_visual_tokens.deinit(allocator);
            var frame_count: u32 = 0;

            {
                // Iterate extracted PNGs using Io.Dir
                var scan_dir = Io.Dir.cwd().openDir(g_io, tmp_dir_slice, .{ .iterate = true }) catch {
                    eprint("Error: could not open video temp directory: {s}\n", .{tmp_dir_slice});
                    return false;
                };
                var scan_buf: [Io.Dir.Reader.min_buffer_len]u8 align(@alignOf(usize)) = undefined;
                var reader = Io.Dir.Reader.init(scan_dir, &scan_buf);
                var frame_names: std.ArrayList([]u8) = .empty;
                defer {
                    for (frame_names.items) |n| allocator.free(n);
                    frame_names.deinit(allocator);
                }
                var entries: [8]Io.Dir.Entry = undefined;
                while (reader.read(g_io, &entries) catch null) |n| {
                    for (entries[0..n]) |entry| {
                        if (!std.mem.endsWith(u8, entry.name, ".png")) continue;
                        const nc = allocator.dupe(u8, entry.name) catch continue;
                        frame_names.append(allocator, nc) catch {
                            allocator.free(nc);
                        };
                    }
                    if (n == 0) break;
                }
                if (scan_dir.handle != Io.Dir.cwd().handle) scan_dir.close(g_io);
                std.mem.sort([]u8, frame_names.items, {}, struct {
                    fn lt(_: void, a: []u8, b: []u8) bool {
                        return std.mem.lessThan(u8, a, b);
                    }
                }.lt);
                for (frame_names.items) |name| {
                    var frame_path_buf2: [512]u8 = undefined;
                    const frame_path = std.fmt.bufPrint(&frame_path_buf2, "{s}/{s}", .{ tmp_dir_slice, name }) catch continue;
                    const img_pixels = loadImage(allocator, frame_path, ve.image_size) catch continue;
                    defer allocator.free(img_pixels);
                    const tokens = ve.encode(img_pixels) catch continue;
                    all_visual_tokens.appendSlice(allocator, tokens) catch {
                        eprint("Error: out of memory collecting video frame embeddings\n", .{});
                        return false;
                    };
                    frame_count += 1;
                }
            }
            if (frame_count > 0 and ve.projection_dim > 0) {
                n_visual_tokens = @intCast(all_visual_tokens.items.len / ve.projection_dim);
                const owned = all_visual_tokens.toOwnedSlice(allocator) catch &.{};
                const pad_id: u32 = if (img_tokens) |it| it.pad else 0;
                model_if.setImageEmbeddings(owned, n_visual_tokens, pad_id);
                if (!g_quiet) eprint("video: encoded {d} frames → {d} visual tokens\n", .{ frame_count, n_visual_tokens });
            } else {
                eprint("Warning: no frames extracted from '{s}' (is ffmpeg installed? is --mmproj set?)\n", .{video_path});
            }
        }
    }

    // ── Draft model loading (speculative decoding) ──────────────
    var draft_gguf: ?GGUFFile = null;
    var draft_st: ?SafeTensorsDir = null;
    var draft_mdl_storage: ?ModelStorage = null;
    // ── PFlash scorer model (separate from draft; optional) ──────
    var scorer_gguf: ?GGUFFile = null;
    var scorer_st: ?SafeTensorsDir = null;
    var scorer_mdl_storage: ?ModelStorage = null;
    var scorer_model_if: Model = undefined;
    var scorer_ptr: ?*Model = null;
    defer {
        if (draft_mdl_storage) |*dm| dm.deinit();
        if (draft_gguf) |*g| g.deinit();
        if (draft_st) |*s| s.deinit();
        if (scorer_mdl_storage) |*sm| sm.deinit();
        if (scorer_gguf) |*g| g.deinit();
        if (scorer_st) |*s| s.deinit();
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
        // HF DFlash2 checkpoints declare model_type "qwen3"; fingerprint them
        // by their selector codebooks so they route to the dflash2 drafter.
        if (draft_arch == .qwen35 and draft_fmt.getTensor("candidate_selector.predecessor_codebook") != null) {
            draft_arch = .dflash2;
        }
        if (draft_arch == .nemotron_h and draft_fmt.getTensor("backbone.embeddings.weight") != null)
            draft_arch = .nemotron_nano;
        if (!draft_arch.isEnabled()) {
            eprint("Error: draft model arch {s} disabled at compile time\n", .{draft_arch.displayName()});
            return false;
        }
        draft_mdl_storage = ModelStorage.initFromArch(draft_arch, allocator, draft_fmt, be, cli.ctx_size, .f16, .f16, 0, 0, null, 0, 1) catch |e| {
            eprint("Error: failed to init draft model: {}\n", .{e});
            return false;
        };
        draft_mdl_storage.?.setPool(pool);
        draft_mdl_storage.?.fixBlockAllocator();
        draft_model_if = draft_mdl_storage.?.model();
        draft_ptr = &draft_model_if;
        eprint("draft: {s} · {s}\n", .{ draft_arch.displayName(), Format.getQuantName(draft_fmt) });

        // ── DFlash2 drafter wiring ───────────────────────────────
        // The checkpoint ships no embeddings or LM head; bind the target's.
        // Target-side feature capture feeds the drafter's context injection,
        // and --spec-mode defaults to dflash2 when this drafter is loaded.
        if (draft_mdl_storage.? == .dflash2) {
            if (cli.tp_degree > 1 or cli.pp_degree > 1) {
                eprint("Error: DFlash2 speculative decoding requires tp=1 and pp=1\n", .{});
                return false;
            }
            const emb_t = fmt.getTensor("token_embd.weight") orelse {
                eprint("Error: target model lacks token_embd.weight for DFlash2 binding\n", .{});
                return false;
            };
            const head_t = fmt.getTensor("output.weight") orelse fmt.getTensor("token_embd.weight") orelse {
                eprint("Error: target model lacks output.weight for DFlash2 binding\n", .{});
                return false;
            };
            draft_mdl_storage.?.dflash2.bindTarget(emb_t, head_t, fmt);
            mdl.setCaptureLayers(draft_mdl_storage.?.dflash2.target_layer_ids, draft_mdl_storage.?.dflash2.cap) catch |e| {
                eprint("Error: failed to enable DFlash2 feature capture on target: {}\n", .{e});
                return false;
            };
        }
    } else if (cli.spec_mode != .none) {
        draft_ptr = &model_if;
    }

    // ── PFlash scorer model (separate model for block importance scoring) ──
    if (cli.pflash_scorer_path) |scorer_path| {
        const scorer_is_dir = blk: {
            const d = Io.Dir.cwd().openDir(g_io, scorer_path, .{}) catch break :blk false;
            d.close(g_io);
            break :blk true;
        };
        var scorer_fmt: Format = undefined;
        if (scorer_is_dir) {
            scorer_st = SafeTensorsDir.open(allocator, scorer_path) catch |e| {
                eprint("Error: failed to open pflash scorer '{s}': {}\n", .{ scorer_path, e });
                return false;
            };
            scorer_fmt = scorer_st.?.format();
        } else {
            scorer_gguf = GGUFFile.open(allocator, scorer_path) catch |e| {
                eprint("Error: failed to open pflash scorer '{s}': {}\n", .{ scorer_path, e });
                return false;
            };
            scorer_fmt = scorer_gguf.?.format();
        }
        const scorer_arch_str = scorer_fmt.getMetaStr("general.architecture") orelse
            scorer_fmt.getMetaStr("model_type") orelse "unknown";
        const scorer_arch = Arch.detect(scorer_arch_str) orelse {
            eprint("Error: unsupported pflash scorer architecture '{s}'\n", .{scorer_arch_str});
            return false;
        };
        if (!scorer_arch.isEnabled()) {
            eprint("Error: pflash scorer arch {s} disabled at compile time\n", .{scorer_arch.displayName()});
            return false;
        }
        scorer_mdl_storage = ModelStorage.initFromArch(scorer_arch, allocator, scorer_fmt, be, cli.ctx_size, .f16, .f16, 0, 0, null, 0, 1) catch |e| {
            eprint("Error: failed to init pflash scorer: {}\n", .{e});
            return false;
        };
        scorer_mdl_storage.?.setPool(pool);
        scorer_mdl_storage.?.fixBlockAllocator();
        scorer_model_if = scorer_mdl_storage.?.model();
        scorer_ptr = &scorer_model_if;
        eprint("pflash scorer: {s} · {s}\n", .{ scorer_arch.displayName(), Format.getQuantName(scorer_fmt) });
    }

    if (cli.frontier_bench) {
        runFrontierBench(&model_if, &tok, allocator, cli, eog);
        return true;
    }

    if (cli.benchmark) {
        runBenchmark(&model_if, &tok, allocator, cli, eog);
        return true;
    }

    if (cli.serve) {
        if (arch == .diffusion_gemma) {
            eprint("Warning: DiffusionGemma in server mode uses autoregressive generation only\n", .{});
            eprint("         (block diffusion via --serve not yet implemented)\n", .{});
        }
        // DFlash2 requires per-request capture/ingest plumbing the server
        // scheduler does not provide yet; serve autoregressively instead.
        if (cli.spec_mode == .dflash2) {
            eprint("Warning: DFlash2 speculative decoding is CLI-only right now; server starts without speculation\n", .{});
        }
        // Initialize shared n-gram pool for cross-request history sharing.
        // Server slots use this as a fallback when their own history has no match.
        if (cli.spec_mode == .ngram) {
            ngram_mod.global_pool = ngram_mod.SharedNgramPool{};
        }
        var tok_if = tok.tokenizer();
        const ve_ptr: ?*VisionEncoder = if (vision_enc != null) &vision_enc.? else null;
        const srv_pad_id: u32 = if (img_tokens) |it| it.pad else 0;
        const srv_start_id: u32 = if (img_tokens) |it| it.start else 0;
        const srv_end_id: u32 = if (img_tokens) |it| it.end else 0;
        server.run(.{
            .allocator = allocator,
            .model = &model_if,
            .tokenizer = &tok_if,
            .chat_template = arch.chatTemplateForLayers(minfo.n_layers),
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
            .draft_model = if (cli.spec_mode == .dflash2) null else draft_ptr,
            .spec_tokens = cli.spec_tokens,
            .tree_budget = cli.tree_budget,
            .sleep_after_s = cli.sleep_after_s,
            .max_batch_size = cli.max_batch_size,
            .rate_limit_rpm = cli.rate_limit_rpm,
            .rate_limit_tpm = cli.rate_limit_tpm,
        }) catch |e| {
            eprint("Error: server failed: {}\n", .{e});
            return false;
        };
    } else if (cli.disagg and cli.tp_peers != null) {
        // Disaggregated inference: rank 0 prefills + sends KV, rank 1 receives KV + decodes
        if (allocator.create(TransportMod.Transport) catch |err| blk: {
            eprint("Error: disaggregated inference transport allocation failed: {}\n", .{err});
            break :blk null;
        }) |dtr| disagg_blk: {
            defer {
                dtr.deinit();
                allocator.destroy(dtr);
            }
            const peers_str = cli.tp_peers orelse break :disagg_blk;
            dtr.* = TransportMod.Transport.init(allocator, .tcp, cli.tp_rank, 2) catch |err| {
                eprint("Error: disaggregated transport init failed: {}\n", .{err});
                break :disagg_blk;
            };
            const disagg_peer = parsePeerAddr(peers_str, disagg_default_port) orelse {
                eprint("Error: invalid peer address '{s}'\n", .{peers_str});
                break :disagg_blk;
            };
            const host = disagg_peer.host;
            const port = disagg_peer.port;
            if (cli.tp_rank == 0) {
                // Prefill node: tokenize, prefill, send KV
                var la: std.posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = 0 };
                const ls = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
                if (ls < 0) break :disagg_blk;
                defer _ = std.c.close(ls);
                var one: c_int = 1;
                _ = std.c.setsockopt(ls, std.posix.SOL.SOCKET, std.posix.SO.REUSEADDR, @ptrCast(&one), @sizeOf(c_int));
                if (std.c.bind(ls, @ptrCast(&la), @sizeOf(@TypeOf(la))) != 0) break :disagg_blk;
                if (std.c.listen(ls, 1) != 0) break :disagg_blk;
                std.log.info("Disagg prefill: waiting for decode node on port {d}...", .{port});
                dtr.acceptPeer(ls) catch break :disagg_blk;
                std.log.info("Decode node connected. Prefilling...", .{});

                if (effective_prompt) |prompt| {
                    const tmpl = arch.chatTemplateForLayers(minfo.n_layers);
                    const formatted = tmpl.format(allocator, null, prompt) catch prompt;
                    defer if (formatted.ptr != prompt.ptr) allocator.free(@constCast(formatted));
                    const tok_iface = tok.tokenizer();
                    const token_ids = tok_iface.encode(formatted) catch break :disagg_blk;
                    defer allocator.free(token_ids);

                    var first_tok: u32 = 0;
                    first_tok = mdl.model().forward(token_ids[0]) catch break :disagg_blk;
                    for (token_ids[1..]) |tid| first_tok = mdl.model().forward(tid) catch break :disagg_blk;
                    std.log.info("Prefill done ({d} tokens, first_gen={d}). Sending KV cache...", .{ token_ids.len, first_tok });
                    mdl.sendKvCache(dtr);
                    // Send first generated token
                    var first_f32 = [1]f32{@floatFromInt(first_tok)};
                    dtr.sendBuf(&first_f32, 1);
                    std.log.info("KV cache sent. Prefill node done.", .{});
                }
            } else {
                // Decode node: receive KV, generate tokens
                std.log.info("Disagg decode: connecting to prefill node...", .{});
                dtr.connectPeer(host, port) catch break :disagg_blk;
                std.log.info("Connected. Waiting for KV cache...", .{});
                mdl.recvKvCache(dtr) catch break :disagg_blk;
                const kv_len = mdl.model().kvSeqLen();
                // Receive first generated token from prefill node
                var first_tok_f32: [1]f32 = undefined;
                dtr.recvBuf(&first_tok_f32, 1);
                const raw_tok = first_tok_f32[0];
                var next: u32 = if (raw_tok >= 0 and raw_tok < @as(f32, @floatFromInt(std.math.maxInt(u32))))
                    @intFromFloat(raw_tok)
                else
                    0;
                std.log.info("KV cache received ({d} positions, first_gen={d}). Generating...", .{ kv_len, next });
                var gen_count: u32 = 0;
                const max_gen: u32 = @intCast(cli.max_tokens);
                while (gen_count < max_gen) {
                    var is_eog = false;
                    for (eog.ids[0..eog.len]) |e_id| {
                        if (next == e_id) {
                            is_eog = true;
                            break;
                        }
                    }
                    if (is_eog) break;
                    const tok_slice = [1]u32{next};
                    const text = tok.tokenizer().decode(@constCast(&tok_slice)) catch break;
                    defer allocator.free(text);
                    _ = std.posix.system.write(stdout_file.handle, text.ptr, text.len);
                    gen_count += 1;
                    next = mdl.model().forward(next) catch break;
                }
                _ = std.posix.system.write(stdout_file.handle, "\n", 1);
            }
        }
    } else if (effective_prompt) |prompt| {
        if (arch == .diffusion_gemma) {
            generateDiffusion(allocator, &model_if, tok, cli, tok_kind, arch, prompt, !g_quiet);
        } else {
            // DFlash2 drafter + feature reader, resolved once where storage lives.
            const df2_ptr: ?*DFlash2Model = if (draft_mdl_storage != null and draft_mdl_storage.? == .dflash2) &draft_mdl_storage.?.dflash2 else null;
            const feat_reader: ?ModelStorage.FeatureReader = if (df2_ptr != null) mdl.featureReader() else null;
            generateAndPrint(allocator, &model_if, df2_ptr, feat_reader, tok, cli, tok_kind, eog, arch, prompt, !g_quiet, minfo, display, img_tokens, n_visual_tokens, draft_ptr, scorer_ptr);
        }
    } else {
        runRepl(allocator, &model_if, tok, cli, tok_kind, eog, arch, minfo, display, img_tokens, n_visual_tokens);
    }
    mdl.reportPerf();

    // Save expert activation profile if requested.
    if (cli.expert_profile_out) |out_path| {
        if (expert_profile_opt) |*prof| {
            prof.writeJson(allocator, out_path) catch |e| {
                eprint("Warning: failed to write expert profile to '{s}': {s}\n", .{ out_path, @errorName(e) });
            };
            eprint("expert profile written to '{s}' ({d} tokens profiled)\n", .{ out_path, prof.total_tokens });
        }
    }

    // Report expert cache stats.
    if (expert_cache_opt) |*ec| ec.reportStats();

    return true;
}

// ── Interactive REPL ─────────────────────────────────────────────

/// Runs the interactive read-eval-print loop, reading user prompts from the terminal and generating responses.
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
) void {
    var n_visual_tokens: u32 = n_visual_tokens_init;
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

    const template = arch.chatTemplateForLayers(minfo.n_layers);

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
                _ = std.posix.system.write(stdout_file.handle, repl_help.ptr, repl_help.len);
                continue;
            } else {
                print("Unknown command: {s} (try /help)\n", .{trimmed});
                continue;
            }
        }

        // Add user message to history
        const user_content = allocator.dupe(u8, trimmed) catch {
            eprint("Error: out of memory\n", .{});
            continue;
        };
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

// ── Diffusion generation (DiffusionGemma) ────────────────────────

/// Block diffusion generation loop for DiffusionGemma.
/// Encodes the prompt autoregressively, then iteratively denoises a 256-token
/// canvas using bidirectional attention, accepting high-confidence tokens.
fn generateDiffusion(
    allocator: std.mem.Allocator,
    model: *Model,
    tok: *BpeTokenizer,
    cli: *const CliArgs,
    tok_kind: TokenizerKind,
    arch: Arch,
    prompt: []const u8,
    show_stats: bool,
) void {
    const template = arch.chatTemplateForLayers(model.nLayers());
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

    if (!g_quiet) eprint("diffusion: prompt = {d} tokens\n", .{token_ids.len});

    const start_ms = milliTimestamp(g_io);
    const last_tok = model.prefill(token_ids) catch |e| {
        eprint("Error: prefill failed: {}\n", .{e});
        return;
    };
    const prefill_ms = milliTimestamp(g_io) - start_ms;
    _ = last_tok;

    const max_steps = cli.diffusion_steps;
    const canvas_len = cli.diffusion_canvas;
    const confidence_threshold = cli.diffusion_confidence;

    // Use DiffusionGemmaModel directly for forwardCanvas.
    const DiffusionModel = @import("models/diffusion_gemma.zig").DiffusionGemmaModel;

    var total_generated: u32 = 0;
    var block_count: u32 = 0;
    const max_blocks = (cli.max_tokens + canvas_len - 1) / canvas_len;

    // Track timing.
    const gen_start_ms = milliTimestamp(g_io);

    while (block_count < max_blocks) : (block_count += 1) {
        // Initialize canvas with random tokens (uniform state diffusion).
        var canvas = allocator.alloc(u32, canvas_len) catch {
            eprint("Error: canvas allocation failed\n", .{});
            return;
        };
        defer allocator.free(canvas);
        var canvas_logits = allocator.alloc(f32, canvas_len * model.vocabSize()) catch {
            eprint("Error: canvas logits allocation failed\n", .{});
            return;
        };
        defer allocator.free(canvas_logits);

        // Start with random tokens (uniform state diffusion uses random noise).
        // Mix CLI seed with block index so --seed fully determines the canvas.
        var rng = std.Random.DefaultPrng.init(cli.seed +% @as(u64, @intCast(block_count)) +% 1);
        const vocab_sz = model.vocabSize();
        for (canvas) |*t| t.* = rng.random().intRangeLessThan(u32, 4, @min(vocab_sz - 1, 32000));

        var locked = allocator.alloc(bool, canvas_len) catch {
            eprint("Error: allocation failed\n", .{});
            return;
        };
        defer allocator.free(locked);
        @memset(locked, false);

        var n_locked: u32 = 0;

        // Retrieve DiffusionGemmaModel pointer for forwardCanvas.
        // The model vtable wraps DiffusionGemmaModel; we access it via downcasting.
        // Since we know the arch, the storage is DiffusionGemmaModel.
        // We can't call forwardCanvas through the vtable (it's not there), so we
        // look up the concrete model through the ModelStorage union.
        // NOTE: We pass the model ptr and rely on the vtable's ptr field being
        // the DiffusionGemmaModel directly (Model.from stores m as ptr).
        const concrete: *DiffusionModel = @ptrCast(@alignCast(model.ptr));

        // Denoising loop.
        for (0..max_steps) |step| {
            // Forward pass over canvas with bidirectional attention.
            concrete.forwardCanvas(canvas, canvas_logits) catch |e| {
                eprint("Error: forwardCanvas failed: {}\n", .{e});
                return;
            };

            // Accept tokens above confidence threshold; re-noise the rest.
            var newly_locked: u32 = 0;
            for (0..canvas_len) |i| {
                if (locked[i]) continue;
                const logits = canvas_logits[i * vocab_sz ..][0..vocab_sz];
                const best_tok = math_ops.argmax(logits);
                const best_score = blk: {
                    // Softmax probability of best token.
                    var mx: f32 = -std.math.inf(f32);
                    for (logits) |v| if (v > mx) {
                        mx = v;
                    };
                    var s: f32 = 0;
                    for (logits) |v| s += @exp(v - mx);
                    break :blk @exp(logits[best_tok] - mx) / s;
                };
                if (best_score >= confidence_threshold) {
                    canvas[i] = best_tok;
                    locked[i] = true;
                    newly_locked += 1;
                } else {
                    // Re-noise with a fresh random token.
                    canvas[i] = rng.random().intRangeLessThan(u32, 4, @min(vocab_sz - 1, 32000));
                }
            }
            n_locked += newly_locked;
            if (!g_quiet) eprint("diffusion: step {d}/{d}, {d}/{d} tokens locked\n", .{ step + 1, max_steps, n_locked, canvas_len });

            // Stop when all canvas tokens are locked.
            if (n_locked >= canvas_len) break;
        }

        // Truncate canvas at first EOS token before output.
        const eos_id = model.eosId();
        var eos_pos: usize = canvas_len;
        for (canvas, 0..) |t, i| {
            if (t == eos_id) {
                eos_pos = i;
                break;
            }
        }
        const output_canvas = canvas[0..eos_pos];

        // Output the (possibly truncated) canvas tokens.
        const canvas_text = tok.decodeSpm(output_canvas) catch tok.decode(output_canvas) catch null;
        if (canvas_text) |text| {
            _ = std.posix.system.write(stdout_file.handle, text.ptr, text.len);
            allocator.free(text);
        }
        total_generated += @intCast(output_canvas.len);

        // Stop if EOS was found in this block.
        if (eos_pos < canvas_len) break;

        // Prefill the canvas into the KV cache for the next block.
        _ = model.prefill(canvas) catch break;

        // Stop if max_tokens reached.
        if (total_generated >= cli.max_tokens) break;
    }
    _ = std.posix.system.write(stdout_file.handle, "\n", 1);

    if (show_stats) {
        const gen_ms = milliTimestamp(g_io) - gen_start_ms;
        const tok_per_s = if (gen_ms > 0) @as(f64, @floatFromInt(total_generated)) * 1000.0 / @as(f64, @floatFromInt(gen_ms)) else 0;
        eprint("\nprefill: {d}ms, generated: {d} tokens in {d}ms ({d:.1} tok/s)\n", .{ prefill_ms, total_generated, gen_ms, tok_per_s });
    }
}

// ── Shared generation logic ──────────────────────────────────────

/// Ingest newly captured target features into a DFlash2 drafter's context
/// cache. Positions are processed strictly in order; when the target's capture
/// ring has advanced past `ingested` (long contexts evict old entries), the
/// cursor resyncs to the ring start, matching the drafter's sliding window.
fn dflash2Ingest(
    reader: ModelStorage.FeatureReader,
    drafter: *DFlash2Model,
    ingested: *usize,
    stage: []f32,
) void {
    const begin = reader.begin();
    const end = reader.end();
    if (end <= ingested.*) return;
    var start = ingested.*;
    if (start < begin) start = begin;
    if (start >= end) {
        ingested.* = end;
        return;
    }
    const concat_dim: usize = drafter.target_layer_ids.len * drafter.n_embd;
    if (concat_dim == 0 or stage.len < concat_dim) return;
    const chunk_positions = stage.len / concat_dim;
    var pos = start;
    while (pos < end) {
        const n = @min(end - pos, chunk_positions);
        for (0..n) |i| reader.readAt(pos + i, stage[i * concat_dim ..][0..concat_dim]);
        drafter.ingestContext(stage[0 .. n * concat_dim], pos, n) catch |err| {
            std.log.warn("dflash2: context ingest failed at {d}: {s}", .{ pos, @errorName(err) });
            return;
        };
        pos += n;
    }
    ingested.* = pos;
}

/// Top-level generation entry point: delegates to speculative decoding if a draft model is available, otherwise standard autoregressive generation.
fn generateAndPrint(
    allocator: std.mem.Allocator,
    mdl: *Model,
    df2: ?*DFlash2Model,
    feat_reader: ?ModelStorage.FeatureReader,
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
    pflash_scorer: ?*Model,
) void {
    if (draft_model) |dm| {
        generateSpeculative(allocator, mdl, dm, pflash_scorer, df2, feat_reader, tok, cli, tok_kind, eog, arch, prompt, show_stats);
    } else {
        const response = generateAndPrintInner(allocator, mdl, tok, cli, tok_kind, eog, arch.chatTemplateForLayers(minfo.n_layers), prompt, true, false, show_stats, minfo, display, false, img_tokens, n_visual_tokens);
        if (response) |r| allocator.free(r);
    }
}

/// Runs speculative decoding: tokenizes the prompt, prefills the target and draft models, then decodes with draft-verify speculation.
/// `df2` carries a DFlash2 drafter (with its target feature reader) when the
/// loaded --draft-model is one; both are null for every other mode.
fn generateSpeculative(
    allocator: std.mem.Allocator,
    target: *Model,
    draft_model: *Model,
    pflash_scorer: ?*Model,
    df2: ?*DFlash2Model,
    feat_reader: ?ModelStorage.FeatureReader,
    tok: *BpeTokenizer,
    cli: *const CliArgs,
    tok_kind: TokenizerKind,
    eog: anytype,
    arch: Arch,
    prompt: []const u8,
    show_stats: bool,
) void {
    const emit = emitGeneratedTokens(cli);
    const template = arch.chatTemplateForLayers(target.nLayers());
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
    var first_target: u32 = 0;

    if (cli.spec_mode == .pflash and target.ptr != draft_model.ptr) {
        // PFlash speculative prefill: draft scores blocks, target prefills compressed prompt
        const pflash = @import("spec/pflash.zig");
        const pflash_cfg = pflash.PFlashConfig{
            .alpha = cli.pflash_alpha,
            .block_size = cli.pflash_block_size,
            .max_kept_ratio = 0.20,
            .score_tail = 16,
        };
        var pflash_state = pflash.PFlashState.init(allocator, pflash_cfg, prefill_toks.len) catch |e| {
            eprint("Error: PFlash state init failed: {}\n", .{e});
            return;
        };
        defer pflash_state.deinit(allocator);
        // Use dedicated scorer if provided, else fall back to draft model
        const scoring_model = pflash_scorer orelse draft_model;
        first_target = pflash.pflashPrefill(pflash_cfg, &pflash_state, scoring_model, target, prefill_toks, allocator) catch |e| {
            eprint("Error: PFlash prefill failed: {}\n", .{e});
            return;
        };
        const ratio = pflash.compressionRatio(&pflash_state);
        eprint("pflash: {d} → {d} tokens ({d:.1}% kept)\n", .{
            prefill_toks.len, pflash_state.selected_len, ratio * 100,
        });
    } else {
        first_target = target.prefill(prefill_toks) catch |e| {
            eprint("Error: target prefill failed: {}\n", .{e});
            return;
        };
        // Only prefill draft model separately when it's a different model.
        // DFlash2 drafters take context via feature injection instead.
        if (target.ptr != draft_model.ptr and df2 == null) {
            _ = draft_model.prefill(prefill_toks) catch |e| {
                eprint("Error: draft prefill failed: {}\n", .{e});
                return;
            };
        }
    }
    const prefill_ms = milliTimestamp(g_io) - prefill_start;

    // DFlash2: ingest the prompt-tail features captured during target prefill
    // into the drafter's context cache, and keep ingestion state for the loop.
    var ctx_ingested: usize = 0;
    var df2_stage: []f32 = &.{};
    defer if (df2_stage.len > 0) allocator.free(df2_stage);
    if (df2 != null and feat_reader != null) {
        const concat_dim: usize = df2.?.target_layer_ids.len * df2.?.n_embd;
        const ingest_chunk: usize = 32;
        df2_stage = allocator.alloc(f32, ingest_chunk * concat_dim) catch |e| {
            eprint("Error: DFlash2 ingest staging failed: {}\n", .{e});
            return;
        };
        dflash2Ingest(feat_reader.?, df2.?, &ctx_ingested, df2_stage);
    }

    // Sampling setup. Distributed pairs stay greedy so draft tokens match.
    const use_sampling = cli.temperature > 0 and !distributedLockstep(cli);
    var prng = std.Random.Xoshiro256.init(cli.seed);
    if (use_sampling) {
        first_target = math_ops.sampleToken(target.getLogits(), cli.temperature, cli.top_k, cli.top_p, prng.random());
    }

    // Speculative generation loop
    var spec_state = spec_decode.SpecState.init(allocator, cli.spec_tokens, target.vocabSize()) catch {
        eprint("Error: failed to allocate speculative state\n", .{});
        return;
    };
    spec_state.adaptive_k_enabled = true;
    defer spec_state.deinit(allocator);

    // FR-Spec: load token frequency map if provided
    var fr_spec_mask: ?[]bool = null;
    defer if (fr_spec_mask) |m| allocator.free(m);
    if (cli.spec_token_map) |map_path| {
        fr_spec_mask = spec_decode.buildTokenMask(allocator, map_path, target.vocabSize()) catch |err| blk: {
            std.log.warn("spec-token-map: failed to load ({s}), FR-Spec disabled", .{@errorName(err)});
            break :blk null;
        };
        spec_state.token_mask = fr_spec_mask;
    }

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

    // pflash only changes prefill; for decode, treat like ddtree when a separate draft model exists
    const effective_spec_mode: SpecMode = if (cli.spec_mode == .pflash and target.ptr != draft_model.ptr)
        .ddtree
    else if (df2 != null and (cli.spec_mode == .ddtree or cli.spec_mode == .standard))
        .dflash2 // auto-select: a DFlash2 drafter overrides the default tree path
    else
        cli.spec_mode;
    const use_ddtree = (effective_spec_mode == .ddtree);
    const self_spec = (effective_spec_mode == .self_spec);
    const use_ngram = (effective_spec_mode == .ngram);
    const use_suffix = (effective_spec_mode == .suffix);
    // Medusa CLI alias is normalized to .mtp at parse time.
    const use_mtp = (effective_spec_mode == .mtp);
    const use_eagle = (effective_spec_mode == .eagle);
    const use_eagle3 = (effective_spec_mode == .eagle3);
    const use_mlp = (effective_spec_mode == .mlp);
    const use_lookahead = (effective_spec_mode == .lookahead);
    const use_dspark = (effective_spec_mode == .dspark);
    const use_dflash2 = (effective_spec_mode == .dflash2);
    var la_state = ngram_mod.LookaheadState{};
    if (use_lookahead) la_state.seed(token_ids);
    var ngram_state = ngram_mod.NgramState{};
    if (use_ngram or use_dflash2) {
        // Seed n-gram history with prefill tokens. In dflash2 mode the history
        // powers hybrid block extension and cooldown-time drafting.
        for (token_ids) |tid| ngram_state.push(tid);
        if (!isEogToken(first_target, eog)) ngram_state.push(first_target);
    }
    var suffix_state_opt: ?ngram_mod.SuffixState = null;
    if (use_suffix) {
        suffix_state_opt = ngram_mod.SuffixState.init(allocator) catch blk: {
            std.log.warn("suffix: alloc failed, falling back to ngram", .{});
            break :blk null;
        };
        if (suffix_state_opt) |*ss| {
            // Push non-special prompt tokens for suffix matching context.
            // Skip special tokens (template markers like <｜Assistant｜>, </think>)
            // which cause the suffix to echo chat formatting.
            // User text tokens provide context for faster suffix matching.
            const special_token_start: u32 = 128000;
            for (token_ids) |tid| {
                if (tid < special_token_start) ss.push(tid);
            }
            if (!isEogToken(first_target, eog) and first_target < special_token_start) ss.push(first_target);
        }
    }
    defer if (suffix_state_opt) |*ss| ss.deinit();

    // Self-speculative: auto-detect layer skip range (skip middle 50%)
    const self_spec_skip_divisor = 4;
    const self_spec_default_skip_fraction = 2;
    const skip_start: u32 = if (self_spec) target.nLayers() / self_spec_skip_divisor else 0;
    const skip_end: u32 = if (self_spec) blk: {
        const skip_count = cli.draft_layers orelse (target.nLayers() / self_spec_default_skip_fraction);
        break :blk skip_start + skip_count;
    } else 0;

    // Adaptive spec decode: skip drafting when acceptance rate drops below threshold
    // (e.g., during reasoning/thinking sections where predictions are unreliable)
    const adaptive_window: u32 = 8;
    const adaptive_threshold: f32 = 0.25;
    var recent_accepted: u32 = 0;
    var recent_drafted: u32 = 0;
    var draft_cooldown: u32 = 0;

    while (token_count < cli.max_tokens and !isEogToken(last, eog)) {
        const pre_draft_pos = target.kvSeqLen();
        // Set when the cooldown branch produced pure n-gram drafts; skips the
        // drafter so verification consumes those drafts directly.
        var skip_draft_phase = false;

        // Adaptive: skip drafting during cooldown (low acceptance period)
        if (draft_cooldown > 0) {
            draft_cooldown -= 1;
            // DFlash2 hybrid cooldown: keep speculating with pure n-gram drafts
            // instead of falling back to autoregressive decoding. Greedy only —
            // sampled rounds would need q distributions n-grams cannot supply.
            var cooldown_ng_drafted = false;
            if (use_dflash2 and !use_sampling) {
                const budget = @min(cli.spec_tokens, @as(u32, spec_decode.max_draft_tokens));
                var prop: [spec_decode.max_draft_tokens]u32 = undefined;
                var np: usize = ngram_state.propose(budget, &prop);
                if (np == 0) {
                    if (ngram_mod.global_pool) |*pl| {
                        const tail_start = if (ngram_state.len >= 10) ngram_state.len - 10 else 0;
                        np = pl.propose(ngram_state.history[tail_start..ngram_state.len], budget, &prop);
                    }
                }
                if (np > 0) {
                    @memcpy(spec_state.draft_tokens[0..np], prop[0..np]);
                    spec_state.n_draft = @intCast(np);
                    cooldown_ng_drafted = true;
                    skip_draft_phase = true;
                }
            }
            if (!cooldown_ng_drafted) {
                if (use_suffix) target.setExpertBudget(3);
                last = target.forward(last) catch break;
                if (use_suffix) target.setExpertBudget(0);
                if (use_sampling) {
                    const cl = target.getLogits();
                    if (cli.min_p > 0) math_ops.applyMinP(cl, cli.min_p);
                    if (cli.xtc_probability > 0) math_ops.applyXtc(cl, cli.xtc_probability, cli.xtc_threshold, prng.random());
                    last = math_ops.sampleToken(cl, cli.temperature, cli.top_k, cli.top_p, prng.random());
                }
                if (isEogToken(last, eog)) break;
                if (use_ngram or use_lookahead or use_dflash2) ngram_state.push(last);
                if (token_count < gen_ids_buf.len) {
                    gen_ids_buf[token_count] = last;
                    token_count += 1;
                }
                continue;
            }
        }

        // Draft phase
        if (self_spec) target.setLayerSkip(skip_start, skip_end);
        const is_self_draft = (target.ptr == draft_model.ptr and !self_spec and !use_ngram and !use_mtp and !use_eagle and !use_eagle3 and !use_mlp and !use_dspark and !use_dflash2);
        const effective_k = spec_state.optimalK();
        const n_drafted: u32 = if (skip_draft_phase) blk: {
            break :blk spec_state.n_draft;
        } else if (use_mtp) blk: {
            break :blk spec_decode.draftMtp(&spec_state, target, last);
        } else if (use_suffix) blk: {
            var n: usize = if (suffix_state_opt) |*ss|
                ss.propose(&spec_state.draft_tokens)
            else
                0;
            // Fallback to shared ngram pool if suffix cache has no match (server mode)
            if (n == 0) {
                if (ngram_mod.global_pool) |*pool| {
                    const tail_start = if (ngram_state.len >= 10) ngram_state.len - 10 else 0;
                    n = pool.propose(ngram_state.history[tail_start..ngram_state.len], effective_k, &spec_state.draft_tokens);
                }
            }
            spec_state.n_draft = @intCast(n);
            break :blk @as(u32, @intCast(n));
        } else if (use_ngram) blk: {
            var n = ngram_state.propose(effective_k, &spec_state.draft_tokens);
            // Try shared pool if local history found nothing (server mode)
            if (n == 0) {
                if (ngram_mod.global_pool) |*pool| {
                    const tail_start = if (ngram_state.len >= 10) ngram_state.len - 10 else 0;
                    n = pool.propose(ngram_state.history[tail_start..ngram_state.len], effective_k, &spec_state.draft_tokens);
                }
            }
            spec_state.n_draft = @intCast(n);
            break :blk @as(u32, @intCast(n));
        } else if (use_lookahead) blk: {
            // Lookahead: advance branches then find n-gram match with current context
            const tail_start = if (ngram_state.len >= 8) ngram_state.len - 8 else 0;
            const n = spec_decode.draftLookahead(&spec_state, target, last, &la_state, ngram_state.history[tail_start..ngram_state.len]);
            break :blk n;
        } else if (use_eagle) blk: {
            // EAGLE: condition draft on target's post-norm hidden state (EAGLE-1).
            const n = if (!use_sampling)
                spec_decode.draftEagle(&spec_state, target.*, draft_model.*, last)
            else
                spec_decode.draftEagleWithLogits(&spec_state, target.*, draft_model.*, last);
            break :blk n;
        } else if (use_eagle3) blk: {
            // EAGLE-3: condition draft on pre-output-norm hidden state for richer signal.
            const n = if (!use_sampling)
                spec_decode.draftEagle3(&spec_state, target.*, draft_model.*, last)
            else
                spec_decode.draftEagle3WithLogits(&spec_state, target.*, draft_model.*, last);
            break :blk n;
        } else if (use_mlp) blk: {
            // MLP Speculator: single-context draft (no chain). All K steps use target's hidden.
            const n = if (!use_sampling)
                spec_decode.draftMlpSpeculator(&spec_state, target.*, draft_model.*, last)
            else
                spec_decode.draftMlpSpeculatorWithLogits(&spec_state, target.*, draft_model.*, last);
            break :blk n;
        } else if (use_dspark) blk: {
            // DSpark: draft with confidence-scheduled verification trim.
            // Draft tokens using the existing draft model (any drafter), then apply
            // the hardware-aware prefix scheduler to drop low-survival-prob suffix tokens.
            const n = if (!use_sampling)
                spec_decode.draft(&spec_state, draft_model, last)
            else
                spec_decode.draftWithLogits(&spec_state, draft_model, last);
            // Apply confidence-based trim: use acceptance history as proxy for per-position
            // survival probability and trim draft to tokens with positive expected return.
            if (n > 0) spec_decode.dsparkTrimDraft(&spec_state);
            break :blk spec_state.n_draft;
        } else if (use_dflash2) blk: {
            // DFlash2 block-diffusion drafting: one parallel pass proposes the
            // whole block; the candidate selector picks a coherent chain.
            if (df2 == null) break :blk 0;
            const drafter = df2.?;
            const k_df2 = @min(spec_state.optimalK(), drafter.block_size -| 1);
            // Selector samples at the generation temperature; greedy walks at T=0.
            const sel_temp: f32 = if (use_sampling) cli.temperature else 0;
            var n = spec_decode.draftDFlash2(&spec_state, drafter, last, drafter.contextLen(), sel_temp, prng.random());
            // Hybrid n-gram composition (greedy only): confirm/extend the block
            // with exact-match history continuations.
            if (n > 0 and !use_sampling) {
                spec_decode.dflash2HybridNgram(&spec_state, &ngram_state, if (ngram_mod.global_pool) |*pl| pl else null, k_df2);
                n = spec_state.n_draft;
            }
            break :blk @as(u32, @intCast(n));
        } else if (is_self_draft and !use_sampling)
            spec_decode.draft(&spec_state, draft_model, last)
        else
            spec_decode.draftWithLogits(&spec_state, draft_model, last);
        if (n_drafted == 0) {
            // N-gram / Suffix / Lookahead / DFlash2: no draft — single-token decode
            if (use_ngram or use_suffix or use_lookahead or use_dflash2) {
                // Use reduced expert budget for fallback forward (50% less I/O)
                if (use_suffix) target.setExpertBudget(3);
                last = target.forward(last) catch break;
                if (use_suffix) target.setExpertBudget(0);
                if (use_sampling) {
                    const cl2 = target.getLogits();
                    if (cli.min_p > 0) math_ops.applyMinP(cl2, cli.min_p);
                    if (cli.xtc_probability > 0) math_ops.applyXtc(cl2, cli.xtc_probability, cli.xtc_threshold, prng.random());
                    last = math_ops.sampleToken(cl2, cli.temperature, cli.top_k, cli.top_p, prng.random());
                }
                if (isEogToken(last, eog)) break;
                if (use_ngram or use_lookahead) ngram_state.push(last);
                if (use_suffix) if (suffix_state_opt) |*ss| ss.push(last);
                if (token_count < gen_ids_buf.len) {
                    gen_ids_buf[token_count] = last;
                    token_count += 1;
                }
                continue;
            }
            break;
        }

        // Verify phase
        // SP-MoE-inspired: blast prefetch ALL layers' experts before verification.
        // Warms page cache so sequential verification forwards hit cached pages.
        if (n_drafted > 0 and use_suffix) {
            target.prefetchAllLayers();
            // MoE-Spec (arXiv 2602.16052): reduce expert count during verification.
            target.setExpertBudget(4);
            // forwardTree has no HC state, so skipping early layers is safe (unlike forward()).
            // Colibri-inspired: freeze expert cache during verification.
            // Prevents eviction of cached experts across sequential verify forwards.
            // target.freezeExpertCache(); // disabled: hurts hit rate
        }

        // Trust-mode threshold: after this many tokens, skip verification for suffix.
        // With greedy decoding (t=0.0), suffix matches from model's own history are
        // provably correct. Skipping verification eliminates ALL forwardTree cost.

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
        else if (use_suffix)
            // Note: verifyBatched uses forwardTree which has no HC/experts and gives 0% acceptance.
            // Suffix mode actually goes through is_self_draft (full forward) which gives 100% acceptance.
            // This path is dead code — kept for reference.
            spec_decode.verifyBatched(&spec_state, target, draft_model, last, pre_draft_pos)
        else
            spec_decode.verifySequential(&spec_state, target, draft_model, last, pre_draft_pos);

        // Reset expert budget and layer skip after verification.
        if (use_suffix) {
            target.setExpertBudget(0);
        }

        // Emit accepted draft tokens
        var hit_eog = false;
        for (0..result.accepted) |i| {
            const accepted_tok = spec_state.draft_tokens[i];
            if (token_count >= cli.max_tokens -| 1) break;
            if (token_count >= gen_ids_buf.len) break;
            if (isEogToken(accepted_tok, eog)) {
                hit_eog = true;
                break;
            }
            gen_ids_buf[token_count] = accepted_tok;
            token_count += 1;
        }

        // Emit correction/bonus token
        if (!hit_eog and token_count < cli.max_tokens -| 1 and token_count < gen_ids_buf.len) {
            if (isEogToken(result.next_token, eog)) {
                hit_eog = true;
            } else {
                gen_ids_buf[token_count] = result.next_token;
                token_count += 1;
            }
        }
        last = if (hit_eog) target.eosId() else result.next_token;

        // Adaptive spec decode: track acceptance rate, enter cooldown if low
        recent_accepted += result.accepted;
        recent_drafted += spec_state.n_draft;
        if (recent_drafted >= adaptive_window) {
            const rate = @as(f32, @floatFromInt(recent_accepted)) / @as(f32, @floatFromInt(recent_drafted));
            if (rate < adaptive_threshold) {
                draft_cooldown = adaptive_window;
            }
            // Log rolling acceptance rate every adaptive_window rounds when verbose
            if (g_verbose and spec_state.total_rounds % 10 == 0 and spec_state.total_drafted > 0) {
                std.log.debug("spec: {d}/{d} ({d:.1}%) accepted this window, {d:.1}% overall", .{
                    recent_accepted, recent_drafted,
                    rate * 100.0,    spec_state.acceptanceRate() * 100.0,
                });
            }
            recent_accepted = 0;
            recent_drafted = 0;
        }

        // MTP: sync KV cache position (reset to match target on partial rejection)
        if (use_mtp) {
            target.resetMtpCache();
        }

        // Update suffix cache with accepted tokens (also push to ngram_state for pool fallback context)
        if (use_suffix) {
            if (suffix_state_opt) |*ss| {
                for (0..result.accepted) |i| {
                    if (isEogToken(spec_state.draft_tokens[i], eog)) break;
                    ss.push(spec_state.draft_tokens[i]);
                    ngram_state.push(spec_state.draft_tokens[i]);
                    if (ngram_mod.global_pool) |*pool| pool.push(spec_state.draft_tokens[i]);
                }
                if (!hit_eog) {
                    ss.push(result.next_token);
                    ngram_state.push(result.next_token);
                    if (ngram_mod.global_pool) |*pool| pool.push(result.next_token);
                }
            }
        }

        // Update n-gram history with accepted tokens
        if (use_ngram) {
            for (0..result.accepted) |i| {
                if (isEogToken(spec_state.draft_tokens[i], eog)) break;
                ngram_state.push(spec_state.draft_tokens[i]);
                if (ngram_mod.global_pool) |*pool| pool.push(spec_state.draft_tokens[i]);
            }
            if (!hit_eog) {
                ngram_state.push(result.next_token);
                if (ngram_mod.global_pool) |*pool| pool.push(result.next_token);
            }
        }

        // DFlash2: maintain hybrid history and ingest the newly verified
        // positions' captured features into the drafter's context cache.
        if (use_dflash2 and feat_reader != null and df2 != null) {
            if (pull.getenv("AGAVE_DF2_DEBUG") != null) {
                std.debug.print("df2 round {d}: pos={d} drafted={d} accepted={d} next={d} drafts={any}\n", .{
                    spec_state.total_rounds, pre_draft_pos, spec_state.n_draft, result.accepted, result.next_token,
                    spec_state.draft_tokens[0..@min(spec_state.n_draft, 8)],
                });
            }
            for (0..result.accepted) |i| {
                if (isEogToken(spec_state.draft_tokens[i], eog)) break;
                ngram_state.push(spec_state.draft_tokens[i]);
            }
            if (!hit_eog) ngram_state.push(result.next_token);
            dflash2Ingest(feat_reader.?, df2.?, &ctx_ingested, df2_stage);
        }

        // Stream
        if (emit and token_count - batch_start >= batch_size) {
            flushTokenBatch(tok, tok_kind, allocator, gen_ids_buf[batch_start..@min(token_count, gen_ids_buf.len)], &started_output);
            batch_start = token_count;
        }
    }

    // Flush remaining
    if (emit and token_count > batch_start and token_count <= gen_ids_buf.len) {
        flushTokenBatch(tok, tok_kind, allocator, gen_ids_buf[batch_start..token_count], &started_output);
    }
    if (emit and !g_tty and started_output) {
        _ = std.posix.system.write(stdout_file.handle, "\n", 1);
    }

    const gen_ms = milliTimestamp(g_io) - gen_start;
    if (emit and show_stats and token_count > 0) {
        const tok_per_sec = if (gen_ms > 0) @as(f32, @floatFromInt(token_count)) / @as(f32, @floatFromInt(gen_ms)) * ms_per_second else 0;
        eprint("\n{d} tok · {d:.1} tok/s · {d}ms prefill · spec: {d:.0}% accept ({d:.1} mean)\n", .{
            token_count,
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
        } catch {
            eprint("Error: failed to encode user prefix for image token insertion\n", .{});
            return null;
        };
        defer allocator.free(prefix_tokens);

        const insert_pos: usize = chat_tmpl_mod.findImageInsertPos(text_token_ids, prefix_tokens);

        // Use injectImageTokens which handles architecture-specific wrapping:
        // Gemma 4 (start=end=pad): just pad×N
        // Qwen 3.5 (distinct start/end): [start, pad×N, end]
        // Fail closed: continuing without placeholders silently drops the image.
        const result = chat_tmpl_mod.injectImageTokens(
            allocator,
            text_token_ids,
            insert_pos,
            img_tokens.?,
            n_visual_tokens,
        ) catch {
            eprint("Error: failed to inject image tokens into prompt\n", .{});
            return null;
        };

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
    const use_sampling = cli.temperature > 0 and !distributedLockstep(cli);
    const use_repeat_penalty = cli.repeat_penalty != 1.0;
    var prng = std.Random.Xoshiro256.init(cli.seed);
    var cli_mirostat_mu: f32 = cli.mirostat_tau * 2.0;

    // Grammar-constrained decoding
    var grammar: ?grammar_mod.Grammar = null;
    var grammar_state: ?grammar_mod.GrammarState = null;
    var grammar_source_buf: ?[]u8 = null; // file buffer owned by Grammar (rules borrow into it)
    var json_depth: i32 = 0;
    const json_mode_active = cli.json_output;
    if (json_mode_active) {
        // JSON mode uses brace-depth tracking instead of GBNF grammar
        // Force first token to contain '{', stop when depth returns to 0
    } else if (cli.json_schema) |schema| {
        grammar = grammar_mod.Grammar.fromJsonSchema(allocator, schema) catch |err| blk: {
            eprint("Error: failed to parse JSON schema: {}\n", .{err});
            break :blk null;
        };
        if (grammar) |*g| grammar_state = g.initState() catch |err| blk: {
            eprint("Error: grammar state init failed (OOM): {}\n", .{err});
            break :blk null;
        };
    } else if (cli.grammar_string) |gs| {
        grammar = grammar_mod.Grammar.parse(allocator, gs) catch |err| blk: {
            eprint("Error: failed to parse grammar string: {}\n", .{err});
            break :blk null;
        };
        if (grammar) |*g| grammar_state = g.initState() catch |err| blk: {
            eprint("Error: grammar state init failed (OOM): {}\n", .{err});
            break :blk null;
        };
    } else if (cli.grammar_path) |path| {
        const gf = Io.Dir.cwd().openFile(g_io, path, .{}) catch |err| blk: {
            eprint("Error: could not open grammar file '{s}': {}\n", .{ path, err });
            break :blk null;
        };
        if (gf) |file| {
            defer file.close(g_io);
            const stat = file.stat(g_io) catch |err| blk: {
                eprint("Error: could not stat grammar file '{s}': {}\n", .{ path, err });
                break :blk null;
            };
            if (stat) |s| {
                const buf = allocator.alloc(u8, s.size) catch |err| blk: {
                    eprint("Error: could not allocate {d} bytes for grammar file '{s}': {}\n", .{ s.size, path, err });
                    break :blk null;
                };
                if (buf) |b| {
                    if (file.readPositionalAll(g_io, b, 0)) |_| {
                        grammar = grammar_mod.Grammar.parse(allocator, b) catch |err| blk: {
                            eprint("Error: failed to parse grammar file '{s}': {}\n", .{ path, err });
                            allocator.free(b);
                            break :blk null;
                        };
                        if (grammar != null) grammar_source_buf = b; // Grammar borrows into b
                        if (grammar) |*g| grammar_state = g.initState() catch |err| blk: {
                            eprint("Error: grammar state init failed (OOM): {}\n", .{err});
                            break :blk null;
                        };
                    } else |err| {
                        eprint("Error: could not read grammar file '{s}': {}\n", .{ path, err });
                        allocator.free(b);
                    }
                }
            }
        }
    }
    defer {
        if (grammar_state) |*gs| gs.deinit();
        if (grammar) |*g| g.deinit();
        if (grammar_source_buf) |b| allocator.free(b);
    }
    if (token_ids.len > 0) {
        const first_logits = mdl.getLogits();
        // Grammar masking for first token
        // Grammar masking
        if (grammar_state) |*gs| {
            if (!gs.isComplete()) {
                gs.grammar.maskLogits(gs, first_logits, tok.id_to_token.items) catch |err| {
                    eprint("Error: grammar mask OOM: {s}\n", .{@errorName(err)});
                    return null;
                };
            }
        }
        // JSON mode: force first token to start with {
        if (json_mode_active) {
            const vocab_texts = tok.id_to_token.items;
            for (first_logits, 0..) |*logit, tid| {
                if (tid >= vocab_texts.len) break;
                const t = grammar_mod.Grammar.getEffectiveText(vocab_texts[tid]);
                if (t.len == 0 or t[0] != '{') logit.* = -std.math.inf(f32);
            }
        }
        if (use_sampling) {
            if (cli.mirostat_mode >= 2) {
                first_gen_token = math_ops.sampleMirostat(first_logits, cli.mirostat_tau, cli.mirostat_eta, &cli_mirostat_mu, cli.temperature, prng.random());
            } else {
                if (cli.min_p > 0) math_ops.applyMinP(first_logits, cli.min_p);
                if (cli.xtc_probability > 0) math_ops.applyXtc(first_logits, cli.xtc_probability, cli.xtc_threshold, prng.random());
                first_gen_token = math_ops.sampleToken(first_logits, cli.temperature, cli.top_k, cli.top_p, prng.random());
            }
        } else if (grammar_state != null or json_mode_active) {
            first_gen_token = math_ops.argmax(first_logits);
        }
        // Update grammar state with first accepted token
        if (grammar_state) |*gs| {
            const tok_slice = [1]u32{first_gen_token};
            const text = tok.decode(&tok_slice) catch |err| blk: {
                eprint("Warning: grammar token decode failed (id={d}): {}\n", .{ first_gen_token, err });
                break :blk null;
            };
            defer if (text) |t| allocator.free(t);
            gs.acceptToken(text orelse "");
        }
        // Track JSON brace depth (scan raw token text — avoids allocation)
        if (json_mode_active and first_gen_token < tok.id_to_token.items.len) {
            for (tok.id_to_token.items[first_gen_token]) |c| {
                if (c == '{' or c == '[') json_depth += 1;
                if (c == '}' or c == ']') json_depth -= 1;
            }
        }
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
    var first_is_eog = token_ids.len > 0 and isEogToken(first_gen_token, eog);
    // Grammar completion after first token
    if (grammar_state) |*gs| {
        if (gs.isComplete()) first_is_eog = true;
    }
    // For grammar completion, still output the token that completed the grammar
    const grammar_completed_first = if (grammar_state) |*gs| gs.isComplete() else false;
    var hit_eog = first_is_eog and !grammar_completed_first;
    if (token_ids.len > 0 and (!first_is_eog or grammar_completed_first)) {
        gen_ids_buf[0] = first_gen_token;
        token_count = 1;
        prev_token = first_gen_token;
        repeat_count = 1;
        if (grammar_completed_first) hit_eog = true;
    }

    // Power throttling state: last measured forward-pass duration (ns).
    var power_last_forward_ns: u64 = 0;

    for (0..cli.max_tokens -| 1) |gi| {
        if (first_is_eog or token_ids.len == 0) break;

        // Jump decoding: if grammar allows exactly one token, skip forward pass
        if (grammar_state != null and !use_sampling) {
            if (grammar) |*g| {
                if (grammar_state) |*gs| {
                    if (g.singleValidToken(gs, tok.id_to_token.items)) |jump_tok| {
                        // Use raw vocab text for grammar state — consistent with maskLogits.
                        const jt_raw = if (jump_tok < tok.id_to_token.items.len) tok.id_to_token.items[jump_tok] else "";
                        gs.acceptToken(jt_raw);
                        if (token_count >= gen_ids_buf.len) break;
                        gen_ids_buf[token_count] = jump_tok;
                        token_count += 1;
                        last = jump_tok;
                        if (gs.isComplete()) {
                            hit_eog = true;
                            break;
                        }
                        continue;
                    }
                }
            }
        }

        // Power throttling: sleep before forward to cap GPU utilisation.
        // At P%, we want GPU active P% and idle (100-P)% of the token period.
        // Approximate: measure forward time and sleep (100-P)/P × forward_ns.
        // On first token the sleep is skipped (no timing yet).
        if (cli.power_pct < 100 and gi > 0) {
            const idle_num = @as(u64, 100 - cli.power_pct);
            const idle_ns = power_last_forward_ns * idle_num / cli.power_pct;
            if (idle_ns > 0) {
                const ts = std.posix.timespec{
                    .sec = @intCast(idle_ns / 1_000_000_000),
                    .nsec = @intCast(idle_ns % 1_000_000_000),
                };
                _ = std.posix.system.nanosleep(&ts, null);
            }
        }
        var power_ts0: std.posix.timespec = undefined;
        _ = std.posix.system.clock_gettime(.MONOTONIC, &power_ts0);
        var next = mdl.forward(last) catch |e| {
            eprint("Error: generation failed at token {d}: {}\n", .{ gi + 1, e });
            break;
        };
        var power_ts1: std.posix.timespec = undefined;
        _ = std.posix.system.clock_gettime(.MONOTONIC, &power_ts1);
        const power_delta_ns: i64 = (@as(i64, power_ts1.sec) - @as(i64, power_ts0.sec)) * 1_000_000_000 + (@as(i64, power_ts1.nsec) - @as(i64, power_ts0.nsec));
        power_last_forward_ns = @intCast(@max(0, power_delta_ns));
        // Apply repeat penalty to logits for recently generated tokens
        const logits = mdl.getLogits();
        if (use_repeat_penalty and token_count > 0) {
            math_ops.applyRepeatPenalty(logits, gen_ids_buf[0..token_count], cli.repeat_penalty);
        }
        if (cli.dry_multiplier > 0 and token_count > 0) {
            math_ops.applyDry(logits, gen_ids_buf[0..token_count], cli.dry_multiplier, cli.dry_length);
        }
        // Grammar-constrained decoding: mask disallowed tokens
        const has_grammar = if (grammar_state) |*gs| !gs.isComplete() else false;
        if (has_grammar) {
            const vocab_texts = tok.id_to_token.items;
            grammar_state.?.grammar.maskLogits(&grammar_state.?, logits, vocab_texts) catch |err| {
                eprint("Error: grammar mask OOM: {s}\n", .{@errorName(err)});
                break;
            };
        }
        if (cli.mirostat_mode >= 2 and use_sampling) {
            next = math_ops.sampleMirostat(logits, cli.mirostat_tau, cli.mirostat_eta, &cli_mirostat_mu, cli.temperature, prng.random());
        } else if (use_sampling) {
            if (cli.min_p > 0) math_ops.applyMinP(logits, cli.min_p);
            if (cli.xtc_probability > 0) math_ops.applyXtc(logits, cli.xtc_probability, cli.xtc_threshold, prng.random());
            next = math_ops.sampleToken(logits, cli.temperature, cli.top_k, cli.top_p, prng.random());
        } else if (has_grammar or (use_repeat_penalty and token_count > 0) or cli.dry_multiplier > 0) {
            // Re-argmax after masking or penalty
            next = math_ops.argmax(logits);
        }
        dbg("gen step {d}: token={d}", .{ gi, next });
        if (isEogToken(next, eog)) {
            hit_eog = true;
            break;
        }
        if (token_count >= gen_ids_buf.len) break;
        gen_ids_buf[token_count] = next;
        // Update grammar state with accepted token.
        // Use raw vocab text (id_to_token) — NOT decoded text — so that
        // getEffectiveText strips BPE prefixes consistently with maskLogits.
        if (grammar_state) |*gs| {
            const raw_text = if (next < tok.id_to_token.items.len) tok.id_to_token.items[next] else "";
            gs.acceptToken(raw_text);
            if (gs.isComplete()) {
                hit_eog = true;
                token_count += 1;
                break;
            }
        }
        // Track JSON brace depth and stop at balanced close
        // (scan raw token text directly — avoids per-token allocation)
        if (json_mode_active and next < tok.id_to_token.items.len) {
            for (tok.id_to_token.items[next]) |c| {
                if (c == '{' or c == '[') json_depth += 1;
                if (c == '}' or c == ']') json_depth -= 1;
            }
            if (json_depth <= 0) {
                hit_eog = true;
                token_count += 1;
                break;
            }
        }
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
        if (emitGeneratedTokens(cli) and token_count - batch_start >= batch_size) {
            if (display.mode != .json) {
                flushTokenBatch(tok, tok_kind, allocator, gen_ids_buf[batch_start..@min(token_count, gen_ids_buf.len)], &started_output);
                batch_start = token_count;
            }
        }
    }
    // Flush remaining tokens
    if (emitGeneratedTokens(cli) and display.mode != .json and token_count > batch_start and token_count <= gen_ids_buf.len) {
        flushTokenBatch(tok, tok_kind, allocator, gen_ids_buf[batch_start..token_count], &started_output);
    }
    // Ensure a trailing newline for piped output (not TTY, not JSON)
    if (emitGeneratedTokens(cli) and !g_tty and display.mode != .json and started_output) {
        _ = std.posix.system.write(stdout_file.handle, "\n", 1);
    }
    if (hit_eog and g_verbose) print("\n[EOG]\n", .{});
    const gen_ms = elapsedMs(gen_start);

    // Decode full response text for return value (skip if caller doesn't need it)
    const response_text: ?[]u8 = if (token_count > 0 and (need_response or display.mode == .json))
        switch (tok_kind) {
            .spm, .spm_no_dummy => tok.decodeSpm(gen_ids_buf[0..token_count]) catch |err| blk: {
                eprint("Error: failed to decode {d} generated tokens: {}\n", .{ token_count, err });
                break :blk null;
            },
            .bpe => tok.decode(gen_ids_buf[0..token_count]) catch |err| blk: {
                eprint("Error: failed to decode {d} generated tokens: {}\n", .{ token_count, err });
                break :blk null;
            },
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
    _ = std.posix.system.write(stdout_file.handle, text.ptr, text.len);
}

test {
    // Force test discovery for all modules with test blocks.
    // Zig 0.16 uses lazy test discovery — files imported at the top level
    // but not referenced by any test block are silently excluded.
    _ = @import("cli.zig");
    _ = @import("display.zig");
    _ = @import("ops/split_attention.zig");
    _ = @import("ops/sparse_attn.zig");
    _ = @import("spec/pflash.zig");
    _ = @import("spec/dspark.zig");
    _ = @import("spec/caps.zig");
    _ = @import("lora.zig");
    _ = @import("arch.zig");
    _ = @import("perf.zig");
    _ = @import("recipe.zig");
    _ = @import("chat_template.zig");
    _ = @import("pull.zig");
    _ = @import("calibrate.zig");
    _ = @import("image.zig");
    _ = @import("image_tokens.zig");
    _ = @import("steering.zig");
    _ = @import("eval.zig");
    _ = @import("expert_profile.zig");
    _ = @import("expert_cache.zig");
    _ = @import("term.zig");
    _ = @import("thread_pool.zig");
    _ = @import("ops/kv_quant.zig");
    _ = @import("ops/quant.zig");
    _ = @import("ops/math.zig");
    _ = @import("ops/sampler_stack.zig");
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
    _ = @import("server/tools.zig");
    _ = @import("server/rate_limiter.zig");
    _ = @import("server/metrics.zig");
    _ = @import("server/fixed_buf_stream.zig");
    _ = @import("server/scheduler.zig");
    _ = @import("sim_clock.zig");
    _ = @import("kvcache/block_allocator.zig");
    _ = @import("kvcache/manager.zig");
    _ = @import("kvcache/tiered.zig");
    _ = @import("kvcache/checkpoint.zig");
    _ = @import("models/model.zig");
    _ = @import("models/gemma3.zig");
    _ = @import("models/gemma4.zig");
    _ = @import("models/diffusion_gemma.zig");
    _ = @import("models/qwen35.zig");
    _ = @import("models/gpt_oss.zig");
    _ = @import("models/glm4.zig");
    _ = @import("models/deepseek4.zig");
    _ = @import("models/llama4.zig");
    _ = @import("models/nemotron_nano.zig");
    _ = @import("models/nemotron_h.zig");
    _ = @import("models/vision.zig");
    _ = @import("backend/cpu.zig");
    // Metal links Apple frameworks; importing it off-macOS breaks the test link.
    // Same gate as the backend dispatcher (see backend.zig MetalBackend).
    if (comptime build_options.enable_metal and @import("builtin").os.tag == .macos) {
        _ = @import("backend/metal.zig");
    }
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
    _ = @import("backend/kernels/cpu/gemv_iq_small.zig");
    _ = @import("backend/kernels/cpu/gemv_q_small.zig");
    _ = @import("backend/kernels/cpu/gemv_q4_0.zig");
    _ = @import("backend/kernels/cpu/gemv_q4_k.zig");
    _ = @import("backend/kernels/cpu/gemv_q5_k.zig");
    _ = @import("backend/kernels/cpu/gemv_q6_k.zig");
    _ = @import("backend/kernels/cpu/gemv_q8_0.zig");
    _ = @import("spec/spec_decode.zig");
    _ = @import("spec/ngram.zig");
    _ = @import("fuzz_tests.zig");
    _ = @import("spec/ddtree.zig");
    _ = @import("backend/kernels/cpu/sdpa_tree.zig");
    _ = @import("backend/mega_compose.zig");
    _ = @import("backend/megakernel.zig");
    _ = @import("ops/gptq.zig");
    _ = @import("ops/awq.zig");
    _ = @import("ops/hqq.zig");
}

test "cpu backend rms_norm via tagged union dispatch" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    var bs = BackendState{};
    bs.init(std.testing.allocator, .cpu, threaded.io(), 0);
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
    defer threaded2.deinit();
    var bs = BackendState{};
    bs.init(std.testing.allocator, .cpu, threaded2.io(), 0);
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
    defer threaded3.deinit();
    var bs = BackendState{};
    bs.init(std.testing.allocator, .cpu, threaded3.io(), 0);
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

test "closeMatch detects substitutions" {
    try std.testing.expect(closeMatch("temeprature", "temperature")); // transposition
    try std.testing.expect(closeMatch("temperture", "temperature")); // missing 'a' (insertion match)
    try std.testing.expect(closeMatch("bakend", "backend")); // missing 'c' (insertion match)
    try std.testing.expect(closeMatch("bakcend", "backend")); // same len, 2 subs
    try std.testing.expect(!closeMatch("xyz", "temperature")); // completely different
    try std.testing.expect(!closeMatch("temperature", "temperature")); // exact match not a typo
}

test "insertionMatch detects missing char" {
    try std.testing.expect(insertionMatch("temperatur", "temperature")); // missing trailing 'e'
    try std.testing.expect(insertionMatch("verbos", "verbose")); // missing trailing 'e'
    try std.testing.expect(insertionMatch("bacend", "backend")); // missing 'k'
    try std.testing.expect(!insertionMatch("abc", "xyzwv")); // completely different
}

test "suggestSpec finds known flags" {
    // Known typos should find suggestions
    const s1 = suggestSpec("temeprature");
    try std.testing.expect(s1 != null);
    try std.testing.expectEqualStrings("temperature", s1.?);

    const s2 = suggestSpec("temperatur");
    try std.testing.expect(s2 != null);
    try std.testing.expectEqualStrings("temperature", s2.?);

    // Completely unrelated string should not match
    try std.testing.expect(suggestSpec("foobar") == null);
    try std.testing.expect(suggestSpec("x") == null);
}

test "looksLikeUnknownShortOpt detects short typos" {
    try std.testing.expect(looksLikeUnknownShortOpt("-z"));
    try std.testing.expect(looksLikeUnknownShortOpt("-qv"));
    try std.testing.expect(!looksLikeUnknownShortOpt("-5"));
    try std.testing.expect(!looksLikeUnknownShortOpt("--quiet"));
    try std.testing.expect(!looksLikeUnknownShortOpt("model.gguf"));
    try std.testing.expect(!looksLikeUnknownShortOpt("-"));
}

// Force test discovery for modules only imported at runtime (inside function bodies).
// Without these comptime references, zig test won't find their test blocks.
comptime {
    _ = @import("devices/discovery.zig");
    _ = @import("parallel/peer_discovery.zig");
    _ = @import("parallel/tp.zig");
    _ = @import("kvcache/prefetch.zig");
}

test "emitGeneratedTokens only rank 0 prints in a pair" {
    var cli: CliArgs = undefined;
    cli.tp_rank = 0;
    cli.tp_degree = 2;
    cli.pp_degree = 1;
    try std.testing.expect(emitGeneratedTokens(&cli));
    cli.tp_rank = 1;
    try std.testing.expect(!emitGeneratedTokens(&cli));
    cli.tp_degree = 1;
    cli.pp_degree = 2;
    try std.testing.expect(!emitGeneratedTokens(&cli));
    cli.tp_rank = 0;
    try std.testing.expect(emitGeneratedTokens(&cli));
    cli.tp_rank = 1;
    cli.pp_degree = 1;
    try std.testing.expect(emitGeneratedTokens(&cli));
}

test "distributedLockstep when tp or pp is a pair" {
    var cli: CliArgs = undefined;
    cli.tp_degree = 1;
    cli.pp_degree = 1;
    try std.testing.expect(!distributedLockstep(&cli));
    cli.tp_degree = 2;
    try std.testing.expect(distributedLockstep(&cli));
    cli.tp_degree = 1;
    cli.pp_degree = 2;
    try std.testing.expect(distributedLockstep(&cli));
}

test "parseIpv4 valid addresses" {
    var out: [4]u8 = undefined;
    try std.testing.expect(parseIpv4("192.168.1.1", &out));
    try std.testing.expectEqual([4]u8{ 192, 168, 1, 1 }, out);
    try std.testing.expect(parseIpv4("0.0.0.0", &out));
    try std.testing.expectEqual([4]u8{ 0, 0, 0, 0 }, out);
    try std.testing.expect(parseIpv4("255.255.255.255", &out));
    try std.testing.expectEqual([4]u8{ 255, 255, 255, 255 }, out);
    try std.testing.expect(parseIpv4("10.0.0.1", &out));
}

test "parseIpv4 invalid addresses" {
    var out: [4]u8 = undefined;
    try std.testing.expect(!parseIpv4("", &out));
    try std.testing.expect(!parseIpv4("256.0.0.1", &out));
    try std.testing.expect(!parseIpv4("1.2.3", &out));
    try std.testing.expect(!parseIpv4("1.2.3.4.5", &out));
    try std.testing.expect(!parseIpv4(".1.2.3.4", &out));
    try std.testing.expect(!parseIpv4("1.2.3.4.", &out));
    try std.testing.expect(!parseIpv4("1..2.3.4", &out));
    try std.testing.expect(!parseIpv4("abc", &out));
    try std.testing.expect(!parseIpv4("1.2.x.4", &out));
}

test "parsePeerAddr host only" {
    const pa = parsePeerAddr("10.0.0.1", 8080);
    try std.testing.expect(pa != null);
    try std.testing.expectEqual([4]u8{ 10, 0, 0, 1 }, pa.?.host);
    try std.testing.expectEqual(@as(u16, 8080), pa.?.port);
}

test "parsePeerAddr host:port" {
    const pa = parsePeerAddr("10.0.0.2:9000", 8080);
    try std.testing.expect(pa != null);
    try std.testing.expectEqual([4]u8{ 10, 0, 0, 2 }, pa.?.host);
    try std.testing.expectEqual(@as(u16, 9000), pa.?.port);
}

test "parsePeerAddr invalid" {
    try std.testing.expect(parsePeerAddr("notanip", 8080) == null);
    try std.testing.expect(parsePeerAddr("1.2.3.4:99999", 8080) == null);
    try std.testing.expect(parsePeerAddr("", 8080) == null);
}

test "resolveTransportKind explicit" {
    try std.testing.expectEqual(TransportMod.TransportKind.tcp, try resolveTransportKind(.tcp, "10.0.0.1"));
    try std.testing.expectEqual(TransportMod.TransportKind.shm, try resolveTransportKind(.shm, "10.0.0.1"));
    try std.testing.expectEqual(TransportMod.TransportKind.nccl, try resolveTransportKind(.nccl, "10.0.0.1"));
    try std.testing.expectError(error.TransportNotImplemented, resolveTransportKind(.rdma, "10.0.0.1"));
    try std.testing.expectError(error.TransportNotImplemented, resolveTransportKind(.udp, "10.0.0.1"));
    try std.testing.expectError(error.TransportNotImplemented, resolveTransportKind(.grpc, "10.0.0.1"));
}

test "resolveTransportKind auto localhost" {
    try std.testing.expectEqual(TransportMod.TransportKind.shm, try resolveTransportKind(.auto, "localhost"));
    try std.testing.expectEqual(TransportMod.TransportKind.shm, try resolveTransportKind(.auto, "127.0.0.1"));
    try std.testing.expectEqual(TransportMod.TransportKind.tcp, try resolveTransportKind(.auto, "10.0.0.2"));
}

test "isKnownShort recognizes valid short flags" {
    // 'h' for help, 't' for temperature should be registered
    try std.testing.expect(isKnownShort('h'));
    // An invalid short should return false
    try std.testing.expect(!isKnownShort(0));
    try std.testing.expect(!isKnownShort('Z'));
}

test "isEogToken" {
    const eog = EogTokens{ .ids = .{ 2, 7, 0, 0, 0, 0, 0, 0 }, .len = 2 };
    try std.testing.expect(isEogToken(2, eog));
    try std.testing.expect(isEogToken(7, eog));
    try std.testing.expect(!isEogToken(99, eog));
    try std.testing.expect(!isEogToken(0, eog)); // id 0 not in [2,7]
}

test "parseUint null input" {
    try std.testing.expect(parseUint(u32, null, "n") == null);
    try std.testing.expect(parseUint(u64, null, "k") == null);
}

test "parseUint valid input" {
    try std.testing.expectEqual(@as(?u32, 42), parseUint(u32, "42", "n"));
    try std.testing.expectEqual(@as(?u64, 1024), parseUint(u64, "1024", "k"));
    try std.testing.expectEqual(@as(?u16, 0), parseUint(u16, "0", "port"));
}

test "fuzz: main.zig pure functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // Comptime symbol refs for all functions not easily called at runtime
            comptime {
                _ = &milliTimestamp;
                _ = &nanoTimestamp;
                _ = &readStdinAll;
                _ = &kvTypeOrExit;
                _ = &detectFreeRam;
                _ = &preloadRegion;
                _ = &preloadModel;
                _ = &preloadRegionProgress;
                _ = &checkSubcommand;
                _ = &parseCli;
                _ = &setupTransport;
                _ = &exchangeDeviceCaps;
                _ = &measurePeerRtt;
                _ = &parseU32;
                _ = &parseU64;
                _ = &parseU16;
                _ = &parseF32;
                _ = &rejectEqualsOnFlag;
                _ = &rejectUnknownOptions;
                _ = &rejectFlagAsValue;
                _ = &rejectUnknownShortPositionals;
                _ = &looksLikeUnknownShortOpt;
                _ = &validateFileExists;
                _ = &runBenchmark;
                _ = &printUsage;
                _ = &elapsedMs;
                _ = &getEogTokens;
                _ = &loadImage;
                _ = &initAndRun;
                _ = &runRepl;
                _ = &generateAndPrint;
                _ = &generateSpeculative;
                _ = &generateAndPrintInner;
                _ = &flushTokenBatch;
                _ = &isKnownSpec;
                _ = &suggestSpec;
                _ = &closeMatch;
                _ = &insertionMatch;
                _ = &isKnownShort;
                _ = &isEogToken;
                _ = &isKnownSpec;
            }

            var ip_out: [4]u8 = undefined;
            const s_len = smith.valueWithHash(u8, 0) % 32;
            var ip_buf: [32]u8 = undefined;
            smith.bytesWithHash(&ip_buf, 1);
            _ = parseIpv4(ip_buf[0..s_len], &ip_out);

            // parsePeerAddr with random strings
            _ = parsePeerAddr(ip_buf[0..s_len], smith.valueWithHash(u16, 2));

            // resolveTransportKind all choices (unimplemented transports return error, not exit)
            const choices = [_]TransportChoice{ .auto, .tcp, .shm, .nccl, .rdma, .udp, .grpc };
            for (choices) |c| {
                _ = resolveTransportKind(c, "localhost") catch {};
                _ = resolveTransportKind(c, "10.0.0.1") catch {};
            }

            // parseUint null/valid paths
            _ = parseUint(u32, null, "n");
            const num_str = "42";
            _ = parseUint(u32, num_str, "n");

            // isKnownShort/isEogToken
            const ch = smith.valueWithHash(u8, 3);
            _ = isKnownShort(ch);
            _ = isEogToken(smith.valueWithHash(u32, 4), EogTokens{ .ids = .{ 1, 2, 0, 0, 0, 0, 0, 0 }, .len = 2 });

            // isKnownSpec/suggestSpec/closeMatch/insertionMatch
            _ = isKnownSpec(ip_buf[0..@min(s_len, 31)]);
            _ = suggestSpec(ip_buf[0..@min(s_len, 31)]);
            _ = closeMatch(ip_buf[0..@min(s_len, 15)], ip_buf[16..@min(16 + s_len, 31)]);
        }
    }.f, .{});
}
