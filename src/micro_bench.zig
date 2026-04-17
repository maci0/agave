//! Standalone micro-benchmark binary for per-kernel and end-to-end benchmarking.
//!
//! Runs individual backend kernels with synthetic data, collects timing samples,
//! and outputs results as JSON lines to stdout. Designed for machine consumption
//! and CI regression tracking.
//!
//! Usage:
//!   agave-bench <kernel_name> [--n N] [--k K] [--iters N] [--backend cpu|metal|vulkan|cuda]
//!   agave-bench e2e --model <path> --backend X -n N
//!
//! Examples:
//!   agave-bench gemv_f32 --n 4096 --k 4096 --iters 50
//!   agave-bench rms_norm --n=4096 --backend=metal
//!   agave-bench sdpa --n 32 --k 128 --iters 20

const std = @import("std");

const backend_mod = @import("backend/backend.zig");
const Backend = backend_mod.Backend;
const BackendState = backend_mod.BackendState;
const TensorData = backend_mod.TensorData;
const format_mod = @import("format/format.zig");
const Format = format_mod.Format;
const GGUFFile = format_mod.GGUFFile;
const SafeTensorsDir = format_mod.SafeTensorsDir;
const model_mod = @import("models/model.zig");
const Model = model_mod.Model;
const tok_mod = @import("tokenizer/tokenizer.zig");
const BpeTokenizer = tok_mod.BpeTokenizer;
const arch_mod = @import("arch.zig");
const Arch = arch_mod.Arch;
const TokenizerKind = tok_mod.TokenizerKind;
const display_mod = @import("display.zig");
const kv_quant = @import("ops/kv_quant.zig");
const KvQuantType = kv_quant.KvQuantType;

// ── Named constants ──────────────────────────────────────────────

/// Number of warmup iterations before collecting timing samples.
const warmup_iters: usize = 10;
/// Default number of timed iterations when --iters is not specified.
const default_iters: usize = 100;
/// Maximum number of timing samples to collect.
const max_samples: usize = 1000;
/// Default dimension for GEMV output rows and elementwise ops.
const default_dim: usize = 4096;
/// Default inner dimension for GEMV (columns).
const default_k: usize = 4096;
/// Modulus for synthetic input vector data generation.
const synthetic_x_mod: usize = 17;
/// Modulus for synthetic weight data generation.
const synthetic_w_mod: usize = 31;
/// Default number of attention heads for RoPE/SDPA benchmarks.
const default_n_heads: usize = 32;
/// Default head dimension for RoPE/SDPA benchmarks when not specified.
const default_head_dim: usize = 128;
/// Default RoPE theta value.
const default_rope_theta: f32 = 10000.0;
/// RMS norm epsilon.
const rms_norm_eps: f32 = 1e-6;
/// L2 norm epsilon.
const l2_norm_eps: f32 = 1e-6;
const gemma_fallback_eos = arch_mod.gemma_fallback_eos;
const default_fallback_eos = arch_mod.default_fallback_eos;
const default_bos_id = arch_mod.default_bos_id;
/// Scale factor for synthetic input data.
const synthetic_x_scale: f32 = 0.01;
/// Offset for synthetic input data.
const synthetic_x_offset: f32 = -0.08;
/// Scale factor for synthetic weight data.
const synthetic_w_scale: f32 = 0.001;
/// Offset for synthetic weight data.
const synthetic_w_offset: f32 = -0.015;
/// Buffer size for formatted output.
const output_buf_size: usize = 4096;
/// Q8_0 block bytes. Canonical source: backend/backend.zig.
const q8_0_block_bytes = backend_mod.q8_0_block_bytes;
/// Q4_0 block bytes. Canonical source: backend/backend.zig.
const q4_0_block_bytes = backend_mod.q4_0_block_bytes;
/// Elements per quantization block (Q8_0, Q4_0). Canonical source: backend/backend.zig.
const quant_group_size = backend_mod.quant_block_elems;
/// Synthetic f16 scale value byte 0 (little-endian f16 ~ 0.00875).
const synthetic_scale_byte_0: u8 = 0x1E;
/// Synthetic f16 scale value byte 1 (paired with byte 0 above).
const synthetic_scale_byte_1: u8 = 0x21;
/// Default sequence length for SDPA benchmark.
const default_sdpa_seq_len: usize = 512;
/// Default position index for RoPE benchmark.
const default_rope_pos: usize = 42;
/// SDPA synthetic query modulus.
const sdpa_q_mod: usize = 13;
/// SDPA synthetic query scale.
const sdpa_q_scale: f32 = 0.01;
/// SDPA synthetic query offset.
const sdpa_q_offset: f32 = -0.06;
/// SDPA synthetic key modulus.
const sdpa_k_mod: usize = 19;
/// SDPA synthetic KV scale.
const sdpa_kv_scale: f32 = 0.005;
/// SDPA synthetic key offset.
const sdpa_k_offset: f32 = -0.04;
/// SDPA synthetic value modulus.
const sdpa_v_mod: usize = 23;
/// SDPA synthetic value offset.
const sdpa_v_offset: f32 = -0.05;
/// SDPA synthetic v_new modulus.
const sdpa_v_new_mod: usize = 29;
/// Default number of tokens to generate in e2e mode.
const default_gen_tokens: usize = 10;
/// Default prompt used for e2e benchmarking.
const e2e_prompt = "What is 2+2?";
/// Number of consecutive identical tokens before halting e2e generation.
const e2e_repeat_halt_threshold: u32 = 6;

// ── Output helpers ───────────────────────────────────────────────

/// Standard I/O file handles via std.Io.File (Zig 0.16 idiom).
const stdout_file = std.Io.File.stdout();
const stderr_file = std.Io.File.stderr();

/// Writes all bytes to a file descriptor using raw C write.
fn fdWriteAll(fd: std.posix.fd_t, bytes: []const u8) void {
    var written: usize = 0;
    while (written < bytes.len) {
        const result = std.c.write(fd, bytes[written..].ptr, bytes[written..].len);
        const n: isize = @bitCast(result);
        if (n <= 0) break;
        written += @intCast(n);
    }
}

fn print(comptime fmt: []const u8, args: anytype) void {
    var buf: [output_buf_size]u8 = undefined;
    fdWriteAll(stdout_file.handle, std.fmt.bufPrint(&buf, fmt, args) catch return);
}

fn eprint(comptime fmt: []const u8, args: anytype) void {
    var buf: [output_buf_size]u8 = undefined;
    fdWriteAll(stderr_file.handle, std.fmt.bufPrint(&buf, fmt, args) catch return);
}

// ── CLI parsing ──────────────────────────────────────────────────

const Mode = enum { kernel, e2e };

const BackendChoice = backend_mod.BackendChoice;

const Kernel = enum {
    gemv_f32,
    gemv_bf16,
    gemv_f16,
    gemv_q8_0,
    gemv_q4_0,
    rms_norm,
    silu,
    gelu,
    softmax,
    l2_norm,
    add,
    mul,
    rope,
    sdpa,
    sdpa_turbo4,
    sdpa_turbo3,
    sdpa_turbo2,
};

const CliArgs = struct {
    mode: Mode,
    kernel: ?Kernel,
    /// In kernel mode: output dimension / vector length.
    /// In e2e mode: number of tokens to generate.
    n: usize,
    k: usize,
    iters: usize,
    backend: BackendChoice,
    model_path: ?[]const u8,
};

/// Parses CLI arguments from process args.
/// Returns null on help/version (exit 0) or parse error (exit 1 via std.process.exit).
fn parseCli(proc_args: std.process.Args) ?CliArgs {
    var args_iter = proc_args.iterate();

    _ = args_iter.skip(); // skip program name

    var n_was_set = false;
    var result = CliArgs{
        .mode = .kernel,
        .kernel = null,
        .n = default_dim,
        .k = default_k,
        .iters = default_iters,
        .backend = .cpu,
        .model_path = null,
    };

    // Collect all args to handle --help anywhere in the arg list
    var positional: ?[]const u8 = null;
    var all_args: [32][]const u8 = undefined;
    var n_args: usize = 0;

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return null; // exit 0
        }
        if (std.mem.eql(u8, arg, "--version") or std.mem.eql(u8, arg, "-v")) {
            print("agave-bench {s}\n", .{display_mod.version});
            return null; // exit 0
        }
        if (n_args < all_args.len) {
            all_args[n_args] = arg;
            n_args += 1;
        }
    }

    if (n_args == 0) {
        printUsage();
        return null; // exit 0 (no args = show help)
    }

    // First positional arg: kernel name or "e2e"
    positional = all_args[0];
    if (std.mem.eql(u8, positional.?, "e2e")) {
        result.mode = .e2e;
    } else {
        result.kernel = parseKernelName(positional.?) orelse {
            eprint("Error: unknown kernel '{s}'\n", .{positional.?});
            eprint("  Valid kernels: gemv_f32 gemv_bf16 gemv_f16 gemv_q8_0 gemv_q4_0\n", .{});
            eprint("                 rms_norm silu gelu softmax l2_norm add mul rope\n", .{});
            eprint("                 sdpa sdpa_turbo4 sdpa_turbo3 sdpa_turbo2\n", .{});
            eprint("Run 'agave-bench --help' for more information.\n", .{});
            std.process.exit(1);
        };
    }

    // Parse remaining flags (supports both --key=value and --key value forms).
    const args_slice = all_args[0..n_args];
    var i: usize = 1; // skip positional[0] (kernel name)
    while (i < n_args) : (i += 1) {
        if (getArgValue(args_slice, &i, "--n") orelse getArgValue(args_slice, &i, "-n")) |v| {
            result.n = std.fmt.parseInt(usize, v, 10) catch {
                eprint("Error: invalid value for --n: '{s}'\n", .{v});
                eprint("Run 'agave-bench --help' for more information.\n", .{});
                std.process.exit(1);
            };
            n_was_set = true;
        } else if (getArgValue(args_slice, &i, "--k") orelse getArgValue(args_slice, &i, "-k")) |v| {
            result.k = std.fmt.parseInt(usize, v, 10) catch {
                eprint("Error: invalid value for --k: '{s}'\n", .{v});
                eprint("Run 'agave-bench --help' for more information.\n", .{});
                std.process.exit(1);
            };
        } else if (getArgValue(args_slice, &i, "--iters")) |v| {
            result.iters = std.fmt.parseInt(usize, v, 10) catch {
                eprint("Error: invalid value for --iters: '{s}'\n", .{v});
                eprint("Run 'agave-bench --help' for more information.\n", .{});
                std.process.exit(1);
            };
            if (result.iters > max_samples) {
                eprint("Warning: clamping iters to {d}\n", .{max_samples});
                result.iters = max_samples;
            }
        } else if (getArgValue(args_slice, &i, "--backend")) |v| {
            result.backend = parseBackendName(v) orelse {
                eprint("Error: unknown backend '{s}'\n", .{v});
                eprint("  Valid options: auto, cpu, metal, vulkan, cuda, rocm\n", .{});
                eprint("Run 'agave-bench --help' for more information.\n", .{});
                std.process.exit(1);
            };
        } else if (getArgValue(args_slice, &i, "--model")) |v| {
            result.model_path = v;
        } else {
            eprint("Error: unknown argument '{s}'\n", .{args_slice[i]});
            eprint("Run 'agave-bench --help' for more information.\n", .{});
            std.process.exit(1);
        }
    }

    if (result.mode == .kernel and result.kernel == null) {
        eprint("Error: kernel name required\n", .{});
        printUsage();
        std.process.exit(1);
    }

    // In e2e mode, default -n to gen_tokens count (not vector dimension)
    if (result.mode == .e2e and !n_was_set) {
        result.n = default_gen_tokens;
    }

    // Validate e2e mode requires --model before we spend time initializing backends
    if (result.mode == .e2e and result.model_path == null) {
        eprint("Error: --model is required for e2e mode\n", .{});
        eprint("  Example: agave-bench e2e --model model.gguf --backend cpu\n", .{});
        eprint("Run 'agave-bench --help' for more information.\n", .{});
        std.process.exit(1);
    }

    return result;
}

/// Extracts the value from a "--key=value" argument, or null if the key doesn't match.
fn parseKeyValue(arg: []const u8, key: []const u8) ?[]const u8 {
    const prefix_eq = blk: {
        if (arg.len < key.len + 1) break :blk false;
        if (!std.mem.startsWith(u8, arg, key)) break :blk false;
        if (arg[key.len] == '=') break :blk true;
        break :blk false;
    };
    if (prefix_eq) {
        return arg[key.len + 1 ..];
    }
    return null;
}

/// Extracts the value for a flag, supporting both `--key=value` and `--key value` forms.
/// Advances `i` past the consumed value when using the space-separated form.
fn getArgValue(args: []const []const u8, i: *usize, key: []const u8) ?[]const u8 {
    const arg = args[i.*];
    // Try --key=value form first.
    if (parseKeyValue(arg, key)) |v| return v;
    // Try --key value (space-separated) form.
    if (std.mem.eql(u8, arg, key)) {
        if (i.* + 1 < args.len) {
            i.* += 1;
            return args[i.*];
        }
        eprint("Error: {s} requires a value\n", .{key});
        eprint("Run 'agave-bench --help' for more information.\n", .{});
        std.process.exit(1);
    }
    return null;
}

fn parseKernelName(name: []const u8) ?Kernel {
    return std.meta.stringToEnum(Kernel, name);
}

fn parseBackendName(name: []const u8) ?BackendChoice {
    return std.meta.stringToEnum(BackendChoice, name);
}

fn printUsage() void {
    const usage =
        \\agave-bench — per-kernel and end-to-end micro-benchmark
        \\
        \\USAGE:
        \\  agave-bench <kernel> [OPTIONS]
        \\  agave-bench e2e --model=<path> [OPTIONS]
        \\
        \\MODES:
        \\  <kernel>       Run a single kernel benchmark with synthetic data
        \\  e2e            Load a model and run end-to-end inference timing
        \\
        \\KERNELS:
        \\  gemv_f32  gemv_bf16  gemv_f16  gemv_q8_0  gemv_q4_0
        \\  rms_norm  silu  gelu  softmax  l2_norm  add  mul  rope
        \\  sdpa  sdpa_turbo4  sdpa_turbo3  sdpa_turbo2
        \\
        \\OPTIONS:
        \\  -h, --help       Show this help message
        \\  -v, --version    Print version
        \\  -n, --n <N>      Kernel: output dimension [default: 4096]
        \\                   E2E: tokens to generate [default: 10]
        \\  -k, --k <K>      Input dimension for GEMV [default: 4096]
        \\  --iters <N>      Number of timed iterations [default: 100, max: 1000]
        \\  --backend <X>    Compute backend: auto, cpu, metal, vulkan, cuda, rocm [default: cpu]
        \\  --model <PATH>   Model file or directory (required for e2e mode)
        \\
        \\EXAMPLES:
        \\  agave-bench gemv_f32 --n 4096 --k 4096 --iters 50
        \\  agave-bench rms_norm --n=4096 --backend=metal
        \\  agave-bench e2e --model model.gguf --backend cpu --n 10
        \\
    ;
    fdWriteAll(stdout_file.handle, usage);
}

// ── Timing utilities ─────────────────────────────────────────────

/// Monotonic nanosecond timer using posix clock_gettime (replaces std.time.Timer
/// which was removed in Zig 0.16).
const NanoTimer = struct {
    start_ts: std.posix.timespec,

    fn start() NanoTimer {
        var ts: std.posix.timespec = undefined;
        _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
        return .{ .start_ts = ts };
    }

    fn read(self: *NanoTimer) u64 {
        var now_ts: std.posix.timespec = undefined;
        _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &now_ts);
        const start_ns: i128 = @as(i128, self.start_ts.sec) * 1_000_000_000 + self.start_ts.nsec;
        const now_ns: i128 = @as(i128, now_ts.sec) * 1_000_000_000 + now_ts.nsec;
        return @intCast(now_ns - start_ns);
    }

    fn reset(self: *NanoTimer) void {
        _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &self.start_ts);
    }
};

/// Collects timing samples for a kernel invocation, returning the median in nanoseconds.
/// Runs warmup_iters untimed iterations, then `iters` timed iterations with per-iteration
/// timing via NanoTimer. Returns median of the sorted sample array.
fn collectMedian(
    comptime runFn: fn (*const BenchCtx) void,
    ctx: *const BenchCtx,
    iters: usize,
) u64 {
    // Warmup
    for (0..warmup_iters) |_| {
        runFn(ctx);
    }

    // Collect samples
    var samples: [max_samples]u64 = undefined;
    const n = @min(iters, max_samples);
    var timer = NanoTimer.start();

    for (0..n) |i| {
        timer.reset();
        runFn(ctx);
        samples[i] = timer.read();
    }

    // Sort for median
    std.mem.sort(u64, samples[0..n], {}, std.sort.asc(u64));
    return samples[n / 2];
}

/// Context for benchmark invocations, holding all pre-allocated buffers and parameters.
const BenchCtx = struct {
    be: Backend,
    // Input/output buffers (f32 slices)
    x: []f32,
    y: []f32 = undefined,
    norm_weight: ?[]f32 = null,
    norm_out: ?[]f32 = null,
    // SDPA buffers
    q: ?[]f32 = null,
    keys: ?[]u8 = null,
    values: ?[]u8 = null,
    k_new: ?[]f32 = null,
    v_new: ?[]f32 = null,
    sdpa_out: ?[]f32 = null,
    // KV cache types for turbo SDPA benchmarks
    kv_type_k: KvQuantType = .f32,
    kv_type_v: KvQuantType = .f32,
    // Dimensions
    n: usize,
    k: usize = 0,
    n_heads: usize = 0,
    head_dim: usize = 0,
    seq_len: usize = 0,
    scale: f32 = 0,
    // Tensor data for GEMV dispatch
    td: TensorData = undefined,
};

// ── Kernel runner functions ──────────────────────────────────────

fn runGemv(ctx: *const BenchCtx) void {
    ctx.be.gemv(ctx.x.ptr, ctx.td, ctx.y.ptr, ctx.n, ctx.k);
    ctx.be.sync();
}

fn runRmsNorm(ctx: *const BenchCtx) void {
    ctx.be.rmsNorm(ctx.x.ptr, ctx.norm_weight.?.ptr, ctx.norm_out.?.ptr, ctx.n, rms_norm_eps);
    ctx.be.sync();
}

fn runSilu(ctx: *const BenchCtx) void {
    ctx.be.silu(ctx.x.ptr, ctx.y.ptr, ctx.n);
    ctx.be.sync();
}

fn runGelu(ctx: *const BenchCtx) void {
    ctx.be.gelu(ctx.x.ptr, ctx.y.ptr, ctx.n);
    ctx.be.sync();
}

fn runSoftmax(ctx: *const BenchCtx) void {
    ctx.be.softmax(ctx.x.ptr, ctx.n);
    ctx.be.sync();
}

fn runL2Norm(ctx: *const BenchCtx) void {
    ctx.be.l2Norm(ctx.x.ptr, ctx.n, l2_norm_eps);
    ctx.be.sync();
}

fn runAdd(ctx: *const BenchCtx) void {
    ctx.be.add(ctx.x.ptr, ctx.y.ptr, ctx.norm_out.?.ptr, ctx.n);
    ctx.be.sync();
}

fn runMul(ctx: *const BenchCtx) void {
    ctx.be.mul(ctx.x.ptr, ctx.y.ptr, ctx.norm_out.?.ptr, ctx.n);
    ctx.be.sync();
}

fn runRope(ctx: *const BenchCtx) void {
    ctx.be.rope(ctx.x.ptr, default_rope_pos, ctx.n_heads, ctx.head_dim, ctx.head_dim, default_rope_theta);
    ctx.be.sync();
}

fn runSdpa(ctx: *const BenchCtx) void {
    ctx.be.sdpa(
        ctx.q.?.ptr,
        ctx.keys.?,
        ctx.values.?,
        ctx.k_new.?.ptr,
        ctx.v_new.?.ptr,
        ctx.sdpa_out.?.ptr,
        ctx.n_heads,
        ctx.n_heads, // nkv = n_heads (no GQA for benchmark simplicity)
        ctx.head_dim,
        ctx.seq_len,
        ctx.scale,
        ctx.kv_type_k,
        ctx.kv_type_v,
    );
    ctx.be.sync();
}

// ── Data construction helpers ────────────────────────────────────

/// Fills a f32 slice with synthetic data: v[i] = (i % mod) * scale + offset.
fn fillSyntheticF32(buf: []f32, mod: usize, scale: f32, offset: f32) void {
    for (buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % mod)) * scale + offset;
    }
}

/// Constructs a BF16 weight buffer from synthetic f32 values.
fn fillSyntheticBf16(buf: []u16) void {
    for (buf, 0..) |*v, i| {
        const f: f32 = @as(f32, @floatFromInt(i % synthetic_w_mod)) * synthetic_w_scale + synthetic_w_offset;
        v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
    }
}

/// Constructs an F16 weight buffer from synthetic f32 values.
fn fillSyntheticF16(buf: []f16) void {
    for (buf, 0..) |*v, i| {
        v.* = @floatCast(@as(f32, @floatFromInt(i % synthetic_w_mod)) * synthetic_w_scale + synthetic_w_offset);
    }
}

/// Constructs a Q8_0 weight buffer with synthetic scale + data bytes.
fn fillSyntheticQ8_0(buf: []u8, n_rows: usize, k: usize) void {
    const nb = (k + quant_group_size - 1) / quant_group_size;
    for (buf, 0..) |*v, i| v.* = @truncate(i % 256);
    for (0..n_rows * nb) |blk| {
        buf[blk * q8_0_block_bytes] = synthetic_scale_byte_0;
        buf[blk * q8_0_block_bytes + 1] = synthetic_scale_byte_1;
    }
}

/// Constructs a Q4_0 weight buffer with synthetic scale + nibble data.
fn fillSyntheticQ4_0(buf: []u8, n_rows: usize, k: usize) void {
    const nb = (k + quant_group_size - 1) / quant_group_size;
    for (buf, 0..) |*v, i| v.* = @truncate(i % 256);
    for (0..n_rows * nb) |blk| {
        buf[blk * q4_0_block_bytes] = synthetic_scale_byte_0;
        buf[blk * q4_0_block_bytes + 1] = synthetic_scale_byte_1;
    }
}

// ── Metrics computation ──────────────────────────────────────────

/// Computes bandwidth in GB/s given total bytes transferred and median nanoseconds.
fn computeGbps(total_bytes: usize, median_ns: u64) f64 {
    if (median_ns == 0) return 0.0;
    return @as(f64, @floatFromInt(total_bytes)) / @as(f64, @floatFromInt(median_ns));
}

/// Computes GFLOP/s given total flops and median nanoseconds.
fn computeGflops(total_flops: usize, median_ns: u64) f64 {
    if (median_ns == 0) return 0.0;
    return @as(f64, @floatFromInt(total_flops)) / @as(f64, @floatFromInt(median_ns));
}

// ── JSON output ──────────────────────────────────────────────────

/// Writes a single JSON result line to stdout.
fn emitJson(kernel_name: []const u8, be_name: []const u8, median_ns: u64, gb_s: f64, gflop_s: f64, iters: usize) void {
    print(
        "{{\"mode\":\"kernel\",\"kernel\":\"{s}\",\"backend\":\"{s}\",\"ns_median\":{d},\"gb_s\":{d:.1},\"gflop_s\":{d:.1},\"iters\":{d}}}\n",
        .{ kernel_name, be_name, median_ns, gb_s, gflop_s, iters },
    );
}

// ── Kernel dispatch ──────────────────────────────────────────────

/// Runs the requested kernel benchmark and emits JSON output.
fn benchKernel(kernel: Kernel, be: Backend, be_name: []const u8, n: usize, k: usize, iters: usize) void {
    // page_allocator used intentionally: benchmark buffers need page alignment
    // for zero-copy GPU wrapping, and this is a one-time allocation, not hot path.
    const page = std.heap.page_allocator;

    switch (kernel) {
        .gemv_f32 => {
            const x = page.alloc(f32, k) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);
            const w = page.alloc(f32, n * k) catch return;
            defer page.free(w);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            fillSyntheticF32(w, synthetic_w_mod, synthetic_w_scale, synthetic_w_offset);

            const td = TensorData{ .data = @ptrCast(w.ptr), .dtype = .f32 };
            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .y = y,
                .n = n,
                .k = k,
                .td = td,
            };
            const median_ns = collectMedian(runGemv, &ctx, iters);
            // GEMV: read w[n*k] + x[k], write y[n]. Flops: ~2*n*k (k muls + k-1 adds per row).
            const total_bytes = n * k * @sizeOf(f32) + k * @sizeOf(f32) + n * @sizeOf(f32);
            const total_flops = 2 * n * k;
            emitJson("gemv_f32", be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(total_flops, median_ns), iters);
        },

        .gemv_bf16 => {
            const x = page.alloc(f32, k) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);
            const w = page.alloc(u16, n * k) catch return;
            defer page.free(w);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            fillSyntheticBf16(w);

            const td = TensorData{ .data = @ptrCast(w.ptr), .dtype = .bf16 };
            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .y = y,
                .n = n,
                .k = k,
                .td = td,
            };
            const median_ns = collectMedian(runGemv, &ctx, iters);
            const total_bytes = n * k * @sizeOf(u16) + k * @sizeOf(f32) + n * @sizeOf(f32);
            const total_flops = 2 * n * k;
            emitJson("gemv_bf16", be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(total_flops, median_ns), iters);
        },

        .gemv_f16 => {
            const x = page.alloc(f32, k) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);
            const w = page.alloc(f16, n * k) catch return;
            defer page.free(w);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            fillSyntheticF16(w);

            const td = TensorData{ .data = @ptrCast(w.ptr), .dtype = .f16 };
            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .y = y,
                .n = n,
                .k = k,
                .td = td,
            };
            const median_ns = collectMedian(runGemv, &ctx, iters);
            const total_bytes = n * k * @sizeOf(f16) + k * @sizeOf(f32) + n * @sizeOf(f32);
            const total_flops = 2 * n * k;
            emitJson("gemv_f16", be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(total_flops, median_ns), iters);
        },

        .gemv_q8_0 => {
            const x = page.alloc(f32, k) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);
            const nb = (k + quant_group_size - 1) / quant_group_size;
            const row_bytes = nb * q8_0_block_bytes;
            const total_w = n * row_bytes;
            const w = page.alloc(u8, total_w) catch return;
            defer page.free(w);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            fillSyntheticQ8_0(w, n, k);

            const td = TensorData{ .data = w.ptr, .dtype = .q8_0 };
            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .y = y,

                .n = n,
                .k = k,
                .td = td,
            };
            const median_ns = collectMedian(runGemv, &ctx, iters);
            const total_bytes = total_w + k * @sizeOf(f32) + n * @sizeOf(f32);
            const total_flops = 2 * n * k;
            emitJson("gemv_q8_0", be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(total_flops, median_ns), iters);
        },

        .gemv_q4_0 => {
            const x = page.alloc(f32, k) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);
            const nb = (k + quant_group_size - 1) / quant_group_size;
            const row_bytes = nb * q4_0_block_bytes;
            const total_w = n * row_bytes;
            const w = page.alloc(u8, total_w) catch return;
            defer page.free(w);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            fillSyntheticQ4_0(w, n, k);

            const td = TensorData{ .data = w.ptr, .dtype = .q4_0 };
            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .y = y,

                .n = n,
                .k = k,
                .td = td,
            };
            const median_ns = collectMedian(runGemv, &ctx, iters);
            const total_bytes = total_w + k * @sizeOf(f32) + n * @sizeOf(f32);
            const total_flops = 2 * n * k;
            emitJson("gemv_q4_0", be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(total_flops, median_ns), iters);
        },

        .rms_norm => {
            const x = page.alloc(f32, n) catch return;
            defer page.free(x);
            const w_norm = page.alloc(f32, n) catch return;
            defer page.free(w_norm);
            const out = page.alloc(f32, n) catch return;
            defer page.free(out);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            for (w_norm) |*v| v.* = 1.0;

            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .norm_weight = w_norm,
                .norm_out = out,
                .n = n,
            };
            const median_ns = collectMedian(runRmsNorm, &ctx, iters);
            // Read input[n] + weight[n], write output[n]
            const total_bytes = 3 * n * @sizeOf(f32);
            // Flops: square(n) + sum(n) + rsqrt(1) + mul(n) + mul(n) ~ 4n
            const total_flops = 4 * n;
            emitJson("rms_norm", be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(total_flops, median_ns), iters);
        },

        .silu => {
            const x = page.alloc(f32, n) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);

            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .y = y,
                .n = n,
            };
            const median_ns = collectMedian(runSilu, &ctx, iters);
            // Read input[n], write output[n]
            const total_bytes = 2 * n * @sizeOf(f32);
            // Flops: exp(n) + add(n) + div(n) + mul(n) ~ 4n
            const total_flops = 4 * n;
            emitJson("silu", be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(total_flops, median_ns), iters);
        },

        .gelu => {
            const x = page.alloc(f32, n) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);

            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .y = y,
                .n = n,
            };
            const median_ns = collectMedian(runGelu, &ctx, iters);
            const total_bytes = 2 * n * @sizeOf(f32);
            const total_flops = 8 * n; // tanh + cube + several multiplies
            emitJson("gelu", be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(total_flops, median_ns), iters);
        },

        .softmax => {
            const x = page.alloc(f32, n) catch return;
            defer page.free(x);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);

            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .n = n,
            };
            const median_ns = collectMedian(runSoftmax, &ctx, iters);
            // In-place: read + write n elements. 3 passes (max, exp-sum, div).
            const total_bytes = 2 * n * @sizeOf(f32);
            const total_flops = 5 * n; // max + exp + sum + sub + div
            emitJson("softmax", be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(total_flops, median_ns), iters);
        },

        .l2_norm => {
            const x = page.alloc(f32, n) catch return;
            defer page.free(x);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);

            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .n = n,
            };
            const median_ns = collectMedian(runL2Norm, &ctx, iters);
            // In-place: read + write n elements
            const total_bytes = 2 * n * @sizeOf(f32);
            const total_flops = 3 * n; // square + sum + div
            emitJson("l2_norm", be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(total_flops, median_ns), iters);
        },

        .add => {
            const x = page.alloc(f32, n) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);
            const out = page.alloc(f32, n) catch return;
            defer page.free(out);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            fillSyntheticF32(y, synthetic_w_mod, synthetic_w_scale, synthetic_w_offset);

            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .y = y,
                .norm_out = out,
                .n = n,
            };
            const median_ns = collectMedian(runAdd, &ctx, iters);
            // Read a[n] + b[n], write out[n]
            const total_bytes = 3 * n * @sizeOf(f32);
            const total_flops = n;
            emitJson("add", be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(total_flops, median_ns), iters);
        },

        .mul => {
            const x = page.alloc(f32, n) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);
            const out = page.alloc(f32, n) catch return;
            defer page.free(out);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            fillSyntheticF32(y, synthetic_w_mod, synthetic_w_scale, synthetic_w_offset);

            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .y = y,
                .norm_out = out,
                .n = n,
            };
            const median_ns = collectMedian(runMul, &ctx, iters);
            const total_bytes = 3 * n * @sizeOf(f32);
            const total_flops = n;
            emitJson("mul", be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(total_flops, median_ns), iters);
        },

        .rope => {
            // n = total elements = n_heads * head_dim. Derive head layout.
            const head_dim: usize = if (n >= default_n_heads * 2) n / default_n_heads else default_head_dim;
            const n_heads: usize = if (head_dim > 0) n / head_dim else 1;
            const total = n_heads * head_dim;

            const x = page.alloc(f32, total) catch return;
            defer page.free(x);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);

            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .n = total,
                .n_heads = n_heads,
                .head_dim = head_dim,
            };
            const median_ns = collectMedian(runRope, &ctx, iters);
            // In-place: read + write total elements
            const total_bytes = 2 * total * @sizeOf(f32);
            // Flops: ~4 ops per element (2 rotations × 2 muls); sin/cos precomputed
            const total_flops = 4 * total;
            emitJson("rope", be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(total_flops, median_ns), iters);
        },

        .sdpa, .sdpa_turbo4, .sdpa_turbo3, .sdpa_turbo2 => {
            // For SDPA: --n = n_heads, --k = head_dim. Sequence length from
            // a sensible default or derived from context.
            const n_heads = if (n > 0) n else default_n_heads;
            const head_dim = if (k > 0) k else default_head_dim;
            const seq_len = default_sdpa_seq_len;
            const nkv = n_heads; // no GQA for benchmark simplicity
            const kv_dim = nkv * head_dim;
            const total_q = n_heads * head_dim;
            const total_kv_elems = seq_len * kv_dim;

            // Determine KV cache types
            const kv_type: KvQuantType = switch (kernel) {
                .sdpa_turbo4 => .turbo4,
                .sdpa_turbo3 => .turbo3,
                .sdpa_turbo2 => .turbo2,
                else => .f32,
            };
            const is_turbo = kv_type != .f32;

            const q = page.alloc(f32, total_q) catch return;
            defer page.free(q);
            const k_new = page.alloc(f32, kv_dim) catch return;
            defer page.free(k_new);
            const v_new = page.alloc(f32, kv_dim) catch return;
            defer page.free(v_new);
            const sdpa_out = page.alloc(f32, total_q) catch return;
            defer page.free(sdpa_out);

            fillSyntheticF32(q, sdpa_q_mod, sdpa_q_scale, sdpa_q_offset);
            fillSyntheticF32(k_new, synthetic_x_mod, sdpa_kv_scale, sdpa_k_offset);
            fillSyntheticF32(v_new, sdpa_v_new_mod, sdpa_kv_scale, sdpa_v_offset);

            const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

            // Dummy x/y for BenchCtx (unused by SDPA runner)
            const dummy = page.alloc(f32, 1) catch return;
            defer page.free(dummy);

            // Allocate KV buffers at this scope so defers outlive the benchmark.
            const kv_bytes = if (is_turbo) kv_quant.kvSliceBytes(kv_type, total_kv_elems) else total_kv_elems * @sizeOf(f32);
            const keys_buf = page.alloc(u8, kv_bytes) catch return;
            defer page.free(keys_buf);
            const values_buf = page.alloc(u8, kv_bytes) catch return;
            defer page.free(values_buf);

            if (is_turbo) {
                @memset(keys_buf, 0);
                @memset(values_buf, 0);

                // Pre-fill KV cache with quantized synthetic data
                const tmp = page.alloc(f32, kv_dim) catch return;
                defer page.free(tmp);
                for (0..seq_len - 1) |pos| {
                    fillSyntheticF32(tmp, sdpa_k_mod + pos, sdpa_kv_scale, sdpa_k_offset);
                    const byte_off = kv_quant.kvByteOffset(kv_type, pos * kv_dim);
                    kv_quant.kvStore(keys_buf.ptr + byte_off, tmp.ptr, kv_dim, kv_type);
                    fillSyntheticF32(tmp, sdpa_v_mod + pos, sdpa_kv_scale, sdpa_v_offset);
                    kv_quant.kvStore(values_buf.ptr + byte_off, tmp.ptr, kv_dim, kv_type);
                }
            } else {
                // f32 path: fill synthetic data via f32 view
                const keys_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, keys_buf));
                const values_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, values_buf));
                fillSyntheticF32(keys_f32, sdpa_k_mod, sdpa_kv_scale, sdpa_k_offset);
                fillSyntheticF32(values_f32, sdpa_v_mod, sdpa_kv_scale, sdpa_v_offset);
            }
            const keys = keys_buf;
            const values = values_buf;

            var ctx = BenchCtx{
                .be = be,
                .x = dummy,
                .q = q,
                .keys = keys,
                .values = values,
                .k_new = k_new,
                .v_new = v_new,
                .sdpa_out = sdpa_out,
                .n = total_q,
                .n_heads = n_heads,
                .head_dim = head_dim,
                .seq_len = seq_len - 1, // backend appends k_new at this pos
                .scale = scale,
                .kv_type_k = kv_type,
                .kv_type_v = kv_type,
            };
            const median_ns = collectMedian(runSdpa, &ctx, iters);

            // Bandwidth: turbo reads fewer bytes from KV cache
            const kv_bytes_total = if (is_turbo)
                2 * kv_quant.kvSliceBytes(kv_type, total_kv_elems)
            else
                2 * total_kv_elems * @sizeOf(f32);
            const total_bytes = 2 * total_q * @sizeOf(f32) + kv_bytes_total;
            // Flops: per head: 2*sl*hd (QK^T) + ~3*sl (softmax: max+exp+norm) + 2*sl*hd (attn@V)
            // Turbo adds WHT overhead: ~5*32 adds per 32-element block for dequant
            const total_flops = n_heads * (4 * seq_len * head_dim + 2 * seq_len);
            const kernel_name = switch (kernel) {
                .sdpa_turbo4 => "sdpa_turbo4",
                .sdpa_turbo3 => "sdpa_turbo3",
                .sdpa_turbo2 => "sdpa_turbo2",
                else => "sdpa",
            };
            emitJson(kernel_name, be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(total_flops, median_ns), iters);
        },
    }
}

// ── E2E benchmark ────────────────────────────────────────────────

/// Runs an end-to-end inference benchmark: loads a model, tokenizes a short
/// prompt, runs prefill + generation, and reports timing as a JSON line.
///
/// Parameters:
///   - allocator: Memory allocator for model/tokenizer init.
///   - cli: Parsed CLI arguments (model_path, backend, n = gen_tokens).
fn runE2e(allocator: std.mem.Allocator, cli: CliArgs) u8 {
    // model_path guaranteed by parseCli validation
    const model_path = cli.model_path.?;
    const gen_tokens = cli.n;

    // ── Load model format ────────────────────────────────────────
    const is_dir = blk: {
        const fd = std.posix.openat(std.posix.AT.FDCWD, model_path, .{ .DIRECTORY = true }, 0) catch break :blk false;
        _ = std.c.close(fd);
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
        st_dir = SafeTensorsDir.open(allocator, model_path) catch |e| {
            eprint("Error: failed to open safetensors dir '{s}': {}\n", .{ model_path, e });
            return 1;
        };
        fmt = st_dir.?.format();
    } else {
        gguf_file = GGUFFile.open(allocator, model_path) catch |e| {
            eprint("Error: failed to open '{s}': {}\n", .{ model_path, e });
            return 1;
        };
        fmt = gguf_file.?.format();
    }

    // ── Detect architecture ──────────────────────────────────────
    const arch_str = fmt.getMetaStr("general.architecture") orelse
        fmt.getMetaStr("model_type") orelse "unknown";
    const name = fmt.getMetaStr("general.name") orelse
        fmt.getMetaStr("model_type") orelse "agave";

    var arch = Arch.detect(arch_str) orelse {
        eprint("Error: unsupported architecture '{s}'\n", .{arch_str});
        return 1;
    };

    // SafeTensors Nemotron Nano variant detection
    if (arch == .nemotron_h and fmt.getTensor("backbone.embeddings.weight") != null) {
        arch = .nemotron_nano;
    }

    if (!arch.isEnabled()) {
        eprint("Error: {s} model support disabled at compile time\n", .{arch.displayName()});
        return 1;
    }

    // ── Detect quantization ──────────────────────────────────────
    const quant = getQuantName(fmt);

    // ── Initialize backend ───────────────────────────────────────
    var bs = BackendState{};
    var threaded = std.Io.Threaded.init(allocator, .{}); bs.init(allocator, cli.backend, threaded.io());
    defer if (bs.pool) |*p| p.deinit();
    const be = bs.be;
    const be_name = bs.name;

    // ── Load tokenizer ───────────────────────────────────────────
    var tok = BpeTokenizer.init(allocator);
    defer tok.deinit();

    const vocab = fmt.getVocab();
    const merges = fmt.getMerges();
    const tok_kind: TokenizerKind = if (arch == .gemma3 or arch == .gemma4) .spm_no_dummy else if (merges != null) .bpe else .spm;
    const eos_id = fmt.getMetaU32("tokenizer.ggml.eos_token_id") orelse
        fmt.getMetaU32("eos_token_id") orelse
        if (arch == .gemma3 or arch == .gemma4) gemma_fallback_eos else default_fallback_eos;
    const bos_id = fmt.getMetaU32("tokenizer.ggml.bos_token_id") orelse
        fmt.getMetaU32("bos_token_id") orelse default_bos_id;

    if (vocab) |v| {
        switch (tok_kind) {
            .spm, .spm_no_dummy => tok.loadFromGGUFSpm(v, eos_id) catch |e| {
                eprint("Error: failed to load tokenizer: {}\n", .{e});
                return 1;
            },
            .bpe => tok.loadFromGGUF(v, merges.?, eos_id) catch |e| {
                eprint("Error: failed to load tokenizer: {}\n", .{e});
                return 1;
            },
        }
        tok.bos_token_id = bos_id;
        tok.tok_kind = tok_kind;
    } else {
        eprint("Error: no embedded tokenizer found\n", .{});
        return 1;
    }

    // ── Initialize model and run inference ────────────────────────
    runE2eWithArch(allocator, arch, fmt, be, be_name, &tok, tok_kind, eos_id, gen_tokens, name, quant);
    return 0;
}

/// Dispatches to the correct model type based on architecture, initializes
/// the model, and runs the benchmark inference loop.
fn runE2eWithArch(
    allocator: std.mem.Allocator,
    arch: Arch,
    fmt: Format,
    be: Backend,
    be_name: []const u8,
    tok: *BpeTokenizer,
    tok_kind: TokenizerKind,
    eos_id: u32,
    gen_tokens: usize,
    model_name: []const u8,
    quant: []const u8,
) void {
    const ModelStorage = model_mod.ModelStorage;
    var mdl = ModelStorage.initFromArch(arch, allocator, fmt, be, 0, .f32, .f32, 0, 0, null) catch |e| {
        eprint("Error: failed to initialize {s}: {}\n", .{ arch.displayName(), e });
        return;
    };
    defer mdl.deinit();
    mdl.fixBlockAllocator();

    var model_if = mdl.model();
    runE2eInference(&model_if, tok, tok_kind, eos_id, gen_tokens, be_name, model_name, quant, arch);
}

/// Core e2e inference loop: encode prompt, prefill, generate, and emit JSON.
fn runE2eInference(
    mdl: *Model,
    tok: *BpeTokenizer,
    tok_kind: TokenizerKind,
    eos_id: u32,
    gen_tokens: usize,
    be_name: []const u8,
    model_name: []const u8,
    quant: []const u8,
    arch: Arch,
) void {
    const page_alloc = std.heap.page_allocator;

    // Format prompt with chat template
    const template = arch.chatTemplate();
    const formatted = template.format(page_alloc, null, e2e_prompt) catch {
        eprint("Error: failed to format prompt\n", .{});
        return;
    };
    defer page_alloc.free(formatted);

    // Encode prompt (token_ids allocated by tokenizer's internal allocator)
    const token_ids = switch (tok_kind) {
        .spm => tok.encodeSpm(formatted) catch {
            eprint("Error: failed to encode prompt\n", .{});
            return;
        },
        .spm_no_dummy => tok.encodeSpmNoDummy(formatted) catch {
            eprint("Error: failed to encode prompt\n", .{});
            return;
        },
        .bpe => tok.encode(formatted) catch {
            eprint("Error: failed to encode prompt\n", .{});
            return;
        },
    };
    defer tok.allocator.free(token_ids);

    // Send BOS token if required
    if (tok.bos_token_id > 0) {
        _ = mdl.forward(tok.bos_token_id) catch {
            eprint("Error: BOS forward failed\n", .{});
            return;
        };
    }

    // ── Prefill (timed) ──────────────────────────────────────────
    var prefill_timer = NanoTimer.start();
    var first_gen_token: u32 = 0;
    for (token_ids) |tid| {
        first_gen_token = mdl.forward(tid) catch |e| {
            eprint("Error: prefill failed: {}\n", .{e});
            return;
        };
    }
    const prefill_ns = prefill_timer.read();

    // ── Generation (timed) ───────────────────────────────────────
    var gen_timer = NanoTimer.start();
    var last = first_gen_token;
    var token_count: u32 = 0;
    var prev_token: u32 = 0;
    var repeat_count: u32 = 0;

    // Count first token from prefill if not EOG
    if (token_ids.len > 0 and last != eos_id) {
        token_count = 1;
        prev_token = last;
        repeat_count = 1;
    }

    for (0..gen_tokens -| 1) |_| {
        if (token_ids.len == 0 or last == eos_id) break;
        const next = mdl.forward(last) catch break;
        if (next == eos_id) break;
        last = next;
        token_count += 1;

        if (next == prev_token) {
            repeat_count += 1;
            if (repeat_count >= e2e_repeat_halt_threshold) break;
        } else {
            repeat_count = 1;
            prev_token = next;
        }
    }
    const gen_ns = gen_timer.read();

    // ── Emit JSON result ─────────────────────────────────────────
    const prefill_ms_f = @as(f64, @floatFromInt(prefill_ns)) / 1e6;
    const gen_ms_f = @as(f64, @floatFromInt(gen_ns)) / 1e6;
    const tok_per_sec: f64 = if (gen_ns > 0)
        @as(f64, @floatFromInt(token_count)) / (@as(f64, @floatFromInt(gen_ns)) / 1e9)
    else
        0.0;

    print(
        "{{\"mode\":\"e2e\",\"model\":\"{s}\",\"quant\":\"{s}\",\"backend\":\"{s}\",\"tok_per_sec\":{d:.1},\"prefill_ms\":{d:.0},\"gen_ms\":{d:.0},\"tokens\":{d},\"prefill_tokens\":{d}}}\n",
        .{ model_name, quant, be_name, tok_per_sec, prefill_ms_f, gen_ms_f, token_count, token_ids.len },
    );
}

const getQuantName = Format.getQuantName;

// ── Entry point ──────────────────────────────────────────────────

pub fn main(init: std.process.Init.Minimal) u8 {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cli = parseCli(init.args) orelse return 0;

    // E2E mode: load model and run end-to-end inference benchmark
    if (cli.mode == .e2e) {
        return runE2e(allocator, cli);
    }

    // Initialize backend
    var bs = BackendState{};
    var threaded = std.Io.Threaded.init(allocator, .{}); bs.init(allocator, cli.backend, threaded.io());
    defer if (bs.pool) |*p| p.deinit();
    const be = bs.be;
    const be_name = bs.name;

    // Dispatch to kernel benchmark
    benchKernel(cli.kernel.?, be, be_name, cli.n, cli.k, cli.iters);
    return 0;
}
