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

const warmup_iters: usize = 10;
const default_iters: usize = 100;
const max_samples: usize = 1000;
const default_dim: usize = 4096;
const default_k: usize = 4096;
const synthetic_x_mod: usize = 17;
const synthetic_w_mod: usize = 31;
const default_n_heads: usize = 32;
const default_head_dim: usize = 128;
/// Tokens a batched-op benchmark uses. The failure mode these exist to catch
/// (any token past the first reading stale data) appears at 2.
const batched_n_tok: usize = 4;
/// Tokens in the sdpa_prefill fixture. Larger than `batched_n_tok` so the last
/// token attends over 15 earlier positions, all written during the same call.
const sdpa_prefill_n_tok: usize = 16;
/// Scale for the addScaled fixture; any finite non-unit value exercises it.
const add_scaled_factor: f32 = 0.75;
/// mRoPE position components; distinct so a kernel that ignores one is caught.
const mrope_t_pos: usize = 5;
const mrope_h_pos: usize = 3;
const mrope_w_pos: usize = 7;
/// Row the embedding fixture reads; any in-range id works.
const emb_lookup_token: u32 = 11;
const default_rope_theta: f32 = 10000.0;
const rms_norm_eps: f32 = 1e-6;
const l2_norm_eps: f32 = 1e-6;
const synthetic_x_scale: f32 = 0.01;
const synthetic_x_offset: f32 = -0.08;
const synthetic_w_scale: f32 = 0.001;
const synthetic_w_offset: f32 = -0.015;
const output_buf_size: usize = 4096;
const q8_0_block_bytes = backend_mod.q8_0_block_bytes;
const q4_k_block_bytes = backend_mod.q4_k_block_bytes;
const q4_0_block_bytes = backend_mod.q4_0_block_bytes;
const q5_0_block_bytes = backend_mod.q5_0_block_bytes;
const q6_k_block_bytes = backend_mod.q6_k_block_bytes;
const q4_1_block_bytes = backend_mod.q4_1_block_bytes;
const q2_k_block_bytes = backend_mod.q2_k_block_bytes;
const q3_k_block_bytes = backend_mod.q3_k_block_bytes;
const q5_k_block_bytes = backend_mod.q5_k_block_bytes;
const iq4_nl_block_bytes = backend_mod.iq4_nl_block_bytes;
const iq4_xs_block_bytes = backend_mod.iq4_xs_block_bytes;
const tq1_0_block_bytes = backend_mod.tq1_0_block_bytes;
const tq2_0_block_bytes = backend_mod.tq2_0_block_bytes;
const quant_group_size = backend_mod.quant_block_elems;
/// Little-endian f16 ≈ 0.00875.
const synthetic_scale_byte_0: u8 = 0x1E;
const synthetic_scale_byte_1: u8 = 0x21;
const default_sdpa_seq_len: usize = 512;
const default_rope_pos: usize = 42;
const sdpa_q_mod: usize = 13;
const sdpa_q_scale: f32 = 0.01;
const sdpa_q_offset: f32 = -0.06;
const sdpa_k_mod: usize = 19;
const sdpa_kv_scale: f32 = 0.005;
const sdpa_k_offset: f32 = -0.04;
const sdpa_v_mod: usize = 23;
const sdpa_v_offset: f32 = -0.05;
const sdpa_v_new_mod: usize = 29;
const default_gen_tokens: usize = 10;
const e2e_prompt = "What is 2+2?";
const e2e_repeat_halt_threshold: u32 = 6;

// ── Output helpers ───────────────────────────────────────────────

const stdout_file = std.Io.File.stdout();
const stderr_file = std.Io.File.stderr();

/// Writes all bytes to a file descriptor using raw posix write.
fn fdWriteAll(fd: std.posix.fd_t, bytes: []const u8) void {
    var written: usize = 0;
    while (written < bytes.len) {
        const result = std.posix.system.write(fd, bytes[written..].ptr, bytes[written..].len);
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
    gemv_q4_k,
    gemv_q4_0,
    gemv_q5_0,
    gemv_q6_k,
    gemv_q4_1,
    gemv_q2_k,
    gemv_q3_k,
    gemv_q5_k,
    gemv_iq4_nl,
    gemv_iq4_xs,
    gemv_tq1_0,
    gemv_tq2_0,
    gemv_fp8_e4m3,
    gemv_fp8_e5m2,
    gemm_q8_0,
    rms_norm_batched,
    rope_batched,
    rms_norm_multi,
    add_aliased,
    silu_mul_aliased,
    deinterleave,
    split_q_gate,
    add_rms_norm,
    rms_norm_add,
    sigmoid_mul,
    gelu_mul,
    clamped_silu_mul,
    add_scaled,
    gemv_multi,
    gemv_t,
    rope_mrope,
    emb_lookup,
    all_reduce_add,
    sdpa_prefill,
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
    /// Evict the weight before each GEMV iteration so the transfer is timed too.
    reupload: bool = false,
    /// Re-run the kernel on the CPU backend and report the largest relative
    /// difference, so a GPU kernel that is merely fast can still be caught.
    validate: bool = false,
    backend: BackendChoice,
    model_path: ?[]const u8,
};

/// Parses CLI arguments from process args.
/// Returns null on help/version (caller exits 0). Parse errors call
/// `std.process.exit(2)` (usage error).
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
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "help")) {
            printUsage();
            return null; // exit 0
        }
        if (std.mem.eql(u8, arg, "--version") or std.mem.eql(u8, arg, "-v")) {
            print("agave-bench " ++ display_mod.version ++ "\n", .{});
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
            eprintKernelList();
            eprint("Run 'agave-bench --help' for more information.\n", .{});
            std.process.exit(2);
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
                std.process.exit(2);
            };
            n_was_set = true;
        } else if (getArgValue(args_slice, &i, "--k") orelse getArgValue(args_slice, &i, "-k")) |v| {
            result.k = std.fmt.parseInt(usize, v, 10) catch {
                eprint("Error: invalid value for --k: '{s}'\n", .{v});
                eprint("Run 'agave-bench --help' for more information.\n", .{});
                std.process.exit(2);
            };
        } else if (getArgValue(args_slice, &i, "--iters")) |v| {
            result.iters = std.fmt.parseInt(usize, v, 10) catch {
                eprint("Error: invalid value for --iters: '{s}'\n", .{v});
                eprint("Run 'agave-bench --help' for more information.\n", .{});
                std.process.exit(2);
            };
            if (result.iters > max_samples) {
                eprint("Warning: clamping iters to {d}\n", .{max_samples});
                result.iters = max_samples;
            }
        } else if (std.mem.eql(u8, args_slice[i], "--reupload")) {
            result.reupload = true;
        } else if (std.mem.eql(u8, args_slice[i], "--validate")) {
            result.validate = true;
        } else if (getArgValue(args_slice, &i, "--backend")) |v| {
            result.backend = parseBackendName(v) orelse {
                eprint("Error: unknown backend '{s}'\n", .{v});
                eprint("  Valid options: auto, cpu, metal, vulkan, cuda, rocm, webgpu\n", .{});
                eprint("Run 'agave-bench --help' for more information.\n", .{});
                std.process.exit(2);
            };
        } else if (getArgValue(args_slice, &i, "--model")) |v| {
            result.model_path = v;
        } else {
            eprint("Error: unknown argument '{s}'\n", .{args_slice[i]});
            eprint("Run 'agave-bench --help' for more information.\n", .{});
            std.process.exit(2);
        }
    }

    if (result.mode == .kernel and result.kernel == null) {
        eprint("Error: kernel name required\n", .{});
        printUsage();
        std.process.exit(2);
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
        std.process.exit(2);
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
        std.process.exit(2);
    }
    return null;
}

fn parseKernelName(name: []const u8) ?Kernel {
    return std.meta.stringToEnum(Kernel, name);
}

fn parseBackendName(name: []const u8) ?BackendChoice {
    return std.meta.stringToEnum(BackendChoice, name);
}

/// Space-separated Kernel enum names. Help and unknown-kernel errors share this
/// so the printed list cannot drift from what `parseKernelName` accepts.
const kernel_names_joined = blk: {
    var acc: []const u8 = "";
    for (@typeInfo(Kernel).@"enum".fields) |f| {
        if (acc.len > 0) acc = acc ++ " ";
        acc = acc ++ f.name;
    }
    break :blk acc;
};

/// Wrap column for the kernel name list in --help and unknown-kernel errors.
const kernel_list_wrap_width: usize = 72;

/// Word-wrap `words` (space-separated) into `out`, prefixing each line with `indent`.
fn wrapWords(out: []u8, words: []const u8, indent: []const u8, width: usize) []const u8 {
    var pos: usize = 0;
    const write = struct {
        fn bytes(buf: []u8, p: *usize, s: []const u8) void {
            const n = @min(s.len, buf.len - p.*);
            if (n == 0) return;
            @memcpy(buf[p.*..][0..n], s[0..n]);
            p.* += n;
        }
    }.bytes;

    var col: usize = 0;
    var it = std.mem.tokenizeScalar(u8, words, ' ');
    var first = true;
    while (it.next()) |w| {
        if (first) {
            write(out, &pos, indent);
            col = indent.len;
            first = false;
        } else if (col + 1 + w.len > width) {
            write(out, &pos, "\n");
            write(out, &pos, indent);
            col = indent.len;
        } else {
            write(out, &pos, " ");
            col += 1;
        }
        write(out, &pos, w);
        col += w.len;
    }
    return out[0..pos];
}

fn eprintKernelList() void {
    var buf: [2048]u8 = undefined;
    const list = wrapWords(&buf, kernel_names_joined, "    ", kernel_list_wrap_width);
    eprint("  Valid kernels:\n{s}\n", .{list});
}

fn printUsage() void {
    const usage_head =
        \\agave-bench, per-kernel and end-to-end micro-benchmark
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
        \\
    ;
    const usage_tail =
        \\
        \\OPTIONS:
        \\  -h, --help       Show this help message and exit
        \\  -v, --version    Print version and exit
        \\  -n, --n <N>      Kernel: output dimension [default: 4096]
        \\                   E2E: tokens to generate [default: 10]
        \\  -k, --k <K>      Input dimension for GEMV [default: 4096]
        \\  --iters <N>      Number of timed iterations [default: 100, max: 1000]
        \\  --validate       Re-run the kernel on the CPU backend with the same inputs
        \\                   and report the largest relative difference. Exits non-zero
        \\                   when it exceeds the tolerance, so CI can gate on it
        \\  --reupload       GEMV only: evict the weight before each iteration so the
        \\                   host-to-device transfer is timed too; the difference
        \\                   against a plain run is the upload cost
        \\  --backend <X>    Compute backend: auto, cpu, metal, vulkan, cuda, rocm, webgpu [default: cpu]
        \\  --model <PATH>   Model file or directory (required for e2e mode)
        \\
        \\EXAMPLES:
        \\  agave-bench gemv_f32 --n 4096 --k 4096 --iters 50
        \\  agave-bench rms_norm --n=4096 --backend=metal
        \\  agave-bench e2e --model model.gguf --backend cpu --n 10
        \\
    ;
    fdWriteAll(stdout_file.handle, usage_head);
    var buf: [2048]u8 = undefined;
    const list = wrapWords(&buf, kernel_names_joined, "  ", kernel_list_wrap_width);
    fdWriteAll(stdout_file.handle, list);
    fdWriteAll(stdout_file.handle, "\n");
    fdWriteAll(stdout_file.handle, usage_tail);
}

// ── Timing utilities ─────────────────────────────────────────────

/// Monotonic nanosecond timer using posix clock_gettime (replaces std.time.Timer
/// which was removed in Zig 0.16).
const NanoTimer = struct {
    start_ts: std.posix.timespec,

    fn start() NanoTimer {
        var ts: std.posix.timespec = undefined;
        _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
        return .{ .start_ts = ts };
    }

    fn read(self: *NanoTimer) u64 {
        var now_ts: std.posix.timespec = undefined;
        _ = std.posix.system.clock_gettime(.MONOTONIC, &now_ts);
        const start_ns: i128 = @as(i128, self.start_ts.sec) * 1_000_000_000 + self.start_ts.nsec;
        const now_ns: i128 = @as(i128, now_ts.sec) * 1_000_000_000 + now_ts.nsec;
        return @intCast(now_ns - start_ns);
    }

    fn reset(self: *NanoTimer) void {
        _ = std.posix.system.clock_gettime(.MONOTONIC, &self.start_ts);
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
    // Batched prefill dimensions
    n_tok: usize = 1,
    n_kv_heads: usize = 0,
    positions: ?[]u32 = null,
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

/// Reference Q4_K row dot on host, structure mirrors the validated TileLang
/// reference implementation (research/kernels/tilelang), checked bit-exact
/// against `gguf.dequantize` on real checkpoint tensors.
fn dequantQ4KRowDotHost(row: []const u8, x: []const f32, out_y: *f32) void {
    const nblk = row.len / 144;
    var acc: f32 = 0;
    const wu: [*]const u32 = @ptrCast(@alignCast(row.ptr));
    var sb: usize = 0;
    while (sb < nblk) : (sb += 1) {
        const blk_u32 = sb * 36;
        const head = wu[blk_u32];
        const d: f32 = @floatCast(@as(f16, @bitCast(@as(u16, @truncate(head)))));
        const dmin: f32 = @floatCast(@as(f16, @bitCast(@as(u16, @truncate(head >> 16)))));
        const BB = sb * 144;
        const scl = row[BB + 4 ..][0..12];
        var ln: usize = 0;
        while (ln < 8) : (ln += 1) {
            var scv: u8 = undefined;
            var mnv: u8 = undefined;
            if (ln < 4) {
                scv = scl[ln] & 63;
                mnv = scl[ln + 4] & 63;
            } else {
                scv = (scl[ln + 4] & 0xF) | ((scl[ln - 4] >> 6) << 4);
                mnv = (scl[ln + 4] >> 4) | ((scl[ln] >> 6) << 4);
            }
            const scf: f32 = @floatFromInt(scv);
            const mnf: f32 = @floatFromInt(mnv);
            const g = ln / 2;
            const sh: u5 = @intCast(4 * (ln % 2));
            const wbase = blk_u32 + 4 + g * 8;
            const xb = sb * 256 + ln * 32;
            var jj: usize = 0;
            while (jj < 8) : (jj += 1) {
                const word = wu[wbase + jj];
                const eb = xb + jj * 4;
                acc += (d * scf * @as(f32, @floatFromInt((word >> sh) & 0xF)) - dmin * mnf) * x[eb];
                acc += (d * scf * @as(f32, @floatFromInt((word >> (sh + 8)) & 0xF)) - dmin * mnf) * x[eb + 1];
                acc += (d * scf * @as(f32, @floatFromInt((word >> (sh + 16)) & 0xF)) - dmin * mnf) * x[eb + 2];
                acc += (d * scf * @as(f32, @floatFromInt((word >> (sh + 24)) & 0xF)) - dmin * mnf) * x[eb + 3];
            }
        }
    }
    out_y.* = acc;
}

// ── Kernel runner functions ──────────────────────────────────────

/// Input refills for `validateVsCpu`. In-place kernels need their input restored
/// between the two runs; out-of-place ones only need `x` to be the same, which it
/// already is, so `refillNone` is correct for them and cheaper.
fn refillNone(_: *BenchCtx) void {}

fn refillX(ctx: *BenchCtx) void {
    fillSyntheticF32(ctx.x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
}

/// addRmsNorm mutates its first argument in place and writes its output, so a
/// second run needs both restored.
fn refillAddRmsNorm(ctx: *BenchCtx) void {
    refillX(ctx);
    refillNormOut(ctx);
}

/// For ops that ACCUMULATE into their output (`rmsNormAdd` does `b[i] += ...`)
/// rather than overwriting it. Refilling only the input would leave the second
/// run adding onto the first run's result.
fn refillNormOut(ctx: *BenchCtx) void {
    fillSyntheticF32(ctx.norm_out.?, synthetic_x_mod + 2, synthetic_x_scale, synthetic_x_offset);
}

/// Largest relative difference tolerated between a backend's kernel and the CPU
/// reference. Quantized GEMV accumulates in a different order on a GPU, so exact
/// equality is not the bar; 2% separates reordering noise from a wrong kernel by
/// a wide margin (a broken kernel is off by whole factors, not percent).
const validate_tolerance: f32 = 0.02;

/// Tracks whether any `--validate` comparison failed, so main can exit non-zero.
var validation_failed: bool = false;

/// Re-run the last kernel on the CPU backend with identical inputs and report the
/// largest relative difference against what the benchmarked backend produced.
///
/// `out` is the slice the kernel writes; it still holds the benchmarked
/// backend's result on entry and is overwritten with the CPU's. Swapping only
/// `ctx.be` guarantees both runs see byte-identical inputs, which a separately
/// constructed reference cannot promise.
fn validateVsCpu(
    comptime runFn: fn (*const BenchCtx) void,
    comptime refillFn: fn (*BenchCtx) void,
    ctx: *BenchCtx,
    out: []f32,
    allocator: std.mem.Allocator,
    io: std.Io,
    kernel_name: []const u8,
    be_name: []const u8,
) void {
    if (std.mem.eql(u8, be_name, "CPU")) return; // comparing CPU to itself proves nothing

    // One clean run per backend from the same refilled inputs, not the state the
    // timed loop left behind: an in-place kernel (softmax, rope, l2Norm) has been
    // applied `iters` times by then, and re-running it once more on the CPU would
    // compare different numbers of applications.
    refillFn(ctx);
    runFn(ctx);
    const gpu = allocator.dupe(f32, out) catch {
        eprint("{{\"validate\":\"{s}\",\"error\":\"out of memory\"}}\n", .{kernel_name});
        return;
    };
    defer allocator.free(gpu);

    var cpu_bs = BackendState{};
    cpu_bs.init(allocator, .cpu, io, 0);
    defer if (cpu_bs.pool) |*p| p.deinit();

    const benched = ctx.be;
    ctx.be = cpu_bs.be;
    refillFn(ctx);
    runFn(ctx);
    ctx.be = benched;

    var max_rel: f32 = 0;
    var worst: usize = 0;
    var n_nonfinite: usize = 0;
    for (out, gpu, 0..) |ref, got, i| {
        if (!std.math.isFinite(got)) {
            n_nonfinite += 1;
            continue;
        }
        const rel = @abs(got - ref) / @max(1.0, @abs(ref));
        if (rel > max_rel) {
            max_rel = rel;
            worst = i;
        }
    }
    const bad = n_nonfinite > 0 or max_rel > validate_tolerance;
    if (bad) validation_failed = true;
    eprint(
        "{{\"validate\":\"{s}\",\"backend\":\"{s}\",\"max_rel_err\":{d:.6},\"worst_index\":{d},\"non_finite\":{d},\"tolerance\":{d:.3},\"pass\":{s}}}\n",
        .{ kernel_name, be_name, max_rel, worst, n_nonfinite, validate_tolerance, if (bad) "false" else "true" },
    );
    if (bad and out.len > worst) {
        eprint("  worst element {d}: backend {d:.6} vs cpu {d:.6}\n", .{ worst, gpu[worst], out[worst] });
    }
}

fn runGemv(ctx: *const BenchCtx) void {
    ctx.be.gemv(ctx.x.ptr, ctx.td, ctx.y.ptr, ctx.n, ctx.k);
    ctx.be.sync();
}

/// Same GEMV, but the weight is evicted from the backend's device cache first,
/// so every iteration re-uploads it. Subtracting `runGemv`'s time leaves the
/// cost a `--vram-budget` eviction pays.
///
/// That cost is NOT the PCIe transfer alone: eviction also frees the device
/// buffer and the re-upload allocates a new one, and on ROCm those driver round
/// trips dominate the bytes moved. Read this as re-upload cost, not link
/// bandwidth; a run with `--vram-budget` set recycles the buffer instead and is
/// the figure that matches real decode.
fn runGemvReupload(ctx: *const BenchCtx) void {
    ctx.be.invalidateWeight(ctx.td.data);
    ctx.be.gemv(ctx.x.ptr, ctx.td, ctx.y.ptr, ctx.n, ctx.k);
    ctx.be.sync();
}

/// Batched prefill ops. These are the paths a multi-token prefill takes and the
/// ones no single-token benchmark reaches, so they get their own validation.
fn runGemm(ctx: *const BenchCtx) void {
    ctx.be.gemm(ctx.x.ptr, ctx.td, ctx.y.ptr, ctx.n_tok, ctx.n, ctx.k);
    ctx.be.sync();
}

fn runRmsNormBatched(ctx: *const BenchCtx) void {
    ctx.be.rmsNormBatched(ctx.x.ptr, ctx.norm_weight.?.ptr, ctx.norm_out.?.ptr, ctx.n_tok, ctx.n, rms_norm_eps);
    ctx.be.sync();
}

fn runRopeBatched(ctx: *const BenchCtx) void {
    ctx.be.ropeBatched(ctx.x.ptr, ctx.positions.?.ptr, ctx.n_tok, ctx.n_heads, ctx.head_dim, ctx.head_dim, default_rope_theta);
    ctx.be.sync();
}

fn runSdpaPrefill(ctx: *const BenchCtx) void {
    ctx.be.sdpaPrefill(
        ctx.q.?.ptr,
        ctx.k_new.?.ptr,
        ctx.v_new.?.ptr,
        ctx.keys.?,
        ctx.values.?,
        ctx.sdpa_out.?.ptr,
        ctx.n_heads,
        ctx.n_kv_heads,
        ctx.head_dim,
        ctx.seq_len,
        ctx.n_tok,
        ctx.scale,
        ctx.kv_type_k,
        ctx.kv_type_v,
    );
    ctx.be.sync();
}

/// Attention output is a convex combination of the value vectors, so a fixture
/// whose values straddle zero can cancel to near-zero and then a tiny difference
/// in the softmax weights flips the sign of a small number. That cannot
/// distinguish a wrong kernel from float noise, which is the one thing a
/// validation fixture has to do, so the values here are strictly positive: the
/// output is then bounded by their range and any real disagreement shows as one.
const sdpa_v_offset_positive: f32 = 0.05;

/// sdpaPrefill appends into the KV cache, so a second run must start from the
/// same cache contents or it attends over the first run's leftovers.
fn refillKvCache(ctx: *BenchCtx) void {
    const keys_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, ctx.keys.?));
    const values_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, ctx.values.?));
    fillSyntheticF32(keys_f32, sdpa_k_mod, sdpa_kv_scale, sdpa_k_offset);
    fillSyntheticF32(values_f32, sdpa_v_mod, sdpa_kv_scale, sdpa_v_offset_positive);
}

/// Per-head RMS norm over n_heads vectors. The prefill path calls this with
/// n_tok * n_heads, so a stride or count bug shows only above one token.
fn runRmsNormMulti(ctx: *const BenchCtx) void {
    ctx.be.rmsNormMulti(ctx.x.ptr, ctx.norm_weight.?.ptr, ctx.n_tok * ctx.n_heads, ctx.head_dim, rms_norm_eps);
    ctx.be.sync();
}

/// add() with the destination aliasing the first source, which is how the
/// batched prefill path writes its residual. The GPU backends cache activations
/// by host address, so a buffer that is simultaneously input and output is the
/// case most likely to confuse that bookkeeping.
fn runAddAliased(ctx: *const BenchCtx) void {
    ctx.be.add(ctx.x.ptr, ctx.y.ptr, ctx.x.ptr, ctx.n);
    ctx.be.sync();
}

/// siluMul() writing back over its gate input, as prefillFeedForward does.
fn runSiluMulAliased(ctx: *const BenchCtx) void {
    ctx.be.siluMul(ctx.x.ptr, ctx.y.ptr, ctx.x.ptr, ctx.n);
    ctx.be.sync();
}

/// Ops the model uses but no benchmark reached. Several are on the batched
/// prefill path, which is exactly where the composition bug lived.
fn runDeinterleave(ctx: *const BenchCtx) void {
    ctx.be.deinterleave(ctx.x.ptr, ctx.y.ptr, ctx.norm_out.?.ptr, ctx.head_dim, ctx.n_heads);
    ctx.be.sync();
}

fn runSplitQGate(ctx: *const BenchCtx) void {
    ctx.be.splitQGate(ctx.x.ptr, ctx.y.ptr, ctx.norm_out.?.ptr, ctx.head_dim, ctx.n_heads);
    ctx.be.sync();
}

fn runAddRmsNorm(ctx: *const BenchCtx) void {
    ctx.be.addRmsNorm(ctx.x.ptr, ctx.y.ptr, ctx.norm_weight.?.ptr, ctx.norm_out.?.ptr, ctx.n, rms_norm_eps);
    ctx.be.sync();
}

fn runRmsNormAdd(ctx: *const BenchCtx) void {
    ctx.be.rmsNormAdd(ctx.x.ptr, ctx.norm_weight.?.ptr, ctx.norm_out.?.ptr, ctx.n, rms_norm_eps);
    ctx.be.sync();
}

fn runSigmoidMul(ctx: *const BenchCtx) void {
    ctx.be.sigmoidMul(ctx.x.ptr, ctx.y.ptr, ctx.n);
    ctx.be.sync();
}

fn runGeluMul(ctx: *const BenchCtx) void {
    ctx.be.geluMul(ctx.x.ptr, ctx.y.ptr, ctx.norm_out.?.ptr, ctx.n);
    ctx.be.sync();
}

fn runClampedSiluMul(ctx: *const BenchCtx) void {
    ctx.be.clampedSiluMul(ctx.x.ptr, ctx.y.ptr, ctx.norm_out.?.ptr, ctx.n);
    ctx.be.sync();
}

fn runAddScaled(ctx: *const BenchCtx) void {
    ctx.be.addScaled(ctx.y.ptr, ctx.x.ptr, add_scaled_factor, ctx.n);
    ctx.be.sync();
}

/// Batched multi-output GEMV: several weight matrices against one activation.
/// The model uses it for fused QKV, so a stride bug here is a wrong projection.
fn runGemvMulti(ctx: *const BenchCtx) void {
    const ops = [_]backend_mod.GemvOp{
        .{ .w = ctx.td, .y = ctx.y.ptr, .n = ctx.n },
        .{ .w = ctx.td, .y = ctx.norm_out.?.ptr, .n = ctx.n },
    };
    ctx.be.gemvMulti(ctx.x.ptr, &ops, ctx.k);
    ctx.be.sync();
}

/// Transposed f32 GEMV (weights stored column-major).
fn runGemvT(ctx: *const BenchCtx) void {
    ctx.be.gemvT(ctx.x.ptr, @ptrCast(ctx.td.data), ctx.y.ptr, ctx.n, ctx.k);
    ctx.be.sync();
}

/// Multimodal RoPE: three position components instead of one.
fn runRopeMrope(ctx: *const BenchCtx) void {
    ctx.be.ropeMrope(ctx.x.ptr, mrope_t_pos, mrope_h_pos, mrope_w_pos, ctx.n_heads, ctx.head_dim, ctx.head_dim, default_rope_theta);
    ctx.be.sync();
}

/// Embedding row read. CLAUDE.md allows this one to run on the CPU inside a GPU
/// backend, so the check is that the two agree, not that it dispatches a kernel.
fn runEmbLookup(ctx: *const BenchCtx) void {
    ctx.be.embLookup(ctx.td, emb_lookup_token, ctx.y.ptr, ctx.n);
    ctx.be.sync();
}

/// Tensor-parallel reduction: dst += src.
fn runAllReduceAdd(ctx: *const BenchCtx) void {
    ctx.be.allReduceAdd(ctx.x.ptr, ctx.y.ptr, ctx.n);
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
fn fillSyntheticQ4_K(buf: []u8, n_rows: usize, k: usize) void {
    // Per 256-elem superblock: [f16 d][f16 dmin][12B 6-bit scales][128B nibbles].
    // Scales get small finite fp16 patterns; nibbles arbitrary nibble values.
    // Round up, matching backend.gemvRowBytes: a k that is not a multiple of
    // 256 still gets a whole final super-block on disk, and a kernel reading it
    // must find allocated bytes there.
    const nb = (k + 255) / 256;
    const row_bytes = nb * q4_k_block_bytes;
    var r: usize = 0;
    while (r < n_rows) : (r += 1) {
        const row = buf[r * row_bytes ..][0..row_bytes];
        @memset(row, 0x11); // nibble payload 0x1 pattern
        var blk: usize = 0;
        while (blk < nb) : (blk += 1) {
            const base = blk * q4_k_block_bytes;
            row[base] = synthetic_scale_byte_0; // f16 d lo
            row[base + 1] = synthetic_scale_byte_1; // f16 d hi
            row[base + 2] = synthetic_scale_byte_1; // f16 dmin lo
            row[base + 3] = synthetic_scale_byte_0; // f16 dmin hi
            // 12 bytes of 6-bit scale/min pairs at +4..+15: keep values <=63
            for (4..16) |i| row[base + i] = 0x24;
        }
    }
}

fn fillSyntheticQ8_0(buf: []u8, n_rows: usize, k: usize) void {
    const nb = (k + quant_group_size - 1) / quant_group_size;
    for (buf, 0..) |*v, i| v.* = @truncate(i % 256);
    for (0..n_rows * nb) |blk| {
        buf[blk * q8_0_block_bytes] = synthetic_scale_byte_0;
        buf[blk * q8_0_block_bytes + 1] = synthetic_scale_byte_1;
    }
}

/// Constructs a Q5_0 weight buffer: [f16 d][4B high-bit plane][16B nibbles] per
/// 32 elements. Only the scale is pinned to a finite fp16; the rest is arbitrary
/// bit patterns, which is what a kernel must handle anyway.
fn fillSyntheticQ5_0(buf: []u8, n_rows: usize, k: usize) void {
    const nb = (k + quant_group_size - 1) / quant_group_size;
    for (buf, 0..) |*v, i| v.* = @truncate(i % 256);
    for (0..n_rows * nb) |blk| {
        buf[blk * q5_0_block_bytes] = synthetic_scale_byte_0;
        buf[blk * q5_0_block_bytes + 1] = synthetic_scale_byte_1;
    }
}

/// Constructs a Q6_K weight buffer: 210 bytes per 256 elements, ending in the
/// f16 super-block scale. Only that scale is pinned to a finite value.
fn fillSyntheticQ6_K(buf: []u8, n_rows: usize, k: usize) void {
    const nb = (k + 255) / 256;
    for (buf, 0..) |*v, i| v.* = @truncate(i % 64); // keep 6-bit scales small
    for (0..n_rows * nb) |blk| {
        const d_off = blk * q6_k_block_bytes + q6_k_block_bytes - 2;
        buf[d_off] = synthetic_scale_byte_0;
        buf[d_off + 1] = synthetic_scale_byte_1;
    }
}

/// Fill a quantized weight buffer with bytes that are valid in EVERY block
/// layout, without encoding any of them.
///
/// Every byte is capped at 63, so a little-endian f16 read from any two adjacent
/// bytes is at most 0x3F3F (about 1.81): always finite, never NaN or Inf,
/// whatever offset a format keeps its scale at. Quantized codes are in range by
/// construction since they are bit fields of the same bytes.
///
/// Validation does not need a well-formed quantization, only that the CPU and
/// the GPU read the SAME bytes the same way. Arbitrary-but-finite content tests
/// that harder than a tidy encoding would.
fn fillSyntheticQuantBytes(buf: []u8) void {
    for (buf, 0..) |*v, i| v.* = @truncate(i % 64);
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
fn benchKernel(kernel: Kernel, be: Backend, be_name: []const u8, n: usize, k: usize, iters: usize, reupload: bool, validate: bool, allocator: std.mem.Allocator, io: std.Io) void {
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
            const median_ns = if (reupload) collectMedian(runGemvReupload, &ctx, iters) else collectMedian(runGemv, &ctx, iters);
            if (validate) validateVsCpu(runGemv, refillNone, &ctx, y, allocator, io, @tagName(kernel), be_name);
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
            const median_ns = if (reupload) collectMedian(runGemvReupload, &ctx, iters) else collectMedian(runGemv, &ctx, iters);
            if (validate) validateVsCpu(runGemv, refillNone, &ctx, y, allocator, io, @tagName(kernel), be_name);
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
            const median_ns = if (reupload) collectMedian(runGemvReupload, &ctx, iters) else collectMedian(runGemv, &ctx, iters);
            if (validate) validateVsCpu(runGemv, refillNone, &ctx, y, allocator, io, @tagName(kernel), be_name);
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
            const median_ns = if (reupload) collectMedian(runGemvReupload, &ctx, iters) else collectMedian(runGemv, &ctx, iters);
            if (validate) validateVsCpu(runGemv, refillNone, &ctx, y, allocator, io, @tagName(kernel), be_name);
            const total_bytes = total_w + k * @sizeOf(f32) + n * @sizeOf(f32);
            const total_flops = 2 * n * k;
            emitJson("gemv_q8_0", be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(total_flops, median_ns), iters);
        },

        .gemv_q4_k => {
            const x = page.alloc(f32, k) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);
            // Round up, matching backend.gemvRowBytes: a k that is not a multiple of
            // 256 still gets a whole final super-block on disk, and a kernel reading it
            // must find allocated bytes there.
            const nb = (k + 255) / 256;
            const row_bytes = nb * q4_k_block_bytes;
            const total_w = n * row_bytes;
            const w = page.alloc(u8, total_w) catch return;
            defer page.free(w);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            fillSyntheticQ4_K(w, n, k);

            const td = TensorData{ .data = w.ptr, .dtype = .q4_k };
            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .y = y,

                .n = n,
                .k = k,
                .td = td,
            };
            const median_ns = if (reupload) collectMedian(runGemvReupload, &ctx, iters) else collectMedian(runGemv, &ctx, iters);
            if (validate) validateVsCpu(runGemv, refillNone, &ctx, y, allocator, io, @tagName(kernel), be_name);
            const total_bytes = total_w + k * @sizeOf(f32) + n * @sizeOf(f32);
            const total_flops = 2 * n * k;
            // Validation: compare first rows against host reference.
            {
                const row_bytes_v = nb * q4_k_block_bytes;
                var v: usize = 0;
                var max_err: f32 = 0;
                const check_rows = @min(n, 8);
                while (v < check_rows) : (v += 1) {
                    var ref_y: f32 = 0;
                    dequantQ4KRowDotHost(w[v * row_bytes_v ..][0..row_bytes_v], x, &ref_y);
                    const got = y[v];
                    const err = @abs(got - ref_y) / @max(1.0, @abs(ref_y));
                    if (err > max_err) max_err = err;
                }
                emitJson("gemv_q4_k", be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(total_flops, median_ns), iters);
                var dbg: usize = 0;
                while (dbg < @min(check_rows, 4)) : (dbg += 1) {
                    var ref_y: f32 = 0;
                    dequantQ4KRowDotHost(w[dbg * row_bytes_v ..][0..row_bytes_v], x, &ref_y);
                    var b1: [64]u8 = undefined;
                    var b2: [64]u8 = undefined;
                    const rs = std.fmt.bufPrint(&b1, "{d:.6}", .{ref_y}) catch "?";
                    const gs = std.fmt.bufPrint(&b2, "{d:.6}", .{y[dbg]}) catch "?";
                    var b3: [160]u8 = undefined;
                    const ln = std.fmt.bufPrint(&b3, "{{\"row\":{d},\"gpu\":{s},\"ref\":{s}}}\n", .{ dbg, gs, rs }) catch "";
                    fdWriteAll(stdout_file.handle, ln);
                }
                var err_buf: [64]u8 = undefined;
                const err_str = std.fmt.bufPrint(&err_buf, "{d}", .{max_err}) catch "?";
                var buf: [128]u8 = undefined;
                const msg = std.fmt.bufPrint(&buf, "{{\"validation\":{{\"rows\":{d},\"max_rel_err\":\"{s}\"}}}}\n", .{ check_rows, err_str }) catch "";
                fdWriteAll(stdout_file.handle, msg);
            }
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
            const median_ns = if (reupload) collectMedian(runGemvReupload, &ctx, iters) else collectMedian(runGemv, &ctx, iters);
            if (validate) validateVsCpu(runGemv, refillNone, &ctx, y, allocator, io, @tagName(kernel), be_name);
            const total_bytes = total_w + k * @sizeOf(f32) + n * @sizeOf(f32);
            const total_flops = 2 * n * k;
            emitJson("gemv_q4_0", be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(total_flops, median_ns), iters);
        },

        // ── Batched prefill ops ──────────────────────────────────
        // n_tok is fixed at a small value: the bug class these catch (a batched
        // op mishandling any token past the first) shows at 2 and does not need
        // a realistic chunk to reproduce.
        .gemv_q5_0 => {
            const x = page.alloc(f32, k) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);
            const nb = (k + quant_group_size - 1) / quant_group_size;
            const row_bytes = nb * q5_0_block_bytes;
            const w = page.alloc(u8, n * row_bytes) catch return;
            defer page.free(w);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            fillSyntheticQ5_0(w, n, k);

            var ctx = BenchCtx{ .be = be, .x = x, .y = y, .n = n, .k = k, .td = TensorData{ .data = w.ptr, .dtype = .q5_0 } };
            const median_ns = if (reupload) collectMedian(runGemvReupload, &ctx, iters) else collectMedian(runGemv, &ctx, iters);
            if (validate) validateVsCpu(runGemv, refillNone, &ctx, y, allocator, io, @tagName(kernel), be_name);
            emitJson("gemv_q5_0", be_name, median_ns, computeGbps(n * row_bytes, median_ns), computeGflops(2 * n * k, median_ns), iters);
        },

        .gemv_q6_k => {
            const x = page.alloc(f32, k) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);
            const nb = (k + 255) / 256;
            const row_bytes = nb * q6_k_block_bytes;
            const w = page.alloc(u8, n * row_bytes) catch return;
            defer page.free(w);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            fillSyntheticQ6_K(w, n, k);

            var ctx = BenchCtx{ .be = be, .x = x, .y = y, .n = n, .k = k, .td = TensorData{ .data = w.ptr, .dtype = .q6_k } };
            const median_ns = if (reupload) collectMedian(runGemvReupload, &ctx, iters) else collectMedian(runGemv, &ctx, iters);
            if (validate) validateVsCpu(runGemv, refillNone, &ctx, y, allocator, io, @tagName(kernel), be_name);
            emitJson("gemv_q6_k", be_name, median_ns, computeGbps(n * row_bytes, median_ns), computeGflops(2 * n * k, median_ns), iters);
        },

        // ── Quantized GEMV formats sharing one generic body ──────
        // Block geometry is the only thing that differs, so they are handled
        // together rather than as ten near-identical blocks.
        .gemv_q4_1,
        .gemv_q2_k,
        .gemv_q3_k,
        .gemv_q5_k,
        .gemv_iq4_nl,
        .gemv_iq4_xs,
        .gemv_tq1_0,
        .gemv_tq2_0,
        .gemv_fp8_e4m3,
        .gemv_fp8_e5m2,
        => {
            const geom: struct { dtype: backend_mod.DType, elems: usize, bytes: usize } = switch (kernel) {
                .gemv_q4_1 => .{ .dtype = .q4_1, .elems = quant_group_size, .bytes = q4_1_block_bytes },
                .gemv_q2_k => .{ .dtype = .q2_k, .elems = 256, .bytes = q2_k_block_bytes },
                .gemv_q3_k => .{ .dtype = .q3_k, .elems = 256, .bytes = q3_k_block_bytes },
                .gemv_q5_k => .{ .dtype = .q5_k, .elems = 256, .bytes = q5_k_block_bytes },
                .gemv_iq4_nl => .{ .dtype = .iq4_nl, .elems = quant_group_size, .bytes = iq4_nl_block_bytes },
                .gemv_iq4_xs => .{ .dtype = .iq4_xs, .elems = 256, .bytes = iq4_xs_block_bytes },
                .gemv_tq1_0 => .{ .dtype = .tq1_0, .elems = 256, .bytes = tq1_0_block_bytes },
                .gemv_tq2_0 => .{ .dtype = .tq2_0, .elems = 256, .bytes = tq2_0_block_bytes },
                // fp8 is byte-per-element, not blocked; one "block" of one byte.
                .gemv_fp8_e4m3 => .{ .dtype = .fp8_e4m3, .elems = 1, .bytes = 1 },
                .gemv_fp8_e5m2 => .{ .dtype = .fp8_e5m2, .elems = 1, .bytes = 1 },
                else => unreachable,
            };

            const x = page.alloc(f32, k) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);
            const nb = (k + geom.elems - 1) / geom.elems;
            const row_bytes = nb * geom.bytes;
            const w = page.alloc(u8, n * row_bytes) catch return;
            defer page.free(w);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            fillSyntheticQuantBytes(w);

            var ctx = BenchCtx{ .be = be, .x = x, .y = y, .n = n, .k = k, .td = TensorData{ .data = w.ptr, .dtype = geom.dtype } };
            const median_ns = if (reupload) collectMedian(runGemvReupload, &ctx, iters) else collectMedian(runGemv, &ctx, iters);
            if (validate) validateVsCpu(runGemv, refillNone, &ctx, y, allocator, io, @tagName(kernel), be_name);
            emitJson(@tagName(kernel), be_name, median_ns, computeGbps(n * row_bytes, median_ns), computeGflops(2 * n * k, median_ns), iters);
        },

        .gemm_q8_0 => {
            const n_tok = batched_n_tok;
            const x = page.alloc(f32, n_tok * k) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n_tok * n) catch return;
            defer page.free(y);
            const nb = (k + quant_group_size - 1) / quant_group_size;
            const row_bytes = nb * q8_0_block_bytes;
            const w = page.alloc(u8, n * row_bytes) catch return;
            defer page.free(w);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            fillSyntheticQ8_0(w, n, k);

            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .y = y,
                .n_tok = n_tok,
                .n = n,
                .k = k,
                .td = TensorData{ .data = w.ptr, .dtype = .q8_0 },
            };
            const median_ns = collectMedian(runGemm, &ctx, iters);
            if (validate) validateVsCpu(runGemm, refillNone, &ctx, y, allocator, io, @tagName(kernel), be_name);
            const total_bytes = n * row_bytes + n_tok * (k + n) * @sizeOf(f32);
            emitJson("gemm_q8_0", be_name, median_ns, computeGbps(total_bytes, median_ns), computeGflops(2 * n_tok * n * k, median_ns), iters);
        },

        .rms_norm_batched => {
            const n_tok = batched_n_tok;
            const x = page.alloc(f32, n_tok * n) catch return;
            defer page.free(x);
            const w_norm = page.alloc(f32, n) catch return;
            defer page.free(w_norm);
            const out = page.alloc(f32, n_tok * n) catch return;
            defer page.free(out);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            for (w_norm) |*v| v.* = 1.0;

            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .norm_weight = w_norm,
                .norm_out = out,
                .n_tok = n_tok,
                .n = n,
                .k = k,
            };
            const median_ns = collectMedian(runRmsNormBatched, &ctx, iters);
            if (validate) validateVsCpu(runRmsNormBatched, refillNone, &ctx, out, allocator, io, @tagName(kernel), be_name);
            emitJson("rms_norm_batched", be_name, median_ns, computeGbps(2 * n_tok * n * @sizeOf(f32), median_ns), 0, iters);
        },

        .rope_batched => {
            const n_tok = batched_n_tok;
            const n_heads = default_n_heads;
            const head_dim = default_head_dim;
            const stride = n_heads * head_dim;
            const x = page.alloc(f32, n_tok * stride) catch return;
            defer page.free(x);
            const positions = page.alloc(u32, n_tok) catch return;
            defer page.free(positions);
            for (positions, 0..) |*pos, t| pos.* = @intCast(t);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);

            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .positions = positions,
                .n_tok = n_tok,
                .n_heads = n_heads,
                .head_dim = head_dim,
                .n = n,
                .k = k,
            };
            const median_ns = collectMedian(runRopeBatched, &ctx, iters);
            if (validate) validateVsCpu(runRopeBatched, refillX, &ctx, x, allocator, io, @tagName(kernel), be_name);
            emitJson("rope_batched", be_name, median_ns, computeGbps(2 * n_tok * stride * @sizeOf(f32), median_ns), 0, iters);
        },

        .sdpa_prefill => {
            // The only batched op that writes the KV cache, and the one a
            // single-token benchmark can never reach. Longer than the other
            // batched fixtures so the tail tokens attend over a real history that
            // the same call produced.
            const n_tok = sdpa_prefill_n_tok;
            const n_heads = default_n_heads;
            const n_kv_heads = default_n_heads;
            const head_dim = default_head_dim;
            // No pre-existing history: every position this attends over is written
            // by the kernel itself during the call.
            //
            // Seeding the cache by writing host memory does not work as a fixture.
            // Vulkan uploads the KV cache and would see it; ROCm keeps it
            // device-side and appends, so it would not, and that difference is
            // correct behaviour rather than a bug: in inference every earlier
            // position was written by an earlier sdpa call on the device. A long
            // enough chunk covers the attend-over-history path honestly, because
            // the later tokens attend over positions the earlier ones just wrote.
            const prev_len: usize = 0;
            const kv_dim = n_kv_heads * head_dim;
            const capacity = prev_len + n_tok;

            const q = page.alloc(f32, n_tok * n_heads * head_dim) catch return;
            defer page.free(q);
            const k_new = page.alloc(f32, n_tok * kv_dim) catch return;
            defer page.free(k_new);
            const v_new = page.alloc(f32, n_tok * kv_dim) catch return;
            defer page.free(v_new);
            const out = page.alloc(f32, n_tok * n_heads * head_dim) catch return;
            defer page.free(out);
            const keys = page.alloc(u8, capacity * kv_dim * @sizeOf(f32)) catch return;
            defer page.free(keys);
            const values = page.alloc(u8, capacity * kv_dim * @sizeOf(f32)) catch return;
            defer page.free(values);
            const dummy = page.alloc(f32, 1) catch return;
            defer page.free(dummy);

            fillSyntheticF32(q, sdpa_q_mod, sdpa_q_scale, sdpa_q_offset);
            fillSyntheticF32(k_new, sdpa_k_mod, sdpa_kv_scale, sdpa_k_offset);
            fillSyntheticF32(v_new, sdpa_v_mod, sdpa_kv_scale, sdpa_v_offset_positive);

            var ctx = BenchCtx{
                .be = be,
                .x = dummy,
                .q = q,
                .keys = keys,
                .values = values,
                .k_new = k_new,
                .v_new = v_new,
                .sdpa_out = out,
                .n_tok = n_tok,
                .n_kv_heads = n_kv_heads,
                .n = n,
                .k = k,
                .n_heads = n_heads,
                .head_dim = head_dim,
                .seq_len = prev_len,
                .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
                .kv_type_k = .f32,
                .kv_type_v = .f32,
            };
            refillKvCache(&ctx);
            const median_ns = collectMedian(runSdpaPrefill, &ctx, iters);

            // Validate on a KV pair the timed loop never touched. ROCm keeps the
            // KV cache device-side and keyed by host address, so refilling host
            // memory does not reset it: the comparison would otherwise run
            // against whatever the benchmark iterations left on the GPU. A pair
            // used for the first time here has no device state to inherit.
            if (validate) {
                const keys2 = page.alloc(u8, capacity * kv_dim * @sizeOf(f32)) catch return;
                defer page.free(keys2);
                const values2 = page.alloc(u8, capacity * kv_dim * @sizeOf(f32)) catch return;
                defer page.free(values2);
                ctx.keys = keys2;
                ctx.values = values2;
                validateVsCpu(runSdpaPrefill, refillKvCache, &ctx, out, allocator, io, @tagName(kernel), be_name);
            }
            emitJson("sdpa_prefill", be_name, median_ns, 0, 0, iters);
        },

        .rms_norm_multi => {
            const n_tok = batched_n_tok;
            const n_heads = 8;
            const head_dim = 128;
            const total = n_tok * n_heads * head_dim;
            const x = page.alloc(f32, total) catch return;
            defer page.free(x);
            const w_norm = page.alloc(f32, head_dim) catch return;
            defer page.free(w_norm);

            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            for (w_norm, 0..) |*v, i| v.* = 1.0 + @as(f32, @floatFromInt(i % 7)) * 0.1;

            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .norm_weight = w_norm,
                .n_tok = n_tok,
                .n_heads = n_heads,
                .head_dim = head_dim,
                .n = n,
                .k = k,
            };
            const median_ns = collectMedian(runRmsNormMulti, &ctx, iters);
            if (validate) validateVsCpu(runRmsNormMulti, refillX, &ctx, x, allocator, io, @tagName(kernel), be_name);
            emitJson("rms_norm_multi", be_name, median_ns, computeGbps(2 * total * @sizeOf(f32), median_ns), 0, iters);
        },

        .add_aliased, .silu_mul_aliased => {
            const x = page.alloc(f32, n) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);
            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            fillSyntheticF32(y, synthetic_x_mod + 1, synthetic_x_scale, synthetic_x_offset);

            var ctx = BenchCtx{ .be = be, .x = x, .y = y, .n = n, .k = k };
            const median_ns = if (kernel == .add_aliased)
                collectMedian(runAddAliased, &ctx, iters)
            else
                collectMedian(runSiluMulAliased, &ctx, iters);
            if (validate) {
                if (kernel == .add_aliased)
                    validateVsCpu(runAddAliased, refillX, &ctx, x, allocator, io, @tagName(kernel), be_name)
                else
                    validateVsCpu(runSiluMulAliased, refillX, &ctx, x, allocator, io, @tagName(kernel), be_name);
            }
            emitJson(@tagName(kernel), be_name, median_ns, computeGbps(3 * n * @sizeOf(f32), median_ns), 0, iters);
        },

        // ── Ops the model uses that no benchmark reached ─────────
        .deinterleave, .split_q_gate => {
            const n_heads = 32;
            const head_dim = 128;
            const total = n_heads * head_dim * 2; // interleaved pairs
            const x = page.alloc(f32, total) catch return;
            defer page.free(x);
            const a = page.alloc(f32, total / 2) catch return;
            defer page.free(a);
            const b = page.alloc(f32, total / 2) catch return;
            defer page.free(b);
            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);

            var ctx = BenchCtx{ .be = be, .x = x, .y = a, .norm_out = b, .n_heads = n_heads, .head_dim = head_dim, .n = n, .k = k };
            const median_ns = if (kernel == .deinterleave)
                collectMedian(runDeinterleave, &ctx, iters)
            else
                collectMedian(runSplitQGate, &ctx, iters);
            if (validate) {
                if (kernel == .deinterleave)
                    validateVsCpu(runDeinterleave, refillNone, &ctx, a, allocator, io, @tagName(kernel), be_name)
                else
                    validateVsCpu(runSplitQGate, refillNone, &ctx, a, allocator, io, @tagName(kernel), be_name);
            }
            emitJson(@tagName(kernel), be_name, median_ns, computeGbps(2 * total * @sizeOf(f32), median_ns), 0, iters);
        },

        .add_rms_norm, .rms_norm_add, .gelu_mul, .clamped_silu_mul => {
            const x = page.alloc(f32, n) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);
            const w_norm = page.alloc(f32, n) catch return;
            defer page.free(w_norm);
            const out = page.alloc(f32, n) catch return;
            defer page.free(out);
            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            fillSyntheticF32(y, synthetic_x_mod + 1, synthetic_x_scale, synthetic_x_offset);
            for (w_norm, 0..) |*v, i| v.* = 1.0 + @as(f32, @floatFromInt(i % 5)) * 0.1;

            var ctx = BenchCtx{ .be = be, .x = x, .y = y, .norm_weight = w_norm, .norm_out = out, .n = n, .k = k };
            // addRmsNorm mutates its first argument; rmsNormAdd accumulates into
            // its output. Both need more than the input restored between runs.
            const median_ns = switch (kernel) {
                .add_rms_norm => collectMedian(runAddRmsNorm, &ctx, iters),
                .rms_norm_add => collectMedian(runRmsNormAdd, &ctx, iters),
                .gelu_mul => collectMedian(runGeluMul, &ctx, iters),
                else => collectMedian(runClampedSiluMul, &ctx, iters),
            };
            if (validate) switch (kernel) {
                .add_rms_norm => validateVsCpu(runAddRmsNorm, refillAddRmsNorm, &ctx, out, allocator, io, @tagName(kernel), be_name),
                .rms_norm_add => validateVsCpu(runRmsNormAdd, refillNormOut, &ctx, out, allocator, io, @tagName(kernel), be_name),
                .gelu_mul => validateVsCpu(runGeluMul, refillNone, &ctx, out, allocator, io, @tagName(kernel), be_name),
                else => validateVsCpu(runClampedSiluMul, refillNone, &ctx, out, allocator, io, @tagName(kernel), be_name),
            };
            emitJson(@tagName(kernel), be_name, median_ns, computeGbps(3 * n * @sizeOf(f32), median_ns), 0, iters);
        },

        .sigmoid_mul, .add_scaled => {
            const x = page.alloc(f32, n) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);
            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            fillSyntheticF32(y, synthetic_x_mod + 1, synthetic_x_scale, synthetic_x_offset);

            var ctx = BenchCtx{ .be = be, .x = x, .y = y, .n = n, .k = k };
            // Both write in place through x, so x is refilled between runs.
            const median_ns = if (kernel == .sigmoid_mul)
                collectMedian(runSigmoidMul, &ctx, iters)
            else
                collectMedian(runAddScaled, &ctx, iters);
            if (validate) {
                if (kernel == .sigmoid_mul)
                    validateVsCpu(runSigmoidMul, refillX, &ctx, x, allocator, io, @tagName(kernel), be_name)
                else
                    validateVsCpu(runAddScaled, refillX, &ctx, x, allocator, io, @tagName(kernel), be_name);
            }
            emitJson(@tagName(kernel), be_name, median_ns, computeGbps(2 * n * @sizeOf(f32), median_ns), 0, iters);
        },

        .gemv_multi, .gemv_t, .emb_lookup => {
            const x = page.alloc(f32, k) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);
            const y2 = page.alloc(f32, n) catch return;
            defer page.free(y2);
            // f32 weights: gemvT and embLookup both index raw f32 rows.
            const w = page.alloc(f32, n * k) catch return;
            defer page.free(w);
            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            fillSyntheticF32(w, synthetic_x_mod + 3, synthetic_x_scale, synthetic_x_offset);

            var ctx = BenchCtx{
                .be = be,
                .x = x,
                .y = y,
                .norm_out = y2,
                .n = n,
                .k = k,
                .td = TensorData{ .data = @ptrCast(w.ptr), .dtype = .f32 },
            };
            const median_ns = switch (kernel) {
                .gemv_multi => collectMedian(runGemvMulti, &ctx, iters),
                .gemv_t => collectMedian(runGemvT, &ctx, iters),
                else => collectMedian(runEmbLookup, &ctx, iters),
            };
            if (validate) switch (kernel) {
                .gemv_multi => validateVsCpu(runGemvMulti, refillNone, &ctx, y, allocator, io, @tagName(kernel), be_name),
                .gemv_t => validateVsCpu(runGemvT, refillNone, &ctx, y, allocator, io, @tagName(kernel), be_name),
                else => validateVsCpu(runEmbLookup, refillNone, &ctx, y, allocator, io, @tagName(kernel), be_name),
            };
            emitJson(@tagName(kernel), be_name, median_ns, computeGbps(n * k * @sizeOf(f32), median_ns), 0, iters);
        },

        .rope_mrope => {
            const n_heads = default_n_heads;
            const head_dim = default_head_dim;
            const x = page.alloc(f32, n_heads * head_dim) catch return;
            defer page.free(x);
            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            var ctx = BenchCtx{ .be = be, .x = x, .n_heads = n_heads, .head_dim = head_dim, .n = n, .k = k };
            const median_ns = collectMedian(runRopeMrope, &ctx, iters);
            if (validate) validateVsCpu(runRopeMrope, refillX, &ctx, x, allocator, io, @tagName(kernel), be_name);
            emitJson("rope_mrope", be_name, median_ns, computeGbps(2 * n_heads * head_dim * @sizeOf(f32), median_ns), 0, iters);
        },

        .all_reduce_add => {
            const x = page.alloc(f32, n) catch return;
            defer page.free(x);
            const y = page.alloc(f32, n) catch return;
            defer page.free(y);
            fillSyntheticF32(x, synthetic_x_mod, synthetic_x_scale, synthetic_x_offset);
            fillSyntheticF32(y, synthetic_x_mod + 1, synthetic_x_scale, synthetic_x_offset);
            var ctx = BenchCtx{ .be = be, .x = x, .y = y, .n = n, .k = k };
            const median_ns = collectMedian(runAllReduceAdd, &ctx, iters);
            // Accumulates into x, so x is restored between runs.
            if (validate) validateVsCpu(runAllReduceAdd, refillX, &ctx, x, allocator, io, @tagName(kernel), be_name);
            emitJson("all_reduce_add", be_name, median_ns, computeGbps(2 * n * @sizeOf(f32), median_ns), 0, iters);
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
            if (validate) validateVsCpu(runRmsNorm, refillNone, &ctx, ctx.norm_out.?, allocator, io, @tagName(kernel), be_name);
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
            if (validate) validateVsCpu(runSilu, refillNone, &ctx, y, allocator, io, @tagName(kernel), be_name);
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
            if (validate) validateVsCpu(runGelu, refillNone, &ctx, y, allocator, io, @tagName(kernel), be_name);
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
            if (validate) validateVsCpu(runSoftmax, refillX, &ctx, x, allocator, io, @tagName(kernel), be_name);
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
            if (validate) validateVsCpu(runL2Norm, refillX, &ctx, x, allocator, io, @tagName(kernel), be_name);
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
            if (validate) validateVsCpu(runAdd, refillNone, &ctx, ctx.norm_out.?, allocator, io, @tagName(kernel), be_name);
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
            if (validate) validateVsCpu(runMul, refillNone, &ctx, ctx.norm_out.?, allocator, io, @tagName(kernel), be_name);
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
            if (validate) validateVsCpu(runRope, refillX, &ctx, x, allocator, io, @tagName(kernel), be_name);
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
            if (validate) validateVsCpu(runSdpa, refillNone, &ctx, ctx.sdpa_out.?, allocator, io, @tagName(kernel), be_name);

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
        _ = std.posix.system.close(fd);
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
    var threaded = std.Io.Threaded.init(allocator, .{});
    bs.init(allocator, cli.backend, threaded.io(), 0);
    defer if (bs.pool) |*p| p.deinit();
    const be = bs.be;
    const be_name = bs.name;

    // ── Load tokenizer ───────────────────────────────────────────
    var tok = BpeTokenizer.init(allocator);
    defer tok.deinit();

    const vocab = fmt.getVocab();
    const merges = fmt.getMerges();
    const tok_kind: TokenizerKind = if (arch == .gemma3 or arch == .gemma4 or arch == .diffusion_gemma) .spm_no_dummy else if (merges != null) .bpe else .spm;
    const eos_id = fmt.getMetaU32("tokenizer.ggml.eos_token_id") orelse
        fmt.getMetaU32("eos_token_id") orelse
        arch.defaultEos();
    const bos_id: u32 = if (arch.templateIncludesBos())
        0
    else
        fmt.getMetaU32("tokenizer.ggml.bos_token_id") orelse
            fmt.getMetaU32("bos_token_id") orelse
            arch.defaultBos() orelse 0;

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
    var mdl = ModelStorage.initFromArch(arch, allocator, fmt, be, 0, .f32, .f32, 0, 0, null, 0, 1) catch |e| {
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

/// Entry point for the agave-bench micro-benchmark binary.
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
    var threaded = std.Io.Threaded.init(allocator, .{});
    bs.init(allocator, cli.backend, threaded.io(), 0);
    defer if (bs.pool) |*p| p.deinit();
    const be = bs.be;
    const be_name = bs.name;

    // Dispatch to kernel benchmark
    benchKernel(cli.kernel.?, be, be_name, cli.n, cli.k, cli.iters, cli.reupload, cli.validate, allocator, threaded.io());
    return if (validation_failed) 1 else 0;
}

// ── Tests ──────────────────────────────────────────────────────────

test "parseKeyValue exact match" {
    try std.testing.expectEqualStrings("4096", parseKeyValue("--n=4096", "--n").?);
    try std.testing.expectEqualStrings("metal", parseKeyValue("--backend=metal", "--backend").?);
    try std.testing.expect(parseKeyValue("--n=4096", "--k") == null);
    try std.testing.expect(parseKeyValue("--n", "--n") == null); // no '='
    try std.testing.expect(parseKeyValue("", "--n") == null);
}

test "parseKeyValue empty value" {
    const v = parseKeyValue("--n=", "--n");
    try std.testing.expect(v != null);
    try std.testing.expectEqualStrings("", v.?);
}

test "parseKernelName valid" {
    try std.testing.expectEqual(Kernel.gemv_f32, parseKernelName("gemv_f32").?);
    try std.testing.expectEqual(Kernel.rms_norm, parseKernelName("rms_norm").?);
    try std.testing.expectEqual(Kernel.sdpa, parseKernelName("sdpa").?);
}

test "parseKernelName invalid" {
    try std.testing.expect(parseKernelName("notakernel") == null);
    try std.testing.expect(parseKernelName("") == null);
    try std.testing.expect(parseKernelName("GEMV_F32") == null);
}

test "kernel_names_joined lists every Kernel" {
    inline for (@typeInfo(Kernel).@"enum".fields) |f| {
        try std.testing.expect(std.mem.indexOf(u8, kernel_names_joined, f.name) != null);
    }
    try std.testing.expect(std.mem.indexOf(u8, kernel_names_joined, "rms_norm_multi") != null);
    try std.testing.expect(std.mem.indexOf(u8, kernel_names_joined, "add_aliased") != null);
    try std.testing.expect(std.mem.indexOf(u8, kernel_names_joined, "silu_mul_aliased") != null);
}

test "wrapWords empty and single" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("", wrapWords(&buf, "", "  ", 72));
    try std.testing.expectEqualStrings("  gemv_f32", wrapWords(&buf, "gemv_f32", "  ", 72));
}

test "wrapWords wraps at width" {
    var buf: [64]u8 = undefined;
    const s = wrapWords(&buf, "aaa bbb ccc", "  ", 8);
    try std.testing.expectEqualStrings("  aaa\n  bbb\n  ccc", s);
}

test "parseBackendName valid" {
    try std.testing.expectEqual(BackendChoice.cpu, parseBackendName("cpu").?);
    try std.testing.expectEqual(BackendChoice.metal, parseBackendName("metal").?);
}

test "parseBackendName invalid" {
    try std.testing.expect(parseBackendName("fpga") == null);
    try std.testing.expect(parseBackendName("") == null);
}

test "fuzz: micro_bench pure functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // Comptime symbol refs for I/O and complex functions
            comptime {
                _ = &fdWriteAll;
                _ = &print;
                _ = &eprint;
                _ = &printUsage;
                _ = &eprintKernelList;
                _ = &parseCli;
                _ = &getArgValue;
                _ = &collectMedian;
                _ = &wrapWords;
            }

            var buf: [64]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len: usize = smith.valueWithHash(u8, 1) % 32;
            const s = buf[0..len];

            var wrap_buf: [128]u8 = undefined;
            _ = wrapWords(&wrap_buf, s, "  ", 40);

            // parseKeyValue with random key and arg
            _ = parseKeyValue(s, "--n");
            _ = parseKeyValue("--backend=metal", s);

            // parseKernelName / parseBackendName with random input
            _ = parseKernelName(s);
            _ = parseBackendName(s);
        }
    }.f, .{});
}
