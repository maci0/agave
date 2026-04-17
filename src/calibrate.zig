//! TriAttention calibration — generates per-head, per-frequency-band Q statistics
//! for the trigonometric KV eviction policy.
//!
//! Usage: agave calibrate <model.gguf> [--tokens N] [--output path.cal]
//!
//! Loads a GGUF model, runs N tokens of inference on a fixed calibration prompt,
//! captures pre-RoPE Q vectors after Q projection at each layer, and computes
//! per-head statistics: center norm, center phase, expected norm, and concentration.
//! Saves results to a binary .cal file consumed by TriAttention scoring.

const std = @import("std");
const Allocator = std.mem.Allocator;
const backend_mod = @import("backend/backend.zig");
const format_mod = @import("format/format.zig");
const model_mod = @import("models/model.zig");
const arch_mod = @import("arch.zig");
const Arch = arch_mod.Arch;
const display_mod = @import("display.zig");
const kv_evict = @import("ops/kv_evict.zig");

const Backend = backend_mod.Backend;
const BackendState = backend_mod.BackendState;
const Format = format_mod.Format;
const GGUFFile = format_mod.GGUFFile;
const SafeTensorsDir = format_mod.SafeTensorsDir;

const kv_quant = @import("ops/kv_quant.zig");
const KvQuantType = kv_quant.KvQuantType;

// ── Named constants ──────────────────────────────────────────────────────────

/// Buffer size for stderr formatting output.
const print_buf_size: usize = 4096;
/// Default number of calibration tokens to generate.
const default_calibration_tokens: u32 = 1000;
/// Default output file name for calibration data.
const default_output_filename: []const u8 = "calibration.cal";
/// Magic bytes for the .cal file header.
const cal_magic: [4]u8 = .{ 'A', 'C', 'A', 'L' };
/// File format version.
const cal_version: u32 = 1;
/// Number of random input vectors per layer for weight-based calibration.
const calibration_vectors_per_layer: usize = 512;
/// Seed for the calibration PRNG.
const calibration_seed: u64 = 0xCAFE_BABE_DEAD_BEEF;
/// Number of layer-0 tensor name candidates to probe for Q weight dims.
const max_q_tensor_probes: usize = 2;

const Io = std.Io;

/// Standard I/O file handles via std.Io.File (Zig 0.16 idiom).
const stderr_file = Io.File.stderr();
const stdout_file = Io.File.stdout();

/// Module-level Io instance, set by run() or readCalFile() from caller.
var mod_io: Io = undefined;

// ── Stderr helpers ───────────────────────────────────────────────────────────

fn eprint(comptime fmt: []const u8, args: anytype) void {
    var buf: [print_buf_size]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, fmt, args) catch return;
    _ = std.c.write(stderr_file.handle, text.ptr, text.len);
}

// ── Argument parsing ─────────────────────────────────────────────────────────

/// Parsed command-line arguments for the `calibrate` sub-command.
const CalibrateArgs = struct {
    /// Path to the GGUF model file or SafeTensors directory.
    model_path: []const u8,
    /// Number of calibration vectors to use.
    n_tokens: u32 = default_calibration_tokens,
    /// Output file path for the .cal file.
    output: []const u8 = default_output_filename,
};

/// Print usage information to stdout.
pub fn printUsage() void {
    const usage_text =
        \\agave calibrate — Generate TriAttention calibration statistics
        \\
        \\USAGE:
        \\  agave calibrate [OPTIONS] <model.gguf|model-dir/>
        \\
        \\ARGUMENTS:
        \\  <model.gguf|model-dir/>  Path to GGUF model file or SafeTensors directory
        \\
        \\GENERAL:
        \\  -h, --help               Show this help message
        \\  -v, --version            Print version
        \\
        \\OPTIONS:
        \\      --tokens <N>         Number of calibration vectors [default: 1000]
        \\      --output <PATH>      Output .cal file path [default: calibration.cal]
        \\
        \\DESCRIPTION:
        \\  Generates per-head, per-frequency-band Q projection statistics needed by
        \\  the TriAttention KV cache eviction policy (--kv-eviction tri).
        \\
        \\  The calibration process:
        \\    1. Loads the model's Q weight matrices from each attention layer
        \\    2. Projects random input vectors through Q weights to simulate Q outputs
        \\    3. Splits Q vectors into RoPE frequency bands (complex pairs)
        \\    4. Computes per-band statistics: center norm, phase, expected norm, concentration
        \\    5. Writes results to a binary .cal file
        \\
        \\EXAMPLES:
        \\  agave calibrate model.gguf
        \\  agave calibrate model.gguf --tokens 2000 --output model.cal
        \\  agave calibrate ./safetensors-dir/ --output tri.cal
        \\
    ;
    _ = std.c.write(stdout_file.handle, usage_text.ptr, usage_text.len);
}

/// Parse command-line arguments for the `calibrate` sub-command.
/// Returns null if --help was requested.
fn parseArgs(args_iter: *std.process.Args.Iterator) ?CalibrateArgs {
    var result = CalibrateArgs{ .model_path = "" };
    var have_model = false;

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "help")) {
            printUsage();
            return null;
        } else if (std.mem.eql(u8, arg, "--version") or std.mem.eql(u8, arg, "-v")) {
            display_mod.printVersion();
            return null;
        } else if (std.mem.eql(u8, arg, "--tokens")) {
            const val = args_iter.next() orelse {
                eprint("Error: --tokens requires a value\n", .{});
                eprint("Run 'agave calibrate --help' for more information.\n", .{});
                return null;
            };
            result.n_tokens = std.fmt.parseInt(u32, val, 10) catch {
                eprint("Error: --tokens value '{s}' is not a valid integer\n", .{val});
                return null;
            };
            if (result.n_tokens == 0) {
                eprint("Error: --tokens must be >= 1\n", .{});
                return null;
            }
        } else if (std.mem.eql(u8, arg, "--output")) {
            const val = args_iter.next() orelse {
                eprint("Error: --output requires a path\n", .{});
                eprint("Run 'agave calibrate --help' for more information.\n", .{});
                return null;
            };
            result.output = val;
        } else if (arg.len > 0 and arg[0] == '-') {
            eprint("Error: unknown option '{s}'\n", .{arg});
            eprint("Run 'agave calibrate --help' for more information.\n", .{});
            return null;
        } else {
            if (have_model) {
                eprint("Error: unexpected argument '{s}'\n", .{arg});
                eprint("Run 'agave calibrate --help' for more information.\n", .{});
                return null;
            }
            result.model_path = arg;
            have_model = true;
        }
    }

    if (!have_model) {
        eprint("Error: model path required\n", .{});
        eprint("Usage: agave calibrate <model.gguf|model-dir/>\n", .{});
        eprint("Run 'agave calibrate --help' for more information.\n", .{});
        return null;
    }

    return result;
}

// ── Accumulator for per-head, per-band statistics ────────────────────────────

/// Running statistics for a single query head's frequency bands.
/// Tracks complex running sums and magnitude sums for computing
/// center norm, center phase, expected norm, and concentration.
const BandAccumulator = struct {
    /// Complex running sum of q_f: Σ q_f (real part).
    sum_re: []f32,
    /// Complex running sum of q_f: Σ q_f (imaginary part).
    sum_im: []f32,
    /// Running sum of ||q_f||: Σ ||q_f||.
    sum_norm: []f32,
    /// Number of samples accumulated.
    count: u64,
    /// Number of frequency bands.
    n_bands: usize,

    fn init(allocator: Allocator, n_bands: usize) !BandAccumulator {
        const sum_re = try allocator.alloc(f32, n_bands);
        errdefer allocator.free(sum_re);
        const sum_im = try allocator.alloc(f32, n_bands);
        errdefer allocator.free(sum_im);
        const sum_norm = try allocator.alloc(f32, n_bands);

        @memset(sum_re, 0);
        @memset(sum_im, 0);
        @memset(sum_norm, 0);

        return .{
            .sum_re = sum_re,
            .sum_im = sum_im,
            .sum_norm = sum_norm,
            .count = 0,
            .n_bands = n_bands,
        };
    }

    fn deinit(self: *BandAccumulator, allocator: Allocator) void {
        allocator.free(self.sum_re);
        allocator.free(self.sum_im);
        allocator.free(self.sum_norm);
    }

    /// Accumulate one Q vector of length head_dim.
    /// Splits into frequency bands: q_f = (q[2f], q[2f+1]) as complex number.
    fn accumulate(self: *BandAccumulator, q_vec: []const f32) void {
        std.debug.assert(q_vec.len >= self.n_bands * 2);
        for (0..self.n_bands) |f| {
            const re = q_vec[2 * f];
            const im = q_vec[2 * f + 1];
            self.sum_re[f] += re;
            self.sum_im[f] += im;
            self.sum_norm[f] += @sqrt(re * re + im * im);
        }
        self.count += 1;
    }

    /// Compute final statistics from accumulated data.
    /// Writes to pre-allocated output slices (length n_bands each).
    fn finalize(
        self: *const BandAccumulator,
        q_center_norm: []f32,
        q_center_phase: []f32,
        q_expected_norm: []f32,
        concentration: []f32,
    ) void {
        std.debug.assert(q_center_norm.len == self.n_bands);
        std.debug.assert(self.count > 0);
        const inv_count: f32 = 1.0 / @as(f32, @floatFromInt(self.count));

        for (0..self.n_bands) |f| {
            const mean_re = self.sum_re[f] * inv_count;
            const mean_im = self.sum_im[f] * inv_count;
            const center_norm = @sqrt(mean_re * mean_re + mean_im * mean_im);
            const expected_norm = self.sum_norm[f] * inv_count;

            q_center_norm[f] = center_norm;
            q_center_phase[f] = std.math.atan2(mean_im, mean_re);
            q_expected_norm[f] = expected_norm;
            concentration[f] = if (expected_norm > 1e-10)
                center_norm / expected_norm
            else
                0.0;
        }
    }
};

// ── Weight-based Q projection calibration ────────────────────────────────────

/// Generate calibration data by projecting random vectors through Q weight matrices.
///
/// For each layer:
///   1. Load the Q weight tensor
///   2. Generate n_tokens random input vectors (Gaussian-like)
///   3. For each input, compute Q = W_q @ input (GEMV)
///   4. Split Q into per-head chunks, accumulate per-band statistics
///
/// This avoids modifying the forward pass or adding hooks to the model vtable.
fn runCalibration(
    allocator: Allocator,
    fmt: Format,
    be: Backend,
    arch_str: []const u8,
    n_tokens: u32,
    n_layers: u32,
    n_q_heads: u32,
    head_dim: u32,
    n_embd: u32,
    rope_theta: f32,
) !CalibrationResult {
    const n_bands: u32 = head_dim / 2;
    const qkv_dim: usize = @as(usize, n_q_heads) * head_dim;

    eprint("Calibration: {d} layers, {d} Q heads, head_dim={d}, {d} bands\n", .{
        n_layers, n_q_heads, head_dim, n_bands,
    });
    eprint("Projecting {d} random vectors per layer through Q weights...\n", .{n_tokens});

    // Allocate per-layer, per-head accumulators
    const total_heads: usize = @as(usize, n_layers) * n_q_heads;
    var accumulators = try allocator.alloc(BandAccumulator, total_heads);
    errdefer {
        for (accumulators) |*acc| acc.deinit(allocator);
        allocator.free(accumulators);
    }
    for (0..total_heads) |i| {
        accumulators[i] = try BandAccumulator.init(allocator, n_bands);
    }

    // Scratch buffers for input and Q output
    const input_buf = try allocator.alloc(f32, n_embd);
    defer allocator.free(input_buf);
    const q_output = try allocator.alloc(f32, qkv_dim);
    defer allocator.free(q_output);

    // PRNG for generating pseudo-random input vectors
    var prng = std.Random.Xoshiro256.init(calibration_seed);

    // Detect whether the model uses fused QKV or separate Q weight
    const uses_fused_qkv = fmt.layerTensor(0, "attn_qkv.weight") != null;

    for (0..n_layers) |li| {
        const layer: u32 = @intCast(li);

        // Load Q weight tensor for this layer
        const q_tensor = if (uses_fused_qkv)
            fmt.layerTensor(layer, "attn_qkv.weight")
        else
            fmt.layerTensor(layer, "attn_q.weight");

        if (q_tensor == null) {
            eprint("Warning: Q weight not found for layer {d}, skipping\n", .{layer});
            continue;
        }
        const qw = q_tensor.?;

        // Project n_tokens random inputs
        for (0..n_tokens) |_| {
            // Generate random input vector (approximate standard normal via Box-Muller-like pairs)
            fillRandomGaussian(&prng, input_buf);

            // Q projection: q_output = W_q @ input
            @memset(q_output, 0);
            if (!model_mod.mlxGemv(be, fmt, input_buf.ptr, qw, q_output.ptr, qkv_dim, n_embd)) {
                be.gemv(
                    input_buf.ptr,
                    .{ .data = qw.data_ptr, .dtype = qw.dtype },
                    q_output.ptr,
                    qkv_dim,
                    n_embd,
                );
            }

            // Sync GPU if needed (Q buffer may be written by GPU)
            be.sync();

            // Split into per-head Q vectors and accumulate
            for (0..n_q_heads) |qh| {
                const acc_idx = li * n_q_heads + qh;
                const head_start = qh * head_dim;
                const head_end = head_start + head_dim;
                accumulators[acc_idx].accumulate(q_output[head_start..head_end]);
            }
        }

        // Progress
        if ((li + 1) % 4 == 0 or li + 1 == n_layers) {
            eprint("\r  layer {d}/{d}", .{ li + 1, n_layers });
        }
    }
    eprint("\n", .{});

    // Compute RoPE frequencies
    const rope_freqs = try kv_evict.ropeFrequencies(allocator, head_dim, rope_theta);
    errdefer allocator.free(rope_freqs);

    // Finalize statistics into flat arrays
    const band_count: usize = @as(usize, n_layers) * n_q_heads * n_bands;
    var q_center_norm = try allocator.alloc(f32, band_count);
    errdefer allocator.free(q_center_norm);
    var q_center_phase = try allocator.alloc(f32, band_count);
    errdefer allocator.free(q_center_phase);
    var q_expected_norm = try allocator.alloc(f32, band_count);
    errdefer allocator.free(q_expected_norm);
    var concentration_data = try allocator.alloc(f32, band_count);
    errdefer allocator.free(concentration_data);

    for (0..total_heads) |i| {
        const offset = i * n_bands;
        accumulators[i].finalize(
            q_center_norm[offset..][0..n_bands],
            q_center_phase[offset..][0..n_bands],
            q_expected_norm[offset..][0..n_bands],
            concentration_data[offset..][0..n_bands],
        );
    }

    // Clean up accumulators
    for (accumulators) |*acc| acc.deinit(allocator);
    allocator.free(accumulators);

    _ = arch_str;

    return .{
        .n_layers = n_layers,
        .n_q_heads = n_q_heads,
        .head_dim = head_dim,
        .n_bands = n_bands,
        .rope_theta = rope_theta,
        .q_center_norm = q_center_norm,
        .q_center_phase = q_center_phase,
        .q_expected_norm = q_expected_norm,
        .concentration = concentration_data,
        .rope_freqs = rope_freqs,
    };
}

/// Fill a buffer with pseudo-random values approximating a standard normal distribution.
/// Uses the uniform-to-Gaussian approximation: sum of 4 uniforms, centered and scaled.
fn fillRandomGaussian(prng: *std.Random.Xoshiro256, buf: []f32) void {
    // Number of uniform samples to sum for the central limit approximation.
    const n_sum: comptime_int = 4;
    // Scale factor: sqrt(12/n_sum) for variance normalization.
    const scale: f32 = @sqrt(12.0 / @as(f32, n_sum));
    const rng = prng.random();
    for (buf) |*v| {
        var sum: f32 = 0;
        for (0..n_sum) |_| {
            sum += rng.float(f32);
        }
        // Center at 0 and scale to approximate N(0,1)
        v.* = (sum - @as(f32, n_sum) / 2.0) * scale;
    }
}

/// Result from calibration — flat arrays ready for writing.
const CalibrationResult = struct {
    n_layers: u32,
    n_q_heads: u32,
    head_dim: u32,
    n_bands: u32,
    rope_theta: f32,
    q_center_norm: []f32,
    q_center_phase: []f32,
    q_expected_norm: []f32,
    concentration: []f32,
    rope_freqs: []f32,

    fn deinit(self: *CalibrationResult, allocator: Allocator) void {
        allocator.free(self.q_center_norm);
        allocator.free(self.q_center_phase);
        allocator.free(self.q_expected_norm);
        allocator.free(self.concentration);
        allocator.free(self.rope_freqs);
    }
};

// ── File I/O ─────────────────────────────────────────────────────────────────

/// Write calibration results to a binary .cal file.
///
/// File format:
///   magic:      [4]u8 "ACAL"
///   version:    u32 = 1
///   n_layers:   u32
///   n_q_heads:  u32
///   head_dim:   u32
///   n_bands:    u32 (= head_dim / 2)
///   rope_theta: f32
///   data:       [n_layers * n_q_heads * n_bands] f32 — q_center_norm
///               [n_layers * n_q_heads * n_bands] f32 — q_center_phase
///               [n_layers * n_q_heads * n_bands] f32 — q_expected_norm
///               [n_layers * n_q_heads * n_bands] f32 — concentration
fn writeCalFile(result: *const CalibrationResult, path: []const u8) !void {
    const io = mod_io;
    const file = try Io.Dir.cwd().createFile(io, path, .{ .read = true });
    defer file.close(io);

    var offset: u64 = 0;
    // Helper: write bytes at current offset and advance
    const pwrite = struct {
        fn f(fi: Io.File, iio: Io, data: []const u8, off: *u64) !void {
            try fi.writePositionalAll(iio, data, off.*);
            off.* += data.len;
        }
    }.f;

    try pwrite(file, io, &cal_magic, &offset);
    try pwrite(file, io, std.mem.asBytes(&cal_version), &offset);
    try pwrite(file, io, std.mem.asBytes(&result.n_layers), &offset);
    try pwrite(file, io, std.mem.asBytes(&result.n_q_heads), &offset);
    try pwrite(file, io, std.mem.asBytes(&result.head_dim), &offset);
    try pwrite(file, io, std.mem.asBytes(&result.n_bands), &offset);
    try pwrite(file, io, std.mem.asBytes(&result.rope_theta), &offset);

    try pwrite(file, io, std.mem.sliceAsBytes(result.q_center_norm), &offset);
    try pwrite(file, io, std.mem.sliceAsBytes(result.q_center_phase), &offset);
    try pwrite(file, io, std.mem.sliceAsBytes(result.q_expected_norm), &offset);
    try pwrite(file, io, std.mem.sliceAsBytes(result.concentration), &offset);
}

/// Read calibration data from a .cal file and return TriCalibration structs
/// for each query head. The caller owns all returned memory.
///
/// Returns a slice of [n_layers * n_q_heads] TriCalibration entries.
pub fn readCalFile(allocator: Allocator, io: Io, path: []const u8) ![]kv_evict.TriCalibration {
    const file = try Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    // Read and validate header
    var magic: [4]u8 = undefined;
    const magic_n = try file.readPositionalAll(io, &magic, 0);
    if (magic_n < 4 or !std.mem.eql(u8, &magic, &cal_magic)) return error.InvalidFormat;

    var version_buf: [4]u8 = undefined;
    _ = try file.readPositionalAll(io, &version_buf, 4);
    const version = std.mem.bytesAsValue(u32, &version_buf).*;
    if (version != cal_version) return error.UnsupportedVersion;

    var header_buf: [20]u8 = undefined;
    _ = try file.readPositionalAll(io, &header_buf, 8);
    const n_layers = std.mem.bytesAsValue(u32, header_buf[0..4]).*;
    const n_q_heads = std.mem.bytesAsValue(u32, header_buf[4..8]).*;
    _ = std.mem.bytesAsValue(u32, header_buf[8..12]).*; // head_dim
    const n_bands = std.mem.bytesAsValue(u32, header_buf[12..16]).*;
    const rope_theta = std.mem.bytesAsValue(f32, header_buf[16..20]).*;

    const total_heads: usize = @as(usize, n_layers) * n_q_heads;
    const band_count: usize = total_heads * n_bands;

    // Read data arrays (header is 28 bytes: magic(4) + version(4) + fields(20))
    const data_offset: u64 = 28;
    const array_bytes = band_count * @sizeOf(f32);

    var q_center_norm = try allocator.alloc(f32, band_count);
    errdefer allocator.free(q_center_norm);
    _ = try file.readPositionalAll(io, std.mem.sliceAsBytes(q_center_norm), data_offset);

    var q_center_phase = try allocator.alloc(f32, band_count);
    errdefer allocator.free(q_center_phase);
    _ = try file.readPositionalAll(io, std.mem.sliceAsBytes(q_center_phase), data_offset + array_bytes);

    var q_expected_norm = try allocator.alloc(f32, band_count);
    errdefer allocator.free(q_expected_norm);
    _ = try file.readPositionalAll(io, std.mem.sliceAsBytes(q_expected_norm), data_offset + array_bytes * 2);

    var concentration_data = try allocator.alloc(f32, band_count);
    errdefer allocator.free(concentration_data);
    _ = try file.readPositionalAll(io, std.mem.sliceAsBytes(concentration_data), data_offset + array_bytes * 3);

    // Compute RoPE frequencies
    const head_dim: usize = @as(usize, n_bands) * 2;
    const rope_freqs = try kv_evict.ropeFrequencies(allocator, head_dim, rope_theta);
    errdefer allocator.free(rope_freqs);

    // Build TriCalibration entries
    var calibrations = try allocator.alloc(kv_evict.TriCalibration, total_heads);
    for (0..total_heads) |i| {
        const offset = i * n_bands;
        calibrations[i] = .{
            .q_center_norm = q_center_norm[offset..][0..n_bands],
            .q_center_phase = q_center_phase[offset..][0..n_bands],
            .q_expected_norm = q_expected_norm[offset..][0..n_bands],
            .concentration = concentration_data[offset..][0..n_bands],
            .rope_freqs = rope_freqs,
            .n_bands = n_bands,
        };
    }

    return calibrations;
}

// ── Entry point ──────────────────────────────────────────────────────────────

/// Main entry point for the `agave calibrate` sub-command.
///
/// Parses arguments, loads the model format, runs calibration, and writes results.
/// Returns exit code (0 = success, 1 = error).
pub fn run(allocator: Allocator, io: Io, process_args: std.process.Args) u8 {
    mod_io = io;
    var args_iter = process_args.iterate();
    _ = args_iter.skip(); // Skip program name (argv[0]).
    _ = args_iter.skip(); // Skip "calibrate" subcommand.

    const maybe_args = parseArgs(&args_iter);
    const args = maybe_args orelse return 1;

    // Detect format: directory -> SafeTensors, else -> GGUF
    const is_dir = blk: {
        const dir = Io.Dir.cwd().openDir(mod_io, args.model_path, .{}) catch break :blk false;
        dir.close(mod_io);
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
        st_dir = SafeTensorsDir.open(allocator, args.model_path) catch |e| {
            eprint("Error: failed to open safetensors dir '{s}': {}\n", .{ args.model_path, e });
            return 1;
        };
        fmt = st_dir.?.format();
    } else {
        gguf_file = GGUFFile.open(allocator, args.model_path) catch |e| {
            eprint("Error: failed to open '{s}': {}\n", .{ args.model_path, e });
            return 1;
        };
        fmt = gguf_file.?.format();
    }

    // Read model metadata
    const arch_str = fmt.getMetaStr("general.architecture") orelse
        fmt.getMetaStr("model_type") orelse "unknown";

    const n_layers = fmt.getArchU32(arch_str, "block_count") orelse
        fmt.getMetaU32("num_hidden_layers") orelse {
        eprint("Error: could not determine number of layers from model metadata\n", .{});
        return 1;
    };

    const n_embd = fmt.getArchU32(arch_str, "embedding_length") orelse
        fmt.getMetaU32("hidden_size") orelse {
        eprint("Error: could not determine embedding dimension from model metadata\n", .{});
        return 1;
    };

    const n_q_heads = fmt.getArchU32(arch_str, "attention.head_count") orelse
        fmt.getMetaU32("num_attention_heads") orelse {
        eprint("Error: could not determine number of Q heads from model metadata\n", .{});
        return 1;
    };

    const head_dim_meta = fmt.getArchU32(arch_str, "attention.key_length") orelse
        fmt.getMetaU32("head_dim");
    const head_dim: u32 = head_dim_meta orelse
        (if (n_embd > 0 and n_q_heads > 0) n_embd / n_q_heads else 0);

    if (head_dim == 0 or head_dim % 2 != 0) {
        eprint("Error: invalid head_dim={d} (must be > 0 and even)\n", .{head_dim});
        return 1;
    }

    const rope_theta = fmt.getArchF32(arch_str, "rope.freq_base") orelse
        fmt.getMetaF32("rope_theta") orelse 10000.0;

    const name = fmt.getMetaStr("general.name") orelse
        fmt.getMetaStr("model_type") orelse "unknown";

    eprint("Model: {s} (arch={s})\n", .{ name, arch_str });
    eprint("  n_layers={d}, n_embd={d}, n_q_heads={d}, head_dim={d}, rope_theta={d:.0}\n", .{
        n_layers, n_embd, n_q_heads, head_dim, rope_theta,
    });

    // Initialize backend (CPU only — calibration doesn't need GPU)
    var bs = BackendState{};
    bs.init(allocator, .cpu, io);
    defer if (bs.pool) |*p| p.deinit();
    const be = bs.be;

    // Run calibration
    var result = runCalibration(
        allocator,
        fmt,
        be,
        arch_str,
        args.n_tokens,
        n_layers,
        n_q_heads,
        head_dim,
        n_embd,
        rope_theta,
    ) catch |e| {
        eprint("Error: calibration failed: {}\n", .{e});
        return 1;
    };
    defer result.deinit(allocator);

    // Print summary statistics
    printSummary(&result);

    // Write output file
    writeCalFile(&result, args.output) catch |e| {
        eprint("Error: failed to write '{s}': {}\n", .{ args.output, e });
        return 1;
    };

    const band_count: usize = @as(usize, n_layers) * n_q_heads * result.n_bands;
    const file_size = @sizeOf([4]u8) + @sizeOf(u32) * 5 + @sizeOf(f32) + band_count * @sizeOf(f32) * 4;
    const fsize = display_mod.formatSize(file_size);
    eprint("\nCalibration saved to: {s} ({d:.1} {s})\n", .{
        args.output, fsize.val, fsize.unit,
    });
    eprint("Use with: agave model.gguf --kv-eviction tri\n", .{});

    return 0;
}

/// Print a summary of calibration results to stderr.
fn printSummary(result: *const CalibrationResult) void {
    const n_bands = result.n_bands;
    const total_heads: usize = @as(usize, result.n_layers) * result.n_q_heads;

    // Compute aggregate statistics across all heads
    var avg_concentration: f32 = 0;
    var max_concentration: f32 = 0;
    var min_concentration: f32 = 1.0;

    for (0..total_heads) |h| {
        const offset = h * n_bands;
        for (0..n_bands) |f| {
            const c = result.concentration[offset + f];
            avg_concentration += c;
            max_concentration = @max(max_concentration, c);
            min_concentration = @min(min_concentration, c);
        }
    }
    const total_bands: f32 = @floatFromInt(total_heads * n_bands);
    avg_concentration /= total_bands;

    eprint("\nCalibration summary:\n", .{});
    eprint("  Concentration: avg={d:.4}, min={d:.4}, max={d:.4}\n", .{
        avg_concentration, min_concentration, max_concentration,
    });
    eprint("  (High concentration -> strong directional bias -> tri-scoring effective)\n", .{});
    eprint("  (Low concentration -> uniform distribution -> norm-scoring fallback)\n", .{});
}

// ── Tests ────────────────────────────────────────────────────────────────────

test "BandAccumulator accumulate and finalize" {
    const allocator = std.testing.allocator;
    var acc = try BandAccumulator.init(allocator, 2);
    defer acc.deinit(allocator);

    // Accumulate two vectors with known values
    // q = [1, 0, 0, 1] -> band 0: (1,0), band 1: (0,1)
    const v1 = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    acc.accumulate(&v1);

    // q = [0, 1, 1, 0] -> band 0: (0,1), band 1: (1,0)
    const v2 = [_]f32{ 0.0, 1.0, 1.0, 0.0 };
    acc.accumulate(&v2);

    var cn: [2]f32 = undefined;
    var cp: [2]f32 = undefined;
    var en: [2]f32 = undefined;
    var conc: [2]f32 = undefined;
    acc.finalize(&cn, &cp, &en, &conc);

    // Band 0: mean = (0.5, 0.5), ||mean|| = sqrt(0.5)
    // E[||q||] = (1.0 + 1.0) / 2 = 1.0
    try std.testing.expectApproxEqAbs(@sqrt(@as(f32, 0.5)), cn[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), en[0], 1e-5);
    // concentration = sqrt(0.5) / 1.0
    try std.testing.expectApproxEqAbs(@sqrt(@as(f32, 0.5)), conc[0], 1e-5);
}

test "fillRandomGaussian produces reasonable values" {
    var prng = std.Random.Xoshiro256.init(42);
    var buf: [100]f32 = undefined;
    fillRandomGaussian(&prng, &buf);

    // Check that values are roughly centered and bounded
    var sum: f32 = 0;
    var max_abs: f32 = 0;
    for (buf) |v| {
        sum += v;
        max_abs = @max(max_abs, @abs(v));
    }
    const mean = sum / @as(f32, @floatFromInt(buf.len));
    // Mean should be near 0
    try std.testing.expect(@abs(mean) < 0.5);
    // Values should be bounded (CLT with 4 uniforms gives range ~ [-3.5, 3.5])
    try std.testing.expect(max_abs < 4.0);
}

test "writeCalFile and readCalFile roundtrip" {
    const allocator = std.testing.allocator;
    mod_io = std.testing.io;

    // Create test calibration data
    const n_bands: usize = 2;
    const total_entries: usize = 4; // 1 layer * 2 heads * 2 bands

    var cn = try allocator.alloc(f32, total_entries);
    defer allocator.free(cn);
    var cp = try allocator.alloc(f32, total_entries);
    defer allocator.free(cp);
    var en = try allocator.alloc(f32, total_entries);
    defer allocator.free(en);
    var conc = try allocator.alloc(f32, total_entries);
    defer allocator.free(conc);
    var freqs = try allocator.alloc(f32, n_bands);
    defer allocator.free(freqs);

    cn[0] = 1.0;
    cn[1] = 2.0;
    cn[2] = 3.0;
    cn[3] = 4.0;
    cp[0] = 0.1;
    cp[1] = 0.2;
    cp[2] = 0.3;
    cp[3] = 0.4;
    en[0] = 5.0;
    en[1] = 6.0;
    en[2] = 7.0;
    en[3] = 8.0;
    conc[0] = 0.5;
    conc[1] = 0.6;
    conc[2] = 0.7;
    conc[3] = 0.8;
    freqs[0] = 1.0;
    freqs[1] = 0.01;

    const result = CalibrationResult{
        .n_layers = 1,
        .n_q_heads = 2,
        .head_dim = 4,
        .n_bands = 2,
        .rope_theta = 10000.0,
        .q_center_norm = cn,
        .q_center_phase = cp,
        .q_expected_norm = en,
        .concentration = conc,
        .rope_freqs = freqs,
    };

    // Write to a temp file
    const tmp_path = "test_calibration_roundtrip.cal";
    writeCalFile(&result, tmp_path) catch |e| {
        std.debug.print("writeCalFile failed: {}\n", .{e});
        return error.TestFailed;
    };
    defer Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};

    // Read back and verify
    var calibrations = try readCalFile(allocator, std.testing.io, tmp_path);
    defer {
        // Free the backing arrays (they are shared across calibrations)
        if (calibrations.len > 0) {
            // All entries share the same rope_freqs allocation
            allocator.free(calibrations[0].rope_freqs);
            // The backing data arrays are contiguous, so reconstruct and free them.
            // q_center_norm for entry 0 points to the start of the full array.
            const total = calibrations.len * calibrations[0].n_bands;
            allocator.free(calibrations[0].q_center_norm.ptr[0..total]);
            allocator.free(calibrations[0].q_center_phase.ptr[0..total]);
            allocator.free(calibrations[0].q_expected_norm.ptr[0..total]);
            allocator.free(calibrations[0].concentration.ptr[0..total]);
        }
        allocator.free(calibrations);
    }

    try std.testing.expectEqual(@as(usize, 2), calibrations.len);
    try std.testing.expectEqual(@as(usize, 2), calibrations[0].n_bands);

    // Verify data integrity
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), calibrations[0].q_center_norm[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), calibrations[0].q_center_norm[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), calibrations[1].q_center_norm[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), calibrations[1].q_center_norm[1], 1e-6);

    try std.testing.expectApproxEqAbs(@as(f32, 0.5), calibrations[0].concentration[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), calibrations[1].concentration[1], 1e-6);
}
