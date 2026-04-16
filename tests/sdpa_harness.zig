//! Shared dual-delta SDPA test harness for GPU backend correctness tests.
//! Compares a GPU backend's SDPA output against both a CPU baseline and an FP64
//! oracle, validating that GPU error stays within 2x of CPU error (dual-delta).
//!
//! Used by test_cuda_sdpa.zig and test_metal_sdpa.zig.

const std = @import("std");
const backend = @import("backend");
const Backend = backend.Backend;
const CpuBackend = backend.CpuBackend;
const computeOracleSdpa = @import("sdpa_oracle").computeOracleSdpa;

/// Run the dual-delta SDPA correctness test for a GPU backend.
///
/// Generates deterministic random test data, computes the FP64 oracle reference,
/// runs both CPU and GPU SDPA, then validates that:
///   1. Both CPU and GPU absolute error < 1e-3
///   2. GPU error ≤ max(2× CPU error, 1e-6)
///
/// Parameters:
///   - gpu_backend: Initialized Backend union wrapping the GPU backend under test.
///   - nh: Number of query heads.
///   - nkv: Number of KV heads (GQA when nkv < nh).
///   - hd: Head dimension.
///   - seq_len: Sequence length (number of KV positions).
///   - label: Human-readable label for diagnostic output (e.g., "CUDA warp-parallel").
pub fn runDualDeltaTest(
    gpu_backend: *Backend,
    nh: usize,
    nkv: usize,
    hd: usize,
    seq_len: usize,
    label: []const u8,
) !void {
    const allocator = std.testing.allocator;
    const kvd = nkv * hd;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    // Generate deterministic random test data
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const q = try allocator.alloc(f32, nh * hd);
    defer allocator.free(q);
    for (q) |*v| v.* = random.float(f32) * 2.0 - 1.0;

    // All keys/values across all seq_len positions (for oracle comparison)
    const all_keys = try allocator.alloc(f32, seq_len * kvd);
    defer allocator.free(all_keys);
    for (all_keys) |*v| v.* = random.float(f32) * 2.0 - 1.0;

    const all_values = try allocator.alloc(f32, seq_len * kvd);
    defer allocator.free(all_values);
    for (all_values) |*v| v.* = random.float(f32) * 2.0 - 1.0;

    // FP64 oracle (high-precision reference over all seq_len positions)
    const oracle_output = try allocator.alloc(f64, nh * hd);
    defer allocator.free(oracle_output);
    try computeOracleSdpa(q, all_keys, all_values, oracle_output, nh, nkv, hd, seq_len, scale);

    // Allocate KV cache byte buffers (f32 format, enough for all positions).
    // sdpa() appends the last position internally, so pre-fill 0..seq_len-2.
    const kv_bytes = seq_len * kvd * @sizeOf(f32);
    const cpu_kv_keys = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(cpu_kv_keys);
    const cpu_kv_values = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(cpu_kv_values);
    const gpu_kv_keys = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(gpu_kv_keys);
    const gpu_kv_values = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(gpu_kv_values);

    // Pre-fill positions 0..seq_len-2 in both CPU and GPU KV caches
    const pre_fill = seq_len - 1;
    const pre_bytes = pre_fill * kvd * @sizeOf(f32);
    const pre_key_bytes = std.mem.sliceAsBytes(all_keys[0 .. pre_fill * kvd]);
    const pre_val_bytes = std.mem.sliceAsBytes(all_values[0 .. pre_fill * kvd]);
    @memcpy(cpu_kv_keys[0..pre_bytes], pre_key_bytes);
    @memcpy(cpu_kv_values[0..pre_bytes], pre_val_bytes);
    @memcpy(gpu_kv_keys[0..pre_bytes], pre_key_bytes);
    @memcpy(gpu_kv_values[0..pre_bytes], pre_val_bytes);

    // k_new/v_new: the last position, appended by sdpa()
    const k_new = all_keys[pre_fill * kvd ..][0..kvd];
    const v_new = all_values[pre_fill * kvd ..][0..kvd];

    // Compute CPU SDPA (f32)
    var cpu_be: CpuBackend = .{};
    var cpu_backend: Backend = .{ .cpu = &cpu_be };
    const cpu_output = try allocator.alloc(f32, nh * hd);
    defer allocator.free(cpu_output);
    cpu_backend.sdpa(q.ptr, cpu_kv_keys, cpu_kv_values, k_new.ptr, v_new.ptr, cpu_output.ptr, nh, nkv, hd, pre_fill, scale, .f32, .f32);

    // Compute GPU SDPA
    const gpu_output = try allocator.alloc(f32, nh * hd);
    defer allocator.free(gpu_output);
    gpu_backend.sdpa(q.ptr, gpu_kv_keys, gpu_kv_values, k_new.ptr, v_new.ptr, gpu_output.ptr, nh, nkv, hd, pre_fill, scale, .f32, .f32);
    gpu_backend.sync();

    // Dual-delta validation
    var max_cpu_err: f32 = 0.0;
    var max_gpu_err: f32 = 0.0;
    for (0..nh * hd) |i| {
        // Reject NaN/Inf in outputs — these indicate kernel bugs, not rounding error
        if (std.math.isNan(cpu_output[i]) or std.math.isInf(cpu_output[i])) {
            std.debug.print("CPU output[{d}] is NaN or Inf\n", .{i});
            return error.TestFailed;
        }
        if (std.math.isNan(gpu_output[i]) or std.math.isInf(gpu_output[i])) {
            std.debug.print("GPU output[{d}] is NaN or Inf\n", .{i});
            return error.TestFailed;
        }
        const cpu_err = @abs(cpu_output[i] - @as(f32, @floatCast(oracle_output[i])));
        const gpu_err = @abs(gpu_output[i] - @as(f32, @floatCast(oracle_output[i])));
        max_cpu_err = @max(max_cpu_err, cpu_err);
        max_gpu_err = @max(max_gpu_err, gpu_err);
    }

    std.debug.print("{s} dual-delta test:\n", .{label});
    std.debug.print("  CPU max error: {e:.2}\n", .{max_cpu_err});
    std.debug.print("  GPU max error: {e:.2}\n", .{max_gpu_err});
    if (max_cpu_err > 0) {
        std.debug.print("  GPU/CPU ratio: {d:.2}x\n", .{max_gpu_err / max_cpu_err});
    } else {
        std.debug.print("  GPU/CPU ratio: N/A (CPU exact)\n", .{});
    }

    // FP32 SDPA accumulates rounding error across seq_len dot products + softmax.
    // 1e-3 absolute tolerance is ~10 bits of precision (2^-10 ~ 1e-3) — well within
    // FP32's 23-bit mantissa, accounting for sequential accumulation.
    try std.testing.expect(max_cpu_err < 1e-3);
    try std.testing.expect(max_gpu_err < 1e-3);
    // GPU uses different reduction order than CPU (warp-parallel or tiled),
    // so allow up to 2x CPU error to account for non-associative FP addition.
    // Floor of 1e-6 prevents overly strict comparison when CPU error is near-zero.
    try std.testing.expect(max_gpu_err <= @max(2.0 * max_cpu_err, 1e-6));
}
