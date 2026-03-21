const std = @import("std");
const Backend = @import("../src/backend/backend.zig").Backend;
const CpuBackend = @import("../src/backend/cpu.zig").CpuBackend;
const CudaBackend = @import("../src/backend/cuda.zig").CudaBackend;

test "CUDA SDPA warp-parallel dual-delta correctness" {
    const allocator = std.testing.allocator;

    // Skip if CUDA not available
    var cuda_be = CudaBackend.init(allocator) catch |err| {
        if (err == error.CudaNotAvailable or err == error.DeviceNotFound) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer cuda_be.deinit();

    // Test parameters (moderate size to stress warp-parallel softmax)
    const nh = 8;
    const nkv = 2; // GQA
    const hd = 128;
    const seq_len = 256; // Long enough to stress 32-thread parallelism
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    // Allocate test data
    var q = try allocator.alloc(f32, nh * hd);
    defer allocator.free(q);
    var keys = try allocator.alloc(f32, seq_len * nkv * hd);
    defer allocator.free(keys);
    var values = try allocator.alloc(f32, seq_len * nkv * hd);
    defer allocator.free(values);
    var cpu_output = try allocator.alloc(f32, nh * hd);
    defer allocator.free(cpu_output);
    var gpu_output = try allocator.alloc(f32, nh * hd);
    defer allocator.free(gpu_output);
    var oracle_output = try allocator.alloc(f64, nh * hd);
    defer allocator.free(oracle_output);

    // Initialize with deterministic random values
    var prng = std.random.DefaultPrng.init(42);
    const random = prng.random();
    for (q) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (keys) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (values) |*v| v.* = random.float(f32) * 2.0 - 1.0;

    // Compute FP64 oracle (high-precision CPU reference)
    computeOracleSdpa(q, keys, values, oracle_output, nh, nkv, hd, seq_len, scale);

    // Compute CPU SDPA (f32)
    var cpu_be = try CpuBackend.init(allocator);
    defer cpu_be.deinit();
    var cpu_backend: Backend = .{ .cpu = cpu_be };
    cpu_backend.sdpa(
        q.ptr,
        @ptrCast(keys),
        @ptrCast(values),
        q.ptr + (nh - 1) * hd, // k_new = last Q head (dummy)
        q.ptr + (nh - 1) * hd, // v_new = last Q head (dummy)
        cpu_output.ptr,
        nh,
        nkv,
        hd,
        seq_len,
        scale,
        .f32,
    );

    // Compute CUDA SDPA (f32 GPU with warp-parallel softmax)
    var cuda_backend: Backend = .{ .cuda = cuda_be };
    cuda_backend.sdpa(
        q.ptr,
        @ptrCast(keys),
        @ptrCast(values),
        q.ptr + (nh - 1) * hd,
        q.ptr + (nh - 1) * hd,
        gpu_output.ptr,
        nh,
        nkv,
        hd,
        seq_len,
        scale,
        .f32,
    );
    cuda_backend.sync(); // Download GPU results

    // Dual-delta validation
    var max_cpu_err: f32 = 0.0;
    var max_gpu_err: f32 = 0.0;
    for (0..nh * hd) |i| {
        const cpu_err = @abs(cpu_output[i] - @as(f32, @floatCast(oracle_output[i])));
        const gpu_err = @abs(gpu_output[i] - @as(f32, @floatCast(oracle_output[i])));
        max_cpu_err = @max(max_cpu_err, cpu_err);
        max_gpu_err = @max(max_gpu_err, gpu_err);
    }

    std.debug.print("CUDA SDPA warp-parallel dual-delta test:\n", .{});
    std.debug.print("  CPU max error: {e:.2}\n", .{max_cpu_err});
    std.debug.print("  GPU max error: {e:.2}\n", .{max_gpu_err});
    std.debug.print("  GPU/CPU ratio: {d:.2}×\n", .{max_gpu_err / max_cpu_err});

    // Accept GPU if error <= 2× CPU error
    try std.testing.expect(max_gpu_err <= 2.0 * max_cpu_err);
}

fn computeOracleSdpa(
    q: []const f32,
    keys: []const f32,
    values: []const f32,
    output: []f64,
    nh: usize,
    nkv: usize,
    hd: usize,
    seq_len: usize,
    scale: f32,
) void {
    // High-precision FP64 SDPA implementation
    const hpg = nh / nkv;
    for (0..nh) |h| {
        const kvh = h / hpg;
        for (0..hd) |d| {
            var sum: f64 = 0.0;
            var max_score: f64 = -std.math.inf(f64);

            // Compute scores and find max
            var scores: [512]f64 = undefined; // Max seq_len for test
            for (0..seq_len) |t| {
                var dot: f64 = 0.0;
                for (0..hd) |k| {
                    const q_val: f64 = @floatCast(q[h * hd + k]);
                    const k_val: f64 = @floatCast(keys[t * nkv * hd + kvh * hd + k]);
                    dot += q_val * k_val;
                }
                scores[t] = dot * @as(f64, @floatCast(scale));
                max_score = @max(max_score, scores[t]);
            }

            // Softmax numerator and sum
            var exp_sum: f64 = 0.0;
            for (0..seq_len) |t| {
                scores[t] = @exp(scores[t] - max_score);
                exp_sum += scores[t];
            }

            // Weighted sum over values
            for (0..seq_len) |t| {
                const weight = scores[t] / exp_sum;
                const v_val: f64 = @floatCast(values[t * nkv * hd + kvh * hd + d]);
                sum += weight * v_val;
            }
            output[h * hd + d] = sum;
        }
    }
}
