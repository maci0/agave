//! Cross-backend op parity: the elementwise, GEMV and attention ops the decode
//! loop uses, run on a GPU backend and compared against the CPU backend on
//! identical inputs.
//!
//! This exists because full-model output is wrong on Vulkan and ROCm while every
//! per-kernel benchmark looks fine: the benchmarks measure speed, and the only
//! correctness check any of them carried was `gemv_q4_k`. A wrong op in the
//! middle of a 24-layer forward pass only surfaces as garbled text, which
//! nothing asserted on. It replaces tests/test_rocm_kernel.zig, which was a
//! `return error.SkipZigTest` stub with a TODO in it.
//!
//! Everything here currently passes on an RX 7900 XTX, so it does not yet
//! explain docs/TODO.md bug 10. It does pin down what is *not* at fault.
//!
//! Buffer aliasing hazard: the CUDA/ROCm backends cache device buffers in a map
//! keyed by *host pointer address* (`getOrAllocKvBuf`, `act_cache`, `buf_cache`).
//! Freeing a host buffer and allocating another at the same address hands back
//! the stale device buffer, so a test that reuses shapes with fresh allocations
//! reports failures that are the harness's fault, not the kernel's. Allocate
//! once per backend and keep the buffers alive, as this file does.
//!
//! Skips cleanly when the backend is disabled at build time or no device is
//! present, so it is safe in CI on a machine with no GPU.

const std = @import("std");
const builtin = @import("builtin");
const backend_mod = @import("backend");

const Backend = backend_mod.Backend;
const CpuBackend = backend_mod.CpuBackend;
const TensorData = backend_mod.TensorData;

/// Relative error allowed between a GPU op and the CPU reference. GPU kernels
/// legitimately reassociate f32 reductions and may use fused multiply-add, so
/// exact equality is not the bar; a wrong kernel misses this by orders of
/// magnitude, not by an ulp.
const max_rel_err: f32 = 1e-3;
/// Below this magnitude, compare absolutely: relative error is meaningless near zero.
const abs_floor: f32 = 1e-4;
/// RMS norm epsilon, matching what the model layers pass.
const norm_eps: f32 = 1e-6;
/// Vector length for elementwise ops. Deliberately not a multiple of 64 so the
/// tail handling in SIMD and workgroup-tiled kernels is exercised.
const vec_len: usize = 1000;
/// Q8_0: 32 weights per block, one f16 scale then 32 i8 quants.
const q8_0_block_len: usize = 32;
const q8_0_block_bytes: usize = 2 + q8_0_block_len;
/// GEMV shape: small enough to stay fast, large enough for multi-workgroup rows.
const gemv_n: usize = 128;
const gemv_k: usize = 256;
/// SDPA shape: GQA 4:1 at head_dim 128, what Qwen 3.5 runs every decode step.
const sdpa_nh: usize = 8;
const sdpa_nkv: usize = 2;
const sdpa_hd: usize = 128;
const sdpa_seq_len: usize = 256;

fn fillDeterministic(buf: []f32, seed: u64) void {
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    for (buf) |*v| v.* = r.float(f32) * 2.0 - 1.0;
}

fn expectClose(got: []const f32, want: []const f32, label: []const u8) !void {
    try std.testing.expectEqual(want.len, got.len);
    var worst: f32 = 0;
    var worst_i: usize = 0;
    for (got, want, 0..) |g, w, i| {
        if (!std.math.isFinite(g)) {
            std.debug.print("{s}: non-finite value {d} at index {d}\n", .{ label, g, i });
            return error.NonFiniteResult;
        }
        const err = @abs(g - w) / @max(abs_floor, @abs(w));
        if (err > worst) {
            worst = err;
            worst_i = i;
        }
    }
    if (worst > max_rel_err) {
        std.debug.print(
            "{s}: max rel err {d} at index {d} (gpu {d}, cpu {d})\n",
            .{ label, worst, worst_i, got[worst_i], want[worst_i] },
        );
        return error.ParityMismatch;
    }
}

/// Elementwise ops plus RoPE.
fn compareOps(gpu: Backend, cpu: Backend, allocator: std.mem.Allocator) !void {
    const a = try allocator.alloc(f32, vec_len);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, vec_len);
    defer allocator.free(b);
    const got = try allocator.alloc(f32, vec_len);
    defer allocator.free(got);
    const want = try allocator.alloc(f32, vec_len);
    defer allocator.free(want);

    fillDeterministic(a, 0x9E3779B9);
    fillDeterministic(b, 0x85EBCA6B);

    gpu.add(a.ptr, b.ptr, got.ptr, vec_len);
    gpu.sync();
    cpu.add(a.ptr, b.ptr, want.ptr, vec_len);
    try expectClose(got, want, "add");

    gpu.mul(a.ptr, b.ptr, got.ptr, vec_len);
    gpu.sync();
    cpu.mul(a.ptr, b.ptr, want.ptr, vec_len);
    try expectClose(got, want, "mul");

    gpu.silu(a.ptr, got.ptr, vec_len);
    gpu.sync();
    cpu.silu(a.ptr, want.ptr, vec_len);
    try expectClose(got, want, "silu");

    // `b` doubles as the norm weight vector.
    gpu.rmsNorm(a.ptr, b.ptr, got.ptr, vec_len, norm_eps);
    gpu.sync();
    cpu.rmsNorm(a.ptr, b.ptr, want.ptr, vec_len, norm_eps);
    try expectClose(got, want, "rmsNorm");

    @memcpy(got, a);
    @memcpy(want, a);
    gpu.softmax(got.ptr, vec_len);
    gpu.sync();
    cpu.softmax(want.ptr, vec_len);
    try expectClose(got, want, "softmax");

    @memcpy(got, a);
    @memcpy(want, a);
    gpu.l2Norm(got.ptr, vec_len, norm_eps);
    gpu.sync();
    cpu.l2Norm(want.ptr, vec_len, norm_eps);
    try expectClose(got, want, "l2Norm");

    // Rotate a [n_heads * head_dim] query block at a non-zero position so both
    // the sin and cos halves are exercised.
    const head_dim: usize = 64;
    const n_heads: usize = 8;
    const rope_len = head_dim * n_heads;
    const pos: usize = 7;
    const theta: f32 = 10000.0;
    @memcpy(got[0..rope_len], a[0..rope_len]);
    @memcpy(want[0..rope_len], a[0..rope_len]);
    gpu.rope(got.ptr, pos, n_heads, head_dim, head_dim, theta);
    gpu.sync();
    cpu.rope(want.ptr, pos, n_heads, head_dim, head_dim, theta);
    try expectClose(got[0..rope_len], want[0..rope_len], "rope");
}

/// f32 and Q8_0 GEMV. Q8_0 is what the Qwen 3.5 GGUF checkpoints in the test
/// matrix actually use, and it had no correctness check of any kind before.
fn compareGemv(gpu: Backend, cpu: Backend, allocator: std.mem.Allocator) !void {
    const x = try allocator.alloc(f32, gemv_k);
    defer allocator.free(x);
    const got = try allocator.alloc(f32, gemv_n);
    defer allocator.free(got);
    const want = try allocator.alloc(f32, gemv_n);
    defer allocator.free(want);
    fillDeterministic(x, 0xC2B2AE35);

    const wf = try allocator.alloc(f32, gemv_n * gemv_k);
    defer allocator.free(wf);
    fillDeterministic(wf, 0x27D4EB2F);
    const td_f32 = TensorData{ .data = @ptrCast(wf.ptr), .dtype = .f32 };
    gpu.gemv(x.ptr, td_f32, got.ptr, gemv_n, gemv_k);
    gpu.sync();
    cpu.gemv(x.ptr, td_f32, want.ptr, gemv_n, gemv_k);
    try expectClose(got, want, "gemv f32");

    // Build Q8_0 blocks directly so both backends see byte-identical weights.
    const blocks_per_row = gemv_k / q8_0_block_len;
    const wq = try allocator.alloc(u8, gemv_n * blocks_per_row * q8_0_block_bytes);
    defer allocator.free(wq);
    var rng = std.Random.DefaultPrng.init(0x165667B1);
    const r = rng.random();
    var off: usize = 0;
    while (off < wq.len) : (off += q8_0_block_bytes) {
        const scale: f16 = @floatCast(r.float(f32) * 0.02 + 0.001);
        @memcpy(wq[off..][0..2], std.mem.asBytes(&scale));
        for (wq[off + 2 ..][0..q8_0_block_len]) |*q| {
            q.* = @bitCast(@as(i8, @intCast(r.intRangeAtMost(i32, -127, 127))));
        }
    }
    const td_q8 = TensorData{ .data = wq.ptr, .dtype = .q8_0 };
    gpu.gemv(x.ptr, td_q8, got.ptr, gemv_n, gemv_k);
    gpu.sync();
    cpu.gemv(x.ptr, td_q8, want.ptr, gemv_n, gemv_k);
    try expectClose(got, want, "gemv q8_0");
}

/// SDPA driven the way decode drives it: one position appended per call, over
/// the same KV buffers, comparing the GPU output against CPU at every step.
///
/// The shared `sdpa_harness.runDualDeltaTest` pre-fills the KV cache on the host
/// and makes a single call. That is invalid for the CUDA and ROCm backends,
/// whose `getOrAllocKvBuf` allocates device memory without ever uploading host
/// contents: everything before the appended position reads as uninitialised
/// device memory. Appending incrementally is both the fair comparison and the
/// one that matches how the model actually calls this.
fn compareSdpaIncremental(gpu: Backend, cpu: Backend, allocator: std.mem.Allocator) !void {
    const kvd = sdpa_nkv * sdpa_hd;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(sdpa_hd)));
    const kv_floats = sdpa_seq_len * kvd;

    const q = try allocator.alloc(f32, sdpa_nh * sdpa_hd);
    defer allocator.free(q);
    const all_k = try allocator.alloc(f32, kv_floats);
    defer allocator.free(all_k);
    const all_v = try allocator.alloc(f32, kv_floats);
    defer allocator.free(all_v);
    fillDeterministic(q, 0x2545F491);
    fillDeterministic(all_k, 0x14057B7E);
    fillDeterministic(all_v, 0x5851F42D);

    const gpu_k = try allocator.alloc(u8, kv_floats * @sizeOf(f32));
    defer allocator.free(gpu_k);
    const gpu_v = try allocator.alloc(u8, kv_floats * @sizeOf(f32));
    defer allocator.free(gpu_v);
    const cpu_k = try allocator.alloc(u8, kv_floats * @sizeOf(f32));
    defer allocator.free(cpu_k);
    const cpu_v = try allocator.alloc(u8, kv_floats * @sizeOf(f32));
    defer allocator.free(cpu_v);
    @memset(gpu_k, 0);
    @memset(gpu_v, 0);
    @memset(cpu_k, 0);
    @memset(cpu_v, 0);

    const got = try allocator.alloc(f32, sdpa_nh * sdpa_hd);
    defer allocator.free(got);
    const want = try allocator.alloc(f32, sdpa_nh * sdpa_hd);
    defer allocator.free(want);

    for (0..sdpa_seq_len) |pos| {
        const k_new = all_k[pos * kvd ..][0..kvd];
        const v_new = all_v[pos * kvd ..][0..kvd];
        gpu.sdpa(q.ptr, gpu_k, gpu_v, k_new.ptr, v_new.ptr, got.ptr, sdpa_nh, sdpa_nkv, sdpa_hd, pos, scale, .f32, .f32);
        gpu.sync();
        cpu.sdpa(q.ptr, cpu_k, cpu_v, k_new.ptr, v_new.ptr, want.ptr, sdpa_nh, sdpa_nkv, sdpa_hd, pos, scale, .f32, .f32);
        expectClose(got, want, "sdpa") catch |err| {
            std.debug.print("sdpa diverged at position {d}\n", .{pos});
            return err;
        };
    }
}

fn runParity(comptime tag: std.meta.Tag(Backend), comptime name: []const u8) !void {
    const allocator = std.testing.allocator;

    const BackendType = @typeInfo(@FieldType(Backend, @tagName(tag))).pointer.child;
    var gpu_impl = BackendType.init(allocator, 0) catch |err| {
        std.debug.print("{s}: unavailable ({t}), skipping\n", .{ name, err });
        return error.SkipZigTest;
    };
    defer gpu_impl.deinit();
    var cpu_impl: CpuBackend = .{};

    const gpu = @unionInit(Backend, @tagName(tag), &gpu_impl);
    const cpu = Backend{ .cpu = &cpu_impl };
    try compareOps(gpu, cpu, allocator);
    try compareGemv(gpu, cpu, allocator);
    try compareSdpaIncremental(gpu, cpu, allocator);
}

test "Vulkan matches CPU on every decode-loop op" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    try runParity(.vulkan, "Vulkan");
}

test "ROCm matches CPU on every decode-loop op" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    try runParity(.rocm, "ROCm");
}
