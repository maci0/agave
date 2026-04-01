//! CPU backend with SIMD-optimized kernels.
//! Supports optional multi-threaded GEMV via ThreadPool.

const std = @import("std");
const builtin = @import("builtin");
const TensorData = @import("backend.zig").TensorData;
const DType = @import("backend.zig").DType;
const quant = @import("../ops/quant.zig");
const ThreadPool = @import("../thread_pool.zig").ThreadPool;
const gemv_kernel = @import("kernels/cpu/gemv.zig");
const emb_kernel = @import("kernels/cpu/embedding.zig");
const norm_kernel = @import("kernels/cpu/norm.zig");
const softmax_kernel = @import("kernels/cpu/softmax.zig");
const rope_kernel = @import("kernels/cpu/rope.zig");
const activation_kernel = @import("kernels/cpu/activation.zig");
const elementwise_kernel = @import("kernels/cpu/elementwise.zig");
const sdpa_kernel = @import("kernels/cpu/sdpa.zig");
const sdpa_prefill_kernel = @import("kernels/cpu/sdpa_prefill.zig");
const gemm_kernel = @import("kernels/cpu/gemm.zig");
const deltanet_kernel = @import("kernels/cpu/deltanet.zig");

// ── Buffer sizes for system detection ─────────────────────────────
const cpu_model_buf_size: usize = 128;
const cpuinfo_read_buf_size: usize = 4096;
const meminfo_read_buf_size: usize = 1024;
const memavail_read_buf_size: usize = 2048;
/// Bytes per kilobyte — used for /proc/meminfo and sysfs cache size parsing.
const kb_to_bytes: usize = 1024;
/// Bytes per megabyte — used for sysfs cache size parsing.
const mb_to_bytes: usize = 1024 * 1024;

// ── CPU model detection ─────────────────────────────────────────

var cpu_model_buf: [cpu_model_buf_size]u8 = .{0} ** cpu_model_buf_size;
var cpu_model_len: usize = 0;
var cpu_model_detected: bool = false;

/// Detect CPU model name from the OS. Called once at first backendInfo() call.
fn detectCpuModel() []const u8 {
    if (cpu_model_detected) return cpu_model_buf[0..cpu_model_len];
    cpu_model_detected = true;

    if (comptime builtin.os.tag == .macos) {
        // macOS: sysctlbyname("machdep.cpu.brand_string")
        var len: usize = cpu_model_buf.len;
        const rc = std.c.sysctlbyname("machdep.cpu.brand_string", &cpu_model_buf, &len, null, 0);
        if (rc == 0 and len > 0) {
            // Strip trailing null
            cpu_model_len = if (cpu_model_buf[len - 1] == 0) len - 1 else len;
            return cpu_model_buf[0..cpu_model_len];
        }
    } else if (comptime builtin.os.tag == .linux) {
        // Linux: parse /proc/cpuinfo for "model name"
        const file = std.fs.openFileAbsolute("/proc/cpuinfo", .{}) catch return "";
        defer file.close();
        var read_buf: [cpuinfo_read_buf_size]u8 = undefined;
        const n = file.read(&read_buf) catch return "";
        const data = read_buf[0..n];
        const needle = "model name\t: ";
        if (std.mem.indexOf(u8, data, needle)) |pos| {
            const start = pos + needle.len;
            const end = std.mem.indexOfScalarPos(u8, data, start, '\n') orelse data.len;
            const name_len = @min(end - start, cpu_model_buf.len);
            @memcpy(cpu_model_buf[0..name_len], data[start..][0..name_len]);
            cpu_model_len = name_len;
            return cpu_model_buf[0..cpu_model_len];
        }
    }
    return "";
}

// ── System memory & cache detection ──────────────────────────────

/// Read a u64 value from a macOS sysctl by name. Returns 0 on failure.
fn sysctlU64(comptime name: [*:0]const u8) usize {
    if (comptime builtin.os.tag != .macos) return 0;
    var val: u64 = 0;
    var len: usize = @sizeOf(u64);
    const rc = std.c.sysctlbyname(name, @ptrCast(&val), &len, null, 0);
    if (rc == 0) return @intCast(val);
    return 0;
}

/// Parse a Linux sysfs cache size file (e.g., "32K", "4096K", "16M").
fn parseSysfsCacheSize(comptime path: []const u8) usize {
    if (comptime builtin.os.tag != .linux) return 0;
    const file = std.fs.openFileAbsolute(path, .{}) catch return 0;
    defer file.close();
    var buf: [32]u8 = undefined;
    const n = file.read(&buf) catch return 0;
    const data = std.mem.trimRight(u8, buf[0..n], "\n ");
    if (data.len == 0) return 0;
    // Parse numeric prefix
    var val: usize = 0;
    var i: usize = 0;
    while (i < data.len and data[i] >= '0' and data[i] <= '9') : (i += 1) {
        val = val * 10 + (data[i] - '0');
    }
    // Check suffix: K or M
    if (i < data.len) {
        if (data[i] == 'K') return val * kb_to_bytes;
        if (data[i] == 'M') return val * mb_to_bytes;
    }
    return val;
}

/// Detect total system physical memory in bytes.
pub fn detectSystemMem() usize {
    if (comptime builtin.os.tag == .macos) {
        return sysctlU64("hw.memsize");
    } else if (comptime builtin.os.tag == .linux) {
        const file = std.fs.openFileAbsolute("/proc/meminfo", .{}) catch return 0;
        defer file.close();
        var read_buf: [meminfo_read_buf_size]u8 = undefined;
        const n = file.read(&read_buf) catch return 0;
        const data = read_buf[0..n];
        const needle = "MemTotal:";
        if (std.mem.indexOf(u8, data, needle)) |pos| {
            var i = pos + needle.len;
            while (i < data.len and data[i] == ' ') i += 1;
            var val: usize = 0;
            while (i < data.len and data[i] >= '0' and data[i] <= '9') : (i += 1) {
                val = val * 10 + (data[i] - '0');
            }
            return val * kb_to_bytes; // kB to bytes
        }
    }
    return 0;
}

/// Detect available (free) system memory in bytes.
pub fn detectAvailMem() usize {
    if (comptime builtin.os.tag == .macos) {
        // vm.page_free_count × hw.pagesize — conservative (free pages only)
        const free_pages = sysctlU64("vm.page_free_count");
        const page_size = sysctlU64("hw.pagesize");
        if (free_pages > 0 and page_size > 0) return free_pages * page_size;
        return 0;
    } else if (comptime builtin.os.tag == .linux) {
        const file = std.fs.openFileAbsolute("/proc/meminfo", .{}) catch return 0;
        defer file.close();
        var read_buf: [memavail_read_buf_size]u8 = undefined;
        const n = file.read(&read_buf) catch return 0;
        const data = read_buf[0..n];
        const needle = "MemAvailable:";
        if (std.mem.indexOf(u8, data, needle)) |pos| {
            var i = pos + needle.len;
            while (i < data.len and data[i] == ' ') i += 1;
            var val: usize = 0;
            while (i < data.len and data[i] >= '0' and data[i] <= '9') : (i += 1) {
                val = val * 10 + (data[i] - '0');
            }
            return val * kb_to_bytes; // kB to bytes
        }
    }
    return 0;
}

const CacheSizes = @import("backend.zig").CacheSizes;

/// Detect CPU cache sizes (L1 data, L2, L3) in bytes.
pub fn detectCacheSizes() CacheSizes {
    if (comptime builtin.os.tag == .macos) {
        return .{
            .l1 = sysctlU64("hw.l1dcachesize"),
            .l2 = sysctlU64("hw.l2cachesize"),
            .l3 = sysctlU64("hw.l3cachesize"),
        };
    } else if (comptime builtin.os.tag == .linux) {
        return .{
            .l1 = parseSysfsCacheSize("/sys/devices/system/cpu/cpu0/cache/index0/size"),
            .l2 = parseSysfsCacheSize("/sys/devices/system/cpu/cpu0/cache/index2/size"),
            .l3 = parseSysfsCacheSize("/sys/devices/system/cpu/cpu0/cache/index3/size"),
        };
    }
    return .{};
}

// ── OS version detection ─────────────────────────────────────────

const os_version_buf_size: usize = 128;
/// Length of the "macOS " / "Linux " prefix prepended to OS version strings.
const os_prefix_len: usize = 6;
var os_version_buf: [os_version_buf_size]u8 = .{0} ** os_version_buf_size;
var os_version_len: usize = 0;
var os_version_detected: bool = false;

/// Detect OS version string. Returns "macOS 14.2.1" or "Linux 6.5.0" style strings.
pub fn detectOsVersion() []const u8 {
    if (os_version_detected) return os_version_buf[0..os_version_len];
    os_version_detected = true;

    if (comptime builtin.os.tag == .macos) {
        // macOS: Try kern.osproductversion first (e.g., "14.2.1"), fall back to kern.osrelease
        var len: usize = os_version_buf.len - os_prefix_len;
        const rc = std.c.sysctlbyname("kern.osproductversion", os_version_buf[os_prefix_len..].ptr, &len, null, 0);
        if (rc == 0 and len > 0) {
            @memcpy(os_version_buf[0..os_prefix_len], "macOS ");
            // Strip trailing null
            const total_len = os_prefix_len + (if (os_version_buf[os_prefix_len + len - 1] == 0) len - 1 else len);
            os_version_len = total_len;
            return os_version_buf[0..os_version_len];
        }
    } else if (comptime builtin.os.tag == .linux) {
        // Linux: Use uname to get kernel release (e.g., "6.5.0-14-generic")
        const uts = std.posix.uname();
        @memcpy(os_version_buf[0..os_prefix_len], "Linux ");
        // uts.release is a null-terminated array; find the null
        const release_slice = std.mem.sliceTo(&uts.release, 0);
        const copy_len = @min(release_slice.len, os_version_buf.len - os_prefix_len);
        @memcpy(os_version_buf[os_prefix_len..][0..copy_len], release_slice[0..copy_len]);
        os_version_len = os_prefix_len + copy_len;
        return os_version_buf[0..os_version_len];
    }
    return "";
}

// ── Autotune constants ───────────────────────────────────────────
// These can be overridden by the grid search in research/kernels/autotune.py
// to find the optimal value for each target platform.
const softmax_width: comptime_int = 8; // SIMD width for softmax: 4, 8, or 16

// ── Parallel computation constants ──────────────────────────────
/// Minimum output rows to justify thread pool dispatch overhead.
const parallel_min_rows: usize = 32;
/// Row granularity for work-stealing (aligned to 4-row batch size).
const parallel_grain: usize = 16;
/// Maximum number of SSM v-heads for DeltaNet stack buffers.
const max_deltanet_v_heads: usize = 128;
/// Minimum v-heads to parallelize DeltaNet recurrence across the thread pool.
const deltanet_parallel_min_heads: usize = 4;

/// CPU backend with SIMD-optimized compute kernels.
/// Provides fallback implementations for all backend operations using
/// 8-wide SIMD vectors (V8) where beneficial. Supports all DType
/// quantization formats (Q2-Q8, BF16, F16, F32, FP8, MXFP4, NVFP4, TQ1).
pub const CpuBackend = struct {
    /// Optional thread pool for parallel GEMV. Null = single-threaded.
    pool: ?*ThreadPool = null,

    /// Allocate a KV cache slice — plain allocator on CPU. `n` is byte count.
    pub fn allocKvSlice(_: *CpuBackend, allocator: std.mem.Allocator, n: usize) error{OutOfMemory}![]u8 {
        return allocator.alloc(u8, n);
    }

    /// Free a KV cache slice allocated via allocKvSlice.
    pub fn freeKvSlice(_: *CpuBackend, allocator: std.mem.Allocator, slice: []u8) void {
        allocator.free(slice);
    }

    /// Performs general matrix-vector multiplication: y = W @ x.
    /// When a thread pool is available and n >= parallel_min_rows,
    /// rows are distributed across worker threads automatically.
    pub fn gemv(self: *CpuBackend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        if (self.pool) |pool| {
            if (n >= parallel_min_rows) {
                const rb = gemvRowBytes(w.dtype, k);
                if (rb > 0) {
                    var ctx = GemvCtx{
                        .x = x,
                        .w_data = w.data,
                        .y = y,
                        .k = k,
                        .row_bytes = rb,
                        .dtype = w.dtype,
                    };
                    pool.parallelFor(n, parallel_grain, @ptrCast(&ctx), GemvCtx.work);
                    return;
                }
            }
        }
        gemvSeq(x, w, y, n, k);
    }

    /// Context for parallel GEMV dispatch. Each worker processes a slice of rows.
    const GemvCtx = struct {
        x: [*]const f32,
        w_data: [*]const u8,
        y: [*]f32,
        k: usize,
        row_bytes: usize,
        dtype: DType,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *const GemvCtx = @ptrCast(@alignCast(ctx_ptr));
            const n_rows = end - start;
            const w_off = ctx.w_data + start * ctx.row_bytes;
            gemv_kernel.gemvSeq(ctx.x, w_off, ctx.dtype, ctx.y + start, n_rows, ctx.k);
        }
    };

    /// Bytes per output row for each quantization format.
    /// Returns 0 for unsupported formats (fallback to sequential).
    const gemvRowBytes = gemv_kernel.gemvRowBytes;

    /// Sequential GEMV — delegates to gemv_kernel for dtype-specific dequantization.
    fn gemvSeq(x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        gemv_kernel.gemvSeq(x, w.data, w.dtype, y, n, k);
    }

    /// Applies Root Mean Square Layer Normalization: output[i] = input[i] * weight[i] / rms(input).
    pub fn rmsNorm(self: *CpuBackend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        _ = self;
        norm_kernel.rmsNorm(input, weight, output, n, eps);
    }

    /// Fused add + rms_norm: a[i] = a[i] + b[i], output = rms_norm(a+b, weight, eps).
    pub fn addRmsNorm(self: *CpuBackend, a: [*]f32, b: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        _ = self;
        norm_kernel.addRmsNorm(a, b, weight, output, n, eps);
    }

    /// Applies SiLU (Swish) activation: x * sigmoid(x).
    pub fn silu(self: *CpuBackend, input: [*]const f32, output: [*]f32, n: usize) void {
        _ = self;
        activation_kernel.silu(input, output, n);
    }

    /// Applies GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))).
    pub fn gelu(self: *CpuBackend, input: [*]const f32, output: [*]f32, n: usize) void {
        _ = self;
        activation_kernel.gelu(input, output, n);
    }

    /// Element-wise addition: out[i] = a[i] + b[i].
    pub fn add(self: *CpuBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        _ = self;
        elementwise_kernel.add(a, b, out, n);
    }

    /// Transposed GEMV: y[out_dim] = W^T @ x[in_dim] for Q8_0 3D weights.
    pub fn gemvT(_: *CpuBackend, x: [*]const f32, w: [*]const u8, y: [*]f32, out_dim: usize, in_dim: usize) void {
        const blocks_per_row = (out_dim + quant.quant_block_elems - 1) / quant.quant_block_elems;
        @memset(y[0..out_dim], 0);
        for (0..in_dim) |j| {
            const xj = x[j];
            const row_base = j * blocks_per_row * quant.q8_0_block_bytes;
            for (0..out_dim) |i| {
                const blk_idx = i / quant.quant_block_elems;
                const blk_off = i % quant.quant_block_elems;
                const blk_ptr = w + row_base + blk_idx * quant.q8_0_block_bytes;
                const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, blk_ptr[0..2], .little))));
                const val: i8 = @bitCast(blk_ptr[2 + blk_off]);
                y[i] += @as(f32, @floatFromInt(val)) * scale * xj;
            }
        }
    }

    /// Scaled accumulate: dst[i] += src[i] * scale. SIMD-optimized with V8.
    pub fn addScaled(_: *CpuBackend, src: [*]const f32, dst: [*]f32, scale: f32, n: usize) void {
        const V8 = @Vector(8, f32);
        const sv: V8 = @splat(scale);
        var i: usize = 0;
        while (i + 8 <= n) : (i += 8) {
            const s: V8 = src[i..][0..8].*;
            const d: V8 = dst[i..][0..8].*;
            dst[i..][0..8].* = @mulAdd(V8, s, sv, d);
        }
        while (i < n) : (i += 1) {
            dst[i] = @mulAdd(f32, src[i], scale, dst[i]);
        }
    }

    /// Fused SiLU + multiply: out[i] = silu(a[i]) * b[i].
    pub fn siluMul(self: *CpuBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        _ = self;
        activation_kernel.siluMul(a, b, out, n);
    }

    /// GELU + multiply: out[i] = gelu(a[i]) * b[i].
    /// Sequential two-pass (gelu then mul) — not fused. Acceptable for CPU
    /// where elementwise ops are fast relative to GEMV bottleneck.
    pub fn geluMul(self: *CpuBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        _ = self;
        activation_kernel.gelu(a, out, n);
        elementwise_kernel.mul(out, b, out, n);
    }

    /// In-place per-head rmsNorm: applies same weight to n_heads independent heads.
    pub fn rmsNormMulti(self: *CpuBackend, data: [*]f32, weight: [*]const f32, n_heads: usize, head_dim: usize, eps: f32) void {
        for (0..n_heads) |h| {
            self.rmsNorm(data + h * head_dim, weight, data + h * head_dim, head_dim, eps);
        }
    }

    /// Element-wise multiplication: out[i] = a[i] * b[i].
    pub fn mul(self: *CpuBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        _ = self;
        elementwise_kernel.mul(a, b, out, n);
    }

    /// Applies softmax normalization in-place.
    pub fn softmax(self: *CpuBackend, data: [*]f32, n: usize) void {
        _ = self;
        softmax_kernel.softmaxSimd(softmax_width, data, n);
    }

    /// Applies Rotary Position Embedding (RoPE) in-place.
    pub fn rope(self: *CpuBackend, x: [*]f32, pos: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        _ = self;
        rope_kernel.rope(x, pos, n_heads, head_dim, rope_dim, theta);
    }

    /// Looks up a token embedding row and dequantizes to f32.
    pub fn embLookup(self: *CpuBackend, table: TensorData, token_id: u32, output: [*]f32, dim: usize) void {
        _ = self;
        emb_kernel.embLookup(table.data, table.dtype, token_id, output, dim);
    }

    /// L2 normalizes a vector in-place: x[i] /= sqrt(sum(x^2) + eps).
    pub fn l2Norm(self: *CpuBackend, x: [*]f32, n: usize, eps: f32) void {
        _ = self;
        norm_kernel.l2Norm(x, n, eps);
    }

    /// In-place sigmoid-gated multiply: data[i] *= sigmoid(gate[i]).
    pub fn sigmoidMul(self: *CpuBackend, data: [*]f32, gate: [*]const f32, n: usize) void {
        _ = self;
        elementwise_kernel.sigmoidMul(data, gate, n);
    }

    /// De-interleave paired blocks on CPU.
    pub fn deinterleave(self: *CpuBackend, input: [*]const f32, out_a: [*]f32, out_b: [*]f32, stride: usize, n_pairs: usize) void {
        _ = self;
        elementwise_kernel.deinterleave(input, out_a, out_b, stride, n_pairs);
    }

    /// Split concatenated Q+gate per-head data into separate Q and gate arrays.
    /// Input layout: [Q0..Q_{hd-1}, G0..G_{hd-1}] × nh heads.
    /// Output: q_out[nh*hd], g_out[nh*hd].
    pub fn splitQGate(_: *CpuBackend, qg: [*]const f32, q_out: [*]f32, g_out: [*]f32, hd: usize, nh: usize) void {
        for (0..nh) |h| {
            const src = h * hd * 2;
            const dst = h * hd;
            @memcpy(q_out[dst..][0..hd], qg[src..][0..hd]);
            @memcpy(g_out[dst..][0..hd], qg[src + hd ..][0..hd]);
        }
    }

    /// No-op on CPU. GPU backends flush pending commands here;
    /// CPU ops are immediately visible, so no flush is needed.
    pub fn sync(self: *CpuBackend) void {
        _ = self;
    }

    /// No-op on CPU — no GPU dispatch batching needed.
    pub fn beginBatch(self: *CpuBackend) void {
        _ = self;
    }

    /// No-op on CPU — no GPU dispatch batching needed.
    pub fn endBatch(self: *CpuBackend) void {
        _ = self;
    }

    /// Returns backend information for display. Cache sizes and total memory
    /// are detected once and cached; available memory is always fresh.
    pub fn backendInfo(_: *const CpuBackend) @import("backend.zig").BackendInfo {
        const Static = struct {
            var caches: CacheSizes = .{};
            var sys_mem: usize = 0;
            var detected: bool = false;
        };
        if (!Static.detected) {
            Static.caches = detectCacheSizes();
            Static.sys_mem = detectSystemMem();
            Static.detected = true;
        }
        const avail = detectAvailMem();
        return .{
            .name = "CPU",
            .device_name = detectCpuModel(),
            .total_mem = Static.sys_mem,
            .avail_mem = avail,
            .system_mem = Static.sys_mem,
            .system_avail = avail,
            .l1_cache = Static.caches.l1,
            .l2_cache = Static.caches.l2,
            .l3_cache = Static.caches.l3,
        };
    }

    /// NVFP4 SafeTensors GEMV: separate weight nibble + FP8 E4M3 scale arrays.
    pub fn gemvNvfp4St(self: *CpuBackend, x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: usize, k: usize) void {
        _ = self;
        quant.gemvNvfp4St(x, weight, scale, y, n, k);
    }

    /// MLX affine quantized GEMV: packed integer weights + bf16 scales/biases.
    /// When a thread pool is available, parallelizes across output rows.
    pub fn gemvMlxQ(self: *CpuBackend, x: [*]const f32, weight: [*]const u8, scales: [*]const u8, biases: [*]const u8, y: [*]f32, n: usize, k: usize, bits: u32) void {
        const mlx_ops = @import("../ops/mlx.zig");
        if (self.pool) |pool| {
            if (n >= parallel_min_rows) {
                var ctx = MlxGemvCtx{
                    .x = x,
                    .pw = @ptrCast(@alignCast(weight)),
                    .sc = @ptrCast(@alignCast(scales)),
                    .bi = @ptrCast(@alignCast(biases)),
                    .y = y,
                    .k = k,
                    .bits = bits,
                };
                pool.parallelFor(n, parallel_grain, @ptrCast(&ctx), MlxGemvCtx.work);
                return;
            }
        }
        mlx_ops.mlxGemvRows(x, @ptrCast(@alignCast(weight)), @ptrCast(@alignCast(scales)), @ptrCast(@alignCast(biases)), y, 0, n, k, bits);
    }

    /// Context for parallel MLX GEMV dispatch.
    const MlxGemvCtx = struct {
        x: [*]const f32,
        pw: [*]const u32,
        sc: [*]const u16,
        bi: [*]const u16,
        y: [*]f32,
        k: usize,
        bits: u32,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *const MlxGemvCtx = @ptrCast(@alignCast(ctx_ptr));
            const mlx = @import("../ops/mlx.zig");
            mlx.mlxGemvRows(ctx.x, ctx.pw, ctx.sc, ctx.bi, ctx.y, start, end - start, ctx.k, ctx.bits);
        }
    };

    /// MXFP4 SafeTensors GEMV (U32-packed nibbles, E8M0 scales, no bias).
    pub fn gemvMxfp4St(self: *CpuBackend, x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: usize, k: usize) void {
        const mlx_ops = @import("../ops/mlx.zig");
        if (self.pool) |pool| {
            if (n >= parallel_min_rows) {
                var ctx = Mxfp4StCtx{
                    .x = x,
                    .pw = @ptrCast(@alignCast(weight)),
                    .scales_u8 = scale,
                    .y = y,
                    .k = k,
                };
                pool.parallelFor(n, parallel_grain, @ptrCast(&ctx), Mxfp4StCtx.work);
                return;
            }
        }
        mlx_ops.mlxMxfp4GemvRows(x, @ptrCast(@alignCast(weight)), scale, y, 0, n, k);
    }

    const Mxfp4StCtx = struct {
        x: [*]const f32,
        pw: [*]const u32,
        scales_u8: [*]const u8,
        y: [*]f32,
        k: usize,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *const Mxfp4StCtx = @ptrCast(@alignCast(ctx_ptr));
            const mlx = @import("../ops/mlx.zig");
            mlx.mlxMxfp4GemvRows(ctx.x, ctx.pw, ctx.scales_u8, ctx.y, start, end - start, ctx.k);
        }
    };

    /// Batched GEMV — fuses all ops into a single parallelFor to minimize
    /// thread wake/sleep overhead (~250 GEMV dispatches per token).
    pub fn gemvMulti(self: *CpuBackend, x: [*]const f32, ops: []const @import("backend.zig").GemvOp, k: usize) void {
        if (ops.len == 0) return;

        // Check if all ops can be parallelized (same dtype, known row bytes)
        if (self.pool) |pool| {
            const dtype = ops[0].w.dtype;
            const rb = gemvRowBytes(dtype, k);
            var total_n: usize = 0;
            var all_same = rb > 0;
            for (ops) |op| {
                total_n += op.n;
                if (op.w.dtype != dtype or op.mlx_scales != null) all_same = false;
            }

            if (all_same and total_n >= parallel_min_rows) {
                var ctx = GemvMultiCtx{
                    .x = x,
                    .k = k,
                    .row_bytes = rb,
                    .dtype = dtype,
                    .ops = ops,
                };
                pool.parallelFor(total_n, parallel_grain, @ptrCast(&ctx), GemvMultiCtx.work);
                return;
            }
        }

        // Fallback: sequential per-op dispatch
        for (ops) |op| {
            if (op.mlx_scales != null) {
                const mlx_ops = @import("../ops/mlx.zig");
                mlx_ops.mlxGemvRaw(x, @ptrCast(@alignCast(op.w.data)), @ptrCast(@alignCast(op.mlx_scales.?)), @ptrCast(@alignCast(op.mlx_biases.?)), op.y, op.n, k, op.mlx_bits);
            } else {
                self.gemv(x, op.w, op.y, op.n, k);
            }
        }
    }

    /// Context for batched parallel GEMV. Maps virtual row indices to specific ops.
    const GemvMultiCtx = struct {
        x: [*]const f32,
        k: usize,
        row_bytes: usize,
        dtype: DType,
        ops: []const @import("backend.zig").GemvOp,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *const GemvMultiCtx = @ptrCast(@alignCast(ctx_ptr));
            // Map virtual row range [start, end) to specific ops
            var pos: usize = 0;
            for (ctx.ops) |op| {
                const op_end = pos + op.n;
                if (start < op_end and end > pos) {
                    // This chunk overlaps with this op
                    const local_start = if (start > pos) start - pos else 0;
                    const local_end = if (end < op_end) end - pos else op.n;
                    const n_rows = local_end - local_start;
                    const w_off = op.w.data + local_start * ctx.row_bytes;
                    gemv_kernel.gemvSeq(ctx.x, w_off, ctx.dtype, op.y + local_start, n_rows, ctx.k);
                }
                pos = op_end;
                if (pos >= end) break;
            }
        }
    };

    /// GEMM: Y[n_tok × n_out] = X[n_tok × n_in] @ W[n_out × n_in]^T.
    /// Each token's GEMV dispatches through the thread pool for parallelism.
    pub fn gemm(self: *CpuBackend, x: [*]const f32, w: TensorData, y: [*]f32, n_tok: usize, n_out: usize, n_in: usize) void {
        for (0..n_tok) |t| {
            self.gemv(x + t * n_in, w, y + t * n_out, n_out, n_in);
        }
    }

    /// Apply RMS normalization independently to each of n_tok rows.
    pub fn rmsNormBatched(self: *CpuBackend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n_tok: usize, dim: usize, eps: f32) void {
        for (0..n_tok) |t| self.rmsNorm(input + t * dim, weight, output + t * dim, dim, eps);
    }

    /// Apply RoPE to n_tok vectors at positions[0..n_tok].
    pub fn ropeBatched(_: *CpuBackend, x: [*]f32, positions: [*]const u32, n_tok: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        const stride = n_heads * head_dim;
        for (0..n_tok) |t| rope_kernel.rope(x + t * stride, positions[t], n_heads, head_dim, rope_dim, theta);
    }

    /// Prefill attention: causal self-attention for n_tok new tokens.
    /// Appends all KV data in bulk, then computes attention per token
    /// with parallel head dispatch via the thread pool.
    pub fn sdpaPrefill(self: *CpuBackend, q: [*]const f32, k: [*]const f32, v: [*]const f32, kv_keys: []u8, kv_values: []u8, output: [*]f32, nh: usize, nkv: usize, hd: usize, prev_len: usize, n_tok: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        const kvd = nkv * hd;

        // Bulk KV append: store all n_tok key/value vectors into the cache
        for (0..n_tok) |t| {
            const src_off = t * kvd;
            const dst_elem = (prev_len + t) * kvd;
            const dst_byte_k = kv_quant.kvByteOffset(kv_type_k, dst_elem);
            const dst_byte_v = kv_quant.kvByteOffset(kv_type_v, dst_elem);
            kv_quant.kvStore(kv_keys.ptr + dst_byte_k, k + src_off, kvd, kv_type_k);
            kv_quant.kvStore(kv_values.ptr + dst_byte_v, v + src_off, kvd, kv_type_v);
        }

        // Per-token causal attention using decode sdpa's parallel head dispatch
        if (kv_type_k == .f32 and kv_type_v == .f32) {
            const f32_keys: [*]const f32 = @ptrCast(@alignCast(kv_keys.ptr));
            const f32_values: [*]const f32 = @ptrCast(@alignCast(kv_values.ptr));
            for (0..n_tok) |t| {
                const sl = prev_len + t + 1;
                const q_off = t * nh * hd;
                const out_off = t * nh * hd;
                if (self.pool) |pool| {
                    if (nh >= sdpa_parallel_min_heads) {
                        var ctx = SdpaF32Ctx{
                            .q = q + q_off,
                            .keys = f32_keys,
                            .values = f32_values,
                            .output = output + out_off,
                            .nh = nh,
                            .nkv = nkv,
                            .hd = hd,
                            .sl = sl,
                            .scale = scale,
                        };
                        pool.parallelFor(nh, 1, @ptrCast(&ctx), SdpaF32Ctx.work);
                        continue;
                    }
                }
                sdpa_kernel.sdpaHeads(q + q_off, f32_keys, f32_values, output + out_off, nh, nkv, hd, sl, scale);
            }
        } else {
            for (0..n_tok) |t| {
                const sl = prev_len + t + 1;
                const q_off = t * nh * hd;
                const out_off = t * nh * hd;
                if (self.pool) |pool| {
                    if (nh >= sdpa_parallel_min_heads) {
                        var ctx = SdpaQuantCtx{
                            .q = q + q_off,
                            .keys = kv_keys.ptr,
                            .values = kv_values.ptr,
                            .output = output + out_off,
                            .nh = nh,
                            .nkv = nkv,
                            .hd = hd,
                            .sl = sl,
                            .scale = scale,
                            .kv_type_k = kv_type_k,
                            .kv_type_v = kv_type_v,
                        };
                        pool.parallelFor(nh, 1, @ptrCast(&ctx), SdpaQuantCtx.work);
                        continue;
                    }
                }
                sdpa_kernel.sdpaQuantHeads(q + q_off, kv_keys.ptr, kv_values.ptr, output + out_off, nh, nkv, hd, sl, scale, kv_type_k, kv_type_v);
            }
        }
    }

    /// Minimum query heads to justify parallelizing SDPA across heads.
    const sdpa_parallel_min_heads: usize = 4;

    const kv_quant = @import("../ops/kv_quant.zig");
    const KvQuantType = kv_quant.KvQuantType;

    /// CPU scaled dot-product attention with KV cache append.
    /// Parallelizes across query heads when a thread pool is available.
    /// Supports quantized KV cache: quantizes k_new/v_new on append,
    /// dequantizes during QK dot products and V accumulation.
    pub fn sdpa(self: *CpuBackend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        const kvd = nkv * hd;

        // KV append: quantize k_new/v_new into cache at position seq_len
        const k_byte_off = kv_quant.kvByteOffset(kv_type_k, seq_len * kvd);
        const v_byte_off = kv_quant.kvByteOffset(kv_type_v, seq_len * kvd);
        kv_quant.kvStore(keys.ptr + k_byte_off, k_new, kvd, kv_type_k);
        kv_quant.kvStore(values.ptr + v_byte_off, v_new, kvd, kv_type_v);

        // f32 fast path: cast to [*]f32 and use existing SIMD kernel for zero regression
        if (kv_type_k == .f32 and kv_type_v == .f32) {
            const f32_keys: [*]const f32 = @ptrCast(@alignCast(keys.ptr));
            const f32_vals: [*]const f32 = @ptrCast(@alignCast(values.ptr));
            if (self.pool) |pool| {
                if (nh >= sdpa_parallel_min_heads) {
                    var ctx = SdpaF32Ctx{
                        .q = q,
                        .keys = f32_keys,
                        .values = f32_vals,
                        .output = output,
                        .nh = nh,
                        .nkv = nkv,
                        .hd = hd,
                        .sl = seq_len + 1,
                        .scale = scale,
                    };
                    pool.parallelFor(nh, 1, @ptrCast(&ctx), SdpaF32Ctx.work);
                    return;
                }
            }
            sdpa_kernel.sdpaHeads(q, f32_keys, f32_vals, output, nh, nkv, hd, seq_len + 1, scale);
            return;
        }

        // Quantized path: use kvDot/kvMulAccum
        if (self.pool) |pool| {
            if (nh >= sdpa_parallel_min_heads) {
                var ctx = SdpaQuantCtx{
                    .q = q,
                    .keys = keys.ptr,
                    .values = values.ptr,
                    .output = output,
                    .nh = nh,
                    .nkv = nkv,
                    .hd = hd,
                    .sl = seq_len + 1,
                    .scale = scale,
                    .kv_type_k = kv_type_k,
                    .kv_type_v = kv_type_v,
                };
                pool.parallelFor(nh, 1, @ptrCast(&ctx), SdpaQuantCtx.work);
                return;
            }
        }
        sdpa_kernel.sdpaQuantHeads(q, keys.ptr, values.ptr, output, nh, nkv, hd, seq_len + 1, scale, kv_type_k, kv_type_v);
    }

    /// Context for parallel f32 SDPA dispatch across query heads.
    const SdpaF32Ctx = struct {
        q: [*]const f32,
        keys: [*]const f32,
        values: [*]const f32,
        output: [*]f32,
        nh: usize,
        nkv: usize,
        hd: usize,
        sl: usize,
        scale: f32,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *const SdpaF32Ctx = @ptrCast(@alignCast(ctx_ptr));
            for (start..end) |h| {
                sdpa_kernel.sdpaHead(ctx.q, ctx.keys, ctx.values, ctx.output, h, ctx.nh, ctx.nkv, ctx.hd, ctx.sl, ctx.scale);
            }
        }
    };

    /// Context for parallel quantized SDPA dispatch across query heads.
    const SdpaQuantCtx = struct {
        q: [*]const f32,
        keys: [*]const u8,
        values: [*]const u8,
        output: [*]f32,
        nh: usize,
        nkv: usize,
        hd: usize,
        sl: usize,
        scale: f32,
        kv_type_k: KvQuantType,
        kv_type_v: KvQuantType,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *const SdpaQuantCtx = @ptrCast(@alignCast(ctx_ptr));
            for (start..end) |h| {
                sdpa_kernel.sdpaQuantHead(ctx.q, ctx.keys, ctx.values, ctx.output, h, ctx.nh, ctx.nkv, ctx.hd, ctx.sl, ctx.scale, ctx.kv_type_k, ctx.kv_type_v);
            }
        }
    };

    /// DeltaNet SSM recurrence: conv1d + L2 norm + recurrence + gated output.
    /// When a thread pool is available, parallelizes across v-heads.
    pub fn deltaNet(self: *CpuBackend, conv_in: [*]const f32, conv_out: [*]f32, z_buf: [*]const f32, alpha_buf: [*]const f32, beta_buf: [*]const f32, output: [*]f32, conv_state: [*]f32, ssm_state: []f32, ssm_a: [*]const f32, dt_bias: [*]const f32, conv_w: [*]const f32, ssm_norm_w: [*]const f32, p: @import("backend.zig").DeltaNetParams) void {
        const math_ops = @import("../ops/math.zig");
        const ssm_ops = @import("../ops/ssm.zig");
        const num_v_heads: usize = p.num_v_heads;
        const num_k_heads: usize = p.num_k_heads;
        const head_k_dim: usize = p.head_k_dim;

        // 1. Gate & beta computation
        var gate_vals: [max_deltanet_v_heads]f32 = undefined;
        var beta_vals: [max_deltanet_v_heads]f32 = undefined;
        for (0..num_v_heads) |h| {
            const alpha_biased = alpha_buf[h] + dt_bias[h];
            gate_vals[h] = ssm_a[h] * math_ops.softplus(alpha_biased);
            beta_vals[h] = math_ops.sigmoid(beta_buf[h]);
        }

        // 2. Conv1d + SiLU
        ssm_ops.causalConv1dSilu(conv_out, conv_state, conv_in, conv_w, null, p.conv_ch, p.d_conv);

        // 3. L2 normalize Q and K per head
        // GGUF (llama.cpp) rearranges to Q,K,V order.
        // SafeTensors/HF keeps original K,Q,V order (split at key_dim, 2*key_dim).
        const q_off: usize = if (p.kqv_order) num_k_heads * head_k_dim else 0;
        const k_off: usize = if (p.kqv_order) 0 else num_k_heads * head_k_dim;
        for (0..num_k_heads) |h| {
            inline for ([_]usize{ q_off, k_off }) |base_off| {
                norm_kernel.l2Norm(conv_out + base_off + h * head_k_dim, head_k_dim, p.rms_eps);
            }
        }

        // 4. Recurrence + gated output — parallelized across v-heads
        const q_ptr = conv_out + q_off;
        const k_ptr = conv_out + k_off;
        const v_off: usize = 2 * num_k_heads * head_k_dim;
        const v_ptr = conv_out + v_off;

        if (self.pool) |pool| {
            if (num_v_heads >= deltanet_parallel_min_heads) {
                var ctx = DeltaNetHeadCtx{
                    .gate_vals = &gate_vals,
                    .beta_vals = &beta_vals,
                    .q_ptr = q_ptr,
                    .k_ptr = k_ptr,
                    .v_ptr = v_ptr,
                    .output = output,
                    .ssm_state = ssm_state.ptr,
                    .z_buf = z_buf,
                    .ssm_norm_w = ssm_norm_w,
                    .p = p,
                };
                pool.parallelFor(num_v_heads, 1, @ptrCast(&ctx), DeltaNetHeadCtx.work);
                return;
            }
        }
        // Fallback: sequential
        for (0..num_v_heads) |h| {
            deltanet_kernel.deltaNetHead(h, &gate_vals, &beta_vals, q_ptr, k_ptr, v_ptr, output, ssm_state.ptr, z_buf, ssm_norm_w, p);
        }
    }

    /// Context for parallel DeltaNet dispatch across v-heads.
    const DeltaNetHeadCtx = struct {
        gate_vals: *const [max_deltanet_v_heads]f32,
        beta_vals: *const [max_deltanet_v_heads]f32,
        q_ptr: [*]const f32,
        k_ptr: [*]const f32,
        v_ptr: [*]const f32,
        output: [*]f32,
        ssm_state: [*]f32,
        z_buf: [*]const f32,
        ssm_norm_w: [*]const f32,
        p: @import("backend.zig").DeltaNetParams,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *const DeltaNetHeadCtx = @ptrCast(@alignCast(ctx_ptr));
            for (start..end) |h| {
                deltanet_kernel.deltaNetHead(h, ctx.gate_vals, ctx.beta_vals, ctx.q_ptr, ctx.k_ptr, ctx.v_ptr, ctx.output, ctx.ssm_state, ctx.z_buf, ctx.ssm_norm_w, ctx.p);
            }
        }
    };
};

// ── Autotune tests ───────────────────────────────────────────────

test "softmax autotune — compare SIMD widths" {
    // Generates all 3 variants at comptime, benchmarks each at test time.
    // Run with: zig build test --release=fast
    const n = 1024;
    var data_orig: [n]f32 = undefined;
    for (0..n) |i| data_orig[i] = @as(f32, @floatFromInt(i % 37)) * 0.1 - 1.8;

    const widths = [_]comptime_int{ 4, 8, 16 };
    var ref: [n]f32 = undefined;
    var have_ref = false;
    inline for (widths) |w| {
        var data = data_orig;
        softmax_kernel.softmaxSimd(w, &data, n);
        // Verify: sum should be ~1.0
        var sum: f32 = 0;
        for (0..n) |i| sum += data[i];
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);
        // Verify: all SIMD widths produce consistent output
        if (!have_ref) {
            ref = data;
            have_ref = true;
        } else {
            for (0..n) |i| {
                try std.testing.expectApproxEqAbs(ref[i], data[i], 1e-5);
            }
        }
    }
}
