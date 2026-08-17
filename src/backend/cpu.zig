//! CPU backend with SIMD-optimized kernels.
//! Supports optional multi-threaded GEMV via ThreadPool.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const posix = std.posix;
const backend_mod = @import("backend.zig");

/// Read a small file into buf via raw posix syscalls (no std.fs dependency).
fn readSmallFile(comptime path: []const u8, buf: []u8) []const u8 {
    const fd = posix.openat(posix.AT.FDCWD, path, .{}, 0) catch return "";
    defer _ = posix.system.close(fd);
    const n = posix.read(fd, buf) catch return "";
    return buf[0..n];
}
const TensorData = backend_mod.TensorData;
const DType = backend_mod.DType;
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
const sdpa_tree_kernel = @import("kernels/cpu/sdpa_tree.zig");
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
/// Set only after the buffer is fully written (release). Readers use acquire.
var cpu_model_detected: std.atomic.Value(bool) = .init(false);
/// Spinlock for one-time detection (avoids torn buffer under concurrent first calls).
var cpu_model_init_lock: std.atomic.Value(u8) = .init(0);

/// Detect CPU model name from the OS. Called once at first backendInfo() call.
fn detectCpuModel() []const u8 {
    if (cpu_model_detected.load(.acquire)) return cpu_model_buf[0..cpu_model_len];

    while (cpu_model_init_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null)
        std.atomic.spinLoopHint();
    defer cpu_model_init_lock.store(0, .release);

    if (cpu_model_detected.load(.acquire)) return cpu_model_buf[0..cpu_model_len];

    if (comptime builtin.os.tag == .macos) {
        // macOS: sysctlbyname("machdep.cpu.brand_string")
        var len: usize = cpu_model_buf.len;
        const rc = std.c.sysctlbyname("machdep.cpu.brand_string", &cpu_model_buf, &len, null, 0);
        if (rc == 0 and len > 0) {
            // Strip trailing null
            cpu_model_len = if (cpu_model_buf[len - 1] == 0) len - 1 else len;
            cpu_model_detected.store(true, .release);
            return cpu_model_buf[0..cpu_model_len];
        }
    } else if (comptime builtin.os.tag == .linux) {
        // Linux: parse /proc/cpuinfo for CPU name.
        // x86 uses "model name\t: ...", ARM uses "Model\t: ..." or "Hardware\t: ...".
        var read_buf: [cpuinfo_read_buf_size]u8 = undefined;
        const data = readSmallFile("/proc/cpuinfo", &read_buf);
        if (data.len == 0) {
            cpu_model_detected.store(true, .release);
            return "";
        }
        // Try x86-style first, then ARM/RISC-V fallbacks.
        const needles = [_][]const u8{
            "model name\t: ", // x86, some ARM kernels
            "Model\t: ", // Raspberry Pi, some ARM
            "Hardware\t: ", // older ARM kernels
        };
        for (needles) |needle| {
            if (std.mem.indexOf(u8, data, needle)) |pos| {
                const start = pos + needle.len;
                const end = std.mem.indexOfScalarPos(u8, data, start, '\n') orelse data.len;
                const name_len = @min(end - start, cpu_model_buf.len);
                @memcpy(cpu_model_buf[0..name_len], data[start..][0..name_len]);
                cpu_model_len = name_len;
                cpu_model_detected.store(true, .release);
                return cpu_model_buf[0..cpu_model_len];
            }
        }
        // ARM/RISC-V without a friendly name: return implementer+part as fallback.
        const impl_needle = "CPU implementer\t: ";
        const part_needle = "CPU part\t: ";
        if (std.mem.indexOf(u8, data, impl_needle)) |ipos| {
            const istart = ipos + impl_needle.len;
            const iend = std.mem.indexOfScalarPos(u8, data, istart, '\n') orelse data.len;
            const ppos = std.mem.indexOf(u8, data, part_needle) orelse 0;
            const pstart = if (ppos > 0) ppos + part_needle.len else 0;
            const pend = if (ppos > 0) std.mem.indexOfScalarPos(u8, data, pstart, '\n') orelse data.len else 0;
            if (ppos > 0) {
                const n = std.fmt.bufPrint(&cpu_model_buf, "ARM impl={s} part={s}", .{ data[istart..iend], data[pstart..pend] }) catch {
                    cpu_model_detected.store(true, .release);
                    return "";
                };
                cpu_model_len = n.len;
                cpu_model_detected.store(true, .release);
                return cpu_model_buf[0..cpu_model_len];
            }
        }
    }
    cpu_model_detected.store(true, .release);
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
    var buf: [32]u8 = undefined;
    const raw = readSmallFile(path, &buf);
    if (raw.len == 0) return 0;
    const data = std.mem.trimEnd(u8, raw, "\n ");
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
        var read_buf: [meminfo_read_buf_size]u8 = undefined;
        const data = readSmallFile("/proc/meminfo", &read_buf);
        if (data.len == 0) return 0;
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
        var read_buf: [memavail_read_buf_size]u8 = undefined;
        const data = readSmallFile("/proc/meminfo", &read_buf);
        if (data.len == 0) return 0;
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

const CacheSizes = backend_mod.CacheSizes;

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
var os_version_detected: std.atomic.Value(bool) = .init(false);
var os_version_init_lock: std.atomic.Value(u8) = .init(0);

/// Detect OS version string. Returns "macOS 14.2.1" or "Linux 6.5.0" style strings.
pub fn detectOsVersion() []const u8 {
    if (os_version_detected.load(.acquire)) return os_version_buf[0..os_version_len];

    while (os_version_init_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null)
        std.atomic.spinLoopHint();
    defer os_version_init_lock.store(0, .release);

    if (os_version_detected.load(.acquire)) return os_version_buf[0..os_version_len];

    if (comptime builtin.os.tag == .macos) {
        // macOS: Try kern.osproductversion first (e.g., "14.2.1"), fall back to kern.osrelease
        var len: usize = os_version_buf.len - os_prefix_len;
        const rc = std.c.sysctlbyname("kern.osproductversion", os_version_buf[os_prefix_len..].ptr, &len, null, 0);
        if (rc == 0 and len > 0) {
            @memcpy(os_version_buf[0..os_prefix_len], "macOS ");
            // Strip trailing null
            const total_len = os_prefix_len + (if (os_version_buf[os_prefix_len + len - 1] == 0) len - 1 else len);
            os_version_len = total_len;
            os_version_detected.store(true, .release);
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
        os_version_detected.store(true, .release);
        return os_version_buf[0..os_version_len];
    }
    os_version_detected.store(true, .release);
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
const parallel_grain: usize = 128;
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
    pub fn rmsNorm(_: *CpuBackend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        norm_kernel.rmsNorm(input, weight, output, n, eps);
    }

    /// Fused add + rms_norm: a[i] = a[i] + b[i], output = rms_norm(a+b, weight, eps).
    pub fn addRmsNorm(_: *CpuBackend, a: [*]f32, b: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        norm_kernel.addRmsNorm(a, b, weight, output, n, eps);
    }

    /// Fused rmsNorm + accumulate: b[i] += rmsNorm(a, weight, eps)[i].
    pub fn rmsNormAdd(_: *CpuBackend, a: [*]const f32, weight: [*]const f32, b: [*]f32, n: usize, eps: f32) void {
        norm_kernel.rmsNormAdd(a, weight, b, n, eps);
    }

    /// Applies SiLU (Swish) activation: x * sigmoid(x).
    pub fn silu(_: *CpuBackend, input: [*]const f32, output: [*]f32, n: usize) void {
        activation_kernel.silu(input, output, n);
    }

    /// Applies GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x³))).
    pub fn gelu(_: *CpuBackend, input: [*]const f32, output: [*]f32, n: usize) void {
        activation_kernel.gelu(input, output, n);
    }

    /// Element-wise addition: out[i] = a[i] + b[i].
    pub fn add(_: *CpuBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        elementwise_kernel.add(a, b, out, n);
    }

    /// Transposed GEMV: y[out_dim] = W^T @ x[in_dim] for Q8_0 3D weights.
    /// Block-structured loop reads the Q8_0 scale once per 32-element block
    /// and uses SIMD for the inner accumulation.
    pub fn gemvT(_: *CpuBackend, x: [*]const f32, w: [*]const u8, y: [*]f32, out_dim: usize, in_dim: usize) void {
        const V8 = @Vector(8, f32);
        const block_elems = quant.quant_block_elems; // 32
        const block_bytes = quant.q8_0_block_bytes; // 34
        const blocks_per_row = (out_dim + block_elems - 1) / block_elems;
        @memset(y[0..out_dim], 0);
        for (0..in_dim) |j| {
            const xj = x[j];
            const row_base = j * blocks_per_row * block_bytes;
            for (0..blocks_per_row) |blk_idx| {
                const blk_ptr = w + row_base + blk_idx * block_bytes;
                const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, blk_ptr[0..2], .little))));
                const scaled_x = xj * scale;
                const sv: V8 = @splat(scaled_x);
                const q = blk_ptr + 2;
                const out_base = blk_idx * block_elems;
                const count = @min(block_elems, out_dim - out_base);
                var k: usize = 0;
                while (k + 8 <= count) : (k += 8) {
                    var qv: V8 = undefined;
                    inline for (0..8) |idx| {
                        qv[idx] = @floatFromInt(@as(i8, @bitCast(q[k + idx])));
                    }
                    const cur: V8 = y[out_base + k ..][0..8].*;
                    y[out_base + k ..][0..8].* = @mulAdd(V8, qv, sv, cur);
                }
                while (k < count) : (k += 1) {
                    const val: f32 = @floatFromInt(@as(i8, @bitCast(q[k])));
                    y[out_base + k] = @mulAdd(f32, val, scaled_x, y[out_base + k]);
                }
            }
        }
    }

    /// All-reduce sum: dst[i] += src[i]. Used for tensor parallelism partial result accumulation.
    pub fn allReduceAdd(_: *CpuBackend, dst: [*]f32, src: [*]const f32, n: usize) void {
        const V8 = @Vector(8, f32);
        var i: usize = 0;
        while (i + 8 <= n) : (i += 8) {
            const d: V8 = dst[i..][0..8].*;
            const s: V8 = src[i..][0..8].*;
            dst[i..][0..8].* = d + s;
        }
        while (i < n) : (i += 1) dst[i] += src[i];
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
    pub fn siluMul(_: *CpuBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        activation_kernel.siluMul(a, b, out, n);
    }

    /// SwiGLU with clamped gate/up values to [-10, 10] (prevents exp overflow in SiLU).
    pub fn clampedSiluMul(_: *CpuBackend, gate: [*]const f32, up: [*]const f32, out: [*]f32, n: usize) void {
        for (0..n) |i| {
            const g = @min(@as(f32, 10.0), @max(@as(f32, -10.0), gate[i]));
            const u = @min(@as(f32, 10.0), @max(@as(f32, -10.0), up[i]));
            out[i] = (g / (1.0 + @exp(-g))) * u;
        }
    }

    /// Fused GELU + multiply: out[i] = gelu(a[i]) * b[i].
    /// Single-pass SIMD avoids a second cache traversal over the output buffer.
    pub fn geluMul(_: *CpuBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        activation_kernel.geluMul(a, b, out, n);
    }

    /// In-place per-head rmsNorm: applies same weight to n_heads independent heads.
    pub fn rmsNormMulti(self: *CpuBackend, data: [*]f32, weight: [*]const f32, n_heads: usize, head_dim: usize, eps: f32) void {
        for (0..n_heads) |h| {
            self.rmsNorm(data + h * head_dim, weight, data + h * head_dim, head_dim, eps);
        }
    }

    /// Element-wise multiplication: out[i] = a[i] * b[i].
    pub fn mul(_: *CpuBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        elementwise_kernel.mul(a, b, out, n);
    }

    /// Applies softmax normalization in-place.
    pub fn softmax(_: *CpuBackend, data: [*]f32, n: usize) void {
        softmax_kernel.softmaxSimd(softmax_width, data, n);
    }

    /// Applies Rotary Position Embedding (RoPE) in-place.
    pub fn rope(_: *CpuBackend, x: [*]f32, pos: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        rope_kernel.rope(x, pos, n_heads, head_dim, rope_dim, theta);
    }

    /// Looks up a token embedding row and dequantizes to f32.
    pub fn embLookup(_: *CpuBackend, table: TensorData, token_id: u32, output: [*]f32, dim: usize) void {
        emb_kernel.embLookup(table.data, table.dtype, token_id, output, dim);
    }

    /// L2 normalizes a vector in-place: x[i] /= sqrt(sum(x^2) + eps).
    pub fn l2Norm(_: *CpuBackend, x: [*]f32, n: usize, eps: f32) void {
        norm_kernel.l2Norm(x, n, eps);
    }

    /// In-place sigmoid-gated multiply: data[i] *= sigmoid(gate[i]).
    pub fn sigmoidMul(_: *CpuBackend, data: [*]f32, gate: [*]const f32, n: usize) void {
        elementwise_kernel.sigmoidMul(data, gate, n);
    }

    /// De-interleave paired blocks on CPU.
    pub fn deinterleave(_: *CpuBackend, input: [*]const f32, out_a: [*]f32, out_b: [*]f32, stride: usize, n_pairs: usize) void {
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

    /// No-op on CPU — operations are immediately visible.
    pub fn sync(_: *CpuBackend) void {}

    /// No-op on CPU.
    pub fn beginBatch(_: *CpuBackend) void {}

    /// No-op on CPU.
    pub fn endBatch(_: *CpuBackend) void {}

    /// Returns backend information for display. Cache sizes and total memory
    /// are detected once and cached; available memory is always fresh.
    pub fn backendInfo(_: *const CpuBackend) backend_mod.BackendInfo {
        const Static = struct {
            var caches: CacheSizes = .{};
            var sys_mem: usize = 0;
            var detected: std.atomic.Value(bool) = .init(false);
        };
        if (!Static.detected.load(.acquire)) {
            Static.caches = detectCacheSizes();
            Static.sys_mem = detectSystemMem();
            Static.detected.store(true, .release);
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
    pub fn gemvNvfp4St(_: *CpuBackend, x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: usize, k: usize) void {
        quant.gemvNvfp4St(x, weight, scale, y, n, k);
    }

    /// MLX affine quantized GEMV: packed integer weights + bf16 scales/biases.
    /// When a thread pool is available, parallelizes across output rows.
    pub fn gemvMlxQ(self: *CpuBackend, x: [*]const f32, weight: [*]const u8, scales: [*]const u8, biases: [*]const u8, y: [*]f32, n: usize, k: usize, bits: u32, group_size: u32) void {
        const mlx_ops = @import("../ops/mlx.zig");
        const gs: usize = group_size;
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
                    .gs = gs,
                };
                pool.parallelFor(n, parallel_grain, @ptrCast(&ctx), MlxGemvCtx.work);
                return;
            }
        }
        mlx_ops.mlxGemvRows(x, @ptrCast(@alignCast(weight)), @ptrCast(@alignCast(scales)), @ptrCast(@alignCast(biases)), y, 0, n, k, bits, gs);
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
        gs: usize,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *const MlxGemvCtx = @ptrCast(@alignCast(ctx_ptr));
            const mlx = @import("../ops/mlx.zig");
            mlx.mlxGemvRows(ctx.x, ctx.pw, ctx.sc, ctx.bi, ctx.y, start, end - start, ctx.k, ctx.bits, ctx.gs);
        }
    };

    /// MXFP4 SafeTensors GEMV (U32-packed nibbles, E8M0 scales, no bias).
    pub fn gemvMxfp4St(self: *CpuBackend, x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: usize, k: usize, gs: usize, sf: mlx_mod.Mxfp4ScaleFormat) void {
        if (self.pool) |pool| {
            if (n >= parallel_min_rows) {
                var ctx = Mxfp4StCtx{
                    .x = x,
                    .pw = @ptrCast(@alignCast(weight)),
                    .scales_u8 = scale,
                    .y = y,
                    .k = k,
                    .gs = gs,
                    .sf = sf,
                };
                pool.parallelFor(n, parallel_grain, @ptrCast(&ctx), Mxfp4StCtx.work);
                return;
            }
        }
        mlx_mod.mlxMxfp4GemvRows(x, @ptrCast(@alignCast(weight)), scale, y, 0, n, k, gs, sf);
    }

    const mlx_mod = @import("../ops/mlx.zig");

    const Mxfp4StCtx = struct {
        x: [*]const f32,
        pw: [*]const u32,
        scales_u8: [*]const u8,
        y: [*]f32,
        k: usize,
        gs: usize,
        sf: mlx_mod.Mxfp4ScaleFormat,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *const Mxfp4StCtx = @ptrCast(@alignCast(ctx_ptr));
            mlx_mod.mlxMxfp4GemvRows(ctx.x, ctx.pw, ctx.scales_u8, ctx.y, start, end - start, ctx.k, ctx.gs, ctx.sf);
        }
    };

    /// GPTQ INT4 GEMV with thread pool parallelism.
    pub fn gemvGptq(self: *CpuBackend, x: [*]const f32, qweight: [*]const u32, scales: [*]const u16, qzeros: [*]const u32, y: [*]f32, n: usize, k: usize, group_size: u32) void {
        if (self.pool) |pool| {
            if (n >= parallel_min_rows) {
                var ctx = GptqCtx{ .x = x, .qw = qweight, .scales = scales, .qzeros = qzeros, .y = y, .n = n, .k = k, .gs = group_size };
                pool.parallelFor(n, parallel_grain, @ptrCast(&ctx), GptqCtx.work);
                return;
            }
        }
        const gptq_ops = @import("../ops/gptq.zig");
        gptq_ops.gptqGemv(x, qweight, scales, qzeros, y, n, k, group_size);
    }

    const GptqCtx = struct {
        x: [*]const f32,
        qw: [*]const u32,
        scales: [*]const u16,
        qzeros: [*]const u32,
        y: [*]f32,
        n: usize,
        k: usize,
        gs: u32,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *const GptqCtx = @ptrCast(@alignCast(ctx_ptr));
            const gptq_ops = @import("../ops/gptq.zig");
            gptq_ops.gptqGemvRows(ctx.x, ctx.qw, ctx.scales, ctx.qzeros, ctx.y, start, end - start, ctx.n, ctx.k, ctx.gs);
        }
    };

    /// AWQ INT4 GEMV with thread pool parallelism.
    pub fn gemvAwq(_: *CpuBackend, x: [*]const f32, qweight: [*]const u32, scales: [*]const u16, qzeros: [*]const u32, y: [*]f32, n: usize, k: usize, group_size: u32) void {
        const awq_ops = @import("../ops/awq.zig");
        awq_ops.awqGemv(x, qweight, scales, qzeros, y, n, k, group_size);
    }

    /// HQQ (Half-Quadratic Quantization) GEMV: y = HQQ_dequant(w_q, scale, zero) @ x.
    pub fn gemvHqq(_: *CpuBackend, x: [*]const f32, w_q: [*]const u8, scale: [*]const u8, zero: [*]const u8, y: [*]f32, n: usize, k: usize, group_size: u32) void {
        const hqq_ops = @import("../ops/hqq.zig");
        hqq_ops.hqqGemv(x, w_q, @ptrCast(@alignCast(scale)), @ptrCast(@alignCast(zero)), y, n, k, group_size);
    }

    /// Batched GEMV — fuses all ops into a single parallelFor to minimize
    /// thread wake/sleep overhead (~250 GEMV dispatches per token).
    pub fn gemvMulti(self: *CpuBackend, x: [*]const f32, ops: []const backend_mod.GemvOp, k: usize) void {
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
                mlx_ops.mlxGemvRaw(x, @ptrCast(@alignCast(op.w.data)), @ptrCast(@alignCast(op.mlx_scales.?)), @ptrCast(@alignCast(op.mlx_biases.?)), op.y, op.n, k, op.mlx_bits, op.mlx_group_size);
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
        ops: []const backend_mod.GemvOp,

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
        // Accelerate.framework: full SGEMM for F32 weights (AMX-accelerated, ~4× faster)
        // Only available when Metal/Accelerate is linked (enable_metal controls this).
        if (comptime builtin.os.tag == .macos and build_options.enable_metal) {
            if (w.dtype == .f32) {
                const accel = @import("accelerate.zig");
                accel.sgemm(n_tok, n_out, n_in, x, @ptrCast(@alignCast(w.data)), y);
                return;
            }
        }
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

    /// CPU scaled dot-product attention with KV cache append, returning per-head
    /// softmax statistics (max and sum) for online softmax merge in split-attention.
    /// Same as sdpa() but additionally outputs head_max[nh] and head_sum[nh].
    /// Parallelizes across query heads when a thread pool is available.
    /// Paged SDPA: block-table-indexed attention for non-contiguous KV cache.
    pub fn sdpaPaged(self: *CpuBackend, q: [*]const f32, kv_view: backend_mod.PagedKvView, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, scale: f32, _: KvQuantType, _: KvQuantType) void {
        // Thread-parallel dispatch across query heads
        if (self.pool) |pool| {
            if (nh >= sdpa_parallel_min_heads) {
                // Append KV first (single-threaded — one position)
                const kvd = nkv * hd;
                const k_dst = kv_view.keyPtrMut(kv_view.seq_len);
                const v_dst = kv_view.valuePtrMut(kv_view.seq_len);
                @memcpy(k_dst[0..kvd], k_new[0..kvd]);
                @memcpy(v_dst[0..kvd], v_new[0..kvd]);

                var ctx = SdpaPagedCtx{
                    .q = q,
                    .kv_view = kv_view,
                    .output = output,
                    .nh = nh,
                    .nkv = nkv,
                    .hd = hd,
                    .sl = kv_view.seq_len + 1,
                    .scale = scale,
                };
                pool.parallelFor(nh, 1, @ptrCast(&ctx), SdpaPagedCtx.work);
                return;
            }
        }
        sdpa_kernel.sdpaPagedHeads(q, kv_view, k_new, v_new, output, nh, nkv, hd, scale);
    }

    const SdpaPagedCtx = struct {
        q: [*]const f32,
        kv_view: backend_mod.PagedKvView,
        output: [*]f32,
        nh: usize,
        nkv: usize,
        hd: usize,
        sl: usize,
        scale: f32,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *const SdpaPagedCtx = @ptrCast(@alignCast(ctx_ptr));
            for (start..end) |h| {
                sdpa_kernel.sdpaPagedHead(ctx.q, ctx.kv_view, ctx.output, h, ctx.nh, ctx.nkv, ctx.hd, ctx.sl, ctx.scale);
            }
        }
    };

    /// SDPA with per-head max/sum statistics for online softmax (split-attention merge path).
    pub fn sdpaWithStats(self: *CpuBackend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, head_max: [*]f32, head_sum: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        const kvd = nkv * hd;

        // KV append: quantize k_new/v_new into cache at position seq_len
        const k_byte_off = kv_quant.kvByteOffset(kv_type_k, seq_len * kvd);
        const v_byte_off = kv_quant.kvByteOffset(kv_type_v, seq_len * kvd);
        kv_quant.kvStore(keys.ptr + k_byte_off, k_new, kvd, kv_type_k);
        kv_quant.kvStore(values.ptr + v_byte_off, v_new, kvd, kv_type_v);

        if (self.pool) |pool| {
            if (nh >= sdpa_parallel_min_heads) {
                var ctx = SdpaQuantWithStatsCtx{
                    .q = q,
                    .keys = keys.ptr,
                    .values = values.ptr,
                    .output = output,
                    .head_max = head_max,
                    .head_sum = head_sum,
                    .nh = nh,
                    .nkv = nkv,
                    .hd = hd,
                    .sl = seq_len + 1,
                    .scale = scale,
                    .kv_type_k = kv_type_k,
                    .kv_type_v = kv_type_v,
                };
                pool.parallelFor(nh, 1, @ptrCast(&ctx), SdpaQuantWithStatsCtx.work);
                return;
            }
        }
        sdpa_kernel.sdpaQuantHeadsWithStats(q, keys.ptr, values.ptr, output, nh, nkv, hd, seq_len + 1, scale, kv_type_k, kv_type_v, head_max, head_sum);
    }

    /// Context for parallel quantized SDPA with stats dispatch across query heads.
    const SdpaQuantWithStatsCtx = struct {
        q: [*]const f32,
        keys: [*]const u8,
        values: [*]const u8,
        output: [*]f32,
        head_max: [*]f32,
        head_sum: [*]f32,
        nh: usize,
        nkv: usize,
        hd: usize,
        sl: usize,
        scale: f32,
        kv_type_k: KvQuantType,
        kv_type_v: KvQuantType,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *const SdpaQuantWithStatsCtx = @ptrCast(@alignCast(ctx_ptr));
            for (start..end) |h| {
                sdpa_kernel.sdpaQuantHeadWithStats(ctx.q, ctx.keys, ctx.values, ctx.output, h, ctx.nh, ctx.nkv, ctx.hd, ctx.sl, ctx.scale, ctx.kv_type_k, ctx.kv_type_v, ctx.head_max, ctx.head_sum);
            }
        }
    };

    /// Tree-masked SDPA for DDTree speculative decoding verification.
    /// Processes n_nodes tree queries against shared prefix KV + masked tree KV.
    pub fn sdpaTree(_: *CpuBackend, q_all: [*]const f32, prefix_keys: [*]const u8, prefix_values: [*]const u8, tree_keys: [*]const f32, tree_values: [*]const f32, output: [*]f32, ancestor_masks: [*]const [8]u64, nh: usize, nkv: usize, hd: usize, prefix_len: usize, n_nodes: u32, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        sdpa_tree_kernel.sdpaTree(q_all, prefix_keys, prefix_values, tree_keys, tree_values, output, ancestor_masks, nh, nkv, hd, prefix_len, n_nodes, scale, kv_type_k, kv_type_v);
    }

    /// DeltaNet SSM recurrence: conv1d + L2 norm + recurrence + gated output.
    /// When a thread pool is available, parallelizes across v-heads.
    pub fn deltaNet(self: *CpuBackend, conv_in: [*]const f32, conv_out: [*]f32, z_buf: [*]const f32, alpha_buf: [*]const f32, beta_buf: [*]const f32, output: [*]f32, conv_state: [*]f32, ssm_state: []f32, ssm_a: [*]const f32, dt_bias: [*]const f32, conv_w: [*]const f32, ssm_norm_w: [*]const f32, p: backend_mod.DeltaNetParams) void {
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
        p: backend_mod.DeltaNetParams,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *const DeltaNetHeadCtx = @ptrCast(@alignCast(ctx_ptr));
            for (start..end) |h| {
                deltanet_kernel.deltaNetHead(h, ctx.gate_vals, ctx.beta_vals, ctx.q_ptr, ctx.k_ptr, ctx.v_ptr, ctx.output, ctx.ssm_state, ctx.z_buf, ctx.ssm_norm_w, ctx.p);
            }
        }
    };
};

// ── Tests ───────────────────────────────────────────────────────────

test "CpuBackend — allocKvSlice and freeKvSlice" {
    const allocator = std.testing.allocator;
    var be = CpuBackend{};
    const slice = try be.allocKvSlice(allocator, 1024);
    try std.testing.expectEqual(@as(usize, 1024), slice.len);
    // Verify memory is writable.
    @memset(slice, 0xAA);
    try std.testing.expectEqual(@as(u8, 0xAA), slice[0]);
    try std.testing.expectEqual(@as(u8, 0xAA), slice[1023]);
    be.freeKvSlice(allocator, slice);
}

test "CpuBackend — addScaled" {
    var be = CpuBackend{};
    var dst = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    const src = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0 };
    be.addScaled(&src, &dst, 0.5, 10);
    // dst[i] = dst[i] + src[i] * 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), dst[0], 1e-6); // 1 + 10*0.5
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), dst[1], 1e-6); // 2 + 20*0.5
    try std.testing.expectApproxEqAbs(@as(f32, 60.0), dst[9], 1e-6); // 10 + 100*0.5
}

test "CpuBackend — allReduceAdd" {
    var be = CpuBackend{};
    var dst = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    const src = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0 };
    be.allReduceAdd(&dst, &src, 10);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), dst[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), dst[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 110.0), dst[9], 1e-6);
}

test "CpuBackend — splitQGate" {
    var be = CpuBackend{};
    // 2 heads, head_dim=3 → input is [Q0,Q1,Q2,G0,G1,G2, Q3,Q4,Q5,G3,G4,G5]
    const qg = [_]f32{ 1, 2, 3, 10, 20, 30, 4, 5, 6, 40, 50, 60 };
    var q_out: [6]f32 = undefined;
    var g_out: [6]f32 = undefined;
    be.splitQGate(&qg, &q_out, &g_out, 3, 2);
    // Q: [1,2,3,4,5,6]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), q_out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), q_out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), q_out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), q_out[3], 1e-6);
    // G: [10,20,30,40,50,60]
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), g_out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 40.0), g_out[3], 1e-6);
}

test "CpuBackend — sync and batch are no-ops" {
    var be = CpuBackend{};
    // These must not panic.
    be.sync();
    be.beginBatch();
    be.endBatch();
}

test "CpuBackend — backendInfo returns CPU" {
    var be = CpuBackend{};
    const info = be.backendInfo();
    try std.testing.expectEqualStrings("CPU", info.name);
    // System memory should be non-zero on any real host.
    try std.testing.expect(info.system_mem > 0 or info.total_mem > 0);
}

test "CpuBackend — default pool is null" {
    const be = CpuBackend{};
    try std.testing.expectEqual(@as(?*ThreadPool, null), be.pool);
}

test "CpuBackend — gemvSeq with F32 identity" {
    // Simple F32 GEMV: y[2] = W[2,4] @ x[4]
    const x = [_]f32{ 1, 0, 0, 0 };
    const w = [_]f32{
        1, 2, 3, 4, // row 0
        5, 6, 7, 8, // row 1
    };
    var y: [2]f32 = undefined;
    const w_bytes: [*]const u8 = @ptrCast(&w);
    var be = CpuBackend{};
    be.gemv(&x, .{ .data = w_bytes, .dtype = .f32 }, &y, 2, 4);
    // x = [1,0,0,0] → y = first column of W
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), y[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), y[1], 1e-5);
}

test "detectSystemMem — returns non-zero" {
    const mem = detectSystemMem();
    // On any real machine, total memory should be at least 256MB.
    try std.testing.expect(mem > 256 * 1024 * 1024);
}

test "detectCacheSizes — returns reasonable values" {
    const caches = detectCacheSizes();
    // Values are either zero (unavailable) or within plausible CPU cache bounds.
    const max_l1: usize = 1 * mb_to_bytes;
    const max_l2: usize = 64 * mb_to_bytes;
    const max_l3: usize = 512 * mb_to_bytes;
    try std.testing.expect(caches.l1 <= max_l1);
    try std.testing.expect(caches.l2 <= max_l2);
    try std.testing.expect(caches.l3 <= max_l3);
    if (caches.l1 > 0 and caches.l2 > 0) {
        try std.testing.expect(caches.l2 >= caches.l1);
    }
}

test "detectOsVersion — returns non-empty string" {
    const version = detectOsVersion();
    if (comptime builtin.os.tag == .macos) {
        try std.testing.expect(version.len > 0);
        try std.testing.expect(std.mem.startsWith(u8, version, "macOS "));
    } else if (comptime builtin.os.tag == .linux) {
        try std.testing.expect(version.len > 0);
        try std.testing.expect(std.mem.startsWith(u8, version, "Linux "));
    }
}

test "detectAvailMem — returns something" {
    const avail = detectAvailMem();
    if (comptime builtin.os.tag == .macos or builtin.os.tag == .linux) {
        // Running hosts expose a positive available-memory estimate.
        try std.testing.expect(avail > 0);
        // Sanity upper bound: less than 16 PiB (guards against unit mistakes).
        try std.testing.expect(avail < (@as(usize, 16) << 50));
    }
}

test "parallel constants — values are reasonable" {
    try std.testing.expectEqual(@as(usize, 32), parallel_min_rows);
    try std.testing.expectEqual(@as(usize, 16), parallel_grain);
    try std.testing.expect(parallel_grain <= parallel_min_rows);
}

// ── Autotune tests ───────────────────────────────────────────────

test "parseSysfsCacheSize — returns 0 on non-linux or valid size" {
    if (comptime builtin.os.tag == .linux) {
        // On Linux, index0 is typically L1 data cache (32K-128K).
        const l1 = parseSysfsCacheSize("/sys/devices/system/cpu/cpu0/cache/index0/size");
        // Must be >0 on any real Linux box with sysfs.
        try std.testing.expect(l1 > 0);
        // L1 should be a multiple of 1 KB.
        try std.testing.expectEqual(@as(usize, 0), l1 % kb_to_bytes);
    } else {
        // On non-Linux, parseSysfsCacheSize is a comptime no-op returning 0.
        const val = parseSysfsCacheSize("/sys/devices/system/cpu/cpu0/cache/index0/size");
        try std.testing.expectEqual(@as(usize, 0), val);
    }
}

test "detectCpuModel — returns non-empty on supported platforms" {
    // Reset detection state for a clean test.
    cpu_model_detected.store(false, .release);
    cpu_model_len = 0;
    const model = detectCpuModel();
    if (comptime builtin.os.tag == .macos or builtin.os.tag == .linux) {
        try std.testing.expect(model.len > 0);
    }
    // Calling again should return cached result (idempotent).
    const model2 = detectCpuModel();
    try std.testing.expectEqualStrings(model, model2);
}

test "sysctlU64 — returns 0 on non-macos" {
    if (comptime builtin.os.tag != .macos) {
        const val = sysctlU64("hw.memsize");
        try std.testing.expectEqual(@as(usize, 0), val);
    } else {
        // On macOS, hw.memsize should be at least 1 GB.
        const val = sysctlU64("hw.memsize");
        try std.testing.expect(val >= 1024 * 1024 * 1024);
    }
}

test "detectCacheSizes — L1 is non-zero on real hardware" {
    const caches = detectCacheSizes();
    if (comptime builtin.os.tag == .macos or builtin.os.tag == .linux) {
        // On real hardware L1 data cache is always present (typically 32K-128K).
        try std.testing.expect(caches.l1 > 0);
        // L1 should be at most 1 MB (sanity upper bound).
        try std.testing.expect(caches.l1 <= 1 * mb_to_bytes);
        // L2 should also be present on modern CPUs.
        try std.testing.expect(caches.l2 > 0);
        // L2 > L1 on virtually all architectures.
        try std.testing.expect(caches.l2 >= caches.l1);
    }
}

test "unit conversion constants" {
    try std.testing.expectEqual(@as(usize, 1024), kb_to_bytes);
    try std.testing.expectEqual(@as(usize, 1024 * 1024), mb_to_bytes);
}

test "CpuBackend — silu activation" {
    var be = CpuBackend{};
    var input = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0 };
    var output: [8]f32 = undefined;
    be.silu(&input, &output, 8);
    // silu(0) = 0*sigmoid(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 1e-5);
    // silu(1) = 1*sigmoid(1) ≈ 0.7311
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), output[1], 1e-3);
    // silu(-1) = -1*sigmoid(-1) ≈ -0.2689
    try std.testing.expectApproxEqAbs(@as(f32, -0.2689), output[2], 1e-3);
}

test "CpuBackend — gelu activation" {
    var be = CpuBackend{};
    var input = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.5 };
    var output: [8]f32 = undefined;
    be.gelu(&input, &output, 8);
    // gelu(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 1e-4);
    // gelu(1) ≈ 0.8412
    try std.testing.expectApproxEqAbs(@as(f32, 0.8412), output[1], 1e-3);
    // gelu(-1) ≈ -0.1588
    try std.testing.expectApproxEqAbs(@as(f32, -0.1588), output[2], 1e-3);
}

test "CpuBackend — add element-wise" {
    var be = CpuBackend{};
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var b_arr = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0 };
    var out: [8]f32 = undefined;
    be.add(&a, &b_arr, &out, 8);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 88.0), out[7], 1e-5);
}

test "CpuBackend — mul element-wise" {
    var be = CpuBackend{};
    var a = [_]f32{ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    var b_arr = [_]f32{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
    var out: [8]f32 = undefined;
    be.mul(&a, &b_arr, &out, 8);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), out[7], 1e-5);
}

test "CpuBackend — rmsNorm" {
    var be = CpuBackend{};
    var input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var weight = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var output: [8]f32 = undefined;
    be.rmsNorm(&input, &weight, &output, 8, 1e-6);
    const rms = @sqrt(@as(f32, 25.5) + 1e-6);
    try std.testing.expectApproxEqAbs(1.0 / rms, output[0], 1e-4);
    try std.testing.expectApproxEqAbs(8.0 / rms, output[7], 1e-4);
}

test "CpuBackend — addRmsNorm" {
    var be = CpuBackend{};
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var b_arr = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var weight = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var output: [8]f32 = undefined;
    be.addRmsNorm(&a, &b_arr, &weight, &output, 8, 1e-6);
    // a = [2,3,4,5,6,7,8,9]
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), a[0], 1e-5);
    const rms = @sqrt(@as(f32, 35.5) + 1e-6);
    try std.testing.expectApproxEqAbs(2.0 / rms, output[0], 1e-4);
    try std.testing.expectApproxEqAbs(9.0 / rms, output[7], 1e-4);
}

test "CpuBackend — rope at pos=0 is identity" {
    var be = CpuBackend{};
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    be.rope(&x, 0, 1, 8, 8, 10000.0);
    // At pos=0, angle=0, cos=1, sin=0 → no change
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), x[4], 1e-5);
}

test "CpuBackend — rope at pos=1 rotates" {
    var be = CpuBackend{};
    var x = [_]f32{ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    be.rope(&x, 1, 1, 8, 8, 10000.0);
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1.0)), x[0], 1e-4);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 1.0)), x[4], 1e-4);
}

test "CpuBackend — softmax normalizes to sum=1" {
    var be = CpuBackend{};
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    be.softmax(&data, 8);
    var sum: f32 = 0.0;
    for (data) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
    // Monotonically increasing
    for (1..8) |i| try std.testing.expect(data[i] >= data[i - 1]);
}

test "CpuBackend — embLookup f32" {
    var be = CpuBackend{};
    const table = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var output: [4]f32 = undefined;
    be.embLookup(.{ .data = @ptrCast(&table), .dtype = .f32 }, 1, &output, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), output[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), output[3], 1e-5);
}

test "CpuBackend — embLookup f16" {
    var be = CpuBackend{};
    const table = [_]f16{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var output: [4]f32 = undefined;
    be.embLookup(.{ .data = @ptrCast(&table), .dtype = .f16 }, 0, &output, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), output[3], 1e-3);
}

test "CpuBackend — siluMul" {
    var be = CpuBackend{};
    var a = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0 };
    var b_arr = [_]f32{ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 };
    var out: [8]f32 = undefined;
    be.siluMul(&a, &b_arr, &out, 8);
    // siluMul(0, 2) = silu(0) * 2 = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-5);
    // siluMul(1, 2) ≈ 0.7311 * 2 = 1.4622
    try std.testing.expectApproxEqAbs(@as(f32, 1.4622), out[1], 1e-3);
}

test "CpuBackend — geluMul" {
    var be = CpuBackend{};
    var a = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.5 };
    var b_arr = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var out: [8]f32 = undefined;
    be.geluMul(&a, &b_arr, &out, 8);
    // geluMul(0, 1) = gelu(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-4);
    // geluMul(1, 1) ≈ 0.8412
    try std.testing.expectApproxEqAbs(@as(f32, 0.8412), out[1], 1e-3);
}

test "CpuBackend — deinterleave" {
    var be = CpuBackend{};
    // 2 pairs, stride=2
    var input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var out_a: [4]f32 = undefined;
    var out_b: [4]f32 = undefined;
    be.deinterleave(&input, &out_a, &out_b, 2, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_a[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out_a[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out_a[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out_b[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), out_b[2], 1e-6);
}

test "CpuBackend — l2Norm" {
    var be = CpuBackend{};
    var x = [_]f32{ 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    be.l2Norm(&x, 8, 1e-12);
    // L2 norm = sqrt(9+16) = 5 → x = [0.6, 0.8, 0, ...]
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), x[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), x[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), x[2], 1e-4);
}

test "CpuBackend — sigmoidMul" {
    var be = CpuBackend{};
    var data = [_]f32{ 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0 };
    var gate = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    be.sigmoidMul(&data, &gate, 8);
    // sigmoid(0) = 0.5 → data[i] *= 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), data[7], 1e-5);
}

test "CpuBackend — rmsNormMulti" {
    var be = CpuBackend{};
    var data = [_]f32{
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
    };
    var weight = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    be.rmsNormMulti(&data, &weight, 2, 8, 1e-6);
    // rms([1,1,...]) = 1 → normalized = 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-4);
    // rms([3,3,...]) = 3 → normalized = 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[8], 1e-4);
}

test "CpuBackend — rmsNormBatched" {
    var be = CpuBackend{};
    var input = [_]f32{
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
    };
    var weight = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var output: [16]f32 = undefined;
    be.rmsNormBatched(&input, &weight, &output, 2, 8, 1e-6);
    // Each row is constant → output = 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[8], 1e-4);
}

test "CpuBackend — ropeBatched" {
    var be = CpuBackend{};
    var x = [_]f32{
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    };
    var positions = [_]u32{ 0, 1 };
    be.ropeBatched(&x, &positions, 2, 1, 8, 8, 10000.0);
    // pos=0 → identity
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[0], 1e-5);
    // pos=1 → rotated
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1.0)), x[8], 1e-4);
}

test "CpuBackend — gemvT with Q8_0" {
    var be = CpuBackend{};
    // 1 input, 32 outputs, 1 Q8_0 block
    var w_block: [34]u8 align(2) = undefined;
    // scale = 1.0 → f16 = 0x3C00
    w_block[0] = 0x00;
    w_block[1] = 0x3C;
    @memset(w_block[2..34], 0);
    w_block[2] = 2; // first quantized weight = 2
    var x_in = [_]f32{3.0};
    var y_out: [32]f32 = undefined;
    be.gemvT(&x_in, &w_block, &y_out, 32, 1);
    // y[0] = 3.0 * 1.0 * 2 = 6.0
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), y_out[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y_out[1], 1e-3);
}

test "CpuBackend — gemm f32" {
    var be = CpuBackend{};
    // Y = X @ W^T
    // X = [[1,2],[3,4]], W = [[1,0],[0,1]] (identity) → Y = X
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var w = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var y: [4]f32 = undefined;
    be.gemm(&x, .{ .data = @ptrCast(&w), .dtype = .f32 }, &y, 2, 2, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), y[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), y[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), y[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), y[3], 1e-5);
}

test "CpuBackend — gemvMulti single op" {
    var be = CpuBackend{};
    var x = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var w = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var y: [1]f32 = undefined;
    const ops = [_]backend_mod.GemvOp{
        .{
            .w = .{ .data = @ptrCast(&w), .dtype = .f32 },
            .y = &y,
            .n = 1,
        },
    };
    be.gemvMulti(&x, &ops, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), y[0], 1e-5);
}

test "CpuBackend — gemvMulti empty ops" {
    var be = CpuBackend{};
    var x = [_]f32{1.0};
    const ops = [_]backend_mod.GemvOp{};
    be.gemvMulti(&x, &ops, 1); // Must not panic
}

test "CpuBackend — sdpa f32 single token" {
    var be = CpuBackend{};
    const nh: usize = 1;
    const nkv: usize = 1;
    const hd: usize = 4;
    const kvd = nkv * hd;
    const max_seq: usize = 4;
    var keys: [max_seq * kvd * @sizeOf(f32)]u8 = undefined;
    var values: [max_seq * kvd * @sizeOf(f32)]u8 = undefined;
    var q = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var k_new = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var v_new = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var output: [4]f32 = undefined;
    be.sdpa(&q, &keys, &values, &k_new, &v_new, &output, nh, nkv, hd, 0, 1.0, .f32, .f32);
    // Single token → output = v_new
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[3], 1e-3);
}

test "CpuBackend — sdpaPrefill f32 single token" {
    var be = CpuBackend{};
    const nh: usize = 1;
    const nkv: usize = 1;
    const hd: usize = 4;
    const kvd = nkv * hd;
    const max_seq: usize = 4;
    var keys: [max_seq * kvd * @sizeOf(f32)]u8 = undefined;
    var values: [max_seq * kvd * @sizeOf(f32)]u8 = undefined;
    // Q, K, V for 1 token prefill
    var q = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var k = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var v = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var output: [4]f32 = undefined;
    be.sdpaPrefill(&q, &k, &v, &keys, &values, &output, nh, nkv, hd, 0, 1, 1.0, .f32, .f32);
    // Single token prefill: output = v (softmax of single score = 1.0)
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[3], 1e-3);
}

test "CpuBackend — sdpaWithStats f32 single token" {
    var be = CpuBackend{};
    const nh: usize = 1;
    const nkv: usize = 1;
    const hd: usize = 4;
    const kvd = nkv * hd;
    const max_seq: usize = 4;
    var keys: [max_seq * kvd * @sizeOf(f32)]u8 = undefined;
    var values: [max_seq * kvd * @sizeOf(f32)]u8 = undefined;
    var q = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var k_new = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var v_new = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var output: [4]f32 = undefined;
    var head_max: [1]f32 = undefined;
    var head_sum: [1]f32 = undefined;
    be.sdpaWithStats(&q, &keys, &values, &k_new, &v_new, &output, &head_max, &head_sum, nh, nkv, hd, 0, 1.0, .f32, .f32);
    // Output should match sdpa single token
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[0], 1e-3);
    // Stats should be finite and sum > 0
    try std.testing.expect(std.math.isFinite(head_max[0]));
    try std.testing.expect(std.math.isFinite(head_sum[0]));
    try std.testing.expect(head_sum[0] > 0);
}

test "fuzz: all cpu functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            _ = smith;
            var be = CpuBackend{};
            const allocator = std.testing.allocator;

            // allocKvSlice / freeKvSlice
            const slice = try be.allocKvSlice(allocator, 64);
            be.freeKvSlice(allocator, slice);

            // sync / beginBatch / endBatch (no-ops)
            be.sync();
            be.beginBatch();
            be.endBatch();

            // backendInfo
            _ = be.backendInfo();

            // silu / gelu
            var act_buf = [_]f32{ 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.0, 0.1 };
            var act_out: [8]f32 = undefined;
            be.silu(&act_buf, &act_out, 8);
            be.gelu(&act_buf, &act_out, 8);

            // add / mul
            var a_buf = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            var b_buf = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
            var c_buf: [8]f32 = undefined;
            be.add(&a_buf, &b_buf, &c_buf, 8);
            be.mul(&a_buf, &b_buf, &c_buf, 8);

            // rmsNorm / addRmsNorm
            var w_buf = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
            be.rmsNorm(&a_buf, &w_buf, &c_buf, 8, 1e-6);
            var a2 = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            be.addRmsNorm(&a2, &b_buf, &w_buf, &c_buf, 8, 1e-6);

            // softmax
            var sm = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            be.softmax(&sm, 8);

            // rope
            var rope_buf = [_]f32{ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            be.rope(&rope_buf, 0, 1, 8, 8, 10000.0);

            // embLookup (f32 table)
            const table = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
            var emb_out: [2]f32 = undefined;
            be.embLookup(.{ .data = @ptrCast(&table), .dtype = .f32 }, 0, &emb_out, 2);

            // l2Norm
            var l2 = [_]f32{ 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            be.l2Norm(&l2, 8, 1e-12);

            // sigmoidMul
            var sig_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            var sig_gate = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            be.sigmoidMul(&sig_data, &sig_gate, 8);

            // siluMul / geluMul
            var sm_a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            var sm_b = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
            var sm_out: [8]f32 = undefined;
            be.siluMul(&sm_a, &sm_b, &sm_out, 8);
            be.geluMul(&sm_a, &sm_b, &sm_out, 8);

            // addScaled / allReduceAdd
            var dst = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            var src_buf = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
            be.addScaled(&src_buf, &dst, 0.5, 8);
            be.allReduceAdd(&dst, &src_buf, 8);

            // rmsNormMulti
            be.rmsNormMulti(&dst, &w_buf, 1, 8, 1e-6);

            // deinterleave
            var di_in = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
            var di_a: [2]f32 = undefined;
            var di_b: [2]f32 = undefined;
            be.deinterleave(&di_in, &di_a, &di_b, 2, 1);

            // splitQGate
            var qg = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
            var q_out: [2]f32 = undefined;
            var g_out: [2]f32 = undefined;
            be.splitQGate(&qg, &q_out, &g_out, 2, 1);

            // gemv (f32)
            const gw = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
            const gx = [_]f32{ 1.0, 2.0 };
            var gy: [2]f32 = undefined;
            be.gemv(&gx, .{ .data = @ptrCast(&gw), .dtype = .f32 }, &gy, 2, 2);

            // gemvMulti (empty)
            const ops = [_]backend_mod.GemvOp{};
            be.gemvMulti(&gx, &ops, 2);

            // gemm
            be.gemm(&gx, .{ .data = @ptrCast(&gw), .dtype = .f32 }, &gy, 1, 2, 2);

            // gemvT
            var w_q8: [34]u8 align(2) = undefined;
            w_q8[0] = 0x00;
            w_q8[1] = 0x3C; // scale = 1.0 in f16
            @memset(w_q8[2..34], 0);
            var gt_x = [_]f32{1.0};
            var gt_y: [32]f32 = undefined;
            be.gemvT(&gt_x, &w_q8, &gt_y, 32, 1);

            // rmsNormBatched / ropeBatched
            var rnb_in = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
            var rnb_out: [8]f32 = undefined;
            be.rmsNormBatched(&rnb_in, &w_buf, &rnb_out, 1, 8, 1e-6);
            var rb_x = [_]f32{ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            var rb_pos = [_]u32{0};
            be.ropeBatched(&rb_x, &rb_pos, 1, 1, 8, 8, 10000.0);

            // sdpa (f32 single token)
            const nh: usize = 1;
            const nkv: usize = 1;
            const hd: usize = 4;
            const kvd = nkv * hd;
            var keys: [4 * kvd * @sizeOf(f32)]u8 = undefined;
            var values: [4 * kvd * @sizeOf(f32)]u8 = undefined;
            var sq = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
            var sk = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
            var sv = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
            var s_out: [4]f32 = undefined;
            be.sdpa(&sq, &keys, &values, &sk, &sv, &s_out, nh, nkv, hd, 0, 1.0, .f32, .f32);

            // sdpaWithStats
            var hm: [1]f32 = undefined;
            var hs: [1]f32 = undefined;
            be.sdpaWithStats(&sq, &keys, &values, &sk, &sv, &s_out, &hm, &hs, nh, nkv, hd, 0, 1.0, .f32, .f32);

            // sdpaPrefill
            be.sdpaPrefill(&sq, &sk, &sv, &keys, &values, &s_out, nh, nkv, hd, 0, 1, 1.0, .f32, .f32);

            // sdpaTree (zero nodes — no-op)
            be.sdpaTree(&sq, @as([*]const u8, @ptrCast(&keys)), @as([*]const u8, @ptrCast(&values)), &sq, &sq, &s_out, @as([*]const [8]u64, &.{.{0} ** 8}), 1, 1, 4, 0, 0, 1.0, .f32, .f32);

            // sdpaPaged (CPU fallback)
            // Not easily callable without PagedKvView setup, verified via comptime ref
            comptime {
                _ = &CpuBackend.sdpaPaged;
                _ = &CpuBackend.deltaNet;
                _ = &CpuBackend.gemvNvfp4St;
                _ = &CpuBackend.gemvMlxQ;
                _ = &CpuBackend.gemvMxfp4St;
                _ = &CpuBackend.gemvGptq;
                _ = &CpuBackend.gemvAwq;
            }
        }
    }.f, .{});
}

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
