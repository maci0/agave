//! Backend abstraction for compute operations.
//! Uses a tagged union with `inline else` dispatch for zero-overhead
//! backend selection, no VTable indirection in the hot path.
//!
//! ## UMA (Unified Memory Architecture) contract
//!
//! On UMA platforms (Apple Silicon, NVIDIA Grace/GB10, AMD APU/Ryzen AI Max+),
//! the CPU and GPU share the same physical DRAM. GPU backends on these platforms
//! can wrap existing CPU allocations as GPU buffers with zero copies:
//!
//!   - **Metal**: `newBufferWithBytesNoCopy` + `storageModeShared`
//!   - **Vulkan**: `HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL` memory type
//!   - **CUDA**: `cudaMallocManaged` or `cudaHostAlloc(cudaHostAllocMapped)`
//!   - **ROCm/HIP**: `hipMallocManaged` or `hipHostMalloc(hipHostMallocMapped)`
//!
//! All GPU backends use deferred dispatch, operations are encoded into command
//! buffers without blocking. Models call `be.sync()` only at points where CPU
//! code reads GPU-produced data. On discrete GPUs, `sync()` must also copy
//! results back; on UMA, results are already visible in CPU address space.
//!
//! Implementations: cpu.zig, metal.zig, vulkan.zig, cuda.zig, rocm.zig, webgpu.zig

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const ThreadPool = @import("../thread_pool.zig").ThreadPool;
const quant_ops = @import("../ops/quant.zig");

/// Tensor data with type info for backend dispatch
pub const TensorData = struct {
    data: [*]const u8,
    dtype: DType,
};

/// A single GEMV operation within a batched dispatch: y[n] = W[n,k] @ x.
/// All ops in a batch share the same input vector x and input dimension k.
pub const GemvOp = struct {
    w: TensorData,
    y: [*]f32,
    n: usize,
    /// Optional MLX companion pointers for quantized weights.
    /// When non-null, the GEMV dispatches the MLX-Q kernel instead of standard dequant.
    mlx_scales: ?[*]const u8 = null,
    mlx_biases: ?[*]const u8 = null,
    mlx_bits: u32 = 0,
    /// MLX quantization group size (elements per scale/bias pair). Defaults to 64.
    mlx_group_size: u32 = 64,
};

/// Supported tensor data types, canonical definition in format/format.zig,
/// re-exported here for backend consumers.
pub const DType = @import("../format/format.zig").DType;

/// KV cache quantization type, re-exported for backend consumers.
pub const KvQuantType = @import("../ops/kv_quant.zig").KvQuantType;

/// Paged KV cache view for block-table-indexed SDPA.
pub const PagedKvView = @import("../kvcache/manager.zig").PagedKvView;

/// Parameters for DeltaNet SSM recurrence (Qwen3.5 hybrid model).
/// Passed to `Backend.deltaNet()` to keep the function signature manageable.
pub const DeltaNetParams = struct {
    conv_ch: u32,
    d_conv: u32,
    d_inner: u32,
    num_k_heads: u32,
    head_k_dim: u32,
    num_v_heads: u32,
    head_v_dim: u32,
    q_scale: f32,
    rms_eps: f32,
    /// True when conv_out split order is K,Q,V (HuggingFace/SafeTensors).
    /// False (default) when split order is Q,K,V (GGUF/llama.cpp convention).
    kqv_order: bool = false,
};

/// Backend and system startup information, partially filled by each backend
/// during init, remainder populated by the caller (see "populated by main" fields).
/// Displayed in the system info line after the model banner.
pub const BackendInfo = struct {
    /// Backend display name (e.g., "Metal", "CUDA", "Vulkan", "ROCm", "CPU").
    name: []const u8 = "CPU",

    /// GPU device name (e.g., "Apple M4 Pro", "NVIDIA GB10").
    device_name: []const u8 = "",

    /// Dynamic library loaded via dlopen (e.g., "libcuda.so.1", "libkosmickrisp.dylib").
    lib_name: []const u8 = "",

    /// Number of GPU kernels/pipelines loaded at init.
    n_gpu_kernels: u32 = 0,

    /// Kernel type label (e.g., "MSL", "PTX", "SPIR-V", "HSACO").
    kernel_type: []const u8 = "",

    /// Total device memory in bytes (VRAM or unified memory).
    total_mem: usize = 0,

    /// Available (free) device memory in bytes at query time.
    avail_mem: usize = 0,

    /// Whether the GPU shares memory with the CPU (unified memory architecture).
    is_uma: bool = false,

    /// Compute capability string (e.g., "sm_121" for CUDA, "gfx1100" for ROCm).
    compute_cap: []const u8 = "",

    /// Driver/API version string (e.g., "CUDA 13.0", "Vulkan 1.3").
    driver_version: []const u8 = "",

    // ── System-level info (populated by caller, not backend) ──

    /// OS version string (e.g., "macOS 14.2.1", "Linux 6.5.0"). Populated by main.
    os_version: []const u8 = "",

    /// CPU thread count (populated by main).
    n_threads: u32 = 0,

    /// Total system physical memory in bytes (populated by main).
    system_mem: usize = 0,

    /// Available system memory in bytes at query time (populated by main).
    system_avail: usize = 0,

    /// CPU cache sizes in bytes (0 = not detected). Populated by main.
    l1_cache: usize = 0,
    l2_cache: usize = 0,
    l3_cache: usize = 0,

    /// Host architecture string (e.g., "aarch64", "x86_64").
    arch: []const u8 = @tagName(builtin.cpu.arch),

    /// Host OS string (e.g., "macos", "linux").
    os: []const u8 = @tagName(builtin.os.tag),
};

/// CPU cache sizes returned by detectCacheSizes().
pub const CacheSizes = struct { l1: usize = 0, l2: usize = 0, l3: usize = 0 };

/// Detect total system physical memory in bytes. Re-exported from cpu.zig.
pub const detectSystemMem = @import("cpu.zig").detectSystemMem;

/// Detect CPU cache sizes (L1d, L2, L3). Re-exported from cpu.zig.
pub const detectCacheSizes = @import("cpu.zig").detectCacheSizes;

/// Detect available (free) system memory in bytes. Re-exported from cpu.zig.
pub const detectAvailMem = @import("cpu.zig").detectAvailMem;

/// Physical (non-SMT) core count usable by this process. See `cpu.zig`.
pub const detectPhysicalCores = @import("cpu.zig").detectPhysicalCores;

/// One logical CPU id per physical core, for thread-pool affinity. See `cpu.zig`.
pub const physicalCoreIds = @import("cpu.zig").physicalCoreIds;

/// Core ids collected for pool affinity. The pool caps at 31 workers plus the
/// main thread, so listing more cores than this cannot change the pool size.
const max_pinned_core_ids: usize = 64;

/// A page-aligned byte range covering an arbitrary host pointer and length.
pub const PageRange = struct { base: usize, size: usize };

/// Widen `ptr[0..len]` to whole pages. Every host-memory syscall that takes a
/// range (mlock, madvise, cuMemHostRegister, hipHostRegister) requires this, and
/// they must all round the SAME way or an unregister misses its base address.
pub fn pageAlignRange(ptr: [*]const u8, len: usize) PageRange {
    const page = std.heap.page_size_min;
    const addr = @intFromPtr(ptr);
    const base = addr & ~@as(usize, page - 1);
    return .{ .base = base, .size = std.mem.alignForward(usize, addr + len, page) - base };
}

/// Fault a host range into physical memory, so a later `hostRegister` (or
/// mlock) is pure page locking instead of a page-at-a-time synchronous read.
///
/// `MADV_WILLNEED` starts the read asynchronously with the kernel's readahead
/// window, which is what makes this fast; the driver's own faulting is
/// page-at-a-time and is not. Best effort: an unsupported platform or a range
/// the kernel declines to prefault just means the register pays the fault cost.
pub fn hostPrefault(ptr: [*]const u8, len: usize) void {
    if (comptime builtin.os.tag != .linux and builtin.os.tag != .macos) return;
    if (len == 0) return;
    const r = pageAlignRange(ptr, len);
    const aligned: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(r.base);
    std.posix.madvise(aligned, r.size, std.posix.system.MADV.WILLNEED) catch {};
}

/// Detect OS version string (e.g., "macOS 14.2.1", "Linux 6.5.0"). Re-exported from cpu.zig.
pub const detectOsVersion = @import("cpu.zig").detectOsVersion;

/// CPU softmax kernel, re-exported for the attention module.
/// The windowed attention fallback runs entirely on CPU, so it needs a CPU-only
/// softmax that avoids be.softmax() (which dispatches to GPU, causing an expensive sync).
pub const CpuSoftmax = struct {
    pub const softmaxSimd = @import("kernels/cpu/softmax.zig").softmaxSimd;
};

/// CPU SDPA kernel functions, re-exported for the split-attention module.
/// Split-attention runs CPU SDPA concurrently with GPU SDPA for tiered KV cache
/// offloading, so it needs direct access to per-head CPU kernel functions
/// (bypassing Backend dispatch which would route to the active GPU backend).
pub const CpuSdpa = struct {
    const kernel = @import("kernels/cpu/sdpa.zig");
    /// Compute one attention head with quantized KV (windowed, single-threaded).
    pub const sdpaQuantHead = kernel.sdpaQuantHead;
    /// Same as sdpaQuantHead but also returns softmax stats (max, sum) for online correction.
    pub const sdpaQuantHeadWithStats = kernel.sdpaQuantHeadWithStats;
    /// Multi-head dispatch: splits heads across thread pool workers.
    pub const sdpaQuantHeads = kernel.sdpaQuantHeads;
};

/// Pre-allocated capacity for GPU buffer caches (weights + activations + KV).
/// Used by Metal, CUDA, Vulkan, and ROCm backends to avoid OOM during hot-path cache puts.
pub const buf_cache_initial_capacity: usize = 512;

/// Elements per small quantization block (Q4_0, Q8_0, etc.).
pub const quant_block_elems: usize = quant_ops.quant_block_elems;
/// Elements per large quantization super-block (Q4_K, Q5_K, Q6_K, etc.).
pub const quant_super_block_elems: usize = 256;
/// Elements per NVFP4 block (8 nibble pairs + 1 scale byte).
pub const nvfp4_block_elems: usize = 16;

// ── Element and block byte sizes ──────────────────────────────────────
/// Byte size per element for non-quantized types, and per block for quantized
/// formats. Used by weightBytes, gemvRowBytes, and model dtypeBytes.
/// f32: 4 bytes per element.
pub const f32_elem_bytes: usize = 4;
/// f16 / bf16: 2 bytes per element.
pub const f16_elem_bytes: usize = 2;
/// Q4_0: f16 scale + 16B quants = 18 bytes per 32-element block.
pub const q4_0_block_bytes: usize = quant_ops.q4_0_block_bytes;
/// Q4_1: f16 scale + f16 min + 16B quants = 20 bytes per 32-element block.
pub const q4_1_block_bytes: usize = 20;
/// Q5_0: f16 scale + 4B high bits + 16B quants = 22 bytes per 32-element block.
pub const q5_0_block_bytes: usize = 22;
/// Q8_0: f16 scale + 32B quants = 34 bytes per 32-element block.
pub const q8_0_block_bytes: usize = quant_ops.q8_0_block_bytes;
/// Q2_K: 84 bytes per 256-element super-block.
pub const q2_k_block_bytes: usize = 84;
/// Q3_K: 110 bytes per 256-element super-block.
pub const q3_k_block_bytes: usize = 110;
/// Q4_K: 144 bytes per 256-element super-block.
pub const q4_k_block_bytes: usize = 144;
/// Q5_K: 176 bytes per 256-element super-block.
pub const q5_k_block_bytes: usize = 176;
/// Q6_K: 210 bytes per 256-element super-block.
pub const q6_k_block_bytes: usize = 210;
/// IQ4_NL: 18 bytes per 32-element block (same layout as Q4_0).
pub const iq4_nl_block_bytes: usize = quant_ops.iq4_nl_block_bytes;
/// IQ4_XS: 136 bytes per 256-element super-block.
/// Layout: f16 d (2) + u16 scales_h (2) + u8 scales_l[4] (4) + u8 qs[128] (128).
pub const iq4_xs_block_bytes: usize = quant_ops.iq4_xs_block_bytes;
/// MXFP4: 16B quants + 1B scale = 17 bytes per 32-element block.
pub const mxfp4_block_bytes: usize = 17;
/// NVFP4: 8B quants + 1B scale = 9 bytes per 16-element block.
pub const nvfp4_block_bytes: usize = 9;
/// TQ1_0: 54 bytes per 256-element super-block.
/// Layout: f16 scale (2) + qs[48] (48) + qh[4] (4) = 54.
pub const tq1_0_block_bytes: usize = 54;
/// TQ2_0: 66 bytes per 256-element super-block (f16 scale + 64 bytes data).
pub const tq2_0_block_bytes: usize = 66;
/// IQ3_XXS: 98 bytes per 256-element super-block.
pub const iq3_xxs_block_bytes: usize = 98;
/// IQ3_S: 110 bytes per 256-element super-block.
pub const iq3_s_block_bytes: usize = 110;
/// IQ2_XXS: 66 bytes per 256-element super-block.
pub const iq2_xxs_block_bytes: usize = 66;
/// IQ2_XS: 74 bytes per 256-element super-block.
pub const iq2_xs_block_bytes: usize = 74;
/// IQ2_S: 82 bytes per 256-element super-block.
pub const iq2_s_block_bytes: usize = 82;
/// IQ1_S: 50 bytes per 256-element super-block.
pub const iq1_s_block_bytes: usize = 50;
/// IQ1_M: 56 bytes per 256-element super-block.
pub const iq1_m_block_bytes: usize = 56;

/// Compute raw byte size of a weight matrix [n, k] for a given dtype.
/// Used by GPU backends to determine upload buffer sizes. Accounts for
/// quantization block structure (e.g., Q4_0 = 18 bytes per 32-element block).
pub fn weightBytes(dtype: DType, n: usize, k: usize) usize {
    const nb = (std.math.add(usize, k, quant_block_elems - 1) catch std.math.maxInt(usize)) / quant_block_elems;
    const nsb = (std.math.add(usize, k, quant_super_block_elems - 1) catch std.math.maxInt(usize)) / quant_super_block_elems;
    const nvb = (std.math.add(usize, k, nvfp4_block_elems - 1) catch std.math.maxInt(usize)) / nvfp4_block_elems;
    return switch (dtype) {
        .f32 => std.math.mul(usize, std.math.mul(usize, n, k) catch std.math.maxInt(usize), f32_elem_bytes) catch std.math.maxInt(usize),
        .f16, .bf16 => std.math.mul(usize, std.math.mul(usize, n, k) catch std.math.maxInt(usize), f16_elem_bytes) catch std.math.maxInt(usize),
        .fp8_e4m3, .fp8_e5m2 => std.math.mul(usize, n, k) catch std.math.maxInt(usize),
        .q8_0 => std.math.mul(usize, std.math.mul(usize, n, nb) catch std.math.maxInt(usize), q8_0_block_bytes) catch std.math.maxInt(usize),
        .q4_0 => std.math.mul(usize, std.math.mul(usize, n, nb) catch std.math.maxInt(usize), q4_0_block_bytes) catch std.math.maxInt(usize),
        .q4_1 => std.math.mul(usize, std.math.mul(usize, n, nb) catch std.math.maxInt(usize), q4_1_block_bytes) catch std.math.maxInt(usize),
        .q5_0 => std.math.mul(usize, std.math.mul(usize, n, nb) catch std.math.maxInt(usize), q5_0_block_bytes) catch std.math.maxInt(usize),
        .q4_k => std.math.mul(usize, std.math.mul(usize, n, nsb) catch std.math.maxInt(usize), q4_k_block_bytes) catch std.math.maxInt(usize),
        .q5_k => std.math.mul(usize, std.math.mul(usize, n, nsb) catch std.math.maxInt(usize), q5_k_block_bytes) catch std.math.maxInt(usize),
        .q6_k => std.math.mul(usize, std.math.mul(usize, n, nsb) catch std.math.maxInt(usize), q6_k_block_bytes) catch std.math.maxInt(usize),
        .mxfp4 => std.math.mul(usize, std.math.mul(usize, n, nb) catch std.math.maxInt(usize), mxfp4_block_bytes) catch std.math.maxInt(usize),
        .q2_k => std.math.mul(usize, std.math.mul(usize, n, nsb) catch std.math.maxInt(usize), q2_k_block_bytes) catch std.math.maxInt(usize),
        .q3_k => std.math.mul(usize, std.math.mul(usize, n, nsb) catch std.math.maxInt(usize), q3_k_block_bytes) catch std.math.maxInt(usize),
        .iq4_nl => std.math.mul(usize, std.math.mul(usize, n, nb) catch std.math.maxInt(usize), iq4_nl_block_bytes) catch std.math.maxInt(usize),
        .iq4_xs => std.math.mul(usize, std.math.mul(usize, n, nsb) catch std.math.maxInt(usize), iq4_xs_block_bytes) catch std.math.maxInt(usize),
        .iq3_xxs => std.math.mul(usize, std.math.mul(usize, n, nsb) catch std.math.maxInt(usize), iq3_xxs_block_bytes) catch std.math.maxInt(usize),
        .iq3_s => std.math.mul(usize, std.math.mul(usize, n, nsb) catch std.math.maxInt(usize), iq3_s_block_bytes) catch std.math.maxInt(usize),
        .iq2_xxs => std.math.mul(usize, std.math.mul(usize, n, nsb) catch std.math.maxInt(usize), iq2_xxs_block_bytes) catch std.math.maxInt(usize),
        .iq2_xs => std.math.mul(usize, std.math.mul(usize, n, nsb) catch std.math.maxInt(usize), iq2_xs_block_bytes) catch std.math.maxInt(usize),
        .iq2_s => std.math.mul(usize, std.math.mul(usize, n, nsb) catch std.math.maxInt(usize), iq2_s_block_bytes) catch std.math.maxInt(usize),
        .iq1_s => std.math.mul(usize, std.math.mul(usize, n, nsb) catch std.math.maxInt(usize), iq1_s_block_bytes) catch std.math.maxInt(usize),
        .iq1_m => std.math.mul(usize, std.math.mul(usize, n, nsb) catch std.math.maxInt(usize), iq1_m_block_bytes) catch std.math.maxInt(usize),
        .tq1_0 => std.math.mul(usize, std.math.mul(usize, n, nsb) catch std.math.maxInt(usize), tq1_0_block_bytes) catch std.math.maxInt(usize),
        .tq2_0 => std.math.mul(usize, std.math.mul(usize, n, nsb) catch std.math.maxInt(usize), tq2_0_block_bytes) catch std.math.maxInt(usize),
        .nvfp4 => std.math.mul(usize, std.math.mul(usize, n, nvb) catch std.math.maxInt(usize), nvfp4_block_bytes) catch std.math.maxInt(usize),
        // GPTQ/AWQ: 8 INT4 nibbles per u32 word
        .gptq, .awq => (std.math.mul(usize, n, k) catch std.math.maxInt(usize)) / 2,
        // HQQ: 2 nibbles/byte, companion scale/zero handled separately
        .hqq => (std.math.mul(usize, n, k) catch std.math.maxInt(usize)) / 2,
        // Unsupported dtypes: assume f32 (4 bytes per element).
        .mlx_q, .unknown => std.math.mul(usize, std.math.mul(usize, n, k) catch std.math.maxInt(usize), 4) catch std.math.maxInt(usize),
    };
}

/// Row stride in bytes for a given dtype and column count.
/// Used by parallel GEMV and TP sharding to compute per-row offsets.
pub fn gemvRowBytes(dtype: DType, k: usize) usize {
    const nb = (std.math.add(usize, k, quant_block_elems - 1) catch std.math.maxInt(usize)) / quant_block_elems;
    const nsb = (std.math.add(usize, k, quant_super_block_elems - 1) catch std.math.maxInt(usize)) / quant_super_block_elems;
    const nvb = (std.math.add(usize, k, nvfp4_block_elems - 1) catch std.math.maxInt(usize)) / nvfp4_block_elems;
    return switch (dtype) {
        .q4_0 => std.math.mul(usize, nb, q4_0_block_bytes) catch std.math.maxInt(usize),
        .q4_1 => std.math.mul(usize, nb, q4_1_block_bytes) catch std.math.maxInt(usize),
        .q5_0 => std.math.mul(usize, nb, q5_0_block_bytes) catch std.math.maxInt(usize),
        .q8_0 => std.math.mul(usize, nb, q8_0_block_bytes) catch std.math.maxInt(usize),
        .q2_k => std.math.mul(usize, nsb, q2_k_block_bytes) catch std.math.maxInt(usize),
        .q3_k => std.math.mul(usize, nsb, q3_k_block_bytes) catch std.math.maxInt(usize),
        .q4_k => std.math.mul(usize, nsb, q4_k_block_bytes) catch std.math.maxInt(usize),
        .q5_k => std.math.mul(usize, nsb, q5_k_block_bytes) catch std.math.maxInt(usize),
        .q6_k => std.math.mul(usize, nsb, q6_k_block_bytes) catch std.math.maxInt(usize),
        .iq4_nl => std.math.mul(usize, nb, iq4_nl_block_bytes) catch std.math.maxInt(usize),
        .iq4_xs => std.math.mul(usize, nsb, iq4_xs_block_bytes) catch std.math.maxInt(usize),
        .iq3_xxs => std.math.mul(usize, nsb, iq3_xxs_block_bytes) catch std.math.maxInt(usize),
        .iq3_s => std.math.mul(usize, nsb, iq3_s_block_bytes) catch std.math.maxInt(usize),
        .iq2_xxs => std.math.mul(usize, nsb, iq2_xxs_block_bytes) catch std.math.maxInt(usize),
        .iq2_xs => std.math.mul(usize, nsb, iq2_xs_block_bytes) catch std.math.maxInt(usize),
        .iq2_s => std.math.mul(usize, nsb, iq2_s_block_bytes) catch std.math.maxInt(usize),
        .iq1_s => std.math.mul(usize, nsb, iq1_s_block_bytes) catch std.math.maxInt(usize),
        .iq1_m => std.math.mul(usize, nsb, iq1_m_block_bytes) catch std.math.maxInt(usize),
        .mxfp4 => std.math.mul(usize, nb, mxfp4_block_bytes) catch std.math.maxInt(usize),
        .nvfp4 => std.math.mul(usize, nvb, nvfp4_block_bytes) catch std.math.maxInt(usize),
        .f16, .bf16 => std.math.mul(usize, k, f16_elem_bytes) catch std.math.maxInt(usize),
        .f32 => std.math.mul(usize, k, f32_elem_bytes) catch std.math.maxInt(usize),
        .fp8_e4m3, .fp8_e5m2 => k,
        .tq1_0, .tq2_0, .mlx_q, .gptq, .awq, .hqq, .unknown => 0,
    };
}

/// Placeholder for backends disabled at build time.
/// The tagged union variant exists but can never be instantiated.
/// init() is a @compileError; methods are unreachable stubs for inline else.
pub const NullBackend = struct {
    /// Compile error, this backend was disabled at build time.
    pub fn init(_: std.mem.Allocator, _: u32) error{BackendDisabled}!NullBackend {
        @compileError("this backend was disabled at build time");
    }

    // Stub methods, unreachable because the variant is never constructed.

    pub fn allocKvSlice(_: *NullBackend, _: std.mem.Allocator, _: usize) error{OutOfMemory}![]u8 {
        unreachable;
    }

    pub fn freeKvSlice(_: *NullBackend, _: std.mem.Allocator, _: []u8) void {
        unreachable;
    }

    pub fn hostRegister(_: *NullBackend, _: [*]const u8, _: usize) bool {
        unreachable;
    }

    pub fn hostUnregister(_: *NullBackend, _: [*]const u8, _: usize) void {
        unreachable;
    }

    pub fn gemv(_: *NullBackend, _: [*]const f32, _: TensorData, _: [*]f32, _: usize, _: usize) void {
        unreachable;
    }

    pub fn rmsNorm(_: *NullBackend, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize, _: f32) void {
        unreachable;
    }

    pub fn silu(_: *NullBackend, _: [*]const f32, _: [*]f32, _: usize) void {
        unreachable;
    }

    pub fn gelu(_: *NullBackend, _: [*]const f32, _: [*]f32, _: usize) void {
        unreachable;
    }

    pub fn add(_: *NullBackend, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize) void {
        unreachable;
    }

    pub fn gemvT(_: *NullBackend, _: [*]const f32, _: [*]const u8, _: [*]f32, _: usize, _: usize) void {
        unreachable;
    }

    pub fn addScaled(_: *NullBackend, _: [*]const f32, _: [*]f32, _: f32, _: usize) void {
        unreachable;
    }

    pub fn addRmsNorm(_: *NullBackend, _: [*]f32, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize, _: f32) void {
        unreachable;
    }

    pub fn rmsNormAdd(_: *NullBackend, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize, _: f32) void {
        unreachable;
    }

    pub fn mul(_: *NullBackend, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize) void {
        unreachable;
    }

    pub fn softmax(_: *NullBackend, _: [*]f32, _: usize) void {
        unreachable;
    }

    pub fn rope(_: *NullBackend, _: [*]f32, _: usize, _: usize, _: usize, _: usize, _: f32) void {
        unreachable;
    }

    pub fn ropeMrope(_: *NullBackend, _: [*]f32, _: usize, _: usize, _: usize, _: usize, _: usize, _: usize, _: f32) void {
        unreachable;
    }

    pub fn embLookup(_: *NullBackend, _: TensorData, _: u32, _: [*]f32, _: usize) void {
        unreachable;
    }

    pub fn l2Norm(_: *NullBackend, _: [*]f32, _: usize, _: f32) void {
        unreachable;
    }

    pub fn sigmoidMul(_: *NullBackend, _: [*]f32, _: [*]const f32, _: usize) void {
        unreachable;
    }

    pub fn clampedSiluMul(_: *NullBackend, gate: [*]const f32, up: [*]const f32, out: [*]f32, n: usize) void {
        for (0..n) |i| {
            const g = @min(@as(f32, 10.0), @max(@as(f32, -10.0), gate[i]));
            const u = @min(@as(f32, 10.0), @max(@as(f32, -10.0), up[i]));
            out[i] = (g / (1.0 + @exp(-g))) * u;
        }
    }
    pub fn siluMul(_: *NullBackend, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize) void {
        unreachable;
    }

    pub fn geluMul(_: *NullBackend, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize) void {
        unreachable;
    }

    pub fn rmsNormMulti(_: *NullBackend, _: [*]f32, _: [*]const f32, _: usize, _: usize, _: f32) void {
        unreachable;
    }

    pub fn deinterleave(_: *NullBackend, _: [*]const f32, _: [*]f32, _: [*]f32, _: usize, _: usize) void {
        unreachable;
    }

    pub fn splitQGate(_: *NullBackend, _: [*]const f32, _: [*]f32, _: [*]f32, _: usize, _: usize) void {
        unreachable;
    }

    pub fn sync(_: *NullBackend) void {
        unreachable;
    }

    pub fn sdpa(_: *NullBackend, _: [*]const f32, _: []u8, _: []u8, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize, _: usize, _: usize, _: usize, _: f32, _: KvQuantType, _: KvQuantType) void {
        unreachable;
    }

    pub fn sdpaWithStats(_: *NullBackend, _: [*]const f32, _: []u8, _: []u8, _: [*]const f32, _: [*]const f32, _: [*]f32, _: [*]f32, _: [*]f32, _: usize, _: usize, _: usize, _: usize, _: f32, _: KvQuantType, _: KvQuantType) void {
        unreachable;
    }

    pub fn sdpaTree(_: *NullBackend, _: [*]const f32, _: [*]const u8, _: [*]const u8, _: [*]const f32, _: [*]const f32, _: [*]f32, _: [*]const [8]u64, _: usize, _: usize, _: usize, _: usize, _: u32, _: f32, _: KvQuantType, _: KvQuantType) void {
        unreachable;
    }

    pub fn sdpaPaged(_: *NullBackend, _: [*]const f32, _: PagedKvView, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize, _: usize, _: usize, _: f32, _: KvQuantType, _: KvQuantType) void {
        unreachable;
    }

    pub fn gemvGptq(_: *NullBackend, _: [*]const f32, _: [*]const u32, _: [*]const u16, _: [*]const u32, _: [*]f32, _: usize, _: usize, _: u32) void {
        unreachable;
    }

    pub fn gemvAwq(_: *NullBackend, _: [*]const f32, _: [*]const u32, _: [*]const u16, _: [*]const u32, _: [*]f32, _: usize, _: usize, _: u32) void {
        unreachable;
    }

    pub fn gemvHqq(_: *NullBackend, _: [*]const f32, _: [*]const u8, _: [*]const u8, _: [*]const u8, _: [*]f32, _: usize, _: usize, _: u32) void {
        unreachable;
    }

    pub fn gemvNvfp4St(_: *NullBackend, _: [*]const f32, _: [*]const u8, _: [*]const u8, _: [*]f32, _: usize, _: usize) void {
        unreachable;
    }

    pub fn gemvMlxQ(_: *NullBackend, _: [*]const f32, _: [*]const u8, _: [*]const u8, _: [*]const u8, _: [*]f32, _: usize, _: usize, _: u32, _: u32) void {
        unreachable;
    }
    pub fn gemvMlxQGpu(_: *NullBackend, _: [*]const f32, _: [*]const u8, _: [*]const u8, _: [*]const u8, _: [*]f32, _: usize, _: usize, _: u32, _: u32) void {
        unreachable;
    }

    pub fn gemvMxfp4St(_: *NullBackend, _: [*]const f32, _: [*]const u8, _: [*]const u8, _: [*]f32, _: usize, _: usize, _: usize, _: Mxfp4ScaleFormat) void {
        unreachable;
    }
    pub fn gemvMxfp4StGpu(_: *NullBackend, _: [*]const f32, _: [*]const u8, _: [*]const u8, _: [*]f32, _: usize, _: usize, _: usize, _: Mxfp4ScaleFormat) void {
        unreachable;
    }

    /// Batched MXFP4 expert GEMV (one launch for many slots). Only CUDA
    /// implements it; other backends fall back to per-slot calls.
    pub fn gemvMxfp4StBatched(_: *NullBackend, _: []const u64, _: []const u64, _: []const u64, _: []const [*]f32, _: usize, _: usize, _: usize, _: Mxfp4ScaleFormat) void {
        unreachable;
    }

    /// Device pointer for a weight range (see CudaBackend.getWeightDevicePtr).
    pub fn getWeightDevicePtr(_: *NullBackend, _: [*]const u8, _: usize) u64 {
        unreachable;
    }

    /// Device pointer for an input activation (see CudaBackend.getInputDevicePtr).
    pub fn getInputDevicePtr(_: *NullBackend, _: [*]const f32, _: usize) u64 {
        unreachable;
    }

    const Mxfp4ScaleFormat = @import("../ops/mlx.zig").Mxfp4ScaleFormat;

    pub fn gemvMulti(_: *NullBackend, _: [*]const f32, _: []const GemvOp, _: usize) void {
        unreachable;
    }

    pub fn gemm(_: *NullBackend, _: [*]const f32, _: TensorData, _: [*]f32, _: usize, _: usize, _: usize) void {
        unreachable;
    }

    pub fn rmsNormBatched(_: *NullBackend, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize, _: usize, _: f32) void {
        unreachable;
    }

    pub fn ropeBatched(_: *NullBackend, _: [*]f32, _: [*]const u32, _: usize, _: usize, _: usize, _: usize, _: f32) void {
        unreachable;
    }

    pub fn sdpaPrefill(_: *NullBackend, _: [*]const f32, _: [*]const f32, _: [*]const f32, _: []u8, _: []u8, _: [*]f32, _: usize, _: usize, _: usize, _: usize, _: usize, _: f32, _: KvQuantType, _: KvQuantType) void {
        unreachable;
    }

    pub fn deltaNet(_: *NullBackend, _: [*]const f32, _: [*]f32, _: [*]const f32, _: [*]const f32, _: [*]const f32, _: [*]f32, _: [*]f32, _: []f32, _: [*]const f32, _: [*]const f32, _: [*]const f32, _: [*]const f32, _: DeltaNetParams) void {
        unreachable;
    }

    pub fn beginBatch(_: *NullBackend) void {
        unreachable;
    }

    pub fn endBatch(_: *NullBackend) void {
        unreachable;
    }

    pub fn backendInfo(_: *const NullBackend) BackendInfo {
        unreachable;
    }

    pub fn resetCounters(_: *NullBackend) void {}

    dispatch_count: u32 = 0,
    barrier_count: u32 = 0,
    sync_count: u32 = 0,
};

/// CPU backend, accessed through Backend union dispatch.
/// Public for test access; production code should use Backend union, not concrete types.
pub const CpuBackend = if (build_options.enable_cpu)
    @import("cpu.zig").CpuBackend
else
    NullBackend;

/// Metal GPU backend.
/// On non-macOS platforms, aliases NullBackend so the tagged union remains valid;
/// the .metal variant is simply never constructed.
pub const MetalBackend = if (build_options.enable_metal and builtin.os.tag == .macos)
    @import("metal.zig").MetalBackend
else
    NullBackend;

/// Vulkan GPU backend.
/// Disabled when cross-compiling without Vulkan headers/libs available.
pub const VulkanBackend = if (build_options.enable_vulkan)
    @import("vulkan.zig").VulkanBackend
else
    NullBackend;

/// CUDA GPU backend.
pub const CudaBackend = if (build_options.enable_cuda)
    @import("cuda.zig").CudaBackend
else
    NullBackend;

/// ROCm GPU backend.
pub const RocmBackend = if (build_options.enable_rocm)
    @import("rocm.zig").RocmBackend
else
    NullBackend;

/// WebGPU backend via wgpu-native.
pub const WebGpuBackend = if (build_options.enable_webgpu)
    @import("webgpu.zig").WebGpuBackend
else
    NullBackend;

/// Backend interface, all compute goes through this tagged union.
/// Dispatch is resolved via `inline else`, giving the compiler full visibility
/// into each backend's implementation for inlining and optimization.
/// No VTable pointer indirection, no `*anyopaque` casts.
pub const Backend = union(enum) {
    cpu: *CpuBackend,
    metal: *MetalBackend,
    vulkan: *VulkanBackend,
    cuda: *CudaBackend,
    rocm: *RocmBackend,
    webgpu: *WebGpuBackend,

    /// Allocate a KV cache slice using backend-optimal memory.
    /// `n` is the byte count (caller computes via kvSliceBytes).
    /// On UMA (Metal/Apple Silicon): page-aligned for zero-copy GPU access.
    /// On CUDA UMA: cudaMallocManaged for direct GPU access.
    /// On CUDA discrete: pinned host memory for fast transfers.
    /// On CPU/Vulkan/ROCm: plain allocator.
    /// The allocator is used as fallback; GPU backends may ignore it.
    pub inline fn allocKvSlice(self: Backend, allocator: std.mem.Allocator, n: usize) error{OutOfMemory}![]u8 {
        switch (self) {
            inline else => |be| return be.allocKvSlice(allocator, n),
        }
    }

    /// Free a KV cache slice allocated via allocKvSlice.
    /// Must use the same backend and allocator as allocation.
    pub inline fn freeKvSlice(self: Backend, allocator: std.mem.Allocator, slice: []u8) void {
        switch (self) {
            inline else => |be| be.freeKvSlice(allocator, slice),
        }
    }

    /// Page-lock a host byte range so the backend can DMA out of it directly,
    /// without staging through a driver bounce buffer. Required for async H2D:
    /// `cuMemcpyHtoDAsync` rejects pageable source memory.
    ///
    /// PRECONDITION: the range must already be RESIDENT. Registering a cold
    /// file-backed mapping makes the driver fault every page in one at a time.
    /// Measured on this codebase: registering the 155 GB DS4 checkpoint ran at
    /// ~19 MB/s, 23 minutes for the first 8 shards. Fill or prefault the range
    /// first (`hostPrefault`), then register, so registration is pure page
    /// locking with no I/O behind it.
    ///
    /// Returns false when the backend needs no registration (unified memory) or
    /// the driver refused. Never fatal: an unregistered range still works, it
    /// just goes over the slower pageable copy path.
    pub inline fn hostRegister(self: Backend, ptr: [*]const u8, len: usize) bool {
        switch (self) {
            inline else => |be| return be.hostRegister(ptr, len),
        }
    }

    /// Release a range page-locked by `hostRegister`. Safe on a range that was
    /// never registered (the backend ignores the driver's error).
    pub inline fn hostUnregister(self: Backend, ptr: [*]const u8, len: usize) void {
        switch (self) {
            inline else => |be| be.hostUnregister(ptr, len),
        }
    }

    /// Compute y[n] = W[n,k] @ x[k] with automatic dequantization.
    pub inline fn gemv(self: Backend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        switch (self) {
            inline else => |be| be.gemv(x, w, y, n, k),
        }
    }

    /// Y[n_tok × n_out] = X[n_tok × n_in] @ W[n_out × n_in]^T.
    pub inline fn gemm(self: Backend, x: [*]const f32, w: TensorData, y: [*]f32, n_tok: usize, n_out: usize, n_in: usize) void {
        switch (self) {
            inline else => |be| be.gemm(x, w, y, n_tok, n_out, n_in),
        }
    }

    /// Normalize each of n_tok rows independently.
    pub inline fn rmsNormBatched(self: Backend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n_tok: usize, dim: usize, eps: f32) void {
        switch (self) {
            inline else => |be| be.rmsNormBatched(input, weight, output, n_tok, dim, eps),
        }
    }

    /// Apply RoPE to n_tok vectors at positions[0..n_tok].
    pub inline fn ropeBatched(self: Backend, x: [*]f32, positions: [*]const u32, n_tok: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        switch (self) {
            inline else => |be| be.ropeBatched(x, positions, n_tok, n_heads, head_dim, rope_dim, theta),
        }
    }

    /// Prefill attention: causal self-attention for n_tok new tokens.
    pub inline fn sdpaPrefill(self: Backend, q: [*]const f32, k: [*]const f32, v: [*]const f32, kv_keys: []u8, kv_values: []u8, output: [*]f32, nh: usize, nkv: usize, hd: usize, prev_len: usize, n_tok: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        switch (self) {
            inline else => |be| be.sdpaPrefill(q, k, v, kv_keys, kv_values, output, nh, nkv, hd, prev_len, n_tok, scale, kv_type_k, kv_type_v),
        }
    }

    /// Compute RMS normalization: output = input * weight / rms(input).
    pub inline fn rmsNorm(self: Backend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        switch (self) {
            inline else => |be| be.rmsNorm(input, weight, output, n, eps),
        }
    }

    /// Apply SiLU activation: output = input * sigmoid(input).
    pub inline fn silu(self: Backend, input: [*]const f32, output: [*]f32, n: usize) void {
        switch (self) {
            inline else => |be| be.silu(input, output, n),
        }
    }

    /// GELU activation: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x³))).
    pub inline fn gelu(self: Backend, input: [*]const f32, output: [*]f32, n: usize) void {
        switch (self) {
            inline else => |be| be.gelu(input, output, n),
        }
    }

    /// Element-wise addition: output = a + b.
    pub inline fn add(self: Backend, a: [*]const f32, b: [*]const f32, output: [*]f32, n: usize) void {
        switch (self) {
            inline else => |be| be.add(a, b, output, n),
        }
    }

    /// Transposed GEMV: y[out_dim] = W^T @ x[in_dim] for Q8_0 3D weights.
    /// W is stored as [in_dim rows, out_dim cols] in Q8_0 blocks.
    pub inline fn gemvT(self: Backend, x: [*]const f32, w: [*]const u8, y: [*]f32, out_dim: usize, in_dim: usize) void {
        switch (self) {
            inline else => |be| be.gemvT(x, w, y, out_dim, in_dim),
        }
    }

    /// Scaled accumulate: dst[i] += src[i] * scale.
    /// Used for MoE expert output accumulation to avoid per-expert GPU sync.
    pub inline fn addScaled(self: Backend, src: [*]const f32, dst: [*]f32, scale: f32, n: usize) void {
        switch (self) {
            inline else => |be| be.addScaled(src, dst, scale, n),
        }
    }

    /// Fused add + rms_norm: a[i] = a[i] + b[i], output = rms_norm(a+b, weight, eps).
    /// Replaces separate add + rmsNorm with a single dispatch.
    pub inline fn addRmsNorm(self: Backend, a: [*]f32, b: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        switch (self) {
            inline else => |be| be.addRmsNorm(a, b, weight, output, n, eps),
        }
    }

    /// Fused rmsNorm + accumulate: b[i] += rmsNorm(a, weight, eps)[i].
    /// Replaces rmsNorm(a, w, a) + add(b, a, b), saves one dispatch per post-FFN boundary.
    pub inline fn rmsNormAdd(self: Backend, a: [*]const f32, weight: [*]const f32, b: [*]f32, n: usize, eps: f32) void {
        switch (self) {
            inline else => |be| be.rmsNormAdd(a, weight, b, n, eps),
        }
    }

    /// Element-wise multiplication: output = a * b.
    pub inline fn mul(self: Backend, a: [*]const f32, b: [*]const f32, output: [*]f32, n: usize) void {
        switch (self) {
            inline else => |be| be.mul(a, b, output, n),
        }
    }

    /// Apply softmax normalization in-place.
    pub inline fn softmax(self: Backend, data: [*]f32, n: usize) void {
        switch (self) {
            inline else => |be| be.softmax(data, n),
        }
    }

    /// Apply Rotary Position Embedding (RoPE) in-place.
    pub inline fn rope(self: Backend, x: [*]f32, pos: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        switch (self) {
            inline else => |be| be.rope(x, pos, n_heads, head_dim, rope_dim, theta),
        }
    }

    /// Interleaved 3D multimodal RoPE (Qwen3.5). Equals `rope` when T=H=W.
    pub inline fn ropeMrope(self: Backend, x: [*]f32, t_pos: usize, h_pos: usize, w_pos: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        switch (self) {
            inline else => |be| be.ropeMrope(x, t_pos, h_pos, w_pos, n_heads, head_dim, rope_dim, theta),
        }
    }

    /// Look up a token embedding with automatic dequantization.
    pub inline fn embLookup(self: Backend, table: TensorData, token_id: u32, output: [*]f32, dim: usize) void {
        switch (self) {
            inline else => |be| be.embLookup(table, token_id, output, dim),
        }
    }

    /// L2 normalize a vector in-place.
    pub inline fn l2Norm(self: Backend, x: [*]f32, n: usize, eps: f32) void {
        switch (self) {
            inline else => |be| be.l2Norm(x, n, eps),
        }
    }

    /// In-place sigmoid-gated multiply: data[i] *= sigmoid(gate[i]).
    /// Used to apply attention gates on GPU without a CPU sync.
    pub inline fn sigmoidMul(self: Backend, data: [*]f32, gate: [*]const f32, n: usize) void {
        switch (self) {
            inline else => |be| be.sigmoidMul(data, gate, n),
        }
    }

    /// De-interleave paired blocks: input[n_pairs * 2 * stride] → out_a + out_b.
    /// For each pair h: out_a[h*stride .. (h+1)*stride] = input[(2*h)*stride .. (2*h+1)*stride],
    ///                  out_b[h*stride .. (h+1)*stride] = input[(2*h+1)*stride .. (2*h+2)*stride].
    pub inline fn deinterleave(self: Backend, input: [*]const f32, out_a: [*]f32, out_b: [*]f32, stride: usize, n_pairs: usize) void {
        switch (self) {
            inline else => |be| be.deinterleave(input, out_a, out_b, stride, n_pairs),
        }
    }

    /// Split concatenated Q+gate per head into separate arrays.
    /// Input: [Q0..Q_{hd-1}, G0..G_{hd-1}] × nh heads. Output: q[nh*hd], g[nh*hd].
    pub inline fn splitQGate(self: Backend, qg: [*]const f32, q_out: [*]f32, g_out: [*]f32, hd: usize, nh: usize) void {
        switch (self) {
            inline else => |be| be.splitQGate(qg, q_out, g_out, hd, nh),
        }
    }

    /// Clamped SiLU×mul: gate(-∞,10] + up[-10,10] + silu(gate)*up in one dispatch.
    pub inline fn clampedSiluMul(self: Backend, gate: [*]const f32, up: [*]const f32, out: [*]f32, n: usize) void {
        switch (self) {
            inline else => |be| be.clampedSiluMul(gate, up, out, n),
        }
    }

    /// Fused SiLU activation + multiply: out[i] = silu(a[i]) * b[i].
    /// Used in SwiGLU FFN to replace separate silu + mul dispatches.
    pub inline fn siluMul(self: Backend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        switch (self) {
            inline else => |be| be.siluMul(a, b, out, n),
        }
    }

    /// Fused GELU + multiply: out[i] = gelu(a[i]) * b[i].
    /// Replaces separate gelu + mul dispatches (2 dispatches → 1).
    pub inline fn geluMul(self: Backend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        switch (self) {
            inline else => |be| be.geluMul(a, b, out, n),
        }
    }

    /// In-place rmsNorm applied to n_heads independent heads (each head_dim
    /// elements, contiguous), sharing the same weight vector.
    /// Replaces N separate rmsNorm calls with a single batched dispatch.
    pub inline fn rmsNormMulti(self: Backend, data: [*]f32, weight: [*]const f32, n_heads: usize, head_dim: usize, eps: f32) void {
        switch (self) {
            inline else => |be| be.rmsNormMulti(data, weight, n_heads, head_dim, eps),
        }
    }

    /// Commit pending GPU work and wait for completion.
    /// Call before CPU code reads buffers written by deferred GPU ops.
    /// No-op on CPU and Vulkan (each dispatch already synchronizes).
    /// On Metal: commits the active command buffer. On CUDA/ROCm:
    /// synchronizes the context, downloads dirty activations, and marks
    /// all activation cache entries as stale (so subsequent CPU writes
    /// are re-uploaded on next GPU use).
    /// CPU-only GEMV: bypasses Metal/GPU entirely. Used for SSD-streamed expert
    /// weights that may be on evicted mmap pages (Metal can't handle page faults).
    /// Uses the thread pool for parallelism when available.
    pub inline fn cpuGemv(self: Backend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        // Flush any pending GPU work before CPU reads the input buffer
        switch (self) {
            .metal => {},
            else => {},
        }
        // Get thread pool from the active backend
        const pool: ?*@import("../thread_pool.zig").ThreadPool = switch (self) {
            .metal => |be| be.pool,
            .cpu => |be| be.pool,
            inline else => |be| if (@hasField(@TypeOf(be.*), "pool")) be.pool else null,
        };
        // Dispatch through CPU GEMV kernel with parallelism
        if (pool) |p| {
            const rb = @import("kernels/cpu/gemv.zig").gemvRowBytes(w.dtype, k);
            if (rb > 0 and n >= 32) {
                var ctx = struct {
                    x_ptr: [*]const f32,
                    w_data: [*]const u8,
                    y_ptr: [*]f32,
                    k_val: usize,
                    row_bytes: usize,
                    dt: DType,

                    fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
                        const c: *const @This() = @ptrCast(@alignCast(ctx_ptr));
                        @import("kernels/cpu/gemv.zig").gemvSeq(
                            c.x_ptr,
                            c.w_data + start * c.row_bytes,
                            c.dt,
                            c.y_ptr + start,
                            end - start,
                            c.k_val,
                        );
                    }
                }{ .x_ptr = x, .w_data = w.data, .y_ptr = y, .k_val = k, .row_bytes = rb, .dt = w.dtype };
                p.parallelFor(n, 16, @ptrCast(&ctx), @TypeOf(ctx).work);
                return;
            }
        }
        @import("kernels/cpu/gemv.zig").gemvSeq(x, w.data, w.dtype, y, n, k);
    }

    pub inline fn sync(self: Backend) void {
        switch (self) {
            inline else => |be| be.sync(),
        }
    }

    /// Enable volatile weight mode (SSD streaming). Metal flushes its buffer
    /// cache on each sync to prevent stale references to evicted mmap'd pages.
    /// No-op on non-Metal backends.
    pub inline fn setVolatileWeights(self: Backend, v: bool) void {
        switch (self) {
            .metal => |be| {
                // On non-macOS targets the metal slot is NullBackend, which has
                // neither field, gate at comptime so Linux builds compile.
                if (comptime @hasField(@TypeOf(be.*), "volatile_weights")) {
                    be.volatile_weights = v;
                    // Immediately flush any existing cached buffers that may hold
                    // stale references from a previous mmap'd model or process.
                    if (v and comptime @hasDecl(@TypeOf(be.*), "flushBufferCache")) be.flushBufferCache();
                }
            },
            else => {},
        }
    }

    /// Scaled dot-product attention with KV cache append.
    /// Appends `k_new`/`v_new` at position `seq_len` in the KV cache, then
    /// computes softmax((Q @ K^T) * scale) @ V over `seq_len + 1` positions.
    /// KV cache is stored in `kv_type_k`/`kv_type_v` format (f32, f16, q8_0, etc.).
    /// Keys/values are byte slices; backends quantize on append and dequantize on read.
    /// Each backend handles sync internally. No caller sync needed.
    pub inline fn sdpa(self: Backend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        switch (self) {
            inline else => |be| be.sdpa(q, keys, values, k_new, v_new, output, nh, nkv, hd, seq_len, scale, kv_type_k, kv_type_v),
        }
    }

    /// Paged SDPA: block-table-indexed attention for non-contiguous KV cache.
    /// Uses PagedKvView to walk block table instead of flat byte slices.
    /// Backends without native paged kernels fall back to CPU.
    pub inline fn sdpaPaged(self: Backend, q: [*]const f32, kv_view: PagedKvView, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        switch (self) {
            inline else => |be| be.sdpaPaged(q, kv_view, k_new, v_new, output, nh, nkv, hd, scale, kv_type_k, kv_type_v),
        }
    }

    /// Scaled dot-product attention with KV cache append, returning per-head
    /// softmax statistics for online softmax merge in split-attention.
    /// Same as sdpa() but additionally outputs head_max[nh] and head_sum[nh]:
    ///   - head_max[h]: max QK score before exp (for each head).
    ///   - head_sum[h]: sum of exp(scores - max), the softmax denominator.
    /// These stats enable exact merging of partial attention outputs from
    /// different devices (GPU + CPU) via online softmax correction.
    /// Note: GPU backends fill identity stats (max=0, sum=1), their SDPA
    /// already produces normalized output, so the merge formula treats it
    /// as-is. Only the CPU backend computes real per-head stats.
    pub inline fn sdpaWithStats(self: Backend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, head_max: [*]f32, head_sum: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        switch (self) {
            inline else => |be| be.sdpaWithStats(q, keys, values, k_new, v_new, output, head_max, head_sum, nh, nkv, hd, seq_len, scale, kv_type_k, kv_type_v),
        }
    }

    /// Tree-masked SDPA for DDTree speculative decoding verification.
    /// Prefix KV can be quantized (u8 byte array). Tree KV is always f32.
    pub inline fn sdpaTree(self: Backend, q_all: [*]const f32, prefix_keys: [*]const u8, prefix_values: [*]const u8, tree_keys: [*]const f32, tree_values: [*]const f32, output: [*]f32, ancestor_masks: [*]const [8]u64, nh: usize, nkv: usize, hd: usize, prefix_len: usize, n_nodes: u32, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        switch (self) {
            inline else => |be| be.sdpaTree(q_all, prefix_keys, prefix_values, tree_keys, tree_values, output, ancestor_masks, nh, nkv, hd, prefix_len, n_nodes, scale, kv_type_k, kv_type_v),
        }
    }

    /// Batched GEMV: dispatch multiple y[n] = W[n,k] @ x[k] ops sharing the same
    /// input vector x and dimension k. GPU backends may fuse into a single kernel
    /// launch to reduce dispatch overhead. CPU/fallback backends run sequentially.
    pub inline fn gemvMulti(self: Backend, x: [*]const f32, ops: []const GemvOp, k: usize) void {
        switch (self) {
            inline else => |be| be.gemvMulti(x, ops, k),
        }
    }

    /// DeltaNet SSM recurrence: conv1d + L2 norm + recurrence + gated output.
    /// On GPU backends, runs entirely on the GPU (no CPU sync needed).
    /// On CPU, runs the same computation inline.
    /// Parameters (in order): conv_in, conv_out, z_buf, alpha_buf, beta_buf,
    /// output, conv_state, ssm_state, ssm_a, dt_bias, conv_w, ssm_norm_w, p.
    pub inline fn deltaNet(self: Backend, conv_in: [*]const f32, conv_out: [*]f32, z_buf: [*]const f32, alpha_buf: [*]const f32, beta_buf: [*]const f32, output: [*]f32, conv_state: [*]f32, ssm_state: []f32, ssm_a: [*]const f32, dt_bias: [*]const f32, conv_w: [*]const f32, ssm_norm_w: [*]const f32, p: DeltaNetParams) void {
        switch (self) {
            inline else => |be| be.deltaNet(conv_in, conv_out, z_buf, alpha_buf, beta_buf, output, conv_state, ssm_state, ssm_a, dt_bias, conv_w, ssm_norm_w, p),
        }
    }

    /// Compute y[n] = W[n,k] @ x[k] for NVFP4 SafeTensors layout (separated weight + scale).
    ///
    /// NVFP4 SafeTensors stores weights as packed nibble pairs and FP8 E4M3
    /// scales in separate tensors, with group_size=16 elements per scale.
    /// GPU backends without a native kernel will @panic at runtime.
    ///
    /// Parameters:
    ///   - x: Input vector [k].
    ///   - weight: Packed nibble pairs [n * k/2] bytes.
    ///   - scale: FP8 E4M3 block scales [n * k/16] bytes.
    ///   - y: Output vector [n].
    ///   - n: Number of output rows.
    ///   - k: Number of input columns (must be divisible by 16).
    pub inline fn gemvNvfp4St(self: Backend, x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: usize, k: usize) void {
        switch (self) {
            inline else => |be| be.gemvNvfp4St(x, weight, scale, y, n, k),
        }
    }

    /// Compute y[n] = W[n,k] @ x[k] for MLX affine quantized layout.
    ///
    /// MLX quantization stores weights as packed integer nibbles (2/4/6/8-bit)
    /// in u32 words, with per-group bf16 scales and biases.
    /// Dequant: float_val = scale * int_val + bias.
    ///
    /// Parameters:
    ///   - x: Input vector [k].
    ///   - weight: Packed integer values [n * groups_per_row * words_per_group] as bytes.
    ///   - scales: BF16 per-group scales [n * groups_per_row * 2] bytes.
    ///   - biases: BF16 per-group biases [n * groups_per_row * 2] bytes.
    ///   - y: Output vector [n].
    ///   - n: Number of output rows.
    ///   - k: Number of input columns.
    ///   - bits: Quantization bit width (2, 4, 6, or 8).
    ///   - group_size: Elements per quantization group (e.g. 32 or 64).
    pub inline fn gemvMlxQ(self: Backend, x: [*]const f32, weight: [*]const u8, scales: [*]const u8, biases: [*]const u8, y: [*]f32, n: usize, k: usize, bits: u32, group_size: u32) void {
        switch (self) {
            inline else => |be| be.gemvMlxQ(x, weight, scales, biases, y, n, k, bits, group_size),
        }
    }

    /// GPU-native MLX-Q GEMV for heap-resident weight data.
    /// On Metal, dispatches to the native GPU kernel (no CPU fallback).
    /// On other backends, delegates to gemvMlxQ.
    pub inline fn gemvMlxQGpu(self: Backend, x: [*]const f32, weight: [*]const u8, scales: [*]const u8, biases: [*]const u8, y: [*]f32, n: usize, k: usize, bits: u32, group_size: u32) void {
        switch (self) {
            inline else => |be| be.gemvMlxQGpu(x, weight, scales, biases, y, n, k, bits, group_size),
        }
    }

    /// Scale format for MXFP4 GEMV (re-exported from mlx.zig for dispatcher callers).
    pub const Mxfp4ScaleFormat = @import("../ops/mlx.zig").Mxfp4ScaleFormat;

    /// Compute y[n] = W[n,k] @ x[k] for MXFP4 SafeTensors layout (MLX-style packing).
    ///
    /// MXFP4 stores weights as U32-packed 4-bit nibbles (8 per word) with
    /// per-group U8 scales. `gs` is the quantization group size
    /// (16 for standard MXFP4, 32 for MLX community expert weights).
    /// `sf` selects scale decoding: `.fp8_e4m3` (NVIDIA/GGUF) or `.e8m0` (OCP/MLX experts).
    pub inline fn gemvMxfp4St(self: Backend, x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: usize, k: usize, gs: usize, sf: Mxfp4ScaleFormat) void {
        switch (self) {
            inline else => |be| be.gemvMxfp4St(x, weight, scale, y, n, k, gs, sf),
        }
    }

    /// GPU-native MXFP4 GEMV for heap-resident data. On Metal, uses GPU kernel.
    pub inline fn gemvMxfp4StGpu(self: Backend, x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: usize, k: usize, gs: usize, sf: Mxfp4ScaleFormat) void {
        switch (self) {
            inline else => |be| be.gemvMxfp4StGpu(x, weight, scale, y, n, k, gs, sf),
        }
    }

    /// Batched MXFP4 expert GEMV: one launch for n_slots independent
    /// row-reductions (active experts' gate+up or down). Falls back to the
    /// per-slot path on backends without the batched kernel.
    pub inline fn gemvMxfp4StBatched(self: Backend, x_devs: []const u64, w_devs: []const u64, s_devs: []const u64, y_hosts: []const [*]f32, n: usize, k: usize, gs: usize, sf: Mxfp4ScaleFormat) void {
        switch (self) {
            inline else => |be| {
                if (comptime @hasDecl(@TypeOf(be.*), "gemvMxfp4StBatched")) {
                    be.gemvMxfp4StBatched(x_devs, w_devs, s_devs, y_hosts, n, k, gs, sf);
                } else {
                    unreachable;
                }
            },
        }
    }

    /// Device pointer for a weight range. No-op (0) on non-CUDA backends.
    pub inline fn getWeightDevicePtr(self: Backend, ptr: [*]const u8, size: usize) u64 {
        switch (self) {
            inline else => |be| {
                if (comptime @hasDecl(@TypeOf(be.*), "getWeightDevicePtr")) return be.getWeightDevicePtr(ptr, size);
            },
        }
        return 0;
    }

    /// Device pointer for an input activation. No-op (0) on non-CUDA backends.
    pub inline fn getInputDevicePtr(self: Backend, ptr: [*]const f32, size: usize) u64 {
        switch (self) {
            inline else => |be| {
                if (comptime @hasDecl(@TypeOf(be.*), "getInputDevicePtr")) return be.getInputDevicePtr(ptr, size);
            },
        }
        return 0;
    }

    /// GPTQ INT4 GEMV: y[n] = dequant(qweight[n,k/8]) @ x[k].
    /// Weights packed 8 nibbles per u32, FP16 per-group scales, INT4 packed zero-points.
    pub inline fn gemvGptq(self: Backend, x: [*]const f32, qweight: [*]const u32, scales: [*]const u16, qzeros: [*]const u32, y: [*]f32, n: usize, k: usize, group_size: u32) void {
        switch (self) {
            inline else => |be| be.gemvGptq(x, qweight, scales, qzeros, y, n, k, group_size),
        }
    }

    /// AWQ INT4 GEMV: y[n] = dequant(qweight[k,n/8]) @ x[k].
    /// Weights column-major packed 8 nibbles per u32, FP16 per-group scales, GEMM-order zero-points.
    pub inline fn gemvAwq(self: Backend, x: [*]const f32, qweight: [*]const u32, scales: [*]const u16, qzeros: [*]const u32, y: [*]f32, n: usize, k: usize, group_size: u32) void {
        switch (self) {
            inline else => |be| be.gemvAwq(x, qweight, scales, qzeros, y, n, k, group_size),
        }
    }

    /// HQQ 4-bit GEMV: y[n] = dequant(w_q[n,k/2], scale[n,k/group], zero[n,k/group]) @ x[k].
    /// w_q: uint8, 2 nibbles/byte. scale and zero: bf16 companion tensors.
    pub inline fn gemvHqq(self: Backend, x: [*]const f32, w_q: [*]const u8, scale: [*]const u8, zero: [*]const u8, y: [*]f32, n: usize, k: usize, group_size: u32) void {
        switch (self) {
            inline else => |be| be.gemvHqq(x, w_q, scale, zero, y, n, k, group_size),
        }
    }

    /// Begin a batch of independent GPU dispatches. While active, memory barriers
    /// between dispatches are suppressed so the GPU can overlap execution.
    /// Call endBatch() after the last independent op to insert a single barrier.
    /// No-op on CPU.
    pub inline fn beginBatch(self: Backend) void {
        switch (self) {
            inline else => |be| be.beginBatch(),
        }
    }

    /// End a batch of independent GPU dispatches and insert a memory barrier.
    /// No-op on CPU.
    pub inline fn endBatch(self: Backend) void {
        switch (self) {
            inline else => |be| be.endBatch(),
        }
    }

    /// Returns backend startup information (device name, lib, VRAM, etc.).
    pub inline fn backendInfo(self: Backend) BackendInfo {
        switch (self) {
            inline else => |be| return be.backendInfo(),
        }
    }

    /// Make GPU context current on calling thread (for multi-threaded use).
    pub inline fn setThreadContext(self: Backend) void {
        switch (self) {
            inline else => |be| {
                if (comptime @hasDecl(@TypeOf(be.*), "setThreadContext")) {
                    be.setThreadContext();
                }
            },
        }
    }

    /// Invalidate cached device copy of a host buffer.
    /// Forces re-upload on next GPU access. Used after CPU writes to activation buffers.
    pub inline fn invalidateActivation(self: Backend, ptr: [*]f32) void {
        switch (self) {
            inline else => |be| {
                if (comptime @hasDecl(@TypeOf(be.*), "invalidateAct")) {
                    be.invalidateAct(ptr);
                }
            },
        }
    }

    /// Copy a just-computed GEMV output back to host and mark it stale, so
    /// CPU code can read the result between GPU GEMVs (DS4's interleaved
    /// rmsNorm/pooling passes). No-op on backends without a device cache.
    pub inline fn syncGemvOutput(self: Backend, y: [*]f32, n: usize) void {
        switch (self) {
            inline else => |be| {
                if (comptime @hasDecl(@TypeOf(be.*), "syncGemvOutput")) {
                    be.syncGemvOutput(y, n);
                }
            },
        }
    }

    /// Make a weight range GPU-resident (prefault zero-copy regions, upload
    /// and release host pages otherwise). No-op on backends without CUDA.
    pub inline fn prefaultWeight(self: Backend, ptr: [*]const u8, len: usize) void {
        switch (self) {
            inline else => |be| {
                if (comptime @hasDecl(@TypeOf(be.*), "prefaultWeight")) {
                    be.prefaultWeight(ptr, len);
                }
            },
        }
    }

    /// Make a weight range GPU-resident (permanent device copy or zero-copy
    /// prefault). No-op on backends without CUDA.
    pub inline fn residentWeight(self: Backend, ptr: [*]const u8, len: usize) void {
        switch (self) {
            inline else => |be| {
                if (comptime @hasDecl(@TypeOf(be.*), "residentWeight")) {
                    be.residentWeight(ptr, len);
                }
            },
        }
    }

    /// Restore RANDOM madvise advice on tracked mmap ranges after the
    /// resident copy (which uses SEQUENTIAL readahead). No-op elsewhere.
    pub inline fn restoreMmapHints(self: Backend) void {
        switch (self) {
            inline else => |be| {
                if (comptime @hasDecl(@TypeOf(be.*), "restoreMmapHints")) {
                    be.restoreMmapHints();
                }
            },
        }
    }

    /// Get the CUDA device pointer for a host activation buffer.
    /// Returns 0 if not available or not a CUDA backend.
    pub inline fn getDevicePtr(self: Backend, ptr: anytype) u64 {
        return switch (self) {
            inline else => |be| {
                if (comptime @hasDecl(@TypeOf(be.*), "getDevicePtr"))
                    return be.getDevicePtr(ptr);
                return 0;
            },
        };
    }

    /// Evict a weight buffer from GPU cache so next access re-uploads from host.
    /// Used when the same host buffer (e.g. tp_row_shard_buf) is reused with
    /// different data between TP rank switches.
    pub inline fn invalidateWeight(self: Backend, ptr: anytype) void {
        switch (self) {
            inline else => |be| {
                if (comptime @hasDecl(@TypeOf(be.*), "invalidateWeight")) {
                    be.invalidateWeight(ptr);
                }
            },
        }
    }

    /// All-reduce sum for tensor parallelism: dst[i] += src[i].
    pub inline fn allReduceAdd(self: Backend, dst: [*]f32, src: [*]const f32, n: usize) void {
        switch (self) {
            inline else => |be| {
                if (comptime @hasDecl(@TypeOf(be.*), "allReduceAdd")) {
                    be.allReduceAdd(dst, src, n);
                }
            },
        }
    }

    /// Register a host memory region for UMA zero-copy GPU access.
    /// Call once per mmap'd file (GGUF, SafeTensors) after loading.
    /// No-op on backends that don't support UMA host registration.
    pub inline fn registerHostRegion(self: Backend, base: [*]const u8, size: usize) void {
        switch (self) {
            inline else => |be| {
                if (comptime @hasDecl(@TypeOf(be.*), "registerHostRegion")) {
                    be.registerHostRegion(base, size);
                }
            },
        }
    }
};

/// Backend selection from CLI or micro-bench argument.
pub const BackendChoice = enum { auto, cpu, metal, vulkan, cuda, rocm, webgpu };

/// Holds the mutable backend storage alongside the tagged-union interface.
/// The backend variables must outlive the `Backend` union (which holds pointers
/// to them), so they are bundled together and kept on the caller's stack.
/// Process exit releases most resources; the thread pool is explicitly
/// cleaned up via defer in the caller.
pub const BackendState = struct {
    cpu_be: CpuBackend = .{},
    metal_be: if (builtin.os.tag == .macos) MetalBackend else void = undefined,
    vulkan_be: VulkanBackend = undefined,
    cuda_be: CudaBackend = undefined,
    rocm_be: RocmBackend = undefined,
    webgpu_be: WebGpuBackend = undefined,
    pool: ?ThreadPool = null,
    be: Backend = undefined,
    name: []const u8 = "CPU",

    /// Threads that actually run parallel CPU work: the pool's workers plus the
    /// main thread, which participates in every `parallelFor`. Reported in the
    /// banner, so it must be the real width, not the logical CPU count (the pool
    /// is sized by PHYSICAL cores; see `init`).
    pub fn computeThreads(self: *const BackendState) u32 {
        const workers = if (self.pool) |*p| p.n_workers else 0;
        return @intCast(workers + 1);
    }

    /// Initialize the requested compute backend, with automatic fallback.
    /// Must be called on a stack-allocated `BackendState`, the `be` field
    /// stores pointers into the struct's own backend fields.
    ///
    /// Parameters:
    ///   - allocator: Used for backend-internal allocations (pipeline caches, etc.).
    ///   - backend_choice: Which backend to initialize (or .auto for auto-detection).
    pub fn init(self: *BackendState, allocator: std.mem.Allocator, backend_choice: BackendChoice, io: std.Io, device_id: u32) void {
        // One thread per PHYSICAL core, minus one for the main thread (which
        // also participates). SMT siblings add no memory bandwidth to a
        // bandwidth-bound GEMV and make the spin-wait in parallelFor worse, so
        // sizing by logical CPUs oversubscribes every core on an SMT machine.
        //
        // Workers are pinned to the core they were sized for. Core id 0 is left
        // to the unpinned main thread, so a worker never shares a core with it.
        var core_ids: [max_pinned_core_ids]u32 = undefined;
        const n_cores_listed = physicalCoreIds(&core_ids);
        const cpu_count = if (n_cores_listed > 0) n_cores_listed else detectPhysicalCores();
        const n_workers = if (cpu_count > 1) cpu_count - 1 else 0;
        self.pool = ThreadPool.init(n_workers);
        if (n_cores_listed > 1) self.pool.?.setAffinity(core_ids[1..n_cores_listed]);
        self.pool.?.spawn(io);
        self.cpu_be.pool = &self.pool.?;
        self.be = blk: {
            switch (backend_choice) {
                .cpu => {
                    if (comptime build_options.enable_cpu) {
                        break :blk .{ .cpu = &self.cpu_be };
                    } else {
                        @panic("CPU backend disabled at build time");
                    }
                },
                .metal => {
                    if (comptime build_options.enable_metal and builtin.os.tag == .macos) {
                        self.metal_be = MetalBackend.init(allocator) catch |err| {
                            if (comptime build_options.enable_cpu) {
                                std.log.warn("Metal unavailable ({s}), falling back to CPU", .{@errorName(err)});
                                break :blk .{ .cpu = &self.cpu_be };
                            } else {
                                @panic("Metal unavailable and CPU backend disabled");
                            }
                        };
                        self.metal_be.pool = &self.pool.?;
                        self.name = "Metal";
                        break :blk .{ .metal = &self.metal_be };
                    } else {
                        if (comptime build_options.enable_cpu) {
                            std.log.warn("Metal not available on this platform", .{});
                            break :blk .{ .cpu = &self.cpu_be };
                        } else {
                            @panic("Metal not available and CPU backend disabled");
                        }
                    }
                },
                .vulkan => {
                    if (comptime build_options.enable_vulkan) {
                        self.vulkan_be = VulkanBackend.init(allocator, device_id) catch |err| {
                            if (comptime build_options.enable_cpu) {
                                std.log.warn("Vulkan unavailable ({s}), falling back to CPU", .{@errorName(err)});
                                break :blk .{ .cpu = &self.cpu_be };
                            } else {
                                @panic("Vulkan unavailable and CPU backend disabled");
                            }
                        };
                        self.name = "Vulkan";
                        break :blk .{ .vulkan = &self.vulkan_be };
                    } else {
                        @panic("Vulkan backend disabled at build time");
                    }
                },
                .cuda => {
                    if (comptime build_options.enable_cuda) {
                        self.cuda_be = CudaBackend.init(allocator, device_id) catch |err| {
                            if (comptime build_options.enable_cpu) {
                                std.log.warn("CUDA unavailable ({s}), falling back to CPU", .{@errorName(err)});
                                break :blk .{ .cpu = &self.cpu_be };
                            } else {
                                @panic("CUDA unavailable and CPU backend disabled");
                            }
                        };
                        self.name = "CUDA";
                        self.cuda_be.cpu.pool = &self.pool.?;
                        break :blk .{ .cuda = &self.cuda_be };
                    } else {
                        @panic("CUDA backend disabled at build time");
                    }
                },
                .rocm => {
                    if (comptime build_options.enable_rocm) {
                        self.rocm_be = RocmBackend.init(allocator, device_id) catch |err| {
                            if (comptime build_options.enable_cpu) {
                                std.log.warn("ROCm unavailable ({s}), falling back to CPU", .{@errorName(err)});
                                break :blk .{ .cpu = &self.cpu_be };
                            } else {
                                @panic("ROCm unavailable and CPU backend disabled");
                            }
                        };
                        self.name = "ROCm";
                        self.rocm_be.cpu.pool = &self.pool.?;
                        break :blk .{ .rocm = &self.rocm_be };
                    } else {
                        @panic("ROCm backend disabled at build time");
                    }
                },
                .webgpu => {
                    if (comptime build_options.enable_webgpu) {
                        self.webgpu_be = WebGpuBackend.init(allocator) catch |err| {
                            if (comptime build_options.enable_cpu) {
                                std.log.warn("WebGPU unavailable ({s}), falling back to CPU", .{@errorName(err)});
                                break :blk .{ .cpu = &self.cpu_be };
                            } else {
                                @panic("WebGPU unavailable and CPU backend disabled");
                            }
                        };
                        self.name = "WebGPU";
                        break :blk .{ .webgpu = &self.webgpu_be };
                    } else {
                        @panic("WebGPU backend disabled at build time");
                    }
                },
                .auto => {
                    // Try Metal (macOS only)
                    if (comptime build_options.enable_metal and builtin.os.tag == .macos) {
                        self.metal_be = MetalBackend.init(allocator) catch {
                            if (comptime build_options.enable_cuda) {
                                self.cuda_be = CudaBackend.init(allocator, device_id) catch {
                                    if (comptime build_options.enable_vulkan) {
                                        self.vulkan_be = VulkanBackend.init(allocator, device_id) catch {
                                            if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else @panic("All GPU backends failed and CPU disabled");
                                        };
                                        self.name = "Vulkan";
                                        break :blk .{ .vulkan = &self.vulkan_be };
                                    }
                                    if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else @panic("CUDA failed, no other backends enabled");
                                };
                                self.name = "CUDA";
                                self.cuda_be.cpu.pool = &self.pool.?;
                                break :blk .{ .cuda = &self.cuda_be };
                            } else if (comptime build_options.enable_vulkan) {
                                self.vulkan_be = VulkanBackend.init(allocator, device_id) catch {
                                    if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else @panic("Vulkan failed and CPU disabled");
                                };
                                self.name = "Vulkan";
                                break :blk .{ .vulkan = &self.vulkan_be };
                            } else if (comptime build_options.enable_cpu) {
                                break :blk .{ .cpu = &self.cpu_be };
                            } else {
                                @panic("No backends enabled");
                            }
                        };
                        self.metal_be.pool = &self.pool.?;
                        self.name = "Metal";
                        break :blk .{ .metal = &self.metal_be };
                    }
                    // Try CUDA
                    if (comptime build_options.enable_cuda) {
                        self.cuda_be = CudaBackend.init(allocator, device_id) catch {
                            if (comptime build_options.enable_rocm) {
                                self.rocm_be = RocmBackend.init(allocator, device_id) catch {
                                    if (comptime build_options.enable_vulkan) {
                                        self.vulkan_be = VulkanBackend.init(allocator, device_id) catch {
                                            if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else @panic("All GPU backends failed and CPU disabled");
                                        };
                                        self.name = "Vulkan";
                                        break :blk .{ .vulkan = &self.vulkan_be };
                                    }
                                    if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else @panic("All GPU backends failed and CPU disabled");
                                };
                                self.name = "ROCm";
                                self.rocm_be.cpu.pool = &self.pool.?;
                                break :blk .{ .rocm = &self.rocm_be };
                            }
                            if (comptime build_options.enable_vulkan) {
                                self.vulkan_be = VulkanBackend.init(allocator, device_id) catch {
                                    if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else @panic("All GPU backends failed and CPU disabled");
                                };
                                self.name = "Vulkan";
                                break :blk .{ .vulkan = &self.vulkan_be };
                            }
                            if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else @panic("CUDA failed, no other backends enabled");
                        };
                        self.name = "CUDA";
                        self.cuda_be.cpu.pool = &self.pool.?;
                        break :blk .{ .cuda = &self.cuda_be };
                    }
                    // Try ROCm
                    if (comptime build_options.enable_rocm) {
                        self.rocm_be = RocmBackend.init(allocator, device_id) catch {
                            if (comptime build_options.enable_vulkan) {
                                self.vulkan_be = VulkanBackend.init(allocator, device_id) catch {
                                    if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else @panic("All GPU backends failed and CPU disabled");
                                };
                                self.name = "Vulkan";
                                break :blk .{ .vulkan = &self.vulkan_be };
                            }
                            if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else @panic("ROCm failed, no other backends enabled");
                        };
                        self.name = "ROCm";
                        self.rocm_be.cpu.pool = &self.pool.?;
                        break :blk .{ .rocm = &self.rocm_be };
                    }
                    // Try Vulkan
                    if (comptime build_options.enable_vulkan) {
                        self.vulkan_be = VulkanBackend.init(allocator, device_id) catch {
                            if (comptime build_options.enable_cpu) break :blk .{ .cpu = &self.cpu_be } else @panic("Vulkan failed and CPU disabled");
                        };
                        self.name = "Vulkan";
                        break :blk .{ .vulkan = &self.vulkan_be };
                    }
                    // CPU fallback
                    if (comptime build_options.enable_cpu) {
                        break :blk .{ .cpu = &self.cpu_be };
                    }
                    @panic("No backends enabled");
                },
            }
        };
    }
};

// ── Tests ───────────────────────────────────────────────────────────

test "pageAlignRange, covers the whole request and starts on a page" {
    const page = std.heap.page_size_min;
    var buf: [1]u8 align(std.heap.page_size_min) = undefined;
    const base = @intFromPtr(&buf);

    // Aligned start, exact page length: no widening.
    const exact = pageAlignRange(@ptrCast(&buf), page);
    try std.testing.expectEqual(base, exact.base);
    try std.testing.expectEqual(page, exact.size);

    // One byte past the page start still covers a full page.
    const one = pageAlignRange(@ptrCast(&buf), 1);
    try std.testing.expectEqual(base, one.base);
    try std.testing.expectEqual(page, one.size);

    // An unaligned pointer rounds the base DOWN, so the range still contains
    // every byte the caller asked for; hostUnregister depends on this matching.
    const off: [*]const u8 = @as([*]const u8, @ptrCast(&buf)) + 8;
    const un = pageAlignRange(off, page);
    try std.testing.expectEqual(base, un.base);
    try std.testing.expectEqual(2 * page, un.size);
    try std.testing.expect(un.base + un.size >= @intFromPtr(off) + page);
}

test "pageAlignRange, zero length collapses to the containing page start" {
    var buf: [1]u8 align(std.heap.page_size_min) = undefined;
    const r = pageAlignRange(@ptrCast(&buf), 0);
    try std.testing.expectEqual(@intFromPtr(&buf), r.base);
    try std.testing.expectEqual(@as(usize, 0), r.size);
}

test "hostPrefault, tolerates zero length and unmapped-tail ranges" {
    var buf: [4096]u8 align(std.heap.page_size_min) = undefined;
    hostPrefault(@ptrCast(&buf), 0);
    hostPrefault(@ptrCast(&buf), buf.len);
}

test "DType, gemvRowBytes for F32" {
    // F32: 4 bytes per element, no quantization blocks.
    try std.testing.expectEqual(@as(usize, 4096 * 4), gemvRowBytes(.f32, 4096));
    try std.testing.expectEqual(@as(usize, 1 * 4), gemvRowBytes(.f32, 1));
}

test "DType, gemvRowBytes for F16 and BF16" {
    // F16/BF16: 2 bytes per element.
    try std.testing.expectEqual(@as(usize, 4096 * 2), gemvRowBytes(.f16, 4096));
    try std.testing.expectEqual(@as(usize, 4096 * 2), gemvRowBytes(.bf16, 4096));
}

test "DType, gemvRowBytes for FP8" {
    // FP8: 1 byte per element.
    try std.testing.expectEqual(@as(usize, 4096), gemvRowBytes(.fp8_e4m3, 4096));
    try std.testing.expectEqual(@as(usize, 4096), gemvRowBytes(.fp8_e5m2, 4096));
}

test "DType, gemvRowBytes for Q4_0" {
    // Q4_0: 18 bytes per 32-element block.
    // k=4096 → 128 blocks → 128 * 18 = 2304 bytes per row.
    try std.testing.expectEqual(@as(usize, 128 * q4_0_block_bytes), gemvRowBytes(.q4_0, 4096));
    // k=32 → 1 block → 18 bytes.
    try std.testing.expectEqual(@as(usize, q4_0_block_bytes), gemvRowBytes(.q4_0, 32));
    // k=33 → ceil(33/32) = 2 blocks → 36 bytes.
    try std.testing.expectEqual(@as(usize, 2 * q4_0_block_bytes), gemvRowBytes(.q4_0, 33));
}

test "DType, gemvRowBytes for Q8_0" {
    // Q8_0: 34 bytes per 32-element block.
    try std.testing.expectEqual(@as(usize, 128 * q8_0_block_bytes), gemvRowBytes(.q8_0, 4096));
    try std.testing.expectEqual(@as(usize, q8_0_block_bytes), gemvRowBytes(.q8_0, 32));
}

test "DType, gemvRowBytes for Q4_K super-block" {
    // Q4_K: 144 bytes per 256-element super-block.
    // k=4096 → 16 super-blocks → 16 * 144 = 2304.
    try std.testing.expectEqual(@as(usize, 16 * q4_k_block_bytes), gemvRowBytes(.q4_k, 4096));
    // k=256 → 1 super-block.
    try std.testing.expectEqual(@as(usize, q4_k_block_bytes), gemvRowBytes(.q4_k, 256));
}

test "DType, gemvRowBytes for Q6_K super-block" {
    // Q6_K: 210 bytes per 256-element super-block.
    try std.testing.expectEqual(@as(usize, 16 * q6_k_block_bytes), gemvRowBytes(.q6_k, 4096));
}

test "DType, weightBytes and gemvRowBytes for HQQ" {
    // HQQ 4-bit: 2 nibbles per byte → n*k/2 bytes total.
    try std.testing.expectEqual(@as(usize, 1 * 4096 / 2), weightBytes(.hqq, 1, 4096));
    try std.testing.expectEqual(@as(usize, 8 * 4096 / 2), weightBytes(.hqq, 8, 4096));
    // gemvRowBytes returns 0 (companion tensors, handled at model level).
    try std.testing.expectEqual(@as(usize, 0), gemvRowBytes(.hqq, 4096));
}

test "DType, gemvRowBytes for NVFP4" {
    // NVFP4: 9 bytes per 16-element block.
    // k=4096 → 256 blocks → 256 * 9 = 2304.
    try std.testing.expectEqual(@as(usize, 256 * nvfp4_block_bytes), gemvRowBytes(.nvfp4, 4096));
}

test "DType, gemvRowBytes returns 0 for unsupported formats" {
    // Formats that don't support standard row-based GEMV return 0.
    try std.testing.expectEqual(@as(usize, 0), gemvRowBytes(.tq1_0, 4096));
    try std.testing.expectEqual(@as(usize, 0), gemvRowBytes(.tq2_0, 4096));
    try std.testing.expectEqual(@as(usize, 0), gemvRowBytes(.mlx_q, 4096));
    try std.testing.expectEqual(@as(usize, 0), gemvRowBytes(.gptq, 4096));
    try std.testing.expectEqual(@as(usize, 0), gemvRowBytes(.awq, 4096));
    try std.testing.expectEqual(@as(usize, 0), gemvRowBytes(.unknown, 4096));
}

test "DType, gemvRowBytes all dtypes handled" {
    // Ensure gemvRowBytes doesn't panic for any DType variant.
    inline for (comptime std.enums.values(DType)) |dtype| {
        _ = gemvRowBytes(dtype, 256);
    }
}

test "weightBytes, F32" {
    // F32: n * k * 4 bytes.
    try std.testing.expectEqual(@as(usize, 4 * 4096 * 4096), weightBytes(.f32, 4096, 4096));
    try std.testing.expectEqual(@as(usize, 4 * 10 * 20), weightBytes(.f32, 10, 20));
}

test "weightBytes, F16 and BF16" {
    // F16/BF16: n * k * 2 bytes.
    try std.testing.expectEqual(@as(usize, 2 * 4096 * 4096), weightBytes(.f16, 4096, 4096));
    try std.testing.expectEqual(@as(usize, 2 * 4096 * 4096), weightBytes(.bf16, 4096, 4096));
}

test "weightBytes, Q4_0" {
    // Q4_0: n * blocks * 18 bytes per block.
    // k=4096 → 128 blocks, n=4096.
    try std.testing.expectEqual(@as(usize, 4096 * 128 * q4_0_block_bytes), weightBytes(.q4_0, 4096, 4096));
}

test "weightBytes, Q4_K super-block" {
    // Q4_K: n * super-blocks * 144 bytes.
    // k=4096 → 16 super-blocks, n=4096.
    try std.testing.expectEqual(@as(usize, 4096 * 16 * q4_k_block_bytes), weightBytes(.q4_k, 4096, 4096));
}

test "weightBytes, GPTQ/AWQ" {
    // GPTQ/AWQ: n * k / 2 (4-bit packed).
    try std.testing.expectEqual(@as(usize, 4096 * 4096 / 2), weightBytes(.gptq, 4096, 4096));
    try std.testing.expectEqual(@as(usize, 4096 * 4096 / 2), weightBytes(.awq, 4096, 4096));
}

test "weightBytes, NVFP4" {
    // NVFP4: n * ceil(k/16) * 9.
    try std.testing.expectEqual(@as(usize, 4096 * 256 * nvfp4_block_bytes), weightBytes(.nvfp4, 4096, 4096));
}

test "weightBytes, all dtypes handled" {
    // Ensure weightBytes doesn't panic for any DType variant and stays bounded.
    inline for (comptime std.enums.values(DType)) |dtype| {
        const bytes = weightBytes(dtype, 256, 256);
        // n*k f32 is an upper bound for dense/quantized layouts at these dims.
        try std.testing.expect(bytes <= 256 * 256 * @sizeOf(f32));
    }
}

test "weightBytes, consistency with gemvRowBytes" {
    // For supported dtypes, weightBytes(dtype, n, k) == n * gemvRowBytes(dtype, k).
    const dtypes_to_check = [_]DType{ .f32, .f16, .bf16, .fp8_e4m3, .fp8_e5m2, .q4_0, .q4_1, .q5_0, .q8_0, .q4_k, .q5_k, .q6_k, .q2_k, .q3_k, .iq4_nl, .iq4_xs, .mxfp4, .nvfp4 };
    for (dtypes_to_check) |dtype| {
        const rb = gemvRowBytes(dtype, 4096);
        if (rb > 0) {
            try std.testing.expectEqual(256 * rb, weightBytes(dtype, 256, 4096));
        }
    }
}

test "BackendInfo, default values" {
    const info = BackendInfo{};
    try std.testing.expectEqualStrings("CPU", info.name);
    try std.testing.expectEqualStrings("", info.device_name);
    try std.testing.expectEqualStrings("", info.lib_name);
    try std.testing.expectEqual(@as(u32, 0), info.n_gpu_kernels);
    try std.testing.expectEqual(@as(usize, 0), info.total_mem);
    try std.testing.expectEqual(@as(usize, 0), info.avail_mem);
    try std.testing.expect(!info.is_uma);
    try std.testing.expectEqualStrings("", info.compute_cap);
    try std.testing.expectEqualStrings("", info.driver_version);
    try std.testing.expectEqual(@as(u32, 0), info.n_threads);
    try std.testing.expectEqual(@as(usize, 0), info.system_mem);
    try std.testing.expectEqualStrings(@tagName(builtin.cpu.arch), info.arch);
    try std.testing.expectEqualStrings(@tagName(builtin.os.tag), info.os);
}

test "BackendInfo, custom values" {
    const info = BackendInfo{
        .name = "Metal",
        .device_name = "Apple M4 Pro",
        .n_gpu_kernels = 42,
        .kernel_type = "MSL",
        .total_mem = 36 * 1024 * 1024 * 1024,
        .is_uma = true,
        .n_threads = 12,
    };
    try std.testing.expectEqualStrings("Metal", info.name);
    try std.testing.expectEqualStrings("Apple M4 Pro", info.device_name);
    try std.testing.expectEqual(@as(u32, 42), info.n_gpu_kernels);
    try std.testing.expectEqualStrings("MSL", info.kernel_type);
    try std.testing.expect(info.is_uma);
    try std.testing.expectEqual(@as(u32, 12), info.n_threads);
}

test "CacheSizes, default zeros" {
    const cs = CacheSizes{};
    try std.testing.expectEqual(@as(usize, 0), cs.l1);
    try std.testing.expectEqual(@as(usize, 0), cs.l2);
    try std.testing.expectEqual(@as(usize, 0), cs.l3);
}

test "TensorData, construction" {
    var data = [_]u8{ 0, 1, 2, 3 };
    const td = TensorData{ .data = &data, .dtype = .f32 };
    try std.testing.expectEqual(DType.f32, td.dtype);
    try std.testing.expectEqual(@as(u8, 0), td.data[0]);
}

test "GemvOp, default optional fields" {
    var y_buf: [4]f32 = undefined;
    var w_data = [_]u8{ 0, 0, 0, 0 };
    const op = GemvOp{
        .w = .{ .data = &w_data, .dtype = .q4_0 },
        .y = &y_buf,
        .n = 4,
    };
    try std.testing.expectEqual(@as(?[*]const u8, null), op.mlx_scales);
    try std.testing.expectEqual(@as(?[*]const u8, null), op.mlx_biases);
    try std.testing.expectEqual(@as(u32, 0), op.mlx_bits);
}

test "DeltaNetParams, default kqv_order" {
    const p = DeltaNetParams{
        .conv_ch = 1024,
        .d_conv = 4,
        .d_inner = 2048,
        .num_k_heads = 8,
        .head_k_dim = 128,
        .num_v_heads = 8,
        .head_v_dim = 128,
        .q_scale = 1.0,
        .rms_eps = 1e-6,
    };
    try std.testing.expect(!p.kqv_order);
    try std.testing.expectEqual(@as(u32, 1024), p.conv_ch);
}

test "quant block constants" {
    // Verify block element counts are powers-of-two aligned.
    try std.testing.expectEqual(@as(usize, 32), quant_block_elems);
    try std.testing.expectEqual(@as(usize, 256), quant_super_block_elems);
    try std.testing.expectEqual(@as(usize, 16), nvfp4_block_elems);

    // Verify block byte sizes match expected values from GGML spec.
    try std.testing.expectEqual(@as(usize, 18), q4_0_block_bytes); // f16 scale + 16B quants
    try std.testing.expectEqual(@as(usize, 20), q4_1_block_bytes); // f16 scale + f16 min + 16B
    try std.testing.expectEqual(@as(usize, 34), q8_0_block_bytes); // f16 scale + 32B quants
    try std.testing.expectEqual(@as(usize, 144), q4_k_block_bytes); // 256-elem super-block
    try std.testing.expectEqual(@as(usize, 210), q6_k_block_bytes); // 256-elem super-block
    try std.testing.expectEqual(@as(usize, 9), nvfp4_block_bytes); // 8B quants + 1B scale
    try std.testing.expectEqual(@as(usize, 17), mxfp4_block_bytes); // 16B quants + 1B scale
}

test "BackendChoice, all variants exist" {
    // Verify the enum has all expected variants.
    const choices = [_]BackendChoice{ .auto, .cpu, .metal, .vulkan, .cuda, .rocm, .webgpu };
    try std.testing.expectEqual(@as(usize, 7), choices.len);
}

test "NullBackend, function signatures exist" {
    // Compile-time check that NullBackend has all required method signatures.
    comptime {
        _ = @TypeOf(NullBackend.gemv);
        _ = @TypeOf(NullBackend.rmsNorm);
        _ = @TypeOf(NullBackend.silu);
        _ = @TypeOf(NullBackend.gelu);
        _ = @TypeOf(NullBackend.add);
        _ = @TypeOf(NullBackend.mul);
        _ = @TypeOf(NullBackend.softmax);
        _ = @TypeOf(NullBackend.rope);
        _ = @TypeOf(NullBackend.sync);
        _ = @TypeOf(NullBackend.sdpa);
        _ = @TypeOf(NullBackend.sdpaPaged);
        _ = @TypeOf(NullBackend.gemvMulti);
        _ = @TypeOf(NullBackend.gemm);
        _ = @TypeOf(NullBackend.deltaNet);
        _ = @TypeOf(NullBackend.backendInfo);
        _ = @TypeOf(NullBackend.embLookup);
        _ = @TypeOf(NullBackend.gemvGptq);
        _ = @TypeOf(NullBackend.gemvAwq);
        _ = @TypeOf(NullBackend.gemvHqq);
        _ = @TypeOf(NullBackend.gemvNvfp4St);
        _ = @TypeOf(NullBackend.gemvMlxQ);
        _ = @TypeOf(NullBackend.gemvMxfp4St);
    }
}

test "Backend union, size is reasonable" {
    // Backend is a tagged union of pointers, should be pointer-sized + tag.
    try std.testing.expect(@sizeOf(Backend) <= 16);
}

test "BackendState, default name is CPU" {
    const state = BackendState{};
    try std.testing.expectEqualStrings("CPU", state.name);
}

test "buf_cache_initial_capacity constant" {
    try std.testing.expectEqual(@as(usize, 512), buf_cache_initial_capacity);
}

test "DType, gemvRowBytes for remaining small-block dtypes" {
    // Q4_1: 20 bytes per 32-element block.
    try std.testing.expectEqual(@as(usize, 128 * q4_1_block_bytes), gemvRowBytes(.q4_1, 4096));
    try std.testing.expectEqual(@as(usize, q4_1_block_bytes), gemvRowBytes(.q4_1, 32));
    // Q5_0: 22 bytes per 32-element block.
    try std.testing.expectEqual(@as(usize, 128 * q5_0_block_bytes), gemvRowBytes(.q5_0, 4096));
    // IQ4_NL: 18 bytes per 32-element block (same as Q4_0).
    try std.testing.expectEqual(@as(usize, 128 * iq4_nl_block_bytes), gemvRowBytes(.iq4_nl, 4096));
    // MXFP4: 17 bytes per 32-element block.
    try std.testing.expectEqual(@as(usize, 128 * mxfp4_block_bytes), gemvRowBytes(.mxfp4, 4096));
}

test "DType, gemvRowBytes for remaining super-block dtypes" {
    // Q2_K: 84 bytes per 256-element super-block.
    try std.testing.expectEqual(@as(usize, 16 * q2_k_block_bytes), gemvRowBytes(.q2_k, 4096));
    try std.testing.expectEqual(@as(usize, q2_k_block_bytes), gemvRowBytes(.q2_k, 256));
    // Q3_K: 110 bytes per 256-element super-block.
    try std.testing.expectEqual(@as(usize, 16 * q3_k_block_bytes), gemvRowBytes(.q3_k, 4096));
    // Q5_K: 176 bytes per 256-element super-block.
    try std.testing.expectEqual(@as(usize, 16 * q5_k_block_bytes), gemvRowBytes(.q5_k, 4096));
    // IQ4_XS: 136 bytes per 256-element super-block.
    try std.testing.expectEqual(@as(usize, 16 * iq4_xs_block_bytes), gemvRowBytes(.iq4_xs, 4096));
}

test "DType, gemvRowBytes and weightBytes edge case k=0" {
    // k=0 should produce 0 bytes for all dtypes.
    inline for (comptime std.enums.values(DType)) |dtype| {
        try std.testing.expectEqual(@as(usize, 0), gemvRowBytes(dtype, 0));
        try std.testing.expectEqual(@as(usize, 0), weightBytes(dtype, 1, 0));
        try std.testing.expectEqual(@as(usize, 0), weightBytes(dtype, 0, 256));
    }
}

test "DType, gemvRowBytes non-aligned k rounds up" {
    // Q4_0 with k=33: ceil(33/32) = 2 blocks.
    try std.testing.expectEqual(@as(usize, 2 * q4_0_block_bytes), gemvRowBytes(.q4_0, 33));
    // Q4_K with k=257: ceil(257/256) = 2 super-blocks.
    try std.testing.expectEqual(@as(usize, 2 * q4_k_block_bytes), gemvRowBytes(.q4_k, 257));
    // NVFP4 with k=17: ceil(17/16) = 2 blocks.
    try std.testing.expectEqual(@as(usize, 2 * nvfp4_block_bytes), gemvRowBytes(.nvfp4, 17));
    // Q8_0 with k=31: ceil(31/32) = 1 block.
    try std.testing.expectEqual(@as(usize, 1 * q8_0_block_bytes), gemvRowBytes(.q8_0, 31));
    // Q6_K with k=255: ceil(255/256) = 1 super-block.
    try std.testing.expectEqual(@as(usize, 1 * q6_k_block_bytes), gemvRowBytes(.q6_k, 255));
}

test "weightBytes, remaining dtypes coverage" {
    const k = 4096;
    const n = 128;
    const nb = k / quant_block_elems; // 128
    const nsb = k / quant_super_block_elems; // 16
    // Q8_0
    try std.testing.expectEqual(@as(usize, n * nb * q8_0_block_bytes), weightBytes(.q8_0, n, k));
    // Q4_1
    try std.testing.expectEqual(@as(usize, n * nb * q4_1_block_bytes), weightBytes(.q4_1, n, k));
    // Q5_0
    try std.testing.expectEqual(@as(usize, n * nb * q5_0_block_bytes), weightBytes(.q5_0, n, k));
    // Q2_K
    try std.testing.expectEqual(@as(usize, n * nsb * q2_k_block_bytes), weightBytes(.q2_k, n, k));
    // Q3_K
    try std.testing.expectEqual(@as(usize, n * nsb * q3_k_block_bytes), weightBytes(.q3_k, n, k));
    // Q5_K
    try std.testing.expectEqual(@as(usize, n * nsb * q5_k_block_bytes), weightBytes(.q5_k, n, k));
    // Q6_K
    try std.testing.expectEqual(@as(usize, n * nsb * q6_k_block_bytes), weightBytes(.q6_k, n, k));
    // IQ4_NL
    try std.testing.expectEqual(@as(usize, n * nb * iq4_nl_block_bytes), weightBytes(.iq4_nl, n, k));
    // IQ4_XS
    try std.testing.expectEqual(@as(usize, n * nsb * iq4_xs_block_bytes), weightBytes(.iq4_xs, n, k));
    // MXFP4
    try std.testing.expectEqual(@as(usize, n * nb * mxfp4_block_bytes), weightBytes(.mxfp4, n, k));
    // TQ1_0 / TQ2_0
    try std.testing.expectEqual(@as(usize, n * nsb * tq1_0_block_bytes), weightBytes(.tq1_0, n, k));
    try std.testing.expectEqual(@as(usize, n * nsb * tq2_0_block_bytes), weightBytes(.tq2_0, n, k));
    // FP8 variants: 1 byte per element.
    try std.testing.expectEqual(@as(usize, n * k), weightBytes(.fp8_e4m3, n, k));
    try std.testing.expectEqual(@as(usize, n * k), weightBytes(.fp8_e5m2, n, k));
    // MLX_Q / unknown: 4 bytes per element (assume f32).
    try std.testing.expectEqual(@as(usize, n * k * 4), weightBytes(.mlx_q, n, k));
    try std.testing.expectEqual(@as(usize, n * k * 4), weightBytes(.unknown, n, k));
}

test "weightBytes, NVFP4 non-aligned k" {
    // k=17 → ceil(17/16) = 2 blocks of 9 bytes each.
    try std.testing.expectEqual(@as(usize, 10 * 2 * nvfp4_block_bytes), weightBytes(.nvfp4, 10, 17));
}

test "BackendInfo, cache size fields" {
    const info = BackendInfo{
        .l1_cache = 64 * 1024,
        .l2_cache = 512 * 1024,
        .l3_cache = 16 * 1024 * 1024,
    };
    try std.testing.expectEqual(@as(usize, 64 * 1024), info.l1_cache);
    try std.testing.expectEqual(@as(usize, 512 * 1024), info.l2_cache);
    try std.testing.expectEqual(@as(usize, 16 * 1024 * 1024), info.l3_cache);
}

test "Backend union, all variant tags exist" {
    // Compile-time verification that Backend enum has all expected tags.
    const Tag = std.meta.Tag(Backend);
    comptime {
        _ = @field(Tag, "cpu");
        _ = @field(Tag, "metal");
        _ = @field(Tag, "vulkan");
        _ = @field(Tag, "cuda");
        _ = @field(Tag, "rocm");
        _ = @field(Tag, "webgpu");
    }
    // Exactly 6 backends.
    try std.testing.expectEqual(@as(usize, 6), std.enums.values(Tag).len);
}

test "KvQuantType, re-export accessible" {
    // Verify KvQuantType re-export is usable.
    comptime {
        _ = @TypeOf(KvQuantType);
        _ = @sizeOf(KvQuantType);
    }
}

// ── Backend dispatch tests (via CPU backend) ─────────────────────

test "Backend.add via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var b = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0 };
    var out: [8]f32 = undefined;
    be.add(&a, &b, &out, 8);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), out[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 88.0), out[7], 1e-5);
}

test "Backend.mul via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var a = [_]f32{ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    var b = [_]f32{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
    var out: [8]f32 = undefined;
    be.mul(&a, &b, &out, 8);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), out[7], 1e-5);
}

test "Backend.silu via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var input = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0 };
    var output: [8]f32 = undefined;
    be.silu(&input, &output, 8);
    // silu(0) = 0 * sigmoid(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 1e-5);
    // silu(1) = 1 * sigmoid(1) ≈ 0.7311
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), output[1], 1e-3);
    // silu(-1) = -1 * sigmoid(-1) ≈ -0.2689
    try std.testing.expectApproxEqAbs(@as(f32, -0.2689), output[2], 1e-3);
}

test "Backend.gelu via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
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

test "Backend.rmsNorm via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    // 8 elements: input = [1..8], weight = all 1.0
    var input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var weight = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var output: [8]f32 = undefined;
    be.rmsNorm(&input, &weight, &output, 8, 1e-6);
    // rms = sqrt(mean(x^2)) = sqrt((1+4+9+16+25+36+49+64)/8) = sqrt(204/8) = sqrt(25.5) ≈ 5.0498
    // output[i] = input[i] / rms
    const rms = @sqrt(@as(f32, 25.5) + 1e-6);
    try std.testing.expectApproxEqAbs(1.0 / rms, output[0], 1e-4);
    try std.testing.expectApproxEqAbs(8.0 / rms, output[7], 1e-4);
}

test "Backend.addRmsNorm via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var b = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var weight = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var output: [8]f32 = undefined;
    be.addRmsNorm(&a, &b, &weight, &output, 8, 1e-6);
    // a = a+b = [2,3,4,5,6,7,8,9], rms = sqrt(mean([4,9,16,25,36,49,64,81])) = sqrt(284/8) = sqrt(35.5)
    const rms = @sqrt(@as(f32, 35.5) + 1e-6);
    try std.testing.expectApproxEqAbs(2.0 / rms, output[0], 1e-4);
    try std.testing.expectApproxEqAbs(9.0 / rms, output[7], 1e-4);
    // Verify a was modified in-place
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), a[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), a[7], 1e-5);
}

test "Backend.rmsNormAdd via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var b = [_]f32{ 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0 };
    be.rmsNormAdd(&a, &weight, &b, 8, 1e-6);
    // rms(a) = sqrt(mean([1,4,9,16,25,36,49,64])) = sqrt(204/8) = sqrt(25.5)
    const rms = @sqrt(@as(f32, 25.5) + 1e-6);
    // b[i] = 10.0 + a[i]/rms
    try std.testing.expectApproxEqAbs(10.0 + 1.0 / rms, b[0], 1e-4);
    try std.testing.expectApproxEqAbs(10.0 + 8.0 / rms, b[7], 1e-4);
}

test "Backend.softmax via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    be.softmax(&data, 8);
    // Verify sum to 1.0
    var sum: f32 = 0.0;
    for (data) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
    // Verify monotonicity: higher input → higher probability
    for (1..8) |i| {
        try std.testing.expect(data[i] >= data[i - 1]);
    }
}

test "Backend.rope via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    // 1 head, head_dim=8, rope_dim=8, pos=0 → no rotation (angle=0, cos=1, sin=0)
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    be.rope(&x, 0, 1, 8, 8, 10000.0);
    // At pos=0 all angles are 0, cos(0)=1, sin(0)=0 → x unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), x[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), x[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), x[7], 1e-5);
}

test "Backend.rope pos=1 modifies values" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    // 1 head, head_dim=8, rope_dim=8, pos=1 → values should change
    var x = [_]f32{ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    be.rope(&x, 1, 1, 8, 8, 10000.0);
    // First element pair should be rotated: x[0] = cos(theta), x[4] = sin(theta)
    // theta_0 = 1.0 / 10000^(0/8) = 1.0 → angle = 1.0
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1.0)), x[0], 1e-4);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 1.0)), x[4], 1e-4);
}

test "Backend.embLookup f32 via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    // Simple f32 embedding table: 3 tokens, dim=4
    const table = [_]f32{
        1.0, 2.0, 3.0, 4.0, // token 0
        5.0, 6.0, 7.0, 8.0, // token 1
        9.0, 10.0, 11.0, 12.0, // token 2
    };
    var output: [4]f32 = undefined;
    const td = TensorData{ .data = @ptrCast(&table), .dtype = .f32 };
    be.embLookup(td, 1, &output, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), output[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), output[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), output[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), output[3], 1e-5);
}

test "Backend.embLookup f16 via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    // f16 embedding table: 2 tokens, dim=4
    const table = [_]f16{
        1.0, 2.0, 3.0, 4.0, // token 0
        5.0, 6.0, 7.0, 8.0, // token 1
    };
    var output: [4]f32 = undefined;
    const td = TensorData{ .data = @ptrCast(&table), .dtype = .f16 };
    be.embLookup(td, 0, &output, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), output[3], 1e-3);
}

test "Backend.l2Norm via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var x = [_]f32{ 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    be.l2Norm(&x, 8, 1e-12);
    // L2 norm of [3,4,0,...,0] = 5 → x[0] = 3/5, x[1] = 4/5
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), x[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), x[1], 1e-4);
}

test "Backend.addScaled via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var dst = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var src = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0 };
    be.addScaled(&src, &dst, 0.1, 8);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), dst[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), dst[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), dst[7], 1e-5);
}

test "Backend.siluMul via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var a = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0 };
    var b = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var out: [8]f32 = undefined;
    be.siluMul(&a, &b, &out, 8);
    // siluMul(a, 1) = silu(a) * 1 = silu(a)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), out[1], 1e-3);
}

test "Backend.geluMul via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var a = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.5 };
    var b = [_]f32{ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 };
    var out: [8]f32 = undefined;
    be.geluMul(&a, &b, &out, 8);
    // geluMul(0, 2) = gelu(0) * 2 = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-4);
    // geluMul(1, 2) = gelu(1) * 2 ≈ 0.8412 * 2 = 1.6824
    try std.testing.expectApproxEqAbs(@as(f32, 1.6824), out[1], 2e-3);
}

test "Backend.sigmoidMul via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var gate = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    be.sigmoidMul(&data, &gate, 8);
    // sigmoid(0) = 0.5, so data[i] *= 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), data[7], 1e-5);
}

test "Backend.deinterleave via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    // 2 pairs, stride=2: input = [A0,A1, B0,B1, A2,A3, B2,B3]
    var input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var out_a: [4]f32 = undefined;
    var out_b: [4]f32 = undefined;
    be.deinterleave(&input, &out_a, &out_b, 2, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_a[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out_a[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out_a[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), out_a[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out_b[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out_b[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), out_b[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), out_b[3], 1e-6);
}

test "Backend.splitQGate via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    // 2 heads, hd=2: input = [Q0,Q1,G0,G1, Q2,Q3,G2,G3]
    var qg = [_]f32{ 1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0 };
    var q_out: [4]f32 = undefined;
    var g_out: [4]f32 = undefined;
    be.splitQGate(&qg, &q_out, &g_out, 2, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), q_out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), q_out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), q_out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), q_out[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), g_out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), g_out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), g_out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 40.0), g_out[3], 1e-6);
}

test "Backend.gemv f32 via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    // y = W @ x, W=[2,4], x=[4]
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var w = [_]f32{
        1.0, 0.0, 0.0, 0.0, // row 0 → dot = 1.0
        0.0, 1.0, 0.0, 0.0, // row 1 → dot = 2.0
    };
    var y: [2]f32 = undefined;
    be.gemv(&x, .{ .data = @ptrCast(&w), .dtype = .f32 }, &y, 2, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), y[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), y[1], 1e-5);
}

test "Backend.gemm f32 via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    // Y[2,2] = X[2,2] @ W[2,2]^T with W stored row-major
    // X = [[1,0],[0,1]], W = [[2,3],[4,5]] → Y = X @ W^T = [[2,4],[3,5]]
    var x = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var w = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    var y: [4]f32 = undefined;
    be.gemm(&x, .{ .data = @ptrCast(&w), .dtype = .f32 }, &y, 2, 2, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), y[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), y[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), y[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), y[3], 1e-5);
}

test "Backend.rmsNormMulti via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    // 2 heads, head_dim=8, weight = all 1.0
    var data = [_]f32{
        1.0, 2.0, 3.0, 4.0, 5.0,  6.0,  7.0,  8.0,
        2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
    };
    var weight = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    be.rmsNormMulti(&data, &weight, 2, 8, 1e-6);
    // Head 0: rms([1..8])
    const rms0 = @sqrt(@as(f32, 25.5) + 1e-6);
    try std.testing.expectApproxEqAbs(1.0 / rms0, data[0], 1e-4);
    // Head 1: rms([2,4,..16]) = 2 * rms([1..8])
    const rms1 = @sqrt(@as(f32, 25.5 * 4.0) + 1e-6);
    try std.testing.expectApproxEqAbs(2.0 / rms1, data[8], 1e-4);
}

test "Backend.rmsNormBatched via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    // 2 tokens, dim=8
    var input = [_]f32{
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // token 0, rms = 1.0
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, // token 1, rms = 2.0
    };
    var weight = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var output: [16]f32 = undefined;
    be.rmsNormBatched(&input, &weight, &output, 2, 8, 1e-6);
    // rms([1,1,...]) = 1 → output = 1/1 = 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 1e-4);
    // rms([2,2,...]) = 2 → output = 2/2 = 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[8], 1e-4);
}

test "Backend.ropeBatched via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    // 2 tokens, 1 head, head_dim=8, rope_dim=8
    var x = [_]f32{
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // token 0
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // token 1
    };
    var positions = [_]u32{ 0, 1 };
    be.ropeBatched(&x, &positions, 2, 1, 8, 8, 10000.0);
    // pos=0: no rotation → first token unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[0], 1e-5);
    // pos=1: rotated → x[8] = cos(1.0), x[12] = sin(1.0)
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1.0)), x[8], 1e-4);
}

test "Backend.sync via CPU dispatch, no-op" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    be.sync(); // Must not panic
}

test "Backend.beginBatch/endBatch via CPU dispatch, no-ops" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    be.beginBatch();
    be.endBatch();
}

test "Backend.backendInfo via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    const info = be.backendInfo();
    try std.testing.expectEqualStrings("CPU", info.name);
}

test "Backend.allocKvSlice and freeKvSlice via CPU dispatch" {
    const allocator = std.testing.allocator;
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    const slice = try be.allocKvSlice(allocator, 256);
    try std.testing.expectEqual(@as(usize, 256), slice.len);
    @memset(slice, 0xFF);
    try std.testing.expectEqual(@as(u8, 0xFF), slice[0]);
    be.freeKvSlice(allocator, slice);
}

test "Backend.gemvMulti via CPU dispatch, single op" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var x = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var w = [_]f32{ 1.0, 2.0, 3.0, 4.0 }; // single row
    var y: [1]f32 = undefined;
    const ops = [_]GemvOp{
        .{
            .w = .{ .data = @ptrCast(&w), .dtype = .f32 },
            .y = &y,
            .n = 1,
        },
    };
    be.gemvMulti(&x, &ops, 4);
    // dot([1,1,1,1], [1,2,3,4]) = 10
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), y[0], 1e-5);
}

test "Backend.gemvMulti via CPU dispatch, empty ops" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var x = [_]f32{1.0};
    const ops = [_]GemvOp{};
    be.gemvMulti(&x, &ops, 1); // Must not panic
}

test "Backend.gemvT via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    // Q8_0 block: 2-byte f16 scale (1.0 → 0x3C00) + 32 i8 quants; first quant=1, rest=0
    var w_block: [34]u8 align(2) = undefined;
    w_block[0] = 0x00;
    w_block[1] = 0x3C;
    @memset(w_block[2..34], 0);
    w_block[2] = 1;
    var x_in = [_]f32{1.0};
    var y_out: [32]f32 = undefined;
    be.gemvT(&x_in, &w_block, &y_out, 32, 1);
    // y[0] = 1.0 * 1.0 * 1.0; y[1..] = 0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), y_out[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y_out[1], 1e-3);
}

test "Backend.sdpa f32 KV via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    // 1 head, 1 kv head, head_dim=4, seq_len=0 (first token)
    const nh: usize = 1;
    const nkv: usize = 1;
    const hd: usize = 4;
    const kvd = nkv * hd;
    const max_seq: usize = 4;
    // KV cache: enough for max_seq tokens
    var keys: [max_seq * kvd * 4]u8 = undefined;
    var values: [max_seq * kvd * 4]u8 = undefined;
    // Q, K_new, V_new
    var q = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var k_new = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var v_new = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var output: [4]f32 = undefined;
    be.sdpa(&q, &keys, &values, &k_new, &v_new, &output, nh, nkv, hd, 0, 1.0, .f32, .f32);
    // With seq_len=0 (first token), attention is just the new token with weight 1.0
    // output = softmax([q·k]) @ v = 1.0 * v = v_new
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[3], 1e-3);
}

test "Backend.sdpa nvfp4_ds_mla via CPU dispatch" {
    const kv_quant = @import("../ops/kv_quant.zig");
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    const nh: usize = 1;
    const nkv: usize = 1;
    const hd: usize = kv_quant.ds_mla_latent_dim;
    const rec = kv_quant.kvSliceBytes(.nvfp4_ds_mla, hd);
    var keys: [1024]u8 align(4) = undefined;
    var values: [1024]u8 align(4) = undefined;
    try std.testing.expect(rec <= keys.len);
    var q: [hd]f32 = @splat(0.02);
    var k_new: [hd]f32 = undefined;
    var v_new: [hd]f32 = undefined;
    for (0..hd) |i| {
        k_new[i] = @as(f32, @floatFromInt(i % 11)) * 0.2 - 1.0;
        v_new[i] = @as(f32, @floatFromInt(i % 7)) * 0.1 + 0.25;
    }
    var output: [hd]f32 = undefined;
    be.sdpa(&q, keys[0..], values[0..], &k_new, &v_new, &output, nh, nkv, hd, 0, 1.0, .nvfp4_ds_mla, .nvfp4_ds_mla);
    for (0..kv_quant.ds_mla_rope_dim) |i| {
        const idx = kv_quant.ds_mla_nope_dim + i;
        const expected: f32 = @floatCast(@as(f16, @floatCast(v_new[idx])));
        try std.testing.expectApproxEqAbs(expected, output[idx], 1e-4);
    }
}

test "Backend.sdpaPrefill f32 via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    const nh: usize = 1;
    const nkv: usize = 1;
    const hd: usize = 4;
    const kvd = nkv * hd;
    const max_seq: usize = 4;
    var keys: [max_seq * kvd * 4]u8 = undefined;
    var values: [max_seq * kvd * 4]u8 = undefined;
    var q = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var k = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var v = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var output: [4]f32 = undefined;
    be.sdpaPrefill(&q, &k, &v, &keys, &values, &output, nh, nkv, hd, 0, 1, 1.0, .f32, .f32);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[3], 1e-3);
}

test "Backend.setThreadContext via CPU dispatch, no-op" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    be.setThreadContext(); // Must not panic, CpuBackend has no setThreadContext
}

test "Backend.invalidateActivation via CPU dispatch, no-op" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var buf = [_]f32{ 1.0, 2.0 };
    be.invalidateActivation(&buf); // CpuBackend has no invalidateAct, no-op
}

test "Backend.getDevicePtr via CPU dispatch returns 0" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var buf = [_]f32{1.0};
    const ptr = be.getDevicePtr(@as([*]const f32, &buf));
    try std.testing.expectEqual(@as(u64, 0), ptr);
}

test "Backend.invalidateWeight via CPU dispatch, no-op" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var buf = [_]u8{ 0, 1, 2, 3 };
    be.invalidateWeight(@as([*]const u8, &buf)); // CpuBackend has no invalidateWeight, no-op
}

test "Backend.allReduceAdd via CPU dispatch, no-op" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var dst = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var src = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0 };
    be.allReduceAdd(&dst, &src, 8);
    // CpuBackend.allReduceAdd does dst[i] += src[i]
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), dst[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 88.0), dst[7], 1e-5);
}

test "Backend.registerHostRegion via CPU dispatch, no-op" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var data = [_]u8{ 0, 1, 2, 3 };
    be.registerHostRegion(&data, 4); // CpuBackend has no registerHostRegion, no-op
}

test "NullBackend, all method signatures are consistent with Backend" {
    // Compile-time check that NullBackend has all methods matching Backend dispatch.
    comptime {
        _ = @TypeOf(NullBackend.allocKvSlice);
        _ = @TypeOf(NullBackend.freeKvSlice);
        _ = @TypeOf(NullBackend.addRmsNorm);
        _ = @TypeOf(NullBackend.rmsNormAdd);
        _ = @TypeOf(NullBackend.addScaled);
        _ = @TypeOf(NullBackend.l2Norm);
        _ = @TypeOf(NullBackend.sigmoidMul);
        _ = @TypeOf(NullBackend.siluMul);
        _ = @TypeOf(NullBackend.geluMul);
        _ = @TypeOf(NullBackend.rmsNormMulti);
        _ = @TypeOf(NullBackend.deinterleave);
        _ = @TypeOf(NullBackend.splitQGate);
        _ = @TypeOf(NullBackend.sdpaWithStats);
        _ = @TypeOf(NullBackend.sdpaTree);
        _ = @TypeOf(NullBackend.sdpaPrefill);
        _ = @TypeOf(NullBackend.rmsNormBatched);
        _ = @TypeOf(NullBackend.ropeBatched);
        _ = @TypeOf(NullBackend.gemvT);
        _ = @TypeOf(NullBackend.beginBatch);
        _ = @TypeOf(NullBackend.endBatch);
    }
}

test "weightBytes, TQ1_0 super-block" {
    // TQ1_0: 54 bytes per 256-element super-block.
    try std.testing.expectEqual(@as(usize, 54), weightBytes(.tq1_0, 1, 256));
    try std.testing.expectEqual(@as(usize, 4 * 16 * tq1_0_block_bytes), weightBytes(.tq1_0, 4, 4096));
    // TQ2_0: 66 bytes per 256-element super-block.
    try std.testing.expectEqual(@as(usize, 66), weightBytes(.tq2_0, 1, 256));
    try std.testing.expectEqual(@as(usize, 4 * 16 * tq2_0_block_bytes), weightBytes(.tq2_0, 4, 4096));
}

test "DType, gemvRowBytes for MXFP4" {
    // MXFP4: 17 bytes per 32-element block.
    // k=4096 → 128 blocks → 128 × 17 = 2176
    try std.testing.expectEqual(@as(usize, 128 * mxfp4_block_bytes), gemvRowBytes(.mxfp4, 4096));
}

test "DType, gemvRowBytes for IQ4_XS super-block" {
    // IQ4_XS: 136 bytes per 256-element super-block.
    try std.testing.expectEqual(@as(usize, 16 * iq4_xs_block_bytes), gemvRowBytes(.iq4_xs, 4096));
}

test "BackendInfo, driver and compute fields" {
    const info = BackendInfo{
        .compute_cap = "sm_121",
        .driver_version = "CUDA 13.0",
        .os_version = "Linux 6.5.0",
    };
    try std.testing.expectEqualStrings("sm_121", info.compute_cap);
    try std.testing.expectEqualStrings("CUDA 13.0", info.driver_version);
    try std.testing.expectEqualStrings("Linux 6.5.0", info.os_version);
}

test "Backend.gemv f32 all-ones dot product via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    // y = W @ x with all ones → dot product = k
    var x = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var w = [_]f32{
        1.0, 1.0, 1.0, 1.0, // row 0 → dot = 4.0
        2.0, 2.0, 2.0, 2.0, // row 1 → dot = 8.0
    };
    var y: [2]f32 = undefined;
    be.gemv(&x, .{ .data = @ptrCast(&w), .dtype = .f32 }, &y, 2, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), y[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), y[1], 1e-5);
}

test "Backend.gemvMulti via CPU dispatch, two ops" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var x = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var w0 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var w1 = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var y0: [1]f32 = undefined;
    var y1: [1]f32 = undefined;
    const ops = [_]GemvOp{
        .{ .w = .{ .data = @ptrCast(&w0), .dtype = .f32 }, .y = &y0, .n = 1 },
        .{ .w = .{ .data = @ptrCast(&w1), .dtype = .f32 }, .y = &y1, .n = 1 },
    };
    be.gemvMulti(&x, &ops, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), y0[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 26.0), y1[0], 1e-5);
}

test "Backend.rope partial rope_dim via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    // 1 head, head_dim=8, rope_dim=4 (only first 4 elements rotated)
    // Split-complex layout: pairs are (x[i], x[i+half]) where half = rope_dim/2 = 2
    // So pairs are (x[0], x[2]) and (x[1], x[3]). x[4..7] are untouched.
    var x = [_]f32{ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    be.rope(&x, 1, 1, 8, 4, 10000.0);
    // Elements beyond rope_dim should be unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), x[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), x[5], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), x[6], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), x[7], 1e-5);
    // Rotated elements finite
    for (x[0..4]) |v| try std.testing.expect(std.math.isFinite(v));
}

test "Backend.softmax single element via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    var data = [_]f32{42.0};
    be.softmax(&data, 1);
    // softmax of single element = 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-5);
}

test "Backend.l2Norm zero vector via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    // With eps > 0, a zero vector should produce zeros (0 / sqrt(eps))
    var x = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    be.l2Norm(&x, 8, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), x[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), x[7], 1e-6);
}

test "GemvOp, MLX companion pointers set" {
    var y_buf: [4]f32 = undefined;
    var w_data = [_]u8{ 0, 0, 0, 0 };
    var scales = [_]u8{ 1, 2, 3 };
    var biases = [_]u8{ 4, 5, 6 };
    const op = GemvOp{
        .w = .{ .data = &w_data, .dtype = .mlx_q },
        .y = &y_buf,
        .n = 4,
        .mlx_scales = &scales,
        .mlx_biases = &biases,
        .mlx_bits = 4,
    };
    try std.testing.expectEqual(@as(u32, 4), op.mlx_bits);
    try std.testing.expect(op.mlx_scales != null);
    try std.testing.expect(op.mlx_biases != null);
}

test "DeltaNetParams, kqv_order true" {
    const p = DeltaNetParams{
        .conv_ch = 512,
        .d_conv = 4,
        .d_inner = 1024,
        .num_k_heads = 4,
        .head_k_dim = 64,
        .num_v_heads = 4,
        .head_v_dim = 64,
        .q_scale = 0.5,
        .rms_eps = 1e-5,
        .kqv_order = true,
    };
    try std.testing.expect(p.kqv_order);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), p.q_scale, 1e-6);
}

test "Backend.sdpaWithStats f32 KV via CPU dispatch" {
    var cpu = CpuBackend{};
    const be = Backend{ .cpu = &cpu };
    const nh: usize = 1;
    const nkv: usize = 1;
    const hd: usize = 4;
    const kvd = nkv * hd;
    const max_seq: usize = 4;
    var keys: [max_seq * kvd * 4]u8 = undefined;
    var values: [max_seq * kvd * 4]u8 = undefined;
    var q = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var k_new = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var v_new = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var output: [4]f32 = undefined;
    var head_max: [1]f32 = undefined;
    var head_sum: [1]f32 = undefined;
    be.sdpaWithStats(&q, &keys, &values, &k_new, &v_new, &output, &head_max, &head_sum, nh, nkv, hd, 0, 1.0, .f32, .f32);
    // Output should match sdpa
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[0], 1e-3);
    // Stats should be finite
    try std.testing.expect(std.math.isFinite(head_max[0]));
    try std.testing.expect(std.math.isFinite(head_sum[0]));
    try std.testing.expect(head_sum[0] > 0);
}

test "fuzz: all backend functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // weightBytes: exercise every DType with random n, k
            {
                const n_raw = smith.valueWithHash(u16, 0) | 1;
                const k_raw = smith.valueWithHash(u16, 1) | 1;
                const n: usize = @intCast(n_raw);
                const k: usize = @intCast(k_raw);
                inline for (comptime std.enums.values(DType)) |dtype| {
                    _ = weightBytes(dtype, n, k);
                }
            }
            // gemvRowBytes: exercise every DType with random k
            {
                const k: usize = @intCast(smith.valueWithHash(u16, 2));
                inline for (comptime std.enums.values(DType)) |dtype| {
                    _ = gemvRowBytes(dtype, k);
                }
            }
            // detectSystemMem, detectAvailMem, detectCacheSizes, detectOsVersion
            {
                try std.testing.expect(detectSystemMem() > 0);
                _ = detectAvailMem();
                const cs = detectCacheSizes();
                _ = cs.l1;
                _ = cs.l2;
                _ = cs.l3;
                _ = detectOsVersion();
            }
            // TensorData
            {
                var data = [_]u8{ smith.valueWithHash(u8, 3), 0, 0, 0 };
                _ = (TensorData{ .data = &data, .dtype = .f32 }).dtype;
            }
            // GemvOp
            {
                var y_buf: [1]f32 = .{0};
                var w_data = [_]u8{0} ** 4;
                const op = GemvOp{ .w = .{ .data = &w_data, .dtype = .f32 }, .y = &y_buf, .n = 1, .mlx_bits = smith.valueWithHash(u32, 4) & 0x7 };
                _ = op.mlx_scales;
                _ = op.mlx_biases;
            }
            // DeltaNetParams
            {
                const p = DeltaNetParams{ .conv_ch = smith.valueWithHash(u32, 5) | 1, .d_conv = (smith.valueWithHash(u32, 6) & 0xF) | 1, .d_inner = smith.valueWithHash(u32, 7) | 1, .num_k_heads = (smith.valueWithHash(u32, 8) & 0xF) | 1, .head_k_dim = (smith.valueWithHash(u32, 9) & 0xFF) | 1, .num_v_heads = (smith.valueWithHash(u32, 10) & 0xF) | 1, .head_v_dim = (smith.valueWithHash(u32, 11) & 0xFF) | 1, .q_scale = 1.0, .rms_eps = 1e-6, .kqv_order = (smith.valueWithHash(u8, 12) & 1) != 0 };
                _ = p.conv_ch;
            }
            // BackendInfo
            {
                const info = BackendInfo{ .n_gpu_kernels = smith.valueWithHash(u32, 13), .total_mem = @intCast(smith.valueWithHash(u32, 14)), .is_uma = (smith.valueWithHash(u8, 15) & 1) != 0, .n_threads = smith.valueWithHash(u32, 16) };
                _ = info.name;
                _ = info.arch;
                _ = info.os;
            }
            // CacheSizes
            {
                const cs = CacheSizes{ .l1 = @intCast(smith.valueWithHash(u16, 17)), .l2 = @intCast(smith.valueWithHash(u16, 18)), .l3 = @intCast(smith.valueWithHash(u16, 19)) };
                try std.testing.expect(cs.l1 <= std.math.maxInt(u16));
            }
            // BackendChoice
            {
                const choices = [_]BackendChoice{ .auto, .cpu, .metal, .vulkan, .cuda, .rocm, .webgpu };
                _ = choices[smith.valueWithHash(u8, 20) % choices.len];
            }
            // BackendState defaults
            try std.testing.expectEqualStrings("CPU", (BackendState{}).name);
            // Constants
            try std.testing.expect(buf_cache_initial_capacity > 0);
            try std.testing.expect(quant_block_elems > 0);
            try std.testing.expect(quant_super_block_elems > 0);
            try std.testing.expect(nvfp4_block_elems > 0);
            try std.testing.expect(f32_elem_bytes == 4);
            try std.testing.expect(f16_elem_bytes == 2);
            try std.testing.expect(q4_0_block_bytes > 0);
            try std.testing.expect(q4_1_block_bytes > 0);
            try std.testing.expect(q5_0_block_bytes > 0);
            try std.testing.expect(q8_0_block_bytes > 0);
            try std.testing.expect(q2_k_block_bytes > 0);
            try std.testing.expect(q3_k_block_bytes > 0);
            try std.testing.expect(q4_k_block_bytes > 0);
            try std.testing.expect(q5_k_block_bytes > 0);
            try std.testing.expect(q6_k_block_bytes > 0);
            try std.testing.expect(iq4_nl_block_bytes > 0);
            try std.testing.expect(iq4_xs_block_bytes > 0);
            try std.testing.expect(mxfp4_block_bytes > 0);
            try std.testing.expect(nvfp4_block_bytes > 0);
            try std.testing.expect(tq1_0_block_bytes > 0);
            try std.testing.expect(tq2_0_block_bytes > 0);
            // Backend dispatch via CPU
            var cpu = CpuBackend{};
            const be = Backend{ .cpu = &cpu };
            const dim = 8;
            var buf_a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            var buf_b = [_]f32{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
            var buf_w = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
            var out: [dim]f32 = undefined;
            be.add(&buf_a, &buf_b, &out, dim);
            for (&out) |v| try std.testing.expect(std.math.isFinite(v));
            be.mul(&buf_a, &buf_b, &out, dim);
            for (&out) |v| try std.testing.expect(std.math.isFinite(v));
            be.silu(&buf_a, &out, dim);
            for (&out) |v| try std.testing.expect(std.math.isFinite(v));
            be.gelu(&buf_a, &out, dim);
            for (&out) |v| try std.testing.expect(std.math.isFinite(v));
            be.rmsNorm(&buf_a, &buf_w, &out, dim, 1e-6);
            for (&out) |v| try std.testing.expect(std.math.isFinite(v));
            var buf_a2 = buf_a;
            be.addRmsNorm(&buf_a2, &buf_b, &buf_w, &out, dim, 1e-6);
            for (&out) |v| try std.testing.expect(std.math.isFinite(v));
            var addsc_dst = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            be.addScaled(&buf_b, &addsc_dst, 0.5, dim);
            for (&addsc_dst) |v| try std.testing.expect(std.math.isFinite(v));
            var sm_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            be.softmax(&sm_data, dim);
            var sm_sum: f32 = 0;
            for (&sm_data) |v| sm_sum += v;
            try std.testing.expectApproxEqAbs(@as(f32, 1.0), sm_sum, 1e-4);
            var rope_buf = [_]f32{ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            be.rope(&rope_buf, 0, 1, dim, dim, 10000.0);
            try std.testing.expect(std.math.isFinite(rope_buf[0]));
            var l2_buf = [_]f32{ 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            be.l2Norm(&l2_buf, dim, 1e-12);
            for (&l2_buf) |v| try std.testing.expect(std.math.isFinite(v));
            var sig_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            var sig_gate = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            be.sigmoidMul(&sig_data, &sig_gate, dim);
            for (&sig_data) |v| try std.testing.expect(std.math.isFinite(v));
            be.siluMul(&buf_a, &buf_b, &out, dim);
            for (&out) |v| try std.testing.expect(std.math.isFinite(v));
            be.geluMul(&buf_a, &buf_b, &out, dim);
            for (&out) |v| try std.testing.expect(std.math.isFinite(v));
            var rnm_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            var rnm_w = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
            be.rmsNormMulti(&rnm_data, &rnm_w, 2, 4, 1e-6);
            for (&rnm_data) |v| try std.testing.expect(std.math.isFinite(v));
            var di_in = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            var di_a: [4]f32 = undefined;
            var di_b: [4]f32 = undefined;
            be.deinterleave(&di_in, &di_a, &di_b, 2, 2);
            for (&di_a) |v| try std.testing.expect(std.math.isFinite(v));
            var qg_in = [_]f32{ 1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0 };
            var q_out: [4]f32 = undefined;
            var g_out: [4]f32 = undefined;
            be.splitQGate(&qg_in, &q_out, &g_out, 2, 2);
            for (&q_out) |v| try std.testing.expect(std.math.isFinite(v));
            const emb_table = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            var emb_out: [4]f32 = undefined;
            be.embLookup(.{ .data = @ptrCast(&emb_table), .dtype = .f32 }, 0, &emb_out, 4);
            for (&emb_out) |v| try std.testing.expect(std.math.isFinite(v));
            var gemv_x = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
            var gemv_w = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
            var gemv_y: [1]f32 = undefined;
            be.gemv(&gemv_x, .{ .data = @ptrCast(&gemv_w), .dtype = .f32 }, &gemv_y, 1, 4);
            try std.testing.expect(std.math.isFinite(gemv_y[0]));
            var gemm_x = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
            var gemm_w = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
            var gemm_y: [4]f32 = undefined;
            be.gemm(&gemm_x, .{ .data = @ptrCast(&gemm_w), .dtype = .f32 }, &gemm_y, 2, 2, 2);
            for (&gemm_y) |v| try std.testing.expect(std.math.isFinite(v));
            var w_block: [34]u8 align(2) = undefined;
            w_block[0] = 0x00;
            w_block[1] = 0x3C;
            @memset(w_block[2..34], 0);
            w_block[2] = 1;
            var gt_x = [_]f32{1.0};
            var gt_y: [32]f32 = undefined;
            be.gemvT(&gt_x, &w_block, &gt_y, 32, 1);
            try std.testing.expect(std.math.isFinite(gt_y[0]));
            var gm_x = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
            var gm_w = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
            var gm_y: [1]f32 = undefined;
            const gm_ops = [_]GemvOp{.{ .w = .{ .data = @ptrCast(&gm_w), .dtype = .f32 }, .y = &gm_y, .n = 1 }};
            be.gemvMulti(&gm_x, &gm_ops, 4);
            try std.testing.expect(std.math.isFinite(gm_y[0]));
            be.rmsNormBatched(&buf_a, &buf_w, &out, 1, dim, 1e-6);
            for (&out) |v| try std.testing.expect(std.math.isFinite(v));
            var rb_x = [_]f32{ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            var rb_pos = [_]u32{0};
            be.ropeBatched(&rb_x, &rb_pos, 1, 1, dim, dim, 10000.0);
            try std.testing.expect(std.math.isFinite(rb_x[0]));
            be.sync();
            be.beginBatch();
            be.endBatch();
            try std.testing.expectEqualStrings("CPU", be.backendInfo().name);
            be.setThreadContext();
            be.invalidateActivation(&out);
            try std.testing.expectEqual(@as(u64, 0), be.getDevicePtr(@as([*]const f32, &out)));
            be.invalidateWeight(@as([*]const u8, @ptrCast(&out)));
            var ar_dst = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
            var ar_src = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
            be.allReduceAdd(&ar_dst, &ar_src, dim);
            var host_region = [_]u8{0} ** 16;
            be.registerHostRegion(&host_region, 16);
            const kv_slice = be.allocKvSlice(std.testing.allocator, 64) catch return;
            try std.testing.expectEqual(@as(usize, 64), kv_slice.len);
            be.freeKvSlice(std.testing.allocator, kv_slice);
            // sdpa
            {
                const hd = 4;
                var kv_keys: [4 * 4 * 4]u8 = undefined;
                var kv_vals: [4 * 4 * 4]u8 = undefined;
                var s_q = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
                var s_k = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
                var s_v = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
                var s_out: [hd]f32 = undefined;
                be.sdpa(&s_q, &kv_keys, &kv_vals, &s_k, &s_v, &s_out, 1, 1, hd, 0, 1.0, .f32, .f32);
                for (&s_out) |v| try std.testing.expect(std.math.isFinite(v));
            }
            // sdpaWithStats
            {
                const hd = 4;
                var kv_keys: [4 * 4 * 4]u8 = undefined;
                var kv_vals: [4 * 4 * 4]u8 = undefined;
                var s_q = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
                var s_k = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
                var s_v = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
                var s_out: [hd]f32 = undefined;
                var h_max: [1]f32 = undefined;
                var h_sum: [1]f32 = undefined;
                be.sdpaWithStats(&s_q, &kv_keys, &kv_vals, &s_k, &s_v, &s_out, &h_max, &h_sum, 1, 1, hd, 0, 1.0, .f32, .f32);
                for (&s_out) |v| try std.testing.expect(std.math.isFinite(v));
                try std.testing.expect(std.math.isFinite(h_max[0]));
                try std.testing.expect(std.math.isFinite(h_sum[0]));
            }
            // sdpaPrefill
            {
                const hd = 4;
                var kv_keys: [4 * 4 * 4]u8 = undefined;
                var kv_vals: [4 * 4 * 4]u8 = undefined;
                var s_q = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
                var s_k = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
                var s_v = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
                var s_out: [hd]f32 = undefined;
                be.sdpaPrefill(&s_q, &s_k, &s_v, &kv_keys, &kv_vals, &s_out, 1, 1, hd, 0, 1, 1.0, .f32, .f32);
                for (&s_out) |v| try std.testing.expect(std.math.isFinite(v));
            }
            // Comptime: NullBackend methods (init skipped, @compileError by design)
            comptime {
                _ = &NullBackend.allocKvSlice;
                _ = &NullBackend.freeKvSlice;
                _ = &NullBackend.gemv;
                _ = &NullBackend.rmsNorm;
                _ = &NullBackend.silu;
                _ = &NullBackend.gelu;
                _ = &NullBackend.add;
                _ = &NullBackend.gemvT;
                _ = &NullBackend.addScaled;
                _ = &NullBackend.addRmsNorm;
                _ = &NullBackend.rmsNormAdd;
                _ = &NullBackend.mul;
                _ = &NullBackend.softmax;
                _ = &NullBackend.rope;
                _ = &NullBackend.embLookup;
                _ = &NullBackend.l2Norm;
                _ = &NullBackend.sigmoidMul;
                _ = &NullBackend.siluMul;
                _ = &NullBackend.geluMul;
                _ = &NullBackend.rmsNormMulti;
                _ = &NullBackend.deinterleave;
                _ = &NullBackend.splitQGate;
                _ = &NullBackend.sync;
                _ = &NullBackend.sdpa;
                _ = &NullBackend.sdpaWithStats;
                _ = &NullBackend.sdpaTree;
                _ = &NullBackend.sdpaPaged;
                _ = &NullBackend.gemvGptq;
                _ = &NullBackend.gemvAwq;
                _ = &NullBackend.gemvHqq;
                _ = &NullBackend.gemvNvfp4St;
                _ = &NullBackend.gemvMlxQ;
                _ = &NullBackend.gemvMxfp4St;
                _ = &NullBackend.gemvMulti;
                _ = &NullBackend.gemm;
                _ = &NullBackend.rmsNormBatched;
                _ = &NullBackend.ropeBatched;
                _ = &NullBackend.sdpaPrefill;
                _ = &NullBackend.deltaNet;
                _ = &NullBackend.beginBatch;
                _ = &NullBackend.endBatch;
                _ = &NullBackend.backendInfo;
            }
            // Comptime: Backend union methods
            comptime {
                _ = &Backend.allocKvSlice;
                _ = &Backend.freeKvSlice;
                _ = &Backend.gemv;
                _ = &Backend.gemm;
                _ = &Backend.rmsNormBatched;
                _ = &Backend.ropeBatched;
                _ = &Backend.sdpaPrefill;
                _ = &Backend.rmsNorm;
                _ = &Backend.silu;
                _ = &Backend.gelu;
                _ = &Backend.add;
                _ = &Backend.gemvT;
                _ = &Backend.addScaled;
                _ = &Backend.addRmsNorm;
                _ = &Backend.rmsNormAdd;
                _ = &Backend.mul;
                _ = &Backend.softmax;
                _ = &Backend.rope;
                _ = &Backend.embLookup;
                _ = &Backend.l2Norm;
                _ = &Backend.sigmoidMul;
                _ = &Backend.deinterleave;
                _ = &Backend.splitQGate;
                _ = &Backend.siluMul;
                _ = &Backend.geluMul;
                _ = &Backend.rmsNormMulti;
                _ = &Backend.sync;
                _ = &Backend.sdpa;
                _ = &Backend.sdpaPaged;
                _ = &Backend.sdpaWithStats;
                _ = &Backend.sdpaTree;
                _ = &Backend.gemvMulti;
                _ = &Backend.deltaNet;
                _ = &Backend.gemvNvfp4St;
                _ = &Backend.gemvMlxQ;
                _ = &Backend.gemvMxfp4St;
                _ = &Backend.gemvGptq;
                _ = &Backend.gemvAwq;
                _ = &Backend.beginBatch;
                _ = &Backend.endBatch;
                _ = &Backend.backendInfo;
                _ = &Backend.setThreadContext;
                _ = &Backend.invalidateActivation;
                _ = &Backend.getDevicePtr;
                _ = &Backend.invalidateWeight;
                _ = &Backend.allReduceAdd;
                _ = &Backend.registerHostRegion;
            }
            // Comptime: BackendState.init, GPU backend types, re-exports
            comptime {
                _ = &BackendState.init;
                _ = @sizeOf(CpuBackend);
                _ = @sizeOf(MetalBackend);
                _ = @sizeOf(VulkanBackend);
                _ = @sizeOf(CudaBackend);
                _ = @sizeOf(RocmBackend);
                _ = @sizeOf(WebGpuBackend);
                _ = &CpuSoftmax.softmaxSimd;
                _ = &CpuSdpa.sdpaQuantHead;
                _ = &CpuSdpa.sdpaQuantHeadWithStats;
                _ = &CpuSdpa.sdpaQuantHeads;
                _ = @sizeOf(DType);
                _ = @sizeOf(KvQuantType);
                _ = @sizeOf(PagedKvView);
            }
        }
    }.f, .{});
}
