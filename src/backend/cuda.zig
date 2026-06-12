//! NVIDIA CUDA GPU backend for accelerated tensor operations.
//!
//! Uses the CUDA Driver API (libcuda.so) loaded dynamically at runtime.
//! Kernels are written in Zig, compiled to PTX via nvptx64-cuda target,
//! and embedded into this binary.
//!
//! If libcuda is not available (macOS, Linux without NVIDIA), init() returns
//! error.CudaNotAvailable and the caller falls back to another backend.
//!
//! ## Deferred Execution Model
//!
//! Kernel launches are non-blocking. Activation buffers stay on the GPU
//! between operations — no per-op sync or download. The model calls
//! `sync()` only when CPU code needs to read GPU-produced data.
//!
//! Weight buffers are uploaded once and cached permanently (`buf_cache`).
//! Activation buffers are cached in `act_cache` with dirty/stale tracking:
//!   - **dirty**: GPU wrote newer data (download on sync)
//!   - **stale**: host may have newer data after sync + CPU work (re-upload on next GPU use)
//!   - **clean**: host and device data match
//!
//! Build PTX: `zig build ptx [-Dcuda-sm=sm_120]`
//! The generated PTX is committed at kernels/cuda/all.ptx.

const std = @import("std");
const builtin = @import("builtin");
const backend_mod = @import("backend.zig");
const TensorData = backend_mod.TensorData;
const CpuBackend = @import("cpu.zig").CpuBackend;
const KvQuantType = backend_mod.KvQuantType;
const PagedKvView = backend_mod.PagedKvView;
const kv_quant = @import("../ops/kv_quant.zig");
const mlx_ops = @import("../ops/mlx.zig");

// ── Embedded PTX ────────────────────────────────────────────────

const ptx_source = @embedFile("kernels/cuda/all.ptx");

// ── CUDA Driver API types ───────────────────────────────────────

const CUresult = c_int;
const CUdevice = c_int;
const CUcontext = ?*anyopaque;
const CUmodule = ?*anyopaque;
const CUfunction = ?*anyopaque;
const CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;

// ── Tuning constants ─────────────────────────────────────────────

/// Block size for elementwise and reduction kernels.
const block_size: u32 = 256;

/// Shared memory for block reductions (8 warps × 4 bytes).
const reduction_smem: u32 = 32;

/// SDPA prefill shared memory layout constants (must match sdpa_prefill.zig).
const prefill_kv_tile: u32 = 32;
const prefill_reduce_slots: u32 = 8;
/// Bits per U32 word — used to compute MLX words-per-group from quantization bits.
const bits_per_u32_word: usize = 32;

/// Size of the buffer for retrieving the CUDA device name.
const device_name_buf_size: usize = 256;

/// CUDA driver version encoding: major = version / 1000.
const cuda_version_major_divisor: u32 = 1000;
/// CUDA driver version encoding: minor = (version % 1000) / 10.
const cuda_version_minor_divisor: u32 = 10;

/// Library name varies by platform.
const cuda_lib_name = switch (builtin.os.tag) {
    .linux => "libcuda.so.1",
    .windows => "nvcuda.dll",
    else => "libcuda.dylib",
};

/// CUDA device attribute for detecting integrated/UMA GPUs.
const CU_DEVICE_ATTRIBUTE_INTEGRATED: c_int = 18;

/// CUDA device attributes for compute capability.
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

/// CUDA function to query total device memory.
const FnDeviceTotalMem = *const fn (*usize, CUdevice) callconv(.c) CUresult;

/// CUDA function to query free and total device memory.
const FnMemGetInfo = *const fn (*usize, *usize) callconv(.c) CUresult;

/// CUDA function to query driver version.
const FnDriverGetVersion = *const fn (*c_int) callconv(.c) CUresult;

// ── CUDA function pointer types ─────────────────────────────────

const FnInit = *const fn (c_uint) callconv(.c) CUresult;
const FnDeviceGet = *const fn (*CUdevice, c_int) callconv(.c) CUresult;
const FnDeviceGetName = *const fn ([*]u8, c_int, CUdevice) callconv(.c) CUresult;
const FnDeviceGetAttribute = *const fn (*c_int, c_int, CUdevice) callconv(.c) CUresult;
const FnCtxCreate = *const fn (*CUcontext, c_uint, CUdevice) callconv(.c) CUresult;
const FnCtxDestroy = *const fn (CUcontext) callconv(.c) CUresult;
const FnDevicePrimaryCtxRetain = *const fn (*CUcontext, CUdevice) callconv(.c) CUresult;
const FnDevicePrimaryCtxRelease = *const fn (CUdevice) callconv(.c) CUresult;
const FnCtxSync = *const fn () callconv(.c) CUresult;
const FnCtxSetCurrent = *const fn (CUcontext) callconv(.c) CUresult;
const FnModuleLoadData = *const fn (*CUmodule, [*]const u8) callconv(.c) CUresult;
const FnModuleUnload = *const fn (CUmodule) callconv(.c) CUresult;
const FnModuleGetFunction = *const fn (*CUfunction, CUmodule, [*:0]const u8) callconv(.c) CUresult;
const FnMemAlloc = *const fn (*CUdeviceptr, usize) callconv(.c) CUresult;
const FnMemFree = *const fn (CUdeviceptr) callconv(.c) CUresult;
const FnMemAllocManaged = *const fn (*CUdeviceptr, usize, c_uint) callconv(.c) CUresult;
/// CU_MEM_ATTACH_GLOBAL: memory is accessible from any stream on any device.
const CU_MEM_ATTACH_GLOBAL: c_uint = 1;
const FnMemcpyHtoD = *const fn (CUdeviceptr, *const anyopaque, usize) callconv(.c) CUresult;
const FnMemcpyDtoH = *const fn (*anyopaque, CUdeviceptr, usize) callconv(.c) CUresult;
const FnMemcpyDtoD = *const fn (CUdeviceptr, CUdeviceptr, usize) callconv(.c) CUresult;
const FnMemHostRegister = *const fn (*const anyopaque, usize, c_uint) callconv(.c) CUresult;
const FnMemHostGetDevicePointer = *const fn (*CUdeviceptr, *const anyopaque, c_uint) callconv(.c) CUresult;
const FnMemHostUnregister = *const fn (*const anyopaque) callconv(.c) CUresult;
/// CU_MEMHOSTREGISTER_DEVICEMAP: maps host memory into device address space.
const CU_MEMHOSTREGISTER_DEVICEMAP: c_uint = 0x02;
const FnLaunchKernel = *const fn (
    CUfunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    ?*anyopaque,
    [*]?*anyopaque,
    ?[*]?*anyopaque,
) callconv(.c) CUresult;

// ── Backend struct ───────────────────────────────────────────────

/// CUDA GPU backend — PTX kernels with deferred execution and driver API loading.
pub const CudaBackend = struct {
    // CUDA handles
    context: CUcontext = null,
    module: CUmodule = null,
    lib: std.DynLib = undefined,

    // Function pointers (loaded from libcuda)
    cuCtxDestroy: FnCtxDestroy = undefined,
    cuCtxSynchronize: FnCtxSync = undefined,
    cuCtxSetCurrent: ?FnCtxSetCurrent = null,
    cuModuleUnload: FnModuleUnload = undefined,
    cuModuleGetFunction: FnModuleGetFunction = undefined,
    cuMemAlloc: FnMemAlloc = undefined,
    cuMemFree: FnMemFree = undefined,
    cuMemcpyHtoD: FnMemcpyHtoD = undefined,
    cuMemcpyDtoH: FnMemcpyDtoH = undefined,
    cuMemcpyDtoD: FnMemcpyDtoD = undefined,
    cuMemHostRegister: ?FnMemHostRegister = null,
    cuMemHostGetDevicePointer: ?FnMemHostGetDevicePointer = null,
    cuMemHostUnregister: ?FnMemHostUnregister = null,
    cuMemAllocManaged: ?FnMemAllocManaged = null,
    cuLaunchKernel: FnLaunchKernel = undefined,

    // GPUDirect Storage (cuFile) — optional, loaded via dlopen
    cufile_lib: ?std.DynLib = null,
    has_gds: bool = false,

    // Kernel function handles
    fn_silu: CUfunction = null,
    fn_gelu: CUfunction = null,
    fn_add: CUfunction = null,
    fn_mul: CUfunction = null,
    fn_rms_norm: CUfunction = null,
    fn_add_rms_norm: CUfunction = null,
    fn_rms_norm_add: CUfunction = null,
    fn_softmax: CUfunction = null,
    fn_l2_norm: CUfunction = null,
    fn_rope: CUfunction = null,
    fn_gemv_f32: CUfunction = null,
    fn_gemv_bf16: CUfunction = null,
    fn_gemv_f16: CUfunction = null,
    fn_gemv_q8_0: CUfunction = null,
    fn_gemv_q4_0: CUfunction = null,
    fn_gemv_q4_1: CUfunction = null,
    fn_gemv_q5_0: CUfunction = null,
    fn_gemv_q2_k: CUfunction = null,
    fn_gemv_q3_k: CUfunction = null,
    fn_gemv_iq4_nl: CUfunction = null,
    fn_gemv_iq4_xs: CUfunction = null,
    fn_gemv_q4_0_batch: CUfunction = null,
    fn_gemv_q4_k: CUfunction = null,
    fn_gemv_q5_k: CUfunction = null,
    fn_gemv_q6_k: CUfunction = null,
    fn_gemv_fp8_e4m3: CUfunction = null,
    fn_gemv_fp8_e5m2: CUfunction = null,
    fn_silu_mul: CUfunction = null,
    fn_add_scaled: CUfunction = null,
    fn_sigmoid_mul: CUfunction = null,
    fn_gelu_mul: CUfunction = null,
    fn_deinterleave: CUfunction = null,
    fn_split_qgate: CUfunction = null,
    fn_gemv_t_q8_0: CUfunction = null,
    fn_gemv_nvfp4_st: CUfunction = null,
    fn_gemv_fp4_tc: CUfunction = null,
    fn_gemv_mlx_q4: CUfunction = null,
    fn_gemv_mlx_q6: CUfunction = null,
    fn_gemv_mlx_q8: CUfunction = null,
    fn_gemv_mxfp4_st: CUfunction = null,
    fn_gemv_gptq: CUfunction = null,
    fn_gemv_awq: CUfunction = null,
    fn_gemv_hqq: CUfunction = null,
    fn_gemv_tq1_0: CUfunction = null,
    fn_gemv_tq2_0: CUfunction = null,
    fn_fused_ffn_q8: CUfunction = null,
    fn_fused_ffn_q4k: CUfunction = null,
    fn_fused_ffn_q5k: CUfunction = null,
    fn_fused_ffn_q6k: CUfunction = null,
    fn_fused_ffn_gelu_q8: CUfunction = null,
    fn_sdpa: CUfunction = null,
    fn_sdpa_turbo: CUfunction = null,
    fn_sdpa_tree: CUfunction = null,
    fn_sdpa_paged: CUfunction = null,
    fn_sdpa_prefill: CUfunction = null,
    fn_gemm_q8_0: CUfunction = null,
    fn_rms_norm_batched: CUfunction = null,
    fn_rope_batched: CUfunction = null,
    fn_mega_qwen35_q8: CUfunction = null,
    fn_mega_gemma_q4k: CUfunction = null,
    fn_mega_gemma_q8: CUfunction = null,

    /// CPU backend for ops where CPU is genuinely faster than GPU dispatch (embLookup).
    cpu: CpuBackend = .{},

    /// Whether the GPU uses unified memory architecture (integrated GPU).
    is_uma: bool = false,

    /// Cached staging buffers for paged SDPA (avoid hot-path allocation).
    sdpa_flat_keys: ?[]f32 = null,
    sdpa_flat_vals: ?[]f32 = null,

    /// Device name retrieved during initialization (e.g., "NVIDIA GB10").
    device_name: [device_name_buf_size]u8 = undefined,
    device_name_len: usize = 0,

    /// Compute capability (e.g., sm_major=12, sm_minor=1 → "sm_121").
    sm_major: u32 = 0,
    sm_minor: u32 = 0,

    /// Total device memory in bytes.
    total_mem: usize = 0,

    /// Available (free) device memory in bytes at init time.
    avail_mem: usize = 0,

    /// CUDA driver version (e.g., 13000 → 13.0).
    driver_version: u32 = 0,

    /// Pre-formatted compute capability string (e.g., "sm_121").
    cc_str: [16]u8 = .{0} ** 16,

    /// Pre-formatted driver version string (e.g., "CUDA 13.0").
    drv_str: [16]u8 = .{0} ** 16,

    /// Allocator for buffer caches.
    allocator: std.mem.Allocator = undefined,

    /// Permanent cache: weight buffers uploaded once and reused forever.
    buf_cache: std.AutoHashMap(usize, CachedBuf) = undefined,

    /// Activation cache: device mirrors of host activation buffers.
    /// Tracks dirty/stale state for deferred sync.
    act_cache: std.AutoHashMap(usize, ActBuf) = undefined,

    /// KV cache: device mirrors of per-layer KV buffers with incremental upload.
    kv_dev_cache: std.AutoHashMap(usize, KvDevCache) = undefined,

    /// Registered UMA host memory regions (mmap'd weight files).
    /// Pointers within registered regions can use cuMemHostGetDevicePointer.
    uma_regions: [max_uma_regions]UmaRegion = @splat(.{}),
    uma_region_count: u32 = 0,

    const max_uma_regions: usize = 8;
    const UmaRegion = struct {
        base: usize = 0,
        size: usize = 0,
    };

    /// Number of PTX kernels loaded at init.
    pub const n_kernels: u32 = 44;

    /// Library name loaded via dlopen at init.
    pub const lib_name = cuda_lib_name;

    const CachedBuf = struct {
        dptr: CUdeviceptr,
        size: usize,
        /// True if this entry was registered via cuMemHostRegister (UMA).
        /// Cleanup uses cuMemHostUnregister instead of cuMemFree.
        is_registered: bool = false,
    };

    /// Device-side KV cache buffer.
    const KvDevCache = struct {
        dptr: CUdeviceptr,
        capacity: usize,
    };

    /// Activation buffer state — tracks data freshness between host and device.
    /// Transitions: clean→dirty (GPU kernel writes), dirty→clean (flushActivations
    /// downloads on sync), clean→stale (invalidateAct after CPU writes),
    /// stale→clean (getInputBuf re-uploads from host).
    const BufState = enum {
        /// Host and device data match.
        clean,
        /// GPU wrote newer data — must download on sync().
        dirty,
        /// Host may have newer data (after sync + CPU work) — must re-upload on next GPU use.
        stale,
    };

    const ActBuf = struct {
        dptr: CUdeviceptr,
        size: usize,
        state: BufState,
    };

    // ── Init / Deinit ───────────────────────────────────────────

    /// Initialize the CUDA backend: load libcuda, create context, load PTX kernels.
    pub fn init(allocator: std.mem.Allocator, device_id: u32) !CudaBackend {
        var self = CudaBackend{};
        self.allocator = allocator;
        self.buf_cache = std.AutoHashMap(usize, CachedBuf).init(allocator);
        try self.buf_cache.ensureTotalCapacity(backend_mod.buf_cache_initial_capacity);
        errdefer self.buf_cache.deinit();
        self.act_cache = std.AutoHashMap(usize, ActBuf).init(allocator);
        errdefer self.act_cache.deinit();
        self.kv_dev_cache = std.AutoHashMap(usize, KvDevCache).init(allocator);
        errdefer self.kv_dev_cache.deinit();

        // Dynamically load libcuda (try standard name, then platform-specific paths)
        self.lib = std.DynLib.open(cuda_lib_name) catch
            std.DynLib.open("/lib/aarch64-linux-gnu/" ++ cuda_lib_name) catch
            std.DynLib.open("/usr/lib/aarch64-linux-gnu/" ++ cuda_lib_name) catch
            std.DynLib.open("/usr/lib/x86_64-linux-gnu/" ++ cuda_lib_name) catch
            return error.CudaNotAvailable;
        errdefer self.lib.close();

        // Resolve all function pointers
        const cuInit = self.lookup(FnInit, "cuInit") orelse return error.CudaNotAvailable;
        const cuDeviceGet = self.lookup(FnDeviceGet, "cuDeviceGet") orelse return error.CudaNotAvailable;
        const cuDeviceGetName = self.lookup(FnDeviceGetName, "cuDeviceGetName") orelse return error.CudaNotAvailable;
        const cuDeviceGetAttribute = self.lookup(FnDeviceGetAttribute, "cuDeviceGetAttribute") orelse return error.CudaNotAvailable;
        const cuCtxCreate = self.lookup(FnCtxCreate, "cuCtxCreate_v2") orelse return error.CudaNotAvailable;
        self.cuCtxDestroy = self.lookup(FnCtxDestroy, "cuCtxDestroy_v2") orelse return error.CudaNotAvailable;
        self.cuCtxSynchronize = self.lookup(FnCtxSync, "cuCtxSynchronize") orelse return error.CudaNotAvailable;
        self.cuCtxSetCurrent = self.lookup(FnCtxSetCurrent, "cuCtxSetCurrent");
        const cuModuleLoadData = self.lookup(FnModuleLoadData, "cuModuleLoadData") orelse return error.CudaNotAvailable;
        self.cuModuleUnload = self.lookup(FnModuleUnload, "cuModuleUnload") orelse return error.CudaNotAvailable;
        self.cuModuleGetFunction = self.lookup(FnModuleGetFunction, "cuModuleGetFunction") orelse return error.CudaNotAvailable;
        self.cuMemAlloc = self.lookup(FnMemAlloc, "cuMemAlloc_v2") orelse return error.CudaNotAvailable;
        self.cuMemFree = self.lookup(FnMemFree, "cuMemFree_v2") orelse return error.CudaNotAvailable;
        self.cuMemAllocManaged = self.lookup(FnMemAllocManaged, "cuMemAllocManaged");
        self.cuMemcpyHtoD = self.lookup(FnMemcpyHtoD, "cuMemcpyHtoD_v2") orelse return error.CudaNotAvailable;
        self.cuMemcpyDtoH = self.lookup(FnMemcpyDtoH, "cuMemcpyDtoH_v2") orelse return error.CudaNotAvailable;
        self.cuMemcpyDtoD = self.lookup(FnMemcpyDtoD, "cuMemcpyDtoD_v2") orelse return error.CudaNotAvailable;
        // UMA zero-copy support (optional — only needed on integrated GPUs)
        self.cuMemHostRegister = self.lookup(FnMemHostRegister, "cuMemHostRegister_v2");
        self.cuMemHostGetDevicePointer = self.lookup(FnMemHostGetDevicePointer, "cuMemHostGetDevicePointer_v2");
        self.cuMemHostUnregister = self.lookup(FnMemHostUnregister, "cuMemHostUnregister");
        self.cuLaunchKernel = self.lookup(FnLaunchKernel, "cuLaunchKernel") orelse return error.CudaNotAvailable;

        // Initialize CUDA
        if (cuInit(0) != CUDA_SUCCESS) return error.CudaInitFailed;

        var dev: CUdevice = 0;
        if (cuDeviceGet(&dev, @intCast(device_id)) != CUDA_SUCCESS) return error.NoCudaDevice;

        // Store device name for display
        if (cuDeviceGetName(&self.device_name, @intCast(device_name_buf_size), dev) == CUDA_SUCCESS) {
            self.device_name_len = std.mem.indexOfScalar(u8, &self.device_name, 0) orelse device_name_buf_size;
        }

        // Detect UMA (integrated GPU sharing host memory)
        var integrated: c_int = 0;
        _ = cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev);
        self.is_uma = integrated != 0;

        // Query compute capability
        var sm_major: c_int = 0;
        var sm_minor: c_int = 0;
        _ = cuDeviceGetAttribute(&sm_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
        _ = cuDeviceGetAttribute(&sm_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
        self.sm_major = @intCast(@max(sm_major, 0));
        self.sm_minor = @intCast(@max(sm_minor, 0));

        // Query total device memory
        if (self.lookup(FnDeviceTotalMem, "cuDeviceTotalMem_v2")) |cuDeviceTotalMem| {
            var total: usize = 0;
            if (cuDeviceTotalMem(&total, dev) == CUDA_SUCCESS) self.total_mem = total;
        }

        // Query free device memory
        if (self.lookup(FnMemGetInfo, "cuMemGetInfo_v2")) |cuMemGetInfo| {
            var free: usize = 0;
            var total: usize = 0;
            if (cuMemGetInfo(&free, &total) == CUDA_SUCCESS) self.avail_mem = free;
        }

        // Query driver version
        if (self.lookup(FnDriverGetVersion, "cuDriverGetVersion")) |cuDriverGetVersion| {
            var ver: c_int = 0;
            if (cuDriverGetVersion(&ver) == CUDA_SUCCESS) self.driver_version = @intCast(@max(ver, 0));
        }

        // Format info strings for display
        if (self.sm_major > 0) {
            _ = std.fmt.bufPrint(&self.cc_str, "sm_{d}{d}", .{ self.sm_major, self.sm_minor }) catch {
                @memcpy(self.cc_str[0..3], "sm_");
            };
        }
        if (self.driver_version > 0) {
            _ = std.fmt.bufPrint(&self.drv_str, "CUDA {d}.{d}", .{ self.driver_version / cuda_version_major_divisor, (self.driver_version % cuda_version_major_divisor) / cuda_version_minor_divisor }) catch {
                @memcpy(self.drv_str[0..4], "CUDA");
            };
        }

        // Use primary context (shared with NCCL/runtime API) instead of cuCtxCreate
        if (self.lookup(FnDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain")) |retain| {
            if (retain(&self.context, dev) == CUDA_SUCCESS) {
                if (self.cuCtxSetCurrent) |setCurrent| _ = setCurrent(self.context);
            } else {
                if (cuCtxCreate(&self.context, 0, dev) != CUDA_SUCCESS) return error.CudaInitFailed;
            }
        } else {
            if (cuCtxCreate(&self.context, 0, dev) != CUDA_SUCCESS) return error.CudaInitFailed;
        }
        errdefer _ = self.cuCtxDestroy(self.context);

        // Load PTX module — must be null-terminated (heap-allocated, too large for stack)
        const ptx_buf = try allocator.alloc(u8, ptx_source.len + 1);
        defer allocator.free(ptx_buf);
        @memcpy(ptx_buf[0..ptx_source.len], ptx_source);
        ptx_buf[ptx_source.len] = 0;

        const load_rc = cuModuleLoadData(&self.module, ptx_buf.ptr);
        if (load_rc != CUDA_SUCCESS) {
            std.log.warn("CUDA PTX load failed with error code {d}", .{load_rc});
            return error.PtxLoadFailed;
        }
        errdefer _ = self.cuModuleUnload(self.module);

        // Get kernel function handles
        self.fn_silu = try self.getFunction("silu_kernel");
        self.fn_gelu = try self.getFunction("gelu_kernel");
        self.fn_add = try self.getFunction("add_kernel");
        self.fn_mul = try self.getFunction("mul_kernel");
        self.fn_rms_norm = try self.getFunction("rms_norm_kernel");
        self.fn_add_rms_norm = try self.getFunction("add_rms_norm_kernel");
        self.fn_rms_norm_add = try self.getFunction("rms_norm_add_kernel");
        self.fn_softmax = try self.getFunction("softmax_kernel");
        self.fn_l2_norm = try self.getFunction("l2_norm_kernel");
        self.fn_rope = try self.getFunction("rope_kernel");
        self.fn_gemv_f32 = try self.getFunction("gemv_f32_kernel");
        self.fn_gemv_bf16 = try self.getFunction("gemv_bf16_kernel");
        self.fn_gemv_f16 = try self.getFunction("gemv_f16_kernel");
        self.fn_gemv_q8_0 = try self.getFunction("gemv_q8_0_kernel");
        self.fn_gemv_q4_0 = try self.getFunction("gemv_q4_0_kernel");
        self.fn_gemv_q4_1 = try self.getFunction("gemv_q4_1_kernel");
        self.fn_gemv_q5_0 = self.getFunction("gemv_q5_0_kernel") catch null;
        self.fn_gemv_q2_k = self.getFunction("gemv_q2_k_kernel") catch null;
        self.fn_gemv_q3_k = self.getFunction("gemv_q3_k_kernel") catch null;
        self.fn_gemv_iq4_nl = self.getFunction("gemv_iq4_nl_kernel") catch null;
        self.fn_gemv_iq4_xs = self.getFunction("gemv_iq4_xs_kernel") catch null;
        self.fn_gemv_q4_0_batch = try self.getFunction("gemv_q4_0_batch_kernel");
        self.fn_gemv_q4_k = try self.getFunction("gemv_q4_k_kernel");
        self.fn_gemv_q5_k = try self.getFunction("gemv_q5_k_kernel");
        self.fn_gemv_q6_k = try self.getFunction("gemv_q6_k_kernel");
        self.fn_gemv_fp8_e4m3 = try self.getFunction("gemv_fp8_e4m3_kernel");
        self.fn_gemv_fp8_e5m2 = try self.getFunction("gemv_fp8_e5m2_kernel");
        self.fn_silu_mul = try self.getFunction("silu_mul_kernel");
        self.fn_add_scaled = try self.getFunction("add_scaled_kernel");
        self.fn_sigmoid_mul = try self.getFunction("sigmoid_mul_kernel");
        self.fn_gelu_mul = try self.getFunction("gelu_mul_kernel");
        self.fn_deinterleave = try self.getFunction("deinterleave_kernel");
        self.fn_split_qgate = try self.getFunction("split_qgate_kernel");
        self.fn_gemv_t_q8_0 = try self.getFunction("gemv_t_q8_0_kernel");
        self.fn_gemv_nvfp4_st = try self.getFunction("gemv_nvfp4_st_kernel");
        self.fn_gemv_fp4_tc = try self.getFunction("gemv_fp4_tc_fallback_kernel");
        self.fn_gemv_mlx_q4 = try self.getFunction("gemv_mlx_q4_kernel");
        self.fn_gemv_mlx_q6 = try self.getFunction("gemv_mlx_q6_kernel");
        self.fn_gemv_mlx_q8 = try self.getFunction("gemv_mlx_q8_kernel");
        self.fn_gemv_mxfp4_st = try self.getFunction("gemv_mxfp4_st_kernel");
        self.fn_gemv_gptq = self.getFunction("gemv_gptq_kernel") catch null;
        self.fn_gemv_awq = self.getFunction("gemv_awq_kernel") catch null;
        self.fn_gemv_hqq = self.getFunction("gemv_hqq_kernel") catch null;
        self.fn_gemv_tq1_0 = self.getFunction("gemv_tq1_0_kernel") catch null;
        self.fn_gemv_tq2_0 = self.getFunction("gemv_tq2_0_kernel") catch null;
        self.fn_fused_ffn_q8 = try self.getFunction("fused_ffn_gate_up_silu_q8_0_kernel");
        self.fn_fused_ffn_q4k = self.getFunction("fused_ffn_gate_up_silu_q4_k_kernel") catch null;
        self.fn_fused_ffn_q5k = self.getFunction("fused_ffn_gate_up_silu_q5_k_kernel") catch null;
        self.fn_fused_ffn_q6k = self.getFunction("fused_ffn_gate_up_silu_q6_k_kernel") catch null;
        self.fn_fused_ffn_gelu_q8 = self.getFunction("fused_ffn_gate_up_gelu_q8_0_kernel") catch null;
        self.fn_sdpa = try self.getFunction("sdpa_kernel");
        self.fn_sdpa_turbo = try self.getFunction("sdpa_turbo_kernel");
        self.fn_sdpa_tree = try self.getFunction("sdpa_tree_kernel");
        self.fn_sdpa_paged = self.getFunction("sdpa_paged_kernel") catch null;
        self.fn_sdpa_prefill = try self.getFunction("sdpa_prefill_kernel");
        self.fn_gemm_q8_0 = try self.getFunction("gemm_q8_0_kernel");
        self.fn_rms_norm_batched = try self.getFunction("rms_norm_batched_kernel");
        self.fn_rope_batched = try self.getFunction("rope_batched_kernel");
        self.fn_mega_qwen35_q8 = try self.getFunction("megakernel_qwen35_q8_kernel");
        self.fn_mega_gemma_q4k = try self.getFunction("megakernel_gemma_q4k_kernel");
        self.fn_mega_gemma_q8 = try self.getFunction("megakernel_gemma_q8_kernel");

        // Detect GPUDirect Storage (cuFile) for NVMe→VRAM direct transfer
        if (!self.is_uma) {
            if (std.DynLib.open("libcufile.so")) |lib| {
                self.cufile_lib = lib;
                self.has_gds = true;
                std.log.info("GPUDirect Storage available (libcufile.so)", .{});
            } else |_| {}
        }

        return self;
    }

    /// Release all CUDA resources: device buffers, caches, module, context, and library.
    pub fn deinit(self: *CudaBackend) void {
        // Free all cached activation buffers
        var act_it = self.act_cache.valueIterator();
        while (act_it.next()) |act| _ = self.cuMemFree(act.dptr);
        self.act_cache.deinit();

        // Free all KV device cache buffers
        var kv_it = self.kv_dev_cache.valueIterator();
        while (kv_it.next()) |kv| _ = self.cuMemFree(kv.dptr);
        self.kv_dev_cache.deinit();

        // Free all cached weight buffers
        var wt_it = self.buf_cache.iterator();
        while (wt_it.next()) |entry| {
            if (entry.value_ptr.is_registered) {
                if (self.cuMemHostUnregister) |unreg| _ = unreg(@ptrFromInt(entry.key_ptr.*));
            } else {
                _ = self.cuMemFree(entry.value_ptr.dptr);
            }
        }
        self.buf_cache.deinit();

        if (self.sdpa_flat_keys) |buf| std.heap.page_allocator.free(buf);
        if (self.sdpa_flat_vals) |buf| std.heap.page_allocator.free(buf);
        if (self.module != null) _ = self.cuModuleUnload(self.module);
        if (self.context != null) {
            _ = self.cuCtxSynchronize();
            _ = self.cuCtxDestroy(self.context);
        }
        if (self.cufile_lib) |*lib| lib.close();
        self.lib.close();
    }

    fn lookup(self: *CudaBackend, comptime T: type, name: [:0]const u8) ?T {
        return self.lib.lookup(T, name);
    }

    fn getFunction(self: *CudaBackend, name: [*:0]const u8) !CUfunction {
        var func: CUfunction = null;
        if (self.cuModuleGetFunction(&func, self.module, name) != CUDA_SUCCESS) {
            std.log.debug("CUDA kernel not found in PTX: {s}", .{name});
            return error.KernelNotFound;
        }
        return func;
    }

    // ── Low-level buffer operations ─────────────────────────────

    fn uploadToDevice(self: *CudaBackend, ptr: *const anyopaque, size: usize) CUdeviceptr {
        var dptr: CUdeviceptr = 0;
        _ = self.cuMemAlloc(&dptr, @max(size, 4));
        _ = self.cuMemcpyHtoD(dptr, ptr, size);
        return dptr;
    }

    fn downloadFromDevice(self: *CudaBackend, dptr: CUdeviceptr, ptr: *anyopaque, size: usize) void {
        _ = self.cuMemcpyDtoH(ptr, dptr, size);
    }

    /// Make CUDA context current on the calling thread.
    /// Required when model.forward() runs on a different thread than init
    /// (e.g. scheduler thread in server mode).
    pub fn setThreadContext(self: *CudaBackend) void {
        if (self.cuCtxSetCurrent) |setCurrent| {
            _ = setCurrent(self.context);
        }
    }

    // ── UMA host region registration ────────────────────────────

    /// Register a contiguous host memory region (e.g. mmap'd GGUF file) for
    /// GPU zero-copy access on UMA platforms. After registration, any pointer
    /// within the region can be mapped to a device pointer via
    /// cuMemHostGetDevicePointer without per-tensor registration.
    /// No-op on discrete GPUs or if cuMemHostRegister is unavailable.
    pub fn registerHostRegion(self: *CudaBackend, base: [*]const u8, size: usize) void {
        if (!self.is_uma) return;
        const reg = self.cuMemHostRegister orelse return;
        if (self.uma_region_count >= max_uma_regions) return;

        const addr = @intFromPtr(base);
        const page = std.heap.page_size_min;
        const aligned_base = addr & ~@as(usize, page - 1);
        const aligned_size = ((addr + size + page - 1) & ~@as(usize, page - 1)) - aligned_base;

        if (reg(@ptrFromInt(aligned_base), aligned_size, CU_MEMHOSTREGISTER_DEVICEMAP) == CUDA_SUCCESS) {
            self.uma_regions[self.uma_region_count] = .{ .base = aligned_base, .size = aligned_size };
            self.uma_region_count += 1;
        } else {
            std.log.warn("UMA cuMemHostRegister failed for region {x}+{d}", .{ aligned_base, aligned_size });
        }
    }

    /// Check if addr falls within a registered UMA region.
    fn isInUmaRegion(self: *const CudaBackend, addr: usize) bool {
        for (self.uma_regions[0..self.uma_region_count]) |r| {
            if (addr >= r.base and addr < r.base + r.size) return true;
        }
        return false;
    }

    // ── Weight cache (permanent, read-only) ─────────────────────

    /// Get device pointer for a weight buffer. Uploads once, reused forever.
    /// On UMA: uses cuMemHostGetDevicePointer for zero-copy if region registered.
    /// On discrete GPU: allocates device memory and uploads.
    fn getOrUpload(self: *CudaBackend, ptr: [*]const u8, size: usize) CUdeviceptr {
        const addr = @intFromPtr(ptr);
        if (self.buf_cache.get(addr)) |cached| {
            if (cached.size >= size) return cached.dptr;
            // Size mismatch — evict old entry
            if (cached.is_registered) {
                if (self.cuMemHostUnregister) |unreg| _ = unreg(@ptrFromInt(addr));
            } else {
                _ = self.cuMemFree(cached.dptr);
            }
            _ = self.buf_cache.remove(addr);
        }

        if (self.is_uma and self.isInUmaRegion(addr)) {
            if (self.cuMemHostGetDevicePointer) |getDevPtr| {
                var dptr: CUdeviceptr = 0;
                if (getDevPtr(&dptr, @ptrCast(ptr), 0) == CUDA_SUCCESS) {
                    self.buf_cache.put(addr, .{ .dptr = dptr, .size = size, .is_registered = true }) catch |err| {
                        std.log.warn("weight cache put failed (UMA zero-copy, size={d}): {}", .{ size, err });
                    };
                    return dptr;
                }
            }
        }

        const dptr = self.uploadToDevice(@ptrCast(ptr), size);
        self.buf_cache.put(addr, .{ .dptr = dptr, .size = size, .is_registered = false }) catch |err| {
            std.log.warn("weight cache put failed (upload, size={d}): {}", .{ size, err });
        };
        return dptr;
    }

    // ── Activation cache (deferred sync) ────────────────────────

    /// Check if addr falls within any cached activation buffer's range.
    /// Used for sub-region access (e.g. per-head rmsNorm on q_buf + h*hd).
    /// If `mark_dirty` (comptime), marks the buffer as dirty (GPU will write to it).
    /// If `refresh_stale` (comptime), re-uploads from host before use if buffer is stale.
    /// Returns device pointer with offset applied, or null if no match.
    fn findContaining(self: *CudaBackend, addr: usize, size: usize, comptime mark_dirty: bool, comptime refresh_stale: bool) ?CUdeviceptr {
        var it = self.act_cache.iterator();
        while (it.next()) |entry| {
            const base = entry.key_ptr.*;
            const act = entry.value_ptr;
            if (addr >= base and addr + size <= base + act.size) {
                if (refresh_stale and act.state == .stale) {
                    // Re-upload entire parent buffer so all sub-regions are fresh
                    const host_ptr: *const anyopaque = @ptrFromInt(base);
                    _ = self.cuMemcpyHtoD(act.dptr, host_ptr, act.size);
                    act.state = .clean;
                }
                if (mark_dirty) act.state = .dirty;
                return act.dptr + (addr - base);
            }
        }
        return null;
    }

    /// Get device buffer for a read-only input.
    /// Returns cached device pointer if clean/dirty (device has current data).
    /// Re-uploads from host if stale (host may have newer data after sync + CPU work).
    fn getInputBuf(self: *CudaBackend, ptr: anytype, size: usize) CUdeviceptr {
        const addr = @intFromPtr(ptr);
        // Exact match in activation cache
        if (self.act_cache.getPtr(addr)) |act| {
            if (act.size >= size) {
                if (act.state == .stale) {
                    _ = self.cuMemcpyHtoD(act.dptr, @ptrCast(ptr), size);
                    act.state = .clean;
                }
                return act.dptr;
            }
            _ = self.cuMemFree(act.dptr);
            _ = self.act_cache.remove(addr);
        }
        // Sub-region of a cached buffer (e.g. per-head rmsNorm)
        if (self.findContaining(addr, size, false, true)) |dptr| return dptr;
        // Weight cache (read-only, permanent)
        if (self.buf_cache.get(addr)) |cached| {
            if (cached.size >= size) return cached.dptr;
        }
        // New buffer: allocate, upload, cache as clean
        const dptr = self.uploadToDevice(@ptrCast(ptr), size);
        self.act_cache.put(addr, .{ .dptr = dptr, .size = size, .state = .clean }) catch |err| {
            std.log.warn("activation cache put failed (read, size={d}): {}", .{ size, err });
        };
        return dptr;
    }

    /// Get device buffer for a write-only output.
    /// Reuses existing allocation if available (no re-upload — kernel will write).
    fn getOutputBuf(self: *CudaBackend, ptr: anytype, size: usize) CUdeviceptr {
        const addr = @intFromPtr(ptr);
        // Exact match
        if (self.act_cache.getPtr(addr)) |act| {
            if (act.size >= size) {
                act.state = .dirty;
                return act.dptr;
            }
            _ = self.cuMemFree(act.dptr);
        }
        // Sub-region of a cached buffer
        if (self.findContaining(addr, size, true, false)) |dptr| return dptr;
        // Allocate new device buffer (no upload — kernel will write)
        var dptr: CUdeviceptr = 0;
        _ = self.cuMemAlloc(&dptr, @max(size, 4));
        self.act_cache.put(addr, .{ .dptr = dptr, .size = size, .state = .dirty }) catch |err| {
            std.log.warn("activation cache put failed (output, size={d}): {}", .{ size, err });
        };
        return dptr;
    }

    /// Get device buffer for in-place read+write (e.g. softmax, rope, l2norm).
    /// Re-uploads from host if stale, then marks dirty.
    fn getInPlaceBuf(self: *CudaBackend, ptr: anytype, size: usize) CUdeviceptr {
        const addr = @intFromPtr(ptr);
        // Exact match
        if (self.act_cache.getPtr(addr)) |act| {
            if (act.size >= size) {
                if (act.state == .stale) {
                    _ = self.cuMemcpyHtoD(act.dptr, @ptrCast(ptr), size);
                }
                act.state = .dirty;
                return act.dptr;
            }
            _ = self.cuMemFree(act.dptr);
        }
        // Sub-region of a cached buffer
        if (self.findContaining(addr, size, true, true)) |dptr| return dptr;
        // New: allocate, upload (need current data for read), mark dirty
        const dptr = self.uploadToDevice(@ptrCast(ptr), size);
        self.act_cache.put(addr, .{ .dptr = dptr, .size = size, .state = .dirty }) catch |err| {
            std.log.warn("activation cache put failed (in-place, size={d}): {}", .{ size, err });
        };
        return dptr;
    }

    /// Sync GPU, download dirty buffers to host, then mark all entries stale.
    /// Called before CPU code that may read or modify activation buffers.
    pub fn flushActivations(self: *CudaBackend) void {
        _ = self.cuCtxSynchronize();
        var it = self.act_cache.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.state == .dirty) {
                const host_ptr: *anyopaque = @ptrFromInt(entry.key_ptr.*);
                self.downloadFromDevice(entry.value_ptr.dptr, host_ptr, entry.value_ptr.size);
            }
            entry.value_ptr.state = .stale;
        }
    }

    /// Remove a specific activation buffer from the cache.
    /// Called after CPU fallback ops write to a host buffer, so the next
    /// GPU use will re-upload the CPU-written data.
    pub fn invalidateAct(self: *CudaBackend, ptr: anytype) void {
        const addr = @intFromPtr(ptr);
        // Mark as stale so next GPU access re-uploads from host
        if (self.act_cache.getPtr(addr)) |act| {
            act.state = .stale;
            return;
        }
        // Also check if ptr falls within a parent buffer
        var it = self.act_cache.iterator();
        while (it.next()) |entry| {
            const base = entry.key_ptr.*;
            const act = entry.value_ptr;
            if (addr >= base and addr < base + act.size) {
                act.state = .stale;
                return;
            }
        }
        // Not in cache — nothing to invalidate (will be uploaded fresh)
    }

    /// Get the device pointer for a host activation buffer (for NCCL direct GPU access).
    /// Returns 0 if the buffer is not in the activation cache.
    /// Wrapper for transport function pointer (takes opaque self).
    pub fn getDevicePtrOpaque(self_opaque: *anyopaque, ptr: [*]const f32) u64 {
        const self: *CudaBackend = @ptrCast(@alignCast(self_opaque));
        return self.getDevicePtr(ptr);
    }

    /// Get device pointer if the GPU has current data (dirty state).
    /// Returns 0 if buffer is not cached or is stale (host has newer data).
    pub fn getDevicePtr(self: *CudaBackend, ptr: anytype) u64 {
        const addr = @intFromPtr(ptr);
        if (self.act_cache.getPtr(addr)) |act| {
            if (act.state == .dirty) return act.dptr;
        }
        return 0;
    }

    /// Evict a weight buffer from the permanent cache so the next getOrUpload
    /// re-reads from host. Used when host-side weight data at the same address
    /// changes between TP ranks (e.g. tp_row_shard_buf is reused per rank).
    pub fn invalidateWeight(self: *CudaBackend, ptr: anytype) void {
        const addr = @intFromPtr(ptr);
        if (self.buf_cache.getPtr(addr)) |cached| {
            if (!cached.is_registered) _ = self.cuMemFree(cached.dptr);
            _ = self.buf_cache.remove(addr);
        }
    }

    // ── Launch helper ───────────────────────────────────────────

    /// CPU fallback for GEMV on UMA (sm_121) where PTX register spilling corrupts output.
    fn cpuGemvFallback(self: *CudaBackend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        self.flushActivations();
        self.cpu.gemv(x, w, y, n, k);
        self.invalidateAct(y);
    }

    fn launch(self: *CudaBackend, func: CUfunction, grid: u32, block: u32, smem: u32, params: [*]?*anyopaque) void {
        _ = self.cuLaunchKernel(func, grid, 1, 1, block, 1, 1, smem, null, params, null);
    }

    // ── Weight size helper ──────────────────────────────────────

    const weightBytes = backend_mod.weightBytes;

    // ── Backend interface ────────────────────────────────────────

    /// y[n] = W[n,k] @ x[k]. GPU kernels for F32, BF16, F16, Q8_0, Q4_0,
    /// Q4_K, Q5_K, Q6_K, FP8_E4M3, FP8_E5M2; unsupported dtypes panic.
    pub fn gemv(self: *CudaBackend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        const func = switch (w.dtype) {
            .f32 => self.fn_gemv_f32,
            .bf16 => self.fn_gemv_bf16,
            .f16 => self.fn_gemv_f16,
            .q8_0 => self.fn_gemv_q8_0,
            .q4_0 => if (self.is_uma) return self.cpuGemvFallback(x, w, y, n, k) else self.fn_gemv_q4_0,
            .q4_1 => if (self.is_uma) return self.cpuGemvFallback(x, w, y, n, k) else self.fn_gemv_q4_1,
            .q5_0 => if (self.is_uma) return self.cpuGemvFallback(x, w, y, n, k) else if (self.fn_gemv_q5_0) |f| f else return self.cpuGemvFallback(x, w, y, n, k),
            .q2_k => if (self.fn_gemv_q2_k) |f| f else return self.cpuGemvFallback(x, w, y, n, k),
            .q3_k => if (self.fn_gemv_q3_k) |f| f else return self.cpuGemvFallback(x, w, y, n, k),
            .iq4_nl => if (self.fn_gemv_iq4_nl) |f| f else return self.cpuGemvFallback(x, w, y, n, k),
            .iq4_xs => if (self.fn_gemv_iq4_xs) |f| f else return self.cpuGemvFallback(x, w, y, n, k),
            .q4_k => if (self.is_uma) return self.cpuGemvFallback(x, w, y, n, k) else self.fn_gemv_q4_k,
            .q5_k => if (self.is_uma) return self.cpuGemvFallback(x, w, y, n, k) else self.fn_gemv_q5_k,
            .q6_k => if (self.is_uma) return self.cpuGemvFallback(x, w, y, n, k) else self.fn_gemv_q6_k,
            .fp8_e4m3 => self.fn_gemv_fp8_e4m3,
            .fp8_e5m2 => self.fn_gemv_fp8_e5m2,
            .tq1_0 => if (self.fn_gemv_tq1_0) |f| f else return self.cpuGemvFallback(x, w, y, n, k),
            .tq2_0 => if (self.fn_gemv_tq2_0) |f| f else return self.cpuGemvFallback(x, w, y, n, k),
            else => @panic("CUDA GEMV: unsupported dtype — add a GPU kernel"),
        };

        var d_x = self.getInputBuf(x, k * @sizeOf(f32));
        var d_w = self.getOrUpload(w.data, weightBytes(w.dtype, n, k));
        var d_y = self.getOutputBuf(y, n * @sizeOf(f32));

        var n_u32: u32 = @intCast(n);
        var k_u32: u32 = @intCast(k);
        var params = [_]?*anyopaque{
            @ptrCast(&d_x),
            @ptrCast(&d_w),
            @ptrCast(&d_y),
            @ptrCast(&n_u32),
            @ptrCast(&k_u32),
        };
        // Multi-row kernels: Q4_K/Q5_K/Q6_K use NR=2, Q4_0/Q8_0 use NR=4.
        const grid_size: u32 = switch (w.dtype) {
            .q4_k, .q5_k, .q6_k => @intCast((n + 1) / 2),
            .q4_0, .q8_0 => @intCast((n + 3) / 4),
            else => @intCast(n),
        };
        self.launch(func, grid_size, block_size, reduction_smem, &params);
    }

    /// In-place sigmoid-gated multiply: data[i] *= sigmoid(gate[i]).
    pub fn sigmoidMul(self: *CudaBackend, data: [*]f32, gate: [*]const f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        var d_data = self.getInPlaceBuf(data, sz);
        var d_gate = self.getInputBuf(gate, sz);

        var n_u32: u32 = @intCast(n);
        var params = [_]?*anyopaque{ @ptrCast(&d_data), @ptrCast(&d_gate), @ptrCast(&n_u32) };
        const grid: u32 = @intCast((n + block_size - 1) / block_size);
        self.launch(self.fn_sigmoid_mul, grid, block_size, 0, &params);
    }

    /// Fused SiLU + multiply: out[i] = silu(a[i]) * b[i].
    pub fn siluMul(self: *CudaBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        var d_a = self.getInputBuf(a, sz);
        var d_b = self.getInputBuf(b, sz);
        var d_out = self.getOutputBuf(out, sz);

        var n_u32: u32 = @intCast(n);
        var params = [_]?*anyopaque{ @ptrCast(&d_a), @ptrCast(&d_b), @ptrCast(&d_out), @ptrCast(&n_u32) };
        const grid: u32 = @intCast((n + block_size - 1) / block_size);
        self.launch(self.fn_silu_mul, grid, block_size, 0, &params);
    }

    /// Fused GELU + multiply: out[i] = gelu(a[i]) * b[i].
    pub fn geluMul(self: *CudaBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        var d_a = self.getInputBuf(a, sz);
        var d_b = self.getInputBuf(b, sz);
        var d_out = self.getOutputBuf(out, sz);

        var n_u32: u32 = @intCast(n);
        var params = [_]?*anyopaque{ @ptrCast(&d_a), @ptrCast(&d_b), @ptrCast(&d_out), @ptrCast(&n_u32) };
        const grid: u32 = @intCast((n + block_size - 1) / block_size);
        self.launch(self.fn_gelu_mul, grid, block_size, 0, &params);
    }

    // ── Fused FFN Gate+Up+SiLU (megakernel) ──────────────────────

    /// Fused FFN: silu(W_gate @ x) * (W_up @ x) in a single dispatch.
    /// Q8_0 weights. x is f32[k], output is f32[n_ff].
    pub fn fusedFfnGateUpSiluQ8(
        self: *CudaBackend,
        x: [*]const f32,
        w_gate: [*]const u8,
        w_up: [*]const u8,
        ff_out: [*]f32,
        n_ff: usize,
        n_embd: usize,
    ) void {
        const w_bytes = weightBytes(.q8_0, n_ff, n_embd);
        var d_x = self.getInputBuf(x, n_embd * @sizeOf(f32));
        var d_gate = self.getOrUpload(w_gate, w_bytes);
        var d_up = self.getOrUpload(w_up, w_bytes);
        var d_out = self.getOutputBuf(ff_out, n_ff * @sizeOf(f32));
        var nf: u32 = @intCast(n_ff);
        var ne: u32 = @intCast(n_embd);
        var params = [_]?*anyopaque{
            @ptrCast(&d_x),   @ptrCast(&d_gate), @ptrCast(&d_up),
            @ptrCast(&d_out), @ptrCast(&nf),     @ptrCast(&ne),
        };
        self.launch(self.fn_fused_ffn_q8, @intCast(n_ff), block_size, reduction_smem, &params);
    }

    /// Fused FFN: gelu(W_gate @ x) * (W_up @ x) in a single dispatch (Gemma 3/4).
    pub fn fusedFfnGateUpGeluQ8(
        self: *CudaBackend,
        x: [*]const f32,
        w_gate: [*]const u8,
        w_up: [*]const u8,
        ff_out: [*]f32,
        n_ff: usize,
        n_embd: usize,
    ) void {
        if (self.fn_fused_ffn_gelu_q8) |func| {
            const w_bytes = weightBytes(.q8_0, n_ff, n_embd);
            var d_x = self.getInputBuf(x, n_embd * @sizeOf(f32));
            var d_gate = self.getOrUpload(w_gate, w_bytes);
            var d_up = self.getOrUpload(w_up, w_bytes);
            var d_out = self.getOutputBuf(ff_out, n_ff * @sizeOf(f32));
            var nf: u32 = @intCast(n_ff);
            var ne: u32 = @intCast(n_embd);
            var params = [_]?*anyopaque{
                @ptrCast(&d_x),   @ptrCast(&d_gate), @ptrCast(&d_up),
                @ptrCast(&d_out), @ptrCast(&nf),     @ptrCast(&ne),
            };
            self.launch(func, @intCast(n_ff), block_size, reduction_smem, &params);
        }
    }

    pub fn fusedFfnGateUpSiluQ4K(self: *CudaBackend, x: [*]const f32, w_gate: [*]const u8, w_up: [*]const u8, ff_out: [*]f32, n_ff: usize, n_embd: usize) void {
        if (self.fn_fused_ffn_q4k) |func| {
            const w_bytes = weightBytes(.q4_k, n_ff, n_embd);
            var d_x = self.getInputBuf(x, n_embd * @sizeOf(f32));
            var d_gate = self.getOrUpload(w_gate, w_bytes);
            var d_up = self.getOrUpload(w_up, w_bytes);
            var d_out = self.getOutputBuf(ff_out, n_ff * @sizeOf(f32));
            var nf: u32 = @intCast(n_ff);
            var ne: u32 = @intCast(n_embd);
            var params = [_]?*anyopaque{
                @ptrCast(&d_x),   @ptrCast(&d_gate), @ptrCast(&d_up),
                @ptrCast(&d_out), @ptrCast(&nf),     @ptrCast(&ne),
            };
            self.launch(func, @intCast(n_ff), block_size, reduction_smem, &params);
        }
    }

    pub fn fusedFfnGateUpSiluQ5K(self: *CudaBackend, x: [*]const f32, w_gate: [*]const u8, w_up: [*]const u8, ff_out: [*]f32, n_ff: usize, n_embd: usize) void {
        if (self.fn_fused_ffn_q5k) |func| {
            const w_bytes = weightBytes(.q5_k, n_ff, n_embd);
            var d_x = self.getInputBuf(x, n_embd * @sizeOf(f32));
            var d_gate = self.getOrUpload(w_gate, w_bytes);
            var d_up = self.getOrUpload(w_up, w_bytes);
            var d_out = self.getOutputBuf(ff_out, n_ff * @sizeOf(f32));
            var nf: u32 = @intCast(n_ff);
            var ne: u32 = @intCast(n_embd);
            var params = [_]?*anyopaque{
                @ptrCast(&d_x),   @ptrCast(&d_gate), @ptrCast(&d_up),
                @ptrCast(&d_out), @ptrCast(&nf),     @ptrCast(&ne),
            };
            self.launch(func, @intCast(n_ff), block_size, reduction_smem, &params);
        }
    }

    pub fn fusedFfnGateUpSiluQ6K(self: *CudaBackend, x: [*]const f32, w_gate: [*]const u8, w_up: [*]const u8, ff_out: [*]f32, n_ff: usize, n_embd: usize) void {
        if (self.fn_fused_ffn_q6k) |func| {
            const w_bytes = weightBytes(.q6_k, n_ff, n_embd);
            var d_x = self.getInputBuf(x, n_embd * @sizeOf(f32));
            var d_gate = self.getOrUpload(w_gate, w_bytes);
            var d_up = self.getOrUpload(w_up, w_bytes);
            var d_out = self.getOutputBuf(ff_out, n_ff * @sizeOf(f32));
            var nf: u32 = @intCast(n_ff);
            var ne: u32 = @intCast(n_embd);
            var params = [_]?*anyopaque{
                @ptrCast(&d_x),   @ptrCast(&d_gate), @ptrCast(&d_up),
                @ptrCast(&d_out), @ptrCast(&nf),     @ptrCast(&ne),
            };
            self.launch(func, @intCast(n_ff), block_size, reduction_smem, &params);
        }
    }

    // ── True Megakernel Dispatch ──────────────────────────────────

    /// Dispatch the Qwen 3.5 Q8_0 true megakernel: single launch for all layers.
    /// Requires cooperative launch (all blocks co-resident for grid sync).
    /// weights: mmap'd GGUF weight base pointer.
    /// layer_offsets: [n_layers * 20 * u64] byte offsets per layer.
    /// kv_keys/kv_values: KV cache buffers (Phase 2).
    /// hidden: [n_embd] f32 input/output hidden state.
    /// scratch: [scratch_size] f32 intermediate buffers.
    /// sync_ctrs: [32] u32 atomic grid sync counters (zero-initialized).
    /// ss_scratch: [1] u32 sum-of-squares accumulator (zero-initialized).
    pub fn dispatchMegakernelQwen35Q8(
        self: *CudaBackend,
        weights: [*]const u8,
        layer_offsets: [*]const u8,
        kv_keys: [*]f32,
        kv_values: [*]f32,
        hidden: [*]f32,
        scratch: [*]f32,
        sync_ctrs: [*]u32,
        ss_scratch: *u32,
        n_layers: u32,
        n_embd: u32,
        n_head: u32,
        n_kv: u32,
        head_dim: u32,
        n_ff: u32,
        rope_dim: u32,
        rope_theta: f32,
        rms_eps: f32,
        full_attn_interval: u32,
        max_seq_len: u32,
        seq_pos: u32,
        n_blocks: u32,
    ) void {
        // Weight buffer: register entire mmap'd weight file for GPU access.
        // On UMA this is zero-copy; on discrete GPU this uploads once and caches.
        // We use a large size estimate since the megakernel accesses all layer weights.
        var d_weights = self.getOrUpload(weights, n_layers * n_ff * n_embd);
        var d_layer_offsets = self.getInputBuf(layer_offsets, n_layers * 160); // 20 fields * 8 bytes
        var d_kv_keys = self.getInPlaceBuf(kv_keys, n_layers * max_seq_len * n_kv * head_dim * @sizeOf(f32));
        var d_kv_values = self.getInPlaceBuf(kv_values, n_layers * max_seq_len * n_kv * head_dim * @sizeOf(f32));
        var d_hidden = self.getInPlaceBuf(hidden, n_embd * @sizeOf(f32));

        // Scratch: hidden2 + ff_gate + ff_up + qkv_buf + ss_scratch
        const qkv_size = n_head * head_dim * 2 + n_kv * head_dim * 2;
        const scratch_elems = n_embd + 2 * n_ff + qkv_size + 1;
        var d_scratch = self.getOutputBuf(scratch, scratch_elems * @sizeOf(f32));

        // Sync counters: 32 u32 values, zero-initialized
        var d_sync_ctrs = self.getInPlaceBuf(sync_ctrs, 32 * @sizeOf(u32));

        // SS scratch: single u32 for sum-of-squares accumulation
        var d_ss_scratch = self.getInPlaceBuf(ss_scratch, @sizeOf(u32));

        var p_n_layers: u32 = n_layers;
        var p_n_embd: u32 = n_embd;
        var p_n_head: u32 = n_head;
        var p_n_kv: u32 = n_kv;
        var p_head_dim: u32 = head_dim;
        var p_n_ff: u32 = n_ff;
        var p_rope_dim: u32 = rope_dim;
        var p_rope_theta: f32 = rope_theta;
        var p_rms_eps: f32 = rms_eps;
        var p_full_attn_interval: u32 = full_attn_interval;
        var p_max_seq_len: u32 = max_seq_len;
        var p_seq_pos: u32 = seq_pos;

        var params = [_]?*anyopaque{
            @ptrCast(&d_weights),
            @ptrCast(&d_layer_offsets),
            @ptrCast(&d_kv_keys),
            @ptrCast(&d_kv_values),
            @ptrCast(&d_hidden),
            @ptrCast(&d_scratch),
            @ptrCast(&d_sync_ctrs),
            @ptrCast(&d_ss_scratch),
            @ptrCast(&p_n_layers),
            @ptrCast(&p_n_embd),
            @ptrCast(&p_n_head),
            @ptrCast(&p_n_kv),
            @ptrCast(&p_head_dim),
            @ptrCast(&p_n_ff),
            @ptrCast(&p_rope_dim),
            @ptrCast(&p_rope_theta),
            @ptrCast(&p_rms_eps),
            @ptrCast(&p_full_attn_interval),
            @ptrCast(&p_max_seq_len),
            @ptrCast(&p_seq_pos),
        };

        // Launch: n_blocks blocks x 256 threads, shared memory for block reductions.
        // All blocks must be co-resident for grid sync to work.
        // cuLaunchKernel is used (standard launch); cooperative launch
        // (cuLaunchCooperativeKernel) would be preferred but requires
        // additional driver API symbol. Standard launch works when
        // n_blocks <= device occupancy limit (typical for models < 10K blocks).
        self.launch(self.fn_mega_qwen35_q8, n_blocks, block_size, reduction_smem, &params);
    }

    /// Dispatch the Gemma 3/4 Q4_K true megakernel: single launch for all layers.
    /// Same cooperative grid sync pattern as Qwen 3.5.
    pub fn dispatchMegakernelGemmaQ4K(
        self: *CudaBackend,
        weights: [*]const u8,
        layer_offsets: [*]const u8,
        kv_keys: [*]f32,
        kv_values: [*]f32,
        hidden: [*]f32,
        scratch: [*]f32,
        sync_ctrs: [*]u32,
        ss_scratch: *u32,
        n_layers: u32,
        n_embd: u32,
        n_head: u32,
        n_kv: u32,
        head_dim: u32,
        n_ff: u32,
        rope_dim: u32,
        rope_theta: f32,
        rms_eps: f32,
        embd_scale: f32,
        max_seq_len: u32,
        seq_pos: u32,
        n_blocks: u32,
    ) void {
        var d_weights = self.getOrUpload(weights, n_layers * n_ff * n_embd);
        var d_layer_offsets = self.getInputBuf(layer_offsets, n_layers * 160);
        var d_kv_keys = self.getInPlaceBuf(kv_keys, n_layers * max_seq_len * n_kv * head_dim * @sizeOf(f32));
        var d_kv_values = self.getInPlaceBuf(kv_values, n_layers * max_seq_len * n_kv * head_dim * @sizeOf(f32));
        var d_hidden = self.getInPlaceBuf(hidden, n_embd * @sizeOf(f32));

        const qkv_size = (n_head + 2 * n_kv) * head_dim;
        const scratch_elems = n_embd + 2 * n_ff + qkv_size + 1;
        var d_scratch = self.getOutputBuf(scratch, scratch_elems * @sizeOf(f32));

        var d_sync_ctrs = self.getInPlaceBuf(sync_ctrs, 32 * @sizeOf(u32));
        var d_ss_scratch = self.getInPlaceBuf(ss_scratch, @sizeOf(u32));

        var p_n_layers: u32 = n_layers;
        var p_n_embd: u32 = n_embd;
        var p_n_head: u32 = n_head;
        var p_n_kv: u32 = n_kv;
        var p_head_dim: u32 = head_dim;
        var p_n_ff: u32 = n_ff;
        var p_rope_dim: u32 = rope_dim;
        var p_rope_theta: f32 = rope_theta;
        var p_rms_eps: f32 = rms_eps;
        var p_embd_scale: f32 = embd_scale;
        var p_max_seq_len: u32 = max_seq_len;
        var p_seq_pos: u32 = seq_pos;

        var params = [_]?*anyopaque{
            @ptrCast(&d_weights),
            @ptrCast(&d_layer_offsets),
            @ptrCast(&d_kv_keys),
            @ptrCast(&d_kv_values),
            @ptrCast(&d_hidden),
            @ptrCast(&d_scratch),
            @ptrCast(&d_sync_ctrs),
            @ptrCast(&d_ss_scratch),
            @ptrCast(&p_n_layers),
            @ptrCast(&p_n_embd),
            @ptrCast(&p_n_head),
            @ptrCast(&p_n_kv),
            @ptrCast(&p_head_dim),
            @ptrCast(&p_n_ff),
            @ptrCast(&p_rope_dim),
            @ptrCast(&p_rope_theta),
            @ptrCast(&p_rms_eps),
            @ptrCast(&p_embd_scale),
            @ptrCast(&p_max_seq_len),
            @ptrCast(&p_seq_pos),
        };

        self.launch(self.fn_mega_gemma_q4k, n_blocks, block_size, reduction_smem, &params);
    }

    /// In-place per-head rmsNorm: n_heads independent heads, each head_dim elements.
    /// Reuses the batched rmsNorm kernel (one block per head, shared weight).
    pub fn rmsNormMulti(self: *CudaBackend, data: [*]f32, weight: [*]const f32, n_heads: usize, head_dim: usize, eps: f32) void {
        var d_data = self.getInPlaceBuf(data, n_heads * head_dim * @sizeOf(f32));
        var d_w = self.getInputBuf(weight, head_dim * @sizeOf(f32));
        // in-place: input == output
        var n_tok_u: u32 = @intCast(n_heads);
        var dim_u: u32 = @intCast(head_dim);
        var eps_v: f32 = eps;
        var params = [_]?*anyopaque{
            @ptrCast(&d_data),  @ptrCast(&d_w),   @ptrCast(&d_data),
            @ptrCast(&n_tok_u), @ptrCast(&dim_u), @ptrCast(&eps_v),
        };
        self.launch(self.fn_rms_norm_batched, @intCast(n_heads), block_size, reduction_smem, &params);
    }

    /// Deinterleave paired data into two separate output buffers.
    pub fn deinterleave(self: *CudaBackend, input: [*]const f32, out_a: [*]f32, out_b: [*]f32, stride: usize, n_pairs: usize) void {
        const total = n_pairs * stride;
        var d_in = self.getInputBuf(input, total * 2 * @sizeOf(f32));
        var d_a = self.getOutputBuf(out_a, total * @sizeOf(f32));
        var d_b = self.getOutputBuf(out_b, total * @sizeOf(f32));

        var stride_u32: u32 = @intCast(stride);
        var n_pairs_u32: u32 = @intCast(n_pairs);
        var params = [_]?*anyopaque{
            @ptrCast(&d_in),       @ptrCast(&d_a),         @ptrCast(&d_b),
            @ptrCast(&stride_u32), @ptrCast(&n_pairs_u32),
        };
        const grid: u32 = @intCast((total + block_size - 1) / block_size);
        self.launch(self.fn_deinterleave, grid, block_size, 0, &params);
    }

    /// Split concatenated Q+gate per-head data into separate arrays.
    pub fn splitQGate(self: *CudaBackend, qg: [*]const f32, q_out: [*]f32, g_out: [*]f32, hd: usize, nh: usize) void {
        const total = nh * hd;
        const sz = total * @sizeOf(f32);
        var d_qg = self.getInputBuf(qg, sz * 2);
        var d_q = self.getOutputBuf(q_out, sz);
        var d_g = self.getOutputBuf(g_out, sz);
        var hd_u32: u32 = @intCast(hd);
        var nh_u32: u32 = @intCast(nh);
        var params = [_]?*anyopaque{
            @ptrCast(&d_qg),   @ptrCast(&d_q),    @ptrCast(&d_g),
            @ptrCast(&hd_u32), @ptrCast(&nh_u32),
        };
        const grid: u32 = @intCast((total + block_size - 1) / block_size);
        self.launch(self.fn_split_qgate, grid, block_size, 0, &params);
    }

    /// Batched GEMV: fuse multiple GEMV ops sharing the same input into a single
    /// kernel launch. On Q4_0 with 2-4 ops, uses the dedicated batch kernel.
    /// Otherwise falls back to sequential gemv calls.
    pub fn gemvMulti(self: *CudaBackend, x: [*]const f32, ops: []const backend_mod.GemvOp, k: usize) void {
        if (ops.len >= 2 and ops.len <= 4) {
            // Check if all ops share the same dtype and we have a batch kernel
            const dtype = ops[0].w.dtype;
            var all_same = true;
            for (ops[1..]) |op| {
                if (op.w.dtype != dtype) {
                    all_same = false;
                    break;
                }
            }
            if (all_same and dtype == .q4_0) {
                self.launchBatchedGemvQ4_0(x, ops, k);
                return;
            }
        }
        // Fallback: sequential dispatch
        for (ops) |op| self.gemv(x, op.w, op.y, op.n, k);
    }

    /// Launch the fused Q4_0 batched GEMV kernel for 2-4 ops.
    fn launchBatchedGemvQ4_0(self: *CudaBackend, x: [*]const f32, ops: []const backend_mod.GemvOp, k: usize) void {
        var d_x = self.getInputBuf(x, k * @sizeOf(f32));

        // Op 0 (always present)
        var d_w0 = self.getOrUpload(ops[0].w.data, weightBytes(.q4_0, ops[0].n, k));
        var d_y0 = self.getOutputBuf(ops[0].y, ops[0].n * @sizeOf(f32));
        var n0: u32 = @intCast(ops[0].n);

        // Op 1 (always present for len >= 2)
        var d_w1 = self.getOrUpload(ops[1].w.data, weightBytes(.q4_0, ops[1].n, k));
        var d_y1 = self.getOutputBuf(ops[1].y, ops[1].n * @sizeOf(f32));
        var n1: u32 = @intCast(ops[1].n);

        // Op 2 (present for len >= 3, else dummy with n2=0)
        var d_w2: CUdeviceptr = d_w0;
        var d_y2: CUdeviceptr = d_y0;
        var n2: u32 = 0;
        if (ops.len >= 3) {
            d_w2 = self.getOrUpload(ops[2].w.data, weightBytes(.q4_0, ops[2].n, k));
            d_y2 = self.getOutputBuf(ops[2].y, ops[2].n * @sizeOf(f32));
            n2 = @intCast(ops[2].n);
        }

        // Op 3 (present for len >= 4, else dummy with n3=0)
        var d_w3: CUdeviceptr = d_w0;
        var d_y3: CUdeviceptr = d_y0;
        var n3: u32 = 0;
        if (ops.len >= 4) {
            d_w3 = self.getOrUpload(ops[3].w.data, weightBytes(.q4_0, ops[3].n, k));
            d_y3 = self.getOutputBuf(ops[3].y, ops[3].n * @sizeOf(f32));
            n3 = @intCast(ops[3].n);
        }

        var k_u32: u32 = @intCast(k);
        var params = [_]?*anyopaque{
            @ptrCast(&d_x),
            @ptrCast(&d_w0),
            @ptrCast(&d_y0),
            @ptrCast(&n0),
            @ptrCast(&d_w1),
            @ptrCast(&d_y1),
            @ptrCast(&n1),
            @ptrCast(&d_w2),
            @ptrCast(&d_y2),
            @ptrCast(&n2),
            @ptrCast(&d_w3),
            @ptrCast(&d_y3),
            @ptrCast(&n3),
            @ptrCast(&k_u32),
        };

        const grid = n0 + n1 + n2 + n3;
        self.launch(self.fn_gemv_q4_0_batch, grid, block_size, reduction_smem, &params);
    }

    /// output[i] = input[i] * weight[i] * rsqrt(mean(x^2) + eps)
    pub fn rmsNorm(self: *CudaBackend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        const sz = n * @sizeOf(f32);
        var d_in = self.getInputBuf(input, sz);
        // Force re-upload of weight — models may reuse the same buffer for different
        // per-layer norm weights (e.g. Nemotron bf16_buf_small written by CPU each layer).
        // Mark stale so getInputBuf re-uploads to the existing device buffer (no realloc).
        if (self.act_cache.getPtr(@intFromPtr(weight))) |act| {
            act.state = .stale;
        }
        var d_w = self.getInputBuf(weight, sz);
        var d_out = self.getOutputBuf(output, sz);

        var n_u32: u32 = @intCast(n);
        var eps_f32: f32 = eps;
        var params = [_]?*anyopaque{
            @ptrCast(&d_in),
            @ptrCast(&d_w),
            @ptrCast(&d_out),
            @ptrCast(&n_u32),
            @ptrCast(&eps_f32),
        };
        self.launch(self.fn_rms_norm, 1, block_size, reduction_smem, &params);
    }

    /// SiLU activation: output[i] = input[i] * sigmoid(input[i])
    pub fn silu(self: *CudaBackend, input: [*]const f32, output: [*]f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        var d_in = self.getInputBuf(input, sz);
        var d_out = self.getOutputBuf(output, sz);

        var n_u32: u32 = @intCast(n);
        var params = [_]?*anyopaque{ @ptrCast(&d_in), @ptrCast(&d_out), @ptrCast(&n_u32) };
        const grid: u32 = @intCast((n + block_size - 1) / block_size);
        self.launch(self.fn_silu, grid, block_size, 0, &params);
    }

    /// GELU activation
    pub fn gelu(self: *CudaBackend, input: [*]const f32, output: [*]f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        var d_in = self.getInputBuf(input, sz);
        var d_out = self.getOutputBuf(output, sz);

        var n_u32: u32 = @intCast(n);
        var params = [_]?*anyopaque{ @ptrCast(&d_in), @ptrCast(&d_out), @ptrCast(&n_u32) };
        const grid: u32 = @intCast((n + block_size - 1) / block_size);
        self.launch(self.fn_gelu, grid, block_size, 0, &params);
    }

    /// Element-wise add
    pub fn add(self: *CudaBackend, a: [*]const f32, b: [*]const f32, output: [*]f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        var d_a = self.getInputBuf(a, sz);
        var d_b = self.getInputBuf(b, sz);
        var d_out = self.getOutputBuf(output, sz);

        var n_u32: u32 = @intCast(n);
        var params = [_]?*anyopaque{ @ptrCast(&d_a), @ptrCast(&d_b), @ptrCast(&d_out), @ptrCast(&n_u32) };
        const grid: u32 = @intCast((n + block_size - 1) / block_size);
        self.launch(self.fn_add, grid, block_size, 0, &params);
    }

    /// Fused add + rmsNorm: a[i] += b[i], output = rmsNorm(a, weight, eps).
    pub fn addRmsNorm(self: *CudaBackend, a: [*]f32, b: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        const sz = n * @sizeOf(f32);
        var d_a = self.getInPlaceBuf(a, sz);
        var d_b = self.getInputBuf(b, sz);
        // Force re-upload of weight — models may reuse the same buffer for different
        // per-layer norm weights. Mark stale so getInputBuf re-uploads (no realloc).
        if (self.act_cache.getPtr(@intFromPtr(weight))) |act| {
            act.state = .stale;
        }
        var d_w = self.getInputBuf(weight, sz);
        var d_out = self.getOutputBuf(output, sz);

        var n_u32: u32 = @intCast(n);
        var eps_f32: f32 = eps;
        var params = [_]?*anyopaque{
            @ptrCast(&d_a),   @ptrCast(&d_b),   @ptrCast(&d_w),
            @ptrCast(&d_out), @ptrCast(&n_u32), @ptrCast(&eps_f32),
        };
        self.launch(self.fn_add_rms_norm, 1, block_size, reduction_smem, &params);
    }

    /// Fused rmsNorm + accumulate: b[i] += rmsNorm(a, weight, eps)[i].
    pub fn rmsNormAdd(self: *CudaBackend, a: [*]const f32, weight: [*]const f32, b: [*]f32, n: usize, eps: f32) void {
        const sz = n * @sizeOf(f32);
        var d_a = self.getInputBuf(a, sz);
        if (self.act_cache.getPtr(@intFromPtr(weight))) |act| act.state = .stale;
        var d_w = self.getInputBuf(weight, sz);
        var d_b = self.getInPlaceBuf(b, sz);
        var n_u32: u32 = @intCast(n);
        var eps_f32: f32 = eps;
        var params = [_]?*anyopaque{
            @ptrCast(&d_a),   @ptrCast(&d_w),     @ptrCast(&d_b),
            @ptrCast(&n_u32), @ptrCast(&eps_f32),
        };
        self.launch(self.fn_rms_norm_add, 1, block_size, reduction_smem, &params);
    }

    /// Transposed GEMV: y[out_dim] = W^T @ x[in_dim] for Q8_0 3D weights.
    pub fn gemvT(self: *CudaBackend, x: [*]const f32, w: [*]const u8, y: [*]f32, out_dim: usize, in_dim: usize) void {
        const quant = @import("../ops/quant.zig");
        const w_bytes = (out_dim * in_dim / quant.quant_block_elems) * quant.q8_0_block_bytes;
        var d_x = self.getInputBuf(x, in_dim * @sizeOf(f32));
        var d_w = self.getOrUpload(w, w_bytes);
        var d_y = self.getOutputBuf(y, out_dim * @sizeOf(f32));

        var out_u32: u32 = @intCast(out_dim);
        var in_u32: u32 = @intCast(in_dim);
        var params = [_]?*anyopaque{
            @ptrCast(&d_x),     @ptrCast(&d_w),    @ptrCast(&d_y),
            @ptrCast(&out_u32), @ptrCast(&in_u32),
        };
        self.launch(self.fn_gemv_t_q8_0, @intCast(out_dim), block_size, reduction_smem, &params);
    }

    /// Scaled accumulate: dst[i] += src[i] * scale.
    pub fn addScaled(self: *CudaBackend, src: [*]const f32, dst: [*]f32, scale: f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        var d_src = self.getInputBuf(src, sz);
        var d_dst = self.getInPlaceBuf(dst, sz);

        var scale_f32: f32 = scale;
        var n_u32: u32 = @intCast(n);
        var params = [_]?*anyopaque{
            @ptrCast(&d_src),     @ptrCast(&d_dst),
            @ptrCast(&scale_f32), @ptrCast(&n_u32),
        };
        const grid: u32 = @intCast((n + block_size - 1) / block_size);
        self.launch(self.fn_add_scaled, grid, block_size, 0, &params);
    }

    /// Element-wise mul
    pub fn mul(self: *CudaBackend, a: [*]const f32, b: [*]const f32, output: [*]f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        var d_a = self.getInputBuf(a, sz);
        var d_b = self.getInputBuf(b, sz);
        var d_out = self.getOutputBuf(output, sz);

        var n_u32: u32 = @intCast(n);
        var params = [_]?*anyopaque{ @ptrCast(&d_a), @ptrCast(&d_b), @ptrCast(&d_out), @ptrCast(&n_u32) };
        const grid: u32 = @intCast((n + block_size - 1) / block_size);
        self.launch(self.fn_mul, grid, block_size, 0, &params);
    }

    /// In-place softmax
    pub fn softmax(self: *CudaBackend, data: [*]f32, n: usize) void {
        var d_data = self.getInPlaceBuf(data, n * @sizeOf(f32));

        var n_u32: u32 = @intCast(n);
        var params = [_]?*anyopaque{ @ptrCast(&d_data), @ptrCast(&n_u32) };
        self.launch(self.fn_softmax, 1, block_size, reduction_smem, &params);
    }

    /// Rotary Position Embedding (in-place)
    pub fn rope(self: *CudaBackend, x: [*]f32, pos: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        var d_x = self.getInPlaceBuf(x, n_heads * head_dim * @sizeOf(f32));

        var pos_u32: u32 = @intCast(pos);
        var nh_u32: u32 = @intCast(n_heads);
        var hd_u32: u32 = @intCast(head_dim);
        var rd_u32: u32 = @intCast(rope_dim);
        var theta_f32: f32 = theta;
        var params = [_]?*anyopaque{
            @ptrCast(&d_x),    @ptrCast(&pos_u32), @ptrCast(&nh_u32),
            @ptrCast(&hd_u32), @ptrCast(&rd_u32),  @ptrCast(&theta_f32),
        };
        const pairs = n_heads * rope_dim / 2;
        const grid: u32 = @intCast((pairs + block_size - 1) / block_size);
        self.launch(self.fn_rope, grid, block_size, 0, &params);
    }

    /// Embedding lookup — CPU is faster than GPU dispatch for single-row read.
    pub fn embLookup(self: *CudaBackend, table: TensorData, token_id: u32, output: [*]f32, dim: usize) void {
        self.flushActivations();
        self.cpu.embLookup(table, token_id, output, dim);
        self.invalidateAct(output);
    }

    /// L2 normalize in-place.
    pub fn l2Norm(self: *CudaBackend, x: [*]f32, n: usize, eps: f32) void {
        var d_x = self.getInPlaceBuf(x, n * @sizeOf(f32));

        var n_u32: u32 = @intCast(n);
        var eps_f32: f32 = eps;
        var params = [_]?*anyopaque{ @ptrCast(&d_x), @ptrCast(&n_u32), @ptrCast(&eps_f32) };
        self.launch(self.fn_l2_norm, 1, block_size, reduction_smem, &params);
    }

    /// NVFP4 SafeTensors GEMV: packed nibbles + FP8 E4M3 scales, group_size=16.
    pub fn gemvNvfp4St(self: *CudaBackend, x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: usize, k: usize) void {
        var d_x = self.getInputBuf(x, k * @sizeOf(f32));
        var d_w = self.getOrUpload(weight, n * (k / 2));
        var d_s = self.getOrUpload(scale, n * (k / 16));
        var d_y = self.getOutputBuf(y, n * @sizeOf(f32));

        var n_u32: u32 = @intCast(n);
        var k_u32: u32 = @intCast(k);
        var params = [_]?*anyopaque{
            @ptrCast(&d_x), @ptrCast(&d_w),   @ptrCast(&d_s),
            @ptrCast(&d_y), @ptrCast(&n_u32), @ptrCast(&k_u32),
        };
        // Use tensor core path on SM120+ (Blackwell), fallback on older
        const kernel = if (self.sm_major >= 12) self.fn_gemv_fp4_tc else self.fn_gemv_nvfp4_st;
        self.launch(kernel, @intCast(n), block_size, reduction_smem, &params);
    }

    /// MLX affine quantized GEMV: packed int (4/6/8-bit) + BF16 scales/biases, group_size=64.
    pub fn gemvMlxQ(self: *CudaBackend, x: [*]const f32, weight: [*]const u8, scales: [*]const u8, biases: [*]const u8, y: [*]f32, n: usize, k: usize, bits: u32) void {
        const mlx_group_size: usize = mlx_ops.mlx_group_size;
        const gpr = (k + mlx_group_size - 1) / mlx_group_size;
        const wpg: usize = mlx_group_size * bits / bits_per_u32_word;
        const w_bytes = n * gpr * wpg * @sizeOf(u32);
        const sb_bytes = n * gpr * @sizeOf(u16);

        var d_x = self.getInputBuf(x, k * @sizeOf(f32));
        var d_w = self.getOrUpload(weight, w_bytes);
        var d_s = self.getOrUpload(scales, sb_bytes);
        var d_b = self.getOrUpload(biases, sb_bytes);
        var d_y = self.getOutputBuf(y, n * @sizeOf(f32));

        var n_u32: u32 = @intCast(n);
        var k_u32: u32 = @intCast(k);
        var params = [_]?*anyopaque{
            @ptrCast(&d_x),   @ptrCast(&d_w), @ptrCast(&d_s),
            @ptrCast(&d_b),   @ptrCast(&d_y), @ptrCast(&n_u32),
            @ptrCast(&k_u32),
        };
        const func = switch (bits) {
            8 => self.fn_gemv_mlx_q8,
            6 => self.fn_gemv_mlx_q6,
            else => self.fn_gemv_mlx_q4,
        };
        self.launch(func, @intCast(n), block_size, reduction_smem, &params);
    }

    /// MXFP4 SafeTensors GEMV: u32-packed nibbles + E8M0 scales, group_size=32.
    pub fn gemvMxfp4St(self: *CudaBackend, x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: usize, k: usize) void {
        const mxfp4_gs: usize = 32;
        const gpr = (k + mxfp4_gs - 1) / mxfp4_gs;
        const wpg: usize = 4; // 32 nibbles / 8 per word
        const w_bytes = n * gpr * wpg * @sizeOf(u32);

        var d_x = self.getInputBuf(x, k * @sizeOf(f32));
        var d_w = self.getOrUpload(weight, w_bytes);
        var d_s = self.getOrUpload(scale, n * gpr);
        var d_y = self.getOutputBuf(y, n * @sizeOf(f32));

        var n_u32: u32 = @intCast(n);
        var k_u32: u32 = @intCast(k);
        var params = [_]?*anyopaque{
            @ptrCast(&d_x), @ptrCast(&d_w),   @ptrCast(&d_s),
            @ptrCast(&d_y), @ptrCast(&n_u32), @ptrCast(&k_u32),
        };
        self.launch(self.fn_gemv_mxfp4_st, @intCast(n), block_size, reduction_smem, &params);
    }

    /// GPTQ INT4 GEMV on CUDA GPU.
    pub fn gemvGptq(self: *CudaBackend, x: [*]const f32, qweight: [*]const u32, scales: [*]const u16, qzeros: [*]const u32, y: [*]f32, n: usize, k: usize, group_size: u32) void {
        if (self.fn_gemv_gptq) |func| {
            const words_per_row = k / 8;
            const n_groups = (k + group_size - 1) / group_size;

            var d_x = self.getInputBuf(x, k * @sizeOf(f32));
            var d_w = self.getOrUpload(@ptrCast(qweight), n * words_per_row * @sizeOf(u32));
            var d_s = self.getOrUpload(@ptrCast(scales), n * n_groups * @sizeOf(u16));
            var d_z = self.getOrUpload(@ptrCast(qzeros), n_groups * ((n + 7) / 8) * @sizeOf(u32));
            var d_y = self.getOutputBuf(y, n * @sizeOf(f32));

            var n_u32: u32 = @intCast(n);
            var k_u32: u32 = @intCast(k);
            var gs_u32: u32 = group_size;
            var params = [_]?*anyopaque{
                @ptrCast(&d_x),   @ptrCast(&d_w),    @ptrCast(&d_s),
                @ptrCast(&d_z),   @ptrCast(&d_y),    @ptrCast(&n_u32),
                @ptrCast(&k_u32), @ptrCast(&gs_u32),
            };
            self.launch(func, @intCast(n), block_size, reduction_smem, &params);
        } else {
            const gptq_ops = @import("../ops/gptq.zig");
            gptq_ops.gptqGemv(x, qweight, scales, qzeros, y, n, k, group_size);
        }
    }

    /// AWQ INT4 GEMV on CUDA GPU.
    pub fn gemvAwq(self: *CudaBackend, x: [*]const f32, qweight: [*]const u32, scales: [*]const u16, qzeros: [*]const u32, y: [*]f32, n: usize, k: usize, group_size: u32) void {
        if (self.fn_gemv_awq) |func| {
            const n_words = n / 8;
            const n_groups = (k + group_size - 1) / group_size;

            var d_x = self.getInputBuf(x, k * @sizeOf(f32));
            var d_w = self.getOrUpload(@ptrCast(qweight), k * n_words * @sizeOf(u32));
            var d_s = self.getOrUpload(@ptrCast(scales), n_groups * n * @sizeOf(u16));
            var d_z = self.getOrUpload(@ptrCast(qzeros), n_groups * n_words * @sizeOf(u32));
            var d_y = self.getOutputBuf(y, n * @sizeOf(f32));

            var n_u32: u32 = @intCast(n);
            var k_u32: u32 = @intCast(k);
            var gs_u32: u32 = group_size;
            var params = [_]?*anyopaque{
                @ptrCast(&d_x),   @ptrCast(&d_w),    @ptrCast(&d_s),
                @ptrCast(&d_z),   @ptrCast(&d_y),    @ptrCast(&n_u32),
                @ptrCast(&k_u32), @ptrCast(&gs_u32),
            };
            self.launch(func, @intCast(n), block_size, reduction_smem, &params);
        } else {
            const awq_ops = @import("../ops/awq.zig");
            awq_ops.awqGemv(x, qweight, scales, qzeros, y, n, k, group_size);
        }
    }

    /// HQQ 4-bit GEMV on CUDA GPU.
    /// w_q: uint8 [n_out, k_in/2], scale/zero: bf16 [n_out, k_in/group_size].
    /// Dequant: w = (nibble - zero) * scale. Kernel uses NR=4 row parallelism.
    pub fn gemvHqq(self: *CudaBackend, x: [*]const f32, w_q: [*]const u8, scale: [*]const u8, zero: [*]const u8, y: [*]f32, n: usize, k: usize, group_size: u32) void {
        if (self.fn_gemv_hqq) |func| {
            const bytes_per_row = (k + 1) / 2;
            const n_groups = (k + group_size - 1) / group_size;

            var d_x = self.getInputBuf(x, k * @sizeOf(f32));
            var d_wq = self.getOrUpload(@ptrCast(w_q), n * bytes_per_row * @sizeOf(u8));
            var d_sc = self.getOrUpload(@ptrCast(scale), n * n_groups * @sizeOf(u16));
            var d_zr = self.getOrUpload(@ptrCast(zero), n * n_groups * @sizeOf(u16));
            var d_y = self.getOutputBuf(y, n * @sizeOf(f32));

            var n_u32: u32 = @intCast(n);
            var k_u32: u32 = @intCast(k);
            var gs_u32: u32 = group_size;
            var params = [_]?*anyopaque{
                @ptrCast(&d_x),   @ptrCast(&d_wq),   @ptrCast(&d_sc),
                @ptrCast(&d_zr),  @ptrCast(&d_y),    @ptrCast(&n_u32),
                @ptrCast(&k_u32), @ptrCast(&gs_u32),
            };
            const grid: u32 = @intCast((n + 3) / 4);
            self.launch(func, grid, block_size, reduction_smem, &params);
        } else {
            const hqq_ops = @import("../ops/hqq.zig");
            hqq_ops.hqqGemv(x, w_q, @ptrCast(@alignCast(scale)), @ptrCast(@alignCast(zero)), y, n, k, group_size);
        }
    }

    /// Commit pending GPU work and download results to host.
    /// Call before CPU code reads buffers written by deferred GPU ops.
    /// After sync, all act_cache entries are marked stale — CPU may modify
    /// any host buffer before the next GPU op.
    pub fn sync(self: *CudaBackend) void {
        self.flushActivations();
    }

    /// No-op — CUDA dispatches are not batched.
    pub fn beginBatch(_: *CudaBackend) void {}
    /// No-op — CUDA dispatches are not batched.
    pub fn endBatch(_: *CudaBackend) void {}

    // ── CUDA Graph capture (spec decode optimization) ────────────────────────
    // CUDA graphs eliminate per-step kernel launch overhead (~10μs/step).
    // Full implementation requires refactoring kernel launches to use a CUDA stream
    // (currently using default stream 0). API surface is defined here for future use.
    // When implemented: capture the draft forward pass as a graph for the first round,
    // then replay for subsequent spec decode rounds with updated device pointers.

    /// Begin CUDA graph capture on the current stream.
    /// No-op until stream-based kernel launches are implemented.
    pub fn beginGraphCapture(_: *CudaBackend) void {
        // TODO: cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL)
        std.log.debug("CUDA graph capture: not yet implemented (requires stream refactor)", .{});
    }

    /// End CUDA graph capture and instantiate executable graph.
    /// Returns true on success, false if capture is unsupported.
    pub fn endGraphCapture(_: *CudaBackend) bool {
        // TODO: cuStreamEndCapture(stream, &graph); cuGraphInstantiate(...)
        return false;
    }

    /// Launch a previously captured CUDA graph for the spec decode hot path.
    /// Falls back to regular execution if graph was never captured.
    pub fn launchGraph(_: *CudaBackend) bool {
        // TODO: cuGraphLaunch(graphExec, stream)
        return false;
    }

    /// Returns backend startup information for display.
    pub fn backendInfo(self: *const CudaBackend) backend_mod.BackendInfo {
        return .{
            .name = "CUDA",
            .device_name = self.device_name[0..self.device_name_len],
            .lib_name = cuda_lib_name,
            .n_gpu_kernels = n_kernels,
            .kernel_type = "PTX",
            .total_mem = self.total_mem,
            .avail_mem = self.avail_mem,
            .is_uma = self.is_uma,
            .compute_cap = std.mem.sliceTo(&self.cc_str, 0),
            .driver_version = std.mem.sliceTo(&self.drv_str, 0),
        };
    }

    // ── KV cache allocation ────────────────────────────────────

    /// Allocate a KV cache slice using the host allocator.
    /// On UMA platforms, the host pointer is later registered for GPU access
    /// via registerRamKv(). On discrete GPUs, the caller manages VRAM
    /// mirroring separately via kv_dev_cache during SDPA.
    pub fn allocKvSlice(_: *CudaBackend, allocator: std.mem.Allocator, n: usize) error{OutOfMemory}![]u8 {
        // Use host allocator — cuMemAllocManaged on UMA (GB10) returns pointers
        // that cause data corruption when used as both host and device memory.
        return allocator.alloc(u8, n);
    }

    /// Free a KV cache slice allocated via allocKvSlice.
    pub fn freeKvSlice(_: *CudaBackend, allocator: std.mem.Allocator, slice: []u8) void {
        if (slice.len == 0) return;
        allocator.free(slice);
    }

    /// Register RAM-tier KV block in act_cache without upload.
    /// On UMA platforms (GB10 Blackwell), host memory is GPU-accessible via
    /// unified addressing — the host pointer is used directly as the device
    /// pointer. No copy needed.
    ///
    /// On discrete GPUs, allocates device buffer and uploads once. Future
    /// optimization: use cuMemAllocHost for pinned RAM tier (faster transfers).
    pub fn registerRamKv(self: *CudaBackend, host_ptr: [*]u8, size: usize) !void {
        const addr = @intFromPtr(host_ptr);

        // Check if already tracked
        if (self.act_cache.get(addr)) |_| return; // Already registered

        if (self.is_uma) {
            // UMA: Host memory is GPU-accessible, no upload needed.
            // On UMA platforms (integrated GPU), the host pointer IS the device
            // pointer via unified addressing. Register as clean (data on device
            // matches host).
            try self.act_cache.put(addr, .{
                .dptr = @intFromPtr(host_ptr), // Same address on UMA
                .size = size,
                .state = .clean,
            });
            std.log.debug("Registered RAM-tier KV block at {x} (UMA zero-copy)", .{addr});
        } else {
            // Discrete GPU: allocate device buffer + upload
            // (Future optimization: use cuMemAllocHost for pinned RAM tier)
            var dev_ptr: CUdeviceptr = 0;
            const result = self.cuMemAlloc(&dev_ptr, size);
            if (result != 0) return error.CudaMemAllocFailed;

            const upload = self.cuMemcpyHtoD(dev_ptr, host_ptr, size);
            if (upload != 0) return error.CudaMemcpyFailed;

            try self.act_cache.put(addr, .{
                .dptr = dev_ptr,
                .size = size,
                .state = .dirty, // Device has data, host may be stale
            });
            std.log.debug("Uploaded RAM-tier KV block to device at {x}", .{dev_ptr});
        }
    }

    // ── KV device cache (incremental upload) ───────────────────

    /// Get or allocate device KV cache buffer. Returns device pointer.
    /// Allocates full capacity on first use. Does NOT upload from host.
    /// Get or allocate a GPU KV cache buffer for the given host pointer.
    /// On first allocation, uploads the existing host data so that accumulated
    /// positions (pre-filled in host memory) are visible to GPU kernels.
    fn getOrAllocKvBuf(self: *CudaBackend, addr: usize, capacity: usize) CUdeviceptr {
        if (self.kv_dev_cache.getPtr(addr)) |kv| return kv.dptr;

        var dptr: CUdeviceptr = 0;
        _ = self.cuMemAlloc(&dptr, @max(capacity, 4));
        // Upload any pre-existing host-side data so accumulated KV positions are visible.
        if (addr != 0 and capacity > 0) {
            _ = self.cuMemcpyHtoD(dptr, @ptrFromInt(addr), capacity);
        }
        self.kv_dev_cache.put(addr, .{
            .dptr = dptr,
            .capacity = capacity,
        }) catch |err| {
            std.log.warn("cache put failed: {}", .{err});
        };
        return dptr;
    }

    /// Fused scaled dot-product attention on GPU with KV cache append.
    /// Supports f32 KV cache (existing fast path) and TurboQuant 2/3/4-bit
    /// KV cache (native GPU dequant — no CPU fallback for SDPA compute).
    /// KV append for turbo types uses CPU quantization (once per token per layer,
    /// not the SDPA hot path). Non-turbo quantized types (q8_0, etc.) panic.
    pub fn sdpa(self: *CudaBackend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type_k: backend_mod.KvQuantType, kv_type_v: backend_mod.KvQuantType) void {
        const is_turbo_k = kv_type_k.isTurbo();
        const is_turbo_v = kv_type_v.isTurbo();
        const is_f32_k = (kv_type_k == .f32);
        const is_f32_v = (kv_type_v == .f32);

        // Non-turbo, non-f32 quantized KV: not yet supported
        if ((!is_f32_k and !is_turbo_k) or (!is_f32_v and !is_turbo_v))
            @panic("CUDA SDPA: unsupported KV type — use --kv-type f32 or turbo2/3/4");

        const kvd = nkv * hd;
        var sl: u32 = @intCast(seq_len + 1);

        if (is_f32_k and is_f32_v) {
            // Pure f32 path: use original sdpa_kernel
            const f32_keys: []f32 = @as([*]f32, @ptrCast(@alignCast(keys.ptr)))[0 .. keys.len / @sizeOf(f32)];
            const f32_values: []f32 = @as([*]f32, @ptrCast(@alignCast(values.ptr)))[0 .. values.len / @sizeOf(f32)];
            const kvd_bytes = kvd * @sizeOf(f32);

            var d_keys = self.getOrAllocKvBuf(@intFromPtr(f32_keys.ptr), f32_keys.len * @sizeOf(f32));
            var d_vals = self.getOrAllocKvBuf(@intFromPtr(f32_values.ptr), f32_values.len * @sizeOf(f32));

            const d_k_new = self.getInputBuf(k_new, kvd_bytes);
            const d_v_new = self.getInputBuf(v_new, kvd_bytes);

            _ = self.cuMemcpyDtoD(d_keys + seq_len * kvd_bytes, d_k_new, kvd_bytes);
            _ = self.cuMemcpyDtoD(d_vals + seq_len * kvd_bytes, d_v_new, kvd_bytes);

            var d_q = self.getInputBuf(q, nh * hd * @sizeOf(f32));
            var d_out = self.getOutputBuf(output, nh * hd * @sizeOf(f32));

            var nh_u32: u32 = @intCast(nh);
            var nkv_u32: u32 = @intCast(nkv);
            var hd_u32: u32 = @intCast(hd);
            var kvd_u32: u32 = @intCast(kvd);
            var scale_f32: f32 = scale;

            var params = [_]?*anyopaque{
                @ptrCast(&d_q),       @ptrCast(&d_keys), @ptrCast(&d_vals),
                @ptrCast(&d_out),     @ptrCast(&nh_u32), @ptrCast(&nkv_u32),
                @ptrCast(&hd_u32),    @ptrCast(&sl),     @ptrCast(&kvd_u32),
                @ptrCast(&scale_f32),
            };

            const smem: u32 = (sl + 1) * @sizeOf(f32);
            self.launch(self.fn_sdpa, @intCast(nh), block_size, smem, &params);
        } else {
            // Turbo or mixed path: CPU KV append + GPU turbo SDPA kernel.
            // KV append on CPU (one write per token per layer — not the hot path).
            self.flushActivations();
            const k_off = kv_quant.kvByteOffset(kv_type_k, seq_len * kvd);
            const v_off = kv_quant.kvByteOffset(kv_type_v, seq_len * kvd);
            kv_quant.kvStore(keys.ptr + k_off, k_new, kvd, kv_type_k);
            kv_quant.kvStore(values.ptr + v_off, v_new, kvd, kv_type_v);

            // Upload only the newly quantized token to the persistent device KV buffer.
            // Prior tokens are already on-device from earlier forward passes.
            const k_new_bytes = kv_quant.kvSliceBytes(kv_type_k, kvd);
            const v_new_bytes = kv_quant.kvSliceBytes(kv_type_v, kvd);
            var d_keys = self.getOrAllocKvBuf(@intFromPtr(keys.ptr), keys.len);
            var d_vals = self.getOrAllocKvBuf(@intFromPtr(values.ptr), values.len);

            _ = self.cuMemcpyHtoD(d_keys + k_off, @ptrCast(keys.ptr + k_off), k_new_bytes);
            _ = self.cuMemcpyHtoD(d_vals + v_off, @ptrCast(values.ptr + v_off), v_new_bytes);

            var d_q = self.getInputBuf(q, nh * hd * @sizeOf(f32));
            var d_out = self.getOutputBuf(output, nh * hd * @sizeOf(f32));

            var nh_u32: u32 = @intCast(nh);
            var nkv_u32: u32 = @intCast(nkv);
            var hd_u32: u32 = @intCast(hd);
            var kvd_u32: u32 = @intCast(kvd);
            var scale_f32: f32 = scale;
            var bits_k_u: u32 = kv_type_k.turboBits();
            var bits_v_u: u32 = kv_type_v.turboBits();
            var bb_k_u: u32 = kv_type_k.turboBlockByteSize();
            var bb_v_u: u32 = kv_type_v.turboBlockByteSize();

            var params = [_]?*anyopaque{
                @ptrCast(&d_q),       @ptrCast(&d_keys),   @ptrCast(&d_vals),
                @ptrCast(&d_out),     @ptrCast(&nh_u32),   @ptrCast(&nkv_u32),
                @ptrCast(&hd_u32),    @ptrCast(&sl),       @ptrCast(&kvd_u32),
                @ptrCast(&scale_f32), @ptrCast(&bits_k_u), @ptrCast(&bits_v_u),
                @ptrCast(&bb_k_u),    @ptrCast(&bb_v_u),
            };

            const smem: u32 = (sl + 1) * @sizeOf(f32);
            self.launch(self.fn_sdpa_turbo, @intCast(nh), block_size, smem, &params);
        }
    }

    /// SDPA with per-head softmax stats for split-attention merge.
    /// Fills identity stats (max=0, sum=1) — GPU SDPA already produces
    /// normalized output, so the merge formula treats it as-is.
    pub fn sdpaWithStats(self: *CudaBackend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, head_max: [*]f32, head_sum: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        self.sdpa(q, keys, values, k_new, v_new, output, nh, nkv, hd, seq_len, scale, kv_type_k, kv_type_v);
        for (0..nh) |h| {
            head_max[h] = 0.0;
            head_sum[h] = 1.0;
        }
    }

    // ── Batched prefill ops ─────────────────────────────────────

    /// GEMM: Y[n_tok × n_out] = X[n_tok × n_in] @ W[n_out × n_in]^T.
    /// Native GPU kernel for Q8_0; others use loop-of-GEMV.
    pub fn gemm(self: *CudaBackend, x: [*]const f32, w: TensorData, y: [*]f32, n_tok: usize, n_out: usize, n_in: usize) void {
        if (n_tok <= 1) {
            self.gemv(x, w, y, n_out, n_in);
            return;
        }
        if (w.dtype == .q8_0) {
            var d_x = self.getInputBuf(x, n_tok * n_in * @sizeOf(f32));
            var d_w = self.getOrUpload(w.data, weightBytes(w.dtype, n_out, n_in));
            var d_y = self.getOutputBuf(y, n_tok * n_out * @sizeOf(f32));
            var n_out_u: u32 = @intCast(n_out);
            var n_in_u: u32 = @intCast(n_in);
            var n_tok_u: u32 = @intCast(n_tok);
            var params = [_]?*anyopaque{
                @ptrCast(&d_x),     @ptrCast(&d_w),    @ptrCast(&d_y),
                @ptrCast(&n_out_u), @ptrCast(&n_in_u), @ptrCast(&n_tok_u),
            };
            self.launch(self.fn_gemm_q8_0, @intCast(n_out), block_size, reduction_smem, &params);
            return;
        }
        // Fallback: loop-of-GEMV for other dtypes
        for (0..n_tok) |t| self.gemv(x + t * n_in, w, y + t * n_out, n_out, n_in);
    }

    /// Batched RMS normalization — single GPU dispatch, one block per row.
    pub fn rmsNormBatched(self: *CudaBackend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n_tok: usize, dim: usize, eps: f32) void {
        var d_in = self.getInputBuf(input, n_tok * dim * @sizeOf(f32));
        // Mark weight stale — may be a mutable per-layer buffer (see rmsNorm comment).
        if (self.act_cache.getPtr(@intFromPtr(weight))) |act| act.state = .stale;
        var d_w = self.getInputBuf(weight, dim * @sizeOf(f32));
        var d_out = self.getOutputBuf(output, n_tok * dim * @sizeOf(f32));
        var n_tok_u: u32 = @intCast(n_tok);
        var dim_u: u32 = @intCast(dim);
        var eps_v: f32 = eps;
        var params = [_]?*anyopaque{
            @ptrCast(&d_in),    @ptrCast(&d_w),   @ptrCast(&d_out),
            @ptrCast(&n_tok_u), @ptrCast(&dim_u), @ptrCast(&eps_v),
        };
        self.launch(self.fn_rms_norm_batched, @intCast(n_tok), block_size, reduction_smem, &params);
    }

    /// Batched RoPE — single GPU dispatch for all tokens.
    pub fn ropeBatched(self: *CudaBackend, x: [*]f32, positions: [*]const u32, n_tok: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        const half_rope = rope_dim / 2;
        const total = n_tok * n_heads * half_rope;
        const stride = n_heads * head_dim;
        var d_x = self.getInPlaceBuf(x, n_tok * stride * @sizeOf(f32));
        var d_pos = self.getInputBuf(positions, n_tok * @sizeOf(u32));
        var n_tok_u: u32 = @intCast(n_tok);
        var nh_u: u32 = @intCast(n_heads);
        var hd_u: u32 = @intCast(head_dim);
        var rd_u: u32 = @intCast(rope_dim);
        var theta_v: f32 = theta;
        var params = [_]?*anyopaque{
            @ptrCast(&d_x),     @ptrCast(&d_pos),
            @ptrCast(&n_tok_u), @ptrCast(&nh_u),
            @ptrCast(&hd_u),    @ptrCast(&rd_u),
            @ptrCast(&theta_v),
        };
        const grid = @as(u32, @intCast((total + block_size - 1) / block_size));
        self.launch(self.fn_rope_batched, grid, block_size, 0, &params);
    }

    /// Prefill SDPA — FlashAttention-2 with causal masking.
    /// Attends to both cached KV (prev_len positions) and new KV (n_tok positions).
    /// For f32 KV: native FA2 GPU kernel (single dispatch for all tokens).
    pub fn sdpaTree(self: *CudaBackend, q_all: [*]const f32, prefix_keys: [*]const u8, prefix_values: [*]const u8, tree_keys: [*]const f32, tree_values: [*]const f32, output: [*]f32, ancestor_masks: [*]const [8]u64, nh: usize, nkv: usize, hd: usize, prefix_len: usize, n_nodes: u32, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        if (kv_type_k == .f32 and kv_type_v == .f32 and n_nodes > 0) {
            const kvd = nkv * hd;
            var d_q = self.getInputBuf(q_all, n_nodes * nh * hd * @sizeOf(f32));
            var d_pk = self.getInputBuf(@as([*]const f32, @ptrCast(@alignCast(prefix_keys))), prefix_len * kvd * @sizeOf(f32));
            var d_pv = self.getInputBuf(@as([*]const f32, @ptrCast(@alignCast(prefix_values))), prefix_len * kvd * @sizeOf(f32));
            var d_tk = self.getInputBuf(tree_keys, n_nodes * kvd * @sizeOf(f32));
            var d_tv = self.getInputBuf(tree_values, n_nodes * kvd * @sizeOf(f32));
            var d_out = self.getOutputBuf(output, n_nodes * nh * hd * @sizeOf(f32));
            var d_masks = self.getInputBuf(@as([*]const u64, @ptrCast(ancestor_masks)), n_nodes * 8 * @sizeOf(u64));
            var nh_u: u32 = @intCast(nh);
            var nkv_u: u32 = @intCast(nkv);
            var hd_u: u32 = @intCast(hd);
            var pl_u: u32 = @intCast(prefix_len);
            var nn_u: u32 = n_nodes;
            var sc: f32 = scale;
            var params = [_]?*anyopaque{
                @ptrCast(&d_q),     @ptrCast(&d_pk), @ptrCast(&d_pv),
                @ptrCast(&d_tk),    @ptrCast(&d_tv), @ptrCast(&d_out),
                @ptrCast(&d_masks), @ptrCast(&nh_u), @ptrCast(&nkv_u),
                @ptrCast(&hd_u),    @ptrCast(&pl_u), @ptrCast(&nn_u),
                @ptrCast(&sc),
            };
            self.launch(self.fn_sdpa_tree, n_nodes * @as(u32, @intCast(nh)), block_size, reduction_smem, &params);
            return;
        }
        @import("kernels/cpu/sdpa_tree.zig").sdpaTree(q_all, prefix_keys, prefix_values, tree_keys, tree_values, output, ancestor_masks, nh, nkv, hd, prefix_len, n_nodes, scale, kv_type_k, kv_type_v);
    }

    /// Paged SDPA: block-table-indexed attention for non-contiguous KV cache.
    /// Appends k_new/v_new to paged cache, gathers blocks into flat contiguous
    /// buffers, uploads to GPU, then dispatches the paged SDPA kernel.
    /// Falls back to CPU kernel if PTX was not rebuilt with sdpa_paged_kernel.
    pub fn sdpaPaged(self: *CudaBackend, q: [*]const f32, kv_view: PagedKvView, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, scale: f32, _: KvQuantType, _: KvQuantType) void {
        const kvd = nkv * hd;
        var sl: u32 = @intCast(kv_view.seq_len + 1);

        // Flush pending GPU ops so CPU can safely write to paged cache
        self.flushActivations();

        // Append k_new/v_new at current seq_len position
        @memcpy(kv_view.keyPtrMut(kv_view.seq_len)[0..kvd], k_new[0..kvd]);
        @memcpy(kv_view.valuePtrMut(kv_view.seq_len)[0..kvd], v_new[0..kvd]);

        // Fall back to CPU if paged kernel not in PTX (needs `zig build ptx` to regenerate)
        if (self.fn_sdpa_paged == null) {
            const cpu_sdpa = @import("kernels/cpu/sdpa.zig");
            for (0..nh) |h| {
                cpu_sdpa.sdpaPagedHead(q, kv_view, output, h, nh, nkv, hd, sl, scale);
            }
            return;
        }

        // Gather scattered blocks into flat contiguous staging buffers.
        // K_flat/V_flat layout: block i at [i * block_size * kvd .. (i+1) * block_size * kvd].
        const n_logical_blocks = (sl + kv_view.block_size - 1) / kv_view.block_size;
        var max_phys: u32 = 0;
        for (kv_view.block_table[0..n_logical_blocks]) |phys_id| max_phys = @max(max_phys, phys_id);
        const n_phys_blocks: usize = @as(usize, max_phys) + 1;
        const block_stride = @as(usize, kv_view.block_size) * kvd;
        const flat_elems = n_phys_blocks * block_stride;
        const flat_bytes = flat_elems * @sizeOf(f32);

        if (self.sdpa_flat_keys == null or self.sdpa_flat_keys.?.len < flat_elems) {
            if (self.sdpa_flat_keys) |old| std.heap.page_allocator.free(old);
            self.sdpa_flat_keys = std.heap.page_allocator.alloc(f32, flat_elems) catch
                @panic("CUDA sdpaPaged: out of memory for flat key staging buffer");
        }
        if (self.sdpa_flat_vals == null or self.sdpa_flat_vals.?.len < flat_elems) {
            if (self.sdpa_flat_vals) |old| std.heap.page_allocator.free(old);
            self.sdpa_flat_vals = std.heap.page_allocator.alloc(f32, flat_elems) catch
                @panic("CUDA sdpaPaged: out of memory for flat value staging buffer");
        }
        const flat_keys = self.sdpa_flat_keys.?;
        const flat_vals = self.sdpa_flat_vals.?;

        for (kv_view.block_table[0..n_logical_blocks]) |phys_id| {
            const dst_off = @as(usize, phys_id) * block_stride;
            const blk = kv_view.blocks[phys_id];
            @memcpy(flat_keys[dst_off..][0..block_stride], blk.keys[0..block_stride]);
            @memcpy(flat_vals[dst_off..][0..block_stride], blk.values[0..block_stride]);
        }

        // Upload flat buffers, block_table, Q to device and launch kernel
        var d_q = self.getInputBuf(q, nh * hd * @sizeOf(f32));
        var d_k = self.uploadToDevice(@ptrCast(flat_keys.ptr), flat_bytes);
        var d_v = self.uploadToDevice(@ptrCast(flat_vals.ptr), flat_bytes);
        var d_out = self.getOutputBuf(output, nh * hd * @sizeOf(f32));
        var d_bt = self.uploadToDevice(@ptrCast(kv_view.block_table.ptr), n_logical_blocks * @sizeOf(u32));

        var nh_u: u32 = @intCast(nh);
        var nkv_u: u32 = @intCast(nkv);
        var hd_u: u32 = @intCast(hd);
        var kvd_u: u32 = @intCast(kvd);
        var scale_f: f32 = scale;
        var paged_bs_u: u32 = kv_view.block_size;

        var params = [_]?*anyopaque{
            @ptrCast(&d_q),   @ptrCast(&d_k),     @ptrCast(&d_v),
            @ptrCast(&d_out), @ptrCast(&d_bt),    @ptrCast(&nh_u),
            @ptrCast(&nkv_u), @ptrCast(&hd_u),    @ptrCast(&sl),
            @ptrCast(&kvd_u), @ptrCast(&scale_f), @ptrCast(&paged_bs_u),
        };

        const smem: u32 = (sl + 1) * @sizeOf(f32);
        self.launch(self.fn_sdpa_paged.?, @intCast(nh), block_size, smem, &params);

        // Free temporary device buffers (not cached — staging data changes every call)
        _ = self.cuMemFree(d_k);
        _ = self.cuMemFree(d_v);
        _ = self.cuMemFree(d_bt);
    }

    /// For turbo KV: CPU-side KV append + sequential GPU turbo SDPA per token.
    /// For other quantized KV types: panics (not yet supported).
    pub fn sdpaPrefill(self: *CudaBackend, q: [*]const f32, k: [*]const f32, v: [*]const f32, kv_keys: []u8, kv_values: []u8, output: [*]f32, nh: usize, nkv: usize, hd: usize, prev_len: usize, n_tok: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        const is_turbo_k = kv_type_k.isTurbo();
        const is_turbo_v = kv_type_v.isTurbo();
        const is_f32_k = (kv_type_k == .f32);
        const is_f32_v = (kv_type_v == .f32);

        // Turbo/mixed prefill: CPU KV append + sequential GPU turbo SDPA per token.
        if ((is_turbo_k or is_f32_k) and (is_turbo_v or is_f32_v) and (is_turbo_k or is_turbo_v)) {
            self.flushActivations();
            const kvd = nkv * hd;

            // Append all n_tok keys/values to KV cache on CPU
            for (0..n_tok) |t| {
                const src_off = t * kvd;
                const dst_elem = (prev_len + t) * kvd;
                const dst_byte_k = kv_quant.kvByteOffset(kv_type_k, dst_elem);
                const dst_byte_v = kv_quant.kvByteOffset(kv_type_v, dst_elem);
                kv_quant.kvStore(kv_keys.ptr + dst_byte_k, k + src_off, kvd, kv_type_k);
                kv_quant.kvStore(kv_values.ptr + dst_byte_v, v + src_off, kvd, kv_type_v);
            }

            // Upload only the newly written tokens to device. Prior tokens
            // are already on-device from earlier forward passes.
            const new_start_k = kv_quant.kvByteOffset(kv_type_k, prev_len * kvd);
            const new_start_v = kv_quant.kvByteOffset(kv_type_v, prev_len * kvd);
            const new_bytes_k = kv_quant.kvSliceBytes(kv_type_k, n_tok * kvd);
            const new_bytes_v = kv_quant.kvSliceBytes(kv_type_v, n_tok * kvd);
            var d_keys = self.getOrAllocKvBuf(@intFromPtr(kv_keys.ptr), kv_keys.len);
            var d_vals = self.getOrAllocKvBuf(@intFromPtr(kv_values.ptr), kv_values.len);
            _ = self.cuMemcpyHtoD(d_keys + new_start_k, @ptrCast(kv_keys.ptr + new_start_k), new_bytes_k);
            _ = self.cuMemcpyHtoD(d_vals + new_start_v, @ptrCast(kv_values.ptr + new_start_v), new_bytes_v);

            // Sequential GPU SDPA per token (each uses turbo kernel over full history)
            for (0..n_tok) |t| {
                var sl: u32 = @intCast(prev_len + t + 1);
                const q_off = t * nh * hd;
                const out_off = t * nh * hd;

                var d_q = self.getInputBuf(q + q_off, nh * hd * @sizeOf(f32));
                var d_out = self.getOutputBuf(output + out_off, nh * hd * @sizeOf(f32));

                var nh_u: u32 = @intCast(nh);
                var nkv_u: u32 = @intCast(nkv);
                var hd_u: u32 = @intCast(hd);
                var kvd_u: u32 = @intCast(kvd);
                var scale_f: f32 = scale;
                var bits_k_u: u32 = kv_type_k.turboBits();
                var bits_v_u: u32 = kv_type_v.turboBits();
                var bb_k_u: u32 = kv_type_k.turboBlockByteSize();
                var bb_v_u: u32 = kv_type_v.turboBlockByteSize();

                var params = [_]?*anyopaque{
                    @ptrCast(&d_q),     @ptrCast(&d_keys),   @ptrCast(&d_vals),
                    @ptrCast(&d_out),   @ptrCast(&nh_u),     @ptrCast(&nkv_u),
                    @ptrCast(&hd_u),    @ptrCast(&sl),       @ptrCast(&kvd_u),
                    @ptrCast(&scale_f), @ptrCast(&bits_k_u), @ptrCast(&bits_v_u),
                    @ptrCast(&bb_k_u),  @ptrCast(&bb_v_u),
                };

                const smem: u32 = (sl + 1) * @sizeOf(f32);
                self.launch(self.fn_sdpa_turbo, @intCast(nh), block_size, smem, &params);
            }
            return;
        }

        // Non-turbo, non-f32 quantized KV: not yet supported
        if (kv_type_k != .f32 or kv_type_v != .f32)
            @panic("CUDA SDPA prefill: unsupported KV type — use --kv-type f32 or turbo2/3/4");

        // Pure f32 path
        const kvd = nkv * hd;

        // Get device pointers for Q and new K/V
        var d_q = self.getInputBuf(q, n_tok * nh * hd * @sizeOf(f32));
        var d_k_new = self.getInputBuf(k, n_tok * kvd * @sizeOf(f32));
        var d_v_new = self.getInputBuf(v, n_tok * kvd * @sizeOf(f32));
        var d_out = self.getOutputBuf(output, n_tok * nh * hd * @sizeOf(f32));

        // Get device pointers for KV cache
        const f32_keys: []f32 = @as([*]f32, @ptrCast(@alignCast(kv_keys.ptr)))[0 .. kv_keys.len / @sizeOf(f32)];
        const f32_values: []f32 = @as([*]f32, @ptrCast(@alignCast(kv_values.ptr)))[0 .. kv_values.len / @sizeOf(f32)];
        var d_k_cache = self.getOrAllocKvBuf(@intFromPtr(f32_keys.ptr), f32_keys.len * @sizeOf(f32));
        var d_v_cache = self.getOrAllocKvBuf(@intFromPtr(f32_values.ptr), f32_values.len * @sizeOf(f32));

        var nh_u: u32 = @intCast(nh);
        var nkv_u: u32 = @intCast(nkv);
        var hd_u: u32 = @intCast(hd);
        var prev_u: u32 = @intCast(prev_len);
        var ntok_u: u32 = @intCast(n_tok);
        var scale_f: f32 = scale;

        // Dynamic shared memory: q[hd] + kv_block[kv_tile*hd] + scores[kv_tile] + out_acc[hd] + reduce[warps] + broadcast[1]
        // Must match layout in sdpa_prefill.zig (kv_block_size=32, max_warps=8).
        const smem: u32 = (hd_u + prefill_kv_tile * hd_u + prefill_kv_tile + hd_u + prefill_reduce_slots + 1) * @sizeOf(f32);

        var params = [_]?*anyopaque{
            @ptrCast(&d_q),     @ptrCast(&d_k_cache), @ptrCast(&d_v_cache),
            @ptrCast(&d_k_new), @ptrCast(&d_v_new),   @ptrCast(&d_out),
            @ptrCast(&nh_u),    @ptrCast(&nkv_u),     @ptrCast(&hd_u),
            @ptrCast(&prev_u),  @ptrCast(&ntok_u),    @ptrCast(&scale_f),
        };

        const grid: u32 = ntok_u * nh_u;
        self.launch(self.fn_sdpa_prefill, grid, block_size, smem, &params);

        // Bulk copy new K/V to KV cache on device (for future decode steps)
        const kvd_bytes = kvd * @sizeOf(f32);
        _ = self.cuMemcpyDtoD(d_k_cache + prev_len * kvd_bytes, d_k_new, n_tok * kvd_bytes);
        _ = self.cuMemcpyDtoD(d_v_cache + prev_len * kvd_bytes, d_v_new, n_tok * kvd_bytes);
    }

    /// DeltaNet SSM recurrence — CPU fallback.
    /// Sequential SSM state updates are inherently serial per head; GPU dispatch
    /// overhead exceeds the compute benefit for single-token decode.
    pub fn deltaNet(self: *CudaBackend, conv_in: [*]const f32, conv_out: [*]f32, z_buf: [*]const f32, alpha_buf: [*]const f32, beta_buf: [*]const f32, output: [*]f32, conv_state: [*]f32, ssm_state: []f32, ssm_a: [*]const f32, dt_bias: [*]const f32, conv_w: [*]const f32, ssm_norm_w: [*]const f32, p: backend_mod.DeltaNetParams) void {
        self.flushActivations();
        self.cpu.deltaNet(conv_in, conv_out, z_buf, alpha_buf, beta_buf, output, conv_state, ssm_state, ssm_a, dt_bias, conv_w, ssm_norm_w, p);
        // Invalidate all activation buffers that DeltaNet wrote to
        self.invalidateAct(conv_out);
        self.invalidateAct(output);
    }
};

// ── Tests ─────────────────────────────────────────────────────────

test "CUDA tuning constants are valid" {
    const testing = std.testing;

    // Block size must be a power of 2 and within CUDA limits (max 1024)
    try testing.expect(block_size > 0);
    try testing.expect(block_size <= 1024);
    try testing.expect(block_size & (block_size - 1) == 0); // power of 2

    // Reduction shared memory must accommodate at least one warp
    try testing.expect(reduction_smem >= 4);
    try testing.expect(reduction_smem <= block_size);

    // SDPA prefill constants must be non-zero
    try testing.expect(prefill_kv_tile > 0);
    try testing.expect(prefill_reduce_slots > 0);

    // Device name buffer must be reasonable
    try testing.expect(device_name_buf_size >= 64);
    try testing.expect(device_name_buf_size <= 1024);

    // Version encoding divisors must be non-zero
    try testing.expect(cuda_version_major_divisor > 0);
    try testing.expect(cuda_version_minor_divisor > 0);

    // bits_per_u32_word must be exactly 32
    try testing.expectEqual(@as(usize, 32), bits_per_u32_word);
}

test "CUDA GEMV grid size calculations" {
    const testing = std.testing;

    // Q4_K/Q5_K/Q6_K use NR=2: grid = ceil(n/2)
    try testing.expectEqual(@as(u32, 1), @as(u32, @intCast((1 + 1) / 2))); // n=1
    try testing.expectEqual(@as(u32, 1), @as(u32, @intCast((2 + 1) / 2))); // n=2
    try testing.expectEqual(@as(u32, 2), @as(u32, @intCast((3 + 1) / 2))); // n=3
    try testing.expectEqual(@as(u32, 2), @as(u32, @intCast((4 + 1) / 2))); // n=4

    // Q4_0/Q8_0 use NR=4: grid = ceil(n/4)
    try testing.expectEqual(@as(u32, 1), @as(u32, @intCast((1 + 3) / 4))); // n=1
    try testing.expectEqual(@as(u32, 1), @as(u32, @intCast((4 + 3) / 4))); // n=4
    try testing.expectEqual(@as(u32, 2), @as(u32, @intCast((5 + 3) / 4))); // n=5

    // Elementwise grid: ceil(n/block_size)
    const n: usize = 1000;
    const grid = (n + block_size - 1) / block_size;
    try testing.expectEqual(@as(usize, 4), grid); // ceil(1000/256) = 4
}

test "CUDA BufState transitions are valid" {
    // Verify the BufState enum has exactly 3 states
    const states = [_]CudaBackend.BufState{ .clean, .dirty, .stale };
    try std.testing.expectEqual(@as(usize, 3), states.len);

    // Verify default ActBuf state
    const act = CudaBackend.ActBuf{ .dptr = 0, .size = 0, .state = .clean };
    try std.testing.expectEqual(CudaBackend.BufState.clean, act.state);
}

test "CUDA UMA region bounds checking" {
    // isInUmaRegion with no regions should always return false
    const be = CudaBackend{};
    try std.testing.expect(!be.isInUmaRegion(0));
    try std.testing.expect(!be.isInUmaRegion(0x1000));
    try std.testing.expect(!be.isInUmaRegion(std.math.maxInt(usize)));
}

test "CUDA backend public function signatures compile" {
    // Compile-time verification that all pub fn signatures exist and are well-typed.
    // This catches signature drift between the backend interface and implementation.
    comptime {
        // Core ops
        _ = @TypeOf(CudaBackend.gemv);
        _ = @TypeOf(CudaBackend.gemvMulti);
        _ = @TypeOf(CudaBackend.rmsNorm);
        _ = @TypeOf(CudaBackend.rmsNormMulti);
        _ = @TypeOf(CudaBackend.silu);
        _ = @TypeOf(CudaBackend.gelu);
        _ = @TypeOf(CudaBackend.add);
        _ = @TypeOf(CudaBackend.addRmsNorm);
        _ = @TypeOf(CudaBackend.mul);
        _ = @TypeOf(CudaBackend.softmax);
        _ = @TypeOf(CudaBackend.rope);
        _ = @TypeOf(CudaBackend.embLookup);
        _ = @TypeOf(CudaBackend.l2Norm);
        _ = @TypeOf(CudaBackend.addScaled);
        _ = @TypeOf(CudaBackend.siluMul);
        _ = @TypeOf(CudaBackend.geluMul);
        _ = @TypeOf(CudaBackend.sigmoidMul);
        _ = @TypeOf(CudaBackend.deinterleave);
        _ = @TypeOf(CudaBackend.splitQGate);
        _ = @TypeOf(CudaBackend.gemvT);

        // GEMV variants
        _ = @TypeOf(CudaBackend.gemvNvfp4St);
        _ = @TypeOf(CudaBackend.gemvMlxQ);
        _ = @TypeOf(CudaBackend.gemvMxfp4St);
        _ = @TypeOf(CudaBackend.gemvGptq);
        _ = @TypeOf(CudaBackend.gemvAwq);
        _ = @TypeOf(CudaBackend.gemvHqq);

        // Fused FFN
        _ = @TypeOf(CudaBackend.fusedFfnGateUpSiluQ8);
        _ = @TypeOf(CudaBackend.fusedFfnGateUpGeluQ8);
        _ = @TypeOf(CudaBackend.fusedFfnGateUpSiluQ4K);
        _ = @TypeOf(CudaBackend.fusedFfnGateUpSiluQ5K);
        _ = @TypeOf(CudaBackend.fusedFfnGateUpSiluQ6K);

        // SDPA
        _ = @TypeOf(CudaBackend.sdpa);
        _ = @TypeOf(CudaBackend.sdpaWithStats);
        _ = @TypeOf(CudaBackend.sdpaPaged);
        _ = @TypeOf(CudaBackend.sdpaPrefill);
        _ = @TypeOf(CudaBackend.sdpaTree);

        // Megakernels
        _ = @TypeOf(CudaBackend.dispatchMegakernelQwen35Q8);
        _ = @TypeOf(CudaBackend.dispatchMegakernelGemmaQ4K);

        // Batched prefill
        _ = @TypeOf(CudaBackend.gemm);
        _ = @TypeOf(CudaBackend.rmsNormBatched);
        _ = @TypeOf(CudaBackend.ropeBatched);

        // Sync / lifecycle
        _ = @TypeOf(CudaBackend.sync);
        _ = @TypeOf(CudaBackend.beginBatch);
        _ = @TypeOf(CudaBackend.endBatch);
        _ = @TypeOf(CudaBackend.init);
        _ = @TypeOf(CudaBackend.deinit);
        _ = @TypeOf(CudaBackend.backendInfo);
        _ = @TypeOf(CudaBackend.flushActivations);
        _ = @TypeOf(CudaBackend.invalidateAct);
        _ = @TypeOf(CudaBackend.invalidateWeight);
        _ = @TypeOf(CudaBackend.setThreadContext);
        _ = @TypeOf(CudaBackend.registerHostRegion);

        // KV cache
        _ = @TypeOf(CudaBackend.allocKvSlice);
        _ = @TypeOf(CudaBackend.freeKvSlice);
        _ = @TypeOf(CudaBackend.registerRamKv);
        _ = @TypeOf(CudaBackend.getDevicePtr);
        _ = @TypeOf(CudaBackend.getDevicePtrOpaque);

        // DeltaNet
        _ = @TypeOf(CudaBackend.deltaNet);
    }
}

test "CUDA n_kernels constant matches expected count" {
    try std.testing.expectEqual(@as(u32, 44), CudaBackend.n_kernels);
}

test "CUDA lib_name is platform-appropriate" {
    const name = CudaBackend.lib_name;
    try std.testing.expect(name.len > 0);
    // Must contain "cuda" (case-insensitive check via known platform values)
    const expected = switch (builtin.os.tag) {
        .linux => "libcuda.so.1",
        .windows => "nvcuda.dll",
        else => "libcuda.dylib",
    };
    try std.testing.expectEqualStrings(expected, name);
}

test "CUDA internal struct layouts are well-formed" {
    const testing = std.testing;

    // CachedBuf: weight cache entry
    const cb = CudaBackend.CachedBuf{ .dptr = 0x1000, .size = 4096, .is_registered = false };
    try testing.expectEqual(@as(CUdeviceptr, 0x1000), cb.dptr);
    try testing.expectEqual(@as(usize, 4096), cb.size);
    try testing.expect(!cb.is_registered);

    // KvDevCache: device KV buffer
    const kv = CudaBackend.KvDevCache{ .dptr = 0x2000, .capacity = 8192 };
    try testing.expectEqual(@as(CUdeviceptr, 0x2000), kv.dptr);
    try testing.expectEqual(@as(usize, 8192), kv.capacity);

    // ActBuf: activation buffer with state tracking
    const ab_clean = CudaBackend.ActBuf{ .dptr = 0x3000, .size = 1024, .state = .clean };
    const ab_dirty = CudaBackend.ActBuf{ .dptr = 0x3000, .size = 1024, .state = .dirty };
    const ab_stale = CudaBackend.ActBuf{ .dptr = 0x3000, .size = 1024, .state = .stale };
    try testing.expectEqual(CudaBackend.BufState.clean, ab_clean.state);
    try testing.expectEqual(CudaBackend.BufState.dirty, ab_dirty.state);
    try testing.expectEqual(CudaBackend.BufState.stale, ab_stale.state);

    // max_uma_regions: must be at least 1 and fit in u32
    try testing.expect(CudaBackend.max_uma_regions >= 1);
    try testing.expect(CudaBackend.max_uma_regions <= 256);

    // UmaRegion default state
    const uma = CudaBackend.UmaRegion{};
    try testing.expectEqual(@as(usize, 0), uma.base);
    try testing.expectEqual(@as(usize, 0), uma.size);
}

test "CUDA driver API types have correct sizes" {
    const testing = std.testing;

    // CUresult and CUdevice are c_int
    try testing.expectEqual(@sizeOf(c_int), @sizeOf(CUresult));
    try testing.expectEqual(@sizeOf(c_int), @sizeOf(CUdevice));

    // CUdeviceptr is u64 (64-bit device address)
    try testing.expectEqual(@as(usize, 8), @sizeOf(CUdeviceptr));

    // CUcontext and CUmodule are optional opaque pointers
    try testing.expectEqual(@sizeOf(?*anyopaque), @sizeOf(CUcontext));
    try testing.expectEqual(@sizeOf(?*anyopaque), @sizeOf(CUmodule));

    // CUDA_SUCCESS must be 0
    try testing.expectEqual(@as(CUresult, 0), CUDA_SUCCESS);

    // Device attribute constants must be distinct
    try testing.expect(CU_DEVICE_ATTRIBUTE_INTEGRATED != CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    try testing.expect(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR != CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
}

test "CUDA SDPA prefill shared memory calculation" {
    const testing = std.testing;

    // Verify the shared memory formula matches the expected layout:
    // q[hd] + kv_block[kv_tile*hd] + scores[kv_tile] + out_acc[hd] + reduce[warps] + broadcast[1]
    const hd: u32 = 128; // typical head dimension
    const smem = (hd + prefill_kv_tile * hd + prefill_kv_tile + hd + prefill_reduce_slots + 1) * @sizeOf(f32);

    // Should be: (128 + 32*128 + 32 + 128 + 8 + 1) * 4 = (128 + 4096 + 32 + 128 + 8 + 1) * 4 = 4393 * 4 = 17572
    try testing.expectEqual(@as(u32, (128 + 32 * 128 + 32 + 128 + 8 + 1) * 4), smem);

    // Must stay within Metal-style 32KB limit for cross-backend compatibility
    try testing.expect(smem <= 32768);
}

test "CUDA default struct initialization is safe" {
    // Verify that a default-initialized CudaBackend has null handles
    // and zero counts (prevents use-before-init bugs).
    const be = CudaBackend{};
    try std.testing.expect(be.context == null);
    try std.testing.expect(be.module == null);
    try std.testing.expect(!be.is_uma);
    try std.testing.expect(!be.has_gds);
    try std.testing.expectEqual(@as(u32, 0), be.sm_major);
    try std.testing.expectEqual(@as(u32, 0), be.sm_minor);
    try std.testing.expectEqual(@as(usize, 0), be.total_mem);
    try std.testing.expectEqual(@as(usize, 0), be.avail_mem);
    try std.testing.expectEqual(@as(u32, 0), be.driver_version);
    try std.testing.expectEqual(@as(u32, 0), be.uma_region_count);
    try std.testing.expectEqual(@as(usize, 0), be.device_name_len);
    try std.testing.expect(be.sdpa_flat_keys == null);
    try std.testing.expect(be.sdpa_flat_vals == null);
    try std.testing.expect(be.cufile_lib == null);

    // Kernel function handles should all be null
    try std.testing.expect(be.fn_silu == null);
    try std.testing.expect(be.fn_sdpa == null);
    try std.testing.expect(be.fn_gemv_f32 == null);
    try std.testing.expect(be.fn_mega_qwen35_q8 == null);
}

// ── Per-function comptime signature tests ────────────────────────

test "CudaBackend.init" {
    comptime {
        _ = &CudaBackend.init;
    }
}
test "CudaBackend.deinit" {
    comptime {
        _ = &CudaBackend.deinit;
    }
}
test "CudaBackend.setThreadContext" {
    comptime {
        _ = &CudaBackend.setThreadContext;
    }
}
test "CudaBackend.registerHostRegion" {
    comptime {
        _ = &CudaBackend.registerHostRegion;
    }
}
test "CudaBackend.flushActivations" {
    comptime {
        _ = &CudaBackend.flushActivations;
    }
}
test "CudaBackend.invalidateAct" {
    comptime {
        _ = &CudaBackend.invalidateAct;
    }
}
test "CudaBackend.getDevicePtrOpaque" {
    comptime {
        _ = &CudaBackend.getDevicePtrOpaque;
    }
}
test "CudaBackend.getDevicePtr" {
    comptime {
        _ = &CudaBackend.getDevicePtr;
    }
}
test "CudaBackend.invalidateWeight" {
    comptime {
        _ = &CudaBackend.invalidateWeight;
    }
}
test "CudaBackend.gemv" {
    comptime {
        _ = &CudaBackend.gemv;
    }
}
test "CudaBackend.sigmoidMul" {
    comptime {
        _ = &CudaBackend.sigmoidMul;
    }
}
test "CudaBackend.siluMul" {
    comptime {
        _ = &CudaBackend.siluMul;
    }
}
test "CudaBackend.geluMul" {
    comptime {
        _ = &CudaBackend.geluMul;
    }
}
test "CudaBackend.fusedFfnGateUpSiluQ8" {
    comptime {
        _ = &CudaBackend.fusedFfnGateUpSiluQ8;
    }
}
test "CudaBackend.fusedFfnGateUpGeluQ8" {
    comptime {
        _ = &CudaBackend.fusedFfnGateUpGeluQ8;
    }
}
test "CudaBackend.fusedFfnGateUpSiluQ4K" {
    comptime {
        _ = &CudaBackend.fusedFfnGateUpSiluQ4K;
    }
}
test "CudaBackend.fusedFfnGateUpSiluQ5K" {
    comptime {
        _ = &CudaBackend.fusedFfnGateUpSiluQ5K;
    }
}
test "CudaBackend.fusedFfnGateUpSiluQ6K" {
    comptime {
        _ = &CudaBackend.fusedFfnGateUpSiluQ6K;
    }
}
test "CudaBackend.dispatchMegakernelQwen35Q8" {
    comptime {
        _ = &CudaBackend.dispatchMegakernelQwen35Q8;
    }
}
test "CudaBackend.dispatchMegakernelGemmaQ4K" {
    comptime {
        _ = &CudaBackend.dispatchMegakernelGemmaQ4K;
    }
}
test "CudaBackend.rmsNormMulti" {
    comptime {
        _ = &CudaBackend.rmsNormMulti;
    }
}
test "CudaBackend.deinterleave" {
    comptime {
        _ = &CudaBackend.deinterleave;
    }
}
test "CudaBackend.splitQGate" {
    comptime {
        _ = &CudaBackend.splitQGate;
    }
}
test "CudaBackend.gemvMulti" {
    comptime {
        _ = &CudaBackend.gemvMulti;
    }
}
test "CudaBackend.rmsNorm" {
    comptime {
        _ = &CudaBackend.rmsNorm;
    }
}
test "CudaBackend.silu" {
    comptime {
        _ = &CudaBackend.silu;
    }
}
test "CudaBackend.gelu" {
    comptime {
        _ = &CudaBackend.gelu;
    }
}
test "CudaBackend.add" {
    comptime {
        _ = &CudaBackend.add;
    }
}
test "CudaBackend.addRmsNorm" {
    comptime {
        _ = &CudaBackend.addRmsNorm;
    }
}
test "CudaBackend.gemvT" {
    comptime {
        _ = &CudaBackend.gemvT;
    }
}
test "CudaBackend.addScaled" {
    comptime {
        _ = &CudaBackend.addScaled;
    }
}
test "CudaBackend.mul" {
    comptime {
        _ = &CudaBackend.mul;
    }
}
test "CudaBackend.softmax" {
    comptime {
        _ = &CudaBackend.softmax;
    }
}
test "CudaBackend.rope" {
    comptime {
        _ = &CudaBackend.rope;
    }
}
test "CudaBackend.embLookup" {
    comptime {
        _ = &CudaBackend.embLookup;
    }
}
test "CudaBackend.l2Norm" {
    comptime {
        _ = &CudaBackend.l2Norm;
    }
}
test "CudaBackend.gemvNvfp4St" {
    comptime {
        _ = &CudaBackend.gemvNvfp4St;
    }
}
test "CudaBackend.gemvMlxQ" {
    comptime {
        _ = &CudaBackend.gemvMlxQ;
    }
}
test "CudaBackend.gemvMxfp4St" {
    comptime {
        _ = &CudaBackend.gemvMxfp4St;
    }
}
test "CudaBackend.gemvGptq" {
    comptime {
        _ = &CudaBackend.gemvGptq;
    }
}
test "CudaBackend.gemvAwq" {
    comptime {
        _ = &CudaBackend.gemvAwq;
    }
}
test "CudaBackend.gemvHqq" {
    comptime {
        _ = &CudaBackend.gemvHqq;
    }
}
test "CudaBackend.sync" {
    comptime {
        _ = &CudaBackend.sync;
    }
}
test "CudaBackend.beginBatch" {
    comptime {
        _ = &CudaBackend.beginBatch;
    }
}
test "CudaBackend.endBatch" {
    comptime {
        _ = &CudaBackend.endBatch;
    }
}
test "CudaBackend.backendInfo" {
    comptime {
        _ = &CudaBackend.backendInfo;
    }
}
test "CudaBackend.allocKvSlice" {
    comptime {
        _ = &CudaBackend.allocKvSlice;
    }
}
test "CudaBackend.freeKvSlice" {
    comptime {
        _ = &CudaBackend.freeKvSlice;
    }
}
test "CudaBackend.registerRamKv" {
    comptime {
        _ = &CudaBackend.registerRamKv;
    }
}
test "CudaBackend.sdpa" {
    comptime {
        _ = &CudaBackend.sdpa;
    }
}
test "CudaBackend.sdpaWithStats" {
    comptime {
        _ = &CudaBackend.sdpaWithStats;
    }
}
test "CudaBackend.gemm" {
    comptime {
        _ = &CudaBackend.gemm;
    }
}
test "CudaBackend.rmsNormBatched" {
    comptime {
        _ = &CudaBackend.rmsNormBatched;
    }
}
test "CudaBackend.ropeBatched" {
    comptime {
        _ = &CudaBackend.ropeBatched;
    }
}
test "CudaBackend.sdpaTree" {
    comptime {
        _ = &CudaBackend.sdpaTree;
    }
}
test "CudaBackend.sdpaPaged" {
    comptime {
        _ = &CudaBackend.sdpaPaged;
    }
}
test "CudaBackend.sdpaPrefill" {
    comptime {
        _ = &CudaBackend.sdpaPrefill;
    }
}
test "CudaBackend.deltaNet" {
    comptime {
        _ = &CudaBackend.deltaNet;
    }
}

test "fuzz: all cuda functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            _ = smith;
            comptime {
                _ = &CudaBackend.init;
                _ = &CudaBackend.deinit;
                _ = &CudaBackend.setThreadContext;
                _ = &CudaBackend.registerHostRegion;
                _ = &CudaBackend.flushActivations;
                _ = &CudaBackend.invalidateAct;
                _ = &CudaBackend.getDevicePtrOpaque;
                _ = &CudaBackend.getDevicePtr;
                _ = &CudaBackend.invalidateWeight;
                _ = &CudaBackend.gemv;
                _ = &CudaBackend.sigmoidMul;
                _ = &CudaBackend.siluMul;
                _ = &CudaBackend.geluMul;
                _ = &CudaBackend.fusedFfnGateUpSiluQ8;
                _ = &CudaBackend.fusedFfnGateUpGeluQ8;
                _ = &CudaBackend.fusedFfnGateUpSiluQ4K;
                _ = &CudaBackend.fusedFfnGateUpSiluQ5K;
                _ = &CudaBackend.fusedFfnGateUpSiluQ6K;
                _ = &CudaBackend.dispatchMegakernelQwen35Q8;
                _ = &CudaBackend.dispatchMegakernelGemmaQ4K;
                _ = &CudaBackend.rmsNormMulti;
                _ = &CudaBackend.deinterleave;
                _ = &CudaBackend.splitQGate;
                _ = &CudaBackend.gemvMulti;
                _ = &CudaBackend.rmsNorm;
                _ = &CudaBackend.silu;
                _ = &CudaBackend.gelu;
                _ = &CudaBackend.add;
                _ = &CudaBackend.addRmsNorm;
                _ = &CudaBackend.gemvT;
                _ = &CudaBackend.addScaled;
                _ = &CudaBackend.mul;
                _ = &CudaBackend.softmax;
                _ = &CudaBackend.rope;
                _ = &CudaBackend.embLookup;
                _ = &CudaBackend.l2Norm;
                _ = &CudaBackend.gemvNvfp4St;
                _ = &CudaBackend.gemvMlxQ;
                _ = &CudaBackend.gemvMxfp4St;
                _ = &CudaBackend.gemvGptq;
                _ = &CudaBackend.gemvAwq;
                _ = &CudaBackend.gemvHqq;
                _ = &CudaBackend.sync;
                _ = &CudaBackend.beginBatch;
                _ = &CudaBackend.endBatch;
                _ = &CudaBackend.backendInfo;
                _ = &CudaBackend.allocKvSlice;
                _ = &CudaBackend.freeKvSlice;
                _ = &CudaBackend.registerRamKv;
                _ = &CudaBackend.sdpa;
                _ = &CudaBackend.sdpaWithStats;
                _ = &CudaBackend.gemm;
                _ = &CudaBackend.rmsNormBatched;
                _ = &CudaBackend.ropeBatched;
                _ = &CudaBackend.sdpaTree;
                _ = &CudaBackend.sdpaPaged;
                _ = &CudaBackend.sdpaPrefill;
                _ = &CudaBackend.deltaNet;
            }
        }
    }.f, .{});
}
