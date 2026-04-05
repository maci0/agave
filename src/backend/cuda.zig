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
const DType = backend_mod.DType;
const CpuBackend = @import("cpu.zig").CpuBackend;
const KvQuantType = backend_mod.KvQuantType;
const kv_quant = @import("../ops/kv_quant.zig");

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

/// TurboQuant 2-bit block: f16 norm (2 bytes) + 8 packed bytes = 10 bytes per 32 elements.
const turbo2_block_bytes: u32 = 10;
/// TurboQuant 3-bit block: f16 norm (2 bytes) + 12 packed bytes = 14 bytes per 32 elements.
const turbo3_block_bytes: u32 = 14;
/// TurboQuant 4-bit block: f16 norm (2 bytes) + 16 packed bytes = 18 bytes per 32 elements.
const turbo4_block_bytes: u32 = 18;

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
const FnCtxSync = *const fn () callconv(.c) CUresult;
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

    // Kernel function handles
    fn_silu: CUfunction = null,
    fn_gelu: CUfunction = null,
    fn_add: CUfunction = null,
    fn_mul: CUfunction = null,
    fn_rms_norm: CUfunction = null,
    fn_add_rms_norm: CUfunction = null,
    fn_softmax: CUfunction = null,
    fn_l2_norm: CUfunction = null,
    fn_rope: CUfunction = null,
    fn_gemv_f32: CUfunction = null,
    fn_gemv_bf16: CUfunction = null,
    fn_gemv_f16: CUfunction = null,
    fn_gemv_q8_0: CUfunction = null,
    fn_gemv_q4_0: CUfunction = null,
    fn_gemv_q4_1: CUfunction = null,
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
    fn_gemv_t_q8_0: CUfunction = null,
    fn_gemv_nvfp4_st: CUfunction = null,
    fn_gemv_mlx_q4: CUfunction = null,
    fn_gemv_mlx_q6: CUfunction = null,
    fn_gemv_mlx_q8: CUfunction = null,
    fn_gemv_mxfp4_st: CUfunction = null,
    fn_sdpa: CUfunction = null,
    fn_sdpa_turbo: CUfunction = null,
    fn_sdpa_prefill: CUfunction = null,
    fn_gemm_q8_0: CUfunction = null,
    fn_rms_norm_batched: CUfunction = null,
    fn_rope_batched: CUfunction = null,

    /// CPU backend for ops where CPU is genuinely faster than GPU dispatch (embLookup).
    cpu: CpuBackend = .{},

    /// Whether the GPU uses unified memory architecture (integrated GPU).
    is_uma: bool = false,

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

    /// Number of PTX kernels loaded at init.
    pub const n_kernels: u32 = 38;

    /// Library name loaded via dlopen at init.
    pub const lib_name = cuda_lib_name;

    const CachedBuf = struct {
        dptr: CUdeviceptr,
        size: usize,
        /// True if this entry was registered via cuMemHostRegister (UMA).
        /// Cleanup uses cuMemHostUnregister instead of cuMemFree.
        is_registered: bool = false,
    };

    /// Device-side KV cache buffer, tracking how many positions have been uploaded.
    const KvDevCache = struct {
        dptr: CUdeviceptr,
        capacity: usize,
        uploaded_sl: usize,
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
    pub fn init(allocator: std.mem.Allocator) !CudaBackend {
        var self = CudaBackend{};
        self.allocator = allocator;
        self.buf_cache = std.AutoHashMap(usize, CachedBuf).init(allocator);
        try self.buf_cache.ensureTotalCapacity(backend_mod.buf_cache_initial_capacity);
        errdefer self.buf_cache.deinit();
        self.act_cache = std.AutoHashMap(usize, ActBuf).init(allocator);
        errdefer self.act_cache.deinit();
        self.kv_dev_cache = std.AutoHashMap(usize, KvDevCache).init(allocator);
        errdefer self.kv_dev_cache.deinit();

        // Dynamically load libcuda
        self.lib = std.DynLib.open(cuda_lib_name) catch return error.CudaNotAvailable;
        errdefer self.lib.close();

        // Resolve all function pointers
        const cuInit = self.lookup(FnInit, "cuInit") orelse return error.CudaNotAvailable;
        const cuDeviceGet = self.lookup(FnDeviceGet, "cuDeviceGet") orelse return error.CudaNotAvailable;
        const cuDeviceGetName = self.lookup(FnDeviceGetName, "cuDeviceGetName") orelse return error.CudaNotAvailable;
        const cuDeviceGetAttribute = self.lookup(FnDeviceGetAttribute, "cuDeviceGetAttribute") orelse return error.CudaNotAvailable;
        const cuCtxCreate = self.lookup(FnCtxCreate, "cuCtxCreate_v2") orelse return error.CudaNotAvailable;
        self.cuCtxDestroy = self.lookup(FnCtxDestroy, "cuCtxDestroy_v2") orelse return error.CudaNotAvailable;
        self.cuCtxSynchronize = self.lookup(FnCtxSync, "cuCtxSynchronize") orelse return error.CudaNotAvailable;
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
        if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return error.NoCudaDevice;

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
            _ = std.fmt.bufPrint(&self.cc_str, "sm_{d}{d}", .{ self.sm_major, self.sm_minor }) catch {};
        }
        if (self.driver_version > 0) {
            _ = std.fmt.bufPrint(&self.drv_str, "CUDA {d}.{d}", .{ self.driver_version / cuda_version_major_divisor, (self.driver_version % cuda_version_major_divisor) / cuda_version_minor_divisor }) catch {};
        }

        if (cuCtxCreate(&self.context, 0, dev) != CUDA_SUCCESS) return error.CudaInitFailed;
        errdefer _ = self.cuCtxDestroy(self.context);

        // Load PTX module — must be null-terminated
        var ptx_buf: [ptx_source.len + 1]u8 = undefined;
        @memcpy(ptx_buf[0..ptx_source.len], ptx_source);
        ptx_buf[ptx_source.len] = 0;

        if (cuModuleLoadData(&self.module, &ptx_buf) != CUDA_SUCCESS) return error.PtxLoadFailed;
        errdefer _ = self.cuModuleUnload(self.module);

        // Get kernel function handles
        self.fn_silu = try self.getFunction("silu_kernel");
        self.fn_gelu = try self.getFunction("gelu_kernel");
        self.fn_add = try self.getFunction("add_kernel");
        self.fn_mul = try self.getFunction("mul_kernel");
        self.fn_rms_norm = try self.getFunction("rms_norm_kernel");
        self.fn_add_rms_norm = try self.getFunction("add_rms_norm_kernel");
        self.fn_softmax = try self.getFunction("softmax_kernel");
        self.fn_l2_norm = try self.getFunction("l2_norm_kernel");
        self.fn_rope = try self.getFunction("rope_kernel");
        self.fn_gemv_f32 = try self.getFunction("gemv_f32_kernel");
        self.fn_gemv_bf16 = try self.getFunction("gemv_bf16_kernel");
        self.fn_gemv_f16 = try self.getFunction("gemv_f16_kernel");
        self.fn_gemv_q8_0 = try self.getFunction("gemv_q8_0_kernel");
        self.fn_gemv_q4_0 = try self.getFunction("gemv_q4_0_kernel");
        self.fn_gemv_q4_1 = try self.getFunction("gemv_q4_1_kernel");
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
        self.fn_gemv_t_q8_0 = try self.getFunction("gemv_t_q8_0_kernel");
        self.fn_gemv_nvfp4_st = try self.getFunction("gemv_nvfp4_st_kernel");
        self.fn_gemv_mlx_q4 = try self.getFunction("gemv_mlx_q4_kernel");
        self.fn_gemv_mlx_q6 = try self.getFunction("gemv_mlx_q6_kernel");
        self.fn_gemv_mlx_q8 = try self.getFunction("gemv_mlx_q8_kernel");
        self.fn_gemv_mxfp4_st = try self.getFunction("gemv_mxfp4_st_kernel");
        self.fn_sdpa = try self.getFunction("sdpa_kernel");
        self.fn_sdpa_turbo = try self.getFunction("sdpa_turbo_kernel");
        self.fn_sdpa_prefill = try self.getFunction("sdpa_prefill_kernel");
        self.fn_gemm_q8_0 = try self.getFunction("gemm_q8_0_kernel");
        self.fn_rms_norm_batched = try self.getFunction("rms_norm_batched_kernel");
        self.fn_rope_batched = try self.getFunction("rope_batched_kernel");

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

        if (self.module != null) _ = self.cuModuleUnload(self.module);
        if (self.context != null) {
            _ = self.cuCtxSynchronize();
            _ = self.cuCtxDestroy(self.context);
        }
        self.lib.close();
    }

    fn lookup(self: *CudaBackend, comptime T: type, name: [:0]const u8) ?T {
        return self.lib.lookup(T, name);
    }

    fn getFunction(self: *CudaBackend, name: [*:0]const u8) !CUfunction {
        var func: CUfunction = null;
        if (self.cuModuleGetFunction(&func, self.module, name) != CUDA_SUCCESS)
            return error.KernelNotFound;
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

    // ── Weight cache (permanent, read-only) ─────────────────────

    /// Get device pointer for a weight buffer. Uploads once, reused forever.
    /// On UMA: registers host memory for GPU access (zero-copy, no allocation).
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

        if (self.is_uma) {
            if (self.cuMemHostRegister) |reg| {
                if (self.cuMemHostGetDevicePointer) |getDevPtr| {
                    // Pin host mmap'd memory for GPU access — zero copy.
                    // cuMemHostRegister requires page-aligned pointers (mmap'd weights are page-aligned).
                    std.debug.assert(std.mem.isAligned(@intFromPtr(ptr), std.heap.page_size_min));
                    if (reg(@ptrCast(ptr), size, CU_MEMHOSTREGISTER_DEVICEMAP) == CUDA_SUCCESS) {
                        var dptr: CUdeviceptr = 0;
                        if (getDevPtr(&dptr, @ptrCast(ptr), 0) == CUDA_SUCCESS) {
                            self.buf_cache.put(addr, .{ .dptr = dptr, .size = size, .is_registered = true }) catch |err| {
                                std.log.warn("cache put failed: {}", .{err});
                            };
                            return dptr;
                        }
                    }
                    // Registration failed — fall through to upload path
                    std.log.warn("UMA cuMemHostRegister failed for {x}, falling back to upload", .{addr});
                }
            }
        }

        const dptr = self.uploadToDevice(@ptrCast(ptr), size);
        self.buf_cache.put(addr, .{ .dptr = dptr, .size = size, .is_registered = false }) catch |err| {
            std.log.warn("cache put failed: {}", .{err});
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
            std.log.warn("cache put failed: {}", .{err});
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
            std.log.warn("cache put failed: {}", .{err});
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
            std.log.warn("cache put failed: {}", .{err});
        };
        return dptr;
    }

    /// Sync GPU, download dirty buffers to host, then mark all entries stale.
    /// Called before CPU code that may read or modify activation buffers.
    fn flushActivations(self: *CudaBackend) void {
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
    fn invalidateAct(self: *CudaBackend, ptr: anytype) void {
        const addr = @intFromPtr(ptr);
        if (self.act_cache.fetchRemove(addr)) |kv| {
            _ = self.cuMemFree(kv.value.dptr);
        }
    }

    // ── Launch helper ───────────────────────────────────────────

    fn launch(self: *CudaBackend, func: CUfunction, grid: u32, block: u32, smem: u32, params: [*]?*anyopaque) void {
        _ = self.cuLaunchKernel(func, grid, 1, 1, block, 1, 1, smem, null, params, null);
    }

    // ── Weight size helper ──────────────────────────────────────

    const weightBytes = @import("backend.zig").weightBytes;

    // ── Backend interface ────────────────────────────────────────

    /// y[n] = W[n,k] @ x[k]. GPU kernels for F32, BF16, F16, Q8_0, Q4_0,
    /// Q4_K, Q5_K, Q6_K, FP8_E4M3, FP8_E5M2; unsupported dtypes panic.
    pub fn gemv(self: *CudaBackend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        const func = switch (w.dtype) {
            .f32 => self.fn_gemv_f32,
            .bf16 => self.fn_gemv_bf16,
            .f16 => self.fn_gemv_f16,
            .q8_0 => self.fn_gemv_q8_0,
            .q4_0 => self.fn_gemv_q4_0,
            .q4_1 => self.fn_gemv_q4_1,
            .q4_k => self.fn_gemv_q4_k,
            .q5_k => self.fn_gemv_q5_k,
            .q6_k => self.fn_gemv_q6_k,
            .fp8_e4m3 => self.fn_gemv_fp8_e4m3,
            .fp8_e5m2 => self.fn_gemv_fp8_e5m2,
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
        self.launch(func, @intCast(n), block_size, reduction_smem, &params);
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
            @ptrCast(&d_data), @ptrCast(&d_w), @ptrCast(&d_data),
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
            @ptrCast(&d_in), @ptrCast(&d_a), @ptrCast(&d_b),
            @ptrCast(&stride_u32), @ptrCast(&n_pairs_u32),
        };
        const grid: u32 = @intCast((total + block_size - 1) / block_size);
        self.launch(self.fn_deinterleave, grid, block_size, 0, &params);
    }

    /// Split concatenated Q+gate per-head data into separate arrays.
    pub fn splitQGate(_: *CudaBackend, _: [*]const f32, _: [*]f32, _: [*]f32, _: usize, _: usize) void {
        @panic("CUDA splitQGate: no GPU kernel — add a CUDA kernel");
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
            @ptrCast(&d_w0), @ptrCast(&d_y0), @ptrCast(&n0),
            @ptrCast(&d_w1), @ptrCast(&d_y1), @ptrCast(&n1),
            @ptrCast(&d_w2), @ptrCast(&d_y2), @ptrCast(&n2),
            @ptrCast(&d_w3), @ptrCast(&d_y3), @ptrCast(&n3),
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
            @ptrCast(&d_a), @ptrCast(&d_b), @ptrCast(&d_w),
            @ptrCast(&d_out), @ptrCast(&n_u32), @ptrCast(&eps_f32),
        };
        self.launch(self.fn_add_rms_norm, 1, block_size, reduction_smem, &params);
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
            @ptrCast(&d_x), @ptrCast(&d_w), @ptrCast(&d_y),
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
            @ptrCast(&d_src), @ptrCast(&d_dst),
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
            @ptrCast(&d_x), @ptrCast(&d_w), @ptrCast(&d_s),
            @ptrCast(&d_y), @ptrCast(&n_u32), @ptrCast(&k_u32),
        };
        self.launch(self.fn_gemv_nvfp4_st, @intCast(n), block_size, reduction_smem, &params);
    }

    /// MLX affine quantized GEMV: packed int (4/6/8-bit) + BF16 scales/biases, group_size=64.
    pub fn gemvMlxQ(self: *CudaBackend, x: [*]const f32, weight: [*]const u8, scales: [*]const u8, biases: [*]const u8, y: [*]f32, n: usize, k: usize, bits: u32) void {
        const mlx_group_size: usize = 64;
        const gpr = (k + mlx_group_size - 1) / mlx_group_size;
        const wpg: usize = mlx_group_size * bits / 32;
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
            @ptrCast(&d_x), @ptrCast(&d_w), @ptrCast(&d_s),
            @ptrCast(&d_b), @ptrCast(&d_y), @ptrCast(&n_u32),
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
            @ptrCast(&d_x), @ptrCast(&d_w), @ptrCast(&d_s),
            @ptrCast(&d_y), @ptrCast(&n_u32), @ptrCast(&k_u32),
        };
        self.launch(self.fn_gemv_mxfp4_st, @intCast(n), block_size, reduction_smem, &params);
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

    /// Returns backend startup information for display.
    pub fn backendInfo(self: *const CudaBackend) @import("backend.zig").BackendInfo {
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

    /// Allocate a KV cache slice.
    /// On UMA (integrated GPU): uses cuMemAllocManaged so both CPU and GPU
    /// can access the same pointer directly — no copies needed.
    /// On discrete GPU: falls back to host allocator; the caller manages
    /// VRAM mirroring separately via kv_dev_cache during SDPA.
    pub fn allocKvSlice(_: *CudaBackend, allocator: std.mem.Allocator, n: usize) error{OutOfMemory}![]u8 {
        // Use host allocator — cuMemAllocManaged on UMA (GB10) returns pointers
        // that cause data corruption when used as both host and device memory.
        // TODO: investigate UMA zero-copy KV cache after fixing correctness.
        return allocator.alloc(u8, n);
    }

    /// Free a KV cache slice allocated via allocKvSlice.
    /// Detects whether the slice was allocated via cuMemAllocManaged (UMA)
    /// or the host allocator, and frees accordingly.
    pub fn freeKvSlice(_: *CudaBackend, allocator: std.mem.Allocator, slice: []u8) void {
        if (slice.len == 0) return;
        allocator.free(slice);
    }

    /// Register RAM-tier KV block in act_cache without upload.
    /// On UMA platforms (GB10 Blackwell), host memory is GPU-accessible via
    /// unified addressing — cuMemAllocManaged provides unified pointers that
    /// work on both CPU and GPU. No copy needed.
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
    fn getOrAllocKvBuf(self: *CudaBackend, addr: usize, capacity: usize) CUdeviceptr {
        if (self.kv_dev_cache.getPtr(addr)) |kv| return kv.dptr;

        var dptr: CUdeviceptr = 0;
        _ = self.cuMemAlloc(&dptr, @max(capacity, 4));
        self.kv_dev_cache.put(addr, .{
            .dptr = dptr,
            .capacity = capacity,
            .uploaded_sl = 0,
        }) catch |err| {
            std.log.warn("cache put failed: {}", .{err});
        };
        return dptr;
    }

    /// Return the turbo bit width for a KV quant type, or 0 for non-turbo types.
    fn turboBits(kv_type: KvQuantType) u32 {
        return switch (kv_type) {
            .turbo2 => 2,
            .turbo3 => 3,
            .turbo4 => 4,
            else => 0,
        };
    }

    /// Return the byte size per 32-element turbo block, or 0 for non-turbo types.
    fn turboBlockByteSize(kv_type: KvQuantType) u32 {
        return switch (kv_type) {
            .turbo2 => turbo2_block_bytes,
            .turbo3 => turbo3_block_bytes,
            .turbo4 => turbo4_block_bytes,
            else => 0,
        };
    }

    /// Check if a KV quant type is a TurboQuant variant.
    fn isTurbo(kv_type: KvQuantType) bool {
        return kv_type == .turbo2 or kv_type == .turbo3 or kv_type == .turbo4;
    }

    /// Fused scaled dot-product attention on GPU with KV cache append.
    /// Supports f32 KV cache (existing fast path) and TurboQuant 2/3/4-bit
    /// KV cache (native GPU dequant — no CPU fallback for SDPA compute).
    /// KV append for turbo types uses CPU quantization (once per token per layer,
    /// not the SDPA hot path). Non-turbo quantized types (q8_0, etc.) panic.
    pub fn sdpa(self: *CudaBackend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type_k: @import("backend.zig").KvQuantType, kv_type_v: @import("backend.zig").KvQuantType) void {
        const is_turbo_k = isTurbo(kv_type_k);
        const is_turbo_v = isTurbo(kv_type_v);
        const is_f32_k = (kv_type_k == .f32);
        const is_f32_v = (kv_type_v == .f32);

        // Non-turbo, non-f32 quantized KV: not yet supported
        if (!is_f32_k and !is_turbo_k or !is_f32_v and !is_turbo_v)
            @panic("CUDA SDPA: unsupported KV quant type — use f32 or turbo2/3/4");

        const kvd = nkv * hd;
        var sl: u32 = @intCast(seq_len + 1);

        if (is_f32_k and is_f32_v) {
            // Pure f32 path: use original sdpa_kernel
            const f32_keys: []f32 = @as([*]f32, @ptrCast(@alignCast(keys.ptr)))[0 .. keys.len / 4];
            const f32_values: []f32 = @as([*]f32, @ptrCast(@alignCast(values.ptr)))[0 .. values.len / 4];
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
                @ptrCast(&d_q),      @ptrCast(&d_keys),  @ptrCast(&d_vals),
                @ptrCast(&d_out),    @ptrCast(&nh_u32),   @ptrCast(&nkv_u32),
                @ptrCast(&hd_u32),   @ptrCast(&sl),       @ptrCast(&kvd_u32),
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

            // Upload entire KV cache to device for GPU SDPA
            const k_cache_bytes = kv_quant.kvSliceBytes(kv_type_k, @as(usize, sl) * kvd);
            const v_cache_bytes = kv_quant.kvSliceBytes(kv_type_v, @as(usize, sl) * kvd);
            var d_keys = self.getOrAllocKvBuf(@intFromPtr(keys.ptr), keys.len);
            var d_vals = self.getOrAllocKvBuf(@intFromPtr(values.ptr), values.len);

            // Re-upload the full cache (turbo data was written by CPU)
            _ = self.cuMemcpyHtoD(d_keys, @ptrCast(keys.ptr), k_cache_bytes);
            _ = self.cuMemcpyHtoD(d_vals, @ptrCast(values.ptr), v_cache_bytes);

            var d_q = self.getInputBuf(q, nh * hd * @sizeOf(f32));
            var d_out = self.getOutputBuf(output, nh * hd * @sizeOf(f32));

            var nh_u32: u32 = @intCast(nh);
            var nkv_u32: u32 = @intCast(nkv);
            var hd_u32: u32 = @intCast(hd);
            var kvd_u32: u32 = @intCast(kvd);
            var scale_f32: f32 = scale;
            var bits_k_u: u32 = turboBits(kv_type_k);
            var bits_v_u: u32 = turboBits(kv_type_v);
            var bb_k_u: u32 = turboBlockByteSize(kv_type_k);
            var bb_v_u: u32 = turboBlockByteSize(kv_type_v);

            var params = [_]?*anyopaque{
                @ptrCast(&d_q),       @ptrCast(&d_keys),   @ptrCast(&d_vals),
                @ptrCast(&d_out),     @ptrCast(&nh_u32),    @ptrCast(&nkv_u32),
                @ptrCast(&hd_u32),    @ptrCast(&sl),         @ptrCast(&kvd_u32),
                @ptrCast(&scale_f32), @ptrCast(&bits_k_u),   @ptrCast(&bits_v_u),
                @ptrCast(&bb_k_u),    @ptrCast(&bb_v_u),
            };

            const smem: u32 = (sl + 1) * @sizeOf(f32);
            self.launch(self.fn_sdpa_turbo, @intCast(nh), block_size, smem, &params);
        }
    }

    /// SDPA with per-head softmax stats for split-attention merge.
    /// GPU stats export not yet implemented — syncs GPU, then runs CPU-side
    /// sdpaQuantHeadsWithStats as fallback. Native GPU stats is future work.
    pub fn sdpaWithStats(self: *CudaBackend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, head_max: [*]f32, head_sum: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        const sdpa_cpu = @import("kernels/cpu/sdpa.zig");
        self.sync();
        const kvd = nkv * hd;
        const k_off = kv_quant.kvByteOffset(kv_type_k, seq_len * kvd);
        const v_off = kv_quant.kvByteOffset(kv_type_v, seq_len * kvd);
        kv_quant.kvStore(keys.ptr + k_off, k_new, kvd, kv_type_k);
        kv_quant.kvStore(values.ptr + v_off, v_new, kvd, kv_type_v);
        sdpa_cpu.sdpaQuantHeadsWithStats(q, keys.ptr, values.ptr, output, nh, nkv, hd, seq_len + 1, scale, kv_type_k, kv_type_v, head_max, head_sum);
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
                @ptrCast(&d_x), @ptrCast(&d_w), @ptrCast(&d_y),
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
            @ptrCast(&d_in), @ptrCast(&d_w), @ptrCast(&d_out),
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
            @ptrCast(&d_x), @ptrCast(&d_pos),
            @ptrCast(&n_tok_u), @ptrCast(&nh_u), @ptrCast(&hd_u),
            @ptrCast(&rd_u), @ptrCast(&theta_v),
        };
        const grid = @as(u32, @intCast((total + block_size - 1) / block_size));
        self.launch(self.fn_rope_batched, grid, block_size, 0, &params);
    }

    /// Prefill SDPA — FlashAttention-2 with causal masking.
    /// Attends to both cached KV (prev_len positions) and new KV (n_tok positions).
    /// For f32 KV: native FA2 GPU kernel (single dispatch for all tokens).
    /// For turbo KV: CPU-side KV append + sequential GPU turbo SDPA per token.
    /// For other quantized KV types: panics (not yet supported).
    pub fn sdpaPrefill(self: *CudaBackend, q: [*]const f32, k: [*]const f32, v: [*]const f32, kv_keys: []u8, kv_values: []u8, output: [*]f32, nh: usize, nkv: usize, hd: usize, prev_len: usize, n_tok: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        const is_turbo_k = isTurbo(kv_type_k);
        const is_turbo_v = isTurbo(kv_type_v);
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

            // Upload KV cache to device
            const total_sl = prev_len + n_tok;
            const k_cache_bytes = kv_quant.kvSliceBytes(kv_type_k, total_sl * kvd);
            const v_cache_bytes = kv_quant.kvSliceBytes(kv_type_v, total_sl * kvd);
            var d_keys = self.getOrAllocKvBuf(@intFromPtr(kv_keys.ptr), kv_keys.len);
            var d_vals = self.getOrAllocKvBuf(@intFromPtr(kv_values.ptr), kv_values.len);
            _ = self.cuMemcpyHtoD(d_keys, @ptrCast(kv_keys.ptr), k_cache_bytes);
            _ = self.cuMemcpyHtoD(d_vals, @ptrCast(kv_values.ptr), v_cache_bytes);

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
                var bits_k_u: u32 = turboBits(kv_type_k);
                var bits_v_u: u32 = turboBits(kv_type_v);
                var bb_k_u: u32 = turboBlockByteSize(kv_type_k);
                var bb_v_u: u32 = turboBlockByteSize(kv_type_v);

                var params = [_]?*anyopaque{
                    @ptrCast(&d_q),       @ptrCast(&d_keys),   @ptrCast(&d_vals),
                    @ptrCast(&d_out),     @ptrCast(&nh_u),      @ptrCast(&nkv_u),
                    @ptrCast(&hd_u),      @ptrCast(&sl),         @ptrCast(&kvd_u),
                    @ptrCast(&scale_f),   @ptrCast(&bits_k_u),   @ptrCast(&bits_v_u),
                    @ptrCast(&bb_k_u),    @ptrCast(&bb_v_u),
                };

                const smem: u32 = (sl + 1) * @sizeOf(f32);
                self.launch(self.fn_sdpa_turbo, @intCast(nh), block_size, smem, &params);
            }
            return;
        }

        // Non-turbo, non-f32 quantized KV: not yet supported
        if (kv_type_k != .f32 or kv_type_v != .f32)
            @panic("CUDA SDPA prefill: unsupported KV quant type — use f32 or turbo2/3/4");

        // Pure f32 path
        const kvd = nkv * hd;

        // Get device pointers for Q and new K/V
        var d_q = self.getInputBuf(q, n_tok * nh * hd * @sizeOf(f32));
        var d_k_new = self.getInputBuf(k, n_tok * kvd * @sizeOf(f32));
        var d_v_new = self.getInputBuf(v, n_tok * kvd * @sizeOf(f32));
        var d_out = self.getOutputBuf(output, n_tok * nh * hd * @sizeOf(f32));

        // Get device pointers for KV cache
        const f32_keys: []f32 = @as([*]f32, @ptrCast(@alignCast(kv_keys.ptr)))[0 .. kv_keys.len / 4];
        const f32_values: []f32 = @as([*]f32, @ptrCast(@alignCast(kv_values.ptr)))[0 .. kv_values.len / 4];
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
            @ptrCast(&d_q),       @ptrCast(&d_k_cache), @ptrCast(&d_v_cache),
            @ptrCast(&d_k_new),   @ptrCast(&d_v_new),   @ptrCast(&d_out),
            @ptrCast(&nh_u),      @ptrCast(&nkv_u),     @ptrCast(&hd_u),
            @ptrCast(&prev_u),    @ptrCast(&ntok_u),     @ptrCast(&scale_f),
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
