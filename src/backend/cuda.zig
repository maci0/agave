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
const CpuBackend = backend_mod.CpuBackend;

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

/// Size of the buffer for retrieving the CUDA device name.
const device_name_buf_size: usize = 256;

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
    cuMemAllocManaged: ?FnMemAllocManaged = null,
    cuLaunchKernel: FnLaunchKernel = undefined,

    // Kernel function handles
    fn_silu: CUfunction = null,
    fn_gelu: CUfunction = null,
    fn_add: CUfunction = null,
    fn_mul: CUfunction = null,
    fn_rms_norm: CUfunction = null,
    fn_softmax: CUfunction = null,
    fn_l2_norm: CUfunction = null,
    fn_rope: CUfunction = null,
    fn_gemv_f32: CUfunction = null,
    fn_gemv_bf16: CUfunction = null,
    fn_gemv_f16: CUfunction = null,
    fn_gemv_q8_0: CUfunction = null,
    fn_gemv_q4_0: CUfunction = null,
    fn_gemv_q4_0_batch: CUfunction = null,
    fn_gemv_q4_k: CUfunction = null,
    fn_gemv_q5_k: CUfunction = null,
    fn_gemv_q6_k: CUfunction = null,
    fn_gemv_fp8_e4m3: CUfunction = null,
    fn_gemv_fp8_e5m2: CUfunction = null,
    fn_sdpa: CUfunction = null,

    /// Allow falling back to CPU for ops without CUDA kernels.
    allow_cpu_fallback: bool = false,

    /// CPU fallback for unsupported ops.
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

    /// One-shot flag: warn only once when NVFP4 SafeTensors GEMV falls back to CPU.
    nvfp4_st_fallback_warned: bool = false,

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
    pub const n_kernels: u32 = 20;

    /// Library name loaded via dlopen at init.
    pub const lib_name = cuda_lib_name;

    const CachedBuf = struct {
        dptr: CUdeviceptr,
        size: usize,
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
        self.buf_cache.ensureTotalCapacity(512) catch {};
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
            _ = std.fmt.bufPrint(&self.drv_str, "CUDA {d}.{d}", .{ self.driver_version / 1000, (self.driver_version % 1000) / 10 }) catch {};
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
        self.fn_softmax = try self.getFunction("softmax_kernel");
        self.fn_l2_norm = try self.getFunction("l2_norm_kernel");
        self.fn_rope = try self.getFunction("rope_kernel");
        self.fn_gemv_f32 = try self.getFunction("gemv_f32_kernel");
        self.fn_gemv_bf16 = try self.getFunction("gemv_bf16_kernel");
        self.fn_gemv_f16 = try self.getFunction("gemv_f16_kernel");
        self.fn_gemv_q8_0 = try self.getFunction("gemv_q8_0_kernel");
        self.fn_gemv_q4_0 = try self.getFunction("gemv_q4_0_kernel");
        self.fn_gemv_q4_0_batch = try self.getFunction("gemv_q4_0_batch_kernel");
        self.fn_gemv_q4_k = try self.getFunction("gemv_q4_k_kernel");
        self.fn_gemv_q5_k = try self.getFunction("gemv_q5_k_kernel");
        self.fn_gemv_q6_k = try self.getFunction("gemv_q6_k_kernel");
        self.fn_gemv_fp8_e4m3 = try self.getFunction("gemv_fp8_e4m3_kernel");
        self.fn_gemv_fp8_e5m2 = try self.getFunction("gemv_fp8_e5m2_kernel");
        self.fn_sdpa = try self.getFunction("sdpa_kernel");

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
        var wt_it = self.buf_cache.valueIterator();
        while (wt_it.next()) |cached| _ = self.cuMemFree(cached.dptr);
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
            if (!self.is_uma) _ = self.cuMemFree(cached.dptr);
            _ = self.buf_cache.remove(addr);
        }

        const dptr = self.uploadToDevice(ptr, size);
        self.buf_cache.put(addr, .{ .dptr = dptr, .size = size }) catch |err| {
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

    /// y[n] = W[n,k] @ x[k]. GPU kernels for F32, BF16, F16, Q8_0, Q4_0;
    /// other dtypes fall back to CPU.
    pub fn gemv(self: *CudaBackend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        const func = switch (w.dtype) {
            .f32 => self.fn_gemv_f32,
            .bf16 => self.fn_gemv_bf16,
            .f16 => self.fn_gemv_f16,
            .q8_0 => self.fn_gemv_q8_0,
            .q4_0 => self.fn_gemv_q4_0,
            .q4_k => self.fn_gemv_q4_k,
            .q5_k => self.fn_gemv_q5_k,
            .q6_k => self.fn_gemv_q6_k,
            .fp8_e4m3 => self.fn_gemv_fp8_e4m3,
            .fp8_e5m2 => self.fn_gemv_fp8_e5m2,
            else => {
                // Unsupported dtype: flush GPU→host, run on CPU, mark output stale for re-upload
                self.flushActivations();
                self.cpu.gemv(x, w, y, n, k);
                self.invalidateAct(y);
                return;
            },
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

    /// In-place sigmoid-gated multiply — CPU fallback.
    pub fn sigmoidMul(self: *CudaBackend, data: [*]f32, gate: [*]const f32, n: usize) void {
        self.flushActivations();
        var cpu = backend_mod.CpuBackend{};
        cpu.sigmoidMul(data, gate, n);
        self.invalidateAct(data);
    }

    /// Fused SiLU + multiply — CPU fallback.
    pub fn siluMul(self: *CudaBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        self.flushActivations();
        var cpu = backend_mod.CpuBackend{};
        cpu.siluMul(a, b, out, n);
        self.invalidateAct(out);
    }

    /// In-place per-head rmsNorm — CPU fallback.
    pub fn rmsNormMulti(self: *CudaBackend, data: [*]f32, weight: [*]const f32, n_heads: usize, head_dim: usize, eps: f32) void {
        self.flushActivations();
        var cpu = backend_mod.CpuBackend{};
        cpu.rmsNormMulti(data, weight, n_heads, head_dim, eps);
        self.invalidateAct(data);
    }

    /// Deinterleave paired data into two separate output buffers — CPU fallback.
    pub fn deinterleave(self: *CudaBackend, input: [*]const f32, out_a: [*]f32, out_b: [*]f32, stride: usize, n_pairs: usize) void {
        self.flushActivations();
        var cpu = backend_mod.CpuBackend{};
        cpu.deinterleave(input, out_a, out_b, stride, n_pairs);
        self.invalidateAct(out_a);
        self.invalidateAct(out_b);
    }

    /// Batched GEMV: fuse multiple GEMV ops sharing the same input into a single
    /// kernel launch. On Q4_0 with 2-3 ops, uses the dedicated batch kernel.
    /// Otherwise falls back to sequential gemv calls.
    pub fn gemvMulti(self: *CudaBackend, x: [*]const f32, ops: []const backend_mod.GemvOp, k: usize) void {
        if (ops.len >= 2 and ops.len <= 3) {
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

    /// Launch the fused Q4_0 batched GEMV kernel for 2-3 ops.
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

        // Op 2 (present for len == 3, else dummy with n2=0)
        var d_w2: CUdeviceptr = d_w0;
        var d_y2: CUdeviceptr = d_y0;
        var n2: u32 = 0;
        if (ops.len >= 3) {
            d_w2 = self.getOrUpload(ops[2].w.data, weightBytes(.q4_0, ops[2].n, k));
            d_y2 = self.getOutputBuf(ops[2].y, ops[2].n * @sizeOf(f32));
            n2 = @intCast(ops[2].n);
        }

        var k_u32: u32 = @intCast(k);
        var params = [_]?*anyopaque{
            @ptrCast(&d_x),
            @ptrCast(&d_w0), @ptrCast(&d_y0), @ptrCast(&n0),
            @ptrCast(&d_w1), @ptrCast(&d_y1), @ptrCast(&n1),
            @ptrCast(&d_w2), @ptrCast(&d_y2), @ptrCast(&n2),
            @ptrCast(&k_u32),
        };

        const grid = n0 + n1 + n2;
        self.launch(self.fn_gemv_q4_0_batch, grid, block_size, reduction_smem, &params);
    }

    /// output[i] = input[i] * weight[i] * rsqrt(mean(x^2) + eps)
    pub fn rmsNorm(self: *CudaBackend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        const sz = n * @sizeOf(f32);
        var d_in = self.getInputBuf(input, sz);
        var d_w = self.getOrUpload(@ptrCast(weight), sz);
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

    /// Fused add + rmsNorm (sequential fallback — no fused CUDA kernel yet).
    pub fn addRmsNorm(self: *CudaBackend, a: [*]f32, b: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        self.add(a, b, a, n);
        self.rmsNorm(a, weight, output, n, eps);
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
            @ptrCast(&d_x),   @ptrCast(&pos_u32), @ptrCast(&nh_u32),
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

    /// NVFP4 SafeTensors GEMV — CPU fallback.
    pub fn gemvNvfp4St(self: *CudaBackend, x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: usize, k: usize) void {
        if (!self.allow_cpu_fallback)
            @panic("NVFP4 SafeTensors GEMV has no CUDA kernel. Pass --allow-cpu-fallback to use CPU fallback.");
        if (!self.nvfp4_st_fallback_warned) {
            self.nvfp4_st_fallback_warned = true;
            std.log.warn("NVFP4 SafeTensors GEMV: no CUDA kernel, using CPU fallback", .{});
        }
        self.flushActivations();
        self.cpu.gemvNvfp4St(x, weight, scale, y, n, k);
        self.invalidateAct(y);
    }

    /// MLX affine quantized GEMV — CPU fallback.
    pub fn gemvMlxQ(self: *CudaBackend, x: [*]const f32, weight: [*]const u8, scales: [*]const u8, biases: [*]const u8, y: [*]f32, n: usize, k: usize, bits: u32) void {
        self.flushActivations();
        self.cpu.gemvMlxQ(x, weight, scales, biases, y, n, k, bits);
        self.invalidateAct(y);
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

    /// Allocate a KV cache slice using GPU-optimal memory.
    /// On UMA (integrated GPU): uses cuMemAllocManaged so both CPU and GPU
    /// can access the same pointer directly — no mirror or D2D copies needed.
    /// On discrete GPU: uses the provided allocator (host memory); the existing
    /// kv_dev_cache will mirror to VRAM on demand during SDPA.
    pub fn allocKvSlice(self: *CudaBackend, allocator: std.mem.Allocator, n: usize) error{OutOfMemory}![]u8 {
        const byte_len = n;
        if (self.is_uma) {
            if (self.cuMemAllocManaged) |allocManaged| {
                var dptr: CUdeviceptr = 0;
                const result = allocManaged(&dptr, @max(byte_len, 4), CU_MEM_ATTACH_GLOBAL);
                if (result == 0 and dptr != 0) {
                    const ptr: [*]u8 = @ptrFromInt(dptr);
                    @memset(ptr[0..n], 0);
                    return ptr[0..n];
                }
            }
        }
        // Fallback: plain host allocation (discrete GPU or managed alloc failed)
        return allocator.alloc(u8, n);
    }

    /// Free a KV cache slice allocated via allocKvSlice.
    /// Detects whether the slice was allocated via cuMemAllocManaged (UMA)
    /// or the host allocator, and frees accordingly.
    pub fn freeKvSlice(self: *CudaBackend, allocator: std.mem.Allocator, slice: []u8) void {
        if (slice.len == 0) return;
        if (self.is_uma and self.cuMemAllocManaged != null) {
            // Try cuMemFree — if the pointer was managed, this succeeds.
            // cuMemFree returns 0 (CUDA_SUCCESS) for managed pointers.
            const dptr: CUdeviceptr = @intFromPtr(slice.ptr);
            const result = self.cuMemFree(dptr);
            if (result == 0) return;
        }
        // Host-allocated fallback
        allocator.free(slice);
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

    /// Fused scaled dot-product attention on GPU with KV cache append.
    /// Reads Q, k_new, v_new from act_cache (still on GPU from prior ops).
    /// Appends k_new/v_new to device KV cache via D2D copy, then launches kernel.
    /// No sync() required — all data stays on GPU.
    pub fn sdpa(self: *CudaBackend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type: @import("backend.zig").KvQuantType) void {
        // Non-f32 KV types: flush to host, run CPU SDPA, invalidate output
        if (kv_type != .f32) {
            self.flushActivations();
            var cpu = @import("cpu.zig").CpuBackend{ .pool = null };
            cpu.sdpa(q, keys, values, k_new, v_new, output, nh, nkv, hd, seq_len, scale, kv_type);
            self.invalidateAct(output);
            return;
        }

        // f32 path: cast byte slices to f32 slices for GPU kernel
        const f32_keys: []f32 = @as([*]f32, @ptrCast(@alignCast(keys.ptr)))[0 .. keys.len / 4];
        const f32_values: []f32 = @as([*]f32, @ptrCast(@alignCast(values.ptr)))[0 .. values.len / 4];

        const kvd = nkv * hd;
        const kvd_bytes = kvd * @sizeOf(f32);
        const sl = seq_len + 1;

        // Get device KV cache buffers (allocated once, reused across tokens)
        var d_keys = self.getOrAllocKvBuf(@intFromPtr(f32_keys.ptr), f32_keys.len * @sizeOf(f32));
        var d_vals = self.getOrAllocKvBuf(@intFromPtr(f32_values.ptr), f32_values.len * @sizeOf(f32));

        // Get k_new/v_new from act_cache (still on GPU from RoPE/gemv — no sync needed)
        const d_k_new = self.getInputBuf(k_new, kvd_bytes);
        const d_v_new = self.getInputBuf(v_new, kvd_bytes);

        // Append k_new/v_new at position seq_len via device-to-device copy
        _ = self.cuMemcpyDtoD(d_keys + seq_len * kvd_bytes, d_k_new, kvd_bytes);
        _ = self.cuMemcpyDtoD(d_vals + seq_len * kvd_bytes, d_v_new, kvd_bytes);

        // Q is an activation buffer, output is an activation
        var d_q = self.getInputBuf(q, nh * hd * @sizeOf(f32));
        var d_out = self.getOutputBuf(output, nh * hd * @sizeOf(f32));

        var nh_u32: u32 = @intCast(nh);
        var nkv_u32: u32 = @intCast(nkv);
        var hd_u32: u32 = @intCast(hd);
        var sl_u32: u32 = @intCast(sl);
        var kvd_u32: u32 = @intCast(kvd);
        var scale_f32: f32 = scale;

        var params = [_]?*anyopaque{
            @ptrCast(&d_q),
            @ptrCast(&d_keys),
            @ptrCast(&d_vals),
            @ptrCast(&d_out),
            @ptrCast(&nh_u32),
            @ptrCast(&nkv_u32),
            @ptrCast(&hd_u32),
            @ptrCast(&sl_u32),
            @ptrCast(&kvd_u32),
            @ptrCast(&scale_f32),
        };

        // Shared memory: sl floats for attention scores (f32 = 4 bytes)
        const smem: u32 = sl_u32 * @sizeOf(f32);
        self.launch(self.fn_sdpa, @intCast(nh), block_size, smem, &params);
    }

    /// DeltaNet SSM recurrence — CPU fallback.
    pub fn deltaNet(self: *CudaBackend, conv_in: [*]const f32, conv_out: [*]f32, z_buf: [*]const f32, alpha_buf: [*]const f32, beta_buf: [*]const f32, output: [*]f32, conv_state: [*]f32, ssm_state: []f32, ssm_a: [*]const f32, dt_bias: [*]const f32, conv_w: [*]const f32, ssm_norm_w: [*]const f32, p: backend_mod.DeltaNetParams) void {
        self.flushActivations();
        self.cpu.deltaNet(conv_in, conv_out, z_buf, alpha_buf, beta_buf, output, conv_state, ssm_state, ssm_a, dt_bias, conv_w, ssm_norm_w, p);
        self.invalidateAct(conv_out);
        self.invalidateAct(output);
    }
};
