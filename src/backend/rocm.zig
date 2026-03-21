//! AMD ROCm/HIP GPU backend for accelerated tensor operations.
//!
//! Uses the HIP Runtime API (libamdhip64.so) loaded dynamically at runtime.
//! Kernels are written in Zig, compiled to AMDGCN ISA via amdgcn-amdhsa target,
//! and embedded as an HSACO (HSA Code Object) in this binary.
//!
//! If libamdhip64 is not available, init() returns error.RocmNotAvailable
//! and the caller falls back to another backend.
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
//! Some GPU kernels are disabled pending validation (marked in code).
//!
//! Build HSACO: `zig build amdgcn [-Drocm-arch=gfx1100]`
//! Output at `zig-out/rocm/kernels.hsaco`. Copy to
//! `src/backend/kernels/rocm/kernels.hsaco` and commit.

const std = @import("std");
const backend_mod = @import("backend.zig");
const TensorData = backend_mod.TensorData;
const DType = backend_mod.DType;
const CpuBackend = backend_mod.CpuBackend;

// ── Embedded HSACO ────────────────────────────────────────────────

const hsaco_data = @embedFile("kernels/rocm/kernels.hsaco");

// ── HIP types ────────────────────────────────────────────────────

const HipError = c_int;
const HipModule = ?*anyopaque;
const HipFunction = ?*anyopaque;
const HipStream = ?*anyopaque;
const DevicePtr = usize;

const HIP_SUCCESS: HipError = 0;

const hipMemcpyHostToDevice: c_int = 1;
const hipMemcpyDeviceToHost: c_int = 2;
const hipMemcpyDeviceToDevice: c_int = 3;

// ── Tuning constants ─────────────────────────────────────────────

/// Workgroup size — must match common.zig block_dim.
const block_size: u32 = 256;

/// LDS bytes for block reductions (8 waves x 4 bytes).
const reduction_smem: u32 = 32;

/// Size of the buffer for retrieving the HIP device name.
const device_name_buf_size: usize = 256;

// ── HIP function pointer types ──────────────────────────────────

const FnInit = *const fn (c_uint) callconv(.c) HipError;
const FnSetDevice = *const fn (c_int) callconv(.c) HipError;
const FnDeviceGetName = *const fn ([*]u8, c_int, c_int) callconv(.c) HipError;
const FnDeviceSynchronize = *const fn () callconv(.c) HipError;
const FnMalloc = *const fn (*?*anyopaque, usize) callconv(.c) HipError;
const FnFree = *const fn (?*anyopaque) callconv(.c) HipError;
const FnMemcpy = *const fn (?*anyopaque, ?*const anyopaque, usize, c_int) callconv(.c) HipError;
const FnModuleLoadData = *const fn (*HipModule, [*]const u8) callconv(.c) HipError;
const FnModuleUnload = *const fn (HipModule) callconv(.c) HipError;
const FnModuleGetFunction = *const fn (*HipFunction, HipModule, [*:0]const u8) callconv(.c) HipError;
const FnRuntimeGetVersion = *const fn (*c_int) callconv(.c) HipError;
const FnDeviceGetAttribute = *const fn (*c_int, c_int, c_int) callconv(.c) HipError;

/// hipDeviceAttribute_t for GCN architecture (gcnArchName equivalent as integer).
const hipDeviceAttributeGcnArch: c_int = 0;

const FnLaunchKernel = *const fn (
    HipFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    HipStream,
    [*]?*anyopaque,
    ?[*]?*anyopaque,
) callconv(.c) HipError;

// ── Backend struct ───────────────────────────────────────────────

/// ROCm GPU backend — HIP/HSA runtime with dynamic library loading.
pub const RocmBackend = struct {
    // HIP handles
    module: HipModule = null,
    lib: std.DynLib = undefined,

    // Function pointers (loaded from libamdhip64)
    hipDeviceSynchronize: FnDeviceSynchronize = undefined,
    hipModuleUnload: FnModuleUnload = undefined,
    hipMalloc: FnMalloc = undefined,
    hipFree: FnFree = undefined,
    hipMemcpy: FnMemcpy = undefined,
    hipLaunchKernel: FnLaunchKernel = undefined,

    // Kernel function handles
    fn_silu: HipFunction = null,
    fn_gelu: HipFunction = null,
    fn_add: HipFunction = null,
    fn_mul: HipFunction = null,
    fn_rms_norm: HipFunction = null,
    fn_softmax: HipFunction = null,
    fn_l2_norm: HipFunction = null,
    fn_rope: HipFunction = null,
    fn_gemv_f32: HipFunction = null,
    fn_gemv_bf16: HipFunction = null,
    fn_gemv_f16: HipFunction = null,
    fn_gemv_q8_0: HipFunction = null,
    fn_gemv_q4_0: HipFunction = null,
    fn_gemv_q4_k: HipFunction = null,
    fn_gemv_q5_k: HipFunction = null,
    fn_gemv_q6_k: HipFunction = null,
    fn_gemv_fp8_e4m3: HipFunction = null,
    fn_gemv_fp8_e5m2: HipFunction = null,
    fn_gemv_mlx_q4: HipFunction = null,
    fn_rms_norm_multi: HipFunction = null,
    fn_sdpa: HipFunction = null,
    fn_sigmoid_mul: HipFunction = null,
    fn_silu_mul: HipFunction = null,
    fn_deinterleave: HipFunction = null,
    fn_deltanet_gate_beta: HipFunction = null,
    fn_deltanet_conv1d: HipFunction = null,

    /// Allow falling back to CPU for ops without ROCm kernels.
    allow_cpu_fallback: bool = false,

    /// CPU fallback for unsupported ops.
    cpu: CpuBackend = .{},

    /// One-shot flag: warn only once when NVFP4 SafeTensors GEMV falls back to CPU.
    nvfp4_st_fallback_warned: bool = false,

    /// Device name retrieved during initialization.
    device_name: [device_name_buf_size]u8 = undefined,
    device_name_len: usize = 0,

    /// Pre-formatted HIP runtime version string (e.g., "HIP 6.2").
    hip_ver_str: [16]u8 = .{0} ** 16,

    /// Pre-formatted GCN architecture string (e.g., "gfx1100").
    gcn_arch_str: [16]u8 = .{0} ** 16,

    /// Allocator for buffer caches.
    allocator: std.mem.Allocator = undefined,

    /// Permanent cache: weight buffers uploaded once and reused forever.
    buf_cache: std.AutoHashMap(usize, CachedBuf) = undefined,

    /// Activation cache: device mirrors of host activation buffers.
    act_cache: std.AutoHashMap(usize, ActBuf) = undefined,

    /// KV cache: device mirrors of per-layer KV buffers with incremental upload.
    kv_dev_cache: std.AutoHashMap(usize, KvDevCache) = undefined,

    /// Number of AMDGCN kernels loaded at init.
    pub const n_kernels: u32 = 26;

    /// Library name loaded via dlopen at init.
    pub const lib_name = "libamdhip64.so";

    const CachedBuf = struct {
        dptr: DevicePtr,
        size: usize,
    };

    /// Device-side KV cache buffer.
    const KvDevCache = struct {
        dptr: DevicePtr,
        capacity: usize,
    };

    /// Activation buffer state — tracks data freshness between host and device.
    const BufState = enum { clean, dirty, stale };

    const ActBuf = struct {
        dptr: DevicePtr,
        size: usize,
        state: BufState,
    };

    // ── Init / Deinit ───────────────────────────────────────────

    /// Initialize the ROCm backend: load libamdhip64, create context, load HSACO kernels.
    pub fn init(allocator: std.mem.Allocator) !RocmBackend {
        var self = RocmBackend{};
        self.allocator = allocator;
        self.buf_cache = std.AutoHashMap(usize, CachedBuf).init(allocator);
        self.buf_cache.ensureTotalCapacity(512) catch {};
        errdefer self.buf_cache.deinit();
        self.act_cache = std.AutoHashMap(usize, ActBuf).init(allocator);
        errdefer self.act_cache.deinit();
        self.kv_dev_cache = std.AutoHashMap(usize, KvDevCache).init(allocator);
        errdefer self.kv_dev_cache.deinit();

        // Dynamically load HIP runtime
        self.lib = std.DynLib.open("libamdhip64.so") catch return error.RocmNotAvailable;
        errdefer self.lib.close();

        // Resolve all function pointers
        const hipInit = self.lookup(FnInit, "hipInit") orelse return error.RocmNotAvailable;
        const hipSetDevice = self.lookup(FnSetDevice, "hipSetDevice") orelse return error.RocmNotAvailable;
        const hipDeviceGetName = self.lookup(FnDeviceGetName, "hipDeviceGetName") orelse return error.RocmNotAvailable;
        self.hipDeviceSynchronize = self.lookup(FnDeviceSynchronize, "hipDeviceSynchronize") orelse return error.RocmNotAvailable;
        const hipModuleLoadData = self.lookup(FnModuleLoadData, "hipModuleLoadData") orelse return error.RocmNotAvailable;
        self.hipModuleUnload = self.lookup(FnModuleUnload, "hipModuleUnload") orelse return error.RocmNotAvailable;
        const hipModuleGetFunction = self.lookup(FnModuleGetFunction, "hipModuleGetFunction") orelse return error.RocmNotAvailable;
        self.hipMalloc = self.lookup(FnMalloc, "hipMalloc") orelse return error.RocmNotAvailable;
        self.hipFree = self.lookup(FnFree, "hipFree") orelse return error.RocmNotAvailable;
        self.hipMemcpy = self.lookup(FnMemcpy, "hipMemcpy") orelse return error.RocmNotAvailable;
        self.hipLaunchKernel = self.lookup(FnLaunchKernel, "hipModuleLaunchKernel") orelse return error.RocmNotAvailable;

        // Initialize HIP
        if (hipInit(0) != HIP_SUCCESS) return error.RocmInitFailed;
        if (hipSetDevice(0) != HIP_SUCCESS) return error.NoRocmDevice;

        // Store device name for display
        if (hipDeviceGetName(&self.device_name, @intCast(device_name_buf_size), 0) == HIP_SUCCESS) {
            self.device_name_len = std.mem.indexOfScalar(u8, &self.device_name, 0) orelse device_name_buf_size;
        }

        // Query HIP runtime version
        if (self.lookup(FnRuntimeGetVersion, "hipRuntimeGetVersion")) |hipRuntimeGetVersion| {
            var ver: c_int = 0;
            if (hipRuntimeGetVersion(&ver) == HIP_SUCCESS and ver > 0) {
                const v: u32 = @intCast(ver);
                _ = std.fmt.bufPrint(&self.hip_ver_str, "HIP {d}.{d}", .{ v / 10_000_000, (v / 100_000) % 100 }) catch {};
            }
        }

        // Query GCN architecture
        if (self.lookup(FnDeviceGetAttribute, "hipDeviceGetAttribute")) |hipDeviceGetAttribute| {
            var gcn: c_int = 0;
            if (hipDeviceGetAttribute(&gcn, hipDeviceAttributeGcnArch, 0) == HIP_SUCCESS and gcn > 0) {
                _ = std.fmt.bufPrint(&self.gcn_arch_str, "gfx{d}", .{@as(u32, @intCast(gcn))}) catch {};
            }
        }

        // Load HSACO module
        if (hsaco_data.len == 0) return error.HsacoEmpty;
        if (hipModuleLoadData(&self.module, hsaco_data.ptr) != HIP_SUCCESS) return error.HsacoLoadFailed;
        errdefer _ = self.hipModuleUnload(self.module);

        // Get kernel function handles
        self.fn_silu = try self.getFunction(hipModuleGetFunction, "silu_kernel");
        self.fn_gelu = try self.getFunction(hipModuleGetFunction, "gelu_kernel");
        self.fn_add = try self.getFunction(hipModuleGetFunction, "add_kernel");
        self.fn_mul = try self.getFunction(hipModuleGetFunction, "mul_kernel");
        self.fn_rms_norm = try self.getFunction(hipModuleGetFunction, "rms_norm_kernel");
        self.fn_softmax = try self.getFunction(hipModuleGetFunction, "softmax_kernel");
        self.fn_l2_norm = try self.getFunction(hipModuleGetFunction, "l2_norm_kernel");
        self.fn_rope = try self.getFunction(hipModuleGetFunction, "rope_kernel");
        self.fn_gemv_f32 = try self.getFunction(hipModuleGetFunction, "gemv_f32_kernel");
        self.fn_gemv_bf16 = try self.getFunction(hipModuleGetFunction, "gemv_bf16_kernel");
        self.fn_gemv_f16 = try self.getFunction(hipModuleGetFunction, "gemv_f16_kernel");
        self.fn_gemv_q8_0 = try self.getFunction(hipModuleGetFunction, "gemv_q8_0_kernel");
        self.fn_gemv_q4_0 = try self.getFunction(hipModuleGetFunction, "gemv_q4_0_kernel");
        self.fn_gemv_q4_k = try self.getFunction(hipModuleGetFunction, "gemv_q4_k_kernel");
        self.fn_gemv_q5_k = try self.getFunction(hipModuleGetFunction, "gemv_q5_k_kernel");
        self.fn_gemv_q6_k = try self.getFunction(hipModuleGetFunction, "gemv_q6_k_kernel");
        self.fn_gemv_fp8_e4m3 = try self.getFunction(hipModuleGetFunction, "gemv_fp8_e4m3_kernel");
        self.fn_gemv_fp8_e5m2 = try self.getFunction(hipModuleGetFunction, "gemv_fp8_e5m2_kernel");
        self.fn_gemv_mlx_q4 = try self.getFunction(hipModuleGetFunction, "gemv_mlx_q4_kernel");
        self.fn_rms_norm_multi = try self.getFunction(hipModuleGetFunction, "rms_norm_multi_kernel"); // loaded but rmsNormMulti() uses CPU fallback — GPU kernel not yet validated
        self.fn_sdpa = try self.getFunction(hipModuleGetFunction, "sdpa_kernel");
        self.fn_sigmoid_mul = try self.getFunction(hipModuleGetFunction, "sigmoid_mul_kernel");
        self.fn_silu_mul = try self.getFunction(hipModuleGetFunction, "silu_mul_kernel");
        self.fn_deinterleave = try self.getFunction(hipModuleGetFunction, "deinterleave_kernel");
        self.fn_deltanet_gate_beta = try self.getFunction(hipModuleGetFunction, "deltanet_gate_beta_kernel");
        self.fn_deltanet_conv1d = try self.getFunction(hipModuleGetFunction, "deltanet_conv1d_kernel");

        return self;
    }

    /// Release all HIP resources: device buffers, caches, module, and library.
    pub fn deinit(self: *RocmBackend) void {
        // Free all cached activation buffers
        var act_it = self.act_cache.valueIterator();
        while (act_it.next()) |act| _ = self.hipFree(@ptrFromInt(act.dptr));
        self.act_cache.deinit();

        // Free all KV device cache buffers
        var kv_it = self.kv_dev_cache.valueIterator();
        while (kv_it.next()) |kv| _ = self.hipFree(@ptrFromInt(kv.dptr));
        self.kv_dev_cache.deinit();

        // Free all cached weight buffers
        var wt_it = self.buf_cache.valueIterator();
        while (wt_it.next()) |cached| _ = self.hipFree(@ptrFromInt(cached.dptr));
        self.buf_cache.deinit();

        if (self.module != null) _ = self.hipModuleUnload(self.module);
        _ = self.hipDeviceSynchronize();
        self.lib.close();
    }

    fn lookup(self: *RocmBackend, comptime T: type, name: [:0]const u8) ?T {
        return self.lib.lookup(T, name);
    }

    fn getFunction(self: *RocmBackend, hipModuleGetFunction: FnModuleGetFunction, name: [*:0]const u8) !HipFunction {
        var func: HipFunction = null;
        if (hipModuleGetFunction(&func, self.module, name) != HIP_SUCCESS)
            return error.KernelNotFound;
        return func;
    }

    // ── Low-level buffer operations ─────────────────────────────

    fn deviceAlloc(self: *RocmBackend, size: usize) DevicePtr {
        var ptr: ?*anyopaque = null;
        const rc = self.hipMalloc(&ptr, @max(size, 4));
        if (rc != HIP_SUCCESS or ptr == null) {
            std.log.err("hipMalloc failed: size={d} ({d} MB) rc={d}", .{ size, size / (1024 * 1024), rc });
            return 0;
        }
        return @intFromPtr(ptr.?);
    }

    fn uploadToDevice(self: *RocmBackend, host_ptr: *const anyopaque, size: usize) DevicePtr {
        const dptr = self.deviceAlloc(size);
        _ = self.hipMemcpy(@ptrFromInt(dptr), host_ptr, size, hipMemcpyHostToDevice);
        return dptr;
    }

    fn downloadFromDevice(self: *RocmBackend, dptr: DevicePtr, host_ptr: *anyopaque, size: usize) void {
        _ = self.hipMemcpy(host_ptr, @ptrFromInt(dptr), size, hipMemcpyDeviceToHost);
    }

    fn memcpyDtoD(self: *RocmBackend, dst: DevicePtr, src: DevicePtr, size: usize) void {
        _ = self.hipMemcpy(@ptrFromInt(dst), @ptrFromInt(src), size, hipMemcpyDeviceToDevice);
    }

    // ── Weight cache (permanent, read-only) ─────────────────────

    /// Get device pointer for a weight buffer. Uploads once, reused forever.
    fn getOrUpload(self: *RocmBackend, ptr: [*]const u8, size: usize) DevicePtr {
        const addr = @intFromPtr(ptr);
        if (self.buf_cache.get(addr)) |cached| {
            if (cached.size >= size) return cached.dptr;
            _ = self.hipFree(@ptrFromInt(cached.dptr));
            _ = self.buf_cache.remove(addr);
        }
        const dptr = self.uploadToDevice(ptr, size);
        self.buf_cache.put(addr, .{ .dptr = dptr, .size = size }) catch |err| {
            std.log.warn("ROCm buf_cache put failed: {}", .{err});
        };
        return dptr;
    }

    // ── Activation cache (deferred sync) ────────────────────────

    /// Check if addr falls within any cached activation buffer's range.
    fn findContaining(self: *RocmBackend, addr: usize, size: usize, comptime mark_dirty: bool, comptime refresh_stale: bool) ?DevicePtr {
        var it = self.act_cache.iterator();
        while (it.next()) |entry| {
            const base = entry.key_ptr.*;
            const act = entry.value_ptr;
            if (addr >= base and addr + size <= base + act.size) {
                if (refresh_stale and act.state == .stale) {
                    _ = self.hipMemcpy(@ptrFromInt(act.dptr), @as(?*const anyopaque, @ptrFromInt(base)), act.size, hipMemcpyHostToDevice);
                    act.state = .clean;
                }
                if (mark_dirty) act.state = .dirty;
                return act.dptr + (addr - base);
            }
        }
        return null;
    }

    /// Get device buffer for a read-only input.
    fn getInputBuf(self: *RocmBackend, ptr: anytype, size: usize) DevicePtr {
        const addr = @intFromPtr(ptr);
        if (self.act_cache.getPtr(addr)) |act| {
            if (act.size >= size) {
                if (act.state == .stale) {
                    _ = self.hipMemcpy(@ptrFromInt(act.dptr), @as(?*const anyopaque, @ptrCast(ptr)), size, hipMemcpyHostToDevice);
                    act.state = .clean;
                }
                return act.dptr;
            }
            _ = self.hipFree(@ptrFromInt(act.dptr));
            _ = self.act_cache.remove(addr);
        }
        if (self.findContaining(addr, size, false, true)) |dptr| return dptr;
        if (self.buf_cache.get(addr)) |cached| {
            if (cached.size >= size) return cached.dptr;
        }
        const dptr = self.uploadToDevice(@ptrCast(ptr), size);
        self.act_cache.put(addr, .{ .dptr = dptr, .size = size, .state = .clean }) catch |err| {
            std.log.warn("ROCm act_cache put failed: {}", .{err});
        };
        return dptr;
    }

    /// Get device buffer for a write-only output.
    fn getOutputBuf(self: *RocmBackend, ptr: anytype, size: usize) DevicePtr {
        const addr = @intFromPtr(ptr);
        if (self.act_cache.getPtr(addr)) |act| {
            if (act.size >= size) {
                act.state = .dirty;
                return act.dptr;
            }
            _ = self.hipFree(@ptrFromInt(act.dptr));
        }
        if (self.findContaining(addr, size, true, false)) |dptr| return dptr;
        const dptr = self.deviceAlloc(size);
        self.act_cache.put(addr, .{ .dptr = dptr, .size = size, .state = .dirty }) catch |err| {
            std.log.warn("ROCm act_cache put failed: {}", .{err});
        };
        return dptr;
    }

    /// Get device buffer for in-place read+write.
    fn getInPlaceBuf(self: *RocmBackend, ptr: anytype, size: usize) DevicePtr {
        const addr = @intFromPtr(ptr);
        if (self.act_cache.getPtr(addr)) |act| {
            if (act.size >= size) {
                if (act.state == .stale) {
                    _ = self.hipMemcpy(@ptrFromInt(act.dptr), @as(?*const anyopaque, @ptrCast(ptr)), size, hipMemcpyHostToDevice);
                }
                act.state = .dirty;
                return act.dptr;
            }
            _ = self.hipFree(@ptrFromInt(act.dptr));
        }
        if (self.findContaining(addr, size, true, true)) |dptr| return dptr;
        const dptr = self.uploadToDevice(@ptrCast(ptr), size);
        self.act_cache.put(addr, .{ .dptr = dptr, .size = size, .state = .dirty }) catch |err| {
            std.log.warn("ROCm act_cache put failed: {}", .{err});
        };
        return dptr;
    }

    /// Sync GPU, download dirty buffers to host, mark all entries stale.
    fn flushActivations(self: *RocmBackend) void {
        _ = self.hipDeviceSynchronize();
        var it = self.act_cache.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.state == .dirty) {
                self.downloadFromDevice(entry.value_ptr.dptr, @ptrFromInt(entry.key_ptr.*), entry.value_ptr.size);
            }
            entry.value_ptr.state = .stale;
        }
    }

    /// Remove a specific activation from cache after CPU writes.
    fn invalidateAct(self: *RocmBackend, ptr: anytype) void {
        const addr = @intFromPtr(ptr);
        if (self.act_cache.fetchRemove(addr)) |kv| {
            _ = self.hipFree(@ptrFromInt(kv.value.dptr));
        }
    }

    // ── Launch helper ───────────────────────────────────────────

    fn launch(self: *RocmBackend, func: HipFunction, grid: u32, block: u32, smem: u32, params: [*]?*anyopaque) void {
        _ = self.hipLaunchKernel(func, grid, 1, 1, block, 1, 1, smem, null, params, null);
    }

    // ── Weight size helper ──────────────────────────────────────

    const weightBytes = @import("backend.zig").weightBytes;

    // ── Backend interface ────────────────────────────────────────

    /// y[n] = W[n,k] @ x[k]. GPU kernels for F32, BF16, F16, Q8_0, Q4_0;
    /// other dtypes fall back to CPU.
    pub fn gemv(self: *RocmBackend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
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

    /// output[i] = input[i] * weight[i] * rsqrt(mean(x^2) + eps)
    pub fn rmsNorm(self: *RocmBackend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
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
    pub fn silu(self: *RocmBackend, input: [*]const f32, output: [*]f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        var d_in = self.getInputBuf(input, sz);
        var d_out = self.getOutputBuf(output, sz);

        var n_u32: u32 = @intCast(n);
        var params = [_]?*anyopaque{ @ptrCast(&d_in), @ptrCast(&d_out), @ptrCast(&n_u32) };
        const grid: u32 = @intCast((n + block_size - 1) / block_size);
        self.launch(self.fn_silu, grid, block_size, 0, &params);
    }

    /// GELU activation
    pub fn gelu(self: *RocmBackend, input: [*]const f32, output: [*]f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        var d_in = self.getInputBuf(input, sz);
        var d_out = self.getOutputBuf(output, sz);

        var n_u32: u32 = @intCast(n);
        var params = [_]?*anyopaque{ @ptrCast(&d_in), @ptrCast(&d_out), @ptrCast(&n_u32) };
        const grid: u32 = @intCast((n + block_size - 1) / block_size);
        self.launch(self.fn_gelu, grid, block_size, 0, &params);
    }

    /// Element-wise add
    pub fn add(self: *RocmBackend, a: [*]const f32, b: [*]const f32, output: [*]f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        var d_a = self.getInputBuf(a, sz);
        var d_b = self.getInputBuf(b, sz);
        var d_out = self.getOutputBuf(output, sz);

        var n_u32: u32 = @intCast(n);
        var params = [_]?*anyopaque{ @ptrCast(&d_a), @ptrCast(&d_b), @ptrCast(&d_out), @ptrCast(&n_u32) };
        const grid: u32 = @intCast((n + block_size - 1) / block_size);
        self.launch(self.fn_add, grid, block_size, 0, &params);
    }

    /// Fused add + rmsNorm
    pub fn addRmsNorm(self: *RocmBackend, a: [*]f32, b: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        self.add(a, b, a, n);
        self.rmsNorm(a, weight, output, n, eps);
    }

    /// Element-wise mul
    pub fn mul(self: *RocmBackend, a: [*]const f32, b: [*]const f32, output: [*]f32, n: usize) void {
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
    pub fn softmax(self: *RocmBackend, data: [*]f32, n: usize) void {
        var d_data = self.getInPlaceBuf(data, n * @sizeOf(f32));

        var n_u32: u32 = @intCast(n);
        var params = [_]?*anyopaque{ @ptrCast(&d_data), @ptrCast(&n_u32) };
        self.launch(self.fn_softmax, 1, block_size, reduction_smem, &params);
    }

    /// Rotary Position Embedding (in-place)
    pub fn rope(self: *RocmBackend, x: [*]f32, pos: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
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
    pub fn embLookup(self: *RocmBackend, table: TensorData, token_id: u32, output: [*]f32, dim: usize) void {
        self.flushActivations();
        self.cpu.embLookup(table, token_id, output, dim);
        self.invalidateAct(output);
    }

    /// L2 normalize in-place.
    pub fn l2Norm(self: *RocmBackend, x: [*]f32, n: usize, eps: f32) void {
        var d_x = self.getInPlaceBuf(x, n * @sizeOf(f32));

        var n_u32: u32 = @intCast(n);
        var eps_f32: f32 = eps;
        var params = [_]?*anyopaque{ @ptrCast(&d_x), @ptrCast(&n_u32), @ptrCast(&eps_f32) };
        self.launch(self.fn_l2_norm, 1, block_size, reduction_smem, &params);
    }

    /// NVFP4 SafeTensors GEMV — CPU fallback.
    pub fn gemvNvfp4St(self: *RocmBackend, x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: usize, k: usize) void {
        if (!self.allow_cpu_fallback)
            @panic("NVFP4 SafeTensors GEMV has no ROCm kernel. Pass --allow-cpu-fallback to use CPU fallback.");
        if (!self.nvfp4_st_fallback_warned) {
            self.nvfp4_st_fallback_warned = true;
            std.log.warn("NVFP4 SafeTensors GEMV: no ROCm kernel, using CPU fallback", .{});
        }
        self.flushActivations();
        self.cpu.gemvNvfp4St(x, weight, scale, y, n, k);
        self.invalidateAct(y);
    }

    /// MLX affine quantized GEMV — CPU fallback.
    pub fn gemvMlxQ(self: *RocmBackend, x: [*]const f32, weight: [*]const u8, scales: [*]const u8, biases: [*]const u8, y: [*]f32, n: usize, k: usize, bits: u32) void {
        self.flushActivations();
        self.cpu.gemvMlxQ(x, weight, scales, biases, y, n, k, bits);
        self.invalidateAct(y);
    }

    /// In-place sigmoid-gated multiply: data[i] *= sigmoid(gate[i])
    pub fn sigmoidMul(self: *RocmBackend, data: [*]f32, gate: [*]const f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        var d_data = self.getOutputBuf(data, sz);
        var d_gate = self.getInputBuf(gate, sz);

        var n_u32: u32 = @intCast(n);
        var params = [_]?*anyopaque{
            @ptrCast(&d_data),
            @ptrCast(&d_gate),
            @ptrCast(&d_data), // in-place output
            @ptrCast(&n_u32),
        };
        const grid: u32 = @intCast((n + block_size - 1) / block_size);
        self.launch(self.fn_sigmoid_mul, grid, block_size, 0, &params);
    }

    /// Fused SiLU + multiply: out[i] = silu(a[i]) * b[i]
    pub fn siluMul(self: *RocmBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        var d_a = self.getInputBuf(a, sz);
        var d_b = self.getInputBuf(b, sz);
        var d_out = self.getOutputBuf(out, sz);

        var n_u32: u32 = @intCast(n);
        var params = [_]?*anyopaque{
            @ptrCast(&d_a),
            @ptrCast(&d_b),
            @ptrCast(&d_out),
            @ptrCast(&n_u32),
        };
        const grid: u32 = @intCast((n + block_size - 1) / block_size);
        self.launch(self.fn_silu_mul, grid, block_size, 0, &params);
    }

    /// In-place per-head rmsNorm — CPU fallback (GPU kernel disabled for debugging).
    pub fn rmsNormMulti(self: *RocmBackend, data: [*]f32, weight: [*]const f32, n_heads: usize, head_dim: usize, eps: f32) void {
        self.flushActivations();
        self.cpu.rmsNormMulti(data, weight, n_heads, head_dim, eps);
        self.invalidateAct(data);
    }

    /// Deinterleave paired data: [A0(stride), B0(stride), ...] → [A0, A1, ...] [B0, B1, ...]
    pub fn deinterleave(self: *RocmBackend, input: [*]const f32, out_a: [*]f32, out_b: [*]f32, stride: usize, n_pairs: usize) void {
        const total = n_pairs * stride;
        const sz = total * @sizeOf(f32);
        var d_in = self.getInputBuf(input, sz * 2);
        var d_a = self.getOutputBuf(out_a, sz);
        var d_b = self.getOutputBuf(out_b, sz);

        var stride_u32: u32 = @intCast(stride);
        var n_pairs_u32: u32 = @intCast(n_pairs);
        var params = [_]?*anyopaque{
            @ptrCast(&d_in),
            @ptrCast(&d_a),
            @ptrCast(&d_b),
            @ptrCast(&stride_u32),
            @ptrCast(&n_pairs_u32),
        };
        const grid: u32 = @intCast((total + block_size - 1) / block_size);
        self.launch(self.fn_deinterleave, grid, block_size, 0, &params);
    }

    /// Batched GEMV — sequential dispatch.
    pub fn gemvMulti(self: *RocmBackend, x: [*]const f32, ops: []const backend_mod.GemvOp, k: usize) void {
        for (ops) |op| self.gemv(x, op.w, op.y, op.n, k);
    }

    /// Commit pending GPU work and download results to host.
    pub fn sync(self: *RocmBackend) void {
        self.flushActivations();
    }

    /// No-op — ROCm dispatches are not batched.
    pub fn beginBatch(_: *RocmBackend) void {}
    /// No-op — ROCm dispatches are not batched.
    pub fn endBatch(_: *RocmBackend) void {}

    /// Returns backend startup information for display.
    pub fn backendInfo(self: *const RocmBackend) @import("backend.zig").BackendInfo {
        return .{
            .name = "ROCm",
            .device_name = self.device_name[0..self.device_name_len],
            .lib_name = lib_name,
            .n_gpu_kernels = n_kernels,
            .kernel_type = "HSACO",
            .compute_cap = std.mem.sliceTo(&self.gcn_arch_str, 0),
            .driver_version = std.mem.sliceTo(&self.hip_ver_str, 0),
        };
    }

    // ── KV cache ────────────────────────────────────────────────

    /// Allocate a KV cache slice using the host allocator. `n` is byte count.
    pub fn allocKvSlice(_: *RocmBackend, allocator: std.mem.Allocator, n: usize) error{OutOfMemory}![]u8 {
        return allocator.alloc(u8, n);
    }

    /// Free a KV cache slice allocated via allocKvSlice.
    pub fn freeKvSlice(_: *RocmBackend, allocator: std.mem.Allocator, slice: []u8) void {
        allocator.free(slice);
    }

    /// Get or grow device KV cache buffer for the given required size.
    /// Starts small and doubles when needed, copying existing data.
    fn getOrAllocKvBuf(self: *RocmBackend, addr: usize, required: usize, max_capacity: usize) DevicePtr {
        if (self.kv_dev_cache.getPtr(addr)) |kv| {
            if (kv.capacity >= required) return kv.dptr;
            // Grow: double until sufficient (capped at max_capacity)
            var new_cap = kv.capacity;
            while (new_cap < required) new_cap = @min(new_cap * 2, max_capacity);
            const new_dptr = self.deviceAlloc(new_cap);
            if (new_dptr != 0) {
                self.memcpyDtoD(new_dptr, kv.dptr, kv.capacity);
                _ = self.hipFree(@ptrFromInt(kv.dptr));
                kv.dptr = new_dptr;
                kv.capacity = new_cap;
            }
            return kv.dptr;
        }
        // First allocation: allocate full capacity to avoid repeated growth.
        const cap = max_capacity;
        const dptr = self.deviceAlloc(cap);
        self.kv_dev_cache.put(addr, .{
            .dptr = dptr,
            .capacity = cap,
        }) catch |err| {
            std.log.warn("ROCm kv_dev_cache put failed: {}", .{err});
        };
        return dptr;
    }

    /// Fused scaled dot-product attention on GPU with KV cache append.
    pub fn sdpa(self: *RocmBackend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type: @import("backend.zig").KvQuantType) void {
        // CPU fallback for quantized KV — GPU kernel only handles f32.
        if (kv_type != .f32) {
            self.flushActivations();
            self.cpu.sdpa(q, keys, values, k_new, v_new, output, nh, nkv, hd, seq_len, scale, kv_type);
            self.invalidateAct(output);
            return;
        }

        // Cast byte slices to f32 for GPU path.
        const f32_keys: []f32 = @as([*]f32, @ptrCast(@alignCast(keys.ptr)))[0 .. keys.len / 4];
        const f32_values: []f32 = @as([*]f32, @ptrCast(@alignCast(values.ptr)))[0 .. values.len / 4];

        const kvd = nkv * hd;
        const kvd_bytes = kvd * @sizeOf(f32);
        const sl = seq_len + 1;

        // Get device KV cache buffers (grow incrementally to avoid pre-allocating full capacity)
        const max_kv_bytes = f32_keys.len * @sizeOf(f32);
        const needed_kv_bytes = sl * kvd_bytes;
        var d_keys = self.getOrAllocKvBuf(@intFromPtr(f32_keys.ptr), needed_kv_bytes, max_kv_bytes);
        var d_vals = self.getOrAllocKvBuf(@intFromPtr(f32_values.ptr), needed_kv_bytes, max_kv_bytes);

        // Get k_new/v_new from act_cache (still on GPU from RoPE/gemv)
        const d_k_new = self.getInputBuf(k_new, kvd_bytes);
        const d_v_new = self.getInputBuf(v_new, kvd_bytes);

        // Append k_new/v_new at position seq_len via device-to-device copy
        self.memcpyDtoD(d_keys + seq_len * kvd_bytes, d_k_new, kvd_bytes);
        self.memcpyDtoD(d_vals + seq_len * kvd_bytes, d_v_new, kvd_bytes);

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

        // LDS: sl floats for attention scores
        const smem: u32 = sl_u32 * @sizeOf(f32);
        self.launch(self.fn_sdpa, @intCast(nh), block_size, smem, &params);
    }

    /// DeltaNet SSM recurrence — CPU fallback.
    /// DeltaNet SSM recurrence — CPU fallback.
    /// GPU kernels exist (gate_beta, conv1d) but full pipeline optimization deferred.
    /// TODO: Implement full GPU DeltaNet pipeline (L2 norm, recurrence kernels).
    pub fn deltaNet(self: *RocmBackend, conv_in: [*]const f32, conv_out: [*]f32, z_buf: [*]const f32, alpha_buf: [*]const f32, beta_buf: [*]const f32, output: [*]f32, conv_state: [*]f32, ssm_state: []f32, ssm_a: [*]const f32, dt_bias: [*]const f32, conv_w: [*]const f32, ssm_norm_w: [*]const f32, p: backend_mod.DeltaNetParams) void {
        self.flushActivations();
        self.cpu.deltaNet(conv_in, conv_out, z_buf, alpha_buf, beta_buf, output, conv_state, ssm_state, ssm_a, dt_bias, conv_w, ssm_norm_w, p);
        self.invalidateAct(conv_out);
        self.invalidateAct(output);
    }
};
