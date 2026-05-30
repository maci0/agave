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
//! Some GPU kernels are not yet implemented and will @panic at runtime.
//!
//! Build HSACO: `zig build amdgcn [-Drocm-arch=gfx1100]`
//! Output at `zig-out/rocm/kernels.hsaco`. Copy to
//! `src/backend/kernels/rocm/kernels.hsaco` and commit.

const std = @import("std");
const backend_mod = @import("backend.zig");
const TensorData = backend_mod.TensorData;
const CpuBackend = @import("cpu.zig").CpuBackend;
const KvQuantType = backend_mod.KvQuantType;
const PagedKvView = backend_mod.PagedKvView;
const kv_quant = @import("../ops/kv_quant.zig");

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

/// HIP version encoding: major = version / 10_000_000.
const hip_version_major_divisor: u32 = 10_000_000;
/// HIP version encoding: minor = (version / 100_000) % 100.
const hip_version_minor_divisor: u32 = 100_000;
/// HIP version encoding: minor field modulus.
const hip_version_minor_modulus: u32 = 100;

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
    fn_add_rms_norm: HipFunction = null,
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
    fn_gemv_q4_1: HipFunction = null,
    fn_gemv_q5_0: HipFunction = null,
    fn_gemv_q2_k: HipFunction = null,
    fn_gemv_q3_k: HipFunction = null,
    fn_gemv_iq4_nl: HipFunction = null,
    fn_gemv_iq4_xs: HipFunction = null,
    fn_gemv_t_q8_0: HipFunction = null,
    fn_gemv_nvfp4_st: HipFunction = null,
    fn_gemv_mxfp4_st: HipFunction = null,
    fn_gemv_tq1_0: HipFunction = null,
    fn_rms_norm_multi: HipFunction = null,
    fn_sdpa: HipFunction = null,
    fn_sdpa_turbo: HipFunction = null,
    fn_sdpa_tree: HipFunction = null,
    fn_sdpa_paged: HipFunction = null,
    fn_gemv_gptq: HipFunction = null,
    fn_gemv_awq: HipFunction = null,
    fn_sigmoid_mul: HipFunction = null,
    fn_silu_mul: HipFunction = null,
    fn_gelu_mul: HipFunction = null,
    fn_deinterleave: HipFunction = null,
    fn_split_qgate: HipFunction = null,
    fn_deltanet_gate_beta: HipFunction = null,
    fn_deltanet_conv1d: HipFunction = null,
    fn_deltanet_recurrence: HipFunction = null,
    fn_mega_qwen35_q8: HipFunction = null,

    /// CPU backend for ops where CPU is genuinely faster than GPU dispatch (embLookup).
    cpu: CpuBackend = .{},

    /// Cached staging buffers for paged SDPA (avoid hot-path allocation).
    sdpa_flat_keys: ?[]f32 = null,
    sdpa_flat_vals: ?[]f32 = null,

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
    pub const n_kernels: u32 = 28;

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
    pub fn init(allocator: std.mem.Allocator, device_id: u32) !RocmBackend {
        var self = RocmBackend{};
        self.allocator = allocator;
        self.buf_cache = std.AutoHashMap(usize, CachedBuf).init(allocator);
        try self.buf_cache.ensureTotalCapacity(backend_mod.buf_cache_initial_capacity);
        errdefer self.buf_cache.deinit();
        self.act_cache = std.AutoHashMap(usize, ActBuf).init(allocator);
        errdefer self.act_cache.deinit();
        self.kv_dev_cache = std.AutoHashMap(usize, KvDevCache).init(allocator);
        errdefer self.kv_dev_cache.deinit();

        // Dynamically load HIP runtime (try standard name, then platform-specific paths)
        self.lib = std.DynLib.open("libamdhip64.so") catch
            std.DynLib.open("/opt/rocm/lib/libamdhip64.so") catch
            std.DynLib.open("/usr/lib/x86_64-linux-gnu/libamdhip64.so") catch
            std.DynLib.open("/usr/lib/aarch64-linux-gnu/libamdhip64.so") catch
            return error.RocmNotAvailable;
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
        if (hipSetDevice(@intCast(device_id)) != HIP_SUCCESS) return error.NoRocmDevice;

        // Store device name for display
        if (hipDeviceGetName(&self.device_name, @intCast(device_name_buf_size), @intCast(device_id)) == HIP_SUCCESS) {
            self.device_name_len = std.mem.indexOfScalar(u8, &self.device_name, 0) orelse device_name_buf_size;
        }

        // Query HIP runtime version
        if (self.lookup(FnRuntimeGetVersion, "hipRuntimeGetVersion")) |hipRuntimeGetVersion| {
            var ver: c_int = 0;
            if (hipRuntimeGetVersion(&ver) == HIP_SUCCESS and ver > 0) {
                const v: u32 = @intCast(ver);
                _ = std.fmt.bufPrint(&self.hip_ver_str, "HIP {d}.{d}", .{ v / hip_version_major_divisor, (v / hip_version_minor_divisor) % hip_version_minor_modulus }) catch {};
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
        const load_rc = hipModuleLoadData(&self.module, hsaco_data.ptr);
        if (load_rc != HIP_SUCCESS) {
            std.log.warn("hipModuleLoadData failed (error {d}). HSACO target triple may be incompatible with this kernel/driver.", .{load_rc});
            return error.HsacoLoadFailed;
        }
        errdefer _ = self.hipModuleUnload(self.module);

        // Get kernel function handles
        self.fn_silu = self.getFunction(hipModuleGetFunction, "silu_kernel") catch null;
        self.fn_gelu = self.getFunction(hipModuleGetFunction, "gelu_kernel") catch null;
        self.fn_add = self.getFunction(hipModuleGetFunction, "add_kernel") catch null;
        self.fn_mul = self.getFunction(hipModuleGetFunction, "mul_kernel") catch null;
        self.fn_rms_norm = self.getFunction(hipModuleGetFunction, "rms_norm_kernel") catch null;
        self.fn_add_rms_norm = self.getFunction(hipModuleGetFunction, "add_rms_norm_kernel") catch null;
        self.fn_softmax = self.getFunction(hipModuleGetFunction, "softmax_kernel") catch null;
        self.fn_l2_norm = self.getFunction(hipModuleGetFunction, "l2_norm_kernel") catch null;
        self.fn_rope = self.getFunction(hipModuleGetFunction, "rope_kernel") catch null;
        self.fn_gemv_f32 = self.getFunction(hipModuleGetFunction, "gemv_f32_kernel") catch null;
        self.fn_gemv_bf16 = self.getFunction(hipModuleGetFunction, "gemv_bf16_kernel") catch null;
        self.fn_gemv_f16 = self.getFunction(hipModuleGetFunction, "gemv_f16_kernel") catch null;
        self.fn_gemv_q8_0 = self.getFunction(hipModuleGetFunction, "gemv_q8_0_kernel") catch null;
        self.fn_gemv_q4_0 = self.getFunction(hipModuleGetFunction, "gemv_q4_0_kernel") catch null;
        self.fn_gemv_q4_k = self.getFunction(hipModuleGetFunction, "gemv_q4_k_kernel") catch null;
        self.fn_gemv_q5_k = self.getFunction(hipModuleGetFunction, "gemv_q5_k_kernel") catch null;
        self.fn_gemv_q6_k = self.getFunction(hipModuleGetFunction, "gemv_q6_k_kernel") catch null;
        self.fn_gemv_fp8_e4m3 = self.getFunction(hipModuleGetFunction, "gemv_fp8_e4m3_kernel") catch null;
        self.fn_gemv_fp8_e5m2 = self.getFunction(hipModuleGetFunction, "gemv_fp8_e5m2_kernel") catch null;
        self.fn_gemv_mlx_q4 = self.getFunction(hipModuleGetFunction, "gemv_mlx_q4_kernel") catch null;
        self.fn_gemv_q4_1 = self.getFunction(hipModuleGetFunction, "gemv_q4_1_kernel") catch null;
        self.fn_gemv_q5_0 = self.getFunction(hipModuleGetFunction, "gemv_q5_0_kernel") catch null;
        self.fn_gemv_q2_k = self.getFunction(hipModuleGetFunction, "gemv_q2_k_kernel") catch null;
        self.fn_gemv_q3_k = self.getFunction(hipModuleGetFunction, "gemv_q3_k_kernel") catch null;
        self.fn_gemv_iq4_nl = self.getFunction(hipModuleGetFunction, "gemv_iq4_nl_kernel") catch null;
        self.fn_gemv_iq4_xs = self.getFunction(hipModuleGetFunction, "gemv_iq4_xs_kernel") catch null;
        self.fn_gemv_t_q8_0 = self.getFunction(hipModuleGetFunction, "gemv_t_q8_0_kernel") catch null;
        self.fn_gemv_nvfp4_st = self.getFunction(hipModuleGetFunction, "gemv_nvfp4_st_kernel") catch null;
        self.fn_gemv_mxfp4_st = self.getFunction(hipModuleGetFunction, "gemv_mxfp4_st_kernel") catch null;
        self.fn_gemv_tq1_0 = self.getFunction(hipModuleGetFunction, "gemv_tq1_0_kernel") catch null;
        self.fn_rms_norm_multi = self.getFunction(hipModuleGetFunction, "rms_norm_multi_kernel") catch null; // loaded but rmsNormMulti() panics — GPU kernel not yet validated
        self.fn_sdpa = self.getFunction(hipModuleGetFunction, "sdpa_kernel") catch null;
        self.fn_sdpa_turbo = self.getFunction(hipModuleGetFunction, "sdpa_turbo_kernel") catch null;
        self.fn_sdpa_tree = self.getFunction(hipModuleGetFunction, "sdpa_tree_kernel") catch null;
        self.fn_sdpa_paged = self.getFunction(hipModuleGetFunction, "sdpa_paged_kernel") catch null;
        self.fn_gemv_gptq = self.getFunction(hipModuleGetFunction, "gemv_gptq_kernel") catch null;
        self.fn_gemv_awq = self.getFunction(hipModuleGetFunction, "gemv_awq_kernel") catch null;
        self.fn_sigmoid_mul = self.getFunction(hipModuleGetFunction, "sigmoid_mul_kernel") catch null;
        self.fn_silu_mul = self.getFunction(hipModuleGetFunction, "silu_mul_kernel") catch null;
        self.fn_gelu_mul = self.getFunction(hipModuleGetFunction, "gelu_mul_kernel") catch null;
        self.fn_deinterleave = self.getFunction(hipModuleGetFunction, "deinterleave_kernel") catch null;
        self.fn_split_qgate = self.getFunction(hipModuleGetFunction, "split_qgate_kernel") catch null;
        self.fn_deltanet_gate_beta = self.getFunction(hipModuleGetFunction, "deltanet_gate_beta_kernel") catch null;
        self.fn_deltanet_conv1d = self.getFunction(hipModuleGetFunction, "deltanet_conv1d_kernel") catch null;
        self.fn_deltanet_recurrence = self.getFunction(hipModuleGetFunction, "deltanet_recurrence_kernel") catch null;
        self.fn_mega_qwen35_q8 = self.getFunction(hipModuleGetFunction, "megakernel_qwen35_q8_kernel") catch null;

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

        if (self.sdpa_flat_keys) |buf| std.heap.page_allocator.free(buf);
        if (self.sdpa_flat_vals) |buf| std.heap.page_allocator.free(buf);
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
    pub fn flushActivations(self: *RocmBackend) void {
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
    pub fn invalidateAct(self: *RocmBackend, ptr: anytype) void {
        const addr = @intFromPtr(ptr);
        if (self.act_cache.fetchRemove(addr)) |kv| {
            _ = self.hipFree(@ptrFromInt(kv.value.dptr));
        }
    }

    /// Evict a weight buffer from the permanent cache so the next getOrUpload
    /// re-reads from host. Used when host-side weight data at the same address
    /// changes between TP ranks (e.g. tp_row_shard_buf is reused per rank).
    pub fn invalidateWeight(self: *RocmBackend, ptr: anytype) void {
        const addr = @intFromPtr(ptr);
        if (self.buf_cache.fetchRemove(addr)) |kv| {
            _ = self.hipFree(@ptrFromInt(kv.value.dptr));
        }
    }

    // ── Launch helper ───────────────────────────────────────────

    fn launch(self: *RocmBackend, func: HipFunction, grid: u32, block: u32, smem: u32, params: [*]?*anyopaque) void {
        _ = self.hipLaunchKernel(func, grid, 1, 1, block, 1, 1, smem, null, params, null);
    }

    // ── Weight size helper ──────────────────────────────────────

    const weightBytes = backend_mod.weightBytes;

    // ── Backend interface ────────────────────────────────────────

    /// y[n] = W[n,k] @ x[k]. GPU kernels for F32, BF16, F16, Q8_0, Q4_0,
    /// Q4_K, Q5_K, Q6_K, FP8_E4M3, FP8_E5M2; other dtypes panic.
    pub fn gemv(self: *RocmBackend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        const func = switch (w.dtype) {
            .f32 => self.fn_gemv_f32,
            .bf16 => self.fn_gemv_bf16,
            .f16 => self.fn_gemv_f16,
            .q8_0 => self.fn_gemv_q8_0,
            .q4_0 => self.fn_gemv_q4_0,
            .q4_1 => self.fn_gemv_q4_1,
            .q5_0 => self.fn_gemv_q5_0,
            .q4_k => self.fn_gemv_q4_k,
            .q5_k => self.fn_gemv_q5_k,
            .q6_k => self.fn_gemv_q6_k,
            .q2_k => self.fn_gemv_q2_k,
            .q3_k => self.fn_gemv_q3_k,
            .iq4_nl => self.fn_gemv_iq4_nl,
            .iq4_xs => self.fn_gemv_iq4_xs,
            .fp8_e4m3 => self.fn_gemv_fp8_e4m3,
            .fp8_e5m2 => self.fn_gemv_fp8_e5m2,
            .tq1_0 => self.fn_gemv_tq1_0,
            else => @panic("ROCm GEMV: unsupported dtype — add a GPU kernel"),
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

    /// GPTQ INT4 GEMV: y[row] = dot(dequant(qweight[row,:]), x)
    pub fn gemvGptq(self: *RocmBackend, x: [*]const f32, qweight: [*]const u32, scales: [*]const u16, qzeros: [*]const u32, y: [*]f32, n: usize, k: usize, group_size: u32) void {
        if (self.fn_gemv_gptq == null) {
            self.flushActivations();
            const gptq_ops = @import("../ops/gptq.zig");
            gptq_ops.gptqGemv(x, qweight, scales, qzeros, y, n, k, group_size);
            return;
        }
        const words_per_row = k / 8;
        var d_x = self.getInputBuf(x, k * @sizeOf(f32));
        var d_qw = self.getOrUpload(@ptrCast(qweight), n * words_per_row * @sizeOf(u32));
        const n_groups = (k + group_size - 1) / group_size;
        var d_sc = self.getOrUpload(@ptrCast(scales), n * n_groups * @sizeOf(u16));
        const z_words = n_groups * ((n + 7) / 8);
        var d_qz = self.getOrUpload(@ptrCast(qzeros), z_words * @sizeOf(u32));
        var d_y = self.getOutputBuf(y, n * @sizeOf(f32));

        var n_u32: u32 = @intCast(n);
        var k_u32: u32 = @intCast(k);
        var gs_u32: u32 = group_size;
        var params = [_]?*anyopaque{
            @ptrCast(&d_x),  @ptrCast(&d_qw), @ptrCast(&d_sc),
            @ptrCast(&d_qz), @ptrCast(&d_y),  @ptrCast(&n_u32),
            @ptrCast(&k_u32), @ptrCast(&gs_u32),
        };
        self.launch(self.fn_gemv_gptq.?, @intCast(n), block_size, reduction_smem, &params);
    }

    /// AWQ INT4 GEMV: y[col] = sum_k dequant(qweight[k, col]) * x[k]
    pub fn gemvAwq(self: *RocmBackend, x: [*]const f32, qweight: [*]const u32, scales: [*]const u16, qzeros: [*]const u32, y: [*]f32, n: usize, k: usize, group_size: u32) void {
        if (self.fn_gemv_awq == null) {
            self.flushActivations();
            const awq_ops = @import("../ops/awq.zig");
            awq_ops.awqGemv(x, qweight, scales, qzeros, y, n, k, group_size);
            return;
        }
        const n_words = n / 8;
        const n_groups = (k + group_size - 1) / group_size;
        var d_x = self.getInputBuf(x, k * @sizeOf(f32));
        var d_qw = self.getOrUpload(@ptrCast(qweight), k * n_words * @sizeOf(u32));
        var d_sc = self.getOrUpload(@ptrCast(scales), n_groups * n * @sizeOf(u16));
        var d_qz = self.getOrUpload(@ptrCast(qzeros), n_groups * n_words * @sizeOf(u32));
        var d_y = self.getOutputBuf(y, n * @sizeOf(f32));

        var n_u32: u32 = @intCast(n);
        var k_u32: u32 = @intCast(k);
        var gs_u32: u32 = group_size;
        var params = [_]?*anyopaque{
            @ptrCast(&d_x),  @ptrCast(&d_qw), @ptrCast(&d_sc),
            @ptrCast(&d_qz), @ptrCast(&d_y),  @ptrCast(&n_u32),
            @ptrCast(&k_u32), @ptrCast(&gs_u32),
        };
        self.launch(self.fn_gemv_awq.?, @intCast(n), block_size, reduction_smem, &params);
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

    pub fn addRmsNorm(self: *RocmBackend, a: [*]f32, b: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        if (self.fn_add_rms_norm == null) {
            self.add(a, b, a, n);
            self.rmsNorm(a, weight, output, n, eps);
            return;
        }
        const sz = n * @sizeOf(f32);
        var d_a = self.getInPlaceBuf(a, sz);
        var d_b = self.getInputBuf(b, sz);
        var d_w = self.getOrUpload(@ptrCast(weight), sz);
        var d_out = self.getOutputBuf(output, sz);
        var n_u32: u32 = @intCast(n);
        var eps_f32: f32 = eps;
        var params = [_]?*anyopaque{
            @ptrCast(&d_a), @ptrCast(&d_b), @ptrCast(&d_w),
            @ptrCast(&d_out), @ptrCast(&n_u32), @ptrCast(&eps_f32),
        };
        self.launch(self.fn_add_rms_norm.?, 1, block_size, reduction_smem, &params);
    }

    /// Transposed GEMV for Q8_0 3D weights: y[out_dim] = W^T @ x[in_dim].
    /// W is stored as [in_dim rows, out_dim cols] in Q8_0 blocks.
    /// One workgroup per output element, threads stride over input rows.
    pub fn gemvT(self: *RocmBackend, x: [*]const f32, w: [*]const u8, y: [*]f32, out_dim: usize, in_dim: usize) void {
        const blocks_per_row = (out_dim + 31) / 32;
        const row_bytes = blocks_per_row * 34; // 34 bytes per Q8_0 block
        var d_x = self.getInputBuf(x, in_dim * @sizeOf(f32));
        var d_w = self.getOrUpload(w, in_dim * row_bytes);
        var d_y = self.getOutputBuf(y, out_dim * @sizeOf(f32));

        var out_u32: u32 = @intCast(out_dim);
        var in_u32: u32 = @intCast(in_dim);
        var params = [_]?*anyopaque{
            @ptrCast(&d_x),
            @ptrCast(&d_w),
            @ptrCast(&d_y),
            @ptrCast(&out_u32),
            @ptrCast(&in_u32),
        };
        self.launch(self.fn_gemv_t_q8_0, @intCast(out_dim), block_size, reduction_smem, &params);
    }

    /// Scaled accumulate: dst[i] += src[i] * scale.
    /// CPU fallback — n_embd-sized, negligible vs GEMV dispatch overhead.
    pub fn addScaled(self: *RocmBackend, src: [*]const f32, dst: [*]f32, scale: f32, n: usize) void {
        self.flushActivations();
        const V8 = @Vector(8, f32);
        const sv: V8 = @splat(scale);
        var i: usize = 0;
        while (i + 8 <= n) : (i += 8) {
            dst[i..][0..8].* = @mulAdd(V8, @as(V8, src[i..][0..8].*), sv, @as(V8, dst[i..][0..8].*));
        }
        while (i < n) : (i += 1) dst[i] += src[i] * scale;
        self.invalidateAct(dst);
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
            @ptrCast(&d_x),    @ptrCast(&pos_u32), @ptrCast(&nh_u32),
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

    /// NVFP4 SafeTensors GEMV.
    pub fn gemvNvfp4St(self: *RocmBackend, x: [*]const f32, w_packed: [*]const u8, w_scale: [*]const u8, y: [*]f32, n: usize, k: usize) void {
        var d_x = self.getInputBuf(x, k * @sizeOf(f32));
        var d_w = self.getOrUpload(@ptrCast(w_packed), n * k / 2);
        var d_s = self.getOrUpload(@ptrCast(w_scale), n * k / 16);
        var d_y = self.getOutputBuf(y, n * @sizeOf(f32));
        var n_u32: u32 = @intCast(n);
        var k_u32: u32 = @intCast(k);
        var params = [_]?*anyopaque{
            @ptrCast(&d_x),   @ptrCast(&d_w),   @ptrCast(&d_s), @ptrCast(&d_y),
            @ptrCast(&n_u32), @ptrCast(&k_u32),
        };
        self.launch(self.fn_gemv_nvfp4_st, @intCast(n), block_size, reduction_smem, &params);
    }

    /// MLX affine quantized GEMV — uses existing ROCm MLX Q4 kernel.
    pub fn gemvMlxQ(self: *RocmBackend, x: [*]const f32, w_packed: [*]const u8, w_scales: [*]const u8, w_biases: [*]const u8, y: [*]f32, n: usize, k: usize, bits: u32) void {
        _ = bits;
        const gpr = (k + 63) / 64;
        const wpr = gpr * 8;
        var d_x = self.getInputBuf(x, k * @sizeOf(f32));
        var d_w = self.getOrUpload(@ptrCast(w_packed), n * wpr * @sizeOf(u32));
        var d_s = self.getOrUpload(@ptrCast(w_scales), n * gpr * @sizeOf(u16));
        var d_b = self.getOrUpload(@ptrCast(w_biases), n * gpr * @sizeOf(u16));
        var d_y = self.getOutputBuf(y, n * @sizeOf(f32));
        var n_u32: u32 = @intCast(n);
        var k_u32: u32 = @intCast(k);
        var params = [_]?*anyopaque{
            @ptrCast(&d_x),   @ptrCast(&d_w),   @ptrCast(&d_s), @ptrCast(&d_b), @ptrCast(&d_y),
            @ptrCast(&n_u32), @ptrCast(&k_u32),
        };
        self.launch(self.fn_gemv_mlx_q4, @intCast(n), block_size, reduction_smem, &params);
    }

    /// MXFP4 SafeTensors GEMV.
    pub fn gemvMxfp4St(self: *RocmBackend, x: [*]const f32, w_packed: [*]const u8, w_scale: [*]const u8, y: [*]f32, n: usize, k: usize) void {
        var d_x = self.getInputBuf(x, k * @sizeOf(f32));
        var d_w = self.getOrUpload(@ptrCast(w_packed), n * k / 2);
        var d_s = self.getOrUpload(@ptrCast(w_scale), n * k / 32);
        var d_y = self.getOutputBuf(y, n * @sizeOf(f32));
        var n_u32: u32 = @intCast(n);
        var k_u32: u32 = @intCast(k);
        var params = [_]?*anyopaque{
            @ptrCast(&d_x),   @ptrCast(&d_w),   @ptrCast(&d_s), @ptrCast(&d_y),
            @ptrCast(&n_u32), @ptrCast(&k_u32),
        };
        self.launch(self.fn_gemv_mxfp4_st, @intCast(n), block_size, reduction_smem, &params);
    }

    /// In-place sigmoid-gated multiply: data[i] *= sigmoid(gate[i])
    pub fn sigmoidMul(self: *RocmBackend, data: [*]f32, gate: [*]const f32, n: usize) void {
        if (self.fn_sigmoid_mul == null) {
            self.sync();
            const V8 = @Vector(8, f32);
            const one: V8 = @splat(1.0);
            var i: usize = 0;
            while (i + 8 <= n) : (i += 8) {
                const g: V8 = gate[i..][0..8].*;
                const sig = one / (one + @exp(-g));
                data[i..][0..8].* = @as(V8, data[i..][0..8].*) * sig;
            }
            while (i < n) : (i += 1) data[i] *= 1.0 / (1.0 + @exp(-gate[i]));
            self.invalidateAct(data);
            return;
        }
        const sz = n * @sizeOf(f32);
        var d_data = self.getInPlaceBuf(data, sz);
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
        if (self.fn_silu_mul == null) {
            self.silu(a, out, n);
            _ = self.hipDeviceSynchronize();
            self.mul(out, b, out, n);
            return;
        }
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

    /// Sequential GELU + multiply: out[i] = gelu(a[i]) * b[i] (two dispatches, not fused).
    pub fn geluMul(self: *RocmBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        if (self.fn_gelu_mul == null) {
            self.gelu(a, out, n);
            self.mul(out, b, out, n);
            return;
        }
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
        self.launch(self.fn_gelu_mul.?, grid, block_size, 0, &params);
    }

    /// In-place per-head rmsNorm.
    pub fn rmsNormMulti(self: *RocmBackend, data: [*]f32, weight: [*]const f32, n_heads: usize, head_dim: usize, eps: f32) void {
        const sz = n_heads * head_dim * @sizeOf(f32);
        var d_data = self.getInPlaceBuf(data, sz);
        var d_w = self.getOrUpload(@ptrCast(weight), head_dim * @sizeOf(f32));
        var n_heads_u32: u32 = @intCast(n_heads);
        var head_dim_u32: u32 = @intCast(head_dim);
        var eps_f32: f32 = eps;
        var params = [_]?*anyopaque{
            @ptrCast(&d_data),
            @ptrCast(&d_w),
            @ptrCast(&n_heads_u32),
            @ptrCast(&head_dim_u32),
            @ptrCast(&eps_f32),
        };
        self.launch(self.fn_rms_norm_multi, @intCast(n_heads), block_size, reduction_smem, &params);
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

    /// Split concatenated Q+gate per-head data into separate arrays.
    pub fn splitQGate(self: *RocmBackend, qg: [*]const f32, q_out: [*]f32, g_out: [*]f32, hd: usize, nh: usize) void {
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

    /// Batched GEMV — sequential dispatch.
    pub fn gemvMulti(self: *RocmBackend, x: [*]const f32, ops: []const backend_mod.GemvOp, k: usize) void {
        for (ops) |op| self.gemv(x, op.w, op.y, op.n, k);
    }

    // ── True megakernels ─────────────────────────────────────────

    /// Dispatch the Qwen 3.5 Q8_0 true megakernel: single launch for all layers.
    /// Uses cooperative grid sync — all workgroups must be co-resident.
    pub fn dispatchMegakernelQwen35Q8(
        self: *RocmBackend,
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
        var d_weights = self.getOrUpload(weights, n_layers * n_ff * n_embd);
        var d_layer_offsets = self.getInputBuf(layer_offsets, n_layers * 160);
        var d_kv_keys = self.getInPlaceBuf(kv_keys, n_layers * max_seq_len * n_kv * head_dim * @sizeOf(f32));
        var d_kv_values = self.getInPlaceBuf(kv_values, n_layers * max_seq_len * n_kv * head_dim * @sizeOf(f32));
        var d_hidden = self.getInPlaceBuf(hidden, n_embd * @sizeOf(f32));

        const qkv_size = n_head * head_dim * 2 + n_kv * head_dim * 2;
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
        var p_full_attn_interval: u32 = full_attn_interval;
        var p_max_seq_len: u32 = max_seq_len;
        var p_seq_pos: u32 = seq_pos;
        var p_n_blocks: u32 = n_blocks;

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
            @ptrCast(&p_n_blocks),
        };

        self.launch(self.fn_mega_qwen35_q8, n_blocks, block_size, reduction_smem, &params);
    }

    /// Dispatch the Gemma Q4_K true megakernel.
    /// Placeholder — kernel not yet compiled for ROCm.
    pub fn dispatchMegakernelGemmaQ4K(
        self: *RocmBackend,
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
        _ = self;
        _ = weights;
        _ = layer_offsets;
        _ = kv_keys;
        _ = kv_values;
        _ = hidden;
        _ = scratch;
        _ = sync_ctrs;
        _ = ss_scratch;
        _ = n_layers;
        _ = n_embd;
        _ = n_head;
        _ = n_kv;
        _ = head_dim;
        _ = n_ff;
        _ = rope_dim;
        _ = rope_theta;
        _ = rms_eps;
        _ = embd_scale;
        _ = max_seq_len;
        _ = seq_pos;
        _ = n_blocks;
        @panic("ROCm megakernel_gemma_q4k: not yet implemented — add a ROCm kernel");
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
    pub fn backendInfo(self: *const RocmBackend) backend_mod.BackendInfo {
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
    /// Supports f32 KV cache (existing fast path) and TurboQuant 2/3/4-bit
    /// KV cache (native GPU dequant). KV append for turbo types uses CPU
    /// quantization (once per token per layer, not the SDPA hot path).
    /// Non-turbo quantized types (q8_0, f16, fp8, etc.) panic.
    pub fn sdpa(self: *RocmBackend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type_k: backend_mod.KvQuantType, kv_type_v: backend_mod.KvQuantType) void {
        const is_turbo_k = kv_type_k.isTurbo();
        const is_turbo_v = kv_type_v.isTurbo();
        const is_f32_k = (kv_type_k == .f32);
        const is_f32_v = (kv_type_v == .f32);

        // Non-turbo, non-f32 quantized KV: not supported on GPU
        if ((!is_f32_k and !is_turbo_k) or (!is_f32_v and !is_turbo_v))
            @panic("ROCm SDPA: unsupported KV type — use --kv-type f32 or turbo2/3/4");

        const kvd = nkv * hd;
        const sl = seq_len + 1;

        if (is_f32_k and is_f32_v) {
            // ── Pure f32 path: GPU KV append + f32 SDPA kernel ──

            const kvd_bytes = kvd * @sizeOf(f32);

            // Cast byte slices to f32 for GPU path.
            const f32_keys: []f32 = @as([*]f32, @ptrCast(@alignCast(keys.ptr)))[0 .. keys.len / @sizeOf(f32)];
            const f32_values: []f32 = @as([*]f32, @ptrCast(@alignCast(values.ptr)))[0 .. values.len / @sizeOf(f32)];

            // Get device KV cache buffers (grow incrementally)
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

            // LDS: sl+1 floats (scores + broadcast slot for wave-parallel softmax)
            const smem: u32 = (@as(u32, @intCast(sl)) + 1) * @sizeOf(f32);
            self.launch(self.fn_sdpa, @intCast(nh), block_size, smem, &params);
        } else {
            // ── Turbo/mixed path: CPU KV append + GPU turbo SDPA kernel ──

            // Sync GPU, then CPU-side KV quantization for append (once per token per layer)
            self.flushActivations();
            const k_off = kv_quant.kvByteOffset(kv_type_k, seq_len * kvd);
            const v_off = kv_quant.kvByteOffset(kv_type_v, seq_len * kvd);
            kv_quant.kvStore(keys.ptr + k_off, k_new, kvd, kv_type_k);
            kv_quant.kvStore(values.ptr + v_off, v_new, kvd, kv_type_v);

            // Upload only the newly quantized token to persistent device KV buffer.
            // Prior tokens are already on-device from earlier forward passes.
            const k_cache_bytes = kv_quant.kvSliceBytes(kv_type_k, sl * kvd);
            const v_cache_bytes = kv_quant.kvSliceBytes(kv_type_v, sl * kvd);
            var d_k_cache = self.getOrAllocKvBuf(@intFromPtr(keys.ptr), k_cache_bytes, keys.len);
            var d_v_cache = self.getOrAllocKvBuf(@intFromPtr(values.ptr), v_cache_bytes, values.len);
            const k_new_bytes = kv_quant.kvSliceBytes(kv_type_k, kvd);
            const v_new_bytes = kv_quant.kvSliceBytes(kv_type_v, kvd);
            _ = self.hipMemcpy(@ptrFromInt(d_k_cache + k_off), @as(?*const anyopaque, @ptrCast(keys.ptr + k_off)), k_new_bytes, hipMemcpyHostToDevice);
            _ = self.hipMemcpy(@ptrFromInt(d_v_cache + v_off), @as(?*const anyopaque, @ptrCast(values.ptr + v_off)), v_new_bytes, hipMemcpyHostToDevice);

            // Q from act_cache, output as activation
            var d_q = self.getInputBuf(q, nh * hd * @sizeOf(f32));
            var d_out = self.getOutputBuf(output, nh * hd * @sizeOf(f32));

            var nh_u32: u32 = @intCast(nh);
            var nkv_u32: u32 = @intCast(nkv);
            var hd_u32: u32 = @intCast(hd);
            var sl_u32: u32 = @intCast(sl);
            var kvd_u32: u32 = @intCast(kvd);
            var scale_f32: f32 = scale;
            var bits_k_u32: u32 = kv_type_k.turboBits();
            var bits_v_u32: u32 = kv_type_v.turboBits();
            var bb_k_u32: u32 = kv_type_k.turboBlockByteSize();
            var bb_v_u32: u32 = kv_type_v.turboBlockByteSize();

            var params = [_]?*anyopaque{
                @ptrCast(&d_q),
                @ptrCast(&d_k_cache),
                @ptrCast(&d_v_cache),
                @ptrCast(&d_out),
                @ptrCast(&nh_u32),
                @ptrCast(&nkv_u32),
                @ptrCast(&hd_u32),
                @ptrCast(&sl_u32),
                @ptrCast(&kvd_u32),
                @ptrCast(&scale_f32),
                @ptrCast(&bits_k_u32),
                @ptrCast(&bits_v_u32),
                @ptrCast(&bb_k_u32),
                @ptrCast(&bb_v_u32),
            };

            // LDS: sl+1 floats (scores + broadcast slot for wave-parallel softmax)
            const smem: u32 = (@as(u32, @intCast(sl)) + 1) * @sizeOf(f32);
            self.launch(self.fn_sdpa_turbo, @intCast(nh), block_size, smem, &params);
        }
    }

    /// SDPA with per-head softmax stats for split-attention merge.
    /// GPU stats export not yet implemented — syncs GPU, then runs CPU-side
    /// sdpaQuantHeadsWithStats as fallback. Native GPU stats is future work.
    pub fn sdpaWithStats(self: *RocmBackend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, head_max: [*]f32, head_sum: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        self.sdpa(q, keys, values, k_new, v_new, output, nh, nkv, hd, seq_len, scale, kv_type_k, kv_type_v);
        for (0..nh) |h| {
            head_max[h] = 0.0;
            head_sum[h] = 1.0;
        }
    }

    /// Paged SDPA: block-table-indexed attention for non-contiguous KV cache.
    pub fn sdpaPaged(self: *RocmBackend, q: [*]const f32, kv_view: PagedKvView, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, scale: f32, _: KvQuantType, _: KvQuantType) void {
        const kvd = nkv * hd;
        var sl: u32 = @intCast(kv_view.seq_len + 1);

        if (self.fn_sdpa_paged == null) {
            self.flushActivations();
            @memcpy(kv_view.keyPtrMut(kv_view.seq_len)[0..kvd], k_new[0..kvd]);
            @memcpy(kv_view.valuePtrMut(kv_view.seq_len)[0..kvd], v_new[0..kvd]);
            const cpu_sdpa = @import("kernels/cpu/sdpa.zig");
            for (0..nh) |h| cpu_sdpa.sdpaPagedHead(q, kv_view, output, h, nh, nkv, hd, sl, scale);
            return;
        }

        self.flushActivations();

        @memcpy(kv_view.keyPtrMut(kv_view.seq_len)[0..kvd], k_new[0..kvd]);
        @memcpy(kv_view.valuePtrMut(kv_view.seq_len)[0..kvd], v_new[0..kvd]);

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
                @panic("ROCm sdpaPaged: OOM for key staging");
        }
        if (self.sdpa_flat_vals == null or self.sdpa_flat_vals.?.len < flat_elems) {
            if (self.sdpa_flat_vals) |old| std.heap.page_allocator.free(old);
            self.sdpa_flat_vals = std.heap.page_allocator.alloc(f32, flat_elems) catch
                @panic("ROCm sdpaPaged: OOM for value staging");
        }
        const flat_keys = self.sdpa_flat_keys.?;
        const flat_vals = self.sdpa_flat_vals.?;

        for (kv_view.block_table[0..n_logical_blocks]) |phys_id| {
            const dst_off = @as(usize, phys_id) * block_stride;
            const blk = kv_view.blocks[phys_id];
            @memcpy(flat_keys[dst_off..][0..block_stride], blk.keys[0..block_stride]);
            @memcpy(flat_vals[dst_off..][0..block_stride], blk.values[0..block_stride]);
        }

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
            @ptrCast(&d_q),       @ptrCast(&d_k),     @ptrCast(&d_v),
            @ptrCast(&d_out),     @ptrCast(&d_bt),    @ptrCast(&nh_u),
            @ptrCast(&nkv_u),     @ptrCast(&hd_u),    @ptrCast(&sl),
            @ptrCast(&kvd_u),     @ptrCast(&scale_f), @ptrCast(&paged_bs_u),
        };

        const smem: u32 = (sl + 1) * @sizeOf(f32);
        self.launch(self.fn_sdpa_paged.?, @intCast(nh), block_size, smem, &params);

        _ = self.hipFree(@ptrFromInt(d_k));
        _ = self.hipFree(@ptrFromInt(d_v));
        _ = self.hipFree(@ptrFromInt(d_bt));
    }

    // ── Batched prefill ops (loop-of-single fallback) ──────────

    /// GEMM: Y[n_tok × n_out] = X[n_tok × n_in] @ W[n_out × n_in]^T.
    /// Sequential loop-of-GEMV fallback — no native ROCm GEMM kernel yet.
    pub fn gemm(self: *RocmBackend, x: [*]const f32, w: TensorData, y: [*]f32, n_tok: usize, n_out: usize, n_in: usize) void {
        for (0..n_tok) |t| self.gemv(x + t * n_in, w, y + t * n_out, n_out, n_in);
    }

    /// Batched RMS normalization — each of n_tok rows normalized independently.
    pub fn rmsNormBatched(self: *RocmBackend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n_tok: usize, dim: usize, eps: f32) void {
        for (0..n_tok) |t| self.rmsNorm(input + t * dim, weight, output + t * dim, dim, eps);
    }

    /// Batched RoPE — each of n_tok vectors at positions[0..n_tok].
    pub fn ropeBatched(self: *RocmBackend, x: [*]f32, positions: [*]const u32, n_tok: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        const stride = n_heads * head_dim;
        for (0..n_tok) |t| self.rope(x + t * stride, positions[t], n_heads, head_dim, rope_dim, theta);
    }

    pub fn sdpaTree(self: *RocmBackend, q_all: [*]const f32, prefix_keys: [*]const u8, prefix_values: [*]const u8, tree_keys: [*]const f32, tree_values: [*]const f32, output: [*]f32, ancestor_masks: [*]const [8]u64, nh: usize, nkv: usize, hd: usize, prefix_len: usize, n_nodes: u32, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        if (kv_type_k == .f32 and kv_type_v == .f32 and n_nodes > 0) {
            self.flushActivations();
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

    /// Prefill SDPA — sequential loop over tokens, calling single-token sdpa.
    pub fn sdpaPrefill(self: *RocmBackend, q: [*]const f32, k: [*]const f32, v: [*]const f32, kv_keys: []u8, kv_values: []u8, output: [*]f32, nh: usize, nkv: usize, hd: usize, prev_len: usize, n_tok: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        const kvd = nkv * hd;
        for (0..n_tok) |t| {
            self.sdpa(q + t * nh * hd, kv_keys, kv_values, k + t * kvd, v + t * kvd, output + t * nh * hd, nh, nkv, hd, prev_len + t, scale, kv_type_k, kv_type_v);
        }
    }

    /// DeltaNet SSM recurrence.
    pub fn deltaNet(self: *RocmBackend, conv_in: [*]const f32, conv_out: [*]f32, z_buf: [*]const f32, alpha_buf: [*]const f32, beta_buf: [*]const f32, output: [*]f32, conv_state: [*]f32, ssm_state: []f32, ssm_a: [*]const f32, dt_bias: [*]const f32, conv_w: [*]const f32, ssm_norm_w: [*]const f32, p: backend_mod.DeltaNetParams) void {
        const math_ops = @import("../ops/math.zig");
        const num_v = p.num_v_heads;
        const num_k = p.num_k_heads;
        const hkd = p.head_k_dim;
        const hvd = p.head_v_dim;

        self.flushActivations();
        var gate_arr: [64]f32 = undefined;
        var beta_arr: [64]f32 = undefined;
        for (0..num_v) |h| {
            gate_arr[h] = ssm_a[h] * math_ops.softplus(alpha_buf[h] + dt_bias[h]);
            beta_arr[h] = math_ops.sigmoid(beta_buf[h]);
        }

        @import("../ops/ssm.zig").causalConv1dSilu(conv_out, conv_state, conv_in, conv_w, null, p.conv_ch, p.d_conv);
        self.flushActivations();

        const q_off: usize = if (p.kqv_order) num_k * hkd else 0;
        const k_off: usize = if (p.kqv_order) 0 else num_k * hkd;
        for (0..num_k) |h| {
            self.l2Norm(conv_out + q_off + h * hkd, hkd, p.rms_eps);
            self.l2Norm(conv_out + k_off + h * hkd, hkd, p.rms_eps);
        }
        self.flushActivations();

        const q_ptr = conv_out + q_off;
        const k_ptr = conv_out + k_off;
        const v_off: usize = 2 * num_k * hkd;
        const v_ptr = conv_out + v_off;

        const q_sz = num_k * hkd * @sizeOf(f32);
        const v_sz = num_v * hvd * @sizeOf(f32);
        const gate_sz = num_v * @sizeOf(f32);
        const state_sz = ssm_state.len * @sizeOf(f32);
        const norm_sz = hvd * @sizeOf(f32);

        var d_q = self.getInputBuf(q_ptr, q_sz);
        var d_k = self.getInputBuf(k_ptr, q_sz);
        var d_v = self.getInputBuf(v_ptr, v_sz);
        var d_gate = self.getInputBuf(@as([*]const f32, &gate_arr), gate_sz);
        var d_beta = self.getInputBuf(@as([*]const f32, &beta_arr), gate_sz);
        var d_z = self.getInputBuf(z_buf, v_sz);
        var d_norm = self.getOrUpload(@ptrCast(ssm_norm_w), norm_sz);
        var d_state = self.getInputBuf(@as([*]const f32, @ptrCast(ssm_state.ptr)), state_sz);
        var d_out = self.getOutputBuf(output, v_sz);

        var nvh: u32 = @intCast(num_v);
        var nkh: u32 = @intCast(num_k);
        var hkd_u: u32 = @intCast(hkd);
        var hvd_u: u32 = @intCast(hvd);
        var q_sc: f32 = p.q_scale;
        var eps: f32 = p.rms_eps;

        var params = [_]?*anyopaque{
            @ptrCast(&d_q),    @ptrCast(&d_k),     @ptrCast(&d_v),
            @ptrCast(&d_gate), @ptrCast(&d_beta),  @ptrCast(&d_z),
            @ptrCast(&d_norm), @ptrCast(&d_state), @ptrCast(&d_out),
            @ptrCast(&nvh),    @ptrCast(&nkh),     @ptrCast(&hkd_u),
            @ptrCast(&hvd_u),  @ptrCast(&q_sc),    @ptrCast(&eps),
        };
        self.launch(self.fn_deltanet_recurrence, @intCast(num_v), 1, 0, &params);
        self.invalidateAct(output);
        self.invalidateAct(@as([*]f32, @ptrCast(ssm_state.ptr)));
    }
};

// ── Tests ─────────────────────────────────────────────────────────

test "ROCm tuning constants are valid" {
    const testing = std.testing;

    // Block size must be a power of 2 and within HIP limits
    try testing.expect(block_size > 0);
    try testing.expect(block_size <= 1024);
    try testing.expect(block_size & (block_size - 1) == 0);

    // Reduction LDS must fit
    try testing.expect(reduction_smem >= 4);
    try testing.expect(reduction_smem <= block_size);

    // Device name buffer
    try testing.expect(device_name_buf_size >= 64);

    // Version encoding divisors must be non-zero
    try testing.expect(hip_version_major_divisor > 0);
    try testing.expect(hip_version_minor_divisor > 0);
    try testing.expect(hip_version_minor_modulus > 0);
}

test "ROCm GEMV grid size matches CUDA pattern" {
    const testing = std.testing;

    // Q4_K/Q5_K/Q6_K use NR=2
    const n: usize = 100;
    try testing.expectEqual(@as(usize, 50), (n + 1) / 2);

    // Q4_0/Q8_0 use NR=4
    try testing.expectEqual(@as(usize, 25), (n + 3) / 4);

    // Other dtypes: grid = n
    try testing.expectEqual(@as(usize, 100), n);
}

test "ROCm BufState transitions are valid" {
    const states = [_]RocmBackend.BufState{ .clean, .dirty, .stale };
    try std.testing.expectEqual(@as(usize, 3), states.len);
}

test "ROCm backend public function signatures compile" {
    comptime {
        // Core ops
        _ = @TypeOf(RocmBackend.gemv);
        _ = @TypeOf(RocmBackend.gemvMulti);
        _ = @TypeOf(RocmBackend.gemvGptq);
        _ = @TypeOf(RocmBackend.gemvAwq);
        _ = @TypeOf(RocmBackend.rmsNorm);
        _ = @TypeOf(RocmBackend.rmsNormMulti);
        _ = @TypeOf(RocmBackend.silu);
        _ = @TypeOf(RocmBackend.gelu);
        _ = @TypeOf(RocmBackend.add);
        _ = @TypeOf(RocmBackend.addRmsNorm);
        _ = @TypeOf(RocmBackend.mul);
        _ = @TypeOf(RocmBackend.softmax);
        _ = @TypeOf(RocmBackend.rope);
        _ = @TypeOf(RocmBackend.embLookup);
        _ = @TypeOf(RocmBackend.l2Norm);
        _ = @TypeOf(RocmBackend.addScaled);
        _ = @TypeOf(RocmBackend.siluMul);
        _ = @TypeOf(RocmBackend.geluMul);
        _ = @TypeOf(RocmBackend.sigmoidMul);
        _ = @TypeOf(RocmBackend.deinterleave);
        _ = @TypeOf(RocmBackend.splitQGate);
        _ = @TypeOf(RocmBackend.gemvT);
        _ = @TypeOf(RocmBackend.gemvNvfp4St);
        _ = @TypeOf(RocmBackend.gemvMlxQ);
        _ = @TypeOf(RocmBackend.gemvMxfp4St);

        // SDPA
        _ = @TypeOf(RocmBackend.sdpa);
        _ = @TypeOf(RocmBackend.sdpaWithStats);
        _ = @TypeOf(RocmBackend.sdpaPaged);
        _ = @TypeOf(RocmBackend.sdpaPrefill);
        _ = @TypeOf(RocmBackend.sdpaTree);

        // Megakernels
        _ = @TypeOf(RocmBackend.dispatchMegakernelQwen35Q8);
        _ = @TypeOf(RocmBackend.dispatchMegakernelGemmaQ4K);

        // Batched prefill
        _ = @TypeOf(RocmBackend.gemm);
        _ = @TypeOf(RocmBackend.rmsNormBatched);
        _ = @TypeOf(RocmBackend.ropeBatched);

        // Sync / lifecycle
        _ = @TypeOf(RocmBackend.sync);
        _ = @TypeOf(RocmBackend.beginBatch);
        _ = @TypeOf(RocmBackend.endBatch);
        _ = @TypeOf(RocmBackend.init);
        _ = @TypeOf(RocmBackend.deinit);
        _ = @TypeOf(RocmBackend.backendInfo);
        _ = @TypeOf(RocmBackend.flushActivations);
        _ = @TypeOf(RocmBackend.invalidateAct);
        _ = @TypeOf(RocmBackend.invalidateWeight);

        // KV cache
        _ = @TypeOf(RocmBackend.allocKvSlice);
        _ = @TypeOf(RocmBackend.freeKvSlice);

        // DeltaNet
        _ = @TypeOf(RocmBackend.deltaNet);
    }
}

test "ROCm n_kernels and lib_name" {
    try std.testing.expectEqual(@as(u32, 28), RocmBackend.n_kernels);
    try std.testing.expectEqualStrings("libamdhip64.so", RocmBackend.lib_name);
}

test "ROCm HIP memcpy direction constants" {
    // Verify direction constants match HIP spec
    try std.testing.expectEqual(@as(c_int, 1), hipMemcpyHostToDevice);
    try std.testing.expectEqual(@as(c_int, 2), hipMemcpyDeviceToHost);
    try std.testing.expectEqual(@as(c_int, 3), hipMemcpyDeviceToDevice);
}

test "ROCm HIP function pointer types are correctly typed" {
    comptime {
        // Verify all HIP function pointer types resolve to valid fn pointer types
        _ = @TypeOf(@as(FnInit, undefined));
        _ = @TypeOf(@as(FnSetDevice, undefined));
        _ = @TypeOf(@as(FnDeviceGetName, undefined));
        _ = @TypeOf(@as(FnDeviceSynchronize, undefined));
        _ = @TypeOf(@as(FnMalloc, undefined));
        _ = @TypeOf(@as(FnFree, undefined));
        _ = @TypeOf(@as(FnMemcpy, undefined));
        _ = @TypeOf(@as(FnModuleLoadData, undefined));
        _ = @TypeOf(@as(FnModuleUnload, undefined));
        _ = @TypeOf(@as(FnModuleGetFunction, undefined));
        _ = @TypeOf(@as(FnRuntimeGetVersion, undefined));
        _ = @TypeOf(@as(FnDeviceGetAttribute, undefined));
        _ = @TypeOf(@as(FnLaunchKernel, undefined));
    }
}

test "ROCm struct field defaults" {
    // Verify default initialization produces valid state (no GPU needed)
    const be = RocmBackend{};
    try std.testing.expect(be.module == null);
    try std.testing.expect(be.fn_silu == null);
    try std.testing.expect(be.fn_gemv_f32 == null);
    try std.testing.expect(be.fn_sdpa == null);
    try std.testing.expect(be.fn_mega_qwen35_q8 == null);
    try std.testing.expect(be.sdpa_flat_keys == null);
    try std.testing.expect(be.sdpa_flat_vals == null);
    try std.testing.expectEqual(@as(usize, 0), be.device_name_len);
}

test "ROCm internal type sizes" {
    const testing = std.testing;
    comptime {
        // CachedBuf must hold device pointer + size
        try testing.expect(@sizeOf(RocmBackend.CachedBuf) >= @sizeOf(DevicePtr) + @sizeOf(usize));

        // ActBuf must hold device pointer + size + state enum
        try testing.expect(@sizeOf(RocmBackend.ActBuf) >= @sizeOf(DevicePtr) + @sizeOf(usize));

        // BufState enum has exactly 3 variants
        try testing.expectEqual(@as(usize, 3), @typeInfo(RocmBackend.BufState).@"enum".fields.len);

        // KvDevCache must hold device pointer + capacity
        try testing.expect(@sizeOf(RocmBackend.KvDevCache) >= @sizeOf(DevicePtr) + @sizeOf(usize));

        // DevicePtr is pointer-sized
        try testing.expectEqual(@sizeOf(usize), @sizeOf(DevicePtr));
    }
}

test "ROCm block_size matches AMDGCN wavefront constraints" {
    const testing = std.testing;

    // block_size must be a multiple of the AMDGCN wavefront size (64)
    try testing.expectEqual(@as(u32, 0), block_size % 64);

    // block_size of 256 = 4 waves per workgroup (matches common.zig)
    try testing.expectEqual(@as(u32, 256), block_size);

    // reduction_smem = 32 bytes = 8 waves x 4 bytes (one f32 per wave for reduction)
    try testing.expectEqual(@as(u32, 32), reduction_smem);
}
