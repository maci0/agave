//! Device discovery for multi-GPU tensor/pipeline parallelism.
//! Enumerates available compute devices across all enabled backends.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

const max_devices: usize = 16;
const name_buf_size: usize = 64;
const cc_buf_size: usize = 16;

pub const BackendKind = enum { cpu, metal, cuda, rocm, vulkan };

pub const DeviceInfo = struct {
    backend: BackendKind,
    device_id: u32,
    name: [name_buf_size]u8 = .{0} ** name_buf_size,
    name_len: usize = 0,
    total_mem: usize = 0,
    avail_mem: usize = 0,
    is_uma: bool = false,
    compute_cap: [cc_buf_size]u8 = .{0} ** cc_buf_size,
    cc_len: usize = 0,

    pub fn displayName(self: *const DeviceInfo) []const u8 {
        return self.name[0..self.name_len];
    }

    pub fn ccString(self: *const DeviceInfo) []const u8 {
        return self.compute_cap[0..self.cc_len];
    }
};

pub const DeviceList = struct {
    devices: [max_devices]DeviceInfo = undefined,
    count: usize = 0,

    fn add(self: *DeviceList, dev: DeviceInfo) void {
        if (self.count < max_devices) {
            self.devices[self.count] = dev;
            self.count += 1;
        }
    }

    pub fn slice(self: *const DeviceList) []const DeviceInfo {
        return self.devices[0..self.count];
    }
};

pub fn enumerate() DeviceList {
    var list = DeviceList{};

    // GPU backends
    if (comptime build_options.enable_metal and builtin.os.tag == .macos) enumerateMetal(&list);
    if (comptime build_options.enable_cuda) enumerateCuda(&list);
    if (comptime build_options.enable_rocm) enumerateRocm(&list);
    if (comptime build_options.enable_vulkan) enumerateVulkan(&list);

    // CPU always available
    enumerateCpu(&list);

    return list;
}

// ── CPU ──────────────────────────────────────────────────────────────

fn enumerateCpu(list: *DeviceList) void {
    const n_threads = std.Thread.getCpuCount() catch 1;
    var dev = DeviceInfo{ .backend = .cpu, .device_id = 0 };
    const msg = std.fmt.bufPrint(&dev.name, "{d} threads", .{n_threads}) catch "";
    dev.name_len = msg.len;
    if (comptime builtin.os.tag != .freestanding) {
        const backend_mod = @import("../backend/backend.zig");
        dev.total_mem = backend_mod.detectSystemMem();
    }
    list.add(dev);
}

// ── Metal ────────────────────────────────────────────────────────────

fn enumerateMetal(list: *DeviceList) void {
    if (comptime builtin.os.tag != .macos) return;
    const objc = @import("../backend/objc.zig");

    // MTLCopyAllDevices() returns NSArray of MTLDevice
    const NSArray = objc.getClass("NSArray") orelse return;
    _ = NSArray;
    const devices_arr: ?objc.id = MTLCopyAllDevices();
    if (devices_arr == null) {
        // Fallback: single default device
        const default_dev = objc.MTLCreateSystemDefaultDevice() orelse return;
        addMetalDevice(list, default_dev, 0);
        return;
    }
    const arr = devices_arr.?;
    const count: u64 = objc.msgSend(u64, arr, objc.sel("count"), .{});
    if (count == 0) return;
    var i: u64 = 0;
    while (i < count and i < max_devices) : (i += 1) {
        const dev: objc.id = objc.msgSend(objc.id, arr, objc.sel("objectAtIndex:"), .{i});
        addMetalDevice(list, dev, @intCast(i));
    }
}

extern "c" fn MTLCopyAllDevices() ?@import("../backend/objc.zig").id;

fn addMetalDevice(list: *DeviceList, mtl_dev: @import("../backend/objc.zig").id, idx: u32) void {
    const objc = @import("../backend/objc.zig");
    var dev = DeviceInfo{ .backend = .metal, .device_id = idx, .is_uma = true };

    // Device name via ObjC: [device name] → NSString → UTF8String
    const name_ns: objc.id = objc.msgSend(objc.id, mtl_dev, objc.sel("name"), .{});
    const name_cstr: ?[*:0]const u8 = objc.msgSend(?[*:0]const u8, name_ns, objc.sel("UTF8String"), .{});
    if (name_cstr) |cstr| {
        const name_slice = std.mem.sliceTo(cstr, 0);
        const copy_len = @min(name_slice.len, name_buf_size);
        @memcpy(dev.name[0..copy_len], name_slice[0..copy_len]);
        dev.name_len = copy_len;
    }

    // Recommended max working set size (approximate VRAM)
    dev.total_mem = objc.msgSend(u64, mtl_dev, objc.sel("recommendedMaxWorkingSetSize"), .{});

    // Metal GPU family for compute cap string
    const cc = std.fmt.bufPrint(&dev.compute_cap, "Metal", .{}) catch "";
    dev.cc_len = cc.len;

    list.add(dev);
}

// ── CUDA ─────────────────────────────────────────────────────────────

fn enumerateCuda(list: *DeviceList) void {
    const CUresult = c_int;
    const CUDA_SUCCESS: CUresult = 0;
    const CUdevice = c_int;
    const FnInit = *const fn (c_uint) callconv(.c) CUresult;
    const FnGetCount = *const fn (*c_int) callconv(.c) CUresult;
    const FnGetDev = *const fn (*CUdevice, c_int) callconv(.c) CUresult;
    const FnGetName = *const fn ([*]u8, c_int, CUdevice) callconv(.c) CUresult;
    const FnGetAttr = *const fn (*c_int, c_int, CUdevice) callconv(.c) CUresult;
    const FnMemInfo = *const fn (*usize, *usize) callconv(.c) CUresult;
    const FnCtxCreate = *const fn (*?*anyopaque, c_uint, CUdevice) callconv(.c) CUresult;
    const FnCtxDestroy = *const fn (?*anyopaque) callconv(.c) CUresult;

    const cuda_lib_name = if (builtin.os.tag == .linux) "libcuda.so.1" else "libcuda.dylib";
    var lib = std.DynLib.open(cuda_lib_name) catch return;
    defer lib.close();

    const cuInit = lib.lookup(FnInit, "cuInit") orelse return;
    if (cuInit(0) != CUDA_SUCCESS) return;

    const cuDeviceGetCount = lib.lookup(FnGetCount, "cuDeviceGetCount") orelse return;
    const cuDeviceGet = lib.lookup(FnGetDev, "cuDeviceGet") orelse return;
    const cuDeviceGetName = lib.lookup(FnGetName, "cuDeviceGetName") orelse return;
    const cuDeviceGetAttribute = lib.lookup(FnGetAttr, "cuDeviceGetAttribute") orelse return;
    const cuDeviceComputeCapability_major: c_int = 75;
    const cuDeviceComputeCapability_minor: c_int = 76;
    const cuDeviceAttr_integrated: c_int = 18;

    var count: c_int = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS) return;

    // Optional: context for memory query
    const cuCtxCreate = lib.lookup(FnCtxCreate, "cuCtxCreate_v2");
    const cuCtxDestroy = lib.lookup(FnCtxDestroy, "cuCtxDestroy_v2");
    const cuMemGetInfo = lib.lookup(FnMemInfo, "cuMemGetInfo_v2");

    var i: c_int = 0;
    while (i < count and i < max_devices) : (i += 1) {
        var cuda_dev: CUdevice = 0;
        if (cuDeviceGet(&cuda_dev, i) != CUDA_SUCCESS) continue;

        var dev = DeviceInfo{ .backend = .cuda, .device_id = @intCast(i) };

        // Name
        var name_c: [name_buf_size]u8 = .{0} ** name_buf_size;
        if (cuDeviceGetName(&name_c, name_buf_size, cuda_dev) == CUDA_SUCCESS) {
            dev.name_len = std.mem.indexOfScalar(u8, &name_c, 0) orelse name_buf_size;
            @memcpy(dev.name[0..dev.name_len], name_c[0..dev.name_len]);
        }

        // Compute capability
        var sm_major: c_int = 0;
        var sm_minor: c_int = 0;
        _ = cuDeviceGetAttribute(&sm_major, cuDeviceComputeCapability_major, cuda_dev);
        _ = cuDeviceGetAttribute(&sm_minor, cuDeviceComputeCapability_minor, cuda_dev);
        const cc = std.fmt.bufPrint(&dev.compute_cap, "sm_{d}{d}", .{ sm_major, sm_minor }) catch "";
        dev.cc_len = cc.len;

        // UMA
        var integrated: c_int = 0;
        _ = cuDeviceGetAttribute(&integrated, cuDeviceAttr_integrated, cuda_dev);
        dev.is_uma = integrated != 0;

        // Memory (requires temp context)
        if (cuCtxCreate) |ctxCreate| {
            if (cuCtxDestroy) |ctxDest| {
                if (cuMemGetInfo) |memInfo| {
                    var ctx: ?*anyopaque = null;
                    if (ctxCreate(&ctx, 0, cuda_dev) == CUDA_SUCCESS) {
                        var free_mem: usize = 0;
                        var total_mem: usize = 0;
                        if (memInfo(&free_mem, &total_mem) == CUDA_SUCCESS) {
                            dev.total_mem = total_mem;
                            dev.avail_mem = free_mem;
                        }
                        _ = ctxDest(ctx);
                    }
                }
            }
        }

        list.add(dev);
    }
}

// ── ROCm ─────────────────────────────────────────────────────────────

fn enumerateRocm(list: *DeviceList) void {
    const HipResult = c_int;
    const HIP_SUCCESS: HipResult = 0;
    const FnInit = *const fn (c_uint) callconv(.c) HipResult;
    const FnGetCount = *const fn (*c_int) callconv(.c) HipResult;

    const lib_name = "libamdhip64.so";
    var lib = std.DynLib.open(lib_name) catch return;
    defer lib.close();

    const hipInit = lib.lookup(FnInit, "hipInit") orelse return;
    if (hipInit(0) != HIP_SUCCESS) return;

    const hipGetDeviceCount = lib.lookup(FnGetCount, "hipGetDeviceCount") orelse return;

    var count: c_int = 0;
    if (hipGetDeviceCount(&count) != HIP_SUCCESS) return;

    const FnSetDev = *const fn (c_int) callconv(.c) HipResult;
    const hipSetDevice = lib.lookup(FnSetDev, "hipSetDevice") orelse return;

    // hipDeviceProp_t is huge (>800 bytes). Query name via hipDeviceGetName instead.
    const FnGetName = *const fn ([*]u8, c_int, c_int) callconv(.c) HipResult;
    const hipDeviceGetName = lib.lookup(FnGetName, "hipDeviceGetName");

    const FnMemInfo = *const fn (*usize, *usize) callconv(.c) HipResult;
    const hipMemGetInfo = lib.lookup(FnMemInfo, "hipMemGetInfo");

    var i: c_int = 0;
    while (i < count and i < max_devices) : (i += 1) {
        var dev = DeviceInfo{ .backend = .rocm, .device_id = @intCast(i) };

        if (hipDeviceGetName) |getName| {
            var name_c: [name_buf_size]u8 = .{0} ** name_buf_size;
            if (getName(&name_c, name_buf_size, i) == HIP_SUCCESS) {
                dev.name_len = std.mem.indexOfScalar(u8, &name_c, 0) orelse name_buf_size;
                @memcpy(dev.name[0..dev.name_len], name_c[0..dev.name_len]);
            }
        }

        if (hipMemGetInfo) |memInfo| {
            _ = hipSetDevice(i);
            var free_mem: usize = 0;
            var total_mem: usize = 0;
            if (memInfo(&free_mem, &total_mem) == HIP_SUCCESS) {
                dev.total_mem = total_mem;
                dev.avail_mem = free_mem;
            }
        }

        list.add(dev);
    }
}

// ── Vulkan ───────────────────────────────────────────────────────────

fn enumerateVulkan(list: *DeviceList) void {
    // Vulkan enumeration requires instance creation which is heavyweight.
    // Skip for now — Vulkan devices are detected at backend init time.
    // TODO: lightweight vkEnumeratePhysicalDevices probe
    _ = list;
}

// ── Display ──────────────────────────────────────────────────────────

pub fn printDeviceTable(list: *const DeviceList) void {
    var buf: [4096]u8 = undefined;
    var pos: usize = 0;
    const header = "\nAvailable devices:\n";
    @memcpy(buf[pos..][0..header.len], header);
    pos += header.len;

    for (list.slice()) |dev| {
        const mem_gb = @as(f64, @floatFromInt(dev.total_mem)) / (1024 * 1024 * 1024);
        const uma_str: []const u8 = if (dev.is_uma) "UMA" else "   ";
        const line = std.fmt.bufPrint(buf[pos..], "  {s}:{d}  {s:<24} {d:>5.1} GB  {s}  {s}\n", .{
            @tagName(dev.backend),
            dev.device_id,
            dev.displayName(),
            mem_gb,
            uma_str,
            dev.ccString(),
        }) catch break;
        pos += line.len;
    }
    buf[pos] = '\n';
    pos += 1;
    _ = std.posix.system.write(1, buf[0..pos].ptr, pos);
}
