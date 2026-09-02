//! Device discovery for multi-GPU tensor/pipeline parallelism.
//! Enumerates available compute devices across all enabled backends.
//!
//! Distinct from `parallel/peer_discovery.zig` (UDP LAN peer join for TP/PP).
//! This module only lists local GPUs/CPUs for `--list-devices` and `--device N`.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

const max_devices: usize = 16;
const name_buf_size: usize = 64;
const cc_buf_size: usize = 16;

/// Discrete / host backends that participate in `--list-devices` and TP/PP topology.
/// WebGPU is intentionally omitted: it exposes one logical adapter (browser or wgpu),
/// not a multi-device mesh. Inference still selects it via `BackendChoice.webgpu`.
pub const BackendKind = enum { cpu, metal, cuda, rocm, vulkan };

/// Describes a single compute device (GPU or CPU) discovered on the local host.
/// Backends populate these during `enumerate` for `--list-devices` and TP/PP topology selection.
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

    /// Returns the human-readable device name (e.g. "Apple M2 Max") as a slice.
    pub fn displayName(self: *const DeviceInfo) []const u8 {
        return self.name[0..self.name_len];
    }

    /// Returns the compute-capability string (e.g. "sm_90" for CUDA) as a slice.
    /// Empty for backends that do not report a compute capability.
    pub fn ccString(self: *const DeviceInfo) []const u8 {
        return self.compute_cap[0..self.cc_len];
    }
};

/// Fixed-capacity list of discovered compute devices (up to 16).
/// Populated by `enumerate` and consumed by `--list-devices` output and backend selection.
pub const DeviceList = struct {
    devices: [max_devices]DeviceInfo = undefined,
    count: usize = 0,

    fn add(self: *DeviceList, dev: DeviceInfo) void {
        if (self.count < max_devices) {
            self.devices[self.count] = dev;
            self.count += 1;
        } else {
            std.log.warn("device list full ({d}); dropping {s} device_id={d}", .{
                max_devices,
                @tagName(dev.backend),
                dev.device_id,
            });
        }
    }

    /// Returns the populated portion of the device array as a const slice.
    pub fn slice(self: *const DeviceList) []const DeviceInfo {
        return self.devices[0..self.count];
    }
};

/// Probes all enabled backends (Metal, CUDA, ROCm, Vulkan) and the CPU,
/// returning a `DeviceList` of every device found on this host.
/// Always includes at least one entry (CPU). Used by `--list-devices` and device selection.
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
    const backend_mod = @import("../backend/backend.zig");
    var buf: [max_devices]backend_mod.MetalDeviceListEntry = undefined;
    const n = backend_mod.listMetalDevices(&buf);
    for (buf[0..n], 0..) |src, idx| {
        var dev = DeviceInfo{ .backend = .metal, .device_id = @intCast(idx), .is_uma = true };
        const copy_len = @min(src.name_len, name_buf_size);
        @memcpy(dev.name[0..copy_len], src.name[0..copy_len]);
        dev.name_len = copy_len;
        dev.total_mem = src.total_mem;
        const cc = std.fmt.bufPrint(&dev.compute_cap, "Metal", .{}) catch "";
        dev.cc_len = cc.len;
        list.add(dev);
    }
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
    var lib = @import("../dynlib.zig").open(cuda_lib_name) orelse return;
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
    var lib = @import("../dynlib.zig").open(lib_name) orelse return;
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
    const VkResult = c_int;
    const VK_SUCCESS: VkResult = 0;
    const VkInstance = ?*anyopaque;
    const VkPhysicalDevice = ?*anyopaque;

    const VkApplicationInfo = extern struct {
        sType: u32 = 0, // VK_STRUCTURE_TYPE_APPLICATION_INFO
        pNext: ?*const anyopaque = null,
        pApplicationName: ?[*:0]const u8 = null,
        applicationVersion: u32 = 0,
        pEngineName: ?[*:0]const u8 = null,
        engineVersion: u32 = 0,
        apiVersion: u32 = 0,
    };
    const VkInstanceCreateInfo = extern struct {
        sType: u32 = 1, // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
        pNext: ?*const anyopaque = null,
        flags: u32 = 0,
        pApplicationInfo: ?*const VkApplicationInfo = null,
        enabledLayerCount: u32 = 0,
        ppEnabledLayerNames: ?*const ?[*:0]const u8 = null,
        enabledExtensionCount: u32 = 0,
        ppEnabledExtensionNames: ?*const ?[*:0]const u8 = null,
    };
    const VkPhysicalDeviceProperties = extern struct {
        apiVersion: u32 = 0,
        driverVersion: u32 = 0,
        vendorID: u32 = 0,
        deviceID: u32 = 0,
        deviceType: u32 = 0,
        deviceName: [256]u8 = .{0} ** 256,
        pipelineCacheUUID: [16]u8 = .{0} ** 16,
        limits: [504]u8 = .{0} ** 504,
        sparseProperties: [20]u8 = .{0} ** 20,
    };
    const VkPhysicalDeviceMemoryProperties = extern struct {
        memoryTypeCount: u32 = 0,
        memoryTypes: [32 * 8]u8 = .{0} ** (32 * 8),
        memoryHeapCount: u32 = 0,
        memoryHeaps: [16 * 16]u8 = .{0} ** (16 * 16),
    };

    const FnCreateInstance = *const fn (*const VkInstanceCreateInfo, ?*const anyopaque, *VkInstance) callconv(.c) VkResult;
    const FnDestroyInstance = *const fn (VkInstance, ?*const anyopaque) callconv(.c) void;
    const FnEnumPhysDevices = *const fn (VkInstance, *u32, ?[*]VkPhysicalDevice) callconv(.c) VkResult;
    const FnGetPhysDevProps = *const fn (VkPhysicalDevice, *VkPhysicalDeviceProperties) callconv(.c) void;
    const FnGetPhysDevMemProps = *const fn (VkPhysicalDevice, *VkPhysicalDeviceMemoryProperties) callconv(.c) void;

    const vk_lib_name = if (builtin.os.tag == .macos) "libvulkan.1.dylib" else "libvulkan.so.1";
    var lib = @import("../dynlib.zig").open(vk_lib_name) orelse return;
    defer lib.close();

    const vkCreateInstance = lib.lookup(FnCreateInstance, "vkCreateInstance") orelse return;
    const vkDestroyInstance = lib.lookup(FnDestroyInstance, "vkDestroyInstance") orelse return;
    const vkEnumeratePhysicalDevices = lib.lookup(FnEnumPhysDevices, "vkEnumeratePhysicalDevices") orelse return;
    const vkGetPhysicalDeviceProperties = lib.lookup(FnGetPhysDevProps, "vkGetPhysicalDeviceProperties") orelse return;

    const app_info = VkApplicationInfo{ .pApplicationName = "agave-probe", .apiVersion = (1 << 22) | (0 << 12) };
    const ci = VkInstanceCreateInfo{ .pApplicationInfo = &app_info };
    var instance: VkInstance = null;
    if (vkCreateInstance(&ci, null, &instance) != VK_SUCCESS) return;
    defer vkDestroyInstance(instance, null);

    var count: u32 = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, null) != VK_SUCCESS) return;
    if (count == 0) return;

    var phys_devs: [max_devices]VkPhysicalDevice = .{null} ** max_devices;
    var n: u32 = @intCast(@min(count, max_devices));
    if (vkEnumeratePhysicalDevices(instance, &n, &phys_devs) != VK_SUCCESS) return;

    const vkGetPhysicalDeviceMemoryProperties = lib.lookup(FnGetPhysDevMemProps, "vkGetPhysicalDeviceMemoryProperties");

    for (0..n) |i| {
        var props: VkPhysicalDeviceProperties = .{};
        vkGetPhysicalDeviceProperties(phys_devs[i], &props);

        var dev = DeviceInfo{ .backend = .vulkan, .device_id = @intCast(i) };
        dev.name_len = std.mem.indexOfScalar(u8, &props.deviceName, 0) orelse props.deviceName.len;
        @memcpy(dev.name[0..@min(dev.name_len, name_buf_size)], props.deviceName[0..@min(dev.name_len, name_buf_size)]);
        if (dev.name_len > name_buf_size) dev.name_len = name_buf_size;

        // Memory, sum device-local heaps
        if (vkGetPhysicalDeviceMemoryProperties) |getMemProps| {
            var mem_props: VkPhysicalDeviceMemoryProperties = .{};
            getMemProps(phys_devs[i], &mem_props);
            const heap_count = @min(mem_props.memoryHeapCount, 16);
            for (0..heap_count) |hi| {
                const heap_base = hi * 16;
                const heap_size = std.mem.readInt(u64, mem_props.memoryHeaps[heap_base..][0..8], .little);
                const heap_flags = std.mem.readInt(u32, mem_props.memoryHeaps[heap_base + 8 ..][0..4], .little);
                if (heap_flags & 1 != 0) dev.total_mem += heap_size; // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
            }
        }

        list.add(dev);
    }
}

// ── Display ──────────────────────────────────────────────────────────

// ── Tests ───────────────────────────────────────────────────────────

test "DeviceInfo, displayName returns name slice" {
    var dev = DeviceInfo{ .backend = .cpu, .device_id = 0 };
    const name = "Test GPU";
    @memcpy(dev.name[0..name.len], name);
    dev.name_len = name.len;
    try @import("std").testing.expectEqualStrings("Test GPU", dev.displayName());
}

test "DeviceInfo, displayName empty when no name" {
    const dev = DeviceInfo{ .backend = .cpu, .device_id = 0 };
    try @import("std").testing.expectEqual(@as(usize, 0), dev.displayName().len);
}

test "DeviceInfo, ccString" {
    var dev = DeviceInfo{ .backend = .cuda, .device_id = 0 };
    const cc = "sm_121";
    @memcpy(dev.compute_cap[0..cc.len], cc);
    dev.cc_len = cc.len;
    try @import("std").testing.expectEqualStrings("sm_121", dev.ccString());
}

test "DeviceInfo, default values" {
    const dev = DeviceInfo{ .backend = .vulkan, .device_id = 3 };
    try @import("std").testing.expectEqual(BackendKind.vulkan, dev.backend);
    try @import("std").testing.expectEqual(@as(u32, 3), dev.device_id);
    try @import("std").testing.expectEqual(@as(usize, 0), dev.total_mem);
    try @import("std").testing.expectEqual(@as(usize, 0), dev.avail_mem);
    try @import("std").testing.expect(!dev.is_uma);
}

test "DeviceList, add and slice" {
    var list = DeviceList{};
    try @import("std").testing.expectEqual(@as(usize, 0), list.count);
    try @import("std").testing.expectEqual(@as(usize, 0), list.slice().len);

    list.add(.{ .backend = .cpu, .device_id = 0 });
    try @import("std").testing.expectEqual(@as(usize, 1), list.count);
    try @import("std").testing.expectEqual(@as(usize, 1), list.slice().len);
    try @import("std").testing.expectEqual(BackendKind.cpu, list.slice()[0].backend);

    list.add(.{ .backend = .metal, .device_id = 0, .is_uma = true });
    try @import("std").testing.expectEqual(@as(usize, 2), list.count);
    try @import("std").testing.expectEqual(BackendKind.metal, list.slice()[1].backend);
    try @import("std").testing.expect(list.slice()[1].is_uma);
}

test "DeviceList, max capacity enforcement" {
    var list = DeviceList{};
    // Fill to max_devices.
    for (0..max_devices) |i| {
        list.add(.{ .backend = .cpu, .device_id = @intCast(i) });
    }
    try @import("std").testing.expectEqual(max_devices, list.count);

    // Adding beyond max should be ignored (and logged at warn).
    list.add(.{ .backend = .cuda, .device_id = 99 });
    try @import("std").testing.expectEqual(max_devices, list.count);
}

test "DeviceList, max_devices constant" {
    try @import("std").testing.expectEqual(@as(usize, 16), max_devices);
}

test "DeviceInfo, name buffer size" {
    try @import("std").testing.expectEqual(@as(usize, 64), name_buf_size);
    try @import("std").testing.expectEqual(@as(usize, 16), cc_buf_size);
}

test "BackendKind, all variants" {
    const fields = @typeInfo(BackendKind).@"enum".fields;
    const expected = [_][]const u8{ "cpu", "metal", "cuda", "rocm", "vulkan" };
    try std.testing.expectEqual(expected.len, fields.len);
    inline for (expected, 0..) |name, i| {
        try std.testing.expectEqualStrings(name, fields[i].name);
    }
}

test "enumerate, always includes CPU" {
    const list = enumerate();
    try @import("std").testing.expect(list.count >= 1);
    // Find CPU in the list (always last, added by enumerateCpu).
    var found_cpu = false;
    for (list.slice()) |dev| {
        if (dev.backend == .cpu) {
            found_cpu = true;
            break;
        }
    }
    try @import("std").testing.expect(found_cpu);
}

test "DeviceInfo, displayName with bufPrint" {
    var dev = DeviceInfo{ .backend = .cpu, .device_id = 0 };
    const msg = std.fmt.bufPrint(&dev.name, "{d} threads", .{@as(u32, 12)}) catch "";
    dev.name_len = msg.len;
    try @import("std").testing.expectEqualStrings("12 threads", dev.displayName());
}

test "DeviceInfo, full construction with all fields" {
    var dev = DeviceInfo{
        .backend = .cuda,
        .device_id = 2,
        .is_uma = true,
        .total_mem = 16 * 1024 * 1024 * 1024, // 16 GB
        .avail_mem = 12 * 1024 * 1024 * 1024, // 12 GB
    };
    const name = "NVIDIA GB10";
    @memcpy(dev.name[0..name.len], name);
    dev.name_len = name.len;
    const cc = "sm_121";
    @memcpy(dev.compute_cap[0..cc.len], cc);
    dev.cc_len = cc.len;

    try std.testing.expectEqual(BackendKind.cuda, dev.backend);
    try std.testing.expectEqual(@as(u32, 2), dev.device_id);
    try std.testing.expect(dev.is_uma);
    try std.testing.expectEqual(@as(usize, 16 * 1024 * 1024 * 1024), dev.total_mem);
    try std.testing.expectEqual(@as(usize, 12 * 1024 * 1024 * 1024), dev.avail_mem);
    try std.testing.expectEqualStrings("NVIDIA GB10", dev.displayName());
    try std.testing.expectEqualStrings("sm_121", dev.ccString());
}

test "DeviceList, slice preserves insertion order" {
    var list = DeviceList{};
    list.add(.{ .backend = .metal, .device_id = 0, .is_uma = true });
    list.add(.{ .backend = .vulkan, .device_id = 1 });
    list.add(.{ .backend = .cpu, .device_id = 0 });

    const s = list.slice();
    try std.testing.expectEqual(@as(usize, 3), s.len);
    try std.testing.expectEqual(BackendKind.metal, s[0].backend);
    try std.testing.expectEqual(@as(u32, 0), s[0].device_id);
    try std.testing.expectEqual(BackendKind.vulkan, s[1].backend);
    try std.testing.expectEqual(@as(u32, 1), s[1].device_id);
    try std.testing.expectEqual(BackendKind.cpu, s[2].backend);
}

test "enumerate, returns DeviceList with at least one device" {
    const list = enumerate();
    // enumerate() always adds CPU, so count >= 1
    try std.testing.expect(list.count >= 1);
    // The slice length must match count
    try std.testing.expectEqual(list.count, list.slice().len);
    // Last device should be CPU (enumerateCpu is called last)
    const last = list.slice()[list.count - 1];
    try std.testing.expectEqual(BackendKind.cpu, last.backend);
    // CPU device should have a name with "threads" in it
    const cpu_name = last.displayName();
    try std.testing.expect(cpu_name.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, cpu_name, "threads") != null);
}

test "printDeviceTable writes a table for mixed devices" {
    const test_stdout = @import("../test_stdout.zig");
    const silencer = try test_stdout.Silencer.init();
    defer silencer.release();

    var list = DeviceList{};
    var cpu = DeviceInfo{ .backend = .cpu, .device_id = 0 };
    const cpu_name = "12 threads";
    @memcpy(cpu.name[0..cpu_name.len], cpu_name);
    cpu.name_len = cpu_name.len;
    list.add(cpu);

    var gpu = DeviceInfo{
        .backend = .cuda,
        .device_id = 1,
        .is_uma = true,
        .total_mem = 16 * 1024 * 1024 * 1024,
    };
    const gpu_name = "NVIDIA GB10";
    @memcpy(gpu.name[0..gpu_name.len], gpu_name);
    gpu.name_len = gpu_name.len;
    const cc = "sm_121";
    @memcpy(gpu.compute_cap[0..cc.len], cc);
    gpu.cc_len = cc.len;
    list.add(gpu);

    printDeviceTable(&list);
    try std.testing.expectEqual(@as(usize, 2), list.count);
}

/// Formats and writes a human-readable device table to stdout.
/// Used by the `--list-devices` CLI flag to display backend, name, memory, UMA, and compute capability.
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
        if (pos >= buf.len) break;
    }
    if (pos < buf.len) {
        buf[pos] = '\n';
        pos += 1;
    }
    _ = std.posix.system.write(1, buf[0..pos].ptr, pos);
}

test "fuzz: all discovery functions" {
    const test_stdout = @import("../test_stdout.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // fd 1 is the test-runner protocol pipe under the server-mode runner;
            // printDeviceTable writes plain text there and would wedge the build.
            const silencer = try test_stdout.Silencer.init();
            defer silencer.release();

            // -- DeviceInfo.displayName --
            var dev = DeviceInfo{ .backend = .cpu, .device_id = smith.valueWithHash(u32, 0) };
            const name_len_raw = smith.valueWithHash(u8, 1);
            dev.name_len = @min(name_len_raw, name_buf_size);
            // Fill name buffer with random bytes
            for (0..dev.name_len) |i| {
                dev.name[i] = smith.valueWithHash(u8, @intCast(i + 100));
            }
            const dn = dev.displayName();
            try std.testing.expectEqual(dev.name_len, dn.len);

            // -- DeviceInfo.ccString --
            const cc_len_raw = smith.valueWithHash(u8, 2);
            dev.cc_len = @min(cc_len_raw, cc_buf_size);
            for (0..dev.cc_len) |i| {
                dev.compute_cap[i] = smith.valueWithHash(u8, @intCast(i + 200));
            }
            const cs = dev.ccString();
            try std.testing.expectEqual(dev.cc_len, cs.len);

            // -- DeviceList.slice --
            var list = DeviceList{};
            const n_devs = smith.valueWithHash(u8, 3) % (max_devices + 2); // may exceed max
            for (0..n_devs) |j| {
                const backend_idx = smith.valueWithHash(u8, @intCast(j + 300)) % 5;
                const backend: BackendKind = @enumFromInt(backend_idx);
                list.add(.{ .backend = backend, .device_id = smith.valueWithHash(u32, @intCast(j + 400)) });
            }
            const sl = list.slice();
            try std.testing.expectEqual(list.count, sl.len);
            try std.testing.expect(list.count <= max_devices);

            // -- enumerate --
            const enum_list = enumerate();
            try std.testing.expect(enum_list.count >= 1);
            // CPU always present
            var found_cpu = false;
            for (enum_list.slice()) |d| {
                if (d.backend == .cpu) found_cpu = true;
            }
            try std.testing.expect(found_cpu);

            // -- printDeviceTable: formats the fuzzed name/cc bytes; output goes to /dev/null --
            var table_list = DeviceList{};
            table_list.add(dev);
            table_list.add(.{ .backend = .cpu, .device_id = 0 });
            printDeviceTable(&table_list);
        }
    }.f, .{});
}
