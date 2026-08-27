//! Minimal CUDA driver-API repro in Zig — replicates the agave's init +
//! first-gemv sequence to isolate a Zig-runtime vs agave-code interaction.
//! Build (native): zig build-exe repro.zig -O ReleaseFast
const std = @import("std");
const libc = std.c;

const CUresult = c_int;
const CUdevice = c_int;
const CUcontext = ?*anyopaque;
const CUdeviceptr = u64;
const CUmodule = ?*anyopaque;
const CUfunction = ?*anyopaque;
const CUstream = ?*anyopaque;

fn lookup(comptime T: type, lib: *std.DynLib, name: [:0]const u8) T {
    return @ptrCast(@alignCast(@constCast(lib.lookup(?*anyopaque, name) orelse @panic("missing symbol"))));
}

pub fn main() !void {
    const alloc = std.heap.page_allocator;

    var lib = try std.DynLib.open("/usr/lib/aarch64-linux-gnu/libcuda.so.1");
    const cuInit = lookup(*const fn (c_uint) callconv(.c) CUresult, &lib, "cuInit");
    const cuDeviceGet = lookup(*const fn (*CUdevice, c_int) callconv(.c) CUresult, &lib, "cuDeviceGet");
    const cuPrimaryRetain = lookup(*const fn (*CUcontext, CUdevice) callconv(.c) CUresult, &lib, "cuDevicePrimaryCtxRetain");
    const cuCtxSetCurrent = lookup(*const fn (CUcontext) callconv(.c) CUresult, &lib, "cuCtxSetCurrent");
    const cuMemAlloc = lookup(*const fn (*CUdeviceptr, usize) callconv(.c) CUresult, &lib, "cuMemAlloc_v2");
    const cuMemcpyHtoD = lookup(*const fn (CUdeviceptr, *const anyopaque, usize) callconv(.c) CUresult, &lib, "cuMemcpyHtoD_v2");
    const cuMemcpyDtoH = lookup(*const fn (*anyopaque, CUdeviceptr, usize) callconv(.c) CUresult, &lib, "cuMemcpyDtoH_v2");
    const cuModuleLoad = lookup(*const fn (*CUmodule, *const anyopaque) callconv(.c) CUresult, &lib, "cuModuleLoadData");
    const cuGetFn = lookup(*const fn (*CUfunction, CUmodule, [*:0]const u8) callconv(.c) CUresult, &lib, "cuModuleGetFunction");
    const cuLaunch = lookup(*const fn (CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, [*]?*anyopaque, ?[*]?*anyopaque) callconv(.c) CUresult, &lib, "cuLaunchKernel");
    const cuCtxSync = lookup(*const fn () callconv(.c) CUresult, &lib, "cuCtxSynchronize");
    _ = cuCtxSync;
    const cuStreamCreate = lookup(*const fn (*CUstream, c_uint) callconv(.c) CUresult, &lib, "cuStreamCreate");
    const cuStreamSync = lookup(*const fn (CUstream) callconv(.c) CUresult, &lib, "cuStreamSynchronize");

    std.debug.print("STEP init\n", .{});
    var ok = true;
    ok = ok and cuInit(0) == 0;
    std.debug.print("STEP init done\n", .{});
    var dev: CUdevice = 0;
    ok = ok and cuDeviceGet(&dev, 0) == 0;
    std.debug.print("STEP device rc={d}\n", .{@intFromBool(ok)});
    var ctx: CUcontext = null;
    const rr = cuPrimaryRetain(&ctx, dev);
    ok = ok and rr == 0;
    _ = cuCtxSetCurrent(ctx);
    std.debug.print("ctx retain rc={d}\n", .{rr});

    // Module: the agave's PTX.
    const ptx_src = @embedFile("all.ptx");
    var mod: CUmodule = null;
    const mr = cuModuleLoad(&mod, ptx_src.ptr);
    ok = ok and mr == 0;
    std.debug.print("module rc={d}\n", .{mr});
    var fn_gemv: CUfunction = null;
    const fr = cuGetFn(&fn_gemv, mod, "gemv_f32_kernel");
    ok = ok and fr == 0;
    std.debug.print("getfn rc={d}\n", .{fr});

    // Alloc + upload (hc gemv: n=24, k=2048).
    const n: u32 = 24;
    const k: u32 = 2048;
    var dx: CUdeviceptr = 0;
    var dw: CUdeviceptr = 0;
    var dy: CUdeviceptr = 0;
    ok = ok and cuMemAlloc(&dx, k * 4) == 0;
    ok = ok and cuMemAlloc(&dw, n * k * 4) == 0;
    ok = ok and cuMemAlloc(&dy, n * 4) == 0;
    const hw = try alloc.alloc(f32, n * k);
    const hx = try alloc.alloc(f32, k);
    for (hw) |*v| v.* = 1.0;
    for (hx) |*v| v.* = 1.0;
    ok = ok and cuMemcpyHtoD(dw, hw.ptr, n * k * 4) == 0;
    ok = ok and cuMemcpyHtoD(dx, hx.ptr, k * 4) == 0;
    std.debug.print("allocs+uploads rc={d}\n", .{@intFromBool(ok)});

    // Launch + sync (on a created stream like the agave).
    var stream: CUstream = null;
    const scr = cuStreamCreate(&stream, 0);
    ok = ok and scr == 0;
    std.debug.print("stream create rc={d}\n", .{scr});
    var nv = n;
    var kv = k;
    var params = [_]?*anyopaque{ @ptrCast(&dx), @ptrCast(&dw), @ptrCast(&dy), @ptrCast(&nv), @ptrCast(&kv) };
    const lr = cuLaunch(fn_gemv, n, 1, 1, 256, 1, 1, 32, stream, &params, null);
    ok = ok and lr == 0;
    std.debug.print("launch rc={d}\n", .{lr});
    const sr = cuStreamSync(stream);
    ok = ok and sr == 0;
    std.debug.print("stream sync rc={d}\n", .{sr});

    // D2H to a Zig heap buffer.
    const hy = try alloc.alloc(f32, n);
    const dr = cuMemcpyDtoH(hy.ptr, dy, n * 4);
    ok = ok and dr == 0;
    std.debug.print("D2H rc={d} hy[0]={d} (expect 2048)\n", .{ dr, @as(f64, @floatCast(hy[0])) });

    std.debug.print("RESULT: {s}\n", .{if (ok) "ALL PASS" else "FAILED"});
}
