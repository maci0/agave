//! Open a GPU/compute dynamic library by soname, trying distro and Homebrew
//! prefixes when the unadorned name is not on the loader path.
//!
//! `dlopen(soname)` is enough when ldconfig/DYLD is set up. The extra prefixes
//! cover Debian multiarch, Fedora `/usr/lib64`, Alpine `/lib`, the CUDA toolkit,
//! ROCm, and macOS Homebrew, which otherwise miss `--list-devices` and backend
//! init on a claimed host.

const std = @import("std");

const path_buf_size: usize = 256;

/// Directories tried after the unadorned soname. Empty string means soname only.
const dynlib_dir_prefixes = [_][]const u8{
    "",
    "/usr/lib64/",
    "/usr/lib/",
    "/usr/lib/x86_64-linux-gnu/",
    "/usr/lib/aarch64-linux-gnu/",
    "/lib/x86_64-linux-gnu/",
    "/lib/aarch64-linux-gnu/",
    "/lib/",
    "/usr/local/lib/",
    "/opt/homebrew/lib/",
    "/usr/local/cuda/lib64/",
    "/usr/local/cuda/lib/",
    "/opt/rocm/lib/",
};

/// Open `soname` from the first location that succeeds. Null if none load.
pub fn open(soname: [:0]const u8) ?std.DynLib {
    var path_buf: [path_buf_size]u8 = undefined;
    for (dynlib_dir_prefixes) |prefix| {
        const path: [:0]const u8 = if (prefix.len == 0) soname else blk: {
            const n = prefix.len + soname.len;
            if (n >= path_buf.len) continue;
            @memcpy(path_buf[0..prefix.len], prefix);
            @memcpy(path_buf[prefix.len..][0..soname.len], soname);
            path_buf[n] = 0;
            break :blk path_buf[0..n :0];
        };
        if (std.DynLib.open(path)) |lib| return lib else |_| continue;
    }
    return null;
}

test "open, missing library is null" {
    try std.testing.expect(open("agave-no-such-library.so") == null);
}

test "prefixes include Homebrew, Fedora lib64, and Alpine lib" {
    var brew = false;
    var lib64 = false;
    var alpine_lib = false;
    for (dynlib_dir_prefixes) |p| {
        if (std.mem.eql(u8, p, "/opt/homebrew/lib/")) brew = true;
        if (std.mem.eql(u8, p, "/usr/lib64/")) lib64 = true;
        if (std.mem.eql(u8, p, "/lib/")) alpine_lib = true;
    }
    try std.testing.expect(brew);
    try std.testing.expect(lib64);
    try std.testing.expect(alpine_lib);
}
