//! DS V4 Flash Multi-Token Prediction (MTP) layer.
//!
//! Loads MTP weights from a separate safetensors file and provides
//! tensor lookup for mtpForward() in deepseek4.zig.
//! The safetensors file is mmap'd; tensor data pointers reference the mmap.

const std = @import("std");
const posix = std.posix;
const Allocator = std.mem.Allocator;
const format_mod = @import("../format/format.zig");
const DType = format_mod.DType;

/// Maximum MTP depth (number of draft tokens per target forward).
pub const max_mtp_depth: u32 = 3;

/// Lightweight tensor reference into mmap'd safetensors data.
pub const MtpTensor = struct {
    data_ptr: [*]const u8,
    dtype: DType,
    shape: [4]u64 = .{ 0, 0, 0, 0 },
    n_dims: u32 = 0,
};

/// MTP weight storage, mmap'd safetensors with tensor name → pointer lookup.
pub const MtpWeights = struct {
    mmap_ptr: ?[*]align(std.heap.page_size_min) const u8 = null,
    mmap_len: usize = 0,
    /// Tensor name → MtpTensor lookup
    tensors: std.StringHashMap(MtpTensor),
    /// Number of MTP depths detected (0, 1, 2, or 3)
    n_depths: u32 = 0,

    pub fn init(allocator: Allocator) MtpWeights {
        return .{ .tensors = std.StringHashMap(MtpTensor).init(allocator) };
    }

    /// Load MTP weights from a safetensors file path.
    pub fn load(self: *MtpWeights, allocator: Allocator, path: anytype) !void {
        // Open file
        const fd = posix.openat(posix.AT.FDCWD, path, .{}, 0) catch return error.FileNotFound;
        defer {
            if (comptime @import("builtin").os.tag == .linux)
                _ = posix.system.close(fd)
            else
                _ = std.c.close(fd);
        }

        // Get file size, same pattern as gguf.zig: statx on Linux (posix fstat
        // wrappers were removed in Zig 0.16), std.c.fstat elsewhere.
        const file_size: usize = blk: {
            if (comptime @import("builtin").os.tag == .linux) {
                var buf: std.os.linux.Statx = undefined;
                const rc = std.os.linux.statx(fd, @ptrCast(""), std.os.linux.AT.EMPTY_PATH, std.os.linux.STATX{ .SIZE = true }, &buf);
                if (rc != 0) return error.FileNotFound;
                break :blk @intCast(buf.size);
            } else {
                var s: posix.Stat = undefined;
                if (std.c.fstat(fd, &s) != 0) return error.FileNotFound;
                break :blk @intCast(s.size);
            }
        };

        // mmap the entire file
        const mapped = try posix.mmap(null, file_size, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0);
        self.mmap_ptr = mapped.ptr;
        self.mmap_len = file_size;

        // Parse safetensors header
        const header_size = std.mem.readInt(u64, mapped[0..8], .little);
        const header_end = 8 + @as(usize, @intCast(header_size));
        if (header_end > file_size) return error.InvalidFormat;

        const header_json = mapped[8..header_end];
        const data_base = mapped.ptr + header_end;

        // Parse JSON to extract tensor metadata
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, header_json, .{});
        defer parsed.deinit();

        const root = parsed.value.object;
        var max_depth: u32 = 0;
        var count: u32 = 0;

        var it = root.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (std.mem.eql(u8, name, "__metadata__")) continue;

            const obj = entry.value_ptr.object;
            const dtype_str = obj.get("dtype").?.string;
            const shape_arr = obj.get("shape").?.array;
            const offsets_arr = obj.get("data_offsets").?.array;

            const start: usize = @intCast(offsets_arr.items[0].integer);

            const dtype = parseDtype(dtype_str);
            var shape: [4]u64 = .{ 0, 0, 0, 0 };
            const n_dims: u32 = @intCast(@min(shape_arr.items.len, 4));
            for (0..n_dims) |i| shape[i] = @intCast(shape_arr.items[i].integer);

            if (std.mem.startsWith(u8, name, "mtp.")) {
                const depth_char = name[4];
                if (depth_char >= '0' and depth_char <= '9') {
                    const d = depth_char - '0' + 1;
                    if (d > max_depth) max_depth = d;
                }
            }

            const name_owned = try allocator.dupe(u8, name);
            try self.tensors.put(name_owned, .{
                .data_ptr = data_base + start,
                .dtype = dtype,
                .shape = shape,
                .n_dims = n_dims,
            });
            count += 1;
        }

        self.n_depths = max_depth;
        std.log.info("MTP: loaded {d} tensors, {d} depths, {d:.1}MB from safetensors", .{
            count, max_depth, @as(f64, @floatFromInt(file_size)) / 1e6,
        });
    }

    /// Look up a tensor by its HF name (e.g., "mtp.0.attn.wq_a.weight").
    pub fn get(self: *const MtpWeights, name: []const u8) ?MtpTensor {
        return self.tensors.get(name);
    }

    pub fn deinit(self: *MtpWeights, allocator: Allocator) void {
        var kit = self.tensors.keyIterator();
        while (kit.next()) |k| allocator.free(k.*);
        self.tensors.deinit();
        if (self.mmap_ptr) |ptr| {
            const slice = @as([*]align(std.heap.page_size_min) const u8, @alignCast(ptr))[0..self.mmap_len];
            posix.munmap(slice);
        }
    }

    fn parseDtype(s: []const u8) DType {
        if (std.mem.eql(u8, s, "F32")) return .f32;
        if (std.mem.eql(u8, s, "F16")) return .f16;
        if (std.mem.eql(u8, s, "BF16")) return .bf16;
        if (std.mem.eql(u8, s, "F8_E4M3")) return .fp8_e4m3;
        if (std.mem.eql(u8, s, "F8_E8M0")) return .unknown; // scale type
        if (std.mem.eql(u8, s, "I8")) return .unknown; // MXFP4 packed
        return .unknown;
    }
};
