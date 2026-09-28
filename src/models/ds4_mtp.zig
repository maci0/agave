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

/// Bytes before the safetensors JSON header: the 8-byte little-endian length.
const safetensors_header_prefix_len: usize = 8;

/// A safetensors data_offsets entry as a byte offset into the data section.
/// Non-integral and negative values are not offsets, so they are rejected
/// rather than rounded: a truncating cast would point a read somewhere else
/// in the mapping.
fn toOffset(v: std.json.Value) ?usize {
    return switch (v) {
        .integer => |i| std.math.cast(usize, i),
        else => null,
    };
}

/// Bytes a tensor of `shape` and `dtype` occupies. `.unknown` has no fixed
/// width here (MXFP4 packs four values per byte), so it validates the range
/// only, which the offset check already does.
fn shapeBytes(shape: []const u64, dtype: DType) usize {
    const elem: usize = switch (dtype) {
        .f32 => 4,
        .f16, .bf16 => 2,
        .fp8_e4m3, .fp8_e5m2 => 1,
        else => return 0,
    };
    var total: usize = elem;
    for (shape) |dim| {
        const d = std.math.cast(usize, dim) orelse return std.math.maxInt(usize);
        total = std.math.mul(usize, total, d) catch return std.math.maxInt(usize);
    }
    return total;
}

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
        // Every later step can fail after the mapping and the dupe'd tensor
        // names exist, and the caller keeps the struct, so release them here
        // and leave a clean, reusable value behind.
        errdefer {
            self.deinit(allocator);
            self.* = init(allocator);
        }
        self.mmap_ptr = mapped.ptr;
        self.mmap_len = file_size;

        // Parse safetensors header. The file is untrusted (any downloaded MTP
        // checkpoint), so both the header length and every tensor extent are
        // checked with std.math: a wrapping sum would pass the `> file_size`
        // test and then slice the mapping out of bounds.
        if (file_size < safetensors_header_prefix_len) return error.InvalidFormat;
        const header_size = std.mem.readInt(u64, mapped[0..8], .little);
        const header_end = std.math.add(usize, safetensors_header_prefix_len, @intCast(header_size)) catch return error.InvalidFormat;
        if (header_end > file_size) return error.InvalidFormat;

        const header_json = mapped[safetensors_header_prefix_len..header_end];
        const data_base = mapped.ptr + header_end;
        const data_len = file_size - header_end;

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
            const dtype_val = obj.get("dtype") orelse continue;
            const shape_val = obj.get("shape") orelse continue;
            const offsets_val = obj.get("data_offsets") orelse continue;
            if (dtype_val != .string or shape_val != .array or offsets_val != .array) continue;
            const dtype_str = dtype_val.string;
            const shape_arr = shape_val.array;
            const offsets_arr = offsets_val.array;
            // A tensor is [start, end); a truncated pair would leave the byte
            // length the consumer multiplies out of shape undefined.
            if (offsets_arr.items.len != 2) continue;

            const start = toOffset(offsets_arr.items[0]) orelse continue;
            const end = toOffset(offsets_arr.items[1]) orelse continue;
            if (end < start or end > data_len) continue;

            const dtype = parseDtype(dtype_str);
            var shape: [4]u64 = .{ 0, 0, 0, 0 };
            const n_dims: u32 = @intCast(@min(shape_arr.items.len, 4));
            for (0..n_dims) |i| {
                shape[i] = std.math.cast(u64, shape_arr.items[i].integer) orelse return error.InvalidFormat;
            }
            // The tensors are read with a count the caller derives from shape,
            // so the mapping must actually hold shape * dtype bytes. Without
            // this a crafted file points a read past the end of the mmap.
            if (end - start < shapeBytes(shape[0..n_dims], dtype)) return error.InvalidFormat;

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

    /// Free the dupe'd tensor names and unmap the safetensors file. `allocator`
    /// must be the one passed to `init` and `load`; tensor data pointers from
    /// `get` dangle afterwards.
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

test "shapeBytes multiplies shape by element width" {
    try std.testing.expectEqual(@as(usize, 64), shapeBytes(&.{ 4, 4 }, .f32));
    try std.testing.expectEqual(@as(usize, 32), shapeBytes(&.{ 4, 4 }, .bf16));
    try std.testing.expectEqual(@as(usize, 8), shapeBytes(&.{ 2, 4 }, .fp8_e4m3));
    // A packed or unknown dtype has no fixed width: range validation only.
    try std.testing.expectEqual(@as(usize, 0), shapeBytes(&.{ 4, 4 }, .unknown));
}

test "shapeBytes saturates instead of wrapping" {
    // A shape the header cannot back must not wrap to a small length that
    // would then pass the "file holds the bytes" check.
    try std.testing.expectEqual(std.math.maxInt(usize), shapeBytes(&.{ std.math.maxInt(u64) / 2, 4 }, .f32));
}

test "toOffset rejects negatives and non-numbers" {
    try std.testing.expectEqual(@as(?usize, 16), toOffset(.{ .integer = 16 }));
    try std.testing.expectEqual(@as(?usize, null), toOffset(.{ .integer = -1 }));
    try std.testing.expectEqual(@as(?usize, null), toOffset(.{ .float = 1.5 }));
    try std.testing.expectEqual(@as(?usize, null), toOffset(.null));
}
