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

        try parseImage(self, allocator, mapped);
    }

    /// Parse a whole safetensors image (8-byte header length, JSON header,
    /// data region) into `self`. Split out of `load` so the header parser runs
    /// on an ordinary buffer, which is what the fuzz target drives; the bytes
    /// are the same either way. Tensor data pointers alias `mapped`, so the
    /// caller must keep it alive (and unmapped last) while it uses them.
    fn parseImage(self: *MtpWeights, allocator: Allocator, mapped: []const u8) !void {
        const file_size = mapped.len;

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

        // A header that is not an object (`[]`, `7`, `"x"`, `null`) carries no
        // tensors. Reading `.object` off another tag is illegal union access
        // and traps in safe modes, so name the tag instead.
        const root = switch (parsed.value) {
            .object => |o| o,
            else => return error.InvalidFormat,
        };
        var max_depth: u32 = 0;
        var count: u32 = 0;

        var it = root.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (std.mem.eql(u8, name, "__metadata__")) continue;

            // Same rule per entry: only an object holds dtype/shape/offsets.
            const obj = switch (entry.value_ptr.*) {
                .object => |o| o,
                else => continue,
            };
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
                // Every element, not just the first, must be read through a
                // tag check: `"shape": ["4"]` is a legal JSON array whose
                // element is a string, and reading `.integer` off it is
                // illegal union access.
                shape[i] = toOffset(shape_arr.items[i]) orelse return error.InvalidFormat;
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

/// Header + data image for `header`, as `load` would read it off disk.
fn imageFor(allocator: Allocator, header: []const u8, data: []const u8) ![]u8 {
    const img = try allocator.alloc(u8, safetensors_header_prefix_len + header.len + data.len);
    std.mem.writeInt(u64, img[0..8], header.len, .little);
    @memcpy(img[8..][0..header.len], header);
    @memcpy(img[8 + header.len ..], data);
    return img;
}

test "mtp: a header that is not an object is refused, not a trap" {
    // Each of these is a legal safetensors *file* whose header parses as JSON
    // but holds no tensor object. Reading `.object` off the tag used to be
    // illegal union access, so this is a crash regression, not a null test.
    const alloc = std.testing.allocator;
    for ([_][]const u8{ "[]", "7", "\"x\"", "null", "true" }) |header| {
        const img = try imageFor(alloc, header, "0123456789abcdef");
        defer alloc.free(img);
        var w = MtpWeights.init(alloc);
        defer w.deinit(alloc);
        try std.testing.expectError(error.InvalidFormat, w.parseImage(alloc, img));
    }
}

test "mtp: a shape element of the wrong type is refused, not a trap" {
    const alloc = std.testing.allocator;
    const header =
        "{\"mtp.0.attn.wq_a.weight\":{\"dtype\":\"F32\",\"shape\":[\"4\",-1,1.5,null],\"data_offsets\":[0,16]}}"
    ;
    const img = try imageFor(alloc, header, "0123456789abcdef");
    defer alloc.free(img);
    var w = MtpWeights.init(alloc);
    defer w.deinit(alloc);
    try std.testing.expectError(error.InvalidFormat, w.parseImage(alloc, img));
}

test "mtp: a header length past the end of the file is refused" {
    const alloc = std.testing.allocator;
    const img = try alloc.alloc(u8, 16);
    defer alloc.free(img);
    @memset(img, 0);
    std.mem.writeInt(u64, img[0..8], 4096, .little);
    var w = MtpWeights.init(alloc);
    defer w.deinit(alloc);
    try std.testing.expectError(error.InvalidFormat, w.parseImage(alloc, img));
}

test "mtp: a well-formed header indexes tensors inside the data region" {
    const alloc = std.testing.allocator;
    const header = "{\"mtp.0.attn.wq_a.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}," ++
        "\"mtp.2.attn.wq_b.weight\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[16,24]}," ++
        "\"out.0.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[24,28]}}";
    const img = try imageFor(alloc, header, "0123456789abcdefghijklmn");
    defer alloc.free(img);
    var w = MtpWeights.init(alloc);
    defer w.deinit(alloc);
    try w.parseImage(alloc, img);
    // Only the `mtp.<digit>` names set a depth: mtp.0 -> 1, mtp.2 -> 3.
    try std.testing.expectEqual(@as(u32, 3), w.n_depths);
    try std.testing.expectEqual(@as(usize, 3), w.tensors.count());
    const t = w.get("mtp.0.attn.wq_a.weight") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(DType.f32, t.dtype);
    try std.testing.expectEqual(@as(u64, 2), t.shape[0]);
    try std.testing.expectEqual(@as(u64, 2), t.shape[1]);
    // The bytes the header says the tensor spans stay inside the image, so a
    // consumer dereferencing data_ptr cannot leave the mapping.
    const base = @intFromPtr(img.ptr);
    var it = w.tensors.valueIterator();
    while (it.next()) |t2| {
        const p0 = @intFromPtr(t2.data_ptr);
        try std.testing.expect(p0 >= base);
        try std.testing.expect(p0 + shapeBytes(t2.shape[0..t2.n_dims], t2.dtype) <= base + img.len);
        try std.testing.expect(t2.n_dims <= 4);
    }
}

test "fuzz: mtp safetensors header, no trap and no out-of-image tensor" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const allocator = std.testing.allocator;

            // Seed a realistic tensor descriptor and vary its fields, so the
            // parser sees the shapes a real checkpoint has alongside the
            // offsets a corrupt or hostile download would claim.
            const names = [_][]const u8{
                "mtp.0.attn.wq_a.weight", "mtp.0.main_proj.scale",
                "mtp.2.ffn.w1.weight",     "out.0.weight",
            };
            const dtypes = [_][]const u8{ "F32", "F16", "BF16", "F8_E4M3", "I8", "MXFP4" };
            const name = names[smith.indexWithHash(names.len, 0)];
            const dtype = dtypes[smith.indexWithHash(dtypes.len, 1)];
            const d0 = smith.valueWithHash(u8, 2) % 8;
            const d1 = smith.valueWithHash(u8, 3) % 8;
            const start = smith.indexWithHash(24, 4);
            const end = start + smith.indexWithHash(24, 5);

            var header_buf: [192]u8 = undefined;
            const header = std.fmt.bufPrint(&header_buf,
                "{{\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}}}"
            , .{ name, dtype, d0, d1, start, end }) catch return;
            const img = try imageFor(allocator, header, "0123456789abcdef0123456789abcdef");
            defer allocator.free(img);

            // Truncate at a fuzzed point: a partial header, a header with no
            // data region, and a length prefix that overruns are all ordinary
            // shapes of a killed download.
            const cut = smith.indexWithHash(img.len + 1, 6);

            var w = MtpWeights.init(allocator);
            defer w.deinit(allocator);
            w.parseImage(allocator, img[0..cut]) catch return;

            const base = @intFromPtr(img.ptr);
            var it = w.tensors.valueIterator();
            while (it.next()) |t| {
                const p0 = @intFromPtr(t.data_ptr);
                // The pointer, and the bytes its shape implies, both stay in
                // the image the consumer will read from.
                try std.testing.expect(p0 >= base);
                try std.testing.expect(p0 + shapeBytes(t.shape[0..t.n_dims], t.dtype) <= base + img.len);
                try std.testing.expect(t.n_dims <= 4);
            }
            // `mtp.<digit>` is the only depth source, so nothing claims more
            // than the three MTP depths DS4 defines.
            try std.testing.expect(w.n_depths <= 3);
        }
    }.f, .{});
}
