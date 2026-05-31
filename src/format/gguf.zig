//! GGUF (GPT-Generated Unified Format) file parser.
//! Provides memory-mapped access to model weights, metadata, and tokenizer data.

const std = @import("std");
const posix = std.posix;
const Allocator = std.mem.Allocator;
const fmt = @import("format.zig");
const Format = fmt.Format;
const FormatTensorInfo = fmt.TensorInfo;
const DType = fmt.DType;

/// GGUF magic bytes: "GGUF" as a little-endian u32.
const gguf_magic: u32 = 0x46554747;
/// Maximum supported GGUF format version (versions 2 and 3 are accepted).
const gguf_version_3: u32 = 3;
/// Minimum valid GGUF file size: 4 magic + 4 version + 8 tensor_count + 8 metadata_kv_count.
const gguf_min_header_size: usize = 24;
/// Maximum allowed metadata key-value count (prevents DoS via crafted files).
/// Real GGUF files have ~50-200 entries; 100K is generous headroom.
const max_metadata_kv_count: usize = 100_000;
/// Maximum allowed tensor count.
const max_tensor_count: usize = 100_000;
/// Maximum allowed metadata array length.
/// Largest real use is tokenizer vocab (~256K tokens); 500K covers that with margin.
const max_array_len: usize = 1_000_000;
/// Maximum allowed alignment (prevents arithmetic issues from crafted metadata).
/// Standard GGUF alignment is 32; values beyond 1 MiB are nonsensical.
const max_alignment: u32 = 1 << 20;
/// Buffer size for tensor/metadata name formatting (must fit longest GGUF key).
const name_buf_size: usize = 256;

/// GGML quantization type identifiers.
pub const GGMLType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 34,
    tq2_0 = 35,
    mxfp4 = 39,
    _,

    /// Returns the number of elements per quantization block for this type.
    pub fn blockSize(self: GGMLType) usize {
        return switch (self) {
            .f32 => 1,
            .f16 => 1,
            .q4_0 => 32,
            .q4_1 => 32,
            .q5_0 => 32,
            .q5_1 => 32,
            .q8_0 => 32,
            .q8_1 => 32,
            .q2_k => 256,
            .q3_k => 256,
            .q4_k => 256,
            .q5_k => 256,
            .q6_k => 256,
            .i8 => 1,
            .i16 => 1,
            .i32 => 1,
            .i64 => 1,
            .f64 => 1,
            .bf16 => 1,
            .iq2_xxs => 256,
            .iq2_xs => 256,
            .iq2_s => 256,
            .iq3_xxs => 256,
            .iq3_s => 256,
            .iq1_s => 256,
            .iq1_m => 256,
            .iq4_nl => 32,
            .iq4_xs => 256,
            .tq1_0 => 256,
            .mxfp4 => 32,
            else => 1,
        };
    }

    /// Returns the byte size of one quantization block for this type.
    pub fn bytesPerBlock(self: GGMLType) usize {
        return switch (self) {
            .f32 => 4,
            .f16 => 2,
            .q4_0 => 18, // f16 scale + 16 bytes (32 nibbles)
            .q4_1 => 20, // f16 scale + f16 min + 16 bytes
            .q5_0 => 22, // f16 scale + 4 bytes high bits + 16 bytes
            .q5_1 => 24, // f16 scale + f16 min + 4 bytes high bits + 16 bytes
            .q8_0 => 34, // f16 scale + 32 bytes
            .q8_1 => 36, // f16 scale + f16 sum + 32 bytes
            .q2_k => 256 / 16 * 2 + 256 / 4 + 2 + 2, // 84
            .q3_k => 256 / 8 + 256 / 4 + 2 + 12, // 110 (hmask + qs + d + scales)
            .q4_k => 2 + 2 + 12 + 256 / 2, // 144
            .q5_k => 2 + 2 + 12 + 256 / 8 + 256 / 2, // 176
            .q6_k => 256 / 2 + 256 / 4 + 256 / 16 + 2, // 210
            .i8 => 1,
            .i16 => 2,
            .i32 => 4,
            .i64 => 8,
            .f64 => 8,
            .bf16 => 2,
            .iq2_xxs => 66, // 256 elements: 2 bytes scale + 2*8 bytes (16 groups × 2B each) + 8*4 bytes (8 blocks × 4B signs) = 66
            .iq2_xs => 74, // 256 elements: 2 bytes scale + 2*8 bytes + 8*4 bytes + 8 bytes scales = 74
            .iq2_s => 82, // 256 elements
            .iq3_xxs => 98, // 256 elements
            .iq3_s => 110, // 256 elements
            .iq1_s => 50, // 256 elements
            .iq1_m => 56, // 256 elements
            .iq4_nl => 18, // f16 scale + 16 bytes (32 nibbles, same as q4_0)
            .iq4_xs => 136, // f16 d (2) + u16 scales_h (2) + scales_l[4] (4) + qs[128] (128)
            .tq1_0 => 64, // f16 scale (2) + qs[40] + qh[13] + padding[9]
            .mxfp4 => 17, // 1 byte E8M0 scale + 16 bytes (32 FP4 nibbles)
            else => 1,
        };
    }

    /// Computes total byte size for a tensor with given element count.
    /// Uses checked arithmetic to prevent silent overflow on crafted metadata.
    pub fn tensorBytes(self: GGMLType, n_elements: usize) usize {
        const bs = self.blockSize();
        const n_blocks = (std.math.add(usize, n_elements, bs - 1) catch
            std.math.maxInt(usize)) / bs;
        return std.math.mul(usize, n_blocks, self.bytesPerBlock()) catch
            std.math.maxInt(usize); // Saturate on overflow — caller validates against file size
    }
};

/// GGUF metadata value type tags (internal to parsing).
const MetaValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool_type = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
    _,
};

/// A parsed GGUF metadata value with type tag and payload.
pub const MetaValue = union(enum) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool_val: bool,
    string: []const u8,
    uint64: u64,
    int64: i64,
    float64: f64,
    array_str: []const []const u8,
    array_u32: []const u32,

    /// Attempts to extract a u32 value, casting from compatible integer types.
    /// Returns null for negative or out-of-range values.
    pub fn asU32(self: MetaValue) ?u32 {
        return switch (self) {
            .uint32 => |v| v,
            .int32 => |v| if (v >= 0) @intCast(v) else null,
            .uint64 => |v| if (v <= std.math.maxInt(u32)) @intCast(v) else null,
            .uint8 => |v| v,
            .uint16 => |v| v,
            else => null,
        };
    }

    /// Attempts to extract an f32 value, casting from f64 if needed.
    pub fn asF32(self: MetaValue) ?f32 {
        return switch (self) {
            .float32 => |v| v,
            .float64 => |v| @floatCast(v),
            else => null,
        };
    }

    /// Attempts to extract a string value.
    pub fn asStr(self: MetaValue) ?[]const u8 {
        return switch (self) {
            .string => |v| v,
            else => null,
        };
    }

    /// Attempts to extract a boolean value.
    pub fn asBool(self: MetaValue) ?bool {
        return switch (self) {
            .bool_val => |v| v,
            else => null,
        };
    }
};

/// Parsed tensor metadata from a GGUF file header.
pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    dims: [4]u64 = .{ 0, 0, 0, 0 },
    ggml_type: GGMLType,
    offset: u64, // relative to data section start

    /// Returns the total number of elements (product of all dimensions).
    pub fn numElements(self: *const TensorInfo) usize {
        var n: usize = 1;
        for (0..self.n_dims) |i| n = std.math.mul(usize, n, std.math.cast(usize, self.dims[i]) orelse return 0) catch return 0;
        return n;
    }

    /// Returns the total byte size of this tensor's data.
    pub fn dataBytes(self: *const TensorInfo) usize {
        return self.ggml_type.tensorBytes(self.numElements());
    }
};

/// A memory-mapped GGUF file providing access to tensors and metadata.
pub const GGUFFile = struct {
    mapped_data: []align(std.heap.page_size_min) u8 = &.{},
    raw_data: []const u8 = &.{},
    is_buffer: bool = false,
    /// File descriptor for GPUDirect Storage (cuFile) direct NVMe→VRAM transfer.
    /// -1 if not available (buffer mode or already closed).
    file_fd: i32 = -1,
    file_size: usize,
    version: u32 = 0,
    tensor_count: u64 = 0,
    metadata: std.StringHashMap(MetaValue),
    tensors: std.StringHashMap(TensorInfo),
    data_offset: usize = 0,
    alignment: usize = 32,
    allocator: Allocator,
    owned_strings: std.ArrayList([]const u8) = .empty,
    owned_arrays: std.ArrayList([*]const u8) = .empty,
    owned_array_lens: std.ArrayList(usize) = .empty,
    owned_u32_arrays: std.ArrayList([]u32) = .empty,

    /// Opens and memory-maps a GGUF file, parsing headers, metadata, and tensor info.
    pub fn open(allocator: Allocator, path: []const u8) !GGUFFile {
        const fd = try posix.openat(posix.AT.FDCWD, path, .{}, 0);
        errdefer {
            if (comptime @import("builtin").os.tag == .linux) {
                _ = posix.system.close(fd);
            } else {
                _ = std.c.close(fd);
            }
        }
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
        if (file_size < gguf_min_header_size) return error.FileTooSmall;

        const mapped = try posix.mmap(null, file_size, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0);
        errdefer posix.munmap(mapped);
        // Hint sequential access for weight loading — enables OS readahead and reduces page faults.
        posix.madvise(mapped.ptr, mapped.len, posix.MADV.SEQUENTIAL) catch {};

        var self = GGUFFile{
            .mapped_data = mapped,
            .file_fd = fd,
            .file_size = file_size,
            .metadata = std.StringHashMap(MetaValue).init(allocator),
            .tensors = std.StringHashMap(TensorInfo).init(allocator),
            .allocator = allocator,
        };
        try self.parseHeader();
        return self;
    }

    /// Open from a raw byte buffer (no file I/O — for WASM or embedded use).
    pub fn fromBuffer(allocator: Allocator, buf: []const u8) !GGUFFile {
        if (buf.len < gguf_min_header_size) return error.FileTooSmall;
        var self = GGUFFile{
            .raw_data = buf,
            .file_size = buf.len,
            .metadata = std.StringHashMap(MetaValue).init(allocator),
            .tensors = std.StringHashMap(TensorInfo).init(allocator),
            .allocator = allocator,
            .is_buffer = true,
        };
        try self.parseHeader();
        return self;
    }

    /// Get the underlying data bytes (works for both mmap and buffer paths).
    inline fn data(self: *const GGUFFile) []const u8 {
        return if (self.is_buffer) self.raw_data else self.mapped_data;
    }

    /// Returns a Format interface backed by this GGUF file.
    pub fn format(self: *GGUFFile) Format {
        return .{ .ptr = self, .vtable = &format_vtable };
    }

    const format_vtable = Format.VTable{
        .get_tensor = @ptrCast(&fmtGetTensor),
        .get_meta_str = @ptrCast(&fmtGetMetaStr),
        .get_meta_u32 = @ptrCast(&fmtGetMetaU32),
        .get_meta_f32 = @ptrCast(&fmtGetMetaF32),
        .get_meta_u32_array = @ptrCast(&fmtGetMetaU32Array),
        .get_vocab = @ptrCast(&fmtGetVocab),
        .get_merges = @ptrCast(&fmtGetMerges),
    };

    fn ggmlToDType(t: GGMLType) DType {
        return switch (t) {
            .f32 => .f32,
            .f16 => .f16,
            .bf16 => .bf16,
            .q2_k => .q2_k,
            .q3_k => .q3_k,
            .q4_0 => .q4_0,
            .q4_1 => .q4_1,
            .q4_k => .q4_k,
            .q5_0 => .q5_0,
            .q5_k => .q5_k,
            .q6_k => .q6_k,
            .q8_0 => .q8_0,
            .iq4_xs => .iq4_xs,
            .iq4_nl => .iq4_nl,
            .tq1_0 => .tq1_0,
            .mxfp4 => .mxfp4,
            else => .unknown,
        };
    }

    fn fmtGetTensor(self: *GGUFFile, name: []const u8) ?FormatTensorInfo {
        const info = self.tensors.getPtr(name) orelse {
            // Fallback: try translating HF-style name to GGUF-style.
            var buf: [name_buf_size]u8 = undefined;
            const gguf_name = hfNameToGguf(name, &buf) orelse return null;
            const info2 = self.tensors.getPtr(gguf_name) orelse return null;
            return .{ .name = info2.name, .n_dims = info2.n_dims, .dims = info2.dims, .dtype = ggmlToDType(info2.ggml_type), .data_ptr = self.tensorData(info2) };
        };
        return .{ .name = info.name, .n_dims = info.n_dims, .dims = info.dims, .dtype = ggmlToDType(info.ggml_type), .data_ptr = self.tensorData(info) };
    }
    fn fmtGetMetaStr(self: *GGUFFile, key: []const u8) ?[]const u8 {
        if (self.getMetaStr(key)) |v| return v;
        return self.getMetaHfKey(key);
    }
    fn fmtGetMetaU32(self: *GGUFFile, key: []const u8) ?u32 {
        if (self.getMetaU32(key)) |v| return v;
        // Fallback: try arch-prefixed GGUF key.
        var buf: [name_buf_size]u8 = undefined;
        const arch_prefix = self.getMetaStr("general.architecture") orelse return null;
        const gguf_suffix = hfKeyToGgufSuffix(key) orelse return null;
        const gguf_key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch_prefix, gguf_suffix }) catch return null;
        return self.getMetaU32(gguf_key);
    }
    fn fmtGetMetaF32(self: *GGUFFile, key: []const u8) ?f32 {
        if (self.getMetaF32(key)) |v| return v;
        var buf: [name_buf_size]u8 = undefined;
        const arch_prefix = self.getMetaStr("general.architecture") orelse return null;
        const gguf_suffix = hfKeyToGgufSuffix(key) orelse return null;
        const gguf_key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch_prefix, gguf_suffix }) catch return null;
        return self.getMetaF32(gguf_key);
    }

    /// Look up a metadata value using arch-prefixed fallback, returning string.
    fn getMetaHfKey(self: *GGUFFile, key: []const u8) ?[]const u8 {
        var buf: [name_buf_size]u8 = undefined;
        const arch_prefix = self.getMetaStr("general.architecture") orelse return null;
        const gguf_suffix = hfKeyToGgufSuffix(key) orelse return null;
        const gguf_key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch_prefix, gguf_suffix }) catch return null;
        return self.getMetaStr(gguf_key);
    }
    fn fmtGetMetaU32Array(self: *GGUFFile, key: []const u8) ?[]const u32 {
        return self.getMetaU32Array(key);
    }
    fn fmtGetVocab(self: *GGUFFile) ?[]const []const u8 {
        return self.getTokenizerVocab();
    }
    fn fmtGetMerges(self: *GGUFFile) ?[]const []const u8 {
        return self.getTokenizerMerges();
    }

    /// Returns the total number of parameters (sum of all tensor element counts).
    pub fn totalParams(self: *const GGUFFile) u64 {
        var total: u64 = 0;
        var it = self.tensors.valueIterator();
        while (it.next()) |info| {
            total += @intCast(info.numElements());
        }
        return total;
    }

    /// Release all resources: metadata, tensors, owned strings, and the mmap.
    pub fn deinit(self: *GGUFFile) void {
        self.metadata.deinit();
        self.tensors.deinit();
        for (self.owned_strings.items) |s| self.allocator.free(s);
        self.owned_strings.deinit(self.allocator);
        for (self.owned_arrays.items, self.owned_array_lens.items) |ptr, len| {
            const slice: []const []const u8 = @as([*]const []const u8, @ptrCast(@alignCast(ptr)))[0..len];
            self.allocator.free(slice);
        }
        self.owned_arrays.deinit(self.allocator);
        self.owned_array_lens.deinit(self.allocator);
        for (self.owned_u32_arrays.items) |s| self.allocator.free(s);
        self.owned_u32_arrays.deinit(self.allocator);
        if (comptime @import("builtin").os.tag != .freestanding) {
            if (!self.is_buffer) posix.munmap(self.mapped_data);
            if (self.file_fd >= 0) {
                if (comptime @import("builtin").os.tag == .linux) {
                    _ = posix.system.close(self.file_fd);
                } else {
                    _ = std.c.close(self.file_fd);
                }
            }
        }
    }

    fn own(self: *GGUFFile, s: []const u8) ![]const u8 {
        const d = try self.allocator.dupe(u8, s);
        errdefer self.allocator.free(d);
        try self.owned_strings.append(self.allocator, d);
        return d;
    }

    fn readU32(self: *const GGUFFile, off: usize) !u32 {
        if (self.file_size < 4 or off > self.file_size - 4) return error.OffsetOutOfBounds;
        return std.mem.readInt(u32, self.data()[off..][0..4], .little);
    }
    fn readU64(self: *const GGUFFile, off: usize) !u64 {
        if (self.file_size < 8 or off > self.file_size - 8) return error.OffsetOutOfBounds;
        return std.mem.readInt(u64, self.data()[off..][0..8], .little);
    }
    fn readI32(self: *const GGUFFile, off: usize) !i32 {
        if (self.file_size < 4 or off > self.file_size - 4) return error.OffsetOutOfBounds;
        return std.mem.readInt(i32, self.data()[off..][0..4], .little);
    }
    fn readF32(self: *const GGUFFile, off: usize) !f32 {
        return @bitCast(try self.readU32(off));
    }
    fn readString(self: *const GGUFFile, off: usize) !struct { str: []const u8, len: usize } {
        const slen: usize = std.math.cast(usize, try self.readU64(off)) orelse return error.OffsetOutOfBounds;
        // Use subtraction to avoid overflow: readU64(off) succeeded so off+8 <= file_size.
        if (slen > self.file_size - off - 8) return error.OffsetOutOfBounds;
        return .{ .str = self.data()[off + 8 ..][0..slen], .len = 8 + slen };
    }

    fn readMetaValue(self: *GGUFFile, off: usize) !struct { val: MetaValue, len: usize } {
        const vtype: MetaValueType = @enumFromInt(try self.readU32(off));
        var pos: usize = off + 4;
        switch (vtype) {
            .uint8 => {
                if (pos + 1 > self.file_size) return error.OffsetOutOfBounds;
                return .{ .val = .{ .uint8 = self.data()[pos] }, .len = 5 };
            },
            .int8 => {
                if (pos + 1 > self.file_size) return error.OffsetOutOfBounds;
                return .{ .val = .{ .int8 = @bitCast(self.data()[pos]) }, .len = 5 };
            },
            .uint16 => {
                if (pos + 2 > self.file_size) return error.OffsetOutOfBounds;
                return .{ .val = .{ .uint16 = std.mem.readInt(u16, self.data()[pos..][0..2], .little) }, .len = 6 };
            },
            .int16 => {
                if (pos + 2 > self.file_size) return error.OffsetOutOfBounds;
                return .{ .val = .{ .int16 = std.mem.readInt(i16, self.data()[pos..][0..2], .little) }, .len = 6 };
            },
            .uint32 => return .{ .val = .{ .uint32 = try self.readU32(pos) }, .len = 8 },
            .int32 => return .{ .val = .{ .int32 = try self.readI32(pos) }, .len = 8 },
            .float32 => return .{ .val = .{ .float32 = try self.readF32(pos) }, .len = 8 },
            .bool_type => {
                if (pos + 1 > self.file_size) return error.OffsetOutOfBounds;
                return .{ .val = .{ .bool_val = self.data()[pos] != 0 }, .len = 5 };
            },
            .string => {
                const s = try self.readString(pos);
                return .{ .val = .{ .string = s.str }, .len = 4 + s.len };
            },
            .uint64 => return .{ .val = .{ .uint64 = try self.readU64(pos) }, .len = 12 },
            .int64 => return .{ .val = .{ .int64 = @bitCast(try self.readU64(pos)) }, .len = 12 },
            .float64 => return .{ .val = .{ .float64 = @bitCast(try self.readU64(pos)) }, .len = 12 },
            .array => {
                const arr_type: MetaValueType = @enumFromInt(try self.readU32(pos));
                const arr_len_u64 = try self.readU64(pos + 4);
                if (arr_len_u64 > max_array_len) return error.ArrayTooLarge;
                const arr_len: usize = std.math.cast(usize, arr_len_u64) orelse return error.ArrayTooLarge;
                pos += 12;
                if (arr_type == .string) {
                    const strings = try self.allocator.alloc([]const u8, arr_len);
                    errdefer self.allocator.free(strings);
                    try self.owned_arrays.append(self.allocator, @ptrCast(strings.ptr));
                    errdefer _ = self.owned_arrays.pop();
                    try self.owned_array_lens.append(self.allocator, arr_len);
                    errdefer _ = self.owned_array_lens.pop();
                    for (0..arr_len) |i| {
                        const s = try self.readString(pos);
                        strings[i] = s.str;
                        pos += s.len;
                    }
                    return .{ .val = .{ .array_str = strings }, .len = pos - off };
                }
                // Parse u32/i32 arrays (used for eog_token_id, head_count_kv, etc.)
                if (arr_type == .uint32 or arr_type == .int32) {
                    const ids = try self.allocator.alloc(u32, arr_len);
                    errdefer self.allocator.free(ids);
                    try self.owned_u32_arrays.append(self.allocator, ids);
                    errdefer _ = self.owned_u32_arrays.pop();
                    for (0..arr_len) |i| {
                        ids[i] = try self.readU32(pos);
                        pos += 4;
                    }
                    return .{ .val = .{ .array_u32 = ids }, .len = pos - off };
                }
                // Parse bool arrays as u32 (used for sliding_window_pattern)
                if (arr_type == .bool_type) {
                    const ids = try self.allocator.alloc(u32, arr_len);
                    errdefer self.allocator.free(ids);
                    try self.owned_u32_arrays.append(self.allocator, ids);
                    errdefer _ = self.owned_u32_arrays.pop();
                    for (0..arr_len) |i| {
                        if (pos >= self.file_size) return error.OffsetOutOfBounds;
                        ids[i] = self.data()[pos];
                        pos += 1;
                    }
                    return .{ .val = .{ .array_u32 = ids }, .len = pos - off };
                }
                // Skip other non-string arrays
                for (0..arr_len) |_| {
                    const elem_size: usize = switch (arr_type) {
                        .uint8, .int8, .bool_type => 1,
                        .uint16, .int16 => 2,
                        .uint32, .int32, .float32 => 4,
                        .uint64, .int64, .float64 => 8,
                        else => return error.UnsupportedGGUFVersion,
                    };
                    if (self.file_size < elem_size or pos > self.file_size - elem_size) return error.OffsetOutOfBounds;
                    pos += elem_size;
                }
                return .{ .val = .{ .uint32 = 0 }, .len = pos - off };
            },
            _ => return .{ .val = .{ .uint32 = 0 }, .len = 4 },
        }
    }

    fn parseHeader(self: *GGUFFile) !void {
        var off: usize = 0;

        const magic = try self.readU32(off);
        off += 4;
        if (magic != gguf_magic) return error.InvalidMagic;

        self.version = try self.readU32(off);
        off += 4;
        if (self.version < 2 or self.version > gguf_version_3) return error.UnsupportedVersion;

        self.tensor_count = try self.readU64(off);
        off += 8;
        if (self.tensor_count > max_tensor_count) return error.TooManyTensors;
        const metadata_kv_count = try self.readU64(off);
        off += 8;
        if (metadata_kv_count > max_metadata_kv_count) return error.TooManyMetadataKeys;

        // Parse metadata
        for (0..@intCast(metadata_kv_count)) |_| {
            const key_info = try self.readString(off);
            const key = try self.own(key_info.str);
            off += key_info.len;
            const val_result = try self.readMetaValue(off);
            off += val_result.len;
            try self.metadata.put(key, val_result.val);
        }

        // Check alignment (must be non-zero to avoid division by zero)
        if (self.metadata.get("general.alignment")) |v| {
            if (v.asU32()) |a| {
                if (a > 0 and a <= max_alignment) self.alignment = a;
            }
        }

        // Parse tensor infos
        for (0..@intCast(self.tensor_count)) |_| {
            const name_info = try self.readString(off);
            const name = try self.own(name_info.str);
            off += name_info.len;

            const n_dims = try self.readU32(off);
            off += 4;
            if (n_dims > 4) return error.TooManyDimensions;

            // Read dims and reverse to standard order (outermost first).
            // GGUF stores dims inner-first; we normalize to [output_rows, input_cols, ...]
            // so all consumers can use dims[0] for output rows regardless of format.
            var raw_dims: [4]u64 = .{ 0, 0, 0, 0 };
            for (0..n_dims) |d| {
                raw_dims[d] = try self.readU64(off);
                off += 8;
            }
            var dims: [4]u64 = .{ 0, 0, 0, 0 };
            for (0..n_dims) |d| dims[d] = raw_dims[n_dims - 1 - d];

            const ggml_type: GGMLType = @enumFromInt(try self.readU32(off));
            off += 4;

            const tensor_offset = try self.readU64(off);
            off += 8;

            try self.tensors.put(name, .{
                .name = name,
                .n_dims = n_dims,
                .dims = dims,
                .ggml_type = ggml_type,
                .offset = tensor_offset,
            });
        }

        // Data section starts after header, aligned.
        // Use checked arithmetic to prevent overflow from crafted alignment values.
        const rem = off % self.alignment;
        if (rem != 0) {
            off = std.math.add(usize, off, self.alignment - rem) catch
                return error.OffsetOutOfBounds;
        }
        if (off > self.file_size) return error.OffsetOutOfBounds;
        self.data_offset = off;

        // Validate all tensor data fits within the mapped file.
        // A crafted GGUF could specify offsets past the mmap'd region,
        // causing out-of-bounds reads when tensorData() is called later.
        var tensor_it = self.tensors.valueIterator();
        while (tensor_it.next()) |info| {
            const info_offset = std.math.cast(usize, info.offset) orelse return error.OffsetOutOfBounds;
            const abs_offset = std.math.add(usize, self.data_offset, info_offset) catch
                return error.OffsetOutOfBounds;
            const data_bytes = info.dataBytes();
            // Detect overflow saturation from tensorBytes() — crafted dimensions.
            if (data_bytes == std.math.maxInt(usize)) return error.OffsetOutOfBounds;
            const tensor_end = std.math.add(usize, abs_offset, data_bytes) catch
                return error.OffsetOutOfBounds;
            if (tensor_end > self.file_size) return error.OffsetOutOfBounds;
        }
    }

    /// Returns raw tensor data bytes for the given tensor info.
    pub fn tensorData(self: *const GGUFFile, info: *const TensorInfo) [*]const u8 {
        return self.data().ptr + self.data_offset + @as(usize, @intCast(info.offset));
    }

    /// Looks up a string metadata value by key.
    pub fn getMetaStr(self: *const GGUFFile, key: []const u8) ?[]const u8 {
        if (self.metadata.get(key)) |v| return v.asStr();
        return null;
    }
    /// Looks up a uint32 metadata value by key.
    pub fn getMetaU32(self: *const GGUFFile, key: []const u8) ?u32 {
        if (self.metadata.get(key)) |v| return v.asU32();
        return null;
    }
    /// Looks up a float32 metadata value by key.
    pub fn getMetaF32(self: *const GGUFFile, key: []const u8) ?f32 {
        if (self.metadata.get(key)) |v| return v.asF32();
        return null;
    }
    /// Looks up a uint32 array metadata value by key.
    pub fn getMetaU32Array(self: *const GGUFFile, key: []const u8) ?[]const u32 {
        if (self.metadata.get(key)) |v| {
            return switch (v) {
                .array_u32 => |arr| arr,
                else => null,
            };
        }
        return null;
    }

    /// Returns the tokenizer vocabulary as a slice of token strings.
    pub fn getTokenizerVocab(self: *const GGUFFile) ?[]const []const u8 {
        if (self.metadata.get("tokenizer.ggml.tokens")) |v| {
            switch (v) {
                .array_str => |arr| return arr,
                else => return null,
            }
        }
        return null;
    }

    /// Returns BPE merge rules as a slice of strings.
    pub fn getTokenizerMerges(self: *const GGUFFile) ?[]const []const u8 {
        if (self.metadata.get("tokenizer.ggml.merges")) |v| {
            switch (v) {
                .array_str => |arr| return arr,
                else => return null,
            }
        }
        return null;
    }

    /// Dequantizes a single Q4_0 block (32 elements) into f32 output.
    pub fn dequantQ4_0(block: [*]const u8, output: [*]f32) void {
        const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little))));
        for (0..16) |i| {
            const byte = block[2 + i];
            const lo: i8 = @as(i8, @intCast(byte & 0x0F)) - 8;
            const hi: i8 = @as(i8, @intCast(byte >> 4)) - 8;
            output[i * 2] = @as(f32, @floatFromInt(lo)) * scale;
            output[i * 2 + 1] = @as(f32, @floatFromInt(hi)) * scale;
        }
    }

    /// Dequantizes a single Q8_0 block (32 elements) into f32 output.
    pub fn dequantQ8_0(block: [*]const u8, output: [*]f32) void {
        const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block[0..2], .little))));
        for (0..32) |i| {
            const val: i8 = @bitCast(block[2 + i]);
            output[i] = @as(f32, @floatFromInt(val)) * scale;
        }
    }
};

// ── HF → GGUF name translation ────────────────────────────────────────────────
// Used as fallback when models (e.g. GLM-4) use HuggingFace-style tensor names
// with a GGUF format file. Reverse of safetensors.zig's ggufToHfName.

/// HF layer component → GGUF layer component (longest-prefix match order).
const hf_gguf_layer_map = [_]struct { []const u8, []const u8 }{
    // MLA attention (DeepSeek2/GLM-4) — longer prefixes first
    .{ "self_attn.q_a_layernorm", "attn_q_a_norm" },
    .{ "self_attn.kv_a_layernorm", "attn_kv_a_norm" },
    .{ "self_attn.q_a_proj", "attn_q_a" },
    .{ "self_attn.q_b_proj", "attn_q_b" },
    .{ "self_attn.kv_a_proj_with_mqa", "attn_kv_a_mqa" },
    .{ "self_attn.embed_q", "attn_k_b" },
    .{ "self_attn.unembed_out", "attn_v_b" },
    .{ "self_attn.o_proj", "attn_output" },
    // Standard attention
    .{ "self_attn.q_norm", "attn_q_norm" },
    .{ "self_attn.k_norm", "attn_k_norm" },
    .{ "self_attn.q_proj", "attn_q" },
    .{ "self_attn.k_proj", "attn_k" },
    .{ "self_attn.v_proj", "attn_v" },
    .{ "input_layernorm", "attn_norm" },
    .{ "post_attention_layernorm", "ffn_norm" },
    .{ "pre_feedforward_layernorm", "ffn_norm" },
    .{ "post_feedforward_layernorm", "post_ffw_norm" },
    // Dense FFN
    .{ "mlp.gate_proj", "ffn_gate" },
    .{ "mlp.up_proj", "ffn_up" },
    .{ "mlp.down_proj", "ffn_down" },
    // MoE routing — longer prefixes first
    .{ "mlp.gate.e_score_correction_bias", "exp_probs_b.bias" },
    .{ "mlp.gate", "ffn_gate_inp" },
    .{ "mlp.router", "ffn_gate_inp" },
    // MoE experts
    .{ "mlp.switch_mlp.gate_proj", "ffn_gate_exps" },
    .{ "mlp.switch_mlp.up_proj", "ffn_up_exps" },
    .{ "mlp.switch_mlp.down_proj", "ffn_down_exps" },
    .{ "mlp.experts.gate_proj", "ffn_gate_exps" },
    .{ "mlp.experts.up_proj", "ffn_up_exps" },
    .{ "mlp.experts.down_proj", "ffn_down_exps" },
    // Shared expert
    .{ "mlp.shared_experts.gate_proj", "ffn_gate_shexp" },
    .{ "mlp.shared_experts.up_proj", "ffn_up_shexp" },
    .{ "mlp.shared_experts.down_proj", "ffn_down_shexp" },
};

/// Translate an HF-style tensor name to GGUF-style.
/// Handles top-level tensors and layer tensors with prefix stripping.
fn hfNameToGguf(name: []const u8, buf: *[name_buf_size]u8) ?[]const u8 {
    // Strip known HF prefixes (multimodal and plain)
    const stripped = for ([_][]const u8{ "language_model.model.", "model." }) |pfx| {
        if (std.mem.startsWith(u8, name, pfx)) break name[pfx.len..];
    } else name;

    // Top-level tensors
    if (std.mem.startsWith(u8, stripped, "embed_tokens.")) {
        return std.fmt.bufPrint(buf, "token_embd.{s}", .{stripped["embed_tokens.".len..]}) catch null;
    }
    if (std.mem.startsWith(u8, stripped, "norm.")) {
        return std.fmt.bufPrint(buf, "output_norm.{s}", .{stripped["norm.".len..]}) catch null;
    }
    if (std.mem.startsWith(u8, stripped, "lm_head.")) {
        return std.fmt.bufPrint(buf, "output.{s}", .{stripped["lm_head.".len..]}) catch null;
    }

    // Layer tensors: layers.{i}.{hf_component}.{attr} → blk.{i}.{gguf_component}.{attr}
    if (std.mem.startsWith(u8, stripped, "layers.")) {
        const rest = stripped["layers.".len..];
        const dot1 = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        const layer_str = rest[0..dot1];
        const suffix = rest[dot1 + 1 ..];

        for (hf_gguf_layer_map) |mapping| {
            if (std.mem.startsWith(u8, suffix, mapping[0])) {
                const attr = suffix[mapping[0].len..]; // e.g., ".weight"
                return std.fmt.bufPrint(buf, "blk.{s}.{s}{s}", .{ layer_str, mapping[1], attr }) catch null;
            }
        }
    }
    return null;
}

/// HF config key → GGUF metadata suffix (without arch prefix).
const hf_gguf_meta_map = [_]struct { []const u8, []const u8 }{
    .{ "num_hidden_layers", "block_count" },
    .{ "hidden_size", "embedding_length" },
    .{ "num_attention_heads", "attention.head_count" },
    .{ "num_key_value_heads", "attention.head_count_kv" },
    .{ "head_dim", "attention.key_length" },
    .{ "intermediate_size", "feed_forward_length" },
    .{ "max_position_embeddings", "context_length" },
    .{ "context_length", "context_length" },
    .{ "rope_theta", "rope.freq_base" },
    .{ "rms_norm_eps", "attention.layer_norm_rms_epsilon" },
    .{ "vocab_size", "vocab_size" },
    // MLA/DeepSeek2 specific
    .{ "q_lora_rank", "attention.q_lora_rank" },
    .{ "kv_lora_rank", "attention.kv_lora_rank" },
    .{ "qk_rope_head_dim", "rope.dimension_count" },
    .{ "n_routed_experts", "expert_count" },
    .{ "num_experts_per_tok", "expert_used_count" },
    .{ "moe_intermediate_size", "expert_feed_forward_length" },
    .{ "first_k_dense_replace", "leading_dense_block_count" },
    .{ "routed_scaling_factor", "expert_weights_scale" },
};

/// Map an HF config.json key to a GGUF metadata suffix (without the arch prefix).
fn hfKeyToGgufSuffix(key: []const u8) ?[]const u8 {
    for (hf_gguf_meta_map) |mapping| {
        if (std.mem.eql(u8, key, mapping[0])) return mapping[1];
    }
    return null;
}

// ── Tests ─────────────────────────────────────────────────────────

test "hfNameToGguf top-level" {
    var buf: [name_buf_size]u8 = undefined;
    try std.testing.expectEqualStrings("token_embd.weight", hfNameToGguf("model.embed_tokens.weight", &buf).?);
    try std.testing.expectEqualStrings("output_norm.weight", hfNameToGguf("model.norm.weight", &buf).?);
    try std.testing.expectEqualStrings("output.weight", hfNameToGguf("model.lm_head.weight", &buf).?);
    try std.testing.expectEqualStrings("output.weight", hfNameToGguf("language_model.model.lm_head.weight", &buf).?);
}

test "hfNameToGguf layer tensors" {
    var buf: [name_buf_size]u8 = undefined;
    try std.testing.expectEqualStrings("blk.0.attn_norm.weight", hfNameToGguf("model.layers.0.input_layernorm.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.5.attn_q_a.weight", hfNameToGguf("model.layers.5.self_attn.q_a_proj.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.1.ffn_gate_inp.weight", hfNameToGguf("model.layers.1.mlp.gate.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.1.exp_probs_b.bias", hfNameToGguf("model.layers.1.mlp.gate.e_score_correction_bias", &buf).?);
    try std.testing.expectEqualStrings("blk.2.ffn_gate_exps.weight", hfNameToGguf("model.layers.2.mlp.switch_mlp.gate_proj.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.3.ffn_down_shexp.weight", hfNameToGguf("model.layers.3.mlp.shared_experts.down_proj.weight", &buf).?);
    try std.testing.expect(hfNameToGguf("model.layers.0.unknown_component.weight", &buf) == null);
}

test "hfKeyToGgufSuffix metadata keys" {
    try std.testing.expectEqualStrings("block_count", hfKeyToGgufSuffix("num_hidden_layers").?);
    try std.testing.expectEqualStrings("embedding_length", hfKeyToGgufSuffix("hidden_size").?);
    try std.testing.expectEqualStrings("rope.freq_base", hfKeyToGgufSuffix("rope_theta").?);
    try std.testing.expectEqualStrings("expert_count", hfKeyToGgufSuffix("n_routed_experts").?);
    try std.testing.expect(hfKeyToGgufSuffix("unknown_key") == null);
}

test "GGMLType blockSize" {
    try std.testing.expectEqual(@as(usize, 32), GGMLType.q4_0.blockSize());
    try std.testing.expectEqual(@as(usize, 32), GGMLType.q8_0.blockSize());
    try std.testing.expectEqual(@as(usize, 256), GGMLType.q6_k.blockSize());
    try std.testing.expectEqual(@as(usize, 256), GGMLType.q2_k.blockSize());
    try std.testing.expectEqual(@as(usize, 1), GGMLType.f32.blockSize());
    try std.testing.expectEqual(@as(usize, 1), GGMLType.f16.blockSize());
    try std.testing.expectEqual(@as(usize, 1), GGMLType.bf16.blockSize());
    try std.testing.expectEqual(@as(usize, 256), GGMLType.tq1_0.blockSize());
    try std.testing.expectEqual(@as(usize, 32), GGMLType.mxfp4.blockSize());
}

test "GGMLType bytesPerBlock" {
    try std.testing.expectEqual(@as(usize, 18), GGMLType.q4_0.bytesPerBlock());
    try std.testing.expectEqual(@as(usize, 34), GGMLType.q8_0.bytesPerBlock());
    try std.testing.expectEqual(@as(usize, 210), GGMLType.q6_k.bytesPerBlock());
    try std.testing.expectEqual(@as(usize, 4), GGMLType.f32.bytesPerBlock());
    try std.testing.expectEqual(@as(usize, 2), GGMLType.f16.bytesPerBlock());
    try std.testing.expectEqual(@as(usize, 17), GGMLType.mxfp4.bytesPerBlock());
    try std.testing.expectEqual(@as(usize, 64), GGMLType.tq1_0.bytesPerBlock());
}

test "GGMLType tensorBytes" {
    // 32 elements of Q4_0 = 1 block × 18 bytes
    try std.testing.expectEqual(@as(usize, 18), GGMLType.q4_0.tensorBytes(32));
    // 64 elements of Q4_0 = 2 blocks × 18 bytes
    try std.testing.expectEqual(@as(usize, 36), GGMLType.q4_0.tensorBytes(64));
    // 256 elements of Q6_K = 1 super-block × 210 bytes
    try std.testing.expectEqual(@as(usize, 210), GGMLType.q6_k.tensorBytes(256));
    // 100 elements of F32 = 100 × 4 bytes
    try std.testing.expectEqual(@as(usize, 400), GGMLType.f32.tensorBytes(100));
}

test "ggmlToDType mapping" {
    try std.testing.expectEqual(DType.f32, GGUFFile.ggmlToDType(.f32));
    try std.testing.expectEqual(DType.f16, GGUFFile.ggmlToDType(.f16));
    try std.testing.expectEqual(DType.bf16, GGUFFile.ggmlToDType(.bf16));
    try std.testing.expectEqual(DType.q4_0, GGUFFile.ggmlToDType(.q4_0));
    try std.testing.expectEqual(DType.q8_0, GGUFFile.ggmlToDType(.q8_0));
    try std.testing.expectEqual(DType.q2_k, GGUFFile.ggmlToDType(.q2_k));
    try std.testing.expectEqual(DType.q3_k, GGUFFile.ggmlToDType(.q3_k));
    try std.testing.expectEqual(DType.q4_k, GGUFFile.ggmlToDType(.q4_k));
    try std.testing.expectEqual(DType.iq4_nl, GGUFFile.ggmlToDType(.iq4_nl));
    try std.testing.expectEqual(DType.iq4_xs, GGUFFile.ggmlToDType(.iq4_xs));
    try std.testing.expectEqual(DType.unknown, GGUFFile.ggmlToDType(.i8));
}

test "MetaValue conversions" {
    const u32_val = MetaValue{ .uint32 = 42 };
    try std.testing.expectEqual(@as(?u32, 42), u32_val.asU32());
    try std.testing.expectEqual(@as(?f32, null), u32_val.asF32());

    const f32_val = MetaValue{ .float32 = 3.14 };
    try std.testing.expectApproxEqAbs(@as(f32, 3.14), f32_val.asF32().?, 0.001);
    try std.testing.expectEqual(@as(?u32, null), f32_val.asU32());

    const str_val = MetaValue{ .string = "hello" };
    try std.testing.expectEqualStrings("hello", str_val.asStr().?);

    const bool_val = MetaValue{ .bool_val = true };
    try std.testing.expectEqual(@as(?bool, true), bool_val.asBool());
}

test "dequantQ4_0 correctness" {
    // Build a Q4_0 block: f16 scale = 0.5, varied nibble patterns
    var block: [18]u8 = undefined;
    // f16 0.5 = 0x3800
    std.mem.writeInt(u16, block[0..2], 0x3800, .little);
    // byte[0] = 0xA3 → low=3 (3-8=-5)*0.5=-2.5, high=0xA=10 (10-8=2)*0.5=1.0
    block[2] = 0xA3;
    // byte[1] = 0x0F → low=0xF=15 (15-8=7)*0.5=3.5, high=0 (0-8=-8)*0.5=-4.0
    block[3] = 0x0F;
    // Fill rest with zero-offset nibbles for simpler verification
    @memset(block[4..], 0x88);

    var output: [32]f32 = undefined;
    GGUFFile.dequantQ4_0(&block, &output);

    try std.testing.expectApproxEqAbs(@as(f32, -2.5), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -4.0), output[3], 0.001);
    // Remaining elements use zero-offset nibbles (8-8)*0.5 = 0
    for (output[4..]) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), v, 0.001);
    }
}

test "dequantQ4_0 nonzero" {
    var block: [18]u8 = undefined;
    // f16 2.0 = 0x4000
    std.mem.writeInt(u16, block[0..2], 0x4000, .little);
    // byte = 0x0F → low nibble = 0xF = 15, high nibble = 0
    // dequant: lo = (15 - 8) * 2.0 = 14.0, hi = (0 - 8) * 2.0 = -16.0
    @memset(block[2..], 0x0F);

    var output: [32]f32 = undefined;
    GGUFFile.dequantQ4_0(&block, &output);
    // Verify ALL 32 outputs, not just the first two.
    // Even indices (from low nibble): (15 - 8) * 2.0 = 14.0
    // Odd indices (from high nibble): (0 - 8) * 2.0 = -16.0
    for (0..16) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 14.0), output[i * 2], 0.001);
        try std.testing.expectApproxEqAbs(@as(f32, -16.0), output[i * 2 + 1], 0.001);
    }
}

test "dequantQ4_0 mixed nibbles" {
    var block: [18]u8 = undefined;
    // f16 1.0 = 0x3C00
    std.mem.writeInt(u16, block[0..2], 0x3C00, .little);
    // byte[0] = 0x19 → low=9 (9-8=1)*1.0=1.0, high=1 (1-8=-7)*1.0=-7.0
    block[2] = 0x19;
    // byte[1] = 0xF0 → low=0 (0-8=-8)*1.0=-8.0, high=15 (15-8=7)*1.0=7.0
    block[3] = 0xF0;
    @memset(block[4..], 0x88); // rest = zero

    var output: [32]f32 = undefined;
    GGUFFile.dequantQ4_0(&block, &output);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7.0), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -8.0), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), output[3], 0.001);
}

test "dequantQ8_0 correctness" {
    var block: [34]u8 = undefined;
    // f16 0.5 = 0x3800
    std.mem.writeInt(u16, block[0..2], 0x3800, .little);
    // Set quant value 0 = 10 (as i8), value 1 = -5 (as i8)
    block[2] = @bitCast(@as(i8, 10));
    block[3] = @bitCast(@as(i8, -5));
    // Set boundary values: max i8 = 127, min i8 = -128
    block[4] = @bitCast(@as(i8, 127));
    block[5] = @bitCast(@as(i8, -128));
    @memset(block[6..], 0);

    var output: [32]f32 = undefined;
    GGUFFile.dequantQ8_0(&block, &output);
    // 10 * 0.5 = 5.0
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), output[0], 0.001);
    // -5 * 0.5 = -2.5
    try std.testing.expectApproxEqAbs(@as(f32, -2.5), output[1], 0.001);
    // 127 * 0.5 = 63.5
    try std.testing.expectApproxEqAbs(@as(f32, 63.5), output[2], 0.001);
    // -128 * 0.5 = -64.0
    try std.testing.expectApproxEqAbs(@as(f32, -64.0), output[3], 0.001);
    // Remaining zeros: 0 * 0.5 = 0.0
    for (output[4..]) |v| try std.testing.expectApproxEqAbs(@as(f32, 0.0), v, 0.001);
}

test "ggmlToDType complete mapping" {
    // Verify all explicitly-mapped GGML types produce the expected DType
    const mapping = [_]struct { GGMLType, DType }{
        .{ .f32, .f32 },
        .{ .f16, .f16 },
        .{ .bf16, .bf16 },
        .{ .q2_k, .q2_k },
        .{ .q3_k, .q3_k },
        .{ .q4_0, .q4_0 },
        .{ .q4_1, .q4_1 },
        .{ .q4_k, .q4_k },
        .{ .q5_0, .q5_0 },
        .{ .q5_k, .q5_k },
        .{ .q6_k, .q6_k },
        .{ .q8_0, .q8_0 },
        .{ .iq4_xs, .iq4_xs },
        .{ .iq4_nl, .iq4_nl },
        .{ .tq1_0, .tq1_0 },
        .{ .mxfp4, .mxfp4 },
    };
    for (mapping) |m| {
        try std.testing.expectEqual(m[1], GGUFFile.ggmlToDType(m[0]));
    }
    // Types that map to unknown
    const unknown_types = [_]GGMLType{ .i8, .i16, .i32, .i64, .f64, .q5_1, .q8_1, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq1_s, .iq1_m, .tq2_0 };
    for (unknown_types) |t| {
        try std.testing.expectEqual(DType.unknown, GGUFFile.ggmlToDType(t));
    }
}

test "GGMLType blockSize all types" {
    // Scalar types: blockSize = 1
    const scalar_types = [_]GGMLType{ .f32, .f16, .i8, .i16, .i32, .i64, .f64, .bf16 };
    for (scalar_types) |t| {
        try std.testing.expectEqual(@as(usize, 1), t.blockSize());
    }
    // 32-element block types
    const block32_types = [_]GGMLType{ .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1, .iq4_nl, .mxfp4 };
    for (block32_types) |t| {
        try std.testing.expectEqual(@as(usize, 32), t.blockSize());
    }
    // 256-element super-block types
    const block256_types = [_]GGMLType{ .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq1_s, .iq1_m, .iq4_xs, .tq1_0 };
    for (block256_types) |t| {
        try std.testing.expectEqual(@as(usize, 256), t.blockSize());
    }
    // Unknown/invalid type falls to else => 1
    try std.testing.expectEqual(@as(usize, 1), (@as(GGMLType, @enumFromInt(99))).blockSize());
}

test "GGMLType bytesPerBlock all types" {
    // Verify known block sizes match GGML spec
    const expected = [_]struct { GGMLType, usize }{
        .{ .f32, 4 },
        .{ .f16, 2 },
        .{ .bf16, 2 },
        .{ .q4_0, 18 },
        .{ .q4_1, 20 },
        .{ .q5_0, 22 },
        .{ .q5_1, 24 },
        .{ .q8_0, 34 },
        .{ .q8_1, 36 },
        .{ .q2_k, 100 },
        .{ .q3_k, 110 },
        .{ .q4_k, 144 },
        .{ .q5_k, 176 },
        .{ .q6_k, 210 },
        .{ .i8, 1 },
        .{ .i16, 2 },
        .{ .i32, 4 },
        .{ .i64, 8 },
        .{ .f64, 8 },
        .{ .iq2_xxs, 66 },
        .{ .iq2_xs, 74 },
        .{ .iq2_s, 82 },
        .{ .iq3_xxs, 98 },
        .{ .iq3_s, 110 },
        .{ .iq1_s, 50 },
        .{ .iq1_m, 56 },
        .{ .iq4_nl, 18 },
        .{ .iq4_xs, 136 },
        .{ .tq1_0, 64 },
        .{ .mxfp4, 17 },
    };
    for (expected) |e| {
        try std.testing.expectEqual(e[1], e[0].bytesPerBlock());
    }
}

test "GGMLType tensorBytes edge cases" {
    // Zero elements
    try std.testing.expectEqual(@as(usize, 0), GGMLType.f32.tensorBytes(0));
    // Non-aligned element count rounds up to next block
    // 33 Q4_0 elements = ceil(33/32) = 2 blocks = 36 bytes
    try std.testing.expectEqual(@as(usize, 36), GGMLType.q4_0.tensorBytes(33));
    // 1 element of Q4_0 = 1 block = 18 bytes
    try std.testing.expectEqual(@as(usize, 18), GGMLType.q4_0.tensorBytes(1));
    // 257 elements of Q6_K = ceil(257/256) = 2 blocks = 420 bytes
    try std.testing.expectEqual(@as(usize, 420), GGMLType.q6_k.tensorBytes(257));
    // Large but valid element count
    try std.testing.expectEqual(@as(usize, 4096 * 4), GGMLType.f32.tensorBytes(4096));
    // Overflow saturation: huge element count shouldn't crash
    const huge = GGMLType.f32.tensorBytes(std.math.maxInt(usize));
    try std.testing.expectEqual(std.math.maxInt(usize), huge);
}

test "GGMLType blockSize and bytesPerBlock consistency" {
    // For scalar types, bytesPerBlock should be the type's natural size
    try std.testing.expectEqual(@as(usize, 4), GGMLType.f32.bytesPerBlock());
    try std.testing.expectEqual(@as(usize, 1), GGMLType.f32.blockSize());
    // tensorBytes(1) = 1 * 4 = 4
    try std.testing.expectEqual(@as(usize, 4), GGMLType.f32.tensorBytes(1));

    // For Q4_K: 256 elements per block, 144 bytes per block
    // 512 elements = 2 blocks = 288 bytes
    try std.testing.expectEqual(@as(usize, 288), GGMLType.q4_k.tensorBytes(512));
    // Verify: n_blocks = ceil(512/256) = 2, 2 * 144 = 288
    const n_blocks = (512 + 256 - 1) / 256;
    try std.testing.expectEqual(@as(usize, 288), n_blocks * GGMLType.q4_k.bytesPerBlock());
}

test "MetaValue asU32 type coercion" {
    // u8 → u32 widening
    try std.testing.expectEqual(@as(?u32, 255), (MetaValue{ .uint8 = 255 }).asU32());
    try std.testing.expectEqual(@as(?u32, 0), (MetaValue{ .uint8 = 0 }).asU32());
    // u16 → u32 widening
    try std.testing.expectEqual(@as(?u32, 65535), (MetaValue{ .uint16 = 65535 }).asU32());
    // i32 → u32 (positive)
    try std.testing.expectEqual(@as(?u32, 100), (MetaValue{ .int32 = 100 }).asU32());
    // i32 → u32 (negative = null)
    try std.testing.expectEqual(@as(?u32, null), (MetaValue{ .int32 = -1 }).asU32());
    // u64 → u32 (in range)
    try std.testing.expectEqual(@as(?u32, 42), (MetaValue{ .uint64 = 42 }).asU32());
    // u64 → u32 (out of range = null)
    try std.testing.expectEqual(@as(?u32, null), (MetaValue{ .uint64 = @as(u64, std.math.maxInt(u32)) + 1 }).asU32());
    // u64 → u32 (max u32 = ok)
    try std.testing.expectEqual(@as(?u32, std.math.maxInt(u32)), (MetaValue{ .uint64 = std.math.maxInt(u32) }).asU32());
    // Types that never convert
    try std.testing.expectEqual(@as(?u32, null), (MetaValue{ .float32 = 1.0 }).asU32());
    try std.testing.expectEqual(@as(?u32, null), (MetaValue{ .string = "42" }).asU32());
    try std.testing.expectEqual(@as(?u32, null), (MetaValue{ .bool_val = true }).asU32());
    try std.testing.expectEqual(@as(?u32, null), (MetaValue{ .int64 = 42 }).asU32());
    try std.testing.expectEqual(@as(?u32, null), (MetaValue{ .float64 = 42.0 }).asU32());
}

test "MetaValue asF32 type coercion" {
    // f32 passthrough
    try std.testing.expectApproxEqAbs(@as(f32, 2.718), (MetaValue{ .float32 = 2.718 }).asF32().?, 0.001);
    // f64 → f32 narrowing
    try std.testing.expectApproxEqAbs(@as(f32, 1e-5), (MetaValue{ .float64 = 1e-5 }).asF32().?, 1e-10);
    // Types that never convert
    try std.testing.expectEqual(@as(?f32, null), (MetaValue{ .uint32 = 42 }).asF32());
    try std.testing.expectEqual(@as(?f32, null), (MetaValue{ .string = "3.14" }).asF32());
}

test "MetaValue asStr and asBool null cases" {
    // asStr returns null for non-string
    try std.testing.expectEqual(@as(?[]const u8, null), (MetaValue{ .uint32 = 42 }).asStr());
    try std.testing.expectEqual(@as(?[]const u8, null), (MetaValue{ .float32 = 1.0 }).asStr());
    // asBool returns null for non-bool
    try std.testing.expectEqual(@as(?bool, null), (MetaValue{ .uint32 = 1 }).asBool());
    try std.testing.expectEqual(@as(?bool, null), (MetaValue{ .string = "true" }).asBool());
    // asBool with false
    try std.testing.expectEqual(@as(?bool, false), (MetaValue{ .bool_val = false }).asBool());
}

test "TensorInfo numElements and dataBytes" {
    const info = TensorInfo{
        .name = "test",
        .n_dims = 2,
        .dims = .{ 4096, 4096, 0, 0 },
        .ggml_type = .q4_k,
        .offset = 0,
    };
    try std.testing.expectEqual(@as(usize, 4096 * 4096), info.numElements());
    // Q4_K: ceil(16777216 / 256) = 65536 blocks * 144 bytes = 9437184
    try std.testing.expectEqual(@as(usize, 65536 * 144), info.dataBytes());
}

test "TensorInfo numElements zero dim" {
    const info = TensorInfo{
        .name = "test",
        .n_dims = 2,
        .dims = .{ 0, 4096, 0, 0 },
        .ggml_type = .f32,
        .offset = 0,
    };
    try std.testing.expectEqual(@as(usize, 0), info.numElements());
}

test "TensorInfo numElements 1D and scalar" {
    // 1D vector
    const vec = TensorInfo{
        .name = "vec",
        .n_dims = 1,
        .dims = .{ 128, 0, 0, 0 },
        .ggml_type = .f32,
        .offset = 0,
    };
    try std.testing.expectEqual(@as(usize, 128), vec.numElements());
    try std.testing.expectEqual(@as(usize, 512), vec.dataBytes()); // 128 * 4

    // Scalar (0-D)
    const scalar = TensorInfo{
        .name = "scalar",
        .n_dims = 0,
        .dims = .{ 0, 0, 0, 0 },
        .ggml_type = .f32,
        .offset = 0,
    };
    try std.testing.expectEqual(@as(usize, 1), scalar.numElements());
    try std.testing.expectEqual(@as(usize, 4), scalar.dataBytes()); // 1 * 4
}

test "hfNameToGguf MLA attention mappings" {
    var buf: [name_buf_size]u8 = undefined;
    // MLA attention (DeepSeek2/GLM-4)
    try std.testing.expectEqualStrings("blk.0.attn_q_a_norm.weight", hfNameToGguf("model.layers.0.self_attn.q_a_layernorm.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.attn_kv_a_norm.weight", hfNameToGguf("model.layers.0.self_attn.kv_a_layernorm.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.attn_q_b.weight", hfNameToGguf("model.layers.0.self_attn.q_b_proj.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.attn_kv_a_mqa.weight", hfNameToGguf("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.attn_k_b.weight", hfNameToGguf("model.layers.0.self_attn.embed_q.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.attn_v_b.weight", hfNameToGguf("model.layers.0.self_attn.unembed_out.weight", &buf).?);
}

test "hfNameToGguf standard attention and FFN" {
    var buf: [name_buf_size]u8 = undefined;
    // Standard attention
    try std.testing.expectEqualStrings("blk.0.attn_q_norm.weight", hfNameToGguf("model.layers.0.self_attn.q_norm.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.attn_k_norm.weight", hfNameToGguf("model.layers.0.self_attn.k_norm.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.attn_k.weight", hfNameToGguf("model.layers.0.self_attn.k_proj.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.attn_v.weight", hfNameToGguf("model.layers.0.self_attn.v_proj.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.attn_output.weight", hfNameToGguf("model.layers.0.self_attn.o_proj.weight", &buf).?);
    // FFN
    try std.testing.expectEqualStrings("blk.0.ffn_gate.weight", hfNameToGguf("model.layers.0.mlp.gate_proj.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.ffn_up.weight", hfNameToGguf("model.layers.0.mlp.up_proj.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.ffn_down.weight", hfNameToGguf("model.layers.0.mlp.down_proj.weight", &buf).?);
    // Norms
    try std.testing.expectEqualStrings("blk.0.ffn_norm.weight", hfNameToGguf("model.layers.0.post_attention_layernorm.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.ffn_norm.weight", hfNameToGguf("model.layers.0.pre_feedforward_layernorm.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.post_ffw_norm.weight", hfNameToGguf("model.layers.0.post_feedforward_layernorm.weight", &buf).?);
}

test "hfNameToGguf MoE expert mappings" {
    var buf: [name_buf_size]u8 = undefined;
    // MoE routing
    try std.testing.expectEqualStrings("blk.0.ffn_gate_inp.weight", hfNameToGguf("model.layers.0.mlp.router.weight", &buf).?);
    // MoE experts (switch_mlp variant)
    try std.testing.expectEqualStrings("blk.0.ffn_up_exps.weight", hfNameToGguf("model.layers.0.mlp.switch_mlp.up_proj.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.ffn_down_exps.weight", hfNameToGguf("model.layers.0.mlp.switch_mlp.down_proj.weight", &buf).?);
    // MoE experts (experts variant)
    try std.testing.expectEqualStrings("blk.0.ffn_gate_exps.weight", hfNameToGguf("model.layers.0.mlp.experts.gate_proj.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.ffn_up_exps.weight", hfNameToGguf("model.layers.0.mlp.experts.up_proj.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.ffn_down_exps.weight", hfNameToGguf("model.layers.0.mlp.experts.down_proj.weight", &buf).?);
    // Shared expert
    try std.testing.expectEqualStrings("blk.0.ffn_gate_shexp.weight", hfNameToGguf("model.layers.0.mlp.shared_experts.gate_proj.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.ffn_up_shexp.weight", hfNameToGguf("model.layers.0.mlp.shared_experts.up_proj.weight", &buf).?);
    try std.testing.expectEqualStrings("blk.0.ffn_down_shexp.weight", hfNameToGguf("model.layers.0.mlp.shared_experts.down_proj.weight", &buf).?);
}

test "hfNameToGguf multimodal prefix stripping" {
    var buf: [name_buf_size]u8 = undefined;
    // language_model.model. prefix
    try std.testing.expectEqualStrings("blk.0.attn_q.weight", hfNameToGguf("language_model.model.layers.0.self_attn.q_proj.weight", &buf).?);
    try std.testing.expectEqualStrings("token_embd.weight", hfNameToGguf("language_model.model.embed_tokens.weight", &buf).?);
}

test "hfNameToGguf returns null for unrecognized names" {
    var buf: [name_buf_size]u8 = undefined;
    try std.testing.expect(hfNameToGguf("completely.unrelated.tensor", &buf) == null);
    try std.testing.expect(hfNameToGguf("model.layers.0.unknown_component.weight", &buf) == null);
    try std.testing.expect(hfNameToGguf("", &buf) == null);
    try std.testing.expect(hfNameToGguf("model.layers.", &buf) == null);
}

test "hfKeyToGgufSuffix complete mapping" {
    // Verify all entries in hf_gguf_meta_map are accessible
    const expected = [_]struct { []const u8, []const u8 }{
        .{ "num_hidden_layers", "block_count" },
        .{ "hidden_size", "embedding_length" },
        .{ "num_attention_heads", "attention.head_count" },
        .{ "num_key_value_heads", "attention.head_count_kv" },
        .{ "head_dim", "attention.key_length" },
        .{ "intermediate_size", "feed_forward_length" },
        .{ "max_position_embeddings", "context_length" },
        .{ "context_length", "context_length" },
        .{ "rope_theta", "rope.freq_base" },
        .{ "rms_norm_eps", "attention.layer_norm_rms_epsilon" },
        .{ "vocab_size", "vocab_size" },
        .{ "q_lora_rank", "attention.q_lora_rank" },
        .{ "kv_lora_rank", "attention.kv_lora_rank" },
        .{ "qk_rope_head_dim", "rope.dimension_count" },
        .{ "n_routed_experts", "expert_count" },
        .{ "num_experts_per_tok", "expert_used_count" },
        .{ "moe_intermediate_size", "expert_feed_forward_length" },
        .{ "first_k_dense_replace", "leading_dense_block_count" },
        .{ "routed_scaling_factor", "expert_weights_scale" },
    };
    for (expected) |e| {
        const result = hfKeyToGgufSuffix(e[0]) orelse {
            std.debug.print("Missing mapping for: {s}\n", .{e[0]});
            return error.MissingMapping;
        };
        try std.testing.expectEqualStrings(e[1], result);
    }
}

test "GGUF fromBuffer invalid magic" {
    const allocator = std.testing.allocator;
    // Valid header structure but wrong magic bytes
    var buf: [24]u8 = undefined;
    std.mem.writeInt(u32, buf[0..4], 0xDEADBEEF, .little); // bad magic
    std.mem.writeInt(u32, buf[4..8], 3, .little); // version
    std.mem.writeInt(u64, buf[8..16], 0, .little); // tensor_count
    std.mem.writeInt(u64, buf[16..24], 0, .little); // metadata_kv_count

    const result = GGUFFile.fromBuffer(allocator, &buf);
    try std.testing.expectError(error.InvalidMagic, result);
}

test "GGUF fromBuffer unsupported version" {
    const allocator = std.testing.allocator;
    var buf: [24]u8 = undefined;
    std.mem.writeInt(u32, buf[0..4], gguf_magic, .little);
    std.mem.writeInt(u32, buf[4..8], 1, .little); // version 1 (too old)
    std.mem.writeInt(u64, buf[8..16], 0, .little);
    std.mem.writeInt(u64, buf[16..24], 0, .little);

    try std.testing.expectError(error.UnsupportedVersion, GGUFFile.fromBuffer(allocator, &buf));

    // Version 4 (too new)
    std.mem.writeInt(u32, buf[4..8], 4, .little);
    try std.testing.expectError(error.UnsupportedVersion, GGUFFile.fromBuffer(allocator, &buf));
}

test "GGUF fromBuffer file too small" {
    const allocator = std.testing.allocator;
    var buf: [4]u8 = .{ 0, 0, 0, 0 };
    try std.testing.expectError(error.FileTooSmall, GGUFFile.fromBuffer(allocator, &buf));
}

test "GGUF fromBuffer minimal valid file" {
    const allocator = std.testing.allocator;
    // Minimal valid GGUF: magic + version 3 + 0 tensors + 0 metadata.
    // Header is 24 bytes, but data_offset is aligned to 32 (default alignment),
    // so the buffer must be at least 32 bytes.
    var buf: [32]u8 = undefined;
    @memset(&buf, 0);
    std.mem.writeInt(u32, buf[0..4], gguf_magic, .little);
    std.mem.writeInt(u32, buf[4..8], 3, .little);
    std.mem.writeInt(u64, buf[8..16], 0, .little);
    std.mem.writeInt(u64, buf[16..24], 0, .little);

    var gguf = try GGUFFile.fromBuffer(allocator, &buf);
    defer gguf.deinit();

    try std.testing.expectEqual(@as(u32, 3), gguf.version);
    try std.testing.expectEqual(@as(u64, 0), gguf.tensor_count);
    try std.testing.expectEqual(@as(usize, 0), gguf.metadata.count());
    try std.testing.expectEqual(@as(usize, 0), gguf.tensors.count());
    try std.testing.expectEqual(@as(u64, 0), gguf.totalParams());
}

test "fuzz: all gguf functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // --- GGMLType: blockSize, bytesPerBlock, tensorBytes ---
            const ggml_raw = smith.valueWithHash(u32, 0);
            const ggml_type: GGMLType = @enumFromInt(ggml_raw);
            const bs = ggml_type.blockSize();
            std.debug.assert(bs >= 1);
            const bpb = ggml_type.bytesPerBlock();
            std.debug.assert(bpb >= 1);
            const n_elems = smith.valueWithHash(usize, 1) % 100_000;
            const tb = ggml_type.tensorBytes(n_elems);
            _ = tb;

            // --- MetaValue: asU32, asF32, asStr, asBool ---
            const mv_tag = smith.valueWithHash(u8, 2) % 7;
            const mv: MetaValue = switch (mv_tag) {
                0 => .{ .uint32 = smith.valueWithHash(u32, 3) },
                1 => .{ .int32 = smith.valueWithHash(i32, 4) },
                2 => .{ .float32 = @bitCast(smith.valueWithHash(u32, 5)) },
                3 => .{ .bool_val = smith.valueWithHash(u8, 6) & 1 == 1 },
                4 => .{ .uint8 = smith.valueWithHash(u8, 7) },
                5 => .{ .uint16 = smith.valueWithHash(u16, 8) },
                else => .{ .uint64 = smith.valueWithHash(u64, 9) },
            };
            _ = mv.asU32();
            _ = mv.asF32();
            _ = mv.asStr();
            _ = mv.asBool();

            // --- TensorInfo: numElements, dataBytes ---
            const ti = TensorInfo{
                .name = "fuzz",
                .n_dims = smith.valueWithHash(u32, 10) % 5,
                .dims = .{
                    smith.valueWithHash(u64, 11) % 4096,
                    smith.valueWithHash(u64, 12) % 4096,
                    smith.valueWithHash(u64, 13) % 64,
                    smith.valueWithHash(u64, 14) % 64,
                },
                .ggml_type = @enumFromInt(smith.valueWithHash(u32, 15) % 40),
                .offset = 0,
            };
            const ne = ti.numElements();
            _ = ne;
            const db = ti.dataBytes();
            _ = db;

            // --- GGUFFile: fromBuffer (exercises parseHeader), deinit ---
            // Build random buffer that might or might not be valid GGUF
            var hdr_buf: [64]u8 = undefined;
            for (&hdr_buf, 0..) |*b, i| b.* = smith.valueWithHash(u8, @intCast(16 + i));
            // Also try a valid header sometimes
            if (smith.valueWithHash(u8, 80) & 1 == 0) {
                std.mem.writeInt(u32, hdr_buf[0..4], gguf_magic, .little);
                std.mem.writeInt(u32, hdr_buf[4..8], 3, .little);
                std.mem.writeInt(u64, hdr_buf[8..16], 0, .little);
                std.mem.writeInt(u64, hdr_buf[16..24], 0, .little);
            }
            const allocator = std.testing.allocator;
            if (GGUFFile.fromBuffer(allocator, &hdr_buf)) |*gguf_ptr| {
                var gguf = gguf_ptr.*;
                // format (returns a Format vtable wrapper)
                _ = gguf.format();
                // totalParams
                _ = gguf.totalParams();
                // getMetaStr, getMetaU32, getMetaF32, getMetaU32Array
                _ = gguf.getMetaStr("fuzz_key");
                _ = gguf.getMetaU32("fuzz_key");
                _ = gguf.getMetaF32("fuzz_key");
                _ = gguf.getMetaU32Array("fuzz_key");
                // getTokenizerVocab, getTokenizerMerges
                _ = gguf.getTokenizerVocab();
                _ = gguf.getTokenizerMerges();
                // tensorData — only safe if tensors exist
                var it = gguf.tensors.valueIterator();
                if (it.next()) |info| _ = gguf.tensorData(info);
                gguf.deinit();
            } else |_| {}

            // --- GGUFFile.open — errors on non-existent path ---
            _ = GGUFFile.open(allocator, "/__nonexistent_fuzz_path__") catch {};

            // --- dequantQ4_0 ---
            var q4_block align(2) = [_]u8{0} ** 18;
            for (&q4_block, 0..) |*b, i| b.* = smith.valueWithHash(u8, @intCast(81 + i));
            var q4_out: [32]f32 = undefined;
            GGUFFile.dequantQ4_0(&q4_block, &q4_out);
            for (q4_out) |v| if (!std.math.isFinite(v)) return; // NaN/Inf from f16 is ok, just bail

            // --- dequantQ8_0 ---
            var q8_block align(2) = [_]u8{0} ** 34;
            for (&q8_block, 0..) |*b, i| b.* = smith.valueWithHash(u8, @intCast(99 + i));
            var q8_out: [32]f32 = undefined;
            GGUFFile.dequantQ8_0(&q8_block, &q8_out);
            for (q8_out) |v| if (!std.math.isFinite(v)) return;
        }
    }.f, .{});
}
