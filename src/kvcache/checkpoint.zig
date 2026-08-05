//! KV cache disk checkpointing — save and restore KV state across restarts.
//!
//! Serializes the KV cache to a binary file with a versioned header so that
//! long system prompts and conversation prefixes don't need re-prefilling
//! after a server restart.
//!
//! File format:
//!   [4 bytes] magic: "KVC\x01"
//!   [4 bytes] version: u32 = 1
//!   [4 bytes] payload_abi: u32 = 1 (bumped when KV layout changes)
//!   [4 bytes] n_layers: u32
//!   [4 bytes] kv_dim: u32 (per-layer K or V dimension)
//!   [4 bytes] n_tokens: u32 (number of cached positions)
//!   [4 bytes] reserved: u32 = 0
//!   [payload] n_layers × n_tokens × kv_dim × sizeof(f32) bytes of K data
//!   [payload] n_layers × n_tokens × kv_dim × sizeof(f32) bytes of V data
//!
//! Usage:
//!   try checkpoint.save(allocator, kv_cache, path, n_layers, kv_dim, n_tokens);
//!   const n_restored = try checkpoint.load(allocator, kv_cache, path, n_layers, kv_dim);
//!
//! Based on the KV checkpoint approach from antirez/ds4.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// File magic bytes: "KVC\x01"
const magic = [4]u8{ 'K', 'V', 'C', 0x01 };
/// Current file format version.
const format_version: u32 = 1;
/// Payload ABI version — bumped when the serialized KV layout changes.
const payload_abi: u32 = 1;

/// Header size in bytes.
const header_size: usize = 28;

/// Checkpoint header.
pub const Header = struct {
    magic: [4]u8,
    version: u32,
    payload_abi: u32,
    n_layers: u32,
    kv_dim: u32,
    n_tokens: u32,
    reserved: u32,
};

/// Validate a checkpoint header against expected model parameters.
pub fn validateHeader(header: Header, expected_n_layers: u32, expected_kv_dim: u32) !void {
    if (!std.mem.eql(u8, &header.magic, &magic)) return error.InvalidMagic;
    if (header.version != format_version) return error.UnsupportedVersion;
    if (header.payload_abi != payload_abi) return error.IncompatiblePayloadAbi;
    if (header.n_layers != expected_n_layers) return error.LayerCountMismatch;
    if (header.kv_dim != expected_kv_dim) return error.KvDimMismatch;
    if (header.n_tokens == 0) return error.EmptyCheckpoint;
}

/// Write header bytes to a buffer.
pub fn writeHeader(buf: *[header_size]u8, n_layers: u32, kv_dim: u32, n_tokens: u32) void {
    @memcpy(buf[0..4], &magic);
    std.mem.writeInt(u32, buf[4..8], format_version, .little);
    std.mem.writeInt(u32, buf[8..12], payload_abi, .little);
    std.mem.writeInt(u32, buf[12..16], n_layers, .little);
    std.mem.writeInt(u32, buf[16..20], kv_dim, .little);
    std.mem.writeInt(u32, buf[20..24], n_tokens, .little);
    std.mem.writeInt(u32, buf[24..28], 0, .little); // reserved
}

/// Read header from a buffer.
pub fn readHeader(buf: *const [header_size]u8) Header {
    return Header{
        .magic = buf[0..4].*,
        .version = std.mem.readInt(u32, buf[4..8], .little),
        .payload_abi = std.mem.readInt(u32, buf[8..12], .little),
        .n_layers = std.mem.readInt(u32, buf[12..16], .little),
        .kv_dim = std.mem.readInt(u32, buf[16..20], .little),
        .n_tokens = std.mem.readInt(u32, buf[20..24], .little),
        .reserved = std.mem.readInt(u32, buf[24..28], .little),
    };
}

// ── Tests ────────────────────────────────────────────────────────

test "header round-trip" {
    var buf: [header_size]u8 = undefined;
    writeHeader(&buf, 42, 2048, 1024);
    const h = readHeader(&buf);
    try std.testing.expectEqualSlices(u8, &magic, &h.magic);
    try std.testing.expectEqual(@as(u32, format_version), h.version);
    try std.testing.expectEqual(@as(u32, payload_abi), h.payload_abi);
    try std.testing.expectEqual(@as(u32, 42), h.n_layers);
    try std.testing.expectEqual(@as(u32, 2048), h.kv_dim);
    try std.testing.expectEqual(@as(u32, 1024), h.n_tokens);
    try std.testing.expectEqual(@as(u32, 0), h.reserved);
}

test "validateHeader rejects wrong magic" {
    const h = Header{ .magic = .{ 'X', 'Y', 'Z', 0 }, .version = 1, .payload_abi = 1, .n_layers = 4, .kv_dim = 128, .n_tokens = 10, .reserved = 0 };
    try std.testing.expectError(error.InvalidMagic, validateHeader(h, 4, 128));
}

test "validateHeader rejects wrong version" {
    const h = Header{ .magic = magic, .version = 99, .payload_abi = 1, .n_layers = 4, .kv_dim = 128, .n_tokens = 10, .reserved = 0 };
    try std.testing.expectError(error.UnsupportedVersion, validateHeader(h, 4, 128));
}

test "validateHeader rejects layer mismatch" {
    const h = Header{ .magic = magic, .version = 1, .payload_abi = 1, .n_layers = 4, .kv_dim = 128, .n_tokens = 10, .reserved = 0 };
    try std.testing.expectError(error.LayerCountMismatch, validateHeader(h, 8, 128));
}

test "validateHeader accepts correct header" {
    const h = Header{ .magic = magic, .version = 1, .payload_abi = 1, .n_layers = 42, .kv_dim = 2048, .n_tokens = 512, .reserved = 0 };
    try validateHeader(h, 42, 2048);
}

test "fuzz: readHeader + validateHeader no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            var buf: [header_size]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const h = readHeader(&buf);
            const expect_layers = smith.valueWithHash(u16, 1);
            const expect_dim = smith.valueWithHash(u16, 2);
            validateHeader(h, expect_layers, expect_dim) catch {};

            // Structure-aware: valid magic/version with fuzzed dimensions
            writeHeader(&buf, smith.valueWithHash(u16, 3), smith.valueWithHash(u16, 4), smith.valueWithHash(u16, 5) +% 1);
            const h2 = readHeader(&buf);
            try std.testing.expectEqualSlices(u8, &magic, &h2.magic);
            try std.testing.expectEqual(format_version, h2.version);
            validateHeader(h2, h2.n_layers, h2.kv_dim) catch |err| {
                try std.testing.expect(err == error.EmptyCheckpoint);
            };
        }
    }.f, .{});
}
