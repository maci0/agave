//! Self-describing header on an exported KV prefix.
//!
//! `exportKvPrefix` used to emit a bare concatenation of the per-layer key and
//! value planes, and `importKvPrefix` accepted anything whose byte count fit.
//! That makes every mismatch silent. A blob from a different architecture, a
//! different cache geometry, or a model whose RoPE base differs restores
//! cleanly and then decodes nonsense, and the bytes arrive over
//! `POST /v1/kv_cache`, so they are untrusted input rather than something the
//! process produced.
//!
//! The header carries what has to agree for stored keys to mean anything at
//! the position they are restored to: cache geometry, and the rotation the
//! keys already carry. A mismatch is refused.
//!
//! Refusing is not repairing. Keys exported at one absolute position cannot be
//! replayed at another without re-rotating them, and no backend has a
//! rope-delta kernel, so a prefix only restores at the offset it was taken
//! from. The header makes that a rejection instead of a wrong answer.

const std = @import("std");

/// Bumped whenever the layout or the meaning of a field changes. A reader
/// refuses anything it does not recognise rather than guessing.
pub const version: u32 = 1;

/// Leading bytes of every exported prefix. Length is part of the layout.
pub const magic = [8]u8{ 'A', 'G', 'V', 'K', 'V', 'P', 'F', 'X' };

const fnv1a_offset_basis: u64 = 0xcbf29ce484222325;
const fnv1a_prime: u64 = 0x100000001b3;

/// Fixed-layout prefix header. `extern` so the field order and padding are
/// the ABI, not whatever the compiler picks: these bytes cross a process
/// boundary.
pub const Header = extern struct {
    magic: [8]u8 = magic,
    version: u32 = @This().current_version,
    n_layers: u32,
    n_tokens: u32,
    /// Bytes of key or value data per token per layer, before the shape hash
    /// disambiguates ragged geometries.
    head_dim: u32,
    _pad: u32 = 0,
    /// Hash of the per-layer KV dimensions. Two models can agree on total
    /// byte count and still lay their caches out differently; this catches
    /// that without carrying a variable-length table.
    shape_hash: u64,
    /// RoPE bases the stored keys were rotated with. Architectures with one
    /// base repeat it in both fields. A model reading back a prefix rotated
    /// at a different base would silently misplace every key.
    rope_theta_a: f32,
    rope_theta_b: f32,

    const current_version = version;

    /// Byte length of the header, and the offset payload data starts at.
    pub const size = @sizeOf(Header);

    /// Whether `other` describes a cache this header's payload can restore
    /// into. Token count is deliberately excluded: it is the payload length,
    /// which the caller checks against its own capacity.
    pub fn compatible(self: Header, other: Header) bool {
        return self.version == other.version and
            self.n_layers == other.n_layers and
            self.head_dim == other.head_dim and
            self.shape_hash == other.shape_hash and
            self.rope_theta_a == other.rope_theta_a and
            self.rope_theta_b == other.rope_theta_b;
    }
};

/// FNV-1a over the per-layer KV dimensions, so a geometry change shows up
/// even when the total size does not.
pub fn shapeHash(kvds: []const usize) u64 {
    var hash: u64 = fnv1a_offset_basis;
    for (kvds) |kvd| {
        const bytes: [8]u8 = @bitCast(@as(u64, kvd));
        for (bytes) |b| {
            hash ^= @as(u64, b);
            hash *%= fnv1a_prime;
        }
    }
    return hash;
}

/// Write `h` at the front of `dst`. Returns false when `dst` is too short,
/// leaving it untouched.
pub fn write(dst: []u8, h: Header) bool {
    if (dst.len < Header.size) return false;
    @memcpy(dst[0..Header.size], std.mem.asBytes(&h));
    return true;
}

/// Read a header off the front of `src`, or null when `src` is too short,
/// does not start with the magic, or carries an unknown version. A caller
/// that gets null must refuse the blob: there is no safe interpretation of
/// bytes whose layout is unknown.
pub fn read(src: []const u8) ?Header {
    if (src.len < Header.size) return null;
    var h: Header = undefined;
    @memcpy(std.mem.asBytes(&h), src[0..Header.size]);
    if (!std.mem.eql(u8, &h.magic, &magic)) return null;
    if (h.version != version) return null;
    return h;
}

test "round trip preserves every field" {
    const h = Header{
        .n_layers = 30,
        .n_tokens = 512,
        .head_dim = 256,
        .shape_hash = shapeHash(&.{ 256, 256, 512 }),
        .rope_theta_a = 10_000.0,
        .rope_theta_b = 1_000_000.0,
    };
    var buf: [Header.size]u8 = undefined;
    try std.testing.expect(write(&buf, h));
    const back = read(&buf) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(h.n_layers, back.n_layers);
    try std.testing.expectEqual(h.n_tokens, back.n_tokens);
    try std.testing.expectEqual(h.head_dim, back.head_dim);
    try std.testing.expectEqual(h.shape_hash, back.shape_hash);
    try std.testing.expectEqual(h.rope_theta_a, back.rope_theta_a);
    try std.testing.expectEqual(h.rope_theta_b, back.rope_theta_b);
    try std.testing.expect(h.compatible(back));
}

test "read rejects short, unmagicked and future blobs" {
    const h = Header{
        .n_layers = 1,
        .n_tokens = 1,
        .head_dim = 8,
        .shape_hash = 0,
        .rope_theta_a = 1,
        .rope_theta_b = 1,
    };
    var buf: [Header.size]u8 = undefined;
    try std.testing.expect(write(&buf, h));

    try std.testing.expect(read(buf[0 .. Header.size - 1]) == null);
    try std.testing.expect(read(&.{}) == null);

    // A raw pre-header dump, which is what the old format looked like.
    var raw: [Header.size]u8 = @splat(0);
    try std.testing.expect(read(&raw) == null);

    var wrong_magic = buf;
    wrong_magic[0] = 'X';
    try std.testing.expect(read(&wrong_magic) == null);

    var future = buf;
    const bumped: [4]u8 = @bitCast(version + 1);
    @memcpy(future[8..12], &bumped);
    try std.testing.expect(read(&future) == null);
}

test "compatible rejects a different cache geometry at the same byte count" {
    // Two layers of 512 and four of 256 are the same bytes per token and a
    // different cache. Without the shape hash this restores and decodes
    // nonsense.
    const wide = Header{
        .n_layers = 2,
        .n_tokens = 8,
        .head_dim = 512,
        .shape_hash = shapeHash(&.{ 512, 512 }),
        .rope_theta_a = 10_000.0,
        .rope_theta_b = 10_000.0,
    };
    const narrow = Header{
        .n_layers = 4,
        .n_tokens = 8,
        .head_dim = 256,
        .shape_hash = shapeHash(&.{ 256, 256, 256, 256 }),
        .rope_theta_a = 10_000.0,
        .rope_theta_b = 10_000.0,
    };
    try std.testing.expect(!wide.compatible(narrow));

    // Same geometry, different RoPE base: the keys carry the wrong rotation.
    var retuned = wide;
    retuned.rope_theta_b = 1_000_000.0;
    try std.testing.expect(!wide.compatible(retuned));

    // Token count is the payload length, not a compatibility field.
    var longer = wide;
    longer.n_tokens = 4096;
    try std.testing.expect(wide.compatible(longer));
}

test "shapeHash separates orderings and lengths" {
    try std.testing.expect(shapeHash(&.{ 1, 2 }) != shapeHash(&.{ 2, 1 }));
    try std.testing.expect(shapeHash(&.{ 1, 2 }) != shapeHash(&.{ 1, 2, 0 }));
    try std.testing.expectEqual(shapeHash(&.{ 4, 4 }), shapeHash(&.{ 4, 4 }));
}
