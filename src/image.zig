//! Minimal PNG image decoder and image utilities for multimodal inference.
//!
//! Supports PNG files with RGB (color_type 2) and RGBA (color_type 6) at
//! 8-bit depth. Ancillary chunks are ignored. JPEG files are detected and
//! rejected with a helpful error message.
//!
//! Decompression uses `std.compress.flate.Decompress` with the zlib container.
//! Scanline filters (None, Sub, Up, Average, Paeth) are reconstructed in-place.

const std = @import("std");
const Allocator = std.mem.Allocator;

// ── Constants ───────────────────────────────────────────────────

/// PNG file signature (8 bytes).
const png_signature = [8]u8{ 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };

/// Maximum image file size to load (64 MB).
const max_file_size: usize = 64 * 1024 * 1024;

/// Maximum decompressed scanline data (100 MP * 4 channels + filter bytes).
const max_decompressed_size: usize = 100 * 1024 * 1024;

/// Number of channels in output RGB pixels.
const rgb_channels: usize = 3;

/// Number of channels in RGBA pixels.
const rgba_channels: usize = 4;

/// Supported PNG bit depth.
const supported_bit_depth: u8 = 8;

/// PNG color type: RGB (no alpha).
const color_type_rgb: u8 = 2;

/// PNG color type: RGBA (with alpha).
const color_type_rgba: u8 = 6;

/// IHDR chunk type identifier.
const chunk_ihdr: [4]u8 = "IHDR".*;

/// IDAT chunk type identifier.
const chunk_idat: [4]u8 = "IDAT".*;

/// IEND chunk type identifier.
const chunk_iend: [4]u8 = "IEND".*;

/// JPEG magic bytes (SOI marker).
const jpeg_magic = [2]u8{ 0xFF, 0xD8 };

/// PPM P6 magic bytes.
const ppm_magic = [2]u8{ 'P', '6' };

// ── Public types ────────────────────────────────────────────────

/// Decoded PNG image with RGB pixel data. Caller owns and must call `deinit()`.
pub const PngImage = struct {
    /// Raw RGB pixel data in row-major order: [width * height * 3] u8.
    pixels: []u8,
    /// Image width in pixels.
    width: u32,
    /// Image height in pixels.
    height: u32,
    /// Allocator used for the pixel buffer — needed by `deinit()`.
    allocator: Allocator,

    pub fn deinit(self: *PngImage) void {
        self.allocator.free(self.pixels);
    }
};

/// Errors returned by PNG decoding and image operations.
pub const ImageError = error{
    InvalidPngSignature,
    UnsupportedColorType,
    UnsupportedBitDepth,
    UnsupportedInterlace,
    MissingIhdr,
    InvalidIhdr,
    NoImageData,
    InvalidFilter,
    InvalidImageFormat,
    InvalidImageSize,
    DecompressionFailed,
    JpegNotSupported,
    OutOfMemory,
};

// ── PNG decoding ────────────────────────────────────────────────

/// Decode a PNG file from raw bytes to RGB pixels.
///
/// Supports color types RGB (2) and RGBA (6) at 8-bit depth.
/// RGBA images are converted to RGB by discarding the alpha channel.
/// Caller owns the returned PngImage and must call deinit() to free.
///
/// Parameters:
///   - allocator: Memory allocator for output pixel buffer and temporaries.
///   - data: Raw PNG file bytes (must start with PNG signature).
///
/// Returns: PngImage with RGB pixel data, or ImageError on failure.
pub fn decodePng(allocator: Allocator, data: []const u8) ImageError!PngImage {
    // Validate PNG signature
    if (data.len < png_signature.len) return error.InvalidPngSignature;
    if (!std.mem.eql(u8, data[0..png_signature.len], &png_signature)) return error.InvalidPngSignature;

    // Parse IHDR (must be the first chunk after signature)
    var pos: usize = png_signature.len;
    const ihdr = readChunk(data, &pos) orelse return error.MissingIhdr;
    if (!std.mem.eql(u8, &ihdr.chunk_type, &chunk_ihdr)) return error.MissingIhdr;
    if (ihdr.chunk_data.len < 13) return error.InvalidIhdr;

    const width = readU32Be(ihdr.chunk_data[0..4]);
    const height = readU32Be(ihdr.chunk_data[4..8]);
    const bit_depth = ihdr.chunk_data[8];
    const color_type = ihdr.chunk_data[9];
    // ihdr.chunk_data[10] = compression method (must be 0)
    // ihdr.chunk_data[11] = filter method (must be 0)
    const interlace = ihdr.chunk_data[12];

    if (bit_depth != supported_bit_depth) return error.UnsupportedBitDepth;
    if (color_type != color_type_rgb and color_type != color_type_rgba) return error.UnsupportedColorType;
    if (interlace != 0) return error.UnsupportedInterlace;
    if (width == 0 or height == 0) return error.InvalidIhdr;

    const bpp: usize = if (color_type == color_type_rgba) rgba_channels else rgb_channels;

    // Collect all IDAT chunk data positions
    // We need to track them to feed into the decompressor
    var idat_chunks: std.ArrayList([]const u8) = .empty;
    defer idat_chunks.deinit(allocator);

    while (pos < data.len) {
        const chunk = readChunk(data, &pos) orelse break;
        if (std.mem.eql(u8, &chunk.chunk_type, &chunk_idat)) {
            idat_chunks.append(allocator, chunk.chunk_data) catch return error.OutOfMemory;
        } else if (std.mem.eql(u8, &chunk.chunk_type, &chunk_iend)) {
            break;
        }
        // Skip ancillary chunks (tEXt, pHYs, etc.)
    }

    if (idat_chunks.items.len == 0) return error.NoImageData;

    // Concatenate all IDAT data into a single buffer for decompression
    var total_idat_size: usize = 0;
    for (idat_chunks.items) |chunk_data| {
        total_idat_size += chunk_data.len;
    }

    const idat_buf = allocator.alloc(u8, total_idat_size) catch return error.OutOfMemory;
    defer allocator.free(idat_buf);
    {
        var offset: usize = 0;
        for (idat_chunks.items) |chunk_data| {
            @memcpy(idat_buf[offset..][0..chunk_data.len], chunk_data);
            offset += chunk_data.len;
        }
    }

    // Decompress using zlib container (PNG uses zlib-wrapped deflate)
    const h: usize = height;
    const w: usize = width;
    const scanline_len = std.math.mul(usize, w, bpp) catch return error.InvalidImageSize;
    const raw_size = std.math.mul(usize, h, std.math.add(usize, 1, scanline_len) catch
        return error.InvalidImageSize) catch return error.InvalidImageSize;

    if (raw_size > max_decompressed_size) return error.InvalidImageSize;

    const decompressed = decompressZlib(allocator, idat_buf, raw_size) catch return error.DecompressionFailed;
    defer allocator.free(decompressed);

    if (decompressed.len < raw_size) return error.InvalidImageSize;

    // Unfilter scanlines in-place
    unfilterScanlines(decompressed, w, h, bpp) catch return error.InvalidFilter;

    // Extract RGB pixels (convert RGBA -> RGB if needed)
    const pixel_count = std.math.mul(usize, w, h) catch return error.InvalidImageSize;
    const rgb_size = std.math.mul(usize, pixel_count, rgb_channels) catch return error.InvalidImageSize;
    const pixels = allocator.alloc(u8, rgb_size) catch return error.OutOfMemory;
    errdefer allocator.free(pixels);

    if (color_type == color_type_rgba) {
        // RGBA -> RGB: copy R, G, B; skip A
        for (0..h) |y| {
            const src_row = decompressed[y * (1 + scanline_len) + 1 ..][0..scanline_len];
            const dst_row = pixels[y * w * rgb_channels ..][0 .. w * rgb_channels];
            for (0..w) |x| {
                dst_row[x * rgb_channels + 0] = src_row[x * rgba_channels + 0];
                dst_row[x * rgb_channels + 1] = src_row[x * rgba_channels + 1];
                dst_row[x * rgb_channels + 2] = src_row[x * rgba_channels + 2];
            }
        }
    } else {
        // RGB: copy directly (skip filter byte per row)
        for (0..h) |y| {
            const src_row = decompressed[y * (1 + scanline_len) + 1 ..][0..scanline_len];
            const dst_row = pixels[y * w * rgb_channels ..][0 .. w * rgb_channels];
            @memcpy(dst_row, src_row);
        }
    }

    return .{
        .pixels = pixels,
        .width = width,
        .height = height,
        .allocator = allocator,
    };
}

/// Detect image format from magic bytes and return a descriptive tag.
///
/// Returns .png, .ppm, .jpeg, or .unknown based on the file header.
pub const ImageFormat = enum { png, ppm, jpeg, unknown };

/// Detect image format from the first few bytes of file data.
pub fn detectFormat(data: []const u8) ImageFormat {
    if (data.len >= png_signature.len and std.mem.eql(u8, data[0..png_signature.len], &png_signature)) return .png;
    if (data.len >= ppm_magic.len and std.mem.eql(u8, data[0..ppm_magic.len], &ppm_magic)) return .ppm;
    if (data.len >= jpeg_magic.len and std.mem.eql(u8, data[0..jpeg_magic.len], &jpeg_magic)) return .jpeg;
    return .unknown;
}

/// Image dimensions (width × height).
pub const ImageDims = struct { width: u32, height: u32 };

/// Read image dimensions from a file without full decoding.
/// Supports PNG (reads IHDR) and PPM (reads header).
pub fn getImageDimensions(allocator: Allocator, path: []const u8) !ImageDims {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    var header: [32]u8 = undefined;
    const n = try file.readAll(&header);
    if (n < 24) return error.InvalidImageSize;

    if (std.mem.eql(u8, header[0..png_signature.len], &png_signature)) {
        // PNG: IHDR chunk starts at offset 8 (4 len + 4 type), width/height at offset 16/20 (big-endian u32)
        const w = std.mem.readInt(u32, header[16..20], .big);
        const h = std.mem.readInt(u32, header[20..24], .big);
        return .{ .width = w, .height = h };
    }
    if (std.mem.eql(u8, header[0..ppm_magic.len], &ppm_magic)) {
        // PPM P6: read header to get width and height
        const max_hdr: usize = 256;
        const hdr_buf = try allocator.alloc(u8, max_hdr);
        defer allocator.free(hdr_buf);
        try file.seekTo(0);
        const hdr_n = try file.readAll(hdr_buf);
        var pos: usize = 3; // skip "P6\n"
        while (pos < hdr_n and hdr_buf[pos] == '#') { // skip comments
            while (pos < hdr_n and hdr_buf[pos] != '\n') pos += 1;
            pos += 1;
        }
        // Parse width and height
        var w: u32 = 0;
        while (pos < hdr_n and hdr_buf[pos] >= '0' and hdr_buf[pos] <= '9') : (pos += 1) {
            w = w * 10 + @as(u32, hdr_buf[pos] - '0');
        }
        pos += 1; // skip space
        var h: u32 = 0;
        while (pos < hdr_n and hdr_buf[pos] >= '0' and hdr_buf[pos] <= '9') : (pos += 1) {
            h = h * 10 + @as(u32, hdr_buf[pos] - '0');
        }
        if (w > 0 and h > 0) return .{ .width = w, .height = h };
    }
    return error.InvalidImageSize;
}

/// Resize an image using bilinear interpolation.
///
/// Parameters:
///   - allocator: Memory allocator for the output buffer.
///   - src: Source RGB pixel data [src_w * src_h * 3].
///   - src_w: Source image width.
///   - src_h: Source image height.
///   - dst_w: Target width.
///   - dst_h: Target height.
///
/// Returns: [dst_w * dst_h * 3] u8 with resized RGB pixels. Caller owns.
pub fn resize(allocator: Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) ![]u8 {
    const dw: usize = dst_w;
    const dh: usize = dst_h;
    const sw: usize = src_w;
    const sh: usize = src_h;
    const out_size = std.math.mul(usize, std.math.mul(usize, dw, dh) catch return error.InvalidImageSize, rgb_channels) catch return error.InvalidImageSize;
    const out = try allocator.alloc(u8, out_size);
    errdefer allocator.free(out);

    // Bilinear interpolation
    for (0..dh) |dy| {
        // Map destination y to source y (center-pixel mapping)
        const sy_f: f64 = (@as(f64, @floatFromInt(dy)) + 0.5) * @as(f64, @floatFromInt(sh)) / @as(f64, @floatFromInt(dh)) - 0.5;
        const sy0: usize = @intFromFloat(@max(0.0, @floor(sy_f)));
        const sy1: usize = @min(sy0 + 1, sh - 1);
        const fy: f64 = sy_f - @as(f64, @floatFromInt(sy0));

        for (0..dw) |dx| {
            const sx_f: f64 = (@as(f64, @floatFromInt(dx)) + 0.5) * @as(f64, @floatFromInt(sw)) / @as(f64, @floatFromInt(dw)) - 0.5;
            const sx0: usize = @intFromFloat(@max(0.0, @floor(sx_f)));
            const sx1: usize = @min(sx0 + 1, sw - 1);
            const fx: f64 = sx_f - @as(f64, @floatFromInt(sx0));

            const dst_idx = (dy * dw + dx) * rgb_channels;

            inline for (0..rgb_channels) |c| {
                const tl: f64 = @floatFromInt(src[(sy0 * sw + sx0) * rgb_channels + c]);
                const tr: f64 = @floatFromInt(src[(sy0 * sw + sx1) * rgb_channels + c]);
                const bl: f64 = @floatFromInt(src[(sy1 * sw + sx0) * rgb_channels + c]);
                const br: f64 = @floatFromInt(src[(sy1 * sw + sx1) * rgb_channels + c]);

                const top = tl * (1.0 - fx) + tr * fx;
                const bot = bl * (1.0 - fx) + br * fx;
                const val = top * (1.0 - fy) + bot * fy;

                out[dst_idx + c] = @intFromFloat(std.math.clamp(val + 0.5, 0.0, 255.0));
            }
        }
    }

    return out;
}

// ── Internal helpers ────────────────────────────────────────────

const Chunk = struct {
    chunk_type: [4]u8,
    chunk_data: []const u8,
};

/// Read a PNG chunk at the given position. Advances pos past the chunk.
/// Returns null if insufficient data remains.
fn readChunk(data: []const u8, pos: *usize) ?Chunk {
    if (pos.* + 8 > data.len) return null;
    const length = readU32Be(data[pos.*..][0..4]);
    const chunk_type = data[pos.* + 4 ..][0..4].*;
    pos.* += 8;

    const end = std.math.add(usize, pos.*, length) catch return null;
    if (end > data.len) return null;
    const chunk_data = data[pos.*..][0..length];
    pos.* += length;

    // Skip CRC (4 bytes)
    if (pos.* + 4 > data.len) return null;
    pos.* += 4;

    return .{
        .chunk_type = chunk_type,
        .chunk_data = chunk_data,
    };
}

/// Read a big-endian u32 from a 4-byte slice.
inline fn readU32Be(b: *const [4]u8) u32 {
    return std.mem.readInt(u32, b, .big);
}

/// Decompress zlib-wrapped data using std.compress.flate.Decompress.
///
/// Allocates and returns the decompressed output buffer.
fn decompressZlib(allocator: Allocator, compressed: []const u8, expected_size: usize) ![]u8 {
    var reader: std.Io.Reader = .fixed(compressed);

    // Use a decompression window buffer
    var window_buf: [std.compress.flate.max_window_len]u8 = undefined;
    var decompress = std.compress.flate.Decompress.init(&reader, .zlib, &window_buf);

    // Allocate output with some extra room
    var aw: std.Io.Writer.Allocating = try .initCapacity(allocator, expected_size);
    errdefer aw.deinit();

    _ = decompress.reader.streamRemaining(&aw.writer) catch return error.DecompressionFailed;

    return aw.toOwnedSlice() catch error.OutOfMemory;
}

/// Reconstruct filtered scanlines in-place.
///
/// Each scanline has a 1-byte filter type prefix followed by the raw
/// filtered data. After unfiltering, the data bytes are the actual
/// pixel values (the filter bytes remain but are skipped during extraction).
fn unfilterScanlines(data: []u8, width: usize, height: usize, bpp: usize) !void {
    const stride = 1 + width * bpp; // filter byte + pixel data

    for (0..height) |y| {
        const row_start = y * stride;
        const filter_type = data[row_start];
        const row = data[row_start + 1 ..][0 .. width * bpp];

        switch (filter_type) {
            0 => {}, // None — no reconstruction needed
            1 => {
                // Sub: recon[i] = raw[i] + recon[i - bpp]
                for (bpp..row.len) |i| {
                    row[i] = row[i] +% row[i - bpp];
                }
            },
            2 => {
                // Up: recon[i] = raw[i] + prior_row[i]
                if (y > 0) {
                    const prior = data[(y - 1) * stride + 1 ..][0 .. width * bpp];
                    for (0..row.len) |i| {
                        row[i] = row[i] +% prior[i];
                    }
                }
                // First row: prior is all zeros, so no-op
            },
            3 => {
                // Average: recon[i] = raw[i] + floor((recon[i-bpp] + prior_row[i]) / 2)
                const prior = if (y > 0) data[(y - 1) * stride + 1 ..][0 .. width * bpp] else null;
                for (0..row.len) |i| {
                    const a: u16 = if (i >= bpp) row[i - bpp] else 0;
                    const b: u16 = if (prior) |p| p[i] else 0;
                    row[i] = row[i] +% @as(u8, @intCast((a + b) / 2));
                }
            },
            4 => {
                // Paeth: recon[i] = raw[i] + PaethPredictor(a, b, c)
                const prior = if (y > 0) data[(y - 1) * stride + 1 ..][0 .. width * bpp] else null;
                for (0..row.len) |i| {
                    const a: i16 = if (i >= bpp) @intCast(row[i - bpp]) else 0;
                    const b: i16 = if (prior) |p| @intCast(p[i]) else 0;
                    const c: i16 = if (i >= bpp) if (prior) |p| @as(i16, @intCast(p[i - bpp])) else 0 else 0;
                    row[i] = row[i] +% paethPredictor(a, b, c);
                }
            },
            else => return error.InvalidFilter,
        }
    }
}

/// Paeth predictor function per PNG specification.
///
/// Parameters a, b, c are the left, above, and upper-left neighbor pixels.
/// Returns the neighbor value closest to the linear predictor p = a + b - c.
inline fn paethPredictor(a: i16, b: i16, c: i16) u8 {
    const p = a + b - c;
    const pa = @as(u16, @intCast(if (p >= a) p - a else a - p));
    const pb = @as(u16, @intCast(if (p >= b) p - b else b - p));
    const pc = @as(u16, @intCast(if (p >= c) p - c else c - p));
    if (pa <= pb and pa <= pc) return @intCast(@as(u16, @intCast(a)));
    if (pb <= pc) return @intCast(@as(u16, @intCast(b)));
    return @intCast(@as(u16, @intCast(c)));
}

// ── Tests ───────────────────────────────────────────────────────

test "detectFormat identifies PNG" {
    const png_data = png_signature ++ [_]u8{ 0, 0, 0, 0 };
    try std.testing.expectEqual(ImageFormat.png, detectFormat(&png_data));
}

test "detectFormat identifies JPEG" {
    const jpeg_data = [_]u8{ 0xFF, 0xD8, 0xFF, 0xE0 };
    try std.testing.expectEqual(ImageFormat.jpeg, detectFormat(&jpeg_data));
}

test "detectFormat identifies PPM" {
    const ppm_data = [_]u8{ 'P', '6', '\n' };
    try std.testing.expectEqual(ImageFormat.ppm, detectFormat(&ppm_data));
}

test "detectFormat returns unknown for random data" {
    const data = [_]u8{ 0x00, 0x01, 0x02, 0x03 };
    try std.testing.expectEqual(ImageFormat.unknown, detectFormat(&data));
}

test "decodePng rejects invalid signature" {
    const data = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7 };
    try std.testing.expectError(error.InvalidPngSignature, decodePng(std.testing.allocator, &data));
}

test "decodePng rejects too-short data" {
    const data = [_]u8{ 0x89, 'P', 'N' };
    try std.testing.expectError(error.InvalidPngSignature, decodePng(std.testing.allocator, &data));
}

test "paethPredictor basic cases" {
    // All zeros -> 0
    try std.testing.expectEqual(@as(u8, 0), paethPredictor(0, 0, 0));
    // a=10, b=0, c=0 -> closest to a
    try std.testing.expectEqual(@as(u8, 10), paethPredictor(10, 0, 0));
    // a=0, b=10, c=0 -> closest to b
    try std.testing.expectEqual(@as(u8, 10), paethPredictor(0, 10, 0));
    // a=10, b=10, c=10 -> p=10, all equal distance, returns a
    try std.testing.expectEqual(@as(u8, 10), paethPredictor(10, 10, 10));
}

test "resize 2x2 to 1x1 averages pixels" {
    const allocator = std.testing.allocator;
    // 2x2 image: red, green, blue, white
    const src = [_]u8{
        255, 0,   0, // red
        0,   255, 0, // green
        0,   0,   255, // blue
        255, 255, 255, // white
    };
    const result = try resize(allocator, &src, 2, 2, 1, 1);
    defer allocator.free(result);
    // With bilinear interpolation at center, should average all 4 pixels
    try std.testing.expectEqual(@as(usize, 3), result.len);
    // Bilinear at (0.5, 0.5) with equal weights → each channel ≈ (255+0+0+255)/4 ≈ 127
    try std.testing.expectApproxEqAbs(@as(f32, 127.5), @as(f32, @floatFromInt(result[0])), 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 127.5), @as(f32, @floatFromInt(result[1])), 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 127.5), @as(f32, @floatFromInt(result[2])), 2.0);
}

test "resize identity (same dimensions)" {
    const allocator = std.testing.allocator;
    const src = [_]u8{ 100, 150, 200, 50, 75, 100, 200, 100, 50, 25, 30, 35 };
    const result = try resize(allocator, &src, 2, 2, 2, 2);
    defer allocator.free(result);
    // Same dimensions should produce very similar output (not necessarily identical due to bilinear)
    try std.testing.expectEqual(@as(usize, 12), result.len);
    // Pixel values should be close to originals (bilinear at same grid points)
    for (0..12) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(src[i])), @as(f32, @floatFromInt(result[i])), 10.0);
    }
}

test "unfilterScanlines None filter" {
    // 2x1 RGB image with None filter
    var data = [_]u8{
        0, // filter=None
        10, 20, 30, 40, 50, 60,
    };
    try unfilterScanlines(&data, 2, 1, 3);
    // None filter: data unchanged
    try std.testing.expectEqualSlices(u8, &[_]u8{ 10, 20, 30, 40, 50, 60 }, data[1..]);
}

test "unfilterScanlines Sub filter" {
    // 2x1 RGB image with Sub filter
    // Sub: recon[i] = raw[i] + recon[i - bpp]
    // For bpp=3: first 3 bytes unchanged, next 3 add previous pixel
    var data = [_]u8{
        1, // filter=Sub
        10, 20, 30, 5, 5, 5,
    };
    try unfilterScanlines(&data, 2, 1, 3);
    // First pixel: 10, 20, 30 (unchanged)
    // Second pixel: 5+10=15, 5+20=25, 5+30=35
    try std.testing.expectEqualSlices(u8, &[_]u8{ 10, 20, 30, 15, 25, 35 }, data[1..]);
}

test "unfilterScanlines Up filter" {
    // 2x1 image, 2 rows, Up filter
    var data = [_]u8{
        0, // filter=None (first row)
        10, 20, 30, // first row pixels
        2, // filter=Up (second row)
        5,  10, 15, // second row raw
    };
    try unfilterScanlines(&data, 1, 2, 3);
    // First row: unchanged (10, 20, 30)
    // Second row: 5+10=15, 10+20=30, 15+30=45
    try std.testing.expectEqualSlices(u8, &[_]u8{ 15, 30, 45 }, data[5..8]);
}

test "unfilterScanlines Average filter" {
    // 1x2 RGB image, row 0=None, row 1=Average
    var data = [_]u8{
        0, // filter=None (first row)
        10, 20, 30, // first row pixels
        3, // filter=Average (second row)
        4,  6,  8, // second row raw
    };
    try unfilterScanlines(&data, 1, 2, 3);
    // First row: unchanged
    // Second row: i < bpp so a=0, b=prior[i]
    //   4 + floor((0 + 10) / 2) = 4 + 5 = 9
    //   6 + floor((0 + 20) / 2) = 6 + 10 = 16
    //   8 + floor((0 + 30) / 2) = 8 + 15 = 23
    try std.testing.expectEqualSlices(u8, &[_]u8{ 9, 16, 23 }, data[5..8]);
}

test "unfilterScanlines Paeth filter" {
    // 2x2 RGB image, row 0=None, row 1=Paeth (exercises left+above+upper-left)
    var data = [_]u8{
        0, // filter=None (first row)
        10, 20, 30, 40, 50, 60, // first row: pixel0=[10,20,30], pixel1=[40,50,60]
        4, // filter=Paeth (second row)
        5,  3,  2,  1,  2,  3, // second row raw
    };
    try unfilterScanlines(&data, 2, 2, 3);
    // First row: unchanged
    // Second row, first pixel (i < bpp): a=0, b=prior[i], c=0
    //   paeth(0,10,0): p=10, pa=10, pb=0, pc=10 → b=10 → 5+10=15
    //   paeth(0,20,0): p=20, pa=20, pb=0, pc=20 → b=20 → 3+20=23
    //   paeth(0,30,0): p=30, pa=30, pb=0, pc=30 → b=30 → 2+30=32
    // Second row, second pixel (i >= bpp): a=recon[i-3], b=prior[i], c=prior[i-3]
    //   paeth(15,40,10): p=45, pa=30, pb=5, pc=35 → b=40 → 1+40=41
    //   paeth(23,50,20): p=53, pa=30, pb=3, pc=33 → b=50 → 2+50=52
    //   paeth(32,60,30): p=62, pa=30, pb=2, pc=32 → b=60 → 3+60=63
    try std.testing.expectEqualSlices(u8, &[_]u8{ 15, 23, 32, 41, 52, 63 }, data[8..14]);
}
