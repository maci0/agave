//! Fixed-capacity byte buffer with a Writer interface.
//!
//! Zig 0.16 removed `std.io.FixedBufStream`. Server response formatting and
//! Prometheus rendering need an allocation-free writer into a stack buffer.
//! Lives here (not in metrics.zig) so HTTP handlers can depend on a small
//! IO helper without importing the metrics collector.

const std = @import("std");

/// Minimal fixed-buffer writer backed by a caller-owned slice.
pub const FixedBufStream = struct {
    buf: []u8,
    pos: usize = 0,

    pub fn init(buf: []u8) FixedBufStream {
        return .{ .buf = buf };
    }

    pub fn writer(self: *FixedBufStream) Writer {
        return .{ .fbs = self };
    }

    pub fn getWritten(self: *const FixedBufStream) []const u8 {
        return self.buf[0..self.pos];
    }

    pub const Writer = struct {
        fbs: *FixedBufStream,

        pub fn writeAll(self: Writer, data: []const u8) !void {
            if (self.fbs.pos + data.len > self.fbs.buf.len) return error.NoSpaceLeft;
            @memcpy(self.fbs.buf[self.fbs.pos..][0..data.len], data);
            self.fbs.pos += data.len;
        }

        pub fn print(self: Writer, comptime fmt: []const u8, args: anytype) !void {
            const written = std.fmt.bufPrint(self.fbs.buf[self.fbs.pos..], fmt, args) catch return error.NoSpaceLeft;
            self.fbs.pos += written.len;
        }

        pub fn writeByte(self: Writer, byte: u8) !void {
            if (self.fbs.pos >= self.fbs.buf.len) return error.NoSpaceLeft;
            self.fbs.buf[self.fbs.pos] = byte;
            self.fbs.pos += 1;
        }
    };
};

test "FixedBufStream: init and getWritten on empty" {
    var buf: [64]u8 = undefined;
    var fbs = FixedBufStream.init(&buf);
    try std.testing.expectEqual(@as(usize, 0), fbs.getWritten().len);
}

test "FixedBufStream: writeAll and getWritten" {
    var buf: [64]u8 = undefined;
    var fbs = FixedBufStream.init(&buf);
    const w = fbs.writer();
    try w.writeAll("hello");
    try std.testing.expectEqualStrings("hello", fbs.getWritten());
    try w.writeAll(" world");
    try std.testing.expectEqualStrings("hello world", fbs.getWritten());
}

test "FixedBufStream: writeByte" {
    var buf: [64]u8 = undefined;
    var fbs = FixedBufStream.init(&buf);
    const w = fbs.writer();
    try w.writeByte('A');
    try w.writeByte('B');
    try std.testing.expectEqualStrings("AB", fbs.getWritten());
}

test "FixedBufStream: print formatted" {
    var buf: [64]u8 = undefined;
    var fbs = FixedBufStream.init(&buf);
    const w = fbs.writer();
    try w.print("count={d}", .{@as(u32, 42)});
    try std.testing.expectEqualStrings("count=42", fbs.getWritten());
}

test "FixedBufStream: overflow returns error" {
    var buf: [4]u8 = undefined;
    var fbs = FixedBufStream.init(&buf);
    const w = fbs.writer();
    try w.writeAll("abcd");
    // Buffer is now full; next write should fail
    try std.testing.expectError(error.NoSpaceLeft, w.writeAll("e"));
    try std.testing.expectError(error.NoSpaceLeft, w.writeByte('x'));
}

test "FixedBufStream: pos tracks correctly" {
    var buf: [64]u8 = undefined;
    var fbs = FixedBufStream.init(&buf);
    try std.testing.expectEqual(@as(usize, 0), fbs.pos);
    const w = fbs.writer();
    try w.writeAll("test");
    try std.testing.expectEqual(@as(usize, 4), fbs.pos);
    try w.writeByte('!');
    try std.testing.expectEqual(@as(usize, 5), fbs.pos);
}
