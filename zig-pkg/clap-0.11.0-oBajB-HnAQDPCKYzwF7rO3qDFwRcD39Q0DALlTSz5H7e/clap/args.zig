/// An example of what methods should be implemented on an arg iterator.
pub const ExampleArgIterator = struct {
    pub fn next(iter: *ExampleArgIterator) ?[]const u8 {
        _ = iter;
        return "2";
    }
};

/// An argument iterator which iterates over a slice of arguments.
/// This implementation does not allocate.
pub const SliceIterator = struct {
    args: []const []const u8,
    index: usize = 0,

    pub fn next(iter: *SliceIterator) ?[]const u8 {
        if (iter.args.len <= iter.index)
            return null;

        defer iter.index += 1;
        return iter.args[iter.index];
    }
};

test "SliceIterator" {
    const args = [_][]const u8{ "A", "BB", "CCC" };
    var iter = SliceIterator{ .args = &args };

    for (args) |a|
        try std.testing.expectEqualStrings(a, iter.next().?);

    try std.testing.expectEqual(@as(?[]const u8, null), iter.next());
}

/// An argument iterator over null-terminated C-style arg pointers (Zig 0.16 compat).
pub const NullTermIterator = struct {
    args: []const [*:0]const u8,
    index: usize = 0,

    pub fn next(iter: *NullTermIterator) ?[]const u8 {
        if (iter.args.len <= iter.index)
            return null;
        defer iter.index += 1;
        return std.mem.sliceTo(iter.args[iter.index], 0);
    }
};

const std = @import("std");
