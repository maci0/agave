//! Test-only helper: point fd 1 at /dev/null around code that prints to stdout.
//!
//! The main suite runs under Zig's server-mode test runner (see `build.zig`),
//! where fd 1 carries the length-prefixed protocol between the test binary and
//! the build runner. A test that writes plain text there desynchronizes the
//! stream, the build runner stops recognizing headers, and `zig build test`
//! blocks forever instead of failing. Wrap such calls in a `Silencer`.

const std = @import("std");
const posix = std.posix;

/// fd 1 restored on `release()`. Not reentrant and not thread-safe: it swaps a
/// process-wide file descriptor, so hold at most one at a time and keep the
/// window free of other threads writing to stdout.
pub const Silencer = struct {
    saved: c_int,

    pub fn init() !Silencer {
        const devnull = try posix.openatZ(posix.AT.FDCWD, "/dev/null", .{ .ACCMODE = .WRONLY }, 0);
        defer _ = std.c.close(devnull);

        const saved = std.c.dup(posix.STDOUT_FILENO);
        if (saved < 0) return error.DupFailed;
        errdefer _ = std.c.close(saved);

        if (std.c.dup2(devnull, posix.STDOUT_FILENO) < 0) return error.DupFailed;
        return .{ .saved = saved };
    }

    /// Restore the original fd 1. Failure here would leave the protocol stream
    /// pointed at /dev/null, which reads as a silent hang, so it panics.
    pub fn release(self: Silencer) void {
        if (std.c.dup2(self.saved, posix.STDOUT_FILENO) < 0) @panic("test_stdout: could not restore stdout");
        _ = std.c.close(self.saved);
    }
};

test "silencer swallows stdout writes and restores fd 1" {
    const before = std.c.dup(posix.STDOUT_FILENO);
    try std.testing.expect(before >= 0);
    defer _ = std.c.close(before);

    const s = try Silencer.init();
    _ = posix.system.write(posix.STDOUT_FILENO, "this must not reach the test runner\n", 36);
    s.release();

    // fd 1 is usable again: a zero-length write to a live descriptor returns 0.
    try std.testing.expectEqual(@as(isize, 0), @as(isize, @intCast(posix.system.write(posix.STDOUT_FILENO, "", 0))));
}
