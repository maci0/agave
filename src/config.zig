//! Environment variable access, shared by every module that reads config.
//!
//! One place decides what an unset variable means, so a debug flag, a secret,
//! and a cache path cannot disagree about the same string. Readers go through
//! here: `src/pull.zig` (HF_TOKEN, HF_HOME, cache roots), `src/main.zig`
//! (AGAVE_* server settings), `src/models/vision.zig` (AGAVE_VISION_DEBUG), and
//! the Vulkan / conversation-store cache paths. Two modules read the
//! environment directly and are the exceptions: `src/display.zig` (TERM,
//! NO_COLOR) and `src/parallel/transport.zig` (per-process IPC discovery).

const std = @import("std");

/// Treat missing, empty, and whitespace-only values as unset.
///
/// Docker Compose forwards optional vars as empty strings (`${VAR:-}`), and
/// sourcing `.env.example` leaves `AGAVE_API_KEY=` until the operator fills it.
/// Callers must not treat those as configured secrets or paths.
pub fn nonemptyEnv(val: ?[]const u8) ?[]const u8 {
    const v = val orelse return null;
    const trimmed = std.mem.trim(u8, v, " \t\r\n");
    return if (trimmed.len == 0) null else trimmed;
}

/// True when a debug/feature env var is exactly `1` after trim.
/// Docs promise `=1`; `0`, empty, and other values stay off.
pub fn envFlagIsOne(val: ?[]const u8) bool {
    const v = nonemptyEnv(val) orelse return false;
    return std.mem.eql(u8, v, "1");
}

/// The value of a debug/feature env var that `envFlagIsOne` will not accept, or
/// null when the var is unset or already `1`.
///
/// `0` is accepted as a deliberate "off". Anything else (`true`, `yes`, `on`)
/// leaves the flag off by design, so callers pass the result to a warning to
/// keep a `.env` typo from silently disabling a debug switch.
pub fn unsupportedEnvFlagValue(val: ?[]const u8) ?[]const u8 {
    const v = nonemptyEnv(val) orelse return null;
    if (std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "0")) return null;
    return v;
}

/// Get an environment variable (Zig 0.16 idiom via C getenv).
/// Empty and whitespace-only values are unset (see `nonemptyEnv`).
pub fn getenv(name: []const u8) ?[]const u8 {
    var buf: [256]u8 = undefined;
    if (name.len >= buf.len) return null;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    const result = std.c.getenv(@ptrCast(buf[0..name.len :0])) orelse return null;
    return nonemptyEnv(std.mem.span(result));
}

test "nonemptyEnv treats empty and whitespace as unset" {
    try std.testing.expect(nonemptyEnv(null) == null);
    try std.testing.expect(nonemptyEnv("") == null);
    try std.testing.expect(nonemptyEnv("   ") == null);
    try std.testing.expect(nonemptyEnv("\t\n") == null);
    try std.testing.expectEqualStrings("abc", nonemptyEnv("abc").?);
    try std.testing.expectEqualStrings("abc", nonemptyEnv("  abc  ").?);
}

test "envFlagIsOne requires trimmed 1" {
    try std.testing.expect(!envFlagIsOne(null));
    try std.testing.expect(!envFlagIsOne(""));
    try std.testing.expect(!envFlagIsOne("0"));
    try std.testing.expect(!envFlagIsOne("true"));
    try std.testing.expect(!envFlagIsOne("yes"));
    try std.testing.expect(envFlagIsOne("1"));
    try std.testing.expect(envFlagIsOne(" 1 "));
}

test "getenv returns null for a variable that is not set" {
    try std.testing.expect(getenv("AGAVE_TEST_UNSET_ENV_VAR_12345") == null);
}

test "unsupportedEnvFlagValue reports only values that leave the flag off" {
    try std.testing.expect(unsupportedEnvFlagValue(null) == null);
    try std.testing.expect(unsupportedEnvFlagValue("") == null);
    try std.testing.expect(unsupportedEnvFlagValue("  ") == null);
    try std.testing.expect(unsupportedEnvFlagValue("1") == null);
    try std.testing.expect(unsupportedEnvFlagValue(" 1 ") == null);
    try std.testing.expect(unsupportedEnvFlagValue("0") == null);
    try std.testing.expectEqualStrings("true", unsupportedEnvFlagValue("true").?);
    try std.testing.expectEqualStrings("yes", unsupportedEnvFlagValue(" yes ").?);
    try std.testing.expectEqualStrings("on", unsupportedEnvFlagValue("on").?);
    // Every value it reports is one envFlagIsOne rejects.
    for ([_][]const u8{ "true", "yes", "on", "2", "01", "TRUE" }) |bad| {
        try std.testing.expect(!envFlagIsOne(bad));
    }
}
