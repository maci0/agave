//! Injectable wall clock for deterministic simulation and tests.
//!
//! Production leaves the override unset so milliNow/nanoNow read the OS clock
//! (REALTIME). Interval timers that must resist NTP skew (profiling, download
//! progress) use CLOCK_MONOTONIC directly and do not go through this module.
//!
//! Tests and a future sim harness call setOverrideMs / advanceMs to drive
//! timeouts, rate-limit refill, and scheduling priority from a single seed.
//!
//! The override is process-global (not thread-local). Tests that set it must
//! `defer setOverrideMs(null)` and must not run concurrently with other tests
//! that also override the clock. Storage is atomic so concurrent milliNow
//! readers never tear a partially-updated override value.

const std = @import("std");
const builtin = @import("builtin");
const is_freestanding = builtin.os.tag == .freestanding;

/// Sentinel: no override installed (milliNow falls through to the wall clock).
const no_override: i64 = std.math.minInt(i64);

/// When not `no_override`, milliNow/nanoNow return this logical millisecond time.
var override_ms: std.atomic.Value(i64) = .init(no_override);

/// Install (or clear) a simulated millisecond clock. Pass null to restore wall time.
pub fn setOverrideMs(ms: ?i64) void {
    override_ms.store(ms orelse no_override, .release);
}

/// Advance the simulated clock by `delta` ms. If no override is set, seeds it
/// from the current wall clock first so subsequent reads stay virtual.
pub fn advanceMs(delta: i64) void {
    while (true) {
        const cur = override_ms.load(.acquire);
        const base = if (cur != no_override) cur else milliNowWall();
        const next = base + delta;
        if (override_ms.cmpxchgWeak(cur, next, .acq_rel, .acquire) == null) return;
        // Lost the race (concurrent advanceMs/setOverrideMs); retry with fresh base.
    }
}

fn milliNowWall() i64 {
    if (comptime is_freestanding) return 0;
    var ts: std.posix.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &ts);
    return @as(i64, ts.sec) * 1000 + @divTrunc(@as(i64, ts.nsec), 1_000_000);
}

/// Milliseconds since epoch (simulated when override is set).
pub fn milliNow() i64 {
    const t = override_ms.load(.acquire);
    if (t != no_override) return t;
    return milliNowWall();
}

/// Nanoseconds since epoch (simulated when override is set).
/// Simulated values are override_ms * 1e6 so seed derivation stays tied to the same clock.
pub fn nanoNow() i96 {
    const t = override_ms.load(.acquire);
    if (t != no_override) return @as(i96, t) * 1_000_000;
    if (comptime is_freestanding) return 0;
    var ts: std.posix.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &ts);
    return @as(i96, ts.sec) * 1_000_000_000 + ts.nsec;
}

test "override freezes milliNow" {
    defer setOverrideMs(null);
    setOverrideMs(1_700_000_000_000);
    try std.testing.expectEqual(@as(i64, 1_700_000_000_000), milliNow());
    advanceMs(500);
    try std.testing.expectEqual(@as(i64, 1_700_000_000_500), milliNow());
    try std.testing.expectEqual(@as(i96, 1_700_000_000_500) * 1_000_000, nanoNow());
}

test "advanceMs accumulates from wall when unset" {
    defer setOverrideMs(null);
    // Seed via advance without prior setOverrideMs.
    advanceMs(0);
    const t0 = milliNow();
    advanceMs(250);
    try std.testing.expectEqual(t0 + 250, milliNow());
}
