//! Injectable wall clock for deterministic simulation and tests.
//!
//! Production leaves the override unset so milliNow/nanoNow read the OS clock
//! (REALTIME). Interval timers that must resist NTP skew (profiling, download
//! progress) use CLOCK_MONOTONIC directly and do not go through this module.
//!
//! Tests and a future sim harness call setOverrideMs / advanceMs to drive
//! timeouts, rate-limit refill, and scheduling priority from a single seed.
//! Under override, sleepNs advances virtual time and returns immediately so
//! poll loops do not block wall-clock time.
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

/// True when milliNow/nanoNow/sleepNs are driven by the override (not the OS clock).
pub fn isOverridden() bool {
    return override_ms.load(.acquire) != no_override;
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
    _ = std.posix.system.clock_gettime(.REALTIME, &ts);
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
    _ = std.posix.system.clock_gettime(.REALTIME, &ts);
    return @as(i96, ts.sec) * 1_000_000_000 + ts.nsec;
}

/// Sleep `ns` nanoseconds. Under a clock override, advances the virtual clock
/// by floor(ns / 1e6) ms and returns immediately (no wall-clock block) so
/// scheduler polls, sleep-mode monitors, and rate-limit waits are sim-driven.
pub fn sleepNs(ns: u64) void {
    if (isOverridden()) {
        const ms: i64 = @intCast(ns / std.time.ns_per_ms);
        if (ms > 0) advanceMs(ms);
        return;
    }
    if (comptime is_freestanding) return;
    const ts = std.posix.timespec{
        .sec = @intCast(ns / std.time.ns_per_s),
        .nsec = @intCast(ns % std.time.ns_per_s),
    };
    _ = std.posix.system.nanosleep(&ts, null);
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

test "sleepNs advances virtual clock when overridden" {
    defer setOverrideMs(null);
    setOverrideMs(1_000_000);
    sleepNs(5 * std.time.ns_per_ms);
    try std.testing.expectEqual(@as(i64, 1_000_005), milliNow());
    // Sub-millisecond sleeps must not move the ms clock.
    sleepNs(500_000);
    try std.testing.expectEqual(@as(i64, 1_000_005), milliNow());
    try std.testing.expect(isOverridden());
}
