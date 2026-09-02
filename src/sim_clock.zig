//! Injectable clocks for deterministic simulation and tests.
//!
//! Production leaves the override unset so each reader hits its natural OS
//! clock: milliNow/nanoNow read REALTIME (wall time: log timestamps, seeds,
//! epoch fields), while monoMilli/monoNano read MONOTONIC (interval math:
//! timeouts, rate-limit refill, scheduling priority, LRU stamps, power
//! throttle) so NTP steps or manual clock changes cannot produce negative
//! or inflated durations. Interval timers outside this module (perf
//! counters, download progress) also use CLOCK_MONOTONIC directly.
//!
//! Tests and a future sim harness call setOverrideMs / advanceMs to drive
//! timeouts, rate-limit refill, and scheduling priority from a single seed.
//! Under override both readers share the one simulated timeline so logic
//! mixing wall and interval reads stays consistent. Under override, sleepNs
//! advances virtual time and returns immediately so poll loops do not block
//! wall-clock time.
//!
//! The override is process-global (not thread-local). Tests that set it must
//! `defer setOverrideMs(null)` and must not run concurrently with other tests
//! that also override the clock. Storage is atomic on every threaded target so
//! concurrent milliNow readers never tear a partially-updated override value.

const std = @import("std");
const builtin = @import("builtin");
const is_freestanding = builtin.os.tag == .freestanding;

/// Sentinel: no override installed (milliNow falls through to the wall clock).
const no_override: i64 = std.math.minInt(i64);

/// Storage for the override, holding `no_override` when unset.
///
/// wasm32-freestanding has no threads and its baseline CPU has no 64-bit
/// atomics (`@atomicLoad` on an i64 is a compile error there), so that target
/// keeps a plain variable. Every other target stores the value atomically so
/// concurrent milliNow readers never see a torn update.
const override_ms = if (is_freestanding) struct {
    var v: i64 = no_override;

    fn load() i64 {
        return v;
    }
    fn store(next: i64) void {
        v = next;
    }
    /// Returns null when the swap succeeded, else the value observed instead.
    fn cmpxchg(expected: i64, next: i64) ?i64 {
        if (v != expected) return v;
        v = next;
        return null;
    }
} else struct {
    var v: std.atomic.Value(i64) = .init(no_override);

    fn load() i64 {
        return v.load(.acquire);
    }
    fn store(next: i64) void {
        v.store(next, .release);
    }
    /// Returns null when the swap succeeded, else the value observed instead.
    fn cmpxchg(expected: i64, next: i64) ?i64 {
        return v.cmpxchgWeak(expected, next, .acq_rel, .acquire);
    }
};

/// Install (or clear) a simulated millisecond clock. Pass null to restore wall time.
pub fn setOverrideMs(ms: ?i64) void {
    override_ms.store(ms orelse no_override);
}

/// True when milliNow/nanoNow/monoMilli/monoNano/sleepNs are driven by the override.
pub fn isOverridden() bool {
    return override_ms.load() != no_override;
}

/// Advance the simulated clock by `delta` ms. If no override is set, seeds it
/// from the current wall clock first so subsequent reads stay virtual.
pub fn advanceMs(delta: i64) void {
    while (true) {
        const cur = override_ms.load();
        const base = if (cur != no_override) cur else milliNowWall();
        const next = base + delta;
        if (override_ms.cmpxchg(cur, next) == null) return;
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
    const t = override_ms.load();
    if (t != no_override) return t;
    return milliNowWall();
}

fn monoNowWall() i64 {
    if (comptime is_freestanding) return 0;
    var ts: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    return @as(i64, ts.sec) * 1000 + @divTrunc(@as(i64, ts.nsec), 1_000_000);
}

/// Monotonic milliseconds for interval math (timeouts, refill, aging, LRU).
///
/// Never steps backwards and never jumps on NTP corrections, unlike
/// milliNow(); durations computed from two monoMilli() reads are always
/// sane. Not comparable across processes or to epoch time. Shares the
/// simulated timeline with milliNow() under override so tests drive both
/// readers from one virtual clock.
pub fn monoMilli() i64 {
    const t = override_ms.load();
    if (t != no_override) return t;
    return monoNowWall();
}

fn monoNowNanoWall() i96 {
    if (comptime is_freestanding) return 0;
    var ts: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    return @as(i96, ts.sec) * 1_000_000_000 + ts.nsec;
}

/// Monotonic nanoseconds for sub-millisecond interval math (power throttle).
///
/// Production reads CLOCK_MONOTONIC so NTP steps cannot inflate or negate
/// a measured duration. Under override, values are override_ms * 1e6 (the
/// same simulated timeline as milliNow/monoMilli/nanoNow) so a frozen
/// virtual clock yields a zero delta and a replay does not inherit host
/// execution speed.
pub fn monoNano() i96 {
    const t = override_ms.load();
    if (t != no_override) return @as(i96, t) * 1_000_000;
    return monoNowNanoWall();
}

/// Nanoseconds since epoch (simulated when override is set).
/// Simulated values are override_ms * 1e6 so seed derivation stays tied to the same clock.
pub fn nanoNow() i96 {
    const t = override_ms.load();
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

test "monoMilli follows override and monotonic clock" {
    defer setOverrideMs(null);
    // Production: two reads never go backwards (CLOCK_MONOTONIC).
    const a = monoMilli();
    const b = monoMilli();
    try std.testing.expect(b >= a);
    // Under override: same simulated timeline as milliNow().
    setOverrideMs(1_700_000_000_000);
    try std.testing.expectEqual(milliNow(), monoMilli());
    advanceMs(250);
    try std.testing.expectEqual(@as(i64, 1_700_000_000_250), monoMilli());
}

test "monoNano follows override and monotonic clock" {
    defer setOverrideMs(null);
    const a = monoNano();
    const b = monoNano();
    try std.testing.expect(b >= a);
    setOverrideMs(1_700_000_000_000);
    try std.testing.expectEqual(nanoNow(), monoNano());
    advanceMs(250);
    try std.testing.expectEqual(@as(i96, 1_700_000_000_250) * 1_000_000, monoNano());
    // A pair of reads with no advance is a zero delta (replay-stable).
    try std.testing.expectEqual(@as(i96, 0), monoNano() - monoNano());
}
