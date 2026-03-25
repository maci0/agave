//! Per-operation profiling for inference bottleneck detection.
//! Enable with `--profile` to sync and time each operation type.
//! Results are printed after generation completes.
//!
//! WARNING: profiling inserts GPU syncs between every operation,
//! which destroys pipeline overlap and roughly halves throughput.
//! The relative percentages are still meaningful for identifying bottlenecks.

const std = @import("std");

/// Microseconds per millisecond — for display conversion.
const us_per_ms: f64 = 1000.0;

/// Multiplier to convert a fraction to a percentage.
const percent_scale: f64 = 100.0;

/// Operation categories for profiling.
pub const Op = enum {
    emb_lookup,
    rms_norm,
    gemv_qkv,
    gemv_out,
    gemv_ffn,
    deinterleave,
    rope,
    sdpa,
    sigmoid_mul,
    silu_mul,
    gelu_mul,
    add,
    deltanet,
    total_layer,
};

const n_ops = @typeInfo(Op).@"enum".fields.len;

/// Buffer size for the profiling report output.
const report_buf_size: usize = 4096;

/// Accumulates wall-clock time per operation type across all tokens.
pub const PerfCounters = struct {
    counts: [n_ops]u64 = [_]u64{0} ** n_ops,
    times_us: [n_ops]u64 = [_]u64{0} ** n_ops,
    n_tokens: u64 = 0,
    enabled: bool = false,

    /// Begin timing an operation. Returns the current timestamp (or 0 if profiling is disabled).
    pub fn start(self: *PerfCounters) i128 {
        if (!self.enabled) return 0;
        return std.time.nanoTimestamp();
    }

    /// End timing for `op`, accumulating elapsed microseconds since `t0`.
    pub fn end(self: *PerfCounters, op: Op, t0: i128) void {
        if (!self.enabled) return;
        const elapsed: u64 = @intCast(@divFloor(std.time.nanoTimestamp() - t0, 1000));
        const idx = @intFromEnum(op);
        self.times_us[idx] += elapsed;
        self.counts[idx] += 1;
    }

    /// Increment the generated token count (used for per-token averaging in reports).
    pub fn addToken(self: *PerfCounters) void {
        if (self.enabled) self.n_tokens += 1;
    }

    /// Print a table of per-op timing breakdown.
    pub fn report(self: *const PerfCounters) void {
        if (!self.enabled or self.n_tokens == 0) return;
        var total_us: u64 = 0;
        for (self.times_us) |t| total_us += t;
        if (total_us == 0) return;

        const stderr = std.fs.File.stderr();
        var buf: [report_buf_size]u8 = undefined;
        const write = struct {
            fn w(f: std.fs.File, b: []u8, comptime fmt: []const u8, args: anytype) void {
                f.writeAll(std.fmt.bufPrint(b, fmt, args) catch return) catch {};
            }
        }.w;

        write(stderr, &buf, "\n\x1b[1m── Profile ({d} tokens, {d:.1}ms avg) ──\x1b[0m\n", .{
            self.n_tokens,
            @as(f64, @floatFromInt(total_us)) / @as(f64, @floatFromInt(self.n_tokens)) / us_per_ms,
        });
        write(stderr, &buf, "{s:<16} {s:>8} {s:>10} {s:>8} {s:>6}\n", .{ "Operation", "Calls", "Total(ms)", "Avg(µs)", "%" });
        write(stderr, &buf, "{s}\n", .{"─" ** 54});

        const fields = @typeInfo(Op).@"enum".fields;
        inline for (fields) |field| {
            const idx = field.value;
            if (self.counts[idx] > 0) {
                const pct = @as(f64, @floatFromInt(self.times_us[idx])) / @as(f64, @floatFromInt(total_us)) * percent_scale;
                const avg = @as(f64, @floatFromInt(self.times_us[idx])) / @as(f64, @floatFromInt(self.counts[idx]));
                write(stderr, &buf, "{s:<16} {d:>8} {d:>10.1} {d:>8.0} {d:>5.1}%\n", .{
                    field.name,
                    self.counts[idx],
                    @as(f64, @floatFromInt(self.times_us[idx])) / us_per_ms,
                    avg,
                    pct,
                });
            }
        }
        write(stderr, &buf, "{s}\n", .{"─" ** 54});
        write(stderr, &buf, "{s:<16} {s:>8} {d:>10.1}\n", .{
            "TOTAL",                                    "",
            @as(f64, @floatFromInt(total_us)) / us_per_ms,
        });
    }
};

test "PerfCounters disabled is no-op" {
    var pc = PerfCounters{};
    const t0 = pc.start();
    try std.testing.expectEqual(@as(i128, 0), t0);
    pc.end(.rope, t0); // should not crash or accumulate
    try std.testing.expectEqual(@as(u64, 0), pc.counts[@intFromEnum(Op.rope)]);
    try std.testing.expectEqual(@as(u64, 0), pc.times_us[@intFromEnum(Op.rope)]);
    pc.addToken(); // should not increment
    try std.testing.expectEqual(@as(u64, 0), pc.n_tokens);
}

test "PerfCounters enabled tracks calls" {
    var pc = PerfCounters{ .enabled = true };
    pc.addToken();
    try std.testing.expectEqual(@as(u64, 1), pc.n_tokens);
    pc.addToken();
    try std.testing.expectEqual(@as(u64, 2), pc.n_tokens);
    // Verify start/end actually accumulates timing
    const t0 = pc.start();
    try std.testing.expect(t0 != 0); // enabled → non-zero timestamp
    pc.end(.rope, t0);
    try std.testing.expectEqual(@as(u64, 1), pc.counts[@intFromEnum(Op.rope)]);
    try std.testing.expect(pc.times_us[@intFromEnum(Op.rope)] > 0);
    // Second end call should increment count
    const t1 = pc.start();
    pc.end(.rope, t1);
    try std.testing.expectEqual(@as(u64, 2), pc.counts[@intFromEnum(Op.rope)]);
}
