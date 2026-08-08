//! Expert usage profiling for MoE models.
//!
//! Tracks which routed experts are activated during inference, building
//! per-layer frequency counts. The resulting profile is used for:
//!   1. Expert hotlist generation (which experts to pin in fast cache)
//!   2. SSD streaming cache policy (which experts to keep resident)
//!   3. Quality diagnostics (expert load balance)
//!
//! Library API (no CLI flag yet; call from model/MoE paths or tests):
//!   var profile = try ExpertProfile.init(allocator, n_layers, n_experts);
//!   defer profile.deinit(allocator);
//!   profile.record(layer, expert_id);  // hot path: zero-alloc
//!   try profile.writeJson(allocator, path);
//!
//! Based on expert profiling from antirez/ds4.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Maximum supported number of layers for stack-allocated iteration.
const max_layers: usize = 256;
/// Maximum supported number of experts per layer.
const max_experts: usize = 512;

pub const ExpertProfile = struct {
    /// Per-layer, per-expert activation counts: [n_layers][n_experts].
    counts: []u64,
    n_layers: u32,
    n_experts: u32,
    /// Total tokens profiled (for computing frequencies).
    total_tokens: u64 = 0,

    pub fn init(allocator: Allocator, n_layers: u32, n_experts: u32) !ExpertProfile {
        if (n_layers > max_layers or n_experts > max_experts) return error.ProfileDimensionTooLarge;
        const total = @as(usize, n_layers) * @as(usize, n_experts);
        const counts = try allocator.alloc(u64, total);
        @memset(counts, 0);
        return ExpertProfile{
            .counts = counts,
            .n_layers = n_layers,
            .n_experts = n_experts,
        };
    }

    pub fn deinit(self: *ExpertProfile, allocator: Allocator) void {
        if (self.counts.len > 0) {
            allocator.free(self.counts);
            self.counts = &.{};
        }
    }

    /// Record an expert activation. Hot path — no allocation.
    pub inline fn record(self: *ExpertProfile, layer: u32, expert_id: u32) void {
        if (layer >= self.n_layers or expert_id >= self.n_experts) return;
        self.counts[@as(usize, layer) * self.n_experts + expert_id] += 1;
    }

    /// Mark a token boundary (increment total_tokens).
    pub inline fn recordToken(self: *ExpertProfile) void {
        self.total_tokens += 1;
    }

    /// Get activation count for a specific expert.
    pub fn getCount(self: *const ExpertProfile, layer: u32, expert_id: u32) u64 {
        if (layer >= self.n_layers or expert_id >= self.n_experts) return 0;
        return self.counts[@as(usize, layer) * self.n_experts + expert_id];
    }

    /// Get the top-K most active experts for a layer (sorted by count, descending).
    /// Returns the number of experts written to `out_ids` (≤ k).
    pub fn topExperts(self: *const ExpertProfile, layer: u32, k: u32, out_ids: []u32) u32 {
        if (layer >= self.n_layers) return 0;
        const ne = self.n_experts;
        const base = @as(usize, layer) * ne;
        const actual_k = @min(k, ne);
        const out_k = @min(actual_k, @as(u32, @intCast(out_ids.len)));

        // Simple selection sort for top-K (experts per layer is small, typically 64-256)
        var used = [_]bool{false} ** max_experts;
        var written: u32 = 0;
        while (written < out_k) {
            var best_id: u32 = 0;
            var best_count: u64 = 0;
            for (0..ne) |e| {
                if (!used[e] and self.counts[base + e] > best_count) {
                    best_count = self.counts[base + e];
                    best_id = @intCast(e);
                }
            }
            if (best_count == 0) break;
            out_ids[written] = best_id;
            used[best_id] = true;
            written += 1;
        }
        return written;
    }

    /// Write profile data as JSON to a file path.
    pub fn writeJson(self: *const ExpertProfile, allocator: Allocator, path: []const u8) !void {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(allocator);

        try buf.appendSlice(allocator, "{\n");
        try buf.print(allocator, "  \"n_layers\": {d},\n", .{self.n_layers});
        try buf.print(allocator, "  \"n_experts\": {d},\n", .{self.n_experts});
        try buf.print(allocator, "  \"total_tokens\": {d},\n", .{self.total_tokens});
        try buf.appendSlice(allocator, "  \"layers\": [\n");

        for (0..self.n_layers) |li| {
            const base = li * self.n_experts;
            try buf.appendSlice(allocator, "    [");
            for (0..self.n_experts) |ei| {
                if (ei > 0) try buf.appendSlice(allocator, ", ");
                try buf.print(allocator, "{d}", .{self.counts[base + ei]});
            }
            try buf.appendSlice(allocator, "]");
            if (li < self.n_layers - 1) try buf.appendSlice(allocator, ",");
            try buf.appendSlice(allocator, "\n");
        }

        try buf.appendSlice(allocator, "  ]\n}\n");

        const open_flags: u32 = if (comptime @import("builtin").os.tag == .linux) (1 | 64 | 512) else (1 | 0x200 | 0x400);
        const fd = std.posix.openat(std.posix.AT.FDCWD, path, @bitCast(open_flags), 0o644) catch |e| return e;
        defer _ = std.posix.system.close(fd);
        var off: usize = 0;
        while (off < buf.items.len) {
            const n = std.c.write(fd, buf.items[off..].ptr, buf.items.len - off);
            if (n <= 0) break;
            off += @intCast(n);
        }
    }

    /// Load a profile written by writeJson. Minimal hand-rolled parser
    /// (avoids pulling in a full JSON library for this single use case).
    pub fn loadJson(allocator: Allocator, path: []const u8) !ExpertProfile {
        const fd = std.posix.openat(std.posix.AT.FDCWD, path, .{}, 0) catch |e| return e;
        defer _ = std.posix.system.close(fd);
        // Stat for size, then read all.
        const fsize: usize = blk: {
            if (comptime @import("builtin").os.tag == .linux) {
                var buf: std.os.linux.Statx = undefined;
                const rc = std.os.linux.statx(fd, @ptrCast(""), std.os.linux.AT.EMPTY_PATH, std.os.linux.STATX{ .SIZE = true }, &buf);
                if (rc != 0) return error.StatFailed;
                break :blk @intCast(buf.size);
            } else {
                var st: std.posix.Stat = undefined;
                if (std.c.fstat(fd, &st) != 0) return error.StatFailed;
                break :blk @intCast(@max(0, st.size));
            }
        };
        if (fsize == 0) return error.EmptyFile;
        const data = try allocator.alloc(u8, fsize);
        var got: usize = 0;
        while (got < fsize) {
            const n = std.posix.read(fd, data[got..]) catch break;
            if (n == 0) break;
            got += n;
        }
        if (got < fsize) return error.ReadFailed;
        defer allocator.free(data);

        // Parse n_layers and n_experts from the header lines.
        var n_layers: u32 = 0;
        var n_experts: u32 = 0;
        var total_tokens: u64 = 0;
        var lines = std.mem.splitScalar(u8, data, '\n');
        while (lines.next()) |line| {
            const t = std.mem.trim(u8, line, " \t\r,");
            if (std.mem.startsWith(u8, t, "\"n_layers\"")) {
                const col = std.mem.lastIndexOfScalar(u8, t, ':') orelse continue;
                n_layers = std.fmt.parseInt(u32, std.mem.trim(u8, t[col + 1 ..], " ,:"), 10) catch continue;
            } else if (std.mem.startsWith(u8, t, "\"n_experts\"")) {
                const col = std.mem.lastIndexOfScalar(u8, t, ':') orelse continue;
                n_experts = std.fmt.parseInt(u32, std.mem.trim(u8, t[col + 1 ..], " ,:"), 10) catch continue;
            } else if (std.mem.startsWith(u8, t, "\"total_tokens\"")) {
                const col = std.mem.lastIndexOfScalar(u8, t, ':') orelse continue;
                total_tokens = std.fmt.parseInt(u64, std.mem.trim(u8, t[col + 1 ..], " ,:"), 10) catch continue;
            }
        }
        if (n_layers == 0 or n_experts == 0) return error.InvalidProfileJson;

        var profile = try ExpertProfile.init(allocator, n_layers, n_experts);
        profile.total_tokens = total_tokens;

        // Parse layers array: each line inside "layers": [ [...], ... ]
        var in_layers = false;
        var layer_idx: usize = 0;
        var lines2 = std.mem.splitScalar(u8, data, '\n');
        while (lines2.next()) |line| {
            const t = std.mem.trim(u8, line, " \t\r");
            if (std.mem.indexOf(u8, t, "\"layers\"") != null) {
                in_layers = true;
                continue;
            }
            if (!in_layers) continue;
            if (!std.mem.startsWith(u8, t, "[")) continue;
            if (layer_idx >= n_layers) break;
            // Parse comma-separated numbers between [ and ]
            const inner_start = std.mem.indexOfScalar(u8, t, '[') orelse continue;
            const inner_end = std.mem.lastIndexOfScalar(u8, t, ']') orelse continue;
            const inner = t[inner_start + 1 .. inner_end];
            var nums = std.mem.splitScalar(u8, inner, ',');
            var ei: usize = 0;
            while (nums.next()) |ns| {
                if (ei >= n_experts) break;
                const v = std.fmt.parseInt(u64, std.mem.trim(u8, ns, " "), 10) catch continue;
                profile.counts[layer_idx * n_experts + ei] = v;
                ei += 1;
            }
            layer_idx += 1;
        }
        return profile;
    }
};

// ── Tests ────────────────────────────────────────────────────────

test "ExpertProfile record and getCount" {
    const allocator = std.testing.allocator;
    var profile = try ExpertProfile.init(allocator, 4, 8);
    defer profile.deinit(allocator);

    profile.record(0, 3);
    profile.record(0, 3);
    profile.record(0, 5);
    profile.record(1, 7);

    try std.testing.expectEqual(@as(u64, 2), profile.getCount(0, 3));
    try std.testing.expectEqual(@as(u64, 1), profile.getCount(0, 5));
    try std.testing.expectEqual(@as(u64, 1), profile.getCount(1, 7));
    try std.testing.expectEqual(@as(u64, 0), profile.getCount(0, 0));
    try std.testing.expectEqual(@as(u64, 0), profile.getCount(99, 0)); // out of range
}

test "ExpertProfile topExperts" {
    const allocator = std.testing.allocator;
    var profile = try ExpertProfile.init(allocator, 2, 4);
    defer profile.deinit(allocator);

    // Layer 0: expert 2 most active, then expert 0
    profile.record(0, 2);
    profile.record(0, 2);
    profile.record(0, 2);
    profile.record(0, 0);
    profile.record(0, 0);
    profile.record(0, 3);

    var top: [4]u32 = undefined;
    const n = profile.topExperts(0, 3, &top);
    try std.testing.expectEqual(@as(u32, 3), n);
    try std.testing.expectEqual(@as(u32, 2), top[0]); // highest: 3 activations
    try std.testing.expectEqual(@as(u32, 0), top[1]); // second: 2 activations
    try std.testing.expectEqual(@as(u32, 3), top[2]); // third: 1 activation
}

test "fuzz: record + topExperts bounds" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const allocator = std.testing.allocator;
            const n_layers: u32 = @as(u32, smith.valueWithHash(u3, 0)) + 1; // 1..8
            const n_experts: u32 = @as(u32, smith.valueWithHash(u4, 1)) + 1; // 1..16
            var profile = try ExpertProfile.init(allocator, n_layers, n_experts);
            defer profile.deinit(allocator);

            // Oversized dims must reject
            try std.testing.expectError(error.ProfileDimensionTooLarge, ExpertProfile.init(allocator, max_layers + 1, 1));
            try std.testing.expectError(error.ProfileDimensionTooLarge, ExpertProfile.init(allocator, 1, max_experts + 1));

            const rounds = smith.valueWithHash(u5, 2) + 1;
            for (0..rounds) |i| {
                profile.record(smith.valueWithHash(u8, @truncate(10 + i)), smith.valueWithHash(u8, @truncate(20 + i)));
                profile.recordToken();
            }

            var out: [16]u32 = undefined;
            const k = smith.valueWithHash(u8, 3);
            const layer = smith.valueWithHash(u8, 4);
            const written = profile.topExperts(layer, k, &out);
            try std.testing.expect(written <= out.len);
            if (layer >= n_layers) {
                try std.testing.expect(written == 0);
            } else {
                try std.testing.expect(written <= @min(@min(k, n_experts), @as(u32, @intCast(out.len))));
            }
            for (out[0..written]) |id| {
                try std.testing.expect(id < n_experts);
            }
            _ = profile.getCount(layer, smith.valueWithHash(u8, 5));
        }
    }.f, .{});
}
