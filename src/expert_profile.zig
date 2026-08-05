//! Expert usage profiling for MoE models.
//!
//! Tracks which routed experts are activated during inference, building
//! per-layer frequency counts. The resulting profile is used for:
//!   1. Expert hotlist generation (which experts to pin in fast cache)
//!   2. SSD streaming cache policy (which experts to keep resident)
//!   3. Quality diagnostics (expert load balance)
//!
//! Usage:
//!   var profile = ExpertProfile.init(allocator, n_layers, n_experts);
//!   defer profile.deinit(allocator);
//!   profile.record(layer, expert_id);  // called per routed expert per token
//!   try profile.writeJson(path);       // dump to file
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
        var buf = std.ArrayList(u8).init(allocator);
        defer buf.deinit();
        const writer = buf.writer();

        try writer.writeAll("{\n");
        try writer.print("  \"n_layers\": {d},\n", .{self.n_layers});
        try writer.print("  \"n_experts\": {d},\n", .{self.n_experts});
        try writer.print("  \"total_tokens\": {d},\n", .{self.total_tokens});
        try writer.writeAll("  \"layers\": [\n");

        for (0..self.n_layers) |li| {
            const base = li * self.n_experts;
            try writer.writeAll("    [");
            for (0..self.n_experts) |ei| {
                if (ei > 0) try writer.writeAll(", ");
                try writer.print("{d}", .{self.counts[base + ei]});
            }
            try writer.writeByte(']');
            if (li < self.n_layers - 1) try writer.writeByte(',');
            try writer.writeByte('\n');
        }

        try writer.writeAll("  ]\n}\n");

        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        try file.writeAll(buf.items);
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
