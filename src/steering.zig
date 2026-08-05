//! Directional steering: runtime activation editing via learned direction vectors.
//!
//! Loads a flat f32 file containing one normalized direction vector per layer
//! (n_layers × n_embd floats). During inference, projects the direction out of
//! (or into) the activation after attention and/or FFN outputs:
//!
//!   y = y - scale * direction[layer] * dot(direction[layer], y)
//!
//! Positive scale removes the direction. Negative scale amplifies it.
//! With no file or zero scales, this is a no-op.
//!
//! Usage:
//!   var steer = try DirectionalSteering.init(allocator, path, n_layers, n_embd, ffn_scale, attn_scale);
//!   defer steer.deinit(allocator);
//!   steer.apply(hidden_ptr, layer_idx, scale);  // hot path: zero-alloc
//!
//! Based on the technique from antirez/ds4's directional steering.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Maximum supported embedding dimension. Larger models are rejected at init
/// so apply() stays a bounded, allocation-free walk of the direction vector.
const max_embd_dim: usize = 16384;

pub const DirectionalSteering = struct {
    /// Per-layer direction vectors, contiguous: [n_layers][n_embd] f32.
    directions: []const f32,
    /// Number of transformer layers.
    n_layers: u32,
    /// Embedding dimension per layer.
    n_embd: u32,
    /// FFN output steering scale (0 = disabled).
    ffn_scale: f32,
    /// Attention output steering scale (0 = disabled).
    attn_scale: f32,

    /// Initialize from pre-loaded f32 data (caller reads the file).
    /// `data` must contain exactly n_layers × n_embd floats.
    /// The steering struct takes ownership of the allocation.
    pub fn initFromData(
        data: []const f32,
        n_layers: u32,
        n_embd: u32,
        ffn_scale: f32,
        attn_scale: f32,
    ) !DirectionalSteering {
        if (n_embd > max_embd_dim) return error.EmbeddingTooLarge;
        const expected_floats: usize = @as(usize, n_layers) * @as(usize, n_embd);
        if (data.len != expected_floats) {
            std.log.err("steering: data size {d} floats != expected {d} ({d} layers × {d} embd)", .{
                data.len, expected_floats, n_layers, n_embd,
            });
            return error.SteeringFileSizeMismatch;
        }

        return DirectionalSteering{
            .directions = data,
            .n_layers = n_layers,
            .n_embd = n_embd,
            .ffn_scale = ffn_scale,
            .attn_scale = attn_scale,
        };
    }

    /// Load direction vectors from a flat f32 file via Io.
    /// File must contain exactly n_layers × n_embd × 4 bytes.
    pub fn init(
        allocator: Allocator,
        io: anytype,
        path: []const u8,
        n_layers: u32,
        n_embd: u32,
        ffn_scale: f32,
        attn_scale: f32,
    ) !DirectionalSteering {
        if (n_embd > max_embd_dim) return error.EmbeddingTooLarge;
        const expected_floats: usize = @as(usize, n_layers) * @as(usize, n_embd);
        const expected_bytes: usize = expected_floats * @sizeOf(f32);
        const Io = @TypeOf(io);
        const Dir = if (@hasDecl(Io, "Dir")) Io.Dir else std.fs.Dir;

        const file = try Dir.cwd().openFile(io, path, .{});
        defer file.close(io);

        const directions = try allocator.alloc(f32, expected_floats);
        errdefer allocator.free(directions);

        const bytes: []u8 = @as([*]u8, @ptrCast(directions.ptr))[0..expected_bytes];
        const bytes_read = file.readPositionalAll(io, bytes, 0) catch |err| {
            std.log.err("steering: read failed: {}", .{err});
            return error.SteeringFileShortRead;
        };
        if (bytes_read != expected_bytes) {
            std.log.err("steering: file size {d} != expected {d} ({d} layers × {d} embd × 4)", .{
                bytes_read, expected_bytes, n_layers, n_embd,
            });
            return error.SteeringFileSizeMismatch;
        }

        return DirectionalSteering{
            .directions = directions,
            .n_layers = n_layers,
            .n_embd = n_embd,
            .ffn_scale = ffn_scale,
            .attn_scale = attn_scale,
        };
    }

    pub fn deinit(self: *DirectionalSteering, allocator: Allocator) void {
        if (self.directions.len > 0) {
            allocator.free(self.directions);
            self.directions = &.{};
        }
    }

    /// Apply directional steering to a hidden state vector.
    /// Hot path: zero allocations, pure arithmetic.
    ///
    /// y = y - scale * direction[layer] * dot(direction[layer], y)
    ///
    /// When scale > 0: removes the direction (suppresses the concept).
    /// When scale < 0: amplifies the direction.
    pub inline fn apply(self: *const DirectionalSteering, y: [*]f32, layer: u32, scale: f32) void {
        if (scale == 0 or layer >= self.n_layers) return;

        const e: usize = self.n_embd;
        const offset: usize = @as(usize, layer) * e;
        const dir = self.directions[offset..][0..e];

        // Dot product: dir · y
        var dot: f32 = 0;
        // Tight f32 accumulation; compiler typically auto-vectorizes.
        for (0..e) |i| {
            dot += dir[i] * y[i];
        }

        // Project: y = y - scale * dot * dir
        const coeff = scale * dot;
        for (0..e) |i| {
            y[i] -= coeff * dir[i];
        }
    }

    /// Apply steering after attention output (if attn_scale != 0).
    pub inline fn applyAttn(self: *const DirectionalSteering, y: [*]f32, layer: u32) void {
        self.apply(y, layer, self.attn_scale);
    }

    /// Apply steering after FFN output (if ffn_scale != 0).
    pub inline fn applyFfn(self: *const DirectionalSteering, y: [*]f32, layer: u32) void {
        self.apply(y, layer, self.ffn_scale);
    }

    /// Returns true if any steering is active.
    pub inline fn isActive(self: *const DirectionalSteering) bool {
        return self.ffn_scale != 0 or self.attn_scale != 0;
    }
};

// ── Tests ────────────────────────────────────────────────────────

test "DirectionalSteering.apply projects out direction" {
    // Direction = [1, 0, 0, 0] (unit vector along dim 0)
    // y = [3, 4, 5, 6], scale = 1
    // dot = 3, coeff = 3
    // result = [3-3, 4-0, 5-0, 6-0] = [0, 4, 5, 6]
    const dir = [_]f32{ 1, 0, 0, 0 };
    var y = [_]f32{ 3, 4, 5, 6 };
    const steer = DirectionalSteering{
        .directions = &dir,
        .n_layers = 1,
        .n_embd = 4,
        .ffn_scale = 1.0,
        .attn_scale = 0,
    };
    steer.apply(&y, 0, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0), y[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), y[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5), y[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6), y[3], 1e-6);
}

test "DirectionalSteering.apply amplifies with negative scale" {
    // Direction = [1, 0, 0, 0], y = [3, 4, 5, 6], scale = -1
    // dot = 3, coeff = -3
    // result = [3-(-3), 4, 5, 6] = [6, 4, 5, 6]
    const dir = [_]f32{ 1, 0, 0, 0 };
    var y = [_]f32{ 3, 4, 5, 6 };
    const steer = DirectionalSteering{
        .directions = &dir,
        .n_layers = 1,
        .n_embd = 4,
        .ffn_scale = 0,
        .attn_scale = 0,
    };
    steer.apply(&y, 0, -1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 6), y[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), y[1], 1e-6);
}

test "DirectionalSteering.apply zero scale is no-op" {
    const dir = [_]f32{ 1, 0, 0, 0 };
    var y = [_]f32{ 3, 4, 5, 6 };
    const steer = DirectionalSteering{
        .directions = &dir,
        .n_layers = 1,
        .n_embd = 4,
        .ffn_scale = 0,
        .attn_scale = 0,
    };
    steer.apply(&y, 0, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 3), y[0], 1e-6);
}

test "DirectionalSteering.apply out-of-range layer is no-op" {
    const dir = [_]f32{ 1, 0, 0, 0 };
    var y = [_]f32{ 3, 4, 5, 6 };
    const steer = DirectionalSteering{
        .directions = &dir,
        .n_layers = 1,
        .n_embd = 4,
        .ffn_scale = 1.0,
        .attn_scale = 0,
    };
    steer.apply(&y, 99, 1.0); // layer 99 > n_layers=1
    try std.testing.expectApproxEqAbs(@as(f32, 3), y[0], 1e-6);
}

test "DirectionalSteering.apply diagonal direction" {
    // Direction = [0.5, 0.5, 0.5, 0.5] (normalized: 1/2 each), scale = 2
    // y = [1, 1, 1, 1]
    // dot = 0.5+0.5+0.5+0.5 = 2, coeff = 4
    // result = [1-2, 1-2, 1-2, 1-2] = [-1, -1, -1, -1]
    const dir = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var y = [_]f32{ 1, 1, 1, 1 };
    const steer = DirectionalSteering{
        .directions = &dir,
        .n_layers = 1,
        .n_embd = 4,
        .ffn_scale = 0,
        .attn_scale = 0,
    };
    steer.apply(&y, 0, 2.0);
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, -1), y[i], 1e-6);
    }
}

test "fuzz: initFromData + apply no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const allocator = std.testing.allocator;
            // Keep dims small so each iteration stays allocation-light
            const n_layers: u32 = @as(u32, smith.valueWithHash(u3, 0)) + 1; // 1..8
            const n_embd: u32 = @as(u32, smith.valueWithHash(u4, 1)) + 1; // 1..16
            const n_floats = @as(usize, n_layers) * @as(usize, n_embd);

            // Size mismatch must error
            {
                const bad = try allocator.alloc(f32, n_floats + 1);
                defer allocator.free(bad);
                try std.testing.expectError(error.SteeringFileSizeMismatch, DirectionalSteering.initFromData(bad, n_layers, n_embd, 1.0, 0));
            }

            const data = try allocator.alloc(f32, n_floats);
            defer allocator.free(data);
            for (data, 0..) |*v, i| {
                v.* = @as(f32, @bitCast(smith.valueWithHash(u32, @truncate(i))));
            }
            const ffn_scale: f32 = @bitCast(smith.valueWithHash(u32, 2));
            const attn_scale: f32 = @bitCast(smith.valueWithHash(u32, 3));
            var steer = try DirectionalSteering.initFromData(data, n_layers, n_embd, ffn_scale, attn_scale);
            // initFromData takes ownership; prevent double-free of `data`
            steer.directions = data;
            defer {
                steer.directions = &.{};
            }

            var y = try allocator.alloc(f32, n_embd);
            defer allocator.free(y);
            for (y, 0..) |*v, i| {
                v.* = @as(f32, @bitCast(smith.valueWithHash(u32, @truncate(100 + i))));
            }

            const layer = smith.valueWithHash(u8, 4);
            const scale: f32 = @bitCast(smith.valueWithHash(u32, 5));
            steer.apply(y.ptr, layer, scale);
            steer.applyAttn(y.ptr, layer);
            steer.applyFfn(y.ptr, layer);
            _ = steer.isActive();
            // Outputs may be non-finite when inputs are NaN/Inf; must not crash
            for (y) |_| {}
        }
    }.f, .{});
}
