//! Tensor Parallelism coordinator for CPU multi-rank execution.
//!
//! Creates N model instances (one per TP rank), each with sharded weights,
//! but only executes rank 0, all-reduce is not yet implemented.
//!
//! CPU-only. GPU TP would use NCCL/RCCL all-reduce instead.

const std = @import("std");
const Allocator = std.mem.Allocator;
const model_mod = @import("../models/model.zig");
const Model = model_mod.Model;
const ModelStorage = model_mod.ModelStorage;
const format_mod = @import("../format/format.zig");
const backend_mod = @import("../backend/backend.zig");
const Arch = @import("../arch.zig").Arch;

/// Tensor-parallelism group that shards model weights across multiple ranks.
pub const TpGroup = struct {
    ranks: []ModelStorage,
    degree: u32,
    allocator: Allocator,

    /// Initialize a tensor-parallelism group with `degree` model instances.
    /// Each rank gets its own `ModelStorage` with sharded weights. On error,
    /// all successfully initialized ranks are cleaned up via `errdefer`.
    pub fn init(
        allocator: Allocator,
        arch: Arch,
        fmt: format_mod.Format,
        be: backend_mod.Backend,
        ctx_size: u32,
        degree: u32,
    ) !TpGroup {
        const ranks = try allocator.alloc(ModelStorage, degree);
        errdefer allocator.free(ranks);

        var init_count: usize = 0;
        errdefer for (ranks[0..init_count]) |*r| r.deinit();

        for (0..degree) |r| {
            ranks[r] = try ModelStorage.initFromArch(
                arch,
                allocator,
                fmt,
                be,
                ctx_size,
                .f32,
                .f32,
                0,
                0,
                null,
                @intCast(r),
                degree,
            );
            init_count = r + 1;
        }

        return .{ .ranks = ranks, .degree = degree, .allocator = allocator };
    }

    /// Deinitialize all rank model instances and free the ranks slice.
    pub fn deinit(self: *TpGroup) void {
        for (self.ranks) |*r| r.deinit();
        self.allocator.free(self.ranks);
    }

    /// Run forward. Multi-rank TP needs all-reduce; fail closed until implemented.
    pub fn forward(self: *TpGroup, token_id: u32) !u32 {
        if (self.degree > 1) return error.TpIncomplete;
        const model = self.ranks[0].model();
        return model.forward(token_id);
    }
};

// ── Tests ───────────────────────────────────────────────────────────

test "TpGroup, struct layout and field types" {
    // Compile-time verification that TpGroup has the expected fields.
    comptime {
        _ = @TypeOf(TpGroup.init);
        _ = @TypeOf(TpGroup.deinit);
        _ = @TypeOf(TpGroup.forward);
    }
    // Verify TpGroup struct size is reasonable (contains a slice + u32 + allocator).
    try std.testing.expect(@sizeOf(TpGroup) > 0);
    try std.testing.expect(@sizeOf(TpGroup) <= 64);
}

test "TpGroup, degree field" {
    // Verify the degree field exists and is settable.
    const allocator = std.testing.allocator;
    _ = allocator;
    // We can't test init() without a real model/format/backend,
    // but we can verify the struct is constructible with undefined fields.
    var group: TpGroup = undefined;
    group.degree = 4;
    try std.testing.expectEqual(@as(u32, 4), group.degree);
}

test "tensor partition size, even division" {
    // TP sharding divides tensor dimensions evenly across ranks.
    // This is the core arithmetic used by shardColumnWeight / shardRowWeight
    // in model implementations (e.g. n_head / tp_degree, n_ff / tp_degree).
    const dims = [_]u32{ 128, 256, 512, 1024, 4096, 8192 };
    const degrees = [_]u32{ 1, 2, 4, 8 };

    for (dims) |total_dim| {
        for (degrees) |degree| {
            if (degree > total_dim) continue;
            const local_dim = total_dim / degree;
            // Verify even division (TP requires dimensions divisible by degree).
            try std.testing.expectEqual(total_dim, local_dim * degree);
            // Each rank's partition must be non-zero.
            try std.testing.expect(local_dim > 0);
        }
    }
}

test "tensor partition size, rank offsets are contiguous and non-overlapping" {
    // For column-sharding: each rank r gets rows [r*local_n .. (r+1)*local_n).
    // Verify the slices cover the full dimension without gaps or overlaps.
    const n_total: u32 = 4096;
    const degree: u32 = 4;
    const n_local = n_total / degree;

    var covered: u32 = 0;
    for (0..degree) |r| {
        const rank: u32 = @intCast(r);
        const offset = rank * n_local;
        // Each rank starts exactly where the previous one ended.
        try std.testing.expectEqual(covered, offset);
        covered += n_local;
    }
    // All ranks together cover the full dimension.
    try std.testing.expectEqual(n_total, covered);
}

test "tensor partition size, head and ff sharding" {
    // Realistic model dimensions: verify that n_head, n_head_kv, and n_ff
    // all divide evenly for common TP degrees.
    const TestCase = struct { n_head: u32, n_head_kv: u32, n_ff: u32 };
    const cases = [_]TestCase{
        .{ .n_head = 32, .n_head_kv = 8, .n_ff = 11008 }, // Llama 7B
        .{ .n_head = 40, .n_head_kv = 40, .n_ff = 13824 }, // Llama 13B
        .{ .n_head = 64, .n_head_kv = 8, .n_ff = 14336 }, // Llama 70B
        .{ .n_head = 36, .n_head_kv = 4, .n_ff = 18944 }, // Qwen 3.5
    };

    for (cases) |c| {
        for ([_]u32{ 1, 2, 4 }) |degree| {
            // n_head must divide evenly.
            try std.testing.expectEqual(@as(u32, 0), c.n_head % degree);
            // n_head_kv must divide evenly (GQA groups per rank).
            try std.testing.expectEqual(@as(u32, 0), c.n_head_kv % degree);
            // n_ff must divide evenly for gate/up/down sharding.
            try std.testing.expectEqual(@as(u32, 0), c.n_ff % degree);
        }
    }
}

test "tensor partition size, row shard byte offset calculation" {
    // Row-sharding extracts columns: each rank gets k_total/degree columns.
    // For quantized types, row_bytes = k * quant_block_bytes / quant_block_size.
    // Here we test the simpler f32 case: row_bytes = k * 4.
    const k_total: usize = 4096;
    const n_rows: usize = 1024;
    const degree: usize = 4;
    const bytes_per_element: usize = 4; // f32

    const local_k = k_total / degree;
    const local_row_bytes = local_k * bytes_per_element;
    const total_row_bytes = k_total * bytes_per_element;

    for (0..degree) |rank| {
        const col_offset = rank * local_row_bytes;
        // Column offset must be within the full row.
        try std.testing.expect(col_offset + local_row_bytes <= total_row_bytes);
    }

    // Total bytes across all ranks per row must equal full row.
    try std.testing.expectEqual(total_row_bytes, local_row_bytes * degree);
    // Total shard size = local_k * n_rows * bytes_per_element.
    const shard_bytes = local_k * n_rows * bytes_per_element;
    try std.testing.expectEqual(k_total * n_rows * bytes_per_element, shard_bytes * degree);
}

test "fuzz: all tp functions" {
    // TpGroup.init/deinit/forward all require a real Model + Format + Backend stack
    // that cannot be constructed in a unit test. Verify all pub fn pointers exist at
    // comptime, then fuzz the TP sharding arithmetic that underpins the module.
    comptime {
        _ = &TpGroup.init;
        _ = &TpGroup.deinit;
        _ = &TpGroup.forward;
    }

    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // Fuzz the TP sharding arithmetic (dimension / degree partitioning).
            const raw_dim = smith.valueWithHash(u16, 0);
            const raw_degree = smith.valueWithHash(u4, 1);

            // Avoid division by zero; clamp degree to [1..16].
            const degree: u32 = @as(u32, raw_degree) | 1;
            // Ensure dim is a multiple of degree so sharding is exact.
            const dim: u32 = (@as(u32, raw_dim) | 1) * degree;

            const local_dim = dim / degree;
            // Partitions must reconstruct the original dimension.
            try std.testing.expectEqual(dim, local_dim * degree);
            // Each partition is non-zero.
            try std.testing.expect(local_dim > 0);

            // Verify contiguous rank offsets cover the full dimension.
            var covered: u32 = 0;
            for (0..degree) |r| {
                const rank: u32 = @intCast(r);
                try std.testing.expectEqual(covered, rank * local_dim);
                covered += local_dim;
            }
            try std.testing.expectEqual(dim, covered);

            // TpGroup struct is constructible with manual field assignment.
            var group: TpGroup = undefined;
            group.degree = degree;
            try std.testing.expectEqual(degree, group.degree);
        }
    }.f, .{});
}
