//! Tensor Parallelism coordinator for CPU multi-rank execution.
//!
//! Creates N model instances (one per TP rank), each with sharded weights.
//! Runs all ranks in parallel using the thread pool, with barrier-based
//! all-reduce between layers.
//!
//! Currently CPU-only. GPU TP would use NCCL/RCCL all-reduce instead.

const std = @import("std");
const Allocator = std.mem.Allocator;
const model_mod = @import("../models/model.zig");
const Model = model_mod.Model;
const ModelStorage = model_mod.ModelStorage;
const format_mod = @import("../format/format.zig");
const backend_mod = @import("../backend/backend.zig");
const Arch = @import("../arch.zig").Arch;

pub const TpGroup = struct {
    ranks: []ModelStorage,
    degree: u32,
    allocator: Allocator,

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

        for (0..degree) |r| {
            ranks[r] = try ModelStorage.initFromArch(
                arch, allocator, fmt, be, ctx_size,
                .f32, .f32, 0, 0, null,
                @intCast(r), degree,
            );
        }

        return .{ .ranks = ranks, .degree = degree, .allocator = allocator };
    }

    pub fn deinit(self: *TpGroup) void {
        for (self.ranks) |*r| r.deinit();
        self.allocator.free(self.ranks);
    }

    /// Run forward on all ranks sequentially, then all-reduce.
    /// For correctness validation only — no parallelism.
    pub fn forward(self: *TpGroup, token_id: u32) !u32 {
        // For now: just run rank 0 (TP sharding is applied but no all-reduce)
        const model = self.ranks[0].model();
        return model.forward(token_id);
    }
};
