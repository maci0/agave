//! LoRA adapter loading and load-time merging into GGUF base models.
//!
//! Supports the GGUF LoRA format produced by llama.cpp's convert_lora_to_gguf.py:
//!   adapter.type       = "lora"
//!   adapter.lora.alpha = <f32>    (scaling factor; effective_scale = alpha / rank)
//!   blk.{i}.{name}.lora_a        [rank, in_features]
//!   blk.{i}.{name}.lora_b        [out_features, rank]
//!
//! Merge strategy: load-time F32 merge.
//!   merged[n, k] = dequant(base[n, k]) + (alpha/rank) * lora_b[n, rank] @ lora_a[rank, k]
//!
//! Merged tensors are stored in GGUFFile.lora_overrides. getTensor() returns the
//! override transparently, so all models see the merged weight without any hot-path overhead.
//!
//! `applyLoraGguf` returns a Handle whose `dispose` unmerges this adapter
//! (restores the previous override or the mmap base). Stacked applies compose;
//! dispose is LIFO.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const gguf = @import("format/gguf.zig");
const quant = @import("ops/quant.zig");
const backend_mod = @import("backend/backend.zig");

/// Revertible LoRA apply. `dispose` restores tensors this apply changed.
pub const Handle = struct {
    const Record = struct {
        key: []const u8,
        prev: ?gguf.GGUFFile.LoraOverride,
    };

    allocator: Allocator,
    file: ?*gguf.GGUFFile = null,
    records: std.ArrayList(Record) = .empty,

    /// Unmerge this adapter. LIFO vs later applies. Idempotent.
    pub fn dispose(self: *Handle) void {
        const file = self.file orelse {
            self.records.deinit(self.allocator);
            self.records = .empty;
            return;
        };
        var i = self.records.items.len;
        while (i > 0) {
            i -= 1;
            const rec = self.records.items[i];
            restoreOverride(file, rec.key, rec.prev);
        }
        self.records.deinit(self.allocator);
        self.records = .empty;
        self.file = null;
    }
};

fn restoreOverride(file: *gguf.GGUFFile, key: []const u8, prev: ?gguf.GGUFFile.LoraOverride) void {
    if (file.lora_overrides.getEntry(key)) |ent| {
        file.allocator.free(ent.value_ptr.data);
        if (prev) |p| {
            ent.value_ptr.* = p;
        } else {
            const removed = file.lora_overrides.fetchRemove(key).?;
            file.allocator.free(removed.key);
        }
    } else if (prev) |p| {
        file.allocator.free(p.data);
    }
}

/// Apply a LoRA adapter GGUF file to a base GGUFFile in place.
/// Modified tensors are stored as F32 in base_gguf.lora_overrides.
/// Caller must keep base_gguf alive until `Handle.dispose` (or GGUFFile.deinit).
pub fn applyLoraGguf(
    allocator: Allocator,
    base_gguf: *gguf.GGUFFile,
    lora_path: []const u8,
) !Handle {
    var lora_file = try gguf.GGUFFile.open(allocator, lora_path);
    defer lora_file.deinit();
    return applyLoraGgufFile(allocator, base_gguf, &lora_file);
}

/// Apply an already-parsed adapter GGUF file to a base GGUFFile in place.
/// Same contract as `applyLoraGguf`; split out so callers holding a parsed
/// file (and tests) skip disk access.
fn applyLoraGgufFile(
    allocator: Allocator,
    base_gguf: *gguf.GGUFFile,
    lora_file: *const gguf.GGUFFile,
) !Handle {
    const adapter_type = lora_file.getMetaStr("adapter.type") orelse
        lora_file.getMetaStr("general.type") orelse "";
    if (!std.mem.eql(u8, adapter_type, "lora")) return error.NotALoraAdapter;

    const alpha = lora_file.getMetaF32("adapter.lora.alpha") orelse 1.0;

    var handle = Handle{ .allocator = allocator, .file = base_gguf };
    errdefer handle.dispose();

    // Iterate lora tensors. Only process lora_a entries; find paired lora_b and base.
    var lora_iter = lora_file.tensors.iterator();
    while (lora_iter.next()) |kv| {
        const lora_a_name = kv.key_ptr.*;
        if (!std.mem.endsWith(u8, lora_a_name, ".lora_a")) continue;

        const base_suffix = lora_a_name[0 .. lora_a_name.len - ".lora_a".len];

        var b_buf: [512]u8 = undefined;
        const lora_b_name = std.fmt.bufPrint(&b_buf, "{s}.lora_b", .{base_suffix}) catch {
            std.log.warn("LoRA: tensor name too long, skipping: {s}", .{base_suffix});
            continue;
        };
        const lora_b_info = lora_file.tensors.get(lora_b_name) orelse continue;
        const lora_a_info = kv.value_ptr.*;

        // rank = lora_a.dims[0], k = lora_a.dims[1], n = lora_b.dims[0], lora_b.dims[1] = rank
        const rank: usize = @intCast(lora_a_info.dims[0]);
        const k: usize = @intCast(lora_a_info.dims[1]);
        const n: usize = @intCast(lora_b_info.dims[0]);
        const rank_b: usize = @intCast(lora_b_info.dims[1]);
        if (rank == 0 or k == 0 or n == 0) {
            std.log.warn("LoRA: skipping '{s}', zero dimension (rank={d}, k={d}, n={d})", .{ base_suffix, rank, k, n });
            continue;
        }
        if (rank_b != rank) {
            std.log.warn("LoRA: skipping '{s}', rank mismatch between lora_a ({d}) and lora_b ({d})", .{ base_suffix, rank, rank_b });
            continue;
        }

        const scale = alpha / @as(f32, @floatFromInt(rank));

        // Find base tensor, try bare name, then with ".weight" suffix
        const base_ti: *gguf.TensorInfo = blk: {
            if (base_gguf.tensors.getPtr(base_suffix)) |p| break :blk p;
            var w_buf: [512]u8 = undefined;
            const w_name = std.fmt.bufPrint(&w_buf, "{s}.weight", .{base_suffix}) catch {
                std.log.warn("LoRA: tensor name too long, skipping: {s}", .{base_suffix});
                continue;
            };
            break :blk base_gguf.tensors.getPtr(w_name) orelse continue;
        };

        const base_n: usize = @intCast(base_ti.dims[0]);
        const base_k: usize = @intCast(base_ti.dims[1]);
        if (base_n != n or base_k != k) {
            std.log.warn("LoRA: skipping '{s}', dimension mismatch: base [{d}, {d}] vs LoRA [{d}, {d}]", .{ base_suffix, base_n, base_k, n, k });
            continue;
        }

        // Dequant lora_a [rank × k] and lora_b [n × rank] to F32
        const la = try allocator.alloc(f32, rank * k);
        defer allocator.free(la);
        quant.dequantToF32(la, lora_file.tensorData(&lora_a_info), gguf.GGUFFile.ggmlToDType(lora_a_info.ggml_type), rank * k);

        const lb = try allocator.alloc(f32, n * rank);
        defer allocator.free(lb);
        quant.dequantToF32(lb, lora_file.tensorData(&lora_b_info), gguf.GGUFFile.ggmlToDType(lora_b_info.ggml_type), n * rank);

        // Allocate merged buffer [n × k]. Start from a previous adapter if present
        // so stacked applies compose; otherwise dequant the mmap base.
        const merged = try allocator.alloc(f32, n * k);
        errdefer allocator.free(merged);
        if (base_gguf.lora_overrides.get(base_ti.name)) |ov| {
            @memcpy(merged, ov.data);
        } else {
            quant.dequantToF32(merged, base_gguf.tensorData(base_ti), gguf.GGUFFile.ggmlToDType(base_ti.ggml_type), n * k);
        }

        // Add LoRA delta: merged += scale * (lb[n,rank] @ la[rank,k])
        // Use Accelerate on macOS for ~4× speedup via AMX; fall back to scalar.
        addLoraMatrix(merged, lb, la, n, rank, k, scale);

        // Insert override keyed by the GGUF canonical name (dupe'd, mmap pointer will be freed).
        const key = try allocator.dupe(u8, base_ti.name);
        errdefer allocator.free(key);
        const gop = try base_gguf.lora_overrides.getOrPut(allocator, key);
        const prev: ?gguf.GGUFFile.LoraOverride = if (gop.found_existing) blk: {
            allocator.free(key);
            break :blk gop.value_ptr.*;
        } else null;
        gop.value_ptr.* = .{
            .data = merged,
            .n_dims = base_ti.n_dims,
            .dims = base_ti.dims,
        };
        try handle.records.append(allocator, .{ .key = gop.key_ptr.*, .prev = prev });
    }
    return handle;
}

test "addLoraMatrix 2×2 rank-1" {
    // base = [[1,2],[3,4]], b = [[2],[1]], a = [[1,1]], scale = 1.0
    // delta = b @ a = [[2,2],[1,1]], merged = base + delta = [[3,4],[4,5]]
    var merged = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 2, 1 };
    const a = [_]f32{ 1, 1 };
    addLoraMatrix(&merged, &b, &a, 2, 1, 2, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), merged[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), merged[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), merged[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), merged[3], 1e-5);
}

test "addLoraMatrix scale" {
    // base = [0,0,0,0], b = [[1],[1]], a = [[1,1]], scale = 0.5 → delta = 0.5 * [[1,1],[1,1]]
    var merged = [_]f32{ 0, 0, 0, 0 };
    const b = [_]f32{ 1, 1 };
    const a = [_]f32{ 1, 1 };
    addLoraMatrix(&merged, &b, &a, 2, 1, 2, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), merged[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), merged[1], 1e-5);
}

/// merged[n,k] += scale * b[n,rank] @ a[rank,k]
fn addLoraMatrix(
    merged: []f32,
    b: []const f32,
    a: []const f32,
    n: usize,
    rank: usize,
    k: usize,
    scale: f32,
) void {
    // macOS: use Accelerate SGEMM (AMX-accelerated, ~4× faster than NEON scalar).
    // merged[n,k] += scale * b[n,rank] @ a[rank,k], beta=1 accumulates in place.
    if (comptime builtin.os.tag == .macos) {
        const build_options = @import("build_options");
        if (comptime build_options.enable_metal) {
            backend_mod.accelerate.sgemmAdd(n, k, rank, scale, b.ptr, a.ptr, merged.ptr);
            return;
        }
    }
    // Scalar fallback (Linux, non-Metal macOS)
    for (0..n) |i| {
        const b_row = b[i * rank ..][0..rank];
        const m_row = merged[i * k ..][0..k];
        for (0..k) |j| {
            var acc: f32 = 0.0;
            for (0..rank) |r| acc += b_row[r] * a[r * k + j];
            m_row[j] += scale * acc;
        }
    }
}

fn testGguf(allocator: Allocator) gguf.GGUFFile {
    return .{
        .is_buffer = true,
        .file_size = 0,
        .metadata = std.StringHashMap(gguf.MetaValue).init(allocator),
        .tensors = std.StringHashMap(gguf.TensorInfo).init(allocator),
        .allocator = allocator,
    };
}

test "lora handle dispose drops override" {
    const allocator = std.testing.allocator;
    var file = testGguf(allocator);
    defer file.deinit();

    const data = try allocator.alloc(f32, 4);
    @memcpy(data, &[_]f32{ 1, 2, 3, 4 });
    const key = try allocator.dupe(u8, "blk.0.weight");
    try file.lora_overrides.put(allocator, key, .{
        .data = data,
        .n_dims = 2,
        .dims = .{ 2, 2, 0, 0 },
    });

    var handle = Handle{ .allocator = allocator, .file = &file };
    try handle.records.append(allocator, .{ .key = key, .prev = null });
    handle.dispose();

    try std.testing.expectEqual(@as(u32, 0), file.lora_overrides.count());
}

test "lora handle dispose restores previous adapter" {
    const allocator = std.testing.allocator;
    var file = testGguf(allocator);
    defer file.deinit();

    const older = try allocator.alloc(f32, 2);
    @memcpy(older, &[_]f32{ 1, 1 });
    const newer = try allocator.alloc(f32, 2);
    @memcpy(newer, &[_]f32{ 9, 9 });
    const key = try allocator.dupe(u8, "tok.weight");
    try file.lora_overrides.put(allocator, key, .{
        .data = newer,
        .n_dims = 1,
        .dims = .{ 2, 0, 0, 0 },
    });

    var handle = Handle{ .allocator = allocator, .file = &file };
    try handle.records.append(allocator, .{
        .key = key,
        .prev = .{
            .data = older,
            .n_dims = 1,
            .dims = .{ 2, 0, 0, 0 },
        },
    });
    handle.dispose();

    const ov = file.lora_overrides.get(key).?;
    try std.testing.expectEqual(@as(f32, 1), ov.data[0]);
    try std.testing.expectEqual(@as(f32, 1), ov.data[1]);
}

test "lora handle dispose is idempotent" {
    const allocator = std.testing.allocator;
    var handle = Handle{ .allocator = allocator };
    handle.dispose();
    handle.dispose();
}

// ── Fuzzing ──────────────────────────────────────────────────────

test "fuzz: LoRA adapter merge over hostile GGUF" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const allocator = std.testing.allocator;
            const n: usize = 4;
            const k: usize = 8;

            // Base tensor data [n × k]: bounded finite floats.
            var base_data: [n * k]f32 = undefined;
            for (&base_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;

            var base_gguf = testGguf(allocator);
            defer base_gguf.deinit();

            // Adapter metadata: type marker and alpha.
            var lora_gguf = testGguf(allocator);
            defer lora_gguf.deinit();
            const type_ok = smith.valueWithHash(u8, 61) & 1 == 0;
            const adapter_type: []const u8 = if (type_ok) "lora" else "model";
            lora_gguf.metadata.put("adapter.type", .{ .string = adapter_type }) catch return;
            const alpha: f32 = @as(f32, @floatFromInt(smith.valueWithHash(u8, 62) % 8)) + 1.0;
            lora_gguf.metadata.put("adapter.lora.alpha", .{ .float32 = alpha }) catch return;

            // Adapter tensors. The well-formed shape is rank ∈ 1..8 paired
            // [rank,k] / [n,rank]; mutations exercise every guard:
            // rank mismatch, zero dims, missing pair, oversized name,
            // oversized dims (alloc failure), non-F32 dtype.
            const rank: usize = smith.indexWithHash(8, 0) + 1;
            const variant = smith.indexWithHash(6, 1);

            var name_buf: [640]u8 = undefined;
            const long_name = variant == 4;
            const p0_suffix: []const u8 = if (long_name) blk: {
                @memset(&name_buf, 'x');
                break :blk name_buf[0 .. name_buf.len - ".lora_a".len];
            } else "p0";

            var a_name_buf: [700]u8 = undefined;
            var b_name_buf: [700]u8 = undefined;
            const a0_name = std.fmt.bufPrint(&a_name_buf, "{s}.lora_a", .{p0_suffix}) catch return;
            const b0_name = std.fmt.bufPrint(&b_name_buf, "{s}.lora_b", .{p0_suffix}) catch return;

            // Data buffers sized for the largest declared shape; declared
            // dims choose how much of each the merge actually reads, mirroring
            // what the GGUF loader validated against real file sizes.
            var a_data: [8 * k]f32 = undefined;
            var b_data: [n * 8]f32 = undefined;
            for (&a_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 10)))) / 10.0;
            for (&b_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 30)))) / 10.0;

            const a_dims: [4]u64 = switch (variant) {
                2 => .{ 0, k, 0, 0 }, // zero rank
                5 => .{ 1 << 20, 1 << 20, 0, 0 }, // alloc-failure scale
                else => .{ rank, k, 0, 0 },
            };
            const b_rank: u64 = if (variant == 1) rank + 1 else rank; // rank mismatch
            // BF16 keeps every converted value finite (top halves of bounded
            // F32s), unlike F16 bitcasts which can produce Inf/NaN payloads.
            const b_dtype: gguf.GGMLType = if (variant == 6) .bf16 else .f32;

            const put_tensor = struct {
                fn put(file: *gguf.GGUFFile, name: []const u8, dims: [4]u64, dt: gguf.GGMLType, data_ptr: *anyopaque) void {
                    file.tensors.put(name, .{
                        .name = name,
                        .n_dims = 2,
                        .dims = dims,
                        .ggml_type = dt,
                        .offset = 0,
                        .abs_ptr = @ptrCast(data_ptr),
                    }) catch unreachable;
                }
            }.put;

            put_tensor(&lora_gguf, a0_name, a_dims, .f32, &a_data);
            if (variant != 3) { // v3: lora_b missing entirely
                put_tensor(&lora_gguf, b0_name, .{ n, b_rank, 0, 0 }, b_dtype, &b_data);
            }
            // Second pair with a short name: always passes the name guard so
            // guard variants isolate to zero records while v4 isolates to one.
            put_tensor(&lora_gguf, "p1.lora_a", a_dims, .f32, &a_data);
            if (variant != 3) {
                put_tensor(&lora_gguf, "p1.lora_b", .{ n, b_rank, 0, 0 }, b_dtype, &b_data);
            }

            // Base tensors: "p0.weight" resolves via the ".weight" fallback,
            // "p1" via the bare-name path. Both share the same shape.
            base_gguf.tensors.put("p0.weight", .{
                .name = "p0.weight",
                .n_dims = 2,
                .dims = .{ n, k, 0, 0 },
                .ggml_type = .f32,
                .offset = 0,
                .abs_ptr = @ptrCast(&base_data),
            }) catch return;
            base_gguf.tensors.put("p1", .{
                .name = "p1",
                .n_dims = 2,
                .dims = .{ n, k, 0, 0 },
                .ggml_type = .f32,
                .offset = 0,
                .abs_ptr = @ptrCast(&base_data),
            }) catch return;
            // Unrelated tensor must be ignored by the lora_a scan.
            put_tensor(&lora_gguf, "decoy.weight", .{ 2, 2, 0, 0 }, .f32, &a_data);

            // Stacked compose target lives on the fallback-resolved base.
            var prev_data: [n * k]f32 = undefined;
            const stacked = smith.valueWithHash(u8, 60) & 1 == 0;
            if (stacked) {
                for (&prev_data, 0..) |*v, i| v.* = 100.0 + @as(f32, @floatFromInt(i));
                const prev_key = allocator.dupe(u8, "p0.weight") catch return;
                base_gguf.lora_overrides.put(allocator, prev_key, .{
                    .data = allocator.dupe(f32, &prev_data) catch {
                        allocator.free(prev_key);
                        return;
                    },
                    .n_dims = 2,
                    .dims = .{ n, k, 0, 0 },
                }) catch return;
            }
            const baseline_count = base_gguf.lora_overrides.count();

            var handle = applyLoraGgufFile(allocator, &base_gguf, &lora_gguf) catch {
                // A failed apply (oversized-dims alloc) must leave overrides
                // exactly as before.
                try std.testing.expectEqual(baseline_count, base_gguf.lora_overrides.count());
                return;
            };

            // Variants 1-3 hit skip guards (rank mismatch, zero dim, missing
            // pair) and merge nothing; v4 skips only the oversized-name pair.
            const expected_total: usize = switch (variant) {
                1, 2, 3 => 0,
                4 => 1,
                else => 2,
            };
            // Pre-dispose checks in a scope so errdefer cannot double-fire
            // after the explicit dispose below.
            {
                errdefer handle.dispose();
                try std.testing.expectEqual(expected_total, handle.records.items.len);
                for (handle.records.items) |rec| {
                    const ov = base_gguf.lora_overrides.get(rec.key).?;
                    try std.testing.expect(ov.data.len == n * k);
                    for (ov.data) |v| try std.testing.expect(std.math.isFinite(v));
                }
                // Resolution-path pins: fallback pair keyed by full base name,
                // bare pair keyed without suffix.
                var saw_fallback = false;
                var saw_bare = false;
                for (handle.records.items) |rec| {
                    if (std.mem.eql(u8, rec.key, "p0.weight")) saw_fallback = true;
                    if (std.mem.eql(u8, rec.key, "p1")) saw_bare = true;
                }
                try std.testing.expect(saw_bare == (expected_total >= 1));
                try std.testing.expect(saw_fallback == (expected_total == 2));
            }

            // Dispose must restore the pre-apply state: planted override back,
            // or keys fully removed when none existed.
            handle.dispose();
            try std.testing.expectEqual(baseline_count, base_gguf.lora_overrides.count());
            if (stacked) {
                const ov = base_gguf.lora_overrides.get("p0.weight").?;
                try std.testing.expectEqualSlices(f32, &prev_data, ov.data);
            } else {
                try std.testing.expect(base_gguf.lora_overrides.get("p0.weight") == null);
            }
            try std.testing.expect(base_gguf.lora_overrides.get("p1") == null);
        }
    }.f, .{});
}
