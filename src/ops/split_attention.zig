//! Split-attention: async CPU-GPU KV cache offloading.
//! Partitions KV cache blocks by tier, dispatches GPU and CPU SDPA concurrently,
//! and merges partial outputs via online softmax correction (FlashAttention-2 style).
//!
//! Architecture overview:
//!   1. `partitionBlocks()` scans a layer's block table and classifies blocks as
//!      GPU-resident (VRAM) or CPU-resident (RAM/SSD) based on `TieredBlock.tier`.
//!   2. `splitAttention()` checks the partition:
//!      - All GPU → fast path: `be.sdpa()` (zero overhead).
//!      - All CPU → CPU SDPA on thread pool (or calling thread if no pool).
//!      - Mixed, host-resident prefix then resident suffix → one partial per
//!        side, combined by `splitSdpaMerge()`. That is the shape a tiered
//!        cache produces, since it demotes oldest-first.
//!      - Mixed and interleaved → full-sequence CPU SDPA after `be.sync()`.
//!        Exact, just not split.
//!   3. `splitSdpaMerge()` combines the partials with FlashAttention-2 online
//!      softmax correction: exact, no approximation. It needs each side's real
//!      per-head max and denominator, which is why `sdpaWithStats` must report
//!      them rather than the identity (max=0, sum=1). A backend with no
//!      stats-reporting kernel delegates to the CPU instead of writing the
//!      identity, which would silently reweight both partials.

const std = @import("std");
const tiered = @import("../kvcache/tiered.zig");
const TieredBlock = tiered.TieredBlock;
const Backend = @import("../backend/backend.zig").Backend;
const kv_quant = @import("kv_quant.zig");
const KvQuantType = kv_quant.KvQuantType;
const ThreadPool = @import("../thread_pool.zig").ThreadPool;
const sdpa_cpu = @import("../backend/backend.zig").CpuSdpa;
/// Block layout for the tier tests below. From the leaf layout module, not the
/// cache manager, so this file never reaches past `tiered.zig` into manager.zig.
const CacheBlock = @import("../kvcache/view.zig").CacheBlock;

/// SIMD vector width (number of f32 lanes) for merge accumulation loops.
const simd_width: usize = 8;

/// Maximum number of contiguous ranges in a partition.
/// Bounded by max blocks per sequence (256 blocks × block_size positions each).
const max_ranges: usize = 256;

/// Head count the merged split path keeps its per-head stats for on the
/// stack. Past this the full-sequence CPU path runs instead, which needs no
/// stats at all; no model in tree comes close, and a heap allocation here
/// would be on the decode path.
const max_split_heads: usize = 128;

/// Boundary where a partition is exactly "CPU prefix, then GPU suffix", or
/// null when the tiers interleave.
///
/// The merged split path handles two contiguous halves. A tiered cache
/// demotes oldest-first, so that is the shape it produces, but a partition
/// that interleaves cannot be served by one slice per side.
fn contiguousPrefixSplit(partition: *const Partition, total: usize) ?usize {
    if (partition.gpu_count == 0) return null;
    const split = partition.total_cpu_positions;
    if (split == 0 or split >= total) return null;
    if (split + partition.total_gpu_positions != total) return null;
    for (partition.cpu[0..partition.cpu_count]) |r| {
        if (r.start + r.len > split) return null;
    }
    for (partition.gpu[0..partition.gpu_count]) |r| {
        if (r.start < split) return null;
    }
    return split;
}

/// A contiguous range of sequence positions assigned to a tier.
pub const SeqRange = struct {
    /// First position (inclusive).
    start: usize,
    /// Number of positions in this range.
    len: usize,
};

/// Partition result: GPU-resident and CPU-resident position ranges for a single layer.
pub const Partition = struct {
    /// GPU (VRAM) position ranges.
    gpu: [max_ranges]SeqRange = undefined,
    /// CPU (RAM/SSD) position ranges.
    cpu: [max_ranges]SeqRange = undefined,
    /// Number of valid entries in `gpu`.
    gpu_count: usize = 0,
    /// Number of valid entries in `cpu`.
    cpu_count: usize = 0,
    /// Total positions across all GPU ranges.
    total_gpu_positions: usize = 0,
    /// Total positions across all CPU ranges.
    total_cpu_positions: usize = 0,
};

/// Partition a layer's block table into GPU and CPU position ranges by inspecting
/// the tier of each physical block in `tiered_blocks`.
///
/// `tier` is written by the prefetch worker under `tier_lock`. Callers that
/// share the cache with a prefetcher must hold that lock (see
/// `partitionBlocksLocked`). Tests that own the block array exclusively may
/// call this unlocked.
///
/// Parameters:
///   - block_table: Per-layer mapping from logical block index to physical block ID.
///   - tiered_blocks: Full array of tiered blocks (indexed by physical block ID).
///   - block_size: Number of positions per block.
///   - seq_len: Total sequence length (positions to scan).
///
/// Returns: Partition with GPU and CPU ranges populated.
pub fn partitionBlocks(
    block_table: []const u32,
    tiered_blocks: []const TieredBlock,
    block_size: usize,
    seq_len: usize,
) Partition {
    var result = Partition{};
    var pos: usize = 0;

    for (block_table) |block_id| {
        if (pos >= seq_len) break;
        const block_len = @min(block_size, seq_len - pos);
        const tier = tiered_blocks[block_id].tier;

        if (tier == .vram) {
            // Coalesce with previous GPU range if contiguous
            if (result.gpu_count > 0 and result.gpu[result.gpu_count - 1].start + result.gpu[result.gpu_count - 1].len == pos) {
                result.gpu[result.gpu_count - 1].len += block_len;
            } else if (result.gpu_count < max_ranges) {
                result.gpu[result.gpu_count] = .{ .start = pos, .len = block_len };
                result.gpu_count += 1;
            }
            result.total_gpu_positions += block_len;
        } else {
            // .ram and .ssd both go to CPU path, coalesce if contiguous
            if (result.cpu_count > 0 and result.cpu[result.cpu_count - 1].start + result.cpu[result.cpu_count - 1].len == pos) {
                result.cpu[result.cpu_count - 1].len += block_len;
            } else if (result.cpu_count < max_ranges) {
                result.cpu[result.cpu_count] = .{ .start = pos, .len = block_len };
                result.cpu_count += 1;
            }
            result.total_cpu_positions += block_len;
        }
        pos += block_size;
    }
    return result;
}

/// `partitionBlocks` with `cache.tier_lock` held so a concurrent prefetch
/// promote cannot tear `tier` mid-scan.
pub fn partitionBlocksLocked(cache: *tiered.TieredKvCache, block_table: []const u32, seq_len: usize) Partition {
    cache.lockTier();
    defer cache.unlockTier();
    return partitionBlocks(block_table, cache.blocks, cache.block_size, seq_len);
}

/// Job context for parallelizing CPU-only SDPA across query heads (no stats needed).
/// Used in the all-CPU fast path where merge is not required.
const CpuSdpaNoStatsJob = struct {
    q: [*]const f32,
    keys: [*]const u8,
    values: [*]const u8,
    output: [*]f32,
    nh: usize,
    nkv: usize,
    hd: usize,
    sl: usize,
    scale: f32,
    kv_type_k: KvQuantType,
    kv_type_v: KvQuantType,

    /// Thread pool work function: process heads [start, end) without stats.
    fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
        const job: *CpuSdpaNoStatsJob = @ptrCast(@alignCast(ctx_ptr));
        for (start..end) |h| {
            sdpa_cpu.sdpaQuantHead(job.q, job.keys, job.values, job.output, h, job.nh, job.nkv, job.hd, job.sl, job.scale, job.kv_type_k, job.kv_type_v);
        }
    }
};

/// Job context for dispatching CPU SDPA across the thread pool.
/// Each thread processes a subset of query heads independently.
const CpuSdpaJob = struct {
    q: [*]const f32,
    keys: [*]const u8,
    values: [*]const u8,
    output: [*]f32,
    head_max: [*]f32,
    head_sum: [*]f32,
    nh: usize,
    nkv: usize,
    hd: usize,
    sl: usize,
    scale: f32,
    kv_type_k: KvQuantType,
    kv_type_v: KvQuantType,

    /// Thread pool work function: process heads [start, end) for this job.
    fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
        const job: *CpuSdpaJob = @ptrCast(@alignCast(ctx_ptr));
        for (start..end) |h| {
            sdpa_cpu.sdpaQuantHeadWithStats(
                job.q,
                job.keys,
                job.values,
                job.output,
                h,
                job.nh,
                job.nkv,
                job.hd,
                job.sl,
                job.scale,
                job.kv_type_k,
                job.kv_type_v,
                job.head_max,
                job.head_sum,
            );
        }
    }
};

/// Async split-attention: partitions KV by tier, runs GPU and CPU SDPA
/// concurrently, merges via online softmax.
///
/// Falls through to regular `be.sdpa()` when all blocks are GPU-resident (zero
/// overhead fast path). When blocks span tiers, GPU SDPA runs with stats export
/// (deferred on Metal/UMA), CPU SDPA runs on the thread pool concurrently, and
/// the two partial results are merged via `splitSdpaMerge()`.
///
/// Parameters:
///   - q: Query buffer [nh * hd].
///   - kv_keys, kv_values: KV cache byte slices for this layer.
///   - k_new, v_new: Current K/V vectors to append.
///   - output: Output buffer [nh * hd].
///   - gpu_out, cpu_out: Pre-allocated output buffers [nh * hd] for split results.
///   - nh, nkv, hd: Head configuration.
///   - seq_len: Current sequence position (before appending).
///   - scale: Attention scale factor.
///   - be: Backend for GPU SDPA dispatch.
///   - kv_type_k, kv_type_v: KV cache quantization formats.
///   - partition: Block tier partition (from partitionBlocks). Passed by pointer
///     because it is an 8 KiB struct read only on this per-layer, per-token path.
///   - pool: Optional thread pool for parallel CPU SDPA.
pub fn splitAttention(
    q: [*]const f32,
    kv_keys: []u8,
    kv_values: []u8,
    k_new: [*]const f32,
    v_new: [*]const f32,
    output: [*]f32,
    gpu_out: [*]f32,
    cpu_out: [*]f32,
    nh: usize,
    nkv: usize,
    hd: usize,
    seq_len: usize,
    scale: f32,
    be: Backend,
    kv_type_k: KvQuantType,
    kv_type_v: KvQuantType,
    partition: *const Partition,
    pool: ?*ThreadPool,
) void {
    // Fast path: all GPU, delegate directly, zero overhead
    if (partition.cpu_count == 0) {
        be.sdpa(q, kv_keys, kv_values, k_new, v_new, output, nh, nkv, hd, seq_len, scale, kv_type_k, kv_type_v);
        return;
    }

    be.sync();
    const kvd = nkv * hd;
    const k_off = kv_quant.kvByteOffset(kv_type_k, seq_len * kvd);
    const v_off = kv_quant.kvByteOffset(kv_type_v, seq_len * kvd);
    kv_quant.kvStore(kv_keys.ptr + k_off, k_new, kvd, kv_type_k);
    kv_quant.kvStore(kv_values.ptr + v_off, v_new, kvd, kv_type_v);

    // Mixed tiers: CPU takes the host-resident prefix, the backend takes the
    // resident suffix, and the two partials merge exactly.
    //
    // This used to run a full-sequence CPU SDPA instead, because GPU
    // backends reported identity softmax stats (max=0, sum=1) and merging
    // against those silently reweights both halves. They report real stats
    // now.
    //
    // Only for the layout a tiered cache actually produces: older positions
    // demoted to RAM or SSD, recent ones resident, so each side is one
    // contiguous slice. Anything more interleaved falls through to the
    // full-sequence CPU path below, which is exact.
    if (partition.gpu_count > 0 and nh <= max_split_heads) {
        if (contiguousPrefixSplit(partition, seq_len + 1)) |split| {
            const start_elems = split * kvd;
            if (split > 0 and
                kv_quant.kvSliceable(kv_type_k, start_elems) and
                kv_quant.kvSliceable(kv_type_v, start_elems))
            {
                var max_cpu: [max_split_heads]f32 = @splat(0);
                var sum_cpu: [max_split_heads]f32 = @splat(0);
                var max_gpu: [max_split_heads]f32 = @splat(0);
                var sum_gpu: [max_split_heads]f32 = @splat(0);

                // Host prefix, positions [0, split). The new token is in the
                // suffix, so this half must not see it.
                if (pool) |p| {
                    var ctx = CpuSdpaJob{
                        .q = q,
                        .keys = kv_keys.ptr,
                        .values = kv_values.ptr,
                        .output = cpu_out,
                        .head_max = &max_cpu,
                        .head_sum = &sum_cpu,
                        .nh = nh,
                        .nkv = nkv,
                        .hd = hd,
                        .sl = split,
                        .scale = scale,
                        .kv_type_k = kv_type_k,
                        .kv_type_v = kv_type_v,
                    };
                    p.parallelFor(nh, 1, @ptrCast(&ctx), CpuSdpaJob.work);
                } else {
                    for (0..nh) |h| {
                        sdpa_cpu.sdpaQuantHeadWithStats(q, kv_keys.ptr, kv_values.ptr, cpu_out, h, nh, nkv, hd, split, scale, kv_type_k, kv_type_v, &max_cpu, &sum_cpu);
                    }
                }

                // Resident suffix, positions [split, seq_len]. The backend
                // re-appends the new K/V at the same bytes it already holds.
                const k_slice_off = kv_quant.kvByteOffset(kv_type_k, start_elems);
                const v_slice_off = kv_quant.kvByteOffset(kv_type_v, start_elems);
                be.sdpaWithStats(
                    q,
                    kv_keys[k_slice_off..],
                    kv_values[v_slice_off..],
                    k_new,
                    v_new,
                    gpu_out,
                    &max_gpu,
                    &sum_gpu,
                    nh,
                    nkv,
                    hd,
                    seq_len - split,
                    scale,
                    kv_type_k,
                    kv_type_v,
                );
                be.sync();

                splitSdpaMerge(cpu_out, &max_cpu, &sum_cpu, gpu_out, &max_gpu, &sum_gpu, output, nh, hd);
                return;
            }
        }
    }
    if (pool) |p| {
        var ctx = CpuSdpaNoStatsJob{
            .q = q,
            .keys = kv_keys.ptr,
            .values = kv_values.ptr,
            .output = output,
            .nh = nh,
            .nkv = nkv,
            .hd = hd,
            .sl = seq_len + 1,
            .scale = scale,
            .kv_type_k = kv_type_k,
            .kv_type_v = kv_type_v,
        };
        p.parallelFor(nh, 1, @ptrCast(&ctx), CpuSdpaNoStatsJob.work);
    } else {
        sdpa_cpu.sdpaQuantHeads(q, kv_keys.ptr, kv_values.ptr, output, nh, nkv, hd, seq_len + 1, scale, kv_type_k, kv_type_v);
    }
}

/// Merge two partial SDPA outputs using online softmax correction.
/// Each split produced its own local softmax (out, max_per_head, sum_per_head).
/// This function combines them into the exact result of full SDPA over all positions.
///
/// The merge is exact, no approximation. Given two splits A and B with
/// independently normalized outputs and their softmax statistics (max, sum),
/// the combined output equals what a single SDPA over all positions would produce.
///
/// Parameters:
///   - out_a: First split normalized attention output [nh * hd].
///   - max_a: Per-head max score before exp in split A [nh].
///   - sum_a: Per-head sum of exp(scores - max) in split A [nh] (softmax denominator).
///   - out_b: Second split normalized attention output [nh * hd].
///   - max_b: Per-head max score before exp in split B [nh].
///   - sum_b: Per-head sum of exp(scores - max) in split B [nh] (softmax denominator).
///   - merged: Output buffer for the merged attention output [nh * hd].
///   - nh: Number of query heads.
///   - hd: Head dimension.
pub fn splitSdpaMerge(
    out_a: [*]const f32,
    max_a: [*]const f32,
    sum_a: [*]const f32,
    out_b: [*]const f32,
    max_b: [*]const f32,
    sum_b: [*]const f32,
    merged: [*]f32,
    nh: usize,
    hd: usize,
) void {
    for (0..nh) |h| {
        const ma = max_a[h];
        const mb = max_b[h];
        const max_all = @max(ma, mb);
        const corr_a = @exp(ma - max_all);
        const corr_b = @exp(mb - max_all);
        const sa = sum_a[h] * corr_a;
        const sb = sum_b[h] * corr_b;
        const sum_all = sa + sb;
        const inv_sum: f32 = if (sum_all > 0) 1.0 / sum_all else 0;

        const base = h * hd;
        const SimdVec = @Vector(simd_width, f32);
        const sa_v: SimdVec = @splat(sa);
        const sb_v: SimdVec = @splat(sb);
        const inv_v: SimdVec = @splat(inv_sum);
        var d: usize = 0;
        while (d + simd_width <= hd) : (d += simd_width) {
            const a_v: SimdVec = out_a[base + d ..][0..simd_width].*;
            const b_v: SimdVec = out_b[base + d ..][0..simd_width].*;
            merged[base + d ..][0..simd_width].* = @mulAdd(SimdVec, a_v, sa_v, b_v * sb_v) * inv_v;
        }
        while (d < hd) : (d += 1) {
            merged[base + d] = (out_a[base + d] * sa + out_b[base + d] * sb) * inv_sum;
        }
    }
}

test "splitSdpaMerge correctness" {
    // Two splits of a 4-position sequence, 1 head, hd=4
    // Split A: positions 0-1, Split B: positions 2-3
    // Verify merge(A, B) matches manual computation

    const hd: usize = 4;
    const nh: usize = 1;

    // Simulated partial outputs (pre-computed from known Q, K, V)
    var out_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var max_a = [_]f32{0.5}; // max score in split A
    var sum_a = [_]f32{2.0}; // sum(exp) in split A

    var out_b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var max_b = [_]f32{0.8}; // max score in split B
    var sum_b = [_]f32{3.0}; // sum(exp) in split B

    var merged = [_]f32{ 0, 0, 0, 0 };

    splitSdpaMerge(&out_a, &max_a, &sum_a, &out_b, &max_b, &sum_b, &merged, nh, hd);

    // Pre-computed expected values (independent of implementation):
    // max_all = max(0.5, 0.8) = 0.8
    // corr_a = exp(0.5 - 0.8) = exp(-0.3) ≈ 0.74082
    // corr_b = exp(0.8 - 0.8) = 1.0
    // w_a = sum_a * corr_a = 2.0 * 0.74082 = 1.48164
    // w_b = sum_b * corr_b = 3.0
    // sum_all = 1.48164 + 3.0 = 4.48164
    // merged[i] = (out_a[i] * w_a + out_b[i] * w_b) / sum_all
    const expected = [4]f32{ 3.6777, 4.6777, 5.6777, 6.6777 };
    for (0..hd) |i| {
        try std.testing.expectApproxEqAbs(expected[i], merged[i], 1e-3);
    }
}

test "partitionBlocksLocked matches unlocked scan" {
    const allocator = std.testing.allocator;
    var cache = try tiered.TieredKvCache.init(allocator, 1, 2, 2, 2, 0, 16, null);
    defer cache.deinit();
    const b0 = try cache.allocBlock();
    const b1 = try cache.allocBlock();
    const table = [_]u32{ b0, b1 };
    const unlocked = partitionBlocks(&table, cache.blocks, cache.block_size, 32);
    const locked = partitionBlocksLocked(&cache, &table, 32);
    try std.testing.expectEqual(unlocked.gpu_count, locked.gpu_count);
    try std.testing.expectEqual(unlocked.cpu_count, locked.cpu_count);
    try std.testing.expectEqual(unlocked.total_gpu_positions, locked.total_gpu_positions);
    try std.testing.expectEqual(@as(u32, 0), cache.tier_lock.load(.monotonic));
}

test "partitionBlocks mixed tiers" {
    // 4 blocks: [vram, ram, vram, ram], block_size=16, seq_len=60
    var blocks: [4]TieredBlock = undefined;
    blocks[0] = .{ .base = CacheBlock{ .keys = &[_]f32{}, .values = &[_]f32{} }, .tier = .vram };
    blocks[1] = .{ .base = CacheBlock{ .keys = &[_]f32{}, .values = &[_]f32{} }, .tier = .ram };
    blocks[2] = .{ .base = CacheBlock{ .keys = &[_]f32{}, .values = &[_]f32{} }, .tier = .vram };
    blocks[3] = .{ .base = CacheBlock{ .keys = &[_]f32{}, .values = &[_]f32{} }, .tier = .ram };
    const table = [_]u32{ 0, 1, 2, 3 };

    const p = partitionBlocks(&table, &blocks, 16, 60);

    // 2 GPU blocks (0,2), 2 CPU blocks (1,3)
    try std.testing.expectEqual(@as(usize, 2), p.gpu_count);
    try std.testing.expectEqual(@as(usize, 2), p.cpu_count);

    // Block 0: pos 0..16 → GPU
    try std.testing.expectEqual(@as(usize, 0), p.gpu[0].start);
    try std.testing.expectEqual(@as(usize, 16), p.gpu[0].len);

    // Block 1: pos 16..32 → CPU
    try std.testing.expectEqual(@as(usize, 16), p.cpu[0].start);
    try std.testing.expectEqual(@as(usize, 16), p.cpu[0].len);

    // Block 2: pos 32..48 → GPU
    try std.testing.expectEqual(@as(usize, 32), p.gpu[1].start);
    try std.testing.expectEqual(@as(usize, 16), p.gpu[1].len);

    // Block 3: pos 48..60 → CPU (partial: seq_len=60, only 12 positions)
    try std.testing.expectEqual(@as(usize, 48), p.cpu[1].start);
    try std.testing.expectEqual(@as(usize, 12), p.cpu[1].len);

    // Totals
    try std.testing.expectEqual(@as(usize, 32), p.total_gpu_positions);
    try std.testing.expectEqual(@as(usize, 28), p.total_cpu_positions);
}

test "partitionBlocks all gpu" {
    var blocks: [2]TieredBlock = undefined;
    blocks[0] = .{ .base = CacheBlock{ .keys = &[_]f32{}, .values = &[_]f32{} }, .tier = .vram };
    blocks[1] = .{ .base = CacheBlock{ .keys = &[_]f32{}, .values = &[_]f32{} }, .tier = .vram };
    const table = [_]u32{ 0, 1 };

    const p = partitionBlocks(&table, &blocks, 16, 32);
    // Adjacent same-tier blocks coalesce into a single range
    try std.testing.expectEqual(@as(usize, 1), p.gpu_count);
    try std.testing.expectEqual(@as(usize, 0), p.cpu_count);
    try std.testing.expectEqual(@as(usize, 32), p.total_gpu_positions);
    try std.testing.expectEqual(@as(usize, 0), p.total_cpu_positions);
    try std.testing.expectEqual(@as(usize, 0), p.gpu[0].start);
    try std.testing.expectEqual(@as(usize, 32), p.gpu[0].len);
}

test "partitionBlocks all cpu" {
    var blocks: [3]TieredBlock = undefined;
    blocks[0] = .{ .base = CacheBlock{ .keys = &[_]f32{}, .values = &[_]f32{} }, .tier = .ram };
    blocks[1] = .{ .base = CacheBlock{ .keys = &[_]f32{}, .values = &[_]f32{} }, .tier = .ssd };
    blocks[2] = .{ .base = CacheBlock{ .keys = &[_]f32{}, .values = &[_]f32{} }, .tier = .ram };
    const table = [_]u32{ 0, 1, 2 };

    const p = partitionBlocks(&table, &blocks, 8, 20);
    try std.testing.expectEqual(@as(usize, 0), p.gpu_count);
    // ram+ssd+ram: all CPU-tier, adjacent blocks coalesce into one range
    try std.testing.expectEqual(@as(usize, 1), p.cpu_count);
    try std.testing.expectEqual(@as(usize, 0), p.total_gpu_positions);
    try std.testing.expectEqual(@as(usize, 20), p.total_cpu_positions);
    // Single coalesced range: 0..20
    try std.testing.expectEqual(@as(usize, 0), p.cpu[0].start);
    try std.testing.expectEqual(@as(usize, 20), p.cpu[0].len);
}

test "partitionBlocks empty" {
    const table = [_]u32{};
    const p = partitionBlocks(&table, &[_]TieredBlock{}, 16, 0);
    try std.testing.expectEqual(@as(usize, 0), p.gpu_count);
    try std.testing.expectEqual(@as(usize, 0), p.cpu_count);
}

test "splitSdpaMerge equal max scores" {
    // When max_a == max_b, both corrections are 1.0 → simple weighted average
    const hd: usize = 4;
    const nh: usize = 1;

    var out_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var max_a = [_]f32{0.5};
    var sum_a = [_]f32{2.0};

    var out_b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var max_b = [_]f32{0.5}; // equal to max_a
    var sum_b = [_]f32{3.0};

    var merged = [_]f32{ 0, 0, 0, 0 };

    splitSdpaMerge(&out_a, &max_a, &sum_a, &out_b, &max_b, &sum_b, &merged, nh, hd);

    // corr_a = corr_b = exp(0) = 1.0, sum_all = 2.0 + 3.0 = 5.0
    // merged[i] = (out_a[i] * 2.0 + out_b[i] * 3.0) / 5.0
    const expected = [4]f32{ 3.4, 4.4, 5.4, 6.4 };
    for (0..hd) |i| {
        try std.testing.expectApproxEqAbs(expected[i], merged[i], 1e-5);
    }
}

test "splitSdpaMerge multi-head" {
    // 2 heads, hd=2. Verify each head merges independently.
    var out_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 }; // head0: [1,2], head1: [3,4]
    var max_a = [_]f32{ 1.0, 0.5 };
    var sum_a = [_]f32{ 1.0, 2.0 };
    var out_b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var max_b = [_]f32{ 0.5, 1.0 };
    var sum_b = [_]f32{ 2.0, 1.0 };
    var merged = [_]f32{ 0, 0, 0, 0 };

    splitSdpaMerge(&out_a, &max_a, &sum_a, &out_b, &max_b, &sum_b, &merged, 2, 2);

    // Head 0: max_all=1.0, corr_a=1.0, corr_b=exp(-0.5)≈0.6065
    //   w_a=1.0, w_b=2.0*0.6065=1.2131, sum=2.2131
    // Head 1: max_all=1.0, corr_a=exp(-0.5)≈0.6065, corr_b=1.0
    //   w_a=2.0*0.6065=1.2131, w_b=1.0, sum=2.2131
    // Pre-computed expected values:
    // h0d0 = (1*1.0 + 5*1.2131) / 2.2131 ≈ 3.1935
    // h0d1 = (2*1.0 + 6*1.2131) / 2.2131 ≈ 4.1935
    // h1d0 = (3*1.2131 + 7*1.0) / 2.2131 ≈ 4.8065
    // h1d1 = (4*1.2131 + 8*1.0) / 2.2131 ≈ 5.8065
    const expected = [4]f32{ 3.1935, 4.1935, 4.8065, 5.8065 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], merged[i], 1e-3);
    }
    // Verify heads are independent (symmetric inputs but swapped → different results)
    try std.testing.expect(merged[0] != merged[2]);
}

test "splitSdpaMerge one split dominates" {
    // When max_a >> max_b, split A should dominate the merged output.
    // This tests the numerical stability of the exp(max_a - max_b) correction.
    const hd: usize = 4;
    const nh: usize = 1;

    var out_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var max_a = [_]f32{10.0}; // much larger
    var sum_a = [_]f32{1.0};

    var out_b = [_]f32{ 9.0, 9.0, 9.0, 9.0 };
    var max_b = [_]f32{0.0}; // much smaller
    var sum_b = [_]f32{1.0};

    var merged = [_]f32{ 0, 0, 0, 0 };

    splitSdpaMerge(&out_a, &max_a, &sum_a, &out_b, &max_b, &sum_b, &merged, nh, hd);

    // corr_a = exp(10-10) = 1.0, corr_b = exp(0-10) ≈ 4.5e-5
    // Split A dominates: merged ≈ out_a
    for (0..hd) |i| {
        try std.testing.expectApproxEqAbs(out_a[i], merged[i], 0.01);
    }
}

test "fuzz: partitionBlocks and splitSdpaMerge" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            var raw: [16]u8 = undefined;
            smith.bytesWithHash(&raw, 0);

            // -- partitionBlocks: random blocks with mixed tiers --
            const n_blocks: usize = (raw[0] % 8) + 1;
            var block_table: [8]u32 = undefined;
            var tiered_blocks: [8]TieredBlock = undefined;
            for (0..n_blocks) |i| {
                block_table[i] = @intCast(i);
                tiered_blocks[i] = .{
                    .base = undefined,
                    .tier = if (raw[1 + i] % 2 == 0) .vram else .ram,
                };
            }
            const block_size: usize = (raw[9] % 4) + 1;
            const seq_len: usize = n_blocks * block_size;

            const partition = partitionBlocks(
                block_table[0..n_blocks],
                tiered_blocks[0..n_blocks],
                block_size,
                seq_len,
            );

            // Invariant: total positions must equal seq_len.
            try std.testing.expectEqual(seq_len, partition.total_gpu_positions + partition.total_cpu_positions);

            // -- splitSdpaMerge: small random merge --
            const hd: usize = 8;
            const nh: usize = 1;
            var out_a: [hd]f32 = undefined;
            var out_b: [hd]f32 = undefined;
            var merged: [hd]f32 = undefined;
            var a_bytes: [hd * 4]u8 = undefined;
            var b_bytes: [hd * 4]u8 = undefined;
            smith.bytesWithHash(&a_bytes, 1);
            smith.bytesWithHash(&b_bytes, 2);
            out_a = @bitCast(a_bytes);
            out_b = @bitCast(b_bytes);
            for (&out_a) |*v| if (!std.math.isFinite(v.*)) {
                v.* = 0.0;
            };
            for (&out_b) |*v| if (!std.math.isFinite(v.*)) {
                v.* = 0.0;
            };

            // Use small finite max values to avoid exp overflow.
            var max_a = [_]f32{@as(f32, @floatFromInt(@as(i8, @bitCast(raw[10])))) / 32.0};
            var max_b = [_]f32{@as(f32, @floatFromInt(@as(i8, @bitCast(raw[11])))) / 32.0};
            var sum_a = [_]f32{@abs(@as(f32, @floatFromInt(@as(i8, @bitCast(raw[12]))))) / 128.0 + 0.01};
            var sum_b = [_]f32{@abs(@as(f32, @floatFromInt(@as(i8, @bitCast(raw[13]))))) / 128.0 + 0.01};

            splitSdpaMerge(&out_a, &max_a, &sum_a, &out_b, &max_b, &sum_b, &merged, nh, hd);

            // Invariant: merged output must be finite.
            for (merged) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

/// Attention over the whole sequence must equal the merge of attention over
/// two halves, using each half's reported softmax stats. That identity is
/// the whole basis of tiered split attention, and it only holds if
/// `sdpaWithStats` reports the real per-head max and denominator: a backend
/// returning the identity (max=0, sum=1) weights the two halves by their
/// position counts instead of their softmax mass.
fn checkSplitMatchesFull(be: Backend, label: []const u8) !void {
    const nh: usize = 2;
    const nkv: usize = 1;
    const hd: usize = 8;
    const kvd: usize = nkv * hd;
    const seq_len: usize = 7; // positions 0..6 cached, new token lands at 7
    const split_at: usize = 3; // part A covers 0..2, part B covers 3..7

    var keys: [512]f32 = @splat(0);
    var vals: [512]f32 = @splat(0);
    // Spread the keys so the two halves have genuinely different maxima;
    // equal maxima would hide a wrong correction factor.
    for (0..seq_len) |t| {
        for (0..hd) |d| {
            keys[t * kvd + d] = @as(f32, @floatFromInt(t)) * 0.3 - 1.0 + @as(f32, @floatFromInt(d)) * 0.05;
            vals[t * kvd + d] = @as(f32, @floatFromInt(t + 1)) + @as(f32, @floatFromInt(d)) * 0.1;
        }
    }
    var k_new: [kvd]f32 = undefined;
    var v_new: [kvd]f32 = undefined;
    for (0..kvd) |d| {
        k_new[d] = 0.7 + @as(f32, @floatFromInt(d)) * 0.02;
        v_new[d] = 9.0 - @as(f32, @floatFromInt(d)) * 0.1;
    }
    var q: [nh * hd]f32 = undefined;
    for (0..nh * hd) |i| q[i] = 0.1 + @as(f32, @floatFromInt(i)) * 0.03;

    const scale: f32 = 0.35;

    // Full-sequence reference.
    var full_keys = keys;
    var full_vals = vals;
    var full_out: [nh * hd]f32 = @splat(0);
    be.sdpa(
        &q,
        std.mem.sliceAsBytes(&full_keys),
        std.mem.sliceAsBytes(&full_vals),
        &k_new,
        &v_new,
        &full_out,
        nh,
        nkv,
        hd,
        seq_len,
        scale,
        .f32,
        .f32,
    );
    be.sync();

    // Split. Both halves read one cache that already holds the new token,
    // so the append each performs rewrites bytes that are already correct.
    var split_keys = keys;
    var split_vals = vals;
    @memcpy(split_keys[seq_len * kvd ..][0..kvd], &k_new);
    @memcpy(split_vals[seq_len * kvd ..][0..kvd], &v_new);

    // Part A: positions 0..split_at-1. Re-appends the key already at
    // split_at-1, which is a no-op write.
    var a_k: [kvd]f32 = undefined;
    var a_v: [kvd]f32 = undefined;
    @memcpy(&a_k, split_keys[(split_at - 1) * kvd ..][0..kvd]);
    @memcpy(&a_v, split_vals[(split_at - 1) * kvd ..][0..kvd]);
    var out_a: [nh * hd]f32 = @splat(0);
    var max_a: [nh]f32 = @splat(0);
    var sum_a: [nh]f32 = @splat(0);
    be.sdpaWithStats(
        &q,
        std.mem.sliceAsBytes(&split_keys),
        std.mem.sliceAsBytes(&split_vals),
        &a_k,
        &a_v,
        &out_a,
        &max_a,
        &sum_a,
        nh,
        nkv,
        hd,
        split_at - 1,
        scale,
        .f32,
        .f32,
    );
    be.sync();

    // Part B: positions split_at..seq_len, read from a shifted base.
    var out_b: [nh * hd]f32 = @splat(0);
    var max_b: [nh]f32 = @splat(0);
    var sum_b: [nh]f32 = @splat(0);
    be.sdpaWithStats(
        &q,
        std.mem.sliceAsBytes(split_keys[split_at * kvd ..]),
        std.mem.sliceAsBytes(split_vals[split_at * kvd ..]),
        &k_new,
        &v_new,
        &out_b,
        &max_b,
        &sum_b,
        nh,
        nkv,
        hd,
        seq_len - split_at,
        scale,
        .f32,
        .f32,
    );
    be.sync();

    var merged: [nh * hd]f32 = @splat(0);
    splitSdpaMerge(&out_a, &max_a, &sum_a, &out_b, &max_b, &sum_b, &merged, nh, hd);

    for (0..nh * hd) |i| {
        std.testing.expectApproxEqAbs(full_out[i], merged[i], 1e-3) catch |err| {
            std.debug.print("{s}: element {d} full={d:.6} merged={d:.6}\n", .{ label, i, full_out[i], merged[i] });
            return err;
        };
    }
}

test "split attention with real stats reproduces full attention on CPU" {
    const BackendState = @import("../backend/backend.zig").BackendState;
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    var bs = BackendState{};
    bs.init(std.testing.allocator, .cpu, threaded.io(), 0);
    defer if (bs.pool) |*p| p.deinit();
    try checkSplitMatchesFull(bs.be, "cpu");
}

test "split attention with real stats reproduces full attention on the GPU" {
    if (@import("builtin").os.tag != .macos) return error.SkipZigTest;
    const BackendState = @import("../backend/backend.zig").BackendState;
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    var bs = BackendState{};
    bs.init(std.testing.allocator, .metal, threaded.io(), 0);
    defer if (bs.pool) |*p| p.deinit();
    if (bs.be != .metal) return error.SkipZigTest;
    defer bs.metal_be.deinit();
    try checkSplitMatchesFull(bs.be, "metal");
}

test "contiguousPrefixSplit accepts a demoted prefix and refuses interleaving" {
    // A tiered cache demotes oldest-first, so the CPU side is a prefix.
    var p = Partition{};
    p.cpu[0] = .{ .start = 0, .len = 4 };
    p.cpu_count = 1;
    p.total_cpu_positions = 4;
    p.gpu[0] = .{ .start = 4, .len = 5 };
    p.gpu_count = 1;
    p.total_gpu_positions = 5;
    try std.testing.expectEqual(@as(?usize, 4), contiguousPrefixSplit(&p, 9));

    // Several CPU ranges are fine as long as they stay inside the prefix.
    var q = Partition{};
    q.cpu[0] = .{ .start = 0, .len = 2 };
    q.cpu[1] = .{ .start = 2, .len = 2 };
    q.cpu_count = 2;
    q.total_cpu_positions = 4;
    q.gpu[0] = .{ .start = 4, .len = 5 };
    q.gpu_count = 1;
    q.total_gpu_positions = 5;
    try std.testing.expectEqual(@as(?usize, 4), contiguousPrefixSplit(&q, 9));

    // Interleaved: a GPU range sits before the end of the CPU side, so one
    // slice per tier cannot describe it and the exact CPU path must run.
    var r = Partition{};
    r.cpu[0] = .{ .start = 0, .len = 2 };
    r.cpu[1] = .{ .start = 4, .len = 2 };
    r.cpu_count = 2;
    r.total_cpu_positions = 4;
    r.gpu[0] = .{ .start = 2, .len = 2 };
    r.gpu[1] = .{ .start = 6, .len = 3 };
    r.gpu_count = 2;
    r.total_gpu_positions = 5;
    try std.testing.expect(contiguousPrefixSplit(&r, 9) == null);

    // All on one tier is not a split.
    var g = Partition{};
    g.gpu[0] = .{ .start = 0, .len = 9 };
    g.gpu_count = 1;
    g.total_gpu_positions = 9;
    try std.testing.expect(contiguousPrefixSplit(&g, 9) == null);

    // Counts that do not add up to the sequence mean a stale partition.
    var bad = Partition{};
    bad.cpu[0] = .{ .start = 0, .len = 4 };
    bad.cpu_count = 1;
    bad.total_cpu_positions = 4;
    bad.gpu[0] = .{ .start = 4, .len = 2 };
    bad.gpu_count = 1;
    bad.total_gpu_positions = 2;
    try std.testing.expect(contiguousPrefixSplit(&bad, 9) == null);
}

test "mixed-tier split attention equals full attention" {
    // End to end through splitAttention: a partition with a demoted prefix
    // must produce exactly what attending over the whole sequence does.
    // Before the backends reported real softmax stats this path ran a
    // full-sequence CPU SDPA to avoid the merge entirely.
    const nh: usize = 2;
    const nkv: usize = 1;
    const hd: usize = 8;
    const kvd: usize = nkv * hd;
    const seq_len: usize = 9;
    const split: usize = 4;

    var keys: [512]f32 = @splat(0);
    var vals: [512]f32 = @splat(0);
    for (0..seq_len) |t| {
        for (0..hd) |d| {
            keys[t * kvd + d] = @as(f32, @floatFromInt(t)) * 0.25 - 1.0 + @as(f32, @floatFromInt(d)) * 0.04;
            vals[t * kvd + d] = @as(f32, @floatFromInt(t + 1)) * 0.5 + @as(f32, @floatFromInt(d)) * 0.1;
        }
    }
    var k_new: [kvd]f32 = undefined;
    var v_new: [kvd]f32 = undefined;
    for (0..kvd) |d| {
        k_new[d] = 0.6 + @as(f32, @floatFromInt(d)) * 0.03;
        v_new[d] = 4.0 - @as(f32, @floatFromInt(d)) * 0.07;
    }
    var q: [nh * hd]f32 = undefined;
    for (0..nh * hd) |i| q[i] = 0.2 + @as(f32, @floatFromInt(i)) * 0.02;
    const scale: f32 = 0.4;

    const BackendState = @import("../backend/backend.zig").BackendState;
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    var bs = BackendState{};
    bs.init(std.testing.allocator, .cpu, threaded.io(), 0);
    defer if (bs.pool) |*p| p.deinit();

    var full_keys = keys;
    var full_vals = vals;
    var expected: [nh * hd]f32 = @splat(0);
    bs.be.sdpa(&q, std.mem.sliceAsBytes(&full_keys), std.mem.sliceAsBytes(&full_vals), &k_new, &v_new, &expected, nh, nkv, hd, seq_len, scale, .f32, .f32);
    bs.be.sync();

    var part = Partition{};
    part.cpu[0] = .{ .start = 0, .len = split };
    part.cpu_count = 1;
    part.total_cpu_positions = split;
    part.gpu[0] = .{ .start = split, .len = seq_len + 1 - split };
    part.gpu_count = 1;
    part.total_gpu_positions = seq_len + 1 - split;

    var split_keys = keys;
    var split_vals = vals;
    var got: [nh * hd]f32 = @splat(0);
    var gpu_out: [nh * hd]f32 = @splat(0);
    var cpu_out: [nh * hd]f32 = @splat(0);
    splitAttention(
        &q,
        std.mem.sliceAsBytes(&split_keys),
        std.mem.sliceAsBytes(&split_vals),
        &k_new,
        &v_new,
        &got,
        &gpu_out,
        &cpu_out,
        nh,
        nkv,
        hd,
        seq_len,
        scale,
        bs.be,
        .f32,
        .f32,
        &part,
        if (bs.pool) |*p| p else null,
    );

    for (0..nh * hd) |i| {
        try std.testing.expectApproxEqAbs(expected[i], got[i], 1e-4);
    }
}
