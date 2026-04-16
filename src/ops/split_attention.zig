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
//!      - Mixed → GPU SDPA with stats (deferred) + CPU SDPA on thread pool
//!        (concurrent on UMA), then `splitSdpaMerge()` to combine.
//!   3. The merge uses FlashAttention-2 online softmax correction — exact, no
//!      approximation.

const std = @import("std");
const tiered = @import("../kvcache/tiered.zig");
const TieredBlock = tiered.TieredBlock;
const Backend = @import("../backend/backend.zig").Backend;
const kv_quant = @import("kv_quant.zig");
const KvQuantType = kv_quant.KvQuantType;
const ThreadPool = @import("../thread_pool.zig").ThreadPool;
const sdpa_cpu = @import("../backend/backend.zig").CpuSdpa;

/// SIMD vector width (number of f32 lanes) for merge accumulation loops.
const simd_width: usize = 8;

/// Maximum number of contiguous ranges in a partition.
/// Bounded by max blocks per sequence (256 blocks × block_size positions each).
const max_ranges: usize = 256;

/// Maximum number of query heads supported for stack-allocated stats buffers.
const max_heads: usize = 256;

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
            // .ram and .ssd both go to CPU path — coalesce if contiguous
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
///   - partition: Block tier partition (from partitionBlocks).
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
    partition: Partition,
    pool: ?*ThreadPool,
) void {
    // Fast path: all GPU — delegate directly, zero overhead
    if (partition.cpu_count == 0) {
        be.sdpa(q, kv_keys, kv_values, k_new, v_new, output, nh, nkv, hd, seq_len, scale, kv_type_k, kv_type_v);
        return;
    }

    // Fast path: all CPU — sync pending GPU ops, run CPU SDPA (parallel if pool available)
    if (partition.gpu_count == 0) {
        be.sync();
        const kvd = nkv * hd;
        const k_off = kv_quant.kvByteOffset(kv_type_k, seq_len * kvd);
        const v_off = kv_quant.kvByteOffset(kv_type_v, seq_len * kvd);
        kv_quant.kvStore(kv_keys.ptr + k_off, k_new, kvd, kv_type_k);
        kv_quant.kvStore(kv_values.ptr + v_off, v_new, kvd, kv_type_v);
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
        return;
    }

    // ── Split path: concurrent GPU + CPU ──────────────────────────
    //
    // Dispatch order (optimized for UMA overlap):
    //   1. KV append (CPU-side for both tiers)
    //   2. Launch GPU SDPA via sdpaWithStats (deferred on Metal)
    //   3. Run CPU SDPA via thread pool (blocks until done; GPU runs concurrently)
    //   4. be.sync() — wait for GPU command buffer
    //   5. Merge partial outputs via online softmax correction

    std.debug.assert(nh <= max_heads);

    // Stack-allocated per-head softmax stats (max 256 heads * 4 bytes = 1KB each)
    var gpu_max: [max_heads]f32 = undefined;
    var gpu_sum: [max_heads]f32 = undefined;
    var cpu_max: [max_heads]f32 = undefined;
    var cpu_sum: [max_heads]f32 = undefined;

    // Step 1: GPU SDPA with stats (deferred dispatch on Metal/UMA)
    be.sdpaWithStats(q, kv_keys, kv_values, k_new, v_new, gpu_out, &gpu_max, &gpu_sum, nh, nkv, hd, seq_len, scale, kv_type_k, kv_type_v);

    // Step 2: CPU SDPA on thread pool (runs concurrently with GPU command buffer)
    var cpu_job = CpuSdpaJob{
        .q = q,
        .keys = kv_keys.ptr,
        .values = kv_values.ptr,
        .output = cpu_out,
        .head_max = &cpu_max,
        .head_sum = &cpu_sum,
        .nh = nh,
        .nkv = nkv,
        .hd = hd,
        .sl = seq_len + 1,
        .scale = scale,
        .kv_type_k = kv_type_k,
        .kv_type_v = kv_type_v,
    };

    if (pool) |p| {
        p.parallelFor(nh, 1, @ptrCast(&cpu_job), CpuSdpaJob.work);
    } else {
        // No thread pool: run CPU SDPA inline
        CpuSdpaJob.work(@ptrCast(&cpu_job), 0, nh);
    }

    // Step 3: Wait for GPU
    be.sync();

    // Step 4: Merge GPU and CPU partial outputs
    splitSdpaMerge(gpu_out, &gpu_max, &gpu_sum, cpu_out, &cpu_max, &cpu_sum, output, nh, hd);
}

/// Merge two partial SDPA outputs using online softmax correction.
/// Each split produced its own local softmax (out, max_per_head, sum_per_head).
/// This function combines them into the exact result of full SDPA over all positions.
///
/// The merge is exact — no approximation. Given two splits A and B with
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

test "partitionBlocks mixed tiers" {
    // 4 blocks: [vram, ram, vram, ram], block_size=16, seq_len=60
    const CacheBlock = @import("../kvcache/manager.zig").CacheBlock;
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
    const CacheBlock = @import("../kvcache/manager.zig").CacheBlock;
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
    const CacheBlock = @import("../kvcache/manager.zig").CacheBlock;
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
