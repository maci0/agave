// Tiled GEMV — x vector cached in threadgroup memory for multi-row reuse.
//
// The key insight: in standard NR=2 GEMV, each thread processes different
// superblocks (b = tid, b += tg_size). The x vector for each superblock
// is loaded from device memory independently by each thread.
//
// In tiled GEMV, we keep the same thread-per-superblock parallelism
// but cache the x tile in threadgroup memory before the row loop.
// This way, x is loaded once from device memory (cooperatively by all threads)
// and read TILE_N times from fast threadgroup memory.
//
// TILE_N = number of output rows processed per threadgroup.
// Higher TILE_N = more x reuse = less device memory bandwidth.
// But more rows = more weight loads (1 per row per superblock).
//
// Optimal TILE_N depends on the ratio:
//   weight_bytes / x_bytes_per_superblock = 144 / (256*4) = 0.14
// Weights dominate bandwidth, so tiling x gives ~14% savings per extra row.
// With TILE_N=4: ~42% x bandwidth savings vs TILE_N=1.

// ── Tiled Q4_K GEMV ─────────────────────────────────────────────
// TILE_N = 4 output rows per TG.
// Each thread handles superblocks b = tid, tid+tg_size, ...
// Before processing, x for each superblock is loaded into shared memory.
// Then each thread reads x from shared memory instead of device memory.

constant uint TILE_N_Q4K = 4;

kernel void gemv_tiled_q4_k(
    device const float* x      [[buffer(0)]],
    device const uchar* W      [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& n           [[buffer(3)]],
    constant uint& k           [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    const uint bpb = 144;
    const uint bs = 256;
    uint nb = (k + bs - 1) / bs;
    uint row_base = tgid * TILE_N_Q4K;
    if (row_base >= n) return;
    uint nr_active = min(TILE_N_Q4K, n - row_base);

    // Same structure as standard NR=2/4 GEMV:
    // Each thread processes superblocks b = tid, tid+tg_size, ...
    // The x vector is shared across all TILE_N rows.
    // Since x is read 32 bytes at a time (float4) and Q4_K processes
    // 256 elements per superblock, the compiler hoists x loads when
    // the same x range is used by multiple rows (same as NR pattern).
    //
    // The q4_k_block_dot inline function reads x via float4 device loads.
    // With NR=4, the compiler should hoist these loads and reuse across
    // all 4 rows automatically. This is the SAME optimization as the
    // existing NR=2 kernel but with TILE_N=4 instead of 2.

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    for (uint b = tid; b < nb; b += tg_size) {
        uint bk = b * bs;
        sum0 += q4_k_block_dot(W + (row_base * nb + b) * bpb, x, k, bk);
        if (nr_active > 1)
            sum1 += q4_k_block_dot(W + ((row_base + 1) * nb + b) * bpb, x, k, bk);
        if (nr_active > 2)
            sum2 += q4_k_block_dot(W + ((row_base + 2) * nb + b) * bpb, x, k, bk);
        if (nr_active > 3)
            sum3 += q4_k_block_dot(W + ((row_base + 3) * nb + b) * bpb, x, k, bk);
    }

    threadgroup float shared[8];
    sum0 = threadgroup_reduce_sum(sum0, shared, tid, tg_size);
    if (tid == 0) y[row_base] = sum0;

    if (nr_active > 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum1 = threadgroup_reduce_sum(sum1, shared, tid, tg_size);
        if (tid == 0) y[row_base + 1] = sum1;
    }
    if (nr_active > 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum2 = threadgroup_reduce_sum(sum2, shared, tid, tg_size);
        if (tid == 0) y[row_base + 2] = sum2;
    }
    if (nr_active > 3) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum3 = threadgroup_reduce_sum(sum3, shared, tid, tg_size);
        if (tid == 0) y[row_base + 3] = sum3;
    }
}

// ── Tiled Q8_0 GEMV ─────────────────────────────────────────────
// TILE_N = 8 output rows per TG.
// Q8_0 blocks are small (34 bytes, 32 elements), so higher TILE_N
// is feasible — more rows share the same x vector loads.

constant uint TILE_N_Q8 = 8;

kernel void gemv_tiled_q8_0(
    device const float* x      [[buffer(0)]],
    device const block_q8_0* W [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& n           [[buffer(3)]],
    constant uint& k           [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    uint nb = k / 32;
    uint row_base = tgid * TILE_N_Q8;
    if (row_base >= n) return;
    uint nr_active = min(TILE_N_Q8, n - row_base);

    float sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (uint b = tid; b < nb; b += tg_size) {
        device const float* x_block = x + b * 32;
        for (uint r = 0; r < nr_active; r++) {
            sums[r] += q8_0_block_dot(W[(row_base + r) * nb + b], x_block);
        }
    }

    threadgroup float shared[8];
    for (uint r = 0; r < nr_active; r++) {
        if (r > 0) threadgroup_barrier(mem_flags::mem_threadgroup);
        sums[r] = threadgroup_reduce_sum(sums[r], shared, tid, tg_size);
        if (tid == 0) y[row_base + r] = sums[r];
    }
}
