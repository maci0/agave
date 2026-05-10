//! Scaled Dot-Product Attention kernel (fused QK·softmax·V).
//! Grid: nh blocks (one per query head), 256 threads per block.
//! Dynamic shared memory: sl floats for attention scores.
//!
//! Softmax is warp-parallel (distributes seq_len across 32 threads),
//! avoiding blockReduce hangs on Blackwell (sm_121) by using warp-only
//! reductions instead of shared memory block reductions. This provides
//! 32× parallelism over serial softmax for long sequences.

const cu = @import("common.zig");

/// Sparse V threshold: skip V positions with negligible softmax weight.
/// At 1e-6, skipped positions contribute < 0.0001% to the output.
const sparse_v_threshold: f32 = 1e-6;

export fn sdpa_kernel(
    q: [*]const f32,
    keys: [*]const f32,
    values: [*]const f32,
    output: [*]f32,
    nh: u32,
    nkv: u32,
    hd: u32,
    sl: u32,
    kvd: u32,
    scale: f32,
) callconv(.kernel) void {
    const tid = cu.threadIdx();
    const head = cu.blockIdx();
    const bdim = cu.blockDim();
    const hpg = nh / nkv;
    const kvh = head / hpg;
    const q_base = head * hd;
    const smem = cu.sharedBase();

    // ── Phase 1: QK dot products → scores in shared memory ──────
    var t = tid;
    while (t < sl) : (t += bdim) {
        const k_off = t * kvd + kvh * hd;
        var dot: f32 = 0.0;
        var d: u32 = 0;
        while (d < hd) : (d += 1) {
            dot += q[q_base + d] * keys[k_off + d];
        }
        smem[t] = dot * scale;
    }
    cu.syncthreads();

    // ── Phase 2: Warp-parallel softmax ──────────────────────────
    // Distribute seq_len across 32 threads (assumes blockDim >= 32).
    // Each thread processes a chunk of scores, then warp-reduces max/sum.
    const warp_size = 32;
    const chunk = (sl + warp_size - 1) / warp_size;
    const start = tid * chunk;
    const end = @min(start + chunk, sl);

    // Phase 2a: Warp-parallel max reduction
    // warpReduceMax returns final result only in lane 0 — broadcast via shared memory.
    var local_max: f32 = cu.neg_f32_max;
    var i = start;
    while (i < end) : (i += 1) {
        local_max = @max(local_max, smem[i]);
    }
    var max_val = cu.warpReduceMax(local_max);
    if (tid == 0) cu.sharedStore(sl, max_val); // Reuse smem slot beyond scores
    cu.syncthreads();
    max_val = cu.sharedLoad(sl);

    // Phase 2b: Warp-parallel exp and sum
    // warpReduceAdd returns final result only in lane 0 — broadcast via shared memory.
    var local_sum: f32 = 0.0;
    i = start;
    while (i < end) : (i += 1) {
        const e = cu.expf(smem[i] - max_val);
        smem[i] = e;
        local_sum += e;
    }
    var sum_val = cu.warpReduceAdd(local_sum);
    if (tid == 0) cu.sharedStore(sl, sum_val);
    cu.syncthreads();
    sum_val = cu.sharedLoad(sl);

    // Phase 2c: Warp-parallel normalization
    const inv = cu.rcpf(sum_val);
    i = start;
    while (i < end) : (i += 1) {
        smem[i] = smem[i] * inv;
    }

    cu.syncthreads();

    // ── Phase 3: V accumulation ─────────────────────────────────
    var d: u32 = tid;
    while (d < hd) : (d += bdim) {
        var acc: f32 = 0.0;
        var tt: u32 = 0;
        while (tt < sl) : (tt += 1) {
            const score = smem[tt];
            if (score < sparse_v_threshold) continue; // Sparse V: skip negligible positions
            acc += score * values[tt * kvd + kvh * hd + d];
        }
        output[q_base + d] = acc;
    }
}

/// Paged SDPA: block-table-indexed attention.
/// K_flat/V_flat store all physical blocks concatenated.
/// block_table maps logical block → physical block ID.
export fn sdpa_paged_kernel(
    q: [*]const f32,
    k_flat: [*]const f32,
    v_flat: [*]const f32,
    output: [*]f32,
    block_table: [*]const u32,
    nh: u32,
    nkv: u32,
    hd: u32,
    sl: u32,
    kvd: u32,
    scale: f32,
    paged_bs: u32,
) callconv(.kernel) void {
    const tid = cu.threadIdx();
    const head = cu.blockIdx();
    const bdim = cu.blockDim();
    const hpg = nh / nkv;
    const kvh = head / hpg;
    const q_base = head * hd;
    const smem = cu.sharedBase();

    // Phase 1: QK dot products with block-table indirection
    var t = tid;
    while (t < sl) : (t += bdim) {
        const bt_idx = t / paged_bs;
        const pos_in_bt = t % paged_bs;
        const phys_id = block_table[bt_idx];
        const k_off = (phys_id * paged_bs + pos_in_bt) * kvd + kvh * hd;
        var dot: f32 = 0.0;
        var d: u32 = 0;
        while (d < hd) : (d += 1) {
            dot += q[q_base + d] * k_flat[k_off + d];
        }
        smem[t] = dot * scale;
    }
    cu.syncthreads();

    // Phase 2: Warp-parallel softmax (identical to non-paged)
    const warp_size = 32;
    const chunk = (sl + warp_size - 1) / warp_size;
    const start = tid * chunk;
    const end = @min(start + chunk, sl);

    var local_max: f32 = cu.neg_f32_max;
    var i = start;
    while (i < end) : (i += 1) local_max = @max(local_max, smem[i]);
    var max_val = cu.warpReduceMax(local_max);
    if (tid == 0) cu.sharedStore(sl, max_val);
    cu.syncthreads();
    max_val = cu.sharedLoad(sl);

    var local_sum: f32 = 0.0;
    i = start;
    while (i < end) : (i += 1) {
        const e = cu.expf(smem[i] - max_val);
        smem[i] = e;
        local_sum += e;
    }
    var sum_val = cu.warpReduceAdd(local_sum);
    if (tid == 0) cu.sharedStore(sl, sum_val);
    cu.syncthreads();
    sum_val = cu.sharedLoad(sl);

    const inv = cu.rcpf(sum_val);
    i = start;
    while (i < end) : (i += 1) smem[i] = smem[i] * inv;
    cu.syncthreads();

    // Phase 3: V accumulation with block-table indirection
    var d: u32 = tid;
    while (d < hd) : (d += bdim) {
        var acc: f32 = 0.0;
        var tt: u32 = 0;
        while (tt < sl) : (tt += 1) {
            const score = smem[tt];
            if (score < sparse_v_threshold) continue;
            const bt_idx = tt / paged_bs;
            const pos_in_bt = tt % paged_bs;
            const phys_id = block_table[bt_idx];
            acc += score * v_flat[(phys_id * paged_bs + pos_in_bt) * kvd + kvh * hd + d];
        }
        output[q_base + d] = acc;
    }
}
