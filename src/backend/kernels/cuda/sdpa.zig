//! Scaled Dot-Product Attention kernel (fused QK·softmax·V).
//! Grid: nh blocks (one per query head), 256 threads per block.
//! Dynamic shared memory: sl floats for attention scores.
//!
//! Softmax is warp-parallel (distributes seq_len across 32 threads),
//! avoiding blockReduce hangs on Blackwell (sm_121) by using warp-only
//! reductions instead of shared memory block reductions. This provides
//! 32× parallelism over serial softmax for long sequences.

const cu = @import("common.zig");

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
    var local_max: f32 = cu.neg_f32_max;
    var i = start;
    while (i < end) : (i += 1) {
        local_max = @max(local_max, smem[i]);
    }
    // All 32 threads participate in warp reduction
    const max_val = cu.warpReduceMax(local_max);

    // Phase 2b: Warp-parallel exp and sum
    var local_sum: f32 = 0.0;
    i = start;
    while (i < end) : (i += 1) {
        const e = cu.expf(smem[i] - max_val);
        smem[i] = e;
        local_sum += e;
    }
    const sum_val = cu.warpReduceAdd(local_sum);

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
            acc += smem[tt] * values[tt * kvd + kvh * hd + d];
        }
        output[q_base + d] = acc;
    }
}
