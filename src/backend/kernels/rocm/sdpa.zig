//! Scaled Dot-Product Attention kernel (fused QK·softmax·V).
//! Grid: nh workgroups (one per query head), 256 threads per workgroup.
//! LDS: sl floats for attention scores.

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

    // ── Phase 1: QK dot products → scores in LDS ──────
    var t = tid;
    while (t < sl) : (t += bdim) {
        const k_off = t * kvd + kvh * hd;
        var dot: f32 = 0.0;
        var d: u32 = 0;
        while (d < hd) : (d += 1) {
            dot += q[q_base + d] * keys[k_off + d];
        }
        cu.sharedStore(t, dot * scale);
    }
    cu.syncthreads();

    // ── Phase 2: Softmax (thread-0 serial for correctness) ──────
    if (tid == 0) {
        // Find max
        var max_val: f32 = cu.sharedLoad(0);
        var i: u32 = 1;
        while (i < sl) : (i += 1) {
            const s = cu.sharedLoad(i);
            if (s > max_val) max_val = s;
        }
        // Exp + sum
        var sum_val: f32 = 0.0;
        i = 0;
        while (i < sl) : (i += 1) {
            const e = cu.expf(cu.sharedLoad(i) - max_val);
            cu.sharedStore(i, e);
            sum_val += e;
        }
        // Normalize
        const inv = cu.rcpf(sum_val);
        i = 0;
        while (i < sl) : (i += 1) {
            cu.sharedStore(i, cu.sharedLoad(i) * inv);
        }
    }
    cu.syncthreads();

    // ── Phase 3: V accumulation ─────────────────────────────────
    var d: u32 = tid;
    while (d < hd) : (d += bdim) {
        var acc: f32 = 0.0;
        var tt: u32 = 0;
        while (tt < sl) : (tt += 1) {
            acc += cu.sharedLoad(tt) * values[tt * kvd + kvh * hd + d];
        }
        output[q_base + d] = acc;
    }
}
