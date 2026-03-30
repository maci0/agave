//! FlashAttention-2 prefill kernel for CUDA.
//! Computes causal multi-head attention over n_tok query tokens against
//! both cached KV (prev_len positions) and new KV (n_tok positions).
//!
//! Grid: n_tok * nh blocks, 256 threads per block.
//! Dynamic shared memory layout (all f32):
//!   [0 .. hd)                                 = q_local
//!   [hd .. hd + kv_block_size * hd)           = kv_block
//!   [hd + kv_block_size * hd .. + kv_block_size) = scores
//!   [.. + hd)                                 = out_acc
//!   [.. + 8)                                  = reduce (warp workspace)
//!   [.. + 1)                                  = broadcast slot
//!
//! Uses inline warp-level reductions with a dedicated workspace area
//! (not blockReduceAdd/Max from common.zig, which would clobber q_local).

const cu = @import("common.zig");

/// KV tile size for shared memory. Kept small to fit in 48KB smem budget.
/// Smem for hd=256: (256 + 32*256 + 32 + 256 + 8 + 1) * 4 = ~34.2KB.
const kv_block_size: u32 = 32;

/// Max warps per block (256 threads / 32 = 8).
const max_warps: u32 = 8;

export fn sdpa_prefill_kernel(
    q: [*]const f32, // [n_tok * nh * hd]
    k_cache: [*]const f32, // [capacity * kvd]
    v_cache: [*]const f32, // [capacity * kvd]
    k_new: [*]const f32, // [n_tok * kvd]
    v_new: [*]const f32, // [n_tok * kvd]
    output: [*]f32, // [n_tok * nh * hd]
    nh: u32,
    nkv: u32,
    hd: u32,
    prev_len: u32,
    n_tok: u32,
    scale: f32,
) callconv(.kernel) void {
    const block_id = cu.blockIdx();
    const tok = block_id / nh;
    const h = block_id % nh;
    if (tok >= n_tok or h >= nh) return;

    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const hpg = nh / nkv;
    const kvh = h / hpg;
    const kvd = nkv * hd;
    const sl = prev_len + tok + 1; // Causal: attend to positions 0..sl-1
    const num_blocks = (sl + kv_block_size - 1) / kv_block_size;

    // Shared memory layout (dynamic, allocated via launch parameter)
    const smem = cu.sharedBase();
    const q_off: u32 = 0;
    const kv_off: u32 = hd;
    const sc_off: u32 = hd + kv_block_size * hd;
    const out_off: u32 = sc_off + kv_block_size;
    const reduce_off: u32 = out_off + hd; // 8 slots for warp workspace
    const bcast_off: u32 = reduce_off + max_warps; // 1 slot for broadcast

    // Load Q for this token+head into shared memory
    const q_base = tok * nh * hd + h * hd;
    var d = tid;
    while (d < hd) : (d += bdim) {
        smem[q_off + d] = q[q_base + d];
    }

    // Initialize output accumulator
    d = tid;
    while (d < hd) : (d += bdim) {
        smem[out_off + d] = 0.0;
    }
    cu.syncthreads();

    // Online softmax state
    var m_i: f32 = cu.neg_f32_max;
    var l_i: f32 = 0.0;

    // Warp identity for inline reductions
    const lane = tid % 32;
    const warp_id = tid / 32;
    const n_warps_actual = (bdim + 31) / 32;

    // FlashAttention-2 outer loop over KV blocks
    var block: u32 = 0;
    while (block < num_blocks) : (block += 1) {
        const block_start = block * kv_block_size;
        const block_len: u32 = @min(kv_block_size, sl - block_start);

        // ── Load K tile ─────────────────────────────────────
        var t = tid;
        while (t < block_len) : (t += bdim) {
            const t_global = block_start + t;
            if (t_global < prev_len) {
                const k_base = t_global * kvd + kvh * hd;
                d = 0;
                while (d < hd) : (d += 1) {
                    smem[kv_off + t * hd + d] = k_cache[k_base + d];
                }
            } else {
                const k_base = (t_global - prev_len) * kvd + kvh * hd;
                d = 0;
                while (d < hd) : (d += 1) {
                    smem[kv_off + t * hd + d] = k_new[k_base + d];
                }
            }
        }
        cu.syncthreads();

        // ── QK dot products ─────────────────────────────────
        t = tid;
        while (t < block_len) : (t += bdim) {
            var dot: f32 = 0.0;
            d = 0;
            while (d < hd) : (d += 1) {
                dot += smem[q_off + d] * smem[kv_off + t * hd + d];
            }
            smem[sc_off + t] = dot * scale;
        }
        cu.syncthreads();

        // ── Online softmax: inline block-reduce max ─────────
        // (Cannot use cu.blockReduceMax — it clobbers smem[0..7] = q_local)
        var local_max: f32 = cu.neg_f32_max;
        t = tid;
        while (t < block_len) : (t += bdim) {
            local_max = @max(local_max, smem[sc_off + t]);
        }
        // Warp-level max
        local_max = cu.warpReduceMax(local_max);
        if (lane == 0) smem[reduce_off + warp_id] = local_max;
        cu.syncthreads();
        // Inter-warp max (warp 0 only), then broadcast
        var m_new: f32 = if (tid < n_warps_actual) smem[reduce_off + tid] else cu.neg_f32_max;
        if (warp_id == 0) m_new = cu.warpReduceMax(m_new);
        if (tid == 0) smem[bcast_off] = m_new;
        cu.syncthreads();
        m_new = smem[bcast_off];

        // Rescale previous accumulator
        const m_prev = m_i;
        m_i = @max(m_i, m_new);
        const rescale = cu.expf(m_prev - m_i);
        l_i *= rescale;
        d = tid;
        while (d < hd) : (d += bdim) {
            smem[out_off + d] = smem[out_off + d] * rescale;
        }
        cu.syncthreads();

        // ── Exp scores ──────────────────────────────────────
        t = tid;
        while (t < block_len) : (t += bdim) {
            smem[sc_off + t] = cu.expf(smem[sc_off + t] - m_i);
        }
        cu.syncthreads();

        // ── Sum reduction ───────────────────────────────────
        var local_sum: f32 = 0.0;
        t = tid;
        while (t < block_len) : (t += bdim) {
            local_sum += smem[sc_off + t];
        }
        // Warp-level sum
        local_sum = cu.warpReduceAdd(local_sum);
        if (lane == 0) smem[reduce_off + warp_id] = local_sum;
        cu.syncthreads();
        // Inter-warp sum (warp 0 only), then broadcast
        var block_sum: f32 = if (tid < n_warps_actual) smem[reduce_off + tid] else 0.0;
        if (warp_id == 0) block_sum = cu.warpReduceAdd(block_sum);
        if (tid == 0) smem[bcast_off] = block_sum;
        cu.syncthreads();
        l_i += smem[bcast_off];

        // ── Load V tile ─────────────────────────────────────
        t = tid;
        while (t < block_len) : (t += bdim) {
            const t_global = block_start + t;
            if (t_global < prev_len) {
                const v_base = t_global * kvd + kvh * hd;
                d = 0;
                while (d < hd) : (d += 1) {
                    smem[kv_off + t * hd + d] = v_cache[v_base + d];
                }
            } else {
                const v_base = (t_global - prev_len) * kvd + kvh * hd;
                d = 0;
                while (d < hd) : (d += 1) {
                    smem[kv_off + t * hd + d] = v_new[v_base + d];
                }
            }
        }
        cu.syncthreads();

        // Accumulate: out_acc[d] += sum_t(scores[t] * V[t,d])
        d = tid;
        while (d < hd) : (d += bdim) {
            var acc: f32 = 0.0;
            t = 0;
            while (t < block_len) : (t += 1) {
                acc += smem[sc_off + t] * smem[kv_off + t * hd + d];
            }
            smem[out_off + d] += acc;
        }
        cu.syncthreads();
    }

    // ── Normalize and write output ──────────────────────────
    const inv_l = cu.rcpf(l_i);
    const o_base = tok * nh * hd + h * hd;
    d = tid;
    while (d < hd) : (d += bdim) {
        output[o_base + d] = smem[out_off + d] * inv_l;
    }
}
