//! Tree-Masked SDPA kernel for CUDA — FlashAttention-2 with ancestor bitmask.
//! One block per (node, head) pair. Threads cooperate on KV block processing.
//! Used during DDTree speculative decoding batch verification.

const cu = @import("common.zig");

const tree_block_size: u32 = 16;

export fn sdpa_tree_kernel(
    q_all: [*]const f32,
    prefix_k: [*]const f32,
    prefix_v: [*]const f32,
    tree_k: [*]const f32,
    tree_v: [*]const f32,
    output: [*]f32,
    ancestor_masks: [*]const u64,
    nh: u32,
    nkv: u32,
    hd: u32,
    prefix_len: u32,
    n_nodes: u32,
    scale: f32,
) callconv(.kernel) void {
    const flat_id = cu.blockIdx();
    const tid = cu.threadIdx();
    const tg_sz = cu.blockDim();

    const node_i = flat_id / nh;
    const h = flat_id % nh;
    if (node_i >= n_nodes or h >= nh) return;

    const hpg = nh / nkv;
    const kvh = h / hpg;
    const kvd = nkv * hd;

    // Load Q for this node/head into registers
    const q_base = node_i * nh * hd + h * hd;

    var m_i: f32 = -3.402823e+38;
    var l_i: f32 = 0.0;

    // Output accumulator in registers (per thread, strided)
    var out_acc: [8]f32 = .{0} ** 8;
    const max_out_per_thread: u32 = 8;

    // Phase 1: Prefix blocks (unconditional)
    const prefix_blocks = (prefix_len + tree_block_size - 1) / tree_block_size;
    var block: u32 = 0;
    while (block < prefix_blocks) : (block += 1) {
        const block_start = block * tree_block_size;
        const block_len = @min(tree_block_size, prefix_len - block_start);

        // Compute scores for this block
        var t: u32 = tid;
        var block_max: f32 = -3.402823e+38;
        var score_buf: [16]f32 = undefined;

        while (t < block_len) : (t += tg_sz) {
            const t_global = block_start + t;
            const k_base = t_global * kvd + kvh * hd;
            var dot_val: f32 = 0.0;
            var d: u32 = 0;
            while (d < hd) : (d += 1) {
                dot_val += q_all[q_base + d] * prefix_k[k_base + d];
            }
            score_buf[t] = dot_val * scale;
            block_max = @max(block_max, score_buf[t]);
        }

        block_max = cu.blockReduceMax(block_max);

        // Rescale
        const m_prev = m_i;
        m_i = @max(m_i, block_max);
        const rescale = @exp(m_prev - m_i);
        l_i *= rescale;
        var oi: u32 = 0;
        while (oi < max_out_per_thread) : (oi += 1) out_acc[oi] *= rescale;

        // Exp + sum
        var block_sum: f32 = 0.0;
        t = tid;
        while (t < block_len) : (t += tg_sz) {
            const w = @exp(score_buf[t] - m_i);
            score_buf[t] = w;
            block_sum += w;
        }
        block_sum = cu.blockReduceAdd(block_sum);
        l_i += block_sum;

        // V accumulate
        t = 0;
        while (t < block_len) : (t += 1) {
            const w = score_buf[t];
            if (w < 1e-6) continue;
            const t_global = block_start + t;
            const v_base = t_global * kvd + kvh * hd;
            var d2: u32 = tid;
            var out_i: u32 = 0;
            while (d2 < hd and out_i < max_out_per_thread) : ({
                d2 += tg_sz;
                out_i += 1;
            }) {
                out_acc[out_i] += w * prefix_v[v_base + d2];
            }
        }
    }

    // Phase 2: Tree nodes (masked by ancestor bitmask)
    var j: u32 = 0;
    while (j < n_nodes) : (j += 1) {
        const mask_word = ancestor_masks[node_i * 8 + j / 64];
        if ((mask_word & (@as(u64, 1) << @as(u6, @intCast(j % 64)))) == 0) continue;

        // Score
        var dot_val: f32 = 0.0;
        if (tid == 0) {
            var d: u32 = 0;
            while (d < hd) : (d += 1) {
                dot_val += q_all[q_base + d] * tree_k[j * kvd + kvh * hd + d];
            }
        }
        const tree_score = dot_val * scale;

        // Online softmax update
        const m_prev = m_i;
        m_i = @max(m_i, tree_score);
        const rescale = @exp(m_prev - m_i);
        l_i *= rescale;
        var oi2: u32 = 0;
        while (oi2 < max_out_per_thread) : (oi2 += 1) out_acc[oi2] *= rescale;

        const w = @exp(tree_score - m_i);
        l_i += w;

        // V accumulate
        var d3: u32 = tid;
        var out_i2: u32 = 0;
        while (d3 < hd and out_i2 < max_out_per_thread) : ({
            d3 += tg_sz;
            out_i2 += 1;
        }) {
            out_acc[out_i2] += w * tree_v[j * kvd + kvh * hd + d3];
        }
    }

    // Normalize and write output
    const inv_l = if (l_i > 0.0) 1.0 / l_i else 0.0;
    const out_base = node_i * nh * hd + h * hd;
    var d4: u32 = tid;
    var out_i3: u32 = 0;
    while (d4 < hd and out_i3 < max_out_per_thread) : ({
        d4 += tg_sz;
        out_i3 += 1;
    }) {
        output[out_base + d4] = out_acc[out_i3] * inv_l;
    }
}
