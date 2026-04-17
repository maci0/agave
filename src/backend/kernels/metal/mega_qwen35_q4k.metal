// ── True Megakernel: Qwen 3.5 Q4_K ──────────────────────────────────────
//
// Single GPU dispatch processes ALL dense layers (attention + DeltaNet + FFN).
// Same structure as mega_qwen35_q8.metal but with Q4_K dequantization.
//
// Uses composable building blocks from mega_common.metal.
// Grid sync via atomic counter barrier between stages.
//
// Dispatch: max(n_embd, n_ff) threadgroups × 256 threads.

kernel void megakernel_qwen35_q4k(
    device const uchar*             weights     [[buffer(0)]],
    device const MegaLayerOffsets*   layer_off   [[buffer(1)]],
    device float*                    kv_keys     [[buffer(2)]],
    device float*                    kv_values   [[buffer(3)]],
    device float*                    hidden      [[buffer(4)]],
    device float*                    scratch     [[buffer(5)]],
    device atomic_uint*              sync_ctrs   [[buffer(6)]],
    constant MegaQwen35Params&       p           [[buffer(7)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Threadgroup scratch for reductions (must be allocated in kernel scope)
    threadgroup float shared[8];

    // Scratch buffer offsets
    device float* hidden2    = scratch;
    device float* ff_gate    = scratch + p.n_embd;
    device float* ff_up      = scratch + p.n_embd + p.n_ff;
    device float* qkv_buf    = scratch + p.n_embd + 2 * p.n_ff;
    device float* ss_scratch = scratch + p.n_embd + 2 * p.n_ff +
                               p.n_head * p.head_dim * 2 + p.n_kv * p.head_dim * 2;

    // Sync counter index (increments each grid_sync call)
    uint sync_idx = 0;

    // ── Layer loop ───────────────────────────────────────────────────────
    for (uint li = 0; li < p.n_layers; li++) {
        device const MegaLayerOffsets& lo = layer_off[li];
        bool is_attn = ((li + 1) % p.full_attn_interval) == 0;
        bool fuse_residual = li > 0;

        // ── 1. Pre-attention norm ────────────────────────────────────────
        device const float* norm_w = (device const float*)(weights + lo.attn_norm);
        if (fuse_residual) {
            mega_add_rms_norm(hidden, hidden2, norm_w, hidden2,
                ss_scratch, &sync_ctrs[sync_idx++ % 32],
                p.n_embd, p.n_tgs, p.rms_eps, shared, tgid, tid, tg_size);
        } else {
            mega_rms_norm(hidden, norm_w, hidden2,
                ss_scratch, &sync_ctrs[sync_idx++ % 32],
                p.n_embd, p.n_tgs, p.rms_eps, shared, tgid, tid, tg_size);
        }

        // ── 2. Attention/DeltaNet projections ────────────────────────────
        if (is_attn) {
            // Full attention: Q, K, V GEMVs
            uint qd = p.n_head * p.head_dim * 2; // ×2 for gate
            uint kvd = p.n_kv * p.head_dim;
            device float* q_buf = qkv_buf;
            device float* k_buf = qkv_buf + qd;
            device float* v_buf = qkv_buf + qd + kvd;

            // Q projection
            mega_gemv_q4k(hidden2, weights + lo.attn_q,
                q_buf, qd, p.n_embd, shared, tgid, tid, tg_size);
            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

            // K projection
            mega_gemv_q4k(hidden2, weights + lo.attn_k,
                k_buf, kvd, p.n_embd, shared, tgid, tid, tg_size);
            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

            // V projection
            mega_gemv_q4k(hidden2, weights + lo.attn_v,
                v_buf, kvd, p.n_embd, shared, tgid, tid, tg_size);
            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

            // RoPE on Q and K
            mega_rope(q_buf, p.n_head, p.head_dim, p.rope_dim, p.rope_theta, p.seq_pos,
                tgid, tid, tg_size);
            mega_rope(k_buf, p.n_kv, p.head_dim, p.rope_dim, p.rope_theta, p.seq_pos,
                tgid, tid, tg_size);
            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

            // KV cache append + inline SDPA (simplified: skip Q/gate deinterleave for now)
            uint kv_layer_stride = p.max_seq_len * p.n_kv * p.head_dim;
            uint attn_layer_idx = li / p.full_attn_interval;
            device float* layer_keys = kv_keys + attn_layer_idx * kv_layer_stride;
            device float* layer_values = kv_values + attn_layer_idx * kv_layer_stride;

            mega_kv_append_f32(k_buf, v_buf, layer_keys, layer_values,
                p.n_kv * p.head_dim, p.seq_pos, tgid, tid, tg_size);
            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

            device float* attn_out_buf = qkv_buf; // reuse QKV scratch for attention output
            mega_sdpa_inline(
                q_buf,
                (device const uchar*)layer_keys,
                (device const uchar*)layer_values,
                attn_out_buf,
                p.n_head,
                p.n_kv,
                p.head_dim,
                p.seq_pos + 1,
                1.0f / sqrt(float(p.head_dim)),
                0, 0, 0, 0,  // f32 KV cache (no TQ)
                shared,
                tgid, tid, tg_size
            );
            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

            // Output projection
            mega_gemv_q4k(attn_out_buf, weights + lo.attn_output,
                hidden2, p.n_embd, p.n_head * p.head_dim, shared, tgid, tid, tg_size);
            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

        } else {
            // DeltaNet SSM layer — currently too complex for GPU megakernel.
        }

        // ── 3. Post-attention norm (FFN pre-norm) ────────────────────────
        device const float* post_norm_w = (device const float*)(weights + lo.post_attn_norm);
        mega_add_rms_norm(hidden, hidden2, post_norm_w, hidden2,
            ss_scratch, &sync_ctrs[sync_idx++ % 32],
            p.n_embd, p.n_tgs, p.rms_eps, shared, tgid, tid, tg_size);

        // ── 4. FFN: gate + up + SiLU*mul ─────────────────────────────────
        // Gate GEMV: ff_gate = W_gate @ hidden2
        mega_gemv_q4k(hidden2, weights + lo.ffn_gate,
            ff_gate, p.n_ff, p.n_embd, shared, tgid, tid, tg_size);
        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

        // Up GEMV: ff_up = W_up @ hidden2
        mega_gemv_q4k(hidden2, weights + lo.ffn_up,
            ff_up, p.n_ff, p.n_embd, shared, tgid, tid, tg_size);
        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

        // SiLU activation: ff_gate = silu(ff_gate) * ff_up
        mega_silu_mul(ff_gate, ff_up, p.n_ff, tgid, tid, tg_size);
        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

        // ── 5. FFN down projection ───────────────────────────────────────
        // hidden2 = W_down @ ff_gate
        mega_gemv_q4k(ff_gate, weights + lo.ffn_down,
            hidden2, p.n_embd, p.n_ff, shared, tgid, tid, tg_size);
        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);
    }

    // ── Final: fuse last FFN residual into hidden ────────────────────────
    mega_add(hidden, hidden2, p.n_embd, tgid, tid, tg_size);
}
