// ── True Megakernel: Gemma 3/4 Dense Q8_0 ────────────────────────────────
// Same structure as mega_gemma_q4k.metal but with Q8_0 dequantization.
// For Q8_0 quantized Gemma models.

kernel void megakernel_gemma_q8(
    device const uchar*             weights     [[buffer(0)]],
    device const MegaLayerOffsets*   layer_off   [[buffer(1)]],
    device float*                    kv_keys     [[buffer(2)]],
    device float*                    kv_values   [[buffer(3)]],
    device float*                    hidden      [[buffer(4)]],
    device float*                    scratch     [[buffer(5)]],
    device atomic_uint*              sync_ctrs   [[buffer(6)]],
    constant MegaGemmaParams&        p           [[buffer(7)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Threadgroup scratch for reductions (must be allocated in kernel scope)
    threadgroup float shared[8];

    device float* hidden2    = scratch;
    device float* ff_gate    = scratch + p.n_embd;
    device float* ff_up      = scratch + p.n_embd + p.n_ff;
    device float* qkv_buf    = scratch + p.n_embd + 2 * p.n_ff;
    device float* ss_scratch = scratch + p.n_embd + 2 * p.n_ff +
                               (p.n_head + 2 * p.n_kv) * p.head_dim;
    uint sync_idx = 0;

    for (uint li = 0; li < p.n_layers; li++) {
        device const MegaLayerOffsets& lo = layer_off[li];

        // 1. Pre-attention norm
        device const float* norm_w = (device const float*)(weights + lo.attn_norm);
        mega_rms_norm(hidden, norm_w, hidden2, ss_scratch,
            &sync_ctrs[sync_idx++ % 32], p.n_embd, p.n_tgs, p.rms_eps, shared, tgid, tid, tg_size);

        // 2. Q/K/V projections (Q8_0)
        uint qd = p.n_head * p.head_dim;
        uint kvd = p.n_kv * p.head_dim;
        device float* q_buf = qkv_buf;
        device float* k_buf = qkv_buf + qd;
        device float* v_buf = qkv_buf + qd + kvd;

        mega_gemv_q8(hidden2, (device const block_q8_0*)(weights + lo.attn_q),
            q_buf, qd, p.n_embd, shared, tgid, tid, tg_size);
        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

        mega_gemv_q8(hidden2, (device const block_q8_0*)(weights + lo.attn_k),
            k_buf, kvd, p.n_embd, shared, tgid, tid, tg_size);
        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

        mega_gemv_q8(hidden2, (device const block_q8_0*)(weights + lo.attn_v),
            v_buf, kvd, p.n_embd, shared, tgid, tid, tg_size);
        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

        // 3. RoPE
        mega_rope(q_buf, p.n_head, p.head_dim, p.rope_dim, p.rope_theta, p.seq_pos,
            tgid, tid, tg_size);
        mega_rope(k_buf, p.n_kv, p.head_dim, p.rope_dim, p.rope_theta, p.seq_pos,
            tgid, tid, tg_size);
        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

        // 4. KV cache append + inline SDPA
        uint kv_layer_stride = p.max_seq_len * p.n_kv * p.head_dim;
        device float* layer_keys = kv_keys + li * kv_layer_stride;
        device float* layer_values = kv_values + li * kv_layer_stride;

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

        // 5. Output projection
        mega_gemv_q8(attn_out_buf, (device const block_q8_0*)(weights + lo.attn_output),
            hidden2, p.n_embd, qd, shared, tgid, tid, tg_size);
        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

        // 6. Post-attention norm + residual
        device const float* post_norm = (device const float*)(weights + lo.post_attn_norm);
        mega_add(hidden, hidden2, p.n_embd, tgid, tid, tg_size);
        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

        mega_rms_norm(hidden, post_norm, hidden2, ss_scratch,
            &sync_ctrs[sync_idx++ % 32], p.n_embd, p.n_tgs, p.rms_eps, shared, tgid, tid, tg_size);

        // 7. FFN gate+up (Q8_0) + GELU
        mega_gemv_q8(hidden2, (device const block_q8_0*)(weights + lo.ffn_gate),
            ff_gate, p.n_ff, p.n_embd, shared, tgid, tid, tg_size);
        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

        mega_gemv_q8(hidden2, (device const block_q8_0*)(weights + lo.ffn_up),
            ff_up, p.n_ff, p.n_embd, shared, tgid, tid, tg_size);
        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

        mega_gelu_mul(ff_gate, ff_up, p.n_ff, tgid, tid, tg_size);
        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

        // 8. FFN down projection
        mega_gemv_q8(ff_gate, (device const block_q8_0*)(weights + lo.ffn_down),
            hidden2, p.n_embd, p.n_ff, shared, tgid, tid, tg_size);
        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

        // 9. Post-FFN residual
        mega_add(hidden, hidden2, p.n_embd, tgid, tid, tg_size);
        mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
        mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);
    }
}
