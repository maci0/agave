// ── True Megakernel: Nemotron-H Q8_0 (Attention + FFN-only layers) ────────
//
// Nemotron-H has THREE layer types:
//   0 = SSM (Mamba): sequential recurrence, breaks out of megakernel
//   1 = Attention: standard Q/K/V → RoPE → SDPA → output
//   2 = FFN-only: squared-ReLU MLP (no gate tensor, single up+down)
//
// The megakernel processes attention and FFN-only layers inline.
// SSM layers break out — the host dispatches standard SSM kernels for those.
//
// All layer types share a pre-norm (attn_norm) and post-norm (post_attn_norm).
// FFN uses SiLU*mul for attention layers, ReLU² for FFN-only layers.
//
// Dispatch: max(n_embd, n_ff, n_ff_ffn_only) threadgroups × 256 threads.
//
// Buffer layout:
//   0: weights_base [*]const u8 — all model weights (mmap'd GGUF)
//   1: layer_offsets [n_layers × 20 × u64] — per-layer byte offsets
//   2: kv_keys [n_attn_layers × max_seq × kv_dim × sizeof(f32)]
//   3: kv_values [n_attn_layers × max_seq × kv_dim × sizeof(f32)]
//   4: hidden [n_embd] f32 — input/output hidden state
//   5: scratch [scratch_size] f32 — intermediate buffers
//   6: sync_counters [32] atomic_uint — grid sync barriers
//   7: params — struct with model dimensions

struct MegaNemotronHParams {
    uint n_layers;
    uint n_embd;
    uint n_head;
    uint n_kv;
    uint head_dim;
    uint n_ff;            // FFN size for attention layers (SiLU*mul gated)
    uint n_ff_ffn_only;   // FFN size for FFN-only layers (ReLU², no gate)
    uint rope_dim;
    float rope_theta;
    float rms_eps;
    uint max_seq_len;
    uint seq_pos;
    uint n_tgs;
};

// Layer type constants (must match nemotron_h.zig LayerType enum)
constant uint LAYER_TYPE_SSM       = 0;
constant uint LAYER_TYPE_ATTENTION = 1;
constant uint LAYER_TYPE_FFN_ONLY  = 2;

kernel void megakernel_nemotron_h_q8(
    device const uchar*             weights     [[buffer(0)]],
    device const MegaLayerOffsets*   layer_off   [[buffer(1)]],
    device float*                    kv_keys     [[buffer(2)]],
    device float*                    kv_values   [[buffer(3)]],
    device float*                    hidden      [[buffer(4)]],
    device float*                    scratch     [[buffer(5)]],
    device atomic_uint*              sync_ctrs   [[buffer(6)]],
    constant MegaNemotronHParams&    p           [[buffer(7)]],
    device const uint*               layer_types [[buffer(8)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Threadgroup scratch for reductions (must be allocated in kernel scope)
    threadgroup float shared[8];

    uint max_ff = max(p.n_ff, p.n_ff_ffn_only);
    device float* hidden2    = scratch;
    device float* ff_gate    = scratch + p.n_embd;
    device float* ff_up      = scratch + p.n_embd + max_ff;
    device float* qkv_buf    = scratch + p.n_embd + 2 * max_ff;
    device float* ss_scratch = scratch + p.n_embd + 2 * max_ff +
                               (p.n_head + 2 * p.n_kv) * p.head_dim;

    uint sync_idx = 0;

    for (uint li = 0; li < p.n_layers; li++) {
        device const MegaLayerOffsets& lo = layer_off[li];
        uint layer_type = layer_types[li];

        // ── 1. Pre-norm (shared across all layer types) ──────────────
        device const float* norm_w = (device const float*)(weights + lo.attn_norm);
        if (li > 0) {
            mega_add_rms_norm(hidden, hidden2, norm_w, hidden2,
                ss_scratch, &sync_ctrs[sync_idx++ % 32],
                p.n_embd, p.n_tgs, p.rms_eps, shared, tgid, tid, tg_size);
        } else {
            mega_rms_norm(hidden, norm_w, hidden2,
                ss_scratch, &sync_ctrs[sync_idx++ % 32],
                p.n_embd, p.n_tgs, p.rms_eps, shared, tgid, tid, tg_size);
        }

        if (layer_type == LAYER_TYPE_SSM) {
            // SSM layers break out of the megakernel.
            // The host will read hidden2 and dispatch standard SSM kernels.
            // We write hidden2 to device memory (already done by rmsNorm above).
            // On resume, the host writes SSM output back into hidden2.
            // For now, skip SSM processing — Phase 2 will handle break/resume.

        } else if (layer_type == LAYER_TYPE_ATTENTION) {
            // ── 2a. Attention: Q/K/V → RoPE → SDPA → output ─────────
            uint qd = p.n_head * p.head_dim;
            uint kvd = p.n_kv * p.head_dim;
            device float* q_buf = qkv_buf;
            device float* k_buf = qkv_buf + qd;
            device float* v_buf = qkv_buf + qd + kvd;

            // Q projection
            mega_gemv_q8(hidden2, (device const block_q8_0*)(weights + lo.attn_q),
                q_buf, qd, p.n_embd, shared, tgid, tid, tg_size);
            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

            // K projection
            mega_gemv_q8(hidden2, (device const block_q8_0*)(weights + lo.attn_k),
                k_buf, kvd, p.n_embd, shared, tgid, tid, tg_size);
            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

            // V projection
            mega_gemv_q8(hidden2, (device const block_q8_0*)(weights + lo.attn_v),
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

            // KV cache append + inline SDPA
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

            // Output projection
            mega_gemv_q8(attn_out_buf, (device const block_q8_0*)(weights + lo.attn_output),
                hidden2, p.n_embd, qd, shared, tgid, tid, tg_size);
            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

            // ── Post-attention norm ──────────────────────────────────
            device const float* post_norm_w = (device const float*)(weights + lo.post_attn_norm);
            mega_add_rms_norm(hidden, hidden2, post_norm_w, hidden2,
                ss_scratch, &sync_ctrs[sync_idx++ % 32],
                p.n_embd, p.n_tgs, p.rms_eps, shared, tgid, tid, tg_size);

            // ── Attention FFN: gate + up + SiLU*mul + down ───────────
            mega_gemv_q8(hidden2, (device const block_q8_0*)(weights + lo.ffn_gate),
                ff_gate, p.n_ff, p.n_embd, shared, tgid, tid, tg_size);
            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

            mega_gemv_q8(hidden2, (device const block_q8_0*)(weights + lo.ffn_up),
                ff_up, p.n_ff, p.n_embd, shared, tgid, tid, tg_size);
            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

            mega_silu_mul(ff_gate, ff_up, p.n_ff, tgid, tid, tg_size);
            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

            mega_gemv_q8(ff_gate, (device const block_q8_0*)(weights + lo.ffn_down),
                hidden2, p.n_embd, p.n_ff, shared, tgid, tid, tg_size);
            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

        } else {
            // ── 2b. FFN-only: ReLU² MLP (no gate tensor) ────────────
            // Post-attention norm (shared)
            device const float* post_norm_w = (device const float*)(weights + lo.post_attn_norm);
            mega_add_rms_norm(hidden, hidden2, post_norm_w, hidden2,
                ss_scratch, &sync_ctrs[sync_idx++ % 32],
                p.n_embd, p.n_tgs, p.rms_eps, shared, tgid, tid, tg_size);

            // Up GEMV: ff_gate = W_up @ hidden2
            mega_gemv_q8(hidden2, (device const block_q8_0*)(weights + lo.ffn_up),
                ff_gate, p.n_ff_ffn_only, p.n_embd, shared, tgid, tid, tg_size);
            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

            // ReLU²: ff_gate[i] = max(0, ff_gate[i])²
            mega_relu_squared(ff_gate, p.n_ff_ffn_only, tgid, tid, tg_size);
            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);

            // Down GEMV: hidden2 = W_down @ ff_gate
            mega_gemv_q8(ff_gate, (device const block_q8_0*)(weights + lo.ffn_down),
                hidden2, p.n_embd, p.n_ff_ffn_only, shared, tgid, tid, tg_size);
            mega_grid_sync(&sync_ctrs[sync_idx++ % 32], p.n_tgs, tgid, tid);
            mega_sync_reset(&sync_ctrs[(sync_idx-1) % 32], tgid, tid);
        }
    }

    // ── Final: fuse last residual into hidden ────────────────────────────
    mega_add(hidden, hidden2, p.n_embd, tgid, tid, tg_size);
}
