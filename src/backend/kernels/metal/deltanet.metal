// DeltaNet SSM kernels (conv1d, L2 norm, recurrence). Used by hybrid attention/SSM models (e.g. Qwen3.5).
// Eliminates GPU-CPU sync by running conv1d, L2 norm, and recurrence on GPU.

// ── Gate & Beta computation ─────────────────────────────────
// gate[h] = ssm_a[h] * softplus(alpha[h] + dt_bias[h])
// beta[h] = sigmoid(beta_in[h])
// Launch: 1 threadgroup, num_v_heads threads.

kernel void deltanet_gate_beta(
    device const float* alpha   [[buffer(0)]],
    device const float* beta_in [[buffer(1)]],
    device const float* ssm_a   [[buffer(2)]],
    device const float* dt_bias [[buffer(3)]],
    device float* gate_out      [[buffer(4)]],
    device float* beta_out      [[buffer(5)]],
    constant uint& n_heads      [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_heads) return;
    float x = alpha[tid] + dt_bias[tid];
    float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
    gate_out[tid] = ssm_a[tid] * sp;
    beta_out[tid] = 1.0f / (1.0f + exp(-beta_in[tid]));
}

// ── Conv1d + SiLU ──────────────────────────────────────────
// Each thread handles one channel. Updates ring buffer in-place.
// conv_state layout: [(d_conv-1) rows] × [conv_ch cols], row-major.
// Launch: ceil(conv_ch/256) threadgroups, 256 threads each.

kernel void deltanet_conv1d(
    device const float* conv_in  [[buffer(0)]],
    device float* conv_state     [[buffer(1)]],
    device const float* conv_w   [[buffer(2)]],
    device float* conv_out       [[buffer(3)]],
    constant uint& conv_ch       [[buffer(4)]],
    constant uint& d_conv        [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= conv_ch) return;
    uint ch = tid;
    uint hist = d_conv - 1;

    float sum = 0.0f;
    for (uint k = 0; k < hist; k++) {
        sum += conv_state[k * conv_ch + ch] * conv_w[ch * d_conv + k];
    }
    sum += conv_in[ch] * conv_w[ch * d_conv + hist];
    conv_out[ch] = sum / (1.0f + exp(-sum));

    // Shift ring buffer left, append current input
    for (uint p = 0; p + 1 < hist; p++) {
        conv_state[p * conv_ch + ch] = conv_state[(p + 1) * conv_ch + ch];
    }
    conv_state[(hist - 1) * conv_ch + ch] = conv_in[ch];
}

// ── Batched L2 Norm ─────────────────────────────────────────
// Normalizes 2*num_k_heads vectors (Q heads then K heads) in-place.
// Launch: 2*num_k_heads threadgroups, 128 threads each.

kernel void deltanet_l2_norm(
    device float* data         [[buffer(0)]],
    constant uint& head_dim    [[buffer(1)]],
    constant uint& num_k_heads [[buffer(2)]],
    constant float& eps        [[buffer(3)]],
    constant uint& q_off       [[buffer(4)]],
    constant uint& k_off       [[buffer(5)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint base_off = (tgid < num_k_heads)
        ? q_off + tgid * head_dim
        : k_off + (tgid - num_k_heads) * head_dim;

    threadgroup float shared[8];
    float sum = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        float v = data[base_off + i];
        sum += v * v;
    }
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    // Broadcast sum to all threads
    if (tid == 0) shared[0] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = shared[0];
    float inv = rsqrt(sum + eps);
    for (uint i = tid; i < head_dim; i += tg_size) {
        data[base_off + i] *= inv;
    }
}

// ── DeltaNet Recurrence + Gated Output ─────────────────────
// One threadgroup per v-head. Threads cooperate on vi dimension.
// Computes: decay → recurrence → algebraic output → RMSNorm + SiLU gate.
// Launch: num_v_heads threadgroups, head_v_dim threads each.
//
// Optimizations:
//   - K/Q vectors cached in threadgroup memory (read once from device, reused per vi row)
//   - float4 vectorized inner loops over head_k_dim (4x throughput)
//   - Fused decay + dot products in a single pass

kernel void deltanet_recurrence(
    device const float* q           [[buffer(0)]],
    device const float* k           [[buffer(1)]],
    device const float* v           [[buffer(2)]],
    device float* state             [[buffer(3)]],
    device const float* gate        [[buffer(4)]],
    device const float* beta_vals   [[buffer(5)]],
    device const float* z           [[buffer(6)]],
    device const float* norm_w      [[buffer(7)]],
    device float* output            [[buffer(8)]],
    constant uint& head_v_dim       [[buffer(9)]],
    constant uint& head_k_dim       [[buffer(10)]],
    constant uint& num_k_heads      [[buffer(11)]],
    constant uint& num_v_heads      [[buffer(12)]],
    constant float& q_scale         [[buffer(13)]],
    constant float& rms_eps         [[buffer(14)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint h = tgid;
    if (h >= num_v_heads) return;

    uint hvd = head_v_dim;
    uint hkd = head_k_dim;
    float decay = exp(gate[h]);
    float beta_h = beta_vals[h];
    uint kh = (num_k_heads == num_v_heads) ? h : h % num_k_heads;
    uint s_off = h * hvd * hkd;
    uint k_base = kh * hkd;

    // Cache K and Q vectors in threadgroup memory — read once from device,
    // reused for every vi row (up to 128 reuses). Max head_k_dim = 256.
    threadgroup float k_local[256];
    threadgroup float q_local[256];
    for (uint i = tid; i < hkd; i += tg_size) {
        k_local[i] = k[k_base + i];
        q_local[i] = q[k_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Precompute dot(K, Q) via cooperative threadgroup reduction
    threadgroup float shared[8];
    float kq_local_sum = 0.0f;
    for (uint ki = tid; ki < hkd; ki += tg_size) {
        kq_local_sum += k_local[ki] * q_local[ki];
    }
    float kq = threadgroup_reduce_sum(kq_local_sum, shared, tid, tg_size);
    // Broadcast kq to all threads (reduce_sum only guarantees tid < 32)
    if (tid == 0) shared[0] = kq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kq = shared[0];

    float4 decay_v4 = float4(decay);

    // Each thread processes assigned vi rows (typically 1 row per thread)
    for (uint vi = tid; vi < hvd; vi += tg_size) {
        uint row_off = s_off + vi * hkd;

        // Pass 1: decay + dot(S_dec, K) + dot(S_dec, Q) — float4 vectorized
        float sk = 0.0f;
        float sq_dec = 0.0f;
        uint hkd4 = hkd & ~3u;
        for (uint ki = 0; ki < hkd4; ki += 4) {
            float4 s_old = *(device float4*)(state + row_off + ki);
            float4 s_dec = s_old * decay_v4;
            *(device float4*)(state + row_off + ki) = s_dec;
            float4 kv = *(threadgroup const float4*)(k_local + ki);
            float4 qv = *(threadgroup const float4*)(q_local + ki);
            sk += dot(s_dec, kv);
            sq_dec += dot(s_dec, qv);
        }
        for (uint ki = hkd4; ki < hkd; ki++) {
            float s_old = state[row_off + ki];
            float s_dec = s_old * decay;
            state[row_off + ki] = s_dec;
            sk += s_dec * k_local[ki];
            sq_dec += s_dec * q_local[ki];
        }

        float delta = beta_h * (v[h * hvd + vi] - sk);
        output[h * hvd + vi] = (sq_dec + delta * kq) * q_scale;

        // Pass 2: state update — float4 vectorized
        float4 delta_v4 = float4(delta);
        for (uint ki = 0; ki < hkd4; ki += 4) {
            float4 s = *(device float4*)(state + row_off + ki);
            float4 kv = *(threadgroup const float4*)(k_local + ki);
            *(device float4*)(state + row_off + ki) = s + kv * delta_v4;
        }
        for (uint ki = hkd4; ki < hkd; ki++) {
            state[row_off + ki] += k_local[ki] * delta;
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Gated output: rms_norm(output) * silu(z)
    uint off = h * hvd;

    float ss = 0.0f;
    for (uint vi = tid; vi < hvd; vi += tg_size) {
        float val = output[off + vi];
        ss += val * val;
    }
    ss = threadgroup_reduce_sum(ss, shared, tid, tg_size);
    // Broadcast ss to all threads
    if (tid == 0) shared[0] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = shared[0];
    float inv_rms = rsqrt(ss / float(hvd) + rms_eps);

    for (uint vi = tid; vi < hvd; vi += tg_size) {
        float normed = output[off + vi] * inv_rms * norm_w[vi];
        float z_val = z[off + vi];
        output[off + vi] = normed * z_val / (1.0f + exp(-z_val));
    }
}
