//! True megakernel: Gemma 3/4 Q4_K — single dispatch for all layers (CUDA).
//!
//! Gemma 3/4 dense models have uniform layer structure — every layer is identical:
//!   pre-norm → Q/K/V → RoPE → SDPA → output → post-attn-norm → residual
//!   pre-norm → gate/up → GELU(gate)*up → down → post-FFN-norm → residual
//!
//! Uses Q4_K dequantization (256 values per super-block, 144 bytes).
//! GELU activation instead of SiLU (Gemma architecture).
//!
//! Dispatch: max(n_embd, n_ff) blocks x 256 threads.

const cu = @import("common.zig");

// ── Q4_K quantization constants ──────────────────────────────────
/// Bytes per Q4_K super-block.
const q4k_bytes_per_block: u32 = 144;
/// Elements per Q4_K super-block.
const q4k_values_per_block: u32 = 256;

/// Number of sync counter slots.
const n_sync_slots: u32 = 32;

/// sqrt(2/pi), used in GELU approximation.
const sqrt_2_over_pi: f32 = 0.7978845608028654;

/// GELU cubic coefficient.
const gelu_coeff: f32 = 0.044715;

// ── Grid sync (same as mega_qwen35_q8.zig) ──────────────────────

/// Read grid dimension (number of blocks in the x dimension).
inline fn gridDim() u32 {
    return asm ("mov.u32 %[ret], %nctaid.x;"
        : [ret] "=r" (-> u32),
    );
}

/// Atomic grid sync: all blocks arrive before any proceed.
fn gridSync(sync_counter: *u32) void {
    if (cu.threadIdx() == 0) {
        _ = asm volatile ("atom.global.add.u32 %[ret], [%[ptr]], 1;"
            : [ret] "=r" (-> u32),
            : [ptr] "l" (sync_counter),
        );
        const n_blocks = gridDim();
        while (true) {
            const val = asm volatile ("ld.global.acquire.gpu.u32 %[ret], [%[ptr]];"
                : [ret] "=r" (-> u32),
                : [ptr] "l" (sync_counter),
            );
            if (val >= n_blocks) break;
        }
    }
    cu.syncthreads();
}

/// Reset sync counter for reuse.
fn gridSyncReset(sync_counter: *u32) void {
    if (cu.blockIdx() == 0 and cu.threadIdx() == 0) {
        asm volatile ("st.global.release.gpu.u32 [%[ptr]], 0;"
            :
            : [ptr] "l" (sync_counter),
        );
    }
    cu.syncthreads();
}

// ── Q4_K block dot product ───────────────────────────────────────
// Same as gemv_q4_k.zig: processes one 256-element super-block.

const f16tof32 = cu.f16tof32;
const getScaleMinK4 = cu.getScaleMinK4;

inline fn q4kBlockDot(
    x: [*]const f32,
    bp: [*]const u8,
    k: u32,
    block_start: u32,
) f32 {
    const d = f16tof32(bp);
    const dmin = f16tof32(bp + 2);
    const scales = bp + 4;
    const qs = bp + 16;
    var sum: f32 = 0.0;

    var g: u32 = 0;
    while (g < 4) : (g += 1) {
        const gi_lo = block_start + g * 64;
        if (gi_lo >= k) break;
        const ql_off = g * 32;

        var sc_lo: u8 = undefined;
        var m_lo: u8 = undefined;
        var sc_hi: u8 = undefined;
        var m_hi: u8 = undefined;
        getScaleMinK4(g * 2, scales, &sc_lo, &m_lo);
        getScaleMinK4(g * 2 + 1, scales, &sc_hi, &m_hi);

        {
            const d_sc = d * @as(f32, @floatFromInt(sc_lo));
            const dm_m = dmin * @as(f32, @floatFromInt(m_lo));
            var q_dot: f32 = 0.0;
            var x_sum: f32 = 0.0;
            for (0..32) |l| {
                const gi = gi_lo + @as(u32, @intCast(l));
                if (gi >= k) break;
                q_dot += x[gi] * @as(f32, @floatFromInt(qs[ql_off + @as(u32, @intCast(l))] & 0x0F));
                x_sum += x[gi];
            }
            sum += d_sc * q_dot - dm_m * x_sum;
        }

        {
            const d_sc = d * @as(f32, @floatFromInt(sc_hi));
            const dm_m = dmin * @as(f32, @floatFromInt(m_hi));
            var q_dot: f32 = 0.0;
            var x_sum: f32 = 0.0;
            for (0..32) |l| {
                const gi = gi_lo + 32 + @as(u32, @intCast(l));
                if (gi >= k) break;
                q_dot += x[gi] * @as(f32, @floatFromInt(qs[ql_off + @as(u32, @intCast(l))] >> 4));
                x_sum += x[gi];
            }
            sum += d_sc * q_dot - dm_m * x_sum;
        }
    }
    return sum;
}

// ── Stage building blocks ────────────────────────────────────────

/// RMS norm: cooperative multi-block, same as mega_qwen35_q8.zig.
fn rmsNormStage(
    input: [*]const f32,
    weight: [*]const f32,
    output: [*]f32,
    ss_buf: *u32,
    sync_ctr: *u32,
    n_dim: u32,
    eps: f32,
) void {
    const bid = cu.blockIdx();
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const n_blocks = gridDim();

    var local_ss: f32 = 0.0;
    var i = bid * bdim + tid;
    while (i < n_dim) : (i += n_blocks * bdim) {
        const v = input[i];
        local_ss += v * v;
    }

    local_ss = cu.blockReduceAdd(local_ss);

    if (tid == 0 and local_ss != 0.0) {
        const bits: u32 = @bitCast(local_ss);
        _ = asm volatile ("atom.global.add.u32 %[ret], [%[ptr]], %[val];"
            : [ret] "=r" (-> u32),
            : [ptr] "l" (ss_buf),
              [val] "r" (bits),
        );
    }

    gridSync(sync_ctr);

    const ss_bits = asm volatile ("ld.global.acquire.gpu.u32 %[ret], [%[ptr]];"
        : [ret] "=r" (-> u32),
        : [ptr] "l" (ss_buf),
    );
    const ss: f32 = @bitCast(ss_bits);
    const inv_rms = cu.rsqrtf(ss / @as(f32, @floatFromInt(n_dim)) + eps);

    i = bid * bdim + tid;
    while (i < n_dim) : (i += n_blocks * bdim) {
        output[i] = input[i] * weight[i] * inv_rms;
    }

    if (bid == 0 and tid == 0) {
        asm volatile ("st.global.release.gpu.u32 [%[ptr]], 0;"
            :
            : [ptr] "l" (ss_buf),
        );
    }

    gridSync(sync_ctr);
    gridSyncReset(sync_ctr);
}

/// Residual add: a[i] += b[i].
fn addStage(
    a: [*]f32,
    b: [*]const f32,
    n: u32,
) void {
    const bid = cu.blockIdx();
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const n_blocks = gridDim();

    var i = bid * bdim + tid;
    while (i < n) : (i += n_blocks * bdim) {
        a[i] += b[i];
    }
}

/// GEMV Q4_K: y[row] = dot(W[row,:], x). One block per output row.
fn gemvQ4kStage(
    x: [*]const f32,
    w: [*]const u8,
    y: [*]f32,
    n_out: u32,
    k: u32,
) void {
    const bid = cu.blockIdx();
    if (bid >= n_out) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const num_blocks = (k + q4k_values_per_block - 1) / q4k_values_per_block;

    var sum: f32 = 0.0;
    var b = tid;
    while (b < num_blocks) : (b += bdim) {
        const block_start = b * q4k_values_per_block;
        sum += q4kBlockDot(x, w + bid * num_blocks * q4k_bytes_per_block + b * q4k_bytes_per_block, k, block_start);
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[bid] = sum;
}

/// RoPE: same as mega_qwen35_q8.zig.
fn ropeStage(
    x: [*]f32,
    n_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    theta: f32,
    seq_pos: u32,
) void {
    const bid = cu.blockIdx();
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const n_blocks = gridDim();

    const ln2: f32 = 0.6931471805599453;
    const half_rd = rope_dim / 2;
    const total = n_heads * half_rd;

    var idx = bid * bdim + tid;
    while (idx < total) : (idx += n_blocks * bdim) {
        const h = idx / half_rd;
        const d = idx % half_rd;

        const neg_log_theta = -cu.log2f(theta) * ln2;
        const freq = cu.expf(neg_log_theta * @as(f32, @floatFromInt(d * 2)) / @as(f32, @floatFromInt(rope_dim)));
        const angle = @as(f32, @floatFromInt(seq_pos)) * freq;

        const cos_a = cu.cosf(angle);
        const sin_a = cu.sinf(angle);

        const idx0 = h * head_dim + d;
        const idx1 = h * head_dim + d + half_rd;
        const x0 = x[idx0];
        const x1 = x[idx1];
        x[idx0] = x0 * cos_a - x1 * sin_a;
        x[idx1] = x0 * sin_a + x1 * cos_a;
    }
}

/// GELU(gate) * up: Gemma uses GELU activation instead of SiLU.
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn geluMulStage(
    gate: [*]f32,
    up: [*]const f32,
    n: u32,
) void {
    const bid = cu.blockIdx();
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const n_blocks = gridDim();

    var i = bid * bdim + tid;
    while (i < n) : (i += n_blocks * bdim) {
        const g = gate[i];
        const inner = sqrt_2_over_pi * (g + gelu_coeff * g * g * g);
        const tanh_val = cu.tanhf(inner);
        gate[i] = 0.5 * g * (1.0 + tanh_val) * up[i];
    }
}

// ── Per-layer offset table ───────────────────────────────────────
const off_attn_norm: u32 = 0;
const off_attn_q: u32 = 8;
const off_attn_k: u32 = 16;
const off_attn_v: u32 = 24;
const off_attn_output: u32 = 48;
const off_post_attn_norm: u32 = 128;
const off_ffn_gate: u32 = 136;
const off_ffn_up: u32 = 144;
const off_ffn_down: u32 = 152;
const layer_offsets_stride: u32 = 160;

inline fn readLayerOffset(layer_offsets: [*]const u8, layer_idx: u32, field_byte_offset: u32) u64 {
    const base = layer_idx * layer_offsets_stride + field_byte_offset;
    const ptr = layer_offsets + base;
    return @as(u64, ptr[0]) |
        (@as(u64, ptr[1]) << 8) |
        (@as(u64, ptr[2]) << 16) |
        (@as(u64, ptr[3]) << 24) |
        (@as(u64, ptr[4]) << 32) |
        (@as(u64, ptr[5]) << 40) |
        (@as(u64, ptr[6]) << 48) |
        (@as(u64, ptr[7]) << 56);
}

// ── Main megakernel entry point ──────────────────────────────────

export fn megakernel_gemma_q4k_kernel(
    weights: [*]const u8,
    layer_offsets: [*]const u8,
    kv_keys: [*]f32,
    kv_values: [*]f32,
    hidden: [*]f32,
    scratch: [*]f32,
    sync_ctrs: [*]u32,
    ss_scratch: *u32,
    n_layers: u32,
    n_embd: u32,
    n_head: u32,
    n_kv: u32,
    head_dim: u32,
    n_ff: u32,
    rope_dim: u32,
    rope_theta: f32,
    rms_eps: f32,
    embd_scale: f32,
    max_seq_len: u32,
    seq_pos: u32,
) callconv(.kernel) void {
    _ = kv_keys;
    _ = kv_values;
    _ = max_seq_len;
    _ = embd_scale;

    // Scratch buffer sub-regions
    const hidden2 = scratch;
    const ff_gate = scratch + n_embd;
    const ff_up = scratch + n_embd + n_ff;
    const qkv_buf = scratch + n_embd + 2 * n_ff;

    var sync_idx: u32 = 0;

    // ── Layer loop ───────────────────────────────────────────────
    var li: u32 = 0;
    while (li < n_layers) : (li += 1) {
        // ── 1. Pre-attention norm ────────────────────────────────
        const norm_off = readLayerOffset(layer_offsets, li, off_attn_norm);
        const norm_w: [*]const f32 = @ptrCast(@alignCast(weights + norm_off));
        rmsNormStage(hidden, norm_w, hidden2, ss_scratch, &sync_ctrs[sync_idx % n_sync_slots], n_embd, rms_eps);
        sync_idx += 1;

        // ── 2. Q/K/V projections ─────────────────────────────────
        const qd = n_head * head_dim;
        const kvd = n_kv * head_dim;
        const q_buf = qkv_buf;
        const k_buf = qkv_buf + qd;
        const v_buf = qkv_buf + qd + kvd;

        const q_off = readLayerOffset(layer_offsets, li, off_attn_q);
        gemvQ4kStage(hidden2, weights + q_off, q_buf, qd, n_embd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        const k_off = readLayerOffset(layer_offsets, li, off_attn_k);
        gemvQ4kStage(hidden2, weights + k_off, k_buf, kvd, n_embd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        const v_off = readLayerOffset(layer_offsets, li, off_attn_v);
        gemvQ4kStage(hidden2, weights + v_off, v_buf, kvd, n_embd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        // ── 3. RoPE ──────────────────────────────────────────────
        ropeStage(q_buf, n_head, head_dim, rope_dim, rope_theta, seq_pos);
        ropeStage(k_buf, n_kv, head_dim, rope_dim, rope_theta, seq_pos);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        // ── 4. SDPA placeholder (Phase 2) ────────────────────────

        // ── 5. Output projection ─────────────────────────────────
        const out_off = readLayerOffset(layer_offsets, li, off_attn_output);
        gemvQ4kStage(q_buf, weights + out_off, hidden2, n_embd, qd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        // ── 6. Post-attention norm + residual ────────────────────
        addStage(hidden, hidden2, n_embd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        const post_norm_off = readLayerOffset(layer_offsets, li, off_post_attn_norm);
        const post_norm_w: [*]const f32 = @ptrCast(@alignCast(weights + post_norm_off));
        rmsNormStage(hidden, post_norm_w, hidden2, ss_scratch, &sync_ctrs[sync_idx % n_sync_slots], n_embd, rms_eps);
        sync_idx += 1;

        // ── 7. FFN: gate + up + GELU*mul ─────────────────────────
        const gate_off = readLayerOffset(layer_offsets, li, off_ffn_gate);
        gemvQ4kStage(hidden2, weights + gate_off, ff_gate, n_ff, n_embd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        const up_off = readLayerOffset(layer_offsets, li, off_ffn_up);
        gemvQ4kStage(hidden2, weights + up_off, ff_up, n_ff, n_embd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        geluMulStage(ff_gate, ff_up, n_ff);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        // ── 8. FFN down projection ───────────────────────────────
        const down_off = readLayerOffset(layer_offsets, li, off_ffn_down);
        gemvQ4kStage(ff_gate, weights + down_off, hidden2, n_embd, n_ff);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        // ── 9. Post-FFN residual ─────────────────────────────────
        addStage(hidden, hidden2, n_embd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;
    }
}
