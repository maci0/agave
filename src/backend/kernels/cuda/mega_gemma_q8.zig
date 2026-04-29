//! True megakernel: Gemma 3/4 Q8_0 — single dispatch for all layers (CUDA).
//!
//! Same structure as mega_gemma_q4k.zig but with Q8_0 dequantization.
//! Gemma uses GELU activation and uniform layer structure.
//!
//! Dispatch: max(n_embd, n_ff) blocks x 256 threads.

const cu = @import("common.zig");

// ── Q8_0 quantization constants ──────────────────────────────────
const q8_0_block_size: u32 = 34;
const q8_0_group_size: u32 = 32;

/// Number of sync counter slots.
const n_sync_slots: u32 = 32;

/// sqrt(2/pi), used in GELU approximation.
const sqrt_2_over_pi: f32 = 0.7978845608028654;

/// GELU cubic coefficient.
const gelu_coeff: f32 = 0.044715;

/// ln(2), used for base conversion.
const ln2: f32 = 0.6931471805599453;

// ── Grid sync ────────────────────────────────────────────────────

inline fn gridDim() u32 {
    return asm ("mov.u32 %[ret], %nctaid.x;"
        : [ret] "=r" (-> u32),
    );
}

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

fn gridSyncReset(sync_counter: *u32) void {
    if (cu.blockIdx() == 0 and cu.threadIdx() == 0) {
        asm volatile ("st.global.release.gpu.u32 [%[ptr]], 0;"
            :
            : [ptr] "l" (sync_counter),
        );
    }
    cu.syncthreads();
}

// ── Q8_0 block dot product ───────────────────────────────────────

inline fn q8BlockDot(x: [*]const f32, block_ptr: [*]const u8, k: u32, base_col: u32) f32 {
    const scale_bits = @as(u16, block_ptr[0]) | (@as(u16, block_ptr[1]) << 8);
    const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));
    const quants = block_ptr + 2;

    var blk_sum: f32 = 0.0;
    var qi: u32 = 0;
    while (qi < q8_0_group_size) : (qi += 1) {
        if (base_col + qi < k) {
            const q: i8 = @bitCast(quants[qi]);
            blk_sum += @as(f32, @floatFromInt(q)) * x[base_col + qi];
        }
    }
    return scale * blk_sum;
}

// ── Stage building blocks ────────────────────────────────────────

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

fn gemvQ8Stage(
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

    const blocks_per_row = (k + q8_0_group_size - 1) / q8_0_group_size;
    const row_bytes = blocks_per_row * q8_0_block_size;

    var sum: f32 = 0.0;
    var blk = tid;
    while (blk < blocks_per_row) : (blk += bdim) {
        const base_col = blk * q8_0_group_size;
        sum += q8BlockDot(x, w + bid * row_bytes + blk * q8_0_block_size, k, base_col);
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[bid] = sum;
}

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

/// GELU(gate) * up: Gemma GELU activation.
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

export fn megakernel_gemma_q8_kernel(
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

    const hidden2 = scratch;
    const ff_gate = scratch + n_embd;
    const ff_up = scratch + n_embd + n_ff;
    const qkv_buf = scratch + n_embd + 2 * n_ff;

    var sync_idx: u32 = 0;

    var li: u32 = 0;
    while (li < n_layers) : (li += 1) {
        // 1. Pre-attention norm
        const norm_off = readLayerOffset(layer_offsets, li, off_attn_norm);
        const norm_w: [*]const f32 = @ptrCast(@alignCast(weights + norm_off));
        rmsNormStage(hidden, norm_w, hidden2, ss_scratch, &sync_ctrs[sync_idx % n_sync_slots], n_embd, rms_eps);
        sync_idx += 1;

        // 2. Q/K/V projections
        const qd = n_head * head_dim;
        const kvd = n_kv * head_dim;
        const q_buf = qkv_buf;
        const k_buf = qkv_buf + qd;
        const v_buf = qkv_buf + qd + kvd;

        const q_off = readLayerOffset(layer_offsets, li, off_attn_q);
        gemvQ8Stage(hidden2, weights + q_off, q_buf, qd, n_embd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        const k_off = readLayerOffset(layer_offsets, li, off_attn_k);
        gemvQ8Stage(hidden2, weights + k_off, k_buf, kvd, n_embd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        const v_off = readLayerOffset(layer_offsets, li, off_attn_v);
        gemvQ8Stage(hidden2, weights + v_off, v_buf, kvd, n_embd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        // 3. RoPE
        ropeStage(q_buf, n_head, head_dim, rope_dim, rope_theta, seq_pos);
        ropeStage(k_buf, n_kv, head_dim, rope_dim, rope_theta, seq_pos);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        // 4. SDPA placeholder (Phase 2)
        // 5. Output projection
        const out_off = readLayerOffset(layer_offsets, li, off_attn_output);
        gemvQ8Stage(q_buf, weights + out_off, hidden2, n_embd, qd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        // 6. Post-attention norm + residual
        addStage(hidden, hidden2, n_embd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        const post_norm_off = readLayerOffset(layer_offsets, li, off_post_attn_norm);
        const post_norm_w: [*]const f32 = @ptrCast(@alignCast(weights + post_norm_off));
        rmsNormStage(hidden, post_norm_w, hidden2, ss_scratch, &sync_ctrs[sync_idx % n_sync_slots], n_embd, rms_eps);
        sync_idx += 1;

        // 7. FFN gate+up + GELU
        const gate_off = readLayerOffset(layer_offsets, li, off_ffn_gate);
        gemvQ8Stage(hidden2, weights + gate_off, ff_gate, n_ff, n_embd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        const up_off = readLayerOffset(layer_offsets, li, off_ffn_up);
        gemvQ8Stage(hidden2, weights + up_off, ff_up, n_ff, n_embd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        geluMulStage(ff_gate, ff_up, n_ff);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        // 8. FFN down projection
        const down_off = readLayerOffset(layer_offsets, li, off_ffn_down);
        gemvQ8Stage(ff_gate, weights + down_off, hidden2, n_embd, n_ff);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        // 9. Post-FFN residual
        addStage(hidden, hidden2, n_embd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots]);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;
    }
}
