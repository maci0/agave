//! True megakernel: Qwen 3.5 Q8_0 — single dispatch for all layers (ROCm/AMDGCN).
//!
//! Eliminates ~330 kernel dispatches per token down to 1 dispatch.
//! Uses atomic grid sync between stages (all workgroups must be co-resident).
//!
//! Dispatch: max(n_embd, n_ff) workgroups x 256 threads.
//! For Qwen 3.5 0.8B: max(1536, 4096) = 4096 workgroups.
//!
//! Scratch layout (for Qwen 3.5 0.8B, n_embd=1536, n_ff=4096):
//!   [0 .. n_embd)                        = hidden2 (normalized)
//!   [n_embd .. n_embd+n_ff)              = ff_gate
//!   [n_embd+n_ff .. n_embd+2*n_ff)       = ff_up
//!   [n_embd+2*n_ff .. +qkv_size)         = Q/K/V buffers
//!   [+qkv_size .. +1)                    = ss_scratch (sum-of-squares)
//!
//! Structurally identical to the CUDA mega_qwen35_q8.zig but uses AMDGCN
//! primitives from rocm/common.zig (LLVM intrinsics, s_barrier, ds_bpermute).

const cu = @import("common.zig");

// ── Q8_0 quantization constants ──────────────────────────────────
/// Bytes per Q8_0 block: 2 bytes f16 scale + 32 bytes i8 quants.
const q8_0_block_size: u32 = 34;
/// Elements per Q8_0 block.
const q8_0_group_size: u32 = 32;

/// ln(2), used for base conversion: ln(x) = log2(x) * ln(2).
const ln2: f32 = 0.6931471805599453;

/// Number of sync counter slots (cycled via modular indexing).
const n_sync_slots: u32 = 32;

// ── Grid sync via global atomic counters ─────────────────────────
// All dispatched workgroups must be co-resident for this to work.
// On AMDGCN, this requires occupancy guarantee from the runtime.

/// Read grid dimension from the implicit kernel argument.
/// On AMDGCN the grid size is passed as an implicit argument, but
/// for the megakernel we pass n_blocks explicitly as a kernel parameter
/// instead. This function is not used — we pass grid_dim directly.
/// Atomic grid sync: all workgroups arrive before any proceed.
/// Uses a global atomic counter; thread 0 of each workgroup increments,
/// then spins until all workgroups have arrived. syncthreads() ensures
/// all threads in each workgroup see the barrier completion.
fn gridSync(sync_counter: *u32, n_blocks: u32) void {
    if (cu.threadIdx() == 0) {
        // Signal arrival via atomic add
        _ = @atomicRmw(u32, sync_counter, .Add, 1, .monotonic);

        // Spin until all workgroups arrive
        while (true) {
            const val = @atomicLoad(u32, sync_counter, .acquire);
            if (val >= n_blocks) break;
        }
    }
    cu.syncthreads();
}

/// Reset sync counter for reuse. Only workgroup 0, thread 0 writes.
fn gridSyncReset(sync_counter: *u32) void {
    if (cu.blockIdx() == 0 and cu.threadIdx() == 0) {
        @atomicStore(u32, sync_counter, 0, .release);
    }
    cu.syncthreads();
}

// ── Q8_0 block dot product ───────────────────────────────────────
// Computes dot(scale * quants, x) for one 32-element quantization block.

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
// Each stage processes one forward-pass operation across all workgroups.
// Workgroups beyond the stage's output dimension are idle.

/// RMS norm: output[i] = input[i] * weight[i] * rsqrt(mean(x^2) + eps).
/// Multi-workgroup cooperative: all workgroups contribute to sum-of-squares via
/// global atomic add, then grid-sync before normalization.
fn rmsNormStage(
    input: [*]const f32,
    weight: [*]const f32,
    output: [*]f32,
    ss_buf: *u32,
    sync_ctr: *u32,
    n_dim: u32,
    n_blocks: u32,
    eps: f32,
) void {
    const bid = cu.blockIdx();
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    // Phase 1: partial sum of squares
    var local_ss: f32 = 0.0;
    var i = bid * bdim + tid;
    while (i < n_dim) : (i += n_blocks * bdim) {
        const v = input[i];
        local_ss += v * v;
    }

    // Intra-workgroup reduction
    local_ss = cu.blockReduceAdd(local_ss);

    // Atomic add to global sum (reinterpret as u32 for atomic)
    if (tid == 0 and local_ss != 0.0) {
        const bits: u32 = @bitCast(local_ss);
        _ = @atomicRmw(u32, ss_buf, .Add, bits, .monotonic);
    }

    // Grid sync: wait for all workgroups to contribute
    gridSync(sync_ctr, n_blocks);

    // Phase 2: normalize (all workgroups read the shared sum)
    const ss_bits = @atomicLoad(u32, ss_buf, .acquire);
    const ss: f32 = @bitCast(ss_bits);
    const inv_rms = cu.rsqrtf(ss / @as(f32, @floatFromInt(n_dim)) + eps);

    i = bid * bdim + tid;
    while (i < n_dim) : (i += n_blocks * bdim) {
        output[i] = input[i] * weight[i] * inv_rms;
    }

    // Reset ss for next norm
    if (bid == 0 and tid == 0) {
        @atomicStore(u32, ss_buf, 0, .release);
    }

    // Sync and reset the sync counter
    gridSync(sync_ctr, n_blocks);
    gridSyncReset(sync_ctr);
}

/// Fused add + RMS norm: a[i] += b[i], output = rmsNorm(a, weight, eps).
/// Same cooperative pattern as rmsNormStage.
fn addRmsNormStage(
    a: [*]f32,
    b: [*]const f32,
    weight: [*]const f32,
    output: [*]f32,
    ss_buf: *u32,
    sync_ctr: *u32,
    n_dim: u32,
    n_blocks: u32,
    eps: f32,
) void {
    const bid = cu.blockIdx();
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    // Phase 1: add + sum of squares
    var local_ss: f32 = 0.0;
    var i = bid * bdim + tid;
    while (i < n_dim) : (i += n_blocks * bdim) {
        const v = a[i] + b[i];
        a[i] = v;
        local_ss += v * v;
    }

    local_ss = cu.blockReduceAdd(local_ss);

    if (tid == 0 and local_ss != 0.0) {
        const bits: u32 = @bitCast(local_ss);
        _ = @atomicRmw(u32, ss_buf, .Add, bits, .monotonic);
    }

    gridSync(sync_ctr, n_blocks);

    const ss_bits = @atomicLoad(u32, ss_buf, .acquire);
    const ss: f32 = @bitCast(ss_bits);
    const inv_rms = cu.rsqrtf(ss / @as(f32, @floatFromInt(n_dim)) + eps);

    i = bid * bdim + tid;
    while (i < n_dim) : (i += n_blocks * bdim) {
        output[i] = a[i] * weight[i] * inv_rms;
    }

    if (bid == 0 and tid == 0) {
        @atomicStore(u32, ss_buf, 0, .release);
    }

    gridSync(sync_ctr, n_blocks);
    gridSyncReset(sync_ctr);
}

/// GEMV Q8_0: y[row] = dot(W[row,:], x). One workgroup per output row.
/// Workgroups beyond n_out are idle.
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

/// RoPE: apply rotary position encoding to Q or K vectors.
/// Distributed across all workgroups (each workgroup handles a subset of pairs).
fn ropeStage(
    x: [*]f32,
    n_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    theta: f32,
    seq_pos: u32,
    n_blocks: u32,
) void {
    const bid = cu.blockIdx();
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const half_rd = rope_dim / 2;
    const total = n_heads * half_rd;

    // Distribute pairs across all workgroups
    var idx = bid * bdim + tid;
    while (idx < total) : (idx += n_blocks * bdim) {
        const h = idx / half_rd;
        const d = idx % half_rd;

        // freq = exp(-log(theta) * 2d / rope_dim)
        const neg_log_theta = -@log(theta);
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

/// SiLU(gate) * up: in-place activation on gate buffer.
fn siluMulStage(
    gate: [*]f32,
    up: [*]const f32,
    n: u32,
    n_blocks: u32,
) void {
    const bid = cu.blockIdx();
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    var i = bid * bdim + tid;
    while (i < n) : (i += n_blocks * bdim) {
        const g = gate[i];
        gate[i] = g * cu.sigmoidf(g) * up[i];
    }
}

/// Residual add: a[i] += b[i].
fn addStage(
    a: [*]f32,
    b: [*]const f32,
    n: u32,
    n_blocks: u32,
) void {
    const bid = cu.blockIdx();
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    var i = bid * bdim + tid;
    while (i < n) : (i += n_blocks * bdim) {
        a[i] += b[i];
    }
}

// ── Per-layer offset table ───────────────────────────────────────
// Must match megakernel.zig LayerOffsets: 20 fields x 8 bytes = 160 bytes.
// Laid out as a flat array of u64 values.

/// Byte offset of attn_norm within a LayerOffsets struct (field 0).
const off_attn_norm: u32 = 0;
/// Byte offset of attn_q (field 1).
const off_attn_q: u32 = 8;
/// Byte offset of attn_k (field 2).
const off_attn_k: u32 = 16;
/// Byte offset of attn_v (field 3).
const off_attn_v: u32 = 24;
/// Byte offset of attn_output (field 6).
const off_attn_output: u32 = 48;
/// Byte offset of post_attn_norm (field 16).
const off_post_attn_norm: u32 = 128;
/// Byte offset of ffn_gate (field 17).
const off_ffn_gate: u32 = 136;
/// Byte offset of ffn_up (field 18).
const off_ffn_up: u32 = 144;
/// Byte offset of ffn_down (field 19).
const off_ffn_down: u32 = 152;

/// Size of one LayerOffsets struct in bytes (20 u64 fields).
const layer_offsets_stride: u32 = 160;

/// Read a u64 offset from the layer_offsets table.
inline fn readLayerOffset(layer_offsets: [*]const u8, layer_idx: u32, field_byte_offset: u32) u64 {
    const base = layer_idx * layer_offsets_stride + field_byte_offset;
    const ptr = layer_offsets + base;
    // Read 8 bytes as little-endian u64
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

export fn megakernel_qwen35_q8_kernel(
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
    full_attn_interval: u32,
    max_seq_len: u32,
    seq_pos: u32,
    n_blocks: u32,
) callconv(.kernel) void {
    // Suppress unused parameter warnings for KV cache (Phase 2)
    _ = kv_keys;
    _ = kv_values;
    _ = max_seq_len;

    // Scratch buffer sub-regions
    const hidden2 = scratch;
    const ff_gate = scratch + n_embd;
    const ff_up = scratch + n_embd + n_ff;
    const qkv_buf = scratch + n_embd + 2 * n_ff;

    var sync_idx: u32 = 0;

    // ── Layer loop ───────────────────────────────────────────────
    var li: u32 = 0;
    while (li < n_layers) : (li += 1) {
        const is_attn = ((li + 1) % full_attn_interval) == 0;
        const fuse_residual = li > 0;

        // ── 1. Pre-attention norm ────────────────────────────────
        const norm_off = readLayerOffset(layer_offsets, li, off_attn_norm);
        const norm_w: [*]const f32 = @ptrCast(@alignCast(weights + norm_off));

        if (fuse_residual) {
            addRmsNormStage(
                hidden,
                hidden2,
                norm_w,
                hidden2,
                ss_scratch,
                &sync_ctrs[sync_idx % n_sync_slots],
                n_embd,
                n_blocks,
                rms_eps,
            );
        } else {
            rmsNormStage(
                hidden,
                norm_w,
                hidden2,
                ss_scratch,
                &sync_ctrs[sync_idx % n_sync_slots],
                n_embd,
                n_blocks,
                rms_eps,
            );
        }
        sync_idx += 1;

        // ── 2. Attention/DeltaNet projections ────────────────────
        if (is_attn) {
            const qd = n_head * head_dim * 2; // x2 for gate
            const kvd = n_kv * head_dim;
            const q_buf = qkv_buf;
            const k_buf = qkv_buf + qd;
            const v_buf = qkv_buf + qd + kvd;

            // Q projection
            const q_off = readLayerOffset(layer_offsets, li, off_attn_q);
            gemvQ8Stage(hidden2, weights + q_off, q_buf, qd, n_embd);
            gridSync(&sync_ctrs[sync_idx % n_sync_slots], n_blocks);
            gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
            sync_idx += 1;

            // K projection
            const k_off = readLayerOffset(layer_offsets, li, off_attn_k);
            gemvQ8Stage(hidden2, weights + k_off, k_buf, kvd, n_embd);
            gridSync(&sync_ctrs[sync_idx % n_sync_slots], n_blocks);
            gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
            sync_idx += 1;

            // V projection
            const v_off = readLayerOffset(layer_offsets, li, off_attn_v);
            gemvQ8Stage(hidden2, weights + v_off, v_buf, kvd, n_embd);
            gridSync(&sync_ctrs[sync_idx % n_sync_slots], n_blocks);
            gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
            sync_idx += 1;

            // RoPE on Q and K
            ropeStage(q_buf, n_head, head_dim, rope_dim, rope_theta, seq_pos, n_blocks);
            ropeStage(k_buf, n_kv, head_dim, rope_dim, rope_theta, seq_pos, n_blocks);
            gridSync(&sync_ctrs[sync_idx % n_sync_slots], n_blocks);
            gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
            sync_idx += 1;

            // NOTE: SDPA (attention scores + softmax + V accumulation) is Phase 2.
            // For now, output projection uses q_buf as placeholder input.

            // Output projection
            const out_off = readLayerOffset(layer_offsets, li, off_attn_output);
            gemvQ8Stage(q_buf, weights + out_off, hidden2, n_embd, n_head * head_dim);
            gridSync(&sync_ctrs[sync_idx % n_sync_slots], n_blocks);
            gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
            sync_idx += 1;
        } else {
            // DeltaNet SSM layer — too complex for GPU megakernel (Phase 2).
            // Sequential recurrence has data dependencies that don't parallelize.
        }

        // ── 3. Post-attention norm (FFN pre-norm) ────────────────
        const post_norm_off = readLayerOffset(layer_offsets, li, off_post_attn_norm);
        const post_norm_w: [*]const f32 = @ptrCast(@alignCast(weights + post_norm_off));
        addRmsNormStage(
            hidden,
            hidden2,
            post_norm_w,
            hidden2,
            ss_scratch,
            &sync_ctrs[sync_idx % n_sync_slots],
            n_embd,
            n_blocks,
            rms_eps,
        );
        sync_idx += 1;

        // ── 4. FFN: gate + up + SiLU*mul ─────────────────────────
        // Gate GEMV: ff_gate = W_gate @ hidden2
        const gate_off = readLayerOffset(layer_offsets, li, off_ffn_gate);
        gemvQ8Stage(hidden2, weights + gate_off, ff_gate, n_ff, n_embd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots], n_blocks);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        // Up GEMV: ff_up = W_up @ hidden2
        const up_off = readLayerOffset(layer_offsets, li, off_ffn_up);
        gemvQ8Stage(hidden2, weights + up_off, ff_up, n_ff, n_embd);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots], n_blocks);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        // SiLU activation: ff_gate = silu(ff_gate) * ff_up
        siluMulStage(ff_gate, ff_up, n_ff, n_blocks);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots], n_blocks);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;

        // ── 5. FFN down projection ───────────────────────────────
        // hidden2 = W_down @ ff_gate
        const down_off = readLayerOffset(layer_offsets, li, off_ffn_down);
        gemvQ8Stage(ff_gate, weights + down_off, hidden2, n_embd, n_ff);
        gridSync(&sync_ctrs[sync_idx % n_sync_slots], n_blocks);
        gridSyncReset(&sync_ctrs[sync_idx % n_sync_slots]);
        sync_idx += 1;
    }

    // ── Final: fuse last FFN residual into hidden ────────────────
    addStage(hidden, hidden2, n_embd, n_blocks);
}
