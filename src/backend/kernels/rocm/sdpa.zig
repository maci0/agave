//! Scaled Dot-Product Attention kernel (fused QK·softmax·V).
//! Grid: nh workgroups (one per query head), 256 threads per workgroup.
//! LDS: sl+1 floats for attention scores + broadcast slot.
//!
//! Includes TurboQuant variant (sdpa_turbo_kernel) that dequantizes
//! packed KV cache blocks in-register using WHT + Lloyd-Max codebook.

const cu = @import("common.zig");

/// Sparse V threshold: skip V positions with negligible softmax weight.
/// At 1e-6, skipped positions contribute < 0.0001% to the output.
const sparse_v_threshold: f32 = 1e-6;

// ── TurboQuant dequantization helpers ────────────────────────────

/// Lloyd-Max optimal centroids for N(0,1) at 2 bits (4 levels).
const lloyd_max_2bit = [4]f32{ -1.510, -0.453, 0.453, 1.510 };
/// Lloyd-Max optimal centroids for N(0,1) at 3 bits (8 levels).
const lloyd_max_3bit = [8]f32{ -2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152 };
/// Lloyd-Max optimal centroids for N(0,1) at 4 bits (16 levels).
const lloyd_max_4bit = [16]f32{ -2.733, -2.069, -1.618, -1.256, -0.942, -0.657, -0.388, -0.128, 0.128, 0.388, 0.657, 0.942, 1.256, 1.618, 2.069, 2.733 };

/// TurboQuant block size: 32 elements (matches WHT-32 transform).
const turbo_block_elems: u32 = 32;
/// Inverse normalization factor: 1/32 (WHT is self-inverse up to factor 32).
const wht_inv_scale: f32 = 1.0 / 32.0;

/// In-place 32-point Walsh-Hadamard Transform (5-stage butterfly network).
/// WHT is self-inverse up to scale factor 32: WHT(WHT(x)) = 32*x.
fn wht32(buf: *[32]f32) void {
    var stride: u32 = 1;
    while (stride <= 16) : (stride *= 2) {
        var i: u32 = 0;
        while (i < 32) : (i += stride * 2) {
            var j: u32 = 0;
            while (j < stride) : (j += 1) {
                const a = buf[i + j];
                const b = buf[i + j + stride];
                buf[i + j] = a + b;
                buf[i + j + stride] = a - b;
            }
        }
    }
}

/// Dequantize one 32-element TurboQuant block into dst[0..32].
/// block_ptr layout: [f16 norm (2 bytes)] [packed centroid indices].
/// bits: 2, 3, or 4 — selects codebook and unpacking strategy.
///
/// Algorithm: unpack indices -> codebook lookup -> inverse WHT -> rescale.
/// Output = norm/32 * WHT(codebook[indices]).
fn turboDequantBlock(block_ptr: [*]const u8, dst: *[32]f32, bits: u32) void {
    // Read f16 norm from block header (2 bytes, little-endian)
    const norm_bits: u16 = @as(*align(1) const u16, @ptrCast(block_ptr)).*;
    const norm: f32 = @as(f32, @floatCast(@as(f16, @bitCast(norm_bits))));
    if (norm == 0.0) {
        for (0..32) |i| dst[i] = 0.0;
        return;
    }
    const packed = block_ptr + 2;

    // Unpack indices and look up codebook values
    if (bits == 4) {
        // 4-bit: 2 indices per byte (16 bytes for 32 elements)
        for (0..16) |i| {
            const byte = packed[i];
            dst[i * 2] = lloyd_max_4bit[byte & 0xF];
            dst[i * 2 + 1] = lloyd_max_4bit[byte >> 4];
        }
    } else if (bits == 2) {
        // 2-bit: 4 indices per byte (8 bytes for 32 elements)
        for (0..8) |i| {
            const byte = packed[i];
            dst[i * 4] = lloyd_max_2bit[byte & 0x3];
            dst[i * 4 + 1] = lloyd_max_2bit[(byte >> 2) & 0x3];
            dst[i * 4 + 2] = lloyd_max_2bit[(byte >> 4) & 0x3];
            dst[i * 4 + 3] = lloyd_max_2bit[(byte >> 6) & 0x3];
        }
    } else {
        // 3-bit: indices span byte boundaries (12 bytes for 32 elements)
        for (0..32) |i| {
            const bit_pos: u32 = @as(u32, @intCast(i)) * 3;
            const byte_idx = bit_pos / 8;
            const bit_off: u5 = @intCast(bit_pos % 8);
            var val: u32 = @as(u32, packed[byte_idx]) >> bit_off;
            if (bit_off + 3 > 8) {
                val |= @as(u32, packed[byte_idx + 1]) << @as(u5, @intCast(8 - bit_off));
            }
            dst[i] = lloyd_max_3bit[val & 0x7];
        }
    }

    // Inverse WHT + rescale: output = norm/32 * WHT(codebook_values)
    wht32(dst);
    const s = norm * wht_inv_scale;
    for (0..32) |i| dst[i] *= s;
}

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

    // ── Phase 2: Wave-parallel softmax ──────────────────────────
    // Distribute seq_len across first wave (32 threads). Each thread
    // processes a chunk of scores, then wave-reduces max/sum.
    // Matches CUDA warp-parallel pattern for 32× speedup over serial.
    const wave_size: u32 = 32;
    const chunk = (sl + wave_size - 1) / wave_size;
    const wstart = tid * chunk;
    const wend = @min(wstart + chunk, sl);

    // Phase 2a: Wave-parallel max reduction
    var local_max: f32 = cu.neg_f32_max;
    var i = wstart;
    while (i < wend) : (i += 1) {
        local_max = @max(local_max, cu.sharedLoad(i));
    }
    var max_val = cu.waveReduceMax(local_max);
    if (tid == 0) cu.sharedStore(sl, max_val);
    cu.syncthreads();
    max_val = cu.sharedLoad(sl);

    // Phase 2b: Wave-parallel exp and sum
    var local_sum: f32 = 0.0;
    i = wstart;
    while (i < wend) : (i += 1) {
        const e = cu.expf(cu.sharedLoad(i) - max_val);
        cu.sharedStore(i, e);
        local_sum += e;
    }
    var sum_val = cu.waveReduceAdd(local_sum);
    if (tid == 0) cu.sharedStore(sl, sum_val);
    cu.syncthreads();
    sum_val = cu.sharedLoad(sl);

    // Phase 2c: Wave-parallel normalization
    const inv = cu.rcpf(sum_val);
    i = wstart;
    while (i < wend) : (i += 1) {
        cu.sharedStore(i, cu.sharedLoad(i) * inv);
    }
    cu.syncthreads();

    // ── Phase 3: V accumulation ─────────────────────────────────
    var d: u32 = tid;
    while (d < hd) : (d += bdim) {
        var acc: f32 = 0.0;
        var tt: u32 = 0;
        while (tt < sl) : (tt += 1) {
            const score = cu.sharedLoad(tt);
            if (score < sparse_v_threshold) continue; // Sparse V: skip negligible positions
            acc += score * values[tt * kvd + kvh * hd + d];
        }
        output[q_base + d] = acc;
    }
}

// ── TurboQuant SDPA kernel ─────────────────────────────────────────
// Same algorithm as sdpa_kernel but reads K/V from packed turbo blocks
// instead of f32 arrays. Dequantization happens in-register per block.
//
// bits_k/bits_v: 0 = f32 passthrough, 2/3/4 = TurboQuant bit width.
// block_bytes_k/block_bytes_v: byte size per 32-element turbo block.
//
// Mixed types supported: f32-K + turbo-V, turbo-K + f32-V, or both turbo.

export fn sdpa_turbo_kernel(
    q: [*]const f32,
    k_cache: [*]const u8,
    v_cache: [*]const u8,
    output: [*]f32,
    nh: u32,
    nkv: u32,
    hd: u32,
    sl: u32,
    kvd: u32,
    scale: f32,
    bits_k: u32,
    bits_v: u32,
    block_bytes_k: u32,
    block_bytes_v: u32,
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
        var dot: f32 = 0.0;

        if (bits_k == 0) {
            // f32 passthrough
            const k_f32: [*]const f32 = @ptrCast(@alignCast(k_cache));
            const k_off = t * kvd + kvh * hd;
            var d: u32 = 0;
            while (d < hd) : (d += 1) {
                dot += q[q_base + d] * k_f32[k_off + d];
            }
        } else {
            // TurboQuant: dequant 32-element blocks
            const elem_base = t * kvd + kvh * hd;
            const n_turbo_blocks = hd / turbo_block_elems;
            var blk: u32 = 0;
            while (blk < n_turbo_blocks) : (blk += 1) {
                const elem_idx = elem_base + blk * turbo_block_elems;
                const turbo_block_idx = elem_idx / turbo_block_elems;
                const byte_off = turbo_block_idx * block_bytes_k;
                var dequant_buf: [32]f32 = undefined;
                turboDequantBlock(k_cache + byte_off, &dequant_buf, bits_k);
                var d: u32 = 0;
                while (d < 32) : (d += 1) {
                    dot += q[q_base + blk * turbo_block_elems + d] * dequant_buf[d];
                }
            }
        }

        cu.sharedStore(t, dot * scale);
    }
    cu.syncthreads();

    // ── Phase 2: Wave-parallel softmax ──────────────────────────
    const wave_size: u32 = 32;
    const chunk = (sl + wave_size - 1) / wave_size;
    const wstart = tid * chunk;
    const wend = @min(wstart + chunk, sl);

    // Phase 2a: Wave-parallel max reduction
    var local_max: f32 = cu.neg_f32_max;
    var i = wstart;
    while (i < wend) : (i += 1) {
        local_max = @max(local_max, cu.sharedLoad(i));
    }
    var max_val = cu.waveReduceMax(local_max);
    if (tid == 0) cu.sharedStore(sl, max_val);
    cu.syncthreads();
    max_val = cu.sharedLoad(sl);

    // Phase 2b: Wave-parallel exp and sum
    var local_sum: f32 = 0.0;
    i = wstart;
    while (i < wend) : (i += 1) {
        const e = cu.expf(cu.sharedLoad(i) - max_val);
        cu.sharedStore(i, e);
        local_sum += e;
    }
    var sum_val = cu.waveReduceAdd(local_sum);
    if (tid == 0) cu.sharedStore(sl, sum_val);
    cu.syncthreads();
    sum_val = cu.sharedLoad(sl);

    // Phase 2c: Wave-parallel normalization
    const inv = cu.rcpf(sum_val);
    i = wstart;
    while (i < wend) : (i += 1) {
        cu.sharedStore(i, cu.sharedLoad(i) * inv);
    }
    cu.syncthreads();

    // ── Phase 3: V accumulation ─────────────────────────────────
    var d: u32 = tid;
    while (d < hd) : (d += bdim) {
        var acc: f32 = 0.0;
        var tt: u32 = 0;
        while (tt < sl) : (tt += 1) {
            const score = cu.sharedLoad(tt);
            if (score < sparse_v_threshold) continue; // Sparse V: skip negligible positions

            if (bits_v == 0) {
                // f32 passthrough
                const v_f32: [*]const f32 = @ptrCast(@alignCast(v_cache));
                acc += score * v_f32[tt * kvd + kvh * hd + d];
            } else {
                // TurboQuant: dequant the block containing dimension d
                const elem_base = tt * kvd + kvh * hd;
                const blk_idx = d / turbo_block_elems;
                const elem_idx = elem_base + blk_idx * turbo_block_elems;
                const turbo_block_idx = elem_idx / turbo_block_elems;
                const byte_off = turbo_block_idx * block_bytes_v;
                var dequant_buf: [32]f32 = undefined;
                turboDequantBlock(v_cache + byte_off, &dequant_buf, bits_v);
                acc += score * dequant_buf[d % turbo_block_elems];
            }
        }
        output[q_base + d] = acc;
    }
}
