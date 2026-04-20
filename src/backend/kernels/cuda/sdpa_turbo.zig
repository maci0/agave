//! TurboQuant SDPA kernel for CUDA — FlashAttention-2 with native KV dequantization.
//!
//! Same algorithm as sdpa.zig but reads K/V from TurboQuant-packed byte buffers
//! instead of f32 arrays. Dequantization (codebook lookup + inverse WHT) happens
//! in registers per 32-element block.
//!
//! Grid: nh blocks (one per query head), 256 threads per block.
//! Dynamic shared memory: sl floats for attention scores + 1 float for broadcast.
//!
//! Parameters bits_k/bits_v: 0 = f32 passthrough, 2/3/4 = TurboQuant.
//! Mixed types supported (f32-K + turbo-V, turbo-K + f32-V, or both turbo).

const cu = @import("common.zig");

/// Sparse V threshold: skip V positions with negligible softmax weight.
/// At 1e-6, skipped positions contribute < 0.0001% to the output.
const sparse_v_threshold: f32 = 1e-6;

// ── TurboQuant constants ────────────────────────────────────────────

/// Lloyd-Max optimal centroids for N(0,1) quantized to 2 bits (4 levels).
const lloyd_max_2bit = [4]f32{ -1.510, -0.453, 0.453, 1.510 };
/// Lloyd-Max optimal centroids for N(0,1) quantized to 3 bits (8 levels).
const lloyd_max_3bit = [8]f32{ -2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152 };
/// Lloyd-Max optimal centroids for N(0,1) quantized to 4 bits (16 levels).
const lloyd_max_4bit = [16]f32{ -2.733, -2.069, -1.618, -1.256, -0.942, -0.657, -0.388, -0.128, 0.128, 0.388, 0.657, 0.942, 1.256, 1.618, 2.069, 2.733 };

/// TurboQuant block size: 32 elements (matches WHT-32 transform).
const turbo_block_size: u32 = 32;

// ── Walsh-Hadamard Transform ────────────────────────────────────────

/// In-place 32-point Walsh-Hadamard Transform (5-stage butterfly network).
/// WHT is self-inverse up to a scale factor of 32: WHT(WHT(x)) = 32*x.
fn wht32(buf: *[32]f32) void {
    var stride: u32 = 1;
    while (stride <= 16) : (stride *= 2) {
        var i: u32 = 0;
        while (i < 32) : (i += stride * 2) {
            var j: u32 = 0;
            while (j < stride) : (j += 1) {
                const a = buf[i + j] + buf[i + j + stride];
                const b = buf[i + j] - buf[i + j + stride];
                buf[i + j] = a;
                buf[i + j + stride] = b;
            }
        }
    }
}

// ── TurboQuant dequantization ───────────────────────────────────────

/// Look up a codebook value by bit width and index.
fn codebookLookup(bits: u32, idx: u32) f32 {
    if (bits == 4) {
        return lloyd_max_4bit[idx & 0xF];
    } else if (bits == 3) {
        return lloyd_max_3bit[idx & 0x7];
    } else {
        return lloyd_max_2bit[idx & 0x3];
    }
}

/// Read a u16 from a byte pointer (unaligned).
fn readU16(ptr: [*]const u8) u16 {
    return @as(u16, ptr[0]) | (@as(u16, ptr[1]) << 8);
}

/// Convert f16 bits to f32.
fn f16ToF32(bits: u16) f32 {
    // Decompose f16: 1 sign + 5 exponent + 10 mantissa
    const sign: u32 = @as(u32, bits >> 15) << 31;
    const exp5: u32 = (bits >> 10) & 0x1F;
    const mant: u32 = bits & 0x3FF;

    if (exp5 == 0) {
        if (mant == 0) {
            return @bitCast(sign); // +-0
        }
        // Denormal f16: normalize
        var m = mant;
        var e: u32 = 1;
        while (m & 0x400 == 0) {
            m <<= 1;
            e += 1;
        }
        const f32_exp: u32 = (127 - 15 + 1 - e) << 23;
        const f32_mant: u32 = (m & 0x3FF) << 13;
        return @bitCast(sign | f32_exp | f32_mant);
    }
    if (exp5 == 31) {
        // Inf/NaN
        return @bitCast(sign | (0xFF << 23) | (mant << 13));
    }
    // Normal f16 → f32: rebias exponent (f16 bias=15, f32 bias=127)
    const f32_exp: u32 = (exp5 + 127 - 15) << 23;
    const f32_mant: u32 = mant << 13;
    return @bitCast(sign | f32_exp | f32_mant);
}

/// Dequantize one 32-element TurboQuant block from packed byte data.
///
/// Block layout: [f16 norm (2 bytes)] [packed centroid indices (bits*32/8 bytes)]
/// Algorithm: unpack indices → codebook lookup → inverse WHT → rescale by norm/32.
fn turboDequantBlock(block_ptr: [*]const u8, dst: *[32]f32, bits: u32) void {
    // Read f16 norm from block header
    const norm_bits = readU16(block_ptr);
    const norm = f16ToF32(norm_bits);
    if (norm == 0.0) {
        var i: u32 = 0;
        while (i < 32) : (i += 1) {
            dst[i] = 0.0;
        }
        return;
    }

    const pack_ptr = block_ptr + 2;

    // Unpack indices and look up codebook values
    if (bits == 4) {
        // 4-bit: 2 indices per byte (16 bytes for 32 elements)
        var i: u32 = 0;
        while (i < 16) : (i += 1) {
            const byte = pack_ptr[i];
            dst[i * 2] = codebookLookup(4, byte & 0xF);
            dst[i * 2 + 1] = codebookLookup(4, byte >> 4);
        }
    } else if (bits == 2) {
        // 2-bit: 4 indices per byte (8 bytes for 32 elements)
        var i: u32 = 0;
        while (i < 8) : (i += 1) {
            const byte = pack_ptr[i];
            dst[i * 4] = codebookLookup(2, byte & 0x3);
            dst[i * 4 + 1] = codebookLookup(2, (byte >> 2) & 0x3);
            dst[i * 4 + 2] = codebookLookup(2, (byte >> 4) & 0x3);
            dst[i * 4 + 3] = codebookLookup(2, (byte >> 6) & 0x3);
        }
    } else {
        // 3-bit: indices span byte boundaries (12 bytes for 32 elements)
        var i: u32 = 0;
        while (i < 32) : (i += 1) {
            const bit_pos = i * 3;
            const byte_idx = bit_pos / 8;
            const bit_off: u5 = @intCast(bit_pos % 8);
            var val: u32 = @as(u32, pack_ptr[byte_idx]) >> bit_off;
            if (bit_off + 3 > 8) {
                val |= @as(u32, pack_ptr[byte_idx + 1]) << @intCast(8 - bit_off);
            }
            dst[i] = codebookLookup(3, val & 0x7);
        }
    }

    // Inverse WHT + rescale: output = norm/32 * WHT(codebook_values)
    wht32(dst);
    const s = norm / 32.0;
    var i: u32 = 0;
    while (i < 32) : (i += 1) {
        dst[i] *= s;
    }
}

// ── SDPA TurboQuant kernel ──────────────────────────────────────────

export fn sdpa_turbo_kernel(
    q: [*]const f32,
    keys: [*]const u8,
    values: [*]const u8,
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
    const smem = cu.sharedBase();

    // ── Phase 1: QK dot products → scores in shared memory ──────
    var t = tid;
    while (t < sl) : (t += bdim) {
        var dot: f32 = 0.0;

        if (bits_k == 0) {
            // f32 passthrough: read K directly as f32
            const f32_keys: [*]const f32 = @ptrCast(@alignCast(keys));
            const k_off = t * kvd + kvh * hd;
            var d: u32 = 0;
            while (d < hd) : (d += 1) {
                dot += q[q_base + d] * f32_keys[k_off + d];
            }
        } else {
            // TurboQuant K: dequant 32-element blocks and dot with Q
            const elem_base = t * kvd + kvh * hd;
            const n_turbo_blocks = hd / turbo_block_size;
            var blk: u32 = 0;
            while (blk < n_turbo_blocks) : (blk += 1) {
                const elem_idx = elem_base + blk * turbo_block_size;
                const turbo_block_idx = elem_idx / turbo_block_size;
                const byte_off = turbo_block_idx * block_bytes_k;
                var dequant_buf: [32]f32 = undefined;
                turboDequantBlock(keys + byte_off, &dequant_buf, bits_k);
                var d: u32 = 0;
                while (d < turbo_block_size) : (d += 1) {
                    dot += q[q_base + blk * turbo_block_size + d] * dequant_buf[d];
                }
            }
        }
        smem[t] = dot * scale;
    }
    cu.syncthreads();

    // ── Phase 2: Warp-parallel softmax ──────────────────────────
    const warp_size = 32;
    const chunk = (sl + warp_size - 1) / warp_size;
    const start = tid * chunk;
    const end = @min(start + chunk, sl);

    // Phase 2a: Warp-parallel max reduction
    var local_max: f32 = cu.neg_f32_max;
    var i = start;
    while (i < end) : (i += 1) {
        local_max = @max(local_max, smem[i]);
    }
    var max_val = cu.warpReduceMax(local_max);
    if (tid == 0) cu.sharedStore(sl, max_val);
    cu.syncthreads();
    max_val = cu.sharedLoad(sl);

    // Phase 2b: Warp-parallel exp and sum
    var local_sum: f32 = 0.0;
    i = start;
    while (i < end) : (i += 1) {
        const e = cu.expf(smem[i] - max_val);
        smem[i] = e;
        local_sum += e;
    }
    var sum_val = cu.warpReduceAdd(local_sum);
    if (tid == 0) cu.sharedStore(sl, sum_val);
    cu.syncthreads();
    sum_val = cu.sharedLoad(sl);

    // Phase 2c: Warp-parallel normalization
    const inv = cu.rcpf(sum_val);
    i = start;
    while (i < end) : (i += 1) {
        smem[i] = smem[i] * inv;
    }
    cu.syncthreads();

    // ── Phase 3: V accumulation ─────────────────────────────────
    var d: u32 = tid;
    while (d < hd) : (d += bdim) {
        var acc: f32 = 0.0;

        if (bits_v == 0) {
            // f32 passthrough: read V directly as f32
            const f32_values: [*]const f32 = @ptrCast(@alignCast(values));
            var tt: u32 = 0;
            while (tt < sl) : (tt += 1) {
                const score = smem[tt];
                if (score < sparse_v_threshold) continue; // Sparse V: skip negligible positions
                acc += score * f32_values[tt * kvd + kvh * hd + d];
            }
        } else {
            // TurboQuant V: dequant the block containing dimension d for each position.
            // Since d is a single dimension within a 32-element turbo block,
            // we need to dequant the whole block and extract dimension d.
            //
            // For each position tt, the element at (tt, kvh*hd + d) is in:
            //   turbo block index = (tt * kvd + kvh * hd + d) / 32
            //   within-block offset = (tt * kvd + kvh * hd + d) % 32
            //
            // Since kvh*hd is constant per head and d is constant per thread,
            // only tt varies. The within-block offset = (kvh*hd + d) % 32
            // is constant across positions if kvd is a multiple of 32 (always true).
            const head_d_off = kvh * hd + d;
            const within_blk = head_d_off % turbo_block_size;
            const blk_idx_base = head_d_off / turbo_block_size;

            var tt: u32 = 0;
            while (tt < sl) : (tt += 1) {
                const score = smem[tt];
                if (score < sparse_v_threshold) continue; // Sparse V: skip negligible positions
                // turbo block index for position tt at this dimension
                const n_blocks_per_pos = kvd / turbo_block_size;
                const turbo_idx = tt * n_blocks_per_pos + blk_idx_base;
                const byte_off = turbo_idx * block_bytes_v;
                var dequant_buf: [32]f32 = undefined;
                turboDequantBlock(values + byte_off, &dequant_buf, bits_v);
                acc += score * dequant_buf[within_blk];
            }
        }
        output[q_base + d] = acc;
    }
}
