//! Shared AMDGCN kernel primitives: thread indexing, math intrinsics,
//! wave-level operations, and block-level reductions.
//!
//! Imported by individual kernel files (silu.zig, rms_norm.zig, etc.)
//! and compiled together to AMDGCN ELF via amdgcn-amdhsa target.
//!
//! Uses LLVM intrinsics for reliable access to hardware registers
//! (work-item ID, workgroup ID) — no fragile physical register reads.

// ── Thread indexing (LLVM intrinsics) ─────────────────────────────

extern fn @"llvm.amdgcn.workitem.id.x"() u32;
extern fn @"llvm.amdgcn.workgroup.id.x"() u32;

/// Returns the thread index within the current workgroup.
pub fn threadIdx() u32 {
    return @"llvm.amdgcn.workitem.id.x"();
}

/// Returns the workgroup index within the grid.
pub fn blockIdx() u32 {
    return @"llvm.amdgcn.workgroup.id.x"();
}

/// Workgroup size — hardcoded to match host launch configuration.
pub const block_dim: u32 = 256;

/// Negative f32 max: identity element for max-reductions on GPU (no std.math on amdgcn).
pub const neg_f32_max: f32 = -3.4028235e+38;

/// Wave size — hardcoded to wave32 for RDNA3 (gfx1100/gfx1101). CDNA (gfx90a,
/// gfx942) uses wave64; manually change this constant to 64 and rebuild.
/// Not controlled by -Drocm-arch — must be edited in this file.
const wave_size: u32 = 32;

/// Number of waves per workgroup.
const n_waves: u32 = block_dim / wave_size;

/// Returns the workgroup size (compile-time constant).
pub fn blockDim() u32 {
    return block_dim;
}

/// Returns the global thread index: blockIdx * blockDim + threadIdx.
pub fn globalIdx() u32 {
    return blockIdx() * block_dim + threadIdx();
}

// ── Math intrinsics ──────────────────────────────────────────────
// LLVM lowers these to native AMDGCN instructions at ReleaseFast:
//   @exp  → v_exp_f32 (base-2; LLVM inserts x * log2(e) conversion)
//   @sqrt → v_sqrt_f32
//   1/x   → v_rcp_f32 (approximate)

/// exp(x) — LLVM lowers to v_exp_f32 (base-2 with log2(e) multiply)
pub fn expf(x: f32) f32 {
    return @exp(x);
}

/// 1/x — LLVM typically lowers to v_rcp_f32 at ReleaseFast
pub fn rcpf(x: f32) f32 {
    return 1.0 / x;
}

/// 1/sqrt(x) — LLVM typically lowers to v_rsq_f32 at ReleaseFast
pub fn rsqrtf(x: f32) f32 {
    return 1.0 / @sqrt(x);
}

/// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
pub fn tanhf(x: f32) f32 {
    const e2x = @exp(2.0 * x);
    return (e2x - 1.0) / (e2x + 1.0);
}

/// sigmoid(x) = 1 / (1 + exp(-x))
pub fn sigmoidf(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// sin(x) — LLVM lowers to v_sin_f32
pub fn sinf(x: f32) f32 {
    return @sin(x);
}

/// cos(x) — LLVM lowers to v_cos_f32
pub fn cosf(x: f32) f32 {
    return @cos(x);
}

// ── Barrier ──────────────────────────────────────────────────────

/// Synchronize all threads in the workgroup (s_barrier).
pub fn syncthreads() void {
    asm volatile ("s_barrier" ::: .{ .memory = true });
}

// ── Wave-level primitives ────────────────────────────────────────
// RDNA3 (gfx1100) uses wave32 (32 threads per wavefront).
// ds_bpermute_b32 provides cross-lane data exchange (uses LDS hardware internally
// but does not consume programmer-allocated LDS space).

extern fn @"llvm.amdgcn.ds.bpermute"(i32, i32) i32;

/// Wave shuffle down: read val from lane (current_lane + offset) % 32.
/// ds_bpermute_b32 is synchronous within the wave — no barrier needed.
pub fn waveShuffleDown(val: f32, offset: u32) f32 {
    const bits: u32 = @bitCast(val);
    const lane = threadIdx() & (wave_size - 1);
    const src_byte_offset: i32 = @intCast(((lane +% offset) & (wave_size - 1)) << 2);
    const result: i32 = @"llvm.amdgcn.ds.bpermute"(src_byte_offset, @as(i32, @bitCast(bits)));
    return @bitCast(@as(u32, @bitCast(result)));
}

/// Reduce-add across a wave (32 lanes). Result valid in lane 0.
pub fn waveReduceAdd(val: f32) f32 {
    var v = val;
    v += waveShuffleDown(v, 16);
    v += waveShuffleDown(v, 8);
    v += waveShuffleDown(v, 4);
    v += waveShuffleDown(v, 2);
    v += waveShuffleDown(v, 1);
    return v;
}

/// Reduce-max across a wave (32 lanes). Result valid in lane 0.
pub fn waveReduceMax(val: f32) f32 {
    var v = val;
    v = @max(v, waveShuffleDown(v, 16));
    v = @max(v, waveShuffleDown(v, 8));
    v = @max(v, waveShuffleDown(v, 4));
    v = @max(v, waveShuffleDown(v, 2));
    v = @max(v, waveShuffleDown(v, 1));
    return v;
}

// ── LDS (shared memory) ─────────────────────────────────────────
// AMDGCN LDS is addrspace(3). We use a fixed-offset approach:
// the host allocates LDS via the kernel descriptor's
// group_segment_fixed_size field (set by LLVM based on usage).

/// LDS base pointer — addrspace(3) at offset 0.
/// Zig maps addrspace(.shared) to LLVM addrspace(3) = LDS on AMDGCN.
/// Uses allowzero because LDS starts at address 0.
fn ldsBase() [*]allowzero addrspace(.shared) volatile f32 {
    return @ptrFromInt(0);
}

/// Store a value to LDS at the given index.
pub fn sharedStore(idx: u32, val: f32) void {
    ldsBase()[idx] = val;
}

/// Load a value from LDS at the given index.
pub fn sharedLoad(idx: u32) f32 {
    return ldsBase()[idx];
}

// ── Shared format-conversion helpers ────────────────────────────
// Used by multiple GEMV kernels. Defined once here to avoid duplication.

/// E2M1 FP4 → float lookup (OCP Microscaling Spec, shared by NVFP4/MXFP4 kernels).
pub const e2m1_lut = [16]f32{
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

/// BF16 → f32: zero-extend mantissa (shift left 16 bits).
pub inline fn bf16ToF32(val: u16) f32 {
    return @bitCast(@as(u32, val) << 16);
}

/// E8M0 → f32: val = 2^(byte - 127). Pure power-of-2 (no mantissa).
pub inline fn e8m0ToF32(byte: u8) f32 {
    if (byte == 0) return 0.0;
    return @bitCast(@as(u32, byte) << 23);
}

/// 6-bit mask for scale extraction in getScaleMinK4.
pub const scale_6bit_mask: u8 = 63;

/// Extract packed scale and min for Q4_K/Q5_K sub-block.
/// Scales are packed in 12 bytes for 8 sub-blocks (6 bits each).
pub inline fn getScaleMinK4(j: usize, q: [*]const u8, sc: *u8, m: *u8) void {
    if (j < 4) {
        sc.* = q[j] & scale_6bit_mask;
        m.* = q[j + 4] & scale_6bit_mask;
    } else {
        sc.* = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m.* = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// ── FP8 format-conversion helpers ──────────────────────────────
// Shared by gemv_fp8_e4m3, gemv_fp8_e5m2, gemv_nvfp4_st.

/// FP8 E4M3 denormal scale: 2^(-6) / 8 = 2^(-9).
const fp8_e4m3_denorm_scale: f32 = 1.0 / 512.0;
/// E4M3 exponent bias offset for F32 conversion: 127 (F32 bias) - 7 (E4M3 bias) = 120.
const fp8_e4m3_exp_rebias: u32 = 120;

/// Compute FP8 E4M3 → f32 at comptime. Bit layout: seeeemmm. No infinities; e=15,m=7 is NaN.
fn fp8e4m3Compute(val: u8) f32 {
    const sign: u32 = @as(u32, val >> 7) << 31;
    const exp: u32 = (val >> 3) & 0x0F;
    const mant: u32 = val & 0x07;
    if (exp == 0x0F and mant == 0x07) return @bitCast(sign | 0x7FC00000);
    if (exp == 0) {
        if (mant == 0) return @bitCast(sign);
        const fmant: f32 = @floatFromInt(mant);
        const val_abs: f32 = fmant * fp8_e4m3_denorm_scale;
        return @bitCast(sign | @as(u32, @bitCast(val_abs)));
    }
    return @bitCast(sign | ((exp + fp8_e4m3_exp_rebias) << 23) | (mant << 20));
}

/// Precomputed FP8 E4M3 → f32 lookup table (256 entries, built at comptime).
pub const fp8e4m3_lut = blk: {
    var table: [256]f32 = undefined;
    for (0..256) |i| table[i] = fp8e4m3Compute(@intCast(i));
    break :blk table;
};

/// Convert FP8 E4M3 to f32 via lookup table.
pub inline fn fp8e4m3ToF32(val: u8) f32 {
    return fp8e4m3_lut[val];
}

/// FP8 E5M2 denormal scale: 2^(-14) / 4 = 2^(-16).
const fp8_e5m2_denorm_scale: f32 = 1.0 / 65536.0;
/// E5M2 exponent bias offset for F32 conversion: 127 (F32 bias) - 15 (E5M2 bias) = 112.
const fp8_e5m2_exp_rebias: u32 = 112;

/// Compute FP8 E5M2 → f32 at comptime. Bit layout: seeeeemm. Has infinities and NaN.
fn fp8e5m2Compute(val: u8) f32 {
    const sign: u32 = @as(u32, val >> 7) << 31;
    const exp: u32 = (val >> 2) & 0x1F;
    const mant: u32 = val & 0x03;
    if (exp == 0x1F) {
        if (mant == 0) return @bitCast(sign | 0x7F800000);
        return @bitCast(sign | 0x7FC00000);
    }
    if (exp == 0) {
        if (mant == 0) return @bitCast(sign);
        const fmant: f32 = @floatFromInt(mant);
        const val_abs: f32 = fmant * fp8_e5m2_denorm_scale;
        return @bitCast(sign | @as(u32, @bitCast(val_abs)));
    }
    return @bitCast(sign | ((exp + fp8_e5m2_exp_rebias) << 23) | (mant << 21));
}

/// Precomputed FP8 E5M2 → f32 lookup table (256 entries, built at comptime).
pub const fp8e5m2_lut = blk: {
    var table: [256]f32 = undefined;
    for (0..256) |i| table[i] = fp8e5m2Compute(@intCast(i));
    break :blk table;
};

/// Convert FP8 E5M2 to f32 via lookup table.
pub inline fn fp8e5m2ToF32(val: u8) f32 {
    return fp8e5m2_lut[val];
}

// ── Block-level reductions (wave reduction + LDS inter-wave) ────
// Uses wave-level ds_bpermute for intra-wave reduction (no barriers),
// then LDS for inter-wave reduction (1 barrier instead of log2(N)).

/// Block-level reduce-add. All threads must participate.
/// Returns the sum in thread 0.
pub fn blockReduceAdd(val: f32) f32 {
    const tid = threadIdx();
    const lane = tid & (wave_size - 1);
    const wave_id = tid / wave_size;

    // Phase 1: intra-wave reduction (no barrier needed)
    const wave_sum = waveReduceAdd(val);
    if (lane == 0) sharedStore(wave_id, wave_sum);
    syncthreads();

    // Phase 2: inter-wave reduction (only wave 0 participates)
    var result: f32 = if (tid < n_waves) sharedLoad(tid) else 0.0;
    if (wave_id == 0) result = waveReduceAdd(result);

    return result;
}

/// Block-level reduce-max. All threads must participate.
/// Returns the max in thread 0.
pub fn blockReduceMax(val: f32) f32 {
    const tid = threadIdx();
    const lane = tid & (wave_size - 1);
    const wave_id = tid / wave_size;

    const wave_max = waveReduceMax(val);
    if (lane == 0) sharedStore(wave_id, wave_max);
    syncthreads();

    var result: f32 = if (tid < n_waves) sharedLoad(tid) else neg_f32_max;
    if (wave_id == 0) result = waveReduceMax(result);

    return result;
}
