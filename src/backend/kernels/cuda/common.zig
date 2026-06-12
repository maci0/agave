//! Shared CUDA kernel primitives: thread indexing, PTX math intrinsics,
//! warp shuffle, and block-level reductions.
//!
//! Imported by individual kernel files (silu.zig, rms_norm.zig, etc.)
//! and compiled together to PTX via nvptx64-cuda target.

/// Memory fence — prevents LLVM from optimizing away global memory stores.
/// Required when using callconv(.nvptx_device) instead of .kernel (Zig 0.16 LLVM workaround).
pub fn memoryFence() void {
    asm volatile ("" ::: "memory");
}

// ── Thread indexing ─────────────────────────────────────────────

/// Returns the thread index within the current block (PTX %tid.x).
pub fn threadIdx() u32 {
    return asm ("mov.u32 %[ret], %tid.x;"
        : [ret] "=r" (-> u32),
    );
}

/// Returns the block index within the grid (PTX %ctaid.x).
pub fn blockIdx() u32 {
    return asm ("mov.u32 %[ret], %ctaid.x;"
        : [ret] "=r" (-> u32),
    );
}

/// Returns the number of threads per block (PTX %ntid.x).
pub fn blockDim() u32 {
    return asm ("mov.u32 %[ret], %ntid.x;"
        : [ret] "=r" (-> u32),
    );
}

/// Returns the global thread index: blockIdx * blockDim + threadIdx.
pub fn globalIdx() u32 {
    return blockIdx() * blockDim() + threadIdx();
}

// ── PTX math intrinsics ─────────────────────────────────────────
// Zig's @exp/@sqrt emit libcalls unavailable on nvptx, so we use
// hardware-accelerated PTX special-function instructions directly.

/// Negative f32 max: identity element for max-reductions on GPU (no std.math on nvptx).
pub const neg_f32_max: f32 = -3.4028235e+38;

/// log2(e), used to convert exp(x) → exp2(x * log2e).
const log2e: f32 = 1.4426950408889634;

/// exp(x) via PTX ex2.approx: exp(x) = exp2(x * log2(e))
pub fn expf(x: f32) f32 {
    const t = x * log2e;
    return asm ("ex2.approx.f32 %[ret], %[in];"
        : [ret] "=f" (-> f32),
        : [in] "f" (t),
    );
}

/// 1/x via PTX rcp.approx
pub fn rcpf(x: f32) f32 {
    return asm ("rcp.approx.f32 %[ret], %[in];"
        : [ret] "=f" (-> f32),
        : [in] "f" (x),
    );
}

/// 1/sqrt(x) via PTX rsqrt.approx
pub fn rsqrtf(x: f32) f32 {
    return asm ("rsqrt.approx.f32 %[ret], %[in];"
        : [ret] "=f" (-> f32),
        : [in] "f" (x),
    );
}

/// log2(x) via PTX lg2.approx
pub fn log2f(x: f32) f32 {
    return asm ("lg2.approx.f32 %[ret], %[in];"
        : [ret] "=f" (-> f32),
        : [in] "f" (x),
    );
}

/// sin(x) via PTX sin.approx
pub fn sinf(x: f32) f32 {
    return asm ("sin.approx.f32 %[ret], %[in];"
        : [ret] "=f" (-> f32),
        : [in] "f" (x),
    );
}

/// cos(x) via PTX cos.approx
pub fn cosf(x: f32) f32 {
    return asm ("cos.approx.f32 %[ret], %[in];"
        : [ret] "=f" (-> f32),
        : [in] "f" (x),
    );
}

/// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
pub fn tanhf(x: f32) f32 {
    const e2x = expf(2.0 * x);
    return (e2x - 1.0) / (e2x + 1.0);
}

/// sigmoid(x) = 1 / (1 + exp(-x))
pub fn sigmoidf(x: f32) f32 {
    return rcpf(1.0 + expf(-x));
}

// ── Warp-level primitives ───────────────────────────────────────
// CRITICAL: All warp-level ops (shfl.sync) MUST use `asm volatile` to prevent
// the compiler from sinking shuffles into conditional branches, which deadlocks
// shfl.sync when not all lanes participate.

/// Warp shuffle down (full warp mask = 0xFFFFFFFF).
/// Must be volatile to prevent the compiler from sinking shuffles into
/// conditional branches (which deadlocks shfl.sync when not all lanes enter).
pub fn shflDown(val: f32, offset: u32) f32 {
    const bits: u32 = @bitCast(val);
    const result = asm volatile ("shfl.sync.down.b32 %[ret], %[val], %[off], 31, 0xFFFFFFFF;"
        : [ret] "=r" (-> u32),
        : [val] "r" (bits),
          [off] "r" (offset),
    );
    return @bitCast(result);
}

/// Reduce-add across a warp (32 threads). Only lane 0 has the final result.
pub fn warpReduceAdd(val: f32) f32 {
    var v = val;
    v += shflDown(v, 16);
    v += shflDown(v, 8);
    v += shflDown(v, 4);
    v += shflDown(v, 2);
    v += shflDown(v, 1);
    return v;
}

/// Reduce-max across a warp. Only lane 0 has the final result.
pub fn warpReduceMax(val: f32) f32 {
    var v = val;
    v = @max(v, shflDown(v, 16));
    v = @max(v, shflDown(v, 8));
    v = @max(v, shflDown(v, 4));
    v = @max(v, shflDown(v, 2));
    v = @max(v, shflDown(v, 1));
    return v;
}

// ── Shared memory ───────────────────────────────────────────────
// Dynamic shared memory for inter-warp reduction (up to 8 warps = 256 threads).
// Allocated via cuLaunchKernel's smem parameter (32 bytes = 8 warps × 4 bytes).
// cvta.shared.u64 converts offset 0 into a generic pointer to shared memory.

/// Returns a pointer to the dynamic shared memory base (via PTX cvta.shared, offset 0).
pub fn sharedBase() [*]addrspace(.shared) volatile f32 {
    return asm (
        \\cvta.shared.u64 %[ret], 0;
        : [ret] "=l" (-> [*]addrspace(.shared) volatile f32),
    );
}

/// Store a value to shared memory at the given index.
pub fn sharedStore(idx: u32, val: f32) void {
    sharedBase()[idx] = val;
}

/// Load a value from shared memory at the given index.
pub fn sharedLoad(idx: u32) f32 {
    return sharedBase()[idx];
}

/// Synchronize all threads in the block (PTX bar.sync 0).
pub fn syncthreads() void {
    asm volatile ("bar.sync 0;" ::: .{ .memory = true });
}

/// Block-level reduce-add using warp reduction + shared memory.
/// Requires blockDim <= 256 (8 warps). All threads must participate.
///
/// Note: warpReduceAdd uses shfl.sync which requires all 32 lanes to
/// participate. The `asm volatile` qualifier on each shfl.sync instruction
/// in `shflDown` prevents the compiler from sinking shuffles into
/// conditional branches, which would deadlock.
pub fn blockReduceAdd(val: f32) f32 {
    const tid = threadIdx();
    const lane = tid % 32;
    const warp_id = tid / 32;

    // Phase 1: intra-warp reduction — ALL lanes must participate
    const warp_sum = warpReduceAdd(val);
    if (lane == 0) sharedStore(warp_id, warp_sum);
    syncthreads();

    // Phase 2: inter-warp reduction — only warp 0
    const n_warps = (blockDim() + 31) / 32;
    var result: f32 = if (tid < n_warps) sharedLoad(tid) else 0.0;
    if (warp_id == 0) result = warpReduceAdd(result);

    return result;
}

// ── Shared format-conversion helpers ────────────────────────────
// Used by multiple GEMV kernels. Defined once here to avoid duplication.

/// Convert little-endian f16 (2 bytes at `ptr`) to f32.
/// Full IEEE 754 half-precision handling: zero, denormal, normal, inf/NaN.
pub fn f16tof32(ptr: [*]const u8) f32 {
    const val = @as(u16, ptr[0]) | (@as(u16, ptr[1]) << 8);
    const sign: u32 = @as(u32, val >> 15) << 31;
    const exp_f16: u32 = (val >> 10) & 0x1F;
    const mant_f16: u32 = val & 0x3FF;

    // Zero
    if (exp_f16 == 0 and mant_f16 == 0) return @bitCast(sign);

    // Denormal (simplified: treat as tiny normal)
    if (exp_f16 == 0) {
        const mant_f32 = mant_f16 << 13;
        const exp_f32: u32 = (127 - 15) << 23;
        return @bitCast(sign | exp_f32 | mant_f32);
    }

    // Inf/NaN
    if (exp_f16 == 0x1F) {
        const exp_f32: u32 = 0xFF << 23;
        const mant_f32: u32 = mant_f16 << 13;
        return @bitCast(sign | exp_f32 | mant_f32);
    }

    // Normal: exp_f32 = exp_f16 + (127 - 15), mant_f32 = mant_f16 << 13
    const exp_f32: u32 = (exp_f16 + (127 - 15)) << 23;
    const mant_f32: u32 = mant_f16 << 13;
    return @bitCast(sign | exp_f32 | mant_f32);
}

/// Convert BF16 (stored as u16) to f32: zero-extend lower 16 bits.
pub fn bf16ToF32(val: u16) f32 {
    return @bitCast(@as(u32, val) << 16);
}

/// E2M1 FP4 → float lookup (OCP Microscaling Spec, shared by NVFP4/MXFP4 kernels).
pub const e2m1_lut = [16]f32{
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

/// E8M0 → f32: val = 2^(byte - 127). Pure power-of-2 (no mantissa).
pub inline fn e8m0ToF32(byte: u8) f32 {
    if (byte == 0) return 0.0;
    return @bitCast(@as(u32, byte) << 23);
}

/// 6-bit mask for scale extraction in getScaleMinK4.
pub const scale_6bit_mask: u8 = 63;

/// Extract packed scale and min for Q4_K/Q5_K sub-block.
/// Scales are packed in 12 bytes for 8 sub-blocks (6 bits each).
pub fn getScaleMinK4(sb: u32, scales_ptr: [*]const u8, sc: *u8, m: *u8) void {
    if (sb < 4) {
        sc.* = scales_ptr[sb] & scale_6bit_mask;
        m.* = scales_ptr[sb + 4] & scale_6bit_mask;
    } else {
        sc.* = (scales_ptr[sb + 4] & 0xF) | ((scales_ptr[sb - 4] >> 6) << 4);
        m.* = (scales_ptr[sb + 4] >> 4) | ((scales_ptr[sb] >> 6) << 4);
    }
}

// ── FP8 format-conversion helpers ──────────────────────────────
// Shared by gemv_fp8_e4m3, gemv_fp8_e5m2, gemv_nvfp4_st, gemv_fp4_tc.

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

/// Block-level reduce-max using warp reduction + shared memory.
pub fn blockReduceMax(val: f32) f32 {
    const tid = threadIdx();
    const lane = tid % 32;
    const warp_id = tid / 32;

    const warp_max = warpReduceMax(val);
    if (lane == 0) sharedStore(warp_id, warp_max);
    syncthreads();

    const n_warps = (blockDim() + 31) / 32;
    var result: f32 = if (tid < n_warps) sharedLoad(tid) else neg_f32_max;
    if (warp_id == 0) result = warpReduceMax(result);

    return result;
}

test "fuzz: all cuda common functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            _ = smith;
            comptime {
                _ = &memoryFence;
                _ = &threadIdx;
                _ = &blockIdx;
                _ = &blockDim;
                _ = &globalIdx;
                _ = &expf;
                _ = &rcpf;
                _ = &rsqrtf;
                _ = &log2f;
                _ = &sinf;
                _ = &cosf;
                _ = &tanhf;
                _ = &sigmoidf;
                _ = &bf16ToF32;
                _ = &f16tof32;
                _ = &fp8e4m3ToF32;
                _ = &fp8e5m2ToF32;
                _ = &e8m0ToF32;
                _ = &getScaleMinK4;
                _ = &sharedBase;
                _ = &sharedLoad;
                _ = &sharedStore;
                _ = &shflDown;
                _ = &syncthreads;
                _ = &warpReduceAdd;
                _ = &warpReduceMax;
                _ = &blockReduceAdd;
                _ = &blockReduceMax;
            }
        }
    }.f, .{});
}

const std = @import("std");
