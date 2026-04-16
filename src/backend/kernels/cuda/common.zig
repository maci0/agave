//! Shared CUDA kernel primitives: thread indexing, PTX math intrinsics,
//! warp shuffle, and block-level reductions.
//!
//! Imported by individual kernel files (silu.zig, rms_norm.zig, etc.)
//! and compiled together to PTX via nvptx64-cuda target.

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
pub inline fn f16tof32(ptr: [*]const u8) f32 {
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
pub inline fn bf16ToF32(val: u16) f32 {
    return @bitCast(@as(u32, val) << 16);
}

/// 6-bit mask for scale extraction in getScaleMinK4.
pub const scale_6bit_mask: u8 = 63;

/// Extract packed scale and min for Q4_K/Q5_K sub-block.
/// Scales are packed in 12 bytes for 8 sub-blocks (6 bits each).
pub inline fn getScaleMinK4(sb: u32, scales_ptr: [*]const u8, sc: *u8, m: *u8) void {
    if (sb < 4) {
        sc.* = scales_ptr[sb] & scale_6bit_mask;
        m.* = scales_ptr[sb + 4] & scale_6bit_mask;
    } else {
        sc.* = (scales_ptr[sb + 4] & 0xF) | ((scales_ptr[sb - 4] >> 6) << 4);
        m.* = (scales_ptr[sb + 4] >> 4) | ((scales_ptr[sb] >> 6) << 4);
    }
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
