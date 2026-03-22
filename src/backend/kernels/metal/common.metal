#include <metal_stdlib>
using namespace metal;

// Shared reduction: simd_sum within SIMD groups, then cross-group reduce.
// tg_size must be ≤ 256 (max 8 SIMD groups of 32).
// NOTE: result is only valid for tid < 32. If all threads need the result,
// callers must broadcast via shared memory (see deltanet.metal for example).
inline float threadgroup_reduce_sum(float val, threadgroup float* shared, uint tid, uint tg_size) {
    val = simd_sum(val);
    uint simd_lane  = tid % 32;
    uint simd_group = tid / 32;
    if (simd_lane == 0) shared[simd_group] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint num_sg = (tg_size + 31) / 32;
    if (tid < num_sg) val = shared[tid]; else val = 0.0f;
    if (tid < 32) val = simd_sum(val);
    return val;
}

// Shared max reduction: simd_max within SIMD groups, then cross-group reduce.
inline float threadgroup_reduce_max(float val, threadgroup float* shared, uint tid, uint tg_size) {
    val = simd_max(val);
    uint simd_lane  = tid % 32;
    uint simd_group = tid / 32;
    if (simd_lane == 0) shared[simd_group] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint num_sg = (tg_size + 31) / 32;
    if (tid < num_sg) val = shared[tid]; else val = -INFINITY;
    if (tid < 32) val = simd_max(val);
    return val;
}
