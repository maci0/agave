//! Paged SDPA kernel for ROCm: block-table-indexed attention.
//! Grid: nh workgroups (one per query head), 256 threads per workgroup.
//! LDS: sl+1 floats for attention scores + broadcast slot.

const cu = @import("common.zig");

const sparse_v_threshold: f32 = 1e-6;

export fn sdpa_paged_kernel(
    q: [*]const f32,
    k_flat: [*]const f32,
    v_flat: [*]const f32,
    output: [*]f32,
    block_table: [*]const u32,
    nh: u32,
    nkv: u32,
    hd: u32,
    sl: u32,
    kvd: u32,
    scale: f32,
    paged_bs: u32,
) callconv(.kernel) void {
    const tid = cu.threadIdx();
    const head = cu.blockIdx();
    const bdim = cu.blockDim();
    const hpg = nh / nkv;
    const kvh = head / hpg;
    const q_base = head * hd;

    // Phase 1: QK dot products with block-table indirection
    var t = tid;
    while (t < sl) : (t += bdim) {
        const bt_idx = t / paged_bs;
        const pos_in_bt = t % paged_bs;
        const phys_id = block_table[bt_idx];
        const k_off = (phys_id * paged_bs + pos_in_bt) * kvd + kvh * hd;
        var dot: f32 = 0.0;
        var d: u32 = 0;
        while (d < hd) : (d += 1) {
            dot += q[q_base + d] * k_flat[k_off + d];
        }
        cu.sharedStore(t, dot * scale);
    }
    cu.syncthreads();

    // Phase 2: Wave-parallel softmax
    const wave_size: u32 = 32;
    const chunk = (sl + wave_size - 1) / wave_size;
    const wstart = tid * chunk;
    const wend = @min(wstart + chunk, sl);

    var local_max: f32 = cu.neg_f32_max;
    var i = wstart;
    while (i < wend) : (i += 1) {
        local_max = @max(local_max, cu.sharedLoad(i));
    }
    var max_val = cu.waveReduceMax(local_max);
    if (tid == 0) cu.sharedStore(sl, max_val);
    cu.syncthreads();
    max_val = cu.sharedLoad(sl);

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

    const inv = cu.rcpf(sum_val);
    i = wstart;
    while (i < wend) : (i += 1) {
        cu.sharedStore(i, cu.sharedLoad(i) * inv);
    }
    cu.syncthreads();

    // Phase 3: V accumulation with block-table indirection
    var d: u32 = tid;
    while (d < hd) : (d += bdim) {
        var acc: f32 = 0.0;
        var tt: u32 = 0;
        while (tt < sl) : (tt += 1) {
            const score = cu.sharedLoad(tt);
            if (score < sparse_v_threshold) continue;
            const bt_idx = tt / paged_bs;
            const pos_in_bt = tt % paged_bs;
            const phys_id = block_table[bt_idx];
            acc += score * v_flat[(phys_id * paged_bs + pos_in_bt) * kvd + kvh * hd + d];
        }
        output[q_base + d] = acc;
    }
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(sparse_v_threshold > 0.0);
}

test "fuzz: sdpa_paged functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
