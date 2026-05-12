//! Fused FFN kernel: gate GEMV + up GEMV + SiLU*mul for Q5_K weights.
//! Replaces 3 kernel launches per FFN layer with 1.
//!
//! Dispatch: n_ff blocks × 256 threads. Each block computes one output element:
//!   ff_out[blockIdx] = silu(dot(W_gate[blockIdx,:], x)) * dot(W_up[blockIdx,:], x)

const cu = @import("common.zig");
const f16tof32 = cu.f16tof32;
const getScaleMinK4 = cu.getScaleMinK4;

const bytes_per_block: u32 = 176;
const values_per_block: u32 = 256;
const q5_k_high_bit_value: f32 = 16.0;

inline fn q5kBlockDot(x: [*]const f32, bp: [*]const u8, k: u32, block_start: u32) f32 {
    const d = f16tof32(bp);
    const dmin = f16tof32(bp + 2);
    const scales = bp + 4;
    const qh = bp + 16;
    const qs = bp + 48;
    var sum: f32 = 0.0;

    var group: u32 = 0;
    while (group < 4) : (group += 1) {
        const j = group * 64;
        const is = group * 2;
        const shift: u3 = @intCast(group * 2);
        const umask1: u8 = @as(u8, 1) << shift;
        const umask2: u8 = @as(u8, 2) << shift;
        const ql_off = group * 32;

        var sc_a: u8 = undefined;
        var m_a: u8 = undefined;
        var sc_b: u8 = undefined;
        var m_b: u8 = undefined;
        getScaleMinK4(is + 0, scales, &sc_a, &m_a);
        getScaleMinK4(is + 1, scales, &sc_b, &m_b);

        const gi_base = block_start + j;
        if (gi_base >= k) break;

        const d_sc_a = d * @as(f32, @floatFromInt(sc_a));
        const dm_m_a = dmin * @as(f32, @floatFromInt(m_a));
        const d_sc_b = d * @as(f32, @floatFromInt(sc_b));
        const dm_m_b = dmin * @as(f32, @floatFromInt(m_b));

        for (0..32) |l| {
            const gi = gi_base + l;
            if (gi >= k) break;
            const lo: f32 = @floatFromInt(qs[ql_off + l] & 0x0F);
            const hi: f32 = if ((qh[l] & umask1) != 0) q5_k_high_bit_value else 0.0;
            sum += x[gi] * (d_sc_a * (lo + hi) - dm_m_a);
        }

        for (0..32) |l| {
            const gi = gi_base + 32 + l;
            if (gi >= k) break;
            const lo: f32 = @floatFromInt(qs[ql_off + l] >> 4);
            const hi: f32 = if ((qh[l] & umask2) != 0) q5_k_high_bit_value else 0.0;
            sum += x[gi] * (d_sc_b * (lo + hi) - dm_m_b);
        }
    }
    return sum;
}

export fn fused_ffn_gate_up_silu_q5_k_kernel(
    x: [*]const f32,
    w_gate: [*]const u8,
    w_up: [*]const u8,
    ff_out: [*]f32,
    n_ff: u32,
    n_embd: u32,
) callconv(.c) void {
    const row = cu.blockIdx();
    if (row >= n_ff) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const blocks_per_row = (n_embd + values_per_block - 1) / values_per_block;
    const row_bytes = blocks_per_row * bytes_per_block;

    var gate_sum: f32 = 0.0;
    var up_sum: f32 = 0.0;

    var blk = tid;
    while (blk < blocks_per_row) : (blk += bdim) {
        const base_col = blk * values_per_block;
        gate_sum += q5kBlockDot(x, w_gate + row * row_bytes + blk * bytes_per_block, n_embd, base_col);
        up_sum += q5kBlockDot(x, w_up + row * row_bytes + blk * bytes_per_block, n_embd, base_col);
    }

    gate_sum = cu.blockReduceAdd(gate_sum);
    cu.syncthreads();
    up_sum = cu.blockReduceAdd(up_sum);

    if (tid == 0) {
        const silu_gate = gate_sum * cu.rcpf(1.0 + cu.expf(-gate_sum));
        ff_out[row] = silu_gate * up_sum;
    }
}
