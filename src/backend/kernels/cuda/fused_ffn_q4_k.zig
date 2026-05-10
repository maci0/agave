//! Fused FFN kernel: gate GEMV + up GEMV + SiLU*mul for Q4_K weights.
//! Replaces 3 kernel launches per FFN layer with 1.
//!
//! Dispatch: n_ff blocks × 256 threads. Each block computes one output element:
//!   ff_out[blockIdx] = silu(dot(W_gate[blockIdx,:], x)) * dot(W_up[blockIdx,:], x)

const cu = @import("common.zig");
const f16tof32 = cu.f16tof32;
const getScaleMinK4 = cu.getScaleMinK4;

const bytes_per_block: u32 = 144;
const values_per_block: u32 = 256;

inline fn q4kBlockDot(x: [*]const f32, bp: [*]const u8, k: u32, block_start: u32) f32 {
    const d = f16tof32(bp);
    const dmin = f16tof32(bp + 2);
    const scales = bp + 4;
    const qs = bp + 16;
    var sum: f32 = 0.0;

    var g: u32 = 0;
    while (g < 4) : (g += 1) {
        const gi_lo = block_start + g * 64;
        if (gi_lo >= k) break;
        const ql_off = g * 32;

        var sc_lo: u8 = undefined;
        var m_lo: u8 = undefined;
        var sc_hi: u8 = undefined;
        var m_hi: u8 = undefined;
        getScaleMinK4(g * 2, scales, &sc_lo, &m_lo);
        getScaleMinK4(g * 2 + 1, scales, &sc_hi, &m_hi);

        {
            const d_sc = d * @as(f32, @floatFromInt(sc_lo));
            const dm_m = dmin * @as(f32, @floatFromInt(m_lo));
            var q_dot: f32 = 0.0;
            var x_sum: f32 = 0.0;
            for (0..32) |l| {
                const gi = gi_lo + l;
                if (gi >= k) break;
                q_dot += x[gi] * @as(f32, @floatFromInt(qs[ql_off + l] & 0x0F));
                x_sum += x[gi];
            }
            sum += d_sc * q_dot - dm_m * x_sum;
        }

        {
            const d_sc = d * @as(f32, @floatFromInt(sc_hi));
            const dm_m = dmin * @as(f32, @floatFromInt(m_hi));
            var q_dot: f32 = 0.0;
            var x_sum: f32 = 0.0;
            for (0..32) |l| {
                const gi = gi_lo + 32 + l;
                if (gi >= k) break;
                q_dot += x[gi] * @as(f32, @floatFromInt(qs[ql_off + l] >> 4));
                x_sum += x[gi];
            }
            sum += d_sc * q_dot - dm_m * x_sum;
        }
    }
    return sum;
}

export fn fused_ffn_gate_up_silu_q4_k_kernel(
    x: [*]const f32,
    w_gate: [*]const u8,
    w_up: [*]const u8,
    ff_out: [*]volatile f32,
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
        gate_sum += q4kBlockDot(x, w_gate + row * row_bytes + blk * bytes_per_block, n_embd, base_col);
        up_sum += q4kBlockDot(x, w_up + row * row_bytes + blk * bytes_per_block, n_embd, base_col);
    }

    gate_sum = cu.blockReduceAdd(gate_sum);
    cu.syncthreads();
    up_sum = cu.blockReduceAdd(up_sum);

    if (tid == 0) {
        const silu_gate = gate_sum * cu.rcpf(1.0 + cu.expf(-gate_sum));
        ff_out[row] = silu_gate * up_sum;
    }
}
