//! Fused FFN kernel: gate GEMV + up GEMV + SiLU*mul for Q6_K weights.

const cu = @import("common.zig");
const gemv_q6k = @import("gemv_q6_k.zig");

const bytes_per_block: usize = 210;
const values_per_block: usize = 256;

export fn fused_ffn_gate_up_silu_q6_k_kernel(
    x: [*]const f32,
    w_gate: [*]const u8,
    w_up: [*]const u8,
    ff_out: [*]f32,
    n_ff: u32,
    n_embd: u32,
) callconv(.kernel) void {
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
        gate_sum += gemv_q6k.q6kBlockDot(x, w_gate + row * row_bytes + blk * bytes_per_block, n_embd, base_col);
        up_sum += gemv_q6k.q6kBlockDot(x, w_up + row * row_bytes + blk * bytes_per_block, n_embd, base_col);
    }

    gate_sum = cu.blockReduceAdd(gate_sum);
    cu.syncthreads();
    up_sum = cu.blockReduceAdd(up_sum);

    if (tid == 0) {
        const silu_gate = gate_sum * cu.rcpf(1.0 + cu.expf(-gate_sum));
        ff_out[row] = silu_gate * up_sum;
    }
}
