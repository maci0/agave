//! Fused FFN kernel: gate GEMV + up GEMV + SiLU*mul in a single dispatch.
//! Replaces 3 kernel launches per FFN layer with 1.
//!
//! Dispatch: n_ff blocks × 256 threads. Each block computes one output element:
//!   ff_out[blockIdx] = silu(dot(W_gate[blockIdx,:], x)) * dot(W_up[blockIdx,:], x)

const cu = @import("common.zig");

/// Bytes per Q8_0 block (32 elements).
const q8_0_block_size: u32 = 34;
/// Elements per Q8_0 block.
const q8_0_group_size: u32 = 32;

/// Compute one Q8_0 block's dot product for a single row.
inline fn q8_0BlockDot(x: [*]const f32, block_ptr: [*]const u8, k: u32, base_col: u32) f32 {
    const scale_bits = @as(u16, block_ptr[0]) | (@as(u16, block_ptr[1]) << 8);
    const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));
    const quants = block_ptr + 2;

    var blk_sum: f32 = 0.0;
    for (0..q8_0_group_size) |qi| {
        if (base_col + qi < k) {
            const q: i8 = @bitCast(quants[qi]);
            blk_sum += @as(f32, @floatFromInt(q)) * x[base_col + qi];
        }
    }
    return scale * blk_sum;
}

/// Fused FFN gate+up+SiLU kernel for Q8_0 weights.
export fn fused_ffn_gate_up_silu_q8_0_kernel(
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

    const blocks_per_row = (n_embd + q8_0_group_size - 1) / q8_0_group_size;
    const row_bytes = blocks_per_row * q8_0_block_size;

    var gate_sum: f32 = 0.0;
    var up_sum: f32 = 0.0;

    var blk = tid;
    while (blk < blocks_per_row) : (blk += bdim) {
        const base_col = blk * q8_0_group_size;
        gate_sum += q8_0BlockDot(x, w_gate + row * row_bytes + blk * q8_0_block_size, n_embd, base_col);
        up_sum += q8_0BlockDot(x, w_up + row * row_bytes + blk * q8_0_block_size, n_embd, base_col);
    }

    gate_sum = cu.blockReduceAdd(gate_sum);
    cu.syncthreads();
    up_sum = cu.blockReduceAdd(up_sum);

    if (tid == 0) {
        // SiLU(gate) * up
        const silu_gate = gate_sum * cu.rcpf(1.0 + cu.expf(-gate_sum));
        ff_out[row] = silu_gate * up_sum;
    }
}
