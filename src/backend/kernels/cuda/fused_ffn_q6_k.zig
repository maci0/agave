//! Fused FFN kernel: gate GEMV + up GEMV + SiLU*mul for Q6_K weights.
//! Replaces 3 kernel launches per FFN layer with 1.
//!
//! Dispatch: n_ff blocks × 256 threads. Each block computes one output element:
//!   ff_out[blockIdx] = silu(dot(W_gate[blockIdx,:], x)) * dot(W_up[blockIdx,:], x)

const cu = @import("common.zig");
const f16tof32 = cu.f16tof32;

const bytes_per_block: u32 = 210;
const values_per_block: u32 = 256;

const q6_k_d_offset: usize = 208;
const q6_k_qh_offset: usize = 128;
const q6_k_sc_offset: usize = 192;
const q6_k_ql_chunk_bytes: usize = 64;
const q6_k_qh_chunk_bytes: usize = 32;
const q6_k_sc_chunk_bytes: usize = 8;
const chunk_elems = values_per_block / 2;
const q6_k_dequant_bias: i8 = -32;
const qh_2bit_mask: u8 = 3;

inline fn q6kBlockDot(x: [*]const f32, bp: [*]const u8, k: u32, block_start: u32) f32 {
    const d = f16tof32(bp + q6_k_d_offset);
    var sum: f32 = 0.0;

    var chunk: u32 = 0;
    while (chunk < 2) : (chunk += 1) {
        const ql = bp + chunk * q6_k_ql_chunk_bytes;
        const qh = bp + q6_k_qh_offset + chunk * q6_k_qh_chunk_bytes;
        const sc: [*]const i8 = @ptrCast(bp + q6_k_sc_offset + chunk * q6_k_sc_chunk_bytes);
        const base = block_start + chunk * chunk_elems;

        for (0..32) |l| {
            const is: usize = l / 16;
            const gi0 = base + l;
            const gi1 = base + l + 32;
            const gi2 = base + l + 64;
            const gi3 = base + l + 96;

            const q1: i8 = @as(i8, @intCast((ql[l] & 0x0F) | ((@as(u8, @truncate(qh[l] >> 0)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
            const q2: i8 = @as(i8, @intCast((ql[l + 32] & 0x0F) | ((@as(u8, @truncate(qh[l] >> 2)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
            const q3: i8 = @as(i8, @intCast((ql[l] >> 4) | ((@as(u8, @truncate(qh[l] >> 4)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;
            const q4: i8 = @as(i8, @intCast((ql[l + 32] >> 4) | ((@as(u8, @truncate(qh[l] >> 6)) & qh_2bit_mask) << 4))) + q6_k_dequant_bias;

            const ds_q1 = d * @as(f32, @floatFromInt(sc[is + 0]));
            const ds_q2 = d * @as(f32, @floatFromInt(sc[is + 2]));
            const ds_q3 = d * @as(f32, @floatFromInt(sc[is + 4]));
            const ds_q4 = d * @as(f32, @floatFromInt(sc[is + 6]));

            if (gi0 < k) sum += x[gi0] * ds_q1 * @as(f32, @floatFromInt(q1));
            if (gi1 < k) sum += x[gi1] * ds_q2 * @as(f32, @floatFromInt(q2));
            if (gi2 < k) sum += x[gi2] * ds_q3 * @as(f32, @floatFromInt(q3));
            if (gi3 < k) sum += x[gi3] * ds_q4 * @as(f32, @floatFromInt(q4));
        }
    }
    return sum;
}

export fn fused_ffn_gate_up_silu_q6_k_kernel(
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
        gate_sum += q6kBlockDot(x, w_gate + row * row_bytes + blk * bytes_per_block, n_embd, base_col);
        up_sum += q6kBlockDot(x, w_up + row * row_bytes + blk * bytes_per_block, n_embd, base_col);
    }

    gate_sum = cu.blockReduceAdd(gate_sum);
    cu.syncthreads();
    up_sum = cu.blockReduceAdd(up_sum);

    if (tid == 0) {
        const silu_gate = gate_sum * cu.rcpf(1.0 + cu.expf(-gate_sum));
        ff_out[row] = silu_gate * up_sum;
    }
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(bytes_per_block > 0);
    comptime std.debug.assert(values_per_block > 0);
    comptime std.debug.assert(q6_k_d_offset > 0);
    comptime std.debug.assert(q6_k_qh_offset > 0);
    comptime std.debug.assert(q6_k_sc_offset > 0);
    comptime std.debug.assert(q6_k_ql_chunk_bytes > 0);
    comptime std.debug.assert(q6_k_qh_chunk_bytes > 0);
    comptime std.debug.assert(q6_k_sc_chunk_bytes > 0);
    comptime std.debug.assert(chunk_elems > 0);
}

test "fuzz: fused_ffn_q6_k functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime { _ = &q6kBlockDot; }
        }
    }.f, .{});
}
