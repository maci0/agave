//! GEMM Q8_0 kernel: Y[n_tok × n_out] = X[n_tok × n_in] @ W_q8[n_out × n_in]^T
//! One block per output row. Weight blocks loaded once, reused across TILE_T=8 tokens.
//! Launch with n_out blocks of 256 threads.

const cu = @import("common.zig");

const q8_0_block_size: u32 = 34;
const q8_0_group_size: u32 = 32;
const tile_t: u32 = 8;

export fn gemm_q8_0_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n_out: u32, n_in: u32, n_tok: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n_out) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const blocks_per_row = (n_in + q8_0_group_size - 1) / q8_0_group_size;
    const row_bytes = blocks_per_row * q8_0_block_size;

    var t_base: u32 = 0;
    while (t_base < n_tok) : (t_base += tile_t) {
        const t_end = @min(t_base + tile_t, n_tok);
        const nt = t_end - t_base;

        var sums: [tile_t]f32 = .{0} ** tile_t;

        var blk = tid;
        while (blk < blocks_per_row) : (blk += bdim) {
            const block_ptr = w + row * row_bytes + blk * q8_0_block_size;

            // Dequantize weight block once
            const scale_bits = @as(u16, block_ptr[0]) | (@as(u16, block_ptr[1]) << 8);
            const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));
            const quants = block_ptr + 2;
            const base_col = blk * q8_0_group_size;

            // Precompute dequantized weights
            var wvals: [q8_0_group_size]f32 = undefined;
            for (0..q8_0_group_size) |qi| {
                const q: i8 = @bitCast(quants[qi]);
                wvals[qi] = @as(f32, @floatFromInt(q)) * scale;
            }

            // Multiply against each token's input
            for (0..nt) |ti| {
                const x_off = (t_base + @as(u32, @intCast(ti))) * n_in + base_col;
                var blk_sum: f32 = 0.0;
                for (0..q8_0_group_size) |qi| {
                    if (base_col + qi < n_in) {
                        blk_sum += wvals[qi] * x[x_off + qi];
                    }
                }
                sums[ti] += blk_sum;
            }
        }

        // Reduce and write each token's result
        for (0..nt) |ti| {
            const s = cu.blockReduceAdd(sums[ti]);
            if (tid == 0) y[(t_base + @as(u32, @intCast(ti))) * n_out + row] = s;
            if (ti + 1 < nt) cu.syncthreads();
        }
        if (t_base + tile_t < n_tok) cu.syncthreads();
    }
}
