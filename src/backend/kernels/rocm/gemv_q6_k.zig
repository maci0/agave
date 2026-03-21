//! GEMV Q6_K kernel: y[row] = dot(W_q6k[row,:], x)
//! Q6_K block: 210 bytes = d(f16) + ql[128] + qh[64] + scales[16].
//! 256 values per block.
//! 6-bit values: 4 low bits in ql[], 2 high bits in qh[].

const cu = @import("common.zig");

/// Bytes per Q6_K block (256 elements).
const q6_k_block_size: u32 = 210;
/// Elements per Q6_K block.
const q6_k_group_size: u32 = 256;

export fn gemv_q6_k_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const blocks_per_row = (k + q6_k_group_size - 1) / q6_k_group_size;
    const row_bytes = blocks_per_row * q6_k_block_size;
    const row_start = @intFromPtr(w) + row * row_bytes;

    var sum: f32 = 0.0;
    var blk = tid;
    while (blk < blocks_per_row) : (blk += bdim) {
        const blk_addr = row_start + blk * q6_k_block_size;

        // Load block header: d (f16)
        const d: f32 = @floatCast(@as(f16, @bitCast(@as(
            *align(1) const u16,
            @ptrFromInt(blk_addr),
        ).*)));

        const ql = @as([*]const u8, @ptrFromInt(blk_addr + 2));
        const qh = @as([*]const u8, @ptrFromInt(blk_addr + 130));
        const scales = @as([*]const i8, @ptrFromInt(blk_addr + 194));

        const base_col = blk * q6_k_group_size;

        // Process 16 sub-blocks of 16 elements each
        for (0..16) |sb| {
            const sub_base = base_col + sb * 16;
            if (sub_base >= k) break;

            const scale: f32 = @floatFromInt(scales[sb]);
            const d_sc = d * scale;

            // Each sub-block: 16 elements = 8 bytes in ql + 4 bytes in qh
            const ql_off = sb * 8;
            const qh_off = sb * 4;

            var sub_sum: f32 = 0.0;
            const count = @min(16, k - sub_base);

            for (0..count) |l| {
                // Low 4 bits from ql
                const lo = ql[ql_off + l / 2];
                const lo_nibble: u8 = if (l % 2 == 0) (lo & 0x0F) else (lo >> 4);

                // High 2 bits from qh
                const qh_idx = qh_off + l / 4;
                const shift: u3 = @intCast((l % 4) * 2);
                const hi_bits: u8 = (qh[qh_idx] >> shift) & 0x03;

                const qval: f32 = @floatFromInt(@as(i8, @intCast(lo_nibble | (hi_bits << 4))) - 32);
                sub_sum += x[sub_base + l] * qval;
            }

            sum += d_sc * sub_sum;
        }
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
