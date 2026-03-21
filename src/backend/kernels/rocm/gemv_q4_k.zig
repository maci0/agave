//! GEMV Q4_K kernel: y[row] = dot(W_q4k[row,:], x)
//! Q4_K block: 144 bytes = d(f16) + dmin(f16) + scales[12] + qs[128].
//! 256 values per super-block, 8 sub-blocks of 32 values each.
//! Launch with n workgroups of 256 threads (one row per workgroup).

const cu = @import("common.zig");

/// Extract scale and minimum from Q4_K packed scale byte array.
inline fn getScaleMinK4(j: usize, q: [*]const u8, sc: *u8, m: *u8) void {
    if (j < 4) {
        sc.* = q[j] & 63;
        m.* = q[j + 4] & 63;
    } else {
        sc.* = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m.* = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/// Bytes per Q4_K super-block (256 elements).
const q4_k_block_size: u32 = 144;
/// Elements per Q4_K super-block.
const q4_k_group_size: u32 = 256;

export fn gemv_q4_k_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const blocks_per_row = (k + q4_k_group_size - 1) / q4_k_group_size;
    const row_bytes = blocks_per_row * q4_k_block_size;
    const row_start = @intFromPtr(w) + row * row_bytes;

    var sum: f32 = 0.0;
    var blk = tid;
    while (blk < blocks_per_row) : (blk += bdim) {
        const blk_addr = row_start + blk * q4_k_block_size;

        // Load super-block header: d (f16), dmin (f16), scales[12]
        const d: f32 = @floatCast(@as(f16, @bitCast(@as(
            *align(1) const u16,
            @ptrFromInt(blk_addr),
        ).*)));
        const dmin: f32 = @floatCast(@as(f16, @bitCast(@as(
            *align(1) const u16,
            @ptrFromInt(blk_addr + 2),
        ).*)));

        const scales = @as([*]const u8, @ptrFromInt(blk_addr + 4));
        const qs = @as([*]const u8, @ptrFromInt(blk_addr + 16));

        const base_col = blk * q4_k_group_size;

        // Process 8 sub-blocks of 32 elements each
        for (0..8) |sb| {
            const sub_base = base_col + sb * 32;
            if (sub_base >= k) break;

            // Extract scale and min for this sub-block
            var sc: u8 = undefined;
            var m: u8 = undefined;
            getScaleMinK4(sb, scales, &sc, &m);

            const d_sc = d * @as(f32, @floatFromInt(sc));
            const dm_m = dmin * @as(f32, @floatFromInt(m));

            // Nibble offset: each sub-block uses 16 bytes (32 nibbles)
            const qi_base = sb * 16;

            // Accumulate over 32 elements (16 bytes)
            var sub_sum: f32 = 0.0;
            if (sub_base + 31 < k) {
                // Full sub-block — unroll 4 iterations of 8 elements
                inline for (0..4) |chunk| {
                    const chunk_base = sub_base + chunk * 8;
                    inline for (0..4) |byte_idx| {
                        const byte = qs[qi_base + chunk * 4 + byte_idx];
                        const lo = @as(f32, @floatFromInt(@as(u4, @truncate(byte))));
                        const hi = @as(f32, @floatFromInt(@as(u4, @truncate(byte >> 4))));
                        sub_sum += x[chunk_base + byte_idx * 2] * lo;
                        sub_sum += x[chunk_base + byte_idx * 2 + 1] * hi;
                    }
                }
            } else {
                // Partial sub-block — scalar loop with bounds check
                for (0..32) |l| {
                    const gi = sub_base + l;
                    if (gi >= k) break;
                    const byte_idx = qi_base + l / 2;
                    const nibble: f32 = @floatFromInt(if (l % 2 == 0)
                        @as(u4, @truncate(qs[byte_idx]))
                    else
                        @as(u4, @truncate(qs[byte_idx] >> 4)));
                    sub_sum += x[gi] * nibble;
                }
            }

            sum += d_sc * sub_sum - dm_m * blk_sum_x(x, @intCast(sub_base), @intCast(@min(32, k - sub_base)));
        }
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}

/// Compute sum of x[base..base+count].
inline fn blk_sum_x(x: [*]const f32, base: u32, count: u32) f32 {
    var s: f32 = 0.0;
    for (0..count) |i| s += x[base + i];
    return s;
}
