//! GEMV Q6_K kernel: y[row] = dot(W_q6k[row,:], x)
//! Q6_K block: 210 bytes = ql[128] + qh[64] + scales[16] + d(f16).
//! 256 values per block.
//! 6-bit values: 4 low bits in ql[], 2 high bits in qh[].
//! NR=2: Launch with ceil(n/2) workgroups, each processes 2 output rows.

const cu = @import("common.zig");

/// Bytes per Q6_K block (256 elements).
const q6_k_block_size: u32 = 210;
/// Elements per Q6_K block.
const q6_k_group_size: u32 = 256;
/// Number of output rows per workgroup.
const nr: u32 = 2;

// Q6_K block layout: [ql(128)] [qh(64)] [scales(16)] [d(2)] = 210 bytes.
// Matches GGUF canonical layout (d at end, same as CUDA/CPU kernels).
const q6_k_ql_offset: u32 = 0; // ql starts at offset 0
const q6_k_qh_offset: u32 = 128; // qh starts after ql(128)
const q6_k_sc_offset: u32 = 192; // scales starts after ql(128) + qh(64)
const q6_k_d_offset: u32 = 208; // d(f16) at end of block

/// Q6_K dequant bias: 6-bit unsigned [0..63] centered to signed [-32..31].
const q6_k_dequant_bias: i32 = -32;
/// Mask for extracting 2-bit high-order field from qh byte.
const qh_2bit_mask: u8 = 0x03;

/// Elements per half super-block: the layout interleaves in 128-element halves.
const q6_k_half_elems: u32 = 128;
/// ql bytes per half (128 elements, 4 bits each, but read as two nibble planes).
const q6_k_ql_half_bytes: u32 = 64;
/// qh bytes per half (128 elements, 2 bits each).
const q6_k_qh_half_bytes: u32 = 32;
/// scale bytes per half.
const q6_k_sc_half_bytes: u32 = 8;

/// Compute one super-block's dot product for a single row.
///
/// Q6_K is NOT a sequential nibble stream. GGML packs each 128-element half so
/// that byte `ql[l]` carries elements `l` (low nibble) and `l + 64` (high
/// nibble), byte `ql[l + 32]` carries `l + 32` and `l + 96`, and `qh[l]` carries
/// the two high bits of all four at shifts 0/2/4/6. Scales are indexed
/// `sc[l/16 + {0,2,4,6}]` for those four. Decoding it as consecutive nibbles
/// reads the right bytes in the wrong order and produces plausible-looking
/// garbage; see `kernels/cpu/gemv_q6_k.zig` for the same mapping.
inline fn q6kBlockDot(x: [*]const f32, blk_addr: usize, k: u32, base_col: u32) f32 {
    const d: f32 = @floatCast(@as(f16, @bitCast(@as(
        *align(1) const u16,
        @ptrFromInt(blk_addr + q6_k_d_offset),
    ).*)));

    var sum: f32 = 0.0;
    var half: u32 = 0;
    while (half < 2) : (half += 1) {
        const ql = @as([*]const u8, @ptrFromInt(blk_addr + q6_k_ql_offset + half * q6_k_ql_half_bytes));
        const qh = @as([*]const u8, @ptrFromInt(blk_addr + q6_k_qh_offset + half * q6_k_qh_half_bytes));
        const sc = @as([*]const i8, @ptrFromInt(blk_addr + q6_k_sc_offset + half * q6_k_sc_half_bytes));
        const base = base_col + half * q6_k_half_elems;
        if (base >= k) break;

        var l: u32 = 0;
        while (l < 32) : (l += 1) {
            const is = l / 16;
            const h = qh[l];
            const lo = ql[l];
            const hi = ql[l + 32];

            const q1: i32 = @as(i32, (lo & 0x0F) | ((h >> 0) & qh_2bit_mask) << 4) + q6_k_dequant_bias;
            const q2: i32 = @as(i32, (hi & 0x0F) | ((h >> 2) & qh_2bit_mask) << 4) + q6_k_dequant_bias;
            const q3: i32 = @as(i32, (lo >> 4) | ((h >> 4) & qh_2bit_mask) << 4) + q6_k_dequant_bias;
            const q4: i32 = @as(i32, (hi >> 4) | ((h >> 6) & qh_2bit_mask) << 4) + q6_k_dequant_bias;

            // The final super-block of a k that is not a multiple of 256 is
            // padded on disk but x stops at k, so each quarter is guarded.
            const e1 = base + l;
            if (e1 < k) sum += d * @as(f32, @floatFromInt(sc[is + 0])) * @as(f32, @floatFromInt(q1)) * x[e1];
            const e2 = base + l + 32;
            if (e2 < k) sum += d * @as(f32, @floatFromInt(sc[is + 2])) * @as(f32, @floatFromInt(q2)) * x[e2];
            const e3 = base + l + 64;
            if (e3 < k) sum += d * @as(f32, @floatFromInt(sc[is + 4])) * @as(f32, @floatFromInt(q3)) * x[e3];
            const e4 = base + l + 96;
            if (e4 < k) sum += d * @as(f32, @floatFromInt(sc[is + 6])) * @as(f32, @floatFromInt(q4)) * x[e4];
        }
    }
    return sum;
}

export fn gemv_q6_k_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row_base = cu.blockIdx() * nr;
    if (row_base >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const nr_active = @min(nr, n - row_base);

    const blocks_per_row = (k + q6_k_group_size - 1) / q6_k_group_size;
    const row_bytes = blocks_per_row * q6_k_block_size;

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;

    const sparse_threshold: f32 = 0.005;
    var blk = tid;
    while (blk < blocks_per_row) : (blk += bdim) {
        const base_col = blk * q6_k_group_size;

        // Sparse skip: check if all 256 input values are near-zero
        var bmax: f32 = 0.0;
        const check_end = @min(base_col + q6_k_group_size, k);
        for (base_col..check_end) |i| {
            const a = @abs(x[i]);
            if (a > bmax) bmax = a;
        }
        if (bmax < sparse_threshold) continue;

        // Row 0
        sum0 += q6kBlockDot(x, @intFromPtr(w) + row_base * row_bytes + blk * q6_k_block_size, k, base_col);

        // Row 1 (if active)
        if (nr_active > 1)
            sum1 += q6kBlockDot(x, @intFromPtr(w) + (row_base + 1) * row_bytes + blk * q6_k_block_size, k, base_col);
    }

    sum0 = cu.blockReduceAdd(sum0);
    if (tid == 0) y[row_base] = sum0;

    if (nr_active > 1) {
        cu.syncthreads();
        sum1 = cu.blockReduceAdd(sum1);
        if (tid == 0) y[row_base + 1] = sum1;
    }
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(q6_k_block_size > 0);
    comptime std.debug.assert(q6_k_group_size > 0);
    comptime std.debug.assert(nr > 0);
    comptime std.debug.assert(q6_k_ql_offset == 0);
    comptime std.debug.assert(q6_k_qh_offset > 0);
    comptime std.debug.assert(q6_k_sc_offset > 0);
    comptime std.debug.assert(q6_k_d_offset > 0);
}

test "fuzz: gemv_q6_k functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = &q6kBlockDot;
            }
        }
    }.f, .{});
}
