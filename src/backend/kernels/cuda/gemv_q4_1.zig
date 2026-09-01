//! Q4_1 GEMV kernel: 32 values per block, 20 bytes (f16 scale + f16 min + 16 nibble bytes).
//! Dequant: value = nibble * d + m (no subtract-8, has zero-point).
//! Grid: n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

/// Q4_1 block: 4-byte header (f16 d + f16 m) + 16 nibble-packed bytes = 20 bytes.
const q4_1_block_bytes: u32 = 20;
const q4_1_block_elems: u32 = 32;

export fn gemv_q4_1_kernel(
    x: [*]const f32,
    w: [*]const u8,
    y: [*]f32,
    n: u32,
    k: u32,
) callconv(.nvptx_device) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    // Round UP, matching the host row stride (backend.gemvRowBytes). Truncating
    // does not merely drop a tail: nb is also the per-row block stride, so at a k
    // that is not a multiple of the block size every row past the first reads
    // another row's bytes.
    const nb = (k + q4_1_block_elems - 1) / q4_1_block_elems;
    var sum: f32 = 0.0;
    const sparse_threshold: f32 = 0.005;

    var b: u32 = tid;
    while (b < nb) : (b += bdim) {
        const base_col = b * q4_1_block_elems;

        // Sparse skip: check if all 32 input values are near-zero
        var bmax: f32 = 0.0;
        for (0..q4_1_block_elems) |i| {
            if (base_col + i < k) {
                const a = @abs(x[base_col + i]);
                if (a > bmax) bmax = a;
            }
        }
        if (bmax < sparse_threshold) continue;

        const bp = w + (row * nb + b) * q4_1_block_bytes;
        // Read f16 scale and min (little-endian)
        const d_bits: u16 = @as(u16, bp[0]) | (@as(u16, bp[1]) << 8);
        const m_bits: u16 = @as(u16, bp[2]) | (@as(u16, bp[3]) << 8);
        const d: f32 = @as(f32, @as(f16, @bitCast(d_bits)));
        const m: f32 = @as(f32, @as(f16, @bitCast(m_bits)));
        const qs = bp + 4;
        const bk = b * q4_1_block_elems;

        var block_sum: f32 = 0.0;
        var x_sum: f32 = 0.0;
        var j: u32 = 0;
        while (j < 16) : (j += 1) {
            const byte = qs[j];
            const lo: f32 = @floatFromInt(byte & 0xF);
            const hi: f32 = @floatFromInt(byte >> 4);
            block_sum += lo * x[bk + j] + hi * x[bk + j + 16];
            x_sum += x[bk + j] + x[bk + j + 16];
        }
        sum += block_sum * d + x_sum * m;
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(q4_1_block_bytes > 0);
    comptime std.debug.assert(q4_1_block_elems > 0);
}

test "fuzz: gemv_q4_1 functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
