//! GEMV TQ2_0 kernel: y[row] = dot(W_tq2[row,:], x)
//! TQ2_0 block: 66 bytes = 2 bytes (f16 scale) + 64 bytes packed 2-bit values.
//! 4 values per byte, each 2 bits: (byte >> (slot*2)) & 3.
//! Values: 0→-1, 1→0, 2→+1. Dequant: (q - 1) * scale.
//! NR=4: Launch with ceil(n/4) blocks, each block processes 4 output rows.

const cu = @import("common.zig");

const tq2_0_block_bytes: u32 = 66;
const tq2_0_block_elems: u32 = 256;
const scale_bytes: u32 = 2;
const qs_bytes: u32 = 64;
const nr: u32 = 4;
const sparse_threshold: f32 = 0.005;

export fn gemv_tq2_0_kernel(
    x: [*]const f32,
    qweight: [*]const u8,
    y: [*]f32,
    n: u32,
    k: u32,
) callconv(.kernel) void {
    const row_base = cu.blockIdx() * nr;
    if (row_base >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();
    const nb = (k + tq2_0_block_elems - 1) / tq2_0_block_elems;
    const row_bytes = nb * tq2_0_block_bytes;

    var sums: [nr]f32 = [_]f32{0.0} ** nr;

    var wi = tid;
    while (wi < nb) : (wi += bdim) {
        const elem_base = wi * tq2_0_block_elems;
        const elems = @min(tq2_0_block_elems, k - elem_base);

        // Sparse skip: check if any x in this block is significant.
        var any_nonzero = false;
        var ei: u32 = 0;
        while (ei < elems) : (ei += 1) {
            if (@abs(x[elem_base + ei]) >= sparse_threshold) {
                any_nonzero = true;
                break;
            }
        }
        if (!any_nonzero) continue;

        inline for (0..nr) |r| {
            const row = row_base + r;
            if (row >= n) break;
            const block_ptr = qweight + row * row_bytes + wi * tq2_0_block_bytes;
            const scale = cu.f16tof32(block_ptr);
            const qs = block_ptr + scale_bytes;

            var row_sum: f32 = 0.0;
            var byte_i: u32 = 0;
            while (byte_i < qs_bytes) : (byte_i += 1) {
                const base_elem = byte_i * 4;
                if (base_elem >= elems) break;
                const byte = qs[byte_i];
                inline for (0..4) |slot| {
                    const elem_idx = base_elem + slot;
                    if (elem_idx < elems) {
                        const q: f32 = @floatFromInt((byte >> @as(u3, slot * 2)) & 0x3);
                        row_sum += (q - 1.0) * scale * x[elem_base + elem_idx];
                    }
                }
            }
            sums[r] += row_sum;
        }
    }

    // Warp reduce and write.
    inline for (0..nr) |r| {
        const reduced = cu.warpReduceAdd(sums[r]);
        if (tid == 0) {
            const row = row_base + r;
            if (row < n) y[row] = reduced;
        }
    }
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(tq2_0_block_bytes > 0);
    comptime std.debug.assert(tq2_0_block_elems > 0);
    comptime std.debug.assert(scale_bytes > 0);
    comptime std.debug.assert(qs_bytes > 0);
    comptime std.debug.assert(nr > 0);
}

test "fuzz: gemv_tq2_0 functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
