//! GEMV HQQ 4-bit kernel: y[row] = dot(dequant(w_q[row,:]), x)
//! HQQ format: w_q uint8 [n_out, k_in/2] — low nibble = even k, high nibble = odd k.
//! scale and zero: bf16, shape [n_out, k_in/group_size].
//! Dequant: w = (nibble - zero) * scale.
//! NR=4: Launch with ceil(n/4) blocks, each block processes 4 output rows.

const cu = @import("common.zig");

const nr: u32 = 4;
const sparse_threshold: f32 = 0.005;

export fn gemv_hqq_kernel(
    x: [*]const f32,
    w_q: [*]const u8,
    scale: [*]const u8,
    zero: [*]const u8,
    y: [*]f32,
    n: u32,
    k: u32,
    group_size: u32,
) callconv(.nvptx_device) void {
    const row_base = cu.blockIdx() * nr;
    if (row_base >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    // Number of packed bytes per output row (2 elements per byte).
    const bytes_per_row = (k + 1) / 2;
    // Number of quantization groups per output row.
    const n_groups = (k + group_size - 1) / group_size;

    var sums: [nr]f32 = [_]f32{0.0} ** nr;

    // Each thread iterates over a strided slice of k-groups.
    var gi = tid;
    while (gi < n_groups) : (gi += bdim) {
        const elem_base = gi * group_size;
        const elems = @min(group_size, k - elem_base);

        // Sparse skip: if no x value in this group is significant, skip.
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

            // Load bf16 scale and zero for this row/group.
            const sc_bits: u16 = @as(*const u16, @ptrCast(@alignCast(scale + (row * n_groups + gi) * 2))).*;
            const zr_bits: u16 = @as(*const u16, @ptrCast(@alignCast(zero + (row * n_groups + gi) * 2))).*;
            const s = cu.bf16ToF32(sc_bits);
            const z = cu.bf16ToF32(zr_bits);

            var row_sum: f32 = 0.0;
            var ki: u32 = 0;
            while (ki < elems) : (ki += 2) {
                const k_abs = elem_base + ki;
                const byte_idx = (row * bytes_per_row) + k_abs / 2;
                const byte = w_q[byte_idx];

                // Even k: low nibble.
                const q0: f32 = @floatFromInt(byte & 0xF);
                row_sum += (q0 - z) * s * x[k_abs];

                // Odd k: high nibble (only if within bounds).
                if (ki + 1 < elems) {
                    const q1: f32 = @floatFromInt(byte >> 4);
                    row_sum += (q1 - z) * s * x[k_abs + 1];
                }
            }
            sums[r] += row_sum;
        }
    }

    // Warp reduce and write output.
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
    comptime std.debug.assert(nr > 0);
    comptime std.debug.assert(sparse_threshold > 0.0);
}

test "fuzz: gemv_hqq functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
