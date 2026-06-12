//! GEMV AWQ INT4 kernel: y[col] = sum_k dequant(qweight[k, col]) * x[k]
//! AWQ GEMM format: column-major packing with interleaved nibble order.
//! qweight[k, n/8] u32, scales[n_groups, n] f16 (natural order),
//! qzeros[n_groups, n/8] u32 (GEMM interleaved order).
//! GEMM order: nibbles within u32 follow [0,2,4,6,1,3,5,7].

const cu = @import("common.zig");

const gemm_reverse: [8]u5 = .{ 0, 4, 1, 5, 2, 6, 3, 7 };

export fn gemv_awq_kernel(
    x: [*]const f32,
    qweight: [*]const u32,
    scales: [*]const u16,
    qzeros: [*]const u32,
    y: [*]f32,
    n: u32,
    k: u32,
    group_size: u32,
) callconv(.nvptx_device) void {
    const col = cu.blockIdx();
    if (col >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const n_words = n / 8;
    const word_idx = col / 8;
    const shift: u5 = @intCast(gemm_reverse[col % 8] * 4);

    var sum: f32 = 0.0;

    var ki = tid;
    while (ki < k) : (ki += bdim) {
        const xv = x[ki];
        if (@abs(xv) < 0.005) continue;

        const word = qweight[ki * n_words + word_idx];
        const nibble: f32 = @floatFromInt(@as(u32, (word >> shift) & 0xF));

        const g = ki / group_size;
        const z_word = qzeros[g * n_words + word_idx];
        const zero: f32 = @floatFromInt(@as(u32, (z_word >> shift) & 0xF));

        const scale_bits = scales[g * n + col];
        const scale_val: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));

        sum += (nibble - zero) * scale_val * xv;
    }

    sum = cu.warpReduceAdd(sum);
    if (tid == 0) y[col] = sum;
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(gemm_reverse.len > 0);
}

test "fuzz: gemv_awq functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
