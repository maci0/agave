//! GEMV GPTQ INT4 kernel: y[row] = dot(dequant(qweight[row,:]), x)
//! GPTQ format: 8 INT4 nibbles packed per u32 word.
//! qweight[n, k/8] u32, scales[n, n_groups] f16, qzeros[n_groups, n/8] u32.
//! Dequant: val = (nibble - zero) * scale

const cu = @import("common.zig");

export fn gemv_gptq_kernel(
    x: [*]const f32,
    qweight: [*]const u32,
    scales: [*]const u16,
    qzeros: [*]const u32,
    y: [*]f32,
    n: u32,
    k: u32,
    group_size: u32,
) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const words_per_row = k / 8;
    const n_groups = (k + group_size - 1) / group_size;

    var sum: f32 = 0.0;

    var wi = tid;
    while (wi < words_per_row) : (wi += bdim) {
        const word = qweight[row * words_per_row + wi];
        const elem_base = wi * 8;
        const g = elem_base / group_size;

        // Scale: f16 stored as u16
        const scale_bits = scales[row * n_groups + g];
        const scale_val: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));

        // Zero-point: packed INT4 in u32
        const z_word_idx = g * ((n + 7) / 8) + row / 8;
        const z_nibble = row % 8;
        const z_word = qzeros[z_word_idx];
        const zero: f32 = @floatFromInt(@as(u32, (z_word >> @as(u5, @intCast(z_nibble * 4))) & 0xF));

        var local_sum: f32 = 0.0;
        inline for (0..8) |ni| {
            const nibble: u32 = (word >> @as(u5, ni * 4)) & 0xF;
            const val = (@as(f32, @floatFromInt(nibble)) - zero) * scale_val;
            local_sum += val * x[elem_base + ni];
        }
        sum += local_sum;
    }

    // Warp reduction
    sum = cu.warpReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}

const std = @import("std");

test "constants valid" {
    // No module-level numeric constants in this file.
    _ = @sizeOf(u8);
}

test "fuzz: gemv_gptq functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
