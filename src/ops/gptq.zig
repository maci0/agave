//! GPTQ quantized GEMV kernel.
//!
//! GPTQ format: INT4 weights packed 8 per u32 word, with per-group
//! FP16 scales and INT4 zero-points (also packed in u32).
//! Dequant: val = (packed_int4 - zero) * scale
//!
//! Supports group_size 32, 64, 128 (most common: 128).

const std = @import("std");
const quant = @import("quant.zig");

const gptq_nibbles_per_u32: usize = 8;

/// GPTQ INT4 GEMV: y[n] = dequant(W[n,k]) @ x[k]
/// qweight: packed INT4 in u32 words, [n, k/8] layout
/// scales: FP16 per group, [n, n_groups] layout
/// qzeros: packed INT4 zero-points in u32, [n_groups, n/8] layout
pub fn gptqGemv(
    x: [*]const f32,
    qweight: [*]const u32,
    scales: [*]const u16,
    qzeros: [*]const u32,
    y: [*]f32,
    n: usize,
    k: usize,
    group_size: u32,
) void {
    gptqGemvRows(x, qweight, scales, qzeros, y, 0, n, n, k, group_size);
}

/// Parallel-friendly GPTQ GEMV over a row range.
/// `n` is the full matrix height (for qzeros row-stride); `start_row`/`n_rows` select the chunk.
pub fn gptqGemvRows(
    x: [*]const f32,
    qweight: [*]const u32,
    scales: [*]const u16,
    qzeros: [*]const u32,
    y: [*]f32,
    start_row: usize,
    n_rows: usize,
    n: usize,
    k: usize,
    group_size: u32,
) void {
    const gs: usize = group_size;
    const n_groups = (k + gs - 1) / gs;
    const words_per_row = k / gptq_nibbles_per_u32;
    const zeros_row_words = (n + gptq_nibbles_per_u32 - 1) / gptq_nibbles_per_u32;

    for (start_row..start_row + n_rows) |row| {
        var sum: f32 = 0.0;
        const w_row = qweight + row * words_per_row;
        const s_row = scales + row * n_groups;

        for (0..n_groups) |g| {
            const scale = f16ToF32(s_row[g]);

            // Extract zero-point for this group+row from packed qzeros
            // qzeros layout: [n_groups, ceil(n/8)] packed INT4 — stride uses full n, not the chunk.
            const z_word_idx = g * zeros_row_words + row / gptq_nibbles_per_u32;
            const z_nibble = row % gptq_nibbles_per_u32;
            const z_word = qzeros[z_word_idx];
            const zero: f32 = @floatFromInt(@as(i32, @intCast((z_word >> @as(u5, @intCast(z_nibble * 4))) & 0xF)));

            const base = g * gs;
            const elems = @min(gs, k - base);
            const full_words = elems / gptq_nibbles_per_u32;

            var group_sum: f32 = 0.0;
            for (0..full_words) |wi| {
                const word = w_row[base / gptq_nibbles_per_u32 + wi];
                inline for (0..8) |ni| {
                    const nibble: u4 = @truncate(word >> @as(u5, ni * 4));
                    const val = (@as(f32, @floatFromInt(@as(i32, nibble))) - zero) * scale;
                    group_sum += val * x[base + wi * 8 + ni];
                }
            }
            sum += group_sum;
        }
        y[row] = sum;
    }
}

/// Convert f16 stored as u16 to f32.
inline fn f16ToF32(val: u16) f32 {
    return @floatCast(@as(f16, @bitCast(val)));
}

// ── Tests ───────────────────────────────────────────────────────

test "gptq dequant basic" {
    // Pack 8 values: [1,2,3,4,5,6,7,0] into one u32
    const word: u32 = 0x07654321;
    const nibble0: u4 = @truncate(word >> 0);
    const nibble1: u4 = @truncate(word >> 4);
    try std.testing.expectEqual(@as(u4, 1), nibble0);
    try std.testing.expectEqual(@as(u4, 2), nibble1);

    // Full GEMV: 1 row, k=8, group_size=8
    // Packed INT4 weights: [1, 2, 3, 0, 1, 2, 3, 0]
    const qweight = [_]u32{0x03210321};
    // Scale = 2.0 as f16
    const scales = [_]u16{@bitCast(@as(f16, 2.0))};
    // Zero-point = 1 in lowest nibble
    const qzeros = [_]u32{0x00000001};
    const x = [_]f32{1.0} ** 8;
    var y = [_]f32{0.0};

    // dequant: (nibble - zero) * scale = [0, 2, 4, -2, 0, 2, 4, -2]
    // dot with all-ones x = 8.0
    gptqGemv(&x, &qweight, &scales, &qzeros, &y, 1, 8, 8);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), y[0], 1e-4);
}

test "gptqGemvRows with start_row offset" {
    // Same setup as basic test but using gptqGemvRows with start_row=0
    const qweight = [_]u32{0x03210321};
    const scales = [_]u16{@bitCast(@as(f16, 2.0))};
    const qzeros = [_]u32{0x00000001};
    const x = [_]f32{1.0} ** 8;
    var y = [_]f32{0.0};

    gptqGemvRows(&x, &qweight, &scales, &qzeros, &y, 0, 1, 1, 8, 8);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), y[0], 1e-4);
}

test "gptqGemvRows chunk uses full-n qzeros stride" {
    // 16 rows so qzeros pack 2 u32 words per group (ceil(16/8)=2).
    // Chunk covering only rows 8..15 must still index zeros with full n=16 stride.
    const n: usize = 16;
    const k: usize = 8;
    const group_size: u32 = 8;
    var qweight = [_]u32{0x03210321} ** n;
    var scales = [_]u16{@bitCast(@as(f16, 1.0))} ** n;
    // Two zero-words: first for rows 0-7 (zero=1), second for rows 8-15 (zero=0).
    const qzeros = [_]u32{ 0x11111111, 0x00000000 };
    const x = [_]f32{1.0} ** k;
    var y = [_]f32{0.0} ** n;

    gptqGemvRows(&x, &qweight, &scales, &qzeros, &y, 8, 8, n, k, group_size);
    // zero=0 → dequant nibbles [1,2,3,0,1,2,3,0] → sum=12
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), y[8], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), y[15], 1e-4);
}

test "fuzz: all gptq functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // Fixed layout: 1 row, k=8, group_size=8 (1 u32 word, 1 group)
            const k: usize = 8;
            const n: usize = 1;
            const group_size: u32 = 8;

            // Random packed INT4 weights (8 nibbles in one u32)
            var qweight = [_]u32{smith.valueWithHash(u32, 0)};
            // Random scale as f16 bits — clamp to finite f16 range
            var scale_bits = smith.valueWithHash(u16, 1);
            // Mask exponent to avoid inf/nan: exponent field [14:10], max 0x7C00 = inf
            // Keep exponent <= 0x1E (30) to stay finite
            const exp = (scale_bits >> 10) & 0x1F;
            if (exp == 0x1F) scale_bits &= 0x83FF; // zero out exponent -> subnormal (finite)
            var scales_arr = [_]u16{scale_bits};
            // Random packed zero-points
            var qzeros = [_]u32{smith.valueWithHash(u32, 2)};

            // Random input vector
            var x: [k]f32 = undefined;
            for (0..k) |i| {
                const bits = smith.valueWithHash(u32, @as(u32, @intCast(i)) +% 100);
                x[i] = @bitCast(bits);
                if (!std.math.isFinite(x[i])) x[i] = 0.0;
            }

            // --- Exercise gptqGemv (pub fn #1) ---
            var y1 = [_]f32{0.0};
            gptqGemv(&x, &qweight, &scales_arr, &qzeros, &y1, n, k, group_size);
            // Result must be finite (finite inputs, finite scale, integer weights)
            if (!std.math.isFinite(y1[0])) return error.TestUnexpectedResult;

            // --- Exercise gptqGemvRows (pub fn #2) ---
            var y2 = [_]f32{0.0};
            gptqGemvRows(&x, &qweight, &scales_arr, &qzeros, &y2, 0, n, n, k, group_size);
            // gptqGemv delegates to gptqGemvRows, so results must match
            if (y1[0] != y2[0]) return error.TestUnexpectedResult;
        }
    }.f, .{});
}
