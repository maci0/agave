//! HQQ (Half-Quadratic Quantization) 4-bit GEMV kernel.
//!
//! Weight format (axis=1, default): W_q [n_out, k_in/2] uint8, 2 nibbles/byte.
//! Nibble packing: byte = (hi << 4) | lo, so:
//!   elem k=0 → byte[k/2] & 0xF (low nibble)
//!   elem k=1 → (byte[k/2] >> 4) & 0xF (high nibble)
//! Scale/zero: [n_out, k_in/group_size] bf16, one pair per group.
//! Dequant: w = (nibble - zero) * scale
//!
//! The W_q, scale, and zero pointers come from the model's companion tensor lookup.

const std = @import("std");

/// HQQ GEMV: y[n_out] = dequant(W_q[n_out, k_in/2]) @ x[k_in]
/// Companion tensors passed separately — they are bf16 and f16 respectively.
pub fn hqqGemv(
    x: [*]const f32,
    w_q: [*]const u8,
    scale: [*]const u16,
    zero: [*]const u16,
    y: [*]f32,
    n_out: usize,
    k_in: usize,
    group_size: u32,
) void {
    hqqGemvRows(x, w_q, scale, zero, y, 0, n_out, k_in, group_size);
}

/// HQQ row-major GEMV: processes a subset of output rows for parallel dispatch.
pub fn hqqGemvRows(
    x: [*]const f32,
    w_q: [*]const u8,
    scale: [*]const u16,
    zero: [*]const u16,
    y: [*]f32,
    start_row: usize,
    n_rows: usize,
    k_in: usize,
    group_size: u32,
) void {
    const gs: usize = group_size;
    const n_groups = (k_in + gs - 1) / gs;
    const half_k = (k_in + 1) / 2; // bytes per row

    const sparse_threshold: f32 = 0.005;

    for (0..n_rows) |ri| {
        const row = start_row + ri;
        var acc: f32 = 0.0;
        const w_row = w_q + row * half_k;
        const s_row = scale + row * n_groups;
        const z_row = zero + row * n_groups;

        for (0..k_in) |ki| {
            const xv = x[ki];
            if (@abs(xv) < sparse_threshold) continue;

            const byte = w_row[ki / 2];
            const nibble: f32 = if (ki % 2 == 0)
                @floatFromInt(byte & 0xF)
            else
                @floatFromInt(byte >> 4);

            const g = ki / gs;
            const s = bf16ToF32(s_row[g]);
            const z = bf16ToF32(z_row[g]);

            acc += (nibble - z) * s * xv;
        }
        y[row] = acc;
    }
}

inline fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

// ── Tests ──────────────────────────────────────────────────────────

test "hqq nibble packing" {
    // byte = (hi << 4) | lo, lo nibble = elem 0, hi nibble = elem 1
    const byte: u8 = 0xA3; // hi=0xA=10, lo=0x3=3
    try std.testing.expectEqual(@as(f32, 3.0), @as(f32, @floatFromInt(byte & 0xF)));
    try std.testing.expectEqual(@as(f32, 10.0), @as(f32, @floatFromInt(byte >> 4)));
}

test "hqqGemv identity row" {
    // 1 output, k=4, group_size=4 → 1 group, 2 bytes
    // W_q: [0x21, 0x43] → nibbles [1,2,3,4]
    // scale=1.0 bf16, zero=0.0 bf16 → w = [1,2,3,4]
    // x = [1,0,0,0] → y = 1*1 = 1.0
    var w_q = [_]u8{ 0x21, 0x43 };
    const one_bf16: u16 = 0x3F80; // 1.0 in bf16
    const zero_bf16: u16 = 0x0000;
    var scale = [_]u16{one_bf16};
    var z = [_]u16{zero_bf16};
    var x = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var y = [_]f32{0.0};
    hqqGemv(&x, &w_q, &scale, &z, &y, 1, 4, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), y[0], 0.01);
}

test "hqqGemv all-ones" {
    // 1 output, k=4, group_size=4
    // W_q: nibbles all 0x8 → 8 each
    // scale=1.0, zero=8.0 → w = (8-8)*1 = 0 → y=0
    var w_q = [_]u8{ 0x88, 0x88 };
    const one_bf16: u16 = 0x3F80;
    const eight_bf16: u16 = 0x4100; // 8.0 in bf16
    var scale = [_]u16{one_bf16};
    var z = [_]u16{eight_bf16};
    var x = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var y = [_]f32{99.0};
    hqqGemv(&x, &w_q, &scale, &z, &y, 1, 4, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[0], 0.01);
}

test "hqqGemv multi-row" {
    // 2 outputs, k=2, group_size=2
    // Row0: [0x21] → nibbles [1,2], scale=2.0, zero=1.0 → w=[0,2]
    // Row1: [0xFE] → nibbles [14,15], scale=1.0, zero=14.0 → w=[0,1]
    // x=[1,1] → y0=0*1+2*1=2, y1=0*1+1*1=1
    var w_q = [_]u8{ 0x21, 0xFE };
    const two_bf16: u16 = 0x4000;
    const one_bf16: u16 = 0x3F80;
    const fourteen_bf16: u16 = 0x4160;
    var scale = [_]u16{ two_bf16, one_bf16 };
    var z = [_]u16{ one_bf16, fourteen_bf16 };
    var x = [_]f32{ 1.0, 1.0 };
    var y = [_]f32{ 0.0, 0.0 };
    hqqGemv(&x, &w_q, &scale, &z, &y, 2, 2, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), y[0], 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), y[1], 0.1);
}

test "fuzz: hqq functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            comptime {
                _ = &hqqGemv;
                _ = &hqqGemvRows;
            }
            const k = 4;
            var w_q: [k / 2]u8 = undefined;
            smith.bytesWithHash(&w_q, 0);
            const scale_raw = smith.valueWithHash(u16, 1) & 0x7F7F; // avoid inf/nan
            const zero_raw = smith.valueWithHash(u16, 2) & 0x7F7F;
            var scale = [_]u16{scale_raw};
            var z = [_]u16{zero_raw};
            var x: [k]f32 = undefined;
            for (&x, 0..) |*v, i| {
                v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 10)))) / 10.0;
            }
            var y = [_]f32{0.0};
            hqqGemv(&x, &w_q, &scale, &z, &y, 1, k, 4);
            try std.testing.expect(!std.math.isNan(y[0]));
        }
    }.f, .{});
}
