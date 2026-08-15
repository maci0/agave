//! CPU GEMV kernels for 4-bit floating-point formats.
//! MXFP4 (E2M1 microscaling) and NVFP4 (FP8 E4M3 block scale).
//! 2-row batched to share x-vector cache reads.

const std = @import("std");
const quant = @import("../../../ops/quant.zig");
const backend_mod = @import("../../backend.zig");

/// MXFP4: 32 values per block, 17 bytes (1 E8M0 scale + 16 nibble-packed bytes)
/// 2-row batched to share x-vector cache reads.
pub fn gemvMXFP4(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb = backend_mod.mxfp4_block_bytes;
    const qk = backend_mod.quant_block_elems;
    const nb = (k + qk - 1) / qk;
    const row_bytes = nb * bpb;

    // Process 2 rows at a time for x-vector cache reuse.
    var row: usize = 0;
    while (row + 2 <= n) : (row += 2) {
        var sum0: f32 = 0.0;
        var sum1: f32 = 0.0;
        const rp0 = w + row * row_bytes;
        const rp1 = w + (row + 1) * row_bytes;
        for (0..nb) |b| {
            const bp0 = rp0 + b * bpb;
            const bp1 = rp1 + b * bpb;
            const d0 = quant.e8m0ToF32(bp0[0]);
            const d1 = quant.e8m0ToF32(bp1[0]);
            const bk = b * qk;
            if (bk + qk - 1 < k) {
                var block_sum0: f32 = 0.0;
                var block_sum1: f32 = 0.0;
                for (0..qk / 2) |j| {
                    const byte0 = bp0[1 + j];
                    const byte1 = bp1[1 + j];
                    const xlo = x[bk + j];
                    const xhi = x[bk + j + qk / 2];
                    block_sum0 += xlo * quant.mxfp4Lookup(byte0 & 0x0F) +
                        xhi * quant.mxfp4Lookup(byte0 >> 4);
                    block_sum1 += xlo * quant.mxfp4Lookup(byte1 & 0x0F) +
                        xhi * quant.mxfp4Lookup(byte1 >> 4);
                }
                sum0 += block_sum0 * d0;
                sum1 += block_sum1 * d1;
            } else {
                for (0..qk / 2) |j| {
                    const byte0 = bp0[1 + j];
                    const byte1 = bp1[1 + j];
                    const gi0 = bk + j;
                    const gi1 = bk + j + qk / 2;
                    if (gi0 < k) {
                        const xv = x[gi0];
                        sum0 += xv * quant.mxfp4Lookup(byte0 & 0x0F) * d0;
                        sum1 += xv * quant.mxfp4Lookup(byte1 & 0x0F) * d1;
                    }
                    if (gi1 < k) {
                        const xv = x[gi1];
                        sum0 += xv * quant.mxfp4Lookup(byte0 >> 4) * d0;
                        sum1 += xv * quant.mxfp4Lookup(byte1 >> 4) * d1;
                    }
                }
            }
        }
        y[row] = sum0;
        y[row + 1] = sum1;
    }

    // Remainder: single row
    while (row < n) : (row += 1) {
        var sum: f32 = 0.0;
        const rp = w + row * row_bytes;
        for (0..nb) |b| {
            const bp = rp + b * bpb;
            const d = quant.e8m0ToF32(bp[0]);
            const bk = b * qk;
            if (bk + qk - 1 < k) {
                var block_sum: f32 = 0.0;
                for (0..qk / 2) |j| {
                    const byte = bp[1 + j];
                    block_sum += x[bk + j] * quant.mxfp4Lookup(byte & 0x0F) +
                        x[bk + j + qk / 2] * quant.mxfp4Lookup(byte >> 4);
                }
                sum += block_sum * d;
            } else {
                for (0..qk / 2) |j| {
                    const byte = bp[1 + j];
                    const v0 = quant.mxfp4Lookup(byte & 0x0F);
                    const v1 = quant.mxfp4Lookup(byte >> 4);
                    const gi0 = bk + j;
                    const gi1 = bk + j + qk / 2;
                    if (gi0 < k) sum += x[gi0] * v0 * d;
                    if (gi1 < k) sum += x[gi1] * v1 * d;
                }
            }
        }
        y[row] = sum;
    }
}


/// SIMD-optimized MXFP4 GEMV. Processes 4 weight bytes (8 values) per vector iteration.
/// Uses @Vector(4, f32) with @mulAdd for FMA accumulation on NEON/SSE.
/// ~2× faster than scalar version on Apple Silicon (M4 Pro measured).
pub fn gemvMXFP4_V(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb = backend_mod.mxfp4_block_bytes; // 17
    const qk: usize = backend_mod.quant_block_elems; // 32
    const nb = (k + qk - 1) / qk;
    const row_bytes = nb * bpb;
    const half_qk = qk / 2; // 16
    const V4 = @Vector(4, f32);
    const lut = comptime [16]f32{
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    };

    for (0..n) |row| {
        var sum_v: V4 = @splat(0.0);
        var sum_s: f32 = 0.0; // scalar remainder
        const rp = w + row * row_bytes;

        for (0..nb) |b| {
            const bp = rp + b * bpb;
            const d = quant.e8m0ToF32(bp[0]);
            if (d == 0.0) continue;
            const dv: V4 = @splat(d);
            const bk = b * qk;

            if (bk + qk <= k) {
                // Full block: process 4 bytes (8 values) per iteration
                var j: usize = 0;
                while (j + 4 <= half_qk) : (j += 4) {
                    // Lo nibbles: 4 weight values at positions bk+j..bk+j+3
                    const wlo: V4 = .{
                        lut[bp[1 + j] & 0x0F], lut[bp[1 + j + 1] & 0x0F],
                        lut[bp[1 + j + 2] & 0x0F], lut[bp[1 + j + 3] & 0x0F],
                    };
                    const xlo: V4 = x[bk + j ..][0..4].*;
                    sum_v = @mulAdd(V4, xlo * wlo, dv, sum_v);

                    // Hi nibbles: 4 weight values at positions bk+half_qk+j..
                    const whi: V4 = .{
                        lut[bp[1 + j] >> 4], lut[bp[1 + j + 1] >> 4],
                        lut[bp[1 + j + 2] >> 4], lut[bp[1 + j + 3] >> 4],
                    };
                    const xhi: V4 = x[bk + half_qk + j ..][0..4].*;
                    sum_v = @mulAdd(V4, xhi * whi, dv, sum_v);
                }
            } else {
                // Partial block: scalar fallback
                for (0..half_qk) |j| {
                    const byte = bp[1 + j];
                    const gi0 = bk + j;
                    const gi1 = bk + j + half_qk;
                    if (gi0 < k) sum_s += x[gi0] * lut[byte & 0x0F] * d;
                    if (gi1 < k) sum_s += x[gi1] * lut[byte >> 4] * d;
                }
            }
        }
        y[row] = @reduce(.Add, sum_v) + sum_s;
    }
}

/// NVFP4: 16-element blocks with FP8 E4M3 block scale.
/// Block layout: 1 byte FP8 scale + 8 bytes packed nibbles = 9 bytes per block.
/// 2-row batched to share x-vector cache reads.
pub fn gemvNVFP4(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb = backend_mod.nvfp4_block_bytes;
    const qk = backend_mod.nvfp4_block_elems;
    const nb = (k + qk - 1) / qk;
    const row_bytes = nb * bpb;

    // Process 2 rows at a time for x-vector cache reuse.
    var row: usize = 0;
    while (row + 2 <= n) : (row += 2) {
        var sum0: f32 = 0.0;
        var sum1: f32 = 0.0;
        const rp0 = w + row * row_bytes;
        const rp1 = w + (row + 1) * row_bytes;
        for (0..nb) |b| {
            const bp0 = rp0 + b * bpb;
            const bp1 = rp1 + b * bpb;
            const scale0 = quant.fp8e4m3ToF32(bp0[0]);
            const scale1 = quant.fp8e4m3ToF32(bp1[0]);
            const bk = b * qk;
            if (bk + qk - 1 < k) {
                var block_sum0: f32 = 0.0;
                var block_sum1: f32 = 0.0;
                for (0..qk / 2) |j| {
                    const byte0 = bp0[1 + j];
                    const byte1 = bp1[1 + j];
                    const xlo = x[bk + j];
                    const xhi = x[bk + j + qk / 2];
                    block_sum0 += xlo * quant.mxfp4Lookup(byte0 & 0x0F) +
                        xhi * quant.mxfp4Lookup(byte0 >> 4);
                    block_sum1 += xlo * quant.mxfp4Lookup(byte1 & 0x0F) +
                        xhi * quant.mxfp4Lookup(byte1 >> 4);
                }
                sum0 += block_sum0 * scale0;
                sum1 += block_sum1 * scale1;
            } else {
                for (0..qk / 2) |j| {
                    const byte0 = bp0[1 + j];
                    const byte1 = bp1[1 + j];
                    const gi0 = bk + j;
                    const gi1 = bk + j + qk / 2;
                    if (gi0 < k) {
                        const xv = x[gi0];
                        sum0 += xv * quant.mxfp4Lookup(byte0 & 0x0F) * scale0;
                        sum1 += xv * quant.mxfp4Lookup(byte1 & 0x0F) * scale1;
                    }
                    if (gi1 < k) {
                        const xv = x[gi1];
                        sum0 += xv * quant.mxfp4Lookup(byte0 >> 4) * scale0;
                        sum1 += xv * quant.mxfp4Lookup(byte1 >> 4) * scale1;
                    }
                }
            }
        }
        y[row] = sum0;
        y[row + 1] = sum1;
    }

    // Remainder: single row
    while (row < n) : (row += 1) {
        var sum: f32 = 0.0;
        const rp = w + row * row_bytes;
        for (0..nb) |b| {
            const bp = rp + b * bpb;
            const scale = quant.fp8e4m3ToF32(bp[0]);
            const bk = b * qk;
            if (bk + qk - 1 < k) {
                var block_sum: f32 = 0.0;
                for (0..qk / 2) |j| {
                    const byte = bp[1 + j];
                    block_sum += x[bk + j] * quant.mxfp4Lookup(byte & 0x0F) +
                        x[bk + j + qk / 2] * quant.mxfp4Lookup(byte >> 4);
                }
                sum += block_sum * scale;
            } else {
                for (0..qk / 2) |j| {
                    const byte = bp[1 + j];
                    const v0 = quant.mxfp4Lookup(byte & 0x0F);
                    const v1 = quant.mxfp4Lookup(byte >> 4);
                    const gi0 = bk + j;
                    const gi1 = bk + j + qk / 2;
                    if (gi0 < k) sum += x[gi0] * v0 * scale;
                    if (gi1 < k) sum += x[gi1] * v1 * scale;
                }
            }
        }
        y[row] = sum;
    }
}

test "gemvMXFP4 uniform weights" {
    // 2x32 GEMV. E8M0 scale=127 → 2^0 = 1.0.
    // All nibbles=2 → mxfp4Lookup(2)=1.0. x = all 1.0.
    // MXFP4 block: 32 elements, 17 bytes (1 E8M0 scale + 16 nibble-packed bytes).
    // y[i] = 1.0 * 32 * 1.0 = 32.0
    const bpb = backend_mod.mxfp4_block_bytes; // 17
    var w: [2 * bpb]u8 = undefined;
    for (0..2) |r| {
        const base = r * bpb;
        w[base] = 127; // e8m0(127) = 1.0
        for (1..17) |i| w[base + i] = 0x22; // lo=2 (1.0), hi=2 (1.0)
    }
    var x: [32]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [2]f32 = undefined;
    gemvMXFP4(&x, &w, &y, 2, 32);
    for (0..2) |i| try std.testing.expectApproxEqAbs(@as(f32, 32.0), y[i], 1e-4);
}

test "gemvMXFP4 scale factor" {
    // 1x32. E8M0 scale=128 → 2^1 = 2.0. nibbles=1 → mxfp4Lookup(1)=0.5.
    // y = 2.0 * 32 * 0.5 = 32.0
    const bpb = backend_mod.mxfp4_block_bytes;
    var w: [bpb]u8 = undefined;
    w[0] = 128; // e8m0(128) = 2.0
    for (1..17) |i| w[i] = 0x11; // lo=1 (0.5), hi=1 (0.5)
    var x: [32]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvMXFP4(&x, &w, &y, 1, 32);
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), y[0], 1e-4);
}

test "gemvNVFP4 uniform weights" {
    // 1x16 NVFP4: FP8 E4M3 scale=0x38 (1.0), nibbles=2 (1.0).
    // y = 1.0 * 16 * 1.0 = 16.0
    const bpb = backend_mod.nvfp4_block_bytes; // 9
    var w: [bpb]u8 = undefined;
    w[0] = 0x38; // FP8 E4M3 1.0
    for (1..9) |i| w[i] = 0x22; // lo=2, hi=2
    var x: [16]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [1]f32 = undefined;
    gemvNVFP4(&x, &w, &y, 1, 16);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), y[0], 1e-4);
}

test "gemvNVFP4 multiple rows" {
    // 3x16 NVFP4: verify multi-row produces independent results.
    const bpb = backend_mod.nvfp4_block_bytes;
    var w: [3 * bpb]u8 = undefined;
    for (0..3) |r| {
        const base = r * bpb;
        // Scale: FP8 E4M3 = 0x38 (1.0) for row 0, 0x40 (2.0) for row 1, 0x38 (1.0) for row 2
        w[base] = if (r == 1) 0x40 else 0x38;
        for (1..9) |i| w[base + i] = 0x22; // all nibbles=2 → mxfp4Lookup(2)=1.0
    }
    var x: [16]f32 = undefined;
    for (&x) |*v| v.* = 1.0;
    var y: [3]f32 = undefined;
    gemvNVFP4(&x, &w, &y, 3, 16);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), y[0], 1e-4); // scale=1.0, sum(1.0*16)
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), y[1], 1e-4); // scale=2.0, sum(1.0*16)*2
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), y[2], 1e-4);
}

test "gemvNVFP4 varying x" {
    // Verify correct element-to-weight correspondence with non-uniform x.
    // 1x16 NVFP4: all weights=1.0, x[i] = i+1 → y = sum(1..16) = 136.
    const bpb = backend_mod.nvfp4_block_bytes;
    var w: [bpb]u8 = undefined;
    w[0] = 0x38; // FP8 E4M3 1.0
    for (1..9) |i| w[i] = 0x22; // all nibbles=2 → 1.0
    var x: [16]f32 = undefined;
    for (0..16) |i| x[i] = @floatFromInt(i + 1);
    var y: [1]f32 = undefined;
    gemvNVFP4(&x, &w, &y, 1, 16);
    // sum(1..16) = 16*17/2 = 136
    try std.testing.expectApproxEqAbs(@as(f32, 136.0), y[0], 1.0);
}

test "gemvMXFP4 varying x" {
    // Verify correct element-to-weight correspondence with non-uniform x.
    // 1x32 MXFP4: all weights=1.0, x[i] = i+1 → y = sum(1..32) = 528.
    const bpb = backend_mod.mxfp4_block_bytes;
    var w: [bpb]u8 = undefined;
    w[0] = 127; // e8m0(127) = 1.0
    for (1..17) |i| w[i] = 0x22; // all nibbles=2 → 1.0
    var x: [32]f32 = undefined;
    for (0..32) |i| x[i] = @floatFromInt(i + 1);
    var y: [1]f32 = undefined;
    gemvMXFP4(&x, &w, &y, 1, 32);
    // sum(1..32) = 32*33/2 = 528
    try std.testing.expectApproxEqAbs(@as(f32, 528.0), y[0], 2.0);
}

test "fuzz: gemvMXFP4 gemvNVFP4" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // -- MXFP4: 2 rows, k=32 --
            {
                const bpb = backend_mod.mxfp4_block_bytes; // 17
                const qk = backend_mod.quant_block_elems; // 32
                const n = 2;
                var x: [qk]f32 = undefined;
                var w: [n * bpb]u8 = undefined;
                var y: [n]f32 = undefined;
                var x_raw: [qk * 4]u8 = undefined;
                smith.bytesWithHash(&x_raw, 0);
                smith.bytesWithHash(&w, 1);
                x = @bitCast(x_raw);
                for (&x) |*v| if (!std.math.isFinite(v.*)) {
                    v.* = 0.0;
                };
                gemvMXFP4(&x, &w, &y, n, qk);
                for (y) |v| try std.testing.expect(std.math.isFinite(v));
            }

            // -- NVFP4: 3 rows, k=16 --
            {
                const bpb = backend_mod.nvfp4_block_bytes; // 9
                const qk = backend_mod.nvfp4_block_elems; // 16
                const n = 3;
                var x: [qk]f32 = undefined;
                var w: [n * bpb]u8 = undefined;
                var y: [n]f32 = undefined;
                var x_raw: [qk * 4]u8 = undefined;
                smith.bytesWithHash(&x_raw, 2);
                smith.bytesWithHash(&w, 3);
                x = @bitCast(x_raw);
                for (&x) |*v| if (!std.math.isFinite(v.*)) {
                    v.* = 0.0;
                };
                // Clamp FP8 E4M3 scale to non-NaN (0x7F = NaN in E4M3).
                for (0..n) |r| {
                    if (w[r * bpb] == 0x7F or w[r * bpb] == 0xFF) w[r * bpb] = 0;
                }
                gemvNVFP4(&x, &w, &y, n, qk);
                for (y) |v| try std.testing.expect(std.math.isFinite(v));
            }
        }
    }.f, .{});
}

/// INT8 activation-quantized MXFP4 GEMV (Colibri-inspired).
/// Quantizes activations to INT8, then uses integer multiply-accumulate.
/// ~2-3× faster than float LUT path on AArch64 with NEON dot product.
pub fn gemvMXFP4_I8(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb = backend_mod.mxfp4_block_bytes; // 17
    const qk: usize = backend_mod.quant_block_elems; // 32
    const nb = (k + qk - 1) / qk;
    const row_bytes = nb * bpb;
    const half_qk = qk / 2; // 16

    // Step 1: Quantize activations to INT8 with per-block scales
    // Block size matches MXFP4 block size (32 elements)
    var x_i8: [16384]i8 = undefined; // max k = 16384
    var x_scales: [512]f32 = undefined; // max blocks = 512
    std.debug.assert(k <= x_i8.len);

    for (0..nb) |b| {
        const bk = b * qk;
        const end = @min(bk + qk, k);
        // Find max absolute value in this block
        var amax: f32 = 0;
        for (bk..end) |i| {
            const abs_val = @abs(x[i]);
            if (abs_val > amax) amax = abs_val;
        }
        // Scale to INT8 range (-127..127)
        const scale = if (amax > 0) 127.0 / amax else 0;
        x_scales[b] = if (amax > 0) amax / 127.0 else 0;
        for (bk..end) |i| {
            const v = x[i] * scale;
            x_i8[i] = @intFromFloat(@max(-127, @min(127, @round(v))));
        }
    }

    // Step 2: For each output row, compute INT4×INT8 dot product
    // MXFP4 LUT as INT8 (×16 to get integer range):
    // 0, 0.5, 1, 1.5, 2, 3, 4, 6 → ×16 → 0, 8, 16, 24, 32, 48, 64, 96
    const lut_i8 = [16]i8{
        0, 8, 16, 24, 32, 48, 64, 96,
        0, -8, -16, -24, -32, -48, -64, -96,
    };
    const inv16: f32 = 1.0 / 16.0;

    for (0..n) |row| {
        var sum: f64 = 0;
        const rp = w + row * row_bytes;

        for (0..nb) |b| {
            const bp = rp + b * bpb;
            const w_scale = quant.e8m0ToF32(bp[0]);
            if (w_scale == 0) continue;
            const bk = b * qk;

            // Compute: Σ lut_i8[nibble] × x_i8[bk+j] for j in 0..32
            // This is an INT8×INT8 dot product that the compiler can vectorize
            var acc: i32 = 0;
            for (0..half_qk) |j| {
                const byte = bp[1 + j];
                const lo_nib: u4 = @truncate(byte & 0x0F);
                const hi_nib: u4 = @truncate(byte >> 4);
                const w_lo: i8 = lut_i8[lo_nib];
                const w_hi: i8 = lut_i8[hi_nib];
                acc += @as(i32, w_lo) * @as(i32, x_i8[bk + j]);
                acc += @as(i32, w_hi) * @as(i32, x_i8[bk + j + half_qk]);
            }

            // Scale: INT32 result × w_scale × x_scale × (1/16)
            sum += @as(f64, @as(f32, @floatFromInt(acc))) * w_scale * x_scales[b] * inv16;
        }

        y[row] = @floatCast(sum);
    }
}
