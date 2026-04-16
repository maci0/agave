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
