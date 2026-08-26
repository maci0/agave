//! GEMV Q4_K kernel: y[row] = dot(W_q4k[row,:], x)
//!
//! TileLang-derived design (see research/kernels/tilelang): ONE output row
//! per workgroup, 256 threads decomposed as 32 copies x 8 lanes. A lane owns
//! one 32-element sub-block slice of a super-block (32 bytes = 8 u32 words);
//! copies split the super-block range so every thread streams a short
//! contiguous dword run instead of striding byte-wise across 144-byte blocks.
//! Nibbles are extracted from whole words with constant shifts; the block-wide
//! reduce produces the dot product.
//!
//! Previous layout (NR=2, tid-strided blocks) left 256-thread workgroups ~90%
//! idle at Qwen3.8 shapes and hit ~34 GB/s; this layout measures >3x.
//!
//! Launch contract (src/backend/rocm.zig): grid = n, block = 256.

const cu = @import("common.zig");
const getScaleMinK4 = cu.getScaleMinK4;

/// Bytes per Q4_K super-block (256 elements).
const q4_k_block_size: u32 = 144;
/// u32 words per Q4_K super-block.
const q4_k_block_words: u32 = q4_k_block_size / 4;
/// Elements per Q4_K super-block.
const elems_per_block: u32 = 256;
/// Lanes per row: one 32-elem sub-block each.
const lanes: u32 = 8;
/// Workgroup size (must match launcher block_size).
const threads: u32 = 256;
/// copies = threads / lanes: super-blocks covered in parallel per pass.
const copies: u32 = threads / lanes;

export fn gemv_q4_k_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const cp = tid / lanes;
    const ln = tid % lanes;

    const nblk = k / elems_per_block;
    const spc = (nblk + copies - 1) / copies;

    // Whole tensor as u32: every super-block base is 144B-strided (4-aligned),
    // and GGUF tensor data is >=4B aligned, so aliasing is safe.
    const wu: [*]const u32 = @ptrCast(@alignCast(w));

    var acc: f32 = 0.0;
    var si: u32 = 0;
    while (si < spc) : (si += 1) {
        const sb = cp * spc + si;
        if (sb >= nblk) break;

        const blk_u32 = (row * nblk + sb) * q4_k_block_words;
        const blk_byte = (row * nblk + sb) * q4_k_block_size;

        // fp16 d at bytes [0,2), dmin at [2,4), read via the u32 view.
        const head = wu[blk_u32];
        const d: f32 = @floatCast(@as(f16, @bitCast(@as(u16, @truncate(head)))));
        const dmin_bits: u16 = @truncate(head >> 16);
        const dmin: f32 = @floatCast(@as(f16, @bitCast(dmin_bits)));

        // 6-bit scale/min pair for this lane's sub-block (bytes 4..15).
        const scales: [*]const u8 = @ptrFromInt(@intFromPtr(w) + blk_byte + 4);
        var scv: u8 = undefined;
        var mnv: u8 = undefined;
        getScaleMinK4(ln, scales, &scv, &mnv);
        const scf: f32 = @floatFromInt(scv);
        const mnf: f32 = @floatFromInt(mnv);

        // Lane's nibble window: group g holds 64 vals; low nibbles are vals
        // [g*64, g*64+32) scaled by pair 2g... here ln IS the sub-block index:
        // group g = ln/2, high/low selected by sh = 4*(ln%2).
        const g = ln / 2;
        const sh: u5 = @intCast(4 * (ln % 2));
        const wbase = blk_u32 + 4 + g * 8;
        const xb = sb * elems_per_block + ln * 32;

        var j: u32 = 0;
        while (j < 8) : (j += 1) {
            const word = wu[wbase + j];
            const eb = xb + j * 4;
            acc += (d * scf * @as(f32, @floatFromInt((word >> sh) & 0xF)) - dmin * mnf) * x[eb];
            acc += (d * scf * @as(f32, @floatFromInt((word >> (sh + 8)) & 0xF)) - dmin * mnf) * x[eb + 1];
            acc += (d * scf * @as(f32, @floatFromInt((word >> (sh + 16)) & 0xF)) - dmin * mnf) * x[eb + 2];
            acc += (d * scf * @as(f32, @floatFromInt((word >> (sh + 24)) & 0xF)) - dmin * mnf) * x[eb + 3];
        }
    }

    const total = cu.blockReduceAdd(acc);
    if (tid == 0) y[row] = total;
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(q4_k_block_size == 144);
    comptime std.debug.assert(q4_k_block_words == 36);
    comptime std.debug.assert(elems_per_block == 256);
    comptime std.debug.assert(copies == 32);
}
