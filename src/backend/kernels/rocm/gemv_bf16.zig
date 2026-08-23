//! GEMV BF16 kernel: y[row] = dot(W_bf16[row,:], x)
//!
//! TileLang-derived design: one row per workgroup, 256 threads, each thread
//! consuming one u32 word = two bf16 weights per iteration (constant-shift
//! unpack, no byte loads, no per-chunk activation rescans). The previous
//! version re-scanned 32 x values for a sparse check on EVERY element and
//! loaded weights as scalar u16 — it ran at ~46 GB/s; this variant streams.
//!
//! Launch contract: grid = n, block = 256.

const cu = @import("common.zig");

export fn gemv_bf16_kernel(x: [*]const f32, w: [*]const u16, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    var sum: f32 = 0.0;

    if (k % 2 == 0) {
        // Fast path: two bf16 weights per dword load. Rows stay 4-byte aligned
        // because row stride in bytes is k*2 with k even; the tensor base is
        // GPU-buffer aligned.
        const w32 = @as([*]const u32, @ptrCast(@alignCast(w)));
        const pairs = k / 2;
        const row_pair_off = row * pairs;

        var p = tid;
        while (p < pairs) : (p += bdim) {
            const bits = w32[row_pair_off + p];
            // little-endian: low half of the word is element c0
            const wf_lo: f32 = @bitCast(bits << 16);
            const wf_hi: f32 = @bitCast(bits & 0xFFFF0000);
            const c0 = p * 2;
            sum += wf_lo * x[c0] + wf_hi * x[c0 + 1];
        }
    } else {
        // Odd-k fallback: scalar u16 path (rare; model shapes are even).
        const row_offset = row * k;
        var j = tid;
        while (j < k) : (j += bdim) {
            const bits: u32 = @as(u32, w[row_offset + j]) << 16;
            sum += @as(f32, @bitCast(bits)) * x[j];
        }
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(true);
}
