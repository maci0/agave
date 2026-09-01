//! Software prefetch hints for the quantized CPU GEMV kernels.
//!
//! A decode-step GEMV reads each weight row exactly once and re-reads the same
//! activation vector for every row, so the two streams want opposite cache
//! treatment:
//!
//!   * **Weights: non-temporal.** They are consumed once and never revisited
//!     within the call. Prefetching them at locality 0 keeps them from evicting
//!     the activations, which are the only thing worth keeping resident.
//!   * **Activations: temporal.** Every row re-reads them, so they belong in L1.
//!
//! The kernels are DRAM-bandwidth-bound, and the interleaved unpack work lowers
//! the L1-miss concurrency the hardware prefetcher sustains on its own, so the
//! explicit hint measurably raises achieved bandwidth even though the access
//! pattern is a pure forward scan.
//!
//! Hints only: an out-of-range or already-resident address costs nothing and
//! faults nothing, so the callers do not bound-check the lookahead against the
//! end of a row.

const std = @import("std");

/// Blocks of lookahead for the weight stream.
///
/// Quantized blocks run 18 to 144 bytes, so this is roughly 1 to 8 KB ahead:
/// far enough to cover DRAM latency, near enough that a row's tail overshoot
/// stays within the next row of the same matrix rather than reaching into
/// another thread's row band, where it would duplicate DRAM traffic at exactly
/// the bandwidth ceiling the kernel is trying to reach.
pub const weight_blocks_ahead: usize = 8;

/// Bytes of lookahead for the activation vector. One 64-byte line per SIMD
/// group is enough: the vector is small, hot, and re-read by every row.
pub const act_bytes_ahead: usize = 128;

/// Hint that a weight block will be read once and not reused.
pub inline fn weight(ptr: [*]const u8, byte_offset: usize) void {
    @prefetch(ptr + byte_offset, .{ .rw = .read, .locality = 0, .cache = .data });
}

/// Hint that an activation range will be read again by later rows.
pub inline fn activation(ptr: [*]const f32, elem_offset: usize) void {
    @prefetch(ptr + elem_offset, .{ .rw = .read, .locality = 3, .cache = .data });
}

/// Weight-stream hint for block `b + weight_blocks_ahead` of a row whose blocks
/// are `block_bytes` apart, starting at `row`. The common call shape: the
/// kernels walk blocks in order and only need the next address.
pub inline fn weightBlock(row: [*]const u8, b: usize, block_bytes: usize) void {
    weight(row, (b + weight_blocks_ahead) * block_bytes);
}

test "prefetch helpers accept in-range and past-the-end offsets" {
    // The point of the test is that a hint past the end of the allocation is
    // inert rather than a fault: the kernels deliberately overshoot a row tail.
    var w: [64]u8 = @splat(0);
    var a: [16]f32 = @splat(0.0);
    weight(&w, 0);
    weight(&w, w.len * 4);
    weightBlock(&w, 0, 18);
    activation(&a, 0);
    activation(&a, a.len * 4);
    try std.testing.expect(weight_blocks_ahead > 0);
}
