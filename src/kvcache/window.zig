//! Slot placement for a sliding-window KV cache.
//!
//! A sliding-window layer at position `p` only ever attends to positions in
//! `(p - window, p]`. Everything older is unreachable, so holding a slot for
//! it is memory the layer can never read. Gemma 4 sizes every layer's cache
//! at the full context, and most of its layers are sliding: at a 32k context
//! with a 1024-token window that is 32x more per sliding layer than the
//! window needs.
//!
//! The obvious fix, a ring buffer, breaks contiguity: once it wraps, the
//! live window is two runs and the attention path can no longer be handed a
//! base pointer and a length. This keeps the window contiguous instead by
//! allocating room for two windows and compacting when the far end is
//! reached: move the live window down to slot 0 and carry on. Each
//! compaction moves `window` positions and buys `window` more writes, so it
//! is one move per position amortised, against a `window / capacity` memory
//! factor.
//!
//! Pure arithmetic: no allocation, no I/O, no knowledge of what a slot
//! holds. The caller owns the buffer and performs the move.

const std = @import("std");

/// Slots to allocate for a sliding-window layer.
///
/// The live window, plus room for a whole prefill batch written after a
/// compaction. The batch term is what lets the base stay put for the length
/// of a batch: tokens in one batch are written and then attended to, so a
/// base that moved between those two steps would leave the earlier tokens'
/// local indices stale.
///
/// Never more than the context. A window at least half the context saves
/// nothing, and the cache must not grow past what it replaces.
pub fn capacityFor(window: usize, max_batch: usize, max_seq_len: usize) usize {
    if (window == 0) return max_seq_len;
    const headroom = @max(window, max_batch);
    const total = std.math.add(usize, window, headroom) catch return max_seq_len;
    return @min(max_seq_len, total);
}

/// Where the token at an absolute position goes, and what has to move first.
pub const Placement = struct {
    /// Local slot index to write at, after any compaction.
    local: usize,
    /// Local index the compaction copies from, or null when none is needed.
    /// The caller moves `keep` positions from here down to slot 0.
    move_from: ?usize = null,
    /// Positions the compaction moves.
    keep: usize = 0,
    /// Absolute position held at local slot 0 after this placement.
    base: usize,
};

/// Place a batch of `n` positions starting at absolute `first` in a cache of
/// `capacity` slots whose slot 0 currently holds absolute position `base`.
///
/// Compacts once, before the batch, when the last position would land past
/// the end, so the base is fixed for the whole batch and every token's local
/// index stays `pos - base` from the write until the attention that reads
/// it. `capacityFor` sizes the cache so a batch always fits afterwards.
///
/// `local` is the slot of `first`. Positions are written in order, so this
/// only ever moves forward.
pub fn placeBatch(window: usize, capacity: usize, base: usize, first: usize, n: usize) Placement {
    std.debug.assert(first >= base);
    std.debug.assert(n > 0);
    const last = first + n - 1;
    if (last - base < capacity) return .{ .local = first - base, .base = base };

    // Keep the live window as of the batch's first position and drop what
    // the layer can no longer reach.
    const keep = @min(first -| windowStart(window, first), capacity);
    const new_base = first - keep;
    std.debug.assert(new_base >= base);
    return .{
        .local = keep,
        .move_from = new_base - base,
        .keep = keep,
        .base = new_base,
    };
}

/// Single-position convenience for the decode path.
pub fn place(window: usize, capacity: usize, base: usize, pos: usize) Placement {
    return placeBatch(window, capacity, base, pos, 1);
}

/// First absolute position still readable at `pos`, given the window.
pub fn windowStart(window: usize, pos: usize) usize {
    if (window == 0) return 0;
    return pos -| (window - 1);
}

test "capacityFor holds two windows, bounded by the context" {
    // Batch fits inside the window, so the window term dominates.
    try std.testing.expectEqual(@as(usize, 2048), capacityFor(1024, 512, 32768));
    // Batch larger than the window widens the headroom instead.
    try std.testing.expectEqual(@as(usize, 2048), capacityFor(512, 1536, 32768));
    try std.testing.expectEqual(@as(usize, 256), capacityFor(128, 64, 4096));
    // A window at least half the context saves nothing; do not grow past it.
    try std.testing.expectEqual(@as(usize, 4096), capacityFor(4096, 512, 4096));
    try std.testing.expectEqual(@as(usize, 4096), capacityFor(3000, 512, 4096));
    // Window 0 means "no window": the layer is global.
    try std.testing.expectEqual(@as(usize, 4096), capacityFor(0, 512, 4096));
}

test "place writes straight through until the cache is full" {
    const window: usize = 4;
    const cap = capacityFor(window, 1, 1024); // 8
    for (0..cap) |pos| {
        const p = place(window, cap, 0, pos);
        try std.testing.expectEqual(pos, p.local);
        try std.testing.expectEqual(@as(?usize, null), p.move_from);
        try std.testing.expectEqual(@as(usize, 0), p.base);
    }
}

test "place compacts once the end is reached and keeps the window contiguous" {
    const window: usize = 4;
    const cap = capacityFor(window, 1, 1024); // 8

    // Position 8 is one past the end: keep positions 5,6,7 and write 8 at
    // local 3, so the live window 5..8 sits contiguously at slots 0..3.
    const p = place(window, cap, 0, 8);
    try std.testing.expectEqual(@as(usize, 3), p.local);
    try std.testing.expectEqual(@as(?usize, 5), p.move_from);
    try std.testing.expectEqual(@as(usize, 3), p.keep);
    try std.testing.expectEqual(@as(usize, 5), p.base);

    // After it, writes run straight through again until the next end.
    for (9..13) |pos| {
        const q = place(window, cap, p.base, pos);
        try std.testing.expectEqual(pos - p.base, q.local);
        try std.testing.expectEqual(@as(?usize, null), q.move_from);
    }
}

test "the live window is always inside the cache and contiguous" {
    // Walk a whole sequence and check the invariant every step: every
    // position the layer can still read maps to a slot below capacity, and
    // they are consecutive.
    const window: usize = 5;
    const cap = capacityFor(window, 1, 4096); // 10
    var base: usize = 0;
    for (0..200) |pos| {
        const p = place(window, cap, base, pos);
        base = p.base;
        try std.testing.expect(p.local < cap);
        if (p.move_from) |from| {
            try std.testing.expect(from + p.keep <= cap);
            try std.testing.expect(p.keep < cap);
        }
        // Everything still readable sits at or after slot 0 and before the
        // write slot, in order.
        const lo = windowStart(window, pos);
        try std.testing.expect(lo >= base);
        try std.testing.expectEqual(p.local, pos - base);
        try std.testing.expect(lo - base <= p.local);
    }
}

test "compaction happens once per window of positions, not per token" {
    const window: usize = 8;
    const cap = capacityFor(window, 1, 4096); // 16
    var base: usize = 0;
    var moves: usize = 0;
    const steps: usize = 800;
    for (0..steps) |pos| {
        const p = place(window, cap, base, pos);
        if (p.move_from != null) moves += 1;
        base = p.base;
    }
    // One move buys `cap - window + 1` further writes, so the count is the
    // sequence length over that, not the sequence length.
    try std.testing.expect(moves > 0);
    try std.testing.expect(moves <= steps / (cap - window) + 2);
}

test "windowStart clamps at the start of the sequence" {
    try std.testing.expectEqual(@as(usize, 0), windowStart(1024, 0));
    try std.testing.expectEqual(@as(usize, 0), windowStart(1024, 512));
    try std.testing.expectEqual(@as(usize, 1), windowStart(1024, 1024));
    try std.testing.expectEqual(@as(usize, 977), windowStart(1024, 2000));
    // Window 0 means global: everything stays readable.
    try std.testing.expectEqual(@as(usize, 0), windowStart(0, 9999));
}

test "placeBatch keeps the base fixed for the whole batch" {
    // Tokens in one batch are written and then attended to, so a base that
    // moved between those steps would leave earlier tokens' local indices
    // pointing at the wrong slots. The compaction happens once, up front.
    const window: usize = 8;
    const batch: usize = 6;
    const cap = capacityFor(window, batch, 4096); // 16
    var base: usize = 0;

    // Three batches straight through: 0..5, 6..11, then 12..17 overflows.
    const p0 = placeBatch(window, cap, base, 0, batch);
    try std.testing.expectEqual(@as(?usize, null), p0.move_from);
    try std.testing.expectEqual(@as(usize, 0), p0.local);
    base = p0.base;

    const p1 = placeBatch(window, cap, base, 6, batch);
    try std.testing.expectEqual(@as(?usize, null), p1.move_from);
    try std.testing.expectEqual(@as(usize, 6), p1.local);
    base = p1.base;

    // 12..17: last is 17, 17 - 0 = 17 >= 16, so compact first.
    const p2 = placeBatch(window, cap, base, 12, batch);
    try std.testing.expect(p2.move_from != null);
    base = p2.base;
    // Every position in the batch lands inside the cache, in order, with
    // this one fixed base.
    for (0..batch) |i| {
        const local = (12 + i) - base;
        try std.testing.expect(local < cap);
        if (i > 0) try std.testing.expect(local == (12 + i - 1) - base + 1);
    }
    // And the window as of the batch's first position survived the move.
    try std.testing.expect(windowStart(window, 12) >= base);
}

test "a batch always fits after one compaction, for any batch up to capacity" {
    // This is the contract capacityFor exists to guarantee. Walk whole
    // sequences at several window and batch sizes and assert every position
    // of every batch lands in range.
    const windows = [_]usize{ 4, 8, 64, 1024 };
    const batches = [_]usize{ 1, 3, 8, 512 };
    for (windows) |window| {
        for (batches) |batch| {
            const cap = capacityFor(window, batch, 1 << 20);
            var base: usize = 0;
            var pos: usize = 0;
            while (pos < window * 12 + batch * 4) {
                const n = @min(batch, (window * 12 + batch * 4) - pos);
                const p = placeBatch(window, cap, base, pos, n);
                base = p.base;
                for (0..n) |i| {
                    const local = (pos + i) - base;
                    try std.testing.expect(local < cap);
                }
                // The live window at the end of the batch is still resident.
                try std.testing.expect(windowStart(window, pos + n - 1) >= base);
                pos += n;
            }
        }
    }
}

test "a windowed cache holds the same window a full cache would" {
    // The point of the whole module: for every position, the slots the
    // layer can still read must contain exactly the values a full-size
    // cache would have at those absolute positions. Simulated here with one
    // value per position so a wrong slot is a wrong number.
    const window: usize = 6;
    const batch: usize = 4;
    const total: usize = 200;
    const cap = capacityFor(window, batch, total);
    try std.testing.expect(cap < total); // actually saving memory

    var full: [total]f32 = @splat(0);
    var windowed: [64]f32 = @splat(0);
    try std.testing.expect(cap <= windowed.len);

    var base: usize = 0;
    var pos: usize = 0;
    while (pos < total) {
        const n = @min(batch, total - pos);
        const p = placeBatch(window, cap, base, pos, n);
        if (p.move_from) |from| {
            std.mem.copyForwards(f32, windowed[0..p.keep], windowed[from..][0..p.keep]);
            base = p.base;
        }
        // Write the batch into both caches.
        for (0..n) |i| {
            const abs = pos + i;
            const v: f32 = @floatFromInt(abs * 3 + 1);
            full[abs] = v;
            windowed[abs - base] = v;
        }
        // After each batch, every readable position must agree.
        const last = pos + n - 1;
        var r = windowStart(window, last);
        while (r <= last) : (r += 1) {
            try std.testing.expect(r >= base);
            try std.testing.expectEqual(full[r], windowed[r - base]);
        }
        pos += n;
    }
}

test "resetting the base restarts a sequence cleanly" {
    // resetCache zeroes the bases. A stale base would place the next
    // sequence's first token deep into the cache, or trip place's assert.
    const window: usize = 4;
    const cap = capacityFor(window, 2, 1024);
    var base: usize = 0;
    for (0..50) |pos| base = place(window, cap, base, pos).base;
    try std.testing.expect(base > 0);

    base = 0; // what resetCache does
    const p = place(window, cap, base, 0);
    try std.testing.expectEqual(@as(usize, 0), p.local);
    try std.testing.expectEqual(@as(usize, 0), p.base);
    try std.testing.expectEqual(@as(?usize, null), p.move_from);
}
