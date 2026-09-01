//! Byte-budgeted LRU bookkeeping for the GPU backends' weight caches.
//!
//! Every GPU backend caches uploaded weights in a map keyed by host address and
//! never evicts, so a model whose weights exceed VRAM cannot run at all: the
//! first pass allocates until the driver refuses. This tracks the same keys
//! under a byte budget and names which ones to drop, turning the cache into a
//! streaming residency window. The backend still owns the device buffers; this
//! only decides what stays.
//!
//! Allocation-free after `init`: the node array is fixed and the index map is
//! pre-sized to it, so `touch` and `admit` run in the token loop without
//! touching the allocator. Both are O(1) (intrusive LRU list), which an
//! eviction-per-layer workload needs: a DeepSeek-V4-scale checkpoint reaches
//! tens of thousands of live keys, where an O(n) victim scan would cost more
//! than the transfer it schedules.
//!
//! Not thread-safe. The backends dispatch weight uploads from the calling
//! thread, one op at a time, which is the same discipline their weight caches
//! already assume.

const std = @import("std");

/// Sentinel for "no node" in the intrusive list and the free chain.
const nil: u32 = std.math.maxInt(u32);

/// Eviction policy. See `WeightBudget.policy`.
pub const Policy = enum { lru, mru };

pub const WeightBudget = struct {
    const Node = struct {
        key: usize = 0,
        bytes: usize = 0,
        /// Toward the least-recently-used end.
        prev: u32 = nil,
        /// Toward the most-recently-used end.
        next: u32 = nil,
        live: bool = false,
    };

    nodes: []Node,
    index: std.AutoHashMapUnmanaged(usize, u32),
    /// Head of the free chain, linked through `next`.
    free_head: u32,
    /// Least-recently-used entry: the next eviction victim.
    lru: u32,
    /// Most-recently-used entry.
    mru: u32,
    budget_bytes: usize,
    used_bytes: usize,
    /// Which end of the recency list to evict from.
    ///
    /// LRU is right when reuse is skewed, as routed MoE experts are. It is the
    /// WORST possible choice for a cyclic scan, which is what a dense
    /// transformer's layer loop is: by the time the loop comes back to layer 0
    /// its weights are the least-recently-used, so a budget below the working set
    /// evicts exactly what is about to be needed and the hit rate collapses to
    /// roughly zero however large the budget is.
    ///
    /// Evicting the most-recently-used entry instead keeps whatever filled the
    /// budget first resident, so the hit rate becomes budget / working set rather
    /// than ~0.
    policy: Policy = .lru,
    /// Entries evicted since init, for reporting a thrashing configuration.
    evictions: u64,

    /// Track at most `capacity` cached weights under `budget_bytes`.
    ///
    /// `capacity` bounds distinct live keys, not bytes: a key beyond it is
    /// reported as un-admittable rather than silently replacing another, so the
    /// caller uploads it untracked instead of losing a buffer it still uses.
    pub fn init(allocator: std.mem.Allocator, capacity: usize, budget_bytes: usize) !WeightBudget {
        std.debug.assert(capacity > 0);
        const nodes = try allocator.alloc(Node, capacity);
        errdefer allocator.free(nodes);
        for (nodes, 0..) |*node, i| {
            node.* = .{ .next = if (i + 1 < capacity) @intCast(i + 1) else nil };
        }
        var index: std.AutoHashMapUnmanaged(usize, u32) = .empty;
        errdefer index.deinit(allocator);
        // Pre-size so admit() never allocates on the token path.
        try index.ensureTotalCapacity(allocator, @intCast(capacity));
        return .{
            .nodes = nodes,
            .index = index,
            .free_head = 0,
            .lru = nil,
            .mru = nil,
            .budget_bytes = budget_bytes,
            .used_bytes = 0,
            .evictions = 0,
        };
    }

    pub fn deinit(self: *WeightBudget, allocator: std.mem.Allocator) void {
        allocator.free(self.nodes);
        self.index.deinit(allocator);
        self.* = undefined;
    }

    /// Whether a budget is in force. Zero means unlimited: the caller keeps its
    /// previous grow-forever behavior and never has to special-case eviction.
    pub fn enabled(self: *const WeightBudget) bool {
        return self.budget_bytes > 0;
    }

    fn unlink(self: *WeightBudget, i: u32) void {
        const node = &self.nodes[i];
        if (node.prev != nil) self.nodes[node.prev].next = node.next else self.lru = node.next;
        if (node.next != nil) self.nodes[node.next].prev = node.prev else self.mru = node.prev;
        node.prev = nil;
        node.next = nil;
    }

    fn linkMru(self: *WeightBudget, i: u32) void {
        const node = &self.nodes[i];
        node.prev = self.mru;
        node.next = nil;
        if (self.mru != nil) self.nodes[self.mru].next = i else self.lru = i;
        self.mru = i;
    }

    /// Mark `key` most-recently used. Returns false if it is not tracked.
    pub fn touch(self: *WeightBudget, key: usize) bool {
        const i = self.index.get(key) orelse return false;
        if (self.mru == i) return true;
        self.unlink(i);
        self.linkMru(i);
        return true;
    }

    /// Bytes currently charged to `key`, or null if untracked.
    pub fn bytesOf(self: *const WeightBudget, key: usize) ?usize {
        const i = self.index.get(key) orelse return null;
        return self.nodes[i].bytes;
    }

    /// Drop `key` and uncharge its bytes. Returns what it held, or null.
    pub fn remove(self: *WeightBudget, key: usize) ?usize {
        const i = self.index.get(key) orelse return null;
        const bytes = self.nodes[i].bytes;
        self.unlink(i);
        self.used_bytes -= bytes;
        self.nodes[i].live = false;
        self.nodes[i].next = self.free_head;
        self.free_head = i;
        _ = self.index.remove(key);
        return bytes;
    }

    /// Result of asking to admit a key.
    pub const Admission = struct {
        /// Keys the caller must free device memory for, written into its buffer.
        evicted: []const usize,
        /// False when the entry cannot be tracked (no free node, or it alone
        /// exceeds the whole budget). The caller should still upload it, just
        /// untracked, so behavior degrades to the old grow-forever path rather
        /// than losing a weight the model needs.
        tracked: bool,
    };

    /// Make room for `bytes` under `key` and take a node for it.
    ///
    /// Evicts least-recently-used entries until the new entry fits, writing
    /// their keys into `evict_buf`; the caller frees exactly those device
    /// buffers. Eviction stops at `evict_buf.len` and the admission is reported
    /// untracked rather than overrunning, so a caller with a small buffer loses
    /// tracking, never memory it still references.
    ///
    /// Re-admitting a live key updates its size in place.
    pub fn admit(self: *WeightBudget, key: usize, bytes: usize, evict_buf: []usize) Admission {
        if (!self.enabled()) return .{ .evicted = evict_buf[0..0], .tracked = false };

        if (self.index.get(key)) |i| {
            self.used_bytes = self.used_bytes - self.nodes[i].bytes + bytes;
            self.nodes[i].bytes = bytes;
            self.unlink(i);
            self.linkMru(i);
            return .{ .evicted = evict_buf[0..0], .tracked = true };
        }

        // A single weight larger than the whole budget can never be resident
        // alongside anything else; tracking it would evict everything on every
        // touch. Leave it untracked and let the caller upload it directly.
        if (bytes > self.budget_bytes) return .{ .evicted = evict_buf[0..0], .tracked = false };

        var n_evicted: usize = 0;
        while (self.used_bytes + bytes > self.budget_bytes or self.free_head == nil) {
            const victim = if (self.policy == .lru) self.lru else self.mru;
            if (victim == nil) break; // nothing left to reclaim
            if (n_evicted == evict_buf.len) return .{ .evicted = evict_buf[0..n_evicted], .tracked = false };
            evict_buf[n_evicted] = self.nodes[victim].key;
            n_evicted += 1;
            _ = self.remove(self.nodes[victim].key);
            self.evictions += 1;
        }

        const i = self.free_head;
        if (i == nil) return .{ .evicted = evict_buf[0..n_evicted], .tracked = false };
        self.free_head = self.nodes[i].next;
        self.nodes[i] = .{ .key = key, .bytes = bytes, .live = true };
        self.linkMru(i);
        self.used_bytes += bytes;
        self.index.putAssumeCapacity(key, i);
        return .{ .evicted = evict_buf[0..n_evicted], .tracked = true };
    }

    /// Change the budget at runtime, naming what no longer fits.
    ///
    /// Raising it evicts nothing. Lowering it evicts least-recently-used entries
    /// until the new budget holds, so VRAM can be handed to another consumer
    /// without reloading weights. Stops at `evict_buf.len`; the caller re-runs
    /// with the same target to continue draining.
    pub fn setBudget(self: *WeightBudget, bytes: usize, evict_buf: []usize) []const usize {
        self.budget_bytes = bytes;
        if (!self.enabled()) return evict_buf[0..0];
        var n: usize = 0;
        while (self.used_bytes > self.budget_bytes and n < evict_buf.len) {
            const victim = if (self.policy == .lru) self.lru else self.mru;
            if (victim == nil) break;
            evict_buf[n] = self.nodes[victim].key;
            n += 1;
            _ = self.remove(self.nodes[victim].key);
            self.evictions += 1;
        }
        return evict_buf[0..n];
    }
};

// ── Tests ────────────────────────────────────────────────────────

const testing = std.testing;

test "WeightBudget, disabled budget tracks nothing" {
    var wb = try WeightBudget.init(testing.allocator, 4, 0);
    defer wb.deinit(testing.allocator);
    var buf: [4]usize = undefined;
    try testing.expect(!wb.enabled());
    const a = wb.admit(1, 100, &buf);
    try testing.expect(!a.tracked);
    try testing.expectEqual(@as(usize, 0), a.evicted.len);
    try testing.expect(!wb.touch(1));
}

test "WeightBudget, admits until the budget is full then evicts LRU" {
    var wb = try WeightBudget.init(testing.allocator, 8, 300);
    defer wb.deinit(testing.allocator);
    var buf: [8]usize = undefined;

    for ([_]usize{ 1, 2, 3 }) |k| {
        const a = wb.admit(k, 100, &buf);
        try testing.expect(a.tracked);
        try testing.expectEqual(@as(usize, 0), a.evicted.len);
    }
    try testing.expectEqual(@as(usize, 300), wb.used_bytes);

    // 1 is the least recently used, so it is the victim.
    const a = wb.admit(4, 100, &buf);
    try testing.expect(a.tracked);
    try testing.expectEqualSlices(usize, &.{1}, a.evicted);
    try testing.expectEqual(@as(usize, 300), wb.used_bytes);
    try testing.expect(!wb.touch(1));
    try testing.expect(wb.touch(2));
}

test "WeightBudget, touch reorders the victim" {
    var wb = try WeightBudget.init(testing.allocator, 8, 300);
    defer wb.deinit(testing.allocator);
    var buf: [8]usize = undefined;
    for ([_]usize{ 1, 2, 3 }) |k| _ = wb.admit(k, 100, &buf);

    try testing.expect(wb.touch(1)); // 1 is now MRU, 2 becomes the victim
    const a = wb.admit(4, 100, &buf);
    try testing.expectEqualSlices(usize, &.{2}, a.evicted);
    try testing.expect(wb.touch(1));
    try testing.expect(wb.touch(3));
}

test "WeightBudget, one entry larger than the budget stays untracked" {
    var wb = try WeightBudget.init(testing.allocator, 4, 100);
    defer wb.deinit(testing.allocator);
    var buf: [4]usize = undefined;
    _ = wb.admit(1, 50, &buf);

    const a = wb.admit(2, 500, &buf);
    try testing.expect(!a.tracked);
    // Nothing was evicted to make room for something that could never fit.
    try testing.expectEqual(@as(usize, 0), a.evicted.len);
    try testing.expect(wb.touch(1));
}

test "WeightBudget, re-admitting a live key resizes in place" {
    var wb = try WeightBudget.init(testing.allocator, 4, 300);
    defer wb.deinit(testing.allocator);
    var buf: [4]usize = undefined;
    _ = wb.admit(1, 100, &buf);
    const a = wb.admit(1, 250, &buf);
    try testing.expect(a.tracked);
    try testing.expectEqual(@as(usize, 0), a.evicted.len);
    try testing.expectEqual(@as(usize, 250), wb.used_bytes);
    try testing.expectEqual(@as(?usize, 250), wb.bytesOf(1));
}

test "WeightBudget, capacity exhaustion evicts to reclaim a node" {
    // Budget is generous; the node array is the binding constraint.
    var wb = try WeightBudget.init(testing.allocator, 2, 1 << 20);
    defer wb.deinit(testing.allocator);
    var buf: [4]usize = undefined;
    _ = wb.admit(1, 10, &buf);
    _ = wb.admit(2, 10, &buf);
    const a = wb.admit(3, 10, &buf);
    try testing.expect(a.tracked);
    try testing.expectEqualSlices(usize, &.{1}, a.evicted);
    try testing.expectEqual(@as(usize, 20), wb.used_bytes);
}

test "WeightBudget, a full evict buffer reports untracked instead of overrunning" {
    var wb = try WeightBudget.init(testing.allocator, 8, 300);
    defer wb.deinit(testing.allocator);
    var big: [8]usize = undefined;
    for ([_]usize{ 1, 2, 3 }) |k| _ = wb.admit(k, 100, &big);

    // Needs three evictions to fit 300 bytes, but only one slot to report them.
    var small: [1]usize = undefined;
    const a = wb.admit(9, 300, &small);
    try testing.expect(!a.tracked);
    try testing.expectEqual(@as(usize, 1), a.evicted.len);
    try testing.expectEqual(@as(usize, 1), a.evicted[0]);
}

test "WeightBudget, remove uncharges and frees the node" {
    var wb = try WeightBudget.init(testing.allocator, 2, 300);
    defer wb.deinit(testing.allocator);
    var buf: [4]usize = undefined;
    _ = wb.admit(1, 100, &buf);
    _ = wb.admit(2, 100, &buf);
    try testing.expectEqual(@as(?usize, 100), wb.remove(1));
    try testing.expectEqual(@as(?usize, null), wb.remove(1));
    try testing.expectEqual(@as(usize, 100), wb.used_bytes);
    // The freed node is reusable.
    const a = wb.admit(3, 100, &buf);
    try testing.expect(a.tracked);
    try testing.expectEqual(@as(usize, 0), a.evicted.len);
}

test "WeightBudget, setBudget shrinks by evicting LRU first" {
    var wb = try WeightBudget.init(testing.allocator, 8, 400);
    defer wb.deinit(testing.allocator);
    var buf: [8]usize = undefined;
    for ([_]usize{ 1, 2, 3, 4 }) |k| _ = wb.admit(k, 100, &buf);

    const dropped = wb.setBudget(200, &buf);
    try testing.expectEqualSlices(usize, &.{ 1, 2 }, dropped);
    try testing.expectEqual(@as(usize, 200), wb.used_bytes);
    try testing.expect(wb.touch(3));
    try testing.expect(wb.touch(4));

    // Raising it evicts nothing.
    const none = wb.setBudget(1000, &buf);
    try testing.expectEqual(@as(usize, 0), none.len);
    try testing.expectEqual(@as(usize, 200), wb.used_bytes);
}

test "WeightBudget, setBudget to zero disables tracking without evicting" {
    var wb = try WeightBudget.init(testing.allocator, 4, 300);
    defer wb.deinit(testing.allocator);
    var buf: [4]usize = undefined;
    _ = wb.admit(1, 100, &buf);
    const dropped = wb.setBudget(0, &buf);
    try testing.expectEqual(@as(usize, 0), dropped.len);
    try testing.expect(!wb.enabled());
}

test "WeightBudget, mru survives a cyclic scan where lru collapses" {
    // A dense transformer's layer loop is a cyclic scan. Under LRU, a budget
    // below the working set evicts exactly the entry the next cycle needs first,
    // so the hit rate is ~0 no matter how large the budget is. MRU keeps whatever
    // filled it, so the hit rate is budget / working set.
    const working_set = 40;
    const budget = 30 * 100; // room for 30 of the 40

    inline for (.{ Policy.lru, Policy.mru }) |policy| {
        var wb = try WeightBudget.init(testing.allocator, 64, budget);
        defer wb.deinit(testing.allocator);
        wb.policy = policy;
        var buf: [64]usize = undefined;

        var hits: usize = 0;
        var refs: usize = 0;
        var cycle: usize = 0;
        while (cycle < 20) : (cycle += 1) {
            for (0..working_set) |i| {
                refs += 1;
                if (wb.touch(i)) hits += 1 else _ = wb.admit(i, 100, &buf);
            }
        }
        if (policy == .lru) {
            // Every reference misses: the entry was evicted one cycle ago.
            try testing.expectEqual(@as(usize, 0), hits);
        } else {
            // 30 of every 40 references hit, minus the first cycle's cold start.
            try testing.expect(hits * 100 > refs * 65);
        }
    }
}

test "WeightBudget, keeps a hot set resident against a cold stream" {
    // A routed-expert stream: a small hot set plus a long cold tail. LRU is
    // worth having exactly when the hot set hits far more often than the tail;
    // it does not promise the hot set is NEVER evicted, and at this slot count
    // a hot key can be pushed out between touches.
    var wb = try WeightBudget.init(testing.allocator, 64, 16 * 100);
    defer wb.deinit(testing.allocator);
    var buf: [64]usize = undefined;

    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rand = prng.random();
    const hot = [_]usize{ 1, 2, 3, 4 };
    var hot_refs: usize = 0;
    var hot_hits: usize = 0;
    var cold_refs: usize = 0;
    var cold_hits: usize = 0;

    var step: usize = 0;
    while (step < 20000) : (step += 1) {
        const is_hot = step % 2 == 0;
        const key = if (is_hot) hot[rand.uintLessThan(usize, hot.len)] else 1000 + rand.uintLessThan(usize, 200);
        const hit = wb.touch(key);
        if (is_hot) {
            hot_refs += 1;
            hot_hits += @intFromBool(hit);
        } else {
            cold_refs += 1;
            cold_hits += @intFromBool(hit);
        }
        if (!hit) _ = wb.admit(key, 100, &buf);
        try testing.expect(wb.used_bytes <= wb.budget_bytes);
    }

    // 4 hot keys in 16 slots against a 200-key tail: the hot set should almost
    // always be resident, the tail almost never.
    try testing.expect(hot_hits * 10 > hot_refs * 9); // > 90%
    try testing.expect(cold_hits * 4 < cold_refs); // < 25%
}

test "WeightBudget, used_bytes stays consistent with the live nodes" {
    var wb = try WeightBudget.init(testing.allocator, 16, 8 * 100);
    defer wb.deinit(testing.allocator);
    var buf: [16]usize = undefined;

    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rand = prng.random();
    var step: usize = 0;
    while (step < 5000) : (step += 1) {
        const key = rand.uintLessThan(usize, 40);
        switch (rand.uintLessThan(u8, 4)) {
            0 => _ = wb.remove(key),
            1 => _ = wb.touch(key),
            else => if (!wb.touch(key)) {
                _ = wb.admit(key, 50 + rand.uintLessThan(usize, 50), &buf);
            },
        }
        // The charged total must equal the sum over live nodes, and the index
        // must name exactly those nodes.
        var sum: usize = 0;
        var live: usize = 0;
        for (wb.nodes) |node| {
            if (node.live) {
                sum += node.bytes;
                live += 1;
            }
        }
        try testing.expectEqual(sum, wb.used_bytes);
        try testing.expectEqual(live, wb.index.count());
        try testing.expect(wb.used_bytes <= wb.budget_bytes);
    }
}
