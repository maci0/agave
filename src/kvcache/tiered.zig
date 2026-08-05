//! Three-tier KV cache (VRAM + RAM + SSD) with automatic promotion/demotion
//! based on memory pressure.
//!
//! Provides memory-bounded serving at scale by spilling cold KV blocks from
//! VRAM to RAM (and eventually SSD) when VRAM budget is exceeded. Blocks are
//! automatically promoted back to VRAM when accessed by active requests.
//!
//! Tier migration uses frequency×cost eviction (NOT simple LRU) to protect
//! shared prefix blocks from thrashing.

const std = @import("std");
const builtin = @import("builtin");
const is_freestanding = builtin.os.tag == .freestanding;
const Io = if (is_freestanding) struct {
    pub const File = void;
} else std.Io;
const SsdFile = if (is_freestanding) void else Io.File;
const sim_clock = @import("../sim_clock.zig");

/// Millisecond timestamp (injectable via sim_clock; 0 on freestanding).
fn milliTimestamp() i64 {
    return sim_clock.milliNow();
}
const Allocator = std.mem.Allocator;

/// Delete a file by path using C unlink (avoids Io dependency).
fn deleteFileByPath(path: []const u8) void {
    if (comptime is_freestanding) return;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len >= buf.len) return;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    _ = std.c.unlink(@ptrCast(buf[0..path.len :0]));
}

/// Write all bytes to a file at the given offset using positioned I/O (pwrite).
/// This avoids seek+write races when multiple threads access the file.
fn pwriteAll(file: anytype, bytes: []const u8, offset: usize) !void {
    if (comptime is_freestanding) return error.WriteError;
    var written: usize = 0;
    while (written < bytes.len) {
        const result = std.c.pwrite(file.handle, bytes[written..].ptr, bytes[written..].len, @intCast(offset + written));
        const n: isize = @bitCast(result);
        if (n <= 0) return error.WriteError;
        written += @intCast(n);
    }
}

/// Read all bytes from a file at the given offset using positioned I/O (pread).
fn preadAll(file: anytype, buf: []u8, offset: usize) !usize {
    if (comptime is_freestanding) return 0;
    var total: usize = 0;
    while (total < buf.len) {
        const result = std.c.pread(file.handle, buf[total..].ptr, buf[total..].len, @intCast(offset + total));
        const n: isize = @bitCast(result);
        if (n <= 0) break;
        total += @intCast(n);
    }
    return total;
}
const manager = @import("manager.zig");
const CacheBlock = manager.CacheBlock;

/// VRAM usage threshold for triggering demotion (90% = evict at 10% free).
const vram_eviction_threshold: f32 = 0.90;
/// Integer numerator/denominator for threshold check without float division:
/// vram_used/vram_total > 0.90  ⟺  vram_used * eviction_denom > vram_total * eviction_numer
const eviction_numer: usize = 9;
const eviction_denom: usize = 10;
/// Eviction cost multiplier for shared prefix blocks (ref_count > 1).
/// Higher cost protects shared prefixes from thrashing.
const shared_prefix_cost: f32 = 100.0;

/// Block tier enum: VRAM (T0), RAM (T1), SSD (T2).
pub const BlockTier = enum {
    vram,
    ram,
    ssd,
};

/// Extended cache block with tier tracking and eviction metadata.
pub const TieredBlock = struct {
    /// Embedded base CacheBlock (keys, values, used, ref_count).
    base: CacheBlock,
    /// Current tier location.
    tier: BlockTier = .vram,
    /// Access count for frequency×cost eviction.
    access_count: u32 = 0,
    /// Last access timestamp (milliseconds) for LRU within tier.
    last_access_ms: i64 = 0,

    /// SSD tier support: file offset for spilled blocks.
    ssd_offset: ?u64 = null,
};

/// Callback for physical data transfer between tiers on discrete GPUs.
/// UMA platforms (Apple Silicon, NVIDIA GB10) use null (no-op — shared memory).
/// Discrete GPUs implement this to copy data via cuMemcpyAsync / hipMemcpyAsync.
pub const TransferCallback = struct {
    /// Copy block data from host (RAM) to device (VRAM).
    /// Called during promoteToVram on discrete GPUs.
    upload: ?*const fn (ctx: *anyopaque, dst_dev: [*]u8, src_host: [*]const u8, bytes: usize) void = null,
    /// Copy block data from device (VRAM) to host (RAM).
    /// Called during demoteToRam on discrete GPUs.
    download: ?*const fn (ctx: *anyopaque, dst_host: [*]u8, src_dev: [*]const u8, bytes: usize) void = null,
    /// Flush all pending async transfers (batch sync point).
    sync: ?*const fn (ctx: *anyopaque) void = null,
    /// Opaque backend context (e.g., CUDA stream, HIP stream).
    ctx: ?*anyopaque = null,

    /// No-op transfer for UMA platforms (default).
    pub const uma_noop = TransferCallback{};

    pub fn isActive(self: TransferCallback) bool {
        return self.upload != null;
    }
};

/// Tiered KV cache allocator supporting VRAM, RAM, and SSD tiers.
/// Blocks are allocated from highest available tier, with automatic demotion
/// when VRAM usage exceeds threshold (90%).
///
/// SSD tier (Plan 03): Sparse file with fixed-size block slots.
/// Block offset = block_id × block_bytes. Blocks spilled to SSD have no RAM
/// backing (empty slices), restored on-demand via promoteFromSsd().
pub const TieredKvCache = struct {
    /// All blocks (VRAM + RAM + SSD).
    blocks: []TieredBlock,
    /// Free list for VRAM tier blocks.
    vram_free_list: std.ArrayList(u32),
    /// Free list for RAM tier blocks.
    ram_free_list: std.ArrayList(u32),
    /// Free list for SSD tier blocks (Plan 03).
    ssd_free_list: std.ArrayList(u32),

    /// Tier budgets (in blocks).
    vram_block_count: usize,
    ram_block_count: usize,
    ssd_block_count: usize,

    /// Number of VRAM blocks currently in use (avoids O(n) scan).
    /// Atomic: accessed by both scheduler and prefetch worker threads.
    vram_used: std.atomic.Value(usize),
    /// Number of RAM blocks currently in use (avoids O(n) scan).
    /// Atomic: accessed by both scheduler and prefetch worker threads.
    ram_used: std.atomic.Value(usize),

    /// Block size (tokens per block).
    block_size: u16,
    /// KV dimension per position.
    kv_dim: usize,
    /// Number of model layers.
    n_layers: usize,
    /// Allocator for block memory.
    allocator: Allocator,

    /// SSD tier support (Plan 03).
    /// Sparse file handle for KV block spill/restore.
    ssd_file: ?SsdFile = null,
    /// Path to SSD sparse file.
    ssd_path: []const u8,
    /// Bytes per block (kv_dim × block_size × @sizeOf(f32) × 2).
    block_bytes: usize,

    /// GPU transfer callback for discrete GPUs (null = UMA zero-copy).
    transfer: TransferCallback = TransferCallback.uma_noop,

    /// Tier-mutation lock: serializes promoteFromSsd/promoteToVram/demoteToRam
    /// across scheduler and prefetch worker threads. Short critical sections
    /// (microseconds) make spin-wait acceptable.
    tier_lock: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),

    pub fn lockTier(self: *TieredKvCache) void {
        while (self.tier_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null)
            std.atomic.spinLoopHint();
    }

    pub fn unlockTier(self: *TieredKvCache) void {
        self.tier_lock.store(0, .release);
    }

    /// Initialize tiered cache with VRAM, RAM, and SSD budgets.
    ///
    /// Parameters:
    ///   - allocator: Memory allocator.
    ///   - n_layers: Number of model layers.
    ///   - kv_dim: KV dimension per position.
    ///   - vram_blocks: Number of VRAM tier blocks (from existing PagedKvCache budget).
    ///   - ram_blocks: Number of RAM tier blocks (auto-detect 50% of free system RAM, or user-specified).
    ///   - ssd_blocks: Number of SSD tier blocks (virtual allocation, no RAM backing).
    ///   - block_size: Tokens per block (default 16).
    ///   - ssd_path: Optional path to SSD sparse file. If null, SSD tier disabled.
    ///
    /// Returns: TieredKvCache with initialized VRAM, RAM, and SSD tiers.
    pub fn init(
        allocator: Allocator,
        n_layers: usize,
        kv_dim: usize,
        vram_blocks: usize,
        ram_blocks: usize,
        ssd_blocks: usize,
        block_size: u16,
        ssd_path: ?[]const u8,
    ) !TieredKvCache {
        const total_blocks = std.math.add(usize, std.math.add(usize, vram_blocks, ram_blocks) catch return error.OutOfMemory, ssd_blocks) catch return error.OutOfMemory;
        const blocks = try allocator.alloc(TieredBlock, total_blocks);
        errdefer allocator.free(blocks);

        var vram_free: std.ArrayList(u32) = .empty;
        errdefer vram_free.deinit(allocator);
        var ram_free: std.ArrayList(u32) = .empty;
        errdefer ram_free.deinit(allocator);
        var ssd_free: std.ArrayList(u32) = .empty;
        errdefer ssd_free.deinit(allocator);

        // Reserve capacity for free lists
        try vram_free.ensureTotalCapacity(allocator, vram_blocks);
        try ram_free.ensureTotalCapacity(allocator, ram_blocks);
        try ssd_free.ensureTotalCapacity(allocator, ssd_blocks);

        // Calculate block bytes (keys + values)
        const slot_size = std.math.mul(usize, @as(usize, block_size), kv_dim) catch return error.OutOfMemory;
        const block_bytes = std.math.mul(usize, slot_size, @sizeOf(f32) * 2) catch return error.OutOfMemory; // keys + values

        // Create sparse file for SSD tier if path provided
        var ssd_file: ?SsdFile = null;
        if (ssd_path) |path| {
            // Reject path traversal, null bytes, and absolute paths in user-supplied SSD cache path.
            if (std.mem.indexOf(u8, path, "..") != null or
                std.mem.indexOfScalar(u8, path, 0) != null or
                (path.len > 0 and (path[0] == '/' or path[0] == '\\')))
                return error.InvalidPath;
            const fd = try std.posix.openat(std.posix.AT.FDCWD, path, .{
                .ACCMODE = .RDWR,
                .CREAT = true,
            }, 0o644);
            ssd_file = .{ .handle = fd, .flags = .{ .nonblocking = false } };
            errdefer {
                _ = std.c.close(ssd_file.?.handle);
                deleteFileByPath(path);
            }

            // Allocate sparse file (write one byte at end to reserve space)
            const total_size = std.math.mul(usize, ssd_blocks, block_bytes) catch return error.OutOfMemory;
            if (total_size > 0) {
                const zero_buf: [1]u8 = .{0};
                const result: isize = @bitCast(std.c.pwrite(ssd_file.?.handle, &zero_buf, 1, @intCast(total_size - 1)));
                if (result != 1) {
                    std.log.err("SSD sparse file allocation failed (pwrite returned {d})", .{result});
                    return error.WriteError;
                }
            }

            std.log.debug("Created SSD sparse file: {s} ({d} blocks, {d} bytes)", .{ path, ssd_blocks, total_size });
        }

        var init_count: usize = 0;
        errdefer {
            for (0..init_count) |i| {
                allocator.free(blocks[i].base.keys);
                allocator.free(blocks[i].base.values);
            }
        }

        // Initialize VRAM tier blocks
        for (0..vram_blocks) |i| {
            const keys = try allocator.alloc(f32, slot_size);
            errdefer allocator.free(keys);
            const values = try allocator.alloc(f32, slot_size);
            errdefer allocator.free(values);
            blocks[i] = .{
                .base = .{ .keys = keys, .values = values, .ref_count = 0 },
                .tier = .vram,
            };
            vram_free.appendAssumeCapacity(@intCast(i));
            init_count = i + 1;
        }

        // Initialize RAM tier blocks
        for (vram_blocks..(vram_blocks + ram_blocks)) |i| {
            const keys = try allocator.alloc(f32, slot_size);
            errdefer allocator.free(keys);
            const values = try allocator.alloc(f32, slot_size);
            errdefer allocator.free(values);
            blocks[i] = .{
                .base = .{ .keys = keys, .values = values, .ref_count = 0 },
                .tier = .ram,
            };
            ram_free.appendAssumeCapacity(@intCast(i));
            init_count = i + 1;
        }

        // Initialize SSD tier blocks (virtual allocation, no RAM backing)
        for ((vram_blocks + ram_blocks)..total_blocks) |i| {
            blocks[i] = .{
                .base = .{
                    .keys = &[_]f32{}, // Empty slice (data on disk)
                    .values = &[_]f32{},
                    .ref_count = 0,
                },
                .tier = .ssd,
                .ssd_offset = null, // Assigned on first spill
            };
            ssd_free.appendAssumeCapacity(@intCast(i));
            init_count = i + 1;
        }

        return .{
            .blocks = blocks,
            .vram_free_list = vram_free,
            .ram_free_list = ram_free,
            .ssd_free_list = ssd_free,
            .vram_used = std.atomic.Value(usize).init(0),
            .ram_used = std.atomic.Value(usize).init(0),
            .vram_block_count = vram_blocks,
            .ram_block_count = ram_blocks,
            .ssd_block_count = ssd_blocks,
            .block_size = block_size,
            .kv_dim = kv_dim,
            .n_layers = n_layers,
            .allocator = allocator,
            .ssd_file = ssd_file,
            .ssd_path = ssd_path orelse "",
            .block_bytes = block_bytes,
        };
    }

    /// Free all cache blocks and free lists.
    pub fn deinit(self: *TieredKvCache) void {
        // Close SSD file if open
        if (self.ssd_file) |f| {
            _ = std.c.close(f.handle);
        }

        for (self.blocks) |*blk| {
            // Only free non-empty slices (SSD blocks have empty slices)
            if (blk.base.keys.len > 0) self.allocator.free(blk.base.keys);
            if (blk.base.values.len > 0) self.allocator.free(blk.base.values);
        }
        self.allocator.free(self.blocks);
        self.vram_free_list.deinit(self.allocator);
        self.ram_free_list.deinit(self.allocator);
        self.ssd_free_list.deinit(self.allocator);
    }

    /// Allocate a block from highest available tier.
    ///
    /// Allocation state machine:
    /// 1. Try VRAM free list first (fastest tier)
    /// 2. If VRAM exhausted AND VRAM usage >90%, demote coldest VRAM block to RAM
    /// 3. Retry allocation from VRAM (now has 1 free block from demotion)
    /// 4. Fallback to RAM tier if VRAM still unavailable
    /// 5. Fallback to SSD tier (promote from SSD to RAM before use)
    /// 6. Return error if all tiers exhausted
    ///
    /// Automatically triggers demotion when VRAM usage exceeds 90% threshold.
    ///
    /// Returns: Physical block ID.
    pub fn allocBlock(self: *TieredKvCache) !u32 {
        self.lockTier();
        defer self.unlockTier();
        return self.allocBlockInner();
    }

    /// Allocate without acquiring tier_lock. Caller must hold tier_lock.
    fn allocBlockInner(self: *TieredKvCache) !u32 {
        // Try VRAM first — loop with bounded demotion attempts (avoids unbounded recursion)
        var demotions: usize = 0;
        const max_demotions: usize = 16;
        while (demotions < max_demotions) : (demotions += 1) {
            if (self.vram_free_list.items.len > 0) {
                const block_id = self.vram_free_list.pop().?;
                self.blocks[block_id].tier = .vram;
                self.blocks[block_id].base.ref_count = 1;
                self.blocks[block_id].base.used = 0;
                _ = self.vram_used.fetchAdd(1, .monotonic);
                return block_id;
            }

            // VRAM exhausted — check if we can evict (trigger at vram_eviction_threshold)
            // Integer comparison avoids float division on allocation path.
            const vram_used = self.vram_used.load(.monotonic);
            const vram_total = self.vram_block_count;

            if (vram_used * eviction_denom > vram_total * eviction_numer) {
                // Demote coldest VRAM block to RAM (block stays in-use, just re-tagged)
                const evicted = try self.demoteToRam();
                std.log.debug("Demoted block {d} from VRAM to RAM (usage: {d:.1}%)", .{
                    evicted,
                    @as(f32, @floatFromInt(vram_used)) / @as(f32, @floatFromInt(vram_total)) * 100.0,
                });
                continue; // retry — vram_used decreased, eventually falls below threshold
            }
            break; // no eviction possible, fall through to RAM/SSD
        }

        // Fallback to RAM tier
        if (self.ram_free_list.items.len > 0) {
            const block_id = self.ram_free_list.pop().?;
            self.blocks[block_id].tier = .ram;
            self.blocks[block_id].base.ref_count = 1;
            self.blocks[block_id].base.used = 0;
            _ = self.ram_used.fetchAdd(1, .monotonic);
            return block_id;
        }

        // Fallback to SSD tier (promote from SSD to RAM before use)
        if (self.ssd_free_list.items.len > 0) {
            const block_id = self.ssd_free_list.pop().?;
            // Promote from SSD to RAM before use (inner: lock already held)
            try self.promoteFromSsdInner(block_id);
            self.blocks[block_id].base.ref_count = 1;
            self.blocks[block_id].base.used = 0;
            return block_id;
        }

        // Out of memory across all tiers
        return error.OutOfKvMemory;
    }

    /// Demote coldest VRAM block to RAM tier.
    ///
    /// Uses frequency×cost eviction (access_count × compute_cost).
    /// Shared prefixes (ref_count > 1) have 100× higher cost to prevent thrashing.
    /// Last block of sequence evicted first (minimize shared prefix eviction).
    ///
    /// If RAM is full, demotes coldest RAM block to SSD first.
    ///
    /// Returns: Block ID of demoted block.
    fn demoteToRam(self: *TieredKvCache) !u32 {
        // Check if RAM is full — if so, demote RAM → SSD first
        const ram_used = self.ram_used.load(.monotonic);
        if (ram_used >= self.ram_block_count and self.ssd_block_count > 0) {
            const ram_victim = try self.selectColdestRamBlock();
            try self.demoteToSsd(ram_victim);
            std.log.debug("RAM full — demoted block {d} to SSD before VRAM→RAM demotion", .{ram_victim});
        }

        var min_score: f32 = std.math.floatMax(f32);
        var victim_id: u32 = 0;
        var found = false;

        for (self.blocks[0..self.vram_block_count], 0..) |*blk, id| {
            if (blk.tier != .vram) continue; // Only demote from VRAM
            if (blk.base.ref_count == 0) continue; // Skip free blocks

            // Frequency × cost metric (shared prefixes prioritized)
            const cost: f32 = if (blk.base.ref_count > 1) shared_prefix_cost else 1.0;
            const score = @as(f32, @floatFromInt(blk.access_count)) * cost;

            if (score < min_score) {
                min_score = score;
                victim_id = @intCast(id);
                found = true;
            }
        }

        if (!found) return error.NoVramBlocksToDemote;

        // Download data from VRAM → RAM on discrete GPUs (no-op on UMA)
        if (self.transfer.isActive()) {
            const vblk = &self.blocks[victim_id];
            const key_bytes = std.mem.sliceAsBytes(vblk.base.keys);
            const val_bytes = std.mem.sliceAsBytes(vblk.base.values);
            self.transfer.download.?(self.transfer.ctx.?, key_bytes.ptr, key_bytes.ptr, key_bytes.len);
            self.transfer.download.?(self.transfer.ctx.?, val_bytes.ptr, val_bytes.ptr, val_bytes.len);
        }

        self.blocks[victim_id].tier = .ram;
        _ = self.vram_used.fetchSub(1, .monotonic);
        _ = self.ram_used.fetchAdd(1, .monotonic);

        return victim_id;
    }

    /// Select coldest RAM block for SSD demotion.
    /// Uses same frequency×cost metric as demoteToRam.
    fn selectColdestRamBlock(self: *const TieredKvCache) !u32 {
        var min_score: f32 = std.math.floatMax(f32);
        var victim_id: u32 = 0;
        var found = false;

        const ram_start = self.vram_block_count;
        const ram_end = ram_start + self.ram_block_count;

        for (self.blocks[ram_start..ram_end], ram_start..) |*blk, id| {
            if (blk.tier != .ram) continue; // Only select RAM blocks
            if (blk.base.ref_count == 0) continue; // Skip free blocks

            const cost: f32 = if (blk.base.ref_count > 1) shared_prefix_cost else 1.0;
            const score = @as(f32, @floatFromInt(blk.access_count)) * cost;

            if (score < min_score) {
                min_score = score;
                victim_id = @intCast(id);
                found = true;
            }
        }

        if (!found) return error.NoRamBlocksToEvict;
        return victim_id;
    }

    /// Spill RAM block to SSD tier.
    ///
    /// Writes block data to SSD file at fixed offset (block_id × block_bytes).
    /// Sparse file format: each block occupies a fixed slot regardless of content.
    ///
    /// After spill, frees RAM backing and sets tier tag to .ssd.
    ///
    /// Returns: void on success.
    fn demoteToSsd(self: *TieredKvCache, block_id: u32) !void {
        var blk = &self.blocks[block_id];
        std.debug.assert(blk.tier == .ram); // Only demote from RAM

        const ssd = self.ssd_file orelse return error.SsdNotConfigured;

        // Calculate fixed offset for this block
        const offset = std.math.mul(u64, @as(u64, block_id), @as(u64, self.block_bytes)) catch return error.Overflow;

        // Write keys + values to SSD using positioned I/O (pwrite) to avoid
        // seek+write races when prefetcher and scheduler access the file concurrently.
        const keys_bytes = std.mem.sliceAsBytes(blk.base.keys);
        const values_bytes = std.mem.sliceAsBytes(blk.base.values);
        try pwriteAll(ssd, keys_bytes, @intCast(offset));
        try pwriteAll(ssd, values_bytes, @intCast(offset + keys_bytes.len));

        // Free RAM backing
        self.allocator.free(blk.base.keys);
        self.allocator.free(blk.base.values);
        blk.base.keys = &[_]f32{};
        blk.base.values = &[_]f32{};

        // Update tier metadata
        blk.tier = .ssd;
        blk.ssd_offset = offset;
        _ = self.ram_used.fetchSub(1, .monotonic);

        std.log.debug("Demoted block {d} from RAM to SSD (offset: {d})", .{ block_id, offset });
    }

    /// Restore SSD block to RAM tier.
    ///
    /// Reads block data from SSD file at stored offset.
    /// Allocates RAM backing and loads keys + values.
    ///
    /// Parameters:
    ///   - block_id: Physical block ID to promote.
    ///
    /// Returns: void on success.
    /// Promote SSD block to RAM. Caller must hold tier_lock.
    fn promoteFromSsdInner(self: *TieredKvCache, block_id: u32) !void {
        var blk = &self.blocks[block_id];
        if (blk.tier != .ssd) return; // Already promoted by another thread

        const ssd = self.ssd_file orelse return error.SsdNotConfigured;
        // Validate offset before allocating so a failed promote never leaves
        // dangling pointers on the block (errdefer would free but not clear).
        const offset = blk.ssd_offset orelse return error.SsdBlockNotSpilled;
        const slot_size = std.math.mul(usize, @as(usize, self.block_size), self.kv_dim) catch return error.OutOfMemory;

        const keys = try self.allocator.alloc(f32, slot_size);
        errdefer self.allocator.free(keys);
        const values = try self.allocator.alloc(f32, slot_size);
        errdefer self.allocator.free(values);

        // Read from SSD using positioned I/O (pread) to avoid
        // seek+read races when prefetcher and scheduler access the file concurrently.
        const keys_bytes = std.mem.sliceAsBytes(keys);
        const values_bytes = std.mem.sliceAsBytes(values);
        _ = try preadAll(ssd, keys_bytes, @intCast(offset));
        _ = try preadAll(ssd, values_bytes, @intCast(offset + keys_bytes.len));

        blk.base.keys = keys;
        blk.base.values = values;
        blk.tier = .ram;
        blk.last_access_ms = milliTimestamp();
        _ = self.ram_used.fetchAdd(1, .monotonic);

        std.log.debug("Promoted block {d} from SSD to RAM", .{block_id});
    }

    /// Promote SSD→RAM with I/O outside tier_lock so the prefetcher worker
    /// does not freeze alloc/free/promote for the duration of pread.
    /// Concurrent promoters of the same block: loser discards its buffers.
    pub fn promoteFromSsd(self: *TieredKvCache, block_id: u32) !void {
        const ssd, const offset, const slot_size = blk: {
            self.lockTier();
            defer self.unlockTier();
            const b = &self.blocks[block_id];
            if (b.tier != .ssd) return;
            const file = self.ssd_file orelse return error.SsdNotConfigured;
            const off = b.ssd_offset orelse return error.SsdBlockNotSpilled;
            const size = std.math.mul(usize, @as(usize, self.block_size), self.kv_dim) catch return error.OutOfMemory;
            break :blk .{ file, off, size };
        };

        var installed = false;
        const keys = try self.allocator.alloc(f32, slot_size);
        errdefer if (!installed) self.allocator.free(keys);
        const values = try self.allocator.alloc(f32, slot_size);
        errdefer if (!installed) self.allocator.free(values);

        const keys_bytes = std.mem.sliceAsBytes(keys);
        const values_bytes = std.mem.sliceAsBytes(values);
        _ = try preadAll(ssd, keys_bytes, @intCast(offset));
        _ = try preadAll(ssd, values_bytes, @intCast(offset + keys_bytes.len));

        self.lockTier();
        defer self.unlockTier();
        var b = &self.blocks[block_id];
        // Drop if another promoter won, or the block was freed while we read
        // (ref_count==0 on free list). Installing into a free SSD block would
        // corrupt ram_used and hand live buffers to a recycled slot.
        if (b.tier != .ssd or b.base.ref_count == 0) {
            self.allocator.free(keys);
            self.allocator.free(values);
            installed = true; // disarm errdefer (already freed)
            return;
        }
        b.base.keys = keys;
        b.base.values = values;
        installed = true;
        b.tier = .ram;
        b.last_access_ms = milliTimestamp();
        _ = self.ram_used.fetchAdd(1, .monotonic);
        std.log.debug("Promoted block {d} from SSD to RAM", .{block_id});
    }

    /// Free a block and return to appropriate tier free list.
    /// Free lists are pre-allocated to tier capacity in init(), so append cannot fail.
    pub fn freeBlock(self: *TieredKvCache, block_id: u32) void {
        self.lockTier();
        defer self.unlockTier();
        self.freeBlockInner(block_id);
    }

    /// Free without acquiring tier_lock. Caller must hold tier_lock.
    fn freeBlockInner(self: *TieredKvCache, block_id: u32) void {
        std.debug.assert(block_id < self.blocks.len);
        var blk = &self.blocks[block_id];
        blk.base.ref_count = 0;
        blk.base.used = 0;
        blk.access_count = 0;

        switch (blk.tier) {
            .vram => {
                _ = self.vram_used.fetchSub(1, .monotonic);
                self.vram_free_list.appendAssumeCapacity(block_id);
            },
            .ram => {
                _ = self.ram_used.fetchSub(1, .monotonic);
                self.ram_free_list.appendAssumeCapacity(block_id);
            },
            .ssd => self.ssd_free_list.appendAssumeCapacity(block_id),
        }
    }

    /// Promote block from RAM or SSD to VRAM tier.
    ///
    /// On UMA platforms, RAM→VRAM is just a tier tag change — no data movement.
    /// Physical memory is shared between CPU and GPU, so "RAM" and "VRAM" are the
    /// same memory. Backends use zero-copy access (Metal newBufferWithBytesNoCopy,
    /// CUDA cuMemAllocManaged, Vulkan HOST_VISIBLE|DEVICE_LOCAL).
    ///
    /// SSD→VRAM promotion requires two steps:
    /// 1. Restore SSD block to RAM (promoteFromSsd)
    /// 2. Promote RAM block to VRAM (tier tag change on UMA)
    ///
    /// On discrete GPUs, this will trigger data upload in Plan 04.
    ///
    /// If VRAM is full, evicts coldest VRAM block to make room.
    ///
    /// Parameters:
    ///   - block_id: Physical block ID to promote.
    pub fn promoteToVram(self: *TieredKvCache, block_id: u32) !void {
        self.lockTier();
        defer self.unlockTier();

        var blk = &self.blocks[block_id];

        if (blk.tier == .vram) return; // Already in VRAM

        // If block is in SSD, promote to RAM first (inner: lock already held)
        if (blk.tier == .ssd) {
            try self.promoteFromSsdInner(block_id);
        }

        // Check if VRAM has space
        if (self.vram_free_list.items.len == 0) {
            // No free VRAM blocks — evict coldest
            const evicted = try self.demoteToRam();
            std.log.debug("Evicted block {d} to make room for promotion of {d}", .{ evicted, block_id });
        }

        // Transfer data from RAM → VRAM on discrete GPUs (no-op on UMA)
        if (self.transfer.isActive()) {
            const key_bytes = std.mem.sliceAsBytes(blk.base.keys);
            const val_bytes = std.mem.sliceAsBytes(blk.base.values);
            self.transfer.upload.?(self.transfer.ctx.?, key_bytes.ptr, key_bytes.ptr, key_bytes.len);
            self.transfer.upload.?(self.transfer.ctx.?, val_bytes.ptr, val_bytes.ptr, val_bytes.len);
        }

        std.debug.assert(blk.tier == .ram);
        blk.tier = .vram;
        blk.last_access_ms = milliTimestamp();
        blk.access_count +|= 1;
        _ = self.ram_used.fetchSub(1, .monotonic);
        _ = self.vram_used.fetchAdd(1, .monotonic);

        std.log.debug("Promoted block {d} to VRAM", .{block_id});
    }

    /// Batch promote multiple blocks to VRAM with a single sync point.
    /// More efficient than individual promoteToVram calls on discrete GPUs
    /// because async transfers are batched and synced once at the end.
    pub fn batchPromoteToVram(self: *TieredKvCache, block_ids: []const u32) !void {
        self.lockTier();
        defer self.unlockTier();

        for (block_ids) |bid| {
            var blk = &self.blocks[bid];
            if (blk.tier == .vram) continue;
            if (blk.tier == .ssd) try self.promoteFromSsdInner(bid);

            if (self.vram_free_list.items.len == 0) {
                _ = try self.demoteToRam();
            }

            if (self.transfer.isActive()) {
                const key_bytes = std.mem.sliceAsBytes(blk.base.keys);
                const val_bytes = std.mem.sliceAsBytes(blk.base.values);
                self.transfer.upload.?(self.transfer.ctx.?, key_bytes.ptr, key_bytes.ptr, key_bytes.len);
                self.transfer.upload.?(self.transfer.ctx.?, val_bytes.ptr, val_bytes.ptr, val_bytes.len);
            }

            blk.tier = .vram;
            blk.last_access_ms = milliTimestamp();
            blk.access_count +|= 1;
            _ = self.ram_used.fetchSub(1, .monotonic);
            _ = self.vram_used.fetchAdd(1, .monotonic);
        }

        // Single sync for all batched transfers
        if (self.transfer.isActive()) {
            if (self.transfer.sync) |sync_fn| sync_fn(self.transfer.ctx.?);
        }
    }

    /// Check if block needs promotion before use in attention.
    ///
    /// Returns true if block is allocated (ref_count > 0) and in a lower tier than VRAM.
    /// Unallocated blocks in lower tiers are on free lists and must not be promoted.
    pub fn needsPromotion(self: *const TieredKvCache, block_id: u32) bool {
        const blk = &self.blocks[block_id];
        return blk.base.ref_count > 0 and blk.tier != .vram;
    }
};

test "TieredKvCache init and deinit" {
    const allocator = std.testing.allocator;
    var cache = try TieredKvCache.init(allocator, 2, 4, 4, 2, 0, 16, null);
    defer cache.deinit();
    try std.testing.expectEqual(@as(usize, 4), cache.vram_block_count);
    try std.testing.expectEqual(@as(usize, 2), cache.ram_block_count);
    try std.testing.expectEqual(@as(usize, 0), cache.ssd_block_count);
    try std.testing.expectEqual(@as(usize, 0), cache.vram_used.load(.monotonic));
}

test "TieredKvCache allocBlock returns VRAM first" {
    const allocator = std.testing.allocator;
    var cache = try TieredKvCache.init(allocator, 1, 2, 2, 2, 0, 16, null);
    defer cache.deinit();

    const b0 = try cache.allocBlock();
    try std.testing.expectEqual(BlockTier.vram, cache.blocks[b0].tier);
    try std.testing.expectEqual(@as(usize, 1), cache.vram_used.load(.monotonic));

    const b1 = try cache.allocBlock();
    try std.testing.expectEqual(BlockTier.vram, cache.blocks[b1].tier);
    try std.testing.expectEqual(@as(usize, 2), cache.vram_used.load(.monotonic));

    // Free both
    cache.freeBlock(b0);
    cache.freeBlock(b1);
    try std.testing.expectEqual(@as(usize, 0), cache.vram_used.load(.monotonic));
}

test "TieredKvCache allocBlock falls back to RAM when VRAM full" {
    const allocator = std.testing.allocator;
    // 1 VRAM block, 1 RAM block, no SSD
    var cache = try TieredKvCache.init(allocator, 1, 2, 1, 1, 0, 16, null);
    defer cache.deinit();

    const b0 = try cache.allocBlock();
    try std.testing.expectEqual(BlockTier.vram, cache.blocks[b0].tier);

    // VRAM full — next alloc demotes b0 to RAM then falls back to RAM free list
    const b1 = try cache.allocBlock();
    try std.testing.expectEqual(BlockTier.ram, cache.blocks[b1].tier);

    cache.freeBlock(b0);
    cache.freeBlock(b1);
}

test "TieredKvCache needsPromotion" {
    const allocator = std.testing.allocator;
    // 2 VRAM + 3 RAM blocks so we always have a spare unallocated RAM block
    var cache = try TieredKvCache.init(allocator, 1, 2, 2, 3, 0, 16, null);
    defer cache.deinit();

    // VRAM block does not need promotion
    const b0 = try cache.allocBlock();
    try std.testing.expect(!cache.needsPromotion(b0));

    // Allocate remaining VRAM
    const b1 = try cache.allocBlock();
    try std.testing.expect(!cache.needsPromotion(b1));

    // Exhaust VRAM — next alloc falls back to RAM
    const b2 = try cache.allocBlock();
    try std.testing.expectEqual(BlockTier.ram, cache.blocks[b2].tier);
    // Allocated RAM block needs promotion
    try std.testing.expect(cache.needsPromotion(b2));

    // Block 2 (first RAM block, still on free list) must NOT need promotion
    // because it has ref_count = 0 (unallocated).
    try std.testing.expect(!cache.needsPromotion(2));

    cache.freeBlock(b0);
    cache.freeBlock(b1);
    cache.freeBlock(b2);
}

test "TieredKvCache freeBlock returns to correct tier" {
    const allocator = std.testing.allocator;
    var cache = try TieredKvCache.init(allocator, 1, 2, 2, 1, 0, 16, null);
    defer cache.deinit();

    const initial_vram_free = cache.vram_free_list.items.len;
    const b0 = try cache.allocBlock();
    try std.testing.expectEqual(initial_vram_free - 1, cache.vram_free_list.items.len);

    cache.freeBlock(b0);
    try std.testing.expectEqual(initial_vram_free, cache.vram_free_list.items.len);
    try std.testing.expectEqual(@as(u32, 0), cache.blocks[b0].base.ref_count);
}

test "TieredKvCache lockTier and unlockTier" {
    const allocator = std.testing.allocator;
    var cache = try TieredKvCache.init(allocator, 1, 2, 2, 1, 0, 16, null);
    defer cache.deinit();

    cache.lockTier();
    try std.testing.expectEqual(@as(u32, 1), cache.tier_lock.load(.monotonic));
    cache.unlockTier();
    try std.testing.expectEqual(@as(u32, 0), cache.tier_lock.load(.monotonic));
}

test "TieredKvCache access_count increments on promote" {
    const allocator = std.testing.allocator;
    var cache = try TieredKvCache.init(allocator, 1, 2, 2, 2, 0, 16, null);
    defer cache.deinit();

    const b0 = try cache.allocBlock();
    const b1 = try cache.allocBlock();
    _ = b0;
    _ = b1;

    // Allocate a RAM block
    const b2 = try cache.allocBlock();
    try std.testing.expectEqual(BlockTier.ram, cache.blocks[b2].tier);

    // Promote b2 to VRAM should increment access_count
    try cache.promoteToVram(b2);
    try std.testing.expectEqual(BlockTier.vram, cache.blocks[b2].tier);
    try std.testing.expectEqual(@as(u32, 1), cache.blocks[b2].access_count);
}

test "TransferCallback.uma_noop is inactive" {
    const cb = TransferCallback.uma_noop;
    try std.testing.expect(!cb.isActive());
    try std.testing.expect(cb.upload == null);
    try std.testing.expect(cb.download == null);
    try std.testing.expect(cb.sync == null);
    try std.testing.expect(cb.ctx == null);
}

test "TieredKvCache batchPromoteToVram empty list" {
    const allocator = std.testing.allocator;
    var cache = try TieredKvCache.init(allocator, 1, 2, 2, 1, 0, 16, null);
    defer cache.deinit();

    const vram_before = cache.vram_used.load(.monotonic);
    const ram_before = cache.ram_used.load(.monotonic);
    try cache.batchPromoteToVram(&[_]u32{});
    try std.testing.expectEqual(vram_before, cache.vram_used.load(.monotonic));
    try std.testing.expectEqual(ram_before, cache.ram_used.load(.monotonic));
}

test "TieredKvCache block_bytes calculation" {
    const allocator = std.testing.allocator;
    var cache = try TieredKvCache.init(allocator, 1, 4, 2, 1, 0, 16, null);
    defer cache.deinit();
    // block_bytes = kv_dim * block_size * sizeof(f32) * 2 = 4 * 16 * 4 * 2 = 512
    try std.testing.expectEqual(@as(usize, 4 * 16 * 4 * 2), cache.block_bytes);
}

test "TieredKvCache SSD round-trip preserves data" {
    const allocator = std.testing.allocator;
    // Use PID-based unique name to avoid collisions between parallel test runs.
    var path_buf: [128]u8 = undefined;
    const ssd_path = std.fmt.bufPrint(&path_buf, "test_ssd_{d}.tmp", .{std.c.getpid()}) catch unreachable;

    // 0 VRAM, 1 RAM, 1 SSD — first allocBlock goes to RAM
    var cache = try TieredKvCache.init(allocator, 1, 4, 0, 1, 1, 16, ssd_path);
    defer {
        cache.deinit();
        deleteFileByPath(ssd_path);
    }

    const b0 = try cache.allocBlock();
    try std.testing.expectEqual(BlockTier.ram, cache.blocks[b0].tier);
    try std.testing.expectEqual(@as(usize, 1), cache.ram_used.load(.monotonic));

    // Write known pattern to block keys and values
    const slot_size = @as(usize, cache.block_size) * cache.kv_dim;
    for (0..slot_size) |i| {
        cache.blocks[b0].base.keys[i] = @floatFromInt(i + 1);
        cache.blocks[b0].base.values[i] = @floatFromInt(i + 100);
    }

    // Demote to SSD: writes data to file, frees RAM backing
    try cache.demoteToSsd(b0);
    try std.testing.expectEqual(BlockTier.ssd, cache.blocks[b0].tier);
    try std.testing.expectEqual(@as(usize, 0), cache.ram_used.load(.monotonic));
    try std.testing.expectEqual(@as(usize, 0), cache.blocks[b0].base.keys.len);

    // Promote from SSD: reads data from file, allocates new RAM backing
    try cache.promoteFromSsd(b0);
    try std.testing.expectEqual(BlockTier.ram, cache.blocks[b0].tier);
    try std.testing.expectEqual(@as(usize, 1), cache.ram_used.load(.monotonic));

    // Verify data survived the round-trip
    for (0..slot_size) |i| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(i + 1)), cache.blocks[b0].base.keys[i]);
        try std.testing.expectEqual(@as(f32, @floatFromInt(i + 100)), cache.blocks[b0].base.values[i]);
    }

    cache.freeBlock(b0);
}

test "fuzz: all tiered functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const allocator = std.testing.allocator;

            // --- BlockTier enum ---
            const tier_val = smith.valueWithHash(u8, 0) % 3;
            const tier: BlockTier = @enumFromInt(tier_val);
            _ = @intFromEnum(tier);

            // --- TransferCallback ---
            const noop = TransferCallback.uma_noop;
            std.debug.assert(!noop.isActive());
            std.debug.assert(noop.upload == null);

            var active_cb = TransferCallback{
                .upload = &struct {
                    fn cb(_: *anyopaque, _: [*]u8, _: [*]const u8, _: usize) void {}
                }.cb,
                .ctx = undefined,
            };
            std.debug.assert(active_cb.isActive());
            active_cb = TransferCallback.uma_noop;

            // --- TieredBlock struct fields ---
            var tb: TieredBlock = undefined;
            tb.tier = .ram;
            tb.access_count = smith.valueWithHash(u32, 1);
            tb.last_access_ms = smith.valueWithHash(i64, 2);
            tb.ssd_offset = null;
            std.debug.assert(tb.tier == .ram);

            // --- TieredKvCache: init, lockTier, unlockTier, allocBlock, freeBlock, needsPromotion, promoteToVram, batchPromoteToVram, promoteFromSsd, deinit ---
            const vram_ct = @as(usize, smith.valueWithHash(u4, 3) % 5) + 2; // 2..6
            const ram_ct = @as(usize, smith.valueWithHash(u4, 4) % 4) + 1; // 1..4
            const kv_dim: usize = @as(usize, smith.valueWithHash(u4, 5) % 4) + 1; // 1..4
            const block_size: u16 = @as(u16, smith.valueWithHash(u4, 6) % 4) + 1; // 1..4

            var cache = TieredKvCache.init(allocator, 1, kv_dim, vram_ct, ram_ct, 0, block_size, null) catch return;
            defer cache.deinit();

            // lockTier / unlockTier
            cache.lockTier();
            std.debug.assert(cache.tier_lock.load(.monotonic) == 1);
            cache.unlockTier();
            std.debug.assert(cache.tier_lock.load(.monotonic) == 0);

            // allocBlock — fill VRAM
            const n_alloc = smith.valueWithHash(u8, 7) % @as(u8, @intCast(vram_ct + ram_ct));
            var alloc_ids: [10]u32 = undefined;
            var alloc_count: usize = 0;
            for (0..n_alloc) |_| {
                const bid = cache.allocBlock() catch break;
                std.debug.assert(bid < cache.blocks.len);
                alloc_ids[alloc_count] = bid;
                alloc_count += 1;
                if (alloc_count >= 10) break;
            }

            // needsPromotion on allocated blocks
            for (alloc_ids[0..alloc_count]) |bid| {
                const needs = cache.needsPromotion(bid);
                if (cache.blocks[bid].tier == .vram) {
                    std.debug.assert(!needs);
                }
            }

            // promoteToVram — promote any RAM blocks
            for (alloc_ids[0..alloc_count]) |bid| {
                if (cache.blocks[bid].tier == .ram) {
                    cache.promoteToVram(bid) catch {};
                }
            }

            // batchPromoteToVram — empty and non-empty
            cache.batchPromoteToVram(&[_]u32{}) catch {};
            if (alloc_count > 0) {
                cache.batchPromoteToVram(alloc_ids[0..alloc_count]) catch {};
            }

            // promoteFromSsd — no SSD tier so should be no-op or error
            if (alloc_count > 0) {
                cache.promoteFromSsd(alloc_ids[0]) catch {};
            }

            // freeBlock
            for (alloc_ids[0..alloc_count]) |bid| {
                cache.freeBlock(bid);
            }

            // Verify counters after freeing all
            std.debug.assert(cache.vram_used.load(.monotonic) == 0);
            std.debug.assert(cache.ram_used.load(.monotonic) == 0);
        }
    }.f, .{});
}
