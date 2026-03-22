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
const Allocator = std.mem.Allocator;
const manager = @import("manager.zig");
const CacheBlock = manager.CacheBlock;

/// VRAM usage threshold for triggering demotion (90% = evict at 10% free).
const vram_eviction_threshold: f32 = 0.90;
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

    /// SSD tier support (used in Plan 03): file offset for spilled blocks.
    ssd_offset: ?u64 = null,
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
    ssd_file: ?std.fs.File = null,
    /// Path to SSD sparse file.
    ssd_path: []const u8,
    /// Bytes per block (kv_dim × block_size × @sizeOf(f32) × 2).
    block_bytes: usize,

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
        const total_blocks = vram_blocks + ram_blocks + ssd_blocks;
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
        const slot_size = @as(usize, block_size) * kv_dim;
        const block_bytes = slot_size * @sizeOf(f32) * 2; // keys + values

        // Create sparse file for SSD tier if path provided
        var ssd_file: ?std.fs.File = null;
        if (ssd_path) |path| {
            ssd_file = try std.fs.cwd().createFile(path, .{ .read = true, .truncate = false });
            errdefer ssd_file.?.close();

            // Allocate sparse file (write one byte at end to reserve space)
            const total_size = ssd_blocks * block_bytes;
            if (total_size > 0) {
                try ssd_file.?.seekTo(total_size - 1);
                var write_buf: [1]u8 = .{0};
                _ = try ssd_file.?.write(&write_buf);
                try ssd_file.?.seekTo(0);
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
            blocks[i] = .{
                .base = .{
                    .keys = try allocator.alloc(f32, slot_size),
                    .values = try allocator.alloc(f32, slot_size),
                },
                .tier = .vram,
            };
            vram_free.appendAssumeCapacity(@intCast(i));
            init_count = i + 1;
        }

        // Initialize RAM tier blocks
        for (vram_blocks..(vram_blocks + ram_blocks)) |i| {
            blocks[i] = .{
                .base = .{
                    .keys = try allocator.alloc(f32, slot_size),
                    .values = try allocator.alloc(f32, slot_size),
                },
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
            f.close();
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
    /// Automatically triggers demotion when VRAM usage exceeds 90% threshold
    /// (per decision D-03: eviction at 10% free).
    ///
    /// Returns: Physical block ID.
    pub fn allocBlock(self: *TieredKvCache) !u32 {
        // Try VRAM first
        if (self.vram_free_list.items.len > 0) {
            const block_id = self.vram_free_list.pop().?;
            self.blocks[block_id].tier = .vram;
            self.blocks[block_id].base.ref_count = 1;
            self.blocks[block_id].base.used = 0;
            return block_id;
        }

        // VRAM exhausted — check if we can evict (per D-03: trigger at 90% usage)
        const vram_used = self.vramUsedBlocks();
        const vram_total = self.vram_block_count;
        const vram_usage = @as(f32, @floatFromInt(vram_used)) / @as(f32, @floatFromInt(vram_total));

        if (vram_usage > vram_eviction_threshold) {
            // Demote coldest VRAM block to RAM
            const evicted = try self.demoteToRam();
            std.log.debug("Demoted block {d} from VRAM to RAM (usage: {d:.1}%)", .{ evicted, vram_usage * 100.0 });
            // Retry allocation
            return try self.allocBlock();
        }

        // Fallback to RAM tier
        if (self.ram_free_list.items.len > 0) {
            const block_id = self.ram_free_list.pop().?;
            self.blocks[block_id].tier = .ram;
            self.blocks[block_id].base.ref_count = 1;
            self.blocks[block_id].base.used = 0;
            return block_id;
        }

        // Fallback to SSD tier (promote from SSD to RAM before use)
        if (self.ssd_free_list.items.len > 0) {
            const block_id = self.ssd_free_list.pop().?;
            // Promote from SSD to RAM before use
            try self.promoteFromSsd(block_id);
            self.blocks[block_id].base.ref_count = 1;
            self.blocks[block_id].base.used = 0;
            return block_id;
        }

        // Out of memory across all tiers
        return error.OutOfKvMemory;
    }

    /// Count number of VRAM blocks currently in use.
    fn vramUsedBlocks(self: *const TieredKvCache) usize {
        var used: usize = 0;
        for (self.blocks[0..self.vram_block_count]) |*blk| {
            if (blk.base.ref_count > 0 and blk.tier == .vram) {
                used += 1;
            }
        }
        return used;
    }

    /// Demote coldest VRAM block to RAM tier.
    ///
    /// Uses frequency×cost eviction from Plan 01 (access_count × compute_cost).
    /// Shared prefixes (ref_count > 1) have 100× higher cost to prevent thrashing.
    /// Last block of sequence evicted first (minimize shared prefix eviction).
    ///
    /// If RAM is full, demotes coldest RAM block to SSD first.
    ///
    /// Returns: Block ID of demoted block.
    fn demoteToRam(self: *TieredKvCache) !u32 {
        // Check if RAM is full — if so, demote RAM → SSD first
        const ram_used = self.ramUsedBlocks();
        if (ram_used >= self.ram_block_count and self.ssd_block_count > 0) {
            const ram_victim = try self.selectColdestRamBlock();
            try self.demoteToSsd(ram_victim);
            std.log.debug("RAM full — demoted block {d} to SSD before VRAM→RAM demotion", .{ram_victim});
        }

        var min_score: f32 = std.math.floatMax(f32);
        var victim_id: u32 = 0;

        for (self.blocks[0..self.vram_block_count], 0..) |*blk, id| {
            if (blk.tier != .vram) continue; // Only demote from VRAM
            if (blk.base.ref_count == 0) continue; // Skip free blocks

            // Frequency × cost metric (shared prefixes prioritized)
            const cost: f32 = if (blk.base.ref_count > 1) shared_prefix_cost else 1.0;
            const score = @as(f32, @floatFromInt(blk.access_count)) * cost;

            if (score < min_score) {
                min_score = score;
                victim_id = @intCast(id);
            }
        }

        // Change tier tag (data stays in place for now — zero-copy in Plan 04)
        self.blocks[victim_id].tier = .ram;

        // Block remains allocated (ref_count > 0), so not on free list
        // Tier tag is sufficient for tracking location

        return victim_id;
    }

    /// Count number of RAM blocks currently in use.
    fn ramUsedBlocks(self: *const TieredKvCache) usize {
        var used: usize = 0;
        const ram_start = self.vram_block_count;
        const ram_end = ram_start + self.ram_block_count;
        for (self.blocks[ram_start..ram_end]) |*blk| {
            if (blk.base.ref_count > 0 and blk.tier == .ram) {
                used += 1;
            }
        }
        return used;
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

        if (self.ssd_file == null) return error.SsdNotConfigured;

        // Calculate fixed offset for this block
        const offset = @as(u64, block_id) * @as(u64, self.block_bytes);

        // Write keys + values to SSD
        try self.ssd_file.?.seekTo(offset);

        const keys_bytes = std.mem.sliceAsBytes(blk.base.keys);
        const values_bytes = std.mem.sliceAsBytes(blk.base.values);
        try self.ssd_file.?.writeAll(keys_bytes);
        try self.ssd_file.?.writeAll(values_bytes);

        // Free RAM backing
        self.allocator.free(blk.base.keys);
        self.allocator.free(blk.base.values);
        blk.base.keys = &[_]f32{};
        blk.base.values = &[_]f32{};

        // Update tier metadata
        blk.tier = .ssd;
        blk.ssd_offset = offset;

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
    pub fn promoteFromSsd(self: *TieredKvCache, block_id: u32) !void {
        var blk = &self.blocks[block_id];
        std.debug.assert(blk.tier == .ssd);

        if (self.ssd_file == null) return error.SsdNotConfigured;

        const slot_size = @as(usize, self.block_size) * self.kv_dim;

        // Allocate RAM backing
        blk.base.keys = try self.allocator.alloc(f32, slot_size);
        errdefer self.allocator.free(blk.base.keys);
        blk.base.values = try self.allocator.alloc(f32, slot_size);
        errdefer self.allocator.free(blk.base.values);

        // Read from SSD
        const offset = blk.ssd_offset.?;
        try self.ssd_file.?.seekTo(offset);

        const keys_bytes = std.mem.sliceAsBytes(blk.base.keys);
        const values_bytes = std.mem.sliceAsBytes(blk.base.values);
        _ = try self.ssd_file.?.readAll(keys_bytes);
        _ = try self.ssd_file.?.readAll(values_bytes);

        // Update tier metadata
        blk.tier = .ram;
        blk.last_access_ms = std.time.milliTimestamp();

        std.log.debug("Promoted block {d} from SSD to RAM", .{block_id});
    }

    /// Free a block and return to appropriate tier free list.
    pub fn freeBlock(self: *TieredKvCache, block_id: u32) !void {
        var blk = &self.blocks[block_id];
        blk.base.ref_count = 0;
        blk.base.used = 0;
        blk.access_count = 0;

        switch (blk.tier) {
            .vram => try self.vram_free_list.append(self.allocator, block_id),
            .ram => try self.ram_free_list.append(self.allocator, block_id),
            .ssd => try self.ssd_free_list.append(self.allocator, block_id),
        }
    }

    /// Promote block from RAM or SSD to VRAM tier.
    ///
    /// On UMA platforms (per D-09), RAM→VRAM is just a tier tag change — no data movement.
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
        var blk = &self.blocks[block_id];

        if (blk.tier == .vram) return; // Already in VRAM

        // If block is in SSD, promote to RAM first
        if (blk.tier == .ssd) {
            try self.promoteFromSsd(block_id);
            // Block is now in RAM, continue to VRAM promotion below
        }

        // Check if VRAM has space
        if (self.vram_free_list.items.len == 0) {
            // No free VRAM blocks — evict coldest
            const evicted = try self.demoteToRam();
            std.log.debug("Evicted block {d} to make room for promotion of {d}", .{ evicted, block_id });
        }

        // Change tier tag (data stays in place on UMA, uploaded in Plan 04 on discrete)
        blk.tier = .vram;
        blk.last_access_ms = std.time.milliTimestamp();
        blk.access_count += 1;

        std.log.debug("Promoted block {d} to VRAM", .{block_id});
    }

    /// Check if block needs promotion before use in attention.
    ///
    /// Returns true if block is in lower tier than VRAM.
    pub fn needsPromotion(self: *const TieredKvCache, block_id: u32) bool {
        return self.blocks[block_id].tier != .vram;
    }
};
