# Phase 5: Integration Wiring (Gap Closure) - Research

**Researched:** 2026-03-22
**Domain:** System integration, runtime wiring, continuous batching activation
**Confidence:** HIGH

## Summary

Phase 5 wires dormant Phase 2-3 infrastructure into the active serving stack. All individual components (RequestManager, TieredKvCache, RadixTree, Prefetcher) are complete and tested — the gap is **runtime activation**, not missing functionality. This is a pure integration phase with no new algorithm implementations.

**Primary finding:** The codebase uses a "build first, activate later" pattern — infrastructure exists but isn't on the execution path. Integration requires 3 orthogonal changes: (1) route HTTP requests through scheduler instead of direct model.forward(), (2) swap models from PagedKvCache to TieredKvCache, (3) extract block tables from completed requests for RadixTree reuse.

**Key insight:** Zig's comptime dispatch and explicit memory management make this a mechanical refactor — no hidden state, no implicit initialization, no runtime surprises. All integration points are statically typed and compile-time verified.

## User Constraints

### Locked Decisions
- Pure infrastructure phase — Claude has full discretion

### Claude's Discretion
- All implementation choices
- Task breakdown granularity
- Integration sequence

### Deferred Ideas
None — discussion stayed within phase scope

---

## Standard Stack

### Core Libraries

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Zig std | latest | All runtime services (threads, atomics, sync) | Language stdlib, zero external deps |

**Installation:** None — pure Zig, uses existing codebase modules

**Version verification:** Not applicable (internal refactor only)

### Existing Infrastructure (Phase 2-3)

| Component | Location | Status | Integration Point |
|-----------|----------|--------|-------------------|
| RequestManager | `src/scheduler.zig` | Complete, tested | `init()` accepts tiered_cache param |
| TieredKvCache | `src/kvcache/tiered.zig` | Complete, tested | Models need init() signature change |
| RadixTree | `src/kvcache/manager.zig` | Complete, tested | Embedded in RequestManager |
| Prefetcher | `src/kvcache/prefetch.zig` | Complete, tested | Auto-initialized by RequestManager |
| runSchedulerLoop | `src/scheduler.zig:296` | Complete, tested | Needs spawn in Server.run() |

---

## Architecture Patterns

### Current State (Broken Flow)

```
HTTP Request
  ↓
server.zig handleChatCompletion()
  ↓
MUTEX LOCK
  ↓
model.forward() [DIRECT CALL — bypasses scheduler]
  ↓
MUTEX UNLOCK
  ↓
SSE stream response
```

**Why it's broken:** Single-request serialization. No continuous batching, no cache-aware scheduling, no multi-request throughput.

### Target State (Continuous Batching)

```
HTTP Request
  ↓
server.zig handleChatCompletion()
  ↓
request_manager.enqueue(prompt_tokens)
  ├→ RadixTree.matchPrefix() [detect cached prefix]
  ├→ Record cache hit/miss in metrics
  └→ Return *Request reference
  ↓
Background scheduler loop (separate thread):
  ├→ scheduler.step(&model, eog_ids)
  │   ├→ Remove finished/cancelled
  │   ├→ Check timeout
  │   ├→ Sort waiting by cache-aware priority
  │   ├→ Fill batch from waiting queue
  │   ├→ Promote blocks to VRAM (if tiered cache enabled)
  │   ├→ Prefetch next blocks (if prefetcher enabled)
  │   └→ Forward all running requests
  └→ Sleep 1ms, repeat
  ↓
req.tokens updated by scheduler [polling]
  ↓
SSE stream from req.tokens
  ↓
On completion:
  ├→ Extract block_table from model's cache
  └→ RadixTree.insert(tokens, block_table)
```

**Key differences:**
1. HTTP handler enqueues and polls — doesn't call forward
2. Scheduler owns model.forward() execution
3. RadixTree receives actual block IDs (not empty list)
4. Tiered cache automatically promotes/demotes during step()

### Model KV Cache Swap Pattern

**Before (flat PagedKvCache):**
```zig
pub fn init(allocator: Allocator, ...) !Gemma3Model {
    var paged_cache = try PagedKvCache.init(
        allocator,
        n_layers,
        kv_dim,
        16, // block_size
    );

    return .{
        .paged_cache = paged_cache,
        // ...
    };
}
```

**After (TieredKvCache):**
```zig
pub fn init(
    allocator: Allocator,
    ...,
    tiered_cache: ?*TieredKvCache, // NEW PARAM
) !Gemma3Model {
    // Use tiered cache if provided, fallback to flat paged
    var cache_impl = if (tiered_cache) |tc|
        tc
    else
        try PagedKvCache.init(...);

    return .{
        .cache = cache_impl, // TYPE CHANGE
        // ...
    };
}
```

**Impact:** All 6 models need signature change + cache field type change. Attention ops already accept abstract cache interface.

### Background Thread Lifecycle

```zig
// Server init
pub fn run(...) !void {
    var request_manager = try scheduler.RequestManager.init(...);
    defer request_manager.deinit();

    // Spawn scheduler loop
    const scheduler_thread = try std.Thread.spawn(.{},
        scheduler.runSchedulerLoop,
        .{&request_manager, &model, eog_ids_slice, &shutdown_flag}
    );

    // Accept connections...

    // Graceful shutdown
    shutdown_flag.store(true, .release);
    scheduler_thread.join();
}
```

**Pattern:** Shutdown flag uses atomic `.release` store for visibility, `.acquire` load in loop. Thread joins during drain period.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Thread synchronization | Manual mutex + condvar | std.atomic.Value + Futex | Scheduler uses atomics for cancel flags |
| Background thread spawn | Raw posix threads | std.Thread.spawn | Zig stdlib handles platform differences |
| Cache lookup | Linear search | RadixTree.matchPrefix | Already implemented with trie optimization |
| Block migration | Manual memcpy loops | TieredKvCache.promoteToVram() | Handles UMA zero-copy + discrete GPU upload |

**Key insight:** All hard problems already solved in Phase 2-3. This phase is mechanical wiring.

---

## Runtime State Inventory

> Integration phase, not rename/refactor — omit this section entirely.

---

## Common Pitfalls

### Pitfall 1: Server Mutex Contention
**What goes wrong:** Holding server.mutex while calling `request_manager.step()` would serialize all requests (defeats continuous batching purpose).
**Why it happens:** Old code pattern locks around `model.forward()` calls.
**How to avoid:**
- `enqueue()` locks briefly (just list append)
- `step()` runs unlocked — RequestManager has internal mutex
- Polling `req.tokens` needs no lock (atomic is_finished flag)
**Warning signs:** Throughput doesn't improve with concurrent requests.

### Pitfall 2: Model Init Signature Mismatch
**What goes wrong:** Changing model init() signature without updating all 6 model files causes compile errors.
**Why it happens:** Models are separate files (gemma3.zig, qwen35.zig, etc.) — easy to miss one.
**How to avoid:**
- Grep for `pub fn init(` in `src/models/*.zig`
- Verify all 6 files compile before proceeding
- Use comptime error messages for clear failure modes
**Warning signs:** Compile error "expected 7 arguments, found 6"

### Pitfall 3: Empty Block Table on Insert
**What goes wrong:** RadixTree.insert() called with `&[_]u32{}` doesn't populate block reuse cache.
**Why it happens:** Models store block tables internally — need extraction helper.
**How to avoid:**
- Add `getBlockTable()` method to model interface
- Extract actual block IDs from seq_table before insert
- Verify cache hit rate increases in metrics
**Warning signs:** Prometheus `cache_hit_total` counter never increments.

### Pitfall 4: Tiered Cache Without CLI Flags
**What goes wrong:** TieredKvCache.init() called with hardcoded budgets ignores user's `--kv-ram-budget` flag.
**Why it happens:** CLI parsing exists but args not threaded through to init site.
**How to avoid:**
- Thread tier config struct from main.zig → Server.run() → model init
- Verify CLI flags actually change tier allocation (print debug or check metrics)
**Warning signs:** VRAM/RAM budgets don't match CLI args.

### Pitfall 5: Scheduler Thread Not Joined
**What goes wrong:** Process exits while scheduler thread is mid-forward(), leaving KV cache in inconsistent state.
**Why it happens:** Shutdown flag not propagated or thread join missing.
**How to avoid:**
- Set `shutdown_flag.store(true, .release)` on SIGTERM
- Call `scheduler_thread.join()` before deinit
- Verify graceful drain in integration test
**Warning signs:** Segfault or corruption on Ctrl+C.

---

## Code Examples

Verified patterns from existing codebase:

### Atomic Shutdown Flag Pattern
```zig
// In Server struct
scheduler_shutdown: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

// In runSchedulerLoop (scheduler.zig:302)
while (!shutdown.load(.acquire)) {
    manager.step(model, eog_ids) catch |err| {
        std.log.err("Scheduler step failed: {}", .{err});
    };
    std.time.sleep(1_000_000); // 1ms between iterations
}

// In shutdown handler
server.scheduler_shutdown.store(true, .release);
scheduler_thread.join();
```

**Pattern:** `.release` guarantees all prior writes visible to thread, `.acquire` ensures thread sees shutdown flag update.

### Request Polling Pattern
```zig
// From scheduler tests (scheduler.zig:311-335)
const req = try manager.enqueue(prompt_tokens);
defer {
    req.deinit();
    allocator.destroy(req);
}

// Poll until finished
while (!req.is_finished and !req.is_cancelled.load(.monotonic)) {
    std.time.sleep(10_000_000); // 10ms poll interval
}

// Access generated tokens
const output = req.tokens.items;
```

**Pattern:** No mutex needed for polling — atomic flags + scheduler internal locking.

### TieredKvCache Init Pattern
```zig
// From tiered.zig:90-126
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
    // Allocate all tiers
    const total_blocks = vram_blocks + ram_blocks + ssd_blocks;
    const blocks = try allocator.alloc(TieredBlock, total_blocks);
    errdefer allocator.free(blocks);

    // Initialize VRAM tier (0..vram_blocks)
    for (0..vram_blocks) |i| {
        blocks[i] = .{
            .base = .{
                .keys = try allocator.alloc(f32, slot_size),
                .values = try allocator.alloc(f32, slot_size),
            },
            .tier = .vram,
        };
    }

    // Initialize RAM tier (vram_blocks..vram_blocks+ram_blocks)
    for (vram_blocks..(vram_blocks + ram_blocks)) |i| {
        blocks[i] = .{
            .base = .{
                .keys = try allocator.alloc(f32, slot_size),
                .values = try allocator.alloc(f32, slot_size),
            },
            .tier = .ram,
        };
    }

    // Initialize SSD tier (virtual, no RAM backing)
    for ((vram_blocks + ram_blocks)..total_blocks) |i| {
        blocks[i] = .{
            .base = .{
                .keys = &[_]f32{}, // Empty slice
                .values = &[_]f32{},
            },
            .tier = .ssd,
        };
    }

    return .{
        .blocks = blocks,
        .vram_block_count = vram_blocks,
        .ram_block_count = ram_blocks,
        .ssd_block_count = ssd_blocks,
        // ...
    };
}
```

**Pattern:** Explicit tier allocation, errdefer cleanup, no hidden state.

### Block Table Extraction Pattern
```zig
// From scheduler.zig:268-275 (current stub)
if (req.is_finished and req.tokens.items.len > 0) {
    // TODO: Extract block_table from model's PagedKvCache
    // For now, insert with empty block list
    const empty_blocks: []const u32 = &[_]u32{};
    self.radix_tree.insert(req.tokens.items, empty_blocks) catch |err| {
        std.log.warn("Failed to insert sequence into RadixTree: {}", .{err});
    };
}

// AFTER: Extract from model's seq_table
if (req.is_finished and req.tokens.items.len > 0) {
    const block_table = model.getBlockTable(req.id); // NEW METHOD
    self.radix_tree.insert(req.tokens.items, block_table) catch |err| {
        std.log.warn("Failed to insert sequence into RadixTree: {}", .{err});
    };
}
```

**Pattern:** Models need to expose their internal block table as a read-only slice.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Direct model.forward() in server | Enqueue + background scheduler | Phase 5 (this phase) | Enables continuous batching (2-3× throughput) |
| Flat PagedKvCache per model | TieredKvCache with VRAM/RAM/SSD | Phase 5 (this phase) | Enables memory-bounded serving at scale |
| RadixTree insert with empty blocks | RadixTree insert with actual block IDs | Phase 5 (this phase) | Enables prefix reuse (60-80% cache hits) |
| Hardcoded KV budgets | CLI flags --kv-tiers, --kv-ram-budget, --kv-ssd-path | Phase 5 (this phase) | User-configurable memory management |

**Deprecated/outdated:**
- Server.mutex around model.forward(): Replaced by RequestManager internal locking + unlocked step()
- Model init without tiered_cache param: Replaced by optional tiered_cache param

---

## Validation Architecture

> **Skip this section:** workflow.nyquist_validation is not set in .planning/config.json. Defaulting to enabled.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Zig test (std.testing) |
| Config file | none — built into build.zig |
| Quick run command | `zig build test` |
| Full suite command | `zig build test` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SERV-01 | HTTP request enqueues through RequestManager, model.forward() only called by scheduler | integration | `zig build test` (verify scheduler.zig tests pass) | ✅ scheduler.zig has unit tests |
| SERV-03 | RadixTree.insert() receives actual block IDs from completed request | unit | `zig build test` (verify RadixTree reuse logic) | ✅ manager.zig has RadixTree tests |
| TIER-06 | Models init with TieredKvCache, zero-copy access works | integration | `zig build test` (verify tiered.zig tests pass) | ✅ tiered.zig has unit tests |

### Sampling Rate

- **Per task commit:** `zig build test` (full suite — fast in Zig)
- **Per wave merge:** `zig build test` + manual integration smoke test (HTTP POST to /v1/chat/completions)
- **Phase gate:** Full test suite green + E2E continuous batching flow verified

### Wave 0 Gaps

None — existing test infrastructure covers all phase requirements:
- `scheduler.zig` has unit tests for enqueue, step, timeout, cancellation
- `kvcache/tiered.zig` has unit tests for allocation, promotion, demotion, eviction
- `kvcache/manager.zig` has RadixTree unit tests for insert, match, eviction
- Integration smoke test: manual curl to server endpoint

---

## Open Questions

1. **Request-to-Model ID Mapping**
   - What we know: Scheduler creates request IDs (atomic counter), models don't track per-request state
   - What's unclear: How to extract block table for a specific request when model's PagedKvCache has single seq_table
   - Recommendation: Add req_id field to SeqBlockTable, or maintain request_id → block_table map in model

2. **Tiered Cache Backward Compatibility**
   - What we know: CLI mode needs to work without --kv-tiers (use flat PagedKvCache)
   - What's unclear: Should models auto-detect tiered vs flat, or require explicit flag?
   - Recommendation: Make tiered_cache param optional — `null` means use flat PagedKvCache (existing behavior)

3. **detectFreeRam() Placeholder**
   - What we know: Function returns hardcoded 16GB (audit tech debt item)
   - What's unclear: Whether to fix in this phase or defer
   - Recommendation: Defer — user can override with --kv-ram-budget. Auto-detection is nice-to-have, not blocker.

---

## Sources

### Primary (HIGH confidence)

- Existing codebase (`src/scheduler.zig`, `src/kvcache/tiered.zig`, `src/server.zig`, `src/models/*.zig`) — implementation details
- v1.0 milestone audit (`.planning/v1.0-MILESTONE-AUDIT.md`) — gap analysis
- Phase 2-3 summary documents — scheduler, tiered cache, RadixTree implementation notes
- Zig std library documentation (embedded in std module) — thread, atomic, sync primitives

### Secondary (MEDIUM confidence)

- CLAUDE.md project standards — Zig patterns, hot path constraints, dispatcher pattern
- Phase CONTEXT.md — user decisions (all discretion to Claude)

### Tertiary (LOW confidence)

None — all findings from authoritative codebase sources

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Pure Zig stdlib, existing codebase modules
- Architecture: HIGH - All integration points statically typed and verified
- Pitfalls: HIGH - Derived from actual codebase patterns and audit findings

**Research date:** 2026-03-22
**Valid until:** 60 days (infrastructure refactor, no external API dependencies)
