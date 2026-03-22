---
phase: 05-integration-wiring
plan: 02
subsystem: kv-cache
tags: [tiered-cache, block-allocator, cli-flags, model-init]

# Dependency graph
requires:
  - phase: 03-memory-optimization
    provides: TieredKvCache, TieredBlock, BlockTier types
  - phase: 05-integration-wiring (plan 01)
    provides: getBlockTable() vtable method, scheduler wiring
provides:
  - TieredBlockAllocator wrapping TieredKvCache for model block allocation
  - All 6 models accept optional TieredKvCache via init() parameter
  - CLI flags (--kv-tiers, --kv-ram-budget, --kv-ssd-path, --kv-ssd-budget) wired to TieredKvCache.init()
  - Server RequestManager receives actual tiered cache pointer
affects: [kv-cache, serving, model-init]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Dual-path KV cache: models check tiered_block_allocator vs block_allocator at runtime"
    - "TieredBlockAllocator mirrors BlockAllocator API for seamless swap"
    - "CLI-driven cache configuration with model metadata for block size calculation"

key-files:
  created: []
  modified:
    - src/kvcache/block_allocator.zig
    - src/models/gemma3.zig
    - src/models/qwen35.zig
    - src/models/gpt_oss.zig
    - src/models/nemotron_h.zig
    - src/models/nemotron_nano.zig
    - src/models/glm4.zig
    - src/main.zig
    - src/server.zig

key-decisions:
  - "TieredBlockAllocator as separate struct (not generic BlockAllocator) to avoid breaking existing API"
  - "Model metadata fallback chain: llama.* keys then SafeTensors keys for cross-format support"
  - "Tiered cache owned by initAndRun scope, outlives model via defer ordering"

patterns-established:
  - "Tiered/paged dual path: if (self.tiered_block_allocator) |*ta| ... else self.block_allocator..."
  - "Block access through tiered: tc.blocks[id].base.keys/values"

requirements-completed: [TIER-06, SERV-01]

# Metrics
duration: 10min
completed: 2026-03-22
---

# Phase 05 Plan 02: Model TieredKvCache Wiring + CLI Flag Threading Summary

**All 6 models accept optional TieredKvCache with CLI-driven VRAM+RAM+SSD tier configuration threaded through to server scheduler**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-22T00:36:26Z
- **Completed:** 2026-03-22T00:46:49Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- TieredBlockAllocator added to block_allocator.zig, providing same API as BlockAllocator but backed by TieredKvCache
- All 6 model architectures (Gemma3, Qwen3.5, GPT-OSS, Nemotron-H, Nemotron-Nano, GLM-4) updated with optional tiered_cache parameter in init(), dual-path block allocation in forward()/resetCache()/deinit(), and tiered-aware getLayerKvView()
- CLI flags --kv-tiers, --kv-ram-budget, --kv-ssd-path, --kv-ssd-budget parsed in main.zig and used to compute block counts from model metadata (n_layers, kv_dim) before initializing TieredKvCache
- Server's RequestManager.init() now receives actual tiered_cache pointer instead of null placeholder

## Task Commits

Each task was committed atomically:

1. **Task 1: Add TieredKvCache support to all 6 model init functions** - `a732781` (feat)
2. **Task 2: Wire CLI flags to TieredKvCache init in main.zig and update server model init call** - `c253f56` (feat)

## Files Created/Modified
- `src/kvcache/block_allocator.zig` - Added TieredBlockAllocator struct with allocateSeqTable/appendBlock/freeSeqTable
- `src/models/gemma3.zig` - Added tiered_cache/tiered_block_allocator fields, dual-path init/forward/reset/deinit
- `src/models/qwen35.zig` - Same tiered cache integration pattern
- `src/models/gpt_oss.zig` - Same tiered cache integration pattern
- `src/models/nemotron_h.zig` - Same tiered cache integration pattern
- `src/models/nemotron_nano.zig` - Same tiered cache integration pattern
- `src/models/glm4.zig` - Same tiered cache integration pattern
- `src/main.zig` - Added kv-tier CLI fields, TieredKvCache.init from metadata, tiered_ptr to model and server
- `src/server.zig` - Added tiered_cache parameter to run(), passed to RequestManager.init()

## Decisions Made
- TieredBlockAllocator implemented as a separate struct rather than making BlockAllocator generic, because the underlying cache types (PagedKvCache vs TieredKvCache) have fundamentally different block structures (CacheBlock vs TieredBlock.base)
- Model metadata keys use a fallback chain (llama.* GGUF keys then SafeTensors config keys) to support both GGUF and SafeTensors model formats when computing tier block counts
- Tiered cache storage is stack-allocated in initAndRun() with defer cleanup, ensuring it outlives the model (model defer runs first, then tiered cache defer)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- TIER-06 integration gap is closed: models can use VRAM+RAM+SSD tiered KV cache via CLI flags
- SERV-01 server integration is complete: scheduler receives tiered cache for cache-aware scheduling
- Pre-existing TODO in detectFreeRam() (returns hardcoded 16GB) means --kv-ram-budget default (50% of "free RAM") currently defaults to 8GB; future work should add platform-specific RAM detection

## Self-Check: PASSED

All files exist, all commits verified.

---
*Phase: 05-integration-wiring*
*Completed: 2026-03-22*
