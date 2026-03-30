# Chapter 5: Memory and Caching

During **autoregressive generation** (generating text one token at a time, where each new token depends on all previous tokens), each new token needs to attend to all previous tokens. Recomputing K and V for every previous position would waste enormous compute. The **KV cache** stores them.

## The KV Cache

```
Token 1: compute K₁, V₁, store in cache
Token 2: compute K₂, V₂, store in cache, attend to [K₁,K₂], [V₁,V₂]
Token 3: compute K₃, V₃, store in cache, attend to [K₁,K₂,K₃], [V₁,V₂,V₃]
```

The cache grows linearly with sequence length. For a model with 30 layers, 5 KV heads, 128-dim heads, and 4096 max tokens:

```
30 × 5 × 128 × 4096 × 2 (K+V) × 4 bytes = 600 MB
```

Quantizing the KV cache (e.g., to f16 or fp8) halves or quarters this cost with minimal quality loss.

## PagedAttention

Allocating a **contiguous** (single continuous memory region) KV cache per **sequence** (a single request or conversation — the tokens for one prompt and its generated response) wastes memory when sequences have different lengths. PagedAttention breaks the cache into fixed-size **blocks** (default 16 positions):

```
physical_block = block_table[position / block_size]
offset = position % block_size
K[position] = blocks[physical_block].keys[offset * kv_dim ...]
```

Benefits:

- **No internal fragmentation** (wasted space within allocated regions) — blocks allocated on demand
- **Memory sharing** — **reference counting** (tracking how many sequences use each block) enables **copy-on-write** (sharing read-only data, duplicating only when modified) between requests
- **Continuous batching** — sequences can grow/shrink independently

Each `CacheBlock` tracks: `keys`, `values`, `used` count, `ref_count` (for sharing), `access_count` (for eviction).

## RadixAttention

RadixAttention builds a **radix tree** (also called a **prefix trie** — a tree data structure where shared prefixes are stored only once) over token sequences to automatically detect and share common prefixes. If two requests share the same system prompt, the KV cache for that prefix is computed once and reused.

```
Request A: "You are helpful. What is 2+2?"     → compute KV for "You are helpful." once
Request B: "You are helpful. Tell me a joke."   → reuse KV, only compute " Tell me a joke."
```

**RadixAttention Tree Visualization:**

```
                           root
                             │
                      "You are helpful."
                     [blocks 0,1,2] (shared)
                             │
                    ┌────────┴────────┐
                    │                 │
              " What is"        " Tell me a"
              [block 3]          [block 3']
                    │                 │
                 " 2+2?"            " joke."
                [block 4]          [block 4']
                    │                 │
              Answer: "4"       Answer: "Why..."
              [block 5]          [block 5']

Request A path: root → "You are helpful." → " What is" → " 2+2?" → "4"
Request B path: root → "You are helpful." → " Tell me a" → " joke." → "Why..."
                         └─────┬─────┘
                        shared prefix (blocks 0,1,2)
                        computed once, reused by both requests

Eviction: Shared blocks (ref_count=2) have 100× cost → preserved longer
Benefit: 3 shared blocks = 48 positions × 2 requests = 96 positions saved
```

Key operations (all at the scheduler layer, never in the token generation hot path):

- **Insert**: Cache a completed sequence's block IDs
- **Lookup**: Find the longest cached prefix for a new prompt
- **Eviction**: **LRU** (Least Recently Used — remove the oldest unused data first) based on access **timestamps** (recorded times when each block was last used); shared prefixes (ref_count > 1) get 100× **eviction cost** (penalty score that makes them harder to remove) to preserve reuse

RadixAttention is the preferred strategy for production serving.

## Chunked Prefill and Bulk KV Population

During **batched prefill**, all prompt tokens are processed through each layer together using GEMM instead of GEMV. The KV cache is populated in bulk — each layer's `sdpaPrefill` kernel appends all N key/value vectors at once.

**Chunked prefill** limits memory usage by splitting long prompts into fixed-size chunks (default 512 tokens). Each chunk is one batched pass through all layers:

```
prefill([2048 tokens], chunk_size=512):
  chunk 0: tokens[0..512]    → GEMM + causal FA2 (prev_len=0)
  chunk 1: tokens[512..1024] → GEMM + causal FA2 (prev_len=512)
  chunk 2: tokens[1024..1536]→ GEMM + causal FA2 (prev_len=1024)
  chunk 3: tokens[1536..2048]→ GEMM + causal FA2 (prev_len=1536)
```

Each chunk's attention is **causal within the chunk AND attends to all previous chunks' KV data** in the cache. The `prev_len` parameter tells the attention kernel how many cached positions precede this chunk.

Prefill buffers are allocated once at model init, sized to `chunk_size × dim`. They are separate from the single-token decode buffers to avoid any regression on the decode path.

---

**In the code:** `src/kvcache/manager.zig` (KvCache, PagedKvCache, RadixTree), `src/kvcache/block_allocator.zig` (block allocation), `src/kvcache/tiered.zig` (VRAM + RAM + SSD tiers), `src/ops/kv_quant.zig` (KV cache quantization), `src/backend/kernels/cpu/sdpa_prefill.zig` (CPU prefill attention), `src/backend/kernels/metal/sdpa.metal` (`sdpa_prefill_fa2` — GPU prefill FA2)

**Next:** [Chapter 6: State Space Models →](06-state-space-models.md)
