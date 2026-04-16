# Future Ideas

> **Recently Implemented** (2026-04): Vision/multimodal (SigLIP-2 + SigLIP + Qwen VL),
> TurboQuant+ (asymmetric KV, boundary V, sparse V), BF16 Metal GEMM, split-attention
> (APEX), Gemma 4 E2B/E4B support, thread-parallel vision attention. See BENCHMARKS.md.

## Model init/deinit/forward Abstraction

All 7 models (gemma3, gemma4, qwen35, gpt_oss, nemotron_h, nemotron_nano, glm4) share
near-identical skeletons for init(), deinit(), forward(), resetCache(), and cancel().
Extracting shared logic could save ~30% LoC per model.

### What's duplicated
- **init()**: Read metadata via getArchU32/getMetaU32, allocate working buffers with
  errdefer, allocate KV cache, return struct
- **deinit()**: Free buffers via inline-for tuple, free KV cache
- **forward()**: KV cache overflow check → embedding lookup → layer loop with
  cancellation check → final norm → logits → argmax → increment kv_seq_len
- **resetCache()/cancel()**: Delegate to model_mod helpers (identical in all models)
- **attention()**: pre-norm → QKV projections → optional per-head norms → RoPE → SDPA →
  output projection → post-norm → residual
- **feedForward()**: pre-norm → gate/up projections → activation → down projection →
  post-norm → residual

### Why it's deferred
Each model is self-contained and easy to modify independently. A generic abstraction
would add comptime complexity (parameterized by activation type, rope config, layer type
dispatch, MoE vs dense FFN, per-head norms, sliding window patterns, etc.) that could
hurt readability. The current ~200-line-per-model duplication is acceptable because:
1. Models rarely change once working
2. Each model has unique quirks (Gemma embedding scaling, GPT-OSS attention sinks,
   Qwen3.5 DeltaNet hybrid, GLM4 MLA compression, etc.)
3. Self-contained files are easier to debug

### If pursued
A reasonable approach would be a `ModelBuilder` in model.zig with:
- `allocBuffers(names_and_sizes)` → allocates + errdefer
- `forwardLoop(embed_fn, layer_fn, final_fn)` → handles KV check + cancellation
- `genericAttention(config)` → parameterized by rope_dim, has_bias, has_per_head_norms
- `genericFfn(activation)` → parameterized by comptime activation enum

Estimated savings: ~600 lines across 7 models. Estimated effort: 2-3 days.

## Speculative Decoding

Current decode is autoregressive — one forward pass per token through the full model.
Speculative decoding uses a small draft model to predict N tokens cheaply, then verifies
all N in a single forward pass of the large target model. Accepted tokens are free;
rejected tokens cost only the wasted draft computation. Typical speedup: 2-3× with no
quality loss (output distribution is mathematically identical to the target model).

### How it works
1. Draft model generates N candidate tokens autoregressively (fast, small model)
2. Target model runs one forward pass on all N candidates (parallel verification)
3. Compare draft and target distributions at each position. Accept token i if
   `r < P_target(t_i) / P_draft(t_i)` (rejection sampling). First rejection at
   position k → accept tokens 0..k-1, resample position k from adjusted distribution.
4. Repeat from the last accepted position

### Architecture
- Load two models simultaneously: target (main) and draft (small variant or same
  architecture with fewer layers / smaller hidden dim)
- Both models need independent KV caches that stay in sync (rollback draft cache on
  rejection, advance target cache on acceptance)
- Draft model can be: same family smaller variant (e.g., 1B drafts for 8B target),
  same model with early-exit after N layers, or a separate lightweight model
- `--draft-model <path>` CLI flag to specify draft model path
- `--draft-tokens <N>` to control speculation depth (default: 5)

### Considerations
- Memory: two models loaded simultaneously — draft model should be small enough that
  the combined VRAM fits. Quantized drafts help.
- Acceptance rate depends on draft/target agreement — higher for similar model families
- Not beneficial for very short generations (overhead of running two models)
- Requires batch prefill for efficient verification pass

### References
- [Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192)
- [SpecInfer: Accelerating LLM Serving with Tree-based Speculative Inference (Miao et al., 2024)](https://arxiv.org/abs/2305.09781)

## Structured Output / Grammar-Constrained Decoding

Constrain model generation to produce valid output matching a schema (JSON, regex, CFG).
Works by masking logits before sampling — at each token position, only tokens that keep
the output on a valid path are allowed. Output quality doesn't degrade because the model
already wants to produce valid output; the constraint just prevents rare failures.

### Supported constraint types
- **JSON Schema**: Generate valid JSON matching a provided schema (required fields,
  types, enums, nested objects). Highest user demand — enables reliable tool calling
  and structured extraction.
- **Regex**: Constrain output to match a regular expression. Useful for dates, IDs,
  formatted strings.
- **Context-Free Grammar (CFG)**: Full grammar support via PDA (pushdown automaton).
  JSON and regex are special cases of CFG.
- **Choice / Enum**: Simple case — restrict output to one of N literal strings.

### Algorithm
1. Pre-compile the grammar/schema into a state machine (DFA for regex, PDA for CFG,
   specialized FSM for JSON schema)
2. Before each token generation, query the FSM: given current state, which tokens
   can lead to a valid continuation? Build a bitmask over the vocabulary.
3. Apply the mask to logits (set disallowed tokens to -inf) before softmax/sampling.
4. After sampling, advance the FSM state with the chosen token's text.

### Considerations
- Vocabulary-aware FSM: tokens are multi-character, so the FSM must handle partial
  matches (a token may partially match the next valid production). Libraries like
  Outlines/lm-format-enforcer solve this with token-level precomputation.
- Performance: bitmask lookup is O(1) per token, but precomputation can be expensive
  for large grammars × large vocabularies. Lazy state expansion helps.
- Server API: `response_format: { "type": "json_schema", "json_schema": {...} }` in
  the OpenAI-compatible chat completion endpoint

### References
- Outlines: Structured Text Generation (Willard & Louf, 2023)
- Efficient Guided Generation for Large Language Models (2024)

## Direct-to-VRAM Model Loading

> **Partially implemented:** Tiered KV cache (VRAM + RAM + SSD) is available via
> `--kv-tiers`. Direct-to-VRAM *weight* loading from NVMe remains future work.

Model weights are currently loaded via `mmap` into system RAM, then uploaded to GPU
memory per-tensor on first use (buffer cache pattern). For large models this means the
full weight file transits: NVMe → CPU RAM → PCIe/fabric → VRAM. Direct storage APIs
can bypass CPU RAM entirely, reading weights from NVMe straight into GPU memory.

### Current loading path
```
NVMe SSD ──mmap──→ CPU RAM ──buffer upload──→ VRAM
           ~7 GB/s            ~32 GB/s (PCIe 4)
```
Bottleneck is the double-copy and CPU involvement. A 27B model (~15 GB quantized)
takes several seconds to fully populate GPU caches.

### Direct storage path
```
NVMe SSD ──GPUDirect/Metal IO──→ VRAM
           ~7 GB/s (per drive, stackable)
```
Single copy, zero CPU involvement. With multiple NVMe drives, bandwidth scales
linearly (4 drives = 28 GB/s). Load time for 15 GB model: ~0.5s vs ~2-4s.

### Platform APIs
- **NVIDIA GPUDirect Storage** (`cuFile`): `cuFileRead()` transfers directly from
  file descriptor to GPU device pointer.
- **Apple Metal**: UMA means "VRAM" is system RAM. Existing zero-copy mmap +
  `newBufferWithBytesNoCopy` is already optimal. No improvement needed.
- **Vulkan**: `VK_KHR_external_memory_fd` + Linux `io_uring` for async reads.
- **AMD ROCm**: Future GDS-equivalent via `hsa_amd_ipc_memory_attach`.

### Considerations
- Alignment requirements: GPUDirect Storage requires 4KB-aligned file offsets.
- UMA platforms (Apple Silicon, NVIDIA GB10): already optimal via zero-copy mmap.
- Compressed formats: decompression must happen on GPU or fall back to CPU-staged.

## Paged SDPA on GPU

> **Partially implemented:** The block allocator (`kvcache/block_allocator.zig`) and
> `PagedKvCache` (`kvcache/manager.zig`) manage paged blocks on the CPU side. Only
> the GPU SDPA kernels need updating to dereference the block table.

Current GPU SDPA kernels only support flat (contiguous) KV cache layouts. With
PagedAttention, KV data is stored in non-contiguous blocks referenced by a block
table. GPU SDPA needs to dereference the block table to find physical KV positions,
adding an indirection layer to the attention kernel.

### What changes
- SDPA kernel accepts a block table (`[]const u32`) alongside K/V cache
- Inner loop iterates over block IDs, loads K/V from physical block addresses
- Block size alignment (16 tokens) naturally maps to SIMD/warp widths

## TriAttention — Frequency-Domain KV Cache Eviction

> **Phase 1+2 implemented** (`--kv-eviction norm` / `--kv-eviction tri`). Phase 1 uses K-norm
> scoring; Phase 2 adds trigonometric frequency-domain scoring with `TriCalibration` stats.
> Periodic compression every 128 tokens. Calibration data generator is future work.

KV cache eviction based on trigonometric frequency analysis of pre-RoPE Q/K vectors.
Instead of scoring tokens by expensive attention computation, it uses statistical
properties of Q/K cluster centers to determine which KV entries are important.
Unimportant entries are pruned from the cache entirely.

### Key results ([Mao et al., 2025](https://github.com/WeianMao/triattention))

- **10.7× KV memory reduction** with accuracy parity on reasoning benchmarks
- **2.5× throughput boost** on AIME25
- Works on Qwen3, DeepSeek-R1, GPT-OSS (models we already support)
- No retraining — inference-only, uses precomputed Q/K frequency statistics

### How it stacks with our existing KV optimizations

| Layer | Technique | Reduction | Status |
|-------|-----------|-----------|--------|
| 1. Bits per entry | TurboQuant turbo4 | 3.8× | ✅ Implemented |
| 2. V-only compression | Asymmetric K=q8_0/V=turbo4 | +quality | ✅ Implemented |
| 3. Skip negligible V | Sparse V dequant (softmax < 1e-6) | +22% decode | ✅ Implemented |
| 4. Evict old entries | TriAttention (norm + frequency) | 10.7× | ✅ Phase 1+2 implemented |
| **Combined** | **1 + 2 + 3 + 4** | **~40×** | |

### Core insight

Pre-RoPE Q and K vectors in reasoning models cluster around fixed frequency centers.
Token importance can be scored cheaply by measuring distance from these centers (via
vector norm and cosine similarity), without computing full attention. This is O(n) per
token vs O(n²) for attention-based importance scoring.

### Implementation plan for agave

**Phase 1 — Heuristic eviction (no precomputed stats):**
1. Add `KvEvictionPolicy` enum to `kv_quant.zig`: `none`, `norm_based`, `tri_frequency`
2. In `PagedKvCache`, track per-block importance scores (running average of K norms)
3. When cache is full, evict the block with lowest importance score
4. Expose as `--kv-eviction norm` CLI flag
5. Integrate with sliding window — evict outside the window first

**Phase 2 — Full TriAttention with precomputed statistics:**
1. Load `.pt` frequency center files per model
2. Score tokens against frequency centers using cosine similarity
3. Per-head and per-layer-per-head pruning strategies
4. In-place KV compaction (shift remaining entries to fill gaps)
5. Expose as `--kv-eviction tri --kv-stats <path>`

**Phase 3 — Dynamic budget:**
1. Auto-tune KV budget based on available memory
2. Adaptive eviction threshold (tighter budget → more aggressive pruning)
3. Preserve recent tokens unconditionally (attention sink pattern)

### Considerations
- Precomputed stats add deployment friction (one `.pt` file per model)
- Phase 1 (norm-based) is simpler and works without stats — good starting point
- Most impactful for long-context (32K+) reasoning chains
- Must preserve the sliding window invariant — never evict within the active window
- Block-level eviction (not token-level) aligns with our paged KV cache design

### References
- [TriAttention: KV Cache Compression via Trigonometric Frequency-Domain Analysis (Mao et al., 2025)](https://github.com/WeianMao/triattention)
- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (Liu et al., 2024)](https://arxiv.org/abs/2402.02750)
- [Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression (Liu et al., 2023)](https://arxiv.org/abs/2305.17118)

## Missing GPU Kernels

See `docs/KERNELS.md` for the full status matrix. Key gaps:
- **CUDA/Vulkan**: sigmoidMul, siluMul, deinterleave, rmsNormMulti, DeltaNet
- **All GPU**: NVFP4 (GGUF)
- **Vulkan/ROCm**: MXFP4 GEMV (available in Metal and CUDA)
- **CUDA/ROCm**: Additional quant formats (q4_1, q5_0, q2_k, q3_k, iq4_nl, iq4_xs)

These currently `@panic` at runtime if a model needs them on a GPU backend.
