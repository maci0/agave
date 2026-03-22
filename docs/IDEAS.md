# Future Ideas

## Model init/deinit/forward Abstraction

All 6 models (gemma3, qwen35, gpt_oss, nemotron_h, nemotron_nano, glm4) share
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
- Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2023)
- SpecInfer: Accelerating LLM Serving with Tree-based Speculative Inference (Miao et al., 2024)

## Batch Prefill

Currently prefill processes prompt tokens one at a time — each token is a separate
`forward()` call with a GEMV (matrix-vector multiply). For a 512-token prompt, that's
512 sequential GEMV calls. Batch prefill processes all prompt tokens in one pass using
GEMM (matrix-matrix multiply), which is compute-bound and much better utilizes GPU
parallelism.

### Performance impact
- GEMV (decode, 1 token): bandwidth-bound, ~5-15% compute utilization on GPU
- GEMM (prefill, N tokens): compute-bound, ~60-80% compute utilization on GPU
- Expected speedup: 10-50× for prefill phase depending on prompt length and hardware
- Directly improves TTFT (time-to-first-token)

### What changes
- **Model forward()**: Accept `[]const u32` (token slice) instead of single `u32`.
  When slice length > 1, use GEMM paths for all projections (QKV, FFN gate/up/down,
  output). When length == 1, fall back to current GEMV paths.
- **Backend**: Add `gemm()` operation alongside existing `gemv()`. CPU: blocked loop
  with SIMD. Metal: MPSMatrixMultiplication or custom kernel. CUDA: cublas-free Zig
  PTX GEMM or tiled shared-memory kernel. Vulkan: compute shader GEMM.
- **KV cache**: Append all K/V vectors for the batch at once instead of one-by-one.
- **Attention**: Prefill attention is over the full prompt — causal mask needed (each
  token attends only to previous tokens). This is where FlashAttention is most
  beneficial (large N×N attention matrix during prefill).
- **RoPE**: Apply positional embeddings to all positions in the batch simultaneously.

### Considerations
- Memory: batch prefill needs activations for all N tokens simultaneously —
  `[N × n_embd]` instead of `[n_embd]`. For 512 tokens × 4096 dim × f32 = 8 MB,
  manageable even on smaller devices.
- Chunked prefill: for very long prompts, process in chunks (e.g., 512 tokens at a
  time) to bound memory usage while still getting GEMM benefits.
- KV cache must be pre-allocated for the full prompt length before prefill starts.

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

## Platform-Specific RAM Detection

`detectFreeRam()` in main.zig currently returns hardcoded 16GB. Should use:
- **macOS**: `sysctl hw.memsize` or `host_statistics64`
- **Linux**: parse `/proc/meminfo` for `MemAvailable`

Used by `--kv-ram-budget` default (50% of detected free RAM for tiered KV cache).
Low effort, high value for correct out-of-box TieredKvCache sizing.

## Paged SDPA on GPU

Current GPU SDPA kernels only support flat (contiguous) KV cache layouts. With
PagedAttention, KV data is stored in non-contiguous blocks referenced by a block
table. GPU SDPA needs to dereference the block table to find physical KV positions,
adding an indirection layer to the attention kernel.

### What changes
- SDPA kernel accepts a block table (`[]const u32`) alongside K/V cache
- Inner loop iterates over block IDs, loads K/V from physical block addresses
- Block size alignment (16 tokens) naturally maps to SIMD/warp widths

## Missing GPU Kernels

See `docs/KERNELS.md` for the full status matrix. Key gaps:
- **CUDA/Vulkan**: sigmoidMul, siluMul, deinterleave, rmsNormMulti, DeltaNet
- **All GPU**: NVFP4 (GGUF), MXFP4 GEMV
- **CUDA/ROCm**: Additional quant formats (q4_1, q5_0, q2_k, q3_k, iq4_nl, iq4_xs)

These currently `@panic` at runtime if a model needs them on a GPU backend.
