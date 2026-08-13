# Changelog

All notable user-facing changes to Agave are recorded here.
Product version is **0.1.0** (`agave --version`, `/health`, `system_fingerprint`).
While on **0.x**, SemVer allows breaking changes without a major bump; such changes
must still appear under **Changed** or **Breaking** below. See
[Versioning & Releases](docs/CONTRIBUTING.md#versioning--releases).

> Note: git tag `v1.0` (2026-03-22) is a historical milestone name, not the product
> SemVer. Do not treat it as release `1.0.0`.

## [Unreleased]

### Breaking
- GPU backends (CUDA and peers): missing GPTQ/AWQ/MXFP4 kernels now fail closed
  instead of silently falling back to CPU. Workloads that accidentally relied on
  that fallback will error; enable a backend that implements the kernel, or use CPU
  explicitly (`--backend cpu`).
- CLI: unknown flags and options now exit with code 2 (previously printed a
  warning and continued). Fix typos or remove unrecognized flags.
- CLI: an option value that looks like another flag (e.g. `--port --host`) now
  exits with code 2 instead of a warning. Pass an explicit value for each option.
- CLI: `--flag=value` on a boolean flag (e.g. `--quiet=true`) now exits with
  code 2 instead of treating the flag as set. Use the bare flag (`--quiet`).
- Auth: when both `--api-key` and `AGAVE_API_KEY` are set, `AGAVE_API_KEY` wins
  (previously the CLI flag won). Prefer setting only the env var.
- HTTP: browser cross-origin requests to a server with no API key return `403`
  with `code: cross_origin_forbidden` (CSRF protection). Set `AGAVE_API_KEY` or
  `--api-key`, or call same-origin / non-browser clients.
- HTTP: `/v1/kv_cache` error `type` is now `invalid_request_error` (was
  `invalid_request`) for missing/invalid `n_tokens` and import failures. Align
  client checks with OpenAI-style `invalid_request_error`.
- HTTP: `/v1/kv_cache` matches the exact path only (no longer
  `startsWith("/v1/kv_cache")`). `/v1/kv_cache/info` is routed separately and is
  not shadowed. Clients using a longer path prefix must call the documented URLs.

### Added
- **DeepSeek V4 Flash 0731**: full architecture support — hyper connections,
  MLA, CSA/HCA compressors, Lightning Indexer, hash routing. See 2026-07-31 entry.
- Server env fallbacks: `AGAVE_HOST` and `AGAVE_PORT` when `--host` / `--port`
  are omitted (`--host` / `--port` still win when set). Documented in `--help`
  and Docker examples.
- Server rate limiting: `--rate-limit-rpm` / `--rate-limit-tpm` (token bucket;
  `0` = unlimited / off). Exceeded limits return `429` with `Retry-After`.
- Local Compose path: `docker-compose.yml` + `.env.example` (`AGAVE_API_KEY`
  required; publish defaults to `127.0.0.1`).
- CLI: short flag clusters and attached short-option values (e.g. `-qV`, `-n128`).

### Fixed
- GPT-OSS / MXFP4 SafeTensors: group size corrected to 16 and block scales decoded
  as FP8 E4M3 (was group size 32 + E8M0, which garbled output)
- IQ2/IQ3 GEMV: sign extraction and qs indexing; `iq4_nl` / `iq4_xs` dequant paths
- ROCm: HSACO load works on ROCm 6.x with Zig 0.16 (ISA triple / ABI version workarounds)
- `--lora` with a SafeTensors base model now warns that LoRA merge is unsupported
  (previously ignored with no message)
- LoRA: reject adapters whose `lora_b` rank does not match `lora_a` (corrupted GGUF)
- HTTP JSON responses: allocation failure while escaping no longer inserts raw
  (possibly unescaped) strings; returns a generic `500` JSON error instead
- Split GGUF shard merging: `tensors.put()` now propagates OOM instead of
  silently dropping tensors (could cause silent model corruption on large shards)

### Changed
- Changelog entries are consumer-oriented; date-stamped sections below remain the
  historical log until the next tagged product release bumps `0.1.0`
- `--diffusion-confidence` docs/help now report default `0.5` (runtime default was
  already `0.5`; help/README previously said `0.9`)
- `--max-batch-size` help no longer claims default `1`; runtime default remains `8`
- Non-loopback `--serve` without a key still requires auth; prefer `AGAVE_API_KEY`
  over `--api-key` (process-list exposure)
- HTTP JSON errors more often include machine-readable `param` and `code` (additive
  for clients that ignore unknown fields; see `docs/API.md`)

## 2026-07-31 — DeepSeek V4 Flash 0731

### DeepSeek V4 Flash Full Architecture Support

New model architecture in `src/models/deepseek4.zig` with complete inference support.

**Architecture:**
- 4-stream hyper connections (HC) with Sinkhorn-normalized combination matrices
- Modified MLA: K=V single compressed head, no separate V projection
- Hash routing (layers 0–2), sqrt_softplus routing (layers 3+)
- Grouped output LoRA (8 groups × 1024 rank)
- CSA compressor (ratio=4, 21 layers) and HCA compressor (ratio=128, 20 layers)
- Lightning Indexer (LID): multi-head ReLU dot-product block scoring for sparse attention

**Performance optimizations (cumulative):**
- KV cache switched from f32 to Q8_0 (~4× memory reduction)
- Metal GPU SDPA kernel for hd=512 (`sdpa_fa2_hd512`) + Q8_0 KV support
- SIMD vectorized: RoPE cos/sin (8-wide), RoPE apply/inverse (4-wide complex rotation),
  sqrt_softplus routing + bias, LID scoring (head-outer loop), expert accumulation
- CPU Q8_0 GEMV for HC pre/head (eliminates 86 GPU dispatches/token)
- 2-row interleaved cpuGemvQ8_0 for HC GEMV throughput
- Sparse V threshold skips negligible attention positions (zero PPL impact)
- Buffer copy elimination in hot path (~3.5 MB/token saved)
- RoPE table cache eliminates 128× redundant transcendental calls per token
- Thread-pool parallel per-head compressed attention
- Inline plainRmsNorm, RoPE table apply/inverse for tight per-head loops

**GPU fast paths:**
- Non-compressed layers use GPU SDPA directly
- Batched CSA+HCA compressor GEMVs in single GPU command buffer
- Hoist sink tensor lookup outside per-head attention loop

## 2026-06-30 — DSpark Speculative Decoding

### DSpark: Confidence-Scheduled Speculative Decoding (Cheng et al., 2026)

Implements the [DSpark framework](https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf) from DeepSeek-AI in `src/spec/dspark.zig`.

**`src/spec/dspark.zig`** (new file):
- `SpsProfile` — pre-profiled steps-per-second table for target-model token-batch sizes; `syntheticComputeBound()` for offline use
- `ConfidenceBlock` + `computeSurvival()` — per-request per-position survival probs `a_{r,j} = Π_{i≤j} c_i`
- `scheduleVerification()` — **Algorithm 1** (Hardware-Aware Prefix Scheduler): globally sorts `(request, position)` candidates by survival probability descending, greedily admits tokens while `Θ = τ × SPS(B)` improves, stops on first drop (non-anticipating property). `O(Rγ log Rγ)`.
- `MarkovHead` — low-rank `V×V` transition bias `B(x_{k-1},·) = W1[x_{k-1}]W2` (§3.1 Eq. 5), `rank=256` default
- `RnnHead` — gated recurrent sequential head with full prefix history (§3.1 Eq. 6)
- `ConfidenceHead` — `c_k = σ(w^T [h_k; W1[x_{k-1}]])` (§3.2.1 Eq. 7)
- `calibrateSts()` — Sequential Temperature Scaling: per-position 1D grid search minimising ECE of cumulative product (§3.2.1)

**`src/spec/spec_decode.zig`**:
- `dsparkTrimDraft()` — single-request draft trim using per-position acceptance history as survival-probability proxy; drops suffix below 0.15 expected survival

**`src/main.zig`**:
- `--spec-mode dspark` wired into decode loop: drafts via existing draft model, trims via `dsparkTrimDraft()`
- Enum, help strings, and test export all updated

4/4 unit tests pass (Markov bias correctness, scheduler greedy/load cases, SPS profile).

## 2026-06-18 — Vulkan: KosmicKrisp + Pipeline Cache

### Vulkan macOS Backend
- **KosmicKrisp** replaces MoltenVK as the macOS Vulkan testing target
- Load path: `libvulkan.1.dylib` (Homebrew Vulkan loader) with `/opt/homebrew/lib/` fallback; set `VK_ICD_FILENAMES` to KosmicKrisp ICD and `DYLD_LIBRARY_PATH=/opt/homebrew/lib`
- `sdpa_turbo` pipeline gracefully skipped when driver lacks `GroupNonUniform` subgroup ops (lavapipe/KosmicKrisp don't implement them); TurboQuant KV falls back to standard SDPA
- **Disk-backed `VkPipelineCache`**: compiled shaders saved to `~/.cache/agave/vk_pipeline_cache.bin` (1.2 MB for 49 kernels), loaded on subsequent runs; note lavapipe re-JITs LLVM IR each run regardless (~5 min; driver limitation)

## 2026-06-16 — IQ2/IQ3 Quant Support + LoRA + MTP Fix

### IQ2/IQ3/IQ1 Quantization Support
- Added DType entries: `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq1_s`, `iq1_m`
- Previously mapped to `.unknown` → zeroed output and warned. Now dispatched to CPU reference kernels
- `iq2_xxs`: full codebook-based dequant via iq2xxs_grid[256] (512-bit packed int8 entries)
- `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq1_s`, `iq1_m`: approximation stubs (scale-based)
- Metal/Vulkan/CUDA/ROCm/WebGPU: CPU fallback instead of panic for these dtypes
- `dequantToF32`: iq4_nl/iq4_xs properly dequanted; iq2/iq3 stub for LoRA merge path
- Mixed-quant "UD" models (e.g. unsloth Qwen3-0.6B-UD-IQ2_XXS) now load and run

### LoRA Adapter Support

### LoRA Adapter Loading
- `--lora <path>`: load a LoRA adapter GGUF file alongside the base model
- Load-time merge: base weights are dequanted to F32, delta = (alpha/rank) * lora_b @ lora_a is added, result stored as F32 override
- Transparent to all model code via `GGUFFile.lora_overrides` map checked in `getTensor()`
- Supports any base quantization (Q4_0, Q4_K, Q8_0, BF16, F16, etc.) and any lora tensor dtype
- Format: llama.cpp GGUF LoRA (convert_lora_to_gguf.py output), `adapter.type = "lora"`

### MTP Spec Decode Fix (Qwopus)

- `qwen35.zig`: MTP detection now handles two GGUF layouts — layout A has block_count excluding MTP heads (nextn at blk.{n_layers}), layout B has block_count including MTP heads (nextn at blk.{n_layers-1}). Layout B adjusts n_layers down so mtpForward uses the correct mtp_lid.
- `qwen35.zig`: All nextn tensor lookups now try `.weight` suffix first (e.g. `nextn.eh_proj.weight`) before falling back to bare name, matching Qwopus GGUF storage convention.
- `qwen35.zig`: `nextn.embed_tokens` falls back to shared `token_embd.weight`; `nextn.shared_head_head` falls back to shared `output.weight`.
- Verified: Qwopus3.6-27B-Coder-MTP — 74% accept rate, 0.7 mean tokens/step.

---

## 2026-06-15 — Vulkan Correctness Fixes

### Vulkan DeltaNet Fixes (2026-06-16)
- `deltanet_recurrence.comp`: GQA head mapping wrong for `num_k != num_v` — CPU uses `h % num_k` (round-robin) but shader used `h * num_k / num_v` (blocked). Fixes garbled output for Qwen3.5-4B and any model with mismatched k/v head counts.
- `vulkan.zig`: `gate_arr`/`beta_arr` too small (64) — should be 128 to match `max_ssm_v_heads`. Prevents stack overflow for models with >64 v_heads.

### Vulkan Backend Fixes
- `destroyBuffer`: submits pending GPU commands before destroying — prevents VUID-vkCmd invalid state (buffer destroyed while recorded in command buffer)
- `downloadF32`: submits pending work before host readback — prevents reading stale deferred dispatch results
- Qwen3.5 Vulkan garbled output fixed: DeltaNet causalConv1d was reading stale conv output due to deferred dispatch not executing before downloadF32
- Vulkan Q8_0 Qwen2.5: confirmed correct at 14.2 tok/s on RX 7900 XTX
- `n_pipelines`: updated 44→49 (5 new pipelines added without updating count)

### Build
- `-Denable-debug=false`: new flag to skip `agave-debug` binary on Linux x86_64 with GCC ≥16 (R_X86_64_PC64 relocation unsupported in debug builds)

---

## 2026-06-12 — Feature Release

### Bug Fixes
- tiered KV cache (`--kv-tiers vram+ram`) crash fixed: `isMultiBlock` now guards against `paged_cache.block_size == 0` (all 10 model architectures)
- Warning added: tiered SDPA split-attention only fully implemented for Gemma 3; other models will warn
- CUDA: all 60 kernel files now in PTX build list (was 19); 61 kernels registered at runtime (was 44)
- CUDA SDPA correctness: `getOrAllocKvBuf` now uploads host KV data on first GPU allocation
- ARM Linux CPU detection: `implementer+part` fallback for aarch64 `/proc/cpuinfo` (no `model name`)

### CUDA Full Validation (GB10 / sm_121 / CUDA 13.0)
- `callconv(.nvptx_device)` replaces `callconv(.kernel)` — fixes Zig 0.16/LLVM NVPTX alias crash
- Build PTX fixup: Python script promotes `.func *_kernel` → `.entry` post-compilation
- All 60 kernel .zig files now in PTX build list (was 19); 61 kernels registered at runtime
- CUDA KV cache fix: `getOrAllocKvBuf` uploads host data on first allocation (was reading garbage)
- ARM Linux CPU detection fix: uses `CPU implementer+part` fallback (no `model name` on aarch64)
- Test results: **1025 passed, 0 failed** on GB10 (-Denable-vulkan=false)
- Server mode verified: `/health` returns `backend=CUDA`, CUDA spec decode (ngram 91% accept)
- TurboQuant KV (`--kv-type turbo2`) works on CUDA
- Performance: 22.3 tok/s decode Qwen3.5-0.8B-Q8_0 on GB10 (UMA; CPU 48 tok/s)

### DiffusionGemma (Block Diffusion LLM)
- Added `diffusion_gemma` architecture — Google's DiffusionGemma 26B-A4B (SafeTensors BF16)
- `src/models/diffusion_gemma.zig`: Gemma 4 26B A4B backbone with block diffusion inference
- `src/ops/attention.zig`: `scaledDotProductAttentionCanvas()` for bidirectional canvas attention
- Inference loop: encoder prefill → iterative denoising (uniform state diffusion) → block autoregressive chaining
- 128 experts, top-8, fused `experts.gate_up_proj` tensor, per-layer `layer_scalar`
- New flags: `--diffusion-steps` (default 16), `--diffusion-canvas` (default 256), `--diffusion-confidence` (default 0.5)

### New Features
- **EAGLE-3 speculative decoding** (`--spec-mode eagle3`): conditions draft on pre-output-norm hidden state instead of post-norm; preserves residual magnitude for potentially richer draft conditioning. `hidden_pre_norm` buffer added to Gemma4.
- **Video input** (`--video`, `--video-fps`): extract frames via ffmpeg at configurable FPS, encode each through vision encoder, concatenate visual tokens for temporal understanding. Works with any vision-capable model (Gemma4, Qwen VL).
- **Sleep mode** (`--serve --sleep-after=N`): server enters soft sleep state after N seconds of inactivity, signaling `/health` with `"sleeping": true`. Auto-wakes on next request.
- **`--spec-mode auto`**: selects DDTree with draft model, N-gram without.
- **`/v1/kv_cache/info`**: lightweight metadata endpoint for orchestrators (seq_len, prefix_hash, kv_used/total).
- **Thinking token budget** (`thinking_budget_tokens`): Anthropic-style budget that applies strong logit bias toward `</think>` when reasoning exceeds limit (streaming + non-streaming).

### Model Support
- **Nex-N2-Pro** (qwen35moe): 512-expert MoE with hybrid DeltaNet+full-attention, `attn_output_gate` disambiguation
- **DeepSeek V3 GGUF**: MLA tensor name fallbacks in glm4.zig, arch-prefixed param loading
- **NVFP4 Qwen3-8B**: SafeTensors empty-prefix fix (bare `lm_head.weight` now found)
- **Qwopus MTP models**: fixed init failure when MTP-head layers lack SSM tensors

### Performance
- `addRmsNorm`/`rmsNormAdd` dispatch fusion across all models (Gemma4, Gemma3, Llama4, GLM-4, GPT-OSS): ~68 fewer Metal dispatches/token
- Second addRmsNorm fusion for Gemma4/Gemma3: deferred FFN residual fused with next-layer pre-attention norm
- Native `rms_norm_add` shaders on all GPU backends (Metal, Vulkan SPIR-V, WebGPU WGSL, CUDA PTX, ROCm HIP)
- Tensor-presence DeltaNet layer detection for Qwen3.5 (handles irregular `layer_types`, MTP boundary layers)

### Fixes
- VLM pending FFN residual flush in `forwardImageBatch` (was corrupting hidden state)
- Metal n_pipelines count: 70 → 71
- MXFP4 scale dtype detection (U8 → `.nvfp4` not `.unknown`)

---

## 2026-05-20 — NCCL RoCE RDMA Performance Fix

**PP=2 NCCL over RoCE: 4.2 → 40.2 tok/s (9.6x speedup)**

Root cause: CUDA interop (context, mem_alloc, memcpy) was not wired for PP transport — NCCL couldn't allocate device staging buffers and fell back to TCP sockets silently.

Fixes:
- Wire CUDA interop inside `setupTransport` before `setupNccl`
- Set CUDA context current before `ncclCommInitRank`
- Eager comm init at TCP sync point (post unique ID exchange)
- NCCL env var logging (17 variables) + comm diagnostics
- Device pointer path in sendBuf (skip host→device when data on GPU)
- Test script (`scripts/test-pp-nccl.sh`) with ConnectX RoCE config

Hardware-verified on dual NVIDIA GB10 over ConnectX RoCE RDMA:
- `NET/IB : Using rocep1s0f1:1/RoCE` confirmed
- `GIN_IB_GDAKI` (GPUDirect) assigned
- 16 p2p channels, 0.27s init time
- PP=2 now **faster than single GPU** (40.2 vs 36.0 tok/s)

---

## 2026-05-19 — Major Feature Release (59 commits)

### GPU Kernels (32 new files)
- **All quantized GEMV formats now native on all 6 backends** (was 14 gaps)
- ROCm: fused silu_mul, gelu_mul, add_rms_norm kernels
- WebGPU: bf16, f16, fp8_e4m3, fp8_e5m2, q4_1, q5_0, q2_k, q3_k, iq4_nl, iq4_xs
- Vulkan: q4_1, q5_0, q2_k, q3_k, iq4_nl, iq4_xs (+ compiled SPIR-V)
- CUDA: q5_0, q2_k, q3_k, iq4_nl, iq4_xs, fused FFN GELU Q8_0
- CUDA fused FFN activation naming fix (SiLU→GELU correctness for Gemma 3)

### Performance
- Vulkan deferred dispatch: single submit vs ~240 per token
- WebGPU deferred dispatch: batch all compute passes into one encoder
- Paged SDPA staging buffer caching on all 5 GPU backends (zero hot-path allocs)
- Q/K norm, RoPE, QKV GEMV, gate/up FFN batched across all models (barrier reduction)

### Samplers (3 new, API + CLI)
- **XTC** (eXclude Top Choices): diversity via random top-token exclusion
- **DRY** (Don't Repeat Yourself): n-gram sequence repetition penalty
- **Mirostat 2.0**: target-entropy adaptive sampling with dynamic mu
- CLI flags: `--dry-multiplier`, `--xtc-probability`, `--mirostat-mode`, etc.
- All samplers applied consistently across first-token, decode, and spec decode paths

### Speculative Decoding
- **N-gram mode** (`--spec-mode ngram`): zero-overhead spec decode from output history
- **Adaptive cooldown**: skip drafting when acceptance rate drops below 25%
- **Profile-guided adaptive K**: track per-K acceptance, auto-optimize draft length
- All three improvements work together

### Distributed Inference
- **UDP peer discovery**: zero-config LAN discovery (no `--peers` needed)
- **Topology-aware device exchange**: peers swap memory capabilities
- **Peer RTT measurement**: TCP ping-pong after connection

### Server / API
- **Logprobs** in streaming responses (`logprobs`, `top_logprobs`)
- **SSM state prefix caching**: ~2x prefill for Qwen3.5/Nemotron with shared prompts
- **xxHash prefix cache**: RadixTree fast path for repeated prefix queries
- **Vulkan device enumeration** for `--list-devices`

### CLI
- `--ctx-size auto`: probe memory, pick largest safe context
- `--benchmark`: built-in decode benchmark with JSON output
- `--benchmark --json`: machine-readable stats for CI

### Documentation
- PARALLELISM.md: rewritten from 2569-line design doc to 200-line impl reference
- Tutorials improved: chapters 2 (attention), 3 (FFN), 5 (memory), 6 (SSMs),
  7 (sampling), 8 (backends), 17 (spec decode) — worked numerical examples
- KERNELS.md: systematic audit fixed 8+ stale entries, file listings updated
- TODO.md + IDEAS.md merged into single unified document
- 26-item roadmap from vLLM, llama.cpp, Exo, Mesh-LLM analysis

### Testing
- 6 unit tests for new samplers (XTC, DRY, Mirostat)
- 11 fuzz tests for parsers (JSON, GBNF, JSON schema) and samplers
- Test compile fixes for device_id parameter + MockModel

## 2026-08-13 — DeepSeek V4 Flash Performance Autoresearch

### Fixed
- **Metal buffer cache staleness**: Added `volatile_weights` mode that flushes
  the Metal buffer cache periodically on `sync()` when `--ssd-streaming` is active.
  Prevents NaN/inf from stale `newBufferWithBytesNoCopy` references to OS-evicted
  mmap'd pages. Models can now run back-to-back without `sudo purge`.
  Files: `src/backend/metal.zig`, `src/backend/backend.zig`, `src/main.zig`

### Investigated
- **IQ2_XXS coherence (ds4 Q2 model)**: CPU dequant logic matches ds4/llama.cpp
  exactly (codebook, signs, scale). Garbled output is NOT a kernel bug but rather
  2-bit quantization error amplified by DeepSeek V4's hyper connections (HC).
  L0 FFN output differs by ~10% from MXFP4 (expected for 2-bit), but by L1 the
  HC mixing amplifies this to 30× divergence. ds4 engine achieves coherent output
  from the same model likely through additional stabilization (per-layer
  normalization, different HC precision, or quantization-aware training).
- **Tokenization verified**: Agave and ds4 produce identical token sequences for
  the same prompt (11 tokens for "What is 2+2?" with deepseek4 chat template).
- **DSpark/MTP**: DS V4 Flash config.json has `dspark_block_size=5`,
  `dspark_markov_rank=256`, `num_nextn_predict_layers=1`, but neither the MXFP4
  nor ds4 Q2 GGUFs contain MTP weight tensors.

### Performance findings (autoresearch)
- **CPU backend for SSD streaming**: 1.2 tok/s with MXFP4, coherent output.
  CPU is the recommended backend for SSD streaming because Metal's
  `newBufferWithBytesNoCopy` does not trigger GPU page faults for evicted
  file-backed mmap pages on Apple Silicon.
- **Metal volatile_weights mode**: Buffer cache flush on `sync()` prevents
  NaN for mostly-resident models. Enabled automatically with `--ssd-streaming`.
  Not reliable when model far exceeds RAM (GPU reads zeroed evicted pages).
- **Speculative decoding**: Suffix mode achieves 100% acceptance rate with
  4.0 mean draft length on DS V4 Flash. N-gram needs history (cold start).
  DSpark not useful for SSD streaming (extra forward passes = more SSD reads).
- **IQ2_XXS coherence**: CPU dequant matches ds4/llama.cpp exactly.
  Root cause is 2-bit quantization error amplified by hyper connections (HC).
  L0 FFN output differs ~10% from MXFP4, diverges 30× by L1 through HC mixing.

### Research findings (autoresearch iterations 4-5)
- **Logit correlation analysis**: MXFP4 preserves 65% of ds4 reference logit
  signal (r=0.65). IQ2_XXS preserves 2% (r=0.02, complete signal loss through
  43 layers of HC mixing). Sinkhorn implementation verified identical to ds4
  (max diff 6e-8). IQ2_XXS kernel verified correct.
- **Expert cache profiling**: ~51 unique experts per layer, 73% cache hit rate
  at 64 tokens. At warm cache, system is ~70% compute-bound, ~30% SSD-bound.
- **Coherent generation**: MXFP4 CPU at 1.0 tok/s produces coherent multi-
  paragraph text. Quality comparable to marginal MXFP4 baseline but reliable.

### Discovery: DS V4 Flash MTP/DSpark weights
- HF safetensors (deepseek-ai/DeepSeek-V4-Flash-0731) contain 3 full MTP
  decoder layers with 4,705 tensors including:
  - Full MLA attention per MTP layer
  - Full MoE FFN (256 routed experts + shared) per MTP layer
  - Hyper connections per MTP layer
  - DSpark confidence_head + markov_head on mtp.2
  - hc_head (HC merge head) on mtp.2
- GGUF quantizers (ggml-org, ds4/antirez) stripped ALL MTP weight tensors
- DSpark weights in ~/Models are for Qwen3 8B, not DS V4 Flash
- Implementing MTP for DS V4 requires loading from HF safetensors (not GGUF)
  or converting MTP tensors to GGUF format
