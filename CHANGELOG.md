# Changelog

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
- tiered KV cache (`--kv-tiers vram+ram`) crash fixed: `isMultiBlock` now guards against `paged_cache.block_size == 0` (all 9 model architectures)
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
