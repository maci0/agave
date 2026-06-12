# Changelog

## 2026-06-12 — Feature Release

### CUDA PTX Build Fix
- `callconv(.nvptx_device)` replaces `callconv(.kernel)` in all 60 CUDA kernel functions
- Fixes `LLVM ERROR: NVPTX aliasee must be a non-kernel function definition` (Zig 0.16/LLVM)
- Build.zig Python fixup converts `.func` device functions → `.entry` kernel entries post-compilation
- All 44 PTX kernels now compile and run on GB10 (sm_121) via CUDA 13.0 compatibility
- Verified: Qwen3.5-0.8B-Q8_0 at 24.2 tok/s decode on NVIDIA GB10

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
