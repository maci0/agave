# Requirements: Agave Production-Ready Inference Engine

**Defined:** 2026-03-21
**Core Value:** Every supported model must produce correct output on every backend at full GPU speed

## v1 Requirements

Requirements for production readiness. Each maps to roadmap phases.

### GPU Kernel Parity

- [ ] **KERN-01**: Metal GPU SDPA kernel produces correct output without CPU fallback
- [ ] **KERN-02**: Metal SDPA uses compute-only path (no blit encoder switching in hot path)
- [ ] **KERN-03**: CUDA GPU GEMV kernel for Q4_K quantization format
- [ ] **KERN-04**: CUDA GPU GEMV kernel for Q5_K quantization format
- [ ] **KERN-05**: CUDA GPU GEMV kernel for Q6_K quantization format
- [ ] **KERN-06**: CUDA GPU GEMV kernel for FP8 E4M3/E5M2 formats
- [ ] **KERN-07**: CUDA parallel SDPA softmax using warp-only reductions (replacing serial thread-0 workaround)
- [ ] **KERN-08**: Vulkan GPU embedding lookup kernel (eliminate CPU fallback)
- [ ] **KERN-09**: Vulkan GPU conv1d kernel for SSM models (eliminate CPU fallback)
- [ ] **KERN-10**: All GPU kernels pass dual-delta numerical tests (GPU error <= 2x CPU error vs FP64 oracle)

### Model Verification

- [ ] **MODL-01**: Nemotron Nano 30B produces correct, coherent output (fix MoE routing instability)
- [ ] **MODL-02**: GLM-4 produces correct output (MLA attention + sigmoid-gated MoE)
- [ ] **MODL-03**: GPT-OSS produces correct output (sliding window + MoE with clamped SwiGLU)
- [ ] **MODL-04**: Nemotron-H produces correct output (hybrid SSM + attention)
- [ ] **MODL-05**: All 6 models verified on CPU backend with golden test output
- [ ] **MODL-06**: All 6 models verified on Metal backend with golden test output
- [ ] **MODL-07**: All 6 models verified on CUDA backend (DGX Spark) with golden test output
- [ ] **MODL-08**: All 6 models verified on Vulkan backend with golden test output
- [ ] **MODL-09**: Automated golden tests comparing output against reference (llama.cpp or HuggingFace)

### Production Serving

- [ ] **SERV-01**: Continuous batching scheduler processes multiple concurrent requests with iteration-level scheduling
- [ ] **SERV-02**: PagedAttention integrated into model inference loop with block tables
- [ ] **SERV-03**: RadixAttention prefix caching integrated into server with automatic prefix detection
- [ ] **SERV-04**: RadixAttention LRU eviction using frequency x cost metric (not simple LRU)
- [ ] **SERV-05**: OpenAI-compatible /v1/chat/completions API with full schema compliance
- [ ] **SERV-06**: SSE streaming with correct OpenAI event format (data: [DONE])
- [ ] **SERV-07**: Per-request timeout with inference cancellation (configurable 30-120s)
- [ ] **SERV-08**: Rate limiting per API key (requests/min + tokens/min, token bucket algorithm)
- [ ] **SERV-09**: API key authentication (--api-key CLI flag, Authorization: Bearer header)
- [ ] **SERV-10**: Prometheus /metrics endpoint (throughput, latency p50/p95/p99, queue depth, KV cache usage)
- [ ] **SERV-11**: Health check endpoints (/health, /ready)
- [ ] **SERV-12**: Graceful shutdown (drain in-flight requests before exit)

### Tiered KV Cache

- [ ] **TIER-01**: PagedKvCache block tier tag (enum { vram, ram, ssd }) with tier-aware allocation
- [ ] **TIER-02**: Automatic demotion of cold KV pages from VRAM to RAM when VRAM budget exceeded
- [ ] **TIER-03**: Automatic promotion of needed KV pages from RAM back to VRAM with LRU eviction
- [ ] **TIER-04**: SSD tier support with async I/O for KV page spill/restore (--kv-ssd-path, --kv-ssd-budget)
- [ ] **TIER-05**: Prefetching of next KV pages from lower tiers during attention compute (overlap I/O with compute)
- [ ] **TIER-06**: Zero-copy access paths per backend (Metal newBufferWithBytesNoCopy for RAM, GPUDirect Storage for CUDA, VK_EXT_external_memory_host for Vulkan)
- [ ] **TIER-07**: CLI flags for tier configuration (--kv-tiers, --kv-ram-budget, --kv-ssd-path, --kv-ssd-budget)

### Parallelism

- [ ] **PARA-01**: DeviceGroup abstraction supporting multiple backends per model
- [ ] **PARA-02**: Weight tensor sharding across devices (column-parallel and row-parallel splits)
- [ ] **PARA-03**: All-reduce communication primitive per backend (Metal, CUDA, Vulkan, CPU)
- [ ] **PARA-04**: Tensor Parallelism (TP) for attention blocks (Q/K/V column-parallel, W_o row-parallel)
- [ ] **PARA-05**: Tensor Parallelism (TP) for FFN blocks (W_gate/W_up column-parallel, W_down row-parallel)
- [ ] **PARA-06**: Pipeline Parallelism (PP) splitting layers across devices with point-to-point activation transfer
- [ ] **PARA-07**: Expert Parallelism (EP) distributing MoE experts across devices with all-to-all token exchange
- [ ] **PARA-08**: KV cache partitioning under TP (each device holds n_kv_heads/TP heads)
- [ ] **PARA-09**: KV cache per-stage assignment under PP
- [ ] **PARA-10**: Hybrid TP + PP configuration support
- [ ] **PARA-11**: CLI auto-detection of available devices and topology (--tp, --pp flags)
- [ ] **PARA-12**: Communication buffers pre-allocated at init (zero allocations in hot path)

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Optimization

- **OPT-01**: FlashAttention-2 tiling for Metal and CUDA SDPA kernels
- **OPT-02**: Fused MLP kernel (SwiGLU gate + activation in single dispatch)
- **OPT-03**: Fused RoPE + Attention kernel
- **OPT-04**: Speculative decoding with draft model (1.3-1.7x speedup)
- **OPT-05**: Chunked prefill (interleave prefill chunks with decode batches)

### Extended Features

- **FEAT-01**: Structured output / constrained decoding (XGrammar integration)
- **FEAT-02**: Function/tool calling support in OpenAI API
- **FEAT-03**: Multi-model serving (model registry + lazy loading)
- **FEAT-04**: CUDA GEMV for NVFP4/MXFP4 microscaled formats
- **FEAT-05**: Vision/multimodal input support
- **FEAT-06**: Multi-LoRA serving
- **FEAT-07**: Disaggregated prefill/decode across GPU pools
- **FEAT-08**: Sequence parallelism and context parallelism

## Out of Scope

| Feature | Reason |
|---------|--------|
| Training / fine-tuning | Different problem domain, massive scope creep |
| Windows support | Linux + macOS only; WSL2 works for Windows users |
| Python bindings | Focus on C API + HTTP API; users use requests library |
| Web UI | Users choose their own UI (Open WebUI, LibreChat, etc.) |
| Model auto-download | HuggingFace Hub already handles this well |
| Built-in RAG/vector search | Orthogonal concern; use dedicated vector DBs |
| Custom templating DSL | Keep chat templates as simple data structures |
| Proprietary binary format | GGUF and SafeTensors are open standards |
| Blockchain/crypto | Zero technical value for inference |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| KERN-01 | Phase 1 | Pending |
| KERN-02 | Phase 1 | Pending |
| KERN-03 | Phase 1 | Pending |
| KERN-04 | Phase 1 | Pending |
| KERN-05 | Phase 1 | Pending |
| KERN-06 | Phase 1 | Pending |
| KERN-07 | Phase 1 | Pending |
| KERN-08 | Phase 1 | Pending |
| KERN-09 | Phase 1 | Pending |
| KERN-10 | Phase 1 | Pending |
| MODL-01 | Phase 1 | Pending |
| MODL-02 | Phase 1 | Pending |
| MODL-03 | Phase 1 | Pending |
| MODL-04 | Phase 1 | Pending |
| MODL-05 | Phase 1 | Pending |
| MODL-06 | Phase 1 | Pending |
| MODL-07 | Phase 1 | Pending |
| MODL-08 | Phase 1 | Pending |
| MODL-09 | Phase 1 | Pending |
| SERV-01 | Phase 2 | Pending |
| SERV-02 | Phase 2 | Pending |
| SERV-03 | Phase 3 | Pending |
| SERV-04 | Phase 3 | Pending |
| SERV-05 | Phase 2 | Pending |
| SERV-06 | Phase 2 | Pending |
| SERV-07 | Phase 2 | Pending |
| SERV-08 | Phase 2 | Pending |
| SERV-09 | Phase 2 | Pending |
| SERV-10 | Phase 2 | Pending |
| SERV-11 | Phase 2 | Pending |
| SERV-12 | Phase 2 | Pending |
| TIER-01 | Phase 3 | Pending |
| TIER-02 | Phase 3 | Pending |
| TIER-03 | Phase 3 | Pending |
| TIER-04 | Phase 3 | Pending |
| TIER-05 | Phase 3 | Pending |
| TIER-06 | Phase 3 | Pending |
| TIER-07 | Phase 3 | Pending |
| PARA-01 | Phase 4 | Pending |
| PARA-02 | Phase 4 | Pending |
| PARA-03 | Phase 4 | Pending |
| PARA-04 | Phase 4 | Pending |
| PARA-05 | Phase 4 | Pending |
| PARA-06 | Phase 4 | Pending |
| PARA-07 | Phase 4 | Pending |
| PARA-08 | Phase 4 | Pending |
| PARA-09 | Phase 4 | Pending |
| PARA-10 | Phase 4 | Pending |
| PARA-11 | Phase 4 | Pending |
| PARA-12 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 50 total
- Mapped to phases: 50
- Unmapped: 0

---
*Requirements defined: 2026-03-21*
*Last updated: 2026-03-21 after initial definition*
