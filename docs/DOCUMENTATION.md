# Agave Documentation

This documentation has been reorganized into focused sections:

## Tutorial: LLM Inference From Scratch

A progressive tutorial series that builds understanding layer by layer:

0. [Getting Started](tutorial/00-getting-started.md): pipeline from model file to sampled text
1. [Tokens and Text](tutorial/01-tokens-and-text.md) — tokenization, BPE, embedding
2. [The Transformer](tutorial/02-the-transformer.md) — attention, GQA, RoPE, normalization
3. [Feed-Forward Networks](tutorial/03-feed-forward-networks.md) — activations, SwiGLU, MoE
4. [Quantization](tutorial/04-quantization.md) — block quant, GEMV, format selection
5. [Memory and Caching](tutorial/05-memory-and-caching.md) — KV cache, PagedAttention, RadixAttention
6. [State Space Models](tutorial/06-state-space-models.md) — DeltaNet, Mamba-2, hybrids
7. [Sampling](tutorial/07-sampling.md) — temperature, top-k, top-p, min-p, XTC, DRY, Mirostat, logit bias, grammar
8. [Backends](tutorial/08-backends.md) — CPU, CUDA, Metal, Vulkan, ROCm, WebGPU, dispatch
9. [CPU SIMD Optimization](tutorial/09-cpu-simd-optimization.md) — @Vector, multi-row batching, quantized GEMV
10. [Memory Safety](tutorial/10-memory-safety.md) — defer, errdefer, leak detection
11. [Metal Backend Internals](tutorial/11-metal-backend-internals.md) — UMA, buffer caching, command buffers
12. [CPU Parallelism](tutorial/12-cpu-parallelism.md) — futex thread pool, work-stealing, atomics
13. [Batched Dispatch and Fusion](tutorial/13-batched-dispatch-and-fusion.md) — gemvMulti, fused ops, batch mode
14. [Format Conventions](tutorial/14-format-conventions.md) — GGUF vs SafeTensors, tensor layout
15. [Chat Templates](tutorial/15-chat-templates.md) — data-driven role markers, EOG tokens
16. [Recipe System](tutorial/16-recipe-system.md) — per-model/hardware defaults, user overrides
17. [Speculative Decoding & DDTree](tutorial/17-speculative-decoding.md) — 12 modes: DDTree, self-spec, n-gram, suffix, lookahead, EAGLE, EAGLE-3, MLP Speculator, Medusa, MTP, FR-Spec, shared pools
18. [Multi-Token Prediction](tutorial/18-multi-token-prediction.md) — MTP heads, +1 offset norm, draft/verify loop
19. [PFlash and Block Sparse Attention](tutorial/19-pflash-and-block-sparse.md) — block sparsity, speculative prefill, alpha tuning, scoring models, PFlash+DDTree composition
20. [Diffusion Language Models](tutorial/20-diffusion-lm.md) — DiffusionGemma, block diffusion, uniform state diffusion, bidirectional canvas attention, confidence-based acceptance, block autoregressive chaining
21. [LoRA Adapters](tutorial/21-lora.md): load-time F32 merge of low-rank adapter weights
22. [Distributed Inference](tutorial/22-distributed-inference.md): tensor/pipeline parallelism, transport selection, device sharding
23. [Server / HTTP API](tutorial/23-server-http-api.md): HTTP request pipeline, sessions, streaming, structured output

**Appendices:**
- [Troubleshooting](tutorial/appendix-troubleshooting.md): symptom → cause → fix for common inference failures
- [Mathematical Operations Reference](tutorial/appendix-math.md) — dot product, softmax, GEMV, convolution
- [Compile-Time Optimization](tutorial/appendix-compile-time.md) — comptime, @embedFile, lookup tables
- [Profiling and Debugging](tutorial/appendix-profiling.md) — --profile flag, dispatch counters
- [Atomic Operations](tutorial/appendix-atomics.md) — std.atomic.Value, memory ordering, lock-free patterns

Start here: **[Chapter 0: Getting Started](tutorial/00-getting-started.md)** (or the full [tutorial index](tutorial/README.md))

## Product Documentation

- **[HTTP API](API.md)** — server endpoints, sampling parameters, structured output, streaming
- **[Architecture](ARCHITECTURE.md)** — project structure, module reference, inference pipeline
- **[Models](MODELS.md)** — supported models, parameters, per-model details, benchmarks
- **[Kernel Status](KERNELS.md)** — per-backend kernel implementation status
- **[Megakernel System](MEGAKERNEL.md)** — three-tier megakernel architecture (fused FFN, true megakernels, composed megakernels)
- **[Benchmarks](BENCHMARKS.md)** — performance data across models, backends, and quantization types
- **[Contributing](CONTRIBUTING.md)** — how to add backends, models, quantization, megakernels, chat templates
- **[Test Matrix](TEST_MATRIX.md)** — model × backend test status and known issues
- **[Parallelism](PARALLELISM.md)** — distributed inference: TP, PP, NCCL, transports
- **[TODO & Roadmap](TODO.md)** — bugs, roadmap, design notes
- **[Changelog](../CHANGELOG.md)** — release history and feature summaries
