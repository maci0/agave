# LLM Inference From Scratch

A tutorial series that builds understanding of LLM inference from the ground up. Written for **systems programmers** who want to understand how LLMs work under the hood — no AI/ML background required.

Each chapter introduces one major concept at a time, explaining both the **why** (the problem being solved) and the **how** (the algorithm and implementation). All AI/ML terminology is explained inline when first mentioned.

**Start here:** [Chapter 0: Getting Started](00-getting-started.md)

**A note on code samples:** these tutorials teach algorithms as language-agnostic pseudocode (fenced as `text`), not as compilable Zig. Each pseudocode block is followed by an **Implementation** link pointing at the real source in `src/`.

## What You'll Learn

By the end of this tutorial, you'll understand:

- **The inference pipeline**: From model file on disk through tokenization, prefill, decode, and sampling ([Chapter 0](00-getting-started.md))
- **How text becomes numbers**: Tokenization, embeddings, and vocabulary projection
- **The transformer architecture**: Attention mechanisms, position encoding, residual connections, and normalization
- **Feed-forward networks**: Activation functions, gated linear units, and Mixture of Experts (MoE)
- **Quantization**: How to compress 32-bit weights down to 4 bits (or less) while maintaining quality
- **Memory management**: KV caching strategies (flat, paged, radix tree) and why they matter for performance
- **State space models**: Linear-time alternatives to quadratic attention (DeltaNet, Mamba-2)
- **Sampling strategies**: Temperature, top-k, top-p, min-p, XTC, DRY, Mirostat, logit bias, grammar-constrained decoding
- **Compute backends**: How CPU, GPU (CUDA, Metal, Vulkan, ROCm, WebGPU) backends execute kernels and manage memory
- **Speculative decoding**: Draft models, DDTree tree construction, self-speculative layer skipping, n-gram, EAGLE, DSpark confidence-scheduled verification, adaptive K
- **Multi-Token Prediction**: MTP heads, +1 offset norm, built-in draft tokens, 70-85% acceptance rates
- **PFlash / block sparse**: Scorer-driven prefill compression for long contexts
- **Diffusion LMs**: Block diffusion (DiffusionGemma), bidirectional canvas, confidence acceptance
- **LoRA adapters**: Load-time merge of low-rank adapter weights into base tensors
- **Distributed inference**: Tensor and pipeline parallelism, transport selection, device sharding
- **Server pipeline**: HTTP request handling, sessions, streaming, and structured output

## Prerequisites

- **Systems programming knowledge**: Comfortable reading code that manages memory, writes tight loops, and thinks about cache locality
- **Basic linear algebra**: If you've forgotten (or never learned) matrix-vector multiply, dot products, etc., see the [Math Reference](appendix-math.md) — we explain everything you need
- **No ML background needed**: We explain transformers, attention, embeddings, etc. from first principles

If you can read Zig, C, or Rust code and understand concepts like "cache line" and "SIMD", you're ready.

## Reading Paths

Different readers have different goals. Here are recommended paths through the tutorials:

### 🧑‍💻 **Beginner Systems Programmer (Knows C, New to Zig/ML)**
You're comfortable with C-style memory management and pointers but haven't touched Zig or ML before. This path front-loads Zig idioms and memory safety before backend/hardware detail, then loops back for the sampling and caching pieces that tie generation together:
- [**Chapter 0: Getting Started**](00-getting-started.md) → [**Chapter 1: Tokens**](01-tokens-and-text.md) → [**Chapter 2: Transformer**](02-the-transformer.md) → [**Chapter 3: FFN**](03-feed-forward-networks.md)
- [**Chapter 10: Memory Safety**](10-memory-safety.md) → [**Chapter 8: Backends**](08-backends.md) → [**Chapter 7: Sampling**](07-sampling.md) → [**Chapter 5: Caching**](05-memory-and-caching.md)

### 🎓 **ML Beginners (Systems Programmers New to ML)**
Start from the beginning and read sequentially. Chapters 0–8 build understanding from first principles:
- [**Chapter 0: Getting Started**](00-getting-started.md) → [**Chapter 1: Tokens**](01-tokens-and-text.md) → [**Chapter 2: Transformer**](02-the-transformer.md) → [**Chapter 3: FFN**](03-feed-forward-networks.md) → [**Chapter 4: Quantization**](04-quantization.md)
- [**Chapter 5: Caching**](05-memory-and-caching.md) → [**Chapter 6: SSMs**](06-state-space-models.md) → [**Chapter 7: Sampling**](07-sampling.md) → [**Chapter 8: Backends**](08-backends.md)
- [**Chapter 9: SIMD**](09-cpu-simd-optimization.md) → [**Chapter 10: Memory Safety**](10-memory-safety.md) → [**Chapter 11: Metal**](11-metal-backend-internals.md) → onward

### 🔧 **Implementation-Focused (Experienced ML Engineers)**
You already know transformers and attention — jump straight to implementation:
- [**Chapter 9: CPU SIMD**](09-cpu-simd-optimization.md) — @Vector patterns, multi-row batching
- [**Chapter 11: Metal Backend**](11-metal-backend-internals.md) — GPU optimization on Apple Silicon
- [**Chapter 13: Batched Dispatch**](13-batched-dispatch-and-fusion.md) — Kernel fusion, dispatch reduction
- [**Appendix: Profiling**](appendix-profiling.md) — Performance debugging techniques

### ⚡ **Performance Optimization**
Focus on chapters that explain speedup techniques:
- [**Chapter 4: Quantization**](04-quantization.md#mlx-affine-quantization) — MLX factored dequantization (fewer arithmetic ops per block)
- [**Chapter 9: CPU SIMD**](09-cpu-simd-optimization.md) — Multi-row GEMV batching (2-4× speedup)
- [**Chapter 13: Batched Dispatch**](13-batched-dispatch-and-fusion.md) — Qwen3.5 optimization journey (15% speedup)
- [**Appendix: Compile-Time**](appendix-compile-time.md) — Lookup tables (20-30× for FP8 dequant)

### 🦀 **Zig-Specific Patterns (Rust/C Programmers)**
Learn Zig idioms used throughout the codebase:
- [**Chapter 9: CPU SIMD**](09-cpu-simd-optimization.md) — @Vector, @reduce, @mulAdd, @splat
- [**Chapter 10: Memory Safety**](10-memory-safety.md) — defer, errdefer, leak detection
- [**Chapter 12: CPU Parallelism**](12-cpu-parallelism.md) — Futex-based thread pool, atomic operations
- [**Appendix: Compile-Time**](appendix-compile-time.md) — comptime, @embedFile, inline else dispatch
- [**Appendix: Atomic Operations**](appendix-atomics.md) — Memory ordering, lock-free patterns

### 📐 **Architecture & Design Patterns**
Understand how the codebase is structured:
- [**Chapter 8: Backends**](08-backends.md) — Tagged union dispatch pattern
- [**Chapter 14: Format Conventions**](14-format-conventions.md) — GGUF vs SafeTensors differences
- [**Chapter 15: Chat Templates**](15-chat-templates.md) — Data-driven configuration
- [**Chapter 16: Recipe System**](16-recipe-system.md) — Per-model/hardware defaults

### 🛠️ **Adding a New Model**
Everything you need to add a new architecture to Agave:
- [**Chapter 14: Format Conventions**](14-format-conventions.md) — Tensor naming, dimension order, format detection
- [**Chapter 15: Chat Templates**](15-chat-templates.md) — Prompt formatting and EOG tokens
- [**Chapter 16: Recipe System**](16-recipe-system.md) — Per-model defaults
- [**Chapter 8: Backends**](08-backends.md) — Dispatcher pattern and kernel interface
- [**Chapter 13: Batched Dispatch**](13-batched-dispatch-and-fusion.md#megakernel-system-three-tier-architecture) — Tier 3 composed megakernels (auto-generated from ModelDesc)

## Reading Order

| # | Chapter | What You'll Learn | ~Time |
| --- | --------- | ------------------- | :---: |
| 0 | [Getting Started](00-getting-started.md) | Pipeline from model file to sampled text tokens | 15 min |
| 1 | [Tokens and Text](01-tokens-and-text.md) | How text becomes numbers the model can process | 12 min |
| 2 | [The Transformer](02-the-transformer.md) | The core architecture: attention, position encoding, normalization | 21 min |
| 3 | [Feed-Forward Networks](03-feed-forward-networks.md) | Activation functions, SwiGLU, MoE, megakernel fusion | 12 min |
| 4 | [Quantization](04-quantization.md) | Compressing weights from 32 bits to 4 bits; MLX, TurboQuant, PlanarQuant | 28 min |
| 5 | [Memory and Caching](05-memory-and-caching.md) | KV cache, PagedAttention, paged SDPA, RadixAttention | 17 min |
| 6 | [State Space Models](06-state-space-models.md) | Linear-time alternatives to attention: DeltaNet and Mamba-2 | 14 min |
| 7 | [Sampling](07-sampling.md) | Temperature, top-k, top-p, min-p, XTC, DRY, Mirostat, logit bias, grammar | 12 min |
| 8 | [Backends](08-backends.md) | CPU, CUDA, Metal, Vulkan, ROCm, WebGPU — dispatchers and paged SDPA | 15 min |
| 9 | [CPU SIMD Optimization](09-cpu-simd-optimization.md) | @Vector, @reduce, @mulAdd, multi-row batching, quantized GEMV | 19 min |
| 10 | [Memory Safety](10-memory-safety.md) | defer, errdefer, guaranteed cleanup, leak detection | 11 min |
| 11 | [Metal Backend Internals](11-metal-backend-internals.md) | UMA, buffer caching, command buffers, batch mode, threadgroup limits | 19 min |
| 12 | [CPU Parallelism](12-cpu-parallelism.md) | Futex-based thread pool, work-stealing, atomic counters | 15 min |
| 13 | [Batched Dispatch and Fusion](13-batched-dispatch-and-fusion.md) | gemvMulti, fused ops, megakernel system (3 tiers) | 22 min |
| 14 | [Format Conventions](14-format-conventions.md) | GGUF vs SafeTensors differences, tensor layout, metadata mapping | 20 min |
| 15 | [Chat Templates](15-chat-templates.md) | Data-driven role markers, EOG tokens, multi-turn formatting | 18 min |
| 16 | [Recipe System](16-recipe-system.md) | Proven defaults per model+hardware, user override semantics | 15 min |
| 17 | [Speculative Decoding & DDTree](17-speculative-decoding.md) | Draft models, DDTree, self-speculative, n-gram, EAGLE, DSpark, adaptive K | 29 min |
| 18 | [Multi-Token Prediction](18-multi-token-prediction.md) | MTP heads, +1 offset norm, draft/verify with built-in heads | 16 min |
| 19 | [PFlash and Block Sparse](19-pflash-and-block-sparse.md) | Block sparse attention, speculative prefill, alpha tuning | 18 min |
| 20 | [Diffusion Language Models](20-diffusion-lm.md) | DiffusionGemma, block diffusion, bidirectional canvas | 14 min |
| 21 | [LoRA Adapters](21-lora.md) | Load-time merge of low-rank adapter weights | 12 min |
| 22 | [Distributed Inference](22-distributed-inference.md) | TP/PP, transport selection, weight sharding | 20 min |
| 23 | [Server / HTTP API](23-server-http-api.md) | HTTP → session → generate → stream/JSON pipeline | 18 min |

## Quick Reference

Looking for one specific topic instead of a full reading path? Jump straight to the relevant chapters:

| Feature | Chapters |
|---------|----------|
| Tokenization / embeddings | 0, 1 |
| Attention / RoPE / GQA | 2 |
| Quantization | 4 |
| KV cache | 5 |
| Sampling / grammar | 7 |
| Backends | 8, 11 |
| Speculative decoding | 17, 18, 19 |
| LoRA | 14, 21 |
| Distributed | 8, 12, 22 |
| Server pipeline | 7, 15, 23 |
| Diffusion LM | 20 |

**Appendices:**
- [Troubleshooting](appendix-troubleshooting.md): Symptom → cause → fix for common inference failures
- [Mathematical Operations Reference](appendix-math.md) — Quick reference for all math operations (dot product, softmax, GEMV, convolution, etc.)
- [Compile-Time Optimization](appendix-compile-time.md) — comptime keyword, @embedFile, lookup tables, feature detection, type specialization
- [Profiling and Debugging](appendix-profiling.md) — --profile flag, dispatch counters, missing kernel policy, regression detection
- [Atomic Operations and Memory Ordering](appendix-atomics.md) — std.atomic.Value, memory ordering semantics, lock-free patterns

## How This Relates to the Code

Each chapter references the Agave source files that implement the concepts discussed. The code follows the same layered structure as these tutorials — understanding the concepts makes the code straightforward to read.

Every chapter ends with a **Glossary** section defining all new terms introduced in that chapter. Terms are explained inline on first use and collected at the end for quick reference.

For product documentation (project structure, module reference, supported models), see:

- [Architecture](../ARCHITECTURE.md) — project structure and module reference
- [Models](../MODELS.md) — supported models and performance benchmarks
- [Kernel Status](../KERNELS.md) — per-backend kernel implementation status

## Further Reading

If these tutorials leave you wanting to go deeper, these are worth your time:

**Foundations**
- *The Little Book of Deep Learning* — François Fleuret. Dense, precise, free. Best mathematical treatment of transformers that fits in 200 pages.
- *Understanding Deep Learning* — Simon Prince. Broader coverage, available free online. Good on the math behind training and loss landscapes.

**Architecture & Systems**
- *Dive into Deep Learning* — d2l.ai. Extremely practical, chapter on attention is excellent. Every concept has working code.
- *GPU Puzzles* — Sasha Rush. Interactive CUDA puzzles. If you want to understand why memory layout matters, work through these.

**Inference & Deployment**
- *LLM Inference Optimization* — NVIDIA TensorRT-LLM docs/blog posts. Dense but accurate on batching, KV cache management, and throughput vs. latency tradeoffs.
- FlashAttention papers (Dao et al. 2022, 2023) — The original papers are readable and explain exactly why the tiled SDPA algorithm works. Chapter 5 of this tutorial is a summary.

**Quantization**
- *A White Paper on Neural Network Quantization* — Nagel et al. (Qualcomm). The most complete treatment of quantization theory: PTQ, QAT, GPTQ, and calibration.
- GGUF/llama.cpp source — `ggml-quants.c` is the reference implementation for every quantization format this engine supports.
