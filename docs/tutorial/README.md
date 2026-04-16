# LLM Inference From Scratch

A tutorial series that builds understanding of LLM inference from the ground up. Written for **systems programmers** who want to understand how LLMs work under the hood — no AI/ML background required.

Each chapter introduces one major concept at a time, explaining both the **why** (the problem being solved) and the **how** (the algorithm and implementation). All AI/ML terminology is explained inline when first mentioned.

## What You'll Learn

By the end of this tutorial, you'll understand:

- **How text becomes numbers**: Tokenization, embeddings, and vocabulary projection
- **The transformer architecture**: Attention mechanisms, position encoding, residual connections, and normalization
- **Feed-forward networks**: Activation functions, gated linear units, and Mixture of Experts (MoE)
- **Quantization**: How to compress 32-bit weights down to 4 bits (or less) while maintaining quality
- **Memory management**: KV caching strategies (flat, paged, radix tree) and why they matter for performance
- **State space models**: Linear-time alternatives to quadratic attention (DeltaNet, Mamba-2)
- **Sampling strategies**: How temperature, top-k, top-p, and repeat penalty control randomness
- **Compute backends**: How CPU, GPU (CUDA, Metal, Vulkan, ROCm) backends execute kernels and manage memory

## Prerequisites

- **Systems programming knowledge**: Comfortable reading code that manages memory, writes tight loops, and thinks about cache locality
- **Basic linear algebra**: If you've forgotten (or never learned) matrix-vector multiply, dot products, etc., see the [Math Reference](appendix-math.md) — we explain everything you need
- **No ML background needed**: We explain transformers, attention, embeddings, etc. from first principles

If you can read Zig, C, or Rust code and understand concepts like "cache line" and "SIMD", you're ready.

## Reading Paths

Different readers have different goals. Here are recommended paths through the tutorials:

### 🎓 **ML Beginners (Systems Programmers New to ML)**
Start from the beginning and read sequentially. Chapters 1-8 build understanding from first principles:
1. **Chapters 1-4** — Core concepts (tokens, transformers, quantization)
2. **Chapters 5-8** — Advanced concepts (caching, SSMs, sampling, backends)
3. **Chapters 9-16** — Implementation patterns (SIMD, memory safety, backend internals)

### 🔧 **Implementation-Focused (Experienced ML Engineers)**
You already know transformers and attention — jump straight to implementation:
- [**Chapter 9: CPU SIMD**](09-cpu-simd-optimization.md) — @Vector patterns, multi-row batching
- [**Chapter 11: Metal Backend**](11-metal-backend-internals.md) — GPU optimization on Apple Silicon
- [**Chapter 13: Batched Dispatch**](13-batched-dispatch-and-fusion.md) — Kernel fusion, dispatch reduction
- [**Appendix: Profiling**](appendix-profiling.md) — Performance debugging techniques

### ⚡ **Performance Optimization**
Focus on chapters that explain speedup techniques:
- [**Chapter 4: Quantization**](04-quantization.md#mlx-affine-quantization) — MLX factored dequantization (30-40% speedup)
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
- [**Chapter 16: Configuration and Recipes**](16-recipe-system.md) — Per-model/hardware defaults

## Reading Order

| # | Chapter | What You'll Learn |
| --- | --------- | ------------------- |
| 1 | [Tokens and Text](01-tokens-and-text.md) | How text becomes numbers the model can process |
| 2 | [The Transformer](02-the-transformer.md) | The core architecture: attention, position encoding, normalization |
| 3 | [Feed-Forward Networks](03-feed-forward-networks.md) | Activation functions, SwiGLU, and Mixture of Experts |
| 4 | [Quantization](04-quantization.md) | Compressing weights from 32 bits to 4 bits; MLX factored dequantization |
| 5 | [Memory and Caching](05-memory-and-caching.md) | KV cache, PagedAttention, and RadixAttention |
| 6 | [State Space Models](06-state-space-models.md) | Linear-time alternatives to attention: DeltaNet and Mamba-2 |
| 7 | [Sampling](07-sampling.md) | Controlling randomness: temperature, top-k, top-p |
| 8 | [Backends](08-backends.md) | CPU, CUDA, Metal, Vulkan, ROCm — how compute backends work |
| 9 | [CPU SIMD Optimization](09-cpu-simd-optimization.md) | @Vector, @reduce, @mulAdd, multi-row batching, quantized GEMV |
| 10 | [Memory Safety](10-memory-safety.md) | defer, errdefer, guaranteed cleanup, leak detection |
| 11 | [Metal Backend Internals](11-metal-backend-internals.md) | UMA, buffer caching, command buffers, batch mode, threadgroup limits |
| 12 | [CPU Parallelism](12-cpu-parallelism.md) | Futex-based thread pool, work-stealing, atomic counters, main thread participation |
| 13 | [Batched Dispatch and Fusion](13-batched-dispatch-and-fusion.md) | gemvMulti, fused ops (addRmsNorm, siluMul, splitQGate), batch mode |
| 14 | [Format Conventions](14-format-conventions.md) | GGUF vs SafeTensors differences, tensor layout, metadata mapping |
| 15 | [Chat Templates](15-chat-templates.md) | Data-driven role markers, EOG tokens, multi-turn formatting |
| 16 | [Configuration and Recipes](16-recipe-system.md) | Proven defaults per model+hardware, user override semantics |

**Appendices:**
- [Mathematical Operations Reference](appendix-math.md) — Quick reference for all math operations (dot product, softmax, GEMV, convolution, etc.)
- [Compile-Time Optimization](appendix-compile-time.md) — comptime keyword, @embedFile, lookup tables, feature detection, type specialization
- [Profiling and Debugging](appendix-profiling.md) — --profile flag, dispatch counters, missing kernel policy, regression detection
- [Atomic Operations and Memory Ordering](appendix-atomics.md) — std.atomic.Value, memory ordering semantics, lock-free patterns

## How This Relates to the Code

Each chapter references the Agave source files that implement the concepts discussed. The code follows the same layered structure as these tutorials — understanding the concepts makes the code straightforward to read.

For product documentation (project structure, module reference, supported models), see:

- [Architecture](../ARCHITECTURE.md) — project structure and module reference
- [Models](../MODELS.md) — supported models and performance benchmarks
- [Kernel Status](../KERNELS.md) — per-backend kernel implementation status
