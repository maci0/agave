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

## Reading Order

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| 1 | [Tokens and Text](01-tokens-and-text.md) | How text becomes numbers the model can process |
| 2 | [The Transformer](02-the-transformer.md) | The core architecture: attention, position encoding, normalization |
| 3 | [Feed-Forward Networks](03-feed-forward-networks.md) | Activation functions, SwiGLU, and Mixture of Experts |
| 4 | [Quantization](04-quantization.md) | Compressing weights from 32 bits to 4 bits without losing quality |
| 5 | [Memory and Caching](05-memory-and-caching.md) | KV cache, PagedAttention, and RadixAttention |
| 6 | [State Space Models](06-state-space-models.md) | Linear-time alternatives to attention: DeltaNet and Mamba-2 |
| 7 | [Sampling](07-sampling.md) | Controlling randomness: temperature, top-k, top-p |
| 8 | [Backends](08-backends.md) | CPU, CUDA, Metal, Vulkan, ROCm — how compute backends work |

**Appendix**: [Mathematical Operations Reference](appendix-math.md) — Quick reference for all math operations (dot product, softmax, GEMV, convolution, etc.)

## How This Relates to the Code

Each chapter references the Agave source files that implement the concepts discussed. The code follows the same layered structure as these tutorials — understanding the concepts makes the code straightforward to read.

For product documentation (project structure, module reference, supported models), see:
- [Architecture](../ARCHITECTURE.md) — project structure and module reference
- [Models](../MODELS.md) — supported models and performance benchmarks
- [Kernel Status](../KERNELS.md) — per-backend kernel implementation status
