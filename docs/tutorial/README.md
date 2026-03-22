# LLM Inference From Scratch

A tutorial series that builds understanding of LLM inference from the ground up. Each chapter introduces the concepts you need, exactly when you need them — no prerequisites beyond basic programming knowledge.

By the end, you'll understand every component of a production inference engine: how text becomes tokens, how tokens flow through transformer layers, why quantization matters, how GPUs execute kernels, and how the KV cache makes generation fast.

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
| 8 | [GPU Backends](08-gpu-backends.md) | CUDA, Metal, Vulkan, ROCm — how GPUs run inference |

## How This Relates to the Code

Each chapter references the Agave source files that implement the concepts discussed. The code follows the same layered structure as these tutorials — understanding the concepts makes the code straightforward to read.

For product documentation (project structure, module reference, supported models), see:
- [Architecture](../ARCHITECTURE.md) — project structure and module reference
- [Models](../MODELS.md) — supported models and performance benchmarks
- [Kernel Status](../KERNELS.md) — per-backend kernel implementation status
