# Agave Documentation

This documentation has been reorganized into focused sections:

## Tutorial: LLM Inference From Scratch

A progressive tutorial series that builds understanding layer by layer:

1. [Tokens and Text](tutorial/01-tokens-and-text.md) — tokenization, BPE, embedding
2. [The Transformer](tutorial/02-the-transformer.md) — attention, GQA, RoPE, normalization
3. [Feed-Forward Networks](tutorial/03-feed-forward-networks.md) — activations, SwiGLU, MoE
4. [Quantization](tutorial/04-quantization.md) — block quant, GEMV, format selection
5. [Memory and Caching](tutorial/05-memory-and-caching.md) — KV cache, PagedAttention, RadixAttention
6. [State Space Models](tutorial/06-state-space-models.md) — DeltaNet, Mamba-2, hybrids
7. [Sampling](tutorial/07-sampling.md) — temperature, top-k, top-p, repeat penalty
8. [Backends](tutorial/08-backends.md) — CPU, CUDA, Metal, Vulkan, ROCm, dispatch

Start here: **[tutorial/README.md](tutorial/README.md)**

## Product Documentation

- **[Architecture](ARCHITECTURE.md)** — project structure, module reference, inference pipeline
- **[Models](MODELS.md)** — supported models, parameters, per-model details, benchmarks
- **[Kernel Status](KERNELS.md)** — per-backend kernel implementation status
- **[Ideas](IDEAS.md)** — future work and optimization ideas
