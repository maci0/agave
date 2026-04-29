# Supported Models (2026-04-08)

## Overview

| Model | Arch ID | Attention | FFN | Special |
|-------|---------|-----------|-----|---------|
| **Gemma 3** | `gemma3` | GQA + QK norm + post-norms | GELU + SwiGLU | Embedding scaling, logit softcap, vision (SigLIP) |
| **Qwen 3.5/3.6** | `qwen35` | GQA (every 4th layer) | SiLU + SwiGLU | DeltaNet SSM hybrid, MoE (3.5-35B, 3.6-35B) |
| **GPT-OSS** | `gpt_oss` | GQA + sliding window + sinks | SiLU + SwiGLU | MoE (top-4 of 32 experts) |
| **Nemotron-H** | `nemotron_h` | GQA (sparse layers) | SiLU + SwiGLU | Mamba-2 SSM hybrid (GGUF) |
| **Nemotron Nano** | `nemotron_nano` | GQA (sparse layers) | ReLU² MoE | SSM + MoE + attention hybrid (NVFP4) |
| **Gemma 4** | `gemma4` | GQA + QK norm + post-norms | GELU + SwiGLU | MoE (top-8) or dense, PLE (E2B/E4B), vision (SigLIP-2), Q4_K/Q5_K/Q6_K GEMM |
| **GLM-4** | `glm4` | MLA (compressed KV) | SiLU + SwiGLU | MoE (64 experts, top-4, sigmoid routing) |

## Model Parameters

| Model | n_embd | n_heads | n_kv_heads | head_dim | ff_dim | n_layers | theta | rope_dim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Gemma3 1B | 1152 | 4 | 1 | 256 | 6912 | 26 | 1M | 256 |
| Qwen3.5 0.8B | 1536 | 16 | 4 | 128 | 4096 | 64 | 10M | 64 |
| Qwen3.6 35B-A3B | 2048 | 16 | 2 | 256 | 512 (MoE×256) | 40 | 10M | 64 |
| GPT-OSS | 2880 | 64 | 8 | 64 | 2880 (MoE) | 24 | 150K | 64 |
| Nemotron-H | 3136 | 40 | 8 | 128 | 12544 | 42 | 10K | 78 |
| Nemotron-Nano | 2688 | 32 | 2 | 128 | 1856 (MoE) | 52 | 10K | 128 |
| Gemma4 E2B | 2304 | 8 | 4 | 256 | 9216 | 28 | 10K | 256 |
| Gemma4 E4B | 2816 | 16 | 8 | 256 | 11264 | 30 | 10K | 256 |
| Gemma4 26B-A4B | 2816 | 16 | 8/2 (sl/gl) | 256/512 (sl/gl) | 2816 + 704/expert (MoE) | 30 | 10K/1M (sl/gl) | 256/128 (sl/gl) |
| GLM-4 | 2048 | 20 | 20 (MLA) | 256 (qk_nope=192 + qk_rope=64) | 10240 (dense) / 1536 (MoE, 64 experts top-4) | 47 | 1M | 64 |

## Model-Specific Details

**Gemma 3**: GGUF converter bakes +1.0 into RMS norm weights (don't add again). Embeddings scaled by `sqrt(n_embd)`. Uses SPM tokenizer (no merges). Tied output embeddings. Vision supported via SigLIP encoder. Supports `--megakernel` (fused FFN GELU, true megakernel Q4K/Q8 on Metal+CUDA).

**Qwen 3.5/3.6**: Hybrid architecture alternating DeltaNet SSM and full attention layers. DeltaNet uses causal conv1d → selective state recurrence with learned decay. Full attention layers have gated output with sigmoid. Qwen 3.6-35B-A3B uses the same architecture with 40 layers, 256 experts (top-8 + shared), hidden_size 2048. Formats: GGUF (Q4_K_M, Q8_0), SafeTensors (BF16, MLX-4bit, NVFP4 compressed-tensors partial). Supports `--megakernel` (fused FFN SiLU, true megakernel Q8/Q4K on Metal+CUDA+ROCm).

**GPT-OSS**: Even layers = 128-token sliding window, odd = full sequence. Learned attention sinks per head. Clamped SwiGLU `[-7.0, +7.0]` in MoE experts.

**Nemotron-H** (GGUF): Mamba-2 SSM with per-group RMS normalization. Layer types (SSM/attention/FFN-only) detected from tensor presence. Squared ReLU for FFN-only layers. Supports `--megakernel` (true megakernel Q8 on Metal).

**Nemotron Nano** (SafeTensors NVFP4): 52-layer hybrid with `hybrid_override_pattern` (M=SSM, E=MoE, *=attention). Mixed quant — most layers NVFP4, 6 SSM layers use BF16. 128 routed experts, top-6 + shared expert.

**Gemma 4**: Three variants — E2B and E4B are dense (no MoE), 26B-A4B uses MoE (128 experts, top-8 softmax) + dense FFN path. All variants use dual attention (sliding-window + global layers) and PLE (Per-Layer Embeddings). Shared KV cache for trailing layers. Channel-based chat template. Vision supported via SigLIP-2 encoder. Supports `--megakernel` (fused FFN GELU for dense+MoE, true megakernel Q4K/Q8 on Metal+CUDA).

**GLM-4** (MLX): MLA compresses K/V into latent space. Sigmoid routing (independent expert gates, not competing). MLX 4/6/8-bit affine quantization. Supports `--megakernel` (fused FFN SiLU on Metal).

## Performance

### Apple M4 Pro (48 GB)

| Model | Quant | Backend | tok/s |
|-------|-------|---------|-------|
| Qwen3.5 0.8B | Q8_0 | Metal | 183 |
| Qwen3.5 0.8B | Q4_0 | Metal | 110 |
| Qwen3.5 9B | Q4_0 | Metal | 34.5 |
| Gemma3 27B | QAT 4-bit | Metal | 11.6 |
| Gemma3 27B | QAT 4-bit | CPU | 3.2 |

### NVIDIA GB10 (Blackwell, UMA)

| Model | Quant | Backend | tok/s |
|-------|-------|---------|-------|
| Gemma3 1B | Q4_0 | CUDA | 40 |
| Gemma3 1B | Q4_0 | CPU | 5.7 |
