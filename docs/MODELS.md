# Supported Models

## Overview

| Model | Arch ID | Attention | FFN | Special |
|-------|---------|-----------|-----|---------|
| **Gemma 3** | `gemma3` | GQA + QK norm + post-norms | GELU + SwiGLU | Embedding scaling, logit softcap |
| **Qwen 3.5** | `qwen35` | GQA (every 4th layer) | SiLU + SwiGLU | DeltaNet SSM hybrid |
| **GPT-OSS** | `gpt_oss` | GQA + sliding window + sinks | SiLU + SwiGLU | MoE (top-4 of 32 experts) |
| **Nemotron-H** | `nemotron_h` | GQA (sparse layers) | SiLU + SwiGLU | Mamba-2 SSM hybrid (GGUF) |
| **Nemotron Nano** | `nemotron_nano` | GQA (sparse layers) | ReLU² MoE | SSM + MoE + attention hybrid (NVFP4) |
| **GLM-4** | `glm4` | MLA (compressed KV) | SiLU + SwiGLU | MoE + sigmoid routing |

## Model Parameters

| Model | n_embd | n_heads | n_kv_heads | head_dim | ff_dim | n_layers | theta | rope_dim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Gemma3 1B | 1152 | 4 | 1 | 256 | 6912 | 26 | 1M | 256 |
| Qwen3.5 0.8B | 1536 | 16 | 4 | 128 | 4096 | 64 | 10M | 64 |
| GPT-OSS | 2880 | 64 | 8 | 64 | 2880 (MoE) | 24 | 150K | 128 |
| Nemotron-H | 3136 | 40 | 8 | 128 | 12544 | 42 | 10K | 78 |
| Nemotron-Nano | 2688 | 32 | 2 | 128 | 1856 (MoE) | 52 | 10K | 128 |
| GLM-4 | 2048 | 20 | 20 (MLA) | 256 | 1536 (MoE) | 47 | 1M | 64 |

## Model-Specific Details

**Gemma 3**: GGUF converter bakes +1.0 into RMS norm weights (don't add again). Embeddings scaled by `sqrt(n_embd)`. Uses SPM tokenizer (no merges). Tied output embeddings.

**Qwen 3.5**: Hybrid architecture alternating DeltaNet SSM and full attention layers. DeltaNet uses causal conv1d → selective state recurrence with learned decay. Full attention layers have gated output with sigmoid.

**GPT-OSS**: Even layers = 128-token sliding window, odd = full sequence. Learned attention sinks per head. Clamped SwiGLU `[-7.0, +7.0]` in MoE experts.

**Nemotron-H** (GGUF): Mamba-2 SSM with per-group RMS normalization. Layer types (SSM/attention/FFN-only) detected from tensor presence. Squared ReLU for FFN-only layers.

**Nemotron Nano** (SafeTensors NVFP4): 52-layer hybrid with `hybrid_override_pattern` (M=SSM, E=MoE, *=attention). Mixed quant — most layers NVFP4, 6 SSM layers use BF16. 128 routed experts, top-6 + shared expert.

**GLM-4** (MLX): MLA compresses K/V into latent space. Sigmoid routing (independent expert gates, not competing). MLX 4/6-bit affine quantization.

## Performance

### Apple M4 Pro (48 GB)

| Model | Quant | Backend | tok/s |
|-------|-------|---------|-------|
| Qwen3.5 0.8B | Q8_0 | Metal | 167 |
| Qwen3.5 0.8B | Q4_0 | Metal | 110 |
| Qwen3.5 9B | Q4_0 | Metal | 34.5 |
| Gemma3 27B | QAT 4-bit | Metal | 11.6 |
| Gemma3 27B | QAT 4-bit | CPU | 3.2 |

### NVIDIA GB10 (Blackwell, UMA)

| Model | Quant | Backend | tok/s |
|-------|-------|---------|-------|
| Gemma3 1B | Q4_0 | CUDA | 40 |
| Gemma3 1B | Q4_0 | CPU | 5.7 |
