# Agave vs llama.cpp — Performance Benchmarks

**Date**: 2026-03-24
**Hardware**: Apple M4 Pro (14-core CPU, 20-core GPU), 48 GB unified memory
**OS**: macOS 26.3.1 (aarch64)
**llama.cpp**: latest (commit ~March 2026), Metal enabled, GGML_CPU_REPACK=OFF
**Agave**: commit 7e3314a + KV cache fix

## Decode Throughput (tok/s, higher is better)

Single-token autoregressive generation (batch=1). This is the primary metric for interactive chat.

| Model | Quant | Size | llama.cpp Metal | Agave Metal | Agave CPU | Ratio (Metal) |
|-------|-------|------|---------------:|------------:|----------:|--------------:|
| Qwen3.5 0.8B | Q8_0 | 764 MB | 140.4 | 183.3 | 57.3 | **1.31x** |
| Qwen3.5 9B | Q8_0 | 8.9 GB | 25.0 | 41.7 | 11.3 | **1.67x** |
| Qwen3.5 9B | Q4_K_M | 5.2 GB | 36.4 | — | 10.1 | — |
| Gemma 3 4B | MLX-Q4 | 2.8 GB | — | 78.1 | 23.9 | — |
| Gemma 3 12B | Q8_0 | 11.6 GB | 18.7 | 22.3 | 6.3 | **1.19x** |

### Notes

- **Agave is 1.2–1.7x faster than llama.cpp on Metal** for decode on supported quant formats.
- llama.cpp Q4_K_M Metal comparison pending (Agave Metal requires small context workaround for Q4_K_M; see Known Issues).
- MLX-Q4 is a SafeTensors format unique to Agave (not comparable to llama.cpp).
- llama.cpp numbers include Metal GPU offload; Agave Metal uses native MSL kernels.
- CPU numbers use all 14 threads (Agave) vs 10 threads (llama.cpp default).

## Prefill Throughput

Agave uses batched GEMM + fused FlashAttention-2 for Gemma 3 prefill. Other models (hybrid SSM/MoE architectures) use sequential `forward()` which is GPU-accelerated but not batched.

| Model | Quant | Prompt | llama.cpp | Agave Sequential | Agave Batched | Speedup |
|-------|-------|--------|----------:|------------------:|--------------:|--------:|
| Gemma 3 12B | Q8_0 | 58 tok | — | 14.9 tok/s | 21.9 tok/s | **1.47×** |
| Gemma 3 12B | Q8_0 | 208 tok | 280 tok/s | 12.4 tok/s | 20.6 tok/s | **1.65×** |
| Gemma 3 1B | Q4_0 | 208 tok (CUDA GB10) | — | — | 44.7 tok/s | **1.19×** |

The batched prefill speedup comes from:
- **GEMM weight reuse**: each weight row loaded once from memory, multiplied against all N input tokens (N× bandwidth savings)
- **GPU kernels**: native Metal GEMM (f32/Q8_0/Q4_0), batched RoPE, FlashAttention-2 with causal masking
- **Zero per-layer flush**: entire layer runs in one GPU command buffer

CLI: `--prefill-batch-size <N>` (default 512). Use `--prefill-batch-size 1` for sequential fallback.

## Supported Quantization Formats

| Format | Agave CPU | Agave Metal | llama.cpp |
|--------|:---------:|:-----------:|:---------:|
| Q8_0 | ✅ | ✅ | ✅ |
| Q4_0 | ✅ | ✅ | ✅ |
| Q4_K_M | ✅ | ✅ | ✅ |
| Q5_K | ✅ | ✅ | ✅ |
| Q6_K | ✅ | ✅ | ✅ |
| bf16 | ✅ | ✅ | ✅ |
| f16 | ✅ | ✅ | ✅ |
| MLX-Q4 | ✅ | ✅ | ❌ |
| NVFP4 (GGUF) | ✅ | ❌ | ✅ |
| NVFP4 (SafeTensors) | ✅ | ✅ | — |
| MXFP4 | ✅ | ✅ | ✅ |
| IQ4_XS/NL | ✅ | ✅ | ✅ |

## Supported Model Architectures

| Architecture | Models | Status |
|-------------|--------|--------|
| Gemma 3 | 4B, 12B, 27B | ✅ Working |
| Qwen3.5 | 0.8B, 9B, 27B | ✅ Working |
| GLM-4 | 4.7B | ⚠️ Poor output quality |
| Nemotron-Nano | 30B | ⚠️ Poor output quality |
| GPT-OSS | 20B | ⚠️ Poor output quality |

## Known Issues

1. **Metal large-context hang**: With default context sizes (2048–4096) and many layers, the PagedKV block pre-allocation is slow. Workaround: use `--ctx-size 128` for benchmarks. Does not affect CPU backend.
2. **Batched prefill gap vs llama.cpp**: Agave's batched prefill achieves 1.5–1.7× over sequential but is still slower than llama.cpp's fully-fused prefill for long prompts. The remaining gap is in GEMM compute density (Agave uses one threadgroup per output row; llama.cpp uses tiled 2D GEMM with shared memory).
3. **Q4_K_M Metal**: Works but requires small context sizes due to the allocation issue above.

## Methodology

- **Decode throughput**: Measured from the stats line output by the engine after generating N tokens with greedy sampling (temperature=0). Prompt: "Hello" with model-appropriate chat template.
- **llama.cpp**: `llama-bench -p 16 -n 32/128 -r 1` with Metal enabled.
- **Agave**: `agave model -n N --backend {cpu,metal} "Hello"` — tok/s from stats output.
- All runs are single-pass (no averaging), cold-start (model loaded from disk each run).
- Memory pressure from other processes was minimal during benchmarks.
