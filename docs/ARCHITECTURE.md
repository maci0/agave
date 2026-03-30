# Architecture

Project structure, module reference, and inference pipeline for the Agave inference engine.

For conceptual background, see the [tutorial series](tutorial/README.md).

## Quick Start

```bash
zig build                                          # Build (ReleaseFast + Debug)
./zig-out/bin/agave model.gguf                     # Interactive REPL
./zig-out/bin/agave model.gguf "What is 2+2?"      # Single prompt
./zig-out/bin/agave model.gguf --serve              # HTTP server (OpenAI + Anthropic API)
./zig-out/bin/agave model.gguf -q "Hello" > out.txt # Quiet mode (pipe-friendly)
./zig-out/bin/agave model.gguf --backend cpu        # Force CPU backend
```

`zig build` produces two binaries:
- `zig-out/bin/agave` — ReleaseFast (optimized, ~2.0 MB)
- `zig-out/bin/agave-debug` — Debug (safety checks, leak detection, ~5.2 MB)

## Project Structure

```
agave/
├── build.zig              # Build config (ReleaseFast default + Debug)
├── build.zig.zon          # Dependencies (clap CLI parser, vaxis terminal UI)
├── src/
│   ├── main.zig           # CLI: arg parsing, format detection, model init, REPL, recipe application
│   ├── arch.zig           # Architecture enum, detection, chat template mapping
│   ├── server/
│   │   ├── server.zig     # HTTP server (OpenAI + Anthropic API + chat UI)
│   │   ├── scheduler.zig  # Continuous batching request scheduler
│   │   ├── metrics.zig    # Prometheus metrics collector
│   │   └── rate_limiter.zig # Token bucket rate limiter
│   ├── display.zig        # Rich CLI output (banner, stats, progress)
│   ├── chat_template.zig  # Data-driven chat prompt templates (ChatML, Gemma, Qwen35, GLM-4, GPT-OSS)
│   ├── recipe.zig         # Optional preset configs per model/hardware/quant combo
│   ├── thread_pool.zig    # Futex-based work-stealing thread pool
│   ├── perf.zig           # Performance timer utilities
│   ├── readline.zig       # Line editor for interactive REPL
│   ├── micro_bench.zig    # Standalone micro-benchmark binary
│   ├── format/
│   │   ├── format.zig     # Format interface (getTensor, getMetaStr, ...)
│   │   ├── gguf.zig       # GGUF v2/v3 parser with mmap
│   │   └── safetensors.zig# Multi-shard SafeTensors loader with config.json
│   ├── models/
│   │   ├── model.zig      # Model interface (forward, prefill, resetCache, cancel)
│   │   ├── gemma3.zig     # Gemma 3 (GQA, GELU, post-norms)
│   │   ├── qwen35.zig     # Qwen 3.5 (hybrid DeltaNet SSM + attention)
│   │   ├── gpt_oss.zig    # GPT-OSS (MoE, sliding window, attention sinks)
│   │   ├── nemotron_h.zig # Nemotron-H (Mamba-2 + attention hybrid)
│   │   ├── glm4.zig       # GLM-4 MoE Lite (MLA + MoE, MLX 4/6-bit)
│   │   └── nemotron_nano.zig # Nemotron Nano (SSM + MoE + attention, NVFP4)
│   ├── ops/
│   │   ├── attention.zig  # Shared SDPA kernel (SIMD, sliding window, backend dispatch)
│   │   ├── math.zig       # argmax, softplus, sigmoid, GELU, sampleToken
│   │   ├── ssm.zig        # SSM ops: causal conv1d, Mamba-2 recurrence, group norm+gate
│   │   ├── quant.zig      # Quantization helpers (bf16, mxfp4, fp8, iq4nl, nvfp4_st)
│   │   ├── kv_quant.zig   # KV cache quantization (f32/f16/q8_0/int8/fp8/nvfp4)
│   │   └── mlx.zig        # MLX 4/6-bit dequant (unpackU4/U6, mlxGemvRaw, mlxEmbLookup)
│   ├── backend/
│   │   ├── backend.zig    # Backend interface (gemv, rmsNorm, softmax, ...)
│   │   ├── cpu.zig        # CPU: V8 SIMD, 4-row GEMV, precomputed RoPE
│   │   ├── metal.zig      # Metal: MSL kernels, simd_sum reduction, buffer cache
│   │   ├── vulkan.zig     # Vulkan: SPIR-V shaders, subgroup reductions, buffer cache
│   │   ├── cuda.zig       # CUDA: PTX kernels from Zig, deferred execution, Driver API
│   │   ├── rocm.zig       # ROCm: HIP Runtime API, HSACO kernels, deferred execution
│   │   ├── objc.zig       # Objective-C runtime bridge for Metal API
│   │   └── kernels/       # GPU kernel source files
│   │       ├── metal/     # MSL compute shaders
│   │       ├── vulkan/    # GLSL compute shaders → compiled SPIR-V (.spv)
│   │       ├── cuda/      # Zig kernels compiled to PTX via nvptx64-cuda target
│   │       └── rocm/      # Zig kernels compiled to HSACO via amdgcn-amdhsa target
│   ├── kvcache/
│   │   ├── manager.zig    # KV cache alloc/free, PagedKvCache, RadixTree
│   │   ├── block_allocator.zig # Block allocation for paged KV cache
│   │   ├── tiered.zig     # Tiered KV cache (VRAM + RAM + SSD)
│   │   └── prefetch.zig   # Async block prefetching for tiered cache
│   └── tokenizer/
│       ├── tokenizer.zig  # Tokenizer interface
│       └── bpe.zig        # BPE + SPM tokenizer with byte-level encoding
```

## The Inference Pipeline

When you run `agave model.gguf "Hello"`:

```
1. LOAD        model.gguf → mmap → Format interface
2. DETECT      "general.architecture" = "gemma3" → Gemma3Model
3. BACKEND     macOS → Metal GPU (auto), --backend cpu → CPU fallback
4. RECIPE      Match arch + backend + quant → apply proven defaults
5. TEMPLATE    arch → ChatTemplate → format prompt with role markers
6. TOKENIZE    formatted prompt → [BOS, 15496, ...] (BPE/SPM encode)
7. PREFILL     model.prefill(prompt_tokens) → fills KV cache (batched)
8. GENERATE    Loop: next = model.forward(last) → sample/argmax → decode → print
9. STATS       "5 tok, 10.4 tok/s, prefill 200ms, gen 480ms"
```

## Module Reference

### Format (`src/format/`)

| Method | Description |
|--------|-------------|
| `getTensor(name)` | Look up tensor by name → `{data_ptr, dtype, dims}` |
| `getMetaStr(key)` | String metadata (architecture name, model name) |
| `getMetaU32(key)` | Integer metadata (num_layers, hidden_size) |
| `getMetaF32(key)` | Float metadata (rope_theta, rms_norm_eps) |
| `getVocab()` | Tokenizer vocabulary array |
| `getMerges()` | BPE merge rules array |
| `layerTensor(li, suffix)` | Shorthand for `getTensor("blk.{li}.{suffix}")` |

### Backend (`src/backend/`)

| Operation | Description | Hot path? |
|-----------|-------------|-----------|
| `gemv(x, W, y, n, k)` | y = W @ x with dequantization | Yes (95% of time) |
| `rmsNorm(in, w, out, n, eps)` | RMS normalization | Yes |
| `sdpa(q, keys, vals, ...)` | Scaled dot-product attention | Yes |
| `softmax(data, n)` | In-place softmax | Yes |
| `rope(x, pos, nh, hd, rd, θ)` | Rotary position encoding | Yes |
| `silu(in, out, n)` | SiLU activation | Yes |
| `gelu(in, out, n)` | GELU activation | Yes |
| `add(a, b, out, n)` | Element-wise add | Yes |
| `mul(a, b, out, n)` | Element-wise multiply | Yes |
| `l2Norm(x, n, eps)` | L2 normalization (DeltaNet) | Yes |
| `addRmsNorm(a, b, w, out, n, eps)` | Fused add + RMS norm | Yes |
| `siluMul(a, b, out, n)` | Fused SiLU(a) × b gate | Yes |
| `sigmoidMul(data, gate, n)` | In-place data × sigmoid(gate) | Yes |
| `deltaNet(...)` | DeltaNet SSM recurrence | Yes |
| `embLookup(table, id, out, d)` | Embedding with dequant | Once per token |
| `sync()` | Flush GPU work | At sync points |

### Chat Templates (`src/chat_template.zig`)

| Preset | Models | EOG Tokens | Notes |
|--------|--------|------------|-------|
| `chatml` | Nemotron-H, Nemotron-Nano | `<\|im_end\|>`, `<\|endoftext\|>` | Standard ChatML |
| `qwen35` | Qwen 3.5 | `<\|im_end\|>`, `<\|endoftext\|>` | ChatML + `<think>\n\n</think>\n\n` generation prefix (disables reasoning) |
| `gemma` | Gemma 3, Gemma 2 | `<end_of_turn>`, `<eos>` | |
| `glm4` | GLM-4 | `<\|endoftext\|>`, `<\|user\|>` | `[gMASK]<sop>` prefix, `</think>` generation prefix |
| `gpt_oss` | GPT-OSS Harmony | `<\|end\|>`, `<\|endoftext\|>` | Includes default system prompt + developer role override |

### Recipes (`src/recipe.zig`)

| Recipe | Arch | Backend | Key Defaults |
|--------|------|---------|--------------|
| Qwen3.5 Q4 Metal | qwen3* | Metal | temp=0.6, top_p=0.9, repeat=1.1 |
| Gemma Q4 Metal | gemma* | Metal | temp=0.7, top_p=0.95 |
| GPT-OSS Metal | gpt* | Metal | temp=0.5, ctx=2048 |
| CPU generic | any | CPU | max_tokens=256, ctx=2048 |

User CLI flags always override recipe defaults.

### Shared Ops (`src/ops/`)

| Function | File | Description |
|----------|------|-------------|
| `scaledDotProductAttention` | attention.zig | Full SDPA with KV cache, GQA, sliding window |
| `sampleToken` | math.zig | Temperature + top-k + top-p nucleus sampling |
| `causalConv1dSilu` | ssm.zig | Causal conv1d with ring buffer + SiLU |
| `mamba2Recurrence` | ssm.zig | Mamba-2 per-head state update + output |
| `groupRmsNormSiluGate` | ssm.zig | Group RMS norm followed by SiLU gate |
| `finalLogits` | math.zig | RMSNorm + GEMV + argmax (forward tail) |
| `expertWeightStride` | model.zig | Byte stride between experts in packed weights |

### Quantization Types

| DType | Bits/val | Block | Models |
|-------|----------|-------|--------|
| `f32` | 32 | 1 | Reference |
| `f16` | 16 | 1 | Embeddings |
| `bf16` | 16 | 1 | Gemma3, Nemotron SSM layers |
| `q8_0` | 8.5 | 32 | General |
| `q6_k` | 6.6 | 256 | General |
| `q5_k` | 5.5 | 256 | General |
| `q4_k` | 4.8 | 256 | General |
| `q5_0` | 5.5 | 32 | Nemotron-H |
| `q4_0` | 4.5 | 32 | General |
| `q4_1` | 5.0 | 32 | General |
| `q3_k` | 3.4 | 256 | Compact |
| `q2_k` | 2.6 | 256 | Ultra-compact |
| `iq4_nl` | 4.5 | 32 | CPU-optimized (lookup table) |
| `iq4_xs` | 4.3 | 256 | CPU-optimized (super-block) |
| `fp8_e4m3` | 8 | 1 | KV cache, weights |
| `fp8_e5m2` | 8 | 1 | KV cache |
| `nvfp4` | 4.25 | 16 | Blackwell+ (GGUF) |
| `mxfp4` | 4.25 | 32 | Microscaled FP4 |
| `tq1_0` | 1.7 | 256 | Ternary quantization |
| `mlx_q` | 4-6 | 64 | MLX models (affine: scale × uint + bias) |
