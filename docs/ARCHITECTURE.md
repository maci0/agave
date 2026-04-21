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
./zig-out/bin/agave model.gguf --megakernel "Hi"    # Fused FFN megakernel (Metal/CUDA)
```

`zig build` produces three binaries:
- `zig-out/bin/agave` — ReleaseFast (optimized, ~3.4 MB)
- `zig-out/bin/agave-debug` — Debug (safety checks, leak detection, ~9.4 MB)
- `zig-out/bin/agave-bench` — ReleaseFast micro-benchmark tool (`src/micro_bench.zig`)

## Project Structure

```
agave/
├── build.zig              # Build config (ReleaseFast default + Debug)
├── build.zig.zon          # Package metadata (zero external dependencies)
├── src/
│   ├── main.zig           # CLI: arg parsing, format detection, model init, REPL, recipe application
│   ├── cli.zig            # Self-contained CLI argument parser (zero deps, replaces clap)
│   ├── arch.zig           # Architecture enum, detection, chat template mapping
│   ├── pull.zig           # Model download from HuggingFace Hub (agave pull <org/repo>)
│   ├── server/
│   │   ├── server.zig     # HTTP server (OpenAI + Anthropic API + chat UI)
│   │   ├── scheduler.zig  # Continuous batching request scheduler
│   │   ├── metrics.zig    # Prometheus metrics collector
│   │   ├── rate_limiter.zig # Token bucket rate limiter
│   │   └── json.zig        # Streaming JSON field extraction
│   ├── display.zig        # Rich CLI output (banner, stats, progress)
│   ├── chat_template.zig  # Data-driven chat prompt templates (ChatML, Gemma, Gemma 4, Qwen35, GLM-4, GPT-OSS)
│   ├── recipe.zig         # Optional preset configs per model/hardware/quant combo
│   ├── thread_pool.zig    # Futex-based work-stealing thread pool
│   ├── perf.zig           # Performance timer utilities
│   ├── readline.zig       # Line editor for interactive REPL
│   ├── term.zig           # Terminal I/O: key parser, ANSI sequences, display width (pure Zig, no libc)
│   ├── image.zig          # PNG/PPM image decoder and resize for multimodal inference
│   ├── micro_bench.zig    # Standalone micro-benchmark binary
│   ├── format/
│   │   ├── format.zig     # Format interface (getTensor, getMetaStr, ...)
│   │   ├── gguf.zig       # GGUF v2/v3 parser with mmap
│   │   └── safetensors.zig# Multi-shard SafeTensors loader with config.json
│   ├── models/
│   │   ├── model.zig      # Model interface (forward, prefill, resetCache, cancel)
│   │   ├── gemma3.zig     # Gemma 3 (GQA, GELU, post-norms)
│   │   ├── gemma4.zig     # Gemma 4 (dual attention, MoE/dense variants, PLE)
│   │   ├── qwen35.zig     # Qwen 3.5 (hybrid DeltaNet SSM + attention)
│   │   ├── gpt_oss.zig    # GPT-OSS (MoE, sliding window, attention sinks)
│   │   ├── nemotron_h.zig # Nemotron-H (Mamba-2 + attention hybrid)
│   │   ├── glm4.zig       # GLM-4 MoE Lite (MLA (DeepSeek-V2) + MoE, MLX 4/6/8-bit)
│   │   ├── nemotron_nano.zig # Nemotron Nano (SSM + MoE + attention, NVFP4)
│   │   └── vision.zig       # SigLIP-2 vision encoder (Gemma 4 multimodal)
│   ├── ops/
│   │   ├── attention.zig  # Shared SDPA kernel (SIMD, sliding window, backend dispatch)
│   │   ├── math.zig       # argmax, softplus, sigmoid, GELU, sampleToken
│   │   ├── ssm.zig        # SSM ops: causal conv1d, Mamba-2 recurrence, group norm+gate
│   │   ├── quant.zig      # Quantization helpers (bf16, mxfp4, fp8, iq4nl, nvfp4_st)
│   │   ├── kv_quant.zig   # KV cache quantization (f32/f16/q8_0/int8/fp8/nvfp4)
│   │   ├── mlx.zig        # MLX 4/6/8-bit dequant (mlxGemvRaw, mlxGemvRows, mlxEmbLookup)
│   │   └── split_attention.zig # Split-attention: async CPU-GPU KV cache offloading
│   ├── backend/
│   │   ├── backend.zig    # Backend interface (gemv, rmsNorm, softmax, ...)
│   │   ├── cpu.zig        # CPU: V8 SIMD, 4-row GEMV, precomputed RoPE
│   │   ├── metal.zig      # Metal: MSL kernels, simd_sum reduction, buffer cache
│   │   ├── vulkan.zig     # Vulkan: SPIR-V shaders, subgroup reductions, buffer cache
│   │   ├── cuda.zig       # CUDA: PTX kernels from Zig, deferred execution, Driver API
│   │   ├── rocm.zig       # ROCm: HIP Runtime API, HSACO kernels, deferred execution
│   │   ├── megakernel.zig # Weight offset computation for fused FFN megakernels
│   │   ├── mega_compose.zig # Composable megakernel generator (ModelDesc → MSL at runtime)
│   │   ├── objc.zig       # Objective-C runtime bridge for Metal API
│   │   └── kernels/       # GPU kernel source files
│   │       ├── metal/     # MSL compute shaders (incl. megakernel.metal, mega_common.metal, mega_*.metal)
│   │       ├── vulkan/    # GLSL compute shaders → compiled SPIR-V (.spv)
│   │       ├── cuda/      # Zig kernels compiled to PTX (incl. fused_ffn_q8_0.zig, mega_*.zig)
│   │       └── rocm/      # Zig kernels compiled to HSACO via amdgcn-amdhsa target (incl. mega_*.zig)
│   ├── kvcache/
│   │   ├── manager.zig    # KV cache alloc/free, PagedKvCache, RadixTree
│   │   ├── block_allocator.zig # Block allocation for paged KV cache
│   │   ├── tiered.zig     # Tiered KV cache (VRAM + RAM + SSD)
│   │   └── prefetch.zig   # Async block prefetching for tiered cache
│   ├── web/
│   │   ├── app.js         # Chat UI JavaScript (SSE streaming, conversation management)
│   │   ├── body.html      # Chat UI HTML body
│   │   ├── head.html      # Chat UI HTML head (meta, styles)
│   │   └── style.css      # Chat UI stylesheet
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
| **Core** | | |
| `gemv(x, W, y, n, k)` | y = W @ x with dequantization | Yes (95% of time) |
| `gemm(x, W, y, n_tok, n_out, n_in)` | Batched matrix multiply (prefill, BF16 on Metal) | Prefill only |
| `rmsNorm(in, w, out, n, eps)` | RMS normalization | Yes |
| `sdpa(q, keys, vals, ...)` | Scaled dot-product attention | Yes |
| `softmax(data, n)` | In-place softmax | Yes |
| `rope(x, pos, nh, hd, rd, θ)` | Rotary position encoding | Yes |
| `silu(in, out, n)` | SiLU activation | Yes |
| `gelu(in, out, n)` | GELU activation | Yes |
| `add(a, b, out, n)` | Element-wise add | Yes |
| `mul(a, b, out, n)` | Element-wise multiply | Yes |
| `l2Norm(x, n, eps)` | L2 normalization (DeltaNet) | Yes |
| `embLookup(table, id, out, d)` | Embedding with dequant | Once per token |
| **Fused** | | |
| `addRmsNorm(a, b, w, out, n, eps)` | Fused add + RMS norm | Yes |
| `siluMul(a, b, out, n)` | Fused SiLU(a) × b (SwiGLU gate) | Yes |
| `geluMul(a, b, out, n)` | Fused GELU(a) × b (Gemma FFN) | Yes |
| `sigmoidMul(data, gate, n)` | In-place data × sigmoid(gate) | Yes |
| `addScaled(src, dst, scale, n)` | dst += src × scale (MoE accumulation) | Yes |
| **Batched (prefill)** | | |
| `rmsNormBatched(in, w, out, n_tok, dim, eps)` | Per-row RMS norm for n_tok rows | Prefill only |
| `ropeBatched(x, positions, n_tok, ...)` | RoPE for n_tok vectors | Prefill only |
| `sdpaPrefill(q, k, v, ...)` | Causal self-attention for n_tok tokens | Prefill only |
| `gemvMulti(x, ops, k)` | Batched GEMV dispatch (fused kernel launch) | Yes |
| **Specialized** | | |
| `gemvT(x, W, y, out_dim, in_dim)` | Transposed GEMV for Q8_0 3D weights (MLA) | Yes |
| `gemvNvfp4St(x, w, scale, y, n, k)` | NVFP4 SafeTensors GEMV (separate scale tensor) | Yes |
| `gemvMlxQ(x, w, scales, biases, y, n, k, bits)` | MLX affine quantized GEMV (4/6/8-bit) | Yes |
| `gemvMxfp4St(x, w, scale, y, n, k)` | MXFP4 SafeTensors GEMV | Yes |
| `rmsNormMulti(data, w, n_heads, hd, eps)` | Per-head RMS norm (QK norm) | Yes |
| `deinterleave(in, out_a, out_b, stride, n)` | Split interleaved Q/K pairs | Yes |
| `splitQGate(qg, q, g, hd, nh)` | Split concatenated Q+gate (Qwen3.5) | Yes |
| `deltaNet(...)` | DeltaNet SSM recurrence | Yes |
| `sdpaWithStats(q, keys, vals, ..., max, sum)` | SDPA returning softmax stats (split-attention) | Yes |
| **Infrastructure** | | |
| `sync()` | Flush GPU work | At sync points |
| `beginBatch()` / `endBatch()` | Suppress/restore GPU memory barriers | GPU only |
| `backendInfo()` | Device name, VRAM, library version | Init only |

### Chat Templates (`src/chat_template.zig`)

| Preset | Models | EOG Tokens | Notes |
|--------|--------|------------|-------|
| `chatml` | Nemotron-H, Nemotron-Nano | `<\|im_end\|>`, `<\|endoftext\|>` | Standard ChatML |
| `qwen35` | Qwen 3.5 | `<\|im_end\|>`, `<\|endoftext\|>` | ChatML + `<think>\n\n</think>\n\n` generation prefix (disables reasoning) |
| `gemma` | Gemma 3, Gemma 2 | `<end_of_turn>`, `<eos>` | |
| `gemma4` | Gemma 4 | `<turn\|>`, `<eos>`, `<channel\|>`, `<\|endoftext\|>`, `<\|end\|>` | `<\|channel>0\n<channel\|>` generation prefix |
| `glm4` | GLM-4 | `<\|endoftext\|>`, `<\|user\|>`, `<\|observation\|>` | `[gMASK]<sop>` prefix, `</think>` generation prefix |
| `gpt_oss` | GPT-OSS Harmony | `<\|end\|>`, `<\|endoftext\|>` | Includes default system prompt + developer role override |

### Recipes (`src/recipe.zig`)

| Recipe | Arch | Backend | Key Defaults |
|--------|------|---------|--------------|
| Qwen3.5 Q4 Metal | qwen3* | Metal | temp=0.6, top_p=0.9, repeat=1.1 |
| Gemma Q4 Metal | gemma* | Metal | temp=0.7, top_p=0.95 |
| GPT-OSS Metal | gpt* | Metal | temp=0.5, ctx=2048 |
| GLM-4 generic | glm4* | any | temp=0.7, repeat=1.1 |
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
| `fp8_e5m2` | 8 | 1 | Weights only |
| `nvfp4` | 4.25 | 16 | Blackwell+ (GGUF) |
| `mxfp4` | 4.25 | 32 | Microscaled FP4 |
| `tq1_0` | 1.7 | 256 | Ternary quantization (parsed but GEMV unsupported — output zeroed) |
| `mlx_q` | 4-8 | 64 | MLX models (affine: scale × uint + bias) |

**KV Cache Quantization Types** (see `src/ops/kv_quant.zig`):

| KvQuantType | Bits/val | Notes |
|-------------|----------|-------|
| `f32` | 32 | Full precision |
| `f16` | 16 | Half precision |
| `q8_0` | 8.5 | Block-quantized |
| `int8` | 8 | Symmetric INT8 |
| `fp8_e4m3` | 8 | FP8 E4M3 |
| `nvfp4` | 4.25 | NVFP4 microscaled |
| `turbo2` | 2.5 | TurboQuant 2-bit (WHT-32 + Lloyd-Max codebook) |
| `turbo3` | 3.5 | TurboQuant 3-bit (WHT-32 + Lloyd-Max codebook) |
| `turbo4` | 4.5 | TurboQuant 4-bit (WHT-32 + Lloyd-Max codebook) |

**TurboQuant+ features:**
- **`turbo` preset** (`--kv-type turbo`): asymmetric K=q8\_0, V=turbo4. Higher K precision protects attention routing accuracy while V compression (3.8x) is nearly free.
- **Boundary V protection**: the turbo preset automatically keeps first/last 2 layers at f16 V to preserve input embedding fidelity and final output quality.
- **Sparse V dequantization**: SDPA kernels dequantize V blocks on-the-fly inside the attention loop rather than pre-expanding, saving memory bandwidth.
- **Native GPU SDPA kernels**: TurboQuant dequantization is fused into SDPA kernels on all GPU backends (Metal, CUDA, Vulkan, ROCm).

**Tiered KV cache and split-attention** (`src/ops/split_attention.zig`):
- Enabled via `--kv-tiers vram+ram`. KV cache blocks are partitioned by tier (GPU-resident VRAM vs CPU-resident RAM).
- **Fast path**: when all blocks are on GPU, dispatches a single `be.sdpa()` with zero overhead.
- **Mixed path**: GPU SDPA with softmax statistics runs concurrently with CPU SDPA on the thread pool, then partial outputs are merged via [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691) online softmax correction (exact, no approximation).
- **CPU-only path**: falls back to CPU SDPA on the thread pool when all blocks have been offloaded.

### KV Cache Eviction

When context grows beyond `--kv-budget`, eviction compresses the cache in-place to stay within budget. Two policies are available:

| Policy | Flag | Calibration | Description |
|--------|------|:-----------:|-------------|
| **Norm** | `--kv-eviction norm` | No | Evicts entries with the smallest K vector L2 norm |
| **Tri** | `--kv-eviction tri` | Yes (`.cal` file) | Trigonometric frequency-domain scoring from [TriAttention (Mao et al., 2025)](https://arxiv.org/abs/2604.04921) |

**Shared behavior:**
- **Attention sink preservation**: the first 4 positions are never evicted (they accumulate disproportionate attention mass in causal models).
- **Recent window**: the most recent positions are always retained regardless of score.
- **Periodic compression**: eviction runs every 128 tokens once the cache exceeds `--kv-budget`.

**Calibration (`agave calibrate`):** The `tri` policy requires per-head Q/K frequency statistics stored in a `.cal` file alongside the model. Run `agave calibrate model.gguf` to generate this data. The calibration pass processes a representative prompt and records the dominant frequency components per attention head.

**Stacking with TurboQuant:** Eviction reduces the *number* of KV entries while TurboQuant reduces the *bits per entry*. Combined, they can achieve ~40x KV memory reduction vs f16 baseline.

## Vision / Multimodal

Vision support is implemented in `src/models/vision.zig` with three auto-detected encoder variants loaded from mmproj GGUF files. The encoder variant is detected at init by probing available tensor names.

### Encoder Architectures

**Gemma 4 [SigLIP-2 (Tschannen et al., 2025)](https://arxiv.org/abs/2502.14786)** (`gemma4_siglip2`):
- 768x768 input, 16x16 patches -> 2304 patches, 3x3 average pooling -> 256 output tokens.
- Conv2D patch embedding (no bias), learned 2D position encoding `[embd_dim, max_pos, 2]`.
- ViT blocks with per-head QK RMSNorm, post-attention/FFN RMSNorm, SwiGLU FFN.
- Input standardization (`scale * x + bias`, replaces CLIP mean/std normalization).
- Single linear projection (`mm.input_projection.weight`) to LLM hidden dimension.

**Gemma 3 [SigLIP (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343)** (`gemma3_siglip`):
- 896x896 input, 14x14 patches -> 4096 patches, no spatial merge.
- Conv2D patch embedding (with bias), learned 1D position embedding `[embd_dim, n_patches]`.
- ViT blocks with LayerNorm (with bias), GELU FFN (up+down, no gate), no QK norms.
- Post-encoder LayerNorm (`v.post_ln`), then `mm.soft_emb_norm` + `mm.input_projection`.

**Qwen VL** (`qwen_vl`):
- Dual Conv2D patch embedding (with bias), learned 1D position embedding.
- ViT blocks with fused QKV projection, LayerNorm (with bias), GELU FFN, no QK norms.
- Post-encoder LayerNorm (`v.post_ln`), then 4x MLP merge projector (`mm.0` + GELU + `mm.2`) -> n\_patches/4 output tokens.
- M-RoPE (multi-dimensional rotary position embedding): 4 sections `[temporal, height, height, width]` with theta=10000.

### Vision Pipeline

```
1. PREPROCESS    decode image -> resize to encoder input size -> normalize pixels
2. PATCH EMBED   Conv2D: [H, W, 3] -> [n_patches, embd_dim]
3. POS EMBED     add learned position embeddings (1D or 2D depending on variant)
4. VIT BLOCKS    N transformer blocks: LayerNorm/RMSNorm -> SDPA -> FFN -> residuals
5. POOL          spatial merge (Gemma 4: 3x3 avg pool, Qwen VL: 4x MLP, Gemma 3: none)
6. STANDARDIZE   input standardization (Gemma 4 only: scale * x + bias)
7. PROJECT       linear projection to LLM hidden dimension
8. NORMALIZE     RMSNorm / soft_emb_norm on projected tokens
```

### LLM Integration

Vision tokens are injected into the LLM via `forwardImageBatch` (Gemma 4). Image embeddings replace placeholder tokens in the input sequence, then the model runs a dedicated forward pass over the image batch using **non-causal (bidirectional) attention** -- image tokens attend to all other image tokens without the causal mask. After this pass, the KV cache contains the image context and subsequent text tokens attend to it normally via causal attention.
