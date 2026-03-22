# Agave — Documentation

**A high-performance LLM inference engine written in Zig.**

Agave loads pre-trained language models, tokenizes input, runs the neural network forward pass, and generates text — all from scratch in Zig with zero external ML libraries.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Concepts](#2-concepts)
   - [Tokens and Tokenization](#tokens-and-tokenization) · [The Forward Pass](#the-forward-pass) · [Attention](#attention) · [SSM](#ssm-state-space-model) (DeltaNet, Mamba-2, Hybrid Patterns)
   - [RoPE](#rope-rotary-position-encoding) · [Quantization](#quantization) · [GEMV](#gemv-general-matrix-vector-multiply) · [Backend](#backend) · [GPU Compute Platforms](#gpu-compute-platforms)
   - [KV Cache](#kv-cache) (Paged, Radix) · [Format](#format-gguf--safetensors) · [Dispatcher Pattern](#dispatcher-pattern) · [Kernels](#kernels)
   - [SDPA](#sdpa-scaled-dot-product-attention) (QK Norm, Sliding Window, Sinks, Sigmoid Gate, Softcapping)
   - [Causal Convolution](#causal-convolution) · [RMS Normalization](#rms-normalization) · [SwiGLU and Activations](#swiglu-and-activation-functions)
   - [MoE](#moe-mixture-of-experts) · [Embedding](#embedding-and-vocabulary-projection) · [Sampling](#decoding-and-sampling-parameters)
3. [Architecture Overview](#3-architecture-overview)
4. [The Inference Pipeline](#4-the-inference-pipeline)
5. [Module Reference](#5-module-reference)
6. [Supported Models](#6-supported-models)
7. [Performance](#7-performance)

---

## 1. Quick Start

```bash
zig build                                          # Build (ReleaseFast + Debug)
./zig-out/bin/agave model.gguf                     # Interactive REPL
./zig-out/bin/agave model.gguf "What is 2+2?"      # Single prompt
./zig-out/bin/agave model.gguf --serve              # HTTP server (OpenAI API)
./zig-out/bin/agave model.gguf -q "Hello" > out.txt # Quiet mode (pipe-friendly)
./zig-out/bin/agave model.gguf --backend cpu        # Force CPU backend
./zig-out/bin/agave models/mlx-community/gemma-3-4b-it-qat-4bit  # SafeTensors directory
```

`zig build` produces two binaries:
- `zig-out/bin/agave` — ReleaseFast (optimized, ~1.7 MB)
- `zig-out/bin/agave-debug` — Debug (safety checks, leak detection, ~4.6 MB)

---

## 2. Concepts

This section explains the core ideas behind LLM inference. If you already know how transformers work, skip to [Architecture Overview](#3-architecture-overview).

### What is inference?

**Training** teaches a model by adjusting billions of weights over trillions of tokens. **Inference** uses those frozen weights to generate new text. Agave only does inference — it loads pre-trained weights and runs the model forward.

### Tokens and Tokenization

Language models don't see text — they see **tokens**, which are integer IDs representing subword pieces. The tokenizer converts between text and tokens:

```
"Hello, world!" → [15496, 11, 995, 0]     (encode)
[15496, 11, 995, 0] → "Hello, world!"     (decode)
```

**BPE (Byte Pair Encoding)** is the most common tokenization algorithm. It works by iteratively merging the most frequent pair of adjacent symbols:

1. Start with individual bytes: `H e l l o`
2. Most frequent pair is `l l` → merge to `ll`: `H e ll o`
3. Next most frequent is `H e` → merge to `He`: `He ll o`
4. Continue until vocabulary is built: `Hello`

The merge rules are learned during training and stored alongside the model. Agave's BPE tokenizer (`src/tokenizer/bpe.zig`) supports two modes:
- **BPE mode** — uses merge rules (Qwen, GPT)
- **SPM mode** — greedy longest-match without merges (Gemma)

### The Forward Pass

The forward pass is the core computation: given a sequence of tokens, predict the next one. It flows through these stages:

```
Token ID → Embedding → N Transformer Layers → Final Norm → Logits → Argmax → Next Token
```

Each **transformer layer** has two sublayers:
1. **Attention** (or SSM) — lets the model look at previous tokens
2. **FFN** (Feed-Forward Network) — processes each position independently

Both sublayers use **residual connections** (`output = input + sublayer(input)`) so information can flow through unchanged.

### Attention

Attention answers: "which previous tokens should I pay attention to?"

For each token position, the model computes:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?" (for each previous position)
- **Value (V)**: "What information do I carry?" (for each previous position)

The attention score between positions i and j is `Q_i · K_j / sqrt(d)`. After softmax normalization, these scores weight the V vectors:

```
output = softmax(Q @ K^T / sqrt(d)) @ V
```

This is **O(n²)** in sequence length — every token attends to every previous token.

**GQA (Grouped Query Attention)** reduces memory by sharing K/V heads across multiple Q heads. With 20 Q heads and 5 KV heads, each KV head serves 4 Q heads, cutting KV cache memory by 4×. The mapping `kvh = h / (nh / nkv)` maps multiple query heads to the same KV head.

| Model | Q heads | KV heads | Ratio |
| :--- | :--- | :--- | :--- |
| Gemma3 1B | 4 | 1 | 4:1 |
| Qwen3.5 | 16 | 4 | 4:1 |
| GPT-OSS | 64 | 8 | 8:1 |
| Nemotron-H | 40 | 8 | 5:1 |

**MLA (Multi-head Latent Attention)** compresses K/V into a low-rank latent space before caching, reducing KV cache memory even further. Used by GLM-4.

### SSM (State Space Model)

SSMs are an alternative to attention that process tokens in **O(1)** time per step instead of O(n²). Instead of re-reading all previous tokens, they maintain a fixed-size **state matrix** that summarizes the past:

```
state[t] = decay * state[t-1] + input[t]    (simplified)
output[t] = state[t] @ query[t]
```

The **decay** factor controls how quickly old information fades. This is like a leaky bucket — new information flows in, old information gradually drains out.

**Mamba-2** (used by Nemotron-H) and **DeltaNet** (used by Qwen3.5) are specific SSM architectures with different state update rules. Both use **causal convolution** (a small sliding window over recent inputs) before the recurrence.

**Hybrid models** combine attention and SSM layers: attention every N layers for global context, SSM for the rest for speed. Qwen3.5 uses attention every 4th layer; Nemotron-H uses attention for ~10% of layers.

#### DeltaNet (Qwen3.5)

DeltaNet replaces quadratic softmax attention with a linear-complexity recurrence. Instead of storing all K/V pairs and computing pairwise scores, it maintains a per-head state matrix `S[v_dim, k_dim]` that accumulates information via the delta rule — error-correcting outer-product updates.

The name comes from the delta rule: the update is proportional to the *error* `(v - S^T * k)`, not just the raw value. This makes the state self-correcting — it learns to predict V from K and only stores the residual.

**Per-timestep algorithm for each V-head `h`:**

```
1. Decay: S[h] *= exp(ssm_a[h] * softplus(alpha[h] + dt_bias[h]))
   - ssm_a is negative (loaded from GGUF as -exp(A_log))
   - So decay = exp(negative * positive) < 1 — state exponentially forgets

2. Delta update:
   sk[vi] = sum_ki(S[h, vi, ki] * k[ki])    // project state onto current key
   delta[vi] = beta[h] * (v[vi] - sk[vi])    // error signal
   S[h, vi, ki] += k[ki] * delta[vi]         // outer product update

3. Output:
   out[vi] = sum_ki(S[h, vi, ki] * q[ki]) / sqrt(head_k_dim)
```

**Q-head indexing (GQA in DeltaNet):** DeltaNet uses separate head counts for Q/K and V:
- `num_v_heads` = `ssm_dt_rank` (16) — the outer loop iterates over V-heads
- `num_k_heads` = `ssm_n_group` (16) — K-heads, potentially different count
- GQA mapping: `kh = h * num_k_heads / num_v_heads` maps each V-head to its K-head
- When `num_k_heads == num_v_heads`, this simplifies to `kh = h` (1:1 mapping)
- State matrix shape per V-head: `[head_v_dim x head_k_dim]` where `head_v_dim = d_inner / num_v_heads`

This is the DeltaNet equivalent of GQA in standard attention — fewer K-heads can be shared across more V-heads to reduce parameters.

**Split order (llama.cpp convention):** After conv1d, the output is split as `[Q | K | V]`. The offsets are:
- `q_off = 0`, `k_off = num_k_heads * head_k_dim`, `v_off = 2 * num_k_heads * head_k_dim`

**Gating:** After the recurrence, output goes through per-head RMS norm, then is multiplied element-wise by `SiLU(z)` where `z` comes from a separate learned gate projection.

#### Mamba-2 (Nemotron-H)

Mamba-2 is a state-space model that learns input-dependent discretization. Unlike a fixed linear system, the `dt` (timestep) parameter is computed from the input, making the model selectively remember or forget.

**Per-head recurrence:**

```
dt_h = softplus(dt_raw[h] + dt_bias[h])     // input-dependent timestep
decay = exp(ssm_a[h] * dt_h)                // decay < 1 (ssm_a is negative)

For each state element [i, j]:
  S[h][i][j] = decay * S[h][i][j] + x[i] * dt_h * B[j]   // state update
  y[i]       = sum_j(S[h][i][j] * C[j]) + D[h] * x[i]    // output + skip
```

**Key differences from DeltaNet:**
- **B/C are input-dependent projections** (selectivity), not just normalized K/Q
- **D skip connection** adds a direct path from input to output
- **Group structure:** Heads are grouped; B and C are shared within a group (`group = h / heads_per_group`)
- **Group RMS norm** on output (not per-head): norm is applied per group of `d_inner / n_group` elements, then gated by SiLU(z)

**State dimensions:** `[num_heads x mamba_head_dim x d_state]` — each head maintains a 2D state matrix.

#### Hybrid Layer Patterns

Several models alternate layer types within a single architecture, combining the global context capacity of attention with the O(1) per-step efficiency of SSMs:

| Model | Pattern | Rule |
| :--- | :--- | :--- |
| Qwen3.5 | DeltaNet + Attention | Attention every `full_attn_interval` layers (default 4) |
| Nemotron-H | Mamba-2 + Attention | Configured by `hybrid_override_pattern` string |
| Nemotron-Nano | SSM + MoE + Attention | 52-layer pattern: `M`=SSM, `E`=MoE, `*`=Attention. 6 SSM layers at positions 4,11,18,25,32,41 |
| GPT-OSS | Sliding + Full attention | Even layers = 128-token window, odd = full sequence |

The layer-type dispatch happens in each model's `forward()` loop and is determined at init from GGUF metadata.

### RoPE (Rotary Position Encoding)

Transformers are position-agnostic by default — "A ate B" and "B ate A" look identical. RoPE encodes position by rotating Q and K vectors by an angle proportional to their position:

```
Q_rotated[i] = rotate(Q[i], position * frequency)
K_rotated[j] = rotate(K[j], position * frequency)
```

The dot product `Q_i · K_j` then depends on the *relative* distance `|i - j|`, not absolute positions. This generalizes to longer sequences than seen during training.

**Rotation formula:** For each pair of dimensions `(i, i + rope_dim/2)` in a head, compute a rotation angle proportional to the token position:

```
freq[i] = 1 / (theta ^ (2i / rope_dim))
angle   = pos * freq[i]

x'[i]              = x[i] * cos(angle) - x[i + half] * sin(angle)
x'[i + half]       = x[i] * sin(angle) + x[i + half] * cos(angle)
```

This is equivalent to multiplying by a 2x2 rotation matrix per dimension pair. Cos/sin values are precomputed once per position and reused across all heads.

**Theta values by model:** Each model family uses a different base frequency, which controls the wavelength range:

| Model | theta | Effect |
| :--- | :--- | :--- |
| Nemotron-H | 10,000 | Standard range |
| GPT-OSS | 150,000 | Extended context |
| Gemma3 | 1,000,000 | Very long context |
| Qwen3.5 | 10,000,000 | Ultra-long context |

Higher theta produces lower-frequency rotations and better long-range position discrimination.

**Partial RoPE**: Some models (Qwen3.5, Nemotron-H) only rotate a subset of dimensions, leaving the rest for non-positional features. Qwen3.5 only rotates the first `rope_dim=64` dimensions of each head, leaving the rest position-independent. This gives the model more capacity for content representation. The `rope_dim` parameter is read from GGUF metadata (`rope.dimension_count`).

### Quantization

Model weights are trained in float32 (32 bits per value) but stored compressed for inference. **Quantization** maps floating-point values to lower-precision integers:

| Format | Bits/weight | Compression | Quality |
|--------|------------|-------------|---------|
| F32 | 32 | 1× | Reference |
| BF16 | 16 | 2× | Near-lossless |
| Q8_0 | 8.5 | 3.8× | Very good |
| Q6_K | 6.6 | 4.8× | Good |
| Q4_0 | 4.5 | 7.1× | Acceptable |
**Block quantization** (Q4_0, Q8_0): Groups of 32 values share a single scale factor. Each value is stored as a small integer, dequantized on-the-fly during computation: `float_value = integer_value * scale`.

**MLX affine quantization** (4-bit or 6-bit): Each group of 64 values has a scale and bias: `float_value = scale * uint_value + bias`. Used by Gemma QAT (4-bit) and GLM-4 (6-bit) MLX models.

**Key principle**: Dequantization happens *inside* the GEMV kernel, not before it. This avoids materializing the full-precision weight matrix in memory.

#### Choosing a Quantization Format

| Use Case | Recommended Format | Rationale |
|----------|-------------------|-----------|
| **Production serving (Blackwell+)** | NVFP4, MXFP4 | Best perf/quality on latest hardware |
| **Balanced quality/speed** | bf16, Q4_K_M | Industry standard, wide support |
| **Maximum compression** | Q2_K, IQ4_XS | Smallest VRAM footprint |
| **CPU inference** | IQ4_NL, Q4_0, Q5_K | Optimized for SIMD on x86/ARM |
| **GPU with limited VRAM** | Q4_K_S, FP8 E4M3 | Good quality/size tradeoff |
| **KV cache (production)** | bf16, FP8 E5M2 | Fast decode, minimal quality loss |
| **Research/golden tests** | f32 | Reference accuracy |

**Quality hierarchy (best to most compressed):**
```
f32 > bf16 > MXFP4/NVFP4 > FP8 E4M3 > Q6_K > Q5_K > Q4_K_M > Q4_K_S > IQ4_NL > Q3_K > Q2_K
```

### GEMV (General Matrix-Vector Multiply)

GEMV is the dominant operation in inference — it accounts for ~95% of compute time. Every linear projection (Q, K, V, output, FFN gate/up/down) is a GEMV:

```
y[i] = sum_j(W[i][j] * x[j])    for each output element i
```

For a 2560×2560 weight matrix, that's 6.5 million multiply-accumulate operations per call, and a typical model does 7 GEMVs per layer × 30 layers = 210 GEMVs per token.

Agave's GEMV kernels fuse dequantization with the dot product, process multiple rows simultaneously for cache reuse, and use SIMD vectors for parallel computation.

### Backend

The **backend** is the compute engine that executes mathematical operations. Agave's backend interface (`src/backend/backend.zig`) defines ~18 primitive operations (gemv, rmsNorm, softmax, rope, silu, gelu, add, mul, embLookup, l2Norm, sdpa, sigmoidMul, siluMul, deinterleave, addRmsNorm, rmsNormMulti, gemvNvfp4St, gemvMlxQ, sync). Each backend implements these for its hardware:

- **CPU** (`cpu.zig`): SIMD-optimized with V8 vectors, 4-row cache-friendly GEMV, precomputed RoPE tables
- **Metal** (`metal.zig`): Apple GPU with MSL compute shaders, threadgroup-level parallel reduction, buffer caching. Tuning constants: `threadgroup_size=256`, `softmax_cpu_threshold=128`, `sdpa_max_seq_len=4096`, `sdpa_max_head_dim=256`
- **CUDA** (`cuda.zig`): NVIDIA GPU with PTX kernels compiled from Zig via `nvptx64-cuda` target, deferred execution model, Driver API loaded dynamically
- **Vulkan** (`vulkan.zig`): Cross-platform GPU with SPIR-V compute shaders, subgroup arithmetic reductions, buffer caching
- **ROCm** (`rocm.zig`): AMD GPU with HIP Runtime API and HSACO kernels compiled from Zig via `amdgcn-amdhsa` target, deferred execution model

The model code calls `self.be.gemv(...)` without knowing which backend is active — the `Backend` tagged union with `inline else` dispatches to the correct implementation at compile time.

### GPU Compute Platforms

Modern GPUs are massively parallel processors — thousands of simple cores executing the same instruction on different data (SIMD/SIMT). To program them, you need a **compute API** that lets you write kernels (small programs) and dispatch them to the GPU. Each hardware vendor has its own API, plus there are cross-platform alternatives:

#### CUDA (NVIDIA)

**Compute Unified Device Architecture.** NVIDIA's proprietary GPU programming platform. The dominant API for ML/AI workloads.

- **Scope**: NVIDIA GPUs only (GeForce, Quadro, Tesla, A100, H100, etc.)
- **Language**: CUDA C++ (extension of C++ with `__global__` kernel functions)
- **IR**: PTX (Parallel Thread Execution) — a virtual ISA that NVIDIA's driver JIT-compiles to the actual GPU microcode
- **Runtime**: CUDA Runtime API (high-level) or CUDA Driver API (low-level)
- **Key features**: Tensor Cores (hardware matrix multiply units), cuBLAS/cuDNN libraries, mature tooling (nsight, nvprof)
- **In Agave**: `src/backend/cuda.zig` (working). Uses the CUDA Driver API loaded dynamically via `std.DynLib`. PTX kernels written in Zig and compiled via `zig build ptx` (cross-compiles to `nvptx64-cuda` target). Deferred execution with activation caching and KV device cache for zero-sync SDPA

#### ROCm (AMD)

**Radeon Open Compute.** AMD's open-source GPU compute platform, designed as a CUDA alternative.

- **Scope**: AMD GPUs only (Radeon, Instinct MI250/MI300/MI350)
- **Language**: HIP (Heterogeneous-compute Interface for Portability) — nearly identical syntax to CUDA C++, designed for easy porting
- **IR**: AMDGCN — AMD's GPU ISA, compiled by LLVM
- **Runtime**: HSA (Heterogeneous System Architecture) — AMD's low-level runtime for memory management and kernel dispatch
- **Key feature**: Unified memory architecture — CPU and GPU share the same physical memory on some AMD APUs, enabling fine-grained SVM (Shared Virtual Memory)
- **In Agave**: `src/backend/rocm.zig` (working). Uses HIP Runtime API loaded dynamically via `std.DynLib`. AMDGCN kernels written in Zig and compiled to HSACO via `zig build amdgcn`. Deferred execution with activation caching, same pattern as CUDA backend

#### Metal (Apple)

**Apple's GPU API** for macOS, iOS, and Apple Silicon (M1/M2/M3/M4).

- **Scope**: Apple GPUs only
- **Language**: MSL (Metal Shading Language) — a C++-like language for GPU kernels
- **IR**: Metal IR (compiled from MSL by Apple's shader compiler, or AIR — Apple Intermediate Representation)
- **Runtime**: Metal framework (Objective-C/Swift API, accessed from Zig via `objc.zig` runtime bridge)
- **Key feature**: Unified memory — on Apple Silicon, CPU and GPU share the same RAM. `newBufferWithBytesNoCopy` wraps existing CPU memory as a GPU buffer with zero copy, which is why Agave can pass mmap'd weight data directly to Metal kernels
- **In Agave**: `src/backend/metal.zig` (working). MSL kernels split across `src/backend/kernels/metal/` (gemv, norm, rope, sdpa, elementwise, common). Uses threadgroup-level `simd_sum` reduction, `float4 dot()` vectorization, and MTLBuffer caching

#### Vulkan

**Cross-platform GPU API** by the Khronos Group (the same consortium behind OpenGL).

- **Scope**: All major GPUs — NVIDIA, AMD, Intel, Qualcomm, Apple (via MoltenVK translation layer)
- **Language**: GLSL or HLSL (compiled to SPIR-V)
- **IR**: SPIR-V (Standard Portable Intermediate Representation) — a binary bytecode format that all Vulkan drivers can consume
- **Runtime**: Vulkan API (very verbose, explicit control over everything — memory, synchronization, command buffers)
- **Key feature**: True cross-platform GPU compute. One SPIR-V kernel runs on NVIDIA, AMD, Intel, and (via MoltenVK) Apple GPUs
- **Trade-off**: More boilerplate than CUDA/Metal, and vendor-specific optimizations (tensor cores, simd_sum) aren't accessible through the generic API
- **In Agave**: `src/backend/vulkan.zig` (working). Pre-compiled SPIR-V compute shaders in `src/backend/kernels/vulkan/`. Uses subgroup arithmetic for reductions, fused single-dispatch normalization/softmax, and buffer caching for weight uploads

#### OpenCL

**Open Computing Language.** An older cross-platform GPU compute standard, also by Khronos.

- **Scope**: NVIDIA, AMD, Intel, ARM, FPGAs — very broad hardware support
- **Language**: OpenCL C (a restricted C dialect for kernels)
- **Status**: Largely superseded by Vulkan Compute for new projects. NVIDIA's OpenCL support is maintained but not prioritized. Apple deprecated OpenCL in favor of Metal.
- **In Agave**: Not used. Vulkan is the preferred cross-platform path

#### MLX (Apple)

**Apple's ML framework**, not a GPU API per se. MLX is a Python/C++ array framework (like NumPy/PyTorch) that runs on Apple Silicon using Metal under the hood.

- **Scope**: Apple Silicon only
- **What it is**: A high-level ML framework, not a low-level compute API. You write `mlx.core.matmul(a, b)`, and MLX dispatches optimized Metal kernels internally
- **Quantization format**: MLX defines its own quantization scheme (affine, group_size=64, configurable bits). The MLX format stores weights as packed values (u4 for 4-bit, u6 for 6-bit) in `u32` words with per-group `bf16` scales and biases
- **In Agave**: MLX itself is not used, but Agave loads MLX-quantized SafeTensors files and implements the dequantization in `src/ops/mlx.zig`. The actual compute runs on Agave's own Metal or CPU backend, not through MLX

#### SPIR-V

**Standard Portable Intermediate Representation** — not a compute API, but a binary IR format used by Vulkan (and OpenCL 2.1+).

- **Role**: The "assembly language" of Vulkan. Kernels written in GLSL/HLSL are compiled to SPIR-V, which Vulkan drivers then JIT-compile to native GPU instructions
- **Analogy**: SPIR-V is to Vulkan what PTX is to CUDA — a portable intermediate form between the source language and the hardware
- **In Agave**: Vulkan backend uses pre-compiled SPIR-V compute shaders in `src/backend/kernels/vulkan/`

#### How They Relate

```
                    ┌─────────────────────────────────────────────┐
                    │              Source Languages                │
                    │  CUDA C++    MSL    HIP    GLSL    OpenCL C │
                    └──────┬───────┬──────┬──────┬───────┬────────┘
                           │       │      │      │       │
                           ▼       ▼      ▼      ▼       ▼
                    ┌─────────────────────────────────────────────┐
                    │           Intermediate Representations       │
                    │    PTX     Metal IR   AMDGCN   SPIR-V       │
                    └──────┬───────┬──────┬──────┬────────────────┘
                           │       │      │      │
                           ▼       ▼      ▼      ▼
                    ┌─────────────────────────────────────────────┐
                    │              GPU Hardware                    │
                    │  NVIDIA     Apple    AMD    Intel/Qualcomm   │
                    └─────────────────────────────────────────────┘

    Vendor-specific:  CUDA ──→ PTX ──→ NVIDIA only
                      Metal ──→ Metal IR ──→ Apple only
                      ROCm/HIP ──→ AMDGCN ──→ AMD only

    Cross-platform:   Vulkan ──→ SPIR-V ──→ All vendors
                      OpenCL ──→ SPIR-V or vendor IR ──→ All vendors (legacy)

    High-level:       MLX ──→ Metal (internally) ──→ Apple only
                      PyTorch ──→ CUDA/ROCm/Metal (internally) ──→ Depends on backend
```

**Agave's strategy**: Use vendor-specific APIs (Metal, CUDA, ROCm) for maximum performance on each platform, with Vulkan as the universal fallback on Linux/Windows. The `Backend` interface abstracts all of these behind a single tagged-union dispatch, so model code is completely hardware-agnostic.

| Platform | Primary Backend | Fallback |
|----------|----------------|----------|
| macOS (Apple Silicon) | Metal | CPU |
| Linux + NVIDIA | CUDA | Vulkan → CPU |
| Linux + AMD | ROCm | Vulkan → CPU |
| Linux + Intel | Vulkan | CPU |
| Windows + NVIDIA | CUDA | Vulkan → CPU |

**Note on macOS + Vulkan**: macOS has no native Vulkan support — Apple only ships Metal. [MoltenVK](https://github.com/KhronosGroup/MoltenVK) is a third-party translation layer that maps Vulkan calls to Metal under the hood. It works for portability but adds overhead and cannot expose Metal-specific optimizations (threadgroup `simd_sum`, zero-copy buffer wrapping, etc.). Agave uses Metal directly on macOS and reserves Vulkan for Linux/Windows where no vendor-specific API is available.

### KV Cache

During autoregressive generation, each new token needs to attend to all previous tokens. Recomputing K and V for every previous position would be wasteful, so we **cache** them:

```
Token 1: compute K₁, V₁, store in cache
Token 2: compute K₂, V₂, store in cache, attend to [K₁,K₂], [V₁,V₂]
Token 3: compute K₃, V₃, store in cache, attend to [K₁,K₂,K₃], [V₁,V₂,V₃]
```

The KV cache grows linearly with sequence length. For a model with 30 layers, 5 KV heads, 128-dim heads, and 4096 max tokens, the cache is: `30 × 5 × 128 × 4096 × 2 (K+V) × 4 bytes = 600MB`.

#### PagedAttention

PagedAttention breaks the KV cache into fixed-size blocks (default 16 positions). A block table maps logical positions to physical blocks, eliminating internal fragmentation:

```
physical_block = block_table[position / block_size]
offset_in_block = position % block_size
K[position] = blocks[physical_block].keys[offset * kvd + head * hd : ...]
```

Each `CacheBlock` contains: `keys[block_size * kv_dim]`, `values[block_size * kv_dim]`, `used: u16`, `ref_count: u16` (copy-on-write support). Benefits include elimination of internal fragmentation, memory sharing between requests via reference counting, and support for continuous batching.

The paged attention kernel (`attention.zig`) walks the block table during score computation and value accumulation, translating logical sequence positions to physical block addresses.

#### RadixAttention

RadixAttention uses a radix tree (prefix trie, fanout=256) over token sequences to automatically detect and share the longest common prefix across requests. Nodes store token sequences and physical block IDs.

Key operations (all at the scheduler layer, **never** in the hot path):
- **Insert:** Walk the tree, create new nodes for diverging suffixes
- **Lookup:** Find the longest cached prefix for a new prompt
- **Eviction:** LRU based on `last_access` timestamp
- **Sharing:** `ref_count` per block for copy-on-write

RadixAttention is the preferred strategy for production serving because it enables prefix caching — if two requests share the same system prompt, the KV cache for that prefix is computed once and reused.

### Format (GGUF / SafeTensors)

Model weights need a container format. Agave supports two:

**GGUF** — Self-contained binary format with weights + metadata + tokenizer in one file. Memory-mapped for zero-copy access. Supports all quantization types. Used by llama.cpp ecosystem.

**SafeTensors** — HuggingFace's format. JSON header + raw tensor data. Multi-shard (weights split across multiple files). Requires separate `config.json` and `tokenizer.json`. Used by MLX quantized models.

The `Format` interface abstracts both: `getTensor(name)` returns a pointer into mmap'd data regardless of the underlying format.

### Dispatcher Pattern

Agave uses **compile-time dispatch** to route operations to the correct implementation without runtime overhead:

```
main.zig → model.zig (interface) → gemma3.zig (implementation)
                                  → qwen35.zig
                                  → ...

model.zig → backend.zig (interface) → cpu.zig (implementation)
                                     → metal.zig
```

The `Backend` is a tagged union (`union(enum) { cpu, metal, ... }`) dispatched via `inline else` — the compiler generates a direct call for each variant with zero vtable overhead. The model code never imports backend implementations directly — it only sees the interface.

### Kernels

In GPU/CPU computing, a **kernel** is a single computational function that runs on the hardware. In Agave, "kernel" refers to the low-level functions inside the backend that do the actual math — one kernel per operation type per data type.

For example, the CPU backend has separate GEMV kernels for each quantization format:
- `gemvQ4_0` — dequantizes 4-bit nibbles and multiplies
- `gemvQ8_0` — dequantizes 8-bit integers and multiplies
- `gemvBF16` — converts bfloat16 to float32 and multiplies
- `gemvF32` — direct float32 multiply

The Metal backend has equivalent MSL (Metal Shading Language) kernels in `src/backend/kernels/metal/` that run on the GPU. Each kernel is a `kernel void` function dispatched to thousands of GPU threads.

**Why separate kernels per dtype?** Each quantization format has a different bit layout, so the dequantization logic is completely different. Fusing dequant into the kernel (rather than dequantizing first, then multiplying) avoids materializing the full-precision weight matrix in memory.

**Kernel fusion** means combining multiple operations into a single kernel to avoid intermediate memory reads/writes. For example, instead of:
```
gemv(gate) → write to memory → read from memory → gelu → write → read → gemv(down)
```
A fused kernel does:
```
fused_mlp: load input once → compute gate and up → gelu in-register → multiply → compute down → write once
```
This is especially important on GPUs where memory bandwidth is the bottleneck.

### SDPA (Scaled Dot-Product Attention)

SDPA is the core attention computation, extracted into a shared kernel (`src/ops/attention.zig`) used by all model architectures:

```
SDPA(Q, K, V, scale) = softmax(Q @ K^T * scale) @ V
```

Agave's implementation (`scaledDotProductAttention`) handles:
- **KV cache append**: Writes current K/V to the cache before computing attention
- **GQA**: Maps each query head to its corresponding KV head group
- **Sliding window**: Optional window parameter for models like GPT-OSS that only attend to recent tokens
- **Attention sinks**: Optional score offset for prepended sink logits
- **SIMD**: V8 vectorized dot products and value accumulation

The function signature captures all these variants:
```zig
pub fn scaledDotProductAttention(
    q, kv_keys, kv_values, k_buf, v_buf,  // data
    attn_out, scores,                       // output + scratch
    nh, nkv, hd, seq_len, scale,           // dimensions
    be,                                     // backend (for softmax)
    window, score_offset,                   // optional sliding window
) void
```

**Naive vs Flash**: The current implementation materializes the full scores vector (`scores[seq_len]`), which grows linearly with context length. **FlashAttention** is an algorithm that computes attention in tiles using online softmax (running max + running sum), never materializing the full scores vector. This reduces memory from O(n) to O(1) per head and improves cache locality. Agave does not yet implement FlashAttention — it's the single biggest optimization opportunity for long sequences.

#### Per-Head QK Normalization

**Used by:** Gemma3, Qwen3.5 (attention layers)

After Q/K projections but before RoPE, each head's Q and K vectors are RMS-normalized with learned weights. This stabilizes attention scores regardless of embedding magnitude:

```
For each head h:
  Q[h] = rmsNorm(Q[h], q_norm_weight)
  K[h] = rmsNorm(K[h], k_norm_weight)
```

In GGUF, the norm weights already have +1.0 baked in (Gemma3 convention).

#### Sliding Window Attention

**Used by:** GPT-OSS

Even-numbered layers restrict attention to the most recent `sliding_window` tokens (default 128). Odd layers attend to the full sequence. This halves the KV cache cost of sliding layers while maintaining global context through alternation:

```
is_sliding = (layer_index % 2 == 0)
window = min(seq_len, sliding_window) if is_sliding else seq_len
start  = max(0, seq_len - sliding_window) if is_sliding else 0
```

#### Learned Attention Sinks

**Used by:** GPT-OSS

Each layer optionally has a learned per-head scalar logit (`attn_sinks.weight[nh]`) that is prepended to the attention scores before softmax. This acts as a "sink" token that absorbs excess attention probability, preventing attention from concentrating too heavily on early positions:

```
scores = [sink_logit[h], QK_score[0], QK_score[1], ..., QK_score[win-1]]
softmax(scores)
// Value accumulation only uses scores[1:] (sink has no value vector)
```

#### Sigmoid Attention Gate

**Used by:** Qwen3.5 (attention layers)

Q projection produces interleaved `[Q, gate]` per head (2x head_dim). After SDPA, the output is gated element-wise:

```
attn_out[i] *= sigmoid(gate[i])
```

This gives the model a learned, per-element control over how much attention output flows to the residual stream.

#### Logit Softcapping

**Used by:** Gemma3

After computing final logits, applies a tanh-based soft clamp that prevents extreme logit values while preserving relative ordering:

```
logits[i] = tanh(logits[i] / cap) * cap
```

Where `cap = final_logit_softcap` (from GGUF metadata). This bounds logits to `[-cap, +cap]` smoothly.

### Causal Convolution

Causal convolution is a 1D convolution that only looks at past inputs (not future ones). It's used by SSM architectures (Mamba-2, DeltaNet) as a preprocessing step before the state recurrence:

```
conv_out[t] = sum(conv_weight[k] * input[t-k] for k in 0..d_conv)
```

With `d_conv=4`, each output depends on the current input and the 3 most recent inputs. A **ring buffer** stores the history so we don't need to keep the full sequence:

```
Ring buffer: [input[t-3], input[t-2], input[t-1]]
New input:   input[t]
Output:      w[0]*input[t-3] + w[1]*input[t-2] + w[2]*input[t-1] + w[3]*input[t]
Shift:       buffer becomes [input[t-2], input[t-1], input[t]]
```

Agave's implementation (`src/ops/ssm.zig: causalConv1dSilu`) fuses the convolution with SiLU activation in a single pass.

**Explicit algorithm:**

```
For each channel ch:
  sum = bias[ch] (if present)
  sum += conv_state[0, ch] * w[ch, 0]    // history position 0
  sum += conv_state[1, ch] * w[ch, 1]    // history position 1
  sum += conv_state[2, ch] * w[ch, 2]    // history position 2
  sum += conv_in[ch]       * w[ch, 3]    // current input
  conv_out[ch] = SiLU(sum)               // x * sigmoid(x)

Shift ring buffer left, append conv_in
```

**Ring buffer layout:** `conv_state[(d_conv-1) x conv_ch]`, row-major. The buffer stores the `d_conv-1` most recent inputs (typically 3). After each step, rows shift left and the current input is appended at the end. This avoids any allocation in the hot path.

**Used by:** Qwen3.5 DeltaNet and Nemotron-H Mamba-2 (both with `d_conv=4`).

### RMS Normalization

RMS (Root Mean Square) normalization stabilizes training and inference by normalizing each vector to unit RMS:

```
rmsNorm(x, weight, eps) = x / sqrt(mean(x²) + eps) * weight
```

Unlike LayerNorm, RMSNorm has no mean subtraction — it only scales by the inverse RMS. This is simpler and empirically works just as well for transformers.

Every transformer layer applies RMSNorm before attention and before FFN (pre-norm architecture). Some models add extra norms:
- **Post-norms** (Gemma3): Additional RMSNorm after attention and after FFN, before the residual add
- **QK norms** (Gemma3, Qwen3.5): Per-head RMSNorm on Q and K vectors before computing attention scores

#### L2 Normalization

Unit-norm normalization without learnable weights:

```
norm = sqrt(sum(x²) + eps)
x[i] /= norm
```

Used in Qwen3.5 DeltaNet to normalize Q and K per head before the recurrence (implemented in `cpu.zig`). This ensures the delta rule operates on unit-length vectors, preventing state magnitude explosion.

### Residual Connections

Every sublayer (attention, FFN) uses a residual connection:
```
output = input + sublayer(input)
```

This lets gradients flow directly through the network during training and prevents the "vanishing gradient" problem in deep networks. During inference, it means the hidden state accumulates information from each layer rather than being completely overwritten.

### SwiGLU and Activation Functions

The FFN (Feed-Forward Network) in each transformer layer expands the hidden dimension, applies a non-linearity, and projects back:

```
FFN(x) = down_proj(activation(gate_proj(x)) * up_proj(x))
```

**SwiGLU** uses SiLU (Sigmoid Linear Unit) as the activation:
```
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

**Full activation function reference:**

| Function | Formula | Used by |
| :--- | :--- | :--- |
| **SiLU/Swish** | `x * sigmoid(x)` = `x / (1 + exp(-x))` | Most FFN layers, conv1d, SSM gating |
| **GELU** | `0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x³)))` | Gemma3 FFN |
| **Softplus** | `log(1 + exp(x))`, linear for x>20 | SSM dt computation |
| **Sigmoid** | `1 / (1 + exp(-x))` | DeltaNet beta, attention gate, MoE routing |

**Clamped SwiGLU (GPT-OSS MoE):** A gated variant with hard clamping to prevent overflow during mixed-precision expert computation:

```
output = clamp(SiLU(gate) * up, -7.0, 7.0)
```

The +/-7.0 clamp bounds the intermediate values before the down projection.

### MoE (Mixture of Experts)

Standard transformers use the same FFN weights for every token. **MoE** models have multiple FFN "experts" and a **router** that selects which experts to use for each token:

```
1. Router: scores = sigmoid(hidden @ gate_weight)    # score each expert
2. Select: top_k = top-4 experts by score            # pick best 4
3. Normalize: weights = softmax(top_k_scores)        # normalize selected scores
4. Compute: output = Σ weight[i] * expert_i(hidden)  # weighted sum of expert outputs
5. Shared: output += shared_expert(hidden)            # always-active shared expert
```

This allows models to have many more parameters (30B total) while only activating a fraction per token (3B active), giving the capacity of a large model with the compute cost of a small one.

GPT-OSS and GLM-4 use MoE. The expert weights are stored as stacked tensors: `switch_mlp.gate_proj.weight` has shape `[n_experts, expert_ff_size, hidden_size]`.

**Routing algorithm detail:**

```
1. router_logits = W_router @ hidden + bias       // [n_experts]
2. Select top-K experts by argmax (greedy, no heap — O(K*N) scan)
3. Softmax over selected experts only → mixing weights
4. For each selected expert:
     expert_out = down_proj(activation(gate_proj(x)) * up_proj(x))
5. output = sum(weight[i] * expert_out[i])
```

Expert selection uses stack-allocated arrays (`max_active_experts` = 8 for GPT-OSS and Nemotron-Nano, 16 for Qwen3.5) — zero heap allocation.

**Expert counts by model:**

| Model | Routed Experts | Top-K | Shared Expert | Notes |
| :--- | :--- | :--- | :--- | :--- |
| GPT-OSS | 32 | 4 | No | Standard softmax routing |
| GLM-4 | varies | varies | No | Sigmoid routing (see below) |
| Nemotron-Nano | 128 | 6 | Yes (1) | Shared expert has 2x hidden dim (3712 vs 1856) |

**Shared expert (Nemotron-Nano):** One expert is always active regardless of router output. Its contribution is added to the routed expert sum unconditionally.

**Sigmoid routing (GLM-4):** Uses `sigmoid(logit)` instead of softmax for expert gating. Each expert's gate is independent — multiple experts can have high activation simultaneously without competing. Top-K selection still applies.

### Embedding and Vocabulary Projection

The first and last operations in the forward pass convert between token IDs and vectors:

**Embedding lookup** (first): Token ID → row in a `[vocab_size × n_embd]` table. This is a simple table lookup, not a matrix multiply. The table may be quantized — the implementation dispatches on weight dtype (f32 copy, bf16/f16 cast, or block dequant for Q4/Q5/Q8/Q6_K/MXFP4). Gemma3 scales embeddings by `sqrt(n_embd)` after lookup, amplifying the embedding signal for its particular architecture.

**Vocabulary projection** (last): Hidden state → logits via `logits = W_output @ hidden`. This IS a matrix multiply — the largest single GEMV in the model (vocab_size rows, often 128K-262K). For models with **tied embeddings** (Gemma3), the output weight matrix is the same as the embedding table, saving parameters.

**Final logits pipeline** (implemented in `math.zig`): The tail of the forward pass is a three-step sequence:
1. RMS normalize the final hidden state
2. Project to vocabulary via LM head GEMV
3. GPU sync + CPU argmax

The sync point after step 2 is mandatory on GPU backends — the GPU wrote the logits buffer, and argmax runs on CPU. Without the sync, argmax would read stale data.

**Argmax**: The logit with the highest value determines the predicted next token. This is greedy decoding — no randomness. The sampling parameters below control how the next token is selected.

### Decoding and Sampling Parameters

After the forward pass produces logits (one score per vocabulary token), the model must **select** the next token. The simplest method is greedy decoding (pick the highest score), but this produces repetitive, deterministic output. Sampling parameters add controlled randomness for more natural text.

#### Max Tokens (`-n`, `--max-tokens`)

The maximum number of tokens to generate before stopping. Generation also stops early if the model produces an EOS (end-of-sequence) token. Default: 512.

```
-n 100    Generate at most 100 tokens
-n 1      Generate exactly 1 token (useful for benchmarking)
```

This is a hard cap — the model will never produce more than this many tokens in one generation, regardless of whether it has "finished" its response.

#### Temperature (`-t`, `--temperature`)

Controls the randomness of token selection by scaling the logits before sampling. Mathematically:

```
adjusted_logits[i] = logits[i] / temperature
probabilities = softmax(adjusted_logits)
next_token = sample(probabilities)
```

| Value | Effect | Use case |
|-------|--------|----------|
| `0` | **Greedy** — always pick the highest logit (argmax, no randomness) | Factual Q&A, code, math |
| `0.1-0.5` | **Low** — mostly picks the top token, occasionally varies | Reliable but slightly varied output |
| `0.7-0.9` | **Medium** — balanced creativity and coherence | General conversation, writing |
| `1.0` | **Neutral** — raw model probabilities, no scaling | Default model behavior |
| `1.5-2.0` | **High** — flattens the distribution, more random picks | Creative writing, brainstorming |

**How it works**: Dividing logits by a small temperature makes the differences between scores larger (the softmax becomes "peakier"), so the top token dominates. Dividing by a large temperature makes scores more similar (the softmax becomes "flatter"), so lower-ranked tokens get a chance.

At temperature=0, Agave skips sampling entirely and uses argmax — this is deterministic (same input always produces the same output).

#### Top-K (`--top-k`)

Restricts sampling to only the K highest-scoring tokens. All other tokens are excluded before sampling.

```
--top-k 40    Only consider the top 40 tokens
--top-k 0     Disabled (consider all tokens) — default
```

**How it works**: Sort tokens by logit score, keep only the top K, renormalize their probabilities, then sample. This prevents the model from ever picking extremely unlikely tokens (which can cause incoherent output at high temperatures).

**Example** with vocabulary of 128K tokens and top-k=40: instead of sampling from 128,000 options, the model only considers the 40 most likely next tokens.

#### Top-P / Nucleus Sampling (`--top-p`)

Restricts sampling to the smallest set of tokens whose cumulative probability exceeds P. More adaptive than top-k because the number of candidates varies based on the model's confidence.

```
--top-p 0.9    Keep tokens until cumulative probability reaches 90%
--top-p 1.0    Disabled (keep all tokens) — default
```

**How it works**: Sort tokens by probability (descending), accumulate probabilities until the sum exceeds P, discard the rest, renormalize, then sample.

**Example**: If the model is very confident (top token has 95% probability), top-p=0.9 might keep only 1-2 tokens. If the model is uncertain (top 50 tokens each have ~2% probability), top-p=0.9 keeps ~45 tokens. This automatically adjusts the candidate pool based on context.

**Top-K vs Top-P**: Top-K always keeps exactly K tokens regardless of confidence. Top-P adapts — fewer candidates when the model is confident, more when it's uncertain. They can be combined: apply top-k first, then top-p on the remaining candidates.

#### Repeat Penalty (`--repeat-penalty`)

Discourages the model from repeating tokens it has already generated. Applied to logits before sampling.

```
--repeat-penalty 1.0    Disabled (no penalty) — default
--repeat-penalty 1.1    Mild penalty — reduces repetition
--repeat-penalty 1.5    Strong penalty — strongly discourages repeats
```

**How it works**: For each token that has already appeared in the generated output, divide its logit by the penalty factor:

```
if token was previously generated:
    logits[token] = logits[token] / repeat_penalty    (if logit > 0)
    logits[token] = logits[token] * repeat_penalty    (if logit < 0)
```

A penalty of 1.0 has no effect. Values above 1.0 make repeated tokens less likely. This prevents the common failure mode where the model gets stuck in a loop ("the the the the...").

#### Combining Parameters

The parameters are applied in order: **temperature → top-k → top-p → repeat penalty → sample**. Common combinations:

```bash
# Deterministic (factual, reproducible)
agave model.gguf -t 0 "What is the capital of France?"

# Balanced (good for most uses)
agave model.gguf -t 0.7 --top-p 0.9 "Tell me a story"

# Creative (varied, surprising)
agave model.gguf -t 1.2 --top-k 50 --top-p 0.95 "Write a poem about"

# Anti-repetition (long-form generation)
agave model.gguf -t 0.8 --repeat-penalty 1.1 -n 1000 "Write an essay about"
```

**Note**: Agave currently implements greedy decoding (temperature=0) only. The sampling parameters are parsed by the CLI but the actual top-k/top-p/temperature sampling logic is not yet implemented — all generation uses argmax regardless of the parameter values. This is a planned feature.

---

## 3. Architecture Overview

```
agave/
├── build.zig              # Build config (ReleaseFast default + Debug)
├── build.zig.zon          # Dependencies (clap CLI parser, vaxis terminal UI)
├── src/
│   ├── main.zig           # CLI: arg parsing, format detection, model init, REPL, recipe application
│   ├── arch.zig           # Architecture enum, detection, chat template mapping
│   ├── server.zig         # HTTP server (OpenAI API + htmx chat UI)
│   ├── display.zig        # Rich CLI output (banner, stats, progress)
│   ├── chat_template.zig  # Data-driven chat prompt templates (ChatML, Gemma, GPT-OSS)
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
│   │   ├── model.zig      # Model interface (forward, resetCache, cancel)
│   │   ├── gemma3.zig     # Gemma 3 (GQA, GELU, post-norms)
│   │   ├── qwen35.zig     # Qwen 3.5 (hybrid DeltaNet SSM + attention)
│   │   ├── gpt_oss.zig    # GPT-OSS (MoE, sliding window, attention sinks)
│   │   ├── nemotron_h.zig # Nemotron-H (Mamba-2 + attention hybrid)
│   │   ├── glm4.zig       # GLM-4 MoE Lite (MLA + MoE, MLX 4/6-bit)
│   │   └── nemotron_nano.zig # Nemotron Nano (SSM + MoE + attention, NVFP4)
│   ├── ops/
│   │   ├── attention.zig  # Shared SDPA kernel (SIMD, sliding window)
│   │   ├── math.zig       # argmax, softplus, sigmoid, applyGelu, finalLogits
│   │   ├── ssm.zig        # SSM ops: causal conv1d, Mamba-2 recurrence, group norm+gate
│   │   ├── quant.zig      # Quantization helpers (bf16, mxfp4, fp8, iq4nl)
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
│   │       ├── metal/     # MSL compute shaders (gemv, norm, rope, sdpa, elementwise)
│   │       ├── vulkan/    # GLSL compute shaders → compiled SPIR-V (.spv)
│   │       ├── cuda/      # Zig kernels compiled to PTX via nvptx64-cuda target
│   │       └── rocm/      # Zig kernels compiled to HSACO via amdgcn-amdhsa target
│   ├── kvcache/
│   │   └── manager.zig    # KV cache alloc/free helpers
│   └── tokenizer/
│       ├── tokenizer.zig  # Tokenizer interface
│       └── bpe.zig        # BPE + SPM tokenizer with byte-level encoding
└── models/                # Model files (GGUF or SafeTensors directories)
```

---

## 4. The Inference Pipeline

When you run `agave model.gguf "Hello"`, this happens:

```
1. LOAD        model.gguf → mmap → Format interface
                config.json + tokenizer.json (if SafeTensors)

2. DETECT      "general.architecture" = "gemma3" → Gemma3Model
               "model_type" = "glm4_moe_lite" → Glm4Model

3. BACKEND     macOS → Metal GPU (auto)
               --backend cpu → CPU fallback

4. RECIPE      Match arch + backend + quant → apply proven defaults
               (user CLI flags always override recipe values)

5. TEMPLATE    arch → ChatTemplate (ChatML, Gemma, GPT-OSS)
               Format prompt: system prefix + user text + assistant prefix

6. TOKENIZE    formatted prompt → [BOS, 15496, ...] (BPE/SPM encode)

7. PREFILL     For each input token:
                 model.forward(token_id) → next_token_id
               (fills KV cache, output discarded except last)

8. GENERATE    Loop:
                 next = model.forward(last_token)
                 if next is EOG: break
                 if repeat_count >= threshold: break
                 print(tokenizer.decode(next))
                 last = next

9. STATS       "5 tok, 10.4 tok/s, prefill 200ms, gen 480ms"
```

### Inside model.forward()

Each call to `forward(token_id)` runs the full transformer stack:

```
token_id
  │
  ▼
Embedding Lookup ─── look up row in [vocab × n_embd] table
  │                  (dequantize from Q4_0/F16/BF16 to f32)
  │
  ▼
┌─── Layer 0 ───────────────────────────────────────────┐
│  Pre-Norm (RMSNorm)                                   │
│  ↓                                                    │
│  Attention / SSM sublayer:                             │
│    Q = gemv(hidden, W_q)     ← quantized               │
│    K = gemv(hidden, W_k)                               │
│    V = gemv(hidden, W_v)                               │
│    RoPE(Q, K, position)                                │
│    Cache K, V                                          │
│    attn_out = softmax(Q·K^T/√d) @ V                   │
│    output = gemv(attn_out, W_o)                        │
│  ↓                                                    │
│  Residual Add: hidden += output                        │
│  ↓                                                    │
│  Post-Norm (RMSNorm)                                  │
│  ↓                                                    │
│  FFN sublayer:                                         │
│    gate = gemv(hidden, W_gate)                         │
│    up   = gemv(hidden, W_up)                           │
│    act  = activation(gate) * up                        │
│    down = gemv(act, W_down)                            │
│  ↓                                                    │
│  Residual Add: hidden += down                          │
└───────────────────────────────────────────────────────┘
  │
  ... repeat for all N layers ...
  │
  ▼
Final RMSNorm
  │
  ▼
Logits = gemv(hidden, W_output)  ← [vocab_size] scores
  │
  ▼
argmax(logits) → predicted next token ID
```

**Activations** vary by model:
- SiLU (Qwen3.5, GPT-OSS, GLM-4, Nemotron-H): `x / (1 + exp(-x))`
- GELU (Gemma3): `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715x³)))`
- ReLU² (Nemotron-Nano MoE): `max(0, x)²`

---

## 5. Module Reference

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
| `softmax(data, n)` | In-place softmax | Yes |
| `rope(x, pos, nh, hd, rd, theta)` | Rotary position encoding | Yes |
| `silu(in, out, n)` | SiLU activation | Yes |
| `add(a, b, out, n)` | Element-wise add | Yes |
| `mul(a, b, out, n)` | Element-wise multiply | Yes |
| `embLookup(table, id, out, dim)` | Embedding with dequant | Once per token |
| `l2Norm(x, n, eps)` | L2 normalization | Rare |

### Chat Templates (`src/chat_template.zig`)

Data-driven chat prompt formatting. Each `ChatTemplate` defines role prefixes/suffixes and EOG (end-of-generation) token names for a model family.

| Preset | Models | EOG Tokens |
|--------|--------|------------|
| `chatml` | Qwen, Nemotron-H, GLM-4 | `<\|im_end\|>`, `<\|endoftext\|>` |
| `gemma` | Gemma 3, Gemma 2 | `<end_of_turn>`, `<eos>` |
| `gpt_oss` | GPT-OSS Harmony | `<\|end\|>`, `<\|endoftext\|>` |

The `format()` method assembles a complete prompt from system message + user message using the template's role markers. Adding a new model family requires only adding a template preset and mapping the arch to it.

### Recipes (`src/recipe.zig`)

Optional proven-default configurations matched by architecture + backend + quantization. A `Recipe` bundles sampling parameters (temperature, top_p, top_k, repeat_penalty), max_tokens, and ctx_size.

| Recipe | Arch | Backend | Quant | Key Defaults |
|--------|------|---------|-------|--------------|
| Qwen3.5 Q4 Metal | qwen3* | Metal | Q4* | temp=0.6, top_p=0.9, repeat=1.1 |
| Gemma Q4 Metal | gemma* | Metal | Q4* | temp=0.7, top_p=0.95 |
| GPT-OSS Metal | gpt* | Metal | any | temp=0.5, ctx=2048 |
| CPU generic | any | CPU | any | max_tokens=256, ctx=2048 |

**User CLI flags always override recipe defaults.** The `Overrides` struct tracks which args the user explicitly set. Recipe matching uses prefix matching on arch and quant names.

### Shared Ops (`src/ops/`)

| Function | File | Description |
|----------|------|-------------|
| `scaledDotProductAttention` | attention.zig | Full SDPA with KV cache, GQA, sliding window |
| `causalConv1dSilu` | ssm.zig | Causal conv1d with ring buffer + SiLU |
| `mamba2Recurrence` | ssm.zig | Mamba-2 per-head state update + output |
| `groupRmsNormSiluGate` | ssm.zig | Group RMS norm followed by SiLU gate |
| `argmax` | math.zig | Index of maximum element |
| `softplus` | math.zig | Numerically stable log(1 + exp(x)) |
| `sigmoid` | math.zig | 1 / (1 + exp(-x)) |
| `applyGelu` | math.zig | GELU activation in-place (named constants: `sqrt_2_over_pi`, `gelu_coeff`) |
| `finalLogits` | math.zig | RMSNorm + GEMV + argmax (forward tail) |
| `mlxGemvRaw` | mlx.zig | MLX 4/6-bit affine dequant GEMV (constants: `mlx_group_size=64`, `mlx_words_per_group=12`) |
| `mlxEmbLookup` | mlx.zig | Dequantize one row from MLX-quantized embedding table |
| `expertWeightStride` | model.zig | Byte stride between experts in packed GGUF weight tensors |

### Quantization Types

| DType | Bits/val | Block | Kernel | Models |
|-------|----------|-------|--------|--------|
| `f32` | 32 | 1 | Direct multiply | Reference |
| `f16` | 16 | 1 | Float cast + multiply | Embeddings |
| `bf16` | 16 | 1 | Bit shift + multiply | Gemma3 |
| `q8_0` | 8.5 | 32 | Scale × int8 | General |
| `q6_k` | 6.6 | 256 | Hierarchical scales | General |
| `q5_k` | 5.5 | 256 | Hierarchical + high bits | General |
| `q5_0` | 5.5 | 32 | Scale × 5-bit int | Nemotron Nano |
| `q4_0` | 4.5 | 32 | Scale × 4-bit nibble | General |
| `q4_1` | 5.0 | 32 | Scale + min × 4-bit | General |
| `mxfp4` | 4.25 | 32 | FP4 microscaling | Advanced |
| `mlx_q` | 6.0 | 64 | Affine: scale × u6 + bias | GLM-4 MLX |

---

## 6. Supported Models

| Model | Arch ID | Attention | FFN | Special |
|-------|---------|-----------|-----|---------|
| **Gemma 3** | `gemma3` | GQA + QK norm + post-norms | GELU + SwiGLU | Embedding scaling, logit softcap |
| **Qwen 3.5** | `qwen35` | GQA (every 4th layer) | SiLU + SwiGLU | DeltaNet SSM hybrid |
| **GPT-OSS** | `gpt_oss` | GQA + sliding window + sinks | SiLU + SwiGLU | MoE (top-k experts) |
| **Nemotron-H** | `nemotron_h` | GQA (sparse layers) | SiLU + SwiGLU | Mamba-2 SSM hybrid (GGUF) |
| **Nemotron Nano 30B** | `nemotron_nano` | GQA (sparse layers) | ReLU² MoE | SSM + MoE + attention hybrid (SafeTensors NVFP4) |
| **GLM-4 MoE Lite** | `glm4` | MLA (compressed KV) | SiLU + SwiGLU | MoE + MLX 6-bit quant |

### Model-Specific Features

**Gemma 3**: GGUF converter bakes +1.0 into RMS norm weights (don't add again). Embeddings scaled by `sqrt(n_embd)`. Uses SPM tokenizer (no merges). Tied output embeddings.

**Qwen 3.5**: Hybrid architecture alternating DeltaNet SSM and full attention layers. DeltaNet uses causal conv1d → selective state recurrence with learned decay. Full attention layers have gated output with sigmoid.

**Nemotron-H** (GGUF): Mamba-2 SSM with per-group RMS normalization on SSM output. Layer types (SSM/attention/FFN-only) detected from tensor presence. Squared ReLU activation for FFN-only layers.

**Nemotron Nano 30B** (SafeTensors NVFP4): 52-layer hybrid with `hybrid_override_pattern` (M=SSM, E=MoE, *=attention). Mixed quantization — most layers NVFP4 (weight nibbles + FP8 E4M3 scales in separate tensors), 6 SSM layers (4,11,18,25,32,41) use BF16. 128 routed experts with top-6 routing + shared expert. Status: forward pass runs end-to-end, tokenizer decode working, but output quality needs improvement (MoE router scoring may need numerical stability work).

---

## 7. Performance

Benchmarks on Apple M-series, Gemma 3 1B, ReleaseFast build:

| Config | tok/s | Notes |
|--------|-------|-------|
| Metal BF16 | 34.5 | Highest quality |
| Metal Q8_0 | 34.1 | Best quality/speed |
| Metal Q4_0 | 21.7 | Best compression/speed |
| CPU BF16 | 16.8 | No GPU needed, high quality |
| CPU Q4_0 | 5.6 | No GPU needed |
| CPU Q8_0 | 4.4 | Higher quality, slower |

### Optimization Techniques

**CPU**:
- V8 SIMD vectors (128-bit NEON, 2 ops per cycle)
- 4-row cache-friendly GEMV (x vector loaded once, applied to 4 weight rows)
- V8 accumulator across blocks with single `@reduce` per row
- Precomputed sin/cos tables for RoPE

**Metal**:
- Threadgroup-level parallel reduction with `simd_sum` (256 threads per row)
- `float4 dot()` vector operations in GEMV inner loops
- MTLBuffer caching (eliminates ~800 ObjC alloc/release per token)
- Command buffer batching for multi-pass ops (rmsNorm, softmax)
- Zero-copy buffer wrapping for mmap'd weights (`newBufferWithBytesNoCopy`)
