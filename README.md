<p align="center">
  <img src="docs/logo.svg" alt="Agave" width="480">
</p>

<p align="center">
  A high-performance LLM inference engine written in Zig.<br>
  Zero external ML libraries — all kernels, quantization, and model logic from scratch.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="docs/CONTRIBUTING.md">Contributing</a> •
  <a href="docs/DOCUMENTATION.md">Docs</a>
</p>

---

## Features

- **7 Model Architectures**: Gemma 3, Gemma 4, Qwen 3.5, GPT-OSS, Nemotron-H, Nemotron Nano, GLM-4
- **6 Backends**: CPU (SIMD-optimized), Metal GPU (Apple Silicon), Vulkan, CUDA, ROCm, WebGPU — individually toggleable at build time
- **Compile-Time Model Selection**: Disable unused model architectures to reduce binary size (1.8 MB → 0.75 MB with all models stripped)
- **2 Formats**: GGUF, SafeTensors (multi-shard, MLX quantized, NVFP4)
- **20+ Quantization Types**: F32, F16, BF16, Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, TQ1_0, IQ4_XS, IQ4_NL, FP8 E4M3, FP8 E5M2, NVFP4, MXFP4, MLX 4/6/8-bit
- **9 KV Cache Quantization Types**: F32, F16, Q8_0, INT8, FP8, NVFP4, TurboQuant 2/3/4-bit — with asymmetric K/V support
- **Tiered KV Cache**: VRAM + RAM + SSD offloading with async prefetch (`--kv-tiers vram+ram+ssd`)
- **Chat Templates**: Data-driven per-architecture prompt formatting (ChatML, Gemma, Gemma 4, Qwen 3.5, GLM-4, GPT-OSS)
- **Recipes**: Optional proven-default configs per model/hardware/quant combo
- **Model Download**: `agave pull <org/repo>` — download GGUF models from HuggingFace Hub with auto quant selection
- **Interactive REPL**: Multi-turn chat with `/help`, `/clear`, `/stats`, `/model`, `/quit`
- **HTTP Server**: OpenAI + Anthropic API compatible, built-in chat UI, Prometheus metrics, rate limiting
- **Multimodal Vision**: Image understanding via Gemma 4 SigLIP-2 and Gemma 3 SigLIP vision encoders — image upload via CLI (`--image`) and HTTP API
- **Batched Prefill**: Chunked GEMM + fused FlashAttention-2 for fast prompt processing
- **~183 tok/s** on Qwen3.5 0.8B Q8_0 (Metal, Apple Silicon M4 Pro), **1.2-1.7x faster than llama.cpp on Q8_0** (Q4_K performance is a [known gap](docs/TODO.md#performance) — active optimization target)

## Quick Start

```bash
# Build (produces both ReleaseFast and Debug binaries)
zig build

# Download a model from HuggingFace
./zig-out/bin/agave pull Qwen/Qwen3.5-0.8B-GGUF

# Interactive REPL
./zig-out/bin/agave model.gguf

# Single prompt
./zig-out/bin/agave model.gguf "What is the capital of France?"

# HTTP server
./zig-out/bin/agave model.gguf --serve --port 8080

# Quiet mode (pipe-friendly, no banner/stats)
./zig-out/bin/agave model.gguf -q "Hello" > output.txt

# Force CPU backend
./zig-out/bin/agave model.gguf --backend cpu

# SafeTensors directory (MLX models)
./zig-out/bin/agave models/mlx-community/gemma-3-4b-it-qat-4bit

# TurboQuant KV cache (2/3/4-bit quantization for longer contexts)
./zig-out/bin/agave model.gguf --kv-type turbo4

# KV cache eviction (extend context past --ctx-size limit)
./zig-out/bin/agave model.gguf --kv-eviction norm --kv-budget 2048
./zig-out/bin/agave model.gguf --kv-eviction tri   # requires .cal file

# Generate TriAttention calibration data
./zig-out/bin/agave calibrate model.gguf

# Vision: describe an image (requires mmproj or built-in vision encoder)
./zig-out/bin/agave model.gguf --image photo.png "Describe this image"

# Override recipe defaults (user flags always win)
./zig-out/bin/agave model.gguf -t 0.9 --top-p 0.95 "Tell me a story"
```

## Supported Models

| Model | Sizes | Status | Quant Types | Notes |
|-------|-------|--------|-------------|-------|
| Gemma 3 | 1B, 4B, 12B, 27B | Working | BF16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, MLX 4-bit | SPM tokenizer, GELU activation, batched prefill |
| Gemma 4 | E2B, E4B, 26B-A4B | Working | Q8_0, Q4_K, MLX 4-bit | MoE (top-8), channel-based chat template, multimodal vision (SigLIP-2) |
| Qwen 3.5 | 0.8B, 9B, 27B | Working | Q4_0, Q4_K_M, Q8_0, BF16, MLX 4-bit | Hybrid DeltaNet SSM + attention |
| GPT-OSS | 20B | Partial | Q4_0 | MoE, sliding window, attention sinks (poor output quality) |
| Nemotron-H | — | Partial | Q5_0 | Mamba-2 + attention hybrid, GGUF (poor output quality) |
| Nemotron Nano | 30B | Partial | MLX 4-bit, NVFP4 | SSM + MoE + attention hybrid, SafeTensors (poor output quality) |
| GLM-4 MoE Lite | 4.7B | Partial | MLX 4/6/8-bit | MLA + MoE (GGUF compatibility issue, poor output quality) |

## Model Download

Download GGUF models from HuggingFace Hub with automatic quantization selection:

```bash
# Download best available quantization (prefers Q4_K_M)
./zig-out/bin/agave pull Qwen/Qwen3.5-0.8B-GGUF

# Request specific quantization
./zig-out/bin/agave pull Qwen/Qwen3.5-0.8B-GGUF --quant Q8_0

# List available GGUF files without downloading
./zig-out/bin/agave pull Qwen/Qwen3.5-0.8B-GGUF --list

# Private repos
HF_TOKEN=hf_xxxxx ./zig-out/bin/agave pull org/private-model
```

Downloads are stored in the standard HuggingFace cache layout with an agave convenience symlink. Supports resume on interrupted downloads.

## Calibration

Generate TriAttention calibration data for frequency-domain KV eviction:

```bash
# Run calibration (produces model.cal alongside model.gguf)
./zig-out/bin/agave calibrate model.gguf
```

The calibration pass records per-head Q/K frequency statistics used by the `--kv-eviction tri` policy. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## HTTP Server

Start with `--serve`. Supports both synchronous JSON and SSE streaming.

```bash
./zig-out/bin/agave model.gguf --serve --port 8080 --api-key sk-mykey
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI chat completion API |
| `/v1/completions` | POST | OpenAI text completion API |
| `/v1/messages` | POST | Anthropic Messages API |
| `/v1/responses` | POST | OpenAI Responses API |
| `/v1/models` | GET | List loaded models |
| `/v1/embeddings` | POST | Embedding generation (stub — returns 501) |
| `/v1/chat` | POST | Built-in web chat UI |
| `/v1/chat/regenerate` | POST | Regenerate last assistant response |
| `/v1/conversations` | GET, POST | Conversation management |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/metrics` | GET | Prometheus metrics |

Server features: up to 64 concurrent connections, request scheduler (batch up to 8, 120s timeout), 30s connection read timeout, rate limiting, Bearer token auth, CORS support.

## Interactive REPL

Launch without a prompt argument for multi-turn chat:

```bash
./zig-out/bin/agave model.gguf
```

**Commands:**

| Command | Description |
|---------|-------------|
| `/clear`, `/reset` | Clear conversation history and KV cache |
| `/context`, `/ctx` | Show context window usage (tokens used / max) |
| `/system <text>` | Set system prompt (clears conversation) |
| `/system` | Show current system prompt |
| `/stats` | Toggle generation statistics display |
| `/verbose` | Toggle technical details (params, EOG tokens) |
| `/debug` | Toggle debug logging (token IDs, layer timing) |
| `/model` | Show model information |
| `/help` | Show REPL help |
| `/quit`, `/exit`, `/q` | Exit |

Keyboard shortcuts: `Ctrl+C` cancel, `Ctrl+D` quit, `Ctrl+L` clear screen, `Ctrl+R` reverse search.

## Benchmarks

Measured on Apple M4 Pro (48 GB unified memory). See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full methodology.

| Model | Quant | Backend | Decode (tok/s) | vs llama.cpp |
|-------|-------|---------|---------------:|-------------:|
| Qwen3.5 0.8B | Q8_0 | Metal | 183.3 | **1.31x** |
| Qwen3.5 9B | Q8_0 | Metal | 41.7 | **1.67x** |
| Gemma 3 4B | MLX-Q4 | Metal | 78.1 | — |
| Gemma 3 12B | Q8_0 | Metal | 22.3 | **1.19x** |
| Gemma 4 E2B | — | Metal | 9.7 | — |
| Gemma 4 E4B | — | Metal | 8.5 | — |
| Gemma 4 26B-A4B | — | Metal | 5.0 | — |
| Gemma 3 27B | QAT 4-bit | Metal | 11.6 | — |
| Qwen3.5 0.8B | MLX-4bit | Metal | 12.7 | — |

## Prerequisites

- **Zig 0.16.0**
- macOS (Metal backend) / Linux (Vulkan, CUDA, ROCm) / any platform (CPU, WebGPU backends)
- GPU backends load drivers at runtime via dlopen — no SDK needed at build time

## CLI Options

```
agave [OPTIONS] <model> [prompt]

  -h, --help               Show help
  -v, --version            Print version
  -q, --quiet              Suppress banner and stats
  -s, --serve              Start HTTP server
  -p, --port <PORT>        Server port [default: 49453]
  -n, --max-tokens <N>     Max tokens to generate [default: 512]
  -t, --temperature <T>    Sampling temperature, 0 = greedy [default: 0]
      --top-p <P>          Nucleus sampling threshold [default: 1.0]
      --top-k <K>          Top-k sampling, 0 = disabled [default: 0]
      --repeat-penalty <R> Repetition penalty [default: 1.0]
      --system <TEXT>      System prompt for chat formatting
      --backend <BE>       auto, cpu, metal, vulkan, cuda, rocm, webgpu [default: auto]
      --ctx-size <N>       Context window size [default: min(model, 4096), 0 = model max]
      --seed <N>           Random seed for sampling [default: random]
      --kv-type <TYPE>     KV cache quantization: f32, f16, q8_0/q8, int8/i8, fp8/fp8_e4m3, nvfp4/fp4, turbo2/tq2, turbo3/tq3, turbo4/tq4 [default: f16]
      --kv-tiers <TIERS>   Enable tiered KV cache: vram+ram, vram+ram+ssd [default: off]
      --kv-ram-budget <GB> RAM tier budget in GB, requires --kv-tiers [default: 50% of free RAM]
      --kv-ssd-path <PATH> SSD tier file path, requires --kv-tiers with ssd
      --kv-ssd-budget <GB> SSD tier budget in GB, requires --kv-tiers with ssd [default: 10]
      --host <ADDR>        Server bind address [default: 127.0.0.1]
      --api-key <KEY>      API key for server authentication (Bearer token)
      --prefill-batch-size <N> Prefill chunk size in tokens [default: 512]
      --no-color           Disable colored output (same as --color=never)
      --color <MODE>       Color mode: auto, always, never [default: auto]
      --kv-type-k <TYPE>   KV key quantization (overrides --kv-type)
      --kv-type-v <TYPE>   KV value quantization (overrides --kv-type)
  -V, --verbose            Show technical details (params, load times, EOG)
      --allow-cpu-fallback Allow GPU backends to fall back to CPU
  -d, --debug              Enable debug logging (token IDs, layer timing)
      --json               Output results as JSON (implies --quiet)
      --model-info         Print model metadata and exit (combine with --json)
      --profile            Profile per-op timing (halves throughput)
      --mmproj <PATH>      Path to vision projector GGUF (mmproj file)
      --image <PATH>       Path to image file for multimodal inference (PNG or PPM)
      --kv-eviction <MODE> KV cache eviction policy: none, norm, tri [default: none]
      --kv-budget <N>      Max KV entries to retain after eviction [default: 80% of ctx-size]
      --mmap               Use lazy mmap instead of preloading weights into RAM
```

## Build Options

All backends and models are enabled by default. Disable individually to reduce binary size or avoid unwanted dependencies.

```bash
# Disable specific backends
zig build -Denable-vulkan=false
zig build -Denable-cuda=false -Denable-rocm=false

# CPU-only build (no GPU backends)
zig build -Denable-metal=false -Denable-vulkan=false -Denable-cuda=false -Denable-rocm=false -Denable-webgpu=false

# GPU-only (disable CPU fallback — compile error if GPU init fails)
zig build -Denable-cpu=false

# Disable specific model architectures
zig build -Denable-glm4=false

# Minimal build: single model (Gemma 3) + single backend (Metal)
zig build -Denable-gemma4=false -Denable-qwen35=false -Denable-gpt-oss=false \
  -Denable-nemotron-h=false -Denable-nemotron-nano=false -Denable-glm4=false \
  -Denable-vulkan=false -Denable-cuda=false -Denable-rocm=false -Denable-webgpu=false

# Override GPU architecture targets
zig build -Dcuda-sm=sm_120        # Blackwell
zig build -Drocm-arch=gfx942      # MI300X

# Cross-compile
zig build -Dtarget=aarch64-linux-gnu -Denable-metal=false
```

**Backend Options:**

| Option | Type | Default | Purpose |
|--------|------|---------|---------|
| `enable-cpu` | bool | true | CPU backend |
| `enable-metal` | bool | true | Metal backend (macOS only) |
| `enable-vulkan` | bool | true | Vulkan backend (runtime dlopen) |
| `enable-cuda` | bool | true | CUDA backend (runtime dlopen) |
| `enable-rocm` | bool | true | ROCm backend (runtime dlopen) |
| `enable-webgpu` | bool | true | WebGPU backend (WGSL shaders) |
| `cuda-sm` | enum | sm_90 | CUDA SM target (sm_50..sm_120) |
| `rocm-arch` | enum | gfx1100 | ROCm GFX target (gfx90a..gfx1151) |

**Model Options:**

| Option | Type | Default | Purpose |
|--------|------|---------|---------|
| `enable-gemma3` | bool | true | Gemma 3 model support |
| `enable-gemma4` | bool | true | Gemma 4 model support |
| `enable-qwen35` | bool | true | Qwen 3.5 model support |
| `enable-gpt-oss` | bool | true | GPT-OSS model support |
| `enable-nemotron-h` | bool | true | Nemotron-H model support |
| `enable-nemotron-nano` | bool | true | Nemotron Nano model support |
| `enable-glm4` | bool | true | GLM-4 model support |

## Recipes

Recipes are optional preset configurations matched by architecture + backend + quantization. They provide proven defaults (temperature, top-p, context size, etc.) while allowing full user override via CLI flags.

```
# Recipe auto-applied, shown in banner:
🌵 agave Qwen3.5-0.8B Q4_0 Metal 32L/4096E/16H (45ms)
recipe: Qwen3.5 Q4 Metal

# User flags always take priority over recipe defaults:
./zig-out/bin/agave model.gguf -t 0  # overrides recipe temperature
```

Current presets: Qwen3.5 Q4 Metal, Gemma Q4 Metal, GPT-OSS Metal, GLM-4 generic, CPU generic. Add new recipes in `src/recipe.zig`.

## Project Structure

```
src/
├── main.zig           # CLI, format detection, model init, REPL, recipe application
├── arch.zig           # Architecture enum, detection, chat template mapping
├── pull.zig           # Model download from HuggingFace Hub (agave pull)
├── server/            # HTTP server
│   ├── server.zig     #   HTTP server (OpenAI + Anthropic API + chat UI)
│   ├── json.zig       #   Streaming JSON parser for API requests
│   ├── scheduler.zig  #   Continuous batching request scheduler
│   ├── metrics.zig    #   Prometheus metrics collector
│   └── rate_limiter.zig #  Token bucket rate limiter
├── display.zig        # Rich CLI output (banner, stats, progress)
├── chat_template.zig  # Data-driven chat prompt templates (ChatML, Gemma, Gemma4, Qwen3.5, GLM-4, GPT-OSS)
├── recipe.zig         # Optional preset configs per model/hardware/quant combo
├── thread_pool.zig    # Futex-based work-stealing thread pool
├── image.zig          # PNG/PPM image decoder and resize for multimodal inference
├── perf.zig           # Performance timer utilities
├── readline.zig       # Line editor for interactive REPL
├── micro_bench.zig    # Standalone micro-benchmark binary
├── format/            # Weight file loaders
│   ├── format.zig     #   Format interface (getTensor, getMetaStr, ...)
│   ├── gguf.zig       #   GGUF v2/v3 with mmap
│   └── safetensors.zig#   Multi-shard SafeTensors + config.json
├── models/            # Model architectures
│   ├── model.zig      #   Model interface + shared helpers (expertWeightStride, etc.)
│   ├── gemma3.zig     #   Gemma 3 (GQA, GELU, post-norms)
│   ├── gemma4.zig     #   Gemma 4 (MoE, dual attention, PLE)
│   ├── qwen35.zig     #   Qwen 3.5 (DeltaNet SSM hybrid)
│   ├── gpt_oss.zig    #   GPT-OSS (MoE, sliding window)
│   ├── nemotron_h.zig #   Nemotron-H (Mamba-2 hybrid, GGUF)
│   ├── nemotron_nano.zig # Nemotron Nano (SSM+MoE+attn, SafeTensors NVFP4)
│   ├── glm4.zig       #   GLM-4 MoE Lite (MLA, MoE)
│   └── vision.zig     #   SigLIP-2 vision encoder for multimodal models
├── ops/               # Shared compute kernels
│   ├── attention.zig  #   SDPA with SIMD + sliding window + backend dispatch
│   ├── math.zig       #   argmax, softplus, sigmoid, GELU, sampleToken
│   ├── ssm.zig        #   SSM ops: causal conv1d, Mamba-2 recurrence, group norm+gate
│   ├── quant.zig      #   Quantization helpers (bf16, mxfp4, fp8, iq4nl, nvfp4_st)
│   ├── kv_quant.zig   #   KV cache quantization (f32/f16/q8_0/int8/fp8/nvfp4)
│   ├── mlx.zig        #   MLX 4/6/8-bit affine dequant
│   └── split_attention.zig # Split-attention: async CPU-GPU KV offloading
├── backend/           # Hardware backends (all individually toggleable)
│   ├── backend.zig    #   Tagged union dispatcher + NullBackend stub
│   ├── cpu.zig        #   CPU with SIMD (AVX2, NEON, SVE)
│   ├── metal.zig      #   Metal GPU (Apple Silicon)
│   ├── vulkan.zig     #   Vulkan GPU (runtime dlopen)
│   ├── cuda.zig       #   CUDA GPU (runtime dlopen, Zig PTX kernels)
│   ├── rocm.zig       #   ROCm GPU (runtime dlopen)
│   ├── webgpu.zig     #   WebGPU (WGSL shaders, browser + native)
│   ├── objc.zig       #   Objective-C runtime bridge for Metal
│   └── kernels/       #   GPU shader/kernel sources
│       ├── metal/     #     MSL compute shaders
│       ├── vulkan/    #     SPIR-V compute shaders
│       ├── cuda/      #     Zig CUDA kernels (compiled to PTX)
│       ├── rocm/      #     AMDGCN kernels (compiled to HSACO)
│       └── webgpu/    #     WGSL compute shaders
├── kvcache/
│   ├── manager.zig    #   KV cache alloc/free, PagedKvCache, RadixTree
│   ├── block_allocator.zig # Block allocation for paged KV cache
│   ├── tiered.zig     #   Tiered KV cache (VRAM + RAM + SSD)
│   └── prefetch.zig   #   Async block prefetching for tiered cache
└── tokenizer/
    ├── tokenizer.zig  #   Tokenizer interface
    └── bpe.zig        #   BPE + SPM tokenizer

research/kernels/          # Kernel research (not part of main build)
├── reference.py           #   PyTorch reference implementations
├── generate_golden.py     #   Golden test data generator
├── autotune.py            #   Benchmarking and optimization orchestrator
├── registry.py            #   Kernel registry and search spaces
└── golden/                #   Generated .bin files for Zig @embedFile
```

## Docker

Build multi-platform images (x86_64 + aarch64) using `docker buildx`:

```bash
# Build for both platforms (all GPU backends enabled, glibc)
docker buildx build --platform linux/amd64,linux/arm64 -t agave .

# Build and load for current platform only
docker buildx build --load -t agave .

# CPU-only build (static musl binary, smaller image)
docker buildx build --load -t agave \
  --build-arg ENABLE_VULKAN=false \
  --build-arg ENABLE_CUDA=false \
  --build-arg ENABLE_ROCM=false .

# Minimal build: single model + CPU only
docker buildx build --load -t agave \
  --build-arg ENABLE_VULKAN=false \
  --build-arg ENABLE_CUDA=false \
  --build-arg ENABLE_ROCM=false \
  --build-arg ENABLE_QWEN35=false \
  --build-arg ENABLE_GPT_OSS=false \
  --build-arg ENABLE_NEMOTRON_H=false \
  --build-arg ENABLE_NEMOTRON_NANO=false \
  --build-arg ENABLE_GLM4=false .

# Run inference
docker run --rm -v /path/to/models:/models agave /models/model.gguf "Hello"

# Run HTTP server
docker run --rm -p 49453:49453 -v /path/to/models:/models agave /models/model.gguf --serve

# Override Zig version at build time
docker buildx build --build-arg ZIG_VERSION=0.16.0 -t agave .
```

GPU backends (CUDA, Vulkan, ROCm) load their drivers at runtime via `dlopen`, which requires glibc. When all three GPU backends are disabled, the build automatically switches to musl for a fully static binary. Zig cross-compiles natively — no QEMU emulation needed during build.

### Static musl builds

For environments where a fully static, dependency-free binary is needed (Alpine containers, embedded systems, minimal distros), disable all dlopen backends:

```bash
# Static musl binary — CPU backend only
zig build -Dtarget=x86_64-linux-musl \
  -Denable-metal=false -Denable-vulkan=false \
  -Denable-cuda=false -Denable-rocm=false

# Cross-compile static ARM64 binary
zig build -Dtarget=aarch64-linux-musl \
  -Denable-metal=false -Denable-vulkan=false \
  -Denable-cuda=false -Denable-rocm=false
```

**Note:** Static musl builds only work with the CPU backend. GPU backends (CUDA, Vulkan, ROCm) depend on `dlopen` to load vendor drivers at runtime, which requires glibc. Attempting to dlopen a glibc-linked `.so` from a musl binary will segfault.

## Documentation

- **[Tutorial: LLM Inference From Scratch](docs/tutorial/README.md)** — 17-chapter progressive tutorial + 4 appendixes
- **[Architecture](docs/ARCHITECTURE.md)** — Project structure, module reference, inference pipeline
- **[Models](docs/MODELS.md)** — Supported models, parameters, per-model details
- **[Benchmarks](docs/BENCHMARKS.md)** — Performance comparisons vs llama.cpp
- **[Kernel Status](docs/KERNELS.md)** — Per-backend kernel implementation status
- **[Contributing](docs/CONTRIBUTING.md)** — How to add backends, models, quantization
- **[CLAUDE.md](CLAUDE.md)** — Engineering standards for contributors
- **[research/kernels/](research/kernels/)** — Kernel research tools (benchmarks, golden tests)

## License

GNU General Public License v3.0
