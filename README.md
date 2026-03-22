# Agave

A high-performance LLM inference engine written in Zig. Zero external ML libraries — all kernels, quantization, and model logic implemented from scratch.

## Features

- **6 Model Architectures**: Gemma 3, Qwen 3.5, GPT-OSS, Nemotron-H, Nemotron Nano, GLM-4
- **5 Backends**: CPU (SIMD-optimized), Metal GPU (Apple Silicon), Vulkan, CUDA, ROCm — individually toggleable at build time
- **Compile-Time Model Selection**: Disable unused model architectures to reduce binary size (1.8 MB → 0.75 MB with all models stripped)
- **2 Formats**: GGUF, SafeTensors (multi-shard, MLX quantized, NVFP4)
- **19 Quantization Types**: F32, F16, BF16, Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, IQ4_XS, IQ4_NL, FP8 E4M3, FP8 E5M2, NVFP4, MXFP4, MLX 4/6-bit
- **Chat Templates**: Data-driven per-architecture prompt formatting (ChatML, Gemma, GPT-OSS)
- **Recipes**: Optional proven-default configs per model/hardware/quant combo
- **Interactive REPL**: Multi-turn chat with `/help`, `/clear`, `/stats`, `/quit`
- **HTTP Server**: OpenAI-compatible API + htmx chat UI
- **~34 tok/s** on Gemma 3 1B Q8_0 (Metal, Apple Silicon)

## Quick Start

```bash
# Build (produces both ReleaseFast and Debug binaries)
zig build

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

# Override recipe defaults (user flags always win)
./zig-out/bin/agave model.gguf -t 0.9 --top-p 0.95 "Tell me a story"
```

## Supported Models

| Model | Status | Quant Types | Notes |
|-------|--------|-------------|-------|
| Gemma 3 | Working | BF16, Q8_0, Q4_0, Q6_K, MLX 4-bit | SPM tokenizer, GELU activation |
| Qwen 3.5 | Working | Q4_0, BF16 | Hybrid DeltaNet SSM + attention |
| GPT-OSS | Working | Q4_0 | MoE, sliding window, attention sinks |
| Nemotron-H | Working | Q5_0 | Mamba-2 + attention hybrid (GGUF) |
| Nemotron Nano 30B | In Progress | NVFP4 + BF16 | SSM + MoE + attention hybrid (SafeTensors); tokenizer decode fixed, MoE routing needs tuning |
| GLM-4 MoE Lite | Partial | MLX 4/6-bit | MLA + MoE (needs debugging) |

## Prerequisites

- **Zig 0.15.2**
- macOS (Metal backend) / Linux (Vulkan, CUDA, ROCm) / any platform (CPU backend)
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
  -t, --temperature <T>    Sampling temperature [default: 0] (greedy only, see note)
      --top-p <P>          Nucleus sampling [default: 1.0] (not yet implemented)
      --top-k <K>          Top-k sampling [default: 0] (not yet implemented)
      --repeat-penalty <R> Repetition penalty [default: 1.0]
      --system <TEXT>      System prompt for chat formatting
      --backend <BE>       auto, cpu, metal, vulkan, cuda, rocm [default: auto]
      --ctx-size <N>       Context window size, 0 = model default [default: 0]
      --no-color           Disable ANSI colors
      --verbose            Show technical details (params, load times, EOG)
      --allow-cpu-fallback Allow GPU backends to fall back to CPU
      --debug              Enable debug logging (token IDs, layer timing)
      --json               Output results as JSON (implies --quiet)
      --model-info         Print model metadata and exit (combine with --json)
      --profile            Profile per-op timing (halves throughput)
      --mmap               Use lazy mmap instead of preloading weights into RAM
```

> **Note**: Sampling is currently greedy only (argmax). The `--temperature`, `--top-p`, and `--top-k` flags are parsed but not yet applied — all generation uses temperature=0 regardless of the values provided.

## Build Options

All backends and models are enabled by default. Disable individually to reduce binary size or avoid unwanted dependencies.

```bash
# Disable specific backends
zig build -Denable-vulkan=false
zig build -Denable-cuda=false -Denable-rocm=false

# CPU-only build (no GPU backends)
zig build -Denable-metal=false -Denable-vulkan=false -Denable-cuda=false -Denable-rocm=false

# GPU-only (disable CPU fallback — compile error if GPU init fails)
zig build -Denable-cpu=false

# Disable specific model architectures
zig build -Denable-glm4=false

# Minimal build: single model + single backend
zig build -Denable-qwen35=false -Denable-gpt-oss=false -Denable-nemotron-h=false \
  -Denable-nemotron-nano=false -Denable-glm4=false \
  -Denable-vulkan=false -Denable-cuda=false -Denable-rocm=false

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
| `cuda-sm` | enum | sm_90 | CUDA SM target (sm_50..sm_120) |
| `rocm-arch` | enum | gfx1100 | ROCm GFX target (gfx90a..gfx1151) |

**Model Options:**

| Option | Type | Default | Purpose |
|--------|------|---------|---------|
| `enable-gemma3` | bool | true | Gemma 3 model support |
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

Current presets: Qwen Q4 Metal, Gemma Q4 Metal, GPT-OSS Metal, CPU generic. Add new recipes in `src/recipe.zig`.

## Project Structure

```
src/
├── main.zig           # CLI, format detection, model init, REPL, recipe application
├── arch.zig           # Architecture enum, detection, chat template mapping
├── server.zig         # HTTP server (OpenAI API + chat UI)
├── display.zig        # Rich CLI output (banner, stats, progress)
├── chat_template.zig  # Data-driven chat prompt templates (ChatML, Gemma, GPT-OSS)
├── recipe.zig         # Optional preset configs per model/hardware/quant combo
├── thread_pool.zig    # Futex-based work-stealing thread pool
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
│   ├── qwen35.zig     #   Qwen 3.5 (DeltaNet SSM hybrid)
│   ├── gpt_oss.zig    #   GPT-OSS (MoE, sliding window)
│   ├── nemotron_h.zig #   Nemotron-H (Mamba-2 hybrid, GGUF)
│   ├── nemotron_nano.zig # Nemotron Nano (SSM+MoE+attn, SafeTensors NVFP4)
│   └── glm4.zig       #   GLM-4 MoE Lite (MLA, MoE)
├── ops/               # Shared compute kernels
│   ├── attention.zig  #   SDPA with SIMD + sliding window
│   ├── math.zig       #   argmax, softplus, sigmoid, GELU
│   ├── ssm.zig        #   SSM ops: causal conv1d, Mamba-2 recurrence, group norm+gate
│   ├── quant.zig      #   Quantization helpers (bf16, mxfp4, fp8, iq4nl, nvfp4_st)
│   └── mlx.zig        #   MLX 4/6-bit affine dequant
├── backend/           # Hardware backends (all individually toggleable)
│   ├── backend.zig    #   Tagged union dispatcher + NullBackend stub
│   ├── cpu.zig        #   CPU with SIMD (AVX2, NEON, SVE)
│   ├── metal.zig      #   Metal GPU (Apple Silicon)
│   ├── vulkan.zig     #   Vulkan GPU (runtime dlopen)
│   ├── cuda.zig       #   CUDA GPU (runtime dlopen, Zig PTX kernels)
│   ├── rocm.zig       #   ROCm GPU (runtime dlopen)
│   ├── objc.zig       #   Objective-C runtime bridge for Metal
│   └── kernels/       #   GPU shader/kernel sources
│       ├── metal/     #     MSL compute shaders
│       ├── vulkan/    #     SPIR-V compute shaders
│       ├── cuda/      #     Zig CUDA kernels (compiled to PTX)
│       └── rocm/      #     AMDGCN kernels (compiled to HSACO)
├── kvcache/
│   └── manager.zig    #   KV cache alloc/free
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
docker run --rm -p 8080:8080 -v /path/to/models:/models agave /models/model.gguf --serve

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

- **[DOCUMENTATION.md](docs/DOCUMENTATION.md)** — Concepts, architecture, module reference
- **[CLAUDE.md](CLAUDE.md)** — Engineering standards for contributors
- **[research/kernels/](research/kernels/)** — Kernel research tools (benchmarks, golden tests, optimization driver)

## License

GNU General Public License v3.0
