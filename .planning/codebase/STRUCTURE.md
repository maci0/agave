# Codebase Structure

**Analysis Date:** 2025-03-21

## Directory Layout

```
agave/
├── src/                         # Core engine source
│   ├── main.zig                 # CLI entry point, generation loop, REPL
│   ├── arch.zig                 # Architecture enum (gemma3, qwen35, etc.)
│   ├── chat_template.zig        # Chat prompt templates (data-driven)
│   ├── recipe.zig               # Preset configs per arch+backend+quant
│   ├── display.zig              # Output formatting (banner, stats, JSON)
│   ├── readline.zig             # Interactive line editor with history
│   ├── server.zig               # HTTP server (OpenAI-compatible API)
│   ├── thread_pool.zig          # Futex-based work-stealing thread pool
│   ├── perf.zig                 # Performance profiling instrumentation
│   ├── micro_bench.zig          # Benchmark CLI tool
│   │
│   ├── backend/                 # Compute backends
│   │   ├── backend.zig          # Dispatcher: union(enum) with inline else
│   │   ├── cpu.zig              # CPU backend (SIMD via std and @Vector)
│   │   ├── metal.zig            # Metal GPU backend (Apple only)
│   │   ├── vulkan.zig           # Vulkan GPU backend (cross-platform)
│   │   ├── cuda.zig             # CUDA GPU backend (NVIDIA)
│   │   ├── rocm.zig             # ROCm GPU backend (AMD)
│   │   ├── objc.zig             # Objective-C runtime bindings (Metal)
│   │   │
│   │   └── kernels/             # GPU kernel source (comptime-embedded or loaded)
│   │       ├── cpu/             # CPU kernel implementations (Zig)
│   │       │   ├── gemv.zig, gemv_q4_0.zig, ... (18 GEMV variants)
│   │       │   ├── norm.zig     # RMS norm, L2 norm
│   │       │   ├── rope.zig     # Rotary position embedding
│   │       │   ├── sdpa.zig     # Scaled dot-product attention
│   │       │   ├── softmax.zig  # Softmax reduction
│   │       │   ├── activation.zig # SiLU, GELU
│   │       │   ├── elementwise.zig # add, mul, sigmoid*mul
│   │       │   ├── embedding.zig # Token embedding lookup
│   │       │   └── deltanet.zig # DeltaNet SSM (Qwen3.5)
│   │       │
│   │       ├── metal/           # Metal shaders (MSL, comptime-embedded)
│   │       │   ├── gemv.metal   # GEMV kernels (all quant formats)
│   │       │   ├── sdpa.metal   # SDPA (unused, falls back to CPU)
│   │       │   ├── norm.metal   # RMS norm, L2 norm
│   │       │   ├── rope.metal   # RoPE
│   │       │   ├── elementwise.metal # Activations, arithmetic, deinterleave
│   │       │   └── common.metal # Shared helpers (simd_sum, fp8 conversion)
│   │       │
│   │       ├── cuda/            # CUDA kernels (Zig compiled to PTX)
│   │       │   ├── gemv_*.zig   # GEMV variants (f32, q4_0, q8_0, etc.)
│   │       │   ├── common.zig   # PTX helpers, warp reduction, shared mem
│   │       │   ├── all.zig      # Aggregator: concatenates all PTX
│   │       │   ├── all.ptx      # Compiled PTX (committed binary)
│   │       │   └── other ops...
│   │       │
│   │       ├── vulkan/          # Vulkan SPIR-V kernels (committed binaries)
│   │       │   ├── *.comp.spv   # 17 compute shaders (pre-compiled)
│   │       │   └── Makefile     # Build rules for SPIR-V
│   │       │
│   │       └── rocm/            # ROCm kernels (placeholder for future)
│   │
│   ├── models/                  # Model architectures
│   │   ├── model.zig            # Model interface (comptime vtable)
│   │   ├── gemma3.zig           # Gemma 3 (standard transformer + SafeTensors support)
│   │   ├── qwen35.zig           # Qwen 3.5 (hybrid DeltaNet + attention)
│   │   ├── gpt_oss.zig          # GPT-OSS (transformer + mixture-of-experts)
│   │   ├── nemotron_h.zig       # Nemotron-H (hybrid Mamba-2 + attention)
│   │   ├── nemotron_nano.zig    # Nemotron-Nano (hybrid Mamba-2 + MoE + attention)
│   │   └── glm4.zig             # GLM-4 (MLA attention + MoE)
│   │
│   ├── ops/                     # Shared operations (algorithm implementations)
│   │   ├── attention.zig        # SDPA with optional sliding window
│   │   ├── ssm.zig              # Causal conv1d, Mamba-2 recurrence, group RMS+gate
│   │   ├── math.zig             # GELU, softplus, sigmoid, sampling, topK
│   │   ├── quant.zig            # Quantization helpers (dequant, bf16, FP8, NVFP4, MXFP4, IQ4)
│   │   ├── kv_quant.zig         # KV cache quantization interface
│   │   └── mlx.zig              # MLX affine quantization (SafeTensors support)
│   │
│   ├── kvcache/                 # KV cache management
│   │   └── manager.zig          # Flat, PagedAttention, RadixAttention (prefix tree)
│   │
│   ├── format/                  # Model format loaders
│   │   ├── format.zig           # Format interface (dispatcher)
│   │   ├── gguf.zig             # GGUF file loader (mmap + index)
│   │   └── safetensors.zig      # SafeTensors directory loader (sharded)
│   │
│   └── tokenizer/               # Text ↔ token ID conversion
│       ├── tokenizer.zig        # Tokenizer interface
│       └── bpe.zig              # BPE and SentencePiece implementations
│
├── build.zig                    # Build system configuration
├── build.zig.zon                # Dependency manifest (version constraints)
├── Dockerfile                   # Container image
├── README.md                    # Usage guide
├── CLAUDE.md                    # Engineering standards (this codebase)
├── docs/                        # Documentation
│   ├── DOCUMENTATION.md         # Core concepts, algorithms, formulas
│   ├── IDEAS.md                 # Future improvements
│   └── KERNELS.md               # Kernel implementation guide
├── research/                    # Experimental prototyping (not in build)
│   └── kernels/                 # Triton, CUTLASS, reference impls
└── models/                      # Local model cache (gitignored)
    └── {org}/{repo}/            # HF-style directory layout
```

## Directory Purposes

**`src/`** - All engine source code
- Main CLI, model dispatch, format loading, tokenization
- Backend implementations (CPU + 4 GPU backends)
- GPU kernel source (Zig, MSL, PTX, SPIR-V)
- 6 model architectures (conditionally compiled)
- Shared ops: attention, SSM, math, quantization
- KV cache managers: flat, paged, radix tree
- Supporting: chat templates, recipes, thread pool, display

**`src/backend/`** - Compute abstraction layer
- `backend.zig`: Tagged union dispatcher + common interface
- `cpu.zig`: Pure Zig SIMD kernels + layout helpers
- `metal.zig`: Apple GPU (MSL shaders), ObjC bridge
- `vulkan.zig`: Cross-platform GPU (SPIR-V shaders, pre-compiled)
- `cuda.zig`: NVIDIA GPU (PTX kernels, dlopen CUDA runtime)
- `rocm.zig`: AMD GPU (placeholder)
- `objc.zig`: ObjC runtime bindings for Metal device/buffer API

**`src/backend/kernels/`** - GPU kernel source
- `cpu/`: 25+ Zig files for GEMV (18 quantization formats), SDPA, activations, normalization, RoPE, elementwise, embedding, DeltaNet
- `metal/`: MSL compute shaders (comptime-embedded as strings in `metal.zig`)
- `cuda/`: Zig source compiled to PTX via `nvptx64-cuda` target; `all.ptx` is committed binary
- `vulkan/`: Pre-compiled SPIR-V shaders (.spv files), Makefile for building from source

**`src/models/`** - Model architectures
- `model.zig`: Vtable interface for polymorphic dispatch
- `gemma3.zig`: Gemma 3 (SafeTensors + GGUF support, no SSM)
- `qwen35.zig`: Qwen 3.5 (hybrid DeltaNet + full attention every N layers)
- `gpt_oss.zig`: GPT-OSS (MoE routing, sliding window attention, MLP-only layers)
- `nemotron_h.zig`: Nemotron-H (hybrid pattern: M=SSM, E=MoE, *=attention)
- `nemotron_nano.zig`: Nemotron-Nano (52-layer hybrid, 128 routed experts, top-6 routing)
- `glm4.zig`: GLM-4 (MLA attention, MoE routing)

**`src/ops/`** - Shared algorithm implementations
- `attention.zig`: SDPA with optional sliding window + KV cache quantization
- `ssm.zig`: Causal conv1d, Mamba-2 recurrence, group RMS norm + SiLU gate
- `math.zig`: GELU, softplus, sigmoid, sampling (temperature, top-k, top-p), topK selection, argmax
- `quant.zig`: Dequantization helpers (BF16, FP8, NVFP4, MXFP4, IQ4_NL)
- `kv_quant.zig`: KV cache quantization enum (f32, f16, q8_0, fp8, nvfp4)
- `mlx.zig`: MLX affine quantization (SafeTensors companion scales/biases)

**`src/kvcache/`** - Memory management for attention state
- `allocKvCache()`: Simple flat per-layer slices
- `PagedKvCache`: Block-based allocation (vLLM PagedAttention)
- `RadixTree`: Prefix tree with LRU tracking (SGLang-style prefix sharing)

**`src/format/`** - Model file loaders
- `format.zig`: Vtable interface (getTensor, getMetaStr, getMetaU32, etc.)
- `gguf.zig`: GGUF v2/v3 file format (mmap'd, indexed)
- `safetensors.zig`: SafeTensors sharded tensors (JSON metadata + binary shards)

**`src/tokenizer/`** - Text ↔ token conversion
- `tokenizer.zig`: Vtable interface (encode, decode, vocabSize)
- `bpe.zig`: BPE (byte-pair merges), SPM (SentencePiece), SPM-no-dummy (Gemma3)

**`build.zig`** - Zig build system
- Defines build targets: `zig build` (binary), `zig build test`, `zig build ptx` (CUDA)
- Configurable backends: `-Denable-metal=true`, `-Denable-cuda=true`, etc.
- Model toggles: `-Denable-gemma3=false` (disables at compile time)
- Dependency management: `build.zig.zon` pins transitive versions

**`docs/`** - Technical documentation
- `DOCUMENTATION.md`: Core concepts (RoPE, attention, quantization, formulas)
- `IDEAS.md`: Future improvements
- `KERNELS.md`: Kernel implementation reference

## Key File Locations

**Entry Points:**
- `src/main.zig`: CLI parsing, model loading, REPL, one-shot generation, format detection (lines 704–1038)
- `src/server.zig`: HTTP server (OpenAI `/v1/chat/completions` endpoint)

**Configuration & Architecture:**
- `src/arch.zig`: Model arch enum + detection + build flags
- `src/chat_template.zig`: Chat prompt templates (role markers, EOG tokens)
- `src/recipe.zig`: Optional preset configs (arch + backend + quant matching)

**Core Logic:**
- `src/backend/backend.zig`: Backend dispatcher (tagged union, ~300 lines)
- `src/models/model.zig`: Model vtable interface (~200 lines)
- `src/format/format.zig`: Format dispatcher (~160 lines)
- `src/tokenizer/tokenizer.zig`: Tokenizer interface (~80 lines)

**Testing & Profiling:**
- `src/micro_bench.zig`: Micro-benchmark harness (individual kernels)
- `src/perf.zig`: Per-layer timing instrumentation

## Naming Conventions

**Files:**
- Zig source: `snake_case.zig` (e.g., `kv_quant.zig`, `gemv_q4_0.zig`)
- Shaders: `operation.metal` (e.g., `rope.metal`, `gemv.metal`)
- Build artifacts: `all.ptx` (CUDA PTX concatenation)
- Models directory: `{org}/{repo}/` (HuggingFace-style path)

**Directories:**
- `snake_case` (e.g., `src/backend`, `src/kvcache`, `src/backend/kernels/cpu`)
- Backend targets: `{backend_name}/` (e.g., `cuda/`, `metal/`)

**Functions:**
- `camelCase` (e.g., `allocKvCache`, `gemv`, `rmsNorm`, `scaledDotProductAttention`)
- Operations: `{op}` (e.g., `rope`, `softmax`, `sdpa`)
- Initialization: `init` (returns instance or error), `deinit` (cleanup)
- Getters: `get{Field}` (e.g., `getTensor`, `getLogits`)

**Types & Structs:**
- `PascalCase` (e.g., `Backend`, `Model`, `CpuBackend`, `MetalBackend`, `KvCache`, `RadixTree`)
- Error sets: `PascalErrorSet` (e.g., `ForwardError`, `TokenizerError`)

**Constants & Variables:**
- Named consts: `snake_case` (e.g., `default_ctx_size`, `simd_width`, `radix_fanout`)
- Abbreviations (comptime-safe): `hd` (head_dim), `nkv` (n_kv_heads), `kvd` (kv_dim)

## Where to Add New Code

**New Feature (Model Layer):**
- Implementation: `src/models/{new_model}.zig`
- Register: Add variant to `Arch` enum in `src/arch.zig`, add dispatcher in `src/main.zig: initAndRun()`
- Build: Add `-Denable-{new_model}` flag in `build.zig`
- Config: Add chat template to `src/chat_template.zig`

**New Backend:**
- Implementation: `src/backend/{new_backend}.zig`
- Kernels: `src/backend/kernels/{new_backend}/` (with all required operations)
- Register: Add variant to `Backend` union in `src/backend/backend.zig`
- Build: Add `-Denable-{new_backend}` flag in `build.zig`
- Dispatch: Update `BackendState.init()` in `src/main.zig` with fallback logic

**New Operation (Shared):**
- Pure algorithm: `src/ops/{operation}.zig`
- Backend-specific: Add method to all 5 backends in their respective files
- Register: Add dispatch method to `Backend` union in `src/backend/backend.zig`
- Use: Call via `be.{operation}(...)` in models or other ops

**New Quantization Format:**
- Format definition: `src/format/format.zig: DType` enum
- Dequant kernel (CPU): Add to `src/ops/quant.zig`
- Dequant kernel (GPU): Add to each backend's GEMV implementation
- Layout helpers: Add utility functions if format has special packing

**New CLI Feature:**
- Argument: Add to `cli_params` clap string in `src/main.zig`
- Parse: Add field to `CliArgs` struct
- Use: Access via `cli.{field}` in generation functions

**Tests:**
- Unit tests: Add `test "name" { ... }` at bottom of relevant module
- Run: `zig build test`
- Integration: Use `std.testing.allocator` for memory leak detection

## Special Directories

**`src/backend/kernels/`:**
- Purpose: GPU kernel source (Zig for CPU/CUDA, MSL for Metal, SPIR-V for Vulkan)
- Generated: Zig source compiled to PTX/SPIR-V at build time
- Committed: Precompiled PTX (.ptx) and SPIR-V (.spv) binaries in git
- Not generated at runtime: Kernels embedded in binary (no .ptx/.spv files needed at runtime on target)

**`research/`:**
- Purpose: Experimental prototypes (Triton, CUTLASS, reference implementations)
- Status: Not part of main build (`build.zig` does not include)
- Gitignored: `.gitignore` excludes `research/`
- Documentation: `research/kernels/README.md` explains prototyping pipeline

**`models/`:**
- Purpose: Local cache of downloaded GGUF/SafeTensors files
- Structure: `models/{org}/{repo}/` (HuggingFace-style)
- Gitignored: `.gitignore` excludes `models/`
- Access: `./zig-out/bin/agave models/meta-llama/llama2-7b/model.gguf`

---

*Structure analysis: 2025-03-21*
