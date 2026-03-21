# Technology Stack

**Analysis Date:** 2026-03-21

## Languages

**Primary:**
- **Zig** (latest) - Core inference engine, all compute kernels, HTTP server, and model implementations
  - Used in: `src/**/*.zig` (all source files except embedded Metal shaders)

**Kernel Shaders:**
- **Metal Shading Language (MSL)** - GPU compute kernels for Apple Silicon (M-series)
  - Used in: `src/backend/kernels/metal/**/*.metal`
  - Compiled at runtime via Metal API

- **CUDA PTX (Parallel Thread Execution)** - NVIDIA GPU kernels, cross-compiled to PTX IR
  - Used in: `src/backend/kernels/cuda/**/*.zig`
  - Cross-compiled via `zig build ptx` with `nvptx64-cuda` target
  - Embedded as `.ptx` assembly files

- **AMDGCN ISA / HSACO** - AMD ROCm GPU kernels
  - Used in: `src/backend/kernels/rocm/**/*.zig`
  - Cross-compiled via `zig build amdgcn` with `amdgcn-amdhsa` target
  - Linked into `.hsaco` (HSA Code Object) for HIP runtime

- **SPIR-V** - Vulkan compute shader IR
  - Used in: `src/backend/kernels/vulkan/**/*.comp`
  - Compiled to binary SPIR-V and embedded into Vulkan backend

## Runtime

**Host Environment:**
- **Zig Standard Library (std)** - All runtime services
  - Memory management: `std.mem.Allocator`, `std.heap.FixedBufferAllocator`, `std.heap.ArenaAllocator`
  - I/O: `std.io`, `std.fs` (mmap via `std.os.mmap`, file reading)
  - Networking: `std.net` (TCP sockets, HTTP parsing, OpenAI-compatible API)
  - POSIX: `std.posix` (memory hints via `madvise`, signal handling for Ctrl+C, CPU affinity)
  - Threading: `std.Thread` (thread spawning, thread-local storage)
  - Atomics: `std.atomic` (lock-free synchronization)
  - Hashing: `std.hash` (hash maps for buffer caching)
  - Collections: `std.ArrayList`, `std.StringHashMap`

**Target Platforms:**
- Linux x86_64, Linux aarch64, macOS aarch64 (Apple Silicon)

## Frameworks

**Core Inference:**
- **Zig-native kernel implementations** - All GEMV, SDPA, quantization, activation kernels written in Zig or native IR
  - No external inference library dependencies (no libggml, no ONNX, no TensorFlow)

**HTTP Server:**
- **std.net with per-connection threads** - Synchronous HTTP/1.1 server
  - Handler threads spawned per TCP connection
  - OpenAI-compatible API endpoints (`/v1/chat/completions`, `/v1/completions`, etc.)
  - SSE (Server-Sent Events) streaming for token generation

**CLI Argument Parsing:**
- **clap** (0.11.0) - Command-line argument parser
  - Used for: `--model`, `--serve`, `--port`, `--backend`, `--prompt`, `--ctx-size`, etc.
  - Source: https://github.com/Hejsil/zig-clap/archive/refs/tags/0.11.0.tar.gz

**Terminal UI:**
- **vaxis** (0.5.1, commit 41fff922) - Cross-platform terminal rendering
  - Used for: Interactive CLI mode with readline-style input
  - Provides: Cursor positioning, color output, terminal mode control
  - Source: git+https://github.com/rockorager/libvaxis.git

## Key Dependencies

**Critical (Vendored via build.zig.zon):**
- `clap` 0.11.0 - Command-line argument parsing library for Zig
  - Hash: `clap-0.11.0-oBajB-HnAQDPCKYzwF7rO3qDFwRcD39Q0DALlTSz5H7e`
  - Purpose: CLI interface

- `vaxis` 0.5.1 - Terminal/TUI rendering library for Zig
  - Hash: `vaxis-0.5.1-BWNV_JJOCQAtdJyLvrYCKbKIhX9q3liQkKMAzujWS4HJ`
  - Commit: 41fff922316dcb8776332ec460e73eaf397d5033
  - Purpose: Interactive terminal mode

**Dynamic Runtime Dependencies (loaded via dlopen):**
- **libcuda.so** / **libcuda.dylib** (optional) - NVIDIA CUDA runtime
  - Loaded dynamically via `std.DynLib.open()`
  - Used only if CUDA backend enabled and available
  - No link-time dependency; graceful fallback to CPU

- **libamdhip64.so** (optional) - AMD ROCm HIP runtime
  - Loaded dynamically
  - Used only if ROCm backend enabled and available
  - Graceful fallback to CPU

- **libMoltenVK.dylib** / **libvulkan.so** (optional) - Vulkan ICD loader
  - Loaded dynamically
  - Used only if Vulkan backend enabled and available
  - Graceful fallback to CPU

- **Metal framework** (Apple Silicon only, link-time) - Metal API for GPU acceleration
  - Linked at compile-time on macOS via `linkFramework("Metal")` and `linkFramework("Foundation")`
  - Used via ObjC runtime bindings in `src/backend/objc.zig`

**Model Format Support:**
- **GGUF (GPT-Generated Unified Format)** - Native mmap-based loading
  - Supports quantization formats: Q2, Q3, Q4, Q5, Q6, Q8, IQ4, FP8, bf16, NVFP4, MXFP4
  - Zero-copy via `std.os.mmap` on all platforms

- **SafeTensors** - Native loader with multi-shard support
  - Supports: MLX quantization, MLX 4-bit, standard float formats
  - Zero-copy via `std.os.mmap` per shard
  - Parses JSON headers in-memory, resolves tensor offsets

## Configuration

**Build Configuration:**
- **build.zig** - Pure Zig build system (no CMake, no Makefiles)
  - Backend flags (all default to true, can disable with `-Denable-backend=false`):
    - `-Denable-cpu` - CPU backend (always enabled)
    - `-Denable-metal` - Metal GPU backend (macOS only)
    - `-Denable-cuda` - CUDA backend (requires CUDA installed)
    - `-Denable-rocm` - ROCm backend (requires ROCm installed)
    - `-Denable-vulkan` - Vulkan backend (requires Vulkan SDK)

  - Model flags (all default to true):
    - `-Denable-gemma3`, `-Denable-qwen35`, `-Denable-gpt-oss`, `-Denable-nemotron-h`, `-Denable-nemotron-nano`, `-Denable-glm4`

  - CUDA SM target (default: sm_90): `-Dcuda-sm=sm_80|sm_90|sm_120`
  - ROCm GFX target (default: gfx1100): `-Drocm-arch=gfx90a|gfx1100|gfx1150`

**Build Modes:**
- **ReleaseFast** - Primary production target, LLVM optimized
- **Debug** - Full safety checks retained

**Build Artifacts:**
- `zig-out/bin/agave` - Main inference engine (ReleaseFast)
- `zig-out/bin/agave-debug` - Debug build
- `zig-out/bin/agave-bench` - Standalone micro-benchmark tool
- `zig-out/ptx/*.ptx` - CUDA kernel PTX assembly (built via `zig build ptx`)
- `zig-out/rocm/kernels.hsaco` - ROCm kernel object (built via `zig build amdgcn`)

**Runtime Configuration:**
- Model weights loaded from GGUF file or SafeTensors directory (user-provided path)
- HTTP server config: port (default 49453), host (0.0.0.0)
- KV cache size: context window in tokens (default 4096)
- Inference backend: auto-selected from available hardware, override with CLI flag
- Chat template: auto-detected by model architecture, configurable via recipe system

## Platform Requirements

**Development:**
- **Zig compiler** (latest development version)
- **CUDA 12.0+** (optional, for CUDA backend)
- **ROCm 5.7+** (optional, for ROCm backend)
- **Vulkan SDK** (optional, for Vulkan backend)
- **Metal SDK** (macOS only, comes with Xcode)

**Runtime:**
- **macOS (Apple Silicon M-series)**: Metal framework (built-in)
- **Linux (x86_64, aarch64)**:
  - For CUDA: NVIDIA driver + CUDA runtime library
  - For ROCm: AMD GPU driver + HIP runtime
  - For Vulkan: Vulkan-capable driver + loader
  - CPU-only: no additional requirements

**Minimum Hardware:**
- **CPU**: x86_64 or aarch64 with SSE2/NEON
- **RAM**: 4GB for quantized models, 16GB+ for larger models
- **GPU** (optional):
  - Apple Silicon (Metal): M1+
  - NVIDIA: Kepler+ (sm_30+), tested on Blackwell (sm_121)
  - AMD: RDNA+ (gfx1100+)

---

*Stack analysis: 2026-03-21*
