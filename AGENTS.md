# Inference Engine Engineering Standards

**Last Updated**: 2026-03-20
**Status**: Active Development (Level 1 target)
**For**: All contributors and AI coding agents

All code generated, refactored, or reviewed within this repository must adhere to the following standards. This project is a high-performance LLM inference engine optimized for extreme cross-platform portability and zero-cost abstraction.

---

## Quick Reference

- **Language**: Zig (following Zig Zen - run `zig zen` to see principles)
- **Target**: High-performance LLM inference engine
- **Key Constraints**: Zero allocations/locks/syscalls in hot path
- **Supported Backends**: CUDA, ROCm, Metal, Vulkan, CPU
- **Quantization**: Q2-Q8, FP8, bf16, NVFP4, MXFP4
- **KV Cache**: RadixAttention preferred (PagedAttention minimum)
- **Build System**: Pure `build.zig` + `build.zig.zon` (no external C/C++ libs)

**Commands:**
```bash
zig build                                      # Build (ReleaseFast + Debug)
zig build test                                 # Run all tests
./zig-out/bin/agave model.gguf "prompt"        # Run inference
./zig-out/bin/agave model.gguf --serve         # HTTP server
./zig-out/bin/agave model.gguf --backend cpu   # Force CPU backend
zig build -Denable-glm4=false                  # Disable a model at compile time
```

**See also:**
- [build.zig](build.zig) - Build configuration
- [src/backend/backend.zig](src/backend/backend.zig) - Core dispatcher implementation
- [src/kvcache/manager.zig](src/kvcache/manager.zig) - KV cache management
- [src/ops/quant.zig](src/ops/quant.zig) - Quantization implementations
- [src/ops/attention.zig](src/ops/attention.zig) - SDPA and paged attention kernels
- [src/ops/ssm.zig](src/ops/ssm.zig) - SSM ops: causal conv1d, Mamba-2 recurrence, group RMS norm + SiLU gate
- [src/ops/math.zig](src/ops/math.zig) - Shared math ops (GELU, softplus, sigmoid, argmax)
- [src/chat_template.zig](src/chat_template.zig) - Chat prompt templates
- [src/recipe.zig](src/recipe.zig) - Preset configurations per model/hardware combo
- [Section 14: Implementation Quick Reference](#14-implementation-quick-reference) - File locations, model parameters, implementation gotchas
- [DOCUMENTATION.md](docs/DOCUMENTATION.md) - Core concepts, algorithm details, mathematical formulations

---

## Glossary

- **TTFT**: Time-to-First-Token
- **HSA**: Heterogeneous System Architecture (AMD ROCm runtime)
- **MSL**: Metal Shading Language
- **MPS**: Metal Performance Shaders
- **SPIR-V**: Standard Portable Intermediate Representation (Vulkan shaders)
- **PTX**: Parallel Thread Execution (NVIDIA CUDA IR)
- **comptime**: Compile-time (Zig's compile-time execution feature)
- **VRAM**: Video RAM (GPU memory)
- **IR**: Intermediate Representation
- **GEMM**: General Matrix Multiply (batch, compute-bound — used during prefill)
- **GEMV**: General Matrix-Vector multiply (single token, bandwidth-bound — used during decode)
- **Q/K/V**: Query, Key, Value — the three projections in attention (see [DOCUMENTATION.md](docs/DOCUMENTATION.md#attention))
- **GQA**: Grouped Query Attention — fewer KV heads shared across Q heads
- **MHA**: Multi-Head Attention — independent parallel attention mechanisms
- **FFN**: Feed-Forward Network — the MLP sub-block in each transformer layer
- **SwiGLU**: Gated FFN variant: `SiLU(W_gate @ x) * (W_up @ x)`
- **MoE**: Mixture of Experts — sparse FFN with learned routing
- **SSM**: State-Space Model — fixed-size recurrent alternative to attention
- **SDPA**: Scaled Dot-Product Attention — `softmax(QK^T / sqrt(d)) @ V`
- **n_embd**: Embedding dimension / hidden state size (e.g., 1152 for Gemma3 1B)
- **n_heads** / **nh**: Number of query attention heads
- **n_kv_heads** / **nkv**: Number of key/value heads (≤ n_heads for GQA)
- **head_dim** / **hd**: Dimension per attention head (`n_embd / n_heads`)
- **kv_dim** / **kvd**: Total KV size per position (`n_kv_heads × head_dim`)
- **ff_dim**: Feed-forward intermediate dimension (typically 4-8× n_embd)
- **n_layers**: Number of transformer layers in the model

---

## 1. Core Philosophy & Language Style
* **Zig Zen:** Follow the Zig Zen. Prefer explicit over implicit. No hidden control flow. No hidden memory allocations.  
  **The agent can actually run the command `zig zen` directly in the terminal (or invoke it programmatically) at any time to display the full official philosophy text and stay perfectly aligned.**
* **Zero-Cost Abstractions:** Logic must be resolved at `comptime` whenever possible to ensure zero runtime overhead. If a hardware feature or type is known at compile-time, it **must** be a `comptime` parameter.
* **Idiomatic Zig:** 
    * Use slices (`[]T`) instead of pointer+length pairs.
    * Use `std` library functions over "C-style" manual implementations.
    * Prefer `error` sets and `try/catch` over returning `null` for failures.
    * **Zig Builtins:** Always prefer Zig's `@` builtins over equivalent manual code or `std` functions whenever they achieve the same semantics. Key examples: `@Vector`/`@reduce`/`@splat` for SIMD, `@exp`/`@sqrt`/`@mulAdd` for math, `@memcpy`/`@memset` for bulk ops, `@bitCast`/`@intCast` for type conversions, `@prefetch` for cache hints.
* **Inference Efficiency:**
    * **No allocations**, syscalls, or locks are permitted within the token generation "hot path."
    * **`inline` for Hot-Path Functions:** Mark performance-critical functions with `inline` (or `inline fn`) when they sit on the hot path and the call overhead or lack of inlining would measurably hurt throughput. Key use cases:
      - **Small math helpers** called per-element inside tight loops (e.g., `silu`, `softplus`, `sigmoid`, `bf16ToF32`) — eliminates function call overhead in inner loops.
      - **Comptime-unrolled dispatch wrappers** like `inline else` on tagged unions — already used in `Backend` dispatch; ensures the compiler resolves the switch at compile time and inlines the selected implementation.
      - **Shared model helpers** called once per layer per token (e.g., `resetInferenceState`, `signalCancel`) — these are small enough that the call frame overhead matters.
      - Avoid `inline` on large functions or those only called during init — it bloats code size without benefit.
    * **Zig Std Efficiency Patterns:**
      - **Zero-Copy I/O**: `std.os.mmap` (weight loading), `std.os.madvise` (hint sequential/random access patterns), `std.os.sendfile` (network serving without userspace copy)
      - **Memory Alignment**: `std.mem.alignForward`, `std.mem.isAligned` (ensure SIMD-friendly layouts), `@alignOf` for cache line alignment (typically 64 bytes)
      - **Allocators**: `std.heap.FixedBufferAllocator` (pre-allocated scratch space for KV cache), `std.heap.ArenaAllocator` (per-request temporary allocations, bulk free), `std.heap.MemoryPool` (fixed-size object pooling)
      - **Buffering**: `std.io.BufferedReader`, `std.io.BufferedWriter` (batch small I/O to reduce syscalls), `std.io.FixedBufferStream` (zero-alloc in-memory I/O)
      - **CPU Affinity**: `std.Thread.setCpuAffinity` (pin inference threads to specific cores/NUMA nodes for cache locality)
      - **Atomics**: `std.atomic.Ordering` (use `.Monotonic` or `.Acquire`/`.Release` instead of `.SeqCst` when full ordering isn't needed)
      - **SIMD Detection**: `std.Target.Cpu.Feature` via `@import("builtin").cpu.features` (check for `avx2`, `avx512f`, `neon`, `sve`, `dotprod` at comptime)
      - **Hashing**: `std.hash.Wyhash` (fast non-cryptographic hash for cache key generation in RadixAttention), `std.hash.XxHash3` (high-quality alternative)
* **Named Constants:** All tuning thresholds, magic numbers, and repeated literals must be extracted into named `const` declarations at module level. This makes the codebase self-documenting and enables future configurability. Examples:
    * Metal backend: `threadgroup_size`, `softmax_cpu_threshold`, `sdpa_max_seq_len`, `sdpa_max_head_dim`
    * Main: `print_buf_size`, `gen_ids_buf_size`, `repeat_halt_threshold`, `tty_batch_size`, `pipe_batch_size`
    * Models: `max_layers` (nemotron_h), `max_active_experts` (gpt_oss), `max_ssm_v_heads` (qwen35)
    * Ops: `mlx_group_size`, `mlx_words_per_group`, `sqrt_2_over_pi`, `gelu_coeff`
* **No CPU Fallbacks in GPU Backends:** GPU backends (Metal, CUDA, Vulkan, ROCm) must **never** silently fall back to CPU for unsupported operations. If a GPU kernel is missing, `@panic` with a clear message naming the missing kernel. The **only** exceptions are operations where CPU is provably faster than GPU dispatch overhead (e.g., single-row embedding lookup, tiny softmax below threshold). These must be documented with a comment explaining the performance justification. This policy prevents silent performance regressions where GPU backends unknowingly run CPU code in the hot path.
* **Concurrency & Synchronization:**
    * Do not spawn unmanaged threads. All CPU-bound parallel execution must utilize a centralized thread pool passed down from the engine's root context.
    * Avoid heavy mutex locks in the inference hot path. Prefer atomic operations (`@atomicLoad`, `@atomicStore`) and lock-free queues where cross-thread communication is strictly necessary.

## 2. Memory Management & Safety
* **Explicit Allocation:** Functions requiring memory **must** accept an `std.mem.Allocator` as an argument. Do not use global or hidden allocators.
* **Ownership & Cleanup:** Structs that allocate memory must provide a `deinit()` function. Clear ownership transfer (e.g., `init` vs `initCopy`) must be documented.  
  **Always use `defer` immediately after acquiring a resource** (e.g. `var obj = try Obj.init(allocator); defer obj.deinit();` or `defer allocator.free(buf);`). This guarantees cleanup on every exit path with zero runtime cost.  
  **Use `errdefer` for resources that should only be cleaned up on error paths** (e.g. `var temp = try allocator.alloc(u8, n); errdefer allocator.free(temp);`). This keeps the success path clean while still guaranteeing cleanup if an error occurs later in the function.
* **Safety:** Use `std.debug.assert` for internal invariants. Ensure code is "comptime-safe" where applicable.
* **Page Allocator Rule:** `std.heap.page_allocator` may only be used during one-time initialization. It is forbidden in any hot path.

**Example - Proper Resource Management:**
```zig
// GOOD: Both defer and errdefer used correctly
pub fn processRequest(allocator: Allocator, config: Config) !Result {
    var buffer = try allocator.alloc(u8, 1024);
    defer allocator.free(buffer); // Always cleanup

    var cache = try KVCache.init(allocator, config.max_tokens);
    errdefer cache.deinit(); // Only cleanup on error path

    try populateCache(cache, buffer);
    return Result{ .cache = cache }; // cache ownership transferred to caller
}
```

```zig
// BAD: Manual cleanup on error paths (fragile!)
pub fn processRequestBad(allocator: Allocator) !Result {
    var buffer = try allocator.alloc(u8, 1024);
    var cache = try KVCache.init(allocator) catch |err| {
        allocator.free(buffer); // Manual cleanup - easy to miss!
        return err;
    };
    // If anything fails here, buffer leaks!
    return Result{ .cache = cache };
}
```

## 3. Quantization & Data Types
* **Explicit Precision:** Always use explicit types for mathematical operations (e.g., `f32`, `f16`, `bf16`, `f8`, `i8`).
* **Sub-Byte & Floating-Point Quantization:** 
  - Use `packed struct` + bit-packing helpers (or `std.bit_pack`) for sub-byte integers in `src/ops/quant.zig`.
  - Use hardware-native floating-point types (`bf16`, FP8 E4M3/E5M2, NVFP4, MXFP4) where supported.
* **Supported Schemes** (must support at minimum; all dequantization **must** happen inside the kernel via comptime-unrolled paths or pre-packed tensors — never full f32 conversion in the hot path):
  - **Integer / GGUF-style**:
    - Q2_K, Q3_K_S/M/L, Q4_0, Q4_K_S/M, Q5_K_S/M, Q6_K, Q8_0
    - IQ4_XS, IQ4_NL (CPU SIMD-optimized)
  - **Floating-Point**:
    - f32, f16, **bf16** (bfloat16 — baseline high-precision weights/KV cache)
    - FP8 (E4M3/E5M2 and MXFP8 — dynamic W8A8 support)
  - **Advanced Microscaled Floating-Point**:
    - **NVFP4** (4-bit hierarchical microscaling: 16-element blocks + FP8 scale + FP32 tensor scale) — required for CUDA backend on Blackwell+ targets
    - **MXFP4** (E2M1 4-bit microscaling with per-block FP32 scale — standard microscaling format, supported on Blackwell+ and equivalent ROCm/Vulkan hardware)
  - **Block-wise**:
    - AWQ/GPTQ (per-group scales), Marlin-compatible layouts
* **Type Erasing:** When passing quantized tensors through the dispatcher, use tagged unions or `comptime` type parameters to handle mixed-precision seamlessly without boxing.
* **Backend Notes:** NVFP4, MXFP4 and MXFP* are handled natively in `backend.cuda.zig` (Blackwell Tensor Cores) and equivalent paths in ROCm/Vulkan. Other backends fall back to equivalent integer or FP8 paths.

For quantization format selection guidance, see [DOCUMENTATION.md — Choosing a Quantization Format](docs/DOCUMENTATION.md#choosing-a-quantization-format).

## 4. Project Structure & Dispatcher Pattern
The project uses a "Dispatcher/Implementation" pattern to maintain clean boundaries and prevent "spaghetti" imports:
* **Modules (The Interface / Contract):** These define the core API and use `comptime` to dispatch to the correct hardware implementation based on the build target.
    * `src/backend/backend.zig`: Core dispatcher for compute.
    * `src/models/model.zig`: Core dispatcher for model architectures.
    * `src/tokenizer/tokenizer.zig`: Core dispatcher for encoding.
    * `src/format/format.zig`: Core dispatcher for weight loading.
    * `src/chat_template.zig`: Data-driven chat prompt templates (role markers, EOG tokens per model family).
    * `src/recipe.zig`: Optional preset configurations matched by arch + backend + quant (user CLI always overrides).
* **Implementations (The Logic):** Concrete logic for specific hardware or architectures (e.g., `cuda.zig`, `metal.zig`, `vulkan.zig`).
    * **Constraint:** Implementation files must **never** be imported directly by high-level logic (e.g., `main.zig`). They must only be accessed through their module dispatcher.

**Example - Dispatcher Pattern:**

The actual backend uses a tagged union with `inline else` dispatch — each backend implements the same set of functions (`gemv`, `rmsNorm`, `rope`, `sdpa`, etc.) and the compiler generates a comptime switch:

```zig
// src/backend/backend.zig (The Dispatcher)
// Backend is a tagged union: union(Enum) { cpu, metal, vulkan, cuda, rocm }
// Dispatch happens via `inline else` — zero VTable overhead.
pub const Backend = union(Enum) {
    cpu: CpuBackend,
    metal: MetalBackend,
    // ...

    pub fn gemv(self: *Backend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        switch (self.*) {
            inline else => |*be| be.gemv(x, w, y, n, k),
        }
    }
};
```

```zig
// main.zig (High-Level Logic)
const backend = @import("backend/backend.zig");

// GOOD: Use dispatcher via Backend union
be.gemv(x, weight, output, n, k);

// BAD: Never do this!
// const cuda = @import("backend/cuda.zig"); // WRONG!
```

## 5. Hardware Support & Cross-Compilation
The engine supports a diverse target matrix. Code must be written to respect the `std.Target` passed by the build system. Do not hardcode host architecture assumptions.

**Hardware & Compute Backends:**

| Vendor | Host-Side Driver (`src/backend/`) | Device Target (Kernels) | Platform |
| :--- | :--- | :--- | :--- |
| **NVIDIA** | `cuda.zig` | `nvptx64-cuda` **(use this one)** | Linux/Windows |
| **AMD** | `rocm.zig` | `amdgcn-amdhsa` **(use this one)** | Linux |
| **Apple** | `metal.zig` | Native MSL / MPS | macOS/iOS |
| **Universal** | `vulkan.zig` | `spirv64-vulkan` | Cross-platform |
| **CPU** | `cpu.zig` | Host Arch (`x86_64`, `aarch64`) | All |

**Note on NVIDIA targets:**  
`nvptx64-cuda` = CUDA runtime (correct for our driver API usage).  
`nvptx64-nvcl` = NVIDIA OpenCL runtime (do **not** use unless adding an OpenCL backend).

**Note on AMD targets:**  
`amdgcn-amdhsa` = HSA/ROCm runtime (correct for our `rocm.zig` and HSA calls).  
`amdgcn-amdpal` = AMD PAL runtime (graphics-focused, proprietary drivers).  
`amdgcn-mesa3d` = Mesa 3D open-source runtime (RADV Vulkan / OpenCL — do **not** use unless adding a Mesa backend).

**Cross-Compilation Matrix:**

| OS | Architecture | Expected Backends |
| :--- | :--- | :--- |
| **Linux** | `x86_64`, `aarch64` | CUDA, ROCm, Vulkan, CPU |
| **macOS** | `aarch64` (M-series) | Metal, CPU |

* **Kernel Compilation:** Kernels should be compiled to their native IR (PTX, SPIR-V, AMDGCN) via the Zig build system (`zig build-obj` with appropriate cross-target) and loaded as blobs or compiled at runtime through the driver API.
* **Vulkan/SPIR-V:** Implementations using Vulkan must target SPIR-V for shader kernels. Prefer Zig-native Vulkan wrappers.
* **Feature Detection:** Use `@import("builtin")` to detect CPU/GPU capabilities (e.g., `avx2`, `neon`, `dotprod`, Blackwell NVFP4/MXFP4) and select the fastest math kernels at compile time.
* **ROCm Specifics:** `rocm.zig` must correctly handle HIP Runtime API calls and memory dispatch.

## 6. KV Cache Management & Attention Patterns
* All models must use a memory-bounded KV cache. Unbounded growth is strictly forbidden.
* **KV Cache Strategies** (in increasing order of sophistication and production preference):
  - Circular / fixed-size buffer (simple single-request baseline)
  - **PagedAttention** with block tables (mandatory for good memory utilization and continuous batching)
  - **RadixAttention** (radix tree / prefix trie structure) — **strongly recommended for production serving**. Enables automatic longest-common-prefix detection, sharing, insertion, lookup, and LRU eviction across multiple requests (SGLang-style).
* KV cache management operations (tree traversal, eviction, reference counting) **must** occur at the request scheduler / batching layer — **never inside the per-token generation hot path**. This preserves zero allocations, zero locks, and zero syscalls during decoding.
* The KV cache logic shall live in `src/kvcache/` and follow the dispatcher pattern (`src/kvcache/manager.zig`).
* Provide a generic `src/ops/attention.zig` interface that individual backends can specialize via `comptime`. The attention kernel must support indirect block access (for both paged and radix layouts).

## 7. Naming Conventions
* Follow the official Zig Coding Conventions (Language Reference):
  - **`camelCase`** for functions and methods.
  - **`snake_case`** for variables, struct fields, and parameters.
  - **`PascalCase`** (TitleCase) for types (structs, enums, unions, error sets) and for functions that return a `type`.
  - File names and directories: `snake_case`.
* Core operations examples: `gemv`, `rmsNorm`, `rope`, `sdpa`, `nvfp4Dequant`, `mxfp4Lookup`.

## 8. Build System & Tooling
* **Native Tooling:** Use `build.zig` and `build.zig.zon` exclusively. No external Makefiles or shell scripts.
* **Pure Zig Rule:** Do not link against external C/C++ inference libraries (e.g., no `libggml`, no `OpenBLAS`). All kernels and mathematical routines must be written in Zig, native IR, or compute shaders.
* **Vendoring:** Any necessary external Zig dependencies must be explicitly pinned via hashes in `build.zig.zon`.
* **Target Awareness:** Every change must maintain the ability to cross-compile for all targets from any host.
* **Build Modes:** `ReleaseFast` is the primary production target. `Debug` must retain full safety checks.
* **Model Toggles:** Individual model architectures can be disabled at compile time via `-Denable-<model>=false` (e.g., `-Denable-glm4=false`). Disabled models are not compiled at all (`@import` is skipped), reducing binary size. All models default to enabled. Available flags: `enable-gemma3`, `enable-qwen35`, `enable-gpt-oss`, `enable-nemotron-h`, `enable-nemotron-nano`, `enable-glm4`.
* **Backend Maturity Levels**:
  - **Level 0**: CPU fallback only
  - **Level 1**: Metal + Vulkan
  - **Level 2**: Full CUDA + ROCm with optimized kernels (including NVFP4/MXFP4 on Blackwell)
  - Current goal: Reach Level 1 across platforms before pushing Level 2 on all vendors.

## 9. API Design & Safety
* **Strict Boundaries:** Use `pub` only for the intended API surface. Keep internal state and hardware handles (like `CUcontext` or `hsa_queue_t`) private to their backend file.
* **Zig-style Interfaces:** Use `comptime` duck-typing or VTable structs to provide clean abstractions without "leaky" implementation details.
* **Re-usability:** Logic should be generic. A tokenizer that works for Gemma should be configurable for Qwen with minimal changes.
* **Error Handling:** Use explicit Zig error sets (e.g., `error.DeviceOutOfMemory`, `error.KernelCompilationFailed`).

## 10. Documentation
* **Doc Comments:** Every public function and struct must have `///` comments explaining:
  - What the function does
  - Parameter semantics (especially ownership transfer)
  - Return value meaning
  - Error conditions
* **Inline Comments:** Use `//` for non-obvious implementation details, but prefer self-documenting code over comments.
* **Module-Level Docs:** Each major module should have a file-level doc comment explaining its purpose and role in the system.

**Example:**
```zig
/// Apply Rotary Position Embedding (RoPE) in-place to the first rope_dim
/// dimensions of each attention head in x.
///
/// Parameters:
///   - x: Interleaved Q or K tensor [n_heads * head_dim], modified in-place.
///   - pos: Absolute position index for this token.
///   - n_heads: Number of attention heads.
///   - head_dim: Dimension per head (rope_dim ≤ head_dim).
///   - rope_dim: Number of dimensions to rotate (must be even).
///   - theta: RoPE base frequency (e.g., 10000.0, 500000.0, or 1000000.0).
pub fn rope(x: [*]f32, pos: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
    // ... implementation
}
```

## 11. Testing
* **In-place Testing:** Place `test` blocks at the bottom of the relevant file, testing the functions defined in that file.
* **Target Testing:** Use target guards (`if (builtin.os.tag == .macos)`) for backend-specific tests.
* **CI Matrix Requirements**:
  - CPU on Linux x86_64 + aarch64 + macOS
  - Metal on macOS runners
  - Vulkan on Linux
  - (Future) CUDA/ROCm on GPU runners when available
* **Correctness:** Maintain golden tests against reference implementations (e.g., llama.cpp outputs with same seed, same quantization).
* **Test Categories:**
  - **Unit tests**: Individual function correctness
  - **Integration tests**: Full inference pipeline with known inputs/outputs
  - **Quantization tests**: Verify dequant output matches reference within tolerance
  - **Memory tests**: Verify no leaks (use `std.testing.allocator`)

**Example - Memory-Safe Test:**
```zig
test "KVCache allocation and cleanup" {
    const allocator = std.testing.allocator;

    var cache = try KVCache.init(allocator, 2048);
    defer cache.deinit(); // Will detect leaks automatically

    try cache.insert(0, test_key, test_value);
    try std.testing.expectEqual(1, cache.num_entries);
}
```

## 12. Profiling & Benchmarking
* **Benchmarking Requirement:** Any change to `src/backend/`, `src/models/`, or `src/kvcache/` must include benchmarks measuring:
  - **Throughput**: tokens/sec (batch=1 and batch=8)
  - **Latency**: Time-to-First-Token (TTFT)
  - **Memory**: VRAM usage and peak allocation
  - **Bandwidth**: Memory bandwidth utilization (GB/s)
  - **Cache Efficiency**: (When RadixAttention is enabled) prefix cache hit rate
  - **Quantization Impact**: Accuracy & throughput comparison (e.g., NVFP4 vs MXFP4 vs bf16 vs Q4_K_M)
* **Tracing:** Instrument critical paths with `std.log` scoped to `.perf`. Use Tracy (vendored) for detailed profiling.
* **Regression Prevention:** Before merging, compare benchmarks against main branch. A >5% regression requires explanation and approval.

**Benchmark Output Example:**
```
Model: Llama-3.1-8B-Q4_K_M
Backend: CUDA (RTX 4090)
Prompt: 512 tokens, Generate: 128 tokens

TTFT:           45ms
Throughput:     156 tok/s (batch=1)
Throughput:     487 tok/s (batch=8)
VRAM Usage:     4.2GB
Bandwidth:      523 GB/s (87% of peak)
Cache Hit Rate: 73% (RadixAttention)
```

## 13. External Kernel Prototyping Tools

* **Triton** (and equivalents: CUTLASS, TVM, MLIR, TileLang, CUDA C++, HIP, etc.) play **no role** in the final engine or any shipped binary.
* They are **permitted exclusively during development/research** (keep experimental code in a separate `research/kernels/` folder that is **not** part of the main build):
  - Rapid prototyping of advanced kernels (e.g., RadixAttention variants, fused FlashAttention, custom dequant+GEMM, NVFP4/MXFP4 fusion)
  - Generating reference outputs for golden tests
  - Exploring algorithmic variants before committing to Zig implementation
* **Mandatory Porting Rule:** Every prototype **must** be manually re-implemented in native Zig + target IR before it can be merged into `src/`.
* **Documentation:** If using prototypes, document the translation plan in `research/kernels/README.md`.

## 14. Implementation Quick Reference

For algorithm explanations and mathematical formulations, see [DOCUMENTATION.md §2 — Concepts](docs/DOCUMENTATION.md#2-concepts). This section covers only implementation-specific details, gotchas, and file locations needed when modifying the code.

### File Locations

| Component | CPU | Metal | Shared |
| :--- | :--- | :--- | :--- |
| RoPE | `src/backend/kernels/cpu/rope.zig` | `src/backend/kernels/metal/rope.metal` | — |
| RMSNorm | `src/backend/kernels/cpu/norm.zig` | `src/backend/kernels/metal/norm.metal` | — |
| L2Norm | `src/backend/kernels/cpu/norm.zig` | `src/backend/kernels/metal/norm.metal` | — |
| SiLU | `src/backend/kernels/cpu/activation.zig` | `src/backend/kernels/metal/elementwise.metal` | — |
| GELU | `src/backend/kernels/cpu/activation.zig` | `src/backend/kernels/metal/elementwise.metal` | `src/ops/math.zig` |
| Softplus | — | — | `src/ops/math.zig` |
| Sigmoid | — | — | `src/ops/math.zig` |
| Embedding | `src/backend/kernels/cpu/embedding.zig` | CPU (faster than GPU dispatch for single row) | — |
| Softmax | `src/backend/kernels/cpu/softmax.zig` | `src/backend/metal.zig` (3-pass GPU) | — |
| Elementwise | `src/backend/kernels/cpu/elementwise.zig` | `src/backend/kernels/metal/elementwise.metal` | — |
| DeltaNet | `src/backend/kernels/cpu/deltanet.zig` | `src/backend/kernels/metal/deltanet.metal` (4 GPU kernels) | — |
| GEMV | `src/backend/kernels/cpu/gemv.zig` (dispatcher) | `src/backend/kernels/metal/gemv.metal` | — |
| SDPA | `src/backend/kernels/cpu/sdpa.zig` | `src/backend/kernels/metal/sdpa.metal` (FlashAttention-2) | `src/ops/attention.zig` |
| PagedSDPA | — | — | `src/ops/attention.zig` |
| Causal Conv1d | — | — | `src/ops/ssm.zig` |
| Mamba-2 Recurrence | — | — | `src/ops/ssm.zig` |
| Group RMSNorm+Gate | — | — | `src/ops/ssm.zig` |
| Final Logits | — | — | `src/ops/math.zig` |
| KV Cache | — | — | `src/kvcache/manager.zig` |

CUDA kernels under `src/backend/kernels/cuda/`, ROCm under `src/backend/kernels/rocm/`, Vulkan under `src/backend/kernels/vulkan/` — see those directories and `docs/KERNELS.md` for the full list.

### Model Parameters

| Model | n_embd | n_heads | n_kv_heads | head_dim | ff_dim | n_layers | theta | rope_dim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Gemma3 1B | 1152 | 4 | 1 | 256 | 6912 | 26 | 1M | 256 |
| Qwen3.5 0.8B | 1536 | 16 | 4 | 128 | 4096 | 64 | 10M | 64 |
| GPT-OSS | 2880 | 64 | 8 | 64 | 2880 (MoE) | 24 | 150K | 128 |
| Nemotron-H | 3136 | 40 | 8 | 128 | 12544 | 42 | 10K | 78 |
| Nemotron-Nano | 2688 | 32 | 2 | 128 | 1856 (MoE) | 52 | 10K | 128 |
| GLM-4 | 2048 | 20 | 20 (MLA) | 256 | 1536 (MoE) | 47 | 1M | 64 |

### Implementation Gotchas

**DeltaNet Q|K|V split order (Qwen3.5):** After conv1d, the output is split as `[Q | K | V]` (llama.cpp convention, not Mamba's V,K,Q). Offsets: `q_off=0`, `k_off=num_k_heads * head_k_dim`, `v_off=2 * num_k_heads * head_k_dim`. Getting this wrong produces garbage output.

**GGUF +1.0 baked into norm weights (Gemma3):** QK norm weights in GGUF already have +1.0 added. Do not add it again in the normalization code.

**Gemma3 embedding scaling:** After embedding lookup, multiply by `sqrt(n_embd)`. This is Gemma-specific — other models do not scale.

**GPU sync before argmax:** Final logits are written by the GPU. CPU argmax must call `be.sync()` first. Missing this reads stale data on UMA platforms.

**SDPA two paths:** `attention.zig` has a fast path (dispatches to `Backend.sdpa()` — GPU kernel on all backends) and a fallback path (inline CPU with SIMD). The fallback is used when sliding window or attention sinks are active.

**Conv1d ring buffer is row-major:** `conv_state[(d_conv-1) × conv_ch]`. Rows shift left after each step. Current input appended at end. Zero allocation in hot path.

**MoE expert selection is stack-allocated:** `max_active_experts` sets stack array size: 8 (GPT-OSS, Nemotron-Nano), 16 (Qwen3.5). Actual top-K per token: 4 (GPT-OSS), 6 (Nemotron-Nano). Uses O(K×N) scan, not a heap — keeps hot path allocation-free.

**Clamped SwiGLU (GPT-OSS MoE):** Output is clamped to `[-7.0, +7.0]` to prevent overflow during mixed-precision expert computation.

**Sigmoid routing (GLM4):** Uses `sigmoid(logit)` instead of softmax for expert gating. Each expert gate is independent — not competing.

**Hybrid layer dispatch:** Layer types determined at init from GGUF metadata, dispatched in each model's `forward()` loop:
- Qwen3.5: attention every `full_attn_interval` layers (default 4), DeltaNet for the rest
- Nemotron-H: configured by `hybrid_override_pattern` string
- Nemotron-Nano: 52-layer pattern `M`=SSM, `E`=MoE, `*`=attention
- GPT-OSS: even layers = sliding window (128 tokens), odd = full attention

**Zig 0.15.2 API Notes:**
- `std.ArrayList(T).init(allocator)` removed → use `.empty`, pass allocator to `deinit(allocator)`, `append(allocator, val)`, `ensureTotalCapacity(allocator, n)`
- `ArrayList.pop()` returns `?T` — unwrap with `.?` after checking `.items.len > 0`
- `std.time.sleep` removed → use `std.Thread.sleep`
- `File.writer()` requires buffer arg → use `File.write(&buf)` for simple writes
- `std.Thread.Futex` requires `std.atomic.Value(u32)`, not `u64`

**Metal threadgroup memory limit:** Must stay ≤ 32KB. The `sdpa_fa2` kernel uses `block_size=16` (not 32) to fit. Calculate total: `q_local + kv_block + out_acc + scores + shared`. Pipeline creation fails silently without the error logging added in `makePipeline`.

**No CPU fallbacks in GPU backends:** GPU backends (Metal, CUDA, Vulkan, ROCm) must `@panic` on missing kernels — never silently delegate to CPU. Only exceptions: `embLookup` (single-row read, CPU faster) and `softmax` below threshold (Metal only). See AGENTS.md Section 1.

**Web UI:** Built-in chat UI at `/v1/chat`. Source files in `src/ui/` (style.css, app.js, head.html, body.html), assembled at comptime via `@embedFile` concatenation in `src/server/server.zig`.

**Build verification:** `zig build test` does NOT compile `agave-bench`. Always run `zig build` (full build) to catch signature mismatches in `src/micro_bench.zig` after changing model init or backend interfaces.

---

## Common Pitfalls & Anti-Patterns

### Memory Management
❌ **Don't**: Use `std.heap.page_allocator` in hot paths
```zig
pub fn processToken() !void {
    var buf = try std.heap.page_allocator.alloc(u8, 1024); // WRONG!
    defer std.heap.page_allocator.free(buf);
}
```
✅ **Do**: Pass allocator as parameter, use only in init
```zig
pub fn processToken(allocator: Allocator) !void {
    var buf = try allocator.alloc(u8, 1024);
    defer allocator.free(buf);
}
```

### Architecture Boundaries
❌ **Don't**: Import backend implementations directly
```zig
const cuda = @import("backend/cuda.zig"); // WRONG! Breaks abstraction
```
✅ **Do**: Access through dispatcher
```zig
const backend = @import("backend/backend.zig");
```

### Concurrency
❌ **Don't**: Spawn threads manually
```zig
const thread = try std.Thread.spawn(.{}, workerFn, .{}); // WRONG!
```
✅ **Do**: Use the centralized thread pool
```zig
try engine.thread_pool.spawn(workerFn, context);
```

### Quantization
❌ **Don't**: Full f32 dequantization before computation
```zig
var f32_weights = try dequantizeToF32(q4_weights); // WRONG! Defeats purpose
defer allocator.free(f32_weights);
be.gemv(input, .{ .data = f32_weights, .dtype = .f32 }, output, n, k);
```
✅ **Do**: Dequantize inside the kernel via comptime paths
```zig
be.gemv(input, .{ .data = q4_weights, .dtype = .q4_0 }, output, n, k); // Dequant happens in-kernel
```

### Error Handling
❌ **Don't**: Swallow errors silently
```zig
be.gemv(x, w, y, n, k) catch {}; // WRONG! Silent failure
```
✅ **Do**: Propagate or handle explicitly
```zig
try be.gemv(x, w, y, n, k); // Propagate to caller
// OR
be.gemv(x, w, y, n, k) catch |err| {
    std.log.err("GEMV failed: {}", .{err});
    return err;
};
```

### Magic Numbers
❌ **Don't**: Hardcode tuning thresholds or repeated literals inline
```zig
if (n < 128) { // WRONG! What is 128? Why 128?
    cpu_fallback();
}
var buf: [8192]u8 = undefined; // WRONG! Why 8192?
```
✅ **Do**: Extract into named module-level constants
```zig
const softmax_cpu_threshold: usize = 128;
const print_buf_size: usize = 8192;

if (n < softmax_cpu_threshold) cpu_fallback();
var buf: [print_buf_size]u8 = undefined;
```

### Resource Cleanup
❌ **Don't**: Manual cleanup on error paths
```zig
var obj = try init() catch |err| {
    cleanup(); // Easy to forget in some paths
    return err;
};
```
✅ **Do**: Use `defer` and `errdefer`
```zig
var obj = try init();
defer obj.deinit(); // Always runs
// OR
var temp = try allocTempResource();
errdefer freeTempResource(temp); // Only on error
```

---

## How To Add...

### How to Add a New Backend

1. **Create implementation file**: `src/backend/yourbackend.zig`
2. **Implement required functions**: `gemv`, `rmsNorm`, `rope`, `sdpa`, `silu`, `gelu`, `add`, `mul`, `softmax`, `embLookup`, `l2Norm`, `sync`, etc. (see `src/backend/cpu.zig` for the full interface)
3. **Add to dispatcher**: Add your backend variant to the `Backend` tagged union in `src/backend/backend.zig`
4. **Add tests**: Backend-specific tests in your implementation file
5. **Update build.zig**: Add target-specific compilation flags if needed
6. **Document**: Add entry to section 5 hardware table and update `docs/KERNELS.md`

**Template:**
```zig
// src/backend/yourbackend.zig
const std = @import("std");
const backend_mod = @import("backend.zig");
const TensorData = backend_mod.TensorData;

pub const YourBackend = struct {
    // ... driver handles, buffer caches, etc.

    pub fn init(allocator: std.mem.Allocator) !YourBackend { }
    pub fn deinit(self: *YourBackend) void { }

    pub fn gemv(self: *YourBackend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void { }
    pub fn rmsNorm(self: *YourBackend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void { }
    pub fn rope(self: *YourBackend, x: [*]f32, pos: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void { }
    pub fn sdpa(self: *YourBackend, q: [*]const f32, keys: []f32, values: []f32, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32) void { }
    pub fn sync(self: *YourBackend) void { }
    // ... see cpu.zig for remaining functions
};
```

### How to Add a New Quantization Scheme

1. **Define format**: Add to `src/ops/quant.zig` (bit layout, block size, scale format)
2. **Implement dequant kernel**: CPU SIMD and native GPU versions (no CPU fallback in GPU backends)
3. **Add to dispatcher**: Update type unions to include new format
4. **Benchmark**: Compare against existing formats
5. **Document**: Add to section 3 supported schemes table
6. **Test**: Golden tests against reference implementation

**Key functions to implement:**
- `quantize()` - Convert f32/bf16 to your format
- `dequantize()` - Convert your format to compute type (in-kernel)
- `kernelSize()` - Report memory footprint

### How to Add a New Model Architecture

1. **Create model file**: `src/models/yourmodel.zig`
2. **Implement interface**: Follow existing model API (init, forward, deinit)
3. **Add to dispatcher**: Update `src/models/model.zig` (conditional import gated by `build_options.enable_yourmodel`)
4. **Add build flag**: Add `enable-yourmodel` option in `build.zig` (both `b.option()` and `backend_options.addOption()`)
5. **Add to main.zig**: Add variant to `Arch` enum with `detect`, `displayName`, `chatTemplate`, `isEnabled`, `buildFlag` methods, and add to `initAndRun` switch
6. **Define config**: Model hyperparameters struct
7. **Add weight loader**: Update `src/format/format.zig` for your weight format
8. **Test**: Golden test against reference implementation (e.g., HuggingFace)

**Required interface** (see `src/models/model.zig` for the vtable contract):
```zig
pub const YourModel = struct {
    // Required fields (read by model.zig vtable):
    eos_token_id: u32,
    vocab_size: u32,
    n_layers: u32,
    n_embd: u32,
    n_head: u32,
    n_head_kv: u32,

    // Implementation fields:
    fmt: Format,
    be: Backend,
    allocator: Allocator,

    pub fn init(allocator: Allocator, fmt: Format, be: Backend) !YourModel { }
    pub fn deinit(self: *YourModel) void { }
    pub fn forward(self: *YourModel, token_id: u32) ForwardError!u32 { }
    pub fn resetCache(self: *YourModel) void { }
    pub fn cancel(self: *YourModel) void { }
};
```

### How to Add a New Chat Template

1. **Define preset**: Add a `pub const` to `src/chat_template.zig` with role prefixes/suffixes and EOG token names
2. **Map arch**: Add the arch → template mapping in `src/arch.zig: Arch.chatTemplate()`
3. **Test**: Add a format test verifying correct prompt assembly

### How to Add a New Recipe

1. **Add preset**: Add a `Preset` entry to the `presets` array in `src/recipe.zig`
2. **Set match criteria**: `arch_prefix` (e.g. "llama"), `backend` (e.g. "Metal"), `quant` (e.g. "Q4") — empty string means "any"
3. **Set defaults**: Only set fields that differ from CLI defaults (null = don't override)
4. **Test**: Run `zig test src/recipe.zig` to verify matching
5. **Document**: Update the recipe table in README.md

**Key principle**: User CLI flags always override recipe defaults. The `Overrides` struct tracks which flags the user explicitly set.

### How to Debug Performance Regressions

1. **Profile per-op timing**: Use the built-in `--profile` flag to instrument individual operations
   ```bash
   ./zig-out/bin/agave model.gguf --profile "prompt"
   # Shows per-op timing breakdown (adds GPU syncs, ~50% throughput loss)
   ```

2. **Run micro-benchmarks**: Use the `bench` build target
   ```bash
   zig build bench
   ./zig-out/bin/agave-bench gemv_f32 --n=4096 --k=4096 --backend=metal
   # Runs individual kernels with synthetic data, outputs JSON timing
   ```

3. **Research kernel benchmarks**: Use the research tooling
   ```bash
   cd research/kernels && uv run run.py bench sdpa --backend cpu
   # Benchmarks individual kernel implementations
   ```

4. **Check allocations**: Use `std.testing.allocator` in tests — it detects leaks automatically

5. **Verify comptime dispatch**: Ensure `inline else` dispatch is still used in `backend.zig` — no runtime vtable calls in the hot path

---

## Agent Meta-Instructions
1. **Analyze the Hot Path:** Before suggesting any code for the inference loop, verify it contains zero syscalls, zero locks, and zero allocations.
2. **Strict Target Adherence:** Before writing code, check the target `cpu` and `os`. Do not suggest Metal code for Linux targets, or CUDA-specific code when the context is focusing on a ROCm implementation.
3. **Comptime First:** If you are writing a dispatcher in `backend.zig`, use `comptime` to switch implementations to avoid runtime vtable overhead where possible.
4. **No Leaks:** Ensure that specific backend types (like a CUDA stream handle) do not appear in the top-level `main.zig` logic.

## Code Review Checklist

**Note**: This checklist is mandatory for all PRs. Use it as a template when reviewing changes.

Before approving any PR, verify:
- [ ] Hot path is allocation-free, lock-free, and syscall-free
- [ ] All memory is explicitly passed via allocator + proper `deinit()`
- [ ] `defer` **and** `errdefer` are used correctly for all resource management (no manual cleanup paths)
- [ ] Comptime is used aggressively for dispatch and type specialization
- [ ] Zig `@` builtins used aggressively where applicable
- [ ] Backend code only accessed through dispatcher
- [ ] No magic numbers — all thresholds and tuning constants are named module-level `const` declarations
- [ ] Chat templates used for prompt formatting (no hardcoded role markers in model code)
- [ ] Benchmarks included and performance regression checked (including prefix hit rate for RadixAttention and NVFP4/MXFP4 throughput)
- [ ] Target cross-compilation still works
- [ ] Quantization stays in-kernel where required (including NVFP4, MXFP4 and bf16 paths)
- [ ] KV cache strategy (Paged or RadixAttention) is clearly documented and bounded
- [ ] All kernels are native Zig / IR (no unported Triton/CUTLASS/etc. artifacts)
- [ ] No CPU fallbacks in GPU backends unless provably faster (embLookup, tiny softmax OK; GEMV, SDPA, DeltaNet NOT OK)

<!-- GSD:project-start source:PROJECT.md -->
## Project

**Agave — Production-Ready LLM Inference Engine**

A high-performance LLM inference engine written in Zig, optimized for extreme cross-platform portability and zero-cost abstraction. Supports multiple GPU backends (Metal, CUDA, Vulkan, ROCm, CPU), multiple model architectures (Gemma3, Qwen3.5, GPT-OSS, Nemotron-H, Nemotron-Nano, GLM-4), and extensive quantization formats (Q2-Q8, FP8, bf16, NVFP4, MXFP4, MLX). The engine provides both CLI and HTTP server interfaces for inference.

**Core Value:** Every supported model must produce correct output on every backend at full GPU speed — no unnecessary CPU fallbacks, no broken models, no unverified paths.

### Constraints

- **Pure Zig**: No external C/C++ inference libraries. All kernels native Zig, MSL, PTX, SPIR-V
- **Hot Path**: Zero allocations, zero syscalls, zero locks in token generation loop
- **Cross-Platform**: Must cross-compile for Linux x86_64, Linux aarch64, macOS aarch64
- **No Regressions**: Performance changes must be benchmarked. >5% regression requires justification
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- **Zig** (latest) - Core inference engine, all compute kernels, HTTP server, and model implementations
- **Metal Shading Language (MSL)** - GPU compute kernels for Apple Silicon (M-series)
- **CUDA PTX (Parallel Thread Execution)** - NVIDIA GPU kernels, cross-compiled to PTX IR
- **AMDGCN ISA / HSACO** - AMD ROCm GPU kernels
- **SPIR-V** - Vulkan compute shader IR
## Runtime
- **Zig Standard Library (std)** - All runtime services
- Linux x86_64, Linux aarch64, macOS aarch64 (Apple Silicon)
## Frameworks
- **Zig-native kernel implementations** - All GEMV, SDPA, quantization, activation kernels written in Zig or native IR
- **std.net with per-connection threads** - Synchronous HTTP/1.1 server
- **clap** (0.11.0) - Command-line argument parser
- **vaxis** (0.5.1, commit 41fff922) - Cross-platform terminal rendering
## Key Dependencies
- `clap` 0.11.0 - Command-line argument parsing library for Zig
- `vaxis` 0.5.1 - Terminal/TUI rendering library for Zig
- **libcuda.so** / **libcuda.dylib** (optional) - NVIDIA CUDA runtime
- **libamdhip64.so** (optional) - AMD ROCm HIP runtime
- **libMoltenVK.dylib** / **libvulkan.so** (optional) - Vulkan ICD loader
- **Metal framework** (Apple Silicon only, link-time) - Metal API for GPU acceleration
- **GGUF (GPT-Generated Unified Format)** - Native mmap-based loading
- **SafeTensors** - Native loader with multi-shard support
## Configuration
- **build.zig** - Pure Zig build system (no CMake, no Makefiles)
- **ReleaseFast** - Primary production target, LLVM optimized
- **Debug** - Full safety checks retained
- `zig-out/bin/agave` - Main inference engine (ReleaseFast)
- `zig-out/bin/agave-debug` - Debug build
- `zig-out/bin/agave-bench` - Standalone micro-benchmark tool
- `zig-out/ptx/*.ptx` - CUDA kernel PTX assembly (built via `zig build ptx`)
- `zig-out/rocm/kernels.hsaco` - ROCm kernel object (built via `zig build amdgcn`)
- Model weights loaded from GGUF file or SafeTensors directory (user-provided path)
- HTTP server config: port (default 49453), host (0.0.0.0)
- KV cache size: context window in tokens (default 4096)
- Inference backend: auto-selected from available hardware, override with CLI flag
- Chat template: auto-detected by model architecture, configurable via recipe system
## Platform Requirements
- **Zig compiler** (latest development version)
- **CUDA 12.0+** (optional, for CUDA backend)
- **ROCm 5.7+** (optional, for ROCm backend)
- **Vulkan SDK** (optional, for Vulkan backend)
- **Metal SDK** (macOS only, comes with Xcode)
- **macOS (Apple Silicon M-series)**: Metal framework (built-in)
- **Linux (x86_64, aarch64)**:
- **CPU**: x86_64 or aarch64 with SSE2/NEON
- **RAM**: 4GB for quantized models, 16GB+ for larger models
- **GPU** (optional):
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- `snake_case` for all files and directories (e.g., `cpu.zig`, `gemv.zig`, `thread_pool.zig`)
- `camelCase` for all functions and methods (e.g., `rmsNorm()`, `rope()`, `bf16ToF32()`, `mxfp4Lookup()`, `gemv()`, `dequantToF32()`)
- Dispatch functions use `inline else` pattern: `pub fn gemv(self: *Backend, ...)` dispatches via `switch(self.*) { inline else => |*be| be.gemv(...) }`
- Getter functions don't use `get` prefix unless there's significant computation (e.g., `backendInfo()` not `getBackendInfo()`)
- `snake_case` for local variables and struct fields (e.g., `kv_bytes_per_layer`, `n_workers`, `task_total`, `local_gen`, `block_size`)
- Field names are descriptive: `mapped_data`, `current_offset`, `init_count`, `avail_mem`
- Loop variables: short names acceptable in tight loops (`i`, `j`, `k`, `n`, `b`, `g`)
- `PascalCase` for structs, enums, unions, error sets (e.g., `TensorData`, `GGMLType`, `Backend`, `ThreadPool`, `KvCache`, `CacheBlock`, `SeqBlockTable`)
- Type functions (returning `type`) also use `PascalCase`
- `snake_case` for all `const` declarations, including module-level tuning constants (e.g., `quant_block_elems`, `print_buf_size`, `default_port`, `max_workers`, `min_grain`, `softmax_cpu_threshold`)
- Named constants MUST be extracted from inline literals — no magic numbers allowed
- Constants are placed at module level for visibility and configurability
## Code Style
- Zig stdlib default: 4-space indentation (consistent across all files)
- Max line length: no hard limit; clarity is preferred over wrapping
- Use blank lines between logical sections (imports, constants, types, functions)
- No external linter enforced; rely on `zig fmt` for auto-formatting
- `zig build` runs `zig fmt` check on changed files
- Always specify types explicitly in function signatures (no inference at boundaries)
- Use explicit type annotations for quantized data: `f32`, `f16`, `bf16`, `i8`, `u8`, etc.
- Example: `pub fn bf16ToF32(val: u16) f32` — types clear at call site
## Import Organization
- No global path aliases; use relative `@import` paths
- Dispatcher pattern: high-level modules import the dispatcher (e.g., `@import("backend/backend.zig")`), never implementation directly
- Re-exports used to avoid exposing internal types: `pub const DType = @import("../format/format.zig").DType;`
## Error Handling
- Use explicit error sets: `error.DeviceOutOfMemory`, `error.OffsetOutOfBounds`, `error.FileTooSmall`
- Propagate errors up with `try`: `const val = try self.readU32(off);`
- Use `catch` with logging for non-critical paths: `catch |err| { std.log.err("detail: {}", .{err}); return err; }`
- Silent error swallowing (`catch {}`) is forbidden except in shutdown paths
- Define at module or function level, near where they're used
- Example (from `gguf.zig`): return statements use inferred error types `!u64` from `return error.OffsetOutOfBounds;`
- `try` is preferred for error propagation in regular paths
- Explicit catch blocks for recovery: `const x = foo() catch |err| { cleanup(); return err; };`
- Never swallow with `catch {}` in production code
## Logging
- Log levels: `.debug`, `.info`, `.warn`, `.err`
- Scoped logs use `@import("std").log.scoped()` or implicit scope from file context
- Performance-critical paths use `.perf` scope (gated by `if (g_debug)` at callsite)
- Example: `std.log.warn("warning detail: {}", .{val});`
- Global flags in `main.zig`: `g_debug`, `g_quiet`, `g_verbose`, `g_color`, `g_tty`
- Use `eprint()` helper for immediate stderr (for CLI feedback)
- Use `print()` helper for stdout with custom formatting buffer
- Tests use `std.testing.expectEqual()`, not print statements
## Comments
- Document non-obvious algorithm details (e.g., bit-packing semantics, quantization block structure)
- Explain invariants and constraints (e.g., "zero allocations in hot path")
- DON'T comment obvious code ("increment i by 1" not needed)
- Required on all public functions and structs
- Include: purpose, parameter semantics (especially ownership), return value meaning, error conditions
- Example (from `quant.zig`):
- Use for implementation details that aren't self-documenting
- Example: `// Wake workers by bumping generation`
- Keep comments close to the code they describe
- First lines of each file explain purpose and role
- Example (from `thread_pool.zig`): `//! Lightweight thread pool for parallel-for workloads.`
## Function Design
- Small is preferred; aim for <50 lines per function
- Dispatch wrappers (inline else) can be exceptions
- Large functions should be broken into helpers with clear names
- Explicit names; no positional inference (e.g., `fn rope(x: [*]f32, pos: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32)`)
- Use structs for >4 related parameters (e.g., `DeltaNetParams`, `BackendInfo`)
- Pass allocators explicitly; never hide allocators in globals
- Output buffers are `[*]T` (pointers) or `[]T` (slices) — be consistent
- Error types explicit: `!Result` for fallible, `Result` for infallible
- Use error propagation (`try`) unless recovery needed
- Multiple return values via struct or named return type
- Example: `fn readMetaValue(self: *GGUFFile, off: usize) !struct { val: MetaValue, len: usize }`
- `inline fn` or `inline` for hot-path math helpers (e.g., `bf16ToF32()`, `mxfp4Lookup()`, `fp8e4m3ToF32()`)
- Use for comptime dispatch: `inline else` in backend switch statements
- Avoid on large functions or one-time init code (wastes code size)
## Memory Management
- All functions requiring memory accept `allocator: Allocator` as explicit parameter
- Never use global or hidden allocators
- No `std.heap.page_allocator` in hot paths (initialization only)
- `defer` immediately after resource acquisition: `var obj = try init(allocator); defer obj.deinit();`
- `errdefer` for partial cleanup on error paths: `var temp = try allocator.alloc(u8, n); errdefer allocator.free(temp);`
- Never rely on manual cleanup in catch blocks (error-prone)
- Require `deinit()` method: `pub fn deinit(self: *MyType) void`
- Clear ownership semantics: `init(allocator)` vs `initCopy(allocator, data)`
- Document transfer of ownership in doc comments
## Module Design
- Use `pub` only for intended API surface; keep internals private
- Re-export dependencies for convenience (e.g., `pub const DType = @import("format.zig").DType;`)
- Hide backend-specific types (e.g., `CUcontext` private to `cuda.zig`)
- High-level modules (main.zig, models) import dispatcher, never implementations
- Dispatcher uses tagged union with `inline else` for zero-overhead dispatch
- Each backend implements the same interface (duck-typing + comptime)
- Use `@Vector`, `@reduce`, `@splat` for SIMD instead of manual loops
- Use `@memcpy`, `@memset` for bulk ops
- Use `@bitCast`, `@intCast` for type conversions
- Use `@exp`, `@sqrt`, `@mulAdd` for math (avoid libcalls on nvptx)
- Example (from `quant.zig`): `return @bitCast(@as(u32, val) << 16);`
## Concurrency
- No manual `std.Thread.spawn()` in inference code (use centralized thread pool)
- No locks in hot paths; prefer atomics: `std.atomic.Value(T)`
- Use `std.Thread.Futex` for sleep/wake synchronization
- All parallel work goes through `ThreadPool` in `src/thread_pool.zig`
- Example (from `thread_pool.zig`): `parallelFor(total, grain, ctx, func)` splits work into chunks
- Workers capture pool by pointer (must be at final memory location before `spawn()`)
## Comptime Usage
- Backend/model selection: `comptime` checks on `build_options` and `builtin`
- Cost-amortized dispatch: `comptime` in if/else branches to eliminate dead code
- Compile-time lookup tables: `const table = blk: { ... break :blk table; };`
- Format detection: `if (comptime builtin.os.tag == .macos)` gates platform-specific code
- `fp8e4m3_lut`: 256-entry f32 table computed at comptime
- `iq4nl_table`: i8 dequant LUT (hardcoded, comptime-verified)
- Backend dispatch: `inline else` on tagged union eliminates VTable calls
## Hot-Path Constraints
- No allocations (`allocator.alloc()`) in token generation loop
- No syscalls in inner loops
- No locks in inference path
- All I/O (model loading, streaming) happens outside the hot path
- Mark hot-path functions with comments: `// Hot path: no allocs, locks, syscalls`
- Use `std.debug.assert()` for invariants
- Benchmark any changes to backend/model forward() functions
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Zero-overhead backend selection via tagged union dispatch (no VTable indirection)
- Comptime vtable generation for model abstraction (duck-typing)
- Deferred GPU execution on UMA (Unified Memory Architecture) platforms
- Multi-model support with build-time architecture toggles
- Strict boundary enforcement: implementations accessed only through dispatchers
## Layers
- Purpose: CLI interface, REPL, HTTP server, output formatting
- Location: `src/main.zig` (CLI + generation logic), `src/server/server.zig` (HTTP API), `src/display.zig` (formatting), `src/readline.zig` (interactive input)
- Contains: Argument parsing, generation loop, prompt formatting, token streaming
- Depends on: Model interface, Tokenizer interface, Format interface
- Used by: Entry point only
- Purpose: Unified interface to all supported architectures (Gemma3, Qwen3.5, GPT-OSS, Nemotron-H, Nemotron-Nano, GLM-4)
- Location: `src/models/model.zig` (interface), `src/models/{gemma3,qwen35,gpt_oss,nemotron_h,nemotron_nano,glm4}.zig` (implementations)
- Contains: Forward pass, KV cache management, layer dispatch, quantization handling
- Depends on: Backend interface, Format interface, KV cache managers, Shared ops (attention, SSM, quantization)
- Used by: Main (via `Model.from()` interface)
- Purpose: Compute abstraction for CPU, Metal, Vulkan, CUDA, ROCm backends
- Location: `src/backend/backend.zig` (dispatcher), `src/backend/{cpu,metal,vulkan,cuda,rocm}.zig` (implementations)
- Contains: Kernel launchers, memory management, sync points, deferred execution handling
- Depends on: Kernel implementations (in `src/backend/kernels/`), system APIs (Metal Framework, CUDA Runtime, Vulkan, etc.)
- Used by: Model layer, Shared ops layer
- Purpose: Algorithm implementations reused across models
- Location: `src/ops/`
- Contains:
- Depends on: Backend interface
- Used by: Model layer
- Purpose: Model file loading (GGUF, SafeTensors)
- Location: `src/format/format.zig` (interface), `src/format/{gguf,safetensors}.zig` (implementations)
- Contains: Tensor lookup, metadata extraction, weight loading, format detection
- Depends on: None (filesystem only)
- Used by: Main, Model layer
- Purpose: Text ↔ token ID conversion
- Location: `src/tokenizer/tokenizer.zig` (interface), `src/tokenizer/bpe.zig` (implementation)
- Contains: BPE encoding/decoding, SentencePiece (SPM) encoding, special tokens
- Depends on: None
- Used by: Main, Server
- Purpose: Manage key/value cache across sequences
- Location: `src/kvcache/manager.zig`
- Contains: Flat allocation, PagedAttention (block-based), RadixAttention (prefix tree), LRU eviction
- Depends on: None
- Used by: Backend layer, Model layer
- `src/chat_template.zig` - Role markers, EOG tokens per model family (data-driven)
- `src/recipe.zig` - Optional preset configurations (arch + backend + quant matching)
- `src/thread_pool.zig` - Futex-based work-stealing thread pool for CPU parallelism
- `src/arch.zig` - Model architecture enum with detection and build flags
- `src/perf.zig` - Per-layer profiling instrumentation
## Data Flow
- Tokens decoded in small batches (4 for TTY, 32 for pipe) for responsive streaming
- Optional stats printed: TTFT, throughput, prefill/decode breakdown
- JSON mode outputs structured result with metadata
- `be.sync()` called only when CPU reads GPU data (e.g., before argmax, before embedding lookup)
- UMA platforms: sync flushes command buffer but no D2H copy needed (memory is shared)
- Discrete GPU: sync also downloads activations to host
- Models minimize sync calls: SDPA handles its own internal sync, final logits stay on GPU until argmax
## Key Abstractions
- Type: `union(enum) { cpu, metal, vulkan, cuda, rocm }`
- Dispatch: `inline else` switch — compiler resolves at compile-time for each backend variant
- Zero overhead: no function pointer table, no runtime type checks
- Interface: All backends implement same set of operations (gemv, rmsNorm, rope, sdpa, silu, gelu, add, mul, softmax, embLookup, l2Norm, etc.)
- Pattern: Operations that differ per-backend (e.g., GPU-accelerated) dispatch to backend; ops that are universal (e.g., CPU-only math) call shared helpers
- Type: `struct { ptr: *anyopaque, vtable: *const VTable }`
- Generated: Comptime vtable creation via `Model.from(ConcreteType, &instance)`
- Interface: `forward(token_id) !u32`, `resetCache()`, `cancel()`, field accessors
- Benefit: High-level code (`main.zig`, `server.zig`) works with any model type without generics
- Type: `struct { ptr: *anyopaque, vtable: *const VTable }`
- Methods: `getTensor(name)`, `getMetaStr/U32/F32(key)`, `getVocab()`, `getMerges()`
- Implementations: GGUF (mmap'd file with index), SafeTensors (sharded tensors with JSON metadata)
- Type: `struct { ptr: *anyopaque, vtable: *const VTable }`
- Methods: `encode(text)`, `decode(tokens)`, `vocabSize()`
- Modes: BPE (byte-pair merges), SPM (SentencePiece greedy), SPM-no-dummy (Gemma3)
- Flat: Simple per-layer byte slices (single-sequence, no sharing)
- Paged: Block-based allocation per vLLM PagedAttention spec (multi-sequence, shared blocks)
- Radix: Prefix tree for longest-common-prefix detection and copy-on-write sharing (SGLang-style)
## Entry Points
- Location: `src/main.zig` `runRepl()`
- Triggers: No prompt or server flag provided
- Responsibilities:
- Location: `src/main.zig` `generateAndPrint()`
- Triggers: Prompt provided (CLI arg or piped stdin)
- Responsibilities:
- Location: `src/server/server.zig`
- Triggers: `--serve` flag
- Responsibilities:
- Location: `src/main.zig` `initAndRun()`
- Triggers: Called by all three entry points
- Responsibilities:
## Error Handling
```zig
```
- Main catches all initialization errors, prints diagnostic, exits
- Generation errors (forward fails) logged and generation aborts
- Tokenizer errors (encode/decode fails) logged and prompt/output skipped
- Server catches all request errors, returns HTTP 500 with error message
## Cross-Cutting Concerns
- CLI: `eprint()` for errors/warnings, `print()` for output
- Debug mode: `dbg()` macro (conditional on `g_debug` flag)
- Server: logs to stderr with request context
- No global logger; output via `std.fs.File.stderr()` and `std.fs.File.stdout()`
- Format: Metadata consistency checks (architecture name, tensor names)
- Model: Weight shapes validated during load, layer count asserted
- Tokenizer: Vocabulary size must match model vocab
- Backend: Device availability checked at init; GPU backends panic on missing kernels (no silent CPU fallback)
- Models loaded from local filesystem only (no remote URLs)
- Server: No authentication (localhost only by default, or behind proxy)
- No secrets in code (env vars for API keys, not committed)
- Single-threaded generation (one token at a time through model)
- CPU backend parallelizes GEMV rows via thread pool (`src/thread_pool.zig`)
- Metal/Vulkan/CUDA: GPU handles parallelism; CPU thread is event loop (no explicit threading)
- Server: Each request queued sequentially (future: continuous batching could interleave multiple requests)
- Build-time toggles: `-Denable-gemma3=false` disables model at compile time
- Runtime CLI args: backend, context size, sampling params, KV quantization
- Recipes: Optional preset configs matched by arch + backend + quant (user CLI overrides)
- Chat templates: Data-driven per-architecture (no hardcoded prompts)
- Hot path (token generation): Zero allocations, zero syscalls, no locks
- Memory: KV cache allocation happens once at init; activations reused across tokens
- GPU: Deferred dispatch (command buffers); CPU reads data only at sync points
- Quantization: Dequantization happens in-kernel (on GPU or SIMD on CPU), not pre-converted to f32
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
