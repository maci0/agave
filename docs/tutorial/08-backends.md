# Chapter 8: Backends

Inference can run on different compute backends: **CPU** (universal, always available), **GPU** (massively parallel — thousands of cores executing the same instruction on different data via **SIMD** — Single Instruction Multiple Data), or specialized **accelerators** (purpose-built hardware like TPUs, NPUs, or FPGAs optimized for specific workloads). Each backend provides a **compute API** that lets you write **kernels** (small programs that run on the hardware) and dispatch them.

## The GPU Landscape

Each hardware **vendor** (manufacturer — NVIDIA, Apple, AMD, etc.) has its own API:

| Platform | Vendor | Language | Compiled Format | Scope |
|----------|--------|----------|-----------------|-------|
| **CPU** | All | Zig + SIMD | Native (NEON/AVX2) | All platforms |
| **CUDA** | NVIDIA | Zig → PTX | PTX | NVIDIA GPUs only |
| **Metal** | Apple | MSL | Metal IR | Apple Silicon only |
| **ROCm/HIP** | AMD | Zig → HSACO | AMDGCN | AMD GPUs only |
| **Vulkan** | Khronos | GLSL | SPIR-V | All vendors (cross-platform) |

The "Compiled Format" column shows the **IR** (Intermediate Representation — compiled bytecode that the GPU driver converts to native machine code at runtime, not final executable code).

```
Vendor-specific:  CUDA ──→ PTX ──→ NVIDIA only
                  Metal ──→ Metal IR ──→ Apple only
                  ROCm/HIP ──→ AMDGCN ──→ AMD only

Cross-platform:   Vulkan ──→ SPIR-V ──→ All vendors
```

**Agave's strategy**: Use vendor-specific APIs for maximum performance, with Vulkan as the universal fallback. The `Backend` interface abstracts all of these behind a single dispatch.

| Platform | Primary | Fallback |
|----------|---------|----------|
| macOS (Apple Silicon) | Metal | CPU |
| Linux + NVIDIA | CUDA | Vulkan → CPU |
| Linux + AMD | ROCm | Vulkan → CPU |
| Linux + Intel | Vulkan | CPU |

## Kernels

A **kernel** is a single computational function dispatched to the GPU. Agave has separate kernels per operation per data type — for example, the CPU backend has `gemvQ4_0`, `gemvQ8_0`, `gemvBF16`, `gemvF32` because each quantization format has completely different bit layout.

**Kernel fusion** combines multiple sequential operations into a single kernel to eliminate intermediate memory traffic. Without fusion, each operation must write its results to memory and the next operation must read them back. With fusion, intermediate results stay in fast registers (on-chip storage, ~100× faster than RAM/VRAM) and never touch slow memory.

```
// Without fusion: 3 memory round-trips (SLOW)
gemv(gate) → write to VRAM → read from VRAM → gelu → write to VRAM → read from VRAM → gemv(down)
            ↑ bottleneck      ↑ bottleneck            ↑ bottleneck      ↑ bottleneck

// With fusion: 1 memory round-trip (FAST)
fused_mlp: load from VRAM → compute gate+up → gelu in-register → multiply in-register → write to VRAM
           ↑ once                              ↑ stays in registers ~100× faster        ↑ once
```

**Why it matters**: GPUs are compute-rich but memory-bandwidth-starved. A modern GPU can do 300+ **TFLOPS** (teraflops — trillion floating-point operations per second) but only read ~900 **GB/s** (gigabytes per second) from VRAM. For small operations like GELU (one input, one output), the GPU spends 95% of its time waiting for memory, not computing. Fusion keeps data on-chip and lets the GPU actually use its compute power.

**Example**: Gemma3's FFN does `down_proj(GELU(gate_proj(x)) * up_proj(x))` — that's 4 matrix operations. Unfused = 8 memory passes. Fused = 2 memory passes (4× speedup from memory reduction alone).

## The Dispatcher Pattern

Model code never imports backend implementations directly. Instead, the `Backend` tagged union with `inline else` dispatch resolves **at compile time** (during compilation, not when the program runs — zero runtime overhead):

```zig
pub const Backend = union(enum) {
    cpu: *CpuBackend,
    metal: *MetalBackend,
    cuda: *CudaBackend,
    // ...

    pub fn gemv(self: Backend, ...) void {
        switch (self) {
            inline else => |be| be.gemv(...),
        }
    }
};
```

This gives zero-overhead dispatch (no **vtable** — virtual function table used for dynamic dispatch in object-oriented languages, no function pointers) while keeping model code hardware-agnostic.

## UMA (Unified Memory Architecture)

On **UMA** platforms (where CPU and GPU share the same physical memory chips, unlike **discrete GPUs** which have separate VRAM) like Apple Silicon and NVIDIA Grace, GPU backends can wrap existing CPU allocations as GPU buffers with zero copies:

- **Metal**: `newBufferWithBytesNoCopy` wraps mmap'd weights directly
- **CUDA**: `cudaMallocManaged` for transparent access
- **Vulkan**: `HOST_VISIBLE | DEVICE_LOCAL` memory type

All GPU backends use **deferred dispatch** — operations are encoded into **command buffers** (queues of GPU operations) without blocking. Models call `be.sync()` only when CPU code needs to read GPU-produced data.

## Batched Prefill Dispatch

During prefill, the backend dispatches **batched** versions of the core ops — GEMM (instead of GEMV), batched RMSNorm, batched RoPE, and fused causal SDPA:

```
Prefill layer pipeline (Gemma 3):
  rmsNormBatched → GEMM(Q,K,V) → rmsNormMulti → ropeBatched
    → sdpaPrefill(FA2) → GEMM(O) → rmsNormBatched → add
    → rmsNormBatched → GEMM(gate,up) → gelu → mul → GEMM(down)
    → rmsNormBatched → add
```

**Metal**: all batched ops are native GPU kernels. The GEMM uses one threadgroup per output row with weight reuse across tokens. The `sdpa_prefill_fa2` kernel reads old K/V from the cache and new K/V directly from GEMM output (dual-source), then a `copy_f32` kernel populates the cache — all in one command buffer with zero CPU-GPU flush.

**CUDA**: native GPU GEMM (Q8_0), batched RMSNorm and RoPE kernels compiled to PTX. The SDPA uses sequential single-token GPU sdpa calls (already fused with KV append on GPU).

**CPU**: parallel GEMV-based GEMM via thread pool, parallel-head SDPA with bulk KV append.

## Backend-Specific Notes

**Metal** (`metal.zig`): MSL compute shaders with **threadgroup**-level (a group of threads that execute together and can share fast on-chip memory) `simd_sum` reduction. Buffer caching eliminates ~800 ObjC alloc/release per token. FlashAttention-2 with block_size=16 (fits 32KB threadgroup memory). Prefill: native GEMM (f32/Q8_0/Q4_0), batched RoPE, dual-source FA2, zero per-layer flush.

**CUDA** (`cuda.zig`): Zig kernels compiled to PTX via `nvptx64-cuda` target — no CUDA C++ dependency. Driver API loaded dynamically via `dlopen`. Deferred execution with activation caching for zero-sync SDPA. Prefill: native GEMM (Q8_0), batched RMSNorm/RoPE.

**Vulkan** (`vulkan.zig`): Pre-compiled SPIR-V compute shaders. Subgroup arithmetic for reductions. Fused single-dispatch normalization/softmax. Works on all vendors including Apple (via MoltenVK).

**ROCm** (`rocm.zig`): HIP Runtime API loaded dynamically. AMDGCN kernels compiled from Zig via `amdgcn-amdhsa` target. Same deferred execution pattern as CUDA.

---

**In the code:** `src/backend/backend.zig` (dispatcher), `src/backend/{cpu,metal,cuda,vulkan,rocm}.zig` (implementations), `src/backend/kernels/` (GPU kernel sources)

**Start over:** [Chapter 1: Tokens and Text →](01-tokens-and-text.md) | **Product docs:** [Architecture](../ARCHITECTURE.md) · [Models](../MODELS.md)
