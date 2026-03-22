# Chapter 8: GPU Backends

Modern GPUs are massively parallel processors — thousands of simple cores executing the same instruction on different data. To use them for inference, you need a **compute API** that lets you write kernels (small programs) and dispatch them to the GPU.

## The GPU Landscape

Each hardware vendor has its own API:

| Platform | Vendor | Language | IR | Scope |
|----------|--------|----------|----|-------|
| **CPU** | All | Zig + SIMD | Native (NEON/AVX2) | All platforms |
| **CUDA** | NVIDIA | Zig → PTX | PTX | NVIDIA GPUs only |
| **Metal** | Apple | MSL | Metal IR | Apple Silicon only |
| **ROCm/HIP** | AMD | Zig → HSACO | AMDGCN | AMD GPUs only |
| **Vulkan** | Khronos | GLSL | SPIR-V | All vendors (cross-platform) |

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

**Kernel fusion** combines multiple operations into one kernel to avoid intermediate memory reads/writes:

```
// Without fusion: 3 memory round-trips
gemv(gate) → write → read → gelu → write → read → gemv(down)

// With fusion: 1 memory round-trip
fused_mlp: load → compute gate+up → gelu in-register → multiply → write
```

This matters enormously on GPUs where memory bandwidth is the bottleneck.

## The Dispatcher Pattern

Model code never imports backend implementations directly. Instead, the `Backend` tagged union with `inline else` dispatch resolves at compile time:

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

This gives zero-overhead dispatch (no vtable, no function pointers) while keeping model code hardware-agnostic.

## UMA (Unified Memory Architecture)

On platforms where CPU and GPU share physical memory (Apple Silicon, NVIDIA Grace), GPU backends can wrap existing CPU allocations as GPU buffers with zero copies:

- **Metal**: `newBufferWithBytesNoCopy` wraps mmap'd weights directly
- **CUDA**: `cudaMallocManaged` for transparent access
- **Vulkan**: `HOST_VISIBLE | DEVICE_LOCAL` memory type

All GPU backends use **deferred dispatch** — operations are encoded into command buffers without blocking. Models call `be.sync()` only when CPU code needs to read GPU-produced data.

## Backend-Specific Notes

**Metal** (`metal.zig`): MSL compute shaders with threadgroup-level `simd_sum` reduction. Buffer caching eliminates ~800 ObjC alloc/release per token. FlashAttention-2 with block_size=16 (fits 32KB threadgroup memory).

**CUDA** (`cuda.zig`): Zig kernels compiled to PTX via `nvptx64-cuda` target — no CUDA C++ dependency. Driver API loaded dynamically via `dlopen`. Deferred execution with activation caching for zero-sync SDPA.

**Vulkan** (`vulkan.zig`): Pre-compiled SPIR-V compute shaders. Subgroup arithmetic for reductions. Fused single-dispatch normalization/softmax. Works on all vendors including Apple (via MoltenVK).

**ROCm** (`rocm.zig`): HIP Runtime API loaded dynamically. AMDGCN kernels compiled from Zig via `amdgcn-amdhsa` target. Same deferred execution pattern as CUDA.

---

**In the code:** `src/backend/backend.zig` (dispatcher), `src/backend/{cpu,metal,cuda,vulkan,rocm}.zig` (implementations), `src/backend/kernels/` (GPU kernel sources)

**Start over:** [Chapter 1: Tokens and Text →](01-tokens-and-text.md) | **Product docs:** [Architecture](../ARCHITECTURE.md) · [Models](../MODELS.md)
