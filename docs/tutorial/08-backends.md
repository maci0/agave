# Chapter 8: Backends

Inference can run on different compute backends: **CPU** (universal, always available), **GPU** (massively parallel — thousands of cores organized into **warps/wavefronts** that execute via **SIMT** — Single Instruction Multiple Thread, where groups of 32-64 threads run the same instruction in lockstep on different data), or specialized **accelerators** (purpose-built hardware like TPUs, NPUs, or FPGAs optimized for specific workloads). Each backend provides a **compute API** that lets you write **kernels** (small programs that run on the hardware) and dispatch them.

**SIMD vs SIMT:** CPUs use **SIMD** (one instruction operates on a vector register of packed values, e.g., 8 f32s in AVX2 — see [Chapter 9](09-cpu-simd-optimization.md)). GPUs use **SIMT** (one instruction is executed by many threads simultaneously, each with its own registers and program counter). The distinction matters: SIMD has no divergence — all lanes do the same thing. SIMT threads can branch independently, but divergent branches serialize.

## The GPU Landscape

Each hardware **vendor** (manufacturer — NVIDIA, Apple, AMD, etc.) has its own API. Every backend compiles kernel source to an intermediate representation, then the GPU driver translates that to native machine code at runtime.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#e8f0fe',
  'primaryTextColor': '#1a1a2e',
  'primaryBorderColor': '#4a6cf7',
  'lineColor': '#4a6cf7',
  'secondaryColor': '#f0f4ff',
  'tertiaryColor': '#f8f9ff',
  'edgeLabelBackground': '#ffffff',
  'clusterBkg': '#f0f4ff',
  'clusterBorder': '#4a6cf7',
  'titleColor': '#1a1a2e',
  'nodeTextColor': '#1a1a2e',
  'fontFamily': 'ui-monospace, SFMono-Regular, monospace'
}}}%%
flowchart LR
    Zig["Zig kernel source"]

    Zig --> PTX["PTX bytecode"]
    Zig --> AMDGCN["AMDGCN bytecode"]
    Zig --> NEON["NEON / AVX2\n(native binary)"]

    PTX --> NVIDIA["NVIDIA GPU"]
    AMDGCN --> AMD["AMD GPU"]
    NEON --> CPU["CPU"]

    MSL["MSL shader"] --> MetalIR["Metal IR"]
    GLSL["GLSL compute shader"] --> SPIRV["SPIR-V bytecode"]
    WGSL["WGSL shader"] --> WGSL2["WGSL (interpreted)"]

    MetalIR --> Apple["Apple Silicon GPU"]
    SPIRV --> AnyGPU["Any GPU\n(Vulkan driver)"]
    WGSL2 --> Browser["Browser / native\n(wgpu)"]

    subgraph Vendor-specific
        PTX
        AMDGCN
        MetalIR
    end

    subgraph Cross-platform
        SPIRV
        WGSL2
    end


| Platform | Vendor | Language | Compiled Format | Scope |
|----------|--------|----------|-----------------|-------|
| **CPU** | All | Zig + SIMD | Native (NEON/AVX2) | All platforms |
| **CUDA** | NVIDIA | Zig → PTX | PTX | NVIDIA GPUs only |
| **Metal** | Apple | MSL | Metal IR | Apple Silicon only |
| **ROCm/HIP** | AMD | Zig → HSACO | AMDGCN | AMD GPUs only |
| **Vulkan** | Khronos | GLSL | SPIR-V | All vendors (cross-platform) |
| **WebGPU** | W3C | WGSL | WGSL source | All vendors (browser + native) |

The "Compiled Format" column shows the **IR** (Intermediate Representation — compiled bytecode that the GPU driver converts to native machine code at runtime, not final executable code).

```
Vendor-specific:  CUDA ──→ PTX ──→ NVIDIA only
                  Metal ──→ Metal IR ──→ Apple only
                  ROCm/HIP ──→ AMDGCN ──→ AMD only

Cross-platform:   Vulkan ──→ SPIR-V ──→ All vendors
                  WebGPU ──→ WGSL ──→ All vendors (browser + native via wgpu)
```

**Agave's strategy**: Use vendor-specific APIs for maximum performance, with Vulkan and WebGPU as cross-platform fallbacks. The `Backend` interface abstracts all six behind a single dispatch.

| Platform | Primary | Fallback |
|----------|---------|----------|
| macOS (Apple Silicon) | Metal | CPU |
| Linux + NVIDIA | CUDA | Vulkan → CPU |
| Linux + AMD | ROCm | Vulkan → CPU |
| Linux + Intel | Vulkan | CPU |
| Browser (any GPU) | WebGPU | N/A |

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

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#e8f0fe',
  'primaryTextColor': '#1a1a2e',
  'primaryBorderColor': '#4a6cf7',
  'lineColor': '#4a6cf7',
  'secondaryColor': '#f0f4ff',
  'tertiaryColor': '#f8f9ff',
  'edgeLabelBackground': '#ffffff',
  'clusterBkg': '#f0f4ff',
  'clusterBorder': '#4a6cf7',
  'titleColor': '#1a1a2e',
  'nodeTextColor': '#1a1a2e',
  'fontFamily': 'ui-monospace, SFMono-Regular, monospace'
}}}%%
flowchart LR
    Input["x\n(hidden state)"]

    subgraph Unfused["Unfused: 8 VRAM reads/writes"]
        direction LR
        G1["gate_proj\nGEMV"] -->|"write gate\nto VRAM"| VRAM1["VRAM"]
        VRAM1 -->|"read gate\nfrom VRAM"| Gelu["GELU\nactivation"]
        U1["up_proj\nGEMV"] -->|"write up\nto VRAM"| VRAM2["VRAM"]
        VRAM2 -->|"read up\nfrom VRAM"| Mul["Element-wise\nmultiply"]
        Gelu --> Mul
        Mul -->|"write mid\nto VRAM"| VRAM3["VRAM"]
        VRAM3 -->|"read mid\nfrom VRAM"| D1["down_proj\nGEMV"]
    end

    subgraph Fused["Fused megakernel: 2 VRAM reads/writes"]
        direction LR
        FG["gate_proj\n+ GELU\n+ up_proj\n+ multiply\n(all in registers)"] -->|"write once\nto VRAM"| FV["VRAM"]
        FV -->|"read once\nfrom VRAM"| FD["down_proj\nGEMV"]
    end

    Input --> G1
    Input --> U1
    Input -->|"load once"| FG


## The Dispatcher Pattern

Model code never imports backend implementations directly. Instead, the `Backend` tagged union with `inline else` dispatch resolves **at compile time** (during compilation, not when the program runs — zero runtime overhead). Every model calls the same `be.gemv()` regardless of which hardware is present.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#e8f0fe',
  'primaryTextColor': '#1a1a2e',
  'primaryBorderColor': '#4a6cf7',
  'lineColor': '#4a6cf7',
  'secondaryColor': '#f0f4ff',
  'tertiaryColor': '#f8f9ff',
  'edgeLabelBackground': '#ffffff',
  'clusterBkg': '#f0f4ff',
  'clusterBorder': '#4a6cf7',
  'titleColor': '#1a1a2e',
  'nodeTextColor': '#1a1a2e',
  'fontFamily': 'ui-monospace, SFMono-Regular, monospace'
}}}%%
flowchart LR
    Model["Model code\nllama.zig / gemma.zig"]
    Dispatcher["Backend dispatcher\nbackend.zig"]

    Model -->|"be.gemv(...)"| Dispatcher

    Dispatcher -->|"inline else\n(compile-time)"| CPU["CpuBackend\ngemvQ4_0 / gemvBF16"]
    Dispatcher --> Metal["MetalBackend\nMSL compute shader"]
    Dispatcher --> CUDA["CudaBackend\nPTX kernel"]
    Dispatcher --> Vulkan["VulkanBackend\nSPIR-V shader"]
    Dispatcher --> ROCm["RocmBackend\nAMDGCN kernel"]
    Dispatcher --> WebGPU["WebGpuBackend\nWGSL shader"]

    subgraph "Never imported by models"
        CPU
        Metal
        CUDA
        Vulkan
        ROCm
        WebGPU
    end


```zig
pub const Backend = union(enum) {
    cpu: *CpuBackend,
    metal: *MetalBackend,
    vulkan: *VulkanBackend,
    cuda: *CudaBackend,
    rocm: *RocmBackend,
    webgpu: *WebGpuBackend,

    pub fn gemv(self: Backend, ...) void {
        switch (self) {
            inline else => |be| be.gemv(...),
        }
    }
};
```

This gives zero-overhead dispatch (no **vtable** — virtual function table used for dynamic dispatch in object-oriented languages, no function pointers) while keeping model code hardware-agnostic.

## UMA (Unified Memory Architecture)

On **UMA** platforms (where CPU and GPU share the same physical memory chips, unlike **discrete GPUs** which have separate VRAM) like Apple Silicon and NVIDIA Grace, GPU backends can wrap existing CPU allocations as GPU buffers with zero copies. This eliminates the biggest bottleneck in traditional GPU inference: copying weights from system RAM across the PCIe bus into separate VRAM.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#e8f0fe',
  'primaryTextColor': '#1a1a2e',
  'primaryBorderColor': '#4a6cf7',
  'lineColor': '#4a6cf7',
  'secondaryColor': '#f0f4ff',
  'tertiaryColor': '#f8f9ff',
  'edgeLabelBackground': '#ffffff',
  'clusterBkg': '#f0f4ff',
  'clusterBorder': '#4a6cf7',
  'titleColor': '#1a1a2e',
  'nodeTextColor': '#1a1a2e',
  'fontFamily': 'ui-monospace, SFMono-Regular, monospace'
}}}%%
flowchart TD
    subgraph Discrete["Discrete GPU (NVIDIA RTX, AMD RX)"]
        direction LR
        SysRAM["System RAM\n(weights loaded here)"]
        PCIe["PCIe Bus\n~64 GB/s"]
        VRAM["VRAM\n(GPU-only memory)"]
        dGPU["GPU Compute"]

        SysRAM -->|"cudaMemcpy\n(explicit copy)"| PCIe --> VRAM --> dGPU
    end

    subgraph UMA["UMA (Apple Silicon, NVIDIA Grace)"]
        direction LR
        SharedMem["Shared Physical Memory\n(weights live here once)"]
        uGPU["GPU Compute"]
        uCPU["CPU Compute"]

        SharedMem -->|"zero-copy pointer\nnewBufferWithBytesNoCopy"| uGPU
        SharedMem -->|"normal pointer"| uCPU
    end


- **Metal**: `newBufferWithBytesNoCopy` wraps mmap'd weights directly
- **CUDA**: `cudaMallocManaged` for transparent access
- **Vulkan**: `HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL` memory type

All GPU backends use **deferred dispatch** — operations are encoded into **command buffers** (queues of GPU operations) without blocking. Models call `be.sync()` only when CPU code needs to read GPU-produced data.

### sdpaWithStats

Extended SDPA that returns per-head softmax statistics for split-attention merge:

```zig
be.sdpaWithStats(q, keys, values, k_new, v_new, output,
                 head_max, head_sum,  // per-head max and sum(exp)
                 nh, nkv, hd, seq_len, scale, kv_type_k, kv_type_v);
```

Used by the split-attention path when KV cache spans GPU and CPU tiers. The `head_max` and `head_sum` arrays enable online softmax merging of partial attention outputs from different devices.

`sdpaWithStats` wraps native SDPA on all backends — no CPU delegates. Stats (head max/sum) are exported alongside the attention output for online softmax merging.

### sdpaPaged

Paged SDPA handles non-contiguous KV cache blocks via `PagedKvView` — a block table that maps logical positions to physical blocks:

```zig
be.sdpaPaged(q, kv_view, k_new, v_new, output, nh, nkv, hd, scale, kv_type_k, kv_type_v);
```

Instead of flat `keys[t * kvd]` offset arithmetic, the kernel computes `block_table[t / block_size]` → physical block → `keys[pos_in_block * kvd]`. Models use 16-token blocks allocated on demand, so memory scales with actual sequence length rather than maximum context window.

CPU backend has native paged SDPA with thread-pool parallelism across query heads. GPU backends use CPU fallback via `@hasDecl` detection.

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

**CUDA**: native GPU GEMM (Q8_0), batched RMSNorm and RoPE kernels compiled to PTX. The f32 SDPA uses a native batched GPU sdpa_prefill kernel. The turbo KV path uses sequential single-token GPU sdpa calls.

**CPU**: parallel GEMV-based GEMM via thread pool, parallel-head SDPA with bulk KV append. On macOS, F32 GEMV and GEMM dispatch to Apple's Accelerate.framework (`cblas_sgemm`), which uses the AMX matrix coprocessor for ~4x speedup over NEON SIMD.

## Backend-Specific Notes

**Metal** (`metal.zig`): MSL compute shaders with **threadgroup**-level (a group of threads that execute together and can share fast on-chip memory) `simd_sum` reduction. Buffer caching eliminates ~800 ObjC alloc/release per token. [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691) with block_size=16 (fits 32KB threadgroup memory). Prefill: native GEMM (f32/Q8_0/Q4_0), batched RoPE, dual-source FA2, zero per-layer flush. **Megakernel**: 70 pipelines including 11 fused FFN kernels and 5 true megakernels with atomic grid sync. Sparse V threshold in SDPA.

**CUDA** (`cuda.zig`): Zig kernels compiled to PTX via `nvptx64-cuda` target — no CUDA C++ dependency. Driver API loaded dynamically via `dlopen`. Deferred execution with activation caching for zero-sync SDPA. Prefill: native GEMM (Q8_0), batched RMSNorm/RoPE. **Megakernel**: 43 kernels including 5 fused FFN kernels (SiLU × Q8_0/Q4_K/Q5_K/Q6_K and GELU × Q8_0) and 3 true megakernels. Sparse V threshold in SDPA.

**WebGPU** (`webgpu.zig`): WGSL compute shaders loaded via wgpu-native C API. Dynamic library loading (`dlopen`). Enabled by default in the build system. **Lazy readback cache**: activation buffers stay on GPU between operations — `cacheGpuResult` registers GPU output in `buf_cache`, and `getOrUpload` finds it on next access. Downloads only happen on `sync()`. This eliminates ~200 CPU↔GPU round-trips per token. 45 WGSL compute shaders covering all core ops including quantized GEMV for all formats. Buffer lifecycle uses deferred destruction — params and cache-evicted buffers are queued for cleanup during `sync()` to avoid destroying buffers still referenced by pending command buffers.

**Vulkan** (`vulkan.zig`): Pre-compiled SPIR-V compute shaders. Subgroup arithmetic for reductions. Fused single-dispatch normalization/softmax. Works on all vendors including Apple (via MoltenVK). No megakernel support.

**ROCm** (`rocm.zig`): HIP Runtime API loaded dynamically. AMDGCN kernels compiled from Zig via `amdgcn-amdhsa` target. Same deferred execution pattern as CUDA. **Megakernel**: 28 kernels including 1 true megakernel (Qwen Q8). Sparse V threshold in SDPA.

---

## Distributed Inference

All GPU backends support distributed inference via `src/parallel/transport.zig`. Three transports: **TCP** (cross-node), **POSIX shm** (same-node zero-copy), **NCCL** (GPU-optimized RoCE RDMA, loaded via `dlopen`). Modes: tensor parallelism (`--tp 2` splits weights), pipeline parallelism (`--pp 2` splits layers), hybrid TP+PP, disaggregated prefill/decode (`--disagg`). Device selection via `--device N`.

NCCL integration requires `cuDevicePrimaryCtxRetain` (not `cuCtxCreate`) — NCCL uses the CUDA primary context and will corrupt a separate driver API context. Device pointer `allReduceAdd` passes GPU activation cache pointers directly to NCCL when data is dirty on device; when CPU fallback has written to host (stale on GPU), uploads to a device staging buffer first.

See [Parallelism docs](../PARALLELISM.md) for full details.

---

## Common Pitfalls

**Never import backend implementations directly**: Model code uses `@import("backend/backend.zig")`, never `@import("backend/cuda.zig")`. Backend-specific types (`CUcontext`, `MTLDevice`) stay private to their backend file.

**Missing `be.sync()` before CPU reads**: GPU operations are asynchronous. If you need to read GPU-produced data on CPU (e.g., argmax on logits), call `be.sync()` first. On CPU backend, sync is a no-op.

**Metal threadgroup memory limit**: Must stay under 32KB total. Calculate: `q_local + kv_block + out_acc + scores + shared`. Pipeline creation fails silently without the error logging in `makePipeline`.

**WebGPU buffer cache generation**: The lazy readback cache uses `upload_generation` to track freshness. Every `sync()` bumps the generation, invalidating all cached activation buffers. Weight buffers survive (they're uploaded once and never invalidated).

**In the code:** [src/backend/backend.zig](../../src/backend/backend.zig) (dispatcher), [src/backend/](../../src/backend/) (cpu, metal, cuda, vulkan, rocm implementations), [src/backend/kernels/](../../src/backend/kernels/) (GPU kernel sources)

**Next:** [Chapter 9: CPU SIMD Optimization →](09-cpu-simd-optimization.md) | **Back:** [Chapter 7: Sampling ←](07-sampling.md) | **Product docs:** [Architecture](../ARCHITECTURE.md) · [Models](../MODELS.md)
