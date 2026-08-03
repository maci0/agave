# Chapter 8: Backends

**Prerequisites:** [Chapter 0: Getting Started](00-getting-started.md), [Chapter 1: Tokens and Text](01-tokens-and-text.md), [Chapter 2: The Transformer](02-the-transformer.md), [Chapter 3: Feed-Forward Networks](03-feed-forward-networks.md) (all helpful, not required)

**Time:** ~20 min

Inference can run on different compute backends: **CPU** (universal, always available), **GPU** (massively parallel — thousands of cores organized into **warps/wavefronts** that execute via **SIMT** — Single Instruction Multiple Thread, where groups of 32-64 threads run the same instruction in lockstep on different data), or specialized **accelerators** (purpose-built hardware like TPUs, NPUs, or FPGAs optimized for specific workloads). Each backend provides a **compute API** that lets you write **kernels** (small programs that run on the hardware) and dispatch them.

**SIMD vs SIMT:** CPUs use **SIMD** (one instruction operates on a vector register of packed values, e.g., 8 f32s in AVX2 — see [Chapter 9](09-cpu-simd-optimization.md)). GPUs use **SIMT** (one instruction is executed by many threads simultaneously, each with its own registers and program counter). The distinction matters: SIMD has no divergence — all lanes do the same thing. SIMT threads can branch independently, but divergent branches serialize.

## The GPU Landscape

Each hardware **vendor** (manufacturer — NVIDIA, Apple, AMD, etc.) has its own API. Every backend compiles kernel source to an intermediate representation, then the GPU driver translates that to native machine code at runtime.

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Zig["Zig kernel source"]:::setup

    PTX["PTX bytecode"]:::migration
    AMDGCN["AMDGCN bytecode"]:::migration
    NEON["NEON / AVX2\n(native binary)"]:::migration
    MSL["MSL shader"]:::setup
    GLSL["GLSL compute shader"]:::setup
    WGSL["WGSL shader"]:::setup
    MetalIR["Metal IR"]:::migration
    SPIRV["SPIR-V bytecode"]:::migration
    WGSL2["WGSL (interpreted)"]:::migration
    NVIDIA["NVIDIA GPU"]:::success
    AMD["AMD GPU"]:::success
    CPU["CPU"]:::success
    Apple["Apple Silicon GPU"]:::success
    AnyGPU["Any GPU\n(Vulkan driver)"]:::success
    Browser["Browser / native\n(wgpu)"]:::success

    Zig --> PTX
    Zig --> AMDGCN
    Zig --> NEON

    PTX --> NVIDIA
    AMDGCN --> AMD
    NEON --> CPU

    MSL --> MetalIR
    GLSL --> SPIRV
    WGSL --> WGSL2

    MetalIR --> Apple
    SPIRV --> AnyGPU
    WGSL2 --> Browser

    subgraph Vendor-specific
        PTX
        AMDGCN
        MetalIR
    end

    subgraph Cross-platform
        SPIRV
        WGSL2
    end
```

| Platform | Vendor | Language | Compiled Format | Scope |
|----------|--------|----------|-----------------|-------|
| **CPU** | All | Zig + SIMD | Native (NEON/AVX2) | All platforms |
| **CUDA** | NVIDIA | Zig → PTX | PTX | NVIDIA GPUs only |
| **Metal** | Apple | MSL | Metal IR | Apple Silicon only |
| **ROCm/HIP** | AMD | Zig → HSACO | AMDGCN | AMD GPUs only |
| **Vulkan** | Khronos | GLSL | SPIR-V | All vendors (cross-platform) |
| **WebGPU** | W3C | WGSL | WGSL source | All vendors (browser + native) |

The "Compiled Format" column shows the **IR** (Intermediate Representation — compiled bytecode that the GPU driver converts to native machine code at runtime, not final executable code).

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
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Input["x\n(hidden state)"]:::setup

    subgraph Unfused["Unfused: 8 VRAM reads/writes"]
        direction LR
        G1["gate_proj\nGEMV"]:::sync
        VRAM1["VRAM"]:::danger
        Gelu["GELU\nactivation"]:::sync
        U1["up_proj\nGEMV"]:::sync
        VRAM2["VRAM"]:::danger
        Mul["Element-wise\nmultiply"]:::sync
        VRAM3["VRAM"]:::danger
        D1["down_proj\nGEMV"]:::sync

        G1 -->|"write gate\nto VRAM"| VRAM1
        VRAM1 -->|"read gate\nfrom VRAM"| Gelu
        U1 -->|"write up\nto VRAM"| VRAM2
        VRAM2 -->|"read up\nfrom VRAM"| Mul
        Gelu --> Mul
        Mul -->|"write mid\nto VRAM"| VRAM3
        VRAM3 -->|"read mid\nfrom VRAM"| D1
    end

    subgraph Fused["Fused megakernel: 2 VRAM reads/writes"]
        direction LR
        FG["gate_proj\n+ GELU\n+ up_proj\n+ multiply\n(all in registers)"]:::sync
        FV["VRAM"]:::migration
        FD["down_proj\nGEMV"]:::sync

        FG -->|"write once\nto VRAM"| FV
        FV -->|"read once\nfrom VRAM"| FD
    end

    Input --> G1
    Input --> U1
    Input -->|"load once"| FG
```

## The Dispatcher Pattern

Model code never imports backend implementations directly. Instead, the `Backend` tagged union with `inline else` dispatch resolves **at compile time** (during compilation, not when the program runs — zero runtime overhead). Every model calls the same `be.gemv()` regardless of which hardware is present.

```text
Backend = union { cpu, metal, vulkan, cuda, rocm, webgpu }
gemv(args):
  match backend: each variant.gemv(args)   # inline else at compile time
```

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Model["Model code\nllama.zig / gemma.zig"]:::setup
    Dispatcher["Backend dispatcher\nbackend.zig"]:::migration
    CPU["CpuBackend\ngemvQ4_0 / gemvBF16"]:::sync
    Metal["MetalBackend\nMSL compute shader"]:::sync
    CUDA["CudaBackend\nPTX kernel"]:::sync
    Vulkan["VulkanBackend\nSPIR-V shader"]:::sync
    ROCm["RocmBackend\nAMDGCN kernel"]:::sync
    WebGPU["WebGpuBackend\nWGSL shader"]:::sync

    Model -->|"be.gemv(...)"| Dispatcher

    Dispatcher -->|"inline else\n(compile-time)"| CPU
    Dispatcher --> Metal
    Dispatcher --> CUDA
    Dispatcher --> Vulkan
    Dispatcher --> ROCm
    Dispatcher --> WebGPU

    subgraph "Never imported by models"
        CPU
        Metal
        CUDA
        Vulkan
        ROCm
        WebGPU
    end
```

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
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    subgraph Discrete["Discrete GPU (NVIDIA RTX, AMD RX)"]
        direction LR
        SysRAM["System RAM\n(weights loaded here)"]:::setup
        PCIe["PCIe Bus\n~64 GB/s"]:::danger
        VRAM["VRAM\n(GPU-only memory)"]:::migration
        dGPU["GPU Compute"]:::success

        SysRAM -->|"cudaMemcpy\n(explicit copy)"| PCIe --> VRAM --> dGPU
    end

    subgraph UMA["UMA (Apple Silicon, NVIDIA Grace)"]
        direction LR
        SharedMem["Shared Physical Memory\n(weights live here once)"]:::setup
        uGPU["GPU Compute"]:::success
        uCPU["CPU Compute"]:::success

        SharedMem -->|"zero-copy pointer\nnewBufferWithBytesNoCopy"| uGPU
        SharedMem -->|"normal pointer"| uCPU
    end
```

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

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Token["Logical token position t"]:::setup
    BlockIdx["Block table index\nblock_table[t / block_size]"]:::migration
    PhysBlock["Physical block ID\n(16-token block, on-demand alloc)"]:::migration
    KVSlot["KV slot within block\nkeys[pos_in_block * kvd]"]:::migration
    SDPA["SDPA kernel\nattention output"]:::success

    Token -->|"t / block_size"| BlockIdx
    BlockIdx -->|"dereference"| PhysBlock
    PhysBlock -->|"t % block_size"| KVSlot

    subgraph BlockTable["PagedKvView — block table indirection"]
        direction LR
        BT0["block_table[0] = phys#4"]:::setup
        BT1["block_table[1] = phys#11"]:::setup
        BT2["block_table[2] = phys#2"]:::setup
        BT3["block_table[3] = phys#7"]:::setup
    end

    subgraph PhysPool["Physical KV pool (non-contiguous blocks)"]
        direction LR
        P2["Block #2\ntokens 32-47"]:::sync
        P4["Block #4\ntokens 0-15"]:::sync
        P7["Block #7\ntokens 48-63"]:::sync
        P11["Block #11\ntokens 16-31"]:::sync
    end

    BlockIdx --> BlockTable
    BlockTable -->|"physical block ID"| PhysPool
    PhysPool --> KVSlot

    KVSlot -->|"all heads in parallel\n(thread pool)"| SDPA
```

## Batched Prefill Dispatch

During prefill, the backend dispatches **batched** versions of the core ops — GEMM (instead of GEMV), batched RMSNorm, batched RoPE, and fused causal SDPA:

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Tokens["Input tokens\n[seq_len, hidden_dim]"]:::setup
    NextLayer["Layer N+1 ..."]:::success

    subgraph Attention["Attention block"]
        direction TB
        RN1["rmsNormBatched\nnormalize hidden states"]:::sync
        QKV["GEMM(Q, K, V)\none-shot projection\n[seq_len, head_dim * n_heads]"]:::sync
        RN2["rmsNormMulti\nnorm Q and K (Gemma3-style)"]:::sync
        RoPE["ropeBatched\nrotary position embeddings"]:::sync
        SDPA["sdpaPrefill (FA2)\ncausal self-attention\ndual-source K/V from GEMM"]:::sync
        ProjO["GEMM(O)\noutput projection\n[seq_len, hidden_dim]"]:::sync
        Add1["residual add"]:::migration

        RN1 --> QKV --> RN2 --> RoPE --> SDPA --> ProjO --> Add1
    end

    subgraph FFN["Feed-forward block"]
        direction TB
        RN3["rmsNormBatched\nnormalize after attention"]:::sync
        GateUp["GEMM(gate, up)\ndual projection\n[seq_len, ffn_dim]"]:::sync
        Gelu["GELU activation\n(in-register for megakernel)"]:::sync
        Mul["element-wise multiply\ngate * up"]:::sync
        Down["GEMM(down)\nproject back to hidden\n[seq_len, hidden_dim]"]:::sync
        Add2["residual add"]:::migration

        RN3 --> GateUp --> Gelu --> Mul --> Down --> Add2
    end

    Tokens --> Attention
    Attention --> FFN
    FFN -->|"next layer"| NextLayer
```

**Metal**: all batched ops are native GPU kernels. The GEMM uses one threadgroup per output row with weight reuse across tokens. The `sdpa_prefill_fa2` kernel reads old K/V from the cache and new K/V directly from GEMM output (dual-source), then a `copy_f32` kernel populates the cache — all in one command buffer with zero CPU-GPU flush.

**CUDA**: native GPU GEMM (Q8_0), batched RMSNorm and RoPE kernels compiled to PTX. The f32 SDPA uses a native batched GPU sdpa_prefill kernel. The turbo KV path uses sequential single-token GPU sdpa calls.

**CPU**: parallel GEMV-based GEMM via thread pool, parallel-head SDPA with bulk KV append. On macOS, F32 GEMV and GEMM dispatch to Apple's Accelerate.framework (`cblas_sgemm`), which uses the AMX matrix coprocessor for ~4x speedup over NEON SIMD.

## Backend-Specific Notes

**Metal** (`metal.zig`): MSL compute shaders with **threadgroup**-level (a group of threads that execute together and can share fast on-chip memory) `simd_sum` reduction. Buffer caching eliminates ~800 ObjC alloc/release per token. [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691) with block_size=16 (fits 32KB threadgroup memory). Prefill: native GEMM (f32/Q8_0/Q4_0), batched RoPE, dual-source FA2, zero per-layer flush. **Megakernel**: 70 pipelines including 11 fused FFN kernels and 5 true megakernels with atomic grid sync. Sparse V threshold in SDPA.

**CUDA** (`cuda.zig`): Zig kernels compiled to PTX via `nvptx64-cuda` target — no CUDA C++ dependency. Driver API loaded dynamically via `dlopen`. Deferred execution with activation caching for zero-sync SDPA. Prefill: native GEMM (Q8_0), batched RMSNorm/RoPE. **Megakernel**: 43 kernels including 5 fused FFN kernels (SiLU × Q8_0/Q4_K/Q5_K/Q6_K and GELU × Q8_0) and 3 true megakernels. Sparse V threshold in SDPA.

**WebGPU** (`webgpu.zig`): WGSL compute shaders loaded via wgpu-native C API. Dynamic library loading (`dlopen`). Enabled by default in the build system. **Lazy readback cache**: activation buffers stay on GPU between operations — `cacheGpuResult` registers GPU output in `buf_cache`, and `getOrUpload` finds it on next access. Downloads only happen on `sync()`. This eliminates ~200 CPU↔GPU round-trips per token. ~48 WGSL compute shaders covering all core ops including quantized GEMV for all formats. Buffer lifecycle uses deferred destruction — params and cache-evicted buffers are queued for cleanup during `sync()` to avoid destroying buffers still referenced by pending command buffers.

**Vulkan** (`vulkan.zig`): Pre-compiled SPIR-V compute shaders. Subgroup arithmetic for reductions. Fused single-dispatch normalization/softmax. Works on all vendors including Apple (via KosmicKrisp — use `libvulkan.1.dylib` loader, not MoltenVK directly). `sdpa_turbo` (TurboQuant KV) requires `GroupNonUniform` subgroup ops and is skipped gracefully on drivers that lack it (e.g. lavapipe/KosmicKrisp). **Disk-backed VkPipelineCache** at `~/.cache/agave/vk_pipeline_cache.bin` (1.2 MB for 49 shaders); speeds up re-init on drivers that honour it. No megakernel support.

**ROCm** (`rocm.zig`): HIP Runtime API loaded dynamically. AMDGCN kernels compiled from Zig via `amdgcn-amdhsa` target. Same deferred execution pattern as CUDA. **Megakernel**: 28 kernels including 1 true megakernel (Qwen Q8). Sparse V threshold in SDPA.

---

## Distributed Inference

See [Chapter 22: Distributed Inference](22-distributed-inference.md) for the full walkthrough of tensor and pipeline parallelism; this section covers only how the transport layer sits on top of the single-device dispatcher above.

All GPU backends support distributed inference via `src/parallel/transport.zig`. Three transports: **TCP** (cross-node), **POSIX shm** (same-node zero-copy), **NCCL** (GPU-optimized RoCE RDMA, loaded via `dlopen`). Modes: tensor parallelism (`--tp 2` splits weights), pipeline parallelism (`--pp 2` splits layers), hybrid TP+PP, disaggregated prefill/decode (`--disagg`). Device selection via `--device N`.

NCCL integration requires `cuDevicePrimaryCtxRetain` (not `cuCtxCreate`) — NCCL uses the CUDA primary context and will corrupt a separate driver API context. Device pointer `allReduceAdd` passes GPU activation cache pointers directly to NCCL when data is dirty on device; when CPU fallback has written to host (stale on GPU), uploads to a device staging buffer first.

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    subgraph Transports["Transport Layer (transport.zig)"]
        direction LR
        TCP["TCP\ncross-node\n(any network)"]:::setup
        SHM["POSIX shm\nsame-node\nzero-copy"]:::setup
        NCCL["NCCL\nRoCE RDMA\nGPU-optimized"]:::optional
    end

    subgraph Modes["Parallelism Modes"]
        direction TB

        subgraph TP["Tensor Parallelism (--tp N)"]
            TP0["GPU 0\nweight shard 0\nallReduceAdd after each layer"]:::sync
            TP1["GPU 1\nweight shard 1\nallReduceAdd after each layer"]:::sync
            TP0 <-->|"allReduce\n(shm or NCCL)"| TP1
        end

        subgraph PP["Pipeline Parallelism (--pp N)"]
            PP0["GPU 0\nlayers 0..L/2\nprefill + decode"]:::sync
            PP1["GPU 1\nlayers L/2..L\nprefill + decode"]:::sync
            PP0 -->|"activation\ntransfer"| PP1
        end

        subgraph Hybrid["Hybrid TP+PP"]
            H0["GPU 0\nlayer shard A"]:::sync
            H1["GPU 1\nlayer shard A"]:::sync
            H2["GPU 2\nlayer shard B"]:::sync
            H3["GPU 3\nlayer shard B"]:::sync
            H0 <-->|"TP allReduce"| H1
            H2 <-->|"TP allReduce"| H3
            H1 -->|"PP activation"| H2
        end

        subgraph Disagg["Disaggregated (--disagg)"]
            PNode["Prefill node\nfull context ingestion"]:::setup
            DNode["Decode node\ntoken-by-token generation"]:::success
            PNode -->|"KV cache transfer\n(TCP or NCCL)"| DNode
        end
    end

    Transports --> Modes
```

See [Parallelism docs](../PARALLELISM.md) for full details.

---

## Gotchas

**Never import backend implementations directly**: Model code uses `@import("backend/backend.zig")`, never `@import("backend/cuda.zig")`. Backend-specific types (`CUcontext`, `MTLDevice`) stay private to their backend file.

**Missing `be.sync()` before CPU reads**: GPU operations are asynchronous. If you need to read GPU-produced data on CPU (e.g., argmax on logits), call `be.sync()` first. On UMA platforms this is easy to miss because the CPU pointer and GPU pointer are the same memory. The read succeeds without crashing; it just returns stale data from before the GPU finished writing. On CPU backend, sync is a no-op.

**Metal threadgroup memory limit**: Must stay under 32KB total. Calculate: `q_local + kv_block + out_acc + scores + shared`. Pipeline creation fails silently without the error logging in `makePipeline`.

**WebGPU buffer cache generation**: The lazy readback cache uses `upload_generation` to track freshness. Every `sync()` bumps the generation, invalidating all cached activation buffers. Weight buffers survive (they're uploaded once and never invalidated).

**In the code:** [src/backend/backend.zig](../../src/backend/backend.zig) (dispatcher), [src/backend/](../../src/backend/) (cpu, metal, cuda, vulkan, rocm implementations), [src/backend/kernels/](../../src/backend/kernels/) (GPU kernel sources)

**Next:** [Chapter 9: CPU SIMD Optimization →](09-cpu-simd-optimization.md) | **Back:** [Chapter 7: Sampling ←](07-sampling.md) | **Product docs:** [Architecture](../ARCHITECTURE.md) · [Models](../MODELS.md)

---

## Glossary

**AMX (Apple Matrix coprocessor)** — Dedicated matrix multiplication hardware on Apple Silicon, accessed via Accelerate.framework.

**backend** — An abstraction layer that routes compute operations to a specific hardware implementation (CPU, Metal, CUDA, Vulkan, ROCm, WebGPU).

**command buffer** — A queue of GPU operations submitted together for execution.

**compute API** — A vendor-specific programming interface for dispatching work to a processor (e.g., Metal, CUDA, Vulkan).

**CUDA (Compute Unified Device Architecture)** — NVIDIA's GPU compute platform.

**deferred dispatch** — Encoding GPU operations into command buffers without blocking; execution happens when the buffer is committed.

**disaggregated inference** — Separating prefill (prompt processing) and decode (token generation) onto different nodes.

**dispatcher pattern** — A compile-time dispatch using Zig's tagged union with `inline else` to route calls to the correct backend at zero runtime cost.

**dlopen** — A POSIX function for loading shared libraries at runtime, avoiding compile-time dependencies on vendor SDKs.

**GLSL (OpenGL Shading Language)** — The shader language for Vulkan compute kernels; compiled to SPIR-V.

**HIP (Heterogeneous-compute Interface for Portability)** — AMD's GPU programming interface, API-compatible with CUDA.

**HSACO (HSA Code Object)** — The compiled binary format for AMD GPU kernels.

**IR (Intermediate Representation)** — Compiled bytecode (PTX, SPIR-V, Metal IR) that a GPU driver translates to native machine code at runtime.

**kernel fusion** — Combining multiple sequential operations into a single kernel to eliminate intermediate memory traffic.

**mmap (memory-mapped I/O)** — Mapping a file directly into virtual memory so the OS handles paging, avoiding explicit read calls.

**MSL (Metal Shading Language)** — Apple's GPU shader/compute language for Metal.

**NCCL (NVIDIA Collective Communications Library)** — A library for multi-GPU collective operations (all-reduce, broadcast) over PCIe or network.

**PCIe (Peripheral Component Interconnect Express)** — The bus connecting discrete GPUs to the CPU.

**pipeline parallelism** — Distributing transformer layers across multiple GPUs; activations flow sequentially between stages.

**PTX (Parallel Thread Execution)** — NVIDIA's intermediate assembly language for CUDA kernels, JIT-compiled to native GPU code.

**RDMA (Remote Direct Memory Access)** — Hardware-level network data transfer bypassing the CPU and OS, for ultra-low-latency GPU communication.

**RoCE (RDMA over Converged Ethernet)** — RDMA over standard Ethernet infrastructure.

**ROCm (Radeon Open Compute)** — AMD's open GPU compute platform.

**SPIR-V (Standard Portable Intermediate Representation)** — Vulkan's binary shader format.

**tensor parallelism** — Distributing weight shards across multiple GPUs; each computes a partial result, then all-reduce merges them.

**WGSL (WebGPU Shading Language)** — The shader language for WebGPU compute kernels.
