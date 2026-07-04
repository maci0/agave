# Glossary & Unexplained-Term Audit — Chapters 8–14

---

## Chapter 8: Backends

### Acronyms/terms used WITHOUT explanation on first use

| Term | Status |
|------|--------|
| **TPU** | Used in list of accelerators; never expanded (Tensor Processing Unit) |
| **NPU** | Used in list of accelerators; never expanded (Neural Processing Unit) |
| **FPGA** | Used in list of accelerators; never expanded (Field-Programmable Gate Array) |
| **f32** | Data type used throughout without explanation (32-bit floating-point) |
| **AVX2** | Mentioned alongside SIMD; not expanded until Ch 9 (Advanced Vector Extensions 2) |
| **CUDA** | Used as platform name; never expanded (Compute Unified Device Architecture) |
| **MSL** | Used in table; never expanded (Metal Shading Language) |
| **PTX** | Used in table; never expanded (Parallel Thread Execution) |
| **ROCm** | Used as platform name; not expanded (Radeon Open Compute) |
| **HIP** | Mentioned once; never expanded (Heterogeneous-Compute Interface for Portability) |
| **AMDGCN** | Used as bytecode format; never expanded (AMD Graphics Core Next) |
| **GLSL** | Used in table; never expanded (OpenGL Shading Language) |
| **SPIR-V** | Used in table; never expanded (Standard Portable Intermediate Representation — Vulkan) |
| **WGSL** | Used in table; never expanded (WebGPU Shading Language) |
| **HSACO** | Used in table; never expanded (HSA Code Object) |
| **mmap** | Used multiple times; never explained (memory-mapped file I/O) |
| **GELU** | Used without expansion (Gaussian Error Linear Unit) |
| **FFN** | Used without expansion (Feed-Forward Network) |
| **PCIe** | Used; never expanded (Peripheral Component Interconnect Express) |
| **SDPA** | Used without expansion (Scaled Dot-Product Attention) |
| **GQA** | Implied but not explicitly defined here (Grouped-Query Attention) |
| **KV cache** | Used without definition (Key-Value cache — stores past K/V states for autoregressive decoding) |
| **RMSNorm** | Used without expansion (Root Mean Square Normalization) |
| **RoPE** | Used without expansion (Rotary Position Embedding) |
| **FA2** | Used in "sdpaPrefill (FA2)"; never expanded (FlashAttention-2) |
| **AMX** | Used once; never expanded (Apple Matrix coprocessor / Advanced Matrix Extensions) |
| **NCCL** | Used; never expanded (NVIDIA Collective Communications Library) |
| **RDMA** | Used; never expanded (Remote Direct Memory Access) |
| **RoCE** | Used; never expanded (RDMA over Converged Ethernet) |
| **ObjC** | Abbreviation for Objective-C; never stated |
| **GEMV** | Used heavily; never expanded here (General Matrix-Vector multiply) |
| **GEMM** | Used; never expanded here (General Matrix-Matrix multiply) |
| **SiLU** | Used once; never expanded here (Sigmoid Linear Unit) |
| **dlopen** | Used; never explained (dynamic library open — POSIX function for runtime shared library loading) |
| **BF16** | Used as data type; not expanded here (Brain Floating-Point 16) |
| **F16** | Used as data type; not expanded (IEEE 754 half-precision floating-point) |
| **Q4_0 / Q8_0 / Q4_K / Q5_K / Q6_K** | Quantization format names used without definition in this chapter |
| **cblas_sgemm** | Function name; not explained (C BLAS single-precision GEMM) |
| **MLX** | Used in "MLX_Q4"; not expanded (Apple's ML framework) |
| **TP / PP** | Used in CLI flags; only briefly glossed (Tensor Parallelism / Pipeline Parallelism) |

### Glossary of terms introduced in Chapter 8

- **Backend** — an abstraction layer that routes compute operations to a specific hardware implementation (CPU, Metal, CUDA, Vulkan, ROCm, WebGPU)
- **Compute API** — a vendor-specific programming interface for dispatching work to a processor (e.g., Metal, CUDA, Vulkan)
- **Kernel** — a small computational function dispatched to the GPU (or CPU SIMD unit) for parallel execution
- **SIMT** — Single Instruction Multiple Thread; GPU execution model where groups of 32–64 threads execute the same instruction on different data
- **SIMD** — Single Instruction Multiple Data; CPU execution model where one instruction operates on a vector register of packed values
- **Warp / Wavefront** — a group of GPU threads (32 on NVIDIA, 64 on AMD) that execute in lockstep under SIMT
- **IR (Intermediate Representation)** — compiled bytecode (e.g., PTX, SPIR-V, Metal IR) that a GPU driver translates to native machine code at runtime
- **Kernel fusion** — combining multiple sequential operations into a single kernel to eliminate intermediate memory traffic
- **Registers** — on-chip storage in the processor core; ~100× faster than RAM/VRAM
- **TFLOPS** — teraflops; trillion floating-point operations per second
- **GB/s** — gigabytes per second; unit of memory bandwidth
- **VRAM** — Video RAM; dedicated GPU memory on discrete graphics cards
- **Dispatcher pattern** — a compile-time dispatch mechanism using Zig's tagged union with `inline else` to route calls to the correct backend at zero runtime cost
- **vtable** — virtual function table used for dynamic dispatch in OOP languages; avoided in Agave
- **UMA (Unified Memory Architecture)** — hardware design where CPU and GPU share the same physical memory, eliminating data copies
- **Discrete GPU** — a GPU with its own separate VRAM, connected to the CPU via PCIe
- **Deferred dispatch** — encoding GPU operations into command buffers without blocking; execution happens when the buffer is committed
- **Command buffer** — a queue of GPU operations that are submitted together for execution
- **sdpaWithStats** — extended Scaled Dot-Product Attention variant that returns per-head softmax statistics (max and sum) for merging partial attention outputs across devices
- **sdpaPaged** — SDPA variant that handles non-contiguous KV cache blocks via a block table (PagedKvView)
- **PagedKvView** — a block-table indirection structure mapping logical token positions to physical KV cache blocks
- **Tensor parallelism** — distributing weight shards across multiple GPUs; each GPU computes a partial result, then all-reduce merges them
- **Pipeline parallelism** — distributing transformer layers across multiple GPUs; activations flow sequentially between stages
- **Disaggregated inference** — separating prefill (prompt ingestion) and decode (token generation) onto different nodes
- **allReduceAdd** — collective operation that sums partial results across all participating GPUs
- **Prefill** — the initial phase of inference where all prompt tokens are processed in parallel (using GEMM), before autoregressive decode begins
- **Megakernel** — a GPU kernel that executes an entire transformer layer (or large fused subgraph) in a single dispatch

---

## Chapter 9: CPU SIMD Optimization

### Acronyms/terms used WITHOUT explanation on first use

| Term | Status |
|------|--------|
| **ARM** | Processor architecture; never expanded (Advanced RISC Machines, though commonly just "ARM") |
| **x86_64** | Processor architecture; not explained |
| **AVX-512** | Mentioned but not expanded (Advanced Vector Extensions 512-bit) |
| **NEON** | Named as ARM SIMD; not expanded (NEON Advanced SIMD) |
| **YMM registers** | Mentioned; not explained (256-bit SIMD registers on x86) |
| **L1/L2 cache** | Used without definition (Level 1 / Level 2 processor cache) |
| **ALU** | Used once; never expanded (Arithmetic Logic Unit) |
| **MAC** | Used once in "dequant + MAC"; never expanded (Multiply-Accumulate) |
| **BF16** | Used as data type; not expanded here |
| **F16** | Used as data type; not expanded here |
| **NR** | Used as "NR=2" and "NR=4"; not expanded (Number of Rows per batch) |
| **K-quant** | Used; format family not defined here (K-type quantization formats from llama.cpp) |
| **Q4_0** | Quantization format; meaning only partially explained via code |
| **nibble** | Used without definition (a 4-bit value, half a byte) |

### Glossary of terms introduced in Chapter 9

- **@Vector** — Zig's portable SIMD type that maps to hardware vector registers; e.g., `@Vector(8, f32)` is 8 packed f32s
- **SIMD register** — a wide hardware register (128–512 bits) that holds multiple data elements for parallel processing
- **Memory alignment** — the requirement that data addresses be multiples of a specific byte count (e.g., 32 bytes for AVX2) for optimal SIMD load performance
- **@splat** — Zig builtin that broadcasts a scalar value to all lanes of a SIMD vector
- **@reduce** — Zig builtin that collapses a SIMD vector to a scalar via a specified operation (e.g., `.Add` for horizontal sum)
- **Reduction tree** — a pair-wise hierarchical reduction pattern that sums vector lanes in log₂(N) steps
- **@mulAdd (FMA)** — Fused Multiply-Add; a single instruction that computes `a*b+c` with no intermediate rounding
- **FMA unit** — a dedicated hardware execution unit for fused multiply-add operations, separate from regular ALUs
- **Multi-row GEMV batching** — processing multiple output rows simultaneously to amortize the cost of loading the input vector from memory
- **Register pressure** — the constraint imposed by having a finite number of hardware SIMD registers; exceeding it causes spills to slower stack memory
- **Tail loop** — a scalar cleanup loop that handles remaining elements when the data length is not a multiple of the SIMD vector width
- **Dequantize** — convert quantized (compressed) weight values back to floating-point within the inner loop, without materializing the full matrix
- **Q4_0 block** — a quantization block encoding 32 elements into 18 bytes: a 2-byte f16 scale + 16 bytes of packed 4-bit nibbles
- **Cache locality** — the property of accessing memory in sequential order to maximize CPU cache hits and minimize cache misses
- **Prefetching** — hinting the CPU to load data into cache before it is needed, to hide memory latency
- **ReLU** — Rectified Linear Unit; activation function defined as max(0, x)
- **SoftPlus** — activation function defined as log(1 + eˣ)
- **RMSNorm** — Root Mean Square Normalization; a two-pass operation (compute RMS, then normalize and weight)
- **Activation sparsity** — the phenomenon where a significant fraction (~40% for SiLU) of activation values are near-zero after nonlinear activation, allowing those computations to be skipped
- **isBlockSparse** — a SIMD max-abs check that determines whether all input values in a block are below a threshold, enabling the block to be skipped

---

## Chapter 10: Memory Safety

### Acronyms/terms used WITHOUT explanation on first use

| Term | Status |
|------|--------|
| **GPA** | Mentioned in arena diagram; never expanded (General Purpose Allocator) |
| **AST** | Used in arena allocator section; never expanded (Abstract Syntax Tree) |
| **HTTP** | Used as example; never expanded (Hypertext Transfer Protocol) |

### Glossary of terms introduced in Chapter 10

- **defer** — Zig keyword that schedules a statement to execute when the current scope exits, regardless of whether the exit is normal or via error
- **errdefer** — Zig keyword that schedules a statement to execute only if the current scope exits via an error return
- **Stack unwinding** — the reverse-order execution pattern of deferred statements: last declared runs first
- **Explicit allocation** — Zig's memory model where every allocation must be paired with a manual free; no garbage collector
- **Partial initialization cleanup** — using `errdefer` to free resources already acquired when a multi-step initialization fails partway through
- **Arena allocator** — a bulk allocator (`std.heap.ArenaAllocator`) that frees all its allocations at once via `deinit()`, useful for short-lived temporary data
- **std.testing.allocator** — Zig's test allocator that tracks all allocations and automically detects memory leaks when a test completes
- **deinit() pattern** — convention where structs with owned resources provide a `deinit()` method that releases all internal allocations
- **Scope** — a block of code delimited by `{` `}` within which defer statements are bound; defers run at the end of their enclosing scope

---

## Chapter 11: Metal Backend Internals

### Acronyms/terms used WITHOUT explanation on first use

| Term | Status |
|------|--------|
| **DRAM** | Used; never expanded (Dynamic Random-Access Memory) |
| **D2H / H2D** | Used in diagram; never expanded (Device-to-Host / Host-to-Device transfer) |
| **ObjC** | Abbreviation used; not expanded (Objective-C) |
| **MTLBuffer** | Metal API type; not explained beyond context |
| **MTLCommandQueue** | Metpe; not explained |
| **MTLCommandBuffer** | Metal API type; not explained |
| **MTLComputeCommandEncoder** | Metal API type; not explained |
| **MSL** | Used; not re-expanded here (Metal Shading Language) |
| **simd_sum** | Metal SIMD function; not explained |
| **threadgroup** | Partially explained in Ch 8; used heavily without re-definition |
| **pipeline** | Used as "Metal pipeline" / "pipeline state"; refers to a compiled GPU program object |
| **SigLIP / SigLIP-2** | Used; never expanded (Sigmoid Loss for Language-Image Pre-training) |
| **ViT** | Used; never expanded (Vision Transformer) |
| **mmproj** | Used; not expanded here (multimodal projector) |
| **n_patches** | Used; not defined (number of image patches produced by the vision encoder) |
| **JIT** | Implied by runtime compile; never stated (Just-In-Time compilation) |

### Glossary of terms introduced in Chapter 11

- **Zero-copy buffer wrapping** — creating a Metal GPU buffer object (`MTLBuffer`) that references existing CPU memory without copying data, via `newBufferWithBytesNoCopy`
- **MTLResourceStorageModeShared** — Metal storage mode where CPU and GPU access the same memory region (UMA)
- **MTLResourceStorageModePrivate** — Metal storage mode where only the GPU can access the memory (used for scratch buffers)
- **Buffer caching** — storing `MTLBuffer` wrappers keyed by host pointer address to avoid repeated ObjC allocation overhead (~800 wrappers per token)
- **BufRef** — a struct containing a Metal buffer object and a byte offset, used to reference sub-regions within a page-aligned buffer
- **Page alignment** — the requirement that `newBufferWithBytesNoCopy` pointers be aligned to 16384-byte (16 KB) page boundaries on Apple Silicon
- **Command buffer batching** — maintaining a persistent command buffer and compute encoder across multiple kernel dispatches, committing all work at once
- **Memory barrier** — a GPU synchronization primitive (`memoryBarrierWithScope`) that ensures write visibility between kernel dispatches
- **Batch mode** — a Metal backend mode (`beginBatch`/`endBatch`) that suppresses intermediate memory barriers between independent operations
- **Flush** — committing the active command buffer, submitting it to the GPU, and waiting for completion
- **Sync point** — a moment where the CPU blocks until the GPU completes pending work, so CPU code can safely read GPU-produced data
- **Threadgroup memory** — fast on-chip shared memory accessible by all threads in a threadgroup; limited to 32 KB on Apple Silicon
- **Profiling counters** — runtime counters (dispatch_count, barrier_count, sync_count) tracked when `--profile` is enabled
- **GEMM dispatch decision** — the Metal backend's logic to select GEMV (single token) vs GEMM (batched tokens) based on `n_tok`
- **Token tiling** — GEMM optimization where multiple input tokens share a single weight load (e.g., TILE_T=8 for Q8_0)
- **Megakernel three-tier system** — Metal's fusion architecture: Tier 1 (fused FFN), Tier 2 (true megakernels with atomic grid sync), Tier 3 (auto-generated MSL from ModelDesc)
- **ModelDesc** — a struct describing model architecture (layers, dims, quant, activation) used by `mega_compose.zig` to auto-generate megakernel MSL

---

## Chapter 12: CPU Parallelism

### Acronyms/terms used WITHOUT explanation on first use

| Term | Status |
|------|--------|
| **futex** | Defined inline, but the expansion is informal ("fast userspace mutex"); the actual meaning is "fast userspace lock" from Linux kernel |
| **CAS** | Used in diagram ("cmpxchgWeak"); never expanded (Compare-And-Swap) |
| **Io** | Zig 0.16 type used without explanation of what it represents |

### Glossary of terms introduced in Chapter 12

- **Thread pool** — a set of persistent worker threads that sleep when idle and wake on demand, avoiding the overhead of thread creation per operation
- **Futex** — fast userspace mutex; a kernel primitive for efficient thread sleep/wake (`futexWait` and `futexWake`)
- **Generation counter** — an atomic variable that workers sleep on; incrementing it and calling futexWake signals new work
- **Atomic counter (work stealing)** — a shared counter that threads atomically increment to claim the next chunk of work, enabling dynamic load balancing
- **Grain size** — the number of work units (e.g., matrix rows) assigned per atomic fetch-add operation; controls the trade-off between contention and load balance
- **Main thread participation** — the pattern where the main thread does useful work alongside pool workers instead of idly waiting
- **Spin-wait** — busy-looping (checking a condition repeatedly) instead of sleeping; appropriate for microsecond-scale waits where futex overhead would dominate
- **spinLoopHint** — a CPU instruction hint (e.g., x86 `PAUSE`) that reduces power consumption during spin-wait loops
- **Memory ordering** — the guarantees about when writes by one thread become visible to other threads
- **.monotonic** — weakest atomic ordering; guarantees atomicity but no cross-thread synchronization of surrounding memory
- **.acquire** — atomic load ordering that guarantees all subsequent reads see writes that happened before a corresponding `.release` store
- **.release** — atomic store ordering that guarantees all prior writes are visible before this store becomes visible
- **.seq_cst** — sequential consistency; strongest atomic ordering where all threads observe the same total order of operations
- **False sharing** — performance degradation when different threads write to different variables that share the same CPU cache line, causing cache-line ping-pong between cores
- **Cache line** — the smallest unit of data transfer between CPU cache levels; typically 64 bytes
- **Cache-line padding** — inserting unused bytes to ensure that frequently-written variables by different threads occupy separate cache lines
- **fetchAdd** — atomic operation that reads the current value, adds a delta, and returns the original value, all in a single indivisible step
- **cmpxchgWeak** — atomic compare-and-exchange that may spuriously fail; used for lock-free state transitions
- **Inline threshold** — the minimum work size below which threading overhead exceeds benefit; work is run directly on the calling thread instead

---

## Chapter 13: Batched Dispatch and Fusion

### Acronyms/terms used WITHOUT explanation on first use

| Term | Status |
|------|--------|
| **MoE** | Used; never expanded (Mixture of Experts) |
| **SwiGLU** | Used as activation pattern name; never expanded (Swish-Gated Linear Unit) |
| **DeltaNet** | Architecture name used without definition; a linear-attention-based layer type used in Qwen3.5 |
| **TurboQuant / TurboQuant+** | Used multiple times; never defined (Agave's inline KV cache quantization format) |
| **TQ+** | Abbreviation of TurboQuant+; not expanded |
| **GQA** | Used; not expanded here (Grouped-Query Attention) |

### Glossary of terms introduced in Chapter 13

- **Dispatch overhead** — the CPU-side cost (~5–10 µs) of setting up pipeline state, binding buffers, and launching a GPU kernel
- **gemvMulti** — a batched GEMV interface that dispatches multiple matrix-vector multiplies sharing the same input vector in a single GPU command, reducing barriers
- **GemvOp** — a struct describing one GEMV operation within a gemvMulti batch: weight data, output buffer, and row count
- **Fusion** — combining sequential GPU operations into a single kernel so intermediate results stay in registers and never touch VRAM
- **addRmsNorm** — a fused operation that performs residual addition and RMS normalization in a single kernel dispatch
- **siluMul** — a fused operation computing `silu(a) * b` in one kernel, eliminating the intermediate activation buffer
- **SiLU** — Sigmoid Linear Unit; activation function defined as x · σ(x)
- **splitQGate** — a GPU kernel that deinterleaves Q and gate values from a single interleaved buffer into separate contiguous buffers, eliminating CPU–GPU sync round-trips
- **addScaled** — a fused operation computing `dst += src * scale` on GPU, used for MoE expert accumulation without CPU synchronization
- **Megakernel system** — Agave's three-tier GPU fusion architecture for eliminating dispatch overhead
- **Tier 1 (Fused FFN)** — combines gate GEMV + up GEMV + activation into a single dispatch (3→1 per FFN layer)
- **Tier 2 (True Megakernel)** — executes an entire transformer layer in a single dispatch using composable building blocks and atomic grid sync
- **Tier 3 (Composed Megakernel)** — auto-generates model-specific MSL source at runtime from a ModelDesc struct; no hand-written shader code needed
- **mega_grid_sync** — an atomic-counter-based grid-level barrier that synchronizes all threadgroups within a megakernel dispatch (Metal has no native grid barrier)
- **composeMSL** — a function in `mega_compose.zig` that generates MSL shader source code from a `ModelDesc` descriptor
- **beginBatch / endBatch** — Metal backend API for suppressing intermediate memory barriers between independent GPU operations; a single barrier is inserted at endBatch

---

## Chapter 14: Format Conventions

### Acronyms/terms used WITHOUT explanation on first use

| Term | Status |
|------|--------|
| **GGUF** | Partially introduced ("a single-file binary format designed by the llama.cpp project"); the acronym itself not expanded (GPT-Generated Unified Format) |
| **HF** | Used as shorthand; never expanded (HuggingFace) |
| **conv1d** | Used; never expanded (one-dimensional convolution) |
| **SSM** | Used in metadata keys ("ssm.conv_kernel"); never expanded (State Space Model) |
| **DeltaNet** | Architecture name used without definition |
| **GQA** | Used without expansion (Grouped-Query Attention) |
| **MLX** | Used; not expanded (Apple's array framework for ML research) |
| **SigLIP / SigLIP-2** | Used; never expanded (Sigmoid Loss for Language-Image Pre-training) |
| **ViT** | Used; never expanded (Vision Transformer) |
| **VL** | Used in "Qwen VL"; not expanded (Vision Language) |
| **mmproj** | Partially explained via context; abbreviation not formally expanded (multimodal projector) |
| **HSA** | Used in HSACO; never expanded (Heterogeneous System Architecture) |
| **JSON** | Used; never expanded (JavaScript Object Notation) |
| **MLP** | Used; never expanded (Multi-Layer Perceptron) |
| **BLAS** | Implied by cblas; never expanded (Basic Linear Algebra Subprograms) |

### Glossary of terms introduced in Chapter 14

- **GGUF** — GPT-Generated Unified Format; a single-file binary format designed by the llama.cpp project for mmap-friendly quantized model storage
- **SafeTensors** — a multi-file format from HuggingFace for storing PyTorch model weights safely (no pickle); uses JSON metadata headers
- **Format convention** — the set of assumptions about tensor layout, split order, metadata keys, and dimension ordering that differ between GGUF and SafeTensors
- **is_safetensors flag** — a boolean on the `Format` interface that decouples format detection from convention selection
- **Silent correctness failure** — a bug where the model runs without errors but produces garbage output due to mismatched format conventions
- **DeltaNet conv output split order** — the order in which Q, K, V slices are packed in the conv1d output buffer: Q,K,V for GGUF vs. K,Q,V for HuggingFace
- **GQA head mapping (tiling vs. interleaved)** — two different schemes for mapping query heads to KV heads: modulo-based tiling (GGUF, `h % n_kv`) vs. block-based grouping (HF, `h * n_kv / n_q`)
- **A_log pre-conversion** — the convention difference where GGUF stores SSM decay as pre-computed `-exp(A_log)` while SafeTensors stores raw `A_log` values requiring init-time conversion
- **Q/Gate split layout** — the memory layout difference for interleaved Q and gate values: element-interleaved per head (GGUF) vs. concatenated halves per head (SafeTensors)
- **Tensor name mapping** — translation between GGUF short names (e.g., `attn_qkv`) and HuggingFace full paths (e.g., `linear_attn.in_proj_qkv.weight`)
- **Attribute-less tensor** — a SafeTensors tensor name that lacks a `.weight` suffix (e.g., `A_log`), requiring special handling in the name translation
- **Metadata key mapping (gguf_hf_meta_map)** — a bidirectional table translating between GGUF metadata keys (e.g., `ssm.conv_kernel`) and HF config.json keys (e.g., `linear_conv_kernel_dim`)
- **Dimension order normalization** — reversing GGUF's inner-first dimension order during parsing so `dims[0]` always means the output (row) dimension, matching SafeTensors/PyTorch convention
- **Norm weight caching** — using a per-tensor fixed-size array cache for dequantized norm weights, avoiding stale GPU buffer reads caused by scratch-buffer reuse
- **mmproj file** — a separate GGUF file containing vision encoder and multimodal projector weights, loaded alongside the main language model
- **v. prefix** — tensor name prefix for vision encoder layers in mmproj GGUF files (e.g., `v.blk.0.attn_q.weight`)
- **mm. prefix** — tensor name prefix for multimodal projection head tensors (e.g., `mm.input_projection.weight`)
- **projection_dim** — the output embedding dimension of the vision encoder, which must match the language model's `n_embd` for embedding replacement
- **VTable** — a struct of function pointers implementing the Format interface's polymorphism (getTensor, getMetaU32, etc.)
- **rope_parameters nesting** — SafeTensors convention where `rope_theta` may be stored in a doubly-nested JSON path (`text_config.rope_parameters.rope_theta`)

---

## Coverage Status

- **All 7 chapters**: Read in full, every section analyzed. ✅
- **Acronyms**: Identified all acronyms/technical terms used without being explained on first use within each chapter. ✅
- **Glossaries**: Every new concept introduced in each chapter has a one-line definition entry. ✅
- **Cross-chapter dependencies**: Some terms (SIMD, GEMV, VRAM, etc.) are explained in one chapter but used without re-explanation in later chapters. The audit above captures terms unexplained *within each individual chapter*.
- **Limitation**: Some acronyms (e.g., f32, Q4_0) may have been defined in earlier chapters (1–7) not covered in this batch. The audit flags them as unexplained within the chapter scope examined.
