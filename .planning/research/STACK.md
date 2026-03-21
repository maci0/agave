# Technology Stack — Production Inference Engine

**Project:** Agave LLM Inference Engine
**Researched:** 2026-03-21
**Domain:** GPU kernel optimization and production serving for existing multi-backend inference engine

---

## Executive Summary

This research focuses on the specific technologies needed to eliminate CPU fallbacks in Metal/CUDA backends and add production serving capabilities to an existing Zig-based inference engine. The stack recommendations are prescriptive and based on 2025 industry best practices from vLLM, SGLang, FlashAttention-3, and TensorRT-LLM.

**Key Finding:** The project should adopt FlashAttention-2 tiling patterns (not FlashAttention-3), online softmax for numerical stability, hierarchical warp+block reductions, and SGLang-style RadixAttention with LRU eviction for production serving.

---

## 1. GPU SDPA Kernel Implementation

### Core Algorithm: FlashAttention-2 (not FA-3)

| Technology | Version/Variant | Why | Confidence |
|------------|-----------------|-----|------------|
| **FlashAttention-2 tiling** | Two-pass algorithm | FA-3 requires Hopper H100+, CUDA 12.3+, limited head dims, no compilation support yet. FA-2 is production-ready, cross-platform, and sufficient for target GPUs (M4 Pro, GB10 Blackwell). | **HIGH** |
| **Online softmax** | Milakov & Gimelshein algorithm | Single-pass numerically stable softmax using running max+sum. Eliminates 3-pass approach, reduces memory traffic by 1.33×. Standard in all modern attention kernels. | **HIGH** |
| **Tiling strategy** | Block sizes: Q/K 128×128, shared dim 32 | Balances SRAM/register pressure with occupancy. Smaller tiles (64×64) better for short sequences (<1024), larger (128×128) for long context (16K-64K tokens). Auto-tuning recommended per input shape. | **HIGH** |

**Rationale:**
- **FA-2 over FA-3:** FlashAttention-3 is beta-only, H100-exclusive, and doesn't yet support model compilation. FA-2 is battle-tested across CUDA, ROCm, and Metal (community ports exist).
- **Online softmax is mandatory:** Achieves numerical stability without the max subtraction + separate normalization passes. Computes `softmax(x) = exp(x - max) / sum(exp(x - max))` in a single pass using induction on running statistics.
- **Tiling enables SRAM residency:** Load Q/K/V blocks into fast on-chip memory (Metal threadgroup, CUDA shared memory), compute attention, write output. Minimizes HBM round-trips from O(N²) to O(N).

**Metal-Specific:**

| Issue | Root Cause | Solution | Confidence |
|-------|------------|----------|------------|
| **GPU SDPA produces wrong output** | Likely race condition in compute-based KV append or incorrect threadgroup barrier placement | Use `threadgroup_barrier(mem_flags::mem_threadgroup)` after shared memory writes. Verify tile indexing matches Q/K/V strides. Consider flush + CPU SDPA as fallback until debugged. | **MEDIUM** |
| **Encoder switching overhead** | Blit encoder for KV append → compute encoder for SDPA causes ~150ms/layer stalls | Fuse KV append into compute shader. Use `threadgroup` memory staging or direct device buffer writes. Avoid blit commands in hot path. | **HIGH** |
| **No native FP32 atomics** | `metal::atomic<float>` is emulated, slow | Use int32 atomics + `as_type<float>` reinterpretation, or redesign to avoid atomics (use threadgroup reductions instead). | **HIGH** |

**CUDA-Specific:**

| Issue | Root Cause | Solution | Confidence |
|-------|------------|----------|------------|
| **blockReduce hangs on Blackwell sm_121** | Likely compiler codegen bug with dynamic shared memory on new arch | Use warp-level `__shfl_down_sync` reductions exclusively (no block-level shared memory reduce) until CUDA toolkit fixes the issue, OR use serial thread-0 softmax as current workaround. File bug with NVIDIA. | **MEDIUM** |
| **Parallel softmax required for performance** | Serial softmax doesn't scale to long sequences | Implement two-tier reduction: warp shuffle (`__shfl_down_sync`) for intra-warp max/sum, then shared memory (array size = num_warps, typically 8) for inter-warp reduction. | **HIGH** |

**Implementation Checklist:**
- [ ] Implement online softmax (running max, running sum, induction-based correction)
- [ ] Use hierarchical reduction (warp shuffle → shared[8] → warp shuffle)
- [ ] Tile Q/K/V into SRAM blocks (default 128×128 for Q/K, 32 for shared dim)
- [ ] Fuse operations: load tile → compute attention → write output (minimize HBM traffic)
- [ ] Add auto-tuning for tile size based on sequence length and head dimensions
- [ ] **Metal:** Replace blit-based KV append with compute shader, verify barrier placement
- [ ] **CUDA:** Test warp-only reduction path on sm_121 before attempting block-level shared memory reductions

---

## 2. CUDA Quantized GEMV Kernels (Q4_K, Q5_K, Q6_K, FP8)

### Quantization-Aware Kernel Design

| Format | Block Structure | Dequant Strategy | Performance Target | Confidence |
|--------|----------------|------------------|-------------------|------------|
| **Q4_K** | 256-element blocks, 6 sub-blocks with independent scales | In-kernel dequant using warp shuffle for scale broadcast. Process 2-4 rows in parallel (share input vector across rows). | Match or exceed llama.cpp Metal Q4_K GEMV throughput. | **HIGH** |
| **Q5_K** | Similar to Q4_K but 5-bit weights | Unpack 5-bit → f32 using LUT or shift+mask. Vectorize inner loop (float4 for coalesced access). | ~90% of Q4_K throughput (slightly more dequant overhead). | **HIGH** |
| **Q6_K** | 6-bit weights, 256-element blocks | Eliminate bounds checks for full 64-element groups. Pre-compute scales for chunk processing. | ~80% of Q4_K throughput (more dequant ops but better accuracy). | **HIGH** |
| **FP8 E4M3/E5M2** | Native FP8 on Ada/Hopper+ | Use CUDA `__nv_fp8_e4m3` intrinsics if available, else 256-entry LUT for dequant. Tensor Core FP8 MMA on Hopper. | 2-3× faster than Q4_K on Hopper (native FP8 tensor cores). | **MEDIUM** |

**Key Optimizations (from 2025 research):**

1. **Multi-row GEMV:** Process 2-4 output rows per threadgroup. Share input vector `x` in shared memory across rows. Increases arithmetic intensity (compute/memory ratio).

2. **Row-wise quantization layout:** Q4_K/Q5_K/Q6_K use per-row scales. Assign thread blocks to rows, process row-local scales without synchronization.

3. **Warp shuffle for reduction:** Use `__shfl_down_sync(0xFFFFFFFF, val, offset)` for warp-local dot product accumulation (32 threads → 1 sum). Faster than shared memory for small reductions.

4. **Coalesced memory access:** Use `float4` or `uint4` loads for weight data. Ensures 128-bit aligned reads (maximizes memory bandwidth utilization).

5. **Dynamic shared memory:** Allocate shared memory for input vector `x` via kernel launch parameter (typically 32 bytes = 8 warps × 4 bytes for partial sums).

**W4A8 Considerations:**

For mixed-precision (W4 weights, A8 activations), existing kernels often bottleneck on CUDA Core dequantization. LiquidGEMM (2025) shows 2-instruction dequant path: `shift` + `fma` (fused multiply-add with scale). This enables overflow-safe dequant that keeps pace with Tensor Core throughput.

**Implementation Priority:**
1. **Q4_K first** (most common quantization for 7B-13B models)
2. **Q6_K second** (used for critical layers in mixed-quant models)
3. **Q5_K third** (less common, between Q4/Q6 in accuracy)
4. **FP8 last** (newer format, requires Ada/Hopper, high value for native tensor core path)

**Testing Strategy:**
- Golden tests against llama.cpp or ggml reference implementations
- Verify numerical accuracy within tolerance (Q4_K: ±0.01, Q6_K: ±0.001)
- Benchmark throughput (tokens/sec) on target hardware (GB10 Blackwell)
- Profile memory bandwidth utilization (target >80% of theoretical peak)

---

## 3. Continuous Batching & Request Scheduling

### Scheduler Architecture (vLLM-style)

| Component | Technology | Why | Confidence |
|-----------|-----------|-----|------------|
| **Batching algorithm** | Continuous batching (step-level scheduling) | Decouple batch processing from request lifecycle. Add/remove requests dynamically at each decode step instead of waiting for batch completion. Standard in vLLM, SGLang, TRT-LLM (2025). | **HIGH** |
| **Request queue** | Priority queue with earliest-deadline-first | Minimizes time-to-first-token (TTFT) for interactive apps. Use `std.PriorityQueue` with deadline as key. | **HIGH** |
| **Memory management** | PagedAttention block tables | Treat GPU KV cache like virtual memory. Fixed-size blocks (16 tokens typical) enable non-contiguous allocation, sharing, copy-on-write. Reduces fragmentation by 19-27% vs. contiguous allocation. | **HIGH** |
| **Concurrency primitive** | Lock-free work-stealing queue (Zig futex-based thread pool) | Distribute decode steps across CPU threads without mutex contention. Use atomics (`@atomicLoad`, `@atomicStore`) for state transitions. | **HIGH** |

**Scheduler Workflow (per decode step):**

```
1. Check request queue for new arrivals
2. Assemble batch from active sequences (up to max_batch_size)
3. If capacity remains, pull waiting requests from queue
4. Allocate/reuse KV cache blocks via PagedAttention
5. Dispatch batch to GPU (single kernel launch for entire batch)
6. Update sequence states (advance position, check EOS)
7. Remove completed sequences, free their KV blocks
8. Return to step 1
```

**Key Metrics:**
- **Throughput:** Total tokens/sec across all concurrent requests (target: 2-4× higher than serial processing)
- **TTFT:** Time from request arrival to first token generated (target: <50ms for interactive apps)
- **GPU utilization:** Percentage of time GPU is computing vs. idle (target: 85-92%)
- **Memory efficiency:** KV cache utilization (target: >80% of allocated GPU memory actively used)

**Implementation Notes:**
- Scheduler runs on CPU (lightweight, <2% overhead per SGLang v0.4 benchmarks)
- Avoid locks in hot path — use atomics for request state transitions
- Batch assembly must be fast (<1ms for 32 requests) to avoid bottlenecking GPU
- Support heterogeneous sequence lengths (continuous batching handles this naturally)

**PagedAttention Integration:**

| Requirement | Implementation | Rationale | Confidence |
|-------------|---------------|-----------|------------|
| **Block table** | `std.AutoHashMap(sequence_id, []BlockRef)` | Maps logical KV cache blocks (per sequence) to physical GPU blocks. `BlockRef = struct { device_ptr: usize, offset: usize }`. | **HIGH** |
| **Block allocator** | Doubly-linked free list with sentinel nodes | O(1) allocation and deallocation. Use `prev_free_block` and `next_free_block` pointers. | **HIGH** |
| **Block size** | 16 tokens (12.8 KB for 13B model) | Balances internal fragmentation vs. external fragmentation. vLLM authors tested extensively — 16 is optimal for most workloads. | **HIGH** |
| **Copy-on-Write** | Reference counting + lazy copy | Multiple sequences can share immutable prefix blocks. Only copy on first write (when sequence diverges). | **HIGH** |
| **Eviction policy** | LRU (least recently used) | Evict coldest blocks when GPU memory full. Track last access timestamp per block. | **MEDIUM** |

**Attention Kernel Modification:**

Standard SDPA: `sdpa(q, k, v, output)`

Paged SDPA: `sdpa(q, block_table, k_blocks, v_blocks, output)`

Kernel must:
1. Read block table to get physical block addresses for this sequence
2. Gather K/V from non-contiguous blocks
3. Compute attention as normal
4. Write output

**Performance Considerations:**
- Block indirection adds ~5-10% overhead vs. contiguous KV cache
- This is offset by 2-4× higher batch throughput (more sequences fit in memory)
- Critical: block gathering must be memory-coalesced (use GPU-side block table)

---

## 4. RadixAttention Prefix Caching

### Radix Tree KV Cache (SGLang-style)

| Component | Data Structure | Why | Confidence |
|-----------|---------------|-----|------------|
| **Tree structure** | Radix tree (prefix trie) with sequence-labeled edges | Maps token sequences → KV cache blocks. Edges can represent multi-token sequences (more compact than trie). Enables automatic longest-common-prefix (LCP) detection and sharing. | **HIGH** |
| **Node storage** | `struct RadixNode { children: HashMap<token, *Node>, kv_blocks: []BlockRef, ref_count: u32, last_access: u64 }` | Tracks KV blocks at this prefix, reference count for CoW, LRU timestamp for eviction. | **HIGH** |
| **Eviction policy** | Recursive LRU on leaf nodes | When GPU memory full, recursively evict least-recently-used leaf nodes. Free blocks bottom-up (leaves → root). Never evict nodes with `ref_count > 0`. | **HIGH** |
| **Insertion** | Tree traversal + edge splitting | Find LCP with existing tree. If partial match, split edge. Append new tokens as new branch. O(L) where L = sequence length. | **HIGH** |
| **Lookup** | Tree traversal for prefix match | Start at root, follow edges matching input tokens. Return deepest matching node's KV blocks. O(L) lookup. | **HIGH** |

**Radix Tree Properties:**

- **Space-efficient:** Shares common prefixes across all requests (vs. PagedAttention which shares only within manually specified batch)
- **Automatic reuse:** No user annotation needed — system detects shared prefixes at runtime
- **Compatible with continuous batching:** Scheduler can prioritize requests with long cached prefixes (cache-aware scheduling)

**Cache-Aware Scheduling:**

Extend priority queue to consider cache hit length:

```
Priority = -1 * (deadline + α * cached_prefix_length)
```

Where `α` weights cache hits vs. deadline urgency. Longer cached prefixes get higher priority (negative priority = earlier scheduling).

**LRU Eviction Algorithm:**

```
1. Identify all leaf nodes with ref_count == 0
2. Sort by last_access timestamp (oldest first)
3. Evict oldest leaf:
   a. Free its KV blocks
   b. Remove from parent's children map
   c. If parent becomes leaf and ref_count == 0, recurse
4. Repeat until sufficient memory freed
```

**Performance Expectations (from SGLang benchmarks):**

- **Prefix hit rate:** 60-80% for multi-turn conversations (typical chatbot workload)
- **Speedup:** 1.5-5× faster than vLLM for workloads with shared prefixes (e.g., many requests with same system prompt)
- **Memory overhead:** ~10% additional GPU memory for radix tree metadata (worthwhile for hit rate gains)
- **Scheduling overhead:** <2% of total inference time (SGLang v0.4 result)

**Implementation Phases:**

1. **Phase 1:** Basic radix tree with insert/lookup/evict operations (CPU-only, no GPU integration)
2. **Phase 2:** Integrate with PagedAttention (radix nodes reference paged KV blocks)
3. **Phase 3:** Cache-aware scheduler (prioritize high-prefix-match requests)
4. **Phase 4:** Hierarchical storage (GPU → CPU memory → disk offload for cold prefixes)

**Edge Cases to Handle:**

- **Node splitting:** When new sequence partially matches existing edge, split edge into prefix (shared) + suffix (divergent)
- **Reference counting:** Increment when sequence starts using prefix, decrement when sequence completes
- **Memory pressure:** Evict aggressively enough to avoid OOM, but not so aggressively that cache hit rate drops
- **Zero-length sequences:** Root node represents empty prefix (all sequences start here)

---

## 5. Metal-Specific Optimizations

### Threadgroup & SIMD Primitives

| Primitive | Use Case | Why | Confidence |
|-----------|----------|-----|------------|
| **simdgroup_matrix operations** | GEMM, GEMV, attention QK^T | 3× faster than scalar threadgroup code. Required to match CPU AMX performance. Apple GPUs prioritize SIMD-scoped ops over threadgroup-scoped. | **HIGH** |
| **simdgroup_shuffle** | Reductions (max, sum) | Equivalent to CUDA `__shfl_down_sync`. Faster than shared memory for intra-SIMD communication. | **HIGH** |
| **threadgroup_barrier(mem_flags::mem_threadgroup)** | Synchronization after shared writes | Prevents race conditions in multi-tile algorithms (FlashAttention, multi-row GEMV). | **HIGH** |
| **newBufferWithBytesNoCopy** | Zero-copy UMA | Wraps host memory as Metal buffer (no GPU copy). Requires page-aligned pointers (4KB). Cache by page-aligned base address, return `BufRef{buf, offset}` for sub-regions. | **HIGH** |

**Avoiding Common Pitfalls:**

1. **Shared memory atomics:** Metal's `atomic<float>` is emulated (slow). Use int32 atomics or redesign to use reductions.

2. **Encoder switching:** Avoid mixing blit and compute commands in hot path. Causes pipeline stalls (~150ms observed in Agave SDPA). Fuse all operations into compute shaders.

3. **Alignment requirements:** `newBufferWithBytesNoCopy` requires page alignment. Small activation buffers from GPA may not be aligned — use `newBufferWithLength` + copy for these, cache page-aligned weights with zero-copy.

4. **CPU fallback pattern:** When falling back to CPU (e.g., unsupported dtype), flush pending GPU work BEFORE CPU execution. Use `be.sync()` or `commandBuffer.waitUntilCompleted()`. Otherwise CPU reads stale data.

**Debugging Tools:**

- **Metal Debugger:** Built into Xcode. Use GPU frame capture to inspect buffers, shader execution, and timing.
- **Shader validation:** `MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_FAIL_MODE=allow` to catch errors without crashing.
- **Performance HUD:** Enable via Xcode → Product → Scheme → Run → Options → GPU Frame Capture → Metal System Trace.

---

## 6. CUDA-Specific Optimizations

### Warp-Level Primitives & Memory Hierarchy

| Primitive | Use Case | Why | Confidence |
|-----------|----------|-----|------------|
| **__shfl_down_sync(0xFFFFFFFF, val, offset)** | Warp reductions (sum, max, min) | 10-100× faster than shared memory. Must be `asm volatile` to prevent compiler sinking shuffles into conditional branches (causes deadlock). | **HIGH** |
| **Dynamic shared memory** | Staging input vectors, inter-warp reductions | 100× lower latency than global memory. Allocated per thread block via kernel launch parameter. | **HIGH** |
| **cuMemHostRegister** | Zero-copy UMA on Blackwell | Register page-aligned host memory for direct GPU access (no copy). Only for UMA platforms (check `CU_DEVICE_ATTRIBUTE_INTEGRATED`). | **MEDIUM** |
| **PTX math intrinsics** | Fast approximate math | Use `ex2.approx`, `rcp.approx`, `rsqrt.approx` instead of libcalls (unavailable on nvptx target). | **HIGH** |

**Hierarchical Reduction Pattern (2-tier):**

```cuda
// Warp reduction (intra-warp)
for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
}

// Store warp results to shared memory
__shared__ float warp_results[8]; // 8 warps per block (256 threads / 32)
if (lane_id == 0) {
    warp_results[warp_id] = val;
}
__syncthreads();

// Thread 0 reduces across warps
if (threadIdx.x == 0) {
    float block_sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        block_sum += warp_results[i];
    }
    // block_sum is final result
}
```

**Blackwell sm_121 Known Issues:**

| Issue | Symptom | Workaround | Permanent Fix | Confidence |
|-------|---------|-----------|---------------|------------|
| **blockReduce hangs** | Kernel deadlocks on `__syncthreads()` after shared memory reduction | Use warp-only reductions (no block-level shared memory in softmax) | Wait for CUDA toolkit update or file bug with NVIDIA | **HIGH** |
| **Dynamic shared memory corruption** | Random values in shared memory array | Avoid dynamic shared memory on sm_121 until driver/compiler fix. Use register arrays instead. | Driver/compiler update from NVIDIA | **MEDIUM** |

**When to Use CPU Fallback:**

Even with GPU acceleration, some ops are faster on CPU:
- **Small inputs:** Embedding lookup for single token (GPU dispatch overhead > compute time)
- **Unsupported dtypes:** NVFP4 with SafeTensors layout (complex unpacking logic, rare format)
- **Debug/verification:** Golden test comparison against reference CPU implementation

**Pattern:** `flushActivations()` before CPU op, `invalidateAct(output)` after CPU writes. Prevents stale data reads.

---

## 7. Production Serving Infrastructure

### HTTP Server & API

| Component | Technology | Why | Confidence |
|-----------|-----------|-----|------------|
| **HTTP server** | Zig `std.http.Server` | Native async I/O, zero external dependencies. Matches project's pure-Zig constraint. | **HIGH** |
| **API format** | OpenAI-compatible `/v1/completions`, `/v1/chat/completions` | De facto standard. Clients expect OpenAI API shape (vLLM, SGLang, LM Studio all use this). | **HIGH** |
| **Streaming** | Server-Sent Events (SSE) | Standard for token-by-token streaming. Use `Content-Type: text/event-stream`, `data: {"token": "..."}\n\n` format. | **HIGH** |
| **Rate limiting** | Token bucket algorithm | Prevent request queue overflow. Use `std.atomic` for bucket state (lock-free). | **HIGH** |
| **Request timeouts** | Per-request deadline | Cancel hung requests after N seconds. Propagate cancellation to model via atomic flag (`be.cancel()`). | **HIGH** |

**OpenAI API Compliance:**

Required fields:
- `POST /v1/completions`: `{model, prompt, max_tokens, temperature, stream}`
- `POST /v1/chat/completions`: `{model, messages, max_tokens, temperature, stream}`
- Response: `{id, object, created, model, choices: [{text, index, finish_reason}]}`

Optional extensions:
- `top_p`, `top_k`, `repetition_penalty`, `stop` sequences
- `logprobs`: Return log probabilities for top-K tokens
- `n`: Generate multiple completions per request (batch size)

**Rate Limiting:**

```zig
const token_bucket_capacity: u32 = 100; // max concurrent requests
const refill_rate: f32 = 10.0; // tokens/sec

var bucket_tokens = std.atomic.Atomic(u32).init(token_bucket_capacity);
var last_refill = std.atomic.Atomic(u64).init(0);

fn tryAcquire() bool {
    const now = std.time.milliTimestamp();
    const last = last_refill.load(.Monotonic);
    const elapsed = @as(f32, @floatFromInt(now - last)) / 1000.0;
    const refill = @min(@as(u32, @intFromFloat(elapsed * refill_rate)), token_bucket_capacity);

    _ = bucket_tokens.fetchAdd(refill, .Monotonic);
    last_refill.store(now, .Monotonic);

    while (true) {
        const tokens = bucket_tokens.load(.Monotonic);
        if (tokens == 0) return false;
        if (bucket_tokens.cmpxchgWeak(tokens, tokens - 1, .Monotonic, .Monotonic)) |_| continue;
        return true;
    }
}
```

**Request Cancellation:**

Each model must expose `cancel()` method that sets atomic flag. Hot path checks flag periodically:

```zig
pub fn forward(self: *Model, token_id: u32) !u32 {
    if (self.cancel_flag.load(.Monotonic)) return error.RequestCancelled;

    // ... layer processing ...

    // Check every N layers (avoid overhead on every op)
    if (layer_idx % 4 == 0) {
        if (self.cancel_flag.load(.Monotonic)) return error.RequestCancelled;
    }
}
```

**Monitoring & Observability:**

Expose `/metrics` endpoint with Prometheus-compatible format:
- `agave_requests_total{status}`: Total requests by status (200, 400, 500, 503 rate-limited)
- `agave_request_duration_seconds`: Histogram of request latencies
- `agave_tokens_generated_total`: Total tokens generated
- `agave_active_requests`: Current number of in-flight requests
- `agave_gpu_memory_bytes_used`: Current GPU VRAM usage
- `agave_kv_cache_hit_rate`: Prefix cache hit rate (RadixAttention only)

---

## Anti-Patterns to Avoid

### What NOT to Do

| Anti-Pattern | Why Bad | Instead | Confidence |
|--------------|---------|---------|------------|
| **FlashAttention-3 for current target GPUs** | Requires H100+, CUDA 12.3+, beta quality, limited head dim support | Use FlashAttention-2 (production-ready, cross-platform) | **HIGH** |
| **Three-pass softmax** | Wastes memory bandwidth (3 HBM round-trips vs. 1) | Use online softmax (single-pass with running max/sum) | **HIGH** |
| **Global atomics in attention kernel** | Serializes parallel work, Metal has emulated FP32 atomics (very slow) | Use hierarchical reductions (warp shuffle → shared memory) | **HIGH** |
| **Contiguous KV cache allocation** | Fragments GPU memory, limits batch size, prevents prefix sharing | Use PagedAttention (16-token blocks, block tables) | **HIGH** |
| **Manual prefix detection** | Requires user annotation, error-prone, misses runtime sharing opportunities | Use RadixAttention (automatic LCP detection and sharing) | **HIGH** |
| **Lock-based scheduler** | Mutex contention bottlenecks throughput, violates hot-path zero-lock rule | Use lock-free atomics + work-stealing queue | **HIGH** |
| **Blit encoder for KV append (Metal)** | Encoder switching causes 150ms stalls | Fuse into compute shader (no blit commands in hot path) | **HIGH** |
| **Block-level shared memory on CUDA sm_121** | Hangs due to Blackwell compiler bug | Use warp-only reductions until driver fix available | **MEDIUM** |
| **CPU SDPA in production serving** | Doesn't scale, GPU idle while CPU computes attention | Debug and fix GPU SDPA kernel (worth the investment) | **MEDIUM** |
| **Ignoring numerical stability** | Softmax overflow (exp(1000) = inf), underflow (exp(-1000) = 0) | Always use max subtraction or online softmax algorithm | **HIGH** |

---

## Installation & Dependencies

All recommendations use technologies already in the project (Zig, Metal, CUDA, Vulkan) or standard libraries (no external deps).

**No new dependencies required.** Implementations use:
- Zig `std.http.Server` (built-in)
- Zig `std.PriorityQueue`, `std.AutoHashMap` (built-in)
- Zig atomics (`@atomicLoad`, `@atomicStore`, `std.atomic.Atomic`) (built-in)
- Metal Shading Language (existing)
- CUDA PTX compilation via Zig `nvptx64-cuda` target (existing)

**Build flags:**
- `zig build -Denable-vulkan=true` (already supported)
- `zig build -Dcuda-sm=sm_121` (already supported)
- No new build options needed

---

## Sources & References

### FlashAttention & SDPA

- [FlashAttention & Paged Attention: GPU Sorcery for Blazing-Fast Transformers](https://medium.com/@afafel/flashattention-paged-attention-gpu-sorcery-for-blazing-fast-transformers-9307df8a3f3f)
- [Understanding Flash Attention: Writing Triton Kernel Code](https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/)
- [Reimplementing FlashAttention for performance and giggles](https://aminediro.com/posts/flash_attn/)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/html/2407.08608v2)
- [Tuning Flash Attention for Peak Performance in NVIDIA CUDA Tile](https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/)
- [Accelerate machine learning with Metal - WWDC24](https://developer.apple.com/videos/play/wwdc2024/10218/)
- [kernels-community/metal-flash-sdpa](https://huggingface.co/kernels-community/metal-flash-sdpa)
- [Metal FlashAttention 2.0: pushing forward on-device inference & training on Apple Silicon](https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c)

### Online Softmax & Numerical Stability

- [Learning CUDA by optimizing softmax: A worklog](https://maharshi.bearblog.dev/optimizing-softmax-cuda/)
- [Online Softmax Normalizer Optimization](https://www.emergentmind.com/papers/1805.02867)
- [GPU kernel optimization: Softmax — Part 1](https://medium.com/@hugo.rosenkranz/gpu-kernel-optimization-softmax-part-1-8ff80766cc95)
- [We reverse-engineered Flash Attention 4](https://modal.com/blog/reverse-engineer-flash-attention-4)
- [Implementing Softmax From Scratch: Avoiding the Numerical Stability Trap](https://www.marktechpost.com/2026/01/06/implementing-softmax-from-scratch-avoiding-the-numerical-stability-trap/)
- [Softmax on a Shoestring: The Art of Online Calculation](https://medium.com/@nanda.yugandhar/softmax-on-a-shoestring-the-art-of-online-calculation-fe07424190f6)

### Continuous Batching & vLLM

- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System - Aleksa Gordić](https://www.aleksagordic.com/blog/vllm)
- [LLM Inference Servers Compared: vLLM vs TGI vs SGLang vs Triton (2026)](https://blog.premai.io/llm-inference-servers-compared-vllm-vs-tgi-vs-sglang-vs-triton-2026/)
- [Static, dynamic and continuous batching](https://bentoml.com/llm/inference-optimization/static-dynamic-continuous-batching)
- [vLLM vs SGLang vs LMDeploy: Fastest LLM Inference Engine in 2026?](https://blog.premai.io/vllm-vs-sglang-vs-lmdeploy-fastest-llm-inference-engine-in-2026/)
- [Under the Hood of vLLM: Memory, Scheduling & Batching Strategies](https://www.javacodegeeks.com/2025/10/under-the-hood-of-vllm-memory-scheduling-batching-strategies.html)
- [vLLM Continuous Batching: High-Throughput Serving for Long Contexts 2025](https://www.johal.in/vllm-continuous-batching-high-throughput-serving-for-long-contexts-2025/)

### PagedAttention

- [Paged Attention - vLLM](https://docs.vllm.ai/en/latest/design/paged_attention/)
- [PagedAttention & vLLM for Efficient LLM Inference Woosuk Kwon](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-22-vLLM_woosuk_kwon-1f34697dbb1a1fb5b798daf6eff14b67.pdf)
- [The Architecture Behind vLLM: How PagedAttention Improves Memory Utilization](https://medium.com/@mandeep0405/the-architecture-behind-vllm-how-pagedattention-improves-memory-utilization-2f9b25272110)
- [Paged Attention from First Principles: A View Inside vLLM](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/)
- [How PagedAttention resolves memory waste of LLM systems](https://developers.redhat.com/articles/2025/07/24/how-pagedattention-resolves-memory-waste-llm-systems)

### RadixAttention & SGLang

- [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/)
- [SGLang Learning Series — Part 1: Shared Prefix, KV Cache, and RadixAttention](https://medium.com/@dharamendra1314.kumar/sglang-learning-series-part-1-shared-prefix-kv-cache-and-radixattention-d7a847d20b1f)
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/pdf/2312.07104)
- [Radix Attention: How a Data Structure Inspired a Revolution in Efficient Transformer Inference](https://medium.com/the-synaptic-stack/radix-attention-how-a-data-structure-inspired-a-revolution-in-efficient-transformer-inference-b7d5baa1219a)
- [When to Choose SGLang Over vLLM: Multi-Turn Conversations and KV Cache Reuse](https://www.runpod.io/blog/sglang-vs-vllm-kv-cache)
- [Efficient LLM Inference with SGLang Speaker: Ying Sheng (xAI, LMSYS, UCLA)](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-25-sglang-72edc5043338f59db34d47e5b96ac870.pdf)
- [SGLang Memory Management & Cache](https://muqi1029.github.io/posts/2025/05/mem_cache/)

### CUDA Optimizations

- [7 Step Optimization of Parallel Reduction with CUDA](https://medium.com/@rimikadhara/7-step-optimization-of-parallel-reduction-with-cuda-33a3b2feafd8)
- [GPU MODE Lecture 9: Reductions](https://christianjmills.com/posts/cuda-mode-notes/lecture-009/)
- [How to write a fast Softmax CUDA kernel?](https://github.com/facebookincubator/AITemplate/wiki/How-to-write-a-fast-Softmax-CUDA-kernel%3F)
- [How to Implement an Efficient Softmax CUDA kernel?](https://oneflow2020.medium.com/how-to-implement-an-efficient-softmax-cuda-kernel-oneflow-performance-optimization-sharing-405ad56e9031)
- [Optimizing Parallel Reduction in CUDA Mark Harris NVIDIA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [Warp Shuffle vs Shared Memory: Which Is Faster?](https://medium.com/a-gpu-crash-course-for-embedded-engineers/warp-shuffle-vs-shared-memory-which-is-faster-f8ed254a7c29)
- [Fast Reductions with Warp Shuffles](https://blog.melashri.net/micro/shfl-cuda/)
- [Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)

### Quantized Kernels

- [AI Model Quantization 2025: Master Compression Techniques](https://local-ai-zone.github.io/guides/what-is-ai-quantization-q4-k-m-q8-gguf-guide-2025.html)
- [LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel](https://arxiv.org/html/2509.01229v1)
- [GGUF Quantized Models Complete Guide 2025](https://apatero.com/blog/gguf-quantized-models-complete-guide-2025)
- [NVIDIA Blackwell: The Impact of NVFP4 For LLM Inference](https://www.edge-ai-vision.com/2025/10/nvidia-blackwell-the-impact-of-nvfp4-for-llm-inference/)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
