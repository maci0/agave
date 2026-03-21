# Domain Pitfalls

**Domain:** LLM inference engine production readiness
**Researched:** 2026-03-21

## Critical Pitfalls

Mistakes that cause rewrites, major performance regressions, or broken models.

### Pitfall 1: Encoder Switching Overhead in Metal (Compute ↔ Blit)
**What goes wrong:** Switching between compute and blit command encoders to perform KV cache append operations introduces 100-150ms/layer stalls due to pipeline serialization and working set limits. The GPU must drain the compute encoder, perform blit operations, then rebuild the compute encoder state — even though compute encoders can perform the same memory operations.

**Why it happens:** Developers reach for `MTLBlitCommandEncoder` because it's the "official" way to copy GPU buffers, not realizing that switching encoder types forces synchronization. Metal documentation emphasizes blit for copies without mentioning the performance cliff when mixed with compute passes.

**Consequences:**
- SDPA becomes slower than CPU fallback (11.6 tok/s vs potential 20+ tok/s)
- Read-after-write hazards require explicit synchronization
- Working set limits are exceeded when multiple encoder types share resources

**Prevention:**
- **Always** use compute kernels for memory operations in the hot path — never switch to blit encoders during token generation
- Implement KV append as a simple compute kernel with threadgroup dispatch
- Reserve blit encoders for one-time setup (weight uploads, cache initialization)

**Detection:**
- Profile with Instruments: look for encoder commit/begin overhead > 10ms
- GPU timeline shows gaps between compute passes
- Per-op profiling shows blit operations taking > 1ms

**Phase impact:** GPU Kernel Parity phase must establish this rule early. Any SDPA implementation using blit encoders will need a complete rewrite later.

**Sources:**
- [Metal GPU Programming Guide](https://awesomeagents.ai/guides/metal-gpu-programming-guide/)
- [Apple Metal Blit Passes Documentation](https://developer.apple.com/documentation/metal/blit-passes)

---

### Pitfall 2: CUDA Blackwell (sm_121) Compiler Bugs in Shared Memory Reductions
**What goes wrong:** Block-level reduction primitives (`__syncthreads()` + shared memory accumulation) hang or produce incorrect results on Blackwell GPUs (sm_121) when compiled with CUDA 13.0-13.2 at `-O3`. Warp shuffles (`__shfl_sync`) work correctly, but cross-warp reductions via shared memory deadlock.

**Why it happens:** NVCC compiler optimization bug where the optimizer incorrectly reorders or eliminates synchronization barriers in shared memory reduction patterns. The issue appears specific to sm_121 codegen — same code works on sm_120 and sm_89.

**Consequences:**
- Kernels hang indefinitely (no error, no timeout — just stops)
- Silent data corruption if the kernel occasionally completes
- SDPA parallel softmax becomes impossible (forces serial thread-0 workaround)
- 10-100× performance loss using serial algorithms instead of parallel reductions

**Prevention:**
- **Avoid shared memory reductions on sm_121 until CUDA 13.3+** — use warp-only reductions where possible
- Implement serial workarounds for now (thread 0 computes, broadcasts result via shared[0])
- Pin CUDA version and track NVIDIA bug reports for sm_121 fixes
- Test all reduction kernels with synthetic stress tests (1000+ iterations, verify output)

**Detection:**
- Kernel launch never returns from `cuCtxSynchronize()`
- `nvidia-smi` shows 100% GPU utilization but no progress
- Works on sm_89/sm_120, hangs on sm_121
- Disabling `-O3` or compiling for sm_120 makes it work

**Phase impact:** GPU Kernel Parity phase MUST detect this early. Don't assume Blackwell works like prior architectures. Budget 2-3 weeks for workarounds or alternative algorithms.

**Sources:**
- [llama.cpp Issue #18331: CUDA MUL_MAT crash on Blackwell](https://github.com/ggml-org/llama.cpp/issues/18331)
- [CUTLASS Issue #3096: SM120 NVFP4 MoE Garbage Output](https://github.com/NVIDIA/cutlass/issues/3096)
- [vLLM Issue #36821: No sm_121 support on aarch64](https://github.com/vllm-project/vllm/issues/36821)

---

### Pitfall 3: MoE Router Numerical Overflow with FP8/Low-Precision Quantization
**What goes wrong:** Expert routing scores explode to extreme values (±1e26 or NaN) when router weights are quantized to FP8 without per-block scaling. The softmax or top-K selection produces nonsensical expert assignments, causing the model to generate garbage output despite the rest of the architecture working correctly.

**Why it happens:** Router weights often have large dynamic range (outliers at ±50, typical values near ±0.1). FP8 E4M3 has max value of 448 — any value beyond that overflows to inf. When you quantize without per-block scales (using a single tensor-wide scale), outliers dominate and small values underflow to zero. During routing, `logit * scale` can overflow even if the original logit was in-range.

**Consequences:**
- Model runs end-to-end but produces nonsensical text
- Hard to debug — the error is silent (no NaN, just bad numbers)
- All experts get nearly identical scores → routing becomes random
- Extreme values (-2.3e26) indicate the overflow happened multiple layers back

**Prevention:**
- **Always use per-block (group-wise) quantization for router weights** — never use tensor-wide scaling
- Use BF16 for router weights if model size allows (routers are <1% of total params)
- Implement overflow detection in routing: `assert(score < 1e10 && score > -1e10)`
- Verify router scores match reference implementation (Python/PyTorch) within 1% for first 10 tokens

**Detection:**
- Router scores contain values > 1e10 or < -1e10
- All experts selected with equal probability (top-K becomes random)
- Output quality collapses while perplexity looks normal (model is "confident nonsense")
- Logging `router_logits.max()` and `.min()` reveals the overflow

**Phase impact:** Model Verification phase must catch this. Don't assume "it compiles and runs" means "it's correct." Budget golden tests for every MoE model with reference outputs.

**Sources:**
- [LMSYS: Unified FP8 for Stable MoE RL](https://lmsys.org/blog/2025-11-25-fp8-rl/)
- [arXiv: FP8-Flow-MoE — Casting-Free FP8 Recipe](https://arxiv.org/abs/2511.02302)
- [DeepSeek-V3 FP8 Revolution in Large-Scale AI](https://medium.com/@prashantsahdev/deepseek-v3-blog-5-low-precision-training-the-fp8-revolution-in-large-scale-ai-29fc4b14761e)

---

### Pitfall 4: Pre-Dequantization vs In-Kernel Dequantization Performance Cliff
**What goes wrong:** Dequantizing an entire weight matrix to FP32 before GEMV consumes 4× the memory bandwidth and eliminates the entire benefit of quantization. A Q4_0 matrix that should fit in 32MB becomes 128MB of FP32, saturating memory bandwidth and making the GPU kernel slower than the CPU.

**Why it happens:** Developers assume "GPU needs FP32 for compute" and add a dequantization pass before GEMV. They don't realize modern GPUs have enough registers and compute throughput to dequantize per-element inside the kernel while loading from memory — the bottleneck is memory bandwidth, not compute.

**Consequences:**
- Q4_K GEMV is slower than FP32 GEMV (should be 4× faster)
- VRAM usage explodes (can't fit large models)
- Quantization becomes pointless — loses both speed and accuracy

**Prevention:**
- **Dequantization MUST happen inside the GEMV kernel** — load quantized data, dequantize in registers, accumulate
- Pass quantized tensors through the backend interface as `TensorData{ .data = ptr, .dtype = .q4_k }`
- Kernel reads scales/zero-points per block, dequantizes 32-128 elements at a time
- Never allocate FP32 scratch buffers for weights in the hot path

**Detection:**
- Memory bandwidth utilization is 100% but GEMV is slow (should be compute-bound for large matrices)
- VRAM usage is 4× expected size
- `htop` shows CPU allocating/freeing large buffers during inference
- Profiling shows `dequantize()` function consuming 50%+ of time

**Phase impact:** GPU Kernel Parity phase must enforce this from the start. Refactoring after implementing full dequantization is painful (affects every kernel).

**Sources:**
- [NVIDIA TensorRT: Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- [PyTorch Quantization-Aware Training](https://pytorch.org/blog/quantization-aware-training/)
- [BentoML: LLM Quantization](https://bentoml.com/llm/getting-started/llm-quantization)

---

### Pitfall 5: RadixAttention Block Eviction Order Causes Prefix Thrashing
**What goes wrong:** Evicting radix tree blocks in simple LRU order (oldest-accessed first) causes prefix thrashing where shared prefixes (system prompts, common chat templates) are evicted and immediately re-inserted, wasting memory bandwidth and KV compute.

**Why it happens:** Not all KV cache blocks are equally valuable. The last block of a request hashes more tokens and is less likely to be reused by other requests, but simple LRU treats it the same as the first block (which might be a shared system prompt). Developers implement LRU because it's easy, not realizing that **frequency × compute_cost** is the correct eviction metric.

**Consequences:**
- Prefix cache hit rate drops from 80% to 40% under load
- Long system prompts are repeatedly evicted and re-computed (50-100ms each)
- Short requests starve long requests (greedy scheduling + LRU = unfair)
- Effective cache size is 30% of allocated size

**Prevention:**
- **Evict based on minimum (frequency × compute_cost)**, not LRU
- Track access count per block (not just last-accessed timestamp)
- Prioritize keeping shared prefixes (ref_count > 1) over private continuations
- Evict the **last block** of a sequence first (reverse order during recursive eviction)
- Implement this during Production Serving phase — don't wait for complaints

**Detection:**
- Cache hit rate < 60% despite allocating 8GB for KV cache
- Logging shows same prefix (e.g., system prompt hash) inserted 100+ times
- First token latency has high variance (100ms vs 800ms for identical prompts)
- RadixTree depth stays at 1-2 (shared prefixes are being evicted)

**Phase impact:** Production Serving phase must implement sophisticated eviction from day one. Retrofitting is painful because the API contract (LRU) leaks into the scheduler.

**Sources:**
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [LMSYS: Fast and Expressive LLM Inference with RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)
- [vLLM RFC: Frequency and Cost Aware Eviction Policy](https://github.com/vllm-project/vllm/issues/23641)

---

## Moderate Pitfalls

Cause performance regressions or development delays, but recoverable.

### Pitfall 6: UMA Zero-Copy Memory Coherency Assumptions
**What goes wrong:** Developers assume UMA (Unified Memory Architecture) provides automatic coherency between CPU and GPU caches. They write to a buffer on the CPU, immediately dispatch a GPU kernel reading it, and get stale data because the CPU writes are still in L1/L2 cache, not visible to the GPU.

**Why it happens:** Marketing materials say "unified memory" → developers assume "cache coherent." But only **hardware-coherent systems** (NVIDIA Grace Hopper, AMD Ryzen AI Max+) provide true coherency. Standard UMA (Apple Silicon, NVIDIA Tegra, entry-level Blackwell) requires explicit synchronization.

**Prevention:**
- **Assume non-coherent unless proven otherwise** — insert sync points
- After CPU writes: call backend `sync()` or `flush()` before GPU read
- After GPU writes: call backend `sync()` before CPU read
- Use `cuMemHostRegister` (CUDA) or `newBufferWithBytesNoCopy` (Metal) to register host pages, but still sync
- Test on **both** UMA and discrete GPUs — bugs often appear only on one

**Detection:**
- Output is correct on discrete GPU, wrong on UMA (or vice versa)
- Intermittent corruption (race condition — depends on cache timing)
- Data becomes correct after adding `sleep(1)` (flush happened coincidentally)

**Phase impact:** GPU Kernel Parity and Model Verification — affects every model on every UMA platform.

**Sources:**
- [NVIDIA CUDA Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)
- [NVIDIA Developer Blog: Unified Memory for CUDA Beginners](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
- [Zero-Copy vs Unified Memory CUDA](https://forums.developer.nvidia.com/t/zero-copy-memory-vs-unified-memory-cuda-processing/51790)

---

### Pitfall 7: GPU Kernel Numerical Tolerance Testing Without Dual Baselines
**What goes wrong:** Developers compare GPU kernel output to a reference (e.g., PyTorch) with fixed tolerance (±0.01), mark the kernel as "correct," then discover it produces different results than the CPU kernel or diverges after 100 tokens due to accumulated error.

**Why it happens:** GPU optimizations (fused operations, different accumulation order, lower precision intermediates) cause small numerical differences from the reference. These differences are usually acceptable, but without comparing **both** GPU and CPU to a high-precision oracle, you can't distinguish "acceptable difference" from "semantic bug."

**Prevention:**
- **Dual-delta testing**: Compare both GPU and CPU kernels to an FP64 reference, verify GPU error ≤ 2× CPU error
- Test with both random inputs AND adversarial inputs (large dynamic range, near-overflow values)
- For iterative operations (attention, RNN, SSM), test 100+ steps to catch divergence
- Golden tests with fixed seeds for reproducibility (commit outputs to repo)

**Detection:**
- GPU kernel passes unit tests but model output diverges after N tokens
- Different results on different GPUs (NVIDIA vs AMD vs Apple)
- Error grows with sequence length (1e-5 at token 10, 1e-2 at token 100)

**Phase impact:** GPU Kernel Parity phase must implement this testing discipline from the start. Catching divergence bugs after shipping is embarrassing.

**Sources:**
- [arXiv: Equivalence Checking of ML GPU Kernels](https://arxiv.org/pdf/2511.12638)
- [arXiv: Dual-Delta Testing for Mixed-Precision Computing](https://arxiv.org/html/2602.10605)
- [NVIDIA SOL-ExecBench: Real-World DL Kernel Problems](https://github.com/NVIDIA/SOL-ExecBench)

---

### Pitfall 8: Cross-Compilation with Musl Breaks dlopen of glibc Libraries
**What goes wrong:** Building a static musl binary to "guarantee portability," then calling `dlopen("libcuda.so")` at runtime causes segfaults or mysterious failures because libcuda.so links against glibc, and mixing musl + glibc in the same process is fundamentally broken.

**Why it happens:** Musl and glibc have incompatible ABIs for thread-local storage (TLS), different symbol resolution rules, and musl's `dlclose()` is a no-op. When you dlopen a glibc library from a musl binary, both libc implementations try to manage the same process, and chaos ensues.

**Prevention:**
- **Use `-gnu` target for cross-compilation** (e.g., `aarch64-linux-gnu`) when you need to dlopen GPU drivers
- Do NOT use musl for CUDA/ROCm backends — drivers are always glibc
- If you must support musl, statically link GPU backends at compile time (no dlopen)
- Document this constraint in build system — it's not obvious

**Detection:**
- `dlopen()` succeeds but driver functions segfault
- Works on glibc systems, fails on Alpine/musl systems
- `ldd` shows "not a dynamic executable" but dlopen still called

**Phase impact:** GPU Kernel Parity phase will discover this when testing on different Linux distros. Budget 1 week for build system fixes.

**Sources:**
- [musl libc: Functional Differences from glibc](https://wiki.musl-libc.org/functional-differences-from-glibc.html)
- [musl: dlopen'ing glibc linked libraries](https://musl.openwall.narkive.com/ODEqT9jH/dlopen-ing-glibc-linked-libraries)
- [TuxCare: musl vs glibc — Pros, Cons, and Key Differences](https://tuxcare.com/blog/musl-vs-glibc/)

---

### Pitfall 9: Thread Pool Futex Wake Latency Dominates Small GEMV Operations
**What goes wrong:** Parallel CPU GEMV splits 32 rows across 8 threads using a futex-based work queue. The futex wake syscall takes 5-10μs, and the actual GEMV work is 2μs/row → 50% of time is spent waking threads, not computing.

**Why it happens:** Futex-based thread pools are designed for coarse-grained tasks (100+ μs), not fine-grained parallelism. Waking threads involves a syscall, scheduler overhead, and cache misses. For small matrices, the overhead exceeds the benefit.

**Prevention:**
- **Set minimum work size for parallel dispatch** — e.g., `parallel_min_rows = 32` (skip parallel if rows < 32)
- Use spinlock-based work queues for latency-critical paths (burns CPU but no syscalls)
- Batch small operations (GEMV 10 small matrices instead of parallelizing one)
- Profile with `perf` to measure futex overhead (`perf record -e syscalls:sys_enter_futex`)

**Detection:**
- `perf` shows 30-50% time in `futex_wake` or `futex_wait`
- Disabling parallelism (single-threaded) is FASTER for small inputs
- Thread pool is faster for large matrices but slower for decode (batch=1)

**Phase impact:** GPU Kernel Parity phase may add parallelism without measuring overhead. Benchmark before committing.

**Sources:**
- [TensorFlow Issue #551: Lock Contention in Thread Pool](https://github.com/tensorflow/tensorflow/issues/551)
- [Eli Bendersky: Basics of Futexes](https://eli.thegreenplace.net/2018/basics-of-futexes/)
- [Examining Futexes and Thundering Herd](https://medium.com/@arkarthick/examining-futexes-and-how-it-tackles-thundering-herd-71d1e30e2887)

---

### Pitfall 10: Metal newBufferWithBytesNoCopy Requires Page-Aligned Pointers
**What goes wrong:** Passing a non-page-aligned pointer to `newBufferWithBytesNoCopy` (e.g., a per-head slice like `qk_norm.ptr + h * head_dim`) causes Metal to silently create a **copy** of the data instead of zero-copy mapping, negating the entire point of the API and causing mysterious slowdowns.

**Why it happens:** Metal's unified memory mapping requires page-aligned base addresses (4KB or 16KB depending on platform). Sub-page offsets can't be mapped directly to GPU page tables. Developers assume "it worked" because no error is thrown — Metal just falls back to copying.

**Prevention:**
- **Always wrap the enclosing page-aligned region** — map the parent buffer, return offset
- Implement `getBufRef(ptr) -> {buf, offset}` that caches page-aligned `MTLBuffer` and computes byte offset
- Use the offset when dispatching kernels: `setBytes(&offset, sizeof(offset), 1)`
- Never create per-head or per-token buffers — reuse the parent allocation

**Detection:**
- `Activity Monitor` shows memory usage doubling (Metal copied the buffer)
- VRAM usage exceeds expected size
- Performance is identical to `newBufferWithBytes` (should be 2-3× faster with NoCopy)

**Phase impact:** GPU Kernel Parity phase will hit this when implementing per-head norms or multi-head attention. Budget 2-3 days for the caching abstraction.

**Sources:**
- [Apple Developer: MTLBlitCommandEncoder](https://developer.apple.com/documentation/metal/mtlblitcommandencoder)
- [Metal by Example: Mipmapping and Blit Encoder](https://metalbyexample.com/mipmapping/)

---

## Minor Pitfalls

Annoying but easy to fix once discovered.

### Pitfall 11: Attention Mask Dimension Mismatches with Prompt Tuning
**What goes wrong:** Model adds virtual prompt tokens (e.g., 3 tokens via PEFT), so `token_embeddings` has shape `[56]`, but `attention_mask` still has shape `[53]`, causing SDPA to crash or produce garbage.

**Why it happens:** Prompt tuning modifies embeddings but doesn't update the mask. Developers test with standard inputs (no PEFT) and don't notice.

**Prevention:**
- Validate `token_embeddings.len == attention_mask.len` after embedding lookup
- Unit tests with PEFT adapters enabled

**Detection:** SDPA crashes with dimension error, or outputs all zeros.

**Sources:**
- [HuggingFace Issue #2995: Attention Mask Mismatch with Prompt Tuning](https://github.com/huggingface/sentence-transformers/issues/2995)

---

### Pitfall 12: FP16 Mixed Precision Overflow at 64k
**What goes wrong:** Model activations exceed 64k (max FP16 value), causing inf propagation and NaN loss.

**Why it happens:** `1e4 * 1e4 = 1e8` (overflow in FP16). Matrix multiplications with large activations explode.

**Prevention:**
- Use BF16 instead of FP16 (max value 3.4e38)
- Insert overflow checks with `DebugUnderflowOverflow` hooks (HuggingFace)
- Gradient clipping during training (but this is inference-only, so validate activations)

**Detection:** NaN in logits, inf in attention scores.

**Sources:**
- [HuggingFace: Debugging Transformers](https://huggingface.co/transformers/v4.8.2/debugging.html)

---

### Pitfall 13: GGUF Norm Weights Pre-Baked with +1.0 (Gemma3)
**What goes wrong:** QK norm implementation adds +1.0 to norm weights (standard RMSNorm), but GGUF already baked +1.0 into the weights → double-adds → wrong output.

**Why it happens:** Different model formats have different conventions (GGUF vs SafeTensors).

**Prevention:**
- Check reference implementation for each model format
- Golden tests catch this immediately

**Detection:** Output quality collapses, norm values are 2× expected.

---

### Pitfall 14: Continuous Batching GPU Memory Fragmentation from `--gpu-memory-utilization=0.95`
**What goes wrong:** Setting `--gpu-memory-utilization=0.95` leaves only 5% headroom, causing OOM during large prefill operations (long prompts) even though steady-state decode fits.

**Why it happens:** Engineers maximize KV cache capacity without accounting for activation spikes during prefill.

**Prevention:**
- Safe ceiling: `0.93` for production
- Reserve 7% headroom for prefill activations

**Detection:** OOM errors on long prompts (>2K tokens), not on short prompts.

**Sources:**
- [BentoML: Static, Dynamic, and Continuous Batching](https://bentoml.com/llm/inference-optimization/static-dynamic-continuous-batching)

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|----------------|------------|
| **GPU Kernel Parity** | Encoder switching (Metal SDPA) | Establish "compute-only in hot path" rule day 1 |
| **GPU Kernel Parity** | Blackwell sm_121 reduction hangs | Serial workarounds, pin CUDA version, track bug reports |
| **GPU Kernel Parity** | Pre-dequantization pattern | Enforce "dequant in-kernel" in code review checklist |
| **Model Verification** | MoE router overflow | Per-block quantization + overflow assertions |
| **Model Verification** | Numerical divergence after 100 tokens | Dual-delta testing with FP64 oracle |
| **Model Verification** | UMA coherency bugs | Sync after every CPU write before GPU read |
| **Production Serving** | RadixAttention LRU thrashing | Frequency × cost eviction from day 1 |
| **Production Serving** | Continuous batching OOM | `--gpu-memory-utilization=0.93` max |
| **Cross-Platform** | Musl + glibc dlopen crash | Use `-gnu` targets for GPU backends |
| **Cross-Platform** | Metal page alignment | `getBufRef()` abstraction for sub-page slices |

---

## Pre-Deployment Checklist

Before considering any phase "complete," verify:

- [ ] **Metal SDPA**: Zero blit encoder usage in hot path (profile confirms)
- [ ] **CUDA sm_121**: Shared memory reductions tested with 1000-iteration stress test
- [ ] **MoE models**: Router scores logged for first 10 tokens, all values < 1e10
- [ ] **Quantization**: No FP32 scratch buffers allocated during GEMV (profile confirms)
- [ ] **RadixAttention**: Eviction policy is frequency × cost, not LRU
- [ ] **UMA platforms**: Sync called after every CPU→GPU data transfer
- [ ] **Numerical correctness**: Dual-delta tests pass (GPU vs CPU vs FP64 oracle)
- [ ] **Cross-compilation**: Linux ARM64 build uses `-gnu` target (not musl)
- [ ] **Thread pool**: Parallel dispatch skips small inputs (rows < 32)
- [ ] **Memory alignment**: Metal buffers are page-aligned or use `getBufRef()`
- [ ] **Continuous batching**: `--gpu-memory-utilization ≤ 0.93`

---

## What Might I Have Missed?

**Areas needing deeper investigation:**

1. **Vulkan SPIR-V compiler bugs** — similar to CUDA Blackwell, may have architecture-specific codegen issues (AMD RDNA3 vs NVIDIA, Intel Arc)
2. **ROCm HIP kernel pitfalls** — no research coverage; assume similar issues to CUDA but with different trigger conditions
3. **PagedAttention block table corruption** — when concurrent requests evict/allocate blocks, race conditions in ref counting
4. **Hybrid SSM+Attention models** — state updates preclude rollback, complicates prefix caching (mentioned in sources but not detailed)
5. **Speculative decoding with RadixAttention** — cache invalidation on rejected tokens
6. **KV cache quantization error accumulation** — does Q8_0 KV cache cause divergence after 1000+ tokens?
7. **Multi-GPU tensor parallelism** — this project is single-GPU, but future expansion will hit new coherency/synchronization pitfalls

**Recommended pre-emptive research for Phase 2:**
- Investigate ROCm-specific kernel bugs (AMD compiler, HIP runtime)
- Vulkan driver differences (Mesa vs proprietary NVIDIA vs AMDGPU-PRO)
- PagedAttention block allocator concurrency testing (simulate 100 concurrent requests)

---

## Sources

### GPU Kernel Optimization
- [KernelFoundry: Hardware-Aware Evolutionary GPU Kernel Optimization](https://arxiv.org/html/2603.12440)
- [MoldStud: Common CUDA Programming Pitfalls](https://moldstud.com/articles/p-top-10-cuda-mistakes-that-hurt-gpu-performance)
- [Medium: Writing Fast ML Kernels on Apple Silicon](https://medium.com/@srivarshan02/writing-fast-ml-kernels-on-apple-silicon-123152624078)
- [Advanced GPU Optimization: Metal & Vulkan Compute](https://dev.to/javadinteger/advanced-gpu-optimization-metal-vulkan-compute-from-zero-to-hero-4cfg)

### Continuous Batching and Production Serving
- [BentoML: Static, Dynamic, and Continuous Batching](https://bentoml.com/llm/inference-optimization/static-dynamic-continuous-batching)
- [Anyscale: Continuous Batching — 23x LLM Inference Throughput](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [HuggingFace: Continuous Batching from First Principles](https://huggingface.co/blog/continuous_batching)
- [Hakia: LLM Inference Optimization Techniques 2026](https://www.hakia.com/tech-insights/llm-inference-optimization/)

### SDPA and Attention Kernels
- [PyTorch Issue #127523: SDPA Memory Efficient Kernel — Singleton Dimensions](https://github.com/pytorch/pytorch/issues/127523)
- [PyTorch Issue #124877: SDPA Returns NaNs — Different Q/K Lengths](https://github.com/pytorch/pytorch/issues/124877)
- [PyTorch Issue #109517: SDPA NaNs for Certain Mask Patterns](https://github.com/pytorch/pytorch/issues/109517)

### MoE and Quantization
- [LMSYS: Unified FP8 — Moving Beyond Mixed Precision for Stable MoE RL](https://lmsys.org/blog/2025-11-25-fp8-rl/)
- [arXiv: FP8-Flow-MoE — Casting-Free FP8 Recipe](https://arxiv.org/abs/2511.02302)
- [Medium: DeepSeek-V3 — FP8 Revolution in Large-Scale AI](https://medium.com/@prashantsahdev/deepseek-v3-blog-5-low-precision-training-the-fp8-revolution-in-large-scale-ai-29fc4b14761e)

### RadixAttention and Prefix Caching
- [LMSYS: Fast LLM Inference with RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/)
- [vLLM: Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [vLLM Issue #23641: Frequency and Cost Aware Eviction Policy](https://github.com/vllm-project/vllm/issues/23641)
- [Medium: SGLang — Shared Prefix, KV Cache, and RadixAttention](https://medium.com/@dharamendra1314.kumar/sglang-learning-series-part-1-shared-prefix-kv-cache-and-radixattention-d7a847d20b1f)

### Numerical Correctness and Testing
- [arXiv: Equivalence Checking of ML GPU Kernels](https://arxiv.org/pdf/2511.12638)
- [arXiv: Dual-Delta Testing for Mixed-Precision Computing](https://arxiv.org/html/2602.10605)
- [NVIDIA SOL-ExecBench: Real-World DL Kernel Problems](https://github.com/NVIDIA/SOL-ExecBench)

### CUDA Blackwell Issues
- [llama.cpp Issue #18331: Blackwell Compiler Bug — sm_121](https://github.com/ggml-org/llama.cpp/issues/18331)
- [vLLM Issue #36821: No sm_121 Support on aarch64](https://github.com/vllm-project/vllm/issues/36821)
- [CUTLASS Issue #3096: SM120 NVFP4 MoE Garbage Output](https://github.com/NVIDIA/cutlass/issues/3096)
- [NVIDIA Developer Forums: DGX Spark SM121 Software Support Lacking](https://forums.developer.nvidia.com/t/dgx-spark-sm121-software-support-is-severely-lacking-official-roadmap-needed/357663)

### Unified Memory and Zero-Copy
- [NVIDIA: CUDA Unified Memory Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)
- [NVIDIA Developer Blog: Unified Memory for CUDA Beginners](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
- [NVIDIA Forums: Zero-Copy vs Unified Memory](https://forums.developer.nvidia.com/t/zero-copy-memory-vs-unified-memory-cuda-processing/51790)

### Transformer Debugging
- [HuggingFace: Debugging Transformers](https://huggingface.co/transformers/v4.8.2/debugging.html)
- [HuggingFace Issue #2995: Attention Mask Mismatch](https://github.com/huggingface/sentence-transformers/issues/2995)
- [HuggingFace Issue #31943: Different Output with Attention Mask](https://github.com/huggingface/transformers/issues/31943)

### Cross-Compilation
- [musl libc: Functional Differences from glibc](https://wiki.musl-libc.org/functional-differences-from-glibc.html)
- [musl: dlopen'ing glibc Linked Libraries](https://musl.openwall.narkive.com/ODEqT9jH/dlopen-ing-glibc-linked-libraries)
- [TuxCare: musl vs glibc](https://tuxcare.com/blog/musl-vs-glibc/)

### Metal and Memory
- [Apple Developer: MTLBlitCommandEncoder](https://developer.apple.com/documentation/metal/mtlblitcommandencoder)
- [Metal by Example: Mipmapping and Blit Encoder](https://metalbyexample.com/mipmapping/)
- [Arch Linux: mmap(2) Manual](https://man.archlinux.org/man/mmap.2.en)

### Thread Pools and Synchronization
- [TensorFlow Issue #551: Lock Contention in Thread Pool](https://github.com/tensorflow/tensorflow/issues/551)
- [Eli Bendersky: Basics of Futexes](https://eli.thegreenplace.net/2018/basics-of-futexes/)
- [Medium: Examining Futexes and Thundering Herd](https://medium.com/@arkarthick/examining-futexes-and-how-it-tackles-thundering-herd-71d1e30e2887)

### Quantization
- [NVIDIA TensorRT: Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- [PyTorch: Quantization-Aware Training](https://pytorch.org/blog/quantization-aware-training/)
- [BentoML: LLM Quantization](https://bentoml.com/llm/getting-started/llm-quantization)
