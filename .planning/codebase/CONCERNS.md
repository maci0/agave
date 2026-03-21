# Codebase Concerns

**Analysis Date:** 2025-02-21

## Tech Debt

### Metal SDPA GPU Fallback (High Impact)
- **Issue:** Metal SDPA uses CPU-based attention instead of GPU kernels for all inference
- **Files:** `src/backend/metal.zig`, `src/ops/attention.zig`
- **Impact:** Eliminates GPU speedup for attention operations (~150ms/layer observed on Gemma3 27B). 3.6× slowdown vs correct GPU SDPA
- **Root cause:**
  - Earlier GPU SDPA kernel + blit-based KV append caused ~150ms/layer stalls due to encoder switching (compute→blit→compute)
  - GPU-based compute KV append kernel exists (`kv_append` in `src/backend/kernels/metal/elementwise.metal`) but produces incorrect output
- **Fix approach:**
  - Debug GPU SDPA kernel output correctness when used with compute-based KV append
  - Profile encoder overhead and consider combining SDPA + KV append into single kernel dispatch
  - Alternatively, implement fused GPU SDPA + KV append as single kernel to avoid encoder switching
  - Benchmark against Vulkan SDPA implementation which uses fused single-dispatch kernels

### CUDA Serial Softmax Workaround (Medium-High Impact)
- **Issue:** CUDA SDPA softmax executed serially by thread 0, not parallel across warp
- **Files:** `src/backend/kernels/cuda/sdpa.zig` (lines 5-7), `src/backend/kernels/cuda/common.zig` (lines 177-179)
- **Impact:** Softmax becomes sequential bottleneck in SDPA. For long sequences (2048+), serial softmax may limit throughput vs parallel variants
- **Root cause:** `blockReduceMax` and `blockReduceAdd` hang on Blackwell (sm_121) due to compiler codegen issue with shared memory
- **Fix approach:**
  - Investigate Blackwell-specific compiler codegen behavior with shared memory patterns
  - Test workarounds: different warp reduction patterns, `__syncwarp()` placement, inline vs separate function
  - If compiler bug confirmed, file NVIDIA bug report; escalate to use different parallel softmax algorithm
  - Implement alternative: Welford online reduction or tree-based reduction without block-level sync
- **Workaround duration:** Temporary pending investigation
- **Current status:** Functional but suboptimal; affects model performance on NVIDIA Blackwell GPUs

### Nemotron Nano Output Quality Issue (Medium-High Impact)
- **Issue:** Model runs end-to-end but generates nonsensical text despite all layers executing
- **Files:** `src/models/nemotron_nano.zig`
- **Impact:** Model unusable despite complete implementation. Safety/output validation code seems correct
- **Symptoms:**
  - Model initializes successfully and loads all weights
  - All 52 layers (mixture of SSM, MoE, attention) execute without errors
  - MoE router scores show extreme values (e.g., -2.3e26) suggesting numerical instability
  - Generated tokens are non-coherent (`<unk>` tokens appear despite proper token generation)
- **Suspected causes:**
  - Numerical instability in routing gating (sigmoid or linear layer)
  - Incorrect parameter initialization or scaling in MoE expert computation
  - Misaligned tensor indexing between SafeTensors layout and model assumptions
  - Interaction between BF16 fallback weights (norms, conv1d) and NVFP4-quantized main weights
  - Incorrect SSM state initialization or decay computation
- **Fix approach:**
  - Compare intermediate activations (post-embedding, post-layer outputs) against reference implementation
  - Verify NVFP4 dequantization matches SafeTensors layout exactly (split nibbles, separate FP8 scales)
  - Check MoE gating computation: scale factors, routing probability normalization, expert selection ordering
  - Validate Mamba-2 state scaling and causal conv1d initialization against llama.cpp
  - Add per-layer diagnostic output (layer input norm, activation ranges) to identify which layer introduces corruption
  - Cross-reference with working Nemotron-H implementation which uses GGUF format (different quantization)
- **Risk:** Blocks validation and deployment of Nemotron Nano until resolved

### Vulkan CPU Fallbacks (Medium Impact)
- **Issue:** Vulkan backend falls back to CPU for several critical operations
- **Files:** `src/backend/vulkan.zig`
- **Missing GPU implementations:**
  - `embLookup` — CPU-only, no GPU kernel
  - `nvfp4_st` (SafeTensors NVFP4) — CPU-only, no GPU kernel
  - Paged SDPA — GPU implementation incomplete or disabled
  - `conv1d` (for SSM layers) — CPU-only
- **Impact:** Models with embedding lookups or NVFP4 weights will sync GPU→CPU for these ops, destroying pipeline overlap
- **Fix approach:**
  - Implement GPU kernels for embLookup (standard GEMV with vocabulary dimension) and conv1d
  - Complete NVFP4 SafeTensors GPU kernel using same MXFP4 dequantization as Metal/CUDA
  - Test paged SDPA implementation against flat SDPA for correctness
- **Priority:** Medium — affects Gemma3 (SafeTensors NVFP4) and Nemotron models on Vulkan

### Incomplete CUDA GEMV Quantization Formats (Low-Medium Impact)
- **Issue:** CUDA backend missing GPU GEMV kernels for several quantization formats
- **Files:** `src/backend/cuda.zig`, `src/backend/kernels/cuda/`
- **Missing formats:**
  - Q4_K, Q5_K, Q6_K — large K-block quantization
  - FP8 (E4M3, E5M2) — GPU kernel not implemented
  - NVFP4, MXFP4 — microscaled quantization formats
- **Impact:** Models using these formats fall back to CPU GEMV, 7-10× slower than GPU
- **Fix approach:**
  - Port Q4_K/Q5_K/Q6_K GEMV kernels from Metal to CUDA (same dequantization logic, different block layout)
  - Implement FP8 GEMV via either native FP8 math (if Blackwell supports it) or float conversion
  - Implement NVFP4/MXFP4 GEMV using same hierarchical microscaling as Metal
  - Benchmark each format and include in CI regression tests
- **Priority:** Medium — affects model selection on CUDA; most deployed models already use Q4_0/Q8_0

## Performance Bottlenecks

### GPU Sync Points in Inference Loop (High Impact)
- **Issue:** Multiple unnecessary GPU sync points remain in some code paths
- **Files:** `src/models/*.zig` (model implementations), `src/backend/metal.zig` (Metal CPU fallback)
- **Specific cases:**
  - Metal CPU-based SDPA requires `be.sync()` before final argmax (Gemma3, Qwen3.5)
  - Metal embLookup CPU fallback may not pre-sync GPU results in all code paths
  - Per-head normalization in Gemma3 may cause multiple syncs if head slices trigger buffer cache misses
- **Impact:** Each sync stalls GPU pipeline, ~150ms on Metal M4 Pro. 10 syncs per token = 1.5s latency addition
- **Optimization:**
  - Profile token generation with per-op sync instrumentation (`src/perf.zig` warnings apply)
  - Identify sync points that could be deferred to batch multiple ops
  - Consider fusing GPU operations to reduce dispatch count (e.g., norm + activation)
- **Current baseline:** Qwen3.5 0.8B achieves 167 tok/s on Metal (optimized), suggesting sync overhead is already mitigated in fast path

### RadixAttention LRU Eviction Not Implemented (Low-Medium Impact)
- **Issue:** RadixTree tracks `last_access` timestamps but eviction is not implemented
- **Files:** `src/kvcache/manager.zig` (line 224)
- **Impact:** In production serving with many concurrent requests, prefix cache may grow unbounded. Memory exhaustion possible
- **Current state:** Marked as "not yet implemented" in code comments
- **Fix approach:**
  - Implement `evictOldest()` method triggered when tree exceeds memory budget
  - Add configurable LRU budget and staleness threshold
  - Implement recursive `deinit()` with LRU criteria to free below target size
  - Test with synthetic workload (many requests with varying prefix overlap)
- **Workaround:** Single-request inference mode doesn't trigger issue; multi-request production serving needs this

### Paged Attention Block Table Fragmentation (Low Impact)
- **Issue:** PagedAttention block allocation may fragment memory during long inference sessions
- **Files:** `src/kvcache/manager.zig` (PagedKvCache implementation)
- **Impact:** Free list fragmentation can prevent allocating blocks for new requests even if total free space exists
- **Mitigation:** Consider block compaction pass or memory pool reallocation strategy for production deployments

## Security Considerations

### HTTP Server Input Validation (Medium Impact)
- **Issue:** HTTP request body size capped at 1MB but validation is basic
- **Files:** `src/server.zig` (line 36, line 570)
- **Current measures:**
  - `max_request_body_size = 1_000_000` enforced
  - `max_conversations = 100`, `max_messages_per_conv = 1000` limits enforced
  - `max_message_len = 100_000` per message limit enforced
- **Potential gaps:**
  - No rate limiting per IP or per conversation
  - No timeout enforcement on long-running inference (request timeout could leave server hanging)
  - JSON parsing could be DoS target if parser is O(n²) or unbounded recursion depth
  - Streaming SSE endpoint not protected against slow clients (server may accumulate unbounded output buffer)
- **Recommendations:**
  - Add per-IP request rate limiting (track in HTTP handler thread locals)
  - Implement inference timeout (cancel model after N seconds)
  - Limit JSON nesting depth and array sizes
  - Use bounded buffer for SSE streaming with backpressure (drop client if buffer exceeds threshold)
  - Document security assumptions for single-threaded inference (no concurrent requests)

### Content Security Policy (Low Impact)
- **Issue:** CSP allows `'unsafe-inline'` for scripts and styles
- **Files:** `src/server.zig` (line 589)
- **Current policy:** `Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline' https://cdn.jsdelivr.net; ...`
- **Risk:** XSS vulnerability if user input is not properly escaped in HTML responses
- **Current state:** Web UI uses `DOMPurify` sanitization which mitigates inline execution risk
- **Recommendation:** Generate script nonce at runtime and use `script-src 'nonce-...'` instead of `'unsafe-inline'` to enforce defense-in-depth

## Fragile Areas

### Meta Type Casting with @intCast (Low-Medium Impact)
- **Issue:** Several numeric operations use unsafe `@intCast` without validation
- **Files:** `src/ops/math.zig` (line 36), `src/display.zig` (lines 601), and others
- **Example:**
  ```zig
  best_idx = @intCast(i);  // Could panic if i > u32 max
  const filled: u32 = @intCast(@min(...));
  ```
- **Risk:** Integer overflow panics in Debug mode, undefined behavior in Release
- **Fix approach:**
  - Change loop index types to match target (use `u32` or `u16` for small loops)
  - Use `@truncate()` for deliberate truncation with documented semantics
  - Add assertions before unsafe casts

### Pointer Alignment Assumptions (Low-Medium Impact)
- **Issue:** Metal backend relies on page-aligned pointers for `newBufferWithBytesNoCopy`
- **Files:** `src/backend/metal.zig` (BufRef mechanism with page alignment checks)
- **Risk:** Activation buffers from GPA (General Purpose Allocator) may not be page-aligned; off-by-one offset calculations could fail
- **Current mitigation:** `getBufRef()` finds enclosing page-aligned region and returns offset
- **Risk level:** Medium — if alignment detection fails silently, subsequent GPU operations corrupt memory

### DeltaNet GQA Head Mapping (Fixed but Worth Noting)
- **Issue:** Fixed in recent commit but demonstrates fragile pattern
- **Files:** `src/backend/kernels/cpu/deltanet.zig`, `src/backend/kernels/metal/deltanet.metal`
- **Pattern:** GQA head mapping uses TILING (modulo grouping) not interleaved grouping
- **Risk:** Easy to regress if someone changes mapping logic without understanding GGML conventions
- **Recommendation:** Add invariant assertions and golden tests against llama.cpp output

### Array Indexing in MoE Expert Selection (Medium Impact)
- **Issue:** Expert selection uses stack-allocated fixed-size arrays with expert count limits
- **Files:** `src/models/nemotron_nano.zig` (line 37, `max_active_experts = 8`)
- **Risk:** If `num_experts_per_tok` exceeds `max_active_experts`, assertion fails. Config changes could silently break
- **Recommendation:** Add compile-time validation or dynamic bounds checking with clear error messages

### Zig Build Options Conditional Compilation (Low Impact)
- **Issue:** Model support disabled via compile-time flags, but runtime path still checks
- **Files:** `src/main.zig` (model initialization with `if (comptime build_options.enable_*) ...`)
- **Risk:** If build option accidentally disabled, users see cryptic "model support disabled" error
- **Recommendation:** Clear build-time error messages and CLI help text reflecting disabled models

## Known Bugs

### Metal MLX Q4 GEMV Unaligned Access Performance (Low Impact)
- **Issue:** SafeTensors MLX-quantized weights may have odd byte offsets; Metal kernel uses `packed_uchar4` workaround
- **Files:** `src/backend/metal.zig` (GEMV MLX Q4 kernel dispatch), `src/backend/kernels/metal/gemv.metal`
- **Impact:** Unaligned reads reduce throughput; achieves ~62% of theoretical bandwidth on M4 Pro
- **Workaround:** Data repacking on load would add startup latency; tradeoff accepted for simplicity
- **Status:** Known and accepted; performance is adequate (multiple tok/s on Gemma3 27B)

### CUDA Musl Cross-Compilation Incompatibility (Low Impact)
- **Issue:** CUDA library (libcuda.so) linked against glibc; musl static builds will segfault
- **Files:** `src/backend/cuda.zig` (dynamic library loading)
- **Workaround:** Must use `-Dtarget=aarch64-linux-gnu` (not musl). Documented in code
- **Risk:** Developers attempting musl builds will get mysterious segfaults
- **Recommendation:** Add build-time guard or clearer error message

## Missing Critical Features

### Paged Attention Not Integrated into Models (Medium Impact)
- **Issue:** PagedKvCache implementation exists but not used by any model
- **Files:** `src/kvcache/manager.zig` (PagedKvCache struct), all model files
- **Impact:** Continuous batching optimization (main use case for paged attention) not available
- **Fix approach:**
  - Modify model interface to accept KV cache strategy (flat, paged, radix)
  - Update inference loop to use block table indirection instead of direct sequence offset
  - Test with synthetic workload (many concurrent requests with varying lengths)
  - Benchmark memory efficiency and scheduling latency improvement
- **Priority:** Medium — important for production serving but not blocking current single-request inference

### RadixAttention Integration Gap (Medium Impact)
- **Issue:** RadixTree fully implemented but not integrated into server's multi-request inference
- **Files:** `src/kvcache/manager.zig` (RadixTree), `src/server.zig` (conversation management)
- **Current state:** Server maintains per-conversation KV cache but doesn't use prefix sharing across conversations
- **Impact:** Prefix cache benefits (faster repeat queries, reduced KV memory) not realized
- **Fix approach:**
  - Modify server to maintain global RadixTree instead of per-conversation flat cache
  - Track which cached prefixes apply to active conversation
  - Update conversation switching logic to reuse cached KV prefixes
  - Implement LRU eviction (prerequisite)
  - Benchmark improvement on typical conversation patterns
- **Priority:** Low-Medium — optimization for later phases after core functionality stabilized

## Scaling Limits

### Default KV Cache Context Size (Low-Medium Impact)
- **Issue:** Hardcoded `default_ctx_size = 4096` (line 62 in `src/main.zig`)
- **Files:** `src/main.zig`, `src/server.zig`
- **Current capacity:** Metal SDPA GPU kernel limited to 4096 (threadgroup memory constraint)
- **Constraint note:** Line 61 documents this is "Matches GPU SDPA kernel limit"
- **Problem:** If GPU SDPA is enabled in future, 4096 is actual hard limit; if disabled, CPU SDPA supports arbitrary sequences
- **Fix approach:**
  - Make default configurable via CLI and recipe system
  - Document actual limits per backend (Metal: 4096, Vulkan/CUDA: 16384+ depending on memory)
  - Add validation: warn if user requests context > backend limit
  - Consider dynamic context sizing based on available VRAM

### HTTP Server Max Concurrent Connections (Low Impact)
- **Issue:** `max_concurrent_connections = 64` hardcoded
- **Files:** `src/server.zig` (line 44)
- **Current model:** Single inference lock serializes requests anyway, so connection limit is per-thread queue depth
- **Impact:** Server accepts max 64 concurrent HTTP connections (though only one can run inference at a time)
- **Scaling:** Adequate for single-model serving; would need model sharding or ensemble for higher throughput
- **Recommendation:** Make configurable via CLI, document that inference is still serialized

## Test Coverage Gaps

### Integration Tests Missing for Multi-Model Scenarios (Low-Medium Impact)
- **What's not tested:** Loading different quantization formats and backends in same session
- **Files:** `tests/`, individual model test blocks
- **Risk:** Regressions in backend switching logic and memory cleanup between model loads
- **Test needed:** Load model A (Q4_0, Metal) → unload → Load model B (Q8_0, CUDA) → verify correctness

### GPU Backend Output Correctness Under Stress (Low-Medium Impact)
- **What's not tested:** Long sequences (near context limit), extreme batch sizes, rapid backend switching
- **Files:** No stress test harness
- **Risk:** Subtle GPU memory corruption or synchronization bugs visible only under load
- **Test needed:** Generate 10k tokens with batch inference, verify determinism with seed

### Quantization Accuracy Regression Tests (Low-Medium Impact)
- **What's not tested:** End-to-end model output accuracy for each quantization format
- **Files:** `src/ops/quant.zig` has unit tests but models don't validate output against reference
- **Risk:** Dequantization bugs (e.g., NVFP4 SafeTensors layout) detectable only at model level
- **Test needed:** Golden test each model × quantization format against reference outputs from llama.cpp with same seed

### Server Endpoint Security Tests (Low Impact)
- **What's not tested:** Malicious inputs to HTTP endpoints (oversized JSON, deeply nested structures, rapid-fire requests)
- **Files:** `src/server.zig` (no fuzz tests or adversarial tests)
- **Risk:** DoS or crash via crafted HTTP requests
- **Test needed:** Fuzzer for /v1/chat/completions endpoint, rate limit tests

---

## Summary by Priority

**High Priority (Blocking Quality Goals):**
- Metal SDPA GPU fallback performance regression
- Nemotron Nano output quality issue
- CUDA serial softmax workaround investigation

**Medium Priority (Important for Feature Completeness):**
- Incomplete Vulkan backends (embLookup, conv1d, NVFP4)
- Incomplete CUDA GEMV quantization formats
- RadixAttention LRU eviction implementation
- Paged attention integration
- HTTP server input validation hardening

**Low Priority (Technical Debt, Optimization):**
- Pointer alignment assumption hardening
- Integer cast safety improvements
- GPU sync point reduction
- Build option validation

*Concerns audit: 2025-02-21*
