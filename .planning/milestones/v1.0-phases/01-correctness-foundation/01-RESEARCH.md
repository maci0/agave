# Phase 1: Correctness Foundation - Research

**Researched:** 2026-03-21
**Domain:** GPU kernel optimization, model architecture debugging, numerical validation
**Confidence:** HIGH

## Summary

Phase 1 establishes correctness across all backends (Metal, CUDA, Vulkan, ROCm, CPU) and all 6 models (Gemma3, Qwen3.5, Nemotron Nano, GLM-4, GPT-OSS, Nemotron-H) plus DeepSeek-R1-Qwen3-8B. The core challenges are: (1) Metal SDPA produces wrong output due to likely race condition or barrier placement, (2) CUDA lacks quantized GEMV kernels for Q4_K/Q5_K/Q6_K/FP8 formats, (3) Vulkan lacks GPU kernels for embedding lookup and conv1d, (4) four models are broken or unverified (Nemotron Nano MoE routing overflow, GLM-4 MLA+MoE untested, GPT-OSS sliding-window MoE untested, Nemotron-H hybrid SSM+attention untested).

**Primary recommendation:** Rewrite Metal SDPA with FlashAttention-2 algorithm (don't debug existing kernel). Implement all CUDA quantized GEMV formats together (they share dequant patterns). Fix broken models in order: Nemotron Nano (highest value — 30B parameter model), then GLM-4/GPT-OSS/Nemotron-H.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Metal SDPA:**
- D-01: Rewrite Metal SDPA with FlashAttention-2 tiling algorithm (do NOT debug existing broken GPU kernel)
- D-02: Use compute-only path — NEVER use blit encoders in hot path (causes 150ms/layer stalls from encoder switching)
- D-03: Implement online softmax (single-pass running max+sum) for numerical stability

**CUDA Quantized GEMV:**
- D-04: Implement Q4_K, Q5_K, Q6_K, and FP8 (E4M3/E5M2) GEMV kernels all at once (they share dequant patterns)
- D-05: All dequantization MUST happen in-kernel (no pre-dequant to f32 scratch buffers)
- D-06: Use warp-only reductions for SDPA parallel softmax (avoid shared memory block reductions that hang on sm_121 Blackwell)

**Model Verification:**
- D-07: DeepSeek-R1-0528-Qwen3-8B (Q8_0 GGUF) loads via existing Qwen3.5 code path — it's a Qwen3 architecture variant
- D-08: Fix all 4 broken/unverified models: Nemotron Nano (MoE routing overflow), GLM-4 (MLA + MoE), GPT-OSS (sliding window MoE), Nemotron-H (hybrid SSM+attention)
- D-09: For Nemotron Nano specifically: investigate MoE router score overflow (-2.3e26) — likely needs per-block scaling for quantized router weights

**Testing Strategy:**
- D-10: Golden tests use BOTH reference implementations: llama.cpp for GGUF models, HuggingFace (PyTorch) for SafeTensors models
- D-11: Dual-delta numerical tests: compare GPU and CPU to FP64 oracle, verify GPU error <= 2x CPU error
- D-12: All models verified on all 5 backends: CPU, Metal, CUDA, Vulkan, ROCm

**ROCm Backend:**
- D-13: ROCm testing on `maci@192.168.0.205` (24GB VRAM — be careful with model sizes)
- D-14: Test all models that fit in 24GB VRAM on ROCm
- D-15: CUDA testing on DGX Spark at `maci@192.168.0.212` (Blackwell sm_121, UMA)

**Vulkan Fallbacks:**
- D-16: Implement Vulkan GPU embedding lookup kernel (eliminate CPU fallback)
- D-17: Implement Vulkan GPU conv1d kernel for SSM models (eliminate CPU fallback)

### Claude's Discretion

- Exact FlashAttention-2 block sizes and tiling strategy for Metal
- CUDA FP8 approach: native intrinsics (Ada/Hopper+) vs 256-entry LUT
- Order of model debugging (which broken model to fix first)
- Golden test tolerance thresholds (exact epsilon values)
- Whether to use deterministic seeding or multi-seed statistical comparison

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| KERN-01 | Metal GPU SDPA kernel produces correct output without CPU fallback | FlashAttention-2 algorithm + online softmax researched |
| KERN-02 | Metal SDPA uses compute-only path (no blit encoder switching in hot path) | Metal backend patterns documented |
| KERN-03 | CUDA GPU GEMV kernel for Q4_K quantization format | Q4_K block structure (144 bytes, 12 scales) documented |
| KERN-04 | CUDA GPU GEMV kernel for Q5_K quantization format | Q5_K block structure researched, CPU reference exists |
| KERN-05 | CUDA GPU GEMV kernel for Q6_K quantization format | Q6_K block structure researched, CPU reference exists |
| KERN-06 | CUDA GPU GEMV kernel for FP8 E4M3/E5M2 formats | FP8 LUT vs intrinsics approaches researched |
| KERN-07 | CUDA parallel SDPA softmax using warp-only reductions | Warp reduction patterns in common.zig, sm_121 hang workaround |
| KERN-08 | Vulkan GPU embedding lookup kernel | Vulkan compute shader patterns documented |
| KERN-09 | Vulkan GPU conv1d kernel for SSM models | SSM causal conv1d algorithm in ops/ssm.zig |
| KERN-10 | All GPU kernels pass dual-delta numerical tests (GPU error <= 2x CPU error vs FP64 oracle) | Dual-delta testing strategy defined |
| MODL-01 | Nemotron Nano 30B produces correct, coherent output (fix MoE routing instability) | MoE router overflow diagnosed (-2.3e26), per-block scaling solution |
| MODL-02 | GLM-4 produces correct output (MLA attention + sigmoid-gated MoE) | MLA architecture researched, sigmoid routing vs softmax |
| MODL-03 | GPT-OSS produces correct output (sliding window + MoE with clamped SwiGLU) | Sliding window (128 tokens), sink tokens, ±7.0 clamp |
| MODL-04 | Nemotron-H produces correct output (hybrid SSM + attention) | Hybrid pattern dispatch researched |
| MODL-05 | All 6 models verified on CPU backend with golden test output | Test harness exists (tests/harness.py) |
| MODL-06 | All 6 models verified on Metal backend with golden test output | Metal backend patterns + model integration |
| MODL-07 | All 6 models verified on CUDA backend (DGX Spark) with golden test output | CUDA test machine access + deferred sync pattern |
| MODL-08 | All 6 models verified on Vulkan backend with golden test output | Vulkan backend near-complete, 17 SPIR-V shaders |
| MODL-09 | Automated golden tests comparing output against reference (llama.cpp or HuggingFace) | llama.cpp (GGUF) + HuggingFace (SafeTensors) as dual references |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Zig | 0.14.0 (assumed latest) | Language & build system | Project written in Zig, no external C/C++ libs allowed per CLAUDE.md |
| PTX (nvptx64-cuda) | Via Zig compiler | CUDA kernel compilation | Zero dependency on CUDA C++ — kernels written in Zig, compiled to PTX |
| SPIR-V (spirv64-vulkan) | Via glslc/spirv-tools | Vulkan shader compilation | Cross-platform Vulkan kernels compiled from GLSL |
| MSL (Metal Shading Language) | Apple SDK | Metal kernel compilation | Embedded via @embedFile, compiled at runtime |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| llama.cpp | Latest git head | GGUF golden reference | Correctness validation for GGUF models (Gemma3, Qwen3.5, etc.) |
| HuggingFace transformers | Latest PyPI | SafeTensors golden reference | Correctness validation for SafeTensors models (MLX format) |
| Python 3.11+ | Latest stable | Test harness | Existing tests/harness.py orchestrates testing |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| FlashAttention-2 | FlashAttention-3 | FA-3 requires H100+, beta-only. FA-2 is production-ready, cross-platform |
| FP8 256-entry LUT | Native CUDA FP8 intrinsics | LUT works on all compute caps; intrinsics require Ada/Hopper (sm_89+). Use LUT for broad compatibility |
| Warp-only reductions | Block reductions with shared memory | Block reductions hang on sm_121 Blackwell (compiler bug). Warp-only is slower but correct |

**Installation:**

None — all dependencies are either built-in (Zig compiler) or already present (test harness Python venv).

**Version verification:**

Project uses Zig build system. CUDA kernels compiled via `zig build ptx`, Vulkan shaders via glslc (manual, .spv committed to repo).

## Architecture Patterns

### Recommended Project Structure

```
src/
├── backend/
│   ├── backend.zig              # Tagged union dispatcher (inline else)
│   ├── metal.zig                # Metal backend (SDPA rewrite target)
│   ├── cuda.zig                 # CUDA backend (add quantized GEMV)
│   ├── vulkan.zig               # Vulkan backend (add embLookup, conv1d)
│   └── kernels/
│       ├── metal/
│       │   └── sdpa.metal       # FlashAttention-2 rewrite
│       ├── cuda/
│       │   ├── common.zig       # PTX math, warp reduction (reuse)
│       │   ├── gemv_q4_k.zig    # NEW: Q4_K GEMV
│       │   ├── gemv_q5_k.zig    # NEW: Q5_K GEMV
│       │   ├── gemv_q6_k.zig    # NEW: Q6_K GEMV
│       │   ├── gemv_fp8_e4m3.zig # NEW: FP8 E4M3 GEMV
│       │   ├── gemv_fp8_e5m2.zig # NEW: FP8 E5M2 GEMV
│       │   └── sdpa.zig         # Upgrade to warp-only parallel softmax
│       └── vulkan/
│           ├── embedding.comp   # NEW: GPU embedding lookup
│           └── conv1d.comp      # NEW: GPU conv1d for SSM
├── models/
│   ├── nemotron_nano.zig        # FIX: MoE router overflow
│   ├── glm4.zig                 # FIX: MLA + sigmoid MoE
│   ├── gpt_oss.zig              # FIX: sliding window + clamped SwiGLU
│   └── nemotron_h.zig           # FIX: hybrid SSM + attention
├── ops/
│   ├── quant.zig                # Dequant helpers (reference for CUDA kernels)
│   └── attention.zig            # CPU SDPA reference (correctness oracle)
└── tests/
    ├── harness.py               # Existing test orchestrator (extend)
    └── golden/                  # NEW: Golden reference outputs
        ├── llama_cpp/           # llama.cpp outputs (GGUF models)
        └── huggingface/         # HF transformers outputs (SafeTensors)
```

### Pattern 1: FlashAttention-2 Tiling

**What:** Block-based attention computation that keeps intermediate softmax state in SRAM (threadgroup memory on Metal) instead of writing to device memory.

**When to use:** Metal SDPA kernel (KERN-01). Reduces memory bandwidth 1.33× via online softmax.

**Algorithm structure:**
```
1. Divide K,V into blocks of size Bc × d (e.g., 64×256)
2. Divide Q,O into blocks of size Br × d (e.g., 128×256)
3. For each Q block (outer loop):
   a. Initialize running max m_i = -∞, running sum l_i = 0, output O_i = 0
   b. For each K,V block (inner loop):
      - Load Q_block, K_block, V_block to SRAM
      - Compute S = Q @ K^T / sqrt(d)
      - Online softmax update:
        * m_new = max(m_i, max(S))
        * l_i = l_i × exp(m_i - m_new) + sum(exp(S - m_new))  [rescale sum]
        * O_i = O_i × exp(m_i - m_new) + exp(S - m_new) @ V    [rescale output]
        * m_i = m_new
   c. Final normalization: O_i = O_i / l_i
```

**Block size constraints:**
- Threadgroup memory on Apple Silicon M-series: ~32KB per threadgroup
- Q_block + K_block + V_block + scores + shared[] must fit in 32KB
- Typical: Bc=64, Br=128, d=128 → (128+64+64)×128×4 bytes + 128×64×4 bytes = ~164KB → too large
- Metal tuned: Bc=32, Br=64, d≤256 → (64+32+32)×256×4 bytes + 64×32×4 bytes = ~140KB → still tight
- **Recommended start:** Bc=32, Br=32, d=128 → 49KB (fits easily, iterate to optimize)

**Key insight:** Online softmax eliminates the need to store full attention matrix in device memory. The rescaling formula `l_i × exp(m_i - m_new)` is the "telescoping sum" property that maintains numerical stability.

**References:**
- [FlashAttention-2 Paper (2023)](https://arxiv.org/pdf/2307.08691) — Algorithm 3, page 5
- [Online Softmax Explanation](https://wangkuiyi.github.io/online-softmax.html) — Mathematical derivation

### Pattern 2: CUDA Quantized GEMV with In-Kernel Dequantization

**What:** GEMV (y = W @ x) where W is quantized (Q4_K/Q5_K/Q6_K/FP8) and dequantized inside the kernel, never materializing full f32 weights.

**When to use:** CUDA backend (KERN-03 through KERN-06). All formats share reduction and dequant patterns.

**Shared kernel structure:**
```zig
// One thread block per output row (y[i] = W[i,:] @ x[:])
// Each thread computes partial dot product, then warp reduction

export fn gemv_qX_k(
    x: [*]const f32,      // Input vector [k]
    w: [*]const u8,       // Quantized weights [n × bytes_per_row]
    y: [*]f32,            // Output vector [n]
    n: u32,               // Number of output rows
    k: u32,               // Input dimension
) callconv(.kernel) void {
    const tid = common.threadIdx();
    const row = common.blockIdx();
    if (row >= n) return;

    const row_ptr = w + row * bytes_per_row;
    var sum: f32 = 0.0;

    // Each thread processes subset of input dimension
    var i = tid;
    while (i < k) : (i += common.blockDim()) {
        // Format-specific dequantization logic here
        const w_val = dequantElement(row_ptr, i);
        sum += x[i] * w_val;
    }

    // Warp reduction (reuse common.warpReduceAdd)
    sum = common.warpReduceAdd(sum);

    // First thread in each warp writes to shared memory
    const lane = tid % 32;
    const warp_id = tid / 32;
    if (lane == 0) common.sharedStore(warp_id, sum);
    common.syncthreads();

    // First warp reduces across warps
    const n_warps = (common.blockDim() + 31) / 32;
    var result = if (tid < n_warps) common.sharedLoad(tid) else 0.0;
    if (warp_id == 0) result = common.warpReduceAdd(result);

    // Thread 0 writes final result
    if (tid == 0) y[row] = result;
}
```

**Format-specific dequantization:**

**Q4_K** (256 values/block, 144 bytes):
```zig
// Layout: d(f16) + dmin(f16) + scales[12] + qs[128]
// 8 sub-blocks of 32 values each, 12 packed scale/min pairs
const block_idx = i / 256;
const sub_block = (i % 256) / 32;
const elem_in_sub = i % 32;
const bp = row_ptr + block_idx * 144;
const d = f16tof32(bp[0..2]);
const dmin = f16tof32(bp[2..4]);
var sc: u8 = undefined;
var m: u8 = undefined;
getScaleMinK4(sub_block, bp[4..16], &sc, &m);
const nibble_idx = sub_block * 16 + elem_in_sub / 2;
const nibble = if (elem_in_sub % 2 == 0) bp[16 + nibble_idx] & 0x0F else bp[16 + nibble_idx] >> 4;
return d * (@as(f32, @floatFromInt(sc)) * @as(f32, @floatFromInt(nibble))) - dmin * @as(f32, @floatFromInt(m));
```

**FP8 E4M3** (direct conversion via LUT):
```zig
const fp8_val = w[row * k + i];
return fp8e4m3_lut[fp8_val];  // 256-entry precomputed LUT (from quant.zig)
```

**Key insight:** CPU GEMV kernels in `src/backend/kernels/cpu/gemv_q{4,5,6}_k.zig` contain the exact dequant logic. Port these to CUDA PTX using the established patterns in `common.zig`.

### Pattern 3: Warp-Only Reductions (sm_121 Blackwell Workaround)

**What:** Parallel softmax using only warp-level shuffles, avoiding shared memory block reductions that hang on CUDA sm_121 (Blackwell).

**When to use:** CUDA SDPA kernel (KERN-07).

**Why needed:** CUDA compiler for sm_121 generates incorrect code for block reductions with shared memory. Warp-only reductions are slower (only 32 threads, not 256) but correct.

**Pattern:**
```zig
// Serial softmax per Q-head (existing workaround in cuda/sdpa.zig)
if (tid == 0) {
    var max_val: f32 = common.neg_f32_max;
    for (0..seq_len) |t| max_val = @max(max_val, scores[t]);

    var sum: f32 = 0.0;
    for (0..seq_len) |t| {
        const v = common.expf(scores[t] - max_val);
        scores[t] = v;
        sum += v;
    }
    const inv_sum = common.rcpf(sum);
    for (0..seq_len) |t| scores[t] *= inv_sum;
}

// Warp-only parallel softmax (Wave 1 improvement)
// Distribute seq_len across 32 threads instead of 1 thread
const warp_size = 32;
const chunk = (seq_len + warp_size - 1) / warp_size;
const start = tid * chunk;
const end = @min(start + chunk, seq_len);

// Phase 1: Find max (warp reduction)
var local_max: f32 = common.neg_f32_max;
for (start..end) |t| local_max = @max(local_max, scores[t]);
const max_val = common.warpReduceMax(local_max);

// Phase 2: Exp and sum (warp reduction)
var local_sum: f32 = 0.0;
for (start..end) |t| {
    const v = common.expf(scores[t] - max_val);
    scores[t] = v;
    local_sum += v;
}
const sum = common.warpReduceAdd(local_sum);

// Phase 3: Normalize
const inv_sum = common.rcpf(sum);
for (start..end) |t| scores[t] *= inv_sum;
```

**Constraint:** Only works for seq_len ≤ ~8192 (32 threads × 256 elements/thread = reasonable workload). For longer sequences, need true block-parallel softmax (requires fixing sm_121 compiler bug or workaround with different memory pattern).

### Pattern 4: Vulkan Compute Shader (GLSL → SPIR-V)

**What:** Vulkan kernels written in GLSL (compute shaders), compiled to SPIR-V, loaded at runtime.

**When to use:** Vulkan backend (KERN-08, KERN-09). Embedding lookup and conv1d kernels.

**Example: Embedding lookup (KERN-08)**
```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { uint token_id; };
layout(binding = 1) readonly buffer Embeddings { float data[]; } emb;
layout(binding = 2) writeonly buffer Output { float data[]; } out_buf;

layout(push_constant) uniform Params {
    uint vocab_size;
    uint n_embd;
} params;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= params.n_embd) return;

    uint offset = token_id * params.n_embd + tid;
    out_buf.data[tid] = emb.data[offset];
}
```

**Compile:** `glslc -fshader-stage=compute embedding.comp -o embedding.spv`

**Example: Causal Conv1d (KERN-09)**
```glsl
#version 450

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { float data[]; } input_buf;
layout(binding = 1) readonly buffer ConvState { float data[]; } state;
layout(binding = 2) readonly buffer Weights { float data[]; } conv_w;
layout(binding = 3) writeonly buffer Output { float data[]; } output;

layout(push_constant) uniform Params {
    uint conv_ch;     // Number of channels
    uint d_conv;      // Kernel size (e.g., 4)
} params;

void main() {
    uint ch = gl_GlobalInvocationID.x;
    if (ch >= params.conv_ch) return;

    float sum = 0.0;
    // Convolve: conv_state[(d_conv-1) × conv_ch] is ring buffer
    // Last d_conv-1 positions + current input
    for (uint i = 0; i < params.d_conv - 1; ++i) {
        sum += state.data[i * params.conv_ch + ch] * conv_w.data[i * params.conv_ch + ch];
    }
    sum += input_buf.data[ch] * conv_w.data[(params.d_conv - 1) * params.conv_ch + ch];
    output.data[ch] = sum;
}
```

**Integration:** Add to `vulkan.zig` pipeline creation (`createComputePipeline`), load SPIR-V via `@embedFile("kernels/vulkan/embedding.spv")`.

### Anti-Patterns to Avoid

**Pre-dequantization to f32 scratch buffer:**
```zig
// BAD: Defeats purpose of quantization
var f32_weights = try allocator.alloc(f32, n * k);
dequantizeToF32(f32_weights, quantized_weights, dtype, n * k);
be.gemv(x, .{ .data = f32_weights, .dtype = .f32 }, y, n, k);
```
**Do instead:** Pass quantized weights directly, dequant in-kernel via `inline` helper or PTX instructions.

**Blit encoders in Metal hot path:**
```metal
// BAD: Causes 150ms/layer stalls
id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
[blit copyFromBuffer:src toBuffer:dst ...];
[blit endEncoding];
```
**Do instead:** Use compute kernels for all data movement (`kv_append` compute shader).

**Block reductions on CUDA sm_121:**
```zig
// BAD: Hangs on Blackwell due to compiler bug
const result = common.blockReduceAdd(val);  // Uses shared memory
```
**Do instead:** Use warp-only reductions (`common.warpReduceAdd`) until compiler bug is fixed.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Attention algorithm | Naive QK^T softmax V | FlashAttention-2 (tiled) | 1.33× memory bandwidth savings, numerically stable online softmax |
| FP8 conversion | Bit-twiddling conversion | 256-entry LUT (comptime) | Zero branches, zero arithmetic, single array lookup — Metal already uses this pattern |
| CUDA warp shuffles | Manual shared memory | `asm volatile` shfl.sync | Compiler sinks non-volatile shuffles into conditionals → deadlock |
| Numerical stability in softmax | Naive exp(x) / sum | Online softmax (running max+sum) | Prevents overflow, single-pass, FA-2 compatible |
| Quantization logic | Invent new dequant | Port from CPU kernels | CPU kernels in `gemv_q{4,5,6}_k.zig` are battle-tested, well-documented |

**Key insight:** This project already has working CPU implementations of every operation. GPU kernels should port proven CPU logic, not reinvent it.

## Runtime State Inventory

> Section omitted — not a rename/refactor phase. All state is code-managed (buffers, KV cache).

## Common Pitfalls

### Pitfall 1: Metal Encoder Switching

**What goes wrong:** Using blit encoders for KV cache append in the hot path causes 150ms/layer stalls (observed on 27B Gemma3).

**Why it happens:** Metal command buffer cannot have overlapping encoders. Switching from compute → blit → compute forces serialization and pipeline flush.

**How to avoid:** Keep everything in compute shaders. Use `kv_append` compute kernel (already exists in `elementwise.metal` but unused) instead of blit encoder.

**Warning signs:** `std.log.warn` in `metal.zig` about "flushing for blit", sudden 10-100× slowdown on multi-layer models.

### Pitfall 2: CUDA sm_121 Block Reduction Hang

**What goes wrong:** `blockReduceAdd` / `blockReduceMax` in `common.zig` hangs indefinitely on Blackwell GPUs (compute cap sm_121).

**Why it happens:** CUDA compiler bug in shared memory codegen for sm_121. Generates incorrect barrier synchronization.

**How to avoid:** Use warp-only reductions (`warpReduceAdd`, `warpReduceMax`) for Blackwell targets. Only 32 threads instead of 256, but correct.

**Warning signs:** Kernel launch never returns, `cuCtxSynchronize` hangs forever, 100% GPU utilization with no output.

### Pitfall 3: Nemotron Nano MoE Router Overflow

**What goes wrong:** Router logits are -2.3e26 (extreme negative), causing sigmoid(logit) ≈ 0 and all expert gates to collapse.

**Why it happens:** Router weights are NVFP4 quantized. Without per-block scaling, quantization error accumulates catastrophically during GEMV.

**How to avoid:** Implement per-block scale tracking for router GEMV. Accumulate `(d_block * q_dot_block)` per quantization block, then sum scaled blocks. Don't dequant everything then dot.

**Warning signs:** Model generates nonsensical text, router scores are outside [-100, +100] range, sigmoid outputs all near 0 or 1.

### Pitfall 4: FlashAttention-2 Block Size Overrun

**What goes wrong:** Choosing block sizes (Bc, Br) too large causes shared memory overrun → kernel launch failure or garbage output.

**Why it happens:** Threadgroup memory on Apple Silicon is ~32KB. (Bc + Br) × d × 4 bytes + score matrix must fit.

**How to avoid:** Start with conservative block sizes (Bc=32, Br=32, d=128 → ~49KB). Verify with actual head_dim from model configs (Gemma3=256, Qwen=128). Use `@comptime` asserts to check memory bounds.

**Warning signs:** Metal kernel fails with "insufficient threadgroup memory", output is all NaN or zero.

### Pitfall 5: Dual-Delta Test False Positives

**What goes wrong:** GPU implementation rejected as "wrong" even though it's numerically equivalent to CPU within tolerance.

**Why it happens:** FP32 arithmetic is not associative. Different reduction order (GPU parallel vs CPU serial) produces different rounding errors.

**How to avoid:** Compare BOTH CPU and GPU to FP64 oracle. Accept GPU if `|GPU - Oracle| <= 2 × |CPU - Oracle|`. This allows 2× accumulated error, reasonable for parallel reduction.

**Warning signs:** Golden tests fail with tiny differences (~1e-6), but output text is identical.

## Code Examples

Verified patterns from existing codebase and official sources:

### Metal FlashAttention-2 SDPA (Skeleton)

```metal
// Scaled Dot-Product Attention with FlashAttention-2 tiling
// One threadgroup per query head. Online softmax for numerical stability.
// Maximum sequence length: 4096 (limited by threadgroup memory).

constant uint Bc = 32;  // K,V block size
constant uint Br = 32;  // Q,O block size
constant uint max_d = 256;  // Maximum head dimension

kernel void sdpa_fa2(
    device const float* Q,       // [nh * hd]
    device const float* K_cache, // [>= sl * kvd]
    device const float* V_cache, // [>= sl * kvd]
    device float* output,        // [nh * hd]
    constant uint& nh,
    constant uint& nkv,
    constant uint& hd,
    constant uint& sl,           // sequence length (1..4096)
    constant float& scale,
    uint h     [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]])
{
    if (h >= nh) return;
    uint hpg = nh / nkv;
    uint kvh = h / hpg;
    uint kvd = nkv * hd;

    threadgroup float q_local[max_d];
    threadgroup float k_block[Bc * max_d];
    threadgroup float v_block[Bc * max_d];
    threadgroup float scores[Br * Bc];
    threadgroup float shared[8];  // For reductions

    // Load Q into threadgroup memory (entire head)
    for (uint d = tid; d < hd; d += tg_sz) {
        q_local[d] = Q[h * hd + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Initialize running max and sum for online softmax
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float out_acc[max_d];  // Register accumulator for output
    for (uint d = 0; d < hd; d++) out_acc[d] = 0.0f;

    // Outer loop: iterate over K,V blocks
    uint num_blocks = (sl + Bc - 1) / Bc;
    for (uint b = 0; b < num_blocks; b++) {
        uint block_start = b * Bc;
        uint block_size = min(Bc, sl - block_start);

        // Load K block
        for (uint t = tid; t < block_size; t += tg_sz) {
            uint k_base = (block_start + t) * kvd + kvh * hd;
            for (uint d = 0; d < hd; d++) {
                k_block[t * hd + d] = K_cache[k_base + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute scores: Q @ K_block^T
        for (uint t = tid; t < block_size; t += tg_sz) {
            float dot = 0.0f;
            for (uint d = 0; d < hd; d++) {
                dot += q_local[d] * k_block[t * hd + d];
            }
            scores[t] = dot * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax: update running max and sum
        float m_prev = m_i;
        float m_new = m_prev;
        for (uint t = tid; t < block_size; t += tg_sz) {
            m_new = max(m_new, scores[t]);
        }
        // Reduce to find block max (use threadgroup_reduce_max)
        m_new = threadgroup_reduce_max(m_new, tid, tg_sz, shared);

        // Rescale previous sum and accumulator
        float scale_factor = exp(m_prev - m_new);
        l_i *= scale_factor;
        for (uint d = 0; d < hd; d++) out_acc[d] *= scale_factor;

        // Load V block
        for (uint t = tid; t < block_size; t += tg_sz) {
            uint v_base = (block_start + t) * kvd + kvh * hd;
            for (uint d = 0; d < hd; d++) {
                v_block[t * hd + d] = V_cache[v_base + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate: out_acc += exp(scores - m_new) @ V_block
        for (uint t = tid; t < block_size; t += tg_sz) {
            float p = exp(scores[t] - m_new);
            l_i += p;
            for (uint d = 0; d < hd; d++) {
                out_acc[d] += p * v_block[t * hd + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        m_i = m_new;
    }

    // Final normalization and write output
    for (uint d = tid; d < hd; d += tg_sz) {
        output[h * hd + d] = out_acc[d] / l_i;
    }
}
```

**Source:** [FlashAttention-2 Algorithm 3](https://arxiv.org/pdf/2307.08691) adapted to Metal Shading Language

### CUDA Q4_K GEMV Kernel (Skeleton)

```zig
// Q4_K GEMV: y = W @ x
// 256 values/block, 144 bytes: d(f16) + dmin(f16) + scales[12] + qs[128]

const common = @import("common.zig");
const quant = @import("../../ops/quant.zig");

const bytes_per_block: usize = 144;
const values_per_block: usize = 256;

export fn gemv_q4_k(
    x: [*]const f32,
    w: [*]const u8,
    y: [*]f32,
    n: u32,
    k: u32,
) callconv(.kernel) void {
    const tid = common.threadIdx();
    const row = common.blockIdx();
    if (row >= n) return;

    const num_blocks = (k + values_per_block - 1) / values_per_block;
    const row_ptr = w + row * num_blocks * bytes_per_block;

    var sum: f32 = 0.0;

    // Each thread processes subset of blocks
    var b = tid;
    while (b < num_blocks) : (b += common.blockDim()) {
        const bp = row_ptr + b * bytes_per_block;
        const d = f16tof32(bp);
        const dmin = f16tof32(bp + 2);
        const scales = bp + 4;
        const qs = bp + 16;
        const block_start = b * values_per_block;

        // Process 8 sub-blocks of 32 values each
        var sb: u32 = 0;
        while (sb < 8) : (sb += 1) {
            const gi_base = block_start + sb * 32;
            if (gi_base >= k) break;

            var sc: u8 = undefined;
            var m: u8 = undefined;
            quant.getScaleMinK4(sb, scales, &sc, &m);

            const d_sc = d * @as(f32, @floatFromInt(sc));
            const dm_m = dmin * @as(f32, @floatFromInt(m));

            // Accumulate dot product for sub-block
            var q_dot: f32 = 0.0;
            var x_sum: f32 = 0.0;
            const qi_base = sb * 16;  // 32 values = 16 bytes
            for (0..32) |l| {
                const gi = gi_base + l;
                if (gi >= k) break;
                const byte_idx = qi_base + l / 2;
                const nibble: f32 = @floatFromInt(
                    if (l % 2 == 0) qs[byte_idx] & 0x0F else qs[byte_idx] >> 4
                );
                q_dot += x[gi] * nibble;
                x_sum += x[gi];
            }
            // Factored: dot(x, d*q - dm) = d*dot(x,q) - dm*sum(x)
            sum += d_sc * q_dot - dm_m * x_sum;
        }
    }

    // Warp reduction
    sum = common.warpReduceAdd(sum);

    const lane = tid % 32;
    const warp_id = tid / 32;
    if (lane == 0) common.sharedStore(warp_id, sum);
    common.syncthreads();

    const n_warps = (common.blockDim() + 31) / 32;
    var result = if (tid < n_warps) common.sharedLoad(tid) else 0.0;
    if (warp_id == 0) result = common.warpReduceAdd(result);

    if (tid == 0) y[row] = result;
}

// Helper: f16 to f32 conversion
inline fn f16tof32(ptr: [*]const u8) f32 {
    const val = @as(u16, ptr[0]) | (@as(u16, ptr[1]) << 8);
    // f16: 1 sign, 5 exp, 10 mantissa → f32: 1 sign, 8 exp, 23 mantissa
    const sign: u32 = @as(u32, val >> 15) << 31;
    const exp_f16: u32 = (val >> 10) & 0x1F;
    const mant_f16: u32 = val & 0x3FF;

    if (exp_f16 == 0) {
        if (mant_f16 == 0) return @bitCast(sign);  // Zero
        // Denormal: convert to f32 denormal (simplified — full version needs normalization)
        const mant_f32 = mant_f16 << 13;
        const exp_f32 = (127 - 15) << 23;
        return @bitCast(sign | exp_f32 | mant_f32);
    }
    if (exp_f16 == 0x1F) {
        // Inf/NaN
        const exp_f32: u32 = 0xFF << 23;
        const mant_f32: u32 = mant_f16 << 13;
        return @bitCast(sign | exp_f32 | mant_f32);
    }
    // Normal: bias adjustment (f16 bias=15, f32 bias=127)
    const exp_f32: u32 = (exp_f16 + (127 - 15)) << 23;
    const mant_f32: u32 = mant_f16 << 13;
    return @bitCast(sign | exp_f32 | mant_f32);
}
```

**Source:** Adapted from `src/backend/kernels/cpu/gemv_q4_k.zig` to CUDA PTX

### Dual-Delta Numerical Test Pattern

```python
# Golden test with dual-delta validation
# Compares both CPU and GPU against FP64 oracle

import numpy as np

def dual_delta_test(cpu_output, gpu_output, oracle_fp64, name):
    """
    Validate GPU output using dual-delta criterion.

    GPU passes if: |GPU - Oracle| <= 2 × |CPU - Oracle|

    Rationale: FP32 arithmetic is not associative. Parallel GPU reduction
    accumulates errors differently than serial CPU reduction. Accept GPU
    if its error is within 2× the CPU baseline error.
    """
    cpu_err = np.abs(cpu_output - oracle_fp64)
    gpu_err = np.abs(gpu_output - oracle_fp64)

    max_cpu_err = np.max(cpu_err)
    max_gpu_err = np.max(gpu_err)

    passed = max_gpu_err <= 2.0 * max_cpu_err

    print(f"{name}:")
    print(f"  CPU max error: {max_cpu_err:.2e}")
    print(f"  GPU max error: {max_gpu_err:.2e}")
    print(f"  GPU/CPU ratio: {max_gpu_err / max_cpu_err:.2f}×")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")

    return passed

# Example usage:
# oracle = run_reference_fp64(input)
# cpu = run_agave_cpu(input)
# gpu = run_agave_metal(input)
# dual_delta_test(cpu, gpu, oracle, "Metal SDPA")
```

**Source:** Numerical testing best practices + CLAUDE.md §11 testing requirements

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Naive SDPA (3 passes) | FlashAttention-2 (tiled, online softmax) | 2023 (FA-2 paper) | 1.33× memory bandwidth savings, numerically stable |
| Pre-dequantize to f32 | In-kernel dequantization | Project inception | Zero f32 weight materialization, preserves quantization benefits |
| Serial softmax (CPU-only) | Warp-parallel softmax (GPU) | CUDA backend added | 32× parallelism for softmax (limited by sm_121 bug → 1× on Blackwell) |
| Blit encoders for KV append | Compute-only KV append | Gemma3 27B optimization | Eliminated 150ms/layer stalls on Metal |
| Simple LRU eviction | Frequency × cost metric | SGLang (2024) | Shared prefixes prioritized, better hit rate |

**Deprecated/outdated:**
- **Metal blit encoders in hot path:** Causes 150ms/layer stalls, replaced with compute-only path
- **CUDA block reductions on sm_121:** Hangs due to compiler bug, replaced with warp-only reductions
- **Single reference (llama.cpp only):** Insufficient for SafeTensors models, now dual-reference (llama.cpp + HF)

## Open Questions

1. **FlashAttention-2 block sizes for Metal**
   - What we know: Threadgroup memory ~32KB, must fit Q_block + K_block + V_block + scores + shared[]
   - What's unclear: Optimal (Bc, Br) for different head dimensions (128 vs 256), performance vs memory tradeoff
   - Recommendation: Start with conservative Bc=32, Br=32, benchmark with actual models, tune upward if memory allows

2. **CUDA FP8 conversion approach (intrinsics vs LUT)**
   - What we know: Ada/Hopper (sm_89+) have native `__nv_fp8_e4m3` intrinsics, Metal uses 256-entry LUT
   - What's unclear: Performance difference between intrinsics and LUT on sm_89+, whether to maintain dual paths
   - Recommendation: Start with LUT (works on all compute caps, proven in Metal), add intrinsics path later if benchmarks show significant gain

3. **Nemotron Nano MoE router exact fix**
   - What we know: Router scores overflow to -2.3e26, likely from accumulated quantization error in NVFP4 GEMV
   - What's unclear: Whether per-block scaling fixes it, or if router weights need different quantization (e.g., FP8 instead of NVFP4)
   - Recommendation: Implement per-block scaling first (matches Q4_K pattern), fall back to FP8 router weights if still unstable

4. **Golden test tolerance thresholds**
   - What we know: Dual-delta criterion (GPU error ≤ 2× CPU error), but exact epsilon for "close enough" unclear
   - What's unclear: Absolute epsilon for small values (when CPU error ≈ 0), relative epsilon for large values
   - Recommendation: Use relative epsilon 1e-5 for |oracle| > 1.0, absolute epsilon 1e-6 for |oracle| ≤ 1.0, verify with actual model outputs

5. **Model debugging priority order**
   - What we know: Nemotron Nano (30B) has highest value, GLM-4/GPT-OSS/Nemotron-H unverified
   - What's unclear: Whether to fix all models sequentially or parallelize (Nemotron Nano + GLM-4 in same wave)
   - Recommendation: Sequential — Nemotron Nano first (highest value), then GLM-4 (MLA is novel), then GPT-OSS (sliding window), then Nemotron-H (hybrid SSM)

## Validation Architecture

> Skipped — `workflow.nyquist_validation` is `false` in `.planning/config.json`.

## Sources

### Primary (HIGH confidence)

- [FlashAttention-2 Paper (2023)](https://arxiv.org/pdf/2307.08691) — Algorithm 3, tiling strategy, online softmax
- [FlashAttention Blog: Online Softmax](https://wangkuiyi.github.io/online-softmax.html) — Telescoping sum derivation
- [CUDA Math API: FP8 Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__FP8.html) — `__nv_fp8_e4m3`, `__nv_fp8_e5m2` constructors
- CLAUDE.md — Project standards, backend patterns, quantization rules
- Existing codebase:
  - `src/backend/kernels/cpu/gemv_q{4,5,6}_k.zig` — Dequant reference
  - `src/backend/kernels/cuda/common.zig` — PTX math, warp reductions
  - `src/backend/metal.zig` — UMA contract, page alignment, CPU fallback patterns
  - `src/ops/quant.zig` — FP8 LUT, bf16, NVFP4, MXFP4 helpers
  - `tests/harness.py` — Test orchestration framework

### Secondary (MEDIUM confidence)

- [Online Softmax Explanation](https://dev.to/lewis_won/online-softmax-by-hand-4h13) — Hand-worked example
- [K-Quants Overview](https://medium.com/@michael.hannecke/gguf-optimization-a-technical-deep-dive-for-practitioners-ce84c8987944) — Q4_K/Q5_K/Q6_K structure
- [Numerically Stable Softmax](https://jaykmody.com/blog/stable-softmax/) — max subtraction technique

### Tertiary (LOW confidence)

- [INT4 GEMV Optimization](https://pytorch.org/blog/int4-decoding/) — General quantized GEMV patterns (not CUDA-specific)
- llama.cpp repository — GGUF quantization reference (not directly cited, use as validation only)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — All dependencies already in use (Zig, PTX, SPIR-V, MSL)
- FlashAttention-2 algorithm: HIGH — Official paper + existing Metal SDPA as skeleton
- CUDA quantized GEMV: HIGH — CPU reference implementations exist, PTX patterns established
- Vulkan kernels: HIGH — 17 existing SPIR-V shaders as reference, GLSL straightforward
- Model debugging: MEDIUM — Root causes identified (MoE overflow, untested paths), but exact fixes need validation
- Numerical testing: HIGH — Dual-delta criterion well-defined, test harness exists

**Research date:** 2026-03-21
**Valid until:** 60 days (stable domain — GPU algorithms don't change rapidly, CUDA 13.1 is current)

---

*Research complete. Ready for planning.*
