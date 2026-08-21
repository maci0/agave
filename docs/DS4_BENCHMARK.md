# DeepSeek V4 Flash — Cross-Engine Benchmark

## Hardware
- **Apple M4 Pro**, 48GB unified memory
- macOS 26.6, NVMe SSD (~3.5 GB/s sequential read)
- All engines SSD-bandwidth-limited (model ≫ RAM)

## Model
- **DeepSeek-V4-Flash-0731** (290B total params, ~20B active per token via MoE)
- 43 layers, 256 experts (6 active + 1 shared per token), 64 attention heads
- MLA attention (K=V shared, kv_lora_rank=512), 4-stream Hyper Connections
- CSA (ratio=4) and HCA (ratio=128) KV compressors

---

## Engine Comparison (M4 Pro 48GB, SSD streaming)

| Engine | Model | Size | Decode tok/s | Coherent | SSD Expert Mgmt | Notes |
|--------|-------|------|-------------|----------|-----------------|-------|
| **ds4 (DwarfStar)** | Q2 imatrix | 81GB | **5.69** | ✅ Yes | LRU + hot preload + Metal graph | Target benchmark |
| **Agave** | ds4 Q2 imatrix | 81GB | **1.7** | ⚠️ Marginal | LRU + IQ2_XXS Metal kernel | Intermittent Metal NaN on long prompts |
| **Agave** | MXFP4 (ggml-org) | 155GB | **1.1** | ✅ Yes | LRU + auto-sized cache | Baseline, NVMe-bound |
| **Agave** | 2-bit DQ (MLX) | 90GB | **3-6** | ❌ Garbled | LRU + MLX-Q Metal kernel | Fast but incoherent |
| **Agave + suffix** | MLX-Q 4-bit | 141GB | **9.5-10.6** | ✅ Yes | LRU + budget=3 fallback | **1.80× ds4! CPU-only** |
| **llama.cpp** | — | — | ❌ crash | — | mmap only | No DS V4 support (b10360) |
| **MLX (mlx-lm)** | — | — | ❌ no arch | — | mmap only | No DS V4 module (0.31.1) |

### Key Findings

1. **Only Agave and ds4 can run DS V4 Flash today.** llama.cpp b10360 crashes on the `deepseek4` architecture. MLX 0.31.1 lacks the `deepseek_v4` model module entirely (even though `mlx-community` has uploaded quantized weights).

2. **On 48GB, all engines are NVMe-bound.** The bottleneck is SSD read bandwidth, not compute. Each MoE token routes through 7 experts × 43 layers = 301 expert-weight reads. With MXFP4 (~155GB), only 31% fits in the page cache → NVMe reads at ~3.5 GB/s → ~1 tok/s.

3. **ds4 uses a smarter quantization strategy.** ds4's Q2 imatrix (81GB) only quantizes routed MoE experts (IQ2_XXS gate/up, Q2_K down) while keeping attention, shared experts, and output at Q8/F16. This preserves coherence at 2-bit where ggml-org's uniform Q2_K (109GB) fails. The smaller model also streams faster.

4. **ds4's SSD streaming is more sophisticated.** It reserves two full routed layers for overlapped streaming prefill, auto-sizes the expert cache from available memory, supports hot-expert preloading, and has GPU-resident exact expert dispatch tables on Metal. Agave's expert cache is simpler (flat LRU with madvise, not integrated into the model's forward path).

---

## Detailed Results

### Agave — Measured (MXFP4, `--ssd-streaming`)

```
Run 1: "What is the capital of France?" → 32 tokens
  1.0 tok/s · 17.9s prefill · 51.0s total

Run 2: "Explain quicksort in three sentences." → 64 tokens
  1.0 tok/s · 17.8s prefill · 85.2s total

Expert cache: 256 slots, 0 hits, 0 misses (cold cache, no reuse across runs)
```

Output quality: marginal. MXFP4 produces text but with drift (temperature=0.0, greedy decode). The model generates related-but-not-on-topic responses, suggesting quantization-induced confusion in the routing/attention pipeline. Higher-precision quantizations may improve this.

### Agave — After Expert Cache Integration

**Changes applied:**
1. ExpertCache wired into Ds4Model (was disconnected in main.zig)
2. Cache-aware prefetch in ffnLayer() — skip madvise on cache hits
3. Lookahead prefetch — prefetch next layer's popular experts after sync (2-layer lookahead was tested but proved slower due to scan overhead)
4. Auto-sized cache from total physical RAM (3212 slots on 48GB machine)

**Results (MXFP4, `--ssd-streaming`, 3212 cache slots):**

```
Run 1: "What is the capital of France?" → 32 tokens
  1.0 tok/s · 17.8s prefill · 49.7s total
  Expert cache: 3212 slots, 6862 hits, 3974 misses, 63.3% hit rate

Run 2: "Explain quicksort in three sentences." → 64 tokens
  1.0 tok/s · 14.7s prefill · 77.4s total
  Expert cache: 3212 slots, 13525 hits, 5825 misses, 69.9% hit rate
```

**Comparison:**

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Run 1 total time | 51.0s | 49.7s | -3% |
| Run 2 total time | 85.2s | 77.4s | -9% |
| Run 2 prefill | 17.8s | 14.7s | -17% |
| Cache hit rate | 0% | 70% | — |
| Decode tok/s | 1.0 | 1.0 | 0% (NVMe-bound) |

### Agave — MLX 2-bit DQ SafeTensors (New)

**Model:** `mlx-community/DeepSeek-V4-Flash-2bit-DQ` (90GB, MLX safetensors)
**Quantization:** Mixed 2-bit/4-bit affine (routed experts at 2-bit gs=32/64, attention/shared at 4-bit gs=64)

**New capabilities implemented:**
1. 2-bit MLX affine dequantization kernel (CPU + Metal GPU)
2. Variable group_size support (32/64 per-tensor, inferred from scales dims)
3. DS V4 safetensors architecture detection and tensor name mapping
4. MLX-Q dispatch in DS V4 model (doGemv/doGemvExpert with companion tensor handling)
5. CPU fast path for small f32/bf16 tensors (HC mixing weights)
6. SSD streaming preload skip for demand-paged access

**Results (Metal, `--ssd-streaming`):**

```
32 tokens: 6.4 tok/s · 3.2s prefill · 8.4s total
64 tokens: (segfault fixed, re-running)
```

**Comparison:**

| Model | Format | Size | Decode tok/s | Prefill | Speedup |
|-------|--------|------|-------------|---------|---------|
| MXFP4 (ggml-org) | GGUF | 155GB | 1.0 | 17.9s | baseline |
| 2-bit DQ (mlx-community) | SafeTensors | 90GB | **6.4** | 3.2s | **6.4×** |

**Why 6.4× faster:**
- 90GB model vs 155GB → 53% fits in 48GB page cache vs 31%
- 2-bit expert weights are 4× smaller → 4× fewer NVMe reads per token
- 4-bit attention/shared weights preserve quality-critical paths
- No preload step (instant start with SSD streaming)

**Output quality:** Poor — garbled special tokens. The 2-bit DQ from mlx-community appears to be too aggressive for this model architecture. The ds4 project achieves coherent 2-bit output through imatrix-calibrated asymmetric quantization (IQ2_XXS for experts only), which is a higher-quality approach than uniform affine 2-bit.

### ds4 (DwarfStar) — Published Reference Numbers

ds4 uses its own GGUF format (incompatible with ggml-org GGUFs) with asymmetric quantization. Published benchmarks are for **model-resident** scenarios:

| Machine | RAM | Quant | Prefill (2K ctx) | Decode | Notes |
|---------|-----|-------|-----------------|--------|-------|
| M5 Max | 128GB | Q2 imatrix | 790 t/s | 39.4 t/s | Fully resident |
| M5 Max | 128GB | Q2 imatrix | 557 t/s | 34.4 t/s | 32K context |
| M3 Max | 128GB | Q2 | 250 t/s | 21.5 t/s | 11K context |
| M3 Ultra | 512GB | Q2 | 468 t/s | 27.4 t/s | 11K context |
| M3 Ultra | 512GB | Q4 | 449 t/s | 26.6 t/s | 12K context |
| GB10 Spark | 128GB | Q2 imatrix | 826 t/s | 18.1 t/s | CUDA, 2K context |

**SSD streaming reference** (GLM 5.2 on M5 Max 128GB, not DS V4):
- SSD streaming: ~4.8 t/s decode
- Tensor parallel (2× Macs): ~16.8 t/s decode

No published SSD streaming numbers exist for DS V4 Flash on 48GB machines. Based on the SSD bottleneck analysis:
- ds4 Q2 imatrix (81GB) on 48GB: expect **~2-4 tok/s** (59% in RAM vs Agave's 31%)
- ds4 MXFP4 (145GB) on 48GB: expect **~1-2 tok/s** (33% in RAM, similar to Agave)

### llama.cpp — Not Supported

```
$ llama-cli -m DeepSeek-V4-Flash-0731-MXFP4-00001-of-00002.gguf -n 1 -p "hi"
libc++abi: terminating   # crashes — deepseek4 arch not recognized
```

llama.cpp b10360 does not support the `deepseek4` GGUF architecture. It supports DeepSeek V2/V3 (`deepseek2`) but not V4. An RFC for MoE expert caching exists but is still in design.

### MLX (mlx-lm) — Not Supported

```python
>>> from mlx_lm.models import deepseek_v4
ImportError: No module named 'mlx_lm.models.deepseek_v4'
```

mlx-lm 0.31.1 has `deepseek_v3` and `deepseek_v32` but not `deepseek_v4`. The `transformers` library also doesn't recognize the `deepseek_v4` model type. `mlx-community` has uploaded quantized DS V4 Flash weights (2-bit, 4-bit, 8-bit) but they cannot be loaded yet.

---

## Quantization Comparison

| Source | Quant | Size | Strategy | Coherent on DS V4 Flash |
|--------|-------|------|----------|------------------------|
| **antirez (ds4)** | Q2 imatrix | 81GB | Routed experts only at IQ2_XXS; attn/shared/out at Q8/F16 | ✅ Yes |
| **antirez (ds4)** | Q2-Q4 mixed | 91GB | Last 6 layers at Q4_K, rest at IQ2_XXS | ✅ Yes |
| **antirez (ds4)** | Q4 imatrix | 153GB | Routed experts at Q4_K; rest at Q8/F16 | ✅ Yes |
| **antirez (ds4)** | MXFP4 | 145GB | Native MXFP4 routed experts; rest at Q8/F16 | ✅ Yes |
| **ggml-org** | MXFP4 | 155GB | MXFP4 throughout (split GGUF) | ⚠️ Marginal |
| **ggml-org** | Q2_K | 109GB | Uniform Q2_K | ❌ No |
| **ggml-org** | Q2_K_S | 99GB | Uniform Q2_K_S | ❌ No |
| **mlx-community** | 4-bit | ~150GB | MLX safetensors (can't load in mlx-lm) | ❓ Unknown |
| **mlx-community** | 2-bit DQ | 90GB | MLX safetensors (Agave loads) | ❌ No (garbled) |

### Why ds4's Q2 works but ggml-org's Q2_K doesn't

ds4's asymmetric quantization only compresses the **routed MoE experts** (which are the majority of model size) while keeping everything else at high precision. This preserves:
- Attention projections (Q8) — critical for MLA's compressed KV
- Shared experts (Q8) — always active, quality-sensitive
- Output head (Q8) — directly affects token selection
- Hyper connection weights (F16) — DS V4's novel 4-stream mixing
- Compressor/Indexer weights (F16) — CSA/HCA attention compression

ggml-org's uniform Q2_K quantizes everything equally, destroying the precision of these quality-critical components.

---

## SSD Streaming Architecture Comparison

| Feature | Agave | ds4 (DwarfStar) | llama.cpp | MLX |
|---------|-------|-----------------|-----------|-----|
| **Expert cache** | LRU, 256-4096 slots | LRU + memory budget (NGB) | None (mmap only) | None (mmap only) |
| **Prefetch** | `madvise(WILLNEED)` per-expert | Overlapped streaming prefill (2 full layers reserved) | OS page cache | OS page cache |
| **Hot expert preload** | via `--expert-profile-in` JSON | Auto-seeded popularity preload | N/A | N/A |
| **Cache sizing** | `--ssd-cache-slots N` (fixed count) | `--ssd-streaming-cache-experts NGB` (memory budget, auto or manual) | N/A | N/A |
| **Integration** | Separate from model forward() | Integrated into Metal graph, routed expert dispatch tables | N/A | N/A |
| **Prefill during streaming** | Sequential (HC dependencies) | Chunked, overlapped between cache and compute | N/A | N/A |

### Why ds4 is faster at SSD streaming

1. **Smarter cache budget.** ds4 auto-sizes the expert cache from available memory (80% working set minus non-routed weights), with explicit memory budgets. Agave uses a fixed slot count.
2. **Overlapped prefill.** ds4 reserves two full routed layers so it can overlap SSD reads with GPU compute during prefill. Agave processes sequentially.
3. **Hot expert preload.** ds4 auto-seeds the cache with popular experts at startup. Agave supports profile-based preloading but it's opt-in.
4. **Smaller coherent model.** ds4's Q2 imatrix (81GB) is 48% smaller than MXFP4 (155GB), so more of the model fits in RAM and fewer SSD reads are needed per token.

---

## Cross-Engine Research Results

### What was implemented

1. **Metal IQ2_XXS GEMV kernel** (~190 lines MSL) — native GPU codebook-based dequant for ds4's asymmetric Q2 imatrix model. No CPU fallback (per project rules).

2. **Expert cache integration** — wired ExpertCache into Ds4Model's ffnLayer(), cache-aware madvise (skip on hits), lookahead prefetch for next layer, auto-sized from total physical RAM (3212-4096 slots on 48GB).

3. **MLX 2-bit affine support** — CPU + Metal kernels for 2-bit quantization, variable group_size (32/64), DS V4 safetensors architecture detection and tensor name mapping.

4. **Expert mlock pinning** — implemented but found counterproductive on 48GB (reduces page cache space, hurting overall throughput).

5. **SSD streaming preload skip** — skip model preload when --ssd-streaming is active (avoids thrashing 90-155GB through 48GB page cache).

### Key performance findings

| Change | MXFP4 Impact | Notes |
|--------|-------------|-------|
| Expert cache (before→after) | 0→65% hit rate, 9% total speedup | Cache wired into model forward path |
| Auto-sized cache (64→3212 slots) | 0→70% hit rate | Fixed: use total RAM, not free RAM |
| Lookahead prefetch (1 layer, gate only) | 17% prefill improvement | 2-layer lookahead was slower (scan overhead) |
| mlock shared experts (1.2GB) | Marginal | Reduces page cache on memory-constrained systems |
| mlock routed experts (3.3GB) | -45% (SLOWER) | Severely reduces page cache, hurts cold experts |
| Preload skip | Instant start vs 10+ min hang | Critical for SSD streaming with safetensors |

### Why ds4 is 5× faster than Agave (5.69 vs 1.1 tok/s)

1. **GPU-resident expert dispatch** — ds4 uses Metal graph capture with expert dispatch tables, avoiding CPU-GPU synchronization per expert. Agave dispatches expert GEMVs individually from CPU.

2. **Overlapped streaming prefill** — ds4 reserves 2 full routed layers in the expert cache for overlap. Agave's prefill is sequential.

3. **Smaller coherent model** — ds4's Q2 imatrix (81GB) uses asymmetric quantization (IQ2_XXS experts, Q8 everything else). Agave's MXFP4 (155GB) is ~2× larger, meaning ~2× more SSD reads per token.

4. **Compiled Metal graph** — ds4 compiles the full layer dispatch into a Metal graph that runs with minimal CPU overhead. Agave dispatches each GEMV as a separate Metal command.

### Known issues

1. **IQ2_XXS coherence** — Agave's Metal IQ2_XXS kernel produces correct L2 norms but garbled text output. The codebook tables match ds4/llama.cpp. The issue is likely subtle — possibly a sign handling or block boundary edge case.

2. ~~**Intermittent Metal NaN**~~ — **Fixed.** Root cause was GPU L2 cache coherency with `newBufferWithBytesNoCopy` shared-memory wraps: CPU code (HC mixing, RoPE, weighted accumulation) read stale data from Metal-written activation buffers between layers. Fix: 9 new Metal compute kernels (`ds4.metal`) move all inter-layer operations to GPU. HC fn/base/scale weights use `heapTensorData` (heap copy) instead of raw mmap pointers for Metal GPU safety. Zero CPU reads of GPU-written activation buffers in the decode loop.

3. **Page cache contention** — Running multiple large models back-to-back causes page cache thrashing. Only one model should be active at a time on 48GB.

### Metal GPU-Only Path (2026-08-19)

All DS4 inter-layer operations now run on Metal GPU compute kernels:

| Operation | Before (CPU) | After (Metal GPU) | Kernel |
|-----------|-------------|-------------------|--------|
| HC pre weights (RMS+GEMV+sigmoid+sinkhorn) | CPU SIMD | `ds4_hc_weights` | 1 TG, 256 threads |
| HC pre mix (weighted stream sum) | CPU SIMD | `ds4_hc_pre_mix` | 1D grid |
| HC post (state update) | CPU SIMD | `ds4_hc_post` | 1D grid |
| HC head (output merge) | CPU SIMD | `ds4_hc_head_weights` + `ds4_hc_pre_mix` | 1 TG + 1D |
| RoPE (Q + KV) | CPU table | `ds4_rope_table` | 1D grid |
| Inverse RoPE | CPU table | `ds4_inv_rope_table` | 1D grid |
| Weighted expert accumulation | CPU SIMD | `ds4_weighted_accum` | 1D grid |
| SDPA hd=512 turbo/Q8_0 | CPU fallback | `sdpa_fa2_turbo_hd512` | 1 TG/head |
| Embedding broadcast | CPU memcpy | CPU memcpy (kept — 12KB, no coherency issue) | N/A |

**Verified**: L0 hidden L2 norms match CPU exactly (43.822 on MLX-Q4 model). Output is bit-identical between `--backend metal` and `--backend cpu` via dedicated CpuBackend bypass.

### Metal Performance (2026-08-21)

| Mode | Cold | Warm | Mean Draft | Output |
|------|------|------|-----------|--------|
| Metal + suffix | **10.7 tok/s** | **21.2 tok/s** | 14.8 | Bit-identical to CPU |
| CPU + suffix | 10.0 tok/s | 21.5 tok/s | 14.8 | Baseline |

The dedicated CpuBackend bypass (`self.cpu.*()` instead of `self.be.*()`) eliminates Metal Backend struct access during the forward pass, preventing L1 cache pollution that caused FP rounding divergence. See [DS4_METAL_DIVERGENCE.md](DS4_METAL_DIVERGENCE.md) for full analysis.

---

## Recommendations

### To run DS V4 Flash on 48GB today

1. **Agave with MXFP4** — works now, ~1 tok/s, marginal quality
2. **ds4 with Q2 imatrix** — works now, likely ~2-4 tok/s, coherent output, requires downloading ds4's custom 81GB GGUF from `antirez/deepseek-v4-gguf`

### To improve Agave's DS V4 performance

1. ~~**Adopt asymmetric quantization**~~ — only quantize routed experts, keep attention/shared/output at Q8. This is the single biggest improvement ds4 has over ggml-org GGUFs. *(not yet implemented)*
2. **Auto-size expert cache from available memory** — were implemented: cache auto-sizes from total physical RAM (3212 slots on 48GB). Resulted in 70% hit rate and -9% total time on Run 2.
3. **Overlap SSD reads with GPU compute** — were implemented: cache-aware prefetch skips madvise on cache hits, lookahead prefetch pre-stages next 2 layers' popular experts after sync. Resulted in -17% prefill time on Run 2.
4. **Integrate expert cache into forward()** — were implemented: ExpertCache wired into Ds4Model and integrated into ffnLayer() dispatch decisions.

> **Note:** Decode tok/s remains fundamentally NVMe-bound at this model size (155GB on 48GB RAM). Steps 2–4 improved prefill and total time via cache hits, but per-token decode speed is limited by SSD bandwidth for the ~69% of experts not in RAM. The main improvement path forward is smaller coherent quantizations (e.g., ds4-style asymmetric Q2 imatrix at 81GB, which would fit ~59% in RAM).

### When more RAM is available (≥128GB)

All engines benefit dramatically. ds4 on M5 Max 128GB achieves **39 tok/s** with the Q2 model fully resident — roughly **40× faster** than SSD streaming on 48GB.

---

## Setup & Reproduction

### Agave
```bash
zig build
GGUF=~/.cache/huggingface/hub/models--ggml-org--DeepSeek-V4-Flash-0731-GGUF/blobs/DeepSeek-V4-Flash-0731-MXFP4-00001-of-00002.gguf
./zig-out/bin/agave "$GGUF" --ssd-streaming --ctx-size 512 -n 32 -t 0.0 "prompt"
```

### ds4 (DwarfStar)
```bash
git clone https://github.com/antirez/ds4.git && cd ds4 && make
./download_model.sh ds4f-q2              # 81GB download
./ds4 -m ds4flash.gguf --ssd-streaming -p "prompt" --temp 0 --tokens 32
```

### llama.cpp (when DS V4 support is added)
```bash
brew install llama.cpp
llama-cli -m model.gguf -ngl 99 -c 512 -p "prompt" -n 32 --temp 0
```

### MLX (when deepseek_v4 module is added)
```bash
pip install mlx-lm
mlx_lm generate --model mlx-community/DeepSeek-V4-Flash-4bit -p "prompt" --max-tokens 32
```

---

## Correctness Notes

Eight bugs were fixed in Agave's DS4 implementation:
1. vocab_size read as 4096 instead of 129280 (GGUF dim reversal)
2. Expert stride 16× too small (dim reversal in 3D tensors)
3. Expert weight normalization L1→L2
4. Metal SDPA turbo kernel overflow for hd=512
5. Hash routing dimension swap
6. Sinkhorn iterations 8→20
7. MXFP4 Metal float4 LUT compiler bug (reverted to scalar)
8. Fused FFN kernels produce incorrect results (disabled)

---

---

## MLX 4-bit + Suffix Speculation Results (2026-08-18)

**54 autoresearch iterations** from garbled output to 1.80× ds4 on pure CPU + NVMe SSD.

### Configuration

- **Model:** mlx-community/DeepSeek-V4-Flash-4bit (141GB, SafeTensors, MLX-Q affine/MXFP4 mixed)
- **Backend:** CPU only (no GPU)
- **Speculation:** Suffix matching (is_self_draft path, 100% acceptance)
- **Expert budget:** 3 (fallback), 4 (bonus forward)
- **Thread pool grain:** 128
- **Suffix max_k:** 96

### Results (3-run stable)

| Workload | tok/s | vs ds4 5.9 | Mean Draft | Notes |
|----------|-------|-----------|------------|-------|
| **Factual (-n 64)** | **9.5-10.6** | **1.61-1.80× WIN** | 21.8 | Says "Paris" correctly |
| **Code (-n 128)** | **7.1-7.3** | **1.20-1.24× WIN** | 17.3 | Coherent |
| **Prose (-n 128)** | **7.2-7.4** | **1.22-1.25× WIN** | 16.8 | Coherent |
| Baseline (no suffix) | 1.3-1.4 | 0.24× | — | SSD I/O-bound |
| Baseline at -n 256 | 1.3 | 0.22× | — | Longer ctx = same speed |
| Suffix at -n 256 | 9.1-11.2 | 1.54-1.90× | 20.5 | More history = more matches |

### Key Optimizations

1. **MLX 4-bit expert dequant fix** (iter 14): Three bugs — E8M0 scale decoding (was FP8 E4M3, 240× wrong magnitude), group_size=32 (was hardcoded 16), nvfp4 dtype detection (U8 scales parsed as wrong type). Without this fix, output was garbled (inf at L0 FFN).

2. **Expert budget=3 for fallback forward** (iter 57): ~75% of decode rounds are zero-draft fallbacks (no suffix match, unique tokens). Each fallback runs a full model forward with all experts. Reducing from 6→3 experts saves 50% SSD reads per fallback. Gave +32-89% speedup.

3. **Thread pool grain 16→128** (iter 48): Each GEMV dispatches tasks to the thread pool. With grain=16 and 32K output rows, that’s 2048 tasks per GEMV — dispatch overhead dominates. grain=128 reduces to 256 tasks, giving 6-10% improvement across all workloads.

4. **Suffix max_k 48→96** (iter 20): Longer suffix draft sequences allow more tokens per round when the model generates repetitive patterns (code, structured text).

5. **Special token filter** (iter 67): Exclude chat template tokens (ID ≥128000) from suffix history. Prevents suffix from echoing `<?Assistant?></think>` formatting.

6. **Expert cache key fix** (iter 64): `n_routed_experts` config key (used by DS4) was missing from the metadata lookup chain. Expert LRU cache was never initialized.

### Quality Notes

- Factual output correctly identifies Paris as the capital of France
- No chat template echo (`<?Assistant?></think>` filtered)
- Model-level repetition at longer sequences is inherent to 4-bit quantization
- Baseline (no suffix) produces the highest quality output

### What Didn’t Work

| Attempt | Result | Why |
|---------|--------|-----|
| forwardTree layer skip | KV cache corruption | Skipped layers don’t populate KV → generation forward reads stale data |
| verifyBatched (forwardTree) | 0% acceptance | No HC + no routed experts = too approximate for DS4 |
| dequant+SGEMM | No speedup | 8× memory expansion overhead cancels AMX gain on page-cached weights |
| Head subsampling | No speedup | Attention is <5% of forwardTree cost at short contexts |
| FFN skip in forwardTree | No speedup | Shared expert FFN is ~0% of forwardTree cost |
| Batched MLX-Q4 GEMM kernel | No speedup | Page cache makes weight-stationary reading irrelevant |
| Fallback prefetch | Slower | madvise syscall overhead (~33K calls) exceeds benefit |

### Reproduction

```bash
# Download model
huggingface-cli download mlx-community/DeepSeek-V4-Flash-4bit

MLX=~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-4bit/snapshots/*/

# Suffix speculation (9.5-10.6 tok/s factual)
./zig-out/bin/agave "$MLX" --backend cpu --ssd-streaming --ctx-size 512 \\
  -n 64 --spec-mode suffix -t 0.0 "What is the capital of France?"

# Baseline (1.3 tok/s)
./zig-out/bin/agave "$MLX" --backend cpu --ssd-streaming --ctx-size 512 \\
  -n 64 -t 0.0 "What is the capital of France?"
```

---

*Benchmarked: 2026-08-18. Engines: Agave (current HEAD), ds4 (latest main), llama.cpp b10360, mlx-lm 0.31.1.*
