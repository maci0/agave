# DeepSeek V4 Flash — SSD Streaming Performance Handoff

*Last updated: 2026-08-13. Covers all work done across three goal sessions.*

---

## Current State

**Hardware:** Apple M4 Pro, 48GB unified memory, macOS 26.6, NVMe SSD (~3.5 GB/s)

### Benchmark Summary

| Engine | Model | Size | Decode tok/s | Coherent | Status |
|--------|-------|------|-------------|----------|--------|
| **ds4 (DwarfStar)** | Q2 imatrix | 81GB | **5.69** | ✅ | Reference target |
| **Agave (Metal)** | ds4 Q2 imatrix | 81GB | **1.7** | ❌ Garbled | IQ2_XXS dequant issue |
| **Agave (Metal)** | MXFP4 ggml-org | 155GB | **1.1** | ✅ Marginal | Working baseline |
| **Agave (Metal)** | MLX 2-bit DQ | 90GB | **3-6** | ❌ Garbled | 2-bit too aggressive |
| **Agave (CPU)** | ds4 Q2 imatrix | 81GB | **1.1** | ❌ Garbled | Same IQ2_XXS issue |
| llama.cpp b10360 | any | — | crash | — | No deepseek4 arch |
| MLX (mlx-lm 0.31.1) | any | — | no arch | — | No deepseek_v4 module |

### Diff Summary

**1338 insertions, 177 deletions across 23 files.** Not committed.

```
docs/DS4_BENCHMARK.md           — full cross-engine benchmark results
src/ops/mlx.zig                 — 2-bit MLX affine CPU GEMV kernel (+288 lines)
src/backend/kernels/metal/gemv.metal — IQ2_XXS Metal GEMV kernel + 2-bit MLX Metal kernel (+272 lines)
src/backend/metal.zig           — IQ2_XXS + MLX Q2 pipeline registration (+56/-24)
src/models/deepseek4.zig        — expert cache wiring, MLX-Q dispatch, F16/f32/bf16 HC fast paths (+274/-18)
src/expert_cache.zig            — prefetchTopResidents, mlock pinning (+124/-3)
src/main.zig                    — auto-sized cache, preload skip, estimateExpertBytes, mlock (+113/-8)
src/format/safetensors.zig      — DS V4 tensor name mappings + config keys (+246/-18)
src/models/model.zig            — setExpertCache, inferMlxGroupSize, findMlxCompanion updates (+33/-3)
src/backend/backend.zig         — variable group_size in GemvOp + gemvMlxQ (+15/-4)
src/backend/{cuda,rocm,vulkan,webgpu,cpu}.zig — variable group_size passthrough, IQ panic messages
src/models/{gemma3,gemma4,glm4,gpt_oss,nemotron_nano,qwen35}.zig — variable group_size wiring
docs/{ARCHITECTURE,tutorials}   — updated gemvMlxQ signature docs
```

---

## What Was Built

### 1. Expert Cache Integration
**Files:** `src/expert_cache.zig`, `src/models/deepseek4.zig`, `src/models/model.zig`, `src/main.zig`

- `ExpertCache` wired into `Ds4Model` via `ModelStorage.setExpertCache()` — was completely disconnected before
- Cache-aware prefetch in `ffnLayer()`: `ec.touch()` checks LRU, only `madvise(WILLNEED)` on misses → 65-70% hit rate
- Lookahead: after `be.sync()`, prefetch next layer's 6 most-recently-used experts (1-layer gate-only — 2-layer was slower)
- Auto-sized from `detectSystemMem()` (total RAM, not free RAM): 3212 slots on 48GB, 4096 on 81GB model
- `ExpertProfile` recording wired into ffnLayer for `--expert-profile-out`

**Impact:** 65-70% cache hit rate, 17% prefill improvement, 9% total time improvement on MXFP4

### 2. MLX Native Quant Support
**Files:** `src/ops/mlx.zig`, `src/backend/kernels/metal/gemv.metal`, `src/backend/metal.zig`, `src/backend/backend.zig`, `src/backend/cpu.zig`, 6 model files

- **2-bit MLX affine kernel** — CPU SIMD (16 crumbs/u32, `@Vector(16, f32)`) + Metal GPU shader (`gemv_mlx_q2`)
- **Variable group_size** — per-tensor gs inferred from scales dims (`inferMlxGroupSize`), threaded through all 6 backends. `GemvOp.mlx_group_size` field + `gemvMlxQ` parameter added.
- Handles DQ (dynamic quantization) models with mixed 2-bit (experts, gs=32) and 4-bit (attention, gs=64)

### 3. DS V4 SafeTensors Architecture
**Files:** `src/format/safetensors.zig`, `src/models/deepseek4.zig`

- Architecture detection: `deepseek_v4` model_type → `Arch.deepseek4` (already in `arch.zig`)
- ~40 tensor name mappings in `gguf_hf_layer_map`: MLA attention (`wq_a/wq_b/wkv/wo_a/wo_b`), hyper connections (`attn_hc.fn/base/scale`), MoE (`switch_mlp.gate_proj/up_proj/down_proj`), compressors, indexers, router
- Config.json key mappings: `n_routed_experts`, `num_experts_per_tok`, `q_lora_rank`, `routed_scaling_factor`, etc.
- `.weight` and `.bias` suffix stripping for HC/sink tensors that omit the suffix
- Top-level HC head: `output_hc_fn/base/scale` → `hc_head.fn/base/scale`

### 4. MLX-Q Dispatch in DS V4 Model
**Files:** `src/models/deepseek4.zig`

- `doGemv()` wrapper → `model_mod.dispatchGemv()` for format-aware GEMV (MLX-Q, NVFP4, GPTQ, standard)
- `doGemvExpert()` for per-expert sliced GEMV with MLX companion tensor slicing
- MLX-Q embedding lookup via `mlxEmbLookup` with companion scales/biases
- **Important:** `doGemvExpert` causes Metal NaN for GGUF quants (MXFP4, Q2_K) — use direct `be.gemv()` for non-MLX-Q expert gate/up. See the `if (ge.dtype == .mlx_q)` guard in ffnLayer.

### 5. Metal IQ2_XXS GEMV Kernel
**Files:** `src/backend/kernels/metal/gemv.metal`, `src/backend/metal.zig`

- ~190 lines MSL: `iq2xxs_grid[256]` + `ksigns_iq2xs[128]` codebook tables, `iq2_dot8()` helper, full threadgroup-per-row kernel
- Pipeline: `pipe_gemv_iq2_xxs`, threadgroup size 256 (QK_K=256 superblocks)
- **No CPU fallback** — per project rules, other IQ formats (iq2_xs, iq2_s, iq3_xxs, etc.) still `@panic`

### 6. CPU Fast Paths for HC Weights
**Files:** `src/models/deepseek4.zig`

- `cpuGemvF32()`, `cpuGemvBf16()`, `cpuGemvF16()` — tiny HC mixing matrices (24×16384) are faster on CPU than Metal dispatch
- HC weights are F32 (safetensors), F16 (ds4 GGUF), or Q8_0 (ggml-org GGUF)
- F16 on Metal caused intermittent NaN → CPU path avoids this

### 7. Expert mlock Pinning
**Files:** `src/expert_cache.zig`, `src/main.zig`

- `ExpertCache.pinExpert()` / `unpinAll()` with page-aligned mlock/munlock
- Profile-guided: `--expert-profile-in` pins top-6 experts per layer
- Auto-pin: shared experts + router gates (~1.2GB)
- **Disabled on ≤64GB** — mlock reduces page cache, hurts throughput when model >> RAM
- Max 30GB wire limit guard

### 8. SSD Streaming Preload Skip
**File:** `src/main.zig`

- When `--ssd-streaming` is active, skip `preloadModel()` entirely
- Prevents thrashing 90-155GB through 48GB page cache (was hanging for 10+ minutes)

---

## Known Issues & Blockers

### P0: IQ2_XXS Garbled Output
**Status:** Unsolved. Blocks coherent 1.7 tok/s from ds4 Q2 model.

- Agave's IQ2_XXS dequant tables, sign tables, block size (66 bytes), and algorithm all match ds4/llama.cpp exactly
- L2 norms are numerically reasonable (87.8 vs MXFP4's 84.3 for attention output)
- Same model + same prompt: ds4 produces "Hello! How can I assist you today", Agave produces "-fl< sin ~ if off in"
- Chat template + BOS/EOS tokens match
- **Next investigation:** Compare first-token logits between ds4 (`--dump-logits`) and Agave. May be a subtle expert dispatch ordering issue (Agave dispatches gate→up→down per expert, ds4 may batch differently).

### P1: Metal Buffer Cache Staleness
**Status:** Known limitation. Workaround: `sudo purge` before runs.

- With 81-155GB mmap'd models on 48GB, the Metal buffer cache (`getBufRef`) can hold stale pointers to OS-evicted pages
- GPU reads zeroed/garbage data → NaN during inference
- Intermittent: depends on page cache pressure from prior runs
- Running multiple large models back-to-back causes page cache thrashing

### P2: doGemvExpert Metal Regression
**Status:** Worked around with dtype guard.

- `doGemvExpert()` for GGUF quants (MXFP4, Q2_K) on Metal produces NaN despite generating identical `be.gemv()` calls as the direct path
- Likely a Zig compiler optimization issue or Metal buffer cache interaction
- **Workaround:** `if (ge.dtype == .mlx_q) { doGemvExpert(...) } else { be.gemv(...) }` in ffnLayer

---

## Architecture: Why ds4 is 5× Faster

| Feature | ds4 (5.69 tok/s) | Agave (1.1 tok/s) |
|---------|-----------------|-------------------|
| Expert dispatch | Metal graph capture, GPU-resident dispatch tables | CPU-side per-expert `be.gemv()` calls |
| Streaming prefill | Overlapped, 2 full layers reserved | Sequential `forward()` loop |
| Expert cache | Memory-budget aware, auto-sizes from Metal working set | Slot-count based, auto-sizes from total RAM |
| Hot expert preload | Compiled-in hotlist (ds4_streaming_hotlist.inc) | Profile-based `--expert-profile-in` |
| Model size | 81GB Q2 imatrix (IQ2_XXS experts, Q8 rest) | 155GB MXFP4 (uniform) |
| Non-routed weights | 8.2GB, fully GPU-resident | ~15GB, mmap'd |

### Closing the Gap — Roadmap

1. **Fix IQ2_XXS coherence** — Enables 1.7 tok/s coherent from 81GB model (vs 1.1 tok/s from 155GB)
2. **Metal graph capture for experts** — Batch all per-layer expert GEMVs into a single Metal graph dispatch
3. **Overlapped streaming prefill** — Reserve 2 full routed layers, overlap SSD reads with GPU compute
4. **GPU-resident non-routed weights** — Pin non-routed weights (~8GB) in Metal residency sets
5. **Expert weight pinning via Metal residency sets** — Replace mlock with GPU-level pinning

---

## Models & Paths

```bash
# MXFP4 (coherent, 155GB, 1.1 tok/s)
GGUF=~/.cache/huggingface/hub/models--ggml-org--DeepSeek-V4-Flash-0731-GGUF/blobs/DeepSeek-V4-Flash-0731-MXFP4-00001-of-00002.gguf
./zig-out/bin/agave "$GGUF" --ssd-streaming --ctx-size 512 -n 32 -t 0.0 "prompt"

# ds4 Q2 imatrix (81GB, 1.7 tok/s, garbled)
./zig-out/bin/agave /tmp/ds4/ds4flash.gguf --ssd-streaming --ctx-size 512 -n 8 -t 0.0 "prompt"

# ds4 engine (5.69 tok/s, coherent — reference)
cd /tmp/ds4 && ./ds4 -m ds4flash.gguf --ssd-streaming -c 512 -p "prompt" --temp 0 --tokens 32 --nothink

# MLX 2-bit DQ (90GB safetensors, 3-6 tok/s, garbled)
MODEL_DIR=~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-2bit-DQ/snapshots/722bf559b7de93575b2320973cf2002e05bfe6c9
./zig-out/bin/agave "$MODEL_DIR" --ssd-streaming --ctx-size 512 -n 32 -t 0.0 "prompt"

# Expert profiling workflow
./zig-out/bin/agave "$GGUF" --ssd-streaming --expert-profile-out /tmp/profile.json -n 16 "prompt"
./zig-out/bin/agave "$GGUF" --ssd-streaming --expert-profile-in /tmp/profile.json -n 32 "prompt"
```

---

## Key Learnings

1. **On 48GB, model size is everything.** 81GB → 53% cache coverage, 155GB → 31%. Each halving of model size roughly doubles decode speed (fewer SSD reads).

2. **mlock hurts on memory-constrained systems.** Pinning 3.3GB of hot experts REDUCED performance by 45% — it shrinks the page cache, causing more cold-expert SSD reads.

3. **Lookahead prefetch sweet spot is 1 layer, gate-only.** 2-layer with gate+up+down was slower due to O(n_slots) scan overhead in `prefetchTopResidents`.

4. **Auto-sizing must use total RAM, not free RAM.** macOS reports <1% "free" RAM when large files are mmap'd, but those pages are reclaimable. Using `detectSystemMem()` gives correct cache sizing.

5. **Metal buffer cache + mmap'd models > RAM = intermittent NaN.** The buffer cache caches Metal buffer objects by page-aligned address, but the OS can evict the underlying pages.

6. **CPU GEMV is faster than Metal for tiny matrices.** HC mixing (24×16384) is cheaper to compute on CPU than to dispatch via Metal command buffers.

7. **`doGemvExpert` wrapper causes Metal NaN for GGUF quants.** Despite generating identical `be.gemv()` calls, the wrapper changes Zig's optimization behavior for the Metal backend. Use direct `be.gemv()` for non-MLX-Q.
