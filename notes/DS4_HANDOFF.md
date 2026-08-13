# DeepSeek V4 Flash — Performance & MTP Handoff

*Last updated: 2026-08-14. Covers all work from the autoresearch loop (14 iterations).*

---

## Current State

**Hardware:** Apple M4 Pro, 48GB unified memory, macOS 26.6, NVMe SSD (~3.5 GB/s)

### Benchmark Summary

| Engine | Model | Size | Decode tok/s | Coherent | Status |
|--------|-------|------|-------------|----------|--------|
| **Agave (CPU)** | MXFP4 ggml-org | 155GB | **1.0–1.3** | ✅ Yes | **Best reliable path** |
| **Agave (CPU+MTP)** | MXFP4 + MTP safetensors | 155+0.6GB | **1.0–1.1** | ✅ Yes | MTP pipeline works, 0% accept |
| Agave (Metal) | MXFP4 ggml-org | 155GB | 0.8–1.1 | ⚠️ Intermittent NaN | volatile_weights partial fix |
| Agave (CPU) | ds4 Q2 imatrix | 81GB | 1.7 | ❌ Garbled | IQ2_XXS HC amplification |
| Agave (Metal) | ds4 Q2 imatrix | 81GB | 1.4–2.0 | ❌ Garbled | Same + GPU page fault risk |
| **ds4 (DwarfStar)** | Q2 imatrix | 81GB | **5.69–7.32** | ✅ Yes | Reference target |
| llama.cpp b10360 | any | — | crash | — | No deepseek4 arch |
| MLX (mlx-lm 0.31.1) | any | — | no arch | — | No deepseek_v4 module |

### Branch: `autoresearch/ds4-perf`

**3,000+ insertions across 30+ files.** Key commits:
```
125e37b  MTP attention + MXFP8 fix for all tensors
4bdf7fb  MTP shared expert FFN + MXFP8 GEMV
49cd77b  MTP pipeline working end-to-end
f8e665c  Discover DS V4 Flash MTP/DSpark weights in HF
e3aa1a3  stable_cache for Metal analysis
09245de  Metal page fault fix, IQ2_XXS analysis
```

---

## What Was Built (This Session)

### 1. Metal Buffer Cache Reliability
**Files:** `src/backend/metal.zig`, `src/backend/backend.zig`, `src/main.zig`

- `volatile_weights` mode: flushes Metal buffer cache on `sync()` when `--ssd-streaming` active
- `stable_cache`: persistent Metal-managed buffer copies (partial fix for page eviction)
- `flushBufferCache()`: explicit cache clear on volatile mode init
- `setVolatileWeights()`: Backend vtable dispatcher

**Root cause found:** Metal's `newBufferWithBytesNoCopy` does NOT trigger GPU page faults for evicted file-backed mmap pages on Apple Silicon. When model >> RAM, the GPU reads zeroed/evicted pages → NaN. Five approaches were tested:

| Approach | Result |
|----------|--------|
| buf_cache flush on sync() | ~66% reliable |
| CPU pre-fault before dispatch | ~50% reliable |
| copy_cache (makeBuffer) | Fails: DMA reads stale pages too |
| Pre-fault + makeBuffer | Worse (race condition) |
| stable_cache (persistent copies) | ~66% reliable |

**Conclusion:** Only pread()-based loading (like ds4) is fully reliable for Metal SSD streaming. CPU backend is the recommended path for model >> RAM.

### 2. IQ2_XXS Coherence Analysis
**Files:** autoresearch.jsonl, notes/DS4_HANDOFF.md

- IQ2_XXS dequant kernel verified correct (codebook + signs match ds4/llama.cpp exactly)
- Sinkhorn HC implementation verified identical to ds4 (max diff 6e-8)
- Tokenization verified identical (11 tokens, same IDs)
- **Root cause:** 2-bit quantization noise cascades through 43 HC layers
  - L0 FFN: 10% error vs MXFP4 → L1: 30× divergence → L43: r=0.02 (random)
  - Logit correlation: MXFP4 r=0.65 vs reference, IQ2_XXS r=0.02
- ds4 engine achieves coherent output from same model (likely GPU-graph-level precision)

### 3. MXFP8 GEMV Kernel
**Files:** `src/backend/kernels/cpu/gemv_fp8.zig`

New `gemvMXFP8` kernel for FP8 E4M3 weights with E8M0 per-tile block scales:
- Per-tile scaled dot product with configurable group_size (128)
- Used for all MTP weight tensors (main_proj, attention, FFN)
- CPU-only (not dispatched through backend vtable — direct call from MTP path)

### 4. SIMD MXFP4 GEMV
**Files:** `src/backend/kernels/cpu/gemv_fp4.zig`, `src/backend/kernels/cpu/gemv.zig`

`gemvMXFP4_V`: @Vector(4,f32) inner loop with @mulAdd FMA. Processes 4 weight bytes (8 values) per iteration. Correct output verified. No measurable speedup (bottleneck is SSD reads, not dequant).

### 5. MTP / Multi-Token Prediction (Major Feature)
**Files:** `src/models/ds4_mtp.zig`, `src/models/deepseek4.zig`, `src/models/model.zig`, `src/main.zig`, `src/spec/spec_decode.zig`

#### Discovery
DS V4 Flash HF safetensors contain **3 full MTP decoder layers** (4,705 tensors) that ALL GGUF quantizers stripped. Each MTP layer has:
- Full MLA attention (wq_a/b, wkv, wo_a/b) in FP8 E4M3
- Full MoE FFN (256 routed experts + shared) — experts in MXFP4, shared in FP8
- Hyper connections (F32)
- DSpark heads on mtp.2: confidence_head [1,4352], markov_head W1/W2 [129280,256] (BF16)

#### Implementation

**MTP weight loader** (`ds4_mtp.zig`):
- Loads safetensors file via mmap + std.json header parsing
- 97 non-expert tensors (595MB) extracted to `/tmp/ds4_mtp_weights.safetensors`
- Tensor lookup by HF name (e.g., `mtp.0.main_proj.weight`)

**MTP forward** (`deepseek4.zig`):
```
mtpForward(token_id, depth):
  1. Build input: concat(target_hidden[4096], prev_mtp_hidden[4096], embed(token)[4096])
  2. main_proj MXFP8 GEMV: [4096, 12288] @ input → hidden2
  3. main_norm: RMS norm on hidden2 (BF16 weights)
  4. Single-token attention: attn_norm → kv_proj → kv_norm → (score=1) → wo_a → wo_b
  5. Shared expert FFN: ffn_norm → gate+up → silu_mul → down (all MXFP8)
  6. Output: MTP-specific norm → shared lm_head → argmax
  Fallback: Markov head (BF16) for bigram-biased draft
```

**Pipeline integration:**
- `--mtp-model` CLI flag loads MTP safetensors
- `n_mtp_layers` field drives `getMtpDepth()` vtable dispatch
- `draftMtp()` in spec_decode.zig chains draft tokens across depths
- Hidden state isolation (hidden2 + mtp_hidden_buf) prevents target corruption
- Markov head adds transition bias to target logits for fallback drafting

**Current status:** 0% acceptance rate. Single-token attention (no KV cache) produces weak drafts. Full MTP KV cache needed for >0% acceptance.

### 6. Expert Cache Profiling
**Files:** `src/expert_cache.zig`, autoresearch.jsonl

- ~51 unique experts per layer (out of 256) for typical prompts
- Cache hit rate: 52% (cold) → 73% (warm, 64 tokens)
- At warm cache: system is ~70% compute-bound, ~30% SSD-bound
- Expert usage is relatively uniform — no extreme hot/cold split
- 3212 cache slots on 48GB (auto-sized from total RAM)

### 7. DSpark / Qwen3 Analysis
**Files:** notes/MTP_DESIGN.md

- DSpark weights in `~/Models/` are for Qwen3 8B (5-layer backbone + confidence/markov heads)
- Agave's `src/spec/dspark.zig` has MarkovHead, RnnHead, ConfidenceHead, SPS scheduler ready
- DS V4 Flash config has DSpark params: block_size=5, markov_rank=256, target_layers=[40,41,42]
- Cannot use Qwen3 DSpark weights for DS V4 (different architecture)

---

## Known Issues & Blockers

### P0: MTP KV Cache (Blocks Speculative Decoding Speedup)
**Status:** Architecturally understood, not yet implemented.

MTP layers need their own KV cache to do full attention over conversation context:
1. Allocate per-MTP-layer KV storage (3 layers × 512 dims × max_seq_len)
2. During MTP forward, append MTP-projected KV to MTP cache
3. Run full SDPA attention against MTP KV cache
4. On speculation rejection, roll back MTP KV cache positions

ds4 approach: `mtp_raw_cache`, `mtp_kv_lora_cache`, `mtp_k_rope_cache` — separate storage for MTP attention. The MTP KV is populated during MTP forward (NOT during target forward).

**Without KV cache:** MTP attention degenerates to single-token (score=1.0, output=V=K passthrough). Drafts don't match target verification → 0% acceptance.

### P1: Metal SSD Streaming (Unreliable for model >> RAM)
**Status:** Root-caused. No user-space fix possible.

Metal's `newBufferWithBytesNoCopy` doesn't trigger GPU page faults for evicted file-backed pages. Only fix: pread()-based expert loading into Metal-managed buffers (like ds4). This is a ~500-line refactor touching format.zig, expert_cache.zig, metal.zig, deepseek4.zig.

**Workaround:** Use `--backend cpu` for SSD streaming.

### P2: IQ2_XXS Coherence (Blocks 1.7+ tok/s from 81GB model)
**Status:** Root-caused. No fix within Agave.

2-bit quantization error amplified 30× per layer through HC mixing. ds4 engine achieves coherent output from the same weights (likely through GPU graph capture that preserves precision differently). Agave's kernel is correct — the issue is fundamental to how HC amplifies noise.

---

## Architecture: Why ds4 is 5× Faster

| Feature | ds4 (5.69 tok/s) | Agave (1.1 tok/s) |
|---------|-----------------|-------------------|
| Expert dispatch | Metal graph capture, GPU-resident tables | CPU-side per-expert `be.gemv()` calls |
| SSD I/O | pread() into Metal buffers | mmap + page cache (NaN risk on Metal) |
| Streaming prefill | Overlapped, 2 full layers reserved | Sequential `forward()` loop |
| Expert cache | Memory-budget aware, GPU-integrated | Slot-count based, CPU madvise |
| Model size | 81GB Q2 imatrix (coherent) | 155GB MXFP4 (coherent but large) |
| MTP | Built-in to GGUF, GPU-graph integrated | Loaded from separate safetensors |

---

## Closing the Gap — Roadmap

### Near-term (this branch)
1. **MTP KV cache** — ~200 lines. Enables >0% acceptance → estimated 2–3× throughput
2. **pread-based Metal expert loading** — ~500 lines. Fixes Metal SSD streaming completely
3. **Overlapped SSD prefetch** — background pread with GPU/CPU compute overlap

### Medium-term
4. **Metal graph capture for expert dispatch** — batch all per-layer expert GEMVs into single dispatch
5. **GPU-resident non-routed weights** — pin attention/HC/output weights in Metal working set
6. **Asymmetric quantization in Agave** — Q8 attention + MXFP4 experts (like ds4's approach)

### Research
7. **MTP routed experts** — load MTP expert weights from shards 46-48 (9.8GB additional)
8. **DSpark confidence head** — trim MTP drafts using confidence prediction
9. **HC stabilization** — investigate ds4's precision handling for coherent IQ2_XXS

---

## Models & Paths

```bash
# MXFP4 (coherent, 155GB, 1.0-1.3 tok/s CPU)
GGUF=$HOME/.cache/huggingface/hub/models--ggml-org--DeepSeek-V4-Flash-0731-GGUF/blobs/DeepSeek-V4-Flash-0731-MXFP4-00001-of-00002.gguf
./zig-out/bin/agave "$GGUF" --backend cpu --ssd-streaming --ctx-size 512 -n 32 -t 0.0 "prompt"

# MXFP4 + MTP (coherent, 1.0-1.1 tok/s, 0% MTP acceptance)
./zig-out/bin/agave "$GGUF" --backend cpu --ssd-streaming --ctx-size 512 -n 32 \
  --mtp-model /tmp/ds4_mtp_weights.safetensors --spec-mode mtp --spec-tokens 3 -t 0.0 "prompt"

# ds4 Q2 imatrix (81GB, 1.7 tok/s CPU, garbled)
./zig-out/bin/agave /tmp/ds4/ds4flash.gguf --backend cpu --ssd-streaming --ctx-size 512 -n 32 -t 0.0 "prompt"

# ds4 engine reference (5.69-7.32 tok/s, coherent)
cd /tmp/ds4 && ./ds4 -m ds4flash.gguf --ssd-streaming -c 512 -p "prompt" --temp 0 --tokens 32 --nothink

# MTP weight extraction (run once)
python3 -c "
from safetensors import safe_open
from safetensors.torch import save_file
snap = '$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062'
mtp = {}
for s in [46,47,48]:
    with safe_open(f'{snap}/model-000{s}-of-00048.safetensors', framework='pt', device='cpu') as f:
        for n in f.keys():
            if n.startswith('mtp.') and ('expert' not in n or 'shared' in n):
                mtp[n] = f.get_tensor(n)
save_file(mtp, '/tmp/ds4_mtp_weights.safetensors')
"

# DSpark weights (Qwen3, not DS V4)
ls ~/Models/dspark_qwen3_8b/  # 4.7GB safetensors, 5-layer backbone
```

---

## Key Learnings

1. **Metal + mmap + model >> RAM = unreliable.** GPU page faults don't work for file-backed pages on Apple Silicon. Use pread() or CPU backend.

2. **2-bit HC cascade is fatal.** Hyper connections amplify 10% per-layer error to complete signal loss (r=0.02) across 43 layers. ds4's GPU graph capture somehow avoids this.

3. **GGUF quantizers strip MTP.** All GGUF converters (ggml-org, antirez/ds4) drop the MTP weight tensors. MTP weights must be loaded from HF safetensors shards 46-48.

4. **MTP without KV cache ≈ useless for speculative decoding.** Single-token attention degenerates to V passthrough. Need full MTP KV cache for meaningful drafts.

5. **MXFP8 = FP8 E4M3 + E8M0 tile scales.** Group size 128 for DS V4 Flash. Must apply tile scale to every FP8 GEMV — unscaled FP8 produces wrong results.

6. **SSD streaming at 73% cache hit is compute-bound.** At warm cache, 70% of per-token time is compute (attention + FFN), only 30% is SSD reads. GEMV optimization matters more than cache optimization at this point.

7. **Expert usage is uniform.** ~51 unique experts per layer out of 256. No extreme hot/cold split. Frequency-based caching doesn't help much over LRU.

8. **Sinkhorn convergence is axis-invariant.** Agave's row-first and ds4's column-first initial softmax both converge to the same doubly-stochastic matrix after 20 iterations (max diff 6e-8). Not a bug source.

---

*Autoresearch: 14 iterations, ~8 hours, branch `autoresearch/ds4-perf`.*
*Benchmarked: 2026-08-13–14. Engines: Agave (HEAD), ds4 (latest main), llama.cpp b10360, mlx-lm 0.31.1.*
