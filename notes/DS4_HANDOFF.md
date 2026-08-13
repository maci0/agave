# DeepSeek V4 Flash — Performance & MTP Handoff

*Last updated: 2026-08-14. Covers 15 autoresearch iterations + 5-iteration MTP ralph loop.*

---

## Current State

**Hardware:** Apple M4 Pro, 48GB unified memory, macOS 26.6, NVMe SSD (~3.5 GB/s)  
**Branch:** `autoresearch/ds4-perf` (3,400+ insertions across 30+ files)

### Benchmark Summary

| Engine | Model | Size | tok/s | Coherent | Status |
|--------|-------|------|-------|----------|--------|
| **Agave CPU** | MXFP4 ggml-org | 155GB | **1.0–1.3** | ✅ | Best reliable path |
| **Agave CPU+MTP** | MXFP4 + MTP safetensors | 155+0.6GB | **1.0** | ✅ | Pipeline works, 0% accept |
| Agave Metal | MXFP4 ggml-org | 155GB | 0.8–1.1 | ⚠️ NaN risk | volatile_weights partial fix |
| Agave CPU | ds4 Q2 imatrix | 81GB | 1.7 | ❌ | IQ2_XXS HC amplification |
| **ds4 engine** | Q2 imatrix | 81GB | **5.7–7.3** | ✅ | Reference target |

---

## What Was Built

### 1. MTP (Multi-Token Prediction) — Major Feature

**Discovery:** DS V4 Flash HF safetensors contain 3 full MTP decoder layers (4,705 tensors) that ALL GGUF quantizers stripped. Extracted 97 non-expert tensors (595MB) for the initial implementation.

**Files delivered:**

| File | Lines | Purpose |
|------|-------|---------|
| `src/models/ds4_mtp.zig` | 142 | Safetensors loader: JSON header parsing, mmap, tensor lookup |
| `src/backend/kernels/cpu/gemv_fp8.zig` | +37 | `gemvMXFP8`: FP8 E4M3 + E8M0 per-tile-scaled GEMV kernel |
| `src/models/deepseek4.zig` | +321 | `mtpForward`, MTP KV cache, `resetMtpCache`, target-forward KV population |
| `src/models/model.zig` | +14 | `setMtpWeights` vtable, `n_mtp_layers` field dispatch |
| `src/main.zig` | +21 | `--mtp-model` CLI flag, MTP weight loading + initialization |
| `src/spec/spec_decode.zig` | +5 | Token chaining in `draftMtp`, debug logging |
| `notes/MTP_DESIGN.md` | 79 | Architecture design + tensor mapping reference |

**MTP forward path:**
```
mtpForward(token_id, depth):
  1. Build input: concat(target_hidden[4096], prev_mtp_hidden[4096], embed(token)[4096])
  2. main_proj: MXFP8 GEMV [4096, 12288] → hidden2[4096]
  3. main_norm: RMS norm (BF16 weights dequanted to f32)
  4. Attention: attn_norm → kv_proj (MXFP8) → kv_a_norm → KV cache append →
     compressed-space dot-product attention → softmax → wo_a → wo_b (all MXFP8)
  5. FFN: ffn_norm → gate_proj + up_proj → clampedSiluMul → down_proj (all MXFP8)
  6. Output: MTP-specific norm → shared lm_head → argmax
  Fallback: Markov head (BF16 W1/W2 bigram transition bias on target logits)
```

**MTP KV cache:**
- `mtp_kv_cache`: flat f32 buffer `[max_seq_len × 512]`
- Populated during EACH target `forward()` call (MTP wkv projection of target hidden)
- Compressed-space attention scores: `kv_current · kv_cached[t] / sqrt(512)`
- `resetMtpCache()`: rolls back `mtp_kv_len` on speculation rejection

**Current status:** 0% acceptance rate. Root cause: compressed-space attention without Q head projection produces degenerate attention (same output across all 64 heads). Full MLA Q projection (wq_a + wq_b + RoPE)e single remaining piece for >0% acceptance.

**Usage:**
```bash
# Extract MTP weights (one-time, requires HF shards 46-48 downloaded):
python3 -c "
from safetensors import safe_open
from safetensors.torch import save_file
snap = '~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062'
mtp = {}
for s in [46,47,48]:
    with safe_open(f'{snap}/model-000{s}-of-00048.safetensors', framework='pt', device='cpu') as f:
        for n in f.keys():
            if n.startswith('mtp.') and ('expert' not in n or 'shared' in n):
                mtp[n] = f.get_tensor(n)
save_file(mtp, '/tmp/ds4_mtp_weights.safetensors')
"

# Run with MTP:
./zig-out/bin/agave model.gguf --backend cpu --ssd-streaming --ctx-size 512 \
  --mtp-model /tmp/ds4_mtp_weights.safetensors --spec-mode mtp --spec-tokens 3 \
  -t 0.0 "prompt"
```

### 2. MXFP8 GEMV Kernel
**File:** `src/backend/kernels/cpu/gemv_fp8.zig`

```zig
pub fn gemvMXFP8(x, w, scale, y, n, k, scale_cols) void
```

Per-tile scaled FP8 dot product. Group size = 128. Scale tensor shape `[n/128, k/128]` with E8M0 encoding. Used for all MTP weight tensors (main_proj, attention Q/K/V/O, FFN gate/up/down). CPU-only; called directly from MTP path (not via backend vtable).

### 3. Metal Buffer Cache Reliability
**Files:** `src/backend/metal.zig`, `src/backend/backend.zig`, `src/main.zig`

- `volatile_weights` mode: flushes Metal buf_cache on `sync()` for `--ssd-streaming`
- `stable_cache`: persistent Metal-managed buffer copies (attempted, partially works)
- `setVolatileWeights()` / `flushBufferCache()` Backend vtable methods

**Root cause:** Metal `newBufferWithBytesNoCopy` does NOT trigger GPU page faults for evicted file-backed mmap pages on Apple Silicon. Five approaches tested, none fully reliable. Only `pread()` into Metal-managed buffers (ds4's approach) would fix this completely (~500-line refactor, deferred).

**Recommendation:** Use `--backend cpu` for SSD streaming when model >> RAM.

### 4. IQ2_XXS Coherence Analysis

- Dequant kernel verified correct (codebook + signs match ds4/llama.cpp)
- Sinkhorn HC verified identical to ds4 (max diff 6e-8)
- Tokenization verified identical (11 tokens, same IDs)
- **Root cause:** 2-bit noise cascades through 43 HC layers → r=0.02 logit correlation (random)
- MXFP4 preserves r=0.65 logit correlation → marginal but coherent output

### 5. SIMD MXFP4 GEMV
**File:** `src/backend/kernels/cpu/gemv_fp4.zig`

`gemvMXFP4_V`: `@Vector(4,f32)` with `@mulAdd` FMA. Correct output, no measurable speedup (SSD-bound at current cache hit rates).

### 6. Expert Cache Profiling

- ~51 unique experts/layer (of 256), uniform distribution
- Cache hit rate: 52% cold → 73% warm (64 tokens)
- At warm cache: ~70% compute-bound, ~30% SSD-bound
- 3212 auto-sized cache slots on 48GB

---

## Known Issues & Blockers

### P0: MTP Acceptance Rate (0%)
**Status:** Root-caused, clear fix path.

Compressed-space attention (no Q heads) produces degenerate attention. Fix: implement full MLA Q projection in mtpForward:
1. `wq_a[1024, 4096]` MXFP8 GEMV → q_compressed
2. `q_a_norm` RMS norm
3. `wq_b[32768, 1024]` MXFP8 GEMV → q_full (64 heads × 512)
4. RoPE on q_full and kv_proj
5. Per-head dot-product attention: `Q[h] · K[t]` for all cached positions
6. Softmax → weighted V sum → inverse RoPE → wo_a → wo_b

**Estimated:** ~90 lines of code, ~30ms per draft depth overhead.

### P1: Metal SSD Streaming
**Status:** Root-caused, needs pread() refactor (~500 lines).

### P2: IQ2_XXS Coherence  
**Status:** Root-caused, no fix within Agave (ds4 GPU graph precision).

---

## Architecture: Why ds4 is 5× Faster

| Feature | ds4 (5.7 tok/s) | Agave (1.1 tok/s) |
|---------|-----------------|-------------------|
| Expert dispatch | Metal graph, GPU-resident tables | CPU per-expert `be.gemv()` |
| SSD I/O | pread() → Metal buffers | mmap + page cache |
| Streaming prefill | Overlapped, 2 layers reserved | Sequential `forward()` |
| Expert cache | Memory-budget, GPU-integrated | Slot-count, CPU madvise |
| Model size | 81GB Q2 imatrix (coherent) | 155GB MXFP4 |
| MTP | Built-in GGUF, GPU-graph | Separate safetensors, CPU |

---

## Roadmap

### Near-term (this branch)
1. **MTP full MLA Q projection** — ~90 lines → >0% acceptance → 2-3× throughput
2. **pread-based Metal expert loading** — ~500 lines → reliable Metal SSD streaming
3. **Overlapped SSD prefetch** — background pread with compute overlap

### Medium-term
4. Metal graph capture for expert dispatch
5. GPU-resident non-routed weights (Metal residency sets)
6. Asymmetric quantization (Q8 attention + MXFP4 experts)

### Research
7. MTP routed experts (load from HF shards 46-48, 9.8GB additional)
8. DSpark confidence head for draft trimming
9. HC stabilization for coherent IQ2_XXS

---

## Reproduction Commands

```bash
# Best current path: MXFP4 CPU (1.0-1.3 tok/s, coherent)
GGUF=$HOME/.cache/huggingface/hub/models--ggml-org--DeepSeek-V4-Flash-0731-GGUF/blobs/DeepSeek-V4-Flash-0731-MXFP4-00001-of-00002.gguf
./zig-out/bin/agave "$GGUF" --backend cpu --ssd-streaming --ctx-size 512 -n 32 -t 0.0 "prompt"

# With MTP (pipeline works, 0% acceptance pending full Q projection)
./zig-out/bin/agave "$GGUF" --backend cpu --ssd-streaming --ctx-size 512 -n 32 \
  --mtp-model /tmp/ds4_mtp_weights.safetensors --spec-mode mtp --spec-tokens 3 -t 0.0 "prompt"

# ds4 reference (5.7-7.3 t
cd /tmp/ds4 && ./ds4 -m ds4flash.gguf --ssd-streaming -c 512 -p "prompt" --temp 0 --tokens 32 --nothink

# DSpark weights (Qwen3 8B, not DS V4)
ls ~/Models/dspark_qwen3_8b/  # 4.7GB safetensors, 5-layer backbone
```

---

## Key Learnings

1. **Metal + mmap + model >> RAM = unreliable.** GPU page faults don't work for file-backed pages. Use pread() or CPU backend.
2. **2-bit HC cascade is fatal.** 10% per-layer → r=0.02 across 43 layers. ds4's GPU graph somehow avoids this.
3. **GGUF quantizers strip MTP.** All converters drop MTP tensors. Must load from HF safetensors.
4. **MTP without Q heads ≈ useless.** Compressed-space attention degenerates without per-head Q projections.
5. **MXFP8 = FP8 E4M3 + E8M0 tile scales.** Group size 128. Must apply scale to every GEMV.
6. **SSD streaming at 73% cache hit is compute-bound.** GEMV optimization > cache optimization.
7. **Expert usage is uniform.** ~51/256 per layer, no hot/cold split. LRU is fine.
8. **Sinkhorn is axis-invariant.** Row-first vs column-first converge identically after 20 iterations.

---

*15 autoresearch iterations + 5 ralph iterations. Branch: `autoresearch/ds4-perf`.*
*Benchmarked: 2026-08-13–14. Engines: Agave (HEAD), ds4 (latest main).*
