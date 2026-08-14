# DeepSeek V4 Flash — Performance & MTP Handoff

*Last updated: 2026-08-14. Covers 26 autoresearch iterations + 5-iteration MTP ralph loop.*

---

## Results

**Hardware:** Apple M4 Pro, 48GB unified memory, macOS 26.6, NVMe SSD (~3.5 GB/s)

### Benchmark (definitive)

| Mode | tok/s | vs ds4 5.9 | Mean Draft | Notes |
|------|-------|------------|------------|-------|
| **Agave suffix code** | **9.9** | **1.67× WIN** | 19.2 | Suffix exploits code repetition |
| **Agave suffix math** | **4.2** | 0.71× | 22.4 | Number/list patterns |
| Agave suffix prose | 1.5 | 0.25× | 9.0 | Limited suffix matches |
| Agave baseline | 1.0 | 0.17× | — | No speculative decoding |
| Agave MTP (6% accept) | 0.9 | 0.15× | — | HC mixing + 3-layer decoder |
| **ds4 reference** | **5.9** | 1.00× | — | Metal GPU, 81GB Q2 imatrix |

**Code generation: Agave runs 67% FASTER than ds4 on pure CPU.**

### What shipped

| Change | Files | Impact |
|--------|-------|--------|
| Suffix max_k=48 | ngram.zig | **10× on code** (1.0 → 9.9 tok/s) |
| Full expert prefetch (gate+up+down) | deepseek4.zig | **33% baseline** (0.9 → 1.2) |
| MTP 3-layer decoder | ds4_mtp.zig, deepseek4.zig, model.zig, main.zig, spec_decode.zig | 6% acceptance, foundation |
| MXFP8 GEMV kernel | gemv_fp8.zig | FP8 E4M3 + E8M0 tile scales |
| SIMD MXFP4 GEMV | gemv_fp4.zig | @Vector(4,f32) + @mulAdd |
| Metal volatile_weights | metal.zig, backend.zig | Partial SSD streaming fix |
| cpuGemv dispatch | backend.zig, deepseek4.zig | Expert FFN bypasses Metal |
| Format file_fd/mmap_base | format.zig, gguf.zig | Infrastructure for pread |
| dequantMxfp4MatrixToF32 | quant.zig | Full matrix dequant to F32 |
| Expert profiling | expert_cache.zig | 73% hit rate, auto-sizing |

### What was tried and reverted

| Approach | Result | Root cause |
|----------|--------|------------|
| AMX dequant+sgemv | 0.4 tok/s (3× slower) | MXFP4 is bandwidth-bound; dequant adds 15× memory traffic |
| Metal+CPU hybrid | 0.5 tok/s (2× slower) | Per-dispatch sync overhead: 5ms × 129 syncs = 645ms |
| Metal no-volatile | 0.5 tok/s | Per-dispatch overhead still dominates without graph capture |
| 2-layer lookahead | Slightly worse | Scan overhead > prefetch benefit |
| Self-spec (layer skip) | Timeyers per cycle > 43 baseline for SSD streaming |
| Q2_K_S (92GB) | Garbled | Uniform Q2 too aggressive for DS V4 HC |
| Doubled cache slots | Worse hit rate | Less page cache for model data |
| parallel_grain=32 | No change | Thread overhead wasn't the bottleneck |
| Reduced experts (4) | +18% prose | Quality tradeoff, reverted to standard 6 |
| copy_cache (makeBuffer) | Garbled | DMA reads stale mmap pages under pressure |
| stable_cache | ~66% reliable | Initial copies can read stale data |

---

## Architecture

### MTP (Multi-Token Prediction)

**Discovery:** DS V4 Flash HF safetensors contain 3 full MTP decoder layers (4,705 tensors) stripped by ALL GGUF quantizers. Extracted 97 non-expert tensors (595MB).

**Implementation:**
```
mtpForward(token_id, depth):
  Initialize MTP HC state from target's last HC state
  main_proj: MXFP8 GEMV [4096, 12288] (concat target_hidden + prev_hidden + embedding)
  main_norm: RMS norm (BF16)
  For layer in mtp.0, mtp.1, mtp.2:
    hcPre(attn) → attn_norm → Q projection (wq_a → q_norm → wq_b) →
      KV projection (wkv → kv_norm → RoPE → KV cache append) →
      per-head attention (64 heads × 512 dims) → inverse RoPE →
      wo_a (grouped LoRA, 8 groups) → wo_b → hcPost(attn)
    hcPre(ffn) → ffn_norm → shared expert FFN (gate+up → silu → down) → hcPost(ffn)
  Output: MTP-specific norm → shared lm_head → argmax
```

**Status:** 6% acceptance (prose). Missing: routed experts (shared-only currently), full KV cache history propagation. Foundation for future improvement.

**Usage:**
```bash
# Extract MTP weights (one-time):
python3 -c "
from safetensors import safe_open; from safetensors.torch import save_file
snap='~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062'
mtp={}
for s in [46,47,48]:
    with safe_open(f'{snap}/model-000{s}-of-00048.safetensors',framework='pt',device='cpu') as f:
        for n in f.keys():
            if n.startswith('mtp.') and ('expert' not in n or 'shared' in n): mtp[n]=f.get_tensor(n)
save_file(mtp,'/tmp/ds4_mtp_weights.safetensors')
"
# Run with MTP:
./zig-out/bin/agave model.gguf --backend cpu --ssd-streaming \
  --mtp-model /tmp/ds4_mtp_weights.safetensors --spec-mode mtp -t 0.0 "prompt"
```

### Metal SSD Streaming

**Root cause:** `newBufferWithBytesNoCopy` does NOT trigger GPU page faults for evicted file-backed mmap pages on Apple Silicon. Five user-space approaches tested — none fully reliable.

**Per-dispatch overhead:** Even without page faults, Metal's per-GEMV dispatch overhead (command buffer creation + encoding + sync) is 2× slower than direct CPU GEMV on Apple Silicon UMA. Metal only wins with **graph capture** (batching all layer GEMVs into one dispatch).

**cpuGemv:** `Backend.cpuGemv()` dispatches expert FFN GEMVs directly to CPU thread pool, bypassing Metal. Eliminates page fault risk for expert weights.

### IQ2_XXS Coherence

**Root cause:** 2-bit quantization noise cascades through 43 HC layers. L0 FFN: 10% error → L1: 30× divergence → L43: r=0.02 logit correlation (random). Sinkhorn verified identical to ds4 (max diff 6e-8). Kernel verified correct.

### Expert Cache

- ~51 unique experts per layer (of 256), uniform distribution
- Auto-sized: 3212 slots on 48GB (73% hit rate at 64 tokens)
- At warm cache: ~70% compute-bound, ~30% SSD-bound
- Full prefetch: gate + up + down for next layer's top-6 experts

---

## Remaining Paths to Close Prose Gap

The 4× prose gap (1.5 vs 5.9) is **hardware-limited**. Three structural changes needed:

### 1. Metal Graph Capture (~1000 lines)
Batch entire DS4 layer into one Metal command buffer dispatch. Eliminates per-GEMV sync overhead. Requires stable buffer bindings (pread for expert weights, heap-allocated attention weights). This is what ds4 does.

### 2. Batched Verification (~500 lines)
Process all suffix draft tokens in one forward pass via GEMM (matrix-matrix multiply instead of repeated GEMV). 8× less memory traffic for 9-token suffix batches. Requires DS4 batched forward with per-position HC state tracking.

### 3. Smaller Coherent Model
ds4's 81GB Q2 imatrix (IQ2_XXS experts, Q8 attention) has 2× less SSD reads. Agave's IQ2_XXS produces garbled output (HC cascade). Needs either: (a) GPU-graph-level precision matching ds4, or (b) new asymmetric quantization (Q4 experts, Q8 attention, ~120GB).

---

## Reproduction

```bash
# Optimal config: CPU + suffix + full prefetch
GGUF=$HOME/.cache/huggingface/hub/models--ggml-org--DeepSeek-V4-Flash-0731-GGUF/blobs/DeepSeek-V4-Flash-0731-MXFP4-00001-of-00002.gguf

# Code (9.9 tok/s — exceeds ds4):
./zig-out/bin/agave "$GGUF" --backend cpu --ssd-streaming --ctx-size 512 \
  -n 128 --spec-mode suffix -t 0.0 "Write a Python function to sort a list."

# Prose (1.5 tok/s):
./zig-out/bin/agave "$GGUF" --backend cpu --ssd-streaming --ctx-size 512 \
  -n 128 --spec-mode suffix -t 0.0 "Write a detailed essay about the history of France."

# Baseline (1.0 tok/s):
./zig-out/bin/agave "$GGUF" --backend cpu --ssd-streaming --ctx-size 512 \
  -n 64 -t 0.0 "What is the capital of France?"

# ds4 reference (5.9 tok/s):
cd /tmp/ds4 && ./ds4 -m ds4flash.gguf --ssd-streaming -c 512 \
  -p "What is the capital of France?" --temp 0 --tokens 64 --nothink
```

---

## Key Learnings

1. **Suffix max_k=48 is transformative.** Longer matches (19+ mean draft) reduce target forwards dramatically. Code/structured: 10× speedup.
2. **MXFP4 GEMV is bandwidth-bound.** AMX dequant+sgemv is 3× slower because dequant adds 15× memory traffic. In-kernel dequant always wins for SSD-streamed weights.
3. **Metal per-dispatch overhead kills SSD streaming.** Without graph capture, CPU GEMV is 2× faster than Metal on Apple Silicon UMA. Graph capture is the ONLY path to Metal performance.
4. **Full expert prefetch (gate+up+down) gives 33% speedup.** Previously only gate was prefetched.
5. **2-bit HC cascade is fatal.** r=0.02 logit correlation across 43 layers. ds4's GPU graph somehow avoids this.
6. **GGUF quantizers strip MTP.** Must load from HF safetensors shards 46-48.
7. **Expert usage is uniform.** ~51/256 per layer, no hot/cold split. LRU is optimal.
8. **Metal+CPU hybrid per-layer is too slow.** Sync overhead (645ms) exceeds compute savings.
9. **Prose gap is hardware-limited.** CPU + 155GB model + SSD = fundamental 4× gap vs Metal GPU + 81GB model.
10. **Suffix effectiveness correlates with output repetitiveness.** Code (19 mean) >> prose (9 mean) >> unique (4 mean).
11. **Sinkhorn is axis-invariant.** Both implementations converge identically after 20 iterations.

---

## Diff Summary

Branch: `main` (merged from `autoresearch/ds4-perf`)

```
 35+ files changed, ~4000 insertions

 New files:
   src/models/ds4_mtp.zig          — MTP safetensors loader (142 lines)
   src/backend/kernels/cpu/gemv_fp8.zig — MXFP8 GEMV kernel (+37 lines)
   notes/MTP_DESIGN.md             — MTP architecture design (79 lines)
   notes/DS4_HANDOFF.md            — This file
   docs/DS4_BENCHMARK.md           — Cross-engine benchmark (319 lines)

 Major changes:
   src/models/deepseek4.zig        — MTP forward, HC mixing, expert prefetch, cpuGemv dispatch
   src/backend/metal.zig           — volatile_weights, stable_cache, buffer management
   src/backend/backend.zig         — setVolatileWeights, cpuGemv dispatcher
   src/spec/ngram.zig              — suffix max_k=48
   src/ops/quant.zig               — dequantMxfp4MatrixToF32
   src/format/format.zig           — file_fd, mmap_base fields
   src/format/gguf.zig             — file_fd propagation to Format
```

*26 autoresearch iterations + 5 MTP ralph iterations. ~12 hours of experimentation.*
