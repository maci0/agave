# Research Ideas for Agave DS V4 Flash Maximum Performance

## Priority 1: IMPLEMENT NOW

### A. Pre-packed AMX SGEMM for Attention (arXiv 2606.25426)
**Paper:** "Exceeding Accelerate at LLM Prefill GEMM on M1 AMX"
**Key finding:** Custom AMX kernel beats Apple Accelerate by 1.58-2.04× for SGEMM!
- Weight PRE-PACKING at model load → amortized packing cost
- Fine multi-thread panels (N_c=64) → engages BOTH AMX blocks
- Accelerate leaves E-cluster AMX block IDLE → 2× wasted hardware
- M4 uses SME not AMX, but MpGEMM (referenced) showed 1.23× on M4 Pro
- End-to-end: 1.44× speedup in llama.cpp for 128-token prefill

**Agave application:**
- Pre-pack Q8_0 attention weights to F32 at model load (~15GB for non-expert weights)
- Use Accelerate SGEMM with pre-packed weights for attention GEMVs
- The key insight: Accelerate re-packs on EVERY call → pre-packing saves ~40% overhead
- For SSD streaming: attention weights are stable (same every token) → pack once
- Expected: ~30% per-forward speedup → 1.0 → 1.3 baseline, 1.5 → 2.0 prose

### B. SP-MoE Expert Prediction During Drafting (arXiv 2510.10302)
**Paper:** "SP-MoE: Speculative Decoding and Prefetching for MoE Inference"
**Key finding:** Use draft model's attention output to predict target experts!
- During suffix drafting: predict which experts verification will need
- Async prefetch predicted experts WHILE drafting runs (idle I/O window)
- Cross-model predictor: draft attention → target gating → 88.94% top-1 accuracy
- Cutoff layer policy: don't over-prefetch (U-shaped perf vs cutoff depth)
- Worker thread with async CUDA stream → continuous background prefetching
- 1.07-3.5× TPOT speedup across models

**Agave application:**
- After suffix match found: use the MATCHED TEXT's hidden state to predict experts
- Run target's gate GEMV on the hidden state → get routing decisions
- Pre-madvise predicted experts for ALL draft positions in the suffix match
- This predicts experts for 9+ positions at once (vs 1 at a time currently)
- Expected: ~50% reduction in verification SSD wait time

### C. PreScope Layer-Aware Expert Prediction (arXiv 2509.23638)
**Paper:** "PreScope: Unleashing Prefetching for MoE Inference"
**Key finding:** Layer-group-aware predictor (LLaPor) achieves 94% accuracy!
- Expert activations have distinct patterns per layer GROUP (input/middle/output)
- Learned lightweight predictor per layer group → much better than LRU
- Cross-layer scheduling: quantify prefetch gain vs loss globally
- Fine-grained async I/O with triple CUDA streams for weight transfer
- Up to 141% throughput improvement over baselines

**Agave application:**
- Replace LRU-based expert cache with LEARNED predictor
- Train per-layer-group predictors offline on calibration data
- Use cross-layer scheduling to decide which experts to prefetch
- Triple-stream async madvise for gate/up/down concurrently
- Expected: cache hit rate 73% → 90%+ → fewer SSD reads

## Priority 2: INVESTIGATE

### D. xHC: Expanded Hyper-Connections (arXiv 2607.14530)
**Paper:** "xHC: Expanded Hyper-Connections"
- DS V4 uses HC with N=4 streams. xHC scales to N=16 with sparse updates.
- Not directly applicable to inference (architectural change) but informs
  understanding of HC's role in the model.

### E. Cross-Model KV Cache Transfer (arXiv 2608.03893)
**Paper:** "Cross-Model KV Cache Transfer in LLM Families"
- Ridge mapper for KV cache transfer between model sizes
- Application: main model KV → MTP KV space
- Blocked: need clean wkv weights for ridge computation

### F. DSpark Paper (arXiv 2607.05147)
**Paper:** "DSpark: Confidence-Scheduled Speculative Decoding"
- Semi-autoregressive generation with Markov head
- Confidence-scheduled verification length trimming
- Already partially implemented in Agave

### G. FlashMemory-DeepSeek-V4 (arXiv 2606.09079)
**Paper:** "Lightning Index Ultra-Long Context via Lookahead Sparse Attention"
- Lookahead sparse attention for long context
- Reduces attention compute for long sequences
- Not critical for 512-token context but useful for scaling

## Priority 3: FUTURE

### H. Custom SME Kernel for M4 Pro
- MpGEMM paper showed 1.23× over Accelerate on M4 Pro
- Custom SME microkernels for MXFP4 dequant + GEMM
- Would require reverse-engineering M4 SME instruction encoding
- High effort (~1000 lines) but potentially 2× compute speedup

### I. Batched Verification with GEMM
- Process multiple draft tokens in one forward pass
- Blocked for quantized weights (GEMM = N×GEMV on CPU)
- Would work with pre-packed F32 weights + AMX SGEMM
- Combine with idea A: pre-pack → AMX GEMM → batched verify

## From Colibri (github.com/JustVugg/colibri)

### I. CFSE Nibble Entropy Compression (Priority: HIGH)
INT4 weights have measured entropy H=2.924 bits/weight (vs 4 bits stored).
rANS order-0 on nibbles achieves 1.37× compression:
- 155GB MXFP4 → ~113GB compressed
- 42GB fewer SSD bytes to read per model pass
- Decompression: ~1 cycle per nibble (negligible vs SSD latency)
- "No context model can do better — it's a theorem" (white nibble statistics)
- Implementation: ~200 lines in fse_coli.h, pure C, no dependencies

### J. pread + O_DIRECT Expert Loading (Priority: HIGH)
Bypass the OS page cache entirely for expert weights:
- Direct SSD → RAM transfer, no page cache pollution
- Eliminates our Metal mmap page fault issue!
- Colibri measured: 6.2 → 4.9 s/token on same hardware (21% faster)
- Requires: open with O_DIRECT flag, aligned buffers, pread syscall
- Implementation: ~100 lines to replace mmap with pread in expert dispatch

### K. Batch Expert Union for Verification (Priority: MEDIUM)  
Colibri's v4_moe_batch_union processes multiple tokens through MoE:
- Route all tokens → compute union of needed experts
- Load each expert ONCE → process all tokens needing it
- Combines with our MoE-Spec: union of top-4 experts across batch
- Implementation: ~200 lines in ffnLayer for batched expert dispatch

### L. Route Trace for Learned Expert Placement (Priority: LOW)
.coli_usage file persists expert activation frequencies across sessions:
- Builds routing histogram over time
- Hot experts get pinned in RAM
- Cold experts stay on SSD
- Replaces our LRU with data-driven placement
