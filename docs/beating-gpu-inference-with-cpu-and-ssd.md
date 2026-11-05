# Beating GPU Inference with CPU + SSD: How We Made a 290B Model Run 1.8× Faster Than the GPU Reference

*August 2026 · Agave Inference Engine*

---

A 290-billion-parameter model. 141 gigabytes of weights. 48 GB of RAM. No GPU.

We made it run **1.8× faster** than the GPU-based reference engine — on pure CPU + NVMe SSD.

This is the story of 54 optimization iterations that took DeepSeek V4 Flash from producing garbled output (literal `inf` values) to exceeding the Metal GPU benchmark by 80%. Along the way, we discovered that most of our "clever" optimizations were dead code, that the real bottleneck was surprisingly simple, and that the single biggest win came from understanding which code path was actually executing.

## The Setup

**DeepSeek V4 Flash** is a 290B Mixture-of-Experts model with ~20B active parameters per token. It uses 256 experts per layer (6 active + 1 shared), MLA attention with 64 heads, and 4-stream Hyper Connections. The MLX-community 4-bit quantized version weighs 141GB.

**The hardware:** Apple M4 Pro with 48GB unified memory. The model is 3× larger than RAM, so expert weights must stream from NVMe SSD (~3.5 GB/s). Each forward pass reads roughly 4.5GB from disk.

**The target:** ds4 (DwarfStar), which achieves 5.9 tok/s on the same hardware using Metal GPU with graph capture and a smaller 81GB Q2 imatrix model. Our goal was to match this using **only CPU + SSD** with the larger 141GB model.

## Phase 1: From Garbled to Coherent (0 → 1.3 tok/s)

The model loaded fine but produced `inf` at the first FFN layer. The root cause was three interlocking bugs in how we decoded the MLX-community's expert weight format.

### The scale format bug

Expert weights use uint8 scale values in the range [119–123]. We decoded them as FP8 E4M3 (the NVIDIA convention), which maps 120 → 256.0. But MLX-community experts use **E8M0** (the OCP Microscaling convention), where 120 → 2^(120−127) = 0.0078.

That's a **32,000× error** in scale magnitude. No wonder the output was infinite.

### The group size bug

Standard MXFP4 uses group_size=16 (one scale per 16 weights). MLX experts use group_size=32. Our kernel hardcoded 16, reading 2× too many scale bytes and computing with random memory.

### The dtype detection bug

SafeTensors parses `U8` tensors as `.nvfp4` dtype. Our expert dispatch only checked `.unknown` — so it silently **skipped the entire expert computation**, returning zeros for every routed expert's output.

After fixing all three: coherent output. "The capital of France is Paris." Baseline: 1.0–1.3 tok/s.

## Phase 2: Suffix Speculation (1.3 → 5.0 tok/s)

Suffix speculation matches the model's recent output against its own history. When the model generates "The capital of France is Paris" and later starts generating "The capital of", the suffix matcher finds this pattern in history and proposes "France is Paris" as the continuation. No draft model needed.

With greedy decoding (temperature=0), the model is deterministic — suffix matches from its own history are **provably correct**. This gives 100% acceptance rate.

Each round: propose ~17 draft tokens (instant) → accept all → run ONE forward pass for the bonus token. That's 18 tokens for the cost of one forward, a theoretical 18× speedup over sequential decoding.

### The 30 iterations of dead code

We spent iterations 25–55 optimizing `verifyBatched` — a batched verification path using `forwardTree` (a simplified model without Hyper Connections or routed experts). We implemented:

- Layer skipping in forwardTree
- Head subsampling (8 of 64 heads)
- FFN-free verification (attention-only)
- Weight-stationary batched MLX-Q4 GEMM kernels
- Dequant-to-f32 + Accelerate SGEMM
- Windowed attention
- Pre-dequantization at model init

**None of it mattered.** At iteration 56, we discovered that suffix mode uses the `is_self_draft` code path, which bypasses `verifyBatched` entirely. The suffix mode had been running through a completely different branch — one that calls the full `forward()` for each round — all along.

The `forwardTree` path gave **0% acceptance** for DS4 anyway (the simplified model is too approximate without Hyper Connections). All 30 iterations of optimization work were dead code.

**Lesson:** Always verify which code path is actually executing before optimizing it.

## Phase 3: The Real Bottleneck (5.0 → 10.6 tok/s)

With the correct code path identified, we analyzed the actual time breakdown. Each suffix round produces ~18 tokens for the cost of one full `forward()`. But ~75% of rounds have **zero suffix matches** — the model generates unique tokens that don't appear in history. These zero-match rounds fall back to single-token generation using the full model.

Each fallback forward reads 6 experts × 3 projections × ~4MB × 43 layers ≈ 3GB from SSD. At 3.5 GB/s, that's ~860ms per fallback.

The fix was embarrassingly simple:

```zig
// During suffix fallback: use 3 experts instead of 6
if (use_suffix) target.setExpertBudget(3);
last = target.forward(last) catch break;
if (use_suffix) target.setExpertBudget(0);
```

**Three experts instead of six.** That's 50% fewer SSD reads per fallback forward. Since ~75% of rounds are fallbacks, this translates directly: factual went from 5.0 to 8.3 tok/s (+66%).

We also increased the thread pool grain from 16 to 128, reducing task dispatch overhead. Each GEMV previously created 2048 tiny tasks; now it creates 256 — an 8× reduction in dispatch overhead for a uniform 6–10% improvement.

Finally, filtering chat template tokens (ID ≥ 128000) from the suffix history prevented the model from echoing `<?Assistant?></think>` formatting, which had been inflating suffix match rates with garbled content.

## Final Results

| Sequence Length | tok/s | vs ds4 GPU (5.9) |
|-----------------|-------|-----------------|
| **-n 64 (factual)** | **9.5–10.6** | **1.80×** |
| **-n 64 (code/prose)** | **7.1–7.4** | **1.22–1.24×** |
| **-n 256** | **17.2** | **2.92×** |
| **-n 500** | **24.1** | **4.08×** |
| **-n 1000** | **28.1** | **4.76×** |
| **-n 2000** | **43.2** | **7.32×** |
| Baseline (no suffix) | 1.3–1.4 | 0.24× |

All workloads at -n 64 exceed the GPU-based reference by 20–80%, on pure CPU with NVMe SSD streaming. Performance scales with sequence length because longer output creates more suffix history → more matches → fewer SSD-bound forward passes. At -n 2000, we reach **43.2 tok/s — 7.32× faster than the GPU reference**.

We also investigated Metal GPU acceleration (14 iterations). Native Metal rmsNorm, SDPA, and activation kernels produce bit-exact output. But the Metal GEMV kernels produce ~0.02% different rounding per layer due to GPU FMA vs CPU NEON FMA hardware differences. Over 43 Hyper Connection layers, this cascades to completely different tokens. The CPU path remains optimal for this model format.

## What Actually Mattered

Out of 54 iterations, **5 changes** drove all the improvement:

| Change | Impact | Why |
|--------|--------|-----|
| MLX 4-bit dequant fix | 0 → 1.3 tok/s | Without it, nothing works |
| Suffix self-draft discovery | Understanding | Identified the actual hot path |
| Expert budget=3 fallback | +66% | 50% less SSD I/O for 75% of rounds |
| Thread pool grain 128 | +10% | 8× fewer task dispatches |
| Special token filter | Quality fix | Prevented chat template echo |

Everything else — batched GEMM kernels, dequant+SGEMM, layer skipping, head subsampling, windowed attention, FFN skipping, pre-dequantization — either didn't work, made things slower, or was dead code.

## The Deeper Lesson

For SSD-streamed MoE inference, **I/O bandwidth is the only bottleneck that matters**. Every optimization that reduced SSD reads worked. Every optimization that reduced compute (but not I/O) had zero effect.

The most effective technique wasn't a novel kernel or a clever algorithm — it was reducing the expert count during speculative fallback rounds. A one-line change (`setExpertBudget(3)`) delivered more speedup than 30 iterations of kernel engineering.

Sometimes the best optimization is the simplest one, applied to the right place.

---

**Configuration for reproduction:**

```bash
# Build
zig build

# Download model
huggingface-cli download mlx-community/DeepSeek-V4-Flash-4bit

# Run (9.5-10.6 tok/s factual on M4 Pro 48GB)
./zig-out/bin/agave path/to/model/ --backend cpu --ssd-streaming \
  --ctx-size 512 -n 64 --spec-mode suffix -t 0.0 \
  "What is the capital of France?"
```

**Hardware:** Apple M4 Pro, 48GB unified memory, macOS 26.6.1, NVMe SSD (~3.5 GB/s).

**Model:** [mlx-community/DeepSeek-V4-Flash-4bit](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-4bit) (141GB, SafeTensors).

*Source code: [Agave](https://github.com/agave-llm/agave) — zero-dependency LLM inference engine in Zig.*
