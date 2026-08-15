# Autoresearch: DS V4 Flash Maximum Performance

**Started:** 2026-08-15
**Target:** Exceed ds4 5.9 tok/s on ALL workloads (prose, code, math)
**Approach:** Literature-driven — arxiv search → implement → benchmark
**Hardware:** Apple M4 Pro 48GB, macOS 26.6, NVMe SSD (~3.5 GB/s)

## Current State
- Code (suffix): 9.9 tok/s ✅ exceeds ds4
- Prose (suffix): 1.5 tok/s ❌ 4× gap
- Baseline (no spec): 1.0 tok/s

## Research Areas
1. MoE expert streaming/caching
2. Speculative decoding advances
3. KV cache compression for MLA
4. Quantized GEMM on Apple Silicon
5. Prefill optimization for HC models
6. DeepSeek-specific papers

## Results Log
| Iter | Paper/Technique | Change | tok/s | Decision |
|------|----------------|--------|-------|----------|
