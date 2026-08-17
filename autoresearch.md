# Autoresearch: DS4 Flash Metal GPU Maximum Performance

**Started:** 2026-08-18
**Target:** Maximum DS4 tok/s on Metal GPU (exceed ds4 5.9 reference)
**Hardware:** Apple M4 Pro 48GB, macOS 26.6, NVMe SSD (~3.5 GB/s)
**Model:** mlx-community/DeepSeek-V4-Flash-4bit (141GB MLX-Q SafeTensors)

## Current State
- Metal backend: status unknown for MLX 4-bit SafeTensors
- CPU baseline: 1.3-1.4 tok/s (no suffix), 9.5-10.6 tok/s (suffix)
- ds4 reference: 5.9 tok/s (Metal GPU, 81GB Q2 imatrix)

## Attack Vectors
1. Get Metal backend running for MLX-Q SafeTensors
2. Metal graph capture (batch layer dispatch)
3. Fused FFN megakernel (gate+up+silu+down)
4. MLX-Q4 Metal GEMV kernel
5. MXFP4 E8M0 Metal expert kernel
6. Metal SDPA for MLA attention
7. Expert streaming with MTLHeap buffers
8. Hybrid CPU+Metal dispatch

## Results Log
| Iter | Change | tok/s | Decision |
|------|--------|-------|----------|
