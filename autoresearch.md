# Autoresearch: DS4 Metal Performance → 10 tok/s

## Target
- Metric: tok/s (higher is better)
- Goal: ≥10 tok/s on Metal+suffix with coherent output
- Model: MLX-community DeepSeek-V4-Flash-4bit (141GB, SafeTensors)
- Hardware: Apple M4 Pro 48GB, macOS 26.6, NVMe SSD

## Baseline
- CPU+suffix historical best: 9.5-10.6 tok/s (expert_budget=3, grain=128, max_k=96)
- CPU+suffix current: 4.6-5.3 tok/s
- Metal+suffix current: 3.9-4.8 tok/s

## Key Constraints
- Output must be coherent English (no garbled text)
- HC mixing sinkhorn must use CPU exp() for precision
- Expert GEMVs must use CPU (mmap'd, GPU can't page-fault)
- Non-expert GEMVs go through CPU fallback (MLX-Q GPU kernel causes FPU drift)

## Hypotheses
1. Metal overhead from conditional sync checks (~15% gap vs CPU)
2. Thread pool grain size not optimized for Metal path
3. Suffix matching could benefit from longer history window
4. Expert budget could be more aggressive
5. The sync at end of attentionLayer may be unnecessary
6. heapTensorData hash lookups are redundant after first token
7. Fused attention projection could eliminate dispatch overhead
8. CPU SDPA could be parallelized across heads via thread pool

## Iteration Log
(populated by autoresearch loop)
