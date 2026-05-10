# Agave TODO

Comprehensive list of bugs, missing features, and improvement opportunities.

**Last updated**: 2026-05-10

---

## Bugs

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | GLM-4.7 Flash — degenerate output (also broken in llama.cpp, likely bad GGUF conversion) | Low (upstream) | Won't fix |

---

## GPU Kernel Coverage

All **correctness-critical** kernels for supported model×quant combinations are implemented. Some specialized ops delegate to CPU where noted. See [KERNELS.md](KERNELS.md) for the full per-backend matrix.

| Backend | Core ops | Notes |
|---------|:--------:|-------|
| Metal | Complete | Paged SDPA via CPU fallback |
| CUDA | Complete | DeltaNet delegates to CPU; paged SDPA via CPU fallback |
| Vulkan | Complete | Paged SDPA via CPU fallback; conv1d lacks bias support |
| WebGPU | Core 12 | Phase 1 ops complete; many specialized ops pending (see KERNELS.md) |
| ROCm | Complete | Paged SDPA via CPU fallback |

### Known CPU fallbacks on GPU backends

| Operation | Backends affected | Rationale |
|-----------|-------------------|----------|
| Paged SDPA | All GPU | Native GPU paged SDPA kernels pending |
| DeltaNet recurrence | CUDA, ROCm | Sequential recurrence is register-heavy, not memory-bound |
| NVFP4 GGUF GEMV | All GPU | GPU backends use SafeTensors NVFP4 path instead |

---

## Performance

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| 1 | Q4_K Metal GEMV slower than llama.cpp | Primary decode bottleneck on quantized models | Optimized — group-level x register preload, needs benchmarking |
| 2 | WebGPU decode 0.7 tok/s | Synchronous per-op dispatch overhead | Optimized — lazy readback cache eliminates CPU↔GPU round-trips, needs benchmarking |
| 3 | Gemma 4 E4B CPU prefill ~60s | Very slow, 42 layers with 4.5GB model | Partially optimized — MoE expert gate+up batched via gemvMulti |
| 4 | NVFP4 model accuracy lower than MLX-4bit | May be community quantization quality, not agave bug | Open |

---

## Feature Gaps

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 1 | Tensor/Pipeline parallelism | Design only | `docs/PARALLELISM.md` has 24-section design doc, no code |
| 2 | Structured output / grammar-constrained decoding | Working | GBNF parser, `--grammar-string`, `--grammar`, `--json-output`, `--json-schema`. Full repetition (`*`/`+`/`?`), grouped expressions, JSON schema→GBNF conversion. HTTP API: `grammar` and `json_schema` fields |
| 3 | TriAttention Phase 3 | Not started | Dynamic budget, auto-tune, calibration data generator |
| 4 | Native GPU tree SDPA for CUDA/ROCm/Vulkan | Done | All backends now have native f32 sdpaTree (CPU fallback only for quantized KV) |
| 5 | Batch `forwardTree()` | Fixed | Was hardcoding KV type as f32, now uses model's kv_type_k/v |
| 6 | Direct NVMe-to-VRAM weight loading | Not started | Tiered KV exists, weight loading still CPU-mediated |
| 7 | CUDA fused FFN megakernels (Q4_K/Q5_K/Q6_K variants) | Not started | Only Q8_0 megakernel exists for CUDA |
| 8 | WebGPU Phase 2 (WASM target) | GGUF+tokenizer done | `zig build wasm` parses GGUF, loads tokenizer, inits model. Forward pass blocked by Zig 0.16 + LLVM 21 wasm32 SIMD codegen bug |
| 9 | Native FP4 tensor cores on Blackwell SM121 | Software fallback done | SM121 routing in CUDA backend. Software FP4 GEMV kernel ready. Tensor core MMA path needs hardware testing |
| 10 | GPTQ SafeTensors support | Working | Parser + dequant kernel + GPU GEMV on Metal + CUDA. CPU thread-pool fallback |

---

## Build / CI / Infra

| # | Issue | Status |
|---|-------|--------|
| 1 | `tests/` directory has test harness but golden tests need model files | By design — manual trigger only |

---

## Documentation

No open documentation issues.

---

## KV Cache Quantization Methods

| Method | Rotation | FMAs (d=128) | Params | Storage | CLI |
|--------|----------|:------------:|:------:|---------|-----|
| TurboQuant | WHT-32 butterfly | ~640 | ~640 | f16 norm + packed indices | `tq2/tq3/tq4` |
| **PlanarQuant** | Givens 2D | **256** | 128 | same | `pq2/pq3/pq4` |
| **IsoQuant** | Quaternion 4D | **512** | 128 | same | `iq2/iq3/iq4` |
| **RotorQuant** | Cl(3,0) rotor 3D | **~2,400** | 372 | same | `rq2/rq3/rq4` |

PlanarQuant uses ~2.5x fewer FMAs than TurboQuant. All methods share the same Lloyd-Max codebook and storage format (2.5/3.5/4.5 bits per element). RotorQuant (~2400 FMAs) is more expensive than WHT due to the Clifford algebra sandwich product, but preserves geometric structure.
