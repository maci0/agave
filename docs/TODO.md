# Agave TODO

Comprehensive list of bugs, missing features, and improvement opportunities.

**Last updated**: 2026-05-18

---

## Bugs

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | GLM-4.7 Flash — degenerate output (also broken in llama.cpp, likely bad GGUF conversion) | Low (upstream) | Won't fix |
| 2 | ROCm GEMV Q8_0 produced wrong results | Fixed | Was using dword-packed loads (loadDword + inline for + accumDword) that had incorrect AMDGCN codegen. Replaced with byte-by-byte implementation (same as CUDA). Now 47-55 tok/s on RX 7900 XTX. Other quant types (Q4_K etc) may have similar dword-packed issues |
| 6 | ROCm HSACO target triple rejected by kernel 7.0.6+ | High | Zig generates `amdgcn-amd-amdhsa5.0.0-unknown-gfx1100` instead of `amdgcn-amd-amdhsa--gfx1100`. Kernel 7.0.5 accepted it, 7.0.6 rejects consistently. GEMV kernels are correct (verified at 47-55 tok/s on 7.0.5). Fix needs Zig std library patch for amdgcn target triple |
| 3 | Vulkan segfault on RADV NAVI31 | Fixed | Was descriptor buffer overflow in dispatch() — [4]→[16] for 9-binding pipelines |
| 4 | Vulkan push descriptor crashes on RADV gfx1100 | Medium | VK_KHR_push_descriptor resolves + device created with extension. Descriptor layouts correctly flagged with PUSH_DESCRIPTOR_BIT. GPU crashes during command buffer execution — no validation errors. Suspected RADV driver issue. Infrastructure in place, disabled pending fix |
| 5 | Vulkan synchronous dispatch bottleneck (2.7 tok/s) | Fixed | Deferred dispatch: per-op descriptor set allocation from pool, single command buffer with compute→compute barriers, submit only at sync(). Push descriptors disabled (RADV crash). Needs benchmarking |
| 7 | CUDA K-quant PTX register spilling on sm_121 | Fixed (workaround) | Q4_K/Q5_K/Q6_K PTX kernels spill registers on GB10's sm_121 JIT, causing data corruption (9B) and 10x slowdown (all sizes). Even NR=1 (single row) spills. CPU fallback with thread pool: 6.8 tok/s 9B, 12.3 tok/s 4B. Q8_0/Q4_0 GPU kernels unaffected |

---

## GPU Kernel Coverage

All **correctness-critical** kernels for supported model×quant combinations are implemented. Some specialized ops delegate to CPU where noted. See [KERNELS.md](KERNELS.md) for the full per-backend matrix.

| Backend | Core ops | Notes |
|---------|:--------:|-------|
| Metal | Complete | Native paged SDPA, GPTQ GEMV |
| CUDA | Complete | Native paged SDPA, GPTQ GEMV, fused FFN Q4_K/Q5_K/Q6_K |
| Vulkan | Complete | Native paged SDPA |
| WebGPU | Complete | Native paged SDPA, lazy readback cache |
| ROCm | Complete | Native paged SDPA, GPTQ GEMV |

### Known CPU fallbacks on GPU backends

| Operation | Backends affected | Rationale |
|-----------|-------------------|----------|
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
| 1 | Tensor/Pipeline parallelism | Working | 6 modes: local TP, distributed TP, distributed PP, hybrid TP+PP, disaggregated prefill/decode, dual-GPU same-node. Transports: TCP, POSIX shm (same-node zero-copy), NCCL (RoCE RDMA, 4x ConnectX, IB+GDAKI). `--transport auto/tcp/shm/nccl`, `--device N`, `--peers`. Fix: `cuDevicePrimaryCtxRetain` for NCCL compatibility. Best results: 9B Q8_0 PP=2 NCCL 8.5 tok/s (93% of single GPU), TP=2 NCCL 5.1 tok/s. 27B Q4_K_M PP=2 2.2 tok/s, TP=2 1.7 tok/s. Heterogeneous x86_64+aarch64 |
| 2 | Structured output / grammar-constrained decoding | Working | GBNF parser, `--grammar-string`, `--grammar`, `--json-output`, `--json-schema`. Full repetition (`*`/`+`/`?`), grouped expressions, JSON schema→GBNF conversion. HTTP API: `grammar` and `json_schema` fields |
| 3 | TriAttention Phase 3 | Wired | CLI `--kv-eviction tri`, .cal auto-loading, scorePositionsTri in evictKvCache. Dynamic budget pending |
| 4 | Native GPU tree SDPA for CUDA/ROCm/Vulkan | Done | All backends now have native f32 sdpaTree (CPU fallback only for quantized KV) |
| 5 | Batch `forwardTree()` | Fixed | Was hardcoding KV type as f32, now uses model's kv_type_k/v |
| 6 | Direct NVMe-to-VRAM weight loading | N/A on UMA | cuFile dlopen detection in CUDA backend, GGUF fd exposed. GB10 UMA uses zero-copy mmap (cuMemHostRegister). cuFileRead only benefits discrete GPUs |
| 7 | CUDA fused FFN megakernels (Q4_K/Q5_K/Q6_K variants) | Done | Q8_0 + Q4_K + Q5_K + Q6_K fused gate+up+SiLU. Q5_K/Q6_K use inlined dequant (aliasee workaround). PTX needs regeneration |
| 8 | WebGPU Phase 2 (WASM target) | GGUF+tokenizer done | `zig build wasm` parses GGUF, loads tokenizer, inits model. Forward pass blocked by Zig 0.16 + LLVM 21 wasm32 SIMD codegen bug |
| 9 | Native FP4 tensor cores on Blackwell SM121 | Working | SM121 routing, software FP4 GEMV. NVIDIA official NVFP4 format supported (per-expert loading + weight_scale/weight_scale_2). Tested: Nemotron-Nano-30B NVFP4 @ 26.4 tok/s CUDA GB10 |
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
