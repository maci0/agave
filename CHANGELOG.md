# Changelog

## 2026-05-19 — Major Feature Release (59 commits)

### GPU Kernels (32 new files)
- **All quantized GEMV formats now native on all 6 backends** (was 14 gaps)
- ROCm: fused silu_mul, gelu_mul, add_rms_norm kernels
- WebGPU: bf16, f16, fp8_e4m3, fp8_e5m2, q4_1, q5_0, q2_k, q3_k, iq4_nl, iq4_xs
- Vulkan: q4_1, q5_0, q2_k, q3_k, iq4_nl, iq4_xs (+ compiled SPIR-V)
- CUDA: q5_0, q2_k, q3_k, iq4_nl, iq4_xs, fused FFN GELU Q8_0
- CUDA fused FFN activation naming fix (SiLU→GELU correctness for Gemma 3)

### Performance
- Vulkan deferred dispatch: single submit vs ~240 per token
- WebGPU deferred dispatch: batch all compute passes into one encoder
- Paged SDPA staging buffer caching on all 5 GPU backends (zero hot-path allocs)
- Q/K norm, RoPE, QKV GEMV, gate/up FFN batched across all models (barrier reduction)

### Samplers (3 new, API + CLI)
- **XTC** (eXclude Top Choices): diversity via random top-token exclusion
- **DRY** (Don't Repeat Yourself): n-gram sequence repetition penalty
- **Mirostat 2.0**: target-entropy adaptive sampling with dynamic mu
- CLI flags: `--dry-multiplier`, `--xtc-probability`, `--mirostat-mode`, etc.
- All samplers applied consistently across first-token, decode, and spec decode paths

### Speculative Decoding
- **N-gram mode** (`--spec-mode ngram`): zero-overhead spec decode from output history
- **Adaptive cooldown**: skip drafting when acceptance rate drops below 25%
- **Profile-guided adaptive K**: track per-K acceptance, auto-optimize draft length
- All three improvements work together

### Distributed Inference
- **UDP peer discovery**: zero-config LAN discovery (no `--peers` needed)
- **Topology-aware device exchange**: peers swap memory capabilities
- **Peer RTT measurement**: TCP ping-pong after connection

### Server / API
- **Logprobs** in streaming responses (`logprobs`, `top_logprobs`)
- **SSM state prefix caching**: ~2x prefill for Qwen3.5/Nemotron with shared prompts
- **xxHash prefix cache**: RadixTree fast path for repeated prefix queries
- **Vulkan device enumeration** for `--list-devices`

### CLI
- `--ctx-size auto`: probe memory, pick largest safe context
- `--benchmark`: built-in decode benchmark with JSON output
- `--benchmark --json`: machine-readable stats for CI

### Documentation
- PARALLELISM.md: rewritten from 2569-line design doc to 200-line impl reference
- Tutorials improved: chapters 2 (attention), 3 (FFN), 5 (memory), 6 (SSMs),
  7 (sampling), 8 (backends), 17 (spec decode) — worked numerical examples
- KERNELS.md: systematic audit fixed 8+ stale entries, file listings updated
- TODO.md + IDEAS.md merged into single unified document
- 26-item roadmap from vLLM, llama.cpp, Exo, Mesh-LLM analysis

### Testing
- 6 unit tests for new samplers (XTC, DRY, Mirostat)
- 11 fuzz tests for parsers (JSON, GBNF, JSON schema) and samplers
- Test compile fixes for device_id parameter + MockModel
