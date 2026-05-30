# Tutorial Claims Audit: Memory/Caching, Sampling, Backends

**Date:** 2025-05-25  
**Scope:** Verify claims from Chapters 5, 7, 8 against actual Agave source code

---

## Evidence Table

| # | Source File | Key Claim | Verdict | Confidence |
|---|-----------|-----------|---------|------------|
| 1 | `src/kvcache/manager.zig:14` | PagedAttention block size default = 16 | **MATCH** | high |
| 2 | `src/models/qwen35.zig:55-59` | Qwen3.5 code defaults: n_layers=32, n_head_kv=4, head_dim=256 | Code defaults differ from tutorial's "9B" figures | high |
| 3 | `src/models/qwen35.zig:149-150,625-629` | Boundary V protection exists, first/last N layers use f16 for V | **MATCH** | high |
| 4 | `src/main.zig:972-973` | Turbo preset sets kv_boundary_v = 2 (first/last 2 layers) | **MATCH** | high |
| 5 | `src/ops/attention.zig:19` | Sparse V threshold = 1e-6 | **MATCH** | high |
| 6 | `src/main.zig:374-378` | --kv-type, --kv-type-k, --kv-type-v CLI flags exist | **MATCH** | high |
| 7 | `src/main.zig:960-969` | --kv-type turbo preset: K=q8_0, V=turbo4 | **MATCH** | high |
| 8 | `src/backend/backend.zig:717-719` | sdpaPaged implemented in Backend union | **MATCH** | high |
| 9 | `src/kvcache/manager.zig:254-361` | RadixTree / RadixAttention implementation exists | **MATCH** | high |
| 10 | `src/main.zig:622,930` | Default top-k=0, top-p=1.0 | **MATCH** | high |
| 11 | `src/main.zig:642,931` | Default min-p=0 | **MATCH** | high |
| 12 | `src/main.zig:624,932` | Default repeat-penalty=1.0 | **MATCH** | high |
| 13 | `src/ops/math.zig:498-500` | temperature==0 → argmax | **MATCH** | high |
| 14 | `src/ops/math.zig:199-203` | DRY sampling exists (applyDry) | **MATCH** | high |
| 15 | `src/ops/math.zig:384-408` | XTC sampling exists (applyXtc) | **MATCH** | high |
| 16 | `src/ops/math.zig:412` | Mirostat 2.0 exists (sampleMirostat) | **MATCH** | high |
| 17 | `src/grammar.zig:1-16` | Grammar-constrained decoding exists (Grammar.parse, maskLogits) | **MATCH** | high |
| 18 | `src/ops/math.zig:242-288` | Frequency + presence penalties (applyPenalties) | **MATCH** | high |
| 19 | `src/backend/backend.zig:584-589` | 6 backends in tagged union: cpu, metal, vulkan, cuda, rocm, webgpu | **MATCH** | high |
| 20 | `src/backend/backend.zig:594-605` | `inline else` dispatch pattern | **MATCH** | high |
| 21 | `src/backend/backend.zig:1040-1150` | Backend auto-selection fallback order | **MATCH** | high |
| 22 | `src/backend/kernels/webgpu/*.wgsl` | WebGPU has 43 shaders | **MATCH** | high |
| 23 | N/A | Sparse V dequant claimed "+22.8% decode throughput" | **NOT FOUND in code** — this is a benchmark result claim, not a code feature | low |

---

## Detailed Findings

### Chapter 5: Memory/Caching

#### 1. KV Cache Per-Token Math (Qwen3.5 9B): PARTIAL MATCH / CANNOT FULLY VERIFY

The tutorial claims: **64 layers × 4 KV heads × 128 dim = 128KB/token at f16**.

The code defaults in `src/models/qwen35.zig` [2] are:
- `n_layers: u32 = 32`
- `n_head_kv: u32 = 4`
- `head_dim: u32 = 256`

These are defaults for the smaller variant; the actual 9B params come from GGUF metadata at load time via `f.getArchU32(arch, "block_count")`, `f.getArchU32(arch, "attention.head_count_kv")`, etc. [2]. The tutorial's math formula structure is correct — the code does compute KV cache size as `n_head_kv * head_dim` per layer [2, line 331: `const kvd: usize = self.n_head_kv * self.head_dim`] — but the specific "64 layers × 4 KV heads × 128 dim" values for the 9B model cannot be verified from code defaults alone. These would need to be checked against the actual Qwen3.5 9B GGUF file metadata.

**Verification of the formula:** 64 × 4 × 128 × 2 bytes(f16) × 2 (K+V) = 131,072 bytes = 128 KB/token. The arithmetic is correct for those parameter values.

#### 2. PagedAttention Block Size Default = 16: **MATCH** [1]

File: `src/kvcache/manager.zig`, line 14:
```zig
const default_block_size: u16 = 16;
```
Comment says: "Default KV cache block size (tokens per block) used across all models." Tests also use 16 as the block size [1, line 331].

#### 3. Boundary V Protection (First/Last 2 Layers at f16): **MATCH** [3][4]

Found in multiple models:
- `src/models/qwen35.zig:149-150`: `kv_boundary_v: u32 = 0` field exists with doc comment: "Number of boundary layers (first/last N) that use f16 V to protect attention quality."
- `src/models/qwen35.zig:625-629`: Logic checks `if (self.kv_boundary_v == 0) return self.kv_type_v` with boundary layer detection.
- Same pattern in `src/models/gemma3.zig:136-137,672-676`, `src/models/gemma4.zig:275-276,1449-1454`, `src/models/llama4.zig:154-155,1007-1011`.
- `src/main.zig:972-973`: Turbo preset sets `kv_boundary_v = 2`:
  ```zig
  .kv_boundary_v = if (res.option("kv-type")) |kv| (if (std.mem.eql(u8, kv, "turbo")) @as(u32, 2) else 0) else 0,
  ```

#### 4. Sparse V Dequant Threshold = 1e-6: **MATCH** [5]

Found consistently across all backends:
- `src/ops/attention.zig:19`: `const sparse_v_threshold: f32 = 1e-6;`
- `src/backend/kernels/cpu/sdpa.zig:10`: `pub const sparse_v_threshold: f32 = 1e-6;`
- `src/backend/kernels/metal/sdpa.metal:23`: `constant float sparse_v_threshold = 1e-6f;`
- `src/backend/kernels/cuda/sdpa.zig:14`: `const sparse_v_threshold: f32 = 1e-6;`
- `src/backend/kernels/rocm/sdpa.zig:12`: `const sparse_v_threshold: f32 = 1e-6;`

The "+22.8% decode throughput" claim is a benchmark/performance result. No such number appears in the source code [23]. This is expected — performance claims wouldn't be in code comments. **Cannot verify from code alone.**

#### 5. --kv-type-k and --kv-type-v CLI Flags: **MATCH** [6]

`src/main.zig:375-378`:
```zig
.{ .long = "kv-type-k", .kind = .option, .help = "KV cache key quantization (overrides --kv-type for keys)." },
.{ .long = "kv-type-v", .kind = .option, .help = "KV cache value quantization (overrides --kv-type for values)." },
.{ .long = "cache-type-k", .kind = .option, .help = "Alias for --kv-type-k." },
.{ .long = "cache-type-v", .kind = .option, .help = "Alias for --kv-type-v." },
```

#### 6. --kv-type turbo Preset: **MATCH** [7]

`src/main.zig:960-969`:
- K: `if (std.mem.eql(u8, kv_str, "turbo")) break :blk KvQuantType.q8_0;`
- V: `if (std.mem.eql(u8, kv_str, "turbo")) break :blk KvQuantType.turbo4;`

Listed in help text at line 147: `"turbo (preset: K=q8_0, V=turbo4)"`.

#### 7. Paged SDPA Implementation: **MATCH** [8]

`sdpaPaged` is defined in the Backend union (`src/backend/backend.zig:717`) with `inline else` dispatch. Implementations exist in:
- `backend/cpu.zig:881` (CpuBackend)
- `backend/metal.zig:2214` (MetalBackend)
- `backend/cuda.zig:1926` (CudaBackend)
- `backend/vulkan.zig:2355` (VulkanBackend)
- `backend/rocm.zig:1309` (RocmBackend)
- `backend/webgpu.zig:1352` (WebGpuBackend)

All 6 model files call `self.be.sdpaPaged(...)`.

#### 8. RadixAttention: **MATCH** [9]

Full implementation in `src/kvcache/manager.zig:200-360`:
- `RadixTree` struct with `init`, `insert`, `matchPrefix`, `deinit`
- `RadixNode` with edge labels, block IDs, children[256], ref_count, last_access
- xxHash64 fast-path cache for repeated prefix queries
- Edge splitting on partial matches
- Comprehensive tests covering insert, match, splitting, empty tree

---

### Chapter 7: Sampling

#### 1. Default Values: **MATCH** [10][11][12]

From `src/main.zig`:
- `temperature`: default 0.0 (line 622: `orelse 0.0`)
- `top_p`: default 1.0 (line 623: `orelse 1.0`) — 1.0 = disabled (includes all tokens)
- `top_k`: default 0 (line 930: `orelse 0`) — 0 = disabled
- `min_p`: default 0 (line 642: `orelse 0.0`) — 0 = disabled
- `repeat_penalty`: default 1.0 (line 624: `orelse 1.0`) — 1.0 = no penalty

**Note:** Tutorial claims default temperature is 0 (disabled/greedy). Code confirms default is 0.0 which maps to argmax.

#### 2. Temperature 0 → Argmax: **MATCH** [13]

`src/ops/math.zig:498-500`:
```zig
pub fn sampleToken(logits: []f32, temperature: f32, top_k: u32, top_p: f32, rng: std.Random) u32 {
    if (logits.len == 0) return 0;
    if (temperature == 0) return argmax(logits);
```
Also confirmed in test at line 754-758: "temperature=0 should return argmax regardless of RNG".

Additionally, `src/main.zig:2849`: `const use_sampling = cli.temperature > 0;` — when temperature is 0, `use_sampling` is false.

#### 3. XTC, DRY, Mirostat: **MATCH** [14][15][16]

All three exist in `src/ops/math.zig`:
- **DRY** (Don't Repeat Yourself): `pub fn applyDry(logits: []f32, recent_ids: []const u32, multiplier: f32, allowed_length: u32)` — line 203
- **XTC** (eXclude Top Choices): `pub fn applyXtc(logits: []f32, xtc_probability: f32, xtc_threshold: f32, rng: std.Random)` — line 387
- **Mirostat 2.0**: `pub fn sampleMirostat(logits: []f32, tau: f32, eta: f32, mu: *f32, temperature: f32, rng: std.Random) u32` — line 412

CLI flags in `src/main.zig:346-352`:
```
--dry-multiplier, --dry-length, --xtc-probability, --xtc-threshold,
--mirostat-mode, --mirostat-tau, --mirostat-eta
```

Fuzz tests in `src/fuzz_tests.zig:136-182` cover all three.

#### 4. Grammar-Constrained Decoding: **MATCH** [17]

`src/grammar.zig` implements GBNF grammar parsing and constrained decoding:
- `Grammar.parse(allocator, gbnf_text)` — parser
- `grammar.maskLogits(&state, logits, vocab)` — constraint application during generation
- `state.acceptToken(token_text)` — state update after sampling
- Called from `src/main.zig:3366`: `grammar_state.?.grammar.maskLogits(&grammar_state.?, logits, vocab_texts);`

#### 5. Frequency and Presence Penalties: **MATCH** [18]

`src/ops/math.zig:242-288`:
```zig
pub fn applyPenalties(logits: []f32, gen_tokens: []const u32, frequency_penalty: f32, presence_penalty: f32) void {
```
- `frequency_penalty`: penalize by `count(token_in_output) * penalty`
- `presence_penalty`: penalize by `1 * penalty` if token appeared at all

Used in server (`server/server.zig:2680-2681`) with clamped values from JSON API (`server/json.zig:64-65,327-328`).

**Note:** These are **server-only** (OpenAI API compatible). The CLI uses `--repeat-penalty` (which is a different mechanism from frequency/presence penalties). The CLI does not expose `--frequency-penalty` or `--presence-penalty` flags directly.

---

### Chapter 8: Backends

#### 1. Six Backends in Tagged Union: **MATCH** [19]

`src/backend/backend.zig:584-589`:
```zig
pub const Backend = union(enum) {
    cpu: *CpuBackend,
    metal: *MetalBackend,
    vulkan: *VulkanBackend,
    cuda: *CudaBackend,
    rocm: *RocmBackend,
    webgpu: *WebGpuBackend,
```

#### 2. Tagged Union Dispatch with `inline else`: **MATCH** [20]

Every method in the Backend union uses the pattern:
```zig
pub inline fn gemv(self: Backend, ...) void {
    switch (self) {
        inline else => |be| be.gemv(...),
    }
}
```
File-level doc comment explicitly describes this: "Uses a tagged union with `inline else` dispatch for zero-overhead backend selection — no VTable indirection in the hot path." [20]

#### 3. Backend Selection/Fallback Order: **MATCH** [21]

`src/backend/backend.zig`, `BackendState.init()` (lines 1040-1150), `.auto` branch:

Auto-detection order:
1. **Metal** (macOS only) → try first on macOS
2. **CUDA** → try if Metal fails or non-macOS
3. **ROCm** → try if CUDA fails
4. **Vulkan** → try if ROCm fails
5. **CPU** → final fallback

Each step uses `catch` to fall through to the next. Explicit fallback messages: `"Metal unavailable ({s}), falling back to CPU"`.

#### 4. WebGPU Has 43 Shaders: **MATCH** [22]

Counted 43 `.wgsl` files in `src/backend/kernels/webgpu/`:
```
add.wgsl, add_rms_norm.wgsl, add_scaled.wgsl, conv1d.wgsl, deinterleave.wgsl,
deltanet_recurrence.wgsl, embedding.wgsl, gelu.wgsl, gelu_mul.wgsl, gemv_bf16.wgsl,
gemv_f16.wgsl, gemv_f32.wgsl, gemv_fp8_e4m3.wgsl, gemv_fp8_e5m2.wgsl, gemv_gptq.wgsl,
gemv_iq4_nl.wgsl, gemv_iq4_xs.wgsl, gemv_mlx_q4.wgsl, gemv_mxfp4_st.wgsl,
gemv_nvfp4_st.wgsl, gemv_q2_k.wgsl, gemv_q3_k.wgsl, gemv_q4_0.wgsl, gemv_q4_1.wgsl,
gemv_q4_k.wgsl, gemv_q5_0.wgsl, gemv_q5_k.wgsl, gemv_q6_k.wgsl, gemv_q8_0.wgsl,
gemv_t_q8_0.wgsl, l2_norm.wgsl, mul.wgsl, rms_norm.wgsl, rms_norm_multi.wgsl,
rope.wgsl, sdpa.wgsl, sdpa_paged.wgsl, sdpa_tree.wgsl, sigmoid_mul.wgsl,
silu.wgsl, silu_mul.wgsl, softmax.wgsl, split_qgate.wgsl
```
The AGENTS.md claim of 43 shaders [22] matches exactly.

---

## Coverage Status

| Area | Status |
|------|--------|
| KV cache block size default | ✅ done — verified = 16 |
| Qwen3.5 9B per-token math | ⚠️ partial — formula structure correct, specific 9B params (64L/4KVH/128dim) cannot be verified from code defaults (code defaults are 32L/4KVH/256dim for smaller variant; actual values come from GGUF metadata) |
| Boundary V protection | ✅ done — exists in qwen35, gemma3, gemma4, llama4 |
| Sparse V threshold | ✅ done — 1e-6 across all backends |
| Sparse V +22.8% throughput | ❌ cannot verify — benchmark claim, not in code |
| CLI flags (kv-type-k, kv-type-v) | ✅ done |
| Turbo preset | ✅ done — K=q8_0, V=turbo4, boundary_v=2 |
| Paged SDPA | ✅ done — all 6 backends |
| RadixAttention | ✅ done — full implementation with tests |
| Sampling defaults | ✅ done — all match |
| Temperature 0 → argmax | ✅ done — verified in code and tests |
| XTC, DRY, Mirostat | ✅ done — all three implemented |
| Grammar-constrained decoding | ✅ done — grammar.zig |
| Frequency/presence penalties | ✅ done — server-side only |
| 6 backends in union | ✅ done |
| `inline else` dispatch | ✅ done |
| Backend fallback order | ✅ done — Metal → CUDA → ROCm → Vulkan → CPU |
| WebGPU 43 shaders | ✅ done — exact count verified |

---

## Sources

1. `src/kvcache/manager.zig` — KV cache manager, PagedKvCache, RadixTree
2. `src/models/qwen35.zig` — Qwen3.5 model definition, default params, boundary V logic
3. `src/main.zig` — CLI argument parsing, defaults, turbo preset, sampling dispatch
4. `src/ops/attention.zig` — Attention ops, sparse V threshold
5. `src/ops/math.zig` — Sampling algorithms (sampleToken, applyDry, applyXtc, sampleMirostat, applyPenalties)
6. `src/backend/backend.zig` — Backend tagged union, dispatch pattern, auto-selection
7. `src/grammar.zig` — GBNF grammar parser and constrained decoding
8. `src/ops/kv_quant.zig` — KV cache quantization types (KvQuantType enum)
9. `src/backend/kernels/webgpu/*.wgsl` — WebGPU shader files (43 total)
10. `src/backend/kernels/cpu/sdpa.zig` — CPU SDPA with sparse V
11. `src/backend/kernels/metal/sdpa.metal` — Metal SDPA with sparse V
12. `src/backend/kernels/cuda/sdpa.zig` — CUDA SDPA with sparse V
13. `src/backend/kernels/rocm/sdpa.zig` — ROCm SDPA with sparse V
14. `AGENTS.md` — Project reference (43 shaders claim source)
15. `src/server/json.zig` — OpenAI API request parsing (frequency/presence penalties)
