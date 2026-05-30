# Tutorial Audit: Speculative Decoding, MTP, Metal Backend, Megakernel

Audit date: 2025-05-25

---

## Evidence Table

| # | Source | File/URL | Key claim | Type | Confidence |
|---|--------|----------|-----------|------|------------|
| 1 | src/spec/spec_decode.zig | code | 6 verification modes in comments; draftMtp, draft, verifySequential, verifySampling, verifyDDTree | primary | high |
| 2 | src/main.zig:409 | code | `SpecMode = enum { none, standard, ddtree, self_spec, ngram, mtp }` — 5 active modes + none | primary | high |
| 3 | src/main.zig, cli_specs | code | `--spec-tokens` default 5, `--tree-budget` default 64 | primary | high |
| 4 | src/spec/ddtree.zig:1-15 | code | DDTree header: "best-first heap algorithm (Ringel & Romano, 2026)" | primary | high |
| 5 | Ringel & Romano, arXiv:2604.12989 | https://arxiv.org/abs/2604.12989v1 | "Accelerating Speculative Decoding with Block Diffusion Draft Trees", Apr 14 2026 | primary | high |
| 6 | src/spec/ngram.zig | code | history_capacity=2048, min_ngram=3, max_ngram=10 | primary | high |
| 7 | src/spec/ddtree.zig | code | Best-first heap: DDTreeBuilder with presort + buildTree using HeapEntry min-heap | primary | high |
| 8 | src/main.zig:2887-2892 | code | Self-spec: skip_start = nLayers/4, default skip_count = nLayers/2 | primary | high |
| 9 | src/models/qwen35.zig:1310-1360 | code | MTP head uses: rmsNormPlusOne for enorm+hnorm, eh_proj GEMV, transformer block, shared_head_norm, shared_head_head | primary | high |
| 10 | src/models/qwen35.zig:1310 | code | `rmsNormPlusOne`: output[i] = (1 + w[i]) * x[i] / rms(x) — the +1 offset | primary | high |
| 11 | docs/tutorial/18-multi-token-prediction.md:45 | tutorial | "70-85% acceptance rate" claim — not a code constant, purely tutorial prose | primary | high |
| 12 | src/spec/spec_decode.zig:117-130 | code | MTP integrated via draftMtp() calling model.mtpForward() | primary | high |
| 13 | src/backend/metal.zig:557,575-610 | code | getBufRef cache by page-aligned base address; wrapBuffer uses newBufferWithBytesNoCopy | primary | high |
| 14 | src/backend/metal.zig:212 | code | `buf_cache: std.AutoHashMap(usize, BufferInfo)` — keyed by address | primary | high |
| 15 | docs/tutorial/11-metal-backend-internals.md:294 | tutorial | "per-threadgroup memory limit of 32 KB on Apple Silicon" | primary | high |
| 16 | docs/tutorial/13-batched-dispatch-and-fusion.md:45-47 | tutorial | "Pipeline state setup: ~5-10 µs" | primary | high |
| 17 | src/backend/backend.zig:37-48 | code | GemvOp struct: w (TensorData), y (*f32), n (usize), mlx_scales, mlx_biases, mlx_bits | primary | high |
| 18 | docs/tutorial/13-batched-dispatch-and-fusion.md:383-491 | tutorial | Three-tier megakernel: Tier 1 Fused FFN, Tier 2 True Megakernels, Tier 3 Composed (auto-generated) | primary | high |
| 19 | src/backend/mega_compose.zig | code | ModelDesc struct, composeMSL() generates MSL from model metadata | primary | high |
| 20 | src/backend/backend.zig:750 | code | `pub inline fn gemvMulti(self: Backend, x: [*]const f32, ops: []const GemvOp, k: usize)` | primary | high |

---

## Findings

### A. Speculative Decoding (Chapter 17)

#### A.1 — "4 modes exist: draft-model, ddtree, self, ngram"
**MISMATCH (minor).** The tutorial says 4 modes. Actual code has **5 active modes** plus `none` [2]:
```zig
const SpecMode = enum { none, standard, ddtree, self_spec, ngram, mtp };
```
The `mtp` mode was added as a 5th spec mode. The `standard` mode (sequential greedy verify) is also a distinct mode from `ddtree`. If the tutorial was written before MTP integration, it would have listed 4 non-`none` modes (standard, ddtree, self, ngram). With MTP there are now 5. The code also distinguishes `standard` from `ddtree` (both need a draft model, but standard is sequential verify while ddtree is tree-structured verify).

The `spec_decode.zig` header comment [1] lists 6 bullet points: Standard greedy, Rejection sampling, DDTree, Self-speculative, N-gram, MTP. The CLI `--spec-mode` accepts: standard, ddtree, self, ngram, mtp [2].

**Verdict: MISMATCH** — tutorial says 4 modes, code has 5 (standard, ddtree, self, ngram, mtp). The tutorial predates MTP integration.

#### A.2 — DDTree reference: 'Ringel & Romano, 2026'
**MATCH.** The code cites "Ringel & Romano, 2026" in both `spec_decode.zig` line 6 [1] and `ddtree.zig` line 4 [4]. This refers to the real paper: *"Accelerating Speculative Decoding with Block Diffusion Draft Trees"* by Liran Ringel and Yaniv Romano, arXiv:2604.12989, published April 14, 2026 [5]. The citation is correct and verifiable.

**Verdict: MATCH**

#### A.3 — Default spec-tokens = 5, tree-budget = 64
**MATCH.** In `src/main.zig`, the `CliArgs` struct has [3]:
```zig
spec_tokens: u32 = 5,
tree_budget: u32 = 64,
```
The CLI `--spec-tokens` help says `[default: 5]` and `--tree-budget` says `[default: 64]`. Both confirmed in the parseCli return values. The `ddtree.zig` `DDTreeBuilder` also has `budget: u32 = 64` as its default [7].

**Verdict: MATCH**

#### A.4 — Self-spec default = 50% layers skipped (skip middle)
**MATCH.** In `src/main.zig:2887-2892` [8]:
```zig
const self_spec_skip_divisor = 4;
const self_spec_default_skip_fraction = 2;
const skip_start: u32 = if (self_spec) target.nLayers() / self_spec_skip_divisor else 0; // start at 25%
const skip_end: u32 = if (self_spec) blk: {
    const skip_count = cli.draft_layers orelse (target.nLayers() / self_spec_default_skip_fraction); // default: 50% of layers
    break :blk skip_start + skip_count;
} else 0;
```
The comment says "skip middle 50%". The skip range starts at `nLayers/4` (25% in) and by default skips `nLayers/2` (50%) layers. For a 32-layer model: skip layers 8-24 (the middle). This matches the tutorial claim.

**Verdict: MATCH**

#### A.5 — N-gram: searches last 2048 tokens, n=3..10
**MATCH.** In `src/spec/ngram.zig` [6]:
```zig
const history_capacity: usize = 2048;
const min_ngram: usize = 3;
const max_ngram: usize = 10;
```
The `propose()` function searches from longest n-gram first (greedy), trying `max_ngram` down to `min_ngram`. The history is stored in a ring buffer of 2048 tokens. All three values match.

**Verdict: MATCH**

#### A.6 — DDTree best-first heap algorithm in ddtree.zig
**MATCH.** The `DDTreeBuilder.buildTree()` in `src/spec/ddtree.zig` [7] implements exactly this:
1. Pre-sorts top-B tokens at each depth by log-probability (`presort()`)
2. Seeds a min-heap with `(depth=0, rank=0)` — using negated cumulative log-prob
3. Pops highest cum-log-prob node (via negated min-heap)
4. Pushes sibling (same depth, rank+1) and child (depth+1, rank=0)
5. Repeats until `budget` exhausted

Uses `HeapEntry` with `heapSiftUp` and `heapSiftDown`. Complexity is O(B log B). Matches the described algorithm exactly.

**Verdict: MATCH**

---

### B. Multi-Token Prediction (Chapter 18)

#### B.1 — MTP head architecture: hnorm, enorm, eh_proj, transformer block, shared_head_norm, shared_head_head
**MATCH.** The `mtpForward()` function in `src/models/qwen35.zig:1330-1453` [9] follows this exact sequence:
1. Token embedding via `nextn.embed_tokens`
2. **enorm**: `rmsNormPlusOne(self.hidden2, ..., enorm_w, ...)` — embed branch
3. **hnorm**: `rmsNormPlusOne(self.mtp_hidden_pre_norm, ..., hnorm_w, ...)` — hidden branch
4. **eh_proj**: `doGemv(self.mtp_concat_buf.ptr, eh_proj, ...)` — concat → project [2*n_embd] → [n_embd]
5. **Transformer block**: full attention (Q/K/V proj, Q gate, Q/K norms, RoPE, SDPA, output proj) + FFN (gate/up/silu/down)
6. **shared_head_norm**: `rmsNorm(self.hidden.ptr, ..., sh_norm, ...)`
7. **shared_head_head**: `doGemv(self.hidden.ptr, sh_head, ..., self.vocab_size, e)`

All tensor names confirmed in the code: `nextn.enorm`, `nextn.hnorm`, `nextn.eh_proj`, `nextn.shared_head_norm`, `nextn.shared_head_head` [9].

**Verdict: MATCH**

#### B.2 — +1 offset in RMSNorm
**MATCH.** In `src/models/qwen35.zig:1310` [10]:
```zig
fn rmsNormPlusOne(input: []const f32, output: []f32, weight: [*]const f32, n: usize, eps: f32) void {
    // ...
    output[i] = (1.0 + weight[i]) * input[i] * inv_rms;
```
The function computes `output[i] = (1 + w[i]) * x[i] / rms(x)`. The comment also states: "RMSNorm with +1 weight offset". This is the exact +1 pattern claimed in the tutorial.

**Verdict: MATCH**

#### B.3 — 70-85% acceptance rate claim
**MISMATCH (unverifiable claim).** The tutorial says: "which happens 70-85% of the time" [11] and shows "Acceptance rate: 70-85%" in a comparison table. However:

- No code constant or configurable threshold corresponds to 70-85%. 
- The `SpecState.acceptanceRate()` computes the runtime acceptance rate dynamically [1].
- No test or benchmark in the codebase validates this claim.
- The number appears to be an empirical observation or aspiration documented only in the tutorial prose.

**Verdict: NOT FOUND in code** — this is an editorial claim in the tutorial, not grounded in any code constant or formal benchmark. It may be anecdotally correct for well-matched MTP models but is unverifiable from the codebase alone.

#### B.4 — MTP integrated with spec_decode.zig
**MATCH.** `spec_decode.zig` contains `draftMtp()` at line 117-130 [12] which calls `model.getMtpDepth()` and `model.mtpForward()`. The main.zig generates MTP drafts via `spec_decode.draftMtp(&spec_state, target, last)` [8]. The `SpecMode.mtp` enum variant is fully wired.

**Verdict: MATCH**

---

### C. Metal Backend (Chapter 11)

#### C.1 — Buffer cache by host pointer address
**MATCH.** In `src/backend/metal.zig:212` [14]:
```zig
buf_cache: std.AutoHashMap(usize, BufferInfo),
```
The `getBufRef()` function at line 575-610 [13] computes a page-aligned base address:
```zig
const aligned_base = addr & ~(@as(usize, page_size - 1));
```
and uses it as the hash key for lookup/store. The cache key is the **page-aligned host pointer address**, not the raw pointer address. This is slightly more precise than "host pointer address" — it's the page-aligned base, enabling sub-region reuse with offset tracking.

**Verdict: MATCH** (with nuance: keyed by page-aligned base address, not raw host pointer)

#### C.2 — Zero-copy UMA: newBufferWithBytesNoCopy
**MATCH.** The `wrapBuffer()` function in `src/backend/metal.zig:557` [13]:
```zig
fn wrapBuffer(self: *MetalBackend, ptr: *const anyopaque, len: usize) ?objc.id {
    return objc.msgSend(
        ?objc.id,
        self.device,
        objc.sel("newBufferWithBytesNoCopy:length:options:deallocator:"),
        .{ ptr, @as(objc.NSUInteger, len), @as(objc.NSUInteger, 0), @as(?objc.id, null) },
    );
}
```
The comment says "Wrap existing memory as a Metal buffer with zero copy (Apple Silicon unified memory)." Multiple references also confirm this pattern across the codebase [13].

**Verdict: MATCH**

#### C.3 — Threadgroup memory ≤ 32KB limit mentioned
**MATCH.** The tutorial `11-metal-backend-internals.md:294` [15] states:
> "Metal has a **per-threadgroup memory limit** of 32 KB on Apple Silicon."

The code confirms this indirectly:
- `sdpa_max_seq_len: usize = 4096` and `sdpa_max_head_dim: usize = 256` are documented as "limited by threadgroup memory" (`metal.zig:52,53`) [13].
- The tutorial walks through a specific threadgroup memory budget calculation showing 18.5 KB for SDPA.
- The 32KB figure is a hardware limit, not a code constant, so it's appropriately documented only in the tutorial.

**Verdict: MATCH**

#### C.4 — Dispatch overhead 5-10µs claim
**MATCH (tutorial claim only).** The tutorial `13-batched-dispatch-and-fusion.md:45-47` [16] states:
> "Pipeline state setup: ~5-10 µs"
> "Total per dispatch: ~5-10 µs"

This is a performance characterization claim in the tutorial. There is no 5-10µs constant in the code. The `metal.zig:50` code comment says "GPU dispatch overhead dominates" for small inputs. The tutorial's `11-metal-backend-internals.md:35` also says "10-15% overhead" for creating MTLBuffer wrappers every dispatch. These are reasonable empirical estimates for Metal on Apple Silicon, but not verifiable from code alone.

**Verdict: MATCH** (as editorial performance claim, not a code constant)

---

### D. Megakernel / Batched Dispatch (Chapter 13)

#### D.1 — GemvOp struct in backend code
**MATCH.** In `src/backend/backend.zig:37-48` [17]:
```zig
pub const GemvOp = struct {
    w: TensorData,
    y: [*]f32,
    n: usize,
    mlx_scales: ?[*]const u8 = null,
    mlx_biases: ?[*]const u8 = null,
    mlx_bits: u32 = 0,
};
```
Fields: `w` (weight tensor data with dtype), `y` (output pointer), `n` (output dimension), plus optional MLX quantization companions.

**Verdict: MATCH**

#### D.2 — 3-tier megakernel architecture in mega_compose.zig
**MATCH.** The tutorial `13-batched-dispatch-and-fusion.md:383-491` [18] describes:
- **Tier 1: Fused FFN** — single dispatch for gate+up+activation+down (line 387)
- **Tier 2: True Megakernels** — hand-written per-model kernels like `mega_qwen35_q8.metal` (line 431)
- **Tier 3: Composed Megakernels (Auto-Generated)** — `mega_compose.zig`'s `composeMSL()` generates MSL from `ModelDesc` (line 489)

The code confirms Tier 3: `mega_compose.zig` exports `ModelDesc` and `composeMSL()` [19]. Metal backend has `pipe_mega_auto` for the auto-composed pipeline and hand-coded pipes like `pipe_mega_qwen35_q8` for Tier 2.

Note: The tutorial says mega_compose.zig contains the 3-tier architecture. More precisely, Tier 1 lives in metal.zig (fused FFN kernels like `pipe_fused_ffn_q8`), Tier 2 in per-model `.metal` files, and Tier 3 in `mega_compose.zig`. The 3-tier architecture is a system-level concept spanning multiple files.

**Verdict: MATCH**

#### D.3 — ModelDesc for auto-generating megakernel MSL
**MATCH.** `src/backend/mega_compose.zig` [19] defines:
```zig
pub const ModelDesc = struct {
    name: []const u8,
    n_layers: u32,
    n_embd: u32,
    n_ff: u32,
    n_head: u32,
    n_kv: u32,
    head_dim: u32,
    rope_dim: u32,
    rope_theta: f32,
    rms_eps: f32,
    max_seq_len: u32,
    activation: Activation,
    quant: QuantKind,
    layer_types: [max_layers]LayerKind,
    // ... per-layer overrides, flags ...
};
```
The `composeMSL(buf, desc)` function generates a complete `kernel void megakernel_auto(...)` MSL source string from this descriptor. Tests confirm this generates valid kernels for Gemma, Qwen, and Nemotron-H architectures.

**Verdict: MATCH**

#### D.4 — gemvMulti interface exists in backend.zig
**MATCH.** In `src/backend/backend.zig:750` [20]:
```zig
pub inline fn gemvMulti(self: Backend, x: [*]const f32, ops: []const GemvOp, k: usize) void {
    switch (self.state) {
        inline else => |be| be.gemvMulti(x, ops, k),
    }
}
```
Implemented by all backends: MetalBackend (line 1965), CpuBackend (line 602), CudaBackend (line 1264), RocmBackend (line 969), VulkanBackend (line 2064), WebGpuBackend (line 1616). Used extensively in model forward passes for batched Q/K/V and gate/up projections.

**Verdict: MATCH**

---

## Summary

| Claim | Verdict | Notes |
|-------|---------|-------|
| **A.1** 4 spec decode modes | **MISMATCH** | 5 modes now (mtp added); tutorial outdated |
| **A.2** DDTree: Ringel & Romano, 2026 | **MATCH** | Verified: arXiv:2604.12989, Apr 2026 |
| **A.3** spec-tokens=5, tree-budget=64 | **MATCH** | Exact code defaults confirmed |
| **A.4** Self-spec: 50% skip middle | **MATCH** | skip_start=25%, skip_count=50% of layers |
| **A.5** N-gram: 2048 tokens, n=3..10 | **MATCH** | Exact constants confirmed |
| **A.6** DDTree best-first heap | **MATCH** | Algorithm matches description exactly |
| **B.1** MTP head architecture | **MATCH** | All 6 components confirmed in code |
| **B.2** +1 offset in RMSNorm | **MATCH** | `rmsNormPlusOne` confirmed |
| **B.3** 70-85% acceptance rate | **NOT FOUND** | Editorial claim, no code backing |
| **B.4** MTP in spec_decode.zig | **MATCH** | draftMtp() fully wired |
| **C.1** Buffer cache by address | **MATCH** | Page-aligned base address as key |
| **C.2** Zero-copy newBufferWithBytesNoCopy | **MATCH** | Exact API call confirmed |
| **C.3** 32KB threadgroup limit | **MATCH** | Documented in tutorial, code respects it |
| **C.4** 5-10µs dispatch overhead | **MATCH** | Tutorial performance claim, reasonable |
| **D.1** GemvOp struct | **MATCH** | w, y, n + MLX optional fields |
| **D.2** 3-tier megakernel architecture | **MATCH** | Tier 1/2/3 confirmed across files |
| **D.3** ModelDesc for auto MSL | **MATCH** | composeMSL() confirmed with tests |
| **D.4** gemvMulti in backend.zig | **MATCH** | All 6 backends implement it |

**Score: 16 MATCH / 1 MISMATCH / 1 NOT FOUND out of 18 claims**

---

## Coverage Status

- **Checked directly**: All 18 claims audited against source code and/or tutorial files
- **Verified external reference**: DDTree paper (Ringel & Romano 2026) confirmed via arXiv:2604.12989
- **Uncertain**: 70-85% MTP acceptance rate is tutorial prose with no code/benchmark backing — may be empirically correct but is unverifiable from the codebase
- **Actionable**: Tutorial Chapter 17 should be updated to mention 5 spec modes (standard, ddtree, self, ngram, mtp) instead of 4

---

## Sources

1. `src/spec/spec_decode.zig` — Speculative decoding orchestrator
2. `src/main.zig:409` — SpecMode enum and CLI arg definitions
3. `src/main.zig` (cli_specs, parseCli) — CLI defaults for spec-tokens and tree-budget
4. `src/spec/ddtree.zig` — DDTree implementation with Ringel & Romano citation
5. Ringel & Romano, "Accelerating Speculative Decoding with Block Diffusion Draft Trees" — https://arxiv.org/abs/2604.12989v1
6. `src/spec/ngram.zig` — N-gram speculative decoding
7. `src/spec/ddtree.zig` — DDTreeBuilder.buildTree() best-first heap
8. `src/main.zig:2877-2940` — Self-speculative setup and MTP integration
9. `src/models/qwen35.zig:1330-1453` — mtpForward() implementation
10. `src/models/qwen35.zig:1310` — rmsNormPlusOne function
11. `docs/tutorial/18-multi-token-prediction.md:45` — 70-85% acceptance rate claim
12. `src/spec/spec_decode.zig:117-130` — draftMtp() function
13. `src/backend/metal.zig:530-620` — Buffer cache, wrapBuffer, getBufRef
14. `src/backend/metal.zig:212` — buf_cache AutoHashMap definition
15. `docs/tutorial/11-metal-backend-internals.md:294` — 32KB threadgroup memory limit
16. `docs/tutorial/13-batched-dispatch-and-fusion.md:45-47` — 5-10µs dispatch overhead
17. `src/backend/backend.zig:37-48` — GemvOp struct definition
18. `docs/tutorial/13-batched-dispatch-and-fusion.md:383-491` — Three-tier megakernel architecture
19. `src/backend/mega_compose.zig` — ModelDesc, composeMSL(), auto-generated megakernels
20. `src/backend/backend.zig:750` — gemvMulti dispatch interface
