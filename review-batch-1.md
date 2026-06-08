# Review Batch 1: Tutorial Chapters 1–4 vs Source Code

## Evidence Table

| # | Source | URL | Key claim verified | Type | Confidence |
|---|--------|-----|--------------------|------|------------|
| 1 | src/format/gguf.zig (bytesPerBlock, blockSize) | local | Q4_K=144B/256el, Q8_0=34B/32el, Q4_0=18B/32el, Q6_K=210B/256el, Q5_K=176B/256el, Q2_K=84B/256el, TQ1_0=64B/256el | primary | high |
| 2 | src/models/qwen35.zig (defaults) | local | 9B: n_embd=4096, n_head=16, n_head_kv=4, head_dim=256, n_ff=12288, n_layers=32, vocab=248320, rope_theta=10M, rope_dim=64 | primary | high |
| 3 | src/models/gemma3.zig (defaults) | local | 1B: n_embd=1152, n_head=4, n_head_kv=1, head_dim=256, n_ff=6912, n_layers=26, rope_theta=1M | primary | high |
| 4 | src/models/gemma4.zig (defaults + E2B) | local | E2B: n_layers=35 (comment), default_n_embd=2816 (26B), default_gl_partial_rotary=0.25 | primary | high |
| 5 | HuggingFace google/gemma-4-E2B config.json | https://huggingface.co/google/gemma-4-E2B/blob/main/config.json | E2B: hidden_size=1536, num_hidden_layers=35, intermediate_size=6144, num_attention_heads=8, num_key_value_heads=1, head_dim=256, sliding_window=512 | primary | high |
| 6 | src/backend/kernels/cpu/rope.zig | local | Split-complex layout: pairs [i, i+half], NOT interleaved [2i, 2i+1] | primary | high |
| 7 | src/backend/kernels/cpu/gemv.zig | local | sparse_threshold = 0.005 | primary | high |
| 8 | src/models/gpt_oss.zig | local | MoE: 32 experts, top-4, softmax routing (lines 666-668: topKExperts then softmax) | primary | high |
| 9 | src/models/qwen35.zig (rope_dim default) | local | rope_dim = 64 (not 78) | primary | high |
| 10 | src/ops/attention.zig | local | sparse_v_threshold = 1e-6 | primary | high |

## Issues Found

---

### [ERROR] 02-the-transformer.md: Gemma4 E2B hidden state size

**Tutorial claims:** `"Token 15496     → embed → [2304 floats]  → 35 layers → [2304 floats]  → norm → [2304 floats]"` and `"Concrete example (Gemma4 E2B, 2.6B parameters)"`

**Source says:** Gemma 4 E2B has `hidden_size=1536` (HuggingFace config.json [5]) and is a ~2B model (the name "E2B" means ~2B). The hidden state is 1536 floats, not 2304. The parameter count is ~2B, not 2.6B.

**Fix:** Replace `[2304 floats]` with `[1536 floats]` throughout the example block. Replace `2.6B parameters` with `2B parameters`. This error also propagates to the text: `"The hidden state (the internal vector representation flowing through each layer) is a fixed-size vector (2304 floats = 9 KB)"` — should be `(1536 floats = 6 KB)`.

---

### [ERROR] 02-the-transformer.md: RoPE dimension pairing visualization

**Tutorial claims:**
```
Original vector:      [x0, x1, x2, x3, x4, x5, x6, x7]
                       └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
                       plane0  plane1  plane2  plane3
```

Also in the Mermaid diagram:
```
D01["dims 0-1"]
D23["dims 2-3"]
D45["dims 4-5"]
D67["dims 6-7"]
```

**Source says:** `src/backend/kernels/cpu/rope.zig` uses **split-complex layout** — pairs `[i, i+half]` where `half = rope_dim/2`. The code comment says: "Uses split-complex layout: pairs `[i, i+half]` are rotated together (matches CUDA convention; NOT interleaved `[2i, 2i+1]`)." For 8 dims: plane0=(x0,x4), plane1=(x1,x5), plane2=(x2,x6), plane3=(x3,x7). [6]

**Fix:** Change the visualization to show split-complex pairing:
```
Original vector:      [x0, x1, x2, x3, x4, x5, x6, x7]
                       │    │    │    │    │    │    │    │
                       └────┼────┼────┼────┘    │    │    │
                            └────┼────┼─────────┘    │    │
                                 └────┼──────────────┘    │
                                      └───────────────────┘
                       plane0  plane1  plane2  plane3
```
Update the Mermaid diagram similarly: `D04["dims 0,4"]`, `D15["dims 1,5"]`, etc.

---

### [ERROR] 02-the-transformer.md: Qwen3.5 partial RoPE dimensions

**Tutorial claims:** `"Some models (Qwen3.5, Nemotron-H) only rotate a subset of dimensions (e.g., first 78 out of 128)"`

**Source says:** `src/models/qwen35.zig` line 63: `rope_dim: u32 = 64`. For the 0.8B model (head_dim=128), this is 64 out of 128, not 78. For the 9B model (head_dim=256), it's 64 out of 256. No variant uses 78. [9]

**Fix:** Change `"first 78 out of 128"` to `"first 64 out of 128"` (or `"first 64 out of head_dim"`).

---

### [ERROR] 03-feed-forward-networks.md: Gemma4 E2B FFN dimensions

**Tutorial claims:** `"the FFN expands the hidden state to a much larger intermediate dimension (e.g., 2304 → 9,216 in Gemma4 E2B)"`

**Source says:** Gemma 4 E2B has `hidden_size=1536` and `intermediate_size=6144` [5]. The expansion is 1536 → 6144, not 2304 → 9,216.

**Fix:** Change to `"(e.g., 1536 → 6,144 in Gemma4 E2B)"`.

---

### [ERROR] 03-feed-forward-networks.md: GPT-OSS MoE routing method (inline comment)

**Tutorial claims:** In the code comment: `"(Qwen 3.5 MoE: softmax+top-8; GPT-OSS: sigmoid+top-4)"`

**Source says:** `src/models/gpt_oss.zig` lines 666-668 show GPT-OSS uses `topKExperts` followed by softmax normalization of selected scores. The MoE table in the same chapter correctly says "GPT-OSS: Softmax" routing. The inline comment contradicts the table. [8]

**Fix:** Change the inline comment to `"(Qwen 3.5 MoE: softmax+top-8; GPT-OSS: softmax+top-4)"`.

---

### [ERROR] 03-feed-forward-networks.md: Mermaid FFN dimensions

**Tutorial claims (Mermaid diagram):**
```
Input["Hidden State\n(e.g. 2304 floats)"] --> Gate["gate_proj\n(2304 → 12288)"]
Input --> Up["up_proj\n(2304 → 12288)"]
...
Down["down_proj\n(12288 → 2304)"]
Down --> Output["FFN Output\n(2304 floats)"]
```

**Source says:** 2304 doesn't match any supported model's `n_embd`. Gemma4 E2B (which the text references) has n_embd=1536, n_ff=6144 [5]. The 12288 is Qwen3.5 9B's n_ff, but Qwen3.5 9B has n_embd=4096, not 2304 [2]. The combination 2304/12288 is fictitious.

**Fix:** Use a real model's dimensions. Either Gemma4 E2B: `1536 → 6144` or Qwen3.5 9B: `4096 → 12288`.

---

## Issues NOT Flagged

- Quantization block sizes and byte counts: all match `gguf.zig` [1]
- GQA table (Gemma3 1B: 4/1, Qwen3.5: 16/4, GPT-OSS: 64/8): matches source [2][3][8]
- Sparse threshold 0.005: matches `gemv.zig` [7]
- Gemma3 uses GELU: confirmed in source [3]
- Gemma3 post-norms: confirmed in source [3]
- Gemma4 global layers 25% partial RoPE: matches `default_gl_partial_rotary = 0.25` [4]
- RoPE formula `x'[i] = x[i]*cos - x[i+half]*sin`: matches `rope.zig` [6]
- GPT-OSS sliding window 128 tokens: matches source [8]
- Qwen3.5 sigmoid gate after SDPA: confirmed in source [2]
- TQ1_0 block size 256 elements, 64 bytes: matches `gguf.zig` [1]
- Nemotron-Nano: 128 experts, top-6, shared expert 2× routed FFN dim: matches source
- `--prefill-batch-size` default 512: matches source

## Coverage Status

- **Checked directly:** All four tutorial files against actual source code for gguf.zig, qwen35.zig, gemma3.zig, gemma4.zig, gpt_oss.zig, nemotron_nano.zig, glm4.zig, rope.zig, gemv.zig, attention.zig, math.zig. Also verified Gemma4 E2B config via HuggingFace.
- **Remaining uncertain:** Exact parameter counts for "2B vs 2.6B" — inferred from model name and hidden_size but not computed from weight shapes.
- **Not checked:** Tutorial Mermaid theme init blocks (excluded per instructions), approximate numbers, simplified examples.

## Sources

1. src/format/gguf.zig — GGMLType.blockSize() and bytesPerBlock()
2. src/models/qwen35.zig — Qwen3.5 model defaults and init
3. src/models/gemma3.zig — Gemma3 model defaults and FFN implementation
4. src/models/gemma4.zig — Gemma4 model defaults and partial rotary factor
5. HuggingFace google/gemma-4-E2B config.json — https://huggingface.co/google/gemma-4-E2B/blob/main/config.json
6. src/backend/kernels/cpu/rope.zig — RoPE implementation (split-complex layout)
7. src/backend/kernels/cpu/gemv.zig — sparse_threshold constant
8. src/models/gpt_oss.zig — GPT-OSS MoE routing and sliding window
9. src/models/qwen35.zig line 63 — rope_dim default = 64
10. src/ops/attention.zig — sparse_v_threshold = 1e-6
