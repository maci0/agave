# Review: docs/tutorial/ Unstaged Diff — Completeness & Gaps

## Summary

The diff adds 117 lines across 7 tutorial chapters: CLI quick-reference tables for sampling (Ch07), distributed (Ch22), and server (Ch23); new architecture notes for Llama 4 iRoPE (Ch02), Nemotron-H Mamba-2 (Ch06), WASM (Ch08); and a `pull` subcommand section (Ch00). The additions are generally accurate and well-sourced to source code. However, there are specific gaps, one factual inaccuracy in the peers flag format, a missing MoE mention for Llama 4, a redundancy in Ch22, and several server-relevant flags omitted from Ch23.

## Strengths

- [S1] Ch07 sampling table is complete — every sampling-related `.long =` option in `src/main.zig` (lines 342–358) has a corresponding row. Defaults are verified correct against source.
- [S2] Ch06 Nemotron-H paragraph is technically precise: the layer distribution (21 SSM / 4 attention / 17 FFN in the 8B variant), the detection-by-tensor-probe mechanism, and the Mamba-2 vs DeltaNet distinction are all accurate per `src/models/nemotron_h.zig`.
- [S3] Ch00 `--quant` flag name and pull subcommand usage match `src/pull.zig` exactly (line 3: `Usage: agave pull <org/repo> [--quant Q4_K_M] [--list]`).
- [S4] The iRoPE explanation accurately reflects the source implementation: `nope_interval` default 4 at line 57, NoPE condition `(layer_id + 1) % nope_interval == 0` at line 76, chunk size from `attention.sliding_window` at line 207.

## Weaknesses

### Ch22 — Distributed Inference

- [W1] **MAJOR: `--peers` format is wrong in the flag table.** The table says `--peers HOST` with examples showing bare IPs (`192.168.0.2`), but the CLI help string at `main.zig:367` reads `"TP peer addresses for distributed inference (e.g. 192.168.0.212:9999)"` — the canonical example includes a port. `parsePeerAddr()` (line 1219) parses both `HOST` and `HOST:PORT` formats, with a fallback port. The docs should show `HOST[:PORT]` as the format, not just `HOST`, to match the source's own help text and avoid misleading users about port configuration.

- [W2] **MAJOR: Redundancy — the TP-blocked note duplicates the first Gotcha verbatim.** The new `> **Note:** --tp > 1 is blocked at the CLI today...` blockquote immediately above the Gotchas section says essentially the same thing as the first Gotcha bullet. The Gotcha already existed before this diff. This creates a redundant read experience — a reader hits the same information twice within ~5 lines of scrolling.

- [W3] **MINOR: `--rank` range not documented.** The Gotchas section (pre-existing) already warns that only two ranks are supported, but the new flag table doesn't mention that valid values are `0` or `1` (since max ranks = 2). A reader seeing `--rank N` with no upper bound may try `--rank 2`.

- [W4] **MINOR: `--backend` used in examples but not in the flag table.** The example `agave model.gguf --backend vulkan --device 1` uses `--backend`, which is not listed in the distributed flags table. Either add it or note it's documented in Ch08.

### Ch23 — Server HTTP API

- [W5] **MAJOR: `--mmap` missing from server flags table.** `--mmap` (line 372) is highly server-relevant — a long-running server benefits from lazy mmap to avoid upfront page-in of multi-GB weights. It's a global flag, but the server table already includes global flags like `--no-kv-cache` and `--device`-class flags appear in Ch22, so the scoping precedent is set. At minimum, a note should mention it.

- [W6] **MINOR: Other server-useful flags missing.** Several flags that a server operator would commonly tune are absent from the Ch23 table: `--ctx-size` (context window), `--backend` (compute backend), `--kv-type` / `--kv-type-k` / `--kv-type-v` (KV cache quantization), `--prefill-batch-size`, `--lora`, and `--mmap`. The table header says "server-related flags" which could justify the narrow scope, but a server deployment guide without mentioning context size or KV quantization has a usability gap.

### Ch02 — Llama 4 / iRoPE

- [W7] **MAJOR: MoE routing not mentioned at all.** Llama 4 is an MoE architecture — `src/models/llama4.zig` lines 7–8 explicitly describe "MoE routing with top-1 expert + optional shared expert" and "Dense FFN fallback for layers without a router tensor." The model has `n_experts`, `n_experts_active`, `router_logits`, and `moe_out` fields. The new iRoPE paragraph covers attention innovations but completely omits MoE, which is a defining characteristic of Llama 4. A reader learning about Llama 4 from this chapter would not know it has mixture-of-experts.

### Ch06 — Nemotron-H

- [W8] **MINOR: Target audience clarity on Mamba-2 vs DeltaNet.** The distinction is technically correct but assumes the reader already knows what "selective-state-space recurrence with causal conv1d and discretized dt gating" means. The earlier chapter sections do cover these concepts, but the new paragraph packs the comparison very densely. A brief parenthetical linking back to the Mamba-2 section header above ("see Mamba-2 section above") would help.

### Ch08 — WASM

- [W9] **MINOR: Codegen bug note is vague about failure mode.** "Full forward-pass inference is currently blocked" tells the reader what doesn't work, but not what happens if they try. Does it crash? Produce wrong output? Fail to compile? The source says "Invalid cast in SIMD vector lowering" (line 13 of `wasm_entry.zig`), which is a compilation/codegen error. Saying "fails to compile for wasm32" or "hits a codegen ICE" would be more actionable than "blocked."

- [W10] **MINOR: No mention of what Zig version would fix it.** The note says "Zig 0.16 + LLVM 21" is the problem but doesn't say whether it's tracked, expected to be fixed in 0.17, or a known upstream LLVM issue. A bare problem statement without a resolution path may confuse readers who want to know if they should wait or work around it.

### Ch00 — Pull Subcommand

- [W11] **MINOR: `--list` flag not described with its own sentence.** The code block shows `--list` but the prose only says "selects the best file(s) based on quantization preference." The `--list` flag's purpose (list available files without downloading) is only visible inside a code comment, not in the text.

- [W12] **MINOR: No mention of `HF_TOKEN` or `HF_HOME` env vars.** `main.zig` lines 1717–1718 document these for pull. Private repos require `HF_TOKEN`. Omitting this means readers with private model repos won't know how to authenticate.

## Questions for Authors

- [Q1] Ch22: Should the two-rank limitation be surfaced in the flag table description of `--rank`, or is the existing Gotcha sufficient?
- [Q2] Ch23: Is the intent to list only server-specific flags (narrowly `--serve`, `--port`, etc.) or all flags a server operator should know about? The current table's scope is ambiguous.
- [Q3] Ch02: Is there a separate chapter covering MoE architectures where Llama 4's MoE would be better placed, or should it be here alongside iRoPE?
- [Q4] Ch08: Is the WASM codegen bug tracked in a Zig issue? If so, linking it would make the note actionable.

## Verdict

The diff is a solid quality-of-life improvement. The sampling CLI table (Ch07) is the strongest addition — complete and accurate. The main risks are:

1. **MoE omission for Llama 4** (W7) is the biggest content gap — a defining architectural feature is invisible.
2. **`--peers` format inaccuracy** (W1) could cause real user confusion with port configuration.
3. **Ch22 redundancy** (W2) is a polish issue but noticeable to any linear reader.
4. **`--mmap` missing from Ch23** (W5) is a practical gap for server operators.

Confidence: **high** — all findings are directly verified against source code.

## Revision Plan

1. **[W7] Add MoE to Ch02 Llama 4 paragraph.** One or two sentences: "Llama 4 is also a Mixture-of-Experts model — top-1 expert routing with an optional shared expert per MoE layer, with some layers falling back to dense FFN when no router tensor is present." Reference line 7 of `llama4.zig`.
2. **[W1] Fix `--peers` format in Ch22 table.** Change `--peers HOST` to `--peers ADDR` with description "Peer address, e.g. `192.168.0.2` or `192.168.0.2:9999` (port optional, defaults to discovery port)."
3. **[W2] Remove or rephrase the Ch22 Note blockquote** to avoid repeating the first Gotcha. Could simply say "See gotchas below for current TP limitations."
4. **[W5] Add `--mmap` to Ch23 server table.** One row: `--mmap | | off | Use lazy mmap instead of eagerly paging weights into RAM`.
5. **[W3] Add rank range note.** In Ch22 flag table, append to `--rank` description: "(valid: 0 to pp-1 or tp-1; only 0–1 currently supported)".
6. **[W9] Clarify WASM failure mode in Ch08.** Change "currently blocked by" to "hits a Zig/LLVM codegen crash (Invalid cast in SIMD vector lowering) with".
7. **[W6, W11, W12] Low-priority polish.** Add `--ctx-size` and `--kv-type` to Ch23 if scope is "server operator guide"; add `HF_TOKEN`/`HF_HOME` mention to Ch00 pull section; describe `--list` in prose.

---

## Inline Annotations

> `--peers HOST` | | Peer address (e.g. `192.168.0.2` or `localhost`)

**[W1] MAJOR:** The source CLI help at `main.zig:367` says `"TP peer addresses for distributed inference (e.g. 192.168.0.212:9999)"` — the canonical example includes `HOST:PORT`. `parsePeerAddr()` accepts both formats. The docs should show `HOST[:PORT]` to match the source and avoid omitting port configuration.

> `--rank N` | `0` | This node's rank for TP/PP/disagg

**[W3] MINOR:** The existing Gotchas section says "Only two ranks are supported, full stop" but the flag table gives no upper bound hint. A reader may try `--rank 2` and get silent misbehavior (per Gotcha #2 about rank/world-size mismatch). Add "(0–1 currently)" to the description.

> **Note:** `--tp > 1` is blocked at the CLI today. The model-layer TP code exists but `main.zig` rejects it before it runs. `--pp` and `--disagg` are fully launchable.

**[W2] MAJOR:** This is nearly a verbatim duplicate of the first Gotcha bullet that immediately follows it. The Gotcha reads: "`--tp > 1` is blocked at the CLI today, not just slow or experimental... `--pp` and `--disagg` have no equivalent gate: both are launchable today." Redundancy within ~5 lines.

> **iRoPE (interleaved RoPE)** (Llama 4): Alternates between local layers with standard RoPE and global NoPE layers that skip rotation entirely. [...] See [`src/models/llama4.zig`](../../src/models/llama4.zig).

**[W7] MAJOR:** `src/models/llama4.zig` header (lines 7–8) describes "MoE routing with top-1 expert + optional shared expert" and "Dense FFN fallback for layers without a router tensor." The model has `n_experts`, `n_experts_active`, `router_logits` fields. MoE is a defining feature of Llama 4 and is completely absent from this paragraph.

> Server-related flags from [`src/main.zig`](../../src/main.zig):

**[W5] MAJOR:** The table omits `--mmap` (line 372), which is especially relevant for server deployments where lazy mmap avoids blocking startup on multi-GB weight reads. The table already includes non-server-specific flags like `--no-kv-cache`, so scope isn't a limiting factor.

> Note: full forward-pass inference is currently blocked by a Zig 0.16 + LLVM 21 wasm32 codegen bug; model init, GGUF parsing, and tokenization work.

**[W9] MINOR:** "Blocked" is ambiguous — the source comment in `wasm_entry.zig:13` clarifies it's "Invalid cast in SIMD vector lowering," a codegen crash, not a runtime error or wrong output. Saying "fails at compile time" or "hits a codegen crash" would be more precise.

> `agave model.gguf --backend vulkan --device 1`

**[W4] MINOR:** `--backend` is used in this example but is not listed in the Ch22 distributed flags table above it. Either add it to the table or note it's documented in Ch08.

> Agave can also download models directly from Hugging Face Hub using the `pull` subcommand

**[W12] MINOR:** `src/main.zig` lines 1717–1718 document `HF_TOKEN` (for private repos) and `HF_HOME` (custom cache dir) as env vars used by pull. Private repo users won't know how to authenticate without this.

> **Nemotron-H's Mamba-2 layers** are distinct from Qwen3.5's DeltaNet layers. Where DeltaNet uses the delta rule (error-correcting outer-product updates) for its recurrence, Mamba-2 uses selective-state-space recurrence with causal conv1d and discretized dt (timestep) gating.

**[W8] MINOR:** The comparison is accurate but dense. "Selective-state-space recurrence with causal conv1d and discretized dt gating" packs three concepts into one clause. A backreference "(see Mamba-2 section above)" would help readers who haven't internalized all terms yet.

---

## Sources

All findings verified directly against source files in the working tree:
- `src/main.zig` — CLI option definitions (lines 335–422), help text (lines 1639–1749), parsePeerAddr (line 1219)
- `src/pull.zig` — pull subcommand usage and `--quant` flag (lines 1–15, 248–334)
- `src/models/llama4.zig` — iRoPE, MoE, chunk attention implementation (lines 1–210)
- `src/models/nemotron_h.zig` — Mamba-2/SSM layer detection (lines 1–210)
- `src/wasm_entry.zig` — WASM entry point and codegen bug note (lines 1–14)