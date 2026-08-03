# Tutorial Expansion Design

**Date:** 2026-08-03  
**Status:** Approved for planning  
**Scope:** Full tutorial list (new chapters + medium/low polish) in one coordinated pass, delivered as vertical slices.

## Goal

Close coverage gaps (Getting Started, LoRA, Distributed Inference, Server/HTTP pipeline, Troubleshooting) and raise consistency across all existing chapters (Code Flow, Gotchas, Prerequisites, stronger code linkage) without turning the tutorial series into a CLI or API reference.

## Non-goals

- Expanding or duplicating CLI flag tables, curl cookbooks, or endpoint field matrices inside tutorials
- Renumbering chapters 1–20
- Documenting runtime LoRA injection (Agave does load-time merge only)
- Changing `src/` implementation

## Hard rules

1. **Tutorials teach concepts, algorithms, and code structure.** Product docs (`README`, `API.md`, `PARALLELISM.md`, `ARCHITECTURE.md`, `BENCHMARKS.md`) own invocation, flags, and endpoint schemas.
2. **At most one optional link** to product docs when a chapter needs an "how do I run this" escape hatch. No flag catalogs, no start-server walkthroughs, no curl catalogs.
3. **Teaching blocks are algorithm-level pseudocode** (plain text or fenced `text`). Diagrams use the same pseudocode level. Real Zig is not pasted as the primary example; link to `src/...` with symbol names instead.
4. **Performance numbers** only from `docs/BENCHMARKS.md`. If a claim lacks a source, point at BENCHMARKS or drop the number.
5. **Accuracy:** every Gotcha and pseudocode block must match current `src/` behavior (review mindset from `docs/DOCS_REVIEW_PROMPT.md`).

## Chapter map

| File | Title | Notes |
|------|-------|--------|
| `docs/tutorial/00-getting-started.md` | Getting Started | New on-ramp; systems story of a first run |
| `docs/tutorial/01-*.md` … `20-*.md` | Existing | Numbers unchanged |
| `docs/tutorial/21-lora.md` | LoRA Adapters | Load-time F32 merge via `lora_overrides` |
| `docs/tutorial/22-distributed-inference.md` | Distributed Inference | Concepts over `PARALLELISM.md` |
| `docs/tutorial/23-server-http-api.md` | Server / HTTP API | Pipeline over `API.md` |
| `docs/tutorial/appendix-troubleshooting.md` | Troubleshooting | Symptom → cause → fix |

**Default Next/Back order:** `0 → 1 → … → 20 → 21 → 22 → 23 → appendices`

**Pedagogical reading-path inserts** (README + Prerequisites, not forced linear order):

- After Ch 14 (Formats) → optional detour to Ch 21 (LoRA)
- After Ch 8 / 12 (Backends / CPU Parallelism) → optional detour to Ch 22 (Distributed)
- After Ch 7 / 15 (Sampling / Chat Templates) → optional detour to Ch 23 (Server)
- Ch 0 is the default start for all reading paths

**Product docs:** add one-line "Tutorial:" cross-links only. Do not merge tutorial narrative into product docs or slim product docs into stubs.

## Shared chapter template

Every chapter (0, 1–20, 21–23) follows this skeleton. Appendices use a lighter variant.

Required sections, in order:

1. Title (`# Chapter N: Title`)
2. **Prerequisites** (omit only for Ch 0) and **Time**
3. One-sentence promise (blockquote)
4. Body sections for the topic
5. **Code Flow**: Mermaid flowchart with pseudocode-level node labels (no Zig identifiers in nodes), plus 2–4 sentences of prose
6. **Gotchas**: 1–2 pitfalls grounded in real behavior
7. **How This Relates to the Code**: markdown link to `src/…` with symbol name, plus a short fenced `text` pseudocode block (not a Zig dump)
8. **Next / Back** (and optional **Product docs** link)
9. **Glossary**

### Template details

| Element | Requirement |
|---------|-------------|
| Prerequisites | All chapters except 0; list conceptual deps |
| Code Flow | ≥1 Mermaid diagram, pseudocode-level nodes |
| Gotchas | 1–2 per chapter; prefer AGENTS.md topics when they match |
| In the code | Link + symbol name + short pseudocode; no multi-line Zig dumps |
| Performance sidebar | Chapters that cite speedups (esp. 4, 9, 11, 13); table from BENCHMARKS only |
| Ch 7 configs | Use-case table of **sampling parameters** (temperature, top-p, etc.), not CLI flags |
| Math appendix | "When to Use What" decision tree or table |
| Troubleshooting | Symptom → likely cause → fix; may point to product docs for invocation errors without becoming a flag list |

## New chapter outlines

### Chapter 0: Getting Started (~15 min)

Mental model of a first inference run as a systems pipeline:

1. Model artifact on disk (GGUF / directory) and arch detection
2. Load weights and runtime buffers
3. Tokenize prompt
4. Prefill → decode loop → sample → detokenize
5. What the output stream represents (tokens, timing as pipeline artifacts)
6. Where the rest of the tutorial series goes (reading paths)

Code Flow: `load → tokenize → prefill → decode loop → sample → text`  
Gotchas: reading GPU logits without sync; arch / format mismatch on load  
Escape hatch: link to project README for building and running (no CLI tour)

### Chapter 21: LoRA Adapters (~12 min)

1. Why adapters exist (specialize without full retrain)
2. GGUF LoRA tensor layout (`adapter.type`, alpha, `lora_a` / `lora_b`)
3. Merge math at **load time**: `W' = dequant(W) + (α/r) · B · A`
4. Override map: merged F32 stored so `getTensor()` returns overrides; no hot-path LoRA math
5. Limits: memory cost of F32 overrides; inconsistent A/B pairs skipped

Code Flow: `open adapter → match base tensors → dequant → merge → override map`  
Source of truth: `src/lora.zig`  
Gotchas: wrong adapter metadata; rank mismatch; merge memory spike  
Escape hatch: README / CLAUDE quick reference for invocation only

### Chapter 22: Distributed Inference (~20 min)

1. When one device is not enough
2. Tensor parallelism vs pipeline parallelism vs hybrid vs disaggregated prefill/decode
3. Transport concepts: TCP, same-node shared memory, NCCL; auto-select intuition
4. Weight/layer sharding and collective / send-recv roles in the forward pass
5. Device enumeration as a concept (discovery), not a flag walkthrough

Code Flow: `init transport → shard → forward with all-reduce or stage send/recv`  
Sources: `src/parallel/`, `src/devices/discovery.zig`  
Gotchas: world-size / rank inconsistency; transport choice vs topology  
Product doc: `PARALLELISM.md` for launch recipes

### Chapter 23: Server / HTTP API (~18 min)

1. HTTP as a front end to the same generation loop
2. Request → session / KV → generate → buffered JSON or SSE
3. How structured output / grammar attaches to sampling
4. Tool calling and vision as content/pipeline extensions (concepts + links)
5. Prefix caching, idle sleep, prefill-only modes: why they exist in a server
6. Speculative decoding composed with server sessions (behavioral interaction)

Code Flow: `accept → parse → session/KV → generate → stream or JSON`  
Sources: `src/server/server.zig`, `src/server/json.zig`  
Gotchas: default greedy sampling behavior; streaming vs non-streaming response shape differences  
Product doc: `API.md` for endpoints and field tables

### Appendix: Troubleshooting

Symptom-oriented. Each entry: symptom → likely cause → fix or chapter link.

Cover at least: OOM / context too large, garbage or wrong output, backend selection confusion, quantization quality artifacts, GGUF vs SafeTensors mismatch, known TEST_MATRIX failures (e.g. GLM-4 GGUF note), distributed hang / rank mismatch, server error classes. Invocation fixes may cite product docs by name without listing flags.

## Polish checklist (chapters 1–20)

Apply the shared template to every existing chapter. Seed Gotchas (verify against code before writing):

| Ch | Focus |
|----|--------|
| 1 | Code Flow token→id→embed; denser diagram if sparse |
| 2 | Code Flow QKV→attn→out; RoPE/GQA shape pitfalls |
| 3 | Strengthen sparse diagram; FFN/MoE Code Flow |
| 4 | In-kernel dequant Code Flow; Performance from BENCHMARKS |
| 5 | Cache write/read Code Flow; inverse RoPE / paged blocks |
| 6 | Stronger diagram; SSM step vs attention |
| 7 | Common Configurations (algo params); sampling Code Flow |
| 8 | Tagged-union dispatch pseudocode; GPU sync before argmax |
| 9 | SIMD Code Flow; Performance sidebar |
| 10 | alloc/defer/errdefer Code Flow |
| 11 | Metal path Code Flow; 32KB threadgroup; Performance |
| 12 | Thread pool Code Flow; no manual spawn |
| 13 | gemvMulti / fusion Code Flow; Performance; `@hasDecl` megakernel guard |
| 14 | Format detect→map Code Flow; SafeTensors U32 ambiguity |
| 15 | messages→template→ids Code Flow |
| 16 | recipe resolve Code Flow; override precedence |
| 17 | draft→verify→rollback Code Flow; KV rollback after reject |
| 18 | MTP draft/verify Code Flow; +1 offset norm |
| 19 | score→threshold→sparse prefill; alpha too aggressive |
| 20 | mask→denoise→accept; canvas vs AR |

**Appendices:** math gets "When to Use What"; compile-time / profiling / atomics get light Prerequisites + Gotchas + pseudocode linkage where thin.

**README / DOCUMENTATION.md:**

- List Ch 0, 21–23, troubleshooting appendix
- Add **Beginner Systems Programmer** reading path (gentler than ML Beginners: C-comfortable, Zig-new)
- Add **Quick Reference** path: feature → chapter list (conceptual index, not CLI)
- Reading paths describe what to read, not commands to run

## Delivery slices

1. **Scaffold:** Commit this design; create stub files for 0/21–23 + troubleshooting with titles, Prerequisites, Next/Back; wire README and DOCUMENTATION.md
2. **New chapters:** Full prose for 0, 21, 22, 23 + troubleshooting
3. **Polish A:** Chapters 1–10 + math appendix
4. **Polish B:** Chapters 11–20 + remaining appendices + README paths + product-doc tutorial links

## File inventory

**Create**

- `docs/tutorial/00-getting-started.md`
- `docs/tutorial/21-lora.md`
- `docs/tutorial/22-distributed-inference.md`
- `docs/tutorial/23-server-http-api.md`
- `docs/tutorial/appendix-troubleshooting.md`
- `docs/superpowers/specs/2026-08-03-tutorial-expansion-design.md` (this file)

**Modify**

- `docs/tutorial/README.md`
- `docs/DOCUMENTATION.md`
- `docs/tutorial/01-*.md` through `20-*.md`
- `docs/tutorial/appendix-math.md` and other appendices as needed
- `docs/PARALLELISM.md`, `docs/API.md`, `docs/ARCHITECTURE.md` (tutorial link lines only)

## Success criteria

- Every chapter has Prerequisites (except 0), Code Flow, Gotchas, link-based "In the code" with pseudocode, Glossary, Next/Back
- No tutorial chapter functions as a CLI or API reference
- Coverage exists for Getting Started, LoRA (load-time), Distributed concepts, Server pipeline, Troubleshooting
- Pseudocode and Gotchas match `src/`; performance claims match `BENCHMARKS.md`
- README offers Beginner Systems Programmer and Quick Reference paths

## LoRA factual note

`src/lora.zig` merges adapters at load time into F32 overrides on the base GGUF (`lora_overrides`). Hot path sees merged weights via `getTensor()` with no per-token LoRA math. Tutorial text must describe this, not "runtime injection into the forward pass."
