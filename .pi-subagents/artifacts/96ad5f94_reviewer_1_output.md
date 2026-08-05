# Style Consistency Review — `docs/tutorial/` unstaged diff

**Reviewer:** AI Research Reviewer (style audit mode)
**Date:** 2026-08-04
**Scope:** 7 files changed across chapters 00, 02, 06, 07, 08, 22, 23

---

## Summary

The diff adds CLI quick-reference tables, code examples, and model-specific technical paragraphs across seven tutorial chapters. The additions are generally well-written and match the project's concise, systems-programmer voice. However, there are several style-consistency issues ranging from table column mismatches between chapters, to a content-redundancy problem in Ch22 where a new blockquote Note repeats the immediately-following Gotcha nearly verbatim, to missing glossary entries for newly introduced terms.

## Strengths

- [S1] **Cross-reference links are correct and consistent.** All new source-file references use the `[src/file.zig](../../src/file.zig)` relative-link pattern. The four referenced files (`src/pull.zig`, `src/models/llama4.zig`, `src/models/nemotron_h.zig`, `src/wasm_entry.zig`) all exist on disk.
- [S2] **Code blocks are consistent.** All bash examples use triple-backtick fencing with the `bash` language tag, matching existing tutorial convention.
- [S3] **Bold inline definitions follow the tutorial pattern** where they appear (e.g., **iRoPE**, **chunked attention**, **temperature scaling** in Ch02; **metadata** reuse in Ch06).
- [S4] **Placement of CLI sections is correct** in Ch07 and Ch23 — both place "CLI Quick Reference" immediately before "Gotchas", matching the established `Content → CLI → Gotchas → Glossary` order visible in the heading structure.
- [S5] **Tone is correct.** New content maintains the concise, systems-programmer voice: no hedging, no marketing, direct declarative statements.

## Weaknesses

- [W1] **MAJOR: Table column inconsistency between Ch22 and Ch07/Ch23.**
  - `docs/tutorial/22-distributed-inference.md` ~line 114: The new CLI table uses a **3-column format** (`Flag | Default | Description`) with no `Short` column.
  - `docs/tutorial/07-sampling.md` ~line 493 and `docs/tutorial/23-server-http-api.md` ~line 86: Both use a **4-column format** (`Flag | Short | Default | Description`).
  - All three are "CLI Quick Reference" tables for the same binary (`agave`). The column structure should be uniform. Even if Ch22 flags have no short forms, the column should be present (with empty cells) for visual consistency, since a reader jumping between chapters will expect the same table shape.

- [W2] **MAJOR: Redundant content — Ch22 Note duplicates the first Gotcha.**
  - `docs/tutorial/22-distributed-inference.md` ~line 150: The new `> **Note:**` blockquote says: _"`--tp > 1` is blocked at the CLI today. The model-layer TP code exists but `main.zig` rejects it before it runs. `--pp` and `--disagg` are fully launchable."_
  - `docs/tutorial/22-distributed-inference.md` ~line 154 (first Gotcha): The existing Gotcha bullet says nearly the same thing with identical phrasing: _"`--tp > 1` is blocked at the CLI today, not just slow or experimental."_ followed by the same explanation.
  - The tutorial does not use blockquote `> **Note:**` callouts elsewhere. This is a stylistic anomaly and a content duplication. Either remove the Note (the Gotcha already covers it) or remove the Gotcha's duplicate wording and cross-reference the CLI section.

- [W3] **MINOR: Ch08 WASM entry breaks the backend naming convention.**
  - `docs/tutorial/08-backends.md` ~line 396: The new entry is titled `**Browser / WASM**` with no parenthetical source file in the bold opener.
  - All other backend entries in the same section follow the pattern `**Name** (\`source_file.zig\`):` — e.g., `**Metal** (\`metal.zig\`)`, `**CUDA** (\`cuda.zig\`)`, `**WebGPU** (\`webgpu.zig\`)`.
  - The WASM entry puts the source file link later in the sentence rather than in the bold title parenthetical. It should be `**Browser / WASM** (\`wasm_entry.zig\`):` to match.

- [W4] **MINOR: Ch00 new section is placed mid-paragraph, breaking the flow.**
  - `docs/tutorial/00-getting-started.md` ~line 16–24: The `pull` subcommand block (prose + code + source link) is inserted between the two bullet points describing artifacts (GGUF / SafeTensors) and the paragraph starting "Agave tells them apart by inspecting the path."
  - That paragraph directly continues the artifact-format discussion — it talks about how Agave distinguishes the two formats. The `pull` download feature is a separate concern (fetching, not loading) and interrupts the artifact → format-detection logical flow. It would be more natural placed after the "Agave tells them apart…" paragraph, or as a subsection.

- [W5] **MINOR: Missing glossary entries for newly introduced terms.**
  - `docs/tutorial/02-the-transformer.md`: Introduces **iRoPE**, **NoPE**, **chunked attention**, and **temperature scaling** (in the Q-vector context). None of these appear in the Ch02 glossary (lines 714+). The chapter consistently glossary-defines other attention variants introduced in the same section (sliding window, attention sinks, etc.).
  - `docs/tutorial/06-state-space-models.md`: Introduces **FFN-only layer** as a new layer type in the Nemotron-H pattern. Not in the Ch06 glossary. The related terms "DeltaNet" and "Mamba-2" are already glossary-defined, but the third layer type in the new paragraph is not.
  - `docs/tutorial/08-backends.md`: Introduces **WASM** / **wasm32 freestanding** as a new backend target. Not in the Ch08 glossary (lines 483+), which defines every other backend technology (CUDA, ROCm, HIP, MSL, WGSL, SPIR-V, etc.).

- [W6] **MINOR: Ch22 section heading level mismatch.**
  - `docs/tutorial/22-distributed-inference.md` ~line 125: `### Examples` is a `###` under `## CLI Invocation`.
  - In Ch07 and Ch23, the CLI Quick Reference sections have no `###` subsection for examples — the code block follows the table directly with no heading.
  - This isn't wrong per se (a `###` under `##` is hierarchically valid), but it's inconsistent with how the same structure is handled in Ch07 and Ch23.

- [W7] **MINOR: Ch02 iRoPE paragraph doesn't follow the established variant-description pattern.**
  - Existing variant descriptions (Per-Head QK Normalization, Sliding Window, Attention Sinks, Sigmoid Gate, Logit Softcapping) all follow the exact pattern:
    `**Name** (Models): Single-sentence description using bold inline parenthetical definitions.`
  - The new iRoPE paragraph spans 4 sentences and is significantly longer and denser than any existing variant description. It reads more like a technical deep-dive paragraph than a variant summary, breaking the uniform rhythm of the section.

## Questions for Authors

- [Q1] Should `--list-devices` and `--device N` be in the Ch22 CLI table? These are general device-selection flags, not distributed-inference-specific. Ch08 (Backends) might be the more natural home, or they could appear in both.
- [Q2] In the Ch06 Nemotron-H paragraph: "21 SSM (Mamba-2) layers on even indices, 4 attention layers at positions 1, 9, 17, 25, and 17 FFN-only layers filling the rest." Is this pattern correct for the 8B variant specifically? The sum is 21+4+17=42 layers, which checks out, but this level of detail may become stale if the model file changes.

## Verdict

The additions are substantively correct and well-written. The two MAJOR issues (W1 table column mismatch, W2 Note/Gotcha redundancy in Ch22) should be fixed before merging because they create visible inconsistency in the most formulaic parts of the tutorial — exactly where a reader expects pattern regularity. The five MINOR issues are polish-level and could be addressed in a follow-up pass.

**Revision priority:** W1 and W2 are quick fixes (add a column, delete duplicate text). W3–W7 are cosmetic but recommended for the merge if time permits.

**Confidence:** High — all findings are based on direct comparison against existing conventions visible in the same files.

## Revision Plan

1. **W1 (table columns):** Add a `Short` column to the Ch22 CLI table, matching Ch07/Ch23. Leave cells empty where no short flag exists.
2. **W2 (redundancy):** Remove the `> **Note:**` blockquote from Ch22 line ~150. The first Gotcha already covers the same information with more detail.
3. **W3 (WASM naming):** Change `**Browser / WASM**: A separate WASM entry point ([...])` → `**Browser / WASM** (\`wasm_entry.zig\`): A separate WASM entry point...` to match other entries.
4. **W4 (placement):** Move the `pull` subcommand block in Ch00 to after the "Agave tells them apart by inspecting the path…" paragraph (after the format-detection discussion concludes).
5. **W5 (glossary):** Add glossary entries for: `iRoPE`, `NoPE`, `chunked attention` (Ch02); `FFN-only layer` (Ch06); `WASM` (Ch08).
6. **W6 (heading):** Either remove the `### Examples` heading in Ch22 and let the code block follow the table directly (matching Ch07/Ch23), or add `### Examples` headings to Ch07/Ch23 for consistency (less preferred — changes more files).
7. **W7 (iRoPE length):** Consider splitting the iRoPE paragraph into a shorter variant-summary line matching the established one-sentence pattern, with the detail pushed into a dedicated subsection or the existing RoPE section.

---

## Inline Annotations

> `| Flag | Default | Description |`
> (docs/tutorial/22-distributed-inference.md, new table header)

**[W1] MAJOR:** This table uses 3 columns while the identical "CLI Quick Reference" tables in Ch07 and Ch23 use 4 columns (`Flag | Short | Default | Description`). Add the `Short` column for cross-chapter consistency, even if all cells are empty.

---

> `> **Note:** \`--tp > 1\` is blocked at the CLI today. The model-layer TP code exists but \`main.zig\` rejects it before it runs. \`--pp\` and \`--disagg\` are fully launchable.`
> (docs/tutorial/22-distributed-inference.md ~line 150)

**[W2] MAJOR:** This is nearly word-for-word identical to the first Gotcha bullet 4 lines later: _"`--tp > 1` is blocked at the CLI today, not just slow or experimental."_ The blockquote `> **Note:**` format is also not used anywhere else in the tutorial. Remove this Note; the Gotcha is the canonical location for this warning.

---

> `**Browser / WASM**: A separate WASM entry point ([`src/wasm_entry.zig`](../../src/wasm_entry.zig)) exports...`
> (docs/tutorial/08-backends.md ~line 396)

**[W3] MINOR:** Every other backend in this section puts the source file in parentheses in the bold title: `**Metal** (\`metal.zig\`)`, `**CUDA** (\`cuda.zig\`)`, etc. This entry should follow the same pattern: `**Browser / WASM** (\`wasm_entry.zig\`):`.

---

> `Agave can also download models directly from Hugging Face Hub using the \`pull\` subcommand...`
> (docs/tutorial/00-getting-started.md ~line 16)

**[W4] MINOR:** This block is inserted between the GGUF/SafeTensors bullet points and the "Agave tells them apart by inspecting the path" paragraph that directly continues the artifact-format discussion. The `pull` feature (downloading) is a different concern from format detection (loading). Moving it after the format-detection paragraph preserves the logical flow.

---

> `**iRoPE (interleaved RoPE)** (Llama 4): Alternates between local layers with standard RoPE and global NoPE layers...` [4 sentences]
> (docs/tutorial/02-the-transformer.md ~line 455)

**[W7] MINOR:** Existing variant descriptions in this section are single sentences. This entry is 4 sentences and significantly denser. Consider shortening to match the one-line pattern and moving the detail elsewhere.

---

> No glossary entries for iRoPE, NoPE, chunked attention, temperature scaling
> (docs/tutorial/02-the-transformer.md, glossary at line 714)

**[W5] MINOR:** All other attention variants introduced in the Attention Variants table have corresponding glossary entries (sliding window, attention sinks, etc.). The four new terms should be added.

---

> `**Nemotron-H's Mamba-2 layers** are distinct from Qwen3.5's DeltaNet layers.`
> (docs/tutorial/06-state-space-models.md ~line 399)

**[W5] MINOR:** Introduces "FFN-only" as a layer type not present in the Ch06 glossary, alongside already-defined "DeltaNet" and "Mamba-2". Add a glossary entry.

---

> `### Examples`
> (docs/tutorial/22-distributed-inference.md ~line 125)

**[W6] MINOR:** Ch07 and Ch23 have no `### Examples` subheading in their CLI Quick Reference sections — code blocks follow the table directly. Remove this heading for consistency, or add matching headings to Ch07 and Ch23.

---

## Sources

All findings based on direct inspection of files in the working tree at `/Users/mwysocki/Experiments/agave/docs/tutorial/`. No external sources consulted.

Files inspected:
- `docs/tutorial/00-getting-started.md`
- `docs/tutorial/02-the-transformer.md`
- `docs/tutorial/06-state-space-models.md`
- `docs/tutorial/07-sampling.md`
- `docs/tutorial/08-backends.md`
- `docs/tutorial/22-distributed-inference.md`
- `docs/tutorial/23-server-http-api.md`
- `src/pull.zig` (existence check)
- `src/wasm_entry.zig` (existence check)
- `src/models/llama4.zig` (existence check)
- `src/models/nemotron_h.zig` (existence check)