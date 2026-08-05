# Task for worker

Apply exactly these 9 fixes to tutorial files. Do not change anything else. Verify each fix against the source code before writing.

## Fix 1: Ch08 WASM exports (docs/tutorial/08-backends.md)
Find the line that says:
`exports `agave_init`, `agave_generate`, `agave_get_output`, and `agave_free` for calling from JavaScript`
Change to:
`exports `agave_init`, `agave_generate`, `agave_get_output`, `agave_free`, `agave_alloc`, and `agave_dealloc` for calling from JavaScript`
Verify the function names exist in src/wasm_entry.zig first.

## Fix 2: Ch08 WASM naming pattern (docs/tutorial/08-backends.md)
Change `**Browser / WASM**: A separate WASM entry point` to `**Browser / WASM** (`wasm_entry.zig`): A separate WASM entry point`
This matches the pattern used by every other backend entry in the same section (e.g., `**Metal** (`metal.zig`):`, `**CUDA** (`cuda.zig`):`).
Remove the later `([src/wasm_entry.zig](../../src/wasm_entry.zig))` inline link since the file is now referenced in the bold title.

## Fix 3: Ch22 remove Note blockquote (docs/tutorial/22-distributed-inference.md)
Delete the entire blockquote that starts with `> **Note:** \`--tp > 1\` is blocked at the CLI today.` because it duplicates the first Gotcha bullet immediately below it.

## Fix 4: Ch22 add Short column to table (docs/tutorial/22-distributed-inference.md)
Change the CLI flag table header from `| Flag | Default | Description |` to `| Flag | Short | Default | Description |` and add empty Short cells to each row. This matches Ch07 and Ch23 table format.

## Fix 5: Ch22 fix --peers format (docs/tutorial/22-distributed-inference.md)
In the CLI table, change `--peers HOST` to `--peers ADDR` and update the description to: `Peer address (e.g. \`192.168.0.2\` or \`192.168.0.2:9999\`)`

## Fix 6: Ch22 remove ### Examples heading (docs/tutorial/22-distributed-inference.md)
Remove the `### Examples` line. Keep the code block that follows. Ch07 and Ch23 don't use a subheading before their examples.

## Fix 7: Ch02 add Llama 4 MoE mention (docs/tutorial/02-the-transformer.md)
In the iRoPE paragraph, after the sentence ending `See [\`src/models/llama4.zig\`]`, add: `Llama 4 also uses Mixture-of-Experts routing (top-1 with an optional shared expert; some layers fall back to dense FFN when no router tensor is present — see [Chapter 3](03-feed-forward-networks.md)).`

## Fix 8: Ch00 move pull block (docs/tutorial/00-getting-started.md)
Move the entire `agave pull` block (from "Agave can also download models" through the `See [src/pull.zig]` line) to AFTER the paragraph that starts "Agave tells them apart by inspecting the path." This preserves the GGUF/SafeTensors → format detection logical flow, then introduces downloading as a separate concern.

## Fix 9: Ch02 add glossary entries (docs/tutorial/02-the-transformer.md)
Add these entries to the Ch02 glossary section (in alphabetical order among existing entries):

**chunked attention** — An attention variant where each token only attends within a fixed-size chunk, reducing cost from O(n²) to O(chunk²); used by Llama 4 local layers.

**iRoPE (interleaved RoPE)** — Llama 4's attention pattern that alternates between local layers with standard RoPE and global NoPE layers that skip rotation.

**NoPE (No Position Encoding)** — An attention layer that skips rotary position encoding entirely, attending to the full sequence with position-independent Q/K vectors and learned temperature scaling.

## Rules
- Only modify docs/tutorial/ files
- Verify function names against source before writing
- Do not change content outside these 9 fixes
- Report changed files and what was done

## Acceptance Contract
Acceptance level: checked
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Implement the requested change without widening scope
- criterion-2: Return evidence sufficient for an independent acceptance review

Required evidence: changed-files, tests-added, commands-run, residual-risks, no-staged-files

Review gate: required by reviewer.

Finish with a fenced JSON block tagged `acceptance-report` in this shape:
Use empty arrays when no items apply; array fields contain strings unless object entries are shown.
`criteriaSatisfied[].status` must be exactly one of: satisfied, not-satisfied, not-applicable.
`commandsRun[].result` must be exactly one of: passed, failed, not-run.
`manualNotes` and `notes` are optional strings; an empty string means no note and does not satisfy `manual-notes` evidence.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "specific proof"
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "specific proof"
    }
  ],
  "changedFiles": [
    "src/file.ts"
  ],
  "testsAddedOrUpdated": [
    "test/file.test.ts"
  ],
  "commandsRun": [
    {
      "command": "command",
      "result": "passed",
      "summary": "short result"
    }
  ],
  "validationOutput": [
    "validation output or concise summary"
  ],
  "residualRisks": [
    "none"
  ],
  "noStagedFiles": true,
  "diffSummary": "short description of the diff",
  "reviewFindings": [
    "blocker: file.ts:12 - issue found, or no blockers"
  ],
  "manualNotes": "anything else the parent should know"
}
```