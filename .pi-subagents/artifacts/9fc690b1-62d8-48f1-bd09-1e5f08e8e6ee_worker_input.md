# Task for worker

Fill tutorial coverage gaps across multiple chapters. Each change should match the existing tutorial style: inline term definitions in bold on first use, Mermaid diagrams where they help, code references to actual source files, and accurate CLI flag names matching src/main.zig.

Before making any edit, verify the actual code to ensure accuracy. Use `grep` and `read` to check flag names, function signatures, defaults, and struct fields.

## Changes to make

### 1. Ch07 (docs/tutorial/07-sampling.md) — Add CLI flags reference

Near the end of the chapter (before Gotchas), add a section "## CLI Quick Reference" with a table of ALL sampling-related CLI flags and their defaults. Check src/main.zig for the exact flag names and defaults. The flags to include are at minimum:
- `-t` / `--temperature` (default 0)
- `--top-k` (default)
- `--top-p` (default)
- `--min-p` (default)
- `--repeat-penalty` (default)
- `--mirostat-mode` (default)
- `--mirostat-tau` (default)
- `--mirostat-eta` (default)
- `--dry-multiplier` (default)
- `--dry-length` (default)
- `--xtc-probability` (default)
- `--xtc-threshold` (default)
- `--seed` (default)
- `--grammar` / `--grammar-string`
- `--json-schema`

Verify each flag and default in src/main.zig before writing.

### 2. Ch22 (docs/tutorial/22-distributed-inference.md) — Add CLI invocation section

Before the Gotchas section, add a "## CLI Invocation" section showing the actual flags. Check src/main.zig for exact names. Include:
- `--tp N` (note: blocked at CLI today for N>1)
- `--pp N`
- `--rank N`
- `--peers HOST`
- `--transport tcp|shm|nccl`
- `--disagg`
- `--list-devices`
- `--device N`

Show concrete usage examples matching the AGENTS.md quick reference.

### 3. Ch23 (docs/tutorial/23-server-http-api.md) — Add CLI flags

In section 11 (Sleep Mode) or a new "## CLI Quick Reference" before Gotchas, add:
- `--sleep-after N` (seconds)
- `--no-kv-cache`
- `--max-batch-size N`
- `--host` / `--port`
- `--api-key`

Verify in src/main.zig.

### 4. Ch02 (docs/tutorial/02-the-transformer.md) — Add Llama 4 to Attention Variants

Find the "### Attention Variants" table and add a row for Llama 4:
- iRoPE (interleaved RoPE/NoPE layers)
- Chunked attention on local layers

Also add a brief paragraph explaining iRoPE after the table entries. Check src/models/llama4.zig for the actual pattern (alternating local RoPE + global NoPE layers). Keep it concise — 3-5 sentences max.

### 5. Ch06 (docs/tutorial/06-state-space-models.md) — Add Nemotron-H Mamba-2 note

In the hybrid architectures table (around line 388), add Nemotron-H. Check src/models/nemotron_h.zig for the actual layer pattern (how many SSM vs attention vs FFN-only layers). Add one brief paragraph noting Nemotron-H uses Mamba-2 (not DeltaNet like Qwen3.5).

### 6. Ch08 (docs/tutorial/08-backends.md) — Add WASM/browser mention

Find where WebGPU is discussed and add a brief note about the WASM entry point (src/wasm_entry.zig) that enables running in the browser. 2-3 sentences max.

### 7. Ch00 (docs/tutorial/00-getting-started.md) — Add model download mention

In section 1 ("The Model Artifact on Disk"), add a brief note that Agave can download models directly from Hugging Face Hub via `agave pull`:

```
agave pull Qwen/Qwen3.5-0.6B-GGUF
```

Reference src/pull.zig. Keep it to 2-3 sentences.

## Rules
- Verify every flag name and default against src/main.zig before writing
- Verify every struct field and function name against actual source files
- Match existing tutorial style (inline bold definitions, code blocks, tables)
- Do not add new tutorial files — only edit existing ones
- Do not modify content outside the gap areas
- Run `wc -l` on each modified file when done to confirm reasonable size

Report: changed files, commands run with exit codes, validation evidence, and anything left undone.

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