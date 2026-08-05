# Task for reviewer

Review the current unstaged diff in docs/tutorial/ for CORRECTNESS against the actual codebase.

Run `git diff docs/tutorial/` to see the changes, then verify every claim against the source code:

1. **Ch07 CLI flags table** — verify every flag name, short alias, and default value against src/main.zig. Check the grammar priority claim.
2. **Ch22 CLI flags** — verify --tp, --pp, --rank, --peers, --transport, --disagg, --list-devices, --device against src/main.zig. Check that --transport default is correct (is it 'auto' or something else?). Check default port for --peers.
3. **Ch23 server flags** — verify --serve, --port, --host, --api-key, --sleep-after, --max-batch-size, --no-kv-cache against src/main.zig. Check the default port (49453?), host (127.0.0.1?), and batch size (8?).
4. **Ch02 Llama 4** — verify iRoPE pattern against src/models/llama4.zig: is `(layer_id + 1) % nope_interval == 0` correct? What is default_nope_interval? Is 'temperature scaling' the right term for Q scaling?
5. **Ch06 Nemotron-H** — verify the layer counts (21 SSM, 4 attn, 17 FFN-only) against src/models/nemotron_h.zig.
6. **Ch08 WASM** — verify the exported function names against src/wasm_entry.zig. Is the codegen bug claim current?
7. **Ch00 pull** — verify `agave pull` syntax, the --quant and --list flags against src/main.zig and src/pull.zig.

For each item, state: CORRECT, INCORRECT (with the actual value), or UNVERIFIABLE.

Do NOT edit any files.

---
**Output:**
Write your findings to exactly this path: /Users/mwysocki/Experiments/agave/.pi-subagents/artifacts/outputs/96ad5f94/parallel-0/0-reviewer/review.md
This path is authoritative for this run.
Ignore any other output filename or output path mentioned elsewhere, including output destinations in the base agent prompt, system prompt, or task instructions.

## Acceptance Contract
Acceptance level: attested
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Return concrete findings with file paths and severity when applicable

Required evidence: review-findings, residual-risks

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