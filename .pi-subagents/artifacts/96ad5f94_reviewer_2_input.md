# Task for reviewer

Review the current unstaged diff in docs/tutorial/ for COMPLETENESS and GAPS.

Run `git diff docs/tutorial/` to see the changes. For each of the 7 modified files, assess:

1. **Ch07 sampling CLI** — Are any sampling-related flags from src/main.zig still missing from the table? Check every `.long =` option definition in the CLI parser related to sampling.
2. **Ch22 distributed** — Are the examples sufficient? Is --rank's range documented? Missing flags?
3. **Ch23 server** — Is --mmap missing (it's server-relevant for memory-mapped weights)? Any other server flags missed?
4. **Ch02 Llama 4** — Does the iRoPE explanation cover enough? Should MoE routing be mentioned (Llama 4 has MoE)?
5. **Ch06 Nemotron-H** — Is the distinction between DeltaNet and Mamba-2 clear enough for the target audience?
6. **Ch08 WASM** — Is the codegen bug note going to be confusing? Should it mention what works vs what doesn't?
7. **Ch00 pull** — Is `--quant Q4_K_M` the right flag name? Does pull actually use that flag?

Also check: does the diff introduce any redundancy with existing content in the same chapters?

List specific findings. Do NOT edit any files.

---
**Output:**
Write your findings to exactly this path: /Users/mwysocki/Experiments/agave/.pi-subagents/artifacts/outputs/96ad5f94/parallel-0/2-reviewer/review.md
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