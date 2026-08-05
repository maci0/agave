# Task for researcher

Audit Agave tutorial files chapters 08-15 plus two appendices against the actual codebase. For each tutorial file listed below, read the full tutorial content, extract every code reference (file paths, function names, struct names, field names, CLI flags, default values, API patterns), then verify each reference against the actual source files in the repo.

Tutorial files to audit:
- docs/tutorial/08-backends.md
- docs/tutorial/09-cpu-simd-optimization.md
- docs/tutorial/10-memory-safety.md
- docs/tutorial/11-metal-backend-internals.md
- docs/tutorial/12-cpu-parallelism.md
- docs/tutorial/13-batched-dispatch-and-fusion.md
- docs/tutorial/14-format-conventions.md
- docs/tutorial/15-chat-templates.md
- docs/tutorial/appendix-compile-time.md
- docs/tutorial/appendix-atomics.md

For EACH tutorial, produce a structured findings list with:
1. Tutorial filename
2. Each claim/reference checked, what the tutorial says vs what the code actually shows
3. VERDICT per claim: MATCH, MISMATCH, STALE, MISSING, or AMBIGUOUS
4. For mismatches: the actual code state with file path and line reference

Focus especially on:
- File paths mentioned (do they exist?)
- Function/struct names (do they exist in the cited file?)
- CLI flags (do they match src/cli.zig?)
- Default values and constants
- Code snippets (do they reflect current Zig 0.16 API patterns?)
- Backend dispatch patterns (tagged union, inline else)

Write your complete findings to outputs/tutorials-audit-batch2.md

---
Update progress at: /Users/mwysocki/Experiments/agave/.pi-subagents/artifacts/progress/e789eeae/progress.md

---
**Output:**
Write your findings to exactly this path: /Users/mwysocki/Experiments/agave/.pi-subagents/artifacts/outputs/e789eeae/outputs/tutorials-audit-batch2.md
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