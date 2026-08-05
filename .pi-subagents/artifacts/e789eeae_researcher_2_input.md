# Task for researcher

Audit Agave tutorial files chapters 16-23 plus remaining appendices against the actual codebase. For each tutorial file listed below, read the full tutorial content, extract every code reference (file paths, function names, struct names, field names, CLI flags, default values, API patterns), then verify each reference against the actual source files in the repo.

Tutorial files to audit:
- docs/tutorial/16-recipe-system.md
- docs/tutorial/17-speculative-decoding.md
- docs/tutorial/18-multi-token-prediction.md
- docs/tutorial/19-pflash-and-block-sparse.md
- docs/tutorial/20-diffusion-lm.md
- docs/tutorial/21-lora.md
- docs/tutorial/22-distributed-inference.md
- docs/tutorial/23-server-http-api.md
- docs/tutorial/appendix-profiling.md
- docs/tutorial/appendix-troubleshooting.md
- docs/tutorial/README.md

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
- Speculative decoding modes and their implementations
- Server API endpoints and parameters
- Distributed inference transport/peer discovery

Write your complete findings to outputs/tutorials-audit-batch3.md

---
Update progress at: /Users/mwysocki/Experiments/agave/.pi-subagents/artifacts/progress/e789eeae/progress.md

---
**Output:**
Write your findings to exactly this path: /Users/mwysocki/Experiments/agave/.pi-subagents/artifacts/outputs/e789eeae/outputs/tutorials-audit-batch3.md
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