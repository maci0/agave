# Task for reviewer

Review the current unstaged diff in docs/tutorial/ for STYLE CONSISTENCY with the existing tutorial conventions.

Run `git diff docs/tutorial/` to see the changes, then check each addition against the established patterns in the surrounding chapters:

1. **Section heading level** — do new sections use the right ## / ### level relative to neighbors?
2. **Table format** — do new tables match the column style of existing tables in the same chapter?
3. **Code block format** — are bash examples consistent (backtick fencing, language tag)?
4. **Bold inline definitions** — are new terms bolded on first use with inline parenthetical definitions, matching the tutorial style?
5. **Cross-references** — do source file references use the `[src/file.zig](../../src/file.zig)` link pattern?
6. **Placement** — are new sections placed logically (e.g., CLI references before Gotchas, not after)?
7. **Tone** — does the new content match the concise, systems-programmer voice of the surrounding text?
8. **Glossary** — should any newly introduced terms be added to the chapter's glossary?

List specific issues with file:line references. Do NOT edit any files.

---
**Output:**
Write your findings to exactly this path: /Users/mwysocki/Experiments/agave/.pi-subagents/artifacts/outputs/96ad5f94/parallel-0/1-reviewer/review.md
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