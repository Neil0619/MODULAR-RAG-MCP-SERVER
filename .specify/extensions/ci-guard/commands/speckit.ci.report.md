---
description: "Generate a machine-readable spec compliance report with requirement coverage metrics"
---

# Spec Compliance Report

Generate a detailed, machine-readable compliance report that maps every spec requirement to its implementation status, test coverage, and file locations. Designed for CI artifact storage, PR comments, and stakeholder dashboards.

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty). The user may specify:
- Output format (e.g., "json", "markdown", "html")
- Report scope (e.g., "requirements only", "full", "summary")
- Target audience (e.g., "engineering", "product", "compliance")
- Specific spec artifact to focus on

## Prerequisites

1. Confirm you are inside a git repository.
2. Verify `.specify/` directory exists with at least `spec.md`.
3. Run `git diff --name-only main...HEAD` (or appropriate base branch) to identify changed files.
4. If `plan.md` exists, read it for technical decisions and architecture context.
5. If `tasks.md` exists, read it for phase completion data.

## Outline

1. **Extract Requirements**: Parse `spec.md` to build a complete requirements list.

   - Extract all items under `## Requirements` section
   - Extract acceptance criteria from `## Success Criteria`
   - Extract user scenarios from `## User Scenarios & Testing`
   - Assign each requirement an ID (REQ-001, REQ-002, etc.)

2. **Map Requirements to Code**: For each requirement, search the codebase for implementing files.

   - Use keyword matching from requirement text against file contents
   - Check git diff for files changed in this branch that relate to each requirement
   - Record file paths and line ranges for each mapping

3. **Map Requirements to Tests**: For each requirement, identify test coverage.

   - Search test directories for test files that exercise each requirement
   - Check if test file names or test descriptions reference the requirement
   - Flag requirements with zero test coverage

4. **Calculate Coverage Metrics**: Compute quantitative compliance scores.

   ```markdown
   ## Coverage Metrics

   | Metric | Value |
   |--------|-------|
   | Total Requirements | 8 |
   | Implemented | 7 (87.5%) |
   | Tested | 6 (75.0%) |
   | Fully Covered (code + tests) | 6 (75.0%) |
   | Gaps | 1 (12.5%) |
   ```

5. **Generate Requirement Traceability Matrix**: Build the full traceability table.

   ```markdown
   ## Requirement Traceability Matrix

   | ID | Requirement | Code | Tests | Status |
   |----|------------|------|-------|--------|
   | REQ-001 | User signup with email | `src/auth/signup.ts` | `signup.test.ts` | ✅ Covered |
   | REQ-002 | JWT token generation | `src/auth/token.ts` | `token.test.ts` | ✅ Covered |
   | REQ-003 | Rate limiting on login | — | — | ❌ Gap |
   ```

6. **Task Phase Summary**: If `tasks.md` exists, summarize phase completion.

   ```markdown
   ## Task Completion by Phase

   | Phase | Total | Done | Progress |
   |-------|-------|------|----------|
   | Phase 1: Setup | 4 | 4 | 100% ✅ |
   | Phase 2: Core | 6 | 5 | 83% ⚠️ |
   | Phase 3: Polish | 3 | 0 | 0% ❌ |
   ```

7. **Gap Analysis**: List all identified gaps with severity.

   ```markdown
   ## Gap Analysis

   | Gap | Severity | Impact |
   |-----|----------|--------|
   | REQ-003: Rate limiting not implemented | High | Security risk |
   | Phase 3 not started | Medium | Polish items pending |
   | No integration tests found | Medium | E2E coverage missing |
   ```

8. **Output Report**: Deliver the full report in the requested format (default: markdown).

   - If JSON requested, wrap all data in a structured JSON object with `metadata`, `requirements`, `coverage`, `gaps`, and `summary` fields
   - If markdown (default), produce a formatted document with all sections above
   - Include a `generated_at` timestamp and `spec_version` reference

## Rules

- **Read-only** — never modify any files, only read and analyze
- **Exhaustive** — cover every requirement found in spec.md, no omissions
- **Traceable** — every claim must reference a specific file path or line
- **Format-flexible** — support markdown (default) and JSON output formats
- **Consistent IDs** — requirement IDs (REQ-001, etc.) must be stable across runs for the same spec
- **Gap-focused** — always highlight what is missing, not just what is present
- **Timestamp all reports** — include generation timestamp for audit trail
