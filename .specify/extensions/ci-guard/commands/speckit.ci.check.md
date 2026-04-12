---
description: "Run all spec compliance checks and report pass/fail status for CI pipelines"
---

# Spec Compliance Check

Run a comprehensive suite of spec compliance checks against the current branch. Produces a structured pass/fail report suitable for CI pipeline integration, pull request comments, and merge gate enforcement.

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty). The user may specify:
- A specific check to run (e.g., "artifacts only", "drift only", "tasks only")
- A custom threshold (e.g., "80% minimum task completion")
- A target branch for diff comparison (e.g., "compare against develop")
- Output format preference (e.g., "markdown", "json", "summary")

## Prerequisites

1. Confirm you are inside a git repository with at least one commit.
2. Look for a `.specify/` directory in the project root. If it does not exist, report `FAIL: No spec artifacts found` and stop.
3. Check for the existence of core spec artifacts: `spec.md`, `plan.md`, `tasks.md`. Record which are present and which are missing.
4. If a `.speckit-ci.yml` or `.speckit-ci.json` configuration file exists in the project root, read it to load custom thresholds and rules. If none exists, use default thresholds.

## Outline

1. **Artifact Existence Check**: Verify that required spec artifacts exist.

   | Check | Status | Details |
   |-------|--------|---------|
   | spec.md exists | ✅ PASS / ❌ FAIL | Required — defines what to build |
   | plan.md exists | ✅ PASS / ⚠️ WARN | Recommended — defines how to build |
   | tasks.md exists | ✅ PASS / ⚠️ WARN | Recommended — defines step-by-step work |
   | constitution.md exists | ✅ PASS / ⚠️ WARN | Optional — defines project rules |

2. **Artifact Completeness Check**: For each existing artifact, verify it contains the expected sections.

   - **spec.md**: Must have `## Requirements` and `## Success Criteria` sections at minimum
   - **plan.md**: Must have `## Summary` and `## Technical Context` sections at minimum
   - **tasks.md**: Must have at least one `## Phase` section with task items

3. **Task Completion Check**: Parse `tasks.md` and calculate completion percentage.

   - Count total tasks (lines matching `- [ ]` or `- [x]`)
   - Count completed tasks (lines matching `- [x]`)
   - Calculate completion percentage
   - Compare against threshold (default: 100% for merge, configurable)

   ```markdown
   ## Task Completion

   - Total tasks: 14
   - Completed: 12
   - Completion: 85.7%
   - Threshold: 80%
   - Status: ✅ PASS
   ```

4. **Spec-Code Alignment Check**: Compare spec requirements against git diff to verify implementation coverage.

   - Extract requirements from `spec.md` `## Requirements` section
   - Check git diff (against main/target branch) for files that address each requirement
   - Flag requirements with no matching code changes

5. **Drift Detection (Quick Scan)**: Run a lightweight drift check.

   - Compare requirement count in spec.md against implemented feature count
   - Flag any spec sections that reference files which don't exist
   - Flag any TODO/FIXME comments that reference spec requirements

6. **Generate Summary Report**: Combine all check results into a single structured report.

   ```markdown
   ## Spec Compliance Report

   | Category | Status | Score |
   |----------|--------|-------|
   | Artifact Existence | ✅ PASS | 4/4 |
   | Artifact Completeness | ✅ PASS | 3/3 |
   | Task Completion | ⚠️ WARN | 85% (threshold: 80%) |
   | Spec-Code Alignment | ✅ PASS | 6/6 requirements covered |
   | Drift Detection | ✅ PASS | No drift detected |

   **Overall: ✅ PASS** (5/5 checks passed)
   ```

7. **Exit Code Recommendation**: State the recommended CI exit code.

   - All checks pass → Exit 0 (success)
   - Any WARN but no FAIL → Exit 0 with warnings
   - Any FAIL → Exit 1 (failure, block merge)

## Rules

- **Read-only** — never modify any files, only read and report
- **Deterministic** — same inputs must produce same outputs across runs
- **Graceful degradation** — if plan.md or tasks.md are missing, run available checks and report warnings instead of failing entirely
- **Machine-parseable** — always include a structured summary table that CI tools can parse
- **No opinions** — report facts from artifacts and code, never add subjective assessments
- **Threshold-aware** — respect custom thresholds from `.speckit-ci.yml` if present, otherwise use sensible defaults (100% task completion for merge gates)
- **Branch-aware** — compare against the correct base branch (main by default, configurable)
