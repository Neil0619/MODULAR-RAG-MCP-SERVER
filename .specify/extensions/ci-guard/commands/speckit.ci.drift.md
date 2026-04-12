---
description: "Detect spec-to-code drift by comparing requirements against current implementation"
---

# Spec Drift Detection

Analyze the gap between what the spec says should exist and what the code actually implements. Produces a detailed drift report that identifies missing implementations, undocumented features, and spec-code misalignment.

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty). The user may specify:
- A specific requirement to check (e.g., "check REQ-003 only")
- Drift direction (e.g., "spec-to-code", "code-to-spec", "both")
- Severity filter (e.g., "critical only", "all")
- Output format (e.g., "markdown", "json")

## Prerequisites

1. Confirm you are inside a git repository with at least one commit.
2. Verify `spec.md` exists in `.specify/` directory. If missing, stop with error: "Cannot detect drift without spec.md".
3. Read `spec.md` completely to extract all requirements, scenarios, and success criteria.
4. If `plan.md` exists, read it to understand intended architecture and technical decisions.
5. If `tasks.md` exists, read it to understand intended implementation phases.
6. Run `git diff --name-only main...HEAD` to identify files changed in the current branch.

## Outline

1. **Extract Spec Intent**: Parse all spec artifacts to build a complete picture of what should exist.

   - From `spec.md`: Extract requirements, user scenarios, success criteria, and assumptions
   - From `plan.md`: Extract technical decisions, architecture choices, and file structure
   - From `tasks.md`: Extract task items, phases, and dependencies
   - Build a unified "intent model" — the complete list of things the spec says should be true

2. **Scan Implementation**: Analyze the current codebase to determine what actually exists.

   - Read source files referenced in plan.md or tasks.md
   - Check for files/directories that the spec expects to exist
   - Search for functions, classes, endpoints, or features mentioned in requirements
   - Identify test files and their coverage targets

3. **Forward Drift Analysis (Spec → Code)**: Find things the spec requires but code doesn't implement.

   ```markdown
   ## Forward Drift (Spec → Code)

   Requirements defined in spec but missing or incomplete in code:

   | Requirement | Expected | Actual | Severity |
   |-------------|----------|--------|----------|
   | Rate limiting on login | `src/middleware/rateLimit.ts` | File not found | 🔴 Critical |
   | Email verification flow | `src/auth/verify.ts` | Function exists but incomplete | 🟡 Warning |
   | Admin dashboard | `src/pages/admin/` | Directory not found | 🔴 Critical |
   ```

4. **Reverse Drift Analysis (Code → Spec)**: Find things the code implements but spec doesn't mention.

   ```markdown
   ## Reverse Drift (Code → Spec)

   Features implemented in code but not documented in spec:

   | Feature | File | Observation |
   |---------|------|-------------|
   | Password reset flow | `src/auth/reset.ts` | Not in spec requirements |
   | GraphQL endpoint | `src/api/graphql.ts` | Spec only mentions REST |
   | Redis caching layer | `src/cache/redis.ts` | Not in plan.md architecture |
   ```

5. **Decision Drift Analysis**: If `plan.md` exists, verify technical decisions were followed.

   ```markdown
   ## Decision Drift

   | Decision (from plan.md) | Expected | Actual | Aligned? |
   |------------------------|----------|--------|----------|
   | Use PostgreSQL | PostgreSQL driver | ✅ pg package found | ✅ Yes |
   | JWT with RS256 | RS256 signing | ❌ HS256 found in code | 🔴 No |
   | Bcrypt cost factor 12 | cost: 12 | ✅ saltRounds = 12 | ✅ Yes |
   ```

6. **Calculate Drift Score**: Produce a quantitative alignment metric.

   ```markdown
   ## Drift Score

   | Direction | Aligned | Drifted | Score |
   |-----------|---------|---------|-------|
   | Spec → Code | 7 | 2 | 77.8% |
   | Code → Spec | 10 | 3 | 76.9% |
   | Decisions | 5 | 1 | 83.3% |
   | **Overall** | **22** | **6** | **78.6%** |

   Threshold: 80% → ⚠️ BELOW THRESHOLD
   ```

7. **Remediation Suggestions**: For each drift item, suggest the fix direction.

   ```markdown
   ## Remediation

   | Drift Item | Suggested Action |
   |------------|-----------------|
   | Rate limiting missing | Implement in code (spec is correct) |
   | Password reset undocumented | Add to spec (code is correct) |
   | JWT algorithm mismatch | Fix code to match plan.md decision |
   ```

8. **Output Report**: Deliver the complete drift analysis.

## Rules

- **Read-only** — never modify any files, only read and analyze
- **Bidirectional** — always check both directions (spec→code AND code→spec) unless user restricts scope
- **Evidence-based** — every drift claim must cite specific file paths, line numbers, or section references
- **Severity-rated** — classify each drift as Critical (🔴), Warning (🟡), or Info (🔵)
- **Actionable** — every identified drift must include a suggested remediation direction
- **Threshold-aware** — compare drift score against configured threshold from `.speckit-ci.yml` if present
- **Non-judgmental** — report drift direction without assuming which side is "correct" — spec may need updating just as much as code
- **Deterministic** — same codebase state must produce same drift report across runs
