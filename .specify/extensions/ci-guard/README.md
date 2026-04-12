# spec-kit-ci-guard

A [Spec Kit](https://github.com/github/spec-kit) extension that adds spec compliance gates to CI/CD pipelines — verify specs exist, check drift, enforce task completion, and block merges when requirements have gaps.

## Problem

Spec Kit generates great artifacts (spec.md, plan.md, tasks.md) — but nothing enforces them. After implementation, there's no automated way to verify:

- Do spec artifacts actually exist before code gets merged?
- Does the implementation match what the spec requires?
- Are all tasks from tasks.md actually completed?
- Has the code drifted from the spec's technical decisions?
- What percentage of spec requirements have matching code and tests?

Teams write detailed specs, then merge code that ignores half the requirements. Without enforcement, SDD becomes documentation theater.

## Solution

The CI Guard extension adds five commands that enforce spec compliance at merge time:

| Command | Purpose | Modifies Files? |
|---------|---------|-----------------|
| `/speckit.ci.check` | Run all spec compliance checks with pass/fail status | No — read-only |
| `/speckit.ci.report` | Generate a requirement traceability matrix with coverage metrics | No — read-only |
| `/speckit.ci.gate` | Configure merge gate rules, thresholds, and profiles | Yes — writes `.speckit-ci.yml` |
| `/speckit.ci.drift` | Detect bidirectional spec-to-code drift with severity ratings | No — read-only |
| `/speckit.ci.badge` | Generate a spec compliance badge for README display | No — read-only |

## Installation

```bash
specify extension add --from https://github.com/Quratulain-bilal/spec-kit-ci-guard/archive/refs/tags/v1.0.0.zip
```

## How It Works

### The Enforcement Gap

```
/speckit.specify    → spec.md      What to build
/speckit.plan       → plan.md      How to build it
/speckit.tasks      → tasks.md     Step-by-step tasks
/speckit.implement  → code         Build it
                    → ???          Did we build what the spec says?

/speckit.ci.check   → compliance   ← CI Guard enforces this
```

### What Gets Checked

**`/speckit.ci.check`** runs the full compliance suite:

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

**`/speckit.ci.report`** generates a full requirement traceability matrix:

```markdown
## Requirement Traceability Matrix

| ID | Requirement | Code | Tests | Status |
|----|------------|------|-------|--------|
| REQ-001 | User signup with email | `src/auth/signup.ts` | `signup.test.ts` | ✅ Covered |
| REQ-002 | JWT token generation | `src/auth/token.ts` | `token.test.ts` | ✅ Covered |
| REQ-003 | Rate limiting on login | — | — | ❌ Gap |

## Coverage Metrics

| Metric | Value |
|--------|-------|
| Total Requirements | 8 |
| Implemented | 7 (87.5%) |
| Tested | 6 (75.0%) |
| Gaps | 1 (12.5%) |
```

**`/speckit.ci.gate`** configures merge rules:

```yaml
# .speckit-ci.yml
version: "1.0"

gates:
  artifacts:
    required: [spec.md, plan.md]
    fail_on_missing_required: true

  task_completion:
    minimum_percentage: 80
    fail_below_threshold: true

  drift_detection:
    enabled: true
    fail_on_drift: false

  test_coverage:
    require_test_files: true
    minimum_requirement_coverage: 75
```

**`/speckit.ci.drift`** detects bidirectional drift:

```markdown
## Drift Score

| Direction | Aligned | Drifted | Score |
|-----------|---------|---------|-------|
| Spec → Code | 7 | 2 | 77.8% |
| Code → Spec | 10 | 3 | 76.9% |
| Decisions | 5 | 1 | 83.3% |
| **Overall** | **22** | **6** | **78.6%** |

## Remediation

| Drift Item | Suggested Action |
|------------|-----------------|
| Rate limiting missing | Implement in code (spec is correct) |
| Password reset undocumented | Add to spec (code is correct) |
| JWT algorithm mismatch | Fix code to match plan.md decision |
```

**`/speckit.ci.badge`** generates compliance badges:

```markdown
![Spec Compliance](https://img.shields.io/badge/spec_compliance-92%25-brightgreen?style=flat-square)
![Tasks](https://img.shields.io/badge/tasks-14%2F14-brightgreen?style=flat-square)
![Drift](https://img.shields.io/badge/drift-87%25_aligned-green?style=flat-square)
```

## Workflow

```
/speckit.specify                     ← Define the feature
       │
       ▼
/speckit.plan                        ← Plan the implementation
       │
       ▼
/speckit.tasks                       ← Generate task list
       │
       ▼
/speckit.implement                   ← Build it
       │
       ▼
/speckit.ci.check                    ← Verify compliance
/speckit.ci.drift                    ← Detect drift
/speckit.ci.report                   ← Generate traceability report
       │
       ▼
Merge with confidence — specs are enforced
```

## Gate Profiles

Three built-in profiles for different team needs:

| Profile | Task Completion | Required Artifacts | Drift Check | Test Coverage |
|---------|----------------|-------------------|-------------|---------------|
| **Strict** | 100% | spec + plan + tasks | Required | Required |
| **Moderate** | 80% | spec + plan | Optional | Recommended |
| **Relaxed** | 50% | spec only | Disabled | Optional |

## Commands

### `/speckit.ci.check`

Runs all spec compliance checks and produces a structured pass/fail report:

- Artifact existence verification (spec.md, plan.md, tasks.md, constitution.md)
- Artifact completeness validation (required sections present)
- Task completion percentage against configurable threshold
- Spec-code alignment check (requirements mapped to changed files)
- Lightweight drift detection
- CI exit code recommendation (0 = pass, 1 = fail)

### `/speckit.ci.report`

Generates a detailed requirement traceability matrix:

- Every spec requirement assigned a stable ID (REQ-001, REQ-002, etc.)
- Each requirement mapped to implementing files and line ranges
- Each requirement mapped to test files and coverage
- Gap analysis with severity ratings
- Phase completion summary from tasks.md
- Machine-readable output (markdown or JSON)

### `/speckit.ci.drift`

Bidirectional drift detection between spec and code:

- **Forward drift**: Spec says it should exist → code doesn't implement it
- **Reverse drift**: Code implements it → spec doesn't document it
- **Decision drift**: Plan says use X → code uses Y
- Quantitative drift score with configurable thresholds
- Remediation suggestions for each drift item
- Severity classification (Critical, Warning, Info)

### `/speckit.ci.gate`

Configure merge gate rules for the project:

- Three preset profiles (strict, moderate, relaxed)
- Custom thresholds for task completion, drift, and test coverage
- Required vs recommended artifact configuration
- Branch protection settings
- CI integration snippets for GitHub Actions, GitLab CI, and generic scripts
- Non-destructive — shows diff before overwriting existing config

### `/speckit.ci.badge`

Generate spec compliance badges for visibility:

- Dynamic compliance percentage based on actual checks
- Color-coded by alignment level (green → red)
- Multiple badge styles (flat, flat-square, for-the-badge)
- Individual badges per metric (tasks, artifacts, drift)
- README integration instructions

## Hooks

The extension registers two optional hooks:

- **before_implement**: Offers to run compliance checks before starting implementation
- **after_implement**: Offers to run drift detection after implementation is complete

## Design Decisions

- **Read-only by default** — 4 of 5 commands never modify files, only analyze and report
- **Bidirectional drift** — checks both spec→code AND code→spec, because specs can be wrong too
- **Profile-based gates** — strict/moderate/relaxed presets prevent analysis paralysis during setup
- **Machine-parseable** — all reports include structured tables that CI tools can parse
- **Deterministic** — same inputs always produce same outputs across runs
- **Graceful degradation** — works with just spec.md if plan.md or tasks.md are missing
- **CI-agnostic** — configuration file is tool-agnostic, with integration snippets for major CI platforms
- **No opinions** — reports facts from artifacts and code, never adds subjective assessments

## Requirements

- Spec Kit >= 0.4.0
- Git >= 2.0.0

## License

MIT
