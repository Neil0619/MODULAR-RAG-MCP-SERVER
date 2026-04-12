---
description: "Generate a spec compliance badge showing current alignment percentage for README display"
---

# Spec Compliance Badge

Generate a compliance badge that shows the current spec alignment percentage. The badge can be embedded in README files, PR descriptions, or dashboards to provide at-a-glance visibility into spec-code alignment.

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty). The user may specify:
- Badge style (e.g., "flat", "flat-square", "for-the-badge", "plastic")
- Custom label (e.g., "SDD Compliance", "Spec Coverage")
- Specific metric to display (e.g., "task completion", "drift score", "requirement coverage")
- Output format (e.g., "markdown", "html", "url-only")

## Prerequisites

1. Confirm you are inside a git repository.
2. Verify `.specify/` directory exists with at least `spec.md`.
3. If `tasks.md` exists, read it to calculate task completion percentage.
4. Run a quick drift check to determine alignment percentage.

## Outline

1. **Calculate Compliance Metrics**: Gather all available compliance data.

   - **Task Completion**: Parse `tasks.md` for completed vs total tasks
   - **Artifact Coverage**: Count existing spec artifacts vs expected set
   - **Requirement Coverage**: Estimate how many spec requirements have matching code
   - Select the primary metric for badge display (default: task completion if tasks.md exists, otherwise artifact coverage)

2. **Determine Badge Color**: Map compliance percentage to color.

   | Range | Color | Meaning |
   |-------|-------|---------|
   | 90-100% | `brightgreen` | Excellent compliance |
   | 75-89% | `green` | Good compliance |
   | 60-74% | `yellow` | Moderate compliance |
   | 40-59% | `orange` | Low compliance |
   | 0-39% | `red` | Critical — major gaps |

3. **Generate Badge Markup**: Produce the badge in requested format.

   ```markdown
   ## Badge Output

   ### Markdown (default)
   ![Spec Compliance](https://img.shields.io/badge/spec_compliance-92%25-brightgreen?style=flat-square)

   ### HTML
   <img src="https://img.shields.io/badge/spec_compliance-92%25-brightgreen?style=flat-square" alt="Spec Compliance: 92%">

   ### Raw URL
   https://img.shields.io/badge/spec_compliance-92%25-brightgreen?style=flat-square
   ```

4. **Generate Multi-Badge Set**: Optionally produce badges for each metric.

   ```markdown
   ## Full Badge Set

   ![Tasks](https://img.shields.io/badge/tasks-14%2F14-brightgreen?style=flat-square)
   ![Artifacts](https://img.shields.io/badge/artifacts-3%2F4-green?style=flat-square)
   ![Drift](https://img.shields.io/badge/drift-87%25_aligned-green?style=flat-square)
   ```

5. **Provide README Integration Instructions**: Show where and how to add the badge.

   ```markdown
   ## Integration

   Add to the top of your README.md, after the project title:

   \```markdown
   # My Project

   ![Spec Compliance](https://img.shields.io/badge/spec_compliance-92%25-brightgreen?style=flat-square)

   Project description here...
   \```
   ```

6. **Output**: Deliver the badge markup and integration instructions.

## Rules

- **Read-only** — never modify README.md or any file, only generate badge markup for the user to add
- **Static badges** — use shields.io static badge format (no server dependency)
- **URL-safe** — properly encode percentages and special characters in badge URLs
- **Accurate** — badge percentage must match the actual compliance calculation, never hardcoded
- **Style-configurable** — support all standard shields.io badge styles
- **Minimal** — generate a single primary badge by default, full set only when requested
