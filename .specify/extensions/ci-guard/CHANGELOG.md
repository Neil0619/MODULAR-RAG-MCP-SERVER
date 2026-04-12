# Changelog

## 1.0.0 (2026-04-10)

- Initial release
- Add `/speckit.ci.check` command for running all spec compliance checks in CI pipelines
- Add `/speckit.ci.report` command for generating machine-readable compliance reports
- Add `/speckit.ci.gate` command for configuring merge gate rules and thresholds
- Add `/speckit.ci.drift` command for detecting spec-to-code drift
- Add `/speckit.ci.badge` command for generating spec compliance badges
- Optional `before_implement` hook for pre-implementation compliance checks
- Optional `after_implement` hook for post-implementation drift detection
- Bridges the gap between SDD workflow and CI/CD pipelines
