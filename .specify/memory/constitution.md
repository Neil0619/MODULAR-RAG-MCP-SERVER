# Modular RAG MCP Server Constitution

## Core Principles

### I. Modular Architecture

The project is a **modular monolith** with five packages sharing a single source tree:

- `src/core/` — shared types, settings, query engine, trace, response assembly
- `src/ingestion/` — document pipeline (load → chunk → transform → embed → store)
- `src/libs/` — pluggable backends with abstract base + factory pattern
- `src/mcp_server/` — MCP protocol handler and tool definitions
- `src/observability/` — logger, Streamlit dashboard, evaluation runner

**Rule**: New code MUST be placed in the correct package based on its responsibility. When in doubt, ask: "Is this a pluggable backend?" → `libs/`. "Is this pipeline orchestration?" → `ingestion/`. "Is this a shared type or setting?" → `core/`.

### II. Plugin Pattern (base + factory + implementation)

Every pluggable subsystem in `src/libs/` follows a three-file convention:

- `base_*.py` — abstract base class defining the contract
- `*_factory.py` — factory function that routes by config key
- `<provider>_<role>.py` — concrete implementation(s)

**Rule**: When adding a new provider (e.g., a new LLM or embedding backend), create only the implementation file. Do not modify the base class unless the contract genuinely needs extension.

### III. Import Discipline

Imports use bare module paths (e.g., `from core.types import ...`), not `from src.core`. This is configured via `pythonpath = ["src"]` in pytest and the hatch build config.

**Rule**: Never use `from src.<module>` imports. Always use `from <module>.<sub> import ...`.

### IV. Dependency Flow

Module dependencies flow in one direction:

```
mcp_server → core, ingestion
observability → core, ingestion
ingestion → core, libs
core → libs
libs → core (types, settings only)
```

**Rule**: `libs` MUST NOT import from `ingestion`, `mcp_server`, or `observability`. `core` MUST NOT import from `ingestion` or `mcp_server`. Circular imports between `core` ↔ `libs` are tolerated only for types and settings.

### V. Configuration via YAML

Runtime configuration lives in `config/settings.yaml` and is loaded through `src/core/settings.py`. Provider keys (e.g., `llm.provider: "openai"`) route to factory functions.

**Rule**: Hard-coded provider names, model names, or API keys are forbidden. All tunable parameters go through `config/settings.yaml`.

### VI. Observability First

All pipeline operations (ingestion, query) emit trace events via `src/core/trace/`. The dashboard (`src/observability/dashboard/`) renders these traces.

**Rule**: New pipeline stages or significant operations MUST emit trace events. Use `observability.logger.get_logger` for logging, not `logging.getLogger` directly.

### VII. Testing Discipline

Tests are organized in three tiers under `tests/`:

- `tests/unit/` — fast, isolated, no external dependencies (`pytest -m unit`)
- `tests/integration/` — multi-component, may use local files (`pytest -m integration`)
- `tests/e2e/` — full pipeline including MCP protocol (`pytest -m e2e`)

Test files follow `test_<feature>_<scenario>.py` naming.

**Rule**: Every new `libs/` implementation MUST have contract tests that validate the base class interface. Every new pipeline stage MUST have at least one integration test.

### VIII. File Naming Conventions

- Python files: `snake_case.py`
- Abstract bases: `base_<role>.py`
- Factories: `<role>_factory.py`
- Implementations: `<provider>_<role>.py`
- Test files: `test_<feature>_<scenario>.py`

**Rule**: No camelCase or kebab-case in Python filenames.

### IX. Commit Style

Conventional Commits with scope and ticket tag:

```
<type>(<scope>): [<ticket>] <description>
```

Types: `feat`, `fix`, `chore`, `test`, `docs`, `refactor`
Scopes: `loader`, `ingestion`, `query`, `trace`, `dashboard`, `evaluation`, `mcp`, `config`
Ticket: incrementing letter+number (e.g., `J5`, `I2`)

**Rule**: Commits MUST follow this format. Scope must match the primary module affected.

### X. Type Safety

`src/core/types.py` defines shared data types (`RetrievalResult`, etc.) used across all modules.

**Rule**: New domain types MUST be added to `core/types.py` or within the relevant `core/` sub-module. Avoid defining shared types in `libs/` or `ingestion/`.

### XI. No External I/O in Libraries

`src/libs/` implementations MAY call external APIs (LLM providers, vector databases) but MUST NOT read project files, access `config/`, or depend on `ingestion/` pipeline state.

**Rule**: Libraries receive all configuration through constructor arguments or settings objects, not by reading files directly.

### XII. Dashboard Pages Follow a Convention

Each Streamlit dashboard page in `src/observability/dashboard/pages/` follows the same pattern: a `render()` function called from the main app, backed by a service module in `src/observability/dashboard/services/`.

**Rule**: New dashboard pages MUST include both a page module (with `render()`) and a service module. No business logic in page modules.

## Development Workflow

### Quality Gates

- `ruff check src/` must pass (target py311, line-length 120)
- `pytest -m unit` must pass for any PR
- New code must include trace instrumentation where applicable
- No `from src.*` imports allowed

### Test Commands

```bash
pytest -m unit          # Fast unit tests only
pytest -m integration   # Integration tests
pytest -m e2e           # End-to-end tests
pytest                  # All tests
pytest --cov=src        # With coverage
ruff check src/         # Lint
```

## Governance

- Constitution supersedes ad-hoc decisions; amendments require updating this file
- When conventions conflict, the constitution wins
- New modules or package restructuring requires a constitution amendment
- Use `DEV_SPEC.md` for development schedule and feature roadmap

**Version**: 1.0.0 | **Ratified**: 2026-04-13 | **Last Amended**: 2026-04-13
