---

description: "Task list template for feature implementation"
---

# Tasks: [FEATURE NAME]

**Input**: Design documents from `/specs/[###-feature-name]/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Include test tasks when the spec requests tests. Use pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- All source code lives under `src/<module>/`
- Tests live under `tests/<tier>/`
- Config lives in `config/settings.yaml`
- Import using bare module paths: `from core.types import ...` (NOT `from src.core`)

## Test Commands

```bash
pytest -m unit              # Fast unit tests only
pytest -m integration       # Integration tests (multi-component)
pytest -m e2e               # End-to-end tests (full pipeline)
pytest tests/unit/test_X.py # Run a specific test file
pytest --cov=src            # With coverage report
ruff check src/             # Lint check (line-length 120, target py311)
```

<!--
  ============================================================================
  The tasks below are SAMPLE TASKS showing project-specific patterns.
  Replace with actual tasks based on spec.md user stories and plan.md design.
  ============================================================================
-->

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and configuration

- [ ] T001 Add new configuration entries to `config/settings.yaml` for [feature]
- [ ] T002 [P] Add new domain types to `src/core/types.py` (if needed)
- [ ] T003 [P] Create abstract base class in `src/libs/<role>/base_<role>.py` (if new pluggable backend)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Register new provider in `src/libs/<role>/<role>_factory.py`
- [ ] T005 Add trace instrumentation points in `src/core/trace/` (if new pipeline stage)
- [ ] T006 [P] Wire new config keys in `src/core/settings.py`

**Checkpoint**: Foundation ready — user story implementation can begin

---

## Phase 3: User Story 1 - [Title] (Priority: P1) 🎯 MVP

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Tests for User Story 1 (if requested)

> **Write tests FIRST, ensure they FAIL before implementation**

- [ ] T007 [P] [US1] Contract test for `<role>` interface in `tests/unit/test_<role>_contract.py`
- [ ] T008 [P] [US1] Integration test for `<pipeline step>` in `tests/integration/test_<feature>.py`

### Implementation for User Story 1

- [ ] T009 [P] [US1] Implement `<provider>_<role>.py` in `src/libs/<role>/`
- [ ] T010 [US1] Implement pipeline stage in `src/ingestion/<stage>/` (depends on T009)
- [ ] T011 [US1] Add trace events to new pipeline stage
- [ ] T012 [US1] Wire into main pipeline in `src/ingestion/pipeline.py`

**Checkpoint**: User Story 1 fully functional and testable independently

---

## Phase 4: User Story 2 - [Title] (Priority: P2)

**Goal**: [Brief description]

**Independent Test**: [How to verify independently]

### Tests for User Story 2 (if requested)

- [ ] T013 [P] [US2] Test in `tests/<tier>/test_<feature>.py`

### Implementation for User Story 2

- [ ] T014 [P] [US2] Implementation in `src/<module>/`
- [ ] T015 [US2] Integration with User Story 1 (if needed)

**Checkpoint**: User Stories 1 AND 2 both work independently

---

[Add more user story phases as needed]

---

## Phase N: Polish & Cross-Cutting

**Purpose**: Improvements that affect multiple user stories

- [ ] TXXX [P] Update dashboard page in `src/observability/dashboard/pages/` + service in `services/`
- [ ] TXXX [P] Update MCP tool in `src/mcp_server/tools/` (if applicable)
- [ ] TXXX [P] Add/update config documentation
- [ ] TXXX Run full test suite: `pytest && ruff check src/`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### Module Dependency Order

When tasks span modules, implement in this order:

1. `core/types.py` — shared types first
2. `libs/` — pluggable backends
3. `ingestion/` or `core/query_engine/` — pipeline stages
4. `mcp_server/tools/` — MCP tools
5. `observability/dashboard/` — dashboard pages

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- `libs/` implementations before pipeline stages
- Pipeline stages before MCP tools or dashboard pages
- Trace events alongside the code they instrument

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All `libs/` implementations for different providers can run in parallel
- Different user stories can be worked on in parallel after Foundational phase
- Dashboard page + service can be developed in parallel with MCP tools

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: `pytest tests/unit/ tests/integration/`
5. Demo if ready

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add User Story 1 → `pytest -m unit` → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Each story adds value without breaking previous stories

---

## Notes

- [P] tasks = different files, no dependencies
- Each user story should be independently completable and testable
- Commit after each task: `feat(<scope>): [<ticket>] <description>`
- Verify tests fail before implementing
- Stop at any checkpoint to validate story independently
