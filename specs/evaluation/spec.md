# Feature Specification: Evaluation

**Status**: migrated
**Created**: 2026-04-13
**Original commits**: H1–H5

## User Scenarios & Testing

### User Story 1 - Evaluate retrieval quality (P1)

As a developer, I want to measure retrieval quality using standard metrics (hit rate, MRR, precision, recall).

**Acceptance Scenarios**:
1. **Given** a golden test set with query → relevant_doc mappings, **When** evaluation runs, **Then** hit_rate, MRR, precision, recall are computed per query and aggregated

### User Story 2 - Use multiple evaluation backends (P2)

As a developer, I want to combine metrics from multiple evaluation backends (custom + Ragas).

**Acceptance Scenarios**:
1. **Given** multiple evaluators configured, **When** `CompositeEvaluator` runs, **Then** metrics from all backends are merged into a single report

### User Story 3 - Graceful degradation when Ragas unavailable (P2)

As a developer, I want the system to work even when optional Ragas dependencies are not installed.

**Acceptance Scenarios**:
1. **Given** Ragas not installed, **When** `RagasEvaluator` is initialized, **Then** it logs a warning and skips Ragas metrics rather than crashing

### User Story 4 - Run evaluations via CLI (P3)

As a developer, I want to run evaluations from the command line.

**Acceptance Scenarios**:
1. **Given** a test set JSON file, **When** `python scripts/evaluate.py --test-set path.json` runs, **Then** evaluation results are printed and saved

## Requirements

### Functional Requirements

- **FR-001**: System MUST implement `BaseEvaluator` abstract class with `evaluate()` method
- **FR-002**: System MUST implement `CustomEvaluator` with hit_rate, MRR, precision, recall
- **FR-003**: System MUST implement `RagasEvaluator` with faithfulness, answer_relevancy, context_precision
- **FR-004**: System MUST implement `CompositeEvaluator` combining multiple backends
- **FR-005**: System MUST implement `EvaluatorFactory` for config-based instantiation
- **FR-006**: System MUST implement `EvalRunner` for running evaluations against golden test sets
- **FR-007**: System MUST provide `scripts/evaluate.py` CLI entry point
- **FR-008**: RagasEvaluator MUST gracefully degrade when optional deps are missing

### Key Entities

- **BaseEvaluator**: Abstract interface for evaluation backends
- **CompositeEvaluator**: Merges metrics from multiple evaluators
- **EvalRunner**: Orchestrates evaluation: loads test set → runs queries → evaluates → reports
- **EvalReport**: Aggregated results with per-case detail

## Success Criteria

- **SC-001**: Custom metrics (hit_rate, MRR) computed correctly
- **SC-002**: Ragas evaluator degrades gracefully without optional deps
- **SC-003**: Composite evaluator merges metrics from multiple backends
- **SC-004**: E2E recall regression test validates threshold

## Gaps Identified

- Ragas implementations are simplified (not full LLM-based scoring)
- No evaluation for generation quality (only retrieval)
- No A/B testing framework
- No benchmark comparison features
