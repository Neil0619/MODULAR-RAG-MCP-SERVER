# Tasks: Evaluation

**Status**: migrated — all tasks completed

## Phase 1: Evaluator Backends

- [x] T001 [H1] Implement `BaseEvaluator` abstract class in `src/libs/evaluator/base_evaluator.py`
- [x] T002 [H1] Implement `RagasEvaluator` with graceful degradation in `src/libs/evaluator/ragas_evaluator.py`
- [x] T003 Implement `CustomEvaluator` with hit_rate, MRR, precision, recall in `src/libs/evaluator/custom_evaluator.py`
- [x] T004 Implement `EvaluatorFactory` in `src/libs/evaluator/evaluator_factory.py`

## Phase 2: Composite + Runner

- [x] T005 [H2] Implement `CompositeEvaluator` with multi-backend merging in `src/observability/evaluation/composite_evaluator.py`
- [x] T006 [H3] Implement `EvalRunner` with golden test set in `src/observability/evaluation/eval_runner.py`
- [x] T007 [H3] Add `scripts/evaluate.py` CLI entry point

## Phase 3: Tests

- [x] T008 Tests for CompositeEvaluator in `tests/unit/test_composite_evaluator.py`
- [x] T009 Tests for CustomEvaluator in `tests/unit/test_custom_evaluator.py`
- [x] T010 Tests for EvalRunner in `tests/unit/test_eval_runner.py`
- [x] T011 Tests for RagasEvaluator in `tests/unit/test_ragas_evaluator.py`
- [x] T012 [H5] E2E recall regression test in `tests/e2e/`

## Gaps

- Ragas implementations are simplified (not full LLM-based)
- No evaluation for generation quality
- No A/B testing framework
