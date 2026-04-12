# Tasks: Trace & Observability

**Status**: migrated — all tasks completed

## Phase 1: Trace Infrastructure

- [x] T001 [F1] Implement `TraceContext` with stage recording and elapsed timing in `src/core/trace/trace_context.py`
- [x] T002 [F1] Implement `TraceCollector` in `src/core/trace/trace_collector.py`

## Phase 2: Persistence + Logging

- [x] T003 [F2] Implement JSONL `write_trace` persistence in `src/observability/logger.py`
- [x] T004 Implement `get_logger()` factory with date-based file rotation in `src/observability/logger.py`

## Phase 3: Instrumentation

- [x] T005 [F3] Add trace instrumentation to Query pipeline in `src/core/query_engine/`
- [x] T006 [F4] Add trace instrumentation to Ingestion pipeline in `src/ingestion/pipeline.py`
- [x] T007 [F5] Add `on_progress` callback to `IngestionPipeline`

## Phase 4: Tests

- [x] T008 Tests for TraceContext in `tests/unit/test_trace_context.py`

## Gaps

- No tests for `TraceCollector`
- No tests for `write_trace` JSONL persistence
- No distributed tracing support
