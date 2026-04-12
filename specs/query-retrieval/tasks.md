# Tasks: Query & Retrieval

**Status**: migrated — all tasks completed

## Phase 1: Query Processing

- [x] T001 [D1] Implement `QueryProcessor` with keyword extraction in `src/core/query_engine/query_processor.py`

## Phase 2: Retrievers

- [x] T002 [D2] Implement `DenseRetriever` in `src/core/query_engine/dense_retriever.py`
- [x] T003 [D3] Implement `SparseRetriever` with BM25 query in `src/core/query_engine/sparse_retriever.py`

## Phase 3: Fusion + Orchestration

- [x] T004 [D4] Implement `reciprocal_rank_fusion()` in `src/core/query_engine/fusion.py`
- [x] T005 [D5] Implement `HybridSearch` orchestration in `src/core/query_engine/hybrid_search.py`

## Phase 4: Reranking

- [x] T006 [D6] Implement `Reranker` with fallback in `src/core/query_engine/reranker.py`

## Phase 5: CLI

- [x] T007 [D7] Add `scripts/query.py` CLI entry point

## Phase 6: Tests

- [x] T008 Tests for QueryProcessor in `tests/unit/test_query_processor.py`
- [x] T009 Tests for DenseRetriever in `tests/unit/test_dense_retriever.py`
- [x] T010 Tests for SparseRetriever in `tests/unit/test_sparse_retriever.py`
- [x] T011 Tests for RRF in `tests/unit/test_fusion_rrf.py`
- [x] T012 Tests for Reranker fallback in `tests/unit/test_reranker_fallback.py`
- [x] T013 Integration tests in `tests/integration/test_hybrid_search.py`

## Gaps

- No unit tests for `query_knowledge_hub` tool
- SparseRetriever imports from `ingestion` (dependency flow violation per constitution)
- No validation for RRF `k` parameter
- No timeout for full search pipeline
