# Feature Specification: Query & Retrieval

**Status**: migrated
**Created**: 2026-04-13
**Original commits**: D1–D7

## User Scenarios & Testing

### User Story 1 - Hybrid search with dense + sparse + fusion (P1)

As a user, I want to query the knowledge base and get ranked results combining semantic similarity and keyword matching.

**Acceptance Scenarios**:
1. **Given** an ingested collection, **When** `HybridSearch.search(query)` is called, **Then** dense and sparse retrieval run in parallel, results are fused via RRF
2. **Given** retrieval results, **When** reranking is enabled, **Then** a reranker re-scores results with fallback on failure

### User Story 2 - Query processing with keyword extraction (P2)

As a user, I want raw queries processed into keywords and filters for sparse retrieval.

**Acceptance Scenarios**:
1. **Given** a natural language query, **When** `QueryProcessor.process()` runs, **Then** keywords are extracted via regex tokenization with stop-word filtering

### User Story 3 - CLI query interface (P3)

As a user, I want to query from the command line with formatted output.

**Acceptance Scenarios**:
1. **Given** a query string, **When** `python scripts/query.py "my query"` runs, **Then** results are printed with text, source, and score

## Requirements

### Functional Requirements

- **FR-001**: System MUST implement `QueryProcessor` with keyword extraction and stop-word filtering
- **FR-002**: System MUST implement `DenseRetriever` using embedding + vector store query
- **FR-003**: System MUST implement `SparseRetriever` using BM25 index lookup
- **FR-004**: System MUST implement `reciprocal_rank_fusion()` (RRF) to merge dense + sparse results
- **FR-005**: System MUST implement `HybridSearch` orchestrating processor → dense → sparse → fusion → rerank
- **FR-006**: System MUST implement `Reranker` with configurable backend and fallback
- **FR-007**: All retrieval stages MUST accept `TraceContext` for observability
- **FR-008**: System MUST provide `scripts/query.py` CLI entry point

### Key Entities

- **QueryProcessor**: Tokenizes queries, extracts keywords, applies filters
- **HybridSearch**: Orchestrator: process → dense retrieve → sparse retrieve → RRF → rerank
- **reciprocal_rank_fusion**: RRF algorithm with configurable `k` parameter

## Success Criteria

- **SC-001**: Hybrid search returns combined dense + sparse results
- **SC-002**: RRF produces meaningful ranking that considers both retrieval signals
- **SC-003**: Reranker falls back gracefully when backend fails
- **SC-004**: All stages emit trace events

## Gaps Identified

- No unit tests for `query_knowledge_hub` tool
- No validation for RRF `k` parameter (should be > 0)
- No timeout for the full search pipeline
- SparseRetriever imports from `ingestion` (dependency flow violation)
