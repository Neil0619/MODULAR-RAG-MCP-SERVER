# Feature Specification: Ingestion Pipeline

**Status**: migrated
**Created**: 2026-04-13
**Original commits**: C5–C15

## User Scenarios & Testing

### User Story 1 - Ingest a document end-to-end (P1)

As a user, I want to run a single command to ingest a document: load → chunk → transform → embed → store.

**Acceptance Scenarios**:
1. **Given** a valid PDF file, **When** `IngestionPipeline.run()` is called, **Then** chunks are embedded, stored in ChromaDB and BM25, and a trace is emitted
2. **Given** an already-processed file, **When** `run()` is called without `force=True`, **Then** the file is skipped
3. **Given** an ingestion in progress, **When** `on_progress` callback is provided, **Then** stage progress is reported

### User Story 2 - Transform chunks with optional LLM (P2)

As a user, I want chunks refined and enriched before embedding, with optional LLM-based processing.

**Acceptance Scenarios**:
1. **Given** chunks from a document, **When** `ChunkRefiner.transform()` runs, **Then** text is cleaned (whitespace, encoding) and optionally LLM-refined
2. **Given** LLM unavailable, **When** transform runs, **Then** it falls back to rule-based processing

### User Story 3 - Encode chunks with dense + sparse vectors (P3)

As a user, I want each chunk encoded as both dense (embedding) and sparse (TF) vectors.

**Acceptance Scenarios**:
1. **Given** a list of chunks, **When** `BatchProcessor.process()` runs, **Then** each chunk gets both `dense_vector` and `sparse_vector`
2. **Given** batch size of 32, **When** processing, **Then** chunks are processed in batches to manage memory

### User Story 4 - Manage document lifecycle across storages (P3)

As a user, I want to list and delete documents across all storage backends (vectors, BM25, images, integrity).

**Acceptance Scenarios**:
1. **Given** an ingested document, **When** `DocumentManager.delete_document()` is called, **Then** chunks are removed from ChromaDB, BM25 index, images, and integrity records

## Requirements

### Functional Requirements

- **FR-001**: System MUST provide `BaseTransform` abstract class with `transform(chunks) -> chunks`
- **FR-002**: System MUST implement `ChunkRefiner` with rule-based cleanup and optional LLM refinement
- **FR-003**: System MUST implement `MetadataEnricher` with rule-based and optional LLM enrichment
- **FR-004**: System MUST implement `ImageCaptioner` with Vision LLM fallback
- **FR-005**: System MUST implement `DenseEncoder` using `EmbeddingFactory`
- **FR-006**: System MUST implement `SparseEncoder` with TF term weights
- **FR-007**: System MUST implement `BatchProcessor` orchestrating dense + sparse encoding
- **FR-008**: System MUST implement `BM25Indexer` with inverted index, IDF scoring, and persistence
- **FR-009**: System MUST implement `VectorUpserter` with idempotent stable IDs
- **FR-010**: System MUST implement `ImageStorage` with file storage and SQLite index
- **FR-011**: System MUST implement `IngestionPipeline` orchestrating the full pipeline
- **FR-012**: System MUST implement `DocumentManager` for cross-storage document lifecycle
- **FR-013**: System MUST provide `scripts/ingest.py` CLI entry point

### Key Entities

- **ChunkRefiner**: Rule-based text cleanup + optional LLM prompt refinement
- **BM25Indexer**: Inverted index with IDF computation and disk persistence
- **VectorUpserter**: ChromaDB upsert with deterministic chunk IDs
- **IngestionPipeline**: Orchestrator with trace instrumentation and progress callbacks
- **DocumentManager**: Coordinates delete/list across all storage backends

## Success Criteria

- **SC-001**: Full pipeline runs end-to-end on a valid PDF
- **SC-002**: Re-ingestion is idempotent (integrity checker + stable IDs)
- **SC-003**: All transforms gracefully degrade when LLM is unavailable
- **SC-004**: BM25 index persists to disk and can be reloaded

## Gaps Identified

- No retry/circuit-breaker for external API calls (embedding, LLM)
- No memory monitoring for large batch processing
- No timeout for individual pipeline stages
