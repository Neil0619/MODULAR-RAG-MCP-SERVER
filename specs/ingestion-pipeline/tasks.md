# Tasks: Ingestion Pipeline

**Status**: migrated — all tasks completed

## Phase 1: Transform Layer

- [x] T001 [C5] Implement `BaseTransform` abstract class in `src/ingestion/transform/base_transform.py`
- [x] T002 [C5] Implement `ChunkRefiner` with rule/LLM refinement in `src/ingestion/transform/chunk_refiner.py`
- [x] T003 [C6] Implement `MetadataEnricher` with rule/LLM enrichment in `src/ingestion/transform/metadata_enricher.py`
- [x] T004 [C7] Implement `ImageCaptioner` with Vision LLM fallback in `src/ingestion/transform/image_captioner.py`

## Phase 2: Embedding Layer

- [x] T005 [C8] Implement `DenseEncoder` with batch embedding in `src/ingestion/embedding/dense_encoder.py`
- [x] T006 [C9] Implement `SparseEncoder` with TF term weights in `src/ingestion/embedding/sparse_encoder.py`
- [x] T007 [C10] Implement `BatchProcessor` orchestrating dense + sparse in `src/ingestion/embedding/batch_processor.py`

## Phase 3: Storage Layer

- [x] T008 [C11] Implement `BM25Indexer` with inverted index + IDF in `src/ingestion/storage/bm25_indexer.py`
- [x] T009 [C12] Implement `VectorUpserter` with idempotent upsert in `src/ingestion/storage/vector_upserter.py`
- [x] T010 [C13] Implement `ImageStorage` with SQLite index in `src/ingestion/storage/image_storage.py`

## Phase 4: Orchestration

- [x] T011 [C14] Implement `IngestionPipeline` in `src/ingestion/pipeline.py`
- [x] T012 [G2] Implement `DocumentManager` in `src/ingestion/document_manager.py`
- [x] T013 [C15] Add `scripts/ingest.py` CLI entry point

## Phase 5: Tests

- [x] T014 Tests for ChunkRefiner in `tests/unit/test_chunk_refiner.py`
- [x] T015 Tests for MetadataEnricher in `tests/unit/test_metadata_enricher_contract.py`
- [x] T016 Tests for ImageCaptioner in `tests/unit/test_image_captioner_fallback.py`
- [x] T017 Tests for DenseEncoder in `tests/unit/test_dense_encoder.py`
- [x] T018 Tests for SparseEncoder in `tests/unit/test_sparse_encoder.py`
- [x] T019 Tests for BatchProcessor in `tests/unit/test_batch_processor.py`
- [x] T020 Tests for BM25Indexer in `tests/unit/test_bm25_indexer_roundtrip.py`
- [x] T021 Tests for VectorUpserter in `tests/unit/test_vector_upserter_idempotency.py`
- [x] T022 Tests for ImageStorage in `tests/unit/test_image_storage.py`
- [x] T023 Tests for DocumentManager in `tests/unit/test_document_manager.py`
- [x] T024 Integration tests in `tests/integration/test_ingestion_pipeline.py`

## Gaps

- No retry/circuit-breaker for external API calls
- No memory monitoring for large batch processing
- No timeout for individual pipeline stages
