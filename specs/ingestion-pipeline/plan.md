# Implementation Plan: Ingestion Pipeline

**Status**: migrated | **Date**: 2026-04-13

## Summary

Implements the full document ingestion pipeline: transforms (refine, enrich, caption), embedding (dense + sparse), storage (BM25, ChromaDB, images), orchestration (IngestionPipeline), and cross-storage lifecycle management (DocumentManager).

## Technical Context

**Language/Version**: Python >= 3.11
**Primary Dependencies**: ChromaDB, sentence-transformers, OpenAI/Doubao embeddings
**Storage**: ChromaDB (vectors), SQLite (images, integrity), JSONL (BM25 index)
**Testing**: pytest (unit + integration)

## Affected Modules

| Module | Impact | Files |
|--------|--------|-------|
| `src/ingestion/transform/` | 4 transform implementations | base, chunk_refiner, metadata_enricher, image_captioner |
| `src/ingestion/embedding/` | 3 encoding implementations | dense_encoder, sparse_encoder, batch_processor |
| `src/ingestion/storage/` | 3 storage implementations | bm25_indexer, vector_upserter, image_storage |
| `src/ingestion/` | Pipeline + DocumentManager | pipeline.py, document_manager.py |
| `scripts/` | CLI entry point | ingest.py |

## Project Structure

```
src/ingestion/transform/base_transform.py     — 37 lines
src/ingestion/transform/chunk_refiner.py      — 164 lines
src/ingestion/transform/metadata_enricher.py  — 151 lines
src/ingestion/transform/image_captioner.py    — 135 lines
src/ingestion/embedding/dense_encoder.py      — 82 lines
src/ingestion/embedding/sparse_encoder.py     — 97 lines
src/ingestion/embedding/batch_processor.py    — 119 lines
src/ingestion/storage/bm25_indexer.py         — 232 lines
src/ingestion/storage/vector_upserter.py      — 118 lines
src/ingestion/storage/image_storage.py        — 207 lines
src/ingestion/pipeline.py                     — 208 lines
src/ingestion/document_manager.py             — 221 lines
scripts/ingest.py                             — 126 lines

tests/unit/ — 10 test files
tests/integration/ — 2 test files
```
