# Tasks: Core Types + Loaders

**Status**: migrated — all tasks completed

## Phase 1: Core Types

- [x] T001 [C1] Define `ImageRef`, `Document`, `Chunk`, `ChunkRecord`, `ProcessedQuery`, `RetrievalResult` in `src/core/types.py`
- [x] T002 [C1] Add `from_dict()` deserialization to all types
- [x] T003 [C1] Implement deterministic ID generation via UUID4

## Phase 2: Loader Infrastructure

- [x] T004 [C3] Implement `BaseLoader` abstract class in `src/libs/loader/base_loader.py`
- [x] T005 [C3] Implement `PdfLoader` with text + image extraction in `src/libs/loader/pdf_loader.py`
- [x] T006 [J1] Implement `LoaderFactory` with extension-based routing in `src/libs/loader/loader_factory.py`
- [x] T007 [C2] Implement `FileIntegrityChecker` + `SQLiteIntegrityChecker` in `src/libs/loader/file_integrity.py`

## Phase 3: Chunking

- [x] T008 [C4] Implement `DocumentChunker` with image distribution in `src/ingestion/chunking/document_chunker.py`

## Phase 4: Tests

- [x] T009 Contract tests for PdfLoader in `tests/unit/test_loader_pdf_contract.py`
- [x] T010 Factory tests in `tests/unit/test_loader_factory.py`
- [x] T011 Integrity checker tests in `tests/unit/test_file_integrity.py`
- [x] T012 Integration tests in `tests/integration/test_ingestion_pipeline.py`

## Gaps

- No tests for `ImageRef`, `ProcessedQuery`, `RetrievalResult`
- No error handling for corrupted/oversized PDFs
- `BaseLoader._validate_path` untested
