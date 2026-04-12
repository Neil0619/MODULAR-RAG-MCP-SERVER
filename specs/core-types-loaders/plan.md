# Implementation Plan: Core Types + Loaders

**Status**: migrated | **Date**: 2026-04-13

## Summary

Defines foundational data types (`Document`, `Chunk`, `ChunkRecord`, `ImageRef`) and implements document loading (PDF via PyMuPDF) with extension-based factory routing, file integrity checking via SQLite, and document chunking with image distribution.

## Technical Context

**Language/Version**: Python >= 3.11
**Primary Dependencies**: PyMuPDF (fitz), langchain-text-splitters
**Storage**: SQLite (file integrity checker)
**Testing**: pytest (unit + integration)
**Project Type**: Modular monolith — foundational layer

## Affected Modules

| Module | Impact | Files |
|--------|--------|-------|
| `src/core/` | Core type definitions | `types.py` |
| `src/libs/loader/` | Loader base + PdfLoader + factory + integrity | 4 files |
| `src/ingestion/chunking/` | Document chunker | `document_chunker.py` |

## Project Structure

```
src/core/types.py                        — 273 lines
src/libs/loader/base_loader.py           — 46 lines
src/libs/loader/pdf_loader.py            — 169 lines
src/libs/loader/loader_factory.py        — 109 lines
src/libs/loader/file_integrity.py        — 173 lines
src/ingestion/chunking/document_chunker.py — 141 lines

tests/unit/test_loader_pdf_contract.py   — 130 lines
tests/unit/test_loader_factory.py        — 80 lines
tests/unit/test_file_integrity.py        — 116 lines
tests/integration/test_ingestion_pipeline.py — 307 lines
```
