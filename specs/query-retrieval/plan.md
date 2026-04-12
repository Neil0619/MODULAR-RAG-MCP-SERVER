# Implementation Plan: Query & Retrieval

**Status**: migrated | **Date**: 2026-04-13

## Summary

Implements the query pipeline: query processing (keyword extraction), dense retrieval (embedding + vector store), sparse retrieval (BM25), Reciprocal Rank Fusion, hybrid search orchestration, reranking with fallback, and a CLI query interface.

## Technical Context

**Language/Version**: Python >= 3.11
**Primary Dependencies**: ChromaDB, sentence-transformers, cross-encoder reranker
**Storage**: ChromaDB (vectors), BM25 index (sparse)
**Testing**: pytest (unit + integration)

## Affected Modules

| Module | Impact | Files |
|--------|--------|-------|
| `src/core/query_engine/` | 6 query components | query_processor, dense_retriever, sparse_retriever, fusion, hybrid_search, reranker |
| `scripts/` | CLI entry point | query.py |

## Project Structure

```
src/core/query_engine/query_processor.py   — 99 lines
src/core/query_engine/dense_retriever.py   — 81 lines
src/core/query_engine/sparse_retriever.py  — 95 lines
src/core/query_engine/fusion.py            — 70 lines
src/core/query_engine/hybrid_search.py     — 222 lines
src/core/query_engine/reranker.py          — 118 lines
scripts/query.py                           — 169 lines

tests/unit/ — 6 test files
tests/integration/ — 1 test file
```
