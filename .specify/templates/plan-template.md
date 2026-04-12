# Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

## Summary

[Extract from feature spec: primary requirement + technical approach]

## Technical Context

**Language/Version**: Python >= 3.11
**Primary Dependencies**: mcp, openai, chromadb, sentence-transformers, streamlit, pymupdf
**Storage**: ChromaDB (vectors), JSONL files (traces), BM25 index (sparse)
**Testing**: pytest with markers (`unit`, `integration`, `e2e`), pytest-cov, pytest-mock
**Target Platform**: Local development server (MCP + Streamlit dashboard)
**Project Type**: Modular monolith — pluggable RAG framework with MCP protocol
**Performance Goals**: [Feature-specific, e.g., embedding batch size, query latency target]
**Constraints**: [Feature-specific, e.g., max memory for document processing]
**Scale/Scope**: [Feature-specific, e.g., number of supported document formats, concurrent users]

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [ ] **Import discipline**: No `from src.*` imports used
- [ ] **Plugin pattern**: New `libs/` code follows `base_*.py` + `*_factory.py` + implementation
- [ ] **Dependency flow**: No upward imports (libs does not import ingestion/mcp_server)
- [ ] **Config via YAML**: No hard-coded providers, models, or keys
- [ ] **Observability**: Trace events emitted for pipeline operations
- [ ] **Type safety**: New domain types in `core/types.py`
- [ ] **Testing**: Contract tests for libs, integration tests for pipeline stages

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
src/
├── core/
│   ├── query_engine/    # Dense/sparse/hybrid retrieval, fusion, reranking
│   ├── response/        # Citation generation, multimodal assembly
│   ├── trace/           # Trace context, collector
│   ├── settings.py      # YAML config loader
│   └── types.py         # Shared domain types
├── ingestion/
│   ├── chunking/        # Document chunker
│   ├── document_manager.py  # Cross-storage coordination
│   ├── embedding/       # Dense/sparse encoders, batch processor
│   ├── pipeline.py      # Orchestrates load → chunk → transform → embed → store
│   ├── storage/         # BM25 indexer, image storage, vector upserter
│   └── transform/       # Chunk refiner, image captioner, metadata enricher
├── libs/
│   ├── embedding/       # Abstract base + factories (OpenAI, Doubao, Azure, Ollama)
│   ├── evaluator/       # Abstract base + factories (Ragas, Custom, Composite)
│   ├── llm/             # Abstract base + factories (OpenAI, Doubao, Azure, DeepSeek, Ollama + vision)
│   ├── loader/          # Abstract base + factories (PDF, DOCX, TXT, MD, HTML, CSV, PPTX, Video)
│   ├── reranker/        # Abstract base + factories (CrossEncoder, LLM)
│   ├── splitter/        # Abstract base + factories (Recursive)
│   └── vector_store/    # Abstract base + factories (ChromaDB)
├── mcp_server/
│   ├── protocol_handler.py  # JSON-RPC protocol
│   ├── server.py        # MCP server entry point
│   └── tools/           # MCP tool definitions (query, summary, list_collections)
└── observability/
    ├── dashboard/       # Streamlit multi-page app (pages/ + services/)
    ├── evaluation/      # Eval runner, composite evaluator
    └── logger.py        # get_logger() factory

tests/
├── unit/                # Fast, isolated tests (pytest -m unit)
├── integration/         # Multi-component tests (pytest -m integration)
├── e2e/                 # Full pipeline tests (pytest -m e2e)
└── fixtures/            # Test data (sample_documents/)

scripts/                 # CLI entry points
config/                  # settings.yaml
```

**Structure Decision**: Uses the existing 5-package modular monolith. New features extend the appropriate package.

## Affected Modules

| Module | Impact | Files to Create/Modify |
|--------|--------|----------------------|
| `src/core/` | [e.g., new type, new pipeline stage] | [specific files] |
| `src/ingestion/` | [e.g., new pipeline step] | [specific files] |
| `src/libs/` | [e.g., new provider] | [specific files] |
| `src/mcp_server/` | [e.g., new tool] | [specific files] |
| `src/observability/` | [e.g., new dashboard page] | [specific files] |
| `config/` | [e.g., new config section] | settings.yaml |
| `tests/` | [e.g., new test files] | [specific files] |

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., cross-module import] | [current need] | [why existing pattern insufficient] |
