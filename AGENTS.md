# AGENTS.md — Module Ownership & Boundaries

This document defines the module boundaries for the Modular RAG MCP Server project. Use it to understand which code belongs where and how modules interact.

## Module Ownership

### `src/core/` — Shared Foundation

**Owner**: All developers
**Responsibility**: Types, settings, query engine orchestration, trace infrastructure, response assembly
**Key files**:
- `types.py` — shared domain types (RetrievalResult, etc.)
- `settings.py` — YAML config loader
- `query_engine/` — dense/sparse/hybrid retrieval, fusion, reranking
- `trace/` — trace context and collector
- `response/` — citation generation, multimodal assembly

**Rules**:
- Can import from `libs` (for types/settings only) and `observability.logger`
- MUST NOT import from `ingestion` or `mcp_server`
- New domain types MUST go here, not in downstream modules

### `src/ingestion/` — Document Pipeline

**Owner**: All developers
**Responsibility**: End-to-end document processing pipeline
**Key files**:
- `pipeline.py` — orchestrates the full pipeline
- `document_manager.py` — cross-storage coordination
- `chunking/` — document chunker
- `embedding/` — dense/sparse encoders, batch processor
- `storage/` — BM25 indexer, image storage, vector upserter
- `transform/` — chunk refiner, image captioner, metadata enricher

**Rules**:
- Can import from `core` and `libs`
- MUST NOT import from `mcp_server` or `observability.dashboard`
- Pipeline stages MUST emit trace events

### `src/libs/` — Pluggable Backends

**Owner**: All developers
**Responsibility**: Abstract interfaces and concrete implementations for all external integrations
**Sub-modules**:
- `embedding/` — embedding providers (OpenAI, Doubao, Azure, Ollama)
- `llm/` — LLM providers (OpenAI, Doubao, Azure, DeepSeek, Ollama + vision variants)
- `loader/` — document loaders (PDF, DOCX, TXT, MD, HTML, CSV, PPTX, Video)
- `reranker/` — reranking backends (CrossEncoder, LLM)
- `splitter/` — text splitting strategies (Recursive)
- `vector_store/` — vector databases (ChromaDB)
- `evaluator/` — evaluation backends (Ragas, Custom, Composite)

**Rules**:
- Can import from `core` (types, settings only)
- MUST NOT import from `ingestion`, `mcp_server`, or `observability`
- Each sub-module follows: `base_*.py` → `*_factory.py` → `<provider>_<role>.py`
- All configuration received via constructor or Settings object

### `src/mcp_server/` — External API

**Owner**: All developers
**Responsibility**: MCP protocol handling and tool definitions
**Key files**:
- `server.py` — MCP server entry point
- `protocol_handler.py` — JSON-RPC protocol implementation
- `tools/` — tool definitions (query_knowledge_hub, get_document_summary, list_collections)

**Rules**:
- Can import from `core` and `ingestion`
- MUST NOT import from `observability` or `libs` directly (go through core/ingestion)
- New tools MUST follow the existing tool pattern and emit trace events

### `src/observability/` — Observability Layer

**Owner**: All developers
**Responsibility**: Logging, dashboard, evaluation
**Sub-modules**:
- `logger.py` — `get_logger()` factory
- `dashboard/` — Streamlit multi-page app (pages/ + services/)
- `evaluation/` — eval runner, composite evaluator

**Rules**:
- Can import from `core` and `ingestion`
- MUST NOT import from `mcp_server`
- Dashboard pages: `render()` function + service module (no business logic in pages)
- Logger is the single source of logging — no `logging.getLogger` elsewhere

## Inter-Module Communication Rules

| From | Can Import | Cannot Import |
|------|-----------|---------------|
| `core` | `libs` (types only), `observability.logger` | `ingestion`, `mcp_server`, `observability.dashboard` |
| `ingestion` | `core`, `libs` | `mcp_server`, `observability.dashboard` |
| `libs` | `core` (types, settings only) | `ingestion`, `mcp_server`, `observability` |
| `mcp_server` | `core`, `ingestion` | `libs` (use via core/ingestion), `observability.dashboard` |
| `observability` | `core`, `ingestion` | `mcp_server` |

## Adding a New Feature — Where Does It Go?

| Feature Type | Target Module | Example |
|-------------|---------------|---------|
| New LLM/embedding provider | `src/libs/<role>/` | Adding Cohere LLM → `libs/llm/cohere_llm.py` |
| New document format | `src/libs/loader/` | Adding EPUB → `libs/loader/epub_loader.py` |
| New pipeline stage | `src/ingestion/` | Adding deduplication → `ingestion/transform/deduplicator.py` |
| New query strategy | `src/core/query_engine/` | Adding ColBERT retrieval → `core/query_engine/colbert_retriever.py` |
| New MCP tool | `src/mcp_server/tools/` | Adding delete_document → `mcp_server/tools/delete_document.py` |
| New dashboard page | `src/observability/dashboard/pages/` | Adding settings page → `dashboard/pages/settings.py` |
| New shared type | `src/core/types.py` | Adding ChunkMetadata → `core/types.py` |
| New config key | `config/settings.yaml` + `src/core/settings.py` | Adding batch_size → settings.yaml + settings.py |
