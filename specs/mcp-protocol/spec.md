# Feature Specification: MCP Protocol

**Status**: migrated
**Created**: 2026-04-13
**Original commits**: E1–E6

## User Scenarios & Testing

### User Story 1 - Query knowledge hub via MCP (P1)

As an MCP client, I want to send a query and receive a structured response with citations.

**Acceptance Scenarios**:
1. **Given** an ingested collection, **When** `query_knowledge_hub` tool is called via MCP, **Then** a JSON response with text and citations is returned
2. **Given** a query, **When** processed, **Then** hybrid search runs and results are formatted with Markdown + citation metadata

### User Story 2 - List available collections (P2)

As an MCP client, I want to see what collections exist and their stats.

**Acceptance Scenarios**:
1. **Given** BM25 index files on disk, **When** `list_collections` is called, **Then** collection names and document counts are returned

### User Story 3 - Get document summary (P3)

As an MCP client, I want to retrieve metadata and stats for a specific document.

**Acceptance Scenarios**:
1. **Given** a `source_path`, **When** `get_document_summary` is called, **Then** document metadata, chunk count, and image count are returned

### User Story 4 - Multimodal responses (P3)

As an MCP client, I want responses that include both text and referenced images.

**Acceptance Scenarios**:
1. **Given** results with images, **When** `MultimodalAssembler.assemble()` runs, **Then** content includes both text parts and image references

## Requirements

### Functional Requirements

- **FR-001**: System MUST implement MCP Server with stdio transport via `mcp` SDK
- **FR-002**: System MUST implement `ProtocolHandler` with JSON-RPC 2.0 (initialize, tools/list, tools/call)
- **FR-003**: System MUST implement `query_knowledge_hub` tool with hybrid search + reranking + citations
- **FR-004**: System MUST implement `list_collections` tool returning collection metadata
- **FR-005**: System MUST implement `get_document_summary` tool returning document stats
- **FR-006**: System MUST implement `CitationGenerator` for extracting citation metadata from results
- **FR-007**: System MUST implement `ResponseBuilder` formatting results with Markdown + citations
- **FR-008**: System MUST implement `MultimodalAssembler` for text + image content assembly

### Key Entities

- **ProtocolHandler**: JSON-RPC 2.0 request router with tool registration
- **CitationGenerator**: Extracts structured citations from `RetrievalResult` list
- **MultimodalAssembler**: Assembles text + image content blocks

## Success Criteria

- **SC-001**: MCP server starts and responds to JSON-RPC requests via stdio
- **SC-002**: All three tools return valid JSON responses
- **SC-003**: Citations include source, page, and relevance score
- **SC-004**: Multimodal responses include both text and image content

## Gaps Identified

- No unit tests for `query_knowledge_hub` tool
- No unit tests for `ProtocolHandler` error handling edge cases
- No retry for transient tool execution failures
