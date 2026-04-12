# Tasks: MCP Protocol

**Status**: migrated — all tasks completed

## Phase 1: Server Foundation

- [x] T001 [E1] Implement MCP Server with stdio transport in `src/mcp_server/server.py`
- [x] T002 [E2] Implement `ProtocolHandler` with JSON-RPC 2.0 in `src/mcp_server/protocol_handler.py`

## Phase 2: MCP Tools

- [x] T003 [E3] Implement `query_knowledge_hub` with citations in `src/mcp_server/tools/query_knowledge_hub.py`
- [x] T004 [E4] Implement `list_collections` in `src/mcp_server/tools/list_collections.py`
- [x] T005 [E5] Implement `get_document_summary` in `src/mcp_server/tools/get_document_summary.py`

## Phase 3: Response Assembly

- [x] T006 [E6] Implement `CitationGenerator` in `src/core/response/citation_generator.py`
- [x] T007 [E6] Implement `ResponseBuilder` in `src/core/response/response_builder.py`
- [x] T008 [E6] Implement `MultimodalAssembler` in `src/core/response/multimodal_assembler.py`

## Phase 4: Tests

- [x] T009 E2E MCP client test in `tests/e2e/test_mcp_client.py`
- [x] T010 Integration server test in `tests/integration/test_mcp_server.py`
- [x] T011 Unit tests for get_document_summary in `tests/unit/test_get_document_summary.py`
- [x] T012 Unit tests for ResponseBuilder in `tests/unit/test_response_builder.py`
- [x] T013 Unit tests for MultimodalAssembler in `tests/unit/test_multimodal_assembler.py`
- [x] T014 Unit tests for list_collections in `tests/unit/test_list_collections.py`

## Gaps

- No unit tests for `query_knowledge_hub` tool
- No unit tests for `ProtocolHandler` error edge cases
- No retry mechanism for transient tool failures
