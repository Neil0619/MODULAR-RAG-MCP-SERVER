"""E2E: MCP Client call simulation.

Tests the MCP server by simulating JSON-RPC 2.0 client requests
through ProtocolHandler, exercising tools/list and tools/call for
query_knowledge_hub.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from mcp_server.protocol_handler import ProtocolHandler, ToolSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SERVER_SCRIPT = _PROJECT_ROOT / "src" / "mcp_server" / "server.py"


def _make_jsonrpc(method: str, params: dict[str, Any] | None = None, req_id: int = 1) -> dict:
    """Build a JSON-RPC 2.0 request dict."""
    req: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "id": req_id}
    if params is not None:
        req["params"] = params
    return req


def _fake_tool_executor(name: str, arguments: dict[str, Any]) -> str:
    """Simulate tool execution for protocol-level tests."""
    if name == "query_knowledge_hub":
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 10)
        return json.dumps({
            "query": query,
            "results": [
                {"chunk_id": "c1", "text": "sample result", "score": 0.95},
            ],
            "total": 1,
            "top_k": top_k,
        })
    raise ValueError(f"Unknown tool: {name}")


def _make_handler() -> ProtocolHandler:
    """Create a ProtocolHandler with registered tools and a fake executor."""
    tools = [
        ToolSchema(
            name="query_knowledge_hub",
            description="Search the knowledge base using hybrid retrieval.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "top_k": {"type": "integer", "description": "Number of results.", "default": 10},
                    "collection": {"type": "string", "description": "Collection name.", "default": "default"},
                },
                "required": ["query"],
            },
        ),
    ]
    return ProtocolHandler(tools=tools, tool_executor=_fake_tool_executor)


# ---------------------------------------------------------------------------
# Tests: Protocol-level simulation (no subprocess)
# ---------------------------------------------------------------------------


class TestMCPProtocolSimulation:
    """Test MCP JSON-RPC protocol handling with fake tool executor."""

    def test_initialize(self) -> None:
        handler = _make_handler()
        resp = handler.handle_request(_make_jsonrpc("initialize"))
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        result = resp["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"]["name"] == "modular-rag-mcp-server"
        assert "tools" in result["capabilities"]

    def test_tools_list(self) -> None:
        handler = _make_handler()
        resp = handler.handle_request(_make_jsonrpc("tools/list"))
        tools = resp["result"]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "query_knowledge_hub"
        assert "inputSchema" in tools[0]

    def test_tools_call_query(self) -> None:
        handler = _make_handler()
        resp = handler.handle_request(_make_jsonrpc("tools/call", {
            "name": "query_knowledge_hub",
            "arguments": {"query": "What is RAG?", "top_k": 5},
        }))
        assert resp["jsonrpc"] == "2.0"
        content = resp["result"]["content"]
        assert len(content) == 1
        data = json.loads(content[0]["text"])
        assert data["query"] == "What is RAG?"
        assert data["top_k"] == 5
        assert len(data["results"]) >= 1

    def test_tools_call_unknown_tool(self) -> None:
        handler = _make_handler()
        resp = handler.handle_request(_make_jsonrpc("tools/call", {
            "name": "nonexistent_tool",
            "arguments": {},
        }))
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_tools_call_missing_name(self) -> None:
        handler = _make_handler()
        resp = handler.handle_request(_make_jsonrpc("tools/call", {
            "arguments": {"query": "test"},
        }))
        assert "error" in resp
        assert resp["error"]["code"] == -32602

    def test_method_not_found(self) -> None:
        handler = _make_handler()
        resp = handler.handle_request(_make_jsonrpc("nonexistent/method"))
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_invalid_jsonrpc_version(self) -> None:
        handler = _make_handler()
        resp = handler.handle_request({"jsonrpc": "1.0", "method": "initialize", "id": 1})
        assert "error" in resp
        assert resp["error"]["code"] == -32600

    def test_notifications_initialized(self) -> None:
        handler = _make_handler()
        resp = handler.handle_request({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        # Notifications return empty dict (no id, no response needed)
        assert resp == {}

    def test_shutdown(self) -> None:
        handler = _make_handler()
        resp = handler.handle_request(_make_jsonrpc("shutdown"))
        assert resp["jsonrpc"] == "2.0"
        assert resp["result"] == {}


# ---------------------------------------------------------------------------
# Tests: Server module importability
# ---------------------------------------------------------------------------


class TestMCPServerImport:

    def test_server_module_importable(self) -> None:
        from mcp_server.server import create_server
        assert callable(create_server)

    def test_server_creates_instance(self) -> None:
        from mcp_server.server import create_server
        server = create_server("test-server")
        assert server is not None

    def test_tool_schemas_defined(self) -> None:
        from mcp_server.server import _TOOL_SCHEMAS
        assert len(_TOOL_SCHEMAS) >= 1
        names = [t.name for t in _TOOL_SCHEMAS]
        assert "query_knowledge_hub" in names


# ---------------------------------------------------------------------------
# Tests: Subprocess server startup (smoke)
# ---------------------------------------------------------------------------


class TestMCPServerSubprocess:
    """Verify the MCP server script can be invoked as a subprocess.

    These tests check that the server module loads without import errors.
    Full stdio MCP communication requires a live client, so we only
    verify startup here.
    """

    def test_server_script_exists(self) -> None:
        assert _SERVER_SCRIPT.exists()

    def test_server_import_no_errors(self) -> None:
        """Verify server module imports cleanly via subprocess."""
        result = subprocess.run(
            [sys.executable, "-c", "from mcp_server.server import create_server; print('OK')"],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT / "src"),
            timeout=10,
        )
        assert result.returncode == 0, f"Import failed: {result.stderr}"
        assert "OK" in result.stdout
