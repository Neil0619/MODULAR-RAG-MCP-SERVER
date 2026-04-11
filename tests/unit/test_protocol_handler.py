"""Unit tests for ProtocolHandler (E2)."""

from __future__ import annotations

from typing import Any

import pytest

from mcp_server.protocol_handler import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    ProtocolHandler,
    ToolSchema,
)


def _tool(name: str = "test_tool", desc: str = "A test tool") -> ToolSchema:
    return ToolSchema(
        name=name,
        description=desc,
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )


def _executor(name: str, arguments: dict[str, Any]) -> str:
    return f"result from {name}: {arguments}"


class TestProtocolHandler:
    """Tests for JSON-RPC 2.0 protocol handling."""

    @pytest.fixture()
    def handler(self) -> ProtocolHandler:
        return ProtocolHandler(
            tools=[_tool()],
            tool_executor=_executor,
        )

    # -- initialize --

    def test_initialize_returns_capabilities(self, handler: ProtocolHandler) -> None:
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        resp = handler.handle_request(req)
        assert resp["id"] == 1
        assert resp["jsonrpc"] == "2.0"
        result = resp["result"]
        assert "capabilities" in result
        assert "tools" in result["capabilities"]
        assert result["serverInfo"]["name"] == "modular-rag-mcp-server"

    def test_initialize_protocol_version(self, handler: ProtocolHandler) -> None:
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        resp = handler.handle_request(req)
        assert resp["result"]["protocolVersion"] == "2024-11-05"

    # -- tools/list --

    def test_tools_list_returns_schemas(self, handler: ProtocolHandler) -> None:
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        resp = handler.handle_request(req)
        tools = resp["result"]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"
        assert "inputSchema" in tools[0]

    def test_tools_list_empty(self) -> None:
        handler = ProtocolHandler(tools=[], tool_executor=_executor)
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        resp = handler.handle_request(req)
        assert resp["result"]["tools"] == []

    # -- tools/call --

    def test_tools_call_routes_correctly(self, handler: ProtocolHandler) -> None:
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "test_tool", "arguments": {"query": "hello"}},
        }
        resp = handler.handle_request(req)
        assert resp["id"] == 3
        content = resp["result"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert "test_tool" in content[0]["text"]

    def test_tools_call_missing_name(self, handler: ProtocolHandler) -> None:
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"arguments": {"query": "hello"}},
        }
        resp = handler.handle_request(req)
        assert resp["error"]["code"] == INVALID_PARAMS

    def test_tools_call_unknown_tool(self, handler: ProtocolHandler) -> None:
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "nonexistent", "arguments": {}},
        }
        resp = handler.handle_request(req)
        assert resp["error"]["code"] == METHOD_NOT_FOUND

    def test_tools_call_no_executor(self) -> None:
        handler = ProtocolHandler(tools=[_tool()], tool_executor=None)
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "test_tool", "arguments": {}},
        }
        resp = handler.handle_request(req)
        assert resp["error"]["code"] == INTERNAL_ERROR

    def test_tools_call_executor_exception(self) -> None:
        def bad_executor(name, args):
            raise RuntimeError("boom")

        handler = ProtocolHandler(tools=[_tool()], tool_executor=bad_executor)
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "test_tool", "arguments": {}},
        }
        resp = handler.handle_request(req)
        # Should return internal error, not leak stack trace
        assert resp["error"]["code"] == INTERNAL_ERROR
        assert "boom" not in resp["error"]["message"]

    def test_tools_call_params_not_dict(self, handler: ProtocolHandler) -> None:
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": "not a dict",
        }
        resp = handler.handle_request(req)
        assert resp["error"]["code"] == INVALID_PARAMS

    # -- method routing errors --

    def test_unknown_method_returns_32601(self, handler: ProtocolHandler) -> None:
        req = {"jsonrpc": "2.0", "id": 4, "method": "foo/bar", "params": {}}
        resp = handler.handle_request(req)
        assert resp["error"]["code"] == METHOD_NOT_FOUND

    def test_missing_jsonrpc_version(self, handler: ProtocolHandler) -> None:
        req = {"id": 1, "method": "initialize", "params": {}}
        resp = handler.handle_request(req)
        assert resp["error"]["code"] == -32600

    def test_wrong_jsonrpc_version(self, handler: ProtocolHandler) -> None:
        req = {"jsonrpc": "1.0", "id": 1, "method": "initialize", "params": {}}
        resp = handler.handle_request(req)
        assert resp["error"]["code"] == -32600

    # -- register_tool --

    def test_register_tool_adds_to_list(self) -> None:
        handler = ProtocolHandler(tools=[], tool_executor=_executor)
        handler.register_tool(_tool("new_tool", "New tool"))
        req = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        resp = handler.handle_request(req)
        assert len(resp["result"]["tools"]) == 1
        assert resp["result"]["tools"][0]["name"] == "new_tool"

    # -- notifications --

    def test_initialized_notification_returns_empty(self, handler: ProtocolHandler) -> None:
        req = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        resp = handler.handle_request(req)
        assert resp == {}
