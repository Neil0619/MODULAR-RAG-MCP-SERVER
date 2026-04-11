"""Protocol Handler — JSON-RPC 2.0 protocol parsing for MCP.

Encapsulates request parsing, method routing, and error handling for
the Model Context Protocol. Methods: ``initialize``, ``tools/list``,
``tools/call``.

Error codes follow JSON-RPC 2.0:
    -32600 Invalid Request
    -32601 Method not found
    -32602 Invalid params
    -32603 Internal error
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# JSON-RPC 2.0 error codes
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"


@dataclass
class ToolSchema:
    """Schema definition for an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


class ProtocolHandler:
    """Handle JSON-RPC 2.0 requests for the MCP protocol.

    Routes ``initialize``, ``tools/list``, and ``tools/call`` methods
    to appropriate handlers with proper error handling.

    Args:
        tools: List of ToolSchema definitions.
        tool_executor: Callable ``(name, arguments) -> Any`` for tool execution.
    """

    def __init__(
        self,
        tools: list[ToolSchema] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> None:
        self._tools = {t.name: t for t in (tools or [])}
        self._tool_executor = tool_executor

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Parse and route a JSON-RPC 2.0 request.

        Args:
            request: Parsed JSON-RPC request object.

        Returns:
            JSON-RPC response dict.
        """
        req_id = request.get("id")

        # Validate JSON-RPC version
        if request.get("jsonrpc") != "2.0":
            return self._error_response(req_id, INVALID_REQUEST, "Invalid Request: missing or wrong jsonrpc version")

        method = request.get("method", "")
        params = request.get("params", {})

        # Route to handler
        if method == "initialize":
            return self._wrap(req_id, self.handle_initialize(params))
        elif method == "tools/list":
            return self._wrap(req_id, self.handle_tools_list())
        elif method == "tools/call":
            return self._wrap(req_id, self.handle_tools_call(params))
        elif method == "notifications/initialized":
            # Notification — no response needed, but return empty for simplicity
            return {}
        elif method == "shutdown":
            return self._wrap(req_id, {})
        else:
            return self._error_response(req_id, METHOD_NOT_FOUND, f"Method not found: {method}")

    def handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle ``initialize`` request.

        Returns server capabilities and info.
        """
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        }

    def handle_tools_list(self) -> dict[str, Any]:
        """Handle ``tools/list`` request.

        Returns registered tool schemas.
        """
        tools = [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in self._tools.values()
        ]
        return {"tools": tools}

    def handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle ``tools/call`` request.

        Routes to the tool executor and wraps results/errors.

        Args:
            params: Must contain ``name`` and optionally ``arguments``.
        """
        if not isinstance(params, dict):
            return self._error_dict(INVALID_PARAMS, "Invalid params: expected object")

        tool_name = params.get("name")
        if not tool_name:
            return self._error_dict(INVALID_PARAMS, "Invalid params: missing 'name'")

        if tool_name not in self._tools:
            return self._error_dict(METHOD_NOT_FOUND, f"Unknown tool: {tool_name}")

        if self._tool_executor is None:
            return self._error_dict(INTERNAL_ERROR, "No tool executor configured")

        arguments = params.get("arguments", {})

        try:
            result = self._tool_executor(tool_name, arguments)
            return {
                "content": [
                    {"type": "text", "text": str(result)},
                ],
            }
        except Exception as exc:
            logger.warning("Tool execution failed: %s", exc)
            return self._error_dict(INTERNAL_ERROR, "Internal error")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wrap(self, req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        """Wrap a result dict into a JSON-RPC response."""
        # If result already contains an error code, return as error
        if "code" in result and "message" in result:
            return {"jsonrpc": "2.0", "id": req_id, "error": result}
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error_response(req_id: Any, code: int, message: str) -> dict[str, Any]:
        """Build a JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }

    @staticmethod
    def _error_dict(code: int, message: str) -> dict[str, Any]:
        """Build an error dict (used inside result wrapping)."""
        return {"code": code, "message": message}

    def register_tool(self, schema: ToolSchema) -> None:
        """Register a tool schema."""
        self._tools[schema.name] = schema
