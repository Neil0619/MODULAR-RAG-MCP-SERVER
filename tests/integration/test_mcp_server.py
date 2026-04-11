"""Integration tests for MCP Server (E1).

Tests the server via subprocess: sends MCP initialize request on stdin,
verifies the JSON-RPC response on stdout and logging on stderr.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SERVER_SCRIPT = _PROJECT_ROOT / "tests" / "integration" / "_mcp_server_helper.py"

# A minimal MCP initialize request (JSON-RPC 2.0)
_INIT_REQUEST = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {
            "name": "test-client",
            "version": "0.1.0",
        },
    },
}

_TOOLS_LIST_REQUEST = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {},
}


def _run_server_subprocess(input_lines: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    """Run the MCP server helper as a subprocess, feed input, capture output."""
    proc = subprocess.run(
        [sys.executable, str(_SERVER_SCRIPT)],
        input="\n".join(input_lines) + "\n",
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(_PROJECT_ROOT),
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": str(_PROJECT_ROOT / "src"),
        },
    )
    return proc


class TestMCPServer:
    """Integration tests for MCP Server via subprocess."""

    def test_initialize_returns_valid_response(self) -> None:
        """Server responds to initialize with capabilities and serverInfo."""
        proc = _run_server_subprocess([json.dumps(_INIT_REQUEST)])
        assert proc.returncode == 0, f"stderr: {proc.stderr}"

        output_lines = [l for l in proc.stdout.strip().split("\n") if l.strip()]
        assert len(output_lines) >= 1, f"No stdout output. stderr: {proc.stderr}"

        response = json.loads(output_lines[0])
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        result = response["result"]
        assert "capabilities" in result
        assert "serverInfo" in result
        assert "tools" in result["capabilities"]

    def test_stderr_has_logging(self) -> None:
        """Stderr should contain log output but stdout should be clean MCP."""
        proc = _run_server_subprocess([json.dumps(_INIT_REQUEST)])
        assert proc.returncode == 0

        # Stderr should have some logging
        assert len(proc.stderr.strip()) > 0

        # Stdout should only contain valid JSON
        for line in proc.stdout.strip().split("\n"):
            if line.strip():
                parsed = json.loads(line)  # should not raise
                assert "jsonrpc" in parsed

    def test_tools_list_returns_schemas(self) -> None:
        """Server responds to tools/list with tool schemas."""
        proc = _run_server_subprocess([
            json.dumps(_INIT_REQUEST),
            json.dumps(_INIT_REQUEST),  # initialized notification (also valid)
            json.dumps(_TOOLS_LIST_REQUEST),
        ])
        assert proc.returncode == 0, f"stderr: {proc.stderr}"

        output_lines = [l for l in proc.stdout.strip().split("\n") if l.strip()]
        # Find the tools/list response (id=2)
        tools_response = None
        for line in output_lines:
            parsed = json.loads(line)
            if parsed.get("id") == 2:
                tools_response = parsed
                break

        assert tools_response is not None, f"No tools/list response found. Output: {output_lines}"
        assert "result" in tools_response
        tools = tools_response["result"].get("tools", [])
        assert len(tools) >= 1

        tool_names = [t["name"] for t in tools]
        assert "query_knowledge_hub" in tool_names

    def test_unknown_method_returns_error(self) -> None:
        """Unknown method returns JSON-RPC error -32601."""
        unknown_req = {"jsonrpc": "2.0", "id": 3, "method": "nonexistent/method", "params": {}}
        proc = _run_server_subprocess([
            json.dumps(_INIT_REQUEST),
            json.dumps(unknown_req),
        ])
        assert proc.returncode == 0, f"stderr: {proc.stderr}"

        output_lines = [l for l in proc.stdout.strip().split("\n") if l.strip()]
        error_response = None
        for line in output_lines:
            parsed = json.loads(line)
            if parsed.get("id") == 3:
                error_response = parsed
                break

        assert error_response is not None, f"No error response found. Output: {output_lines}"
        assert "error" in error_response
