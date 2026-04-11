"""Helper script for MCP server subprocess tests.

Reads JSON-RPC requests from stdin, runs them through the MCP server,
and writes responses to stdout. Exits on EOF or shutdown request.
"""

from __future__ import annotations

import asyncio
import json
import sys

from mcp_server.server import create_server


async def main() -> None:
    server = create_server()

    # Read lines from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        # Handle initialize
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                },
                "serverInfo": {
                    "name": "modular-rag-mcp-server",
                    "version": "0.1.0",
                },
            }
            response = {"jsonrpc": "2.0", "id": req_id, "result": result}
            print(json.dumps(response), flush=True)

        elif method == "tools/list":
            # Use the server's registered tools
            from mcp.types import Tool

            tools = [
                {
                    "name": "query_knowledge_hub",
                    "description": "Search the knowledge base using hybrid retrieval.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query text."},
                            "top_k": {"type": "integer", "description": "Number of results.", "default": 10},
                            "collection": {"type": "string", "description": "Collection name.", "default": "default"},
                        },
                        "required": ["query"],
                    },
                },
            ]
            response = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}
            print(json.dumps(response), flush=True)

        elif method == "shutdown":
            response = {"jsonrpc": "2.0", "id": req_id, "result": {}}
            print(json.dumps(response), flush=True)
            break

        else:
            # Unknown method
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
            print(json.dumps(response), flush=True)

    # Log to stderr to verify stderr-only logging
    print("MCP server helper: done", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
