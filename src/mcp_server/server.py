"""MCP Server entry point with Stdio transport.

Implements the Model Context Protocol server using the ``mcp`` SDK.
Stdout is reserved exclusively for MCP protocol messages; all logging
goes to stderr.

The server exposes tools for knowledge base retrieval (registered in
Phase E via ``mcp_server/tools/``).
"""

from __future__ import annotations

import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from observability.logger import get_logger

logger = get_logger("mcp_server")

# ---------------------------------------------------------------------------
# Tool definitions (schemas only — implementations in tools/)
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS: list[Tool] = [
    Tool(
        name="query_knowledge_hub",
        description="Search the knowledge base using hybrid retrieval (dense + sparse + RRF fusion). Returns top-k relevant document chunks with citations.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query text.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 10).",
                    "default": 10,
                },
                "collection": {
                    "type": "string",
                    "description": "Collection to search (default: 'default').",
                    "default": "default",
                },
            },
            "required": ["query"],
        },
    ),
]


def create_server(name: str = "modular-rag-mcp-server") -> Server:
    """Create and configure the MCP Server instance.

    Args:
        name: Server name for identification.

    Returns:
        Configured ``Server`` with tool registrations.
    """
    server = Server(name)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Return the list of available tools."""
        return _TOOL_SCHEMAS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Route tool calls to their implementations.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            List of TextContent results.
        """
        if name == "query_knowledge_hub":
            from mcp_server.tools.query_knowledge_hub import query_knowledge_hub

            result = await query_knowledge_hub(
                query=arguments["query"],
                top_k=arguments.get("top_k", 10),
                collection=arguments.get("collection", "default"),
            )
            return [TextContent(type="text", text=result)]

        raise ValueError(f"Unknown tool: {name}")

    return server


async def run_server(settings: Any = None) -> None:
    """Run the MCP server with stdio transport.

    Args:
        settings: Application settings. If ``None``, loaded from default config.
    """
    if settings is None:
        from core.settings import load_settings
        settings = load_settings()

    logger.info("Starting MCP Server (stdio transport)")
    logger.info("  LLM: %s/%s", settings.llm.provider, settings.llm.model)
    logger.info("  Embedding: %s/%s", settings.embedding.provider, settings.embedding.model)

    server = create_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

    logger.info("MCP Server stopped")
