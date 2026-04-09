"""Modular RAG MCP Server - Entry point."""

import sys

from core.settings import SettingsError, load_settings


def main() -> None:
    """Start the MCP Server."""
    try:
        settings = load_settings()
    except SettingsError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Modular RAG MCP Server", file=sys.stderr)
    print(f"  LLM: {settings.llm.provider}/{settings.llm.model}", file=sys.stderr)
    print(f"  Embedding: {settings.embedding.provider}/{settings.embedding.model}", file=sys.stderr)
    print(f"  VectorStore: {settings.vector_store.backend}", file=sys.stderr)
    print(f"  Status: ready", file=sys.stderr)

    # MCP Server will be implemented in Phase E


if __name__ == "__main__":
    main()
