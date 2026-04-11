"""query_knowledge_hub — MCP tool for knowledge base retrieval.

Calls HybridSearch + Reranker and builds a formatted response with
inline citations and structured citation metadata.
"""

from __future__ import annotations

import json
from typing import Any

from core.response.response_builder import ResponseBuilder
from core.settings import load_settings
from core.types import RetrievalResult

# Module-level singletons (initialized lazily)
_hybrid = None
_reranker = None
_settings = None


def _get_components() -> tuple[Any, Any]:
    """Lazily initialize and cache HybridSearch + Reranker."""
    global _hybrid, _reranker, _settings
    if _hybrid is None:
        _settings = load_settings()
        from core.query_engine.hybrid_search import HybridSearch
        from core.query_engine.query_processor import QueryProcessor
        from core.query_engine.reranker import Reranker

        qp = QueryProcessor()
        _hybrid = HybridSearch(_settings, query_processor=qp)
        _reranker = Reranker(_settings)
    return _hybrid, _reranker


async def query_knowledge_hub(
    *,
    query: str,
    top_k: int = 10,
    collection: str = "default",
) -> str:
    """Search the knowledge base using hybrid retrieval.

    Args:
        query: Search query text.
        top_k: Number of results to return.
        collection: Collection to search.

    Returns:
        JSON string with ``text`` (Markdown) and ``citations``.
    """
    hybrid, reranker = _get_components()

    # Stage 1: Hybrid search (dense + sparse → RRF fusion)
    results: list[RetrievalResult] = hybrid.search(
        query,
        top_k=top_k,
        collection=collection,
    )

    # Stage 2: Rerank
    if results:
        results = reranker.rerank(query, results, top_k=top_k)

    # Stage 3: Build response
    response = ResponseBuilder.build(results, query)

    return json.dumps(response, ensure_ascii=False)
