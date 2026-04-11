"""Reranker — core-level reranking with fallback to fusion ranking.

Wraps ``libs.reranker`` backends and provides graceful fallback:
if the reranker backend fails or times out, the original fusion
ranking is preserved and ``fallback=True`` is marked in metadata.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from core.types import RetrievalResult
from libs.reranker.base_reranker import BaseReranker, RerankCandidate
from libs.reranker.reranker_factory import RerankerFactory

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


class Reranker:
    """Rerank fusion results with optional backend, fallback on failure.

    Args:
        settings: Application settings.
        backend: Optional pre-created reranker backend (for testing).
    """

    def __init__(
        self,
        settings: Settings,
        backend: BaseReranker | None = None,
    ) -> None:
        self._settings = settings
        self._backend = backend or RerankerFactory.create(settings)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        top_k: int | None = None,
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results with fallback.

        Args:
            query: Original user query.
            results: Fusion results to rerank.
            top_k: Number of results to return. Defaults to ``settings.rerank.top_m``.
            trace: Optional trace context.

        Returns:
            Reranked RetrievalResult list. On backend failure, returns
            original results with ``fallback=True`` in metadata.
        """
        if not results:
            return []

        if top_k is None:
            top_k = self._settings.rerank.top_m

        # Convert to RerankCandidate
        candidates = [
            RerankCandidate(
                id=r.chunk_id,
                text=r.text,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ]

        try:
            reranked = self._backend.rerank(query, candidates, top_k=top_k, trace=trace)
            return [
                RetrievalResult(
                    chunk_id=c.id,
                    score=c.score,
                    text=c.text,
                    metadata={**c.metadata, "reranked": True},
                    source="reranked",
                )
                for c in reranked
            ]
        except Exception:
            logger.warning("Reranker backend failed, using fusion ranking as fallback", exc_info=True)
            return [
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    score=r.score,
                    text=r.text,
                    metadata={**r.metadata, "fallback": True},
                    source=r.source,
                )
                for r in results[:top_k]
            ]
