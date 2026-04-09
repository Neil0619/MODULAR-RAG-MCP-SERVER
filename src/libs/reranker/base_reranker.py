"""Abstract base class for rerankers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RerankCandidate:
    """A candidate for reranking."""

    id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
    """Abstract base for reranking backends."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_k: int | None = None,
        trace: Any | None = None,
    ) -> list[RerankCandidate]:
        """Rerank candidates by relevance to query.

        Args:
            query: The user query.
            candidates: Candidates from fusion/recall stage.
            top_k: Return only top-k results. None = return all.
            trace: Optional TraceContext.

        Returns:
            Reordered candidates with updated scores.
        """
        ...


class NoneReranker(BaseReranker):
    """Pass-through reranker that preserves original order."""

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_k: int | None = None,
        trace: Any | None = None,
    ) -> list[RerankCandidate]:
        result = candidates
        if top_k is not None:
            result = result[:top_k]
        return result
