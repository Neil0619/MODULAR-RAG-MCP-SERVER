"""Abstract base class for evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    """Abstract base for RAG evaluation backends."""

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        *,
        retrieved_texts: list[str] | None = None,
        answer: str | None = None,
        trace: Any | None = None,
    ) -> dict[str, float]:
        """Evaluate retrieval/generation quality.

        Args:
            query: The user query.
            retrieved_ids: IDs returned by the retrieval pipeline.
            golden_ids: Expected (ground truth) IDs.
            retrieved_texts: Optional texts for faithfulness metrics.
            answer: Optional generated answer for answer-quality metrics.
            trace: Optional TraceContext.

        Returns:
            Dict of metric_name -> score (float in [0, 1]).
        """
        ...
