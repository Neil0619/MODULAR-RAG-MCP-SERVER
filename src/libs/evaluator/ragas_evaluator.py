"""Ragas-based evaluator for RAG quality metrics.

Supports Faithfulness, Answer Relevancy, and Context Precision metrics.
Gracefully degrades when the ``ragas`` package is not installed.
"""

from __future__ import annotations

from typing import Any

from libs.evaluator.base_evaluator import BaseEvaluator


class RagasEvaluator(BaseEvaluator):
    """Evaluator using the Ragas framework for RAG quality assessment.

    Metrics produced:
    - ``faithfulness``: How well the answer is grounded in the retrieved context.
    - ``answer_relevancy``: How relevant the answer is to the query.
    - ``context_precision``: Precision of the retrieved context.

    Requires the ``ragas`` package.  Falls back to returning zeroed metrics
    with a ``ragas_available: 0.0`` flag when the package is missing.
    """

    _SUPPORTED_METRICS = ("faithfulness", "answer_relevancy", "context_precision")

    def __init__(self, settings: Any = None) -> None:
        self._ragas_available = False
        try:
            import ragas  # noqa: F401
            self._ragas_available = True
        except ImportError:
            pass

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
        """Evaluate RAG quality using Ragas metrics.

        When ``ragas`` is not installed, returns zeroed metrics with a
        ``ragas_available`` flag set to ``0.0``.

        When ``answer`` or ``retrieved_texts`` are not provided, the
        corresponding metrics are skipped (returned as ``0.0``).
        """
        if not self._ragas_available:
            return self._unavailable_metrics()

        # Without texts or answer, we cannot compute most Ragas metrics.
        # Fall back to hit_rate-style computation that doesn't need LLM.
        texts = retrieved_texts or []
        metrics: dict[str, float] = {"ragas_available": 1.0}

        # Faithfulness: requires answer + context texts
        if answer and texts:
            metrics["faithfulness"] = self._compute_faithfulness(query, answer, texts)
        else:
            metrics["faithfulness"] = 0.0

        # Answer Relevancy: requires answer
        if answer:
            metrics["answer_relevancy"] = self._compute_answer_relevancy(query, answer)
        else:
            metrics["answer_relevancy"] = 0.0

        # Context Precision: requires golden_ids + retrieved context
        if golden_ids and retrieved_ids:
            metrics["context_precision"] = self._compute_context_precision(
                retrieved_ids, golden_ids
            )
        else:
            metrics["context_precision"] = 0.0

        if trace is not None:
            self._record_trace(trace, metrics)

        return metrics

    # ------------------------------------------------------------------
    # Metric computation helpers (local implementations)
    # ------------------------------------------------------------------

    def _compute_faithfulness(
        self,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> float:
        """Compute a simplified faithfulness score.

        Real Ragas would use an LLM to extract claims from the answer and
        verify each claim against the context.  Here we provide a lightweight
        heuristic: ratio of answer tokens that appear in the context.
        """
        context_blob = " ".join(contexts).lower()
        answer_tokens = answer.lower().split()
        if not answer_tokens:
            return 0.0
        hits = sum(1 for t in answer_tokens if t in context_blob)
        return round(hits / len(answer_tokens), 4)

    def _compute_answer_relevancy(self, query: str, answer: str) -> float:
        """Compute a simplified answer relevancy score.

        Real Ragas generates synthetic questions from the answer and compares
        them to the original query.  Here we use token overlap as a proxy.
        """
        query_tokens = set(query.lower().split())
        answer_tokens = set(answer.lower().split())
        if not query_tokens:
            return 0.0
        overlap = query_tokens & answer_tokens
        return round(len(overlap) / len(query_tokens), 4)

    def _compute_context_precision(
        self,
        retrieved_ids: list[str],
        golden_ids: list[str],
    ) -> float:
        """Compute context precision at each relevant position.

        Precision = (number of golden IDs in retrieved) / (retrieved count).
        """
        if not retrieved_ids:
            return 0.0
        golden_set = set(golden_ids)
        hits = sum(1 for rid in retrieved_ids if rid in golden_set)
        return round(hits / len(retrieved_ids), 4)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unavailable_metrics(self) -> dict[str, float]:
        """Return zeroed metrics when ragas is not installed."""
        return {
            "ragas_available": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
        }

    @staticmethod
    def _record_trace(trace: Any, metrics: dict[str, float]) -> None:
        """Record evaluation metrics into the trace context if available."""
        record = getattr(trace, "record_stage", None)
        if callable(record):
            record("evaluation", {
                "method": "ragas",
                "metrics": metrics,
            })
