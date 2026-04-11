"""CompositeEvaluator — runs multiple evaluators and merges their metrics."""

from __future__ import annotations

from typing import Any

from libs.evaluator.base_evaluator import BaseEvaluator


class CompositeEvaluator(BaseEvaluator):
    """Combine multiple :class:`BaseEvaluator` instances.

    Runs each evaluator sequentially and merges all metric dicts into a
    single result.  If two evaluators produce the same metric key, the
    last one wins (later evaluators override earlier ones).

    Each evaluator's metrics are also namespaced under ``<name>::<metric>``
    for full traceability.
    """

    def __init__(self, evaluators: list[tuple[str, BaseEvaluator]]) -> None:
        self._evaluators = evaluators

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
        merged: dict[str, float] = {}

        for name, evaluator in self._evaluators:
            try:
                result = evaluator.evaluate(
                    query,
                    retrieved_ids,
                    golden_ids,
                    retrieved_texts=retrieved_texts,
                    answer=answer,
                    trace=trace,
                )
            except Exception:
                # Individual evaluator failure must not block others
                result = {}

            # Add namespaced keys for traceability
            for key, value in result.items():
                merged[f"{name}::{key}"] = value

            # Also merge flat (later evaluator wins on collision)
            merged.update(result)

        if trace is not None:
            self._record_trace(trace, merged)

        return merged

    @staticmethod
    def _record_trace(trace: Any, metrics: dict[str, float]) -> None:
        record = getattr(trace, "record_stage", None)
        if callable(record):
            record("evaluation", {
                "method": "composite",
                "evaluator_count": len(metrics),
            })

    @classmethod
    def from_settings(cls, settings: Any) -> CompositeEvaluator:
        """Create a CompositeEvaluator from Settings.

        Uses ``settings.evaluation.backends`` to determine which evaluators
        to load via :class:`EvaluatorFactory`.
        """
        from libs.evaluator.evaluator_factory import EvaluatorFactory

        backends: list[str] = []
        if hasattr(settings, "evaluation") and hasattr(settings.evaluation, "backends"):
            backends = settings.evaluation.backends

        if not backends:
            backends = ["custom"]

        evaluators: list[tuple[str, BaseEvaluator]] = []
        for name in backends:
            ev = EvaluatorFactory.create_from_name(name, settings)
            evaluators.append((name, ev))

        return cls(evaluators)
