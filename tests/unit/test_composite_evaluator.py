"""Unit tests for CompositeEvaluator."""

from __future__ import annotations

from typing import Any

import pytest

from libs.evaluator.base_evaluator import BaseEvaluator
from observability.evaluation.composite_evaluator import CompositeEvaluator


# ---------------------------------------------------------------------------
# Stub evaluators
# ---------------------------------------------------------------------------


class _StubEvaluator(BaseEvaluator):
    """Returns a fixed metrics dict."""

    def __init__(self, metrics: dict[str, float]) -> None:
        self._metrics = metrics

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
        return dict(self._metrics)


class _FailingEvaluator(BaseEvaluator):
    """Always raises."""

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
        raise RuntimeError("evaluator failed")


class _FakeTrace:
    def __init__(self) -> None:
        self.stages: list[tuple[str, dict[str, Any]]] = []

    def record_stage(self, name: str, data: dict[str, Any]) -> None:
        self.stages.append((name, data))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompositeEvaluator:

    def test_single_evaluator(self) -> None:
        ev = _StubEvaluator({"hit_rate": 1.0, "mrr": 0.5})
        comp = CompositeEvaluator([("custom", ev)])
        result = comp.evaluate("q", ["a"], ["a"])
        assert result["hit_rate"] == 1.0
        assert result["mrr"] == 0.5

    def test_two_evaluators_merged(self) -> None:
        ev1 = _StubEvaluator({"hit_rate": 1.0})
        ev2 = _StubEvaluator({"faithfulness": 0.9})
        comp = CompositeEvaluator([("custom", ev1), ("ragas", ev2)])
        result = comp.evaluate("q", ["a"], ["a"])
        assert result["hit_rate"] == 1.0
        assert result["faithfulness"] == 0.9

    def test_namespaced_keys(self) -> None:
        ev1 = _StubEvaluator({"hit_rate": 1.0})
        ev2 = _StubEvaluator({"hit_rate": 0.8})
        comp = CompositeEvaluator([("custom", ev1), ("ragas", ev2)])
        result = comp.evaluate("q", ["a"], ["a"])
        # Namespaced keys preserve both
        assert result["custom::hit_rate"] == 1.0
        assert result["ragas::hit_rate"] == 0.8
        # Flat key: last evaluator wins
        assert result["hit_rate"] == 0.8

    def test_failing_evaluator_does_not_block(self) -> None:
        ev1 = _StubEvaluator({"hit_rate": 1.0})
        ev2 = _FailingEvaluator()
        comp = CompositeEvaluator([("custom", ev1), ("bad", ev2)])
        result = comp.evaluate("q", ["a"], ["a"])
        assert result["hit_rate"] == 1.0

    def test_empty_evaluators(self) -> None:
        comp = CompositeEvaluator([])
        result = comp.evaluate("q", ["a"], ["a"])
        assert result == {}

    def test_trace_recorded(self) -> None:
        ev = _StubEvaluator({"hit_rate": 1.0})
        comp = CompositeEvaluator([("custom", ev)])
        trace = _FakeTrace()
        comp.evaluate("q", ["a"], ["a"], trace=trace)
        assert len(trace.stages) == 1
        assert trace.stages[0][0] == "evaluation"
        assert trace.stages[0][1]["method"] == "composite"

    def test_no_trace_when_none(self) -> None:
        ev = _StubEvaluator({"hit_rate": 1.0})
        comp = CompositeEvaluator([("custom", ev)])
        result = comp.evaluate("q", ["a"], ["a"])
        assert result["hit_rate"] == 1.0

    def test_forward_kwargs(self) -> None:
        """Verify retrieved_texts and answer are forwarded."""

        class _SpyEvaluator(BaseEvaluator):
            def __init__(self) -> None:
                self.received_texts: list[str] | None = None
                self.received_answer: str | None = None

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
                self.received_texts = retrieved_texts
                self.received_answer = answer
                return {"ok": 1.0}

        spy = _SpyEvaluator()
        comp = CompositeEvaluator([("spy", spy)])
        comp.evaluate(
            "q", ["a"], ["a"],
            retrieved_texts=["hello"],
            answer="world",
        )
        assert spy.received_texts == ["hello"]
        assert spy.received_answer == "world"


class TestFromSettings:

    def test_from_settings_creates_composite(self) -> None:
        from core.settings import Settings

        settings = Settings()
        settings.evaluation.backends = ["custom"]
        comp = CompositeEvaluator.from_settings(settings)
        result = comp.evaluate("q", ["a"], ["a"])
        assert "hit_rate" in result

    def test_from_settings_multiple_backends(self) -> None:
        from core.settings import Settings

        settings = Settings()
        settings.evaluation.backends = ["custom", "ragas"]
        comp = CompositeEvaluator.from_settings(settings)
        result = comp.evaluate("q", ["a"], ["a"])
        # Should have custom metrics + ragas metrics (or ragas_available=0)
        assert "hit_rate" in result
