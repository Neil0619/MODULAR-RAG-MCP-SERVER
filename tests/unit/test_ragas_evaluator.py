"""Unit tests for RagasEvaluator."""

from __future__ import annotations

from typing import Any

import pytest

from libs.evaluator.ragas_evaluator import RagasEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeTrace:
    """Minimal trace stub that records stage calls."""

    def __init__(self) -> None:
        self.stages: list[tuple[str, dict[str, Any]]] = []

    def record_stage(self, name: str, data: dict[str, Any]) -> None:
        self.stages.append((name, data))


def _make_evaluator() -> RagasEvaluator:
    return RagasEvaluator(settings=None)


# ---------------------------------------------------------------------------
# Unavailable / import guard
# ---------------------------------------------------------------------------


class TestRagasUnavailable:

    def test_returns_zeroed_metrics_when_no_ragas(self) -> None:
        evaluator = _make_evaluator()
        # If ragas is not installed, _ragas_available is False
        if evaluator._ragas_available:
            pytest.skip("ragas is installed; skipping unavailability test")
        result = evaluator.evaluate("q", ["a"], ["a"])
        assert result["ragas_available"] == 0.0
        assert result["faithfulness"] == 0.0
        assert result["answer_relevancy"] == 0.0
        assert result["context_precision"] == 0.0

    def test_unavailable_metrics_has_four_keys(self) -> None:
        evaluator = _make_evaluator()
        metrics = evaluator._unavailable_metrics()
        assert set(metrics.keys()) == {
            "ragas_available",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
        }


# ---------------------------------------------------------------------------
# Core metrics (work even without ragas installed)
# ---------------------------------------------------------------------------


class TestContextPrecision:

    def test_perfect_precision(self) -> None:
        ev = _make_evaluator()
        score = ev._compute_context_precision(["a", "b"], ["a", "b"])
        assert score == 1.0

    def test_partial_precision(self) -> None:
        ev = _make_evaluator()
        score = ev._compute_context_precision(["a", "c", "d"], ["a", "b"])
        # 1 hit out of 3 retrieved
        assert score == pytest.approx(1 / 3, abs=0.01)

    def test_zero_precision(self) -> None:
        ev = _make_evaluator()
        score = ev._compute_context_precision(["x", "y"], ["a", "b"])
        assert score == 0.0

    def test_empty_retrieved(self) -> None:
        ev = _make_evaluator()
        score = ev._compute_context_precision([], ["a"])
        assert score == 0.0


class TestFaithfulness:

    def test_full_overlap(self) -> None:
        ev = _make_evaluator()
        context = ["the cat sat on the mat"]
        score = ev._compute_faithfulness("where is the cat", "the cat sat on the mat", context)
        assert score == 1.0

    def test_partial_overlap(self) -> None:
        ev = _make_evaluator()
        context = ["the cat sat on the mat"]
        score = ev._compute_faithfulness("q", "cat mat table", context)
        # "cat" and "mat" match, "table" does not → 2/3
        assert 0 < score < 1.0

    def test_empty_answer(self) -> None:
        ev = _make_evaluator()
        score = ev._compute_faithfulness("q", "", ["some context"])
        assert score == 0.0


class TestAnswerRelevancy:

    def test_full_overlap(self) -> None:
        ev = _make_evaluator()
        score = ev._compute_answer_relevancy("what is rag", "rag is retrieval augmented generation")
        # "rag" and "is" appear in both (lowercased)
        assert score > 0.0

    def test_no_overlap(self) -> None:
        ev = _make_evaluator()
        score = ev._compute_answer_relevancy("python testing", "java debugging tool")
        # No overlapping tokens
        assert score == 0.0

    def test_empty_query(self) -> None:
        ev = _make_evaluator()
        score = ev._compute_answer_relevancy("", "some answer")
        assert score == 0.0


# ---------------------------------------------------------------------------
# Full evaluate() integration
# ---------------------------------------------------------------------------


class TestEvaluateIntegration:

    def test_returns_all_metric_keys(self) -> None:
        ev = _make_evaluator()
        result = ev.evaluate(
            "what is rag",
            retrieved_ids=["c1", "c2"],
            golden_ids=["c1"],
            retrieved_texts=["rag is a technique"],
            answer="rag stands for retrieval augmented generation",
        )
        assert "faithfulness" in result
        assert "answer_relevancy" in result
        assert "context_precision" in result

    def test_missing_answer_zeros_relevancy(self) -> None:
        ev = _make_evaluator()
        result = ev.evaluate(
            "query",
            retrieved_ids=["a"],
            golden_ids=["a"],
        )
        if not ev._ragas_available:
            pytest.skip("ragas not installed")
        assert result["answer_relevancy"] == 0.0

    def test_missing_texts_zeros_faithfulness(self) -> None:
        ev = _make_evaluator()
        result = ev.evaluate(
            "query",
            retrieved_ids=["a"],
            golden_ids=["a"],
            answer="some answer",
        )
        if not ev._ragas_available:
            pytest.skip("ragas not installed")
        assert result["faithfulness"] == 0.0

    def test_trace_recorded(self) -> None:
        ev = _make_evaluator()
        if not ev._ragas_available:
            pytest.skip("ragas not installed")
        trace = _FakeTrace()
        ev.evaluate(
            "query",
            retrieved_ids=["a"],
            golden_ids=["a"],
            retrieved_texts=["text"],
            answer="answer",
            trace=trace,
        )
        assert len(trace.stages) == 1
        assert trace.stages[0][0] == "evaluation"
        assert trace.stages[0][1]["method"] == "ragas"
        assert "metrics" in trace.stages[0][1]

    def test_no_trace_when_none(self) -> None:
        ev = _make_evaluator()
        # Should not raise
        result = ev.evaluate("q", ["a"], ["a"])
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestFactoryIntegration:

    def test_factory_creates_ragas(self) -> None:
        from core.settings import Settings
        from libs.evaluator.evaluator_factory import EvaluatorFactory

        settings = Settings()
        settings.evaluation.backends = ["ragas"]
        evaluator = EvaluatorFactory.create(settings)
        assert isinstance(evaluator, RagasEvaluator)

    def test_factory_unknown_raises(self) -> None:
        from core.settings import Settings
        from libs.evaluator.evaluator_factory import EvaluatorFactory

        settings = Settings()
        with pytest.raises(ValueError, match="Unknown evaluator"):
            EvaluatorFactory.create_from_name("nonexistent", settings)
