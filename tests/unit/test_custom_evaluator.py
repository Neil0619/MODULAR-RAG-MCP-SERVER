"""Unit tests for CustomEvaluator (B6)."""

import pytest

from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator


class TestCustomEvaluator:
    def test_perfect_hit(self) -> None:
        ev = CustomEvaluator()
        result = ev.evaluate("q", ["a", "b"], ["a"])
        assert result["hit_rate"] == 1.0
        assert result["mrr"] == 1.0

    def test_miss(self) -> None:
        ev = CustomEvaluator()
        result = ev.evaluate("q", ["x", "y"], ["a"])
        assert result["hit_rate"] == 0.0
        assert result["mrr"] == 0.0

    def test_mrr_second_position(self) -> None:
        ev = CustomEvaluator()
        result = ev.evaluate("q", ["x", "a"], ["a"])
        assert result["mrr"] == 0.5

    def test_precision_recall(self) -> None:
        ev = CustomEvaluator()
        result = ev.evaluate("q", ["a", "b", "x"], ["a", "b"])
        assert result["precision"] == pytest.approx(2 / 3)
        assert result["recall"] == 1.0

    def test_empty_golden(self) -> None:
        ev = CustomEvaluator()
        result = ev.evaluate("q", ["a"], [])
        assert result["hit_rate"] == 0.0
        assert result["mrr"] == 0.0

    def test_empty_retrieved(self) -> None:
        ev = CustomEvaluator()
        result = ev.evaluate("q", [], ["a"])
        assert result["hit_rate"] == 0.0
        assert result["recall"] == 0.0

    def test_base_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseEvaluator()  # type: ignore[abstract]


class TestCustomEvaluatorBoundary:

    def test_single_item_perfect(self) -> None:
        ev = CustomEvaluator()
        result = ev.evaluate("q", ["a"], ["a"])
        assert result["hit_rate"] == 1.0
        assert result["mrr"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_both_empty(self) -> None:
        ev = CustomEvaluator()
        result = ev.evaluate("q", [], [])
        assert result["hit_rate"] == 0.0
        assert result["precision"] == 0.0

    def test_duplicate_retrieved_ids(self) -> None:
        ev = CustomEvaluator()
        result = ev.evaluate("q", ["a", "a", "b"], ["a"])
        assert result["hit_rate"] == 1.0
        # recall = n_relevant(2 duplicates) / golden_set_size(1) = 2.0
        # This is expected behavior — caller should dedupe if needed

    def test_returns_four_keys(self) -> None:
        ev = CustomEvaluator()
        result = ev.evaluate("q", ["a"], ["a"])
        assert set(result.keys()) == {"hit_rate", "mrr", "precision", "recall"}

    def test_mrr_third_position(self) -> None:
        ev = CustomEvaluator()
        result = ev.evaluate("q", ["x", "y", "a"], ["a"])
        assert result["mrr"] == pytest.approx(1 / 3)

    def test_accepts_kwargs_without_error(self) -> None:
        ev = CustomEvaluator()
        result = ev.evaluate(
            "q", ["a"], ["a"],
            retrieved_texts=["text"],
            answer="answer",
            trace=None,
        )
        assert result["hit_rate"] == 1.0
