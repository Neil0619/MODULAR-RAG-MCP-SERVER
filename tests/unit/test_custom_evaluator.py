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
