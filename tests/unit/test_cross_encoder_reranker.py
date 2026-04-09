"""Unit tests for CrossEncoderReranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.settings import RerankSettings, Settings
from libs.reranker.base_reranker import RerankCandidate
from libs.reranker.cross_encoder_reranker import CrossEncoderReranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="c-0", text="Python programming language", score=0.0, metadata={}),
        RerankCandidate(id="c-1", text="Java virtual machine", score=0.0, metadata={}),
        RerankCandidate(id="c-2", text="Python web framework Django", score=0.0, metadata={}),
    ]


def _make_reranker_with_mock(mock_model: MagicMock) -> CrossEncoderReranker:
    """Build a CrossEncoderReranker with an injected mock model."""
    settings = Settings(rerank=RerankSettings(model="mock-model"))
    reranker = CrossEncoderReranker(settings)
    reranker._model = mock_model
    return reranker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    def test_reranking_changes_order_based_on_scores(self) -> None:
        mock_model = MagicMock()
        # c-2 should be most relevant, c-0 second, c-1 last
        mock_model.predict.return_value = np.array([0.5, 0.1, 0.9])

        reranker = _make_reranker_with_mock(mock_model)
        result = reranker.rerank("Python framework", _candidates())

        assert len(result) == 3
        assert result[0].id == "c-2"
        assert result[0].score == pytest.approx(0.9)
        assert result[1].id == "c-0"
        assert result[1].score == pytest.approx(0.5)
        assert result[2].id == "c-1"
        assert result[2].score == pytest.approx(0.1)

    def test_error_falls_back_to_original_order(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("Model inference error")

        reranker = _make_reranker_with_mock(mock_model)
        candidates = _candidates()
        result = reranker.rerank("Python framework", candidates)

        assert len(result) == 3
        assert [r.id for r in result] == ["c-0", "c-1", "c-2"]

    def test_top_k_limits_results(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.3, 0.8, 0.5])

        reranker = _make_reranker_with_mock(mock_model)
        result = reranker.rerank("Java", _candidates(), top_k=2)

        assert len(result) == 2
        # Top 2 by score: c-1 (0.8), c-2 (0.5)
        assert result[0].id == "c-1"
        assert result[1].id == "c-2"

    def test_empty_candidates_returns_empty(self) -> None:
        mock_model = MagicMock()
        reranker = _make_reranker_with_mock(mock_model)

        result = reranker.rerank("query", [])

        assert result == []
        mock_model.predict.assert_not_called()

    def test_none_model_falls_back_to_original_order(self) -> None:
        settings = Settings(rerank=RerankSettings(model="nonexistent-model"))
        reranker = CrossEncoderReranker(settings)
        reranker._model = None

        candidates = _candidates()
        result = reranker.rerank("query", candidates)

        assert len(result) == 3
        assert [r.id for r in result] == ["c-0", "c-1", "c-2"]
