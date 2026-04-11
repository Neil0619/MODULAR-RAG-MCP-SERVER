"""Unit tests for Reranker fallback behavior (D6)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.query_engine.reranker import Reranker
from core.settings import LLMSettings, RerankSettings, Settings
from core.types import RetrievalResult
from libs.reranker.base_reranker import BaseReranker, RerankCandidate


def _rr(cid: str, score: float, text: str = "", meta: dict | None = None, source: str = "fusion") -> RetrievalResult:
    return RetrievalResult(chunk_id=cid, score=score, text=text, metadata=meta or {}, source=source)


def _make_settings(backend: str = "none") -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o", api_key="fake"),
        rerank=RerankSettings(backend=backend, top_m=5),
    )


class TestRerankerFallback:
    """Tests for Reranker core layer with fallback."""

    def test_none_reranker_preserves_order(self) -> None:
        settings = _make_settings(backend="none")
        reranker = Reranker(settings)
        results = [_rr("a", 0.9), _rr("b", 0.8), _rr("c", 0.7)]
        out = reranker.rerank("test query", results)
        assert len(out) == 3
        assert out[0].chunk_id == "a"
        assert out[0].metadata.get("reranked") is True

    def test_top_m_limits_results(self) -> None:
        settings = _make_settings(backend="none")
        reranker = Reranker(settings)
        results = [_rr(f"c{i}", 0.9 - i * 0.1) for i in range(10)]
        out = reranker.rerank("test", results)
        assert len(out) == 5  # top_m=5

    def test_explicit_top_k_overrides(self) -> None:
        settings = _make_settings(backend="none")
        reranker = Reranker(settings)
        results = [_rr(f"c{i}", 0.9 - i * 0.1) for i in range(10)]
        out = reranker.rerank("test", results, top_k=3)
        assert len(out) == 3

    def test_empty_input_returns_empty(self) -> None:
        settings = _make_settings(backend="none")
        reranker = Reranker(settings)
        out = reranker.rerank("test", [])
        assert out == []

    def test_backend_failure_triggers_fallback(self) -> None:
        failing_backend = MagicMock(spec=BaseReranker)
        failing_backend.rerank.side_effect = RuntimeError("API timeout")

        settings = _make_settings(backend="none")
        reranker = Reranker(settings, backend=failing_backend)

        results = [_rr("a", 0.9, "text a"), _rr("b", 0.8, "text b")]
        out = reranker.rerank("test query", results)

        # Should return original results with fallback=True
        assert len(out) == 2
        assert out[0].metadata.get("fallback") is True
        assert out[0].chunk_id == "a"
        assert out[0].text == "text a"

    def test_fallback_preserves_original_score(self) -> None:
        failing_backend = MagicMock(spec=BaseReranker)
        failing_backend.rerank.side_effect = RuntimeError("fail")

        settings = _make_settings(backend="none")
        reranker = Reranker(settings, backend=failing_backend)

        results = [_rr("a", 0.95), _rr("b", 0.75)]
        out = reranker.rerank("test", results)

        assert out[0].score == 0.95
        assert out[1].score == 0.75

    def test_fallback_preserves_source(self) -> None:
        failing_backend = MagicMock(spec=BaseReranker)
        failing_backend.rerank.side_effect = RuntimeError("fail")

        settings = _make_settings(backend="none")
        reranker = Reranker(settings, backend=failing_backend)

        results = [_rr("a", 0.9, source="fusion")]
        out = reranker.rerank("test", results)

        assert out[0].source == "fusion"

    def test_successful_rerank_marks_reranked(self) -> None:
        mock_backend = MagicMock(spec=BaseReranker)
        mock_backend.rerank.return_value = [
            RerankCandidate(id="b", text="text b", score=0.99, metadata={}),
            RerankCandidate(id="a", text="text a", score=0.88, metadata={}),
        ]

        settings = _make_settings(backend="none")
        reranker = Reranker(settings, backend=mock_backend)

        results = [_rr("a", 0.9, "text a"), _rr("b", 0.8, "text b")]
        out = reranker.rerank("test", results)

        assert out[0].chunk_id == "b"  # reranked order
        assert out[0].metadata.get("reranked") is True
        assert out[0].source == "reranked"
        assert out[1].chunk_id == "a"

    def test_successful_rerank_no_fallback_flag(self) -> None:
        mock_backend = MagicMock(spec=BaseReranker)
        mock_backend.rerank.return_value = [
            RerankCandidate(id="a", text="text a", score=0.95, metadata={}),
        ]

        settings = _make_settings(backend="none")
        reranker = Reranker(settings, backend=mock_backend)

        results = [_rr("a", 0.9)]
        out = reranker.rerank("test", results)

        assert "fallback" not in out[0].metadata
