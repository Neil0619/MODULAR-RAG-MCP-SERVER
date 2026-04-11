"""E2E tests for the query.py script (D7)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.types import RetrievalResult


def _make_mock_hybrid(results: list[RetrievalResult]) -> MagicMock:
    mock = MagicMock()
    mock.search.return_value = results
    return mock


def _make_mock_reranker(results: list[RetrievalResult]) -> MagicMock:
    mock = MagicMock()
    mock.rerank.side_effect = lambda q, r, **kw: r[:kw.get("top_k", len(r))]
    return mock


_SAMPLE_RESULTS = [
    RetrievalResult(
        chunk_id="c1", score=0.95, text="Azure OpenAI configuration guide",
        metadata={"source_path": "azure.pdf", "chunk_index": 0, "title": "Azure Setup"},
    ),
    RetrievalResult(
        chunk_id="c2", score=0.85, text="RAG pipeline architecture overview",
        metadata={"source_path": "rag.pdf", "chunk_index": 2, "title": "RAG Overview"},
    ),
]


class TestQueryScript:
    """E2E tests for scripts/query.py."""

    def test_basic_query(self) -> None:
        from scripts.query import main as query_main

        mock_hybrid = _make_mock_hybrid(_SAMPLE_RESULTS)
        mock_reranker = _make_mock_reranker(_SAMPLE_RESULTS)

        with (
            patch("scripts.query.load_settings", return_value=MagicMock()),
            patch("scripts.query.HybridSearch", return_value=mock_hybrid),
            patch("scripts.query.Reranker", return_value=mock_reranker),
        ):
            ret = query_main(["--query", "how to configure Azure", "--config", "fake"])

        assert ret == 0
        mock_hybrid.search.assert_called_once()

    def test_top_k_passed(self) -> None:
        from scripts.query import main as query_main

        mock_hybrid = _make_mock_hybrid(_SAMPLE_RESULTS)

        with (
            patch("scripts.query.load_settings", return_value=MagicMock()),
            patch("scripts.query.HybridSearch", return_value=mock_hybrid),
            patch("scripts.query.Reranker", return_value=_make_mock_reranker(_SAMPLE_RESULTS)),
        ):
            ret = query_main(["--query", "test", "--top-k", "5", "--config", "fake"])

        assert ret == 0
        call_kwargs = mock_hybrid.search.call_args
        assert call_kwargs[1]["top_k"] == 5

    def test_collection_passed(self) -> None:
        from scripts.query import main as query_main

        mock_hybrid = _make_mock_hybrid(_SAMPLE_RESULTS)

        with (
            patch("scripts.query.load_settings", return_value=MagicMock()),
            patch("scripts.query.HybridSearch", return_value=mock_hybrid),
            patch("scripts.query.Reranker", return_value=_make_mock_reranker(_SAMPLE_RESULTS)),
        ):
            ret = query_main(["--query", "test", "--collection", "research", "--config", "fake"])

        assert ret == 0
        call_kwargs = mock_hybrid.search.call_args
        assert call_kwargs[1]["collection"] == "research"

    def test_no_results_returns_zero(self) -> None:
        from scripts.query import main as query_main

        mock_hybrid = _make_mock_hybrid([])

        with (
            patch("scripts.query.load_settings", return_value=MagicMock()),
            patch("scripts.query.HybridSearch", return_value=mock_hybrid),
            patch("scripts.query.Reranker", return_value=_make_mock_reranker([])),
        ):
            ret = query_main(["--query", "nonexistent", "--config", "fake"])

        assert ret == 0

    def test_no_rerank_skips_reranker(self) -> None:
        from scripts.query import main as query_main

        mock_hybrid = _make_mock_hybrid(_SAMPLE_RESULTS)

        with (
            patch("scripts.query.load_settings", return_value=MagicMock()),
            patch("scripts.query.HybridSearch", return_value=mock_hybrid),
            patch("scripts.query.Reranker") as MockReranker,
        ):
            ret = query_main(["--query", "test", "--no-rerank", "--config", "fake"])

        assert ret == 0
        MockReranker.assert_not_called()

    def test_verbose_mode(self) -> None:
        from scripts.query import main as query_main

        mock_hybrid = _make_mock_hybrid(_SAMPLE_RESULTS)

        with (
            patch("scripts.query.load_settings", return_value=MagicMock()),
            patch("scripts.query.HybridSearch", return_value=mock_hybrid),
            patch("scripts.query.Reranker", return_value=_make_mock_reranker(_SAMPLE_RESULTS)),
        ):
            ret = query_main(["--query", "test", "--verbose", "--config", "fake"])

        assert ret == 0
