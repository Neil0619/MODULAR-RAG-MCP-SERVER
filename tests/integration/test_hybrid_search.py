"""Integration tests for HybridSearch (D5).

Uses mocked embedding / vector store / BM25 to avoid external deps,
but tests the full orchestration: query → dense + sparse → fusion → filter.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.query_processor import QueryProcessor
from core.settings import (
    EmbeddingSettings,
    LLMSettings,
    RetrievalSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)
from core.types import RetrievalResult
from libs.vector_store.base_vector_store import QueryResult


def _rr(cid: str, score: float, text: str = "", source: str = "", meta: dict | None = None) -> RetrievalResult:
    return RetrievalResult(chunk_id=cid, score=score, text=text, metadata=meta or {}, source=source)


def _make_settings() -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o", api_key="fake"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small", api_key="fake"),
        splitter=SplitterSettings(),
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(rrf_k=60, top_k_dense=20, top_k_sparse=20, top_k_final=10),
    )


def _make_mock_dense(results: list[RetrievalResult]) -> MagicMock:
    mock = MagicMock()
    mock.retrieve.return_value = results
    return mock


def _make_mock_sparse(results: list[RetrievalResult]) -> MagicMock:
    mock = MagicMock()
    mock.retrieve.return_value = results
    return mock


class TestHybridSearch:
    """Integration tests for HybridSearch orchestration."""

    @pytest.fixture()
    def setup(self) -> dict:
        dense_results = [
            _rr("c1", 0.95, "Dense result one", "dense", {"source_path": "a.pdf", "doc_type": "pdf"}),
            _rr("c2", 0.85, "Dense result two", "dense", {"source_path": "a.pdf", "doc_type": "pdf"}),
            _rr("c3", 0.75, "Dense result three", "dense", {"source_path": "b.pdf", "doc_type": "pdf"}),
        ]
        sparse_results = [
            _rr("c2", 3.5, "Sparse result two", "sparse", {"source_path": "a.pdf", "doc_type": "pdf"}),
            _rr("c4", 2.0, "Sparse result four", "sparse", {"source_path": "c.pdf", "doc_type": "md"}),
            _rr("c5", 1.5, "Sparse result five", "sparse", {"source_path": "b.pdf", "doc_type": "pdf"}),
        ]

        settings = _make_settings()
        dense = _make_mock_dense(dense_results)
        sparse = _make_mock_sparse(sparse_results)
        qp = QueryProcessor()

        hs = HybridSearch(
            settings=settings,
            query_processor=qp,
            dense_retriever=dense,
            sparse_retriever=sparse,
        )
        return {"hs": hs, "dense": dense, "sparse": sparse, "settings": settings}

    def test_basic_search(self, setup: dict) -> None:
        hs: HybridSearch = setup["hs"]
        results = hs.search("azure openai setup")
        assert len(results) > 0
        # All results should have source="fusion"
        for r in results:
            assert r.source == "fusion"
        # Results should be sorted by score desc
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_dense_and_sparse_called(self, setup: dict) -> None:
        hs: HybridSearch = setup["hs"]
        hs.search("test query")
        setup["dense"].retrieve.assert_called_once()
        setup["sparse"].retrieve.assert_called_once()

    def test_top_k_limits_results(self, setup: dict) -> None:
        hs: HybridSearch = setup["hs"]
        results = hs.search("test", top_k=2)
        assert len(results) <= 2

    def test_metadata_filter(self, setup: dict) -> None:
        hs: HybridSearch = setup["hs"]
        results = hs.search("test", filters={"doc_type": "pdf"})
        for r in results:
            # Either metadata has no doc_type (passes), or matches "pdf"
            dt = r.metadata.get("doc_type")
            assert dt is None or dt == "pdf"

    def test_metadata_filter_excludes(self, setup: dict) -> None:
        hs: HybridSearch = setup["hs"]
        results = hs.search("test", filters={"doc_type": "md"})
        # Only c4 has doc_type=md, but it may or may not survive fusion + filter
        for r in results:
            dt = r.metadata.get("doc_type")
            assert dt is None or dt == "md"

    def test_collection_passed_to_retrievers(self, setup: dict) -> None:
        hs: HybridSearch = setup["hs"]
        hs.search("test", collection="research")
        dense_call = setup["dense"].retrieve.call_args
        sparse_call = setup["sparse"].retrieve.call_args
        assert dense_call[1]["collection"] == "research"
        assert sparse_call[1]["collection"] == "research"

    def test_dense_failure_degrades_gracefully(self, setup: dict) -> None:
        setup["dense"].retrieve.side_effect = RuntimeError("API down")
        hs: HybridSearch = setup["hs"]
        results = hs.search("test query")
        # Should still return sparse results
        assert len(results) > 0
        assert all(r.source == "fusion" for r in results)

    def test_sparse_failure_degrades_gracefully(self, setup: dict) -> None:
        setup["sparse"].retrieve.side_effect = RuntimeError("Index missing")
        hs: HybridSearch = setup["hs"]
        results = hs.search("test query")
        assert len(results) > 0

    def test_both_failures_returns_empty(self, setup: dict) -> None:
        setup["dense"].retrieve.side_effect = RuntimeError("fail")
        setup["sparse"].retrieve.side_effect = RuntimeError("fail")
        hs: HybridSearch = setup["hs"]
        results = hs.search("test query")
        assert results == []

    def test_keywords_passed_to_sparse(self, setup: dict) -> None:
        hs: HybridSearch = setup["hs"]
        hs.search("machine learning algorithms")
        sparse_call = setup["sparse"].retrieve.call_args
        keywords = sparse_call[0][0]
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "machine" in keywords
        assert "learning" in keywords

    def test_deduplication_across_retrievers(self, setup: dict) -> None:
        hs: HybridSearch = setup["hs"]
        results = hs.search("test")
        chunk_ids = [r.chunk_id for r in results]
        # No duplicates
        assert len(chunk_ids) == len(set(chunk_ids))
