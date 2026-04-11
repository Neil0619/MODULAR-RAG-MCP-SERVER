"""Integration tests for HybridSearch (D5).

Uses mocked embedding / vector store / BM25 to avoid external deps,
but tests the full orchestration: query → dense + sparse → fusion → filter.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.reranker import Reranker
from core.settings import (
    EmbeddingSettings,
    LLMSettings,
    RetrievalSettings,
    RerankSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.reranker.base_reranker import BaseReranker, RerankCandidate
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


class TestHybridSearchTracing:
    """Tests for trace instrumentation in HybridSearch (F3)."""

    @pytest.fixture()
    def traced_setup(self) -> dict:
        dense_results = [
            _rr("c1", 0.95, "Dense one", "dense", {"source_path": "a.pdf"}),
            _rr("c2", 0.85, "Dense two", "dense", {"source_path": "b.pdf"}),
        ]
        sparse_results = [
            _rr("c2", 3.5, "Sparse two", "sparse", {"source_path": "b.pdf"}),
            _rr("c3", 2.0, "Sparse three", "sparse", {"source_path": "c.pdf"}),
        ]
        settings = _make_settings()
        dense = _make_mock_dense(dense_results)
        sparse = _make_mock_sparse(sparse_results)
        qp = QueryProcessor()
        trace = TraceContext(trace_type="query")

        hs = HybridSearch(
            settings=settings,
            query_processor=qp,
            dense_retriever=dense,
            sparse_retriever=sparse,
        )
        return {"hs": hs, "trace": trace, "settings": settings}

    def test_trace_records_query_processing(self, traced_setup: dict) -> None:
        trace: TraceContext = traced_setup["trace"]
        traced_setup["hs"].search("test query", trace=trace)

        assert "query_processing" in trace.stages
        stage = trace.stages["query_processing"]
        assert stage["method"] == "rule_based"
        assert "keywords" in stage
        assert "elapsed_ms" in stage

    def test_trace_records_dense_retrieval(self, traced_setup: dict) -> None:
        trace: TraceContext = traced_setup["trace"]
        traced_setup["hs"].search("test query", trace=trace)

        assert "dense_retrieval" in trace.stages
        stage = trace.stages["dense_retrieval"]
        assert stage["method"] == "embedding"
        assert "hit_count" in stage
        assert "elapsed_ms" in stage

    def test_trace_records_sparse_retrieval(self, traced_setup: dict) -> None:
        trace: TraceContext = traced_setup["trace"]
        traced_setup["hs"].search("test query", trace=trace)

        assert "sparse_retrieval" in trace.stages
        stage = trace.stages["sparse_retrieval"]
        assert stage["method"] == "bm25"
        assert "hit_count" in stage
        assert "elapsed_ms" in stage

    def test_trace_records_fusion(self, traced_setup: dict) -> None:
        trace: TraceContext = traced_setup["trace"]
        traced_setup["hs"].search("test query", trace=trace)

        assert "fusion" in trace.stages
        stage = trace.stages["fusion"]
        assert stage["algorithm"] == "rrf"
        assert "rrf_k" in stage
        assert "result_count" in stage
        assert "elapsed_ms" in stage

    def test_trace_type_is_query(self, traced_setup: dict) -> None:
        trace: TraceContext = traced_setup["trace"]
        traced_setup["hs"].search("test query", trace=trace)

        assert trace.trace_type == "query"
        d = trace.to_dict()
        assert d["trace_type"] == "query"

    def test_no_trace_still_works(self, traced_setup: dict) -> None:
        # Passing trace=None should not break anything
        results = traced_setup["hs"].search("test query", trace=None)
        assert len(results) > 0


class TestRerankerTracing:
    """Tests for trace instrumentation in Reranker (F3)."""

    def test_rerank_traces_success(self) -> None:
        settings = Settings(
            llm=LLMSettings(provider="openai", model="gpt-4o", api_key="fake"),
            embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small", api_key="fake"),
            splitter=SplitterSettings(),
            vector_store=VectorStoreSettings(backend="chroma"),
            retrieval=RetrievalSettings(),
            rerank=RerankSettings(backend="none"),
        )

        class FakeBackend(BaseReranker):
            def rerank(self, query, candidates, *, top_k=10, trace=None):
                return candidates[:top_k]

        reranker = Reranker(settings, backend=FakeBackend())
        trace = TraceContext(trace_type="query")

        results = [
            RetrievalResult(chunk_id="c1", score=0.9, text="text", metadata={}, source="fusion"),
            RetrievalResult(chunk_id="c2", score=0.8, text="text", metadata={}, source="fusion"),
        ]
        reranker.rerank("query", results, trace=trace)

        assert "rerank" in trace.stages
        stage = trace.stages["rerank"]
        assert stage["method"] == "none"
        assert "input_count" in stage
        assert "elapsed_ms" in stage

    def test_rerank_traces_fallback(self) -> None:
        settings = Settings(
            llm=LLMSettings(provider="openai", model="gpt-4o", api_key="fake"),
            embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small", api_key="fake"),
            splitter=SplitterSettings(),
            vector_store=VectorStoreSettings(backend="chroma"),
            retrieval=RetrievalSettings(),
            rerank=RerankSettings(backend="cross_encoder"),
        )

        class FailingBackend(BaseReranker):
            def rerank(self, query, candidates, *, top_k=10, trace=None):
                raise RuntimeError("model not loaded")

        reranker = Reranker(settings, backend=FailingBackend())
        trace = TraceContext(trace_type="query")

        results = [
            RetrievalResult(chunk_id="c1", score=0.9, text="text", metadata={}, source="fusion"),
        ]
        reranker.rerank("query", results, trace=trace)

        assert "rerank" in trace.stages
        stage = trace.stages["rerank"]
        assert stage["fallback"] is True
