"""Unit tests for SparseRetriever (D3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.query_engine.sparse_retriever import SparseRetriever
from core.types import RetrievalResult
from ingestion.storage.bm25_indexer import BM25Indexer


class TestSparseRetriever:
    """Tests for SparseRetriever."""

    @pytest.fixture()
    def mock_bm25(self) -> MagicMock:
        bm25 = MagicMock(spec=BM25Indexer)
        bm25.query.return_value = [
            ("chunk_001", 3.5),
            ("chunk_005", 2.1),
            ("chunk_010", 1.0),
        ]
        return bm25

    @pytest.fixture()
    def mock_store(self) -> MagicMock:
        store = MagicMock()
        store.get_by_ids.return_value = [
            {"id": "chunk_001", "text": "Azure OpenAI setup guide", "metadata": {"source_path": "a.pdf"}},
            {"id": "chunk_005", "text": "RAG pipeline overview", "metadata": {"source_path": "b.pdf"}},
            {"id": "chunk_010", "text": "BM25 indexing details", "metadata": {"source_path": "c.pdf"}},
        ]
        return store

    @pytest.fixture()
    def retriever(self, mock_bm25: MagicMock, mock_store: MagicMock) -> SparseRetriever:
        return SparseRetriever(
            settings=MagicMock(),
            bm25_indexer=mock_bm25,
            store=mock_store,
        )

    def test_basic_retrieve(self, retriever: SparseRetriever) -> None:
        results = retriever.retrieve(["azure", "openai"])
        assert len(results) == 3
        assert results[0].chunk_id == "chunk_001"
        assert results[0].score == 3.5
        assert results[0].text == "Azure OpenAI setup guide"
        assert results[0].source == "sparse"

    def test_results_sorted_by_score_desc(self, retriever: SparseRetriever) -> None:
        results = retriever.retrieve(["test"])
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_results_have_source_sparse(self, retriever: SparseRetriever) -> None:
        results = retriever.retrieve(["query"])
        for r in results:
            assert r.source == "sparse"

    def test_bm25_load_called_with_collection(self, retriever: SparseRetriever, mock_bm25: MagicMock) -> None:
        retriever.retrieve(["test"], collection="research")
        mock_bm25.load.assert_called_once_with("research")

    def test_bm25_query_called_with_keywords(self, retriever: SparseRetriever, mock_bm25: MagicMock) -> None:
        retriever.retrieve(["machine", "learning"])
        mock_bm25.query.assert_called_once_with(["machine", "learning"], top_k=10)

    def test_top_k_passed_to_bm25(self, retriever: SparseRetriever, mock_bm25: MagicMock) -> None:
        retriever.retrieve(["test"], top_k=5)
        mock_bm25.query.assert_called_once_with(["test"], top_k=5)

    def test_store_get_by_ids_called(self, retriever: SparseRetriever, mock_store: MagicMock) -> None:
        retriever.retrieve(["test"], collection="col")
        mock_store.get_by_ids.assert_called_once()
        call_args = mock_store.get_by_ids.call_args
        assert call_args[0][0] == ["chunk_001", "chunk_005", "chunk_010"]
        assert call_args[1]["collection"] == "col"

    def test_empty_keywords_returns_empty(self, retriever: SparseRetriever) -> None:
        results = retriever.retrieve([])
        assert results == []

    def test_no_bm25_hits_returns_empty(self, mock_bm25: MagicMock, mock_store: MagicMock) -> None:
        mock_bm25.query.return_value = []
        retriever = SparseRetriever(
            settings=MagicMock(),
            bm25_indexer=mock_bm25,
            store=mock_store,
        )
        results = retriever.retrieve(["nonexistent"])
        assert results == []

    def test_partial_enrichment(self, mock_bm25: MagicMock, mock_store: MagicMock) -> None:
        """Store returns fewer records than BM25 (some IDs missing)."""
        mock_store.get_by_ids.return_value = [
            {"id": "chunk_001", "text": "Found text", "metadata": {}},
        ]
        retriever = SparseRetriever(
            settings=MagicMock(),
            bm25_indexer=mock_bm25,
            store=mock_store,
        )
        results = retriever.retrieve(["test"])
        assert len(results) == 3  # Still 3 from BM25
        assert results[0].text == "Found text"
        assert results[1].text == ""  # Missing in store
        assert results[2].text == ""

    def test_no_store_uses_empty_text(self) -> None:
        """Without a store, text/metadata are empty."""
        bm25 = MagicMock(spec=BM25Indexer)
        bm25.query.return_value = [("c1", 2.0)]
        retriever = SparseRetriever(
            settings=MagicMock(),
            bm25_indexer=bm25,
            store=None,
        )
        results = retriever.retrieve(["test"])
        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert results[0].text == ""
        assert results[0].metadata == {}

    def test_metadata_preserved(self, retriever: SparseRetriever) -> None:
        results = retriever.retrieve(["test"])
        assert results[0].metadata["source_path"] == "a.pdf"
        assert results[1].metadata["source_path"] == "b.pdf"

    def test_retrieval_result_roundtrip(self) -> None:
        rr = RetrievalResult(
            chunk_id="c1", score=3.5, text="hello",
            metadata={"k": "v"}, source="sparse",
        )
        d = rr.to_dict()
        restored = RetrievalResult.from_dict(d)
        assert restored.chunk_id == rr.chunk_id
        assert restored.score == rr.score
        assert restored.source == "sparse"
