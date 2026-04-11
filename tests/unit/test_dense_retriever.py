"""Unit tests for DenseRetriever (D2)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from core.query_engine.dense_retriever import DenseRetriever
from core.types import RetrievalResult
from libs.vector_store.base_vector_store import QueryResult


def _make_mock_embedding(vectors: list[list[float]] | None = None):
    """Create a mock embedding client."""
    if vectors is None:
        vectors = [[0.1, 0.2, 0.3]]
    mock = MagicMock()
    mock.embed = MagicMock(return_value=vectors)
    return mock


def _make_mock_store(results: list[QueryResult] | None = None):
    """Create a mock vector store."""
    if results is None:
        results = [
            QueryResult(id="chunk_1", score=0.95, text="Hello world", metadata={"source_path": "doc.pdf"}),
            QueryResult(id="chunk_2", score=0.85, text="Another chunk", metadata={"source_path": "doc.pdf"}),
        ]
    mock = MagicMock()
    mock.query = MagicMock(return_value=results)
    return mock


class TestDenseRetriever:
    """Tests for DenseRetriever."""

    @pytest.fixture()
    def retriever(self) -> DenseRetriever:
        return DenseRetriever(
            settings=MagicMock(),
            embedding=_make_mock_embedding(),
            store=_make_mock_store(),
        )

    def test_basic_retrieve(self, retriever: DenseRetriever) -> None:
        results = retriever.retrieve("test query")
        assert len(results) == 2
        assert results[0].chunk_id == "chunk_1"
        assert results[0].score == 0.95
        assert results[0].text == "Hello world"
        assert results[0].source == "dense"

    def test_all_results_have_source_dense(self, retriever: DenseRetriever) -> None:
        results = retriever.retrieve("test")
        for r in results:
            assert r.source == "dense"

    def test_embedding_called_with_query(self) -> None:
        mock_emb = _make_mock_embedding()
        retriever = DenseRetriever(
            settings=MagicMock(),
            embedding=mock_emb,
            store=_make_mock_store(),
        )
        retriever.retrieve("my query text")
        mock_emb.embed.assert_called_once_with(["my query text"])

    def test_store_query_called_with_vector(self) -> None:
        vec = [[0.5, 0.6, 0.7]]
        mock_emb = _make_mock_embedding(vectors=vec)
        mock_store = _make_mock_store()
        retriever = DenseRetriever(
            settings=MagicMock(),
            embedding=mock_emb,
            store=mock_store,
        )
        retriever.retrieve("test")
        mock_store.query.assert_called_once()
        call_kwargs = mock_store.query.call_args
        assert call_kwargs[0][0] == [0.5, 0.6, 0.7]  # positional vector arg

    def test_top_k_passed_to_store(self) -> None:
        mock_store = _make_mock_store()
        retriever = DenseRetriever(
            settings=MagicMock(),
            embedding=_make_mock_embedding(),
            store=mock_store,
        )
        retriever.retrieve("test", top_k=5)
        call_kwargs = mock_store.query.call_args
        assert call_kwargs[1]["top_k"] == 5

    def test_filters_passed_to_store(self) -> None:
        mock_store = _make_mock_store()
        retriever = DenseRetriever(
            settings=MagicMock(),
            embedding=_make_mock_embedding(),
            store=mock_store,
        )
        filters = {"doc_type": "pdf"}
        retriever.retrieve("test", filters=filters)
        call_kwargs = mock_store.query.call_args
        assert call_kwargs[1]["filters"] == filters

    def test_collection_passed_to_store(self) -> None:
        mock_store = _make_mock_store()
        retriever = DenseRetriever(
            settings=MagicMock(),
            embedding=_make_mock_embedding(),
            store=mock_store,
        )
        retriever.retrieve("test", collection="research")
        call_kwargs = mock_store.query.call_args
        assert call_kwargs[1]["collection"] == "research"

    def test_empty_results(self) -> None:
        mock_store = _make_mock_store(results=[])
        retriever = DenseRetriever(
            settings=MagicMock(),
            embedding=_make_mock_embedding(),
            store=mock_store,
        )
        results = retriever.retrieve("obscure query")
        assert results == []

    def test_metadata_preserved(self) -> None:
        results_with_meta = [
            QueryResult(
                id="c1",
                score=0.9,
                text="text",
                metadata={"source_path": "a.pdf", "tags": ["ml"], "chunk_index": 0},
            ),
        ]
        retriever = DenseRetriever(
            settings=MagicMock(),
            embedding=_make_mock_embedding(),
            store=_make_mock_store(results=results_with_meta),
        )
        results = retriever.retrieve("test")
        assert results[0].metadata["source_path"] == "a.pdf"
        assert results[0].metadata["tags"] == ["ml"]
        assert results[0].metadata["chunk_index"] == 0

    def test_retrieval_result_serialization(self) -> None:
        rr = RetrievalResult(
            chunk_id="c1",
            score=0.95,
            text="hello",
            metadata={"key": "val"},
            source="dense",
        )
        d = rr.to_dict()
        assert d["chunk_id"] == "c1"
        assert d["score"] == 0.95
        assert d["source"] == "dense"

        restored = RetrievalResult.from_dict(d)
        assert restored.chunk_id == rr.chunk_id
        assert restored.score == rr.score
        assert restored.text == rr.text
