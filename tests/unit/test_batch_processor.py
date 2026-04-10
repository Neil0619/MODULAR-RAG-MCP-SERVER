"""Unit tests for BatchProcessor (C10)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.settings import (
    EmbeddingSettings,
    LLMSettings,
    RetrievalSettings,
    Settings,
    VectorStoreSettings,
)
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder


# ---------------------------------------------------------------------------
# Fake encoders
# ---------------------------------------------------------------------------


class FakeDenseEncoder:
    dimensions = 4

    def encode(self, chunks: list[Chunk], **kw: Any) -> list[list[float]]:
        return [[float(i) * 0.1] * self.dimensions for i in range(len(chunks))]


class FakeSparseEncoder:
    def encode(self, chunks: list[Chunk], **kw: Any) -> list[dict[str, float]]:
        return [{"word": 0.5} for _ in chunks]


def _make_processor(batch_size: int = 32) -> BatchProcessor:
    settings = Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
    )
    return BatchProcessor(
        settings,
        batch_size=batch_size,
        dense_encoder=FakeDenseEncoder(),  # type: ignore[arg-type]
        sparse_encoder=FakeSparseEncoder(),  # type: ignore[arg-type]
    )


def _make_chunks(n: int) -> list[Chunk]:
    return [Chunk(id=f"c{i}", text=f"text chunk {i}", metadata={"source_path": "/test.pdf"}) for i in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBatchProcessor:
    def test_five_chunks_batch_size_two_gives_three_batches(self) -> None:
        bp = _make_processor(batch_size=2)
        chunks = _make_chunks(5)
        batches = bp._make_batches(chunks)
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_process_returns_correct_count(self) -> None:
        bp = _make_processor(batch_size=2)
        chunks = _make_chunks(5)
        records = bp.process(chunks)
        assert len(records) == 5

    def test_output_are_chunk_records(self) -> None:
        bp = _make_processor(batch_size=2)
        chunks = _make_chunks(3)
        records = bp.process(chunks)
        for r in records:
            assert isinstance(r, ChunkRecord)

    def test_records_have_dense_vectors(self) -> None:
        bp = _make_processor(batch_size=2)
        chunks = _make_chunks(3)
        records = bp.process(chunks)
        for r in records:
            assert r.dense_vector is not None
            assert len(r.dense_vector) == 4

    def test_records_have_sparse_vectors(self) -> None:
        bp = _make_processor(batch_size=2)
        chunks = _make_chunks(3)
        records = bp.process(chunks)
        for r in records:
            assert r.sparse_vector is not None
            assert "word" in r.sparse_vector

    def test_order_preserved(self) -> None:
        bp = _make_processor(batch_size=2)
        chunks = _make_chunks(5)
        records = bp.process(chunks)
        ids = [r.id for r in records]
        assert ids == ["c0", "c1", "c2", "c3", "c4"]

    def test_empty_input_returns_empty(self) -> None:
        bp = _make_processor(batch_size=2)
        assert bp.process([]) == []

    def test_single_chunk(self) -> None:
        bp = _make_processor(batch_size=2)
        chunks = _make_chunks(1)
        records = bp.process(chunks)
        assert len(records) == 1

    def test_exact_batch_size(self) -> None:
        bp = _make_processor(batch_size=3)
        chunks = _make_chunks(6)
        batches = bp._make_batches(chunks)
        assert len(batches) == 2
        assert all(len(b) == 3 for b in batches)

    def test_batch_size_property(self) -> None:
        bp = _make_processor(batch_size=7)
        assert bp.batch_size == 7

    def test_trace_records_batches(self) -> None:
        bp = _make_processor(batch_size=2)
        chunks = _make_chunks(5)
        trace = TraceContext()
        bp.process(chunks, trace=trace)
        assert "batch_0" in trace.stages
        assert "batch_1" in trace.stages
        assert "batch_2" in trace.stages
        assert trace.stages["batch_0"]["chunk_count"] == 2
        assert trace.stages["batch_2"]["chunk_count"] == 1

    def test_metadata_preserved(self) -> None:
        bp = _make_processor(batch_size=2)
        chunk = Chunk(id="c1", text="hello", metadata={"source_path": "/test.pdf", "doc_type": "pdf"})
        records = bp.process([chunk])
        assert records[0].metadata["source_path"] == "/test.pdf"
        assert records[0].metadata["doc_type"] == "pdf"
