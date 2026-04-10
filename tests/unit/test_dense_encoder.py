"""Unit tests for DenseEncoder (C8)."""

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
from core.types import Chunk
from ingestion.embedding.dense_encoder import DenseEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Controllable fake embedder for testing."""

    def __init__(self, settings: Any = None) -> None:
        self._dim = 8

    @property
    def dimensions(self) -> int:
        return self._dim

    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must not be empty")
        return [[float(i) * 0.1] * self._dim for i in range(len(texts))]

    def embed_single(self, text: str, **kwargs: Any) -> list[float]:
        return [0.1] * self._dim


def _make_encoder() -> DenseEncoder:
    with patch(
        "ingestion.embedding.dense_encoder.EmbeddingFactory"
    ) as mock_factory:
        mock_factory.create.return_value = FakeEmbedder()
        settings = Settings(
            llm=LLMSettings(provider="openai", model="gpt-4o"),
            embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
            vector_store=VectorStoreSettings(backend="chroma"),
            retrieval=RetrievalSettings(sparse_backend="bm25"),
        )
        return DenseEncoder(settings)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDenseEncoder:
    def test_encode_returns_correct_count(self) -> None:
        encoder = _make_encoder()
        chunks = [
            Chunk(id="c1", text="hello"),
            Chunk(id="c2", text="world"),
            Chunk(id="c3", text="foo"),
        ]
        vectors = encoder.encode(chunks)
        assert len(vectors) == 3

    def test_encode_dimensions_consistent(self) -> None:
        encoder = _make_encoder()
        chunks = [Chunk(id="c1", text="hello"), Chunk(id="c2", text="world")]
        vectors = encoder.encode(chunks)
        assert all(len(v) == encoder.dimensions for v in vectors)

    def test_encode_single_chunk(self) -> None:
        encoder = _make_encoder()
        chunks = [Chunk(id="c1", text="single")]
        vectors = encoder.encode(chunks)
        assert len(vectors) == 1
        assert len(vectors[0]) == encoder.dimensions

    def test_encode_empty_raises(self) -> None:
        encoder = _make_encoder()
        with pytest.raises(ValueError, match="chunks must not be empty"):
            encoder.encode([])

    def test_dimensions_property(self) -> None:
        encoder = _make_encoder()
        assert encoder.dimensions == 8

    def test_encode_preserves_order(self) -> None:
        encoder = _make_encoder()
        chunks = [
            Chunk(id="c1", text="first"),
            Chunk(id="c2", text="second"),
        ]
        vectors = encoder.encode(chunks)
        # First chunk gets index 0, second gets index 1
        assert vectors[0][0] == 0.0
        assert vectors[1][0] == pytest.approx(0.1)

    def test_encode_texts(self) -> None:
        encoder = _make_encoder()
        vectors = encoder.encode_texts(["hello", "world"])
        assert len(vectors) == 2

    def test_encode_texts_empty_raises(self) -> None:
        encoder = _make_encoder()
        with pytest.raises(ValueError, match="texts must not be empty"):
            encoder.encode_texts([])

    def test_large_batch(self) -> None:
        encoder = _make_encoder()
        chunks = [Chunk(id=f"c{i}", text=f"text {i}") for i in range(50)]
        vectors = encoder.encode(chunks)
        assert len(vectors) == 50
