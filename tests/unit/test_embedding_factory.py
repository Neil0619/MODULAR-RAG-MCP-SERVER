"""Unit tests for Embedding base class and factory (B2)."""

from typing import Any

import pytest

from core.settings import (
    EmbeddingSettings,
    LLMSettings,
    RetrievalSettings,
    Settings,
    VectorStoreSettings,
)
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


# ---------------------------------------------------------------------------
# Fake embedding for testing
# ---------------------------------------------------------------------------


class FakeEmbedding(BaseEmbedding):
    """Deterministic fake: returns a hash-based vector for each text."""

    _DIM = 8

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings

    @property
    def dimensions(self) -> int:
        return self._DIM

    def embed(
        self,
        texts: list[str],
        *,
        trace: Any | None = None,
    ) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must not be empty")
        return [
            [float(hash(t) % 100) / 100.0] * self._DIM
            for t in texts
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(provider: str) -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=EmbeddingSettings(provider=provider, model="test-emb"),
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
    )


# ---------------------------------------------------------------------------
# BaseEmbedding tests
# ---------------------------------------------------------------------------


class TestBaseEmbedding:
    def test_cannot_instantiate_base(self) -> None:
        with pytest.raises(TypeError):
            BaseEmbedding()  # type: ignore[abstract]


class TestFakeEmbedding:
    def test_dimensions(self) -> None:
        emb = FakeEmbedding()
        assert emb.dimensions == 8

    def test_embed_returns_correct_count(self) -> None:
        emb = FakeEmbedding()
        vecs = emb.embed(["hello", "world"])
        assert len(vecs) == 2
        assert all(len(v) == emb.dimensions for v in vecs)

    def test_embed_stable_for_same_input(self) -> None:
        emb = FakeEmbedding()
        v1 = emb.embed(["test"])
        v2 = emb.embed(["test"])
        assert v1 == v2

    def test_embed_different_for_different_input(self) -> None:
        emb = FakeEmbedding()
        v1 = emb.embed(["aaa"])
        v2 = emb.embed(["bbb"])
        assert v1 != v2

    def test_embed_empty_raises(self) -> None:
        emb = FakeEmbedding()
        with pytest.raises(ValueError, match="empty"):
            emb.embed([])

    def test_embed_single_convenience(self) -> None:
        emb = FakeEmbedding()
        vec = emb.embed_single("hello")
        assert len(vec) == emb.dimensions
        assert isinstance(vec[0], float)


# ---------------------------------------------------------------------------
# EmbeddingFactory tests
# ---------------------------------------------------------------------------


class TestEmbeddingFactory:
    def test_unknown_provider_raises(self) -> None:
        settings = _make_settings("nonexistent")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            EmbeddingFactory.create(settings)

    def test_register_and_create(self) -> None:
        EmbeddingFactory.register_provider(
            "fake", f"{FakeEmbedding.__module__}.{FakeEmbedding.__qualname__}"
        )
        emb = EmbeddingFactory.create(_make_settings("fake"))
        assert isinstance(emb, FakeEmbedding)
        vecs = emb.embed(["test"])
        assert len(vecs) == 1

    def test_case_insensitive(self) -> None:
        EmbeddingFactory.register_provider(
            "fake", f"{FakeEmbedding.__module__}.{FakeEmbedding.__qualname__}"
        )
        emb = EmbeddingFactory.create(_make_settings("FAKE"))
        assert isinstance(emb, FakeEmbedding)

    def test_error_lists_available(self) -> None:
        with pytest.raises(ValueError, match="Available:") as exc_info:
            EmbeddingFactory.create(_make_settings("bogus"))
        for name in ("openai", "azure", "ollama"):
            assert name in str(exc_info.value)
