"""Smoke tests for the Ollama embedding provider (B7.4)."""

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
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: Any) -> Settings:
    """Build a Settings configured for Ollama embeddings."""
    emb = EmbeddingSettings(provider="ollama", model="nomic-embed-text", **overrides)
    return Settings(
        llm=LLMSettings(provider="ollama", model="llama3"),
        embedding=emb,
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
    )


def _mock_embedding_response(count: int, dim: int = 768) -> MagicMock:
    """Build a mock embeddings response with *count* vectors."""
    data = []
    for i in range(count):
        item = MagicMock()
        item.embedding = [0.05 + i * 0.01] * dim
        data.append(item)
    mock_response = MagicMock()
    mock_response.data = data
    return mock_response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOllamaEmbedding:
    @patch("libs.embedding.ollama_embedding.OpenAI")
    def test_factory_creates_ollama(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_settings()
        emb = EmbeddingFactory.create(settings)
        assert isinstance(emb, BaseEmbedding)
        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    @patch("libs.embedding.ollama_embedding.OpenAI")
    def test_embed_returns_correct_count(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_embedding_response(3, dim=768)

        settings = _make_settings(dimensions=768)
        emb = EmbeddingFactory.create(settings)
        vecs = emb.embed(["a", "b", "c"])

        assert len(vecs) == 3
        assert all(len(v) == 768 for v in vecs)

    @patch("libs.embedding.ollama_embedding.OpenAI")
    def test_dimensions_property(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_settings(dimensions=512)
        emb = EmbeddingFactory.create(settings)
        assert emb.dimensions == 512

    @patch("libs.embedding.ollama_embedding.OpenAI")
    def test_empty_input_raises(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_settings()
        emb = EmbeddingFactory.create(settings)

        with pytest.raises(ValueError, match="texts must not be empty"):
            emb.embed([])

    @patch("libs.embedding.ollama_embedding.OpenAI")
    def test_api_error_raises_runtime_error(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.side_effect = Exception("Connection refused")

        settings = _make_settings()
        emb = EmbeddingFactory.create(settings)

        with pytest.raises(RuntimeError, match="Ollama embedding request failed"):
            emb.embed(["test"])

    @patch("libs.embedding.ollama_embedding.OpenAI")
    def test_custom_base_url(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_settings(base_url="http://my-ollama:11434/v1")
        EmbeddingFactory.create(settings)

        mock_openai_cls.assert_called_once_with(
            base_url="http://my-ollama:11434/v1",
            api_key="ollama",
        )

    @patch("libs.embedding.ollama_embedding.OpenAI")
    def test_embed_single_convenience(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_embedding_response(1, dim=256)

        settings = _make_settings(dimensions=256)
        emb = EmbeddingFactory.create(settings)
        vec = emb.embed_single("hello")

        assert len(vec) == 256
        assert isinstance(vec[0], float)

    @patch("libs.embedding.ollama_embedding.OpenAI")
    def test_batch_splitting(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.side_effect = [
            _mock_embedding_response(2, dim=64),
            _mock_embedding_response(1, dim=64),
        ]

        settings = _make_settings(dimensions=64, batch_size=2)
        emb = EmbeddingFactory.create(settings)
        vecs = emb.embed(["a", "b", "c"])

        assert len(vecs) == 3
        assert mock_client.embeddings.create.call_count == 2

    @patch("libs.embedding.ollama_embedding.OpenAI")
    def test_model_passed_correctly(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_embedding_response(1)

        settings = _make_settings()
        emb = EmbeddingFactory.create(settings)
        emb.embed(["test"])

        call_kwargs = mock_client.embeddings.create.call_args
        assert call_kwargs.kwargs["model"] == "nomic-embed-text"
