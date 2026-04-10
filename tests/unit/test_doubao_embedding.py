"""Unit tests for DoubaoEmbedding (B10).

All tests mock the openai SDK — no real API calls.
"""

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

_VEC_DIM = 1024


def _make_embed_settings(**overrides: Any) -> Settings:
    kwargs: dict[str, Any] = dict(
        provider="doubao",
        model="doubao-embedding-vision-251215",
        api_key="test-volc-key",
        dimensions=_VEC_DIM,
        batch_size=100,
    )
    kwargs.update(overrides)
    emb = EmbeddingSettings(**kwargs)
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=emb,
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
    )


def _mock_embed_response(count: int, dim: int = _VEC_DIM) -> MagicMock:
    data = []
    for _ in range(count):
        item = MagicMock()
        item.embedding = [0.1] * dim
        data.append(item)
    resp = MagicMock()
    resp.data = data
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDoubaoEmbeddingCreation:
    @patch("libs.embedding.doubao_embedding.OpenAI")
    def test_factory_creates_doubao(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_embed_settings()
        emb = EmbeddingFactory.create(settings)
        assert isinstance(emb, BaseEmbedding)

    @patch("libs.embedding.doubao_embedding.OpenAI")
    def test_client_configured_with_ark_base_url(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_embed_settings()
        EmbeddingFactory.create(settings)

        mock_openai_cls.assert_called_once_with(
            api_key="test-volc-key",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )

    @patch("libs.embedding.doubao_embedding.OpenAI")
    def test_custom_base_url_override(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_embed_settings(base_url="https://custom.ark.example.com/api/v3")
        EmbeddingFactory.create(settings)

        mock_openai_cls.assert_called_once_with(
            api_key="test-volc-key",
            base_url="https://custom.ark.example.com/api/v3",
        )

    @patch("libs.embedding.doubao_embedding.OpenAI")
    def test_dimensions_property(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_embed_settings()
        emb = EmbeddingFactory.create(settings)
        assert emb.dimensions == _VEC_DIM


class TestDoubaoEmbeddingEmbed:
    @patch("libs.embedding.doubao_embedding.OpenAI")
    def test_embed_returns_correct_count(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_embed_response(3)

        settings = _make_embed_settings()
        emb = EmbeddingFactory.create(settings)
        result = emb.embed(["hello", "world", "test"])

        assert len(result) == 3
        assert len(result[0]) == _VEC_DIM

    @patch("libs.embedding.doubao_embedding.OpenAI")
    def test_embed_passes_model_and_dimensions(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_embed_response(1)

        settings = _make_embed_settings()
        emb = EmbeddingFactory.create(settings)
        emb.embed(["test"])

        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == "doubao-embedding-vision-251215"
        assert call_kwargs["dimensions"] == _VEC_DIM

    @patch("libs.embedding.doubao_embedding.OpenAI")
    def test_embed_batch_splitting(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        # First batch of 2, second batch of 1
        mock_client.embeddings.create.side_effect = [
            _mock_embed_response(2),
            _mock_embed_response(1),
        ]

        settings = _make_embed_settings(batch_size=2)
        emb = EmbeddingFactory.create(settings)
        result = emb.embed(["a", "b", "c"])

        assert len(result) == 3
        assert mock_client.embeddings.create.call_count == 2

    @patch("libs.embedding.doubao_embedding.OpenAI")
    def test_embed_empty_input_raises(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_embed_settings()
        emb = EmbeddingFactory.create(settings)

        with pytest.raises(ValueError, match="texts must not be empty"):
            emb.embed([])


class TestDoubaoEmbeddingErrorHandling:
    @patch("libs.embedding.doubao_embedding.OpenAI")
    def test_api_error_raises_runtime_error(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.side_effect = Exception("401 Unauthorized")

        settings = _make_embed_settings()
        emb = EmbeddingFactory.create(settings)

        with pytest.raises(RuntimeError, match="Doubao embedding request failed.*401"):
            emb.embed(["test"])


class TestDoubaoEmbeddingFactoryRouting:
    def test_doubao_in_available_providers(self) -> None:
        from libs.embedding.embedding_factory import _PROVIDER_REGISTRY

        assert "doubao" in _PROVIDER_REGISTRY
