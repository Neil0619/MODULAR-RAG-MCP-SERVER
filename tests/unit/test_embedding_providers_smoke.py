"""Smoke tests for OpenAI and Azure embedding providers (B7.3)."""

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


def _make_embedding_settings(provider: str, **overrides: Any) -> Settings:
    """Build a Settings with the given embedding provider."""
    emb = EmbeddingSettings(provider=provider, model="text-embedding-3-small", **overrides)
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=emb,
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
    )


def _mock_embedding_response(count: int, dim: int = 1536) -> MagicMock:
    """Build a mock OpenAI embeddings response with *count* vectors."""
    data = []
    for i in range(count):
        item = MagicMock()
        item.embedding = [0.1 + i * 0.01] * dim
        data.append(item)
    mock_response = MagicMock()
    mock_response.data = data
    return mock_response


# ---------------------------------------------------------------------------
# OpenAI Embedding tests
# ---------------------------------------------------------------------------


class TestOpenAIEmbedding:
    @patch("libs.embedding.openai_embedding.OpenAI")
    def test_factory_creates_openai(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_embedding_settings("openai", api_key="sk-test")
        emb = EmbeddingFactory.create(settings)
        assert isinstance(emb, BaseEmbedding)
        mock_openai_cls.assert_called_once_with(api_key="sk-test")

    @patch("libs.embedding.openai_embedding.OpenAI")
    def test_embed_returns_correct_count(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_embedding_response(3)

        settings = _make_embedding_settings("openai", api_key="sk-test", dimensions=1536)
        emb = EmbeddingFactory.create(settings)
        vecs = emb.embed(["a", "b", "c"])

        assert len(vecs) == 3
        assert all(len(v) == 1536 for v in vecs)

    @patch("libs.embedding.openai_embedding.OpenAI")
    def test_dimensions_property(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_embedding_settings("openai", api_key="sk-test", dimensions=512)
        emb = EmbeddingFactory.create(settings)
        assert emb.dimensions == 512

    @patch("libs.embedding.openai_embedding.OpenAI")
    def test_empty_input_raises(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_embedding_settings("openai", api_key="sk-test")
        emb = EmbeddingFactory.create(settings)

        with pytest.raises(ValueError, match="texts must not be empty"):
            emb.embed([])

    @patch("libs.embedding.openai_embedding.OpenAI")
    def test_api_error_raises_runtime_error(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.side_effect = Exception("API error")

        settings = _make_embedding_settings("openai", api_key="sk-test")
        emb = EmbeddingFactory.create(settings)

        with pytest.raises(RuntimeError, match="OpenAI embedding request failed"):
            emb.embed(["test"])

    @patch("libs.embedding.openai_embedding.OpenAI")
    def test_batch_splitting(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        # First batch of 2, second batch of 1
        mock_client.embeddings.create.side_effect = [
            _mock_embedding_response(2),
            _mock_embedding_response(1),
        ]

        settings = _make_embedding_settings(
            "openai", api_key="sk-test", dimensions=64, batch_size=2
        )
        emb = EmbeddingFactory.create(settings)
        vecs = emb.embed(["a", "b", "c"])

        assert len(vecs) == 3
        assert mock_client.embeddings.create.call_count == 2

    @patch("libs.embedding.openai_embedding.OpenAI")
    def test_embed_single_convenience(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_embedding_response(1, dim=64)

        settings = _make_embedding_settings("openai", api_key="sk-test", dimensions=64)
        emb = EmbeddingFactory.create(settings)
        vec = emb.embed_single("hello")

        assert len(vec) == 64
        assert isinstance(vec[0], float)


# ---------------------------------------------------------------------------
# Azure Embedding tests
# ---------------------------------------------------------------------------


class TestAzureEmbedding:
    @patch("libs.embedding.azure_embedding.AzureOpenAI")
    def test_factory_creates_azure(self, mock_azure_cls: MagicMock) -> None:
        settings = _make_embedding_settings(
            "azure",
            azure_endpoint="https://test.openai.azure.com",
            api_key="azure-key",
            api_version="2024-02-01",
            deployment_name="embed-deploy",
        )
        emb = EmbeddingFactory.create(settings)
        assert isinstance(emb, BaseEmbedding)
        mock_azure_cls.assert_called_once_with(
            azure_endpoint="https://test.openai.azure.com",
            api_key="azure-key",
            api_version="2024-02-01",
        )

    @patch("libs.embedding.azure_embedding.AzureOpenAI")
    def test_embed_returns_correct_count(self, mock_azure_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_azure_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_embedding_response(2)

        settings = _make_embedding_settings(
            "azure",
            azure_endpoint="https://test.openai.azure.com",
            api_key="azure-key",
            api_version="2024-02-01",
            dimensions=1536,
        )
        emb = EmbeddingFactory.create(settings)
        vecs = emb.embed(["hello", "world"])

        assert len(vecs) == 2
        assert all(len(v) == 1536 for v in vecs)

    @patch("libs.embedding.azure_embedding.AzureOpenAI")
    def test_dimensions_property(self, mock_azure_cls: MagicMock) -> None:
        settings = _make_embedding_settings(
            "azure",
            azure_endpoint="https://test.openai.azure.com",
            api_key="azure-key",
            api_version="2024-02-01",
            dimensions=1024,
        )
        emb = EmbeddingFactory.create(settings)
        assert emb.dimensions == 1024

    @patch("libs.embedding.azure_embedding.AzureOpenAI")
    def test_empty_input_raises(self, mock_azure_cls: MagicMock) -> None:
        settings = _make_embedding_settings(
            "azure",
            azure_endpoint="https://test.openai.azure.com",
            api_key="azure-key",
            api_version="2024-02-01",
        )
        emb = EmbeddingFactory.create(settings)

        with pytest.raises(ValueError, match="texts must not be empty"):
            emb.embed([])

    @patch("libs.embedding.azure_embedding.AzureOpenAI")
    def test_api_error_raises_runtime_error(self, mock_azure_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_azure_cls.return_value = mock_client
        mock_client.embeddings.create.side_effect = Exception("Azure error")

        settings = _make_embedding_settings(
            "azure",
            azure_endpoint="https://test.openai.azure.com",
            api_key="azure-key",
            api_version="2024-02-01",
        )
        emb = EmbeddingFactory.create(settings)

        with pytest.raises(RuntimeError, match="Azure embedding request failed"):
            emb.embed(["test"])

    @patch("libs.embedding.azure_embedding.AzureOpenAI")
    def test_uses_deployment_name_as_model(self, mock_azure_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_azure_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_embedding_response(1)

        settings = _make_embedding_settings(
            "azure",
            azure_endpoint="https://test.openai.azure.com",
            api_key="azure-key",
            api_version="2024-02-01",
            deployment_name="my-embed-deploy",
        )
        emb = EmbeddingFactory.create(settings)
        emb.embed(["test"])

        call_kwargs = mock_client.embeddings.create.call_args
        assert call_kwargs.kwargs["model"] == "my-embed-deploy"
