"""Unit tests for DoubaoLLM (B10).

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
from libs.llm.base_llm import BaseLLM, ChatResponse
from libs.llm.llm_factory import LLMFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_settings(**overrides: Any) -> Settings:
    llm = LLMSettings(
        provider="doubao",
        model="doubao-seed-2-0-pro-260215",
        api_key="test-volc-key",
        temperature=0.0,
        max_tokens=4096,
        **overrides,
    )
    return Settings(
        llm=llm,
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
    )


def _mock_chat_response(content: str = "Hello from Doubao") -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    resp.model = "doubao-seed-2-0-pro-260215"
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 20
    resp.usage.total_tokens = 30
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDoubaoLLMCreation:
    @patch("libs.llm.doubao_llm.OpenAI")
    def test_factory_creates_doubao(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_llm_settings()
        llm = LLMFactory.create(settings)
        assert isinstance(llm, BaseLLM)

    @patch("libs.llm.doubao_llm.OpenAI")
    def test_client_configured_with_ark_base_url(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_llm_settings()
        LLMFactory.create(settings)

        mock_openai_cls.assert_called_once_with(
            api_key="test-volc-key",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )

    @patch("libs.llm.doubao_llm.OpenAI")
    def test_custom_base_url_override(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_llm_settings(base_url="https://custom.ark.example.com/api/v3")
        LLMFactory.create(settings)

        mock_openai_cls.assert_called_once_with(
            api_key="test-volc-key",
            base_url="https://custom.ark.example.com/api/v3",
        )


class TestDoubaoLLMChat:
    @patch("libs.llm.doubao_llm.OpenAI")
    def test_chat_returns_response(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response()

        settings = _make_llm_settings()
        llm = LLMFactory.create(settings)
        resp = llm.chat_str("Hello!")

        assert resp == "Hello from Doubao"

    @patch("libs.llm.doubao_llm.OpenAI")
    def test_chat_with_messages(self, mock_openai_cls: MagicMock) -> None:
        from libs.llm.base_llm import ChatMessage

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response("Reply")

        settings = _make_llm_settings()
        llm = LLMFactory.create(settings)
        response = llm.chat([ChatMessage(role="user", content="Test query")])

        assert response.content == "Reply"
        assert response.model == "doubao-seed-2-0-pro-260215"
        assert response.usage["total_tokens"] == 30

    @patch("libs.llm.doubao_llm.OpenAI")
    def test_chat_passes_model_and_params(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response()

        settings = _make_llm_settings()
        llm = LLMFactory.create(settings)
        llm.chat_str("Hi", temperature=0.5, max_tokens=100)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "doubao-seed-2-0-pro-260215"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100


class TestDoubaoLLMErrorHandling:
    @patch("libs.llm.doubao_llm.OpenAI")
    def test_api_error_raises_runtime_error(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("401 Unauthorized")

        settings = _make_llm_settings()
        llm = LLMFactory.create(settings)

        with pytest.raises(RuntimeError, match="Doubao chat request failed.*401"):
            llm.chat_str("test")

    @patch("libs.llm.doubao_llm.OpenAI")
    def test_connection_error(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Connection refused")

        settings = _make_llm_settings()
        llm = LLMFactory.create(settings)

        with pytest.raises(RuntimeError, match="Doubao chat request failed.*Connection refused"):
            llm.chat_str("test")


class TestDoubaoLLMFactoryRouting:
    def test_unknown_provider_still_errors(self) -> None:
        settings = _make_llm_settings()
        settings.llm.provider = "unknown_provider"
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMFactory.create(settings)

    @patch("libs.llm.doubao_llm.OpenAI")
    def test_doubao_in_available_providers(self, mock_openai_cls: MagicMock) -> None:
        """Verify doubao appears in the factory registry."""
        from libs.llm.llm_factory import _PROVIDER_REGISTRY

        assert "doubao" in _PROVIDER_REGISTRY
