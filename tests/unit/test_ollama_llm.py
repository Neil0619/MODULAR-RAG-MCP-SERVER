"""Smoke tests for the Ollama LLM provider (B7.2)."""

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
from libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from libs.llm.llm_factory import LLMFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: Any) -> Settings:
    """Build a Settings configured for Ollama."""
    llm = LLMSettings(provider="ollama", model="llama3", **overrides)
    return Settings(
        llm=llm,
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
    )


def _mock_chat_response(content: str = "Ollama reply") -> MagicMock:
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 8
    mock_usage.completion_tokens = 3
    mock_usage.total_tokens = 11
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model = "llama3"
    mock_response.usage = mock_usage
    return mock_response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOllamaLLM:
    @patch("libs.llm.ollama_llm.OpenAI")
    def test_factory_creates_ollama(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_settings()
        llm = LLMFactory.create(settings)
        assert isinstance(llm, BaseLLM)
        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    @patch("libs.llm.ollama_llm.OpenAI")
    def test_chat_returns_chat_response(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response()

        settings = _make_settings()
        llm = LLMFactory.create(settings)
        resp = llm.chat([ChatMessage(role="user", content="hello")])

        assert isinstance(resp, ChatResponse)
        assert resp.content == "Ollama reply"
        assert resp.usage["total_tokens"] == 11

    @patch("libs.llm.ollama_llm.OpenAI")
    def test_custom_base_url(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_settings(base_url="http://my-server:11434/v1")
        LLMFactory.create(settings)

        mock_openai_cls.assert_called_once_with(
            base_url="http://my-server:11434/v1",
            api_key="ollama",
        )

    @patch("libs.llm.ollama_llm.OpenAI")
    def test_chat_passes_model(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response()

        settings = _make_settings()
        llm = LLMFactory.create(settings)
        llm.chat([ChatMessage(role="user", content="hi")])

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "llama3"

    @patch("libs.llm.ollama_llm.OpenAI")
    def test_temperature_override(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response()

        settings = _make_settings(temperature=0.7)
        llm = LLMFactory.create(settings)
        llm.chat(
            [ChatMessage(role="user", content="hi")],
            temperature=0.2,
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.2

    @patch("libs.llm.ollama_llm.OpenAI")
    def test_api_error_raises_runtime_error(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Connection refused")

        settings = _make_settings()
        llm = LLMFactory.create(settings)

        with pytest.raises(RuntimeError, match="Ollama chat request failed"):
            llm.chat([ChatMessage(role="user", content="hello")])

    @patch("libs.llm.ollama_llm.OpenAI")
    def test_empty_content_in_response(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = _mock_chat_response("")
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response

        settings = _make_settings()
        llm = LLMFactory.create(settings)
        resp = llm.chat([ChatMessage(role="user", content="hello")])

        assert isinstance(resp, ChatResponse)
        assert resp.content == ""
