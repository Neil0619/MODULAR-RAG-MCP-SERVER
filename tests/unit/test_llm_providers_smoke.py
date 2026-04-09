"""Smoke tests for OpenAI, Azure, and DeepSeek LLM providers (B7.1)."""

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


def _make_settings(provider: str, **overrides: Any) -> Settings:
    """Build a minimal Settings with the given LLM provider."""
    llm = LLMSettings(provider=provider, model="test-model", **overrides)
    return Settings(
        llm=llm,
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
    )


def _mock_chat_response(content: str = "Hello from LLM") -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model = "test-model"
    mock_response.usage = mock_usage
    return mock_response


# ---------------------------------------------------------------------------
# OpenAI LLM tests
# ---------------------------------------------------------------------------


class TestOpenAILLM:
    @patch("libs.llm.openai_llm.OpenAI")
    def test_factory_creates_openai(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_settings("openai", api_key="sk-test")
        llm = LLMFactory.create(settings)
        assert isinstance(llm, BaseLLM)
        mock_openai_cls.assert_called_once()

    @patch("libs.llm.openai_llm.OpenAI")
    def test_chat_returns_chat_response(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response("OpenAI reply")

        settings = _make_settings("openai", api_key="sk-test")
        llm = LLMFactory.create(settings)
        resp = llm.chat([ChatMessage(role="user", content="hello")])

        assert isinstance(resp, ChatResponse)
        assert resp.content == "OpenAI reply"
        assert resp.model == "test-model"
        assert resp.usage["total_tokens"] == 15

    @patch("libs.llm.openai_llm.OpenAI")
    def test_chat_passes_model_and_temperature(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response()

        settings = _make_settings("openai", api_key="sk-test", temperature=0.5)
        llm = LLMFactory.create(settings)
        llm.chat([ChatMessage(role="user", content="hi")])

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "test-model"
        assert call_kwargs.kwargs["temperature"] == 0.5

    @patch("libs.llm.openai_llm.OpenAI")
    def test_chat_override_temperature_and_max_tokens(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response()

        settings = _make_settings("openai", api_key="sk-test")
        llm = LLMFactory.create(settings)
        llm.chat(
            [ChatMessage(role="user", content="hi")],
            temperature=0.9,
            max_tokens=100,
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.9
        assert call_kwargs.kwargs["max_tokens"] == 100

    @patch("libs.llm.openai_llm.OpenAI")
    def test_api_error_raises_runtime_error(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API timeout")

        settings = _make_settings("openai", api_key="sk-test")
        llm = LLMFactory.create(settings)

        with pytest.raises(RuntimeError, match="OpenAI chat request failed"):
            llm.chat([ChatMessage(role="user", content="hello")])

    @patch("libs.llm.openai_llm.OpenAI")
    def test_config_passed_correctly(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_settings(
            "openai",
            api_key="sk-test-123",
            base_url="https://custom.api.com",
        )
        LLMFactory.create(settings)

        mock_openai_cls.assert_called_once_with(
            api_key="sk-test-123",
            base_url="https://custom.api.com",
        )


# ---------------------------------------------------------------------------
# Azure LLM tests
# ---------------------------------------------------------------------------


class TestAzureLLM:
    @patch("libs.llm.azure_llm.AzureOpenAI")
    def test_factory_creates_azure(self, mock_azure_cls: MagicMock) -> None:
        settings = _make_settings(
            "azure",
            azure_endpoint="https://test.openai.azure.com",
            api_key="azure-key",
            api_version="2024-02-01",
            deployment_name="gpt-4-deploy",
        )
        llm = LLMFactory.create(settings)
        assert isinstance(llm, BaseLLM)
        mock_azure_cls.assert_called_once_with(
            azure_endpoint="https://test.openai.azure.com",
            api_key="azure-key",
            api_version="2024-02-01",
        )

    @patch("libs.llm.azure_llm.AzureOpenAI")
    def test_chat_returns_chat_response(self, mock_azure_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_azure_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response("Azure reply")

        settings = _make_settings(
            "azure",
            azure_endpoint="https://test.openai.azure.com",
            api_key="azure-key",
            api_version="2024-02-01",
            deployment_name="gpt-4-deploy",
        )
        llm = LLMFactory.create(settings)
        resp = llm.chat([ChatMessage(role="user", content="hello")])

        assert isinstance(resp, ChatResponse)
        assert resp.content == "Azure reply"

    @patch("libs.llm.azure_llm.AzureOpenAI")
    def test_uses_deployment_name_as_model(self, mock_azure_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_azure_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response()

        settings = _make_settings(
            "azure",
            azure_endpoint="https://test.openai.azure.com",
            api_key="azure-key",
            api_version="2024-02-01",
            deployment_name="my-deployment",
        )
        llm = LLMFactory.create(settings)
        llm.chat([ChatMessage(role="user", content="hi")])

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "my-deployment"

    @patch("libs.llm.azure_llm.AzureOpenAI")
    def test_api_error_raises_runtime_error(self, mock_azure_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_azure_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Azure error")

        settings = _make_settings(
            "azure",
            azure_endpoint="https://test.openai.azure.com",
            api_key="azure-key",
            api_version="2024-02-01",
        )
        llm = LLMFactory.create(settings)

        with pytest.raises(RuntimeError, match="Azure OpenAI chat request failed"):
            llm.chat([ChatMessage(role="user", content="hello")])

    @patch("libs.llm.azure_llm.AzureOpenAI")
    def test_config_passed_correctly(self, mock_azure_cls: MagicMock) -> None:
        settings = _make_settings(
            "azure",
            azure_endpoint="https://test.openai.azure.com",
            api_key="azure-key",
            api_version="2024-06-01",
        )
        LLMFactory.create(settings)

        mock_azure_cls.assert_called_once_with(
            azure_endpoint="https://test.openai.azure.com",
            api_key="azure-key",
            api_version="2024-06-01",
        )


# ---------------------------------------------------------------------------
# DeepSeek LLM tests
# ---------------------------------------------------------------------------


class TestDeepSeekLLM:
    @patch("libs.llm.deepseek_llm.OpenAI")
    def test_factory_creates_deepseek(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_settings("deepseek", api_key="ds-key")
        llm = LLMFactory.create(settings)
        assert isinstance(llm, BaseLLM)
        mock_openai_cls.assert_called_once_with(
            api_key="ds-key",
            base_url="https://api.deepseek.com",
        )

    @patch("libs.llm.deepseek_llm.OpenAI")
    def test_chat_returns_chat_response(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response("DeepSeek reply")

        settings = _make_settings("deepseek", api_key="ds-key")
        llm = LLMFactory.create(settings)
        resp = llm.chat([ChatMessage(role="user", content="hello")])

        assert isinstance(resp, ChatResponse)
        assert resp.content == "DeepSeek reply"

    @patch("libs.llm.deepseek_llm.OpenAI")
    def test_api_error_raises_runtime_error(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("DS error")

        settings = _make_settings("deepseek", api_key="ds-key")
        llm = LLMFactory.create(settings)

        with pytest.raises(RuntimeError, match="DeepSeek chat request failed"):
            llm.chat([ChatMessage(role="user", content="hello")])

    @patch("libs.llm.deepseek_llm.OpenAI")
    def test_config_uses_deepseek_base_url(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_settings("deepseek", api_key="ds-key-abc")
        LLMFactory.create(settings)

        mock_openai_cls.assert_called_once_with(
            api_key="ds-key-abc",
            base_url="https://api.deepseek.com",
        )
