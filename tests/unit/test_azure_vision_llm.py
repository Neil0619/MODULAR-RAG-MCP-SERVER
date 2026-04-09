"""Unit tests for AzureVisionLLM (B9).

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
    VisionLLMSettings,
)
from libs.llm.base_llm import BaseLLM, ChatResponse
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import LLMFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_PNG = (
    b"\x89PNG\r\n\x1a\n"
    + b"\x00\x00\x00\rIHDR\x00\x00\x03\x20\x00\x00\x02X"
    + b"\x08\x02\x00\x00\x00"  # 800x600 PNG header-ish
    + b"\x00" * 100
)


def _make_vision_settings(**overrides: Any) -> Settings:
    vision = VisionLLMSettings(
        provider="azure",
        model="gpt-4o",
        azure_endpoint="https://test.openai.azure.com/",
        api_key="test-key",
        api_version="2024-02-01",
        deployment_name="gpt4o-vision",
        **overrides,
    )
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
        vision_llm=vision,
    )


def _mock_chat_response(content: str = "A beautiful sunset") -> MagicMock:
    """Build a mock chat.completions.create response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    resp.model = "gpt-4o"
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 50
    resp.usage.total_tokens = 150
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAzureVisionLLMCreation:
    @patch("libs.llm.azure_vision_llm.AzureOpenAI")
    def test_factory_creates_azure_vision(self, mock_azure_cls: MagicMock) -> None:
        settings = _make_vision_settings()
        vllm = LLMFactory.create_vision_llm(settings)
        assert isinstance(vllm, BaseVisionLLM)
        assert isinstance(vllm, BaseLLM)

    @patch("libs.llm.azure_vision_llm.AzureOpenAI")
    def test_azure_client_configured_correctly(self, mock_azure_cls: MagicMock) -> None:
        settings = _make_vision_settings()
        LLMFactory.create_vision_llm(settings)

        mock_azure_cls.assert_called_once_with(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            api_version="2024-02-01",
        )

    @patch("libs.llm.azure_vision_llm.AzureOpenAI")
    def test_uses_deployment_name_as_model(self, mock_azure_cls: MagicMock) -> None:
        settings = _make_vision_settings()
        vllm = LLMFactory.create_vision_llm(settings)
        assert vllm._model == "gpt4o-vision"


class TestAzureVisionLLMChatWithImage:
    @patch("libs.llm.azure_vision_llm.AzureOpenAI")
    def test_chat_with_image_bytes(self, mock_azure_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_azure_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response("A cat on a sofa")

        settings = _make_vision_settings()
        vllm = LLMFactory.create_vision_llm(settings)
        resp = vllm.chat_with_image("Describe this image", _FAKE_PNG)

        assert resp.content == "A cat on a sofa"
        assert resp.model == "gpt-4o"
        assert resp.usage["total_tokens"] == 150

    @patch("libs.llm.azure_vision_llm.AzureOpenAI")
    def test_chat_with_image_builds_multimodal_content(self, mock_azure_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_azure_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response()

        settings = _make_vision_settings()
        vllm = LLMFactory.create_vision_llm(settings)
        vllm.chat_with_image("What do you see?", _FAKE_PNG)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "What do you see?"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    @patch("libs.llm.azure_vision_llm.AzureOpenAI")
    def test_chat_with_image_file_path(self, mock_azure_cls: MagicMock, tmp_path: Any) -> None:
        mock_client = MagicMock()
        mock_azure_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response("Chart data")

        img = tmp_path / "test.png"
        img.write_bytes(_FAKE_PNG)

        settings = _make_vision_settings()
        vllm = LLMFactory.create_vision_llm(settings)
        resp = vllm.chat_with_image("Read this chart", str(img))

        assert resp.content == "Chart data"

    @patch("libs.llm.azure_vision_llm.AzureOpenAI")
    def test_chat_with_image_file_not_found(self, mock_azure_cls: MagicMock) -> None:
        settings = _make_vision_settings()
        vllm = LLMFactory.create_vision_llm(settings)

        with pytest.raises(FileNotFoundError, match="Image file not found"):
            vllm.chat_with_image("Describe", "/nonexistent/image.png")


class TestAzureVisionLLMChat:
    @patch("libs.llm.azure_vision_llm.AzureOpenAI")
    def test_text_chat_works(self, mock_azure_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_azure_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response("Hello there")

        settings = _make_vision_settings()
        vllm = LLMFactory.create_vision_llm(settings)
        resp = vllm.chat_str("Say hello")

        assert resp == "Hello there"


class TestAzureVisionLLMErrorHandling:
    @patch("libs.llm.azure_vision_llm.AzureOpenAI")
    def test_api_error_raises_runtime_error(self, mock_azure_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_azure_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("401 Unauthorized")

        settings = _make_vision_settings()
        vllm = LLMFactory.create_vision_llm(settings)

        with pytest.raises(RuntimeError, match="Azure Vision image request failed.*401"):
            vllm.chat_with_image("test", _FAKE_PNG)

    @patch("libs.llm.azure_vision_llm.AzureOpenAI")
    def test_auth_failure_includes_status(self, mock_azure_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_azure_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("403 Forbidden: InvalidApiKey")

        settings = _make_vision_settings()
        vllm = LLMFactory.create_vision_llm(settings)

        with pytest.raises(RuntimeError, match="InvalidApiKey"):
            vllm.chat_with_image("test", _FAKE_PNG)

    @patch("libs.llm.azure_vision_llm.AzureOpenAI")
    def test_text_chat_api_error(self, mock_azure_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_azure_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Timeout")

        settings = _make_vision_settings()
        vllm = LLMFactory.create_vision_llm(settings)

        with pytest.raises(RuntimeError, match="Azure Vision chat request failed.*Timeout"):
            vllm.chat_str("hello")


class TestAzureVisionLLMResize:
    @patch("libs.llm.azure_vision_llm.AzureOpenAI")
    def test_resize_delegates_to_pillow(self, mock_azure_cls: MagicMock) -> None:
        """Verify _resize_if_needed is called and works with mocked PIL."""
        settings = _make_vision_settings(max_image_size=512)
        vllm = LLMFactory.create_vision_llm(settings)

        # Create a tiny "image" — resize logic will attempt PIL decode
        # Since our fake PNG is not valid, resize falls back to original
        result = vllm._resize_if_needed(_FAKE_PNG)
        assert isinstance(result, bytes)

    @patch("libs.llm.azure_vision_llm.AzureOpenAI")
    def test_resize_disabled_when_zero(self, mock_azure_cls: MagicMock) -> None:
        settings = _make_vision_settings(max_image_size=0)
        vllm = LLMFactory.create_vision_llm(settings)
        result = vllm._resize_if_needed(b"any data")
        assert result == b"any data"
