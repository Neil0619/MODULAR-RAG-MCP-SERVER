"""Unit tests for DoubaoVisionLLM (B10).

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
    kwargs: dict[str, Any] = dict(
        provider="doubao",
        model="doubao-seed-2-0-pro-260215",
        api_key="test-volc-key",
        max_image_size=2048,
    )
    kwargs.update(overrides)
    vision = VisionLLMSettings(**kwargs)
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
        vision_llm=vision,
    )


def _mock_chat_response(content: str = "A beautiful sunset") -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    resp.model = "doubao-seed-2-0-pro-260215"
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 50
    resp.usage.total_tokens = 150
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDoubaoVisionLLMCreation:
    @patch("libs.llm.doubao_vision_llm.OpenAI")
    def test_factory_creates_doubao_vision(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_vision_settings()
        vllm = LLMFactory.create_vision_llm(settings)
        assert isinstance(vllm, BaseVisionLLM)
        assert isinstance(vllm, BaseLLM)

    @patch("libs.llm.doubao_vision_llm.OpenAI")
    def test_client_configured_with_ark_base_url(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_vision_settings()
        LLMFactory.create_vision_llm(settings)

        mock_openai_cls.assert_called_once_with(
            api_key="test-volc-key",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )

    @patch("libs.llm.doubao_vision_llm.OpenAI")
    def test_custom_base_url_override(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_vision_settings(base_url="https://custom.ark.example.com/api/v3")
        LLMFactory.create_vision_llm(settings)

        mock_openai_cls.assert_called_once_with(
            api_key="test-volc-key",
            base_url="https://custom.ark.example.com/api/v3",
        )


class TestDoubaoVisionLLMChatWithImage:
    @patch("libs.llm.doubao_vision_llm.OpenAI")
    def test_chat_with_image_bytes(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response("A cat on a sofa")

        settings = _make_vision_settings()
        vllm = LLMFactory.create_vision_llm(settings)
        resp = vllm.chat_with_image("Describe this image", _FAKE_PNG)

        assert resp.content == "A cat on a sofa"
        assert resp.model == "doubao-seed-2-0-pro-260215"
        assert resp.usage["total_tokens"] == 150

    @patch("libs.llm.doubao_vision_llm.OpenAI")
    def test_chat_with_image_builds_multimodal_content(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
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

    @patch("libs.llm.doubao_vision_llm.OpenAI")
    def test_chat_with_image_file_path(self, mock_openai_cls: MagicMock, tmp_path: Any) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response("Chart data")

        img = tmp_path / "test.png"
        img.write_bytes(_FAKE_PNG)

        settings = _make_vision_settings()
        vllm = LLMFactory.create_vision_llm(settings)
        resp = vllm.chat_with_image("Read this chart", str(img))

        assert resp.content == "Chart data"

    @patch("libs.llm.doubao_vision_llm.OpenAI")
    def test_chat_with_image_file_not_found(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_vision_settings()
        vllm = LLMFactory.create_vision_llm(settings)

        with pytest.raises(FileNotFoundError, match="Image file not found"):
            vllm.chat_with_image("Describe", "/nonexistent/image.png")


class TestDoubaoVisionLLMChat:
    @patch("libs.llm.doubao_vision_llm.OpenAI")
    def test_text_chat_works(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_chat_response("Hello there")

        settings = _make_vision_settings()
        vllm = LLMFactory.create_vision_llm(settings)
        resp = vllm.chat_str("Say hello")

        assert resp == "Hello there"


class TestDoubaoVisionLLMErrorHandling:
    @patch("libs.llm.doubao_vision_llm.OpenAI")
    def test_api_error_raises_runtime_error(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("401 Unauthorized")

        settings = _make_vision_settings()
        vllm = LLMFactory.create_vision_llm(settings)

        with pytest.raises(RuntimeError, match="Doubao Vision image request failed.*401"):
            vllm.chat_with_image("test", _FAKE_PNG)

    @patch("libs.llm.doubao_vision_llm.OpenAI")
    def test_text_chat_api_error(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Timeout")

        settings = _make_vision_settings()
        vllm = LLMFactory.create_vision_llm(settings)

        with pytest.raises(RuntimeError, match="Doubao Vision chat request failed.*Timeout"):
            vllm.chat_str("hello")


class TestDoubaoVisionLLMResize:
    @patch("libs.llm.doubao_vision_llm.OpenAI")
    def test_resize_delegates_to_pillow(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_vision_settings(max_image_size=512)
        vllm = LLMFactory.create_vision_llm(settings)

        # Fake PNG is not valid, resize falls back to original
        result = vllm._resize_if_needed(_FAKE_PNG)
        assert isinstance(result, bytes)

    @patch("libs.llm.doubao_vision_llm.OpenAI")
    def test_resize_disabled_when_zero(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_vision_settings(max_image_size=0)
        vllm = LLMFactory.create_vision_llm(settings)
        result = vllm._resize_if_needed(b"any data")
        assert result == b"any data"


class TestDoubaoVisionFactoryRouting:
    def test_doubao_in_available_vision_providers(self) -> None:
        from libs.llm.llm_factory import _VISION_REGISTRY

        assert "doubao" in _VISION_REGISTRY
