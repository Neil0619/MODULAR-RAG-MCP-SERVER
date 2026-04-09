"""Unit tests for BaseVisionLLM and Vision LLM factory (B8)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from core.settings import (
    EmbeddingSettings,
    LLMSettings,
    RetrievalSettings,
    Settings,
    VectorStoreSettings,
    VisionLLMSettings,
)
from libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import LLMFactory


# ---------------------------------------------------------------------------
# Fake Vision LLM for testing
# ---------------------------------------------------------------------------


class FakeVisionLLM(BaseVisionLLM):
    """Minimal Vision LLM for unit tests."""

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace: Any | None = None,
    ) -> ChatResponse:
        return ChatResponse(content="fake text response", model="fake-vision")

    def chat_with_image(
        self,
        text: str,
        image: str | bytes,
        *,
        trace: Any | None = None,
    ) -> ChatResponse:
        return ChatResponse(
            content=f"described: {text}",
            model="fake-vision",
            metadata={"image_type": "bytes" if isinstance(image, bytes) else "path"},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**vision_overrides: Any) -> Settings:
    """Build Settings with vision_llm section."""
    vision = VisionLLMSettings(provider="azure", model="gpt-4o", **vision_overrides)
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
        vision_llm=vision,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaseVisionLLM:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseVisionLLM()  # type: ignore[abstract]

    def test_is_subclass_of_base_llm(self) -> None:
        assert issubclass(BaseVisionLLM, BaseLLM)

    def test_fake_vision_llm_chat_with_image(self) -> None:
        vllm = FakeVisionLLM()
        resp = vllm.chat_with_image("Describe this", b"\x89PNG\r\n\x1a\n")
        assert "described" in resp.content
        assert resp.metadata["image_type"] == "bytes"

    def test_fake_vision_llm_chat_with_image_path(self) -> None:
        vllm = FakeVisionLLM()
        resp = vllm.chat_with_image("Describe", "/tmp/test.png")
        assert resp.metadata["image_type"] == "path"


class TestVisionLLMImageHelpers:
    def test_encode_image_bytes(self) -> None:
        data = b"hello world"
        encoded = BaseVisionLLM.encode_image(data)
        import base64

        assert encoded == base64.b64encode(data).decode("utf-8")

    def test_encode_image_path_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            BaseVisionLLM.encode_image("/nonexistent/image.png")

    def test_encode_image_path_exists(self, tmp_path: Any) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\ndata")
        encoded = BaseVisionLLM.encode_image(str(img))
        import base64

        assert encoded == base64.b64encode(b"\x89PNG\r\n\x1a\ndata").decode("utf-8")

    def test_guess_mime_type_png(self) -> None:
        assert BaseVisionLLM.guess_mime_type("/path/to/img.png") == "image/png"

    def test_guess_mime_type_jpg(self) -> None:
        assert BaseVisionLLM.guess_mime_type("photo.jpg") == "image/jpeg"

    def test_guess_mime_type_jpeg(self) -> None:
        assert BaseVisionLLM.guess_mime_type("photo.jpeg") == "image/jpeg"

    def test_guess_mime_type_unknown_ext(self) -> None:
        assert BaseVisionLLM.guess_mime_type("file.xyz") == "application/octet-stream"

    def test_guess_mime_type_bytes_png(self) -> None:
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        assert BaseVisionLLM.guess_mime_type(png_header) == "image/png"

    def test_guess_mime_type_bytes_jpeg(self) -> None:
        jpeg_header = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        assert BaseVisionLLM.guess_mime_type(jpeg_header) == "image/jpeg"

    def test_guess_mime_type_bytes_unknown(self) -> None:
        assert BaseVisionLLM.guess_mime_type(b"random data") == "application/octet-stream"


class TestVisionLLMFactory:
    def test_unknown_vision_provider_raises(self) -> None:
        settings = _make_settings()
        # Override provider to something unknown
        settings.vision_llm.provider = "bogus_vision"
        with pytest.raises(ValueError, match="Unknown Vision LLM provider"):
            LLMFactory.create_vision_llm(settings)

    def test_register_and_create_vision_provider(self) -> None:
        LLMFactory.register_vision_provider(
            "fake",
            f"{FakeVisionLLM.__module__}.{FakeVisionLLM.__qualname__}",
        )
        settings = _make_settings()
        settings.vision_llm.provider = "fake"
        vllm = LLMFactory.create_vision_llm(settings)
        assert isinstance(vllm, FakeVisionLLM)
        resp = vllm.chat_with_image("test", b"\x89PNG")
        assert "described" in resp.content

    def test_create_vision_llm_is_base_vision_llm(self) -> None:
        LLMFactory.register_vision_provider(
            "fake",
            f"{FakeVisionLLM.__module__}.{FakeVisionLLM.__qualname__}",
        )
        settings = _make_settings()
        settings.vision_llm.provider = "fake"
        vllm = LLMFactory.create_vision_llm(settings)
        assert isinstance(vllm, BaseVisionLLM)

    def test_create_vision_llm_also_base_llm(self) -> None:
        LLMFactory.register_vision_provider(
            "fake",
            f"{FakeVisionLLM.__module__}.{FakeVisionLLM.__qualname__}",
        )
        settings = _make_settings()
        settings.vision_llm.provider = "fake"
        vllm = LLMFactory.create_vision_llm(settings)
        # Vision LLM should also be usable as a plain LLM
        assert isinstance(vllm, BaseLLM)
        resp = vllm.chat_str("hello")
        assert resp == "fake text response"
