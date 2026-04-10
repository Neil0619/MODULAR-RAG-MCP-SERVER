"""Unit tests for ImageCaptioner (C7).

Uses mocked Vision LLM — no real API calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from core.settings import (
    EmbeddingSettings,
    IngestionSettings,
    LLMSettings,
    RetrievalSettings,
    Settings,
    VectorStoreSettings,
    VisionLLMSettings,
)
from core.types import Chunk
from ingestion.transform.image_captioner import ImageCaptioner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(enabled: bool = False) -> Settings:
    ingestion = IngestionSettings(
        image_captioner=IngestionSettings.ImageCaptionerSettings(enabled=enabled),
    )
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
        ingestion=ingestion,
        vision_llm=VisionLLMSettings(provider="azure", model="gpt-4o"),
    )


def _make_chunk_with_images(has_images: bool = True) -> Chunk:
    """Create a chunk with image references."""
    images = [
        {"id": "img_001", "path": "/data/images/test/img_001.png", "page": 0},
        {"id": "img_002", "path": "/data/images/test/img_002.png", "page": 0},
    ] if has_images else []

    meta: dict[str, Any] = {"source_path": "/test.pdf"}
    if has_images:
        meta["image_refs"] = ["img_001", "img_002"]
        meta["images"] = images

    text = "Some text [IMAGE: img_001] and [IMAGE: img_002] more text" if has_images else "No images"
    return Chunk(id="c1", text=text, metadata=meta)


def _make_chunk_without_images() -> Chunk:
    return _make_chunk_with_images(has_images=False)


# ---------------------------------------------------------------------------
# Disabled mode
# ---------------------------------------------------------------------------


class TestDisabledMode:
    def test_disabled_marks_unprocessed(self) -> None:
        captioner = ImageCaptioner(_make_settings(enabled=False))
        chunk = _make_chunk_with_images()
        result = captioner.transform([chunk])
        assert result[0].metadata["has_unprocessed_images"] is True
        assert "image_captions" not in result[0].metadata

    def test_disabled_preserves_image_refs(self) -> None:
        captioner = ImageCaptioner(_make_settings(enabled=False))
        chunk = _make_chunk_with_images()
        result = captioner.transform([chunk])
        assert "image_refs" in result[0].metadata
        assert len(result[0].metadata["image_refs"]) == 2

    def test_no_image_refs_skipped(self) -> None:
        captioner = ImageCaptioner(_make_settings(enabled=False))
        chunk = _make_chunk_without_images()
        result = captioner.transform([chunk])
        assert "has_unprocessed_images" not in result[0].metadata
        assert "image_captions" not in result[0].metadata


# ---------------------------------------------------------------------------
# Enabled mode (mocked Vision LLM)
# ---------------------------------------------------------------------------


class TestEnabledMode:
    def _make_captioner_with_mock(self) -> tuple[ImageCaptioner, MagicMock]:
        mock_vllm = MagicMock()
        mock_vllm.chat_with_image.return_value = MagicMock(
            content="A bar chart showing sales data for Q1 2024."
        )
        settings = _make_settings(enabled=True)
        return ImageCaptioner(settings, vision_llm=mock_vllm), mock_vllm

    def test_captions_generated(self) -> None:
        captioner, _ = self._make_captioner_with_mock()
        chunk = _make_chunk_with_images()
        result = captioner.transform([chunk])
        captions = result[0].metadata.get("image_captions", {})
        assert len(captions) == 2
        assert "img_001" in captions
        assert "img_002" in captions

    def test_caption_content_populated(self) -> None:
        captioner, _ = self._make_captioner_with_mock()
        chunk = _make_chunk_with_images()
        result = captioner.transform([chunk])
        assert "bar chart" in result[0].metadata["image_captions"]["img_001"]

    def test_vision_llm_called_with_image_path(self) -> None:
        captioner, mock_vllm = self._make_captioner_with_mock()
        chunk = _make_chunk_with_images()
        captioner.transform([chunk])
        # Should be called twice (2 images)
        assert mock_vllm.chat_with_image.call_count == 2
        # First call should use img_001 path
        call_args = mock_vllm.chat_with_image.call_args_list[0]
        assert call_args[0][1] == "/data/images/test/img_001.png"

    def test_no_image_refs_not_touched(self) -> None:
        captioner, _ = self._make_captioner_with_mock()
        chunk = _make_chunk_without_images()
        result = captioner.transform([chunk])
        assert "image_captions" not in result[0].metadata
        assert "has_unprocessed_images" not in result[0].metadata


# ---------------------------------------------------------------------------
# Fallback / error handling
# ---------------------------------------------------------------------------


class TestFallback:
    def test_vision_llm_failure_marks_unprocessed(self) -> None:
        mock_vllm = MagicMock()
        mock_vllm.chat_with_image.side_effect = Exception("Vision API error")
        settings = _make_settings(enabled=True)
        captioner = ImageCaptioner(settings, vision_llm=mock_vllm)

        chunk = _make_chunk_with_images()
        result = captioner.transform([chunk])
        assert result[0].metadata.get("has_unprocessed_images") is True

    def test_partial_failure_preserves_good_captions(self) -> None:
        mock_vllm = MagicMock()
        # First image succeeds, second fails
        mock_vllm.chat_with_image.side_effect = [
            MagicMock(content="Chart showing data trends"),
            Exception("Timeout"),
        ]
        settings = _make_settings(enabled=True)
        captioner = ImageCaptioner(settings, vision_llm=mock_vllm)

        chunk = _make_chunk_with_images()
        result = captioner.transform([chunk])
        # img_001 should have caption, img_002 failed
        captions = result[0].metadata.get("image_captions", {})
        assert "img_001" in captions
        assert "img_002" not in captions
        # Partial success still writes captions, no unprocessed marker
        # (only set if NO captions generated)

    def test_vision_llm_none_marks_unprocessed(self) -> None:
        settings = _make_settings(enabled=True)
        captioner = ImageCaptioner(settings, vision_llm=None)
        chunk = _make_chunk_with_images()
        result = captioner.transform([chunk])
        assert result[0].metadata["has_unprocessed_images"] is True

    def test_empty_caption_treated_as_failure(self) -> None:
        mock_vllm = MagicMock()
        mock_vllm.chat_with_image.return_value = MagicMock(content="   ")
        settings = _make_settings(enabled=True)
        captioner = ImageCaptioner(settings, vision_llm=mock_vllm)

        chunk = _make_chunk_with_images()
        result = captioner.transform([chunk])
        assert result[0].metadata.get("has_unprocessed_images") is True

    def test_missing_image_path_skipped(self) -> None:
        mock_vllm = MagicMock()
        mock_vllm.chat_with_image.return_value = MagicMock(content="Caption")
        settings = _make_settings(enabled=True)
        captioner = ImageCaptioner(settings, vision_llm=mock_vllm)

        # Image with no path
        chunk = Chunk(
            id="c1",
            text="text [IMAGE: img_x]",
            metadata={
                "image_refs": ["img_x"],
                "images": [{"id": "img_x"}],  # no path
            },
        )
        result = captioner.transform([chunk])
        mock_vllm.chat_with_image.assert_not_called()


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


class TestPromptLoading:
    def test_default_prompt_has_context(self) -> None:
        captioner = ImageCaptioner(_make_settings(enabled=False))
        assert "{context}" in captioner._prompt_template

    def test_custom_prompt(self, tmp_path: Any) -> None:
        p = tmp_path / "prompt.txt"
        p.write_text("Custom {context}")
        captioner = ImageCaptioner(
            _make_settings(enabled=False),
            prompt_path=str(p),
        )
        assert captioner._prompt_template == "Custom {context}"
