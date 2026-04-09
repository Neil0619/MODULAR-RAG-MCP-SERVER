"""Abstract base class for Vision LLM providers.

Vision LLMs extend standard LLMs with the ability to process images
alongside text. They are used by the ImageCaptioner (C7) to generate
text descriptions of document images.

All Vision LLM implementations (Azure GPT-4o, Qwen-VL, etc.) must
inherit from :class:`BaseVisionLLM` and implement
:meth:`chat_with_image`.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from libs.llm.base_llm import BaseLLM, ChatResponse


class BaseVisionLLM(BaseLLM, ABC):
    """Abstract base for Vision LLM providers.

    Extends :class:`BaseLLM` with a multimodal ``chat_with_image``
    method that accepts text plus an image (file path or raw bytes).

    Subclasses must implement :meth:`chat_with_image`.
    """

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image: str | bytes,
        *,
        trace: Any | None = None,
    ) -> ChatResponse:
        """Send a multimodal chat request with text and an image.

        Args:
            text: Text prompt (e.g. "Describe this image").
            image: Either a file path (``str``) or raw image bytes.
            trace: Optional TraceContext for observability.

        Returns:
            ChatResponse containing the LLM's analysis.
        """
        ...

    # -- Image preprocessing helpers ----------------------------------------

    @staticmethod
    def encode_image(image: str | bytes) -> str:
        """Encode an image to a base64 string.

        Args:
            image: File path (``str``) or raw bytes.

        Returns:
            Base64-encoded string of the image data.
        """
        if isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image}")
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    @staticmethod
    def guess_mime_type(image: str | bytes) -> str:
        """Guess MIME type from image data or file extension.

        Args:
            image: File path (``str``) or raw bytes (falls back to
                   ``application/octet-stream``).

        Returns:
            MIME type string (e.g. ``image/png``).
        """
        if isinstance(image, bytes):
            # Try to detect from magic bytes
            if image[:8] == b"\x89PNG\r\n\x1a\n":
                return "image/png"
            if image[:2] == b"\xff\xd8":
                return "image/jpeg"
            if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
                return "image/webp"
            if image[:4] == b"GIF8":
                return "image/gif"
            return "application/octet-stream"

        ext = Path(image).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
        }
        return mime_map.get(ext, "application/octet-stream")
