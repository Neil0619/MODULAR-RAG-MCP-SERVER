"""Azure Vision LLM provider using Azure OpenAI with vision capabilities.

Supports GPT-4o / GPT-4-Vision-Preview for image understanding tasks
such as captioning, OCR, and visual Q&A.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

from openai import AzureOpenAI

from libs.llm.base_llm import ChatMessage, ChatResponse
from libs.llm.base_vision_llm import BaseVisionLLM

if TYPE_CHECKING:
    from core.settings import Settings


class AzureVisionLLM(BaseVisionLLM):
    """Vision LLM provider backed by Azure OpenAI.

    Reads configuration from ``settings.vision_llm``:
      - ``azure_endpoint``: Azure endpoint URL.
      - ``api_key``: Azure API key.
      - ``api_version``: Azure API version string.
      - ``deployment_name``: Azure deployment name (used as model).
      - ``model``: Fallback model name if deployment_name is empty.
      - ``max_image_size``: Max image dimension in pixels (default 2048).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        vision = settings.vision_llm
        self._model = vision.deployment_name or vision.model
        self._max_image_size = vision.max_image_size

        self._client = AzureOpenAI(
            azure_endpoint=vision.azure_endpoint,
            api_key=vision.api_key,
            api_version=vision.api_version,
        )

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace: Any | None = None,
    ) -> ChatResponse:
        """Send a chat completion request to Azure OpenAI."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[m.to_dict() for m in messages],  # type: ignore[arg-type]
                temperature=temperature or 0.0,
                max_tokens=max_tokens or 4096,
            )
        except Exception as exc:
            raise RuntimeError(f"Azure Vision chat request failed: {exc}") from exc

        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return ChatResponse(content=content, model=response.model or self._model, usage=usage)

    def chat_with_image(
        self,
        text: str,
        image: str | bytes,
        *,
        trace: Any | None = None,
    ) -> ChatResponse:
        """Send a multimodal chat request with text and an image.

        Args:
            text: Text prompt (e.g. "Describe this image in detail").
            image: File path (``str``) or raw image bytes.
            trace: Optional TraceContext for observability.

        Returns:
            ChatResponse with the model's analysis.

        Raises:
            RuntimeError: If the API call fails.
        """
        # Encode & optionally compress image
        image_bytes = self._load_image_bytes(image)
        image_bytes = self._resize_if_needed(image_bytes)
        b64 = self.encode_image(image_bytes)
        mime = self.guess_mime_type(image_bytes)
        data_uri = f"data:{mime};base64,{b64}"

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.0,
                max_tokens=4096,
            )
        except Exception as exc:
            raise RuntimeError(f"Azure Vision image request failed: {exc}") from exc

        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return ChatResponse(content=content, model=response.model or self._model, usage=usage)

    # -- Private helpers -----------------------------------------------------

    @staticmethod
    def _load_image_bytes(image: str | bytes) -> bytes:
        """Load image bytes from a path or pass through raw bytes."""
        if isinstance(image, bytes):
            return image
        from pathlib import Path

        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image}")
        return path.read_bytes()

    def _resize_if_needed(self, image_bytes: bytes) -> bytes:
        """Resize image if its dimensions exceed ``max_image_size``.

        Falls back to returning original bytes if Pillow is not installed
        or if the image cannot be decoded.
        """
        if self._max_image_size <= 0:
            return image_bytes

        try:
            from PIL import Image
        except ImportError:
            return image_bytes

        try:
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size
            max_dim = max(width, height)
            if max_dim <= self._max_image_size:
                return image_bytes

            # Calculate new dimensions preserving aspect ratio
            ratio = self._max_image_size / max_dim
            new_w = int(width * ratio)
            new_h = int(height * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

            buf = io.BytesIO()
            fmt = img.format or "PNG"
            img.save(buf, format=fmt)
            return buf.getvalue()
        except Exception:
            # If resizing fails, use original
            return image_bytes
