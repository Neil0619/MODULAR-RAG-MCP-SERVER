"""ImageCaptioner — generate captions for document images using Vision LLM.

When enabled and a Vision LLM is configured, iterates over chunks that
contain ``image_refs`` and generates text captions for each image via
the Vision LLM. Captions are written into ``metadata["image_captions"]``.

When disabled or on failure, chunks retain ``image_refs`` and are marked
with ``has_unprocessed_images`` — the pipeline continues unblocked.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext

_DEFAULT_PROMPT = (
    "You are an expert image analyst. Given an image from a technical "
    "document, provide a detailed and accurate description.\n\n"
    "Context from surrounding text:\n{context}\n\n"
    "Describe the image in detail.\n\nImage description:\n"
)


class ImageCaptioner(BaseTransform):
    """Generate captions for images referenced in chunks.

    Args:
        settings: Application settings.
        vision_llm: Optional pre-created Vision LLM instance.
        prompt_path: Path to prompt template file.
    """

    def __init__(
        self,
        settings: Settings,
        vision_llm: Any | None = None,
        prompt_path: str | None = None,
    ) -> None:
        self._settings = settings
        self._enabled = settings.ingestion.image_captioner.enabled

        if vision_llm is not None:
            self._vision_llm = vision_llm
        elif self._enabled:
            try:
                from libs.llm.llm_factory import LLMFactory

                self._vision_llm = LLMFactory.create_vision_llm(settings)
            except Exception:
                self._vision_llm = None
        else:
            self._vision_llm = None

        self._prompt_template = self._load_prompt(prompt_path)

    def transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """Generate captions for images referenced in chunks."""
        for chunk in chunks:
            image_refs = chunk.metadata.get("image_refs", [])
            if not image_refs:
                continue

            if not self._enabled or self._vision_llm is None:
                chunk.metadata["has_unprocessed_images"] = True
                continue

            images_meta = chunk.metadata.get("images", [])
            image_map = {img["id"]: img for img in images_meta if "id" in img}
            captions: dict[str, str] = {}

            for img_id in image_refs:
                img_info = image_map.get(img_id)
                if not img_info:
                    continue
                try:
                    caption = self._caption_image(img_info, chunk.text)
                    if caption:
                        captions[img_id] = caption
                except Exception:
                    # Per-image failure should not block others
                    continue

            if captions:
                chunk.metadata["image_captions"] = captions
            else:
                chunk.metadata["has_unprocessed_images"] = True

        return chunks

    def _caption_image(self, img_info: dict[str, Any], context_text: str) -> str | None:
        """Generate a caption for a single image.

        Args:
            img_info: Image metadata dict with ``path`` key.
            context_text: Surrounding chunk text for context.

        Returns:
            Caption string or None on failure.
        """
        img_path = img_info.get("path", "")
        if not img_path:
            return None

        try:
            prompt = self._prompt_template.replace(
                "{context}", context_text[:1000]
            )
            response = self._vision_llm.chat_with_image(prompt, img_path)
            caption = response.content.strip()
            return caption if caption else None
        except Exception:
            return None

    def _load_prompt(self, prompt_path: str | None = None) -> str:
        """Load prompt template from file, falling back to default."""
        if prompt_path is None:
            prompt_path = "config/prompts/image_captioning.txt"

        path = Path(prompt_path)
        if path.exists():
            return path.read_text(encoding="utf-8")

        return _DEFAULT_PROMPT
