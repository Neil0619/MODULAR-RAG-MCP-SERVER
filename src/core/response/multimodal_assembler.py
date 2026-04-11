"""MultimodalAssembler — assemble text + image content for MCP responses.

When retrieval results contain ``image_refs``, reads the image files from
disk and encodes them as base64 for inclusion in MCP tool responses.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from core.types import RetrievalResult


# MIME type mapping by extension
_MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
}


def _guess_mime(path: str) -> str:
    """Guess MIME type from file extension."""
    ext = Path(path).suffix.lower()
    return _MIME_MAP.get(ext, "application/octet-stream")


class MultimodalAssembler:
    """Assemble multimodal content (text + images) from retrieval results.

    Scans results for ``image_refs`` metadata, reads image files,
    and returns a list of content items (text + image) suitable for
    MCP tool responses.
    """

    @staticmethod
    def assemble(
        results: list[RetrievalResult],
        *,
        include_text: bool = True,
        max_images: int = 5,
    ) -> list[dict[str, Any]]:
        """Build multimodal content list from retrieval results.

        Args:
            results: Ranked retrieval results.
            include_text: Whether to include text content items.
            max_images: Maximum number of images to include.

        Returns:
            List of content dicts with ``type`` field (``text`` or ``image``).
        """
        content: list[dict[str, Any]] = []
        images_added = 0

        for r in results:
            # Add text content
            if include_text and r.text:
                content.append({
                    "type": "text",
                    "text": r.text,
                })

            # Add image content from image_refs
            image_refs = r.metadata.get("image_refs", [])
            images_meta = r.metadata.get("images", [])
            image_map = {img["id"]: img for img in images_meta if "id" in img}

            for img_id in image_refs:
                if images_added >= max_images:
                    break

                img_info = image_map.get(img_id)
                if not img_info:
                    continue

                img_path = img_info.get("path", "")
                if not img_path or not Path(img_path).exists():
                    continue

                try:
                    data = Path(img_path).read_bytes()
                    b64 = base64.b64encode(data).decode("ascii")
                    mime = _guess_mime(img_path)

                    content.append({
                        "type": "image",
                        "data": b64,
                        "mimeType": mime,
                    })
                    images_added += 1
                except Exception:
                    continue

        return content

    @staticmethod
    def has_images(results: list[RetrievalResult]) -> bool:
        """Check if any result contains image references."""
        for r in results:
            if r.metadata.get("image_refs"):
                return True
        return False
