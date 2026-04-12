"""PPTX loader — reads .pptx files using python-pptx."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from core.types import Document
from libs.loader.base_loader import BaseLoader

try:
    from pptx import Presentation
except ImportError:
    Presentation = None  # type: ignore[assignment]


class PptxLoader(BaseLoader):
    """Load a PPTX file into a :class:`Document`.

    Extracts slide text, shapes, and speaker notes. Each slide is
    separated by a heading for downstream chunking.

    Requires the ``python-pptx`` package.
    """

    def load(self, path: str, **kwargs: Any) -> Document:
        if Presentation is None:
            raise RuntimeError(
                "python-pptx is required for PPTX loading. "
                "Install with: pip install python-pptx"
            )

        file_path = self._validate_path(path)
        doc_hash = self._file_hash(file_path)
        prs = Presentation(str(file_path))

        text_parts: list[str] = []

        for slide_num, slide in enumerate(prs.slides):
            slide_text = f"## Slide {slide_num + 1}\n"

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text += shape.text + "\n"

            # Speaker notes
            try:
                if slide.notes_slide and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
                    if notes:
                        slide_text += f"\n[Speaker Notes]: {notes}\n"
            except Exception:
                pass

            text_parts.append(slide_text)

        full_text = "\n\n".join(text_parts)

        return Document(
            text=full_text,
            metadata={
                "source_path": str(file_path),
                "doc_hash": doc_hash,
                "doc_type": "pptx",
                "slide_count": len(prs.slides),
                "images": [],
            },
        )

    @staticmethod
    def _file_hash(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
