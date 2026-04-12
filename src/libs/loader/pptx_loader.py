"""PPTX loader — reads .pptx files (placeholder for J3)."""

from __future__ import annotations

from typing import Any

from core.types import Document
from libs.loader.base_loader import BaseLoader

try:
    from pptx import Presentation
except ImportError:
    Presentation = None  # type: ignore[assignment]


class PptxLoader(BaseLoader):
    """Load a PPTX file into a :class:`Document`."""

    def load(self, path: str, **kwargs: Any) -> Document:
        if Presentation is None:
            raise RuntimeError(
                "python-pptx is required for PPTX loading. "
                "Install with: pip install python-pptx"
            )
        raise NotImplementedError("PptxLoader will be implemented in J3")
