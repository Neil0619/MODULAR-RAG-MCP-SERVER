"""DOCX loader — reads .docx files (placeholder for J3)."""

from __future__ import annotations

from typing import Any

from core.types import Document
from libs.loader.base_loader import BaseLoader

try:
    import docx  # python-docx
except ImportError:
    docx = None  # type: ignore[assignment]


class DocxLoader(BaseLoader):
    """Load a DOCX file into a :class:`Document`."""

    def load(self, path: str, **kwargs: Any) -> Document:
        if docx is None:
            raise RuntimeError(
                "python-docx is required for DOCX loading. "
                "Install with: pip install python-docx"
            )
        raise NotImplementedError("DocxLoader will be implemented in J3")
