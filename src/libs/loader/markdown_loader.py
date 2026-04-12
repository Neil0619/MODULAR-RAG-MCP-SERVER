"""Markdown loader — reads .md files with heading extraction."""

from __future__ import annotations

import hashlib
from typing import Any

from core.types import Document
from libs.loader.base_loader import BaseLoader


class MarkdownLoader(BaseLoader):
    """Load a Markdown file into a :class:`Document`."""

    def load(self, path: str, **kwargs: Any) -> Document:
        file_path = self._validate_path(path)
        text = file_path.read_text(encoding="utf-8")
        headings = [line.strip() for line in text.splitlines() if line.startswith("#")]
        doc_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return Document(
            text=text,
            metadata={
                "source_path": str(file_path),
                "doc_hash": doc_hash,
                "doc_type": "markdown",
                "heading_outline": headings,
                "images": [],
            },
        )
