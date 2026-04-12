"""HTML loader — reads .html files using BeautifulSoup."""

from __future__ import annotations

import hashlib
from typing import Any

from core.types import Document
from libs.loader.base_loader import BaseLoader

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment]


class HtmlLoader(BaseLoader):
    """Load an HTML file into a :class:`Document`.

    Strips script/style tags, extracts title and headings.
    Requires the ``beautifulsoup4`` package.
    """

    def load(self, path: str, **kwargs: Any) -> Document:
        if BeautifulSoup is None:
            raise RuntimeError(
                "beautifulsoup4 is required for HTML loading. "
                "Install with: pip install beautifulsoup4"
            )

        from pathlib import Path as P

        file_path = self._validate_path(path)
        html_content = file_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        doc_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        # Extract metadata
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"])]

        return Document(
            text=text,
            metadata={
                "source_path": str(file_path),
                "doc_hash": doc_hash,
                "doc_type": "html",
                "title": title,
                "heading_outline": headings,
                "images": [],
            },
        )
