"""HTML loader — reads .html files (placeholder for J4)."""

from __future__ import annotations

from typing import Any

from core.types import Document
from libs.loader.base_loader import BaseLoader

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment]


class HtmlLoader(BaseLoader):
    """Load an HTML file into a :class:`Document`."""

    def load(self, path: str, **kwargs: Any) -> Document:
        if BeautifulSoup is None:
            raise RuntimeError(
                "beautifulsoup4 is required for HTML loading. "
                "Install with: pip install beautifulsoup4"
            )
        raise NotImplementedError("HtmlLoader will be implemented in J4")
