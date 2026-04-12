"""CSV loader — reads .csv files (placeholder for J4)."""

from __future__ import annotations

from typing import Any

from core.types import Document
from libs.loader.base_loader import BaseLoader


class CsvLoader(BaseLoader):
    """Load a CSV file into a :class:`Document`."""

    def load(self, path: str, **kwargs: Any) -> Document:
        raise NotImplementedError("CsvLoader will be implemented in J4")
