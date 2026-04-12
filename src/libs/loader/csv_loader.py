"""CSV loader — reads .csv files and formats as markdown table."""

from __future__ import annotations

import csv
import hashlib
from typing import Any

from core.types import Document
from libs.loader.base_loader import BaseLoader


class CsvLoader(BaseLoader):
    """Load a CSV file into a :class:`Document`.

    Formats the CSV as a markdown table for better downstream chunking.
    Uses stdlib ``csv`` module — no external dependencies.
    """

    def load(self, path: str, **kwargs: Any) -> Document:
        file_path = self._validate_path(path)

        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            text = ""
            headers: list[str] = []
        else:
            headers = rows[0]
            text_parts = [" | ".join(headers)]
            text_parts.append(" | ".join(["---"] * len(headers)))
            for row in rows[1:]:
                # Pad or trim row to match header length
                padded = row + [""] * (len(headers) - len(row))
                text_parts.append(" | ".join(padded[:len(headers)]))
            text = "\n".join(text_parts)

        doc_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        return Document(
            text=text,
            metadata={
                "source_path": str(file_path),
                "doc_hash": doc_hash,
                "doc_type": "csv",
                "row_count": max(len(rows) - 1, 0),
                "column_count": len(headers),
                "headers": headers,
                "images": [],
            },
        )
