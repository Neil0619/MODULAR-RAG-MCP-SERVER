"""CitationGenerator — generate citation info from retrieval results.

Produces structured citation objects with source, page, chunk_id,
score, and an optional text preview.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.types import RetrievalResult


@dataclass
class Citation:
    """A single citation referencing a retrieved chunk.

    Attributes:
        index: 1-based citation number (e.g. [1], [2]).
        source: Source file path.
        page: Page number in the source document (if available).
        chunk_id: ID of the cited chunk.
        score: Relevance score.
        text_preview: Short text preview of the cited chunk.
    """

    index: int
    source: str = ""
    page: int | None = None
    chunk_id: str = ""
    score: float = 0.0
    text_preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
        }
        if self.page is not None:
            d["page"] = self.page
        if self.text_preview:
            d["text_preview"] = self.text_preview
        return d


class CitationGenerator:
    """Generate citations from retrieval results."""

    @staticmethod
    def generate(results: list[RetrievalResult]) -> list[Citation]:
        """Create citation objects from retrieval results.

        Args:
            results: Ranked retrieval results.

        Returns:
            List of Citation objects with 1-based indices.
        """
        citations: list[Citation] = []
        for i, r in enumerate(results, start=1):
            preview = r.text[:150].replace("\n", " ") if r.text else ""
            if len(r.text) > 150:
                preview += "..."

            citations.append(
                Citation(
                    index=i,
                    source=r.metadata.get("source_path", ""),
                    page=r.metadata.get("page", r.metadata.get("page_count")),
                    chunk_id=r.chunk_id,
                    score=r.score,
                    text_preview=preview,
                )
            )
        return citations
