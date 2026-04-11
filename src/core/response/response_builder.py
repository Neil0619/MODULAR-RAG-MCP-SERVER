"""ResponseBuilder — construct MCP tool responses from retrieval results.

Builds a Markdown body with inline citation markers (e.g. [1], [2])
and a structured citations list for programmatic consumption.
"""

from __future__ import annotations

from typing import Any

from core.response.citation_generator import CitationGenerator, Citation
from core.types import RetrievalResult


class ResponseBuilder:
    """Build formatted MCP responses from retrieval results.

    The response contains:
    - A readable Markdown body with ``[1]``, ``[2]`` citation markers
    - A structured ``citations`` list with source/score/page info
    """

    @staticmethod
    def build(results: list[RetrievalResult], query: str) -> dict[str, Any]:
        """Build a complete MCP tool response.

        Args:
            results: Ranked retrieval results.
            query: Original user query.

        Returns:
            Dict with ``text`` (Markdown) and ``citations`` (list).
        """
        if not results:
            return {
                "text": f"No relevant documents found for: \"{query}\"\n\n"
                        "Please run `ingest.py` to index documents first.",
                "citations": [],
            }

        citations = CitationGenerator.generate(results)
        citation_map = {c.chunk_id: c for c in citations}

        # Build Markdown body with citation markers
        body_parts: list[str] = []
        body_parts.append(f"## Search Results: \"{query}\"\n")
        body_parts.append(f"Found **{len(results)}** relevant chunks.\n")

        for r in results:
            c = citation_map.get(r.chunk_id)
            marker = f"[{c.index}]" if c else ""
            title = r.metadata.get("title", "")
            source = r.metadata.get("source_path", "unknown")

            if title:
                body_parts.append(f"### {marker} {title}")
            else:
                body_parts.append(f"### {marker} Result {c.index if c else '?'}")

            body_parts.append(f"*Source: {source}*  |  Score: {r.score:.4f}\n")
            body_parts.append(r.text)
            body_parts.append("")  # blank line

        # Citations section
        body_parts.append("---\n")
        body_parts.append("## Citations\n")
        for c in citations:
            page_str = f", p.{c.page}" if c.page is not None else ""
            body_parts.append(
                f"[{c.index}] {c.source}{page_str} — score: {c.score:.4f}"
            )

        return {
            "text": "\n".join(body_parts),
            "citations": [c.to_dict() for c in citations],
        }
