"""MetadataEnricher — rule-based + optional LLM metadata enrichment.

Adds ``title``, ``summary``, and ``tags`` to chunk metadata.
When LLM is enabled, generates semantically rich metadata via LLM call.
On failure, falls back to rule-based extraction.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext

_METADATA_PROMPT = (
    "Analyze the following text chunk and extract metadata.\n\n"
    "Respond in this exact format:\n"
    "TITLE: <a concise title, max 10 words>\n"
    "SUMMARY: <a one-sentence summary>\n"
    "TAGS: <comma-separated keywords, max 5>\n\n"
    "Text:\n{text}"
)


class MetadataEnricher(BaseTransform):
    """Enrich chunk metadata with title, summary, and tags.

    Args:
        settings: Application settings.
        llm: Optional pre-created LLM instance.
    """

    def __init__(
        self,
        settings: Settings,
        llm: Any | None = None,
    ) -> None:
        self._settings = settings
        self._use_llm = settings.ingestion.metadata_enricher.use_llm

        if llm is not None:
            self._llm = llm
        elif self._use_llm:
            from libs.llm.llm_factory import LLMFactory

            self._llm = LLMFactory.create(settings)
        else:
            self._llm = None

    def transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """Enrich all chunks with title, summary, and tags metadata."""
        for chunk in chunks:
            try:
                if self._use_llm and self._llm is not None:
                    llm_meta = self._llm_enrich(chunk.text)
                    if llm_meta is not None:
                        chunk.metadata.update(llm_meta)
                        chunk.metadata["enriched_by"] = "llm"
                    else:
                        self._apply_rule_based(chunk)
                        chunk.metadata["enrichment_fallback"] = "llm_failed"
                else:
                    self._apply_rule_based(chunk)
            except Exception:
                self._apply_rule_based(chunk)
                chunk.metadata["enrichment_fallback"] = "exception"

        return chunks

    def _apply_rule_based(self, chunk: Chunk) -> None:
        """Apply rule-based metadata extraction as fallback."""
        text = chunk.text.strip()
        chunk.metadata["enriched_by"] = "rule"

        # Title: first non-empty line, truncated
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            title = lines[0]
            if len(title) > 100:
                title = title[:97] + "..."
            chunk.metadata["title"] = title
        else:
            chunk.metadata["title"] = ""

        # Summary: first 200 chars
        if len(text) > 200:
            chunk.metadata["summary"] = text[:197] + "..."
        else:
            chunk.metadata["summary"] = text

        # Tags: extract significant words (alpha-only, >3 chars)
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        # Deduplicate, take top 5
        seen: set[str] = set()
        tags: list[str] = []
        for w in words:
            if w not in seen:
                seen.add(w)
                tags.append(w)
                if len(tags) >= 5:
                    break
        chunk.metadata["tags"] = tags

    def _llm_enrich(self, text: str) -> dict[str, Any] | None:
        """Use LLM to generate title, summary, tags.

        Returns dict with keys or None on failure.
        """
        if not text.strip():
            return None

        try:
            prompt = _METADATA_PROMPT.replace("{text}", text[:2000])
            response = self._llm.chat_str(prompt)
            return self._parse_llm_response(response)
        except Exception:
            return None

    @staticmethod
    def _parse_llm_response(response: str) -> dict[str, Any] | None:
        """Parse structured LLM response into metadata dict."""
        if not response or not response.strip():
            return None

        result: dict[str, Any] = {}
        title_match = re.search(r"TITLE:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        summary_match = re.search(r"SUMMARY:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        tags_match = re.search(r"TAGS:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)

        if title_match:
            result["title"] = title_match.group(1).strip()
        if summary_match:
            result["summary"] = summary_match.group(1).strip()
        if tags_match:
            tags_str = tags_match.group(1).strip()
            result["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]

        # Must have at least title to be valid
        if "title" not in result:
            return None

        return result
