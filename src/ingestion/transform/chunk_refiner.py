"""ChunkRefiner — rule-based noise removal + optional LLM enhancement.

Applies two-stage text refinement to chunks:
1. **Rule-based cleanup**: Remove page headers/footers, excessive whitespace,
   HTML comments, format artifacts.
2. **Optional LLM refinement**: Use an LLM to further clean text (controlled
   by ``settings.ingestion.chunk_refiner.use_llm``).

When LLM fails or is disabled, the rule-based result is used with a
``refined_by: "rule"`` metadata marker.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext

# Default prompt template (fallback if file not found)
_DEFAULT_PROMPT = (
    "You are a text cleanup specialist. Your task is to refine the following "
    "text chunk extracted from a document, removing noise while preserving "
    "all meaningful content.\n\nOriginal text:\n{text}\n\nRefined text:\n"
)


class ChunkRefiner(BaseTransform):
    """Refine chunk text via rules + optional LLM.

    Args:
        settings: Application settings.
        llm: Optional pre-created LLM instance. If ``None``, created from settings.
        prompt_path: Path to prompt template file.
    """

    def __init__(
        self,
        settings: Settings,
        llm: Any | None = None,
        prompt_path: str | None = None,
    ) -> None:
        self._settings = settings
        self._use_llm = settings.ingestion.chunk_refiner.use_llm

        if llm is not None:
            self._llm = llm
        elif self._use_llm:
            from libs.llm.llm_factory import LLMFactory

            self._llm = LLMFactory.create(settings)
        else:
            self._llm = None

        self._prompt_template = self._load_prompt(prompt_path)

    def transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """Refine all chunks using rule-based + optional LLM approach."""
        for chunk in chunks:
            try:
                rule_text = self._rule_based_refine(chunk.text)

                if self._use_llm and self._llm is not None:
                    llm_text = self._llm_refine(rule_text, trace)
                    if llm_text is not None:
                        chunk.text = llm_text
                        chunk.metadata["refined_by"] = "llm"
                    else:
                        chunk.text = rule_text
                        chunk.metadata["refined_by"] = "rule"
                        chunk.metadata["refinement_fallback"] = "llm_failed"
                else:
                    chunk.text = rule_text
                    chunk.metadata["refined_by"] = "rule"
            except Exception:
                # Single chunk failure should not block others
                chunk.metadata["refined_by"] = "rule"
                chunk.metadata["refinement_fallback"] = "exception"

        return chunks

    def _rule_based_refine(self, text: str) -> str:
        """Apply rule-based noise removal.

        Handles:
        - Excessive whitespace / blank lines
        - Page header/footer patterns (e.g. "Page X of Y")
        - HTML comments
        - Format artifacts (--- separators, *** dividers)
        """
        result = text

        # Remove HTML comments
        result = re.sub(r"<!--.*?-->", "", result, flags=re.DOTALL)

        # Remove page header/footer patterns
        result = re.sub(
            r"^[\s]*(page\s+\d+\s*(of\s+\d+)?|p\.\s*\d+)[\s]*$",
            "",
            result,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        # Remove horizontal rule separators (3+ dashes/equals/asterisks)
        result = re.sub(r"^\s*[-=_*]{3,}\s*$", "", result, flags=re.MULTILINE)

        # Collapse 3+ consecutive blank lines into 1
        result = re.sub(r"\n{3,}", "\n\n", result)

        # Strip leading/trailing whitespace per line (preserve internal spacing)
        lines = result.split("\n")
        cleaned = []
        for line in lines:
            # Remove trailing whitespace, limit leading to reasonable amount
            cleaned.append(line.rstrip())

        result = "\n".join(cleaned)

        # Remove leading/trailing newlines
        result = result.strip("\n")

        return result

    def _llm_refine(self, text: str, trace: TraceContext | None = None) -> str | None:
        """Optional LLM-based refinement.

        Returns ``None`` if LLM call fails (triggers fallback).
        """
        if not text.strip():
            return text

        try:
            prompt = self._prompt_template.replace("{text}", text)
            response = self._llm.chat_str(prompt)
            refined = response.strip()

            if not refined:
                return None

            return refined
        except Exception:
            return None

    def _load_prompt(self, prompt_path: str | None = None) -> str:
        """Load prompt template from file, falling back to default."""
        if prompt_path is None:
            prompt_path = "config/prompts/chunk_refinement.txt"

        path = Path(prompt_path)
        if path.exists():
            return path.read_text(encoding="utf-8")

        return _DEFAULT_PROMPT
