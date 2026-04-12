"""LLM-based reranker that delegates ranking decisions to a language model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.settings import Settings
from libs.llm.base_llm import BaseLLM
from libs.reranker.base_reranker import BaseReranker, RerankCandidate
from observability.logger import get_logger as _get_logger

logger = _get_logger("rag.reranker.llm")

_RERANK_PROMPT_PATH = Path("config/prompts/rerank.txt")


class LLMReranker(BaseReranker):
    """Reranker that uses an LLM to judge and reorder candidates."""

    def __init__(self, settings: Settings, llm: BaseLLM) -> None:
        self._llm = llm
        self._prompt_template = self._load_prompt_template()

    # ------------------------------------------------------------------
    # Prompt template
    # ------------------------------------------------------------------

    @staticmethod
    def _load_prompt_template() -> str:
        if _RERANK_PROMPT_PATH.exists():
            return _RERANK_PROMPT_PATH.read_text()
        # Fallback template if file is missing
        return (
            "You are a relevance judge. Given a query and a list of document passages, "
            "rank them by relevance to the query.\n\n"
            "Query: {query}\n\nPassages:\n{passages}\n\n"
            "Return a JSON array of passage indices ordered by relevance (most relevant first). "
            "Only include passages that are relevant to the query.\n"
            "Example output: [2, 0, 5]\n\nRanked indices:\n"
        )

    # ------------------------------------------------------------------
    # BaseReranker interface
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_k: int | None = None,
        trace: Any | None = None,
    ) -> list[RerankCandidate]:
        if not candidates:
            return []

        # Build numbered passages block
        passages_lines: list[str] = []
        for idx, c in enumerate(candidates):
            passages_lines.append(f"[{idx}] {c.text}")
        passages = "\n".join(passages_lines)

        prompt = self._prompt_template.format(query=query, passages=passages)

        try:
            raw = self._llm.chat_str(prompt)
            ranked_indices = self._parse_response(raw, len(candidates))
        except Exception:
            logger.exception("LLM reranker failed; returning original order")
            result = list(candidates)
            if top_k is not None:
                result = result[:top_k]
            return result

        # Reorder candidates according to LLM ranking
        reordered: list[RerankCandidate] = []
        for pos, idx in enumerate(ranked_indices):
            cand = candidates[idx]
            # Assign a descending relevance score based on rank position
            reordered.append(
                RerankCandidate(
                    id=cand.id,
                    text=cand.text,
                    score=float(len(ranked_indices) - pos),
                    metadata=cand.metadata,
                )
            )

        # Append any candidates not mentioned by the LLM (preserve original order)
        seen = set(ranked_indices)
        for i, cand in enumerate(candidates):
            if i not in seen:
                reordered.append(
                    RerankCandidate(
                        id=cand.id,
                        text=cand.text,
                        score=0.0,
                        metadata=cand.metadata,
                    )
                )

        if top_k is not None:
            reordered = reordered[:top_k]

        return reordered

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str, max_index: int) -> list[int]:
        """Extract a JSON list of integer indices from *raw* LLM output.

        Returns an empty list on any parse failure, which triggers the
        fallback path in :meth:`rerank`.
        """
        # Try to find a JSON array anywhere in the response
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON array found in LLM response")

        try:
            indices = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in LLM response: {exc}") from exc

        if not isinstance(indices, list):
            raise ValueError("LLM response is not a JSON list")

        valid: list[int] = []
        for item in indices:
            if isinstance(item, int) and 0 <= item < max_index:
                valid.append(item)

        return valid
