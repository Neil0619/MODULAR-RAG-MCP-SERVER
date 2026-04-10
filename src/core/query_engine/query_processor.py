"""QueryProcessor — extract keywords and parse filters from a raw query.

Performs rule-based keyword extraction (tokenize + stop-word filtering)
and parses a generic ``filters`` dict from the query or explicit params.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core.types import ProcessedQuery

# Tokenizer: lowercase alpha sequences of 2+ chars
_WORD_RE = re.compile(r"[a-zA-Z]{2,}")

# Common English stop words (shared with SparseEncoder for consistency)
_STOP_WORDS = frozenset({
    "the", "is", "at", "which", "on", "and", "or", "an", "be",
    "this", "that", "with", "for", "are", "but", "not", "you",
    "all", "can", "had", "her", "was", "one", "our", "out",
    "has", "have", "from", "been", "will", "each", "make",
    "like", "were", "they", "them", "what", "when", "their",
    "there", "would", "about", "how", "who", "did", "does",
    "why", "just", "than", "then", "also", "more", "some",
    "could", "into", "very", "most", "only", "other", "should",
    "may", "might", "shall", "these", "those", "such", "any",
    "being", "because", "between", "both", "during", "before",
    "after", "above", "below", "same", "another", "much",
    "its", "it", "we", "he", "she", "me", "my", "your",
    "his", "him", "us", "so", "if", "no", "up", "do",
    # Chinese common stop words (single-char)
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这",
})

# Chinese character tokenization (2+ consecutive CJK chars)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")


class QueryProcessor:
    """Pre-process a raw query into keywords and filters.

    Args:
        use_stop_words: Whether to filter stop words. Defaults to ``True``.
    """

    def __init__(self, use_stop_words: bool = True) -> None:
        self._use_stop_words = use_stop_words

    def process(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> ProcessedQuery:
        """Extract keywords and produce a ProcessedQuery.

        Args:
            query: Raw user query string.
            filters: Optional pre-parsed filters (e.g. collection, doc_type).

        Returns:
            ProcessedQuery with ``keywords`` (non-empty for non-empty input)
            and ``filters`` dict.
        """
        keywords = self._extract_keywords(query)
        return ProcessedQuery(
            original_query=query,
            keywords=keywords,
            filters=filters or {},
        )

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text via tokenization and stop-word removal."""
        if not text or not text.strip():
            return []

        # English tokens
        en_tokens = _WORD_RE.findall(text.lower())

        # Chinese tokens
        cjk_tokens = _CJK_RE.findall(text.lower())

        all_tokens = en_tokens + cjk_tokens

        if self._use_stop_words:
            all_tokens = [t for t in all_tokens if t not in _STOP_WORDS]

        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for token in all_tokens:
            if token not in seen:
                seen.add(token)
                result.append(token)

        return result
