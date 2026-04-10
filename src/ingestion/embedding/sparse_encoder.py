"""SparseEncoder — compute BM25-ready sparse term weights for chunks.

Tokenizes chunk text and computes term frequency weights suitable for
BM25 indexing. The output is a list of ``{term: weight}`` dicts, one
per chunk, matching the ``ChunkRecord.sparse_vector`` contract.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from core.types import Chunk

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

# Tokenizer: lowercase, split on non-alpha, filter short tokens
_WORD_RE = re.compile(r"[a-zA-Z]{2,}")
# Common English stop words
_STOP_WORDS = frozenset({
    "the", "is", "at", "which", "on", "and", "or", "an", "be",
    "this", "that", "with", "for", "are", "but", "not", "you",
    "all", "can", "had", "her", "was", "one", "our", "out",
    "has", "have", "from", "been", "will", "each", "make",
    "like", "has", "been", "were", "they", "them", "what",
    "when", "which", "their", "there", "would", "about",
})


class SparseEncoder:
    """Compute sparse term weights for BM25 indexing.

    Uses TF (term frequency) weighting by default. The output can be
    consumed by BM25Indexer (C11) for IDF re-weighting.
    """

    def __init__(self, use_stop_words: bool = True) -> None:
        self._use_stop_words = use_stop_words

    def encode(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[dict[str, float]]:
        """Compute sparse term weights for each chunk.

        Args:
            chunks: Chunks whose ``text`` will be tokenized.
            trace: Optional trace context.

        Returns:
            List of ``{term: tf_weight}`` dicts, one per chunk.
            Empty chunks produce an empty dict.
        """
        return [self._encode_text(chunk.text) for chunk in chunks]

    def encode_texts(
        self,
        texts: list[str],
        trace: TraceContext | None = None,
    ) -> list[dict[str, float]]:
        """Compute sparse term weights for raw text strings.

        Args:
            texts: List of strings to tokenize and weight.
            trace: Optional trace context.

        Returns:
            List of ``{term: tf_weight}`` dicts.
        """
        return [self._encode_text(t) for t in texts]

    def _encode_text(self, text: str) -> dict[str, float]:
        """Tokenize and compute TF weights for a single text."""
        tokens = self._tokenize(text)
        if not tokens:
            return {}

        counts = Counter(tokens)
        max_freq = max(counts.values())

        # Normalized TF: term_freq / max_term_freq (standard BM25 normalization)
        return {
            term: freq / max_freq
            for term, freq in counts.items()
        }

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase words, filtering stop words."""
        words = _WORD_RE.findall(text.lower())
        if self._use_stop_words:
            words = [w for w in words if w not in _STOP_WORDS]
        return words
