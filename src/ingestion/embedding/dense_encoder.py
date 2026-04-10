"""DenseEncoder — batch-encode chunk text into dense vectors.

Uses ``libs.embedding`` (via ``EmbeddingFactory``) to embed chunk text
into dense vectors for vector store retrieval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.types import Chunk
from libs.embedding.embedding_factory import EmbeddingFactory

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext


class DenseEncoder:
    """Encode chunk text into dense embedding vectors.

    Args:
        settings: Application settings (used to create the embedding provider).
    """

    def __init__(self, settings: Settings) -> None:
        self._embedder = EmbeddingFactory.create(settings)
        self._dimensions = self._embedder.dimensions

    @property
    def dimensions(self) -> int:
        """Dimensionality of the output vectors."""
        return self._dimensions

    def encode(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[list[float]]:
        """Encode all chunk texts into dense vectors.

        Args:
            chunks: Chunks whose ``text`` field will be embedded.
            trace: Optional trace context.

        Returns:
            List of embedding vectors, one per chunk (same order).

        Raises:
            ValueError: If chunks is empty.
            RuntimeError: If embedding fails.
        """
        if not chunks:
            raise ValueError("chunks must not be empty")

        texts = [chunk.text for chunk in chunks]
        return self._embedder.embed(texts)

    def encode_texts(
        self,
        texts: list[str],
        trace: TraceContext | None = None,
    ) -> list[list[float]]:
        """Encode raw text strings into dense vectors.

        Args:
            texts: List of strings to embed.
            trace: Optional trace context.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            raise ValueError("texts must not be empty")

        return self._embedder.embed(texts)
