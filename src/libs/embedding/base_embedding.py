"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEmbedding(ABC):
    """Abstract base for embedding providers.

    Subclasses must implement :meth:`embed`.
    """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        *,
        trace: Any | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of strings to embed.
            trace: Optional TraceContext for observability.

        Returns:
            List of embedding vectors, one per input text.
            Each vector is a list of floats with length == :attr:`dimensions`.

        Raises:
            ValueError: If texts is empty.
        """
        ...

    def embed_single(self, text: str, *, trace: Any | None = None) -> list[float]:
        """Convenience: embed a single string."""
        return self.embed([text], trace=trace)[0]
