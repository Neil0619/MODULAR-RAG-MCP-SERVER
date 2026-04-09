"""Abstract base class for chunk transforms.

Transforms modify chunk text or metadata in-place (e.g. noise removal,
metadata enrichment, image captioning).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from core.types import Chunk

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseTransform(ABC):
    """Abstract base for chunk transformation stages."""

    @abstractmethod
    def transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """Transform a list of chunks.

        Args:
            chunks: Input chunks to transform.
            trace: Optional trace context for observability.

        Returns:
            Transformed chunks (may be the same objects, modified in-place).
        """
        ...
