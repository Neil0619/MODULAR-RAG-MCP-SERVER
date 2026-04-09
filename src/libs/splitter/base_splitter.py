"""Abstract base class for text splitters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSplitter(ABC):
    """Abstract base for text splitting strategies."""

    @abstractmethod
    def split_text(
        self,
        text: str,
        *,
        trace: Any | None = None,
    ) -> list[str]:
        """Split text into chunks.

        Args:
            text: Input text to split.
            trace: Optional TraceContext for observability.

        Returns:
            List of text chunks.
        """
        ...
