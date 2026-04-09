"""Abstract base class for document loaders.

All loader implementations (PDF, DOCX, etc.) must inherit from
:class:`BaseLoader` and implement the :meth:`load` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from core.types import Document


class BaseLoader(ABC):
    """Abstract base for document loaders.

    Subclasses must implement :meth:`load`.
    """

    @abstractmethod
    def load(self, path: str, **kwargs: Any) -> Document:
        """Load a file and return a :class:`Document`.

        Args:
            path: File path to load.
            **kwargs: Additional loader-specific options.

        Returns:
            A :class:`Document` with at least ``metadata.source_path``.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If the file cannot be parsed.
        """
        ...

    def _validate_path(self, path: str) -> Path:
        """Check that the file exists and return its resolved Path."""
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not p.is_file():
            raise ValueError(f"Not a file: {path}")
        return p
