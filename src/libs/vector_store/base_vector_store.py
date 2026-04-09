"""Abstract base class for vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorRecord:
    """A record to upsert into the vector store."""

    id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str = ""


@dataclass
class QueryResult:
    """A single result from a vector store query."""

    id: str
    score: float
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """Abstract base for vector store backends."""

    @abstractmethod
    def upsert(
        self,
        records: list[VectorRecord],
        *,
        collection: str = "default",
        trace: Any | None = None,
    ) -> int:
        """Upsert records. Returns count of upserted records."""
        ...

    @abstractmethod
    def query(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        collection: str = "default",
        trace: Any | None = None,
    ) -> list[QueryResult]:
        """Query by vector similarity. Returns top-k results."""
        ...

    @abstractmethod
    def get_by_ids(
        self,
        ids: list[str],
        *,
        collection: str = "default",
    ) -> list[dict[str, Any]]:
        """Fetch records by IDs. Returns list of {id, text, metadata} dicts."""
        ...

    @abstractmethod
    def delete_by_metadata(
        self,
        filter: dict[str, Any],
        *,
        collection: str = "default",
    ) -> int:
        """Delete records matching metadata filter. Returns deleted count."""
        ...

    @abstractmethod
    def get_collection_stats(
        self,
        collection: str = "default",
    ) -> dict[str, Any]:
        """Return stats for a collection (doc_count, etc.)."""
        ...
