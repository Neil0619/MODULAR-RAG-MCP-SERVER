"""Core data types shared across ingestion, retrieval, and MCP tools.

These types form the contract between pipeline stages:

- :class:`Document` — raw document loaded from a file (PDF, etc.)
- :class:`Chunk` — a text segment split from a Document
- :class:`ChunkRecord` — a Chunk enriched with vectors for storage/retrieval

Image placeholder convention
----------------------------
Images in ``Document.text`` are marked with ``[IMAGE: {image_id}]``.
Metadata carries an ``images`` list with offsets for precise location.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


def _generate_id() -> str:
    """Generate a short unique ID (UUID4 without hyphens, first 16 chars)."""
    return uuid.uuid4().hex[:16]


@dataclass
class ImageRef:
    """Reference to an image embedded within a document.

    Attributes:
        id: Globally unique image identifier (``{doc_hash}_{page}_{seq}``).
        path: File path where the image is stored.
        page: Page number in the source document (optional).
        text_offset: Start position of the ``[IMAGE: ...]`` placeholder in
            the document text (0-indexed).
        text_length: Character length of the placeholder string.
        position: Physical position info (e.g. PDF coordinates, pixel size).
    """

    id: str
    path: str
    page: int | None = None
    text_offset: int = 0
    text_length: int = 0
    position: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "page": self.page,
            "text_offset": self.text_offset,
            "text_length": self.text_length,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageRef:
        return cls(
            id=data["id"],
            path=data["path"],
            page=data.get("page"),
            text_offset=data.get("text_offset", 0),
            text_length=data.get("text_length", 0),
            position=data.get("position", {}),
        )


@dataclass
class Document:
    """A raw document loaded from a file.

    Attributes:
        id: Unique document identifier.
        text: Full document text (may contain ``[IMAGE: {id}]`` placeholders).
        metadata: Must include ``source_path``; ``images`` is a list of
            :class:`ImageRef` dicts for multimodal documents.
    """

    id: str = field(default_factory=_generate_id)
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        return cls(
            id=data.get("id", _generate_id()),
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Chunk:
    """A text segment split from a Document.

    Attributes:
        id: Unique chunk identifier.
        text: Chunk text content.
        metadata: Must include ``source_path``; may carry other info.
        start_offset: Start character offset in the parent Document text.
        end_offset: End character offset (exclusive) in the parent Document.
        source_ref: Reference to parent document ID (optional).
    """

    id: str = field(default_factory=_generate_id)
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    start_offset: int = 0
    end_offset: int = 0
    source_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            id=data.get("id", _generate_id()),
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
            start_offset=data.get("start_offset", 0),
            end_offset=data.get("end_offset", 0),
            source_ref=data.get("source_ref", ""),
        )


@dataclass
class ChunkRecord:
    """A Chunk enriched with embedding vectors for storage/retrieval.

    Populated during ingestion (C8/C9) and consumed by retrieval (D).

    Attributes:
        id: Same as the parent Chunk ID.
        text: Chunk text content.
        metadata: Full metadata from the Chunk.
        dense_vector: Dense embedding (e.g. from OpenAI, Ollama).
        sparse_vector: Sparse embedding (e.g. BM25 term weights).
    """

    id: str = field(default_factory=_generate_id)
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_vector: list[float] | None = None
    sparse_vector: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
        }
        if self.dense_vector is not None:
            result["dense_vector"] = self.dense_vector
        if self.sparse_vector is not None:
            result["sparse_vector"] = self.sparse_vector
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkRecord:
        return cls(
            id=data.get("id", _generate_id()),
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
            dense_vector=data.get("dense_vector"),
            sparse_vector=data.get("sparse_vector"),
        )

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        *,
        dense_vector: list[float] | None = None,
        sparse_vector: dict[str, float] | None = None,
    ) -> ChunkRecord:
        """Create a ChunkRecord from a Chunk, optionally adding vectors."""
        return cls(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )


@dataclass
class ProcessedQuery:
    """A query after preprocessing (keyword extraction, filters, etc.).

    Produced by :class:`QueryProcessor` and consumed by the retrieval pipeline.

    Attributes:
        original_query: The raw user query string.
        keywords: Extracted keywords for sparse retrieval (BM25).
        filters: Generic metadata filters (e.g. collection, doc_type).
    """

    original_query: str = ""
    keywords: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "keywords": self.keywords,
            "filters": self.filters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessedQuery:
        return cls(
            original_query=data.get("original_query", ""),
            keywords=data.get("keywords", []),
            filters=data.get("filters", {}),
        )
