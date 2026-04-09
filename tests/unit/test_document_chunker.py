"""Unit tests for DocumentChunker (C4)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.settings import (
    EmbeddingSettings,
    LLMSettings,
    RetrievalSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)
from core.types import Chunk, Document
from ingestion.chunking.document_chunker import DocumentChunker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**splitter_overrides: Any) -> Settings:
    spl = SplitterSettings(provider="recursive", **splitter_overrides)
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
        splitter=spl,
    )


class FakeSplitter:
    """A controlled splitter for testing."""

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings

    def split_text(self, text: str, **kwargs: Any) -> list[str]:
        # Simple split on double newline
        return [p.strip() for p in text.split("\n\n") if p.strip()]


def _make_chunker() -> DocumentChunker:
    """Create a DocumentChunker with FakeSplitter."""
    with patch(
        "ingestion.chunking.document_chunker.SplitterFactory"
    ) as mock_factory:
        mock_factory.create.return_value = FakeSplitter()
        return DocumentChunker(_make_settings())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDocumentChunkerBasic:
    def test_splits_document_into_chunks(self) -> None:
        chunker = _make_chunker()
        doc = Document(
            id="doc1",
            text="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
            metadata={"source_path": "/test.pdf", "doc_type": "pdf"},
        )
        chunks = chunker.split_document(doc)
        assert len(chunks) == 3
        assert chunks[0].text == "First paragraph."
        assert chunks[1].text == "Second paragraph."
        assert chunks[2].text == "Third paragraph."

    def test_empty_document_returns_empty(self) -> None:
        chunker = _make_chunker()
        doc = Document(id="doc1", text="", metadata={})
        assert chunker.split_document(doc) == []

    def test_whitespace_only_document_returns_empty(self) -> None:
        chunker = _make_chunker()
        doc = Document(id="doc1", text="   \n\n  \t  ", metadata={})
        assert chunker.split_document(doc) == []


class TestChunkIds:
    def test_ids_are_unique(self) -> None:
        chunker = _make_chunker()
        doc = Document(
            id="doc1",
            text="Para one.\n\nPara two.\n\nPara three.",
            metadata={"source_path": "/test.pdf"},
        )
        chunks = chunker.split_document(doc)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_ids_are_deterministic(self) -> None:
        chunker = _make_chunker()
        doc = Document(
            id="doc1",
            text="Para one.\n\nPara two.",
            metadata={"source_path": "/test.pdf"},
        )
        ids1 = [c.id for c in chunker.split_document(doc)]
        ids2 = [c.id for c in chunker.split_document(doc)]
        assert ids1 == ids2

    def test_id_format(self) -> None:
        chunker = _make_chunker()
        doc = Document(
            id="abc123",
            text="Some text.\n\nMore text.",
            metadata={},
        )
        chunks = chunker.split_document(doc)
        # Format: {doc_id}_{index:04d}_{hash_8chars}
        for i, chunk in enumerate(chunks):
            assert chunk.id.startswith(f"abc123_{i:04d}_")
            parts = chunk.id.split("_")
            # doc_id=abc123, index=0000, hash=8chars → 3 parts after split
            # Actually: "abc123_0000_xxxxxxxx" → split by _ gives ["abc123", "0000", "xxxxxxxx"]
            assert len(parts[-1]) == 8


class TestMetadataInheritance:
    def test_metadata_inherits_source_path(self) -> None:
        chunker = _make_chunker()
        doc = Document(
            id="doc1",
            text="Hello\n\nWorld",
            metadata={"source_path": "/data/file.pdf", "doc_type": "pdf"},
        )
        chunks = chunker.split_document(doc)
        for chunk in chunks:
            assert chunk.metadata["source_path"] == "/data/file.pdf"
            assert chunk.metadata["doc_type"] == "pdf"

    def test_chunk_index_set(self) -> None:
        chunker = _make_chunker()
        doc = Document(
            id="doc1",
            text="A\n\nB\n\nC",
            metadata={"source_path": "/test.pdf"},
        )
        chunks = chunker.split_document(doc)
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[1].metadata["chunk_index"] == 1
        assert chunks[2].metadata["chunk_index"] == 2

    def test_doc_images_not_in_chunk_metadata_by_default(self) -> None:
        chunker = _make_chunker()
        doc = Document(
            id="doc1",
            text="Just text\n\nNo images here",
            metadata={
                "source_path": "/test.pdf",
                "images": [{"id": "img1", "path": "/img.png"}],
            },
        )
        chunks = chunker.split_document(doc)
        for chunk in chunks:
            # No [IMAGE: ...] placeholders, so no images field
            assert "images" not in chunk.metadata
            assert "image_refs" not in chunk.metadata


class TestImageDistribution:
    def test_image_refs_distributed_to_correct_chunk(self) -> None:
        chunker = _make_chunker()
        images = [
            {"id": "img_0_0", "path": "/img0.png", "page": 0},
            {"id": "img_0_1", "path": "/img1.png", "page": 0},
        ]
        doc = Document(
            id="doc1",
            text="First chunk\n\nSecond [IMAGE: img_0_0] chunk\n\nThird chunk [IMAGE: img_0_1] here",
            metadata={"source_path": "/test.pdf", "images": images},
        )
        chunks = chunker.split_document(doc)

        # Chunk 0: no images
        assert "images" not in chunks[0].metadata
        assert "image_refs" not in chunks[0].metadata

        # Chunk 1: has img_0_0
        assert chunks[1].metadata["image_refs"] == ["img_0_0"]
        assert len(chunks[1].metadata["images"]) == 1
        assert chunks[1].metadata["images"][0]["id"] == "img_0_0"

        # Chunk 2: has img_0_1
        assert chunks[2].metadata["image_refs"] == ["img_0_1"]
        assert chunks[2].metadata["images"][0]["id"] == "img_0_1"

    def test_multiple_images_in_one_chunk(self) -> None:
        chunker = _make_chunker()
        images = [
            {"id": "img_a", "path": "/a.png"},
            {"id": "img_b", "path": "/b.png"},
        ]
        doc = Document(
            id="doc1",
            text="Chunk with [IMAGE: img_a] and [IMAGE: img_b]",
            metadata={"source_path": "/test.pdf", "images": images},
        )
        chunks = chunker.split_document(doc)
        assert chunks[0].metadata["image_refs"] == ["img_a", "img_b"]
        assert len(chunks[0].metadata["images"]) == 2


class TestSourceRef:
    def test_source_ref_points_to_document_id(self) -> None:
        chunker = _make_chunker()
        doc = Document(
            id="mydoc42",
            text="Hello\n\nWorld",
            metadata={"source_path": "/test.pdf"},
        )
        chunks = chunker.split_document(doc)
        for chunk in chunks:
            assert chunk.source_ref == "mydoc42"


class TestOffsets:
    def test_offsets_match_text_positions(self) -> None:
        chunker = _make_chunker()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        doc = Document(id="doc1", text=text, metadata={})
        chunks = chunker.split_document(doc)
        for chunk in chunks:
            assert doc.text[chunk.start_offset : chunk.end_offset] == chunk.text

    def test_offsets_are_ordered(self) -> None:
        chunker = _make_chunker()
        doc = Document(
            id="doc1",
            text="Alpha\n\nBeta\n\nGamma",
            metadata={},
        )
        chunks = chunker.split_document(doc)
        for i in range(len(chunks) - 1):
            assert chunks[i].start_offset < chunks[i + 1].start_offset
