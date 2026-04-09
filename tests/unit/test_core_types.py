"""Unit tests for core data types (C1)."""

from __future__ import annotations

import json

import pytest

from core.types import Chunk, ChunkRecord, Document, ImageRef


# ---------------------------------------------------------------------------
# ImageRef
# ---------------------------------------------------------------------------


class TestImageRef:
    def test_fields(self) -> None:
        ref = ImageRef(
            id="abc123_0_1",
            path="data/images/default/abc123_0_1.png",
            page=0,
            text_offset=42,
            text_length=20,
            position={"x": 0, "y": 100, "width": 500, "height": 300},
        )
        assert ref.id == "abc123_0_1"
        assert ref.page == 0
        assert ref.text_offset == 42

    def test_defaults(self) -> None:
        ref = ImageRef(id="img1", path="/tmp/img.png")
        assert ref.page is None
        assert ref.text_offset == 0
        assert ref.text_length == 0
        assert ref.position == {}

    def test_to_dict_roundtrip(self) -> None:
        ref = ImageRef(id="i1", path="/p.png", page=3, text_offset=10, text_length=15)
        d = ref.to_dict()
        restored = ImageRef.from_dict(d)
        assert restored.id == ref.id
        assert restored.path == ref.path
        assert restored.page == ref.page
        assert restored.text_offset == ref.text_offset
        assert restored.text_length == ref.text_length

    def test_from_dict_missing_optional(self) -> None:
        ref = ImageRef.from_dict({"id": "x", "path": "/y.png"})
        assert ref.page is None
        assert ref.text_offset == 0


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class TestDocument:
    def test_default_id_generated(self) -> None:
        doc = Document(text="hello")
        assert len(doc.id) == 16

    def test_unique_ids(self) -> None:
        ids = {Document().id for _ in range(100)}
        assert len(ids) == 100

    def test_text_and_metadata(self) -> None:
        doc = Document(
            text="Some content [IMAGE: img_0_1] more text",
            metadata={"source_path": "/data/file.pdf"},
        )
        assert "[IMAGE: img_0_1]" in doc.text
        assert doc.metadata["source_path"] == "/data/file.pdf"

    def test_to_dict_roundtrip(self) -> None:
        doc = Document(
            id="doc1",
            text="content",
            metadata={"source_path": "/a.pdf", "images": [{"id": "i1"}]},
        )
        d = doc.to_dict()
        restored = Document.from_dict(d)
        assert restored.id == "doc1"
        assert restored.text == "content"
        assert restored.metadata["source_path"] == "/a.pdf"

    def test_to_dict_json_serializable(self) -> None:
        doc = Document(
            text="hello",
            metadata={"source_path": "/test.pdf", "page": 1},
        )
        serialized = json.dumps(doc.to_dict())
        assert isinstance(serialized, str)

    def test_from_dict_missing_fields(self) -> None:
        doc = Document.from_dict({})
        assert doc.text == ""
        assert doc.metadata == {}

    def test_images_in_metadata(self) -> None:
        images = [
            ImageRef(
                id="abc_0_1",
                path="data/images/default/abc_0_1.png",
                page=0,
                text_offset=5,
                text_length=len("[IMAGE: abc_0_1]"),
            ).to_dict()
        ]
        doc = Document(
            text="Some [IMAGE: abc_0_1] text",
            metadata={"source_path": "/doc.pdf", "images": images},
        )
        assert len(doc.metadata["images"]) == 1
        assert doc.metadata["images"][0]["id"] == "abc_0_1"
        assert doc.metadata["images"][0]["text_offset"] == 5


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def test_default_id_generated(self) -> None:
        chunk = Chunk(text="chunk text")
        assert len(chunk.id) == 16

    def test_offsets(self) -> None:
        chunk = Chunk(
            text="hello world",
            start_offset=0,
            end_offset=11,
            source_ref="doc1",
        )
        assert chunk.end_offset - chunk.start_offset == len(chunk.text)

    def test_to_dict_roundtrip(self) -> None:
        chunk = Chunk(
            id="c1",
            text="some chunk",
            metadata={"source_path": "/a.pdf"},
            start_offset=50,
            end_offset=60,
            source_ref="doc1",
        )
        d = chunk.to_dict()
        restored = Chunk.from_dict(d)
        assert restored.id == "c1"
        assert restored.start_offset == 50
        assert restored.source_ref == "doc1"

    def test_to_dict_json_serializable(self) -> None:
        chunk = Chunk(text="text", metadata={"source_path": "/x.pdf"})
        serialized = json.dumps(chunk.to_dict())
        assert isinstance(serialized, str)

    def test_from_dict_missing_fields(self) -> None:
        chunk = Chunk.from_dict({"text": "hello"})
        assert chunk.text == "hello"
        assert chunk.source_ref == ""
        assert chunk.start_offset == 0


# ---------------------------------------------------------------------------
# ChunkRecord
# ---------------------------------------------------------------------------


class TestChunkRecord:
    def test_from_chunk(self) -> None:
        chunk = Chunk(
            id="c1",
            text="chunk text",
            metadata={"source_path": "/a.pdf"},
        )
        record = ChunkRecord.from_chunk(chunk, dense_vector=[0.1, 0.2, 0.3])
        assert record.id == "c1"
        assert record.text == "chunk text"
        assert record.dense_vector == [0.1, 0.2, 0.3]
        assert record.sparse_vector is None

    def test_from_chunk_with_sparse(self) -> None:
        chunk = Chunk(id="c2", text="text")
        sparse = {"hello": 0.5, "world": 0.3}
        record = ChunkRecord.from_chunk(chunk, sparse_vector=sparse)
        assert record.sparse_vector == sparse

    def test_to_dict_includes_vectors(self) -> None:
        record = ChunkRecord(
            id="r1",
            text="text",
            dense_vector=[1.0],
            sparse_vector={"foo": 0.5},
        )
        d = record.to_dict()
        assert d["dense_vector"] == [1.0]
        assert d["sparse_vector"] == {"foo": 0.5}

    def test_to_dict_omits_none_vectors(self) -> None:
        record = ChunkRecord(id="r1", text="text")
        d = record.to_dict()
        assert "dense_vector" not in d
        assert "sparse_vector" not in d

    def test_roundtrip(self) -> None:
        record = ChunkRecord(
            id="r1",
            text="text",
            metadata={"source_path": "/x.pdf"},
            dense_vector=[0.1],
        )
        d = record.to_dict()
        restored = ChunkRecord.from_dict(d)
        assert restored.id == "r1"
        assert restored.dense_vector == [0.1]

    def test_to_dict_json_serializable(self) -> None:
        record = ChunkRecord(
            text="text",
            dense_vector=[0.1, 0.2],
            sparse_vector={"word": 1.0},
        )
        serialized = json.dumps(record.to_dict())
        assert isinstance(serialized, str)
