"""Unit tests for DocumentManager (G2)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ingestion.document_manager import (
    CollectionStats,
    DeleteResult,
    DocumentInfo,
    DocumentManager,
)
from libs.vector_store.base_vector_store import BaseVectorStore


# ---------------------------------------------------------------------------
# Helpers — lightweight fakes
# ---------------------------------------------------------------------------


class FakeVectorStore(BaseVectorStore):
    """In-memory vector store backed by a list of dicts."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def upsert(
        self,
        records: list[Any],
        *,
        collection: str = "default",
        trace: Any | None = None,
    ) -> int:
        for r in records:
            self._records.append(
                {"id": r.id, "text": r.text, "metadata": r.metadata}
            )
        return len(records)

    def query(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        collection: str = "default",
        trace: Any | None = None,
    ) -> list[Any]:
        return []

    def get_by_ids(
        self,
        ids: list[str],
        *,
        collection: str = "default",
    ) -> list[dict[str, Any]]:
        return [r for r in self._records if r["id"] in ids]

    def delete_by_metadata(
        self,
        filter: dict[str, Any],
        *,
        collection: str = "default",
    ) -> int:
        key, val = next(iter(filter.items()))
        before = len(self._records)
        self._records = [r for r in self._records if r["metadata"].get(key) != val]
        return before - len(self._records)

    def get_collection_stats(
        self,
        collection: str = "default",
    ) -> dict[str, Any]:
        return {"name": collection, "doc_count": len(self._records)}

    def get_all(
        self,
        *,
        collection: str = "default",
    ) -> list[dict[str, Any]]:
        return list(self._records)

    # test helper
    def add_record(
        self,
        doc_id: str,
        source: str,
        doc_hash: str = "",
        text: str = "chunk text",
    ) -> None:
        self._records.append(
            {
                "id": doc_id,
                "text": text,
                "metadata": {"source_path": source, "doc_hash": doc_hash},
            }
        )


class FakeBM25Indexer:
    def __init__(self) -> None:
        self.removed_ids: list[str] = []
        self.saved_collections: list[str] = []

    def remove_chunks(self, chunk_ids: list[str]) -> int:
        self.removed_ids.extend(chunk_ids)
        return len(chunk_ids)

    def save(self, collection: str) -> None:
        self.saved_collections.append(collection)


class FakeImageStorage:
    def __init__(self) -> None:
        self._images: list[dict[str, Any]] = []

    def add_image(self, doc_hash: str, collection: str) -> None:
        self._images.append({"doc_hash": doc_hash, "collection": collection})

    def get_by_doc_hash(self, doc_hash: str) -> list[dict[str, Any]]:
        return [i for i in self._images if i["doc_hash"] == doc_hash]

    def get_by_collection(self, collection: str) -> list[dict[str, Any]]:
        return [i for i in self._images if i["collection"] == collection]

    def delete_by_doc_hash(self, doc_hash: str, collection: str) -> int:
        before = len(self._images)
        self._images = [
            i
            for i in self._images
            if not (i["doc_hash"] == doc_hash and i["collection"] == collection)
        ]
        return before - len(self._images)


class FakeIntegrity:
    def __init__(self) -> None:
        self._records: dict[str, bool] = {}

    def mark_success(self, file_hash: str, **kw: Any) -> None:
        self._records[file_hash] = True

    def remove_record(self, file_hash: str) -> bool:
        return self._records.pop(file_hash, None) is not None

    def list_processed(self) -> list[dict[str, Any]]:
        return [{"file_hash": h} for h in self._records]

    # unused stubs
    def should_skip(self, file_hash: str) -> bool:
        return file_hash in self._records

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        pass


def _make_manager() -> tuple[DocumentManager, FakeVectorStore, FakeBM25Indexer, FakeImageStorage, FakeIntegrity]:
    vs = FakeVectorStore()
    bm25 = FakeBM25Indexer()
    img = FakeImageStorage()
    integ = FakeIntegrity()
    mgr = DocumentManager(vs, bm25, img, integ)
    return mgr, vs, bm25, img, integ


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------


class TestListDocuments:

    def test_empty_collection(self) -> None:
        mgr, *_ = _make_manager()
        assert mgr.list_documents("default") == []

    def test_single_document(self) -> None:
        mgr, vs, _, img, _ = _make_manager()
        vs.add_record("c1", "doc.pdf", "hash1")
        img.add_image("hash1", "default")

        docs = mgr.list_documents("default")
        assert len(docs) == 1
        assert docs[0].source_path == "doc.pdf"
        assert docs[0].chunk_count == 1
        assert docs[0].image_count == 1
        assert docs[0].doc_hash == "hash1"

    def test_multiple_chunks_grouped(self) -> None:
        mgr, vs, _, _, _ = _make_manager()
        vs.add_record("c1", "doc.pdf", "hash1")
        vs.add_record("c2", "doc.pdf", "hash1")
        vs.add_record("c3", "other.pdf", "hash2")

        docs = mgr.list_documents("default")
        assert len(docs) == 2
        by_source = {d.source_path: d for d in docs}
        assert by_source["doc.pdf"].chunk_count == 2
        assert by_source["other.pdf"].chunk_count == 1

    def test_no_images(self) -> None:
        mgr, vs, _, _, _ = _make_manager()
        vs.add_record("c1", "doc.pdf", "hash1")

        docs = mgr.list_documents("default")
        assert docs[0].image_count == 0

    def test_record_without_source_path_skipped(self) -> None:
        mgr, vs, _, _, _ = _make_manager()
        vs._records.append({"id": "x", "text": "t", "metadata": {}})

        docs = mgr.list_documents("default")
        assert docs == []


# ---------------------------------------------------------------------------
# delete_document
# ---------------------------------------------------------------------------


class TestDeleteDocument:

    def test_delete_existing_document(self) -> None:
        mgr, vs, bm25, img, integ = _make_manager()
        vs.add_record("c1", "doc.pdf", "hash1")
        vs.add_record("c2", "doc.pdf", "hash1")
        img.add_image("hash1", "default")
        integ.mark_success("hash1", file_path="doc.pdf")

        result = mgr.delete_document("doc.pdf", "default")
        assert isinstance(result, DeleteResult)
        assert result.chunks_deleted == 2
        assert result.bm25_postings_removed == 2
        assert result.images_deleted == 1
        assert result.integrity_removed is True

    def test_chunks_removed_from_store(self) -> None:
        mgr, vs, *_ = _make_manager()
        vs.add_record("c1", "doc.pdf", "hash1")
        mgr.delete_document("doc.pdf", "default")
        assert len(vs._records) == 0

    def test_bm25_chunks_removed(self) -> None:
        mgr, vs, bm25, *_ = _make_manager()
        vs.add_record("c1", "doc.pdf", "hash1")
        mgr.delete_document("doc.pdf", "default")
        assert "c1" in bm25.removed_ids

    def test_bm25_saved_after_delete(self) -> None:
        mgr, vs, bm25, *_ = _make_manager()
        vs.add_record("c1", "doc.pdf", "hash1")
        mgr.delete_document("doc.pdf", "default")
        assert "default" in bm25.saved_collections

    def test_delete_nonexistent_document(self) -> None:
        mgr, vs, *_ = _make_manager()
        result = mgr.delete_document("missing.pdf", "default")
        assert result.chunks_deleted == 0
        assert result.bm25_postings_removed == 0
        assert result.images_deleted == 0
        assert result.integrity_removed is False

    def test_delete_only_removes_matching_source(self) -> None:
        mgr, vs, bm25, *_ = _make_manager()
        vs.add_record("c1", "doc_a.pdf", "hash1")
        vs.add_record("c2", "doc_b.pdf", "hash2")

        mgr.delete_document("doc_a.pdf", "default")
        assert len(vs._records) == 1
        assert vs._records[0]["metadata"]["source_path"] == "doc_b.pdf"

    def test_delete_no_doc_hash_skips_image_and_integrity(self) -> None:
        mgr, vs, bm25, img, integ = _make_manager()
        vs.add_record("c1", "doc.pdf", "")

        result = mgr.delete_document("doc.pdf", "default")
        assert result.images_deleted == 0
        assert result.integrity_removed is False


# ---------------------------------------------------------------------------
# get_collection_stats
# ---------------------------------------------------------------------------


class TestGetCollectionStats:

    def test_empty_collection(self) -> None:
        mgr, *_ = _make_manager()
        stats = mgr.get_collection_stats("default")
        assert isinstance(stats, CollectionStats)
        assert stats.document_count == 0
        assert stats.chunk_count == 0
        assert stats.image_count == 0

    def test_stats_with_data(self) -> None:
        mgr, vs, _, img, _ = _make_manager()
        vs.add_record("c1", "a.pdf", "h1")
        vs.add_record("c2", "a.pdf", "h1")
        vs.add_record("c3", "b.pdf", "h2")
        img.add_image("h1", "default")
        img.add_image("h2", "default")

        stats = mgr.get_collection_stats("default")
        assert stats.name == "default"
        assert stats.document_count == 2
        assert stats.chunk_count == 3
        assert stats.image_count == 2

    def test_stats_name_matches_collection(self) -> None:
        mgr, *_ = _make_manager()
        stats = mgr.get_collection_stats("my_col")
        assert stats.name == "my_col"


# ---------------------------------------------------------------------------
# New methods on storage backends
# ---------------------------------------------------------------------------


class TestFileIntegrityNewMethods:
    """Verify remove_record and list_processed on the real SQLite backend."""

    def test_remove_record_existing(self, tmp_path: Any) -> None:
        from libs.loader.file_integrity import SQLiteIntegrityChecker

        db = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(str(db))
        checker.mark_success("hash1", "/tmp/a.pdf")
        assert checker.remove_record("hash1") is True
        assert checker.should_skip("hash1") is False

    def test_remove_record_nonexistent(self, tmp_path: Any) -> None:
        from libs.loader.file_integrity import SQLiteIntegrityChecker

        db = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(str(db))
        assert checker.remove_record("nope") is False

    def test_list_processed(self, tmp_path: Any) -> None:
        from libs.loader.file_integrity import SQLiteIntegrityChecker

        db = tmp_path / "test.db"
        checker = SQLiteIntegrityChecker(str(db))
        checker.mark_success("h1", "/a.pdf")
        checker.mark_success("h2", "/b.pdf")
        checker.mark_failed("h3", "error")

        records = checker.list_processed()
        assert len(records) == 2
        assert {r["file_hash"] for r in records} == {"h1", "h2"}


class TestBM25RemoveChunks:

    def test_remove_existing_chunks(self) -> None:
        from core.types import ChunkRecord
        from ingestion.storage.bm25_indexer import BM25Indexer

        indexer = BM25Indexer()
        records = [
            ChunkRecord(id="c1", text="hello world", metadata={"source_path": "a.pdf"},
                        sparse_vector={"hello": 1.0, "world": 1.0}),
            ChunkRecord(id="c2", text="hello python", metadata={"source_path": "b.pdf"},
                        sparse_vector={"hello": 1.0, "python": 1.0}),
        ]
        indexer.build(records)

        removed = indexer.remove_chunks(["c1"])
        assert removed >= 1

        # c1 should no longer appear in results
        results = indexer.query(["hello", "world"], top_k=10)
        ids = [r[0] for r in results]
        assert "c1" not in ids

    def test_remove_nonexistent_chunks(self) -> None:
        from ingestion.storage.bm25_indexer import BM25Indexer

        indexer = BM25Indexer()
        assert indexer.remove_chunks(["nonexistent"]) == 0

    def test_remove_all_chunks_empties_index(self) -> None:
        from core.types import ChunkRecord
        from ingestion.storage.bm25_indexer import BM25Indexer

        indexer = BM25Indexer()
        records = [
            ChunkRecord(id="c1", text="hello", metadata={},
                        sparse_vector={"hello": 1.0}),
        ]
        indexer.build(records)
        indexer.remove_chunks(["c1"])
        assert indexer.doc_count == 0
        assert indexer.term_count == 0


class TestImageStorageDeleteByDocHash:

    def test_delete_existing_images(self, tmp_path: Any) -> None:
        from ingestion.storage.image_storage import ImageStorage

        db = tmp_path / "img.db"
        storage = ImageStorage(base_dir=str(tmp_path / "imgs"), db_path=str(db))

        # Save a dummy image
        storage.save("img1", b"\x89PNG", "test.png", collection="col1", doc_hash="hash1")

        deleted = storage.delete_by_doc_hash("hash1", "col1")
        assert deleted == 1
        assert storage.get_path("img1") is None

    def test_delete_no_match(self, tmp_path: Any) -> None:
        from ingestion.storage.image_storage import ImageStorage

        db = tmp_path / "img.db"
        storage = ImageStorage(base_dir=str(tmp_path / "imgs"), db_path=str(db))
        assert storage.delete_by_doc_hash("nope", "col1") == 0

    def test_delete_only_matching_collection(self, tmp_path: Any) -> None:
        from ingestion.storage.image_storage import ImageStorage

        db = tmp_path / "img.db"
        storage = ImageStorage(base_dir=str(tmp_path / "imgs"), db_path=str(db))
        storage.save("img1", b"\x00", "a.png", collection="col1", doc_hash="h1")
        storage.save("img2", b"\x00", "b.png", collection="col2", doc_hash="h1")

        deleted = storage.delete_by_doc_hash("h1", "col1")
        assert deleted == 1
        assert storage.get_path("img1") is None
        assert storage.get_path("img2") is not None
