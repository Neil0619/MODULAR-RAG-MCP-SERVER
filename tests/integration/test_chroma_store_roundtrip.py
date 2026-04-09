"""Integration tests for ChromaStore round-trip operations."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.settings import Settings, VectorStoreSettings
from libs.vector_store.base_vector_store import VectorRecord
from libs.vector_store.chroma_store import ChromaStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> ChromaStore:
    settings = Settings(
        vector_store=VectorStoreSettings(persist_path=str(tmp_path / "chroma_db"))
    )
    return ChromaStore(settings)


def _record(idx: int, dim: int = 4) -> VectorRecord:
    """Create a simple VectorRecord with a deterministic vector."""
    vector = [float(idx)] * dim
    return VectorRecord(
        id=f"doc-{idx}",
        vector=vector,
        text=f"Document number {idx}",
        metadata={"source": "test", "index": idx},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestChromaStoreRoundtrip:
    """Round-trip integration tests for ChromaStore."""

    def test_upsert_then_query_roundtrip(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        records = [_record(i) for i in range(5)]
        store.upsert(records, collection="test_col")

        # Query with the same vector as doc-2
        results = store.query(
            vector=[2.0, 2.0, 2.0, 2.0],
            top_k=3,
            collection="test_col",
        )

        assert len(results) == 3
        # The top result should be doc-2 (exact match)
        assert results[0].id == "doc-2"
        assert results[0].score > results[1].score

    def test_upsert_is_idempotent(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        records = [_record(0), _record(1)]
        store.upsert(records, collection="idem")

        # Upsert again with updated text
        updated = [
            VectorRecord(id="doc-0", vector=[0.0] * 4, text="Updated doc 0", metadata={"source": "test", "index": 0}),
        ]
        store.upsert(updated, collection="idem")

        stats = store.get_collection_stats("idem")
        assert stats["doc_count"] == 2  # not 3

        fetched = store.get_by_ids(["doc-0"], collection="idem")
        assert fetched[0]["text"] == "Updated doc 0"

    def test_get_by_ids_returns_correct_records(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        records = [_record(i) for i in range(3)]
        store.upsert(records, collection="getbyid")

        fetched = store.get_by_ids(["doc-0", "doc-2"], collection="getbyid")

        assert len(fetched) == 2
        ids = {r["id"] for r in fetched}
        assert ids == {"doc-0", "doc-2"}
        for r in fetched:
            assert "text" in r
            assert "metadata" in r

    def test_delete_by_metadata_removes_records(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        records = [
            VectorRecord(id=f"doc-{i}", vector=[float(i)] * 4, text=f"Doc {i}", metadata={"group": "A" if i < 3 else "B"})
            for i in range(5)
        ]
        store.upsert(records, collection="delmeta")

        deleted = store.delete_by_metadata({"group": "A"}, collection="delmeta")
        assert deleted == 3

        stats = store.get_collection_stats("delmeta")
        assert stats["doc_count"] == 2

    def test_collection_stats_are_accurate(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)

        stats_before = store.get_collection_stats("stats_col")
        assert stats_before["doc_count"] == 0

        records = [_record(i) for i in range(7)]
        store.upsert(records, collection="stats_col")

        stats_after = store.get_collection_stats("stats_col")
        assert stats_after["doc_count"] == 7
        assert stats_after["name"] == "stats_col"
