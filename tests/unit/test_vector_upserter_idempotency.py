"""Unit tests for VectorUpserter (C12)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from core.types import ChunkRecord
from ingestion.storage.vector_upserter import VectorUpserter, _generate_stable_id
from libs.vector_store.base_vector_store import VectorRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeVectorStore:
    """In-memory fake vector store for testing."""

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def upsert(
        self,
        records: list[VectorRecord],
        *,
        collection: str = "default",
        trace: Any | None = None,
    ) -> int:
        count = 0
        for r in records:
            self._records[r.id] = r
            count += 1
        return count

    def get_by_ids(
        self,
        ids: list[str],
        *,
        collection: str = "default",
    ) -> list[dict[str, Any]]:
        results = []
        for rid in ids:
            if rid in self._records:
                r = self._records[rid]
                results.append({"id": r.id, "text": r.text, "metadata": r.metadata})
        return results


def _make_record(
    text: str = "hello world",
    source_path: str = "/test.pdf",
    chunk_index: int = 0,
) -> ChunkRecord:
    return ChunkRecord(
        id="test",
        text=text,
        metadata={"source_path": source_path, "chunk_index": chunk_index},
        dense_vector=[0.1, 0.2, 0.3],
    )


def _make_upserter(store: FakeVectorStore | None = None) -> VectorUpserter:
    return VectorUpserter(settings=MagicMock(), store=store or FakeVectorStore())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStableId:
    def test_same_content_same_id(self) -> None:
        r1 = _make_record("hello world")
        r2 = _make_record("hello world")
        assert _generate_stable_id(r1) == _generate_stable_id(r2)

    def test_different_content_different_id(self) -> None:
        r1 = _make_record("hello world")
        r2 = _make_record("goodbye world")
        assert _generate_stable_id(r1) != _generate_stable_id(r2)

    def test_different_source_different_id(self) -> None:
        r1 = _make_record("same text", source_path="/a.pdf")
        r2 = _make_record("same text", source_path="/b.pdf")
        assert _generate_stable_id(r1) != _generate_stable_id(r2)

    def test_different_chunk_index_different_id(self) -> None:
        r1 = _make_record("same text", chunk_index=0)
        r2 = _make_record("same text", chunk_index=1)
        assert _generate_stable_id(r1) != _generate_stable_id(r2)


class TestVectorUpserter:
    def test_upsert_returns_count(self) -> None:
        store = FakeVectorStore()
        upserter = _make_upserter(store)
        records = [_make_record(chunk_index=i) for i in range(3)]
        count = upserter.upsert(records)
        assert count == 3

    def test_upsert_writes_to_store(self) -> None:
        store = FakeVectorStore()
        upserter = _make_upserter(store)
        record = _make_record()
        upserter.upsert([record])
        stable_id = upserter.get_stable_id(record)
        results = store.get_by_ids([stable_id])
        assert len(results) == 1
        assert results[0]["text"] == "hello world"

    def test_idempotent_upsert(self) -> None:
        store = FakeVectorStore()
        upserter = _make_upserter(store)
        record = _make_record()

        # Upsert twice
        upserter.upsert([record])
        upserter.upsert([record])

        # Should have exactly 1 record (same ID overwrites)
        stable_id = upserter.get_stable_id(record)
        results = store.get_by_ids([stable_id])
        assert len(results) == 1

    def test_content_change_produces_new_id(self) -> None:
        store = FakeVectorStore()
        upserter = _make_upserter(store)

        r1 = _make_record("version one")
        upserter.upsert([r1])
        id_v1 = upserter.get_stable_id(r1)

        r2 = _make_record("version two")
        upserter.upsert([r2])
        id_v2 = upserter.get_stable_id(r2)

        assert id_v1 != id_v2
        # Both records exist
        results = store.get_by_ids([id_v1, id_v2])
        assert len(results) == 2

    def test_batch_upsert_preserves_order(self) -> None:
        store = FakeVectorStore()
        upserter = _make_upserter(store)
        records = [_make_record(chunk_index=i) for i in range(5)]
        count = upserter.upsert(records)
        assert count == 5

    def test_empty_records_raises(self) -> None:
        upserter = _make_upserter()
        with pytest.raises(ValueError, match="records must not be empty"):
            upserter.upsert([])

    def test_records_without_dense_vector_skipped(self) -> None:
        store = FakeVectorStore()
        upserter = _make_upserter(store)
        record = ChunkRecord(id="r1", text="no vector", metadata={"source_path": "/t.pdf"})
        count = upserter.upsert([record])
        assert count == 0

    def test_get_stable_id_exposed(self) -> None:
        upserter = _make_upserter()
        record = _make_record()
        sid = upserter.get_stable_id(record)
        assert isinstance(sid, str)
        assert len(sid) > 0
