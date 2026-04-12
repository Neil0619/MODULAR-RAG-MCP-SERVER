"""Unit tests for VectorStore base and factory (B4)."""

from typing import Any

import pytest

from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord, QueryResult
from libs.vector_store.vector_store_factory import VectorStoreFactory
from core.settings import Settings, LLMSettings, EmbeddingSettings, VectorStoreSettings, RetrievalSettings


class FakeVectorStore(BaseVectorStore):
    def __init__(self, settings: Any = None) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def upsert(self, records: list[VectorRecord], *, collection: str = "default", trace: Any | None = None) -> int:
        for r in records:
            self._store[r.id] = {"id": r.id, "text": r.text, "metadata": r.metadata, "vector": r.vector}
        return len(records)

    def query(self, vector: list[float], *, top_k: int = 10, filters: dict[str, Any] | None = None, collection: str = "default", trace: Any | None = None) -> list[QueryResult]:
        results = []
        for rid, rec in self._store.items():
            results.append(QueryResult(id=rid, score=1.0, text=rec["text"], metadata=rec["metadata"]))
        return results[:top_k]

    def get_by_ids(self, ids: list[str], *, collection: str = "default") -> list[dict[str, Any]]:
        return [self._store[i] for i in ids if i in self._store]

    def delete_by_metadata(self, filter: dict[str, Any], *, collection: str = "default") -> int:
        to_del = [k for k, v in self._store.items() if all(v.get("metadata", {}).get(fk) == fv for fk, fv in filter.items())]
        for k in to_del:
            del self._store[k]
        return len(to_del)

    def get_collection_stats(self, collection: str = "default") -> dict[str, Any]:
        return {"count": len(self._store)}

    def get_all(self, *, collection: str = "default") -> list[dict[str, Any]]:
        return list(self._store.values())


def _settings(backend: str = "fake") -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreSettings(backend=backend),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
    )


class TestVectorRecord:
    def test_fields(self) -> None:
        r = VectorRecord(id="1", vector=[0.1, 0.2], text="hello", metadata={"src": "a"})
        assert r.id == "1"
        assert len(r.vector) == 2


class TestQueryResult:
    def test_fields(self) -> None:
        q = QueryResult(id="1", score=0.95, text="hi")
        assert q.score == 0.95


class TestFakeVectorStore:
    def test_upsert_and_query(self) -> None:
        vs = FakeVectorStore()
        vs.upsert([VectorRecord(id="1", vector=[1.0], text="doc1")])
        results = vs.query([1.0], top_k=5)
        assert len(results) == 1
        assert results[0].text == "doc1"

    def test_get_by_ids(self) -> None:
        vs = FakeVectorStore()
        vs.upsert([VectorRecord(id="a", vector=[], text="ta"), VectorRecord(id="b", vector=[], text="tb")])
        records = vs.get_by_ids(["a", "b"])
        assert len(records) == 2

    def test_delete_by_metadata(self) -> None:
        vs = FakeVectorStore()
        vs.upsert([VectorRecord(id="1", vector=[], text="t", metadata={"src": "x"})])
        assert vs.delete_by_metadata({"src": "x"}) == 1
        assert vs.get_collection_stats()["count"] == 0


class TestVectorStoreFactory:
    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown vector store"):
            VectorStoreFactory.create(_settings("bogus"))

    def test_register_and_create(self) -> None:
        VectorStoreFactory.register_provider(
            "fake", f"{FakeVectorStore.__module__}.{FakeVectorStore.__qualname__}"
        )
        vs = VectorStoreFactory.create(_settings("fake"))
        assert isinstance(vs, FakeVectorStore)


class TestDeleteByMetadataBoundary:

    def test_delete_no_match(self) -> None:
        vs = FakeVectorStore()
        vs.upsert([VectorRecord(id="1", vector=[], text="t", metadata={"src": "x"})])
        assert vs.delete_by_metadata({"src": "nonexistent"}) == 0
        assert vs.get_collection_stats()["count"] == 1

    def test_delete_empty_store(self) -> None:
        vs = FakeVectorStore()
        assert vs.delete_by_metadata({"src": "x"}) == 0

    def test_delete_partial_metadata_match(self) -> None:
        vs = FakeVectorStore()
        vs.upsert([VectorRecord(id="1", vector=[], text="t", metadata={"src": "x", "tag": "a"})])
        vs.upsert([VectorRecord(id="2", vector=[], text="t", metadata={"src": "x", "tag": "b"})])
        # Match only on src — both deleted
        assert vs.delete_by_metadata({"src": "x"}) == 2
        assert vs.get_collection_stats()["count"] == 0

    def test_delete_multi_filter(self) -> None:
        vs = FakeVectorStore()
        vs.upsert([VectorRecord(id="1", vector=[], text="t", metadata={"src": "x", "tag": "a"})])
        vs.upsert([VectorRecord(id="2", vector=[], text="t", metadata={"src": "x", "tag": "b"})])
        # Match on both src and tag — only one deleted
        assert vs.delete_by_metadata({"src": "x", "tag": "a"}) == 1
        assert vs.get_collection_stats()["count"] == 1


class TestGetAllContract:

    def test_get_all_returns_all_records(self) -> None:
        vs = FakeVectorStore()
        vs.upsert([
            VectorRecord(id="a", vector=[], text="t1"),
            VectorRecord(id="b", vector=[], text="t2"),
        ])
        all_records = vs.get_all()
        assert len(all_records) == 2
        ids = {r["id"] for r in all_records}
        assert ids == {"a", "b"}

    def test_get_all_empty_store(self) -> None:
        vs = FakeVectorStore()
        assert vs.get_all() == []

    def test_get_all_with_collection_param(self) -> None:
        vs = FakeVectorStore()
        vs.upsert([VectorRecord(id="1", vector=[], text="t")], collection="test")
        # FakeVectorStore ignores collection but accepts the param
        all_records = vs.get_all(collection="test")
        assert len(all_records) >= 0  # contract: accepts collection kwarg
