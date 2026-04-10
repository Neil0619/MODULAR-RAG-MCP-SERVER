"""Unit tests for BM25Indexer (C11)."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.types import ChunkRecord
from ingestion.storage.bm25_indexer import BM25Indexer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_records() -> list[ChunkRecord]:
    """Create test records with sparse vectors."""
    return [
        ChunkRecord(
            id="doc1",
            text="machine learning algorithms",
            sparse_vector={"machine": 0.8, "learning": 0.6, "algorithms": 0.4},
        ),
        ChunkRecord(
            id="doc2",
            text="deep learning neural networks",
            sparse_vector={"deep": 0.7, "learning": 0.9, "neural": 0.5, "networks": 0.3},
        ),
        ChunkRecord(
            id="doc3",
            text="machine learning applications",
            sparse_vector={"machine": 0.6, "learning": 0.8, "applications": 0.7},
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBM25Build:
    def test_build_creates_index(self) -> None:
        indexer = BM25Indexer()
        indexer.build(_make_records())
        assert indexer.doc_count == 3
        assert indexer.term_count > 0

    def test_idf_computed(self) -> None:
        indexer = BM25Indexer()
        indexer.build(_make_records())
        terms = indexer.index["terms"]
        # "learning" appears in all 3 docs → lower IDF
        # "deep" appears in 1 doc → higher IDF
        assert "learning" in terms
        assert "deep" in terms
        assert terms["deep"]["idf"] > terms["learning"]["idf"]

    def test_postings_populated(self) -> None:
        indexer = BM25Indexer()
        indexer.build(_make_records())
        terms = indexer.index["terms"]
        # "machine" appears in doc1 and doc3
        assert len(terms["machine"]["postings"]) == 2
        # "deep" appears only in doc2
        assert len(terms["deep"]["postings"]) == 1

    def test_empty_records(self) -> None:
        indexer = BM25Indexer()
        indexer.build([])
        assert indexer.doc_count == 0
        assert indexer.term_count == 0

    def test_records_without_sparse(self) -> None:
        indexer = BM25Indexer()
        records = [
            ChunkRecord(id="d1", text="hello"),
            ChunkRecord(id="d2", text="world"),
        ]
        indexer.build(records)
        assert indexer.doc_count == 2
        assert indexer.term_count == 0


class TestBM25Query:
    def test_query_returns_results(self) -> None:
        indexer = BM25Indexer()
        indexer.build(_make_records())
        results = indexer.query(["machine", "learning"])
        assert len(results) > 0

    def test_query_returns_tuples(self) -> None:
        indexer = BM25Indexer()
        indexer.build(_make_records())
        results = indexer.query(["machine"])
        for chunk_id, score in results:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)

    def test_query_top_k(self) -> None:
        indexer = BM25Indexer()
        indexer.build(_make_records())
        results = indexer.query(["learning"], top_k=2)
        assert len(results) <= 2

    def test_query_rare_term_ranks_higher(self) -> None:
        indexer = BM25Indexer()
        indexer.build(_make_records())
        results = indexer.query(["deep", "learning"])
        # "deep" is rare (1 doc), should boost doc2
        ids = [r[0] for r in results]
        assert "doc2" in ids

    def test_query_unknown_term_returns_empty(self) -> None:
        indexer = BM25Indexer()
        indexer.build(_make_records())
        results = indexer.query(["nonexistent_term"])
        assert len(results) == 0


class TestBM25Persistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        indexer = BM25Indexer(index_dir=str(tmp_path / "bm25"))
        indexer.build(_make_records())
        path = indexer.save("test")
        assert path.exists()

        # Load into new indexer
        indexer2 = BM25Indexer(index_dir=str(tmp_path / "bm25"))
        indexer2.load("test")
        assert indexer2.doc_count == 3
        assert indexer2.term_count == indexer.term_count

    def test_query_after_load_matches(self, tmp_path: Path) -> None:
        indexer = BM25Indexer(index_dir=str(tmp_path / "bm25"))
        indexer.build(_make_records())
        results_before = indexer.query(["machine"], top_k=5)

        indexer.save("test")

        indexer2 = BM25Indexer(index_dir=str(tmp_path / "bm25"))
        indexer2.load("test")
        results_after = indexer2.query(["machine"], top_k=5)

        ids_before = [r[0] for r in results_before]
        ids_after = [r[0] for r in results_after]
        assert ids_before == ids_after

    def test_load_nonexistent_no_error(self, tmp_path: Path) -> None:
        indexer = BM25Indexer(index_dir=str(tmp_path / "bm25"))
        indexer.load("nonexistent")
        assert indexer.doc_count == 0

    def test_rebuild_replaces_index(self) -> None:
        indexer = BM25Indexer()
        records1 = [
            ChunkRecord(id="d1", text="alpha", sparse_vector={"alpha": 1.0}),
        ]
        indexer.build(records1)
        assert indexer.term_count == 1

        records2 = [
            ChunkRecord(id="d1", text="alpha", sparse_vector={"alpha": 1.0}),
            ChunkRecord(id="d2", text="beta gamma", sparse_vector={"beta": 0.8, "gamma": 0.6}),
        ]
        indexer.build(records2)
        assert indexer.term_count == 3  # alpha, beta, gamma
        assert indexer.doc_count == 2
