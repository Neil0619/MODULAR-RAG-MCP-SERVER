"""Integration tests for IngestionPipeline (C14).

Tests the full pipeline: integrity → load → split → transform → encode → store.
Uses mock embedding / vector-store to avoid external API calls.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.settings import (
    EmbeddingSettings,
    IngestionSettings,
    LLMSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)
from core.types import Chunk, ChunkRecord
from ingestion.pipeline import IngestionPipeline, PipelineError
from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeVectorStore(BaseVectorStore):
    """In-memory fake vector store for integration tests."""

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def upsert(
        self,
        records: list[VectorRecord],
        *,
        collection: str = "default",
        trace: Any | None = None,
    ) -> int:
        for rec in records:
            self._records[rec.id] = rec
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
        return [
            {"id": rid, "text": self._records[rid].text, "metadata": self._records[rid].metadata}
            for rid in ids
            if rid in self._records
        ]

    def delete_by_metadata(
        self,
        filter: dict[str, Any],
        *,
        collection: str = "default",
    ) -> int:
        return 0

    def get_collection_stats(
        self,
        collection: str = "default",
    ) -> dict[str, Any]:
        return {"doc_count": len(self._records)}

    def get_all(
        self,
        *,
        collection: str = "default",
    ) -> list[dict[str, Any]]:
        return [
            {"id": rid, "text": rec.text, "metadata": rec.metadata}
            for rid, rec in self._records.items()
        ]


def _make_settings(tmp_path: Path) -> Settings:
    """Create minimal Settings for pipeline testing."""
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o", api_key="fake"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small", api_key="fake"),
        splitter=SplitterSettings(chunk_size=500, chunk_overlap=50),
        vector_store=VectorStoreSettings(backend="chroma", persist_path=str(tmp_path / "chroma")),
        ingestion=IngestionSettings(
            chunk_refiner=IngestionSettings.ChunkRefinerSettings(use_llm=False),
            metadata_enricher=IngestionSettings.MetadataEnricherSettings(use_llm=False),
            image_captioner=IngestionSettings.ImageCaptionerSettings(enabled=False),
        ),
    )


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
SIMPLE_PDF = FIXTURES_DIR / "simple.pdf"
WITH_IMAGES_PDF = FIXTURES_DIR / "with_images.pdf"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIngestionPipeline:
    """Integration tests for the full ingestion pipeline."""

    @pytest.fixture()
    def setup(self, tmp_path: Path) -> dict:
        """Set up pipeline with mocked external deps."""
        settings = _make_settings(tmp_path)
        fake_store = FakeVectorStore()

        # Mock DenseEncoder to return fixed vectors
        mock_dense = MagicMock()
        dim = 8

        def fake_encode(chunks, trace=None):
            return [[float(i) / dim for i in range(dim)] for _ in chunks]

        mock_dense.encode = fake_encode
        mock_dense.dimensions = dim

        with (
            patch("ingestion.pipeline.VectorUpserter", side_effect=lambda s, store=None: __import__("ingestion.storage.vector_upserter", fromlist=["VectorUpserter"]).VectorUpserter(s, store=fake_store)),
            patch("ingestion.pipeline.BatchProcessor") as MockBatch,
        ):
            # Make BatchProcessor use our mock encoder
            from ingestion.embedding.sparse_encoder import SparseEncoder

            real_sparse = SparseEncoder()

            def make_batch(settings, batch_size=32, dense_encoder=None, sparse_encoder=None):
                bp = MagicMock()
                bp.process = lambda chunks, trace=None: [
                    ChunkRecord.from_chunk(
                        c,
                        dense_vector=[float(i) / dim for i in range(dim)],
                        sparse_vector=real_sparse.encode([c])[0] if c.text.strip() else {},
                    )
                    for c in chunks
                ]
                return bp

            MockBatch.side_effect = make_batch

            pipeline = IngestionPipeline(settings, collection="test")
            yield {
                "pipeline": pipeline,
                "settings": settings,
                "fake_store": fake_store,
                "tmp_path": tmp_path,
            }

    def test_full_pipeline_simple_pdf(self, setup: dict) -> None:
        """Run full pipeline on simple.pdf and verify all stages."""
        pipeline: IngestionPipeline = setup["pipeline"]

        if not SIMPLE_PDF.exists():
            pytest.skip("simple.pdf fixture not found")

        summary = pipeline.run(str(SIMPLE_PDF), force=True)

        assert summary["collection"] == "test"
        assert "skipped" not in summary
        assert "stages" in summary

        # Load stage
        assert "load" in summary["stages"]
        assert summary["stages"]["load"]["text_length"] > 0

        # Split stage
        assert "split" in summary["stages"]
        assert summary["stages"]["split"]["chunk_count"] > 0

        # Transform stage
        assert "transform" in summary["stages"]

        # Encode stage
        assert "encode" in summary["stages"]
        assert summary["stages"]["encode"]["record_count"] > 0

        # Store stage
        assert "store" in summary["stages"]
        assert summary["stages"]["store"]["upserted"] > 0

    def test_integrity_skip(self, setup: dict, tmp_path: Path) -> None:
        """Second run without force should skip due to integrity check."""
        from libs.loader.file_integrity import SQLiteIntegrityChecker

        db_path = str(tmp_path / "integrity_test.db")

        if not SIMPLE_PDF.exists():
            pytest.skip("simple.pdf fixture not found")

        file_hash = SQLiteIntegrityChecker.compute_sha256(str(SIMPLE_PDF))

        # Not yet marked — should not skip
        checker1 = SQLiteIntegrityChecker(db_path=db_path)
        assert checker1.should_skip(file_hash) is False

        # Mark success
        checker1.mark_success(file_hash, str(SIMPLE_PDF), chunks=5)
        checker1.close()

        # Now should skip
        checker2 = SQLiteIntegrityChecker(db_path=db_path)
        assert checker2.should_skip(file_hash) is True
        checker2.close()

    def test_force_reruns(self, setup: dict) -> None:
        """Force flag bypasses integrity check."""
        pipeline: IngestionPipeline = setup["pipeline"]

        if not SIMPLE_PDF.exists():
            pytest.skip("simple.pdf fixture not found")

        s1 = pipeline.run(str(SIMPLE_PDF), force=True)
        s2 = pipeline.run(str(SIMPLE_PDF), force=True)

        assert "skipped" not in s1
        assert "skipped" not in s2

    def test_file_not_found(self, setup: dict) -> None:
        """Missing file raises PipelineError."""
        pipeline: IngestionPipeline = setup["pipeline"]

        with pytest.raises(PipelineError, match="File not found"):
            pipeline.run("/nonexistent/file.pdf")

    def test_no_chunks_returns_early(self, setup: dict) -> None:
        """Empty document (no extractable text) returns with skip_reason."""
        pipeline: IngestionPipeline = setup["pipeline"]

        # Create an empty-ish file that PdfLoader can open but yields no text
        # We'll mock the loader to return empty text
        from core.types import Document

        with patch("ingestion.pipeline.LoaderFactory") as MockFactory:
            mock_loader = MagicMock()
            mock_loader.load.return_value = Document(
                text="   ",  # whitespace-only
                metadata={"source_path": "test.pdf", "page_count": 1},
            )
            MockFactory.create_from_path.return_value = mock_loader

            summary = pipeline.run(str(SIMPLE_PDF), force=True)

        # Should have loaded but no chunks
        assert summary["stages"]["split"]["chunk_count"] == 0
        assert summary["stages"].get("skip_reason") == "no_chunks"

    def test_trace_context_populated(self, setup: dict) -> None:
        """Pipeline records stages in TraceContext with elapsed_ms."""
        pipeline: IngestionPipeline = setup["pipeline"]

        if not SIMPLE_PDF.exists():
            pytest.skip("simple.pdf fixture not found")

        pipeline.run(str(SIMPLE_PDF), force=True)

        trace = pipeline.trace
        assert trace.trace_type == "ingestion"

        # All 5 stages should be present
        stage_names = list(trace.stages.keys())
        for expected in ("load", "split", "transform", "encode", "store"):
            assert expected in stage_names, f"Missing stage: {expected}"

        # Each stage should have elapsed_ms and method
        for name, data in trace.stages.items():
            assert isinstance(data, dict), f"Stage {name} data is not a dict"
            assert "elapsed_ms" in data, f"Stage {name} missing elapsed_ms"

        # Check specific method fields
        assert trace.stages["load"]["method"] == "pdf"
        assert trace.stages["split"]["method"] == "recursive"
        assert trace.stages["encode"]["method"] == "dense+sparse"
        assert trace.stages["store"]["method"] == "chroma"

        # Trace should be finished
        assert trace.finished_at is not None

        # to_dict should be JSON-serialisable
        import json
        d = trace.to_dict()
        assert d["trace_type"] == "ingestion"
        json.dumps(d)  # no error
