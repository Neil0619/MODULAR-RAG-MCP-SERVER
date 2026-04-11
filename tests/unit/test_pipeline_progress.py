"""Unit tests for Pipeline progress callback (F5)."""

from __future__ import annotations

from pathlib import Path
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
from core.types import Chunk, ChunkRecord, Document
from ingestion.pipeline import IngestionPipeline

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
SIMPLE_PDF = FIXTURES_DIR / "simple.pdf"


def _make_settings(tmp_path: Path) -> Settings:
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


class TestPipelineProgress:
    """Tests for on_progress callback in IngestionPipeline."""

    @pytest.fixture()
    def setup(self, tmp_path: Path) -> dict:
        settings = _make_settings(tmp_path)

        from ingestion.embedding.sparse_encoder import SparseEncoder

        real_sparse = SparseEncoder()
        dim = 8

        # Create a dummy file so Path.exists() passes
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 fake")

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

        with (
            patch("ingestion.pipeline.PdfLoader") as MockLoader,
            patch("ingestion.pipeline.BatchProcessor") as MockBatch,
        ):
            MockLoader.return_value.load.return_value = Document(
                text="Test paragraph with enough content to produce at least one chunk. " * 10,
                metadata={"source_path": "test.pdf", "page_count": 1},
            )
            MockBatch.side_effect = make_batch

            pipeline = IngestionPipeline(settings, collection="test")
            yield {
                "pipeline": pipeline,
                "settings": settings,
                "tmp_path": tmp_path,
                "dummy_pdf": dummy_pdf,
            }

    def test_callback_called_for_each_stage(self, setup: dict) -> None:
        pipeline: IngestionPipeline = setup["pipeline"]
        callback = MagicMock()

        pipeline.run(str(setup["dummy_pdf"]), force=True, on_progress=callback)

        assert callback.call_count == 5

        # Check stage names in order
        stage_names = [call.args[0] for call in callback.call_args_list]
        assert stage_names == ["load", "split", "transform", "encode", "store"]

    def test_callback_current_increments(self, setup: dict) -> None:
        pipeline: IngestionPipeline = setup["pipeline"]
        callback = MagicMock()

        pipeline.run(str(setup["dummy_pdf"]), force=True, on_progress=callback)

        currents = [call.args[1] for call in callback.call_args_list]
        assert currents == [1, 2, 3, 4, 5]

    def test_callback_total_is_constant(self, setup: dict) -> None:
        pipeline: IngestionPipeline = setup["pipeline"]
        callback = MagicMock()

        pipeline.run(str(setup["dummy_pdf"]), force=True, on_progress=callback)

        totals = [call.args[2] for call in callback.call_args_list]
        assert all(t == 5 for t in totals)

    def test_no_callback_when_none(self, setup: dict) -> None:
        """Passing on_progress=None (default) should not break anything."""
        pipeline: IngestionPipeline = setup["pipeline"]
        summary = pipeline.run(str(setup["dummy_pdf"]), force=True)

        assert "stages" in summary
        assert len(summary["stages"]) >= 3  # load, split, at minimum

    def test_callback_signature(self, setup: dict) -> None:
        """Each call should be (stage_name: str, current: int, total: int)."""
        pipeline: IngestionPipeline = setup["pipeline"]
        callback = MagicMock()

        pipeline.run(str(setup["dummy_pdf"]), force=True, on_progress=callback)

        for call in callback.call_args_list:
            args = call.args
            assert len(args) == 3
            assert isinstance(args[0], str)
            assert isinstance(args[1], int)
            assert isinstance(args[2], int)
            assert 1 <= args[1] <= args[2]

    def test_no_callback_on_skip(self, setup: dict) -> None:
        """Skipped files (integrity check) should not trigger callback."""
        pipeline: IngestionPipeline = setup["pipeline"]
        callback = MagicMock()

        # Mock integrity checker to skip
        with patch("ingestion.pipeline.SQLiteIntegrityChecker") as MockChecker:
            MockChecker.return_value.compute_sha256.return_value = "abc123"
            MockChecker.return_value.should_skip.return_value = True
            pipeline.run(str(setup["dummy_pdf"]), force=False, on_progress=callback)

        callback.assert_not_called()
