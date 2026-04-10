"""E2E tests for the ingest.py script (C15).

Verifies the CLI entry point works with mocked external deps.
"""

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
from core.types import ChunkRecord

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
SIMPLE_PDF = FIXTURES_DIR / "simple.pdf"

_DIM = 8
_DUMMY_VEC = [float(i) / _DIM for i in range(_DIM)]


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


def _mock_batch_processor():
    """Create a mock BatchProcessor that returns dummy vectors."""
    from ingestion.embedding.sparse_encoder import SparseEncoder

    sparse = SparseEncoder()
    mock = MagicMock()

    def fake_process(chunks, trace=None):
        return [
            ChunkRecord.from_chunk(
                c,
                dense_vector=_DUMMY_VEC,
                sparse_vector=sparse.encode([c])[0] if c.text.strip() else {},
            )
            for c in chunks
        ]

    mock.process = fake_process
    return mock


class TestIngestScript:
    """E2E tests for scripts/ingest.py."""

    def test_main_single_file(self, tmp_path: Path) -> None:
        """Process a single PDF file via main()."""
        if not SIMPLE_PDF.exists():
            pytest.skip("simple.pdf fixture not found")

        from scripts.ingest import main as ingest_main

        settings = _make_settings(tmp_path)
        mock_bp = _mock_batch_processor()

        with (
            patch("scripts.ingest.load_settings", return_value=settings),
            patch("ingestion.pipeline.BatchProcessor", return_value=mock_bp),
        ):
            ret = ingest_main([
                "--path", str(SIMPLE_PDF),
                "--collection", "test",
                "--config", "fake",
                "--force",
            ])

        assert ret == 0

    def test_main_directory(self, tmp_path: Path) -> None:
        """Process a directory of PDF files."""
        if not SIMPLE_PDF.exists():
            pytest.skip("simple.pdf fixture not found")

        from scripts.ingest import main as ingest_main

        settings = _make_settings(tmp_path)
        mock_bp = _mock_batch_processor()

        with (
            patch("scripts.ingest.load_settings", return_value=settings),
            patch("ingestion.pipeline.BatchProcessor", return_value=mock_bp),
        ):
            ret = ingest_main([
                "--path", str(FIXTURES_DIR),
                "--collection", "test",
                "--config", "fake",
                "--force",
            ])

        assert ret == 0

    def test_main_no_pdfs(self, tmp_path: Path) -> None:
        """Empty directory returns error code."""
        from scripts.ingest import main as ingest_main

        settings = _make_settings(tmp_path)
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch("scripts.ingest.load_settings", return_value=settings):
            ret = ingest_main([
                "--path", str(empty_dir),
                "--collection", "test",
                "--config", "fake",
            ])

        assert ret == 1

    def test_main_nonexistent_path(self, tmp_path: Path) -> None:
        """Nonexistent path returns error code."""
        from scripts.ingest import main as ingest_main

        settings = _make_settings(tmp_path)

        with patch("scripts.ingest.load_settings", return_value=settings):
            ret = ingest_main([
                "--path", "/nonexistent/path.pdf",
                "--collection", "test",
                "--config", "fake",
            ])

        assert ret == 1

    def test_force_reruns_file(self, tmp_path: Path) -> None:
        """Force flag re-processes an already-ingested file."""
        if not SIMPLE_PDF.exists():
            pytest.skip("simple.pdf fixture not found")

        from scripts.ingest import main as ingest_main

        settings = _make_settings(tmp_path)

        with (
            patch("scripts.ingest.load_settings", return_value=settings),
            patch("ingestion.pipeline.BatchProcessor", return_value=_mock_batch_processor()),
        ):
            ret1 = ingest_main([
                "--path", str(SIMPLE_PDF),
                "--collection", "test",
                "--config", "fake",
                "--force",
            ])
            ret2 = ingest_main([
                "--path", str(SIMPLE_PDF),
                "--collection", "test",
                "--config", "fake",
                "--force",
            ])

        assert ret1 == 0
        assert ret2 == 0

    def test_skip_already_ingested(self, tmp_path: Path) -> None:
        """Second run without --force skips already-ingested file."""
        if not SIMPLE_PDF.exists():
            pytest.skip("simple.pdf fixture not found")

        from scripts.ingest import main as ingest_main

        settings = _make_settings(tmp_path)

        with (
            patch("scripts.ingest.load_settings", return_value=settings),
            patch("ingestion.pipeline.BatchProcessor", return_value=_mock_batch_processor()),
        ):
            # First run with force
            ret1 = ingest_main([
                "--path", str(SIMPLE_PDF),
                "--collection", "test",
                "--config", "fake",
                "--force",
            ])
            # Second run without force (should skip but still succeed)
            ret2 = ingest_main([
                "--path", str(SIMPLE_PDF),
                "--collection", "test",
                "--config", "fake",
            ])

        assert ret1 == 0
        assert ret2 == 0
