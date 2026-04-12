"""Pipeline multiformat tests — verify LoaderFactory routing (J7)."""

from __future__ import annotations

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
from core.types import ChunkRecord, Document
from ingestion.pipeline import IngestionPipeline


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


def _run_with_mocked_encoding(settings: Settings, file_path: str, tmp_path: Path) -> dict[str, Any]:
    """Run pipeline with BatchProcessor mocked to avoid real API calls."""
    dim = 8
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

    with patch("ingestion.pipeline.BatchProcessor", side_effect=make_batch):
        pipeline = IngestionPipeline(settings, collection="test")
        return pipeline.run(file_path, force=True)


class TestPipelineMultiformat:
    """Verify pipeline routes files correctly through LoaderFactory."""

    @pytest.mark.parametrize(
        "ext,doc_type",
        [
            ("txt", "txt"),
            ("md", "markdown"),
            ("csv", "csv"),
            ("html", "html"),
        ],
    )
    def test_text_formats(self, tmp_path: Path, ext: str, doc_type: str) -> None:
        """Pipeline handles plain text-based formats."""
        settings = _make_settings(tmp_path)

        dummy = tmp_path / f"test.{ext}"
        dummy.write_text("Test content for multiformat pipeline. " * 20, encoding="utf-8")

        summary = _run_with_mocked_encoding(settings, str(dummy), tmp_path)

        assert "skipped" not in summary
        assert summary["stages"]["load"]["method"] == doc_type
        assert summary["stages"]["split"]["chunk_count"] > 0

    @pytest.mark.parametrize(
        "ext,doc_type",
        [
            ("docx", "docx"),
            ("pptx", "pptx"),
        ],
    )
    def test_office_formats(self, tmp_path: Path, ext: str, doc_type: str) -> None:
        """Pipeline routes office formats through correct loader."""
        settings = _make_settings(tmp_path)

        fixture = Path(__file__).resolve().parents[1] / "fixtures" / f"sample.{ext}"
        if not fixture.exists():
            pytest.skip(f"sample.{ext} fixture not found")

        summary = _run_with_mocked_encoding(settings, str(fixture), tmp_path)
        assert "skipped" not in summary
        assert summary["stages"]["load"]["method"] == doc_type

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """Unsupported file extension raises PipelineError."""
        settings = _make_settings(tmp_path)

        dummy = tmp_path / "test.xyz"
        dummy.write_text("unknown format", encoding="utf-8")

        with pytest.raises(Exception):
            _run_with_mocked_encoding(settings, str(dummy), tmp_path)

    def test_factory_creates_correct_loader(self) -> None:
        """LoaderFactory returns the right loader type for each extension."""
        from libs.loader.loader_factory import LoaderFactory
        from libs.loader.txt_loader import TxtLoader
        from libs.loader.markdown_loader import MarkdownLoader
        from libs.loader.csv_loader import CsvLoader

        assert isinstance(LoaderFactory.create_from_name("txt"), TxtLoader)
        assert isinstance(LoaderFactory.create_from_name("markdown"), MarkdownLoader)
        assert isinstance(LoaderFactory.create_from_name("csv"), CsvLoader)

    def test_factory_supported_extensions_list(self) -> None:
        """supported_extensions() returns expected formats."""
        from libs.loader.loader_factory import LoaderFactory

        exts = LoaderFactory.supported_extensions()
        for expected in ("txt", "md", "csv", "html", "pdf", "docx", "pptx", "mp4"):
            assert expected in exts, f"Missing extension: {expected}"
