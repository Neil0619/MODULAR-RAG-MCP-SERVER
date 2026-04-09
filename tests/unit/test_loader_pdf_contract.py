"""Unit tests for BaseLoader and PdfLoader (C3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.types import Document
from libs.loader.base_loader import BaseLoader
from libs.loader.pdf_loader import PdfLoader


# ---------------------------------------------------------------------------
# BaseLoader contract
# ---------------------------------------------------------------------------


class TestBaseLoader:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseLoader()  # type: ignore[abstract]

    def test_validate_path_not_found(self) -> None:
        loader = PdfLoader()
        with pytest.raises(FileNotFoundError, match="File not found"):
            loader.load("/nonexistent/file.pdf")

    def test_validate_path_not_a_file(self, tmp_path: Path) -> None:
        loader = PdfLoader()
        d = tmp_path / "adir"
        d.mkdir()
        with pytest.raises(ValueError, match="Not a file"):
            loader.load(str(d))


# ---------------------------------------------------------------------------
# PdfLoader with simple PDF (no images)
# ---------------------------------------------------------------------------


class TestPdfLoaderSimple:
    @pytest.fixture()
    def loader(self, tmp_path: Path) -> PdfLoader:
        return PdfLoader(image_dir=str(tmp_path / "images"))

    def test_load_simple_pdf(self, loader: PdfLoader) -> None:
        doc = loader.load("tests/fixtures/simple.pdf")
        assert isinstance(doc, Document)
        assert "Hello from the test PDF" in doc.text
        assert doc.metadata["source_path"]
        assert doc.metadata["doc_type"] == "pdf"

    def test_metadata_has_source_path(self, loader: PdfLoader) -> None:
        doc = loader.load("tests/fixtures/simple.pdf")
        assert "source_path" in doc.metadata
        assert "simple.pdf" in doc.metadata["source_path"]

    def test_metadata_has_page_count(self, loader: PdfLoader) -> None:
        doc = loader.load("tests/fixtures/simple.pdf")
        assert doc.metadata["page_count"] >= 1

    def test_simple_pdf_no_images(self, loader: PdfLoader) -> None:
        doc = loader.load("tests/fixtures/simple.pdf")
        images = doc.metadata.get("images", [])
        assert len(images) == 0


# ---------------------------------------------------------------------------
# PdfLoader with images PDF
# ---------------------------------------------------------------------------


class TestPdfLoaderWithImages:
    @pytest.fixture()
    def loader(self, tmp_path: Path) -> PdfLoader:
        return PdfLoader(image_dir=str(tmp_path / "images"))

    def test_load_pdf_with_images(self, loader: PdfLoader) -> None:
        doc = loader.load("tests/fixtures/with_images.pdf")
        assert isinstance(doc, Document)
        assert "Page with an image" in doc.text

    def test_image_placeholder_in_text(self, loader: PdfLoader) -> None:
        doc = loader.load("tests/fixtures/with_images.pdf")
        assert "[IMAGE:" in doc.text

    def test_images_metadata_populated(self, loader: PdfLoader) -> None:
        doc = loader.load("tests/fixtures/with_images.pdf")
        images = doc.metadata.get("images", [])
        assert len(images) >= 1
        img = images[0]
        assert "id" in img
        assert "path" in img
        assert "page" in img
        assert "text_offset" in img
        assert "text_length" in img

    def test_image_file_saved_to_disk(self, loader: PdfLoader, tmp_path: Path) -> None:
        doc = loader.load("tests/fixtures/with_images.pdf")
        images = doc.metadata.get("images", [])
        if images:
            img_path = Path(images[0]["path"])
            assert img_path.exists()
            assert img_path.stat().st_size > 0

    def test_text_offset_matches_placeholder(self, loader: PdfLoader) -> None:
        doc = loader.load("tests/fixtures/with_images.pdf")
        for img in doc.metadata.get("images", []):
            placeholder = f"[IMAGE: {img['id']}]"
            offset = img["text_offset"]
            assert doc.text[offset : offset + len(placeholder)] == placeholder


# ---------------------------------------------------------------------------
# PdfLoader error handling
# ---------------------------------------------------------------------------


class TestPdfLoaderErrors:
    def test_non_pdf_file_still_produces_document(self, tmp_path: Path) -> None:
        """PyMuPDF can open text files — it gracefully handles non-PDF input."""
        txt = tmp_path / "test.txt"
        txt.write_text("not a pdf")
        loader = PdfLoader()
        doc = loader.load(str(txt))
        assert isinstance(doc, Document)
        assert doc.metadata["source_path"]
