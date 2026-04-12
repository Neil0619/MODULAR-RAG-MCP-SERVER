"""Contract tests for DOCX and PPTX loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.types import Document

_FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


class TestDocxLoaderContract:

    def test_load_returns_document(self) -> None:
        from libs.loader.docx_loader import DocxLoader
        doc = DocxLoader().load(str(_FIXTURES / "sample.docx"))
        assert isinstance(doc, Document)

    def test_metadata_doc_type(self) -> None:
        from libs.loader.docx_loader import DocxLoader
        doc = DocxLoader().load(str(_FIXTURES / "sample.docx"))
        assert doc.metadata["doc_type"] == "docx"

    def test_metadata_source_path(self) -> None:
        from libs.loader.docx_loader import DocxLoader
        doc = DocxLoader().load(str(_FIXTURES / "sample.docx"))
        assert "source_path" in doc.metadata

    def test_metadata_doc_hash(self) -> None:
        from libs.loader.docx_loader import DocxLoader
        doc = DocxLoader().load(str(_FIXTURES / "sample.docx"))
        assert len(doc.metadata["doc_hash"]) == 16

    def test_metadata_images_list(self) -> None:
        from libs.loader.docx_loader import DocxLoader
        doc = DocxLoader().load(str(_FIXTURES / "sample.docx"))
        assert isinstance(doc.metadata["images"], list)

    def test_text_has_paragraphs(self) -> None:
        from libs.loader.docx_loader import DocxLoader
        doc = DocxLoader().load(str(_FIXTURES / "sample.docx"))
        assert "first paragraph" in doc.text
        assert "second paragraph" in doc.text

    def test_text_has_table(self) -> None:
        from libs.loader.docx_loader import DocxLoader
        doc = DocxLoader().load(str(_FIXTURES / "sample.docx"))
        assert "Col1" in doc.text
        assert "Col2" in doc.text

    def test_missing_file_raises(self) -> None:
        from libs.loader.docx_loader import DocxLoader
        with pytest.raises(FileNotFoundError):
            DocxLoader().load("/nonexistent/file.docx")


class TestPptxLoaderContract:

    def test_load_returns_document(self) -> None:
        from libs.loader.pptx_loader import PptxLoader
        doc = PptxLoader().load(str(_FIXTURES / "sample.pptx"))
        assert isinstance(doc, Document)

    def test_metadata_doc_type(self) -> None:
        from libs.loader.pptx_loader import PptxLoader
        doc = PptxLoader().load(str(_FIXTURES / "sample.pptx"))
        assert doc.metadata["doc_type"] == "pptx"

    def test_metadata_source_path(self) -> None:
        from libs.loader.pptx_loader import PptxLoader
        doc = PptxLoader().load(str(_FIXTURES / "sample.pptx"))
        assert "source_path" in doc.metadata

    def test_metadata_doc_hash(self) -> None:
        from libs.loader.pptx_loader import PptxLoader
        doc = PptxLoader().load(str(_FIXTURES / "sample.pptx"))
        assert len(doc.metadata["doc_hash"]) == 16

    def test_metadata_slide_count(self) -> None:
        from libs.loader.pptx_loader import PptxLoader
        doc = PptxLoader().load(str(_FIXTURES / "sample.pptx"))
        assert doc.metadata["slide_count"] == 2

    def test_text_has_slides(self) -> None:
        from libs.loader.pptx_loader import PptxLoader
        doc = PptxLoader().load(str(_FIXTURES / "sample.pptx"))
        assert "Slide 1" in doc.text
        assert "Slide 2" in doc.text
        assert "Slide 1 Title" in doc.text

    def test_metadata_images_list(self) -> None:
        from libs.loader.pptx_loader import PptxLoader
        doc = PptxLoader().load(str(_FIXTURES / "sample.pptx"))
        assert isinstance(doc.metadata["images"], list)

    def test_missing_file_raises(self) -> None:
        from libs.loader.pptx_loader import PptxLoader
        with pytest.raises(FileNotFoundError):
            PptxLoader().load("/nonexistent/file.pptx")
