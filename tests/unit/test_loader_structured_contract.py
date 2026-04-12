"""Contract tests for HTML and CSV loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.types import Document

_FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


class TestHtmlLoaderContract:

    def test_load_returns_document(self) -> None:
        from libs.loader.html_loader import HtmlLoader
        doc = HtmlLoader().load(str(_FIXTURES / "sample.html"))
        assert isinstance(doc, Document)

    def test_metadata_doc_type(self) -> None:
        from libs.loader.html_loader import HtmlLoader
        doc = HtmlLoader().load(str(_FIXTURES / "sample.html"))
        assert doc.metadata["doc_type"] == "html"

    def test_metadata_source_path(self) -> None:
        from libs.loader.html_loader import HtmlLoader
        doc = HtmlLoader().load(str(_FIXTURES / "sample.html"))
        assert "source_path" in doc.metadata

    def test_metadata_doc_hash(self) -> None:
        from libs.loader.html_loader import HtmlLoader
        doc = HtmlLoader().load(str(_FIXTURES / "sample.html"))
        assert len(doc.metadata["doc_hash"]) == 16

    def test_title_extracted(self) -> None:
        from libs.loader.html_loader import HtmlLoader
        doc = HtmlLoader().load(str(_FIXTURES / "sample.html"))
        assert doc.metadata["title"] == "Sample HTML Page"

    def test_heading_outline(self) -> None:
        from libs.loader.html_loader import HtmlLoader
        doc = HtmlLoader().load(str(_FIXTURES / "sample.html"))
        headings = doc.metadata.get("heading_outline", [])
        assert "Sample HTML" in headings
        assert "Introduction" in headings

    def test_script_stripped(self) -> None:
        from libs.loader.html_loader import HtmlLoader
        doc = HtmlLoader().load(str(_FIXTURES / "sample.html"))
        assert "console.log" not in doc.text

    def test_style_stripped(self) -> None:
        from libs.loader.html_loader import HtmlLoader
        doc = HtmlLoader().load(str(_FIXTURES / "sample.html"))
        assert "color: red" not in doc.text

    def test_text_has_content(self) -> None:
        from libs.loader.html_loader import HtmlLoader
        doc = HtmlLoader().load(str(_FIXTURES / "sample.html"))
        assert "bold text" in doc.text

    def test_missing_file_raises(self) -> None:
        from libs.loader.html_loader import HtmlLoader
        with pytest.raises(FileNotFoundError):
            HtmlLoader().load("/nonexistent/file.html")


class TestCsvLoaderContract:

    def test_load_returns_document(self) -> None:
        from libs.loader.csv_loader import CsvLoader
        doc = CsvLoader().load(str(_FIXTURES / "sample.csv"))
        assert isinstance(doc, Document)

    def test_metadata_doc_type(self) -> None:
        from libs.loader.csv_loader import CsvLoader
        doc = CsvLoader().load(str(_FIXTURES / "sample.csv"))
        assert doc.metadata["doc_type"] == "csv"

    def test_metadata_row_count(self) -> None:
        from libs.loader.csv_loader import CsvLoader
        doc = CsvLoader().load(str(_FIXTURES / "sample.csv"))
        assert doc.metadata["row_count"] == 3

    def test_metadata_column_count(self) -> None:
        from libs.loader.csv_loader import CsvLoader
        doc = CsvLoader().load(str(_FIXTURES / "sample.csv"))
        assert doc.metadata["column_count"] == 3

    def test_metadata_headers(self) -> None:
        from libs.loader.csv_loader import CsvLoader
        doc = CsvLoader().load(str(_FIXTURES / "sample.csv"))
        assert doc.metadata["headers"] == ["name", "age", "city"]

    def test_text_is_markdown_table(self) -> None:
        from libs.loader.csv_loader import CsvLoader
        doc = CsvLoader().load(str(_FIXTURES / "sample.csv"))
        assert "name | age | city" in doc.text
        assert "--- | --- | ---" in doc.text
        assert "Alice | 30 | Beijing" in doc.text

    def test_metadata_images_list(self) -> None:
        from libs.loader.csv_loader import CsvLoader
        doc = CsvLoader().load(str(_FIXTURES / "sample.csv"))
        assert isinstance(doc.metadata["images"], list)

    def test_empty_csv(self, tmp_path: Path) -> None:
        from libs.loader.csv_loader import CsvLoader
        f = tmp_path / "empty.csv"
        f.write_text("")
        doc = CsvLoader().load(str(f))
        assert doc.text == ""

    def test_missing_file_raises(self) -> None:
        from libs.loader.csv_loader import CsvLoader
        with pytest.raises(FileNotFoundError):
            CsvLoader().load("/nonexistent/file.csv")
