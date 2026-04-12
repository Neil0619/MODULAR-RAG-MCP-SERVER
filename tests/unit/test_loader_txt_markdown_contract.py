"""Contract tests for TXT and Markdown loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.types import Document

_FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


class TestTxtLoaderContract:

    def test_load_returns_document(self) -> None:
        from libs.loader.txt_loader import TxtLoader
        doc = TxtLoader().load(str(_FIXTURES / "sample.txt"))
        assert isinstance(doc, Document)

    def test_metadata_source_path(self) -> None:
        from libs.loader.txt_loader import TxtLoader
        doc = TxtLoader().load(str(_FIXTURES / "sample.txt"))
        assert "source_path" in doc.metadata
        assert Path(doc.metadata["source_path"]).exists()

    def test_metadata_doc_type(self) -> None:
        from libs.loader.txt_loader import TxtLoader
        doc = TxtLoader().load(str(_FIXTURES / "sample.txt"))
        assert doc.metadata["doc_type"] == "txt"

    def test_metadata_doc_hash(self) -> None:
        from libs.loader.txt_loader import TxtLoader
        doc = TxtLoader().load(str(_FIXTURES / "sample.txt"))
        assert len(doc.metadata["doc_hash"]) == 16

    def test_metadata_images_list(self) -> None:
        from libs.loader.txt_loader import TxtLoader
        doc = TxtLoader().load(str(_FIXTURES / "sample.txt"))
        assert isinstance(doc.metadata["images"], list)
        assert doc.metadata["images"] == []

    def test_text_content(self) -> None:
        from libs.loader.txt_loader import TxtLoader
        doc = TxtLoader().load(str(_FIXTURES / "sample.txt"))
        assert "sample text file" in doc.text
        assert len(doc.text) > 50

    def test_missing_file_raises(self) -> None:
        from libs.loader.txt_loader import TxtLoader
        with pytest.raises(FileNotFoundError):
            TxtLoader().load("/nonexistent/file.txt")

    def test_empty_file(self, tmp_path: Path) -> None:
        from libs.loader.txt_loader import TxtLoader
        f = tmp_path / "empty.txt"
        f.write_text("")
        doc = TxtLoader().load(str(f))
        assert doc.text == ""
        assert doc.metadata["doc_type"] == "txt"


class TestMarkdownLoaderContract:

    def test_load_returns_document(self) -> None:
        from libs.loader.markdown_loader import MarkdownLoader
        doc = MarkdownLoader().load(str(_FIXTURES / "sample.md"))
        assert isinstance(doc, Document)

    def test_metadata_source_path(self) -> None:
        from libs.loader.markdown_loader import MarkdownLoader
        doc = MarkdownLoader().load(str(_FIXTURES / "sample.md"))
        assert "source_path" in doc.metadata

    def test_metadata_doc_type(self) -> None:
        from libs.loader.markdown_loader import MarkdownLoader
        doc = MarkdownLoader().load(str(_FIXTURES / "sample.md"))
        assert doc.metadata["doc_type"] == "markdown"

    def test_heading_outline(self) -> None:
        from libs.loader.markdown_loader import MarkdownLoader
        doc = MarkdownLoader().load(str(_FIXTURES / "sample.md"))
        headings = doc.metadata.get("heading_outline", [])
        assert len(headings) >= 3
        assert any("Sample Markdown" in h for h in headings)

    def test_text_content(self) -> None:
        from libs.loader.markdown_loader import MarkdownLoader
        doc = MarkdownLoader().load(str(_FIXTURES / "sample.md"))
        assert "**sample**" in doc.text
        assert "```python" in doc.text

    def test_missing_file_raises(self) -> None:
        from libs.loader.markdown_loader import MarkdownLoader
        with pytest.raises(FileNotFoundError):
            MarkdownLoader().load("/nonexistent/file.md")

    def test_metadata_images_list(self) -> None:
        from libs.loader.markdown_loader import MarkdownLoader
        doc = MarkdownLoader().load(str(_FIXTURES / "sample.md"))
        assert isinstance(doc.metadata["images"], list)
