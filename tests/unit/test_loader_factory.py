"""Unit tests for LoaderFactory."""

from __future__ import annotations

import pytest

from libs.loader.loader_factory import LoaderFactory


class TestLoaderFactoryRouting:

    def test_create_from_path_pdf(self) -> None:
        loader = LoaderFactory.create_from_path("report.pdf")
        from libs.loader.pdf_loader import PdfLoader
        assert isinstance(loader, PdfLoader)

    def test_create_from_path_txt(self) -> None:
        loader = LoaderFactory.create_from_path("notes.txt")
        from libs.loader.txt_loader import TxtLoader
        assert isinstance(loader, TxtLoader)

    def test_create_from_path_md(self) -> None:
        loader = LoaderFactory.create_from_path("readme.md")
        from libs.loader.markdown_loader import MarkdownLoader
        assert isinstance(loader, MarkdownLoader)

    def test_create_from_path_markdown(self) -> None:
        loader = LoaderFactory.create_from_path("doc.markdown")
        from libs.loader.markdown_loader import MarkdownLoader
        assert isinstance(loader, MarkdownLoader)

    def test_create_from_path_case_insensitive(self) -> None:
        loader = LoaderFactory.create_from_path("report.PDF")
        from libs.loader.pdf_loader import PdfLoader
        assert isinstance(loader, PdfLoader)

    def test_create_from_path_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported file type"):
            LoaderFactory.create_from_path("data.xyz")

    def test_create_from_path_unknown_lists_supported(self) -> None:
        with pytest.raises(ValueError, match="pdf"):
            LoaderFactory.create_from_path("data.xyz")

    def test_create_from_name_pdf(self) -> None:
        loader = LoaderFactory.create_from_name("pdf")
        from libs.loader.pdf_loader import PdfLoader
        assert isinstance(loader, PdfLoader)

    def test_create_from_name_with_dot(self) -> None:
        loader = LoaderFactory.create_from_name(".txt")
        from libs.loader.txt_loader import TxtLoader
        assert isinstance(loader, TxtLoader)

    def test_create_from_name_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            LoaderFactory.create_from_name("xyz")


class TestLoaderFactoryRegistry:

    def test_supported_extensions(self) -> None:
        exts = LoaderFactory.supported_extensions()
        assert "pdf" in exts
        assert "txt" in exts
        assert "md" in exts
        assert "docx" in exts
        assert "mp4" in exts

    def test_register_extension(self) -> None:
        import libs.loader.loader_factory as lf_mod
        original = len(lf_mod._EXTENSION_REGISTRY)
        LoaderFactory.register_extension("json", "libs.loader.txt_loader.TxtLoader")
        assert len(lf_mod._EXTENSION_REGISTRY) == original + 1
        assert "json" in LoaderFactory.supported_extensions()
        # Verify routing works
        loader = LoaderFactory.create_from_name("json")
        assert isinstance(loader, type(LoaderFactory.create_from_name("txt")))
        # Cleanup
        del lf_mod._EXTENSION_REGISTRY["json"]
