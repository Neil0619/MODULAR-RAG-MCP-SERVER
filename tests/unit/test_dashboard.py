"""Unit tests for Dashboard components (G1 + G3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from observability.dashboard.services.config_service import ConfigService, ComponentInfo, SystemOverview
from observability.dashboard.services.data_service import DataService
from core.settings import (
    EmbeddingSettings,
    LLMSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)
from ingestion.document_manager import DocumentInfo


def _make_settings() -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o", api_key="sk-test"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small", api_key="sk-test"),
        splitter=SplitterSettings(chunk_size=500, chunk_overlap=50),
        vector_store=VectorStoreSettings(backend="chroma"),
    )


class TestConfigService:
    """Tests for ConfigService."""

    def test_get_overview_returns_components(self) -> None:
        service = ConfigService(settings=_make_settings())
        overview = service.get_overview()

        assert isinstance(overview, SystemOverview)
        assert len(overview.components) == 5

    def test_llm_component_info(self) -> None:
        service = ConfigService(settings=_make_settings())
        overview = service.get_overview()
        llm = overview.components[0]

        assert llm.name == "LLM"
        assert llm.provider == "openai"
        assert llm.model == "gpt-4o"
        assert llm.details["api_key_set"] is True

    def test_splitter_component_info(self) -> None:
        service = ConfigService(settings=_make_settings())
        overview = service.get_overview()
        splitter = overview.components[2]

        assert splitter.name == "Splitter"
        assert splitter.provider == "recursive"
        assert splitter.details["chunk_size"] == 500

    def test_reranker_component_info(self) -> None:
        service = ConfigService(settings=_make_settings())
        overview = service.get_overview()
        reranker = overview.components[4]

        assert reranker.name == "Reranker"
        assert reranker.provider == "none"

    def test_get_component_table(self) -> None:
        service = ConfigService(settings=_make_settings())
        rows = service.get_component_table()

        assert len(rows) == 5
        assert rows[0]["Component"] == "LLM"
        assert rows[0]["Provider"] == "openai"

    def test_no_api_key_shows_false(self) -> None:
        settings = Settings(
            llm=LLMSettings(provider="ollama", model="llama3"),
            embedding=EmbeddingSettings(provider="ollama", model="nomic-embed-text"),
            splitter=SplitterSettings(),
            vector_store=VectorStoreSettings(backend="chroma"),
        )
        service = ConfigService(settings=settings)
        overview = service.get_overview()

        llm = overview.components[0]
        assert llm.details["api_key_set"] is False


class TestDashboardImports:
    """Verify dashboard modules are importable."""

    def test_import_app(self) -> None:
        from observability.dashboard.app import main
        assert callable(main)

    def test_import_overview(self) -> None:
        from observability.dashboard.pages.overview import render
        assert callable(render)

    def test_import_placeholder(self) -> None:
        from observability.dashboard.pages.placeholder import render_placeholder
        assert callable(render_placeholder)

    def test_import_config_service(self) -> None:
        from observability.dashboard.services.config_service import ConfigService
        assert ConfigService is not None

    def test_import_data_service(self) -> None:
        from observability.dashboard.services.data_service import DataService
        assert DataService is not None

    def test_import_data_browser(self) -> None:
        from observability.dashboard.pages.data_browser import render
        assert callable(render)

    def test_start_dashboard_script_exists(self) -> None:
        script = Path(__file__).resolve().parents[2] / "scripts" / "start_dashboard.py"
        assert script.exists()


# ---------------------------------------------------------------------------
# DataService unit tests (G3)
# ---------------------------------------------------------------------------


class _FakeDocumentManager:
    def __init__(self, docs: list[DocumentInfo] | None = None) -> None:
        self._docs = docs or []

    def list_documents(self, collection: str = "default") -> list[DocumentInfo]:
        return [d for d in self._docs if d.collection == collection]


class _FakeImageStorage:
    def __init__(self, images: list[dict[str, Any]] | None = None) -> None:
        self._images = images or []

    def get_by_doc_hash(self, doc_hash: str) -> list[dict[str, Any]]:
        return [i for i in self._images if i.get("doc_hash") == doc_hash]

    def get_path(self, image_id: str) -> str | None:
        for i in self._images:
            if i.get("image_id") == image_id:
                return i.get("file_path")
        return None


class TestDataService:

    def test_list_documents_returns_docs(self) -> None:
        docs = [
            DocumentInfo(source_path="a.pdf", collection="default", chunk_count=3),
            DocumentInfo(source_path="b.pdf", collection="test", chunk_count=1),
        ]
        svc = DataService(
            document_manager=_FakeDocumentManager(docs),
            image_storage=_FakeImageStorage(),
        )
        result = svc.list_documents("default")
        assert len(result) == 1
        assert result[0].source_path == "a.pdf"

    def test_list_documents_empty(self) -> None:
        svc = DataService(
            document_manager=_FakeDocumentManager(),
            image_storage=_FakeImageStorage(),
        )
        assert svc.list_documents("default") == []

    def test_get_images_for_doc(self) -> None:
        images = [
            {"image_id": "img1", "doc_hash": "h1", "file_path": "/tmp/img1.png"},
        ]
        svc = DataService(
            document_manager=_FakeDocumentManager(),
            image_storage=_FakeImageStorage(images),
        )
        result = svc.get_images_for_doc("h1")
        assert len(result) == 1
        assert result[0]["image_id"] == "img1"

    def test_get_images_no_match(self) -> None:
        svc = DataService(
            document_manager=_FakeDocumentManager(),
            image_storage=_FakeImageStorage(),
        )
        assert svc.get_images_for_doc("nope") == []

    def test_get_image_path(self) -> None:
        images = [
            {"image_id": "img1", "file_path": "/tmp/img1.png"},
        ]
        svc = DataService(
            document_manager=_FakeDocumentManager(),
            image_storage=_FakeImageStorage(images),
        )
        assert svc.get_image_path("img1") == "/tmp/img1.png"
        assert svc.get_image_path("missing") is None
