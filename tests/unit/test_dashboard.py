"""Unit tests for Dashboard components (G1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observability.dashboard.services.config_service import ConfigService, ComponentInfo, SystemOverview
from core.settings import (
    EmbeddingSettings,
    LLMSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)


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

    def test_start_dashboard_script_exists(self) -> None:
        script = Path(__file__).resolve().parents[2] / "scripts" / "start_dashboard.py"
        assert script.exists()
