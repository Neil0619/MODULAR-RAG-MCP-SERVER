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

    def test_import_ingestion_manager(self) -> None:
        from observability.dashboard.pages.ingestion_manager import render
        assert callable(render)

    def test_import_ingestion_traces(self) -> None:
        from observability.dashboard.pages.ingestion_traces import render
        assert callable(render)

    def test_import_trace_service(self) -> None:
        from observability.dashboard.services.trace_service import TraceService
        assert TraceService is not None

    def test_import_query_traces(self) -> None:
        from observability.dashboard.pages.query_traces import render
        assert callable(render)

    def test_import_evaluation_panel(self) -> None:
        from observability.dashboard.pages.evaluation_panel import render
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


# ---------------------------------------------------------------------------
# TraceService unit tests (G5)
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, traces: list[dict[str, Any]]) -> None:
    """Write traces as JSONL envelopes (same format as write_trace)."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in traces:
            envelope = {
                "timestamp": "2026-04-12T12:00:00",
                "level": "INFO",
                "logger": "rag.trace",
                "message": json.dumps(t, ensure_ascii=False),
            }
            f.write(json.dumps(envelope) + "\n")


class TestTraceService:

    def test_list_traces_empty(self, tmp_path: Path) -> None:
        from observability.dashboard.services.trace_service import TraceService
        svc = TraceService(str(tmp_path / "nosuch.jsonl"))
        assert svc.list_traces() == []

    def test_list_traces_reads_file(self, tmp_path: Path) -> None:
        from observability.dashboard.services.trace_service import TraceService
        f = tmp_path / "traces.jsonl"
        _write_jsonl(f, [
            {"trace_id": "t1", "trace_type": "ingestion"},
            {"trace_id": "t2", "trace_type": "query"},
            {"trace_id": "t3", "trace_type": "ingestion"},
        ])
        svc = TraceService(str(f))
        all_traces = svc.list_traces()
        assert len(all_traces) == 3

    def test_list_traces_filter_by_type(self, tmp_path: Path) -> None:
        from observability.dashboard.services.trace_service import TraceService
        f = tmp_path / "traces.jsonl"
        _write_jsonl(f, [
            {"trace_id": "t1", "trace_type": "ingestion"},
            {"trace_id": "t2", "trace_type": "query"},
            {"trace_id": "t3", "trace_type": "ingestion"},
        ])
        svc = TraceService(str(f))
        ingest = svc.list_traces(trace_type="ingestion")
        assert len(ingest) == 2
        assert all(t["trace_type"] == "ingestion" for t in ingest)

    def test_list_traces_newest_first(self, tmp_path: Path) -> None:
        from observability.dashboard.services.trace_service import TraceService
        f = tmp_path / "traces.jsonl"
        _write_jsonl(f, [
            {"trace_id": "old", "trace_type": "ingestion"},
            {"trace_id": "new", "trace_type": "ingestion"},
        ])
        svc = TraceService(str(f))
        traces = svc.list_traces(trace_type="ingestion")
        assert traces[0]["trace_id"] == "new"
        assert traces[1]["trace_id"] == "old"

    def test_get_trace_by_id(self, tmp_path: Path) -> None:
        from observability.dashboard.services.trace_service import TraceService
        f = tmp_path / "traces.jsonl"
        _write_jsonl(f, [
            {"trace_id": "abc", "trace_type": "query"},
            {"trace_id": "def", "trace_type": "ingestion"},
        ])
        svc = TraceService(str(f))
        assert svc.get_trace("def") is not None
        assert svc.get_trace("def")["trace_type"] == "ingestion"
        assert svc.get_trace("zzz") is None

    def test_get_stage_elapsed(self, tmp_path: Path) -> None:
        from observability.dashboard.services.trace_service import TraceService
        svc = TraceService(str(tmp_path / "x.jsonl"))
        trace = {
            "trace_id": "t1",
            "stages": {
                "load": {"elapsed_ms": 100.0, "method": "markitdown"},
                "split": {"elapsed_ms": 50.0, "method": "recursive"},
            },
        }
        stages = svc.get_stage_elapsed(trace)
        assert len(stages) == 2
        names = {s["stage"] for s in stages}
        assert names == {"load", "split"}
        load = next(s for s in stages if s["stage"] == "load")
        assert load["elapsed_ms"] == 100.0
        assert load["method"] == "markitdown"

    def test_get_stage_elapsed_empty(self, tmp_path: Path) -> None:
        from observability.dashboard.services.trace_service import TraceService
        svc = TraceService(str(tmp_path / "x.jsonl"))
        assert svc.get_stage_elapsed({}) == []
        assert svc.get_stage_elapsed({"stages": {}}) == []

    def test_limit(self, tmp_path: Path) -> None:
        from observability.dashboard.services.trace_service import TraceService
        f = tmp_path / "traces.jsonl"
        _write_jsonl(f, [{"trace_id": f"t{i}", "trace_type": "query"} for i in range(20)])
        svc = TraceService(str(f))
        assert len(svc.list_traces(limit=5)) == 5
