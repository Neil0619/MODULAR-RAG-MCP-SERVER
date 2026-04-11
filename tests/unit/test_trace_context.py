"""Unit tests for TraceContext enhancement (F1) and TraceCollector."""

from __future__ import annotations

import json
import time

import pytest

from core.trace.trace_context import TraceContext
from core.trace.trace_collector import TraceCollector


# ── TraceContext ──────────────────────────────────────────────────


class TestTraceContextBasic:
    """Tests for basic TraceContext creation and defaults."""

    def test_default_trace_type_is_query(self) -> None:
        t = TraceContext()
        assert t.trace_type == "query"

    def test_trace_type_ingestion(self) -> None:
        t = TraceContext(trace_type="ingestion")
        assert t.trace_type == "ingestion"

    def test_trace_id_generated(self) -> None:
        t = TraceContext()
        assert isinstance(t.trace_id, str) and len(t.trace_id) == 16

    def test_trace_id_unique(self) -> None:
        ids = {TraceContext().trace_id for _ in range(100)}
        assert len(ids) == 100

    def test_stages_default_empty(self) -> None:
        assert TraceContext().stages == {}


class TestRecordStage:
    """Tests for record_stage method."""

    def test_record_stage_with_dict_data(self) -> None:
        t = TraceContext()
        t.record_stage("load", {"method": "markitdown", "bytes": 1024})
        assert t.stages["load"]["method"] == "markitdown"
        assert t.stages["load"]["bytes"] == 1024

    def test_record_stage_with_none_data(self) -> None:
        t = TraceContext()
        t.record_stage("split")
        assert isinstance(t.stages["split"], dict)

    def test_record_stage_with_explicit_elapsed(self) -> None:
        t = TraceContext()
        t.record_stage("embed", {"provider": "openai"}, elapsed_ms=123.4)
        assert t.stages["embed"]["provider"] == "openai"
        assert t.stages["embed"]["elapsed_ms"] == 123.4

    def test_record_stage_auto_elapsed(self) -> None:
        t = TraceContext()
        t.start_stage("dense")
        time.sleep(0.01)  # 10ms
        t.record_stage("dense", {"top_k": 20})
        assert t.stages["dense"]["elapsed_ms"] > 0

    def test_record_stage_merges_data(self) -> None:
        t = TraceContext()
        t.record_stage("load", {"method": "markitdown"})
        t.record_stage("load", {"bytes": 2048}, elapsed_ms=50.0)
        assert t.stages["load"]["method"] == "markitdown"
        assert t.stages["load"]["bytes"] == 2048
        assert t.stages["load"]["elapsed_ms"] == 50.0


class TestFinish:
    """Tests for finish() and elapsed_ms()."""

    def test_finish_sets_finished_at(self) -> None:
        t = TraceContext()
        assert t.finished_at is None
        t.finish()
        assert t.finished_at is not None

    def test_total_elapsed_ms_before_finish(self) -> None:
        t = TraceContext()
        time.sleep(0.01)
        elapsed = t.elapsed_ms()
        assert elapsed > 0

    def test_total_elapsed_ms_after_finish(self) -> None:
        t = TraceContext()
        time.sleep(0.01)
        t.finish()
        elapsed = t.elapsed_ms()
        assert elapsed > 0

    def test_elapsed_ms_stabilises_after_finish(self) -> None:
        t = TraceContext()
        t.finish()
        e1 = t.elapsed_ms()
        time.sleep(0.01)
        e2 = t.elapsed_ms()
        assert e1 == e2  # no drift after finish

    def test_elapsed_ms_for_specific_stage(self) -> None:
        t = TraceContext()
        t.record_stage("load", elapsed_ms=42.5)
        assert t.elapsed_ms("load") == 42.5

    def test_elapsed_ms_unknown_stage(self) -> None:
        t = TraceContext()
        assert t.elapsed_ms("nonexistent") == 0.0


class TestToDict:
    """Tests for to_dict serialisation."""

    def test_basic_fields(self) -> None:
        t = TraceContext(trace_type="query")
        d = t.to_dict()
        assert "trace_id" in d
        assert d["trace_type"] == "query"
        assert "started_at" in d
        assert "total_elapsed_ms" in d
        assert "stages" in d

    def test_finished_at_included_after_finish(self) -> None:
        t = TraceContext()
        t.finish()
        d = t.to_dict()
        assert "finished_at" in d

    def test_no_finished_at_before_finish(self) -> None:
        t = TraceContext()
        d = t.to_dict()
        assert "finished_at" not in d

    def test_json_serialisable(self) -> None:
        t = TraceContext(trace_type="ingestion")
        t.record_stage("load", {"method": "markitdown"}, elapsed_ms=10.0)
        t.record_stage("split", {"chunk_count": 5}, elapsed_ms=5.0)
        t.finish()
        s = json.dumps(t.to_dict())
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert parsed["trace_type"] == "ingestion"
        assert parsed["stages"]["load"]["method"] == "markitdown"

    def test_stages_dict_is_copy(self) -> None:
        t = TraceContext()
        t.record_stage("x", {"val": 1})
        d = t.to_dict()
        d["stages"]["x"]["val"] = 999
        assert t.stages["x"]["val"] == 1  # original unchanged


# ── TraceCollector ────────────────────────────────────────────────


class TestTraceCollector:
    """Tests for TraceCollector."""

    def test_collect_finished_trace(self) -> None:
        collector = TraceCollector(persist=False)
        t = TraceContext(trace_type="query")
        t.record_stage("dense", {"top_k": 10}, elapsed_ms=5.0)
        t.finish()
        collector.collect(t)

        assert len(collector.traces) == 1
        assert collector.traces[0]["trace_type"] == "query"

    def test_collect_auto_finishes(self) -> None:
        collector = TraceCollector(persist=False)
        t = TraceContext()
        assert t.finished_at is None
        collector.collect(t)
        assert t.finished_at is not None

    def test_multiple_traces(self) -> None:
        collector = TraceCollector(persist=False)
        for i in range(5):
            t = TraceContext(trace_type="query")
            t.finish()
            collector.collect(t)
        assert len(collector.traces) == 5

    def test_clear(self) -> None:
        collector = TraceCollector(persist=False)
        collector.collect(TraceContext())
        collector.collect(TraceContext())
        assert len(collector.traces) == 2
        collector.clear()
        assert len(collector.traces) == 0

    def test_traces_returns_copy(self) -> None:
        collector = TraceCollector(persist=False)
        collector.collect(TraceContext())
        refs = collector.traces
        refs.clear()
        assert len(collector.traces) == 1

    def test_collect_persists_to_file(self, tmp_path) -> None:
        """collect() writes trace to JSONL file when persist=True."""
        import json
        log_file = tmp_path / "traces.jsonl"

        collector = TraceCollector(persist=True)
        t = TraceContext(trace_type="query")
        t.record_stage("test_stage", {"method": "test"})
        t.finish()

        # Monkey-patch write_trace to use our temp file
        from observability import logger as logger_mod
        _orig = logger_mod._DEFAULT_TRACE_FILE
        logger_mod._DEFAULT_TRACE_FILE = str(log_file)
        try:
            collector.collect(t)
        finally:
            logger_mod._DEFAULT_TRACE_FILE = _orig

        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        envelope = json.loads(lines[0])
        trace_data = json.loads(envelope["message"])
        assert trace_data["trace_type"] == "query"
        assert "test_stage" in trace_data["stages"]


# ── Backward compatibility ────────────────────────────────────────


class TestBackwardCompat:
    """Ensure existing code patterns still work."""

    def test_old_style_record_stage(self) -> None:
        """Existing code uses record_stage(name, data) without elapsed."""
        t = TraceContext()
        t.record_stage("load", {"method": "markitdown", "bytes": 1024})
        assert t.stages["load"]["method"] == "markitdown"

    def test_old_style_stages_access(self) -> None:
        """Existing code accesses .stages directly."""
        t = TraceContext()
        t.record_stage("x", "hello")
        assert "x" in t.stages

    def test_pipeline_trace_type_default(self) -> None:
        """Pipeline creates TraceContext() — should default to 'query'."""
        t = TraceContext()
        assert t.trace_type == "query"

    def test_pipeline_explicit_ingestion(self) -> None:
        t = TraceContext(trace_type="ingestion")
        assert t.trace_type == "ingestion"
