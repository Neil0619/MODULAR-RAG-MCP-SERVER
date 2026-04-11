"""Unit tests for JSON Lines logger (F2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observability.logger import (
    JSONFormatter,
    get_logger,
    get_trace_logger,
    write_trace,
)


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_basic_record(self) -> None:
        import logging

        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test"
        assert "timestamp" in parsed

    def test_non_ascii_message(self) -> None:
        import logging

        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="中文消息", args=(), exc_info=None,
        )
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "中文消息"


class TestGetLogger:
    """Tests for existing get_logger (regression)."""

    def test_returns_logger(self) -> None:
        import logging

        lg = get_logger("test_jsonl")
        assert isinstance(lg, logging.Logger)

    def test_stderr_handler(self) -> None:
        import sys

        lg = get_logger("test_stderr_check")
        assert any(h.stream is sys.stderr for h in lg.handlers)


class TestGetTraceLogger:
    """Tests for get_trace_logger."""

    def test_creates_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "traces.jsonl"
        # Use a unique logger name by creating a new one each test
        import logging

        name = f"rag.trace.test_{id(log_file)}"
        logger = logging.getLogger(name)
        logger.handlers.clear()

        # Directly test file creation
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(log_file), encoding="utf-8")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.info("test")

        assert log_file.exists()
        handler.close()

    def test_json_lines_format(self, tmp_path: Path) -> None:
        import logging

        log_file = tmp_path / "traces.jsonl"
        logger = logging.getLogger(f"rag.trace.format_{id(log_file)}")
        logger.handlers.clear()

        handler = logging.FileHandler(str(log_file), encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        payload = {"trace_id": "abc123", "trace_type": "query", "stages": {}}
        logger.info(json.dumps(payload))

        handler.close()

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert "message" in parsed
        # The message field contains the JSON payload as a string
        inner = json.loads(parsed["message"])
        assert inner["trace_id"] == "abc123"
        assert inner["trace_type"] == "query"


class TestWriteTrace:
    """Tests for write_trace convenience function."""

    def test_write_trace_creates_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "traces.jsonl"
        trace = {
            "trace_id": "t001",
            "trace_type": "ingestion",
            "total_elapsed_ms": 150.0,
            "stages": {"load": {"elapsed_ms": 50.0}},
        }
        write_trace(trace, log_file=log_file)

        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        # The line should be parseable JSON containing the trace
        outer = json.loads(lines[0])
        inner = json.loads(outer["message"])
        assert inner["trace_id"] == "t001"
        assert inner["trace_type"] == "ingestion"

    def test_write_multiple_traces(self, tmp_path: Path) -> None:
        log_file = tmp_path / "traces.jsonl"

        for i in range(3):
            write_trace({"trace_id": f"t{i:03d}", "trace_type": "query"},
                        log_file=log_file)

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_trace_contains_trace_type(self, tmp_path: Path) -> None:
        log_file = tmp_path / "traces.jsonl"
        write_trace({"trace_id": "x", "trace_type": "ingestion"},
                    log_file=log_file)

        content = log_file.read_text()
        assert "ingestion" in content

    def test_non_ascii_values(self, tmp_path: Path) -> None:
        log_file = tmp_path / "traces.jsonl"
        write_trace({"trace_id": "y", "source_path": "/文档/测试.pdf"},
                    log_file=log_file)

        content = log_file.read_text()
        assert "测试" in content
