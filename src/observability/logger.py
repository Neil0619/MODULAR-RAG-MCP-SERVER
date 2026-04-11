"""Structured logging — stderr + JSON Lines trace persistence.

* ``get_logger()`` — human-readable logs to stderr (stdout reserved for MCP).
* ``JSONFormatter`` — formats log records as single-line JSON.
* ``get_trace_logger()`` — logger that writes JSON Lines to a file.
* ``write_trace()`` — convenience function to persist a trace dict.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any


# ── Human-readable stderr logger (existing) ──────────────────────


def get_logger(name: str = "rag") -> logging.Logger:
    """Get a logger that outputs to stderr.

    stdout is reserved for MCP JSON-RPC messages; all logs go to stderr
    to avoid polluting the protocol channel.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ── JSON formatter ────────────────────────────────────────────────


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt or "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


# ── Trace logger ──────────────────────────────────────────────────

_TRACE_LOGGER_NAME = "rag.trace"

# Default path; can be overridden via get_trace_logger(log_file=...)
_DEFAULT_TRACE_FILE = "logs/traces.jsonl"


def get_trace_logger(
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Get a logger that writes JSON Lines to *log_file*.

    Creates the parent directory if it doesn't exist.  Repeated calls
    with the same logger name return the same logger (only one file
    handler is attached).

    Args:
        log_file: Path to the ``.jsonl`` file.  Defaults to
            ``logs/traces.jsonl``.
    """
    logger = logging.getLogger(_TRACE_LOGGER_NAME)

    # Avoid duplicate handlers
    if not any(isinstance(h, _TraceFileHandler) for h in logger.handlers):
        path = Path(log_file) if log_file else Path(_DEFAULT_TRACE_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)

        handler = _TraceFileHandler(path)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


class _TraceFileHandler(logging.FileHandler):
    """FileHandler that remembers its path for tests."""

    def __init__(self, path: Path) -> None:
        super().__init__(str(path), encoding="utf-8")
        self.file_path = path


def write_trace(
    trace_dict: dict[str, Any],
    log_file: str | Path | None = None,
) -> None:
    """Persist a trace dictionary as a single JSON Line.

    Appends one line to the trace log file.  Each line is a JSON object
    with a ``message`` field containing the serialised trace.

    Args:
        trace_dict: Serialisable dict (typically from
            :meth:`TraceContext.to_dict`).
        log_file: Override trace file path.  Defaults to
            ``logs/traces.jsonl``.
    """
    path = Path(log_file) if log_file else Path(_DEFAULT_TRACE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)

    import datetime
    envelope = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "level": "INFO",
        "logger": _TRACE_LOGGER_NAME,
        "message": json.dumps(trace_dict, ensure_ascii=False, default=str),
    }
    line = json.dumps(envelope, ensure_ascii=False, default=str)

    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
