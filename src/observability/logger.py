"""Structured logging — stderr + date-folder/level-split files + trace persistence.

File layout::

    logs/
    ├── 2026-04-12/          <- one folder per day
    │   ├── info.log         <- INFO level
    │   ├── warning.log      <- WARNING level
    │   └── error.log        <- ERROR + CRITICAL
    ├── 2026-04-13/
    │   ├── info.log
    │   ├── warning.log
    │   └── error.log
    └── traces/
        └── traces.jsonl     <- pipeline traces

Old date folders beyond ``_RETENTION_DAYS`` are pruned on startup.
"""

from __future__ import annotations

import datetime
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

_LOG_DIR = Path("logs")
_RETENTION_DAYS = 30

_FMT = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Level → filename mapping
_LEVEL_FILES: dict[int, str] = {
    logging.INFO: "info.log",
    logging.WARNING: "warning.log",
    logging.ERROR: "error.log",
}


# ── Date-level rotating file handler ─────────────────────────────


class _DateLevelFileHandler(logging.Handler):
    """File handler that organises logs into ``logs/YYYY-MM-DD/<level>.log``.

    Automatically creates a new date folder at midnight.  Each log level
    gets its own file inside the date folder.
    """

    def __init__(self, level: int, log_dir: Path = _LOG_DIR) -> None:
        super().__init__(level)
        self._log_dir = log_dir
        self._level_file = _LEVEL_FILES.get(level, f"level_{level}.log")
        self._current_date: str = ""
        self._file_handle: Any = None

    # ── internal helpers ──────────────────────────────────────────

    def _today(self) -> str:
        return datetime.date.today().isoformat()

    def _ensure_open(self) -> None:
        today = self._today()
        if today != self._current_date:
            self._close()
            date_dir = self._log_dir / today
            date_dir.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(
                date_dir / self._level_file, "a", encoding="utf-8",
            )
            self._current_date = today

    def _close(self) -> None:
        if self._file_handle is not None:
            self._file_handle.flush()
            self._file_handle.close()
            self._file_handle = None

    # ── Handler interface ─────────────────────────────────────────

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._ensure_open()
            msg = self.format(record)
            self._file_handle.write(msg + "\n")
            self._file_handle.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self._close()
        super().close()


# ── Log retention: prune old date folders ────────────────────────


def _prune_old_logs(log_dir: Path = _LOG_DIR, retention_days: int = _RETENTION_DAYS) -> None:
    """Remove date folders older than *retention_days*."""
    if not log_dir.exists():
        return
    cutoff = (datetime.date.today() - datetime.timedelta(days=retention_days)).isoformat()
    for child in sorted(log_dir.iterdir()):
        if not child.is_dir():
            continue
        # Only prune directories that look like dates (YYYY-MM-DD)
        if len(child.name) == 10 and child.name < cutoff:
            shutil.rmtree(child, ignore_errors=True)


# ── Public API: get_logger ───────────────────────────────────────


def get_logger(name: str = "rag") -> logging.Logger:
    """Get a logger that outputs to stderr **and** date-folder/level-split files.

    * **stderr** — all messages at INFO and above.
    * ``logs/<date>/info.log`` — INFO messages.
    * ``logs/<date>/warning.log`` — WARNING messages.
    * ``logs/<date>/error.log`` — ERROR and CRITICAL messages.

    Old date folders beyond 30 days are pruned on first call.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _prune_old_logs()

    # 1. Console -> stderr
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_FMT)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # 2. Per-level file handlers
    for level in _LEVEL_FILES:
        handler = _DateLevelFileHandler(level, log_dir=_LOG_DIR)
        handler.setFormatter(_FMT)
        handler.setLevel(level)
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
_DEFAULT_TRACE_FILE = _LOG_DIR / "traces" / "traces.jsonl"


def get_trace_logger(
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Get a logger that writes JSON Lines to *log_file*.

    Args:
        log_file: Path to the ``.jsonl`` file.  Defaults to
            ``logs/traces/traces.jsonl``.
    """
    logger = logging.getLogger(_TRACE_LOGGER_NAME)

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
    """Persist a trace dictionary as a single JSON Line."""
    path = Path(log_file) if log_file else Path(_DEFAULT_TRACE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)

    envelope = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "level": "INFO",
        "logger": _TRACE_LOGGER_NAME,
        "message": json.dumps(trace_dict, ensure_ascii=False, default=str),
    }
    line = json.dumps(envelope, ensure_ascii=False, default=str)

    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
