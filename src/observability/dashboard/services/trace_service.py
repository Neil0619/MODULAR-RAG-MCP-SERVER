"""TraceService — read and parse ``logs/traces.jsonl`` for the Dashboard.

Each line in the JSONL file is a JSON *envelope* with a ``message`` field
containing the serialised :class:`TraceContext` dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Default path — matches observability.logger._DEFAULT_TRACE_FILE
_DEFAULT_TRACE_FILE = "logs/traces.jsonl"


class TraceService:
    """Read traces from a JSONL file and expose them as dicts.

    Args:
        trace_file: Path to the ``.jsonl`` file.
            Defaults to ``logs/traces.jsonl``.
    """

    def __init__(self, trace_file: str = _DEFAULT_TRACE_FILE) -> None:
        self._path = Path(trace_file)

    # ------------------------------------------------------------------
    # Core reads
    # ------------------------------------------------------------------

    def _read_all(self) -> list[dict[str, Any]]:
        """Parse every line in the trace file and return parsed dicts."""
        if not self._path.exists():
            return []

        traces: list[dict[str, Any]] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    envelope = json.loads(line)
                    # The actual trace data is inside envelope["message"]
                    msg = envelope.get("message", "{}")
                    if isinstance(msg, str):
                        trace_data = json.loads(msg)
                    else:
                        trace_data = msg
                    traces.append(trace_data)
                except (json.JSONDecodeError, TypeError):
                    continue
        return traces

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_traces(
        self,
        trace_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return traces, newest first, optionally filtered by ``trace_type``.

        Args:
            trace_type: Filter to ``"query"`` or ``"ingestion"``.
                ``None`` returns all.
            limit: Maximum number of traces to return.
        """
        traces = self._read_all()
        if trace_type is not None:
            traces = [t for t in traces if t.get("trace_type") == trace_type]
        return traces[-limit:][::-1]  # newest first

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Find a single trace by ID."""
        for t in self._read_all():
            if t.get("trace_id") == trace_id:
                return t
        return None

    def get_stage_elapsed(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract ``[{name, elapsed_ms}]`` from a trace's stages dict.

        Returns stages sorted by their recorded order (insertion order).
        """
        stages = trace.get("stages", {})
        result: list[dict[str, Any]] = []
        for name, data in stages.items():
            if isinstance(data, dict):
                result.append({
                    "stage": name,
                    "elapsed_ms": data.get("elapsed_ms", 0.0),
                    **{k: v for k, v in data.items() if k != "elapsed_ms"},
                })
        return result
