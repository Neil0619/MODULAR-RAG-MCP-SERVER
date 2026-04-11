"""TraceContext — per-request trace for observability.

Supports two trace types: ``"query"`` and ``"ingestion"``.  Each stage
records elapsed time and arbitrary detail data.  Call :meth:`finish` to
mark the trace complete, then :meth:`to_dict` to get a JSON-serialisable
snapshot.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceContext:
    """Per-request trace context for tracking pipeline stages."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_type: str = "query"  # "query" | "ingestion"
    stages: dict[str, Any] = field(default_factory=dict)

    # Internal timing — not part of the public API
    _started_at: float = field(default_factory=time.time, repr=False)
    _finished_at: float | None = field(default=None, repr=False)
    _stage_starts: dict[str, float] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_stage(
        self,
        name: str,
        data: Any = None,
        *,
        elapsed_ms: float | None = None,
    ) -> None:
        """Record a pipeline stage.

        Args:
            name: Stage name (e.g. ``"load"``, ``"dense_retrieval"``).
            data: Arbitrary detail dict associated with the stage.
            elapsed_ms: Explicit elapsed time in ms.  When *None* the
                context will auto-compute elapsed from the last call to
                :meth:`start_stage` with the same *name*.
        """
        if elapsed_ms is None and name in self._stage_starts:
            elapsed_ms = (time.time() - self._stage_starts[name]) * 1000

        entry: dict[str, Any] = {}
        if data is not None:
            # Merge existing data if stage recorded multiple times
            if name in self.stages and isinstance(self.stages[name], dict):
                entry = dict(self.stages[name])
                if isinstance(data, dict):
                    entry.update(data)
                else:
                    entry["data"] = data
            else:
                if isinstance(data, dict):
                    entry = dict(data)
                else:
                    entry = {"data": data}
        elif name in self.stages and isinstance(self.stages[name], dict):
            entry = dict(self.stages[name])

        if elapsed_ms is not None:
            entry["elapsed_ms"] = round(elapsed_ms, 3)

        self.stages[name] = entry

    def start_stage(self, name: str) -> None:
        """Mark the beginning of a stage for auto-elapsed computation."""
        self._stage_starts[name] = time.time()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def finish(self) -> None:
        """Mark the trace as finished and record total elapsed time."""
        self._finished_at = time.time()

    @property
    def started_at(self) -> float:
        """Epoch timestamp when the trace was created."""
        return self._started_at

    @property
    def finished_at(self) -> float | None:
        """Epoch timestamp when :meth:`finish` was called, or *None*."""
        return self._finished_at

    # ------------------------------------------------------------------
    # Elapsed helpers
    # ------------------------------------------------------------------

    def elapsed_ms(self, stage_name: str | None = None) -> float:
        """Return elapsed time in milliseconds.

        Args:
            stage_name: If given, return elapsed for that specific stage.
                Otherwise return total elapsed since trace creation.

        Returns:
            Elapsed time in ms (rounded to 3 decimal places).
        """
        if stage_name is not None:
            stage = self.stages.get(stage_name, {})
            if isinstance(stage, dict) and "elapsed_ms" in stage:
                return stage["elapsed_ms"]
            if stage_name in self._stage_starts:
                return round((time.time() - self._stage_starts[stage_name]) * 1000, 3)
            return 0.0

        end = self._finished_at or time.time()
        return round((end - self._started_at) * 1000, 3)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the trace to a JSON-safe dictionary."""
        stages_copy: dict[str, Any] = {}
        for k, v in self.stages.items():
            stages_copy[k] = dict(v) if isinstance(v, dict) else v

        result: dict[str, Any] = {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "total_elapsed_ms": self.elapsed_ms(),
            "stages": stages_copy,
        }
        if self._finished_at is not None:
            result["finished_at"] = self._finished_at
        return result
