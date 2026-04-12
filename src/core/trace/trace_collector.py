"""TraceCollector — collect finished traces and persist to JSON Lines.

When :meth:`collect` is called, the trace is serialised and written
to ``logs/traces.jsonl`` via :func:`observability.logger.write_trace`.
Collected traces are also kept in memory for callers that need
synchronous access (e.g. tests).
"""

from __future__ import annotations

import logging
from typing import Any

from core.trace.trace_context import TraceContext
from observability.logger import get_logger as _get_logger

logger = _get_logger("rag.trace.collector")


class TraceCollector:
    """Collects finished :class:`TraceContext` objects and persists them.

    Usage::

        collector = TraceCollector()
        collector.collect(trace)       # auto-finishes + writes to traces.jsonl
    """

    def __init__(self, *, persist: bool = True) -> None:
        self._traces: list[dict[str, Any]] = []
        self._persist = persist

    def collect(self, trace: TraceContext) -> None:
        """Collect a finished trace and persist it.

        If the trace has not been finished, :meth:`finish` is called
        automatically before serialisation.  The serialised dict is then
        written to ``logs/traces.jsonl`` (unless *persist* was set to
        ``False`` at construction time).
        """
        if trace._finished_at is None:
            trace.finish()

        trace_dict = trace.to_dict()
        self._traces.append(trace_dict)
        logger.debug("Trace collected: %s (%s)", trace.trace_id, trace.trace_type)

        if self._persist:
            from observability.logger import write_trace
            write_trace(trace_dict)

    @property
    def traces(self) -> list[dict[str, Any]]:
        """Return all collected trace dicts."""
        return list(self._traces)

    def clear(self) -> None:
        """Remove all collected traces."""
        self._traces.clear()
