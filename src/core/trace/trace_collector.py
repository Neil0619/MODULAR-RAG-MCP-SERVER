"""TraceCollector — collect finished traces and trigger persistence.

Phase F2 will connect this to the JSON Lines logger.  For now the
collector simply accumulates traces in memory so tests can verify
the data flow.
"""

from __future__ import annotations

import logging
from typing import Any

from core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


class TraceCollector:
    """Collects finished :class:`TraceContext` objects.

    Usage::

        collector = TraceCollector()
        collector.collect(trace)       # trace.finish() should already be called
    """

    def __init__(self) -> None:
        self._traces: list[dict[str, Any]] = []

    def collect(self, trace: TraceContext) -> None:
        """Collect a finished trace.

        If the trace has not been finished, :meth:`finish` is called
        automatically before serialisation.
        """
        if trace._finished_at is None:
            trace.finish()

        trace_dict = trace.to_dict()
        self._traces.append(trace_dict)
        logger.debug("Trace collected: %s (%s)", trace.trace_id, trace.trace_type)

    @property
    def traces(self) -> list[dict[str, Any]]:
        """Return all collected trace dicts."""
        return list(self._traces)

    def clear(self) -> None:
        """Remove all collected traces."""
        self._traces.clear()
