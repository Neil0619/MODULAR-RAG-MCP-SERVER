"""Minimal TraceContext for observability (Phase F will expand)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceContext:
    """Lightweight trace context for tracking pipeline stages.

    Phase F will replace this with a full-featured implementation.
    """

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    stages: dict[str, Any] = field(default_factory=dict)

    def record_stage(self, name: str, data: Any = None) -> None:
        """Record a pipeline stage."""
        self.stages[name] = data
