"""ConfigService — reads Settings and formats component info for Dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.settings import Settings, load_settings


@dataclass
class ComponentInfo:
    """Single pluggable component summary."""

    name: str
    provider: str
    model: str = ""
    details: dict[str, Any] | None = None


@dataclass
class SystemOverview:
    """Aggregated system overview for the Dashboard."""

    components: list[ComponentInfo]
    config_path: str = ""


class ConfigService:
    """Read Settings and format for Dashboard display."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    def get_overview(self) -> SystemOverview:
        """Return a SystemOverview with all pluggable component info."""
        s = self.settings
        components = [
            ComponentInfo(
                name="LLM",
                provider=s.llm.provider,
                model=s.llm.model,
                details={"api_key_set": bool(s.llm.api_key)},
            ),
            ComponentInfo(
                name="Embedding",
                provider=s.embedding.provider,
                model=s.embedding.model,
                details={"api_key_set": bool(s.embedding.api_key)},
            ),
            ComponentInfo(
                name="Splitter",
                provider="recursive",
                model="",
                details={
                    "chunk_size": s.splitter.chunk_size,
                    "chunk_overlap": s.splitter.chunk_overlap,
                },
            ),
            ComponentInfo(
                name="Vector Store",
                provider=s.vector_store.backend,
                model="",
                details={"persist_path": s.vector_store.persist_path},
            ),
            ComponentInfo(
                name="Reranker",
                provider=s.rerank.backend,
                model=s.rerank.model,
                details={"top_m": s.rerank.top_m},
            ),
        ]
        return SystemOverview(components=components)

    def get_component_table(self) -> list[dict[str, str]]:
        """Return a list of dicts suitable for st.dataframe."""
        overview = self.get_overview()
        rows = []
        for c in overview.components:
            rows.append({
                "Component": c.name,
                "Provider": c.provider,
                "Model": c.model,
            })
        return rows
