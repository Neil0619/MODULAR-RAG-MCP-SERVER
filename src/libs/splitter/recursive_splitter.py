"""Recursive character text splitter using LangChain under the hood."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.settings import Settings
from libs.splitter.base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """Wraps LangChain's ``RecursiveCharacterTextSplitter``."""

    def __init__(self, settings: Settings) -> None:
        self._chunk_size: int = settings.splitter.chunk_size
        self._chunk_overlap: int = settings.splitter.chunk_overlap
        self._separators: list[str] = list(settings.splitter.separators)

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=self._separators,
        )

    # ------------------------------------------------------------------
    # BaseSplitter interface
    # ------------------------------------------------------------------

    def split_text(
        self,
        text: str,
        *,
        trace: Any | None = None,
    ) -> list[str]:
        """Split *text* into chunks using recursive character splitting."""
        if not text:
            return []
        return self._splitter.split_text(text)
