"""Unit tests for Splitter base and factory (B3)."""

from typing import Any

import pytest

from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory
from core.settings import Settings, LLMSettings, EmbeddingSettings, VectorStoreSettings, RetrievalSettings, SplitterSettings


class FakeSplitter(BaseSplitter):
    def __init__(self, settings: Any = None) -> None:
        self._settings = settings

    def split_text(self, text: str, *, trace: Any | None = None) -> list[str]:
        return text.split("\n\n")


def _settings(provider: str = "fake") -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
        splitter=SplitterSettings(provider=provider),
    )


class TestBaseSplitter:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseSplitter()  # type: ignore[abstract]


class TestFakeSplitter:
    def test_splits_on_double_newline(self) -> None:
        s = FakeSplitter()
        result = s.split_text("aaa\n\nbbb\n\nccc")
        assert result == ["aaa", "bbb", "ccc"]

    def test_single_paragraph(self) -> None:
        s = FakeSplitter()
        assert s.split_text("just one") == ["just one"]


class TestSplitterFactory:
    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown splitter"):
            SplitterFactory.create(_settings("bogus"))

    def test_register_and_create(self) -> None:
        SplitterFactory.register_provider(
            "fake", f"{FakeSplitter.__module__}.{FakeSplitter.__qualname__}"
        )
        s = SplitterFactory.create(_settings("fake"))
        assert isinstance(s, FakeSplitter)
        assert s.split_text("a\n\nb") == ["a", "b"]
