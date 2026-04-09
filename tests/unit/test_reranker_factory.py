"""Unit tests for Reranker base, NoneReranker, and factory (B5)."""

from typing import Any

import pytest

from libs.reranker.base_reranker import BaseReranker, NoneReranker, RerankCandidate
from libs.reranker.reranker_factory import RerankerFactory
from core.settings import Settings, LLMSettings, EmbeddingSettings, VectorStoreSettings, RetrievalSettings, RerankSettings


def _candidates(n: int = 3) -> list[RerankCandidate]:
    return [RerankCandidate(id=f"c{i}", text=f"text {i}", score=float(i)) for i in range(n)]


class TestRerankCandidate:
    def test_fields(self) -> None:
        c = RerankCandidate(id="1", text="hello", score=0.5, metadata={"k": "v"})
        assert c.metadata["k"] == "v"


class TestNoneReranker:
    def test_preserves_order(self) -> None:
        r = NoneReranker()
        cands = _candidates(3)
        result = r.rerank("query", cands)
        assert [c.id for c in result] == ["c0", "c1", "c2"]

    def test_top_k(self) -> None:
        r = NoneReranker()
        result = r.rerank("query", _candidates(5), top_k=2)
        assert len(result) == 2

    def test_empty_candidates(self) -> None:
        r = NoneReranker()
        assert r.rerank("query", []) == []


class TestRerankerFactory:
    def _settings(self, backend: str) -> Settings:
        return Settings(
            llm=LLMSettings(provider="openai", model="gpt-4o"),
            embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
            vector_store=VectorStoreSettings(backend="chroma"),
            retrieval=RetrievalSettings(sparse_backend="bm25"),
            rerank=RerankSettings(backend=backend),
        )

    def test_none_backend_returns_none_reranker(self) -> None:
        r = RerankerFactory.create(self._settings("none"))
        assert isinstance(r, NoneReranker)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown reranker"):
            RerankerFactory.create(self._settings("bogus"))

    def test_base_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseReranker()  # type: ignore[abstract]
