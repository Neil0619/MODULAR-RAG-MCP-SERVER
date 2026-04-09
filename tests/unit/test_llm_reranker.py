"""Unit tests for LLMReranker."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.settings import Settings
from libs.llm.base_llm import BaseLLM, ChatResponse
from libs.reranker.base_reranker import RerankCandidate
from libs.reranker.llm_reranker import LLMReranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidates(n: int = 4) -> list[RerankCandidate]:
    return [
        RerankCandidate(id=f"c-{i}", text=f"Candidate text {i}", score=0.0, metadata={"idx": i})
        for i in range(n)
    ]


def _fake_llm(response_content: str) -> MagicMock:
    """Create a mock BaseLLM whose ``chat_str`` returns *response_content*."""
    llm = MagicMock(spec=BaseLLM)
    llm.chat_str.return_value = response_content
    return llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLLMReranker:
    """Tests for LLMReranker."""

    def test_returns_reranked_candidates(self) -> None:
        llm = _fake_llm("[2, 0, 3, 1]")
        reranker = LLMReranker(Settings(), llm)
        candidates = _candidates(4)

        result = reranker.rerank("test query", candidates)

        assert len(result) == 4
        # First should be the one originally at index 2
        assert result[0].id == "c-2"
        assert result[0].score > result[1].score
        # Last (unmentioned original-order append) should still be present
        assert result[0].score == 4.0
        assert result[1].score == 3.0

    def test_invalid_llm_response_falls_back_to_original_order(self) -> None:
        llm = _fake_llm("I cannot rank these documents because garbled output")
        reranker = LLMReranker(Settings(), llm)
        candidates = _candidates(3)

        result = reranker.rerank("test query", candidates)

        assert len(result) == 3
        ids = [r.id for r in result]
        assert ids == ["c-0", "c-1", "c-2"]

    def test_empty_candidates_returns_empty(self) -> None:
        llm = _fake_llm("[0]")
        reranker = LLMReranker(Settings(), llm)

        result = reranker.rerank("test query", [])

        assert result == []

    def test_top_k_limits_results(self) -> None:
        llm = _fake_llm("[3, 1, 0, 2]")
        reranker = LLMReranker(Settings(), llm)
        candidates = _candidates(4)

        result = reranker.rerank("test query", candidates, top_k=2)

        assert len(result) == 2
        assert result[0].id == "c-3"
        assert result[1].id == "c-1"

    def test_llm_exception_falls_back(self) -> None:
        llm = _fake_llm("")
        llm.chat_str.side_effect = RuntimeError("LLM unavailable")
        reranker = LLMReranker(Settings(), llm)
        candidates = _candidates(3)

        result = reranker.rerank("test query", candidates)

        assert len(result) == 3
        assert [r.id for r in result] == ["c-0", "c-1", "c-2"]

    def test_out_of_range_indices_are_ignored(self) -> None:
        llm = _fake_llm("[0, 99, 1]")
        reranker = LLMReranker(Settings(), llm)
        candidates = _candidates(2)

        result = reranker.rerank("test query", candidates)

        # Index 99 is out of range so only 0 and 1 are ranked
        ranked_ids = [r.id for r in result if r.score > 0.0]
        assert "c-0" in ranked_ids
        assert "c-1" in ranked_ids
