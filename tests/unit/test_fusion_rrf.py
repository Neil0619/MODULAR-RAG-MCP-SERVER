"""Unit tests for RRF Fusion (D4)."""

from __future__ import annotations

import pytest

from core.query_engine.fusion import reciprocal_rank_fusion
from core.types import RetrievalResult


def _rr(cid: str, score: float, source: str = "") -> RetrievalResult:
    return RetrievalResult(chunk_id=cid, score=score, text=f"text_{cid}", metadata={}, source=source)


class TestReciprocalRankFusion:
    """Tests for RRF fusion."""

    def test_empty_input(self) -> None:
        assert reciprocal_rank_fusion([]) == []

    def test_single_list(self) -> None:
        dense = [_rr("a", 0.9, "dense"), _rr("b", 0.8, "dense"), _rr("c", 0.7, "dense")]
        result = reciprocal_rank_fusion([dense])
        assert len(result) == 3
        assert result[0].chunk_id == "a"  # rank 1 → highest RRF
        assert result[0].source == "fusion"

    def test_two_lists_basic(self) -> None:
        dense = [_rr("a", 0.95), _rr("b", 0.85), _rr("c", 0.75)]
        sparse = [_rr("b", 3.0), _rr("c", 2.0), _rr("d", 1.0)]
        result = reciprocal_rank_fusion([dense, sparse], k=60)

        # "b" appears at rank 2 (dense) + rank 1 (sparse)
        # "c" appears at rank 3 (dense) + rank 2 (sparse)
        # "a" appears at rank 1 (dense) only
        # "d" appears at rank 3 (sparse) only
        # b: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.03226
        # a: 1/(60+1) = 0.01639
        # c: 1/(60+3) + 1/(60+2) = 1/63 + 1/62 ≈ 0.03199
        # So order should be b > c > a > d
        ids = [r.chunk_id for r in result]
        assert ids[0] == "b"
        assert ids[1] == "c"

    def test_deterministic(self) -> None:
        dense = [_rr("x", 0.9), _rr("y", 0.8)]
        sparse = [_rr("y", 3.0), _rr("x", 2.0)]
        r1 = reciprocal_rank_fusion([dense, sparse])
        r2 = reciprocal_rank_fusion([dense, sparse])
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]
        assert [r.score for r in r1] == [r.score for r in r2]

    def test_k_parameter(self) -> None:
        dense = [_rr("a", 0.9), _rr("b", 0.8)]
        sparse = [_rr("b", 3.0), _rr("a", 2.0)]

        r_k60 = reciprocal_rank_fusion([dense, sparse], k=60)
        r_k1 = reciprocal_rank_fusion([dense, sparse], k=1)

        # With smaller k, rank differences matter more
        # Both should still have 2 results
        assert len(r_k60) == 2
        assert len(r_k1) == 2
        # Scores should differ
        assert r_k60[0].score != r_k1[0].score

    def test_top_k_limit(self) -> None:
        dense = [_rr(f"d{i}", 0.9 - i * 0.1) for i in range(10)]
        result = reciprocal_rank_fusion([dense], top_k=3)
        assert len(result) == 3

    def test_deduplication(self) -> None:
        """Same chunk in both lists gets fused score."""
        dense = [_rr("x", 0.9)]
        sparse = [_rr("x", 5.0)]
        result = reciprocal_rank_fusion([dense, sparse])
        assert len(result) == 1
        assert result[0].chunk_id == "x"
        # Score is 1/(60+1) + 1/(60+1)
        expected = round(1.0 / 61 + 1.0 / 61, 6)
        assert result[0].score == expected

    def test_empty_list_in_lists(self) -> None:
        dense = [_rr("a", 0.9)]
        result = reciprocal_rank_fusion([dense, []])
        assert len(result) == 1
        assert result[0].chunk_id == "a"

    def test_all_empty_lists(self) -> None:
        result = reciprocal_rank_fusion([[], []])
        assert result == []

    def test_three_lists(self) -> None:
        l1 = [_rr("a", 0.9), _rr("b", 0.8)]
        l2 = [_rr("b", 3.0), _rr("c", 2.0)]
        l3 = [_rr("a", 0.5), _rr("c", 0.4)]
        result = reciprocal_rank_fusion([l1, l2, l3])
        ids = [r.chunk_id for r in result]
        # "a": rank 1 (l1) + rank 1 (l3) = strong
        # "b": rank 2 (l1) + rank 1 (l2) = strong
        # "c": rank 2 (l2) + rank 2 (l3) = weaker
        assert len(result) == 3
        assert "c" in ids[-1]  # c should be last

    def test_score_is_rounded(self) -> None:
        dense = [_rr("a", 0.9)]
        result = reciprocal_rank_fusion([dense])
        # score should be rounded to 6 decimal places
        assert result[0].score == round(result[0].score, 6)

    def test_result_source_is_fusion(self) -> None:
        dense = [_rr("a", 0.9, "dense")]
        sparse = [_rr("a", 3.0, "sparse")]
        result = reciprocal_rank_fusion([dense, sparse])
        assert result[0].source == "fusion"

    def test_text_preserved_from_first_occurrence(self) -> None:
        dense = [_rr("a", 0.9)]
        sparse = [RetrievalResult(chunk_id="a", score=3.0, text="different", metadata={}, source="sparse")]
        result = reciprocal_rank_fusion([dense, sparse])
        # Should use the text from whichever list first encounters the ID
        assert result[0].text == "text_a"  # from dense (first list)
