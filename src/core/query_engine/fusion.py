"""RRF (Reciprocal Rank Fusion) — merge dense and sparse retrieval results.

Combines ranked lists from multiple retrievers into a single ranking
using the RRF formula: ``score(d) = Σ 1 / (k + rank_i(d))``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from core.types import RetrievalResult

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    *,
    k: int = 60,
    top_k: int = 10,
    trace: TraceContext | None = None,
) -> list[RetrievalResult]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: One or more ranked lists of RetrievalResult.
            Each list is assumed sorted by score descending.
        k: RRF constant (default 60). Higher k dampens the effect of
            individual rankings.
        top_k: Number of fused results to return.
        trace: Optional trace context.

    Returns:
        Fused list of RetrievalResult sorted by RRF score descending.
    """
    if not result_lists:
        return []

    # Accumulate RRF scores per chunk_id
    rrf_scores: dict[str, float] = defaultdict(float)
    # Keep the best RetrievalResult per chunk_id (from highest-ranked list)
    best_results: dict[str, RetrievalResult] = {}

    for ranked_list in result_lists:
        for rank, result in enumerate(ranked_list, start=1):
            cid = result.chunk_id
            rrf_scores[cid] += 1.0 / (k + rank)
            if cid not in best_results:
                best_results[cid] = result

    # Sort by fused RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Build final results with fused score
    fused: list[RetrievalResult] = []
    for cid in sorted_ids[:top_k]:
        base = best_results[cid]
        fused.append(
            RetrievalResult(
                chunk_id=cid,
                score=round(rrf_scores[cid], 6),
                text=base.text,
                metadata=base.metadata,
                source="fusion",
            )
        )

    return fused
