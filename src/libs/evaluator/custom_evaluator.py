"""Custom evaluator with hit_rate and MRR metrics."""

from __future__ import annotations

from libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """Lightweight evaluator computing hit_rate and MRR."""

    def __init__(self, settings: Any = None) -> None:
        pass

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        *,
        retrieved_texts: list[str] | None = None,
        answer: str | None = None,
        trace: Any | None = None,
    ) -> dict[str, float]:
        if not golden_ids:
            return {"hit_rate": 0.0, "mrr": 0.0, "precision": 0.0, "recall": 0.0}

        golden_set = set(golden_ids)

        # Hit rate: any golden ID in retrieved?
        hits = [1 if rid in golden_set else 0 for rid in retrieved_ids]
        hit_rate = float(any(hits))

        # MRR: reciprocal rank of first hit
        mrr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in golden_set:
                mrr = 1.0 / rank
                break

        # Precision / Recall
        n_relevant = sum(hits)
        precision = n_relevant / len(retrieved_ids) if retrieved_ids else 0.0
        recall = n_relevant / len(golden_set) if golden_set else 0.0

        return {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "precision": precision,
            "recall": recall,
        }


# Fix missing import
from typing import Any  # noqa: E402
