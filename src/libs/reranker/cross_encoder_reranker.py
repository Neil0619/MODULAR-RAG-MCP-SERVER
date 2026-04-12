"""Cross-encoder reranker using sentence-transformers."""

from __future__ import annotations

import logging
from typing import Any

from core.settings import Settings
from libs.reranker.base_reranker import BaseReranker, RerankCandidate
from observability.logger import get_logger as _get_logger

logger = _get_logger("rag.reranker.cross_encoder")


class CrossEncoderReranker(BaseReranker):
    """Reranker backed by a ``sentence-transformers`` CrossEncoder model."""

    def __init__(self, settings: Settings) -> None:
        self._model_name: str = settings.rerank.model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._model = self._load_model()

    def _load_model(self):  # noqa: ANN202
        """Lazy-import and instantiate the CrossEncoder."""
        try:
            from sentence_transformers import CrossEncoder

            return CrossEncoder(self._model_name)
        except Exception:
            logger.exception("Failed to load CrossEncoder model %s", self._model_name)
            return None

    # ------------------------------------------------------------------
    # BaseReranker interface
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_k: int | None = None,
        trace: Any | None = None,
    ) -> list[RerankCandidate]:
        if not candidates:
            return []

        if self._model is None:
            logger.warning("CrossEncoder model unavailable; returning original order")
            result = list(candidates)
            if top_k is not None:
                result = result[:top_k]
            return result

        try:
            return self._do_rerank(query, candidates, top_k)
        except Exception:
            logger.exception("CrossEncoder reranking failed; returning original order")
            result = list(candidates)
            if top_k is not None:
                result = result[:top_k]
            return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: int | None,
    ) -> list[RerankCandidate]:
        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)

        scored: list[tuple[float, RerankCandidate]] = [
            (float(s), c) for s, c in zip(scores, candidates)
        ]
        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[RerankCandidate] = []
        for rank, (score, cand) in enumerate(scored):
            results.append(
                RerankCandidate(
                    id=cand.id,
                    text=cand.text,
                    score=score,
                    metadata=cand.metadata,
                )
            )

        if top_k is not None:
            results = results[:top_k]

        return results
