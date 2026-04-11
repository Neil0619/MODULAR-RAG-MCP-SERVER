"""HybridSearch — orchestrate dense + sparse retrieval with RRF fusion.

Combines QueryProcessor, DenseRetriever, SparseRetriever, and RRF Fusion
into a single ``search()`` call. Includes metadata post-filtering and
graceful degradation when one retriever fails.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import reciprocal_rank_fusion
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.sparse_retriever import SparseRetriever
from core.types import RetrievalResult
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.vector_store_factory import VectorStoreFactory

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


class HybridSearch:
    """Hybrid search combining dense + sparse retrieval with RRF fusion.

    Args:
        settings: Application settings.
        query_processor: Optional pre-created QueryProcessor.
        dense_retriever: Optional pre-created DenseRetriever.
        sparse_retriever: Optional pre-created SparseRetriever.
    """

    def __init__(
        self,
        settings: Settings,
        query_processor: QueryProcessor | None = None,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
    ) -> None:
        self._settings = settings
        self._query_processor = query_processor or QueryProcessor()

        if dense_retriever is not None:
            self._dense = dense_retriever
        else:
            self._dense = DenseRetriever(settings)

        if sparse_retriever is not None:
            self._sparse = sparse_retriever
        else:
            self._sparse = SparseRetriever(settings)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        collection: str = "default",
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        """Execute a hybrid search: query process → dense + sparse → fusion → filter.

        Args:
            query: Raw user query string.
            top_k: Final number of results to return.
            filters: Optional metadata filters.
            collection: Collection to search.
            trace: Optional trace context.

        Returns:
            List of RetrievalResult sorted by fused score descending.
        """
        # Stage 1: Query processing
        if trace:
            trace.start_stage("query_processing")
        processed = self._query_processor.process(query, filters=filters)
        if trace:
            trace.record_stage("query_processing", {
                "method": "rule_based",
                "keywords": processed.keywords,
                "original_query": processed.original_query,
            })

        retrieval_cfg = self._settings.retrieval
        dense_top_k = retrieval_cfg.top_k_dense
        sparse_top_k = retrieval_cfg.top_k_sparse

        # Stage 2: Dense + Sparse retrieval (with graceful degradation)
        result_lists: list[list[RetrievalResult]] = []

        if trace:
            trace.start_stage("dense_retrieval")
        dense_results = self._safe_dense_retrieve(
            processed.original_query,
            top_k=dense_top_k,
            filters=filters,
            collection=collection,
            trace=trace,
        )
        if dense_results:
            result_lists.append(dense_results)
        if trace:
            trace.record_stage("dense_retrieval", {
                "method": "embedding",
                "hit_count": len(dense_results),
            })

        if trace:
            trace.start_stage("sparse_retrieval")
        sparse_results = self._safe_sparse_retrieve(
            processed.keywords,
            top_k=sparse_top_k,
            collection=collection,
            trace=trace,
        )
        if sparse_results:
            result_lists.append(sparse_results)
        if trace:
            trace.record_stage("sparse_retrieval", {
                "method": "bm25",
                "hit_count": len(sparse_results),
            })

        if not result_lists:
            return []

        # Stage 3: RRF Fusion
        if trace:
            trace.start_stage("fusion")
        fused = reciprocal_rank_fusion(
            result_lists,
            k=retrieval_cfg.rrf_k,
            top_k=top_k * 3,  # over-fetch before filtering
        )
        if trace:
            trace.record_stage("fusion", {
                "algorithm": "rrf",
                "rrf_k": retrieval_cfg.rrf_k,
                "result_count": len(fused),
            })

        # Stage 4: Metadata post-filtering
        if filters:
            fused = self._apply_metadata_filters(fused, filters)

        return fused[:top_k]

    def _safe_dense_retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None,
        collection: str,
        trace: TraceContext | None,
    ) -> list[RetrievalResult]:
        """Dense retrieval with graceful error handling."""
        try:
            return self._dense.retrieve(
                query,
                top_k=top_k,
                filters=filters,
                collection=collection,
                trace=trace,
            )
        except Exception:
            logger.warning("Dense retrieval failed, degrading to sparse-only", exc_info=True)
            return []

    def _safe_sparse_retrieve(
        self,
        keywords: list[str],
        *,
        top_k: int,
        collection: str,
        trace: TraceContext | None,
    ) -> list[RetrievalResult]:
        """Sparse retrieval with graceful error handling."""
        try:
            return self._sparse.retrieve(
                keywords,
                top_k=top_k,
                collection=collection,
                trace=trace,
            )
        except Exception:
            logger.warning("Sparse retrieval failed, degrading to dense-only", exc_info=True)
            return []

    @staticmethod
    def _apply_metadata_filters(
        candidates: list[RetrievalResult],
        filters: dict[str, Any],
    ) -> list[RetrievalResult]:
        """Post-filter results by metadata key-value pairs.

        Only applies filters for keys that exist in the result metadata.
        """
        if not filters:
            return candidates

        filtered: list[RetrievalResult] = []
        for r in candidates:
            match = True
            for key, value in filters.items():
                meta_val = r.metadata.get(key)
                if meta_val is None:
                    continue  # key not in metadata, don't filter
                if meta_val != value:
                    match = False
                    break
            if match:
                filtered.append(r)

        return filtered
