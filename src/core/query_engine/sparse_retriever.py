"""SparseRetriever — keyword-based retrieval via BM25 index.

Loads a pre-built BM25 index from disk, queries it with keywords,
and enriches results with text/metadata from the vector store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.types import RetrievalResult
from ingestion.storage.bm25_indexer import BM25Indexer

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext
    from libs.vector_store.base_vector_store import BaseVectorStore


class SparseRetriever:
    """Retrieve chunks by BM25 keyword matching.

    Args:
        settings: Application settings.
        bm25_indexer: Optional pre-created BM25Indexer (for testing).
        store: Optional pre-created vector store (for text/metadata lookup).
        index_dir: Directory where BM25 indices are stored.
    """

    def __init__(
        self,
        settings: Settings,
        bm25_indexer: BM25Indexer | None = None,
        store: BaseVectorStore | None = None,
        index_dir: str = "data/db/bm25",
    ) -> None:
        self._bm25 = bm25_indexer or BM25Indexer(index_dir=index_dir)
        self._store = store
        self._settings = settings

    def retrieve(
        self,
        keywords: list[str],
        *,
        top_k: int = 10,
        collection: str = "default",
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        """Query the BM25 index and enrich results with text/metadata.

        Args:
            keywords: Pre-tokenized query terms (from QueryProcessor).
            top_k: Number of results to return.
            collection: Collection name for the BM25 index.
            trace: Optional trace context.

        Returns:
            List of RetrievalResult sorted by BM25 score descending.
        """
        if not keywords:
            return []

        # Load index for this collection
        self._bm25.load(collection)

        # Query BM25: returns [(chunk_id, score), ...]
        scored = self._bm25.query(keywords, top_k=top_k)

        if not scored:
            return []

        # Enrich with text/metadata from vector store if available
        chunk_ids = [cid for cid, _ in scored]
        score_map = {cid: score for cid, score in scored}

        if self._store is not None:
            enriched = self._store.get_by_ids(chunk_ids, collection=collection)
            enriched_map = {rec["id"]: rec for rec in enriched}
        else:
            enriched_map: dict[str, dict[str, Any]] = {}

        results: list[RetrievalResult] = []
        for cid, score in scored:
            info = enriched_map.get(cid, {})
            results.append(
                RetrievalResult(
                    chunk_id=cid,
                    score=score,
                    text=info.get("text", ""),
                    metadata=info.get("metadata", {}),
                    source="sparse",
                )
            )

        return results
