"""DenseRetriever — semantic retrieval via embedding + vector store query.

Embeds the user query into a dense vector, queries the vector store,
and normalizes results into :class:`RetrievalResult` objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.types import RetrievalResult
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.vector_store_factory import VectorStoreFactory

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext
    from libs.embedding.base_embedding import BaseEmbedding
    from libs.vector_store.base_vector_store import BaseVectorStore


class DenseRetriever:
    """Retrieve chunks by dense vector similarity.

    Args:
        settings: Application settings (used to create embedding + vector store).
        embedding: Optional pre-created embedding client (for testing).
        store: Optional pre-created vector store (for testing).
    """

    def __init__(
        self,
        settings: Settings,
        embedding: BaseEmbedding | None = None,
        store: BaseVectorStore | None = None,
    ) -> None:
        self._embedding = embedding or EmbeddingFactory.create(settings)
        self._store = store or VectorStoreFactory.create(settings)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        collection: str = "default",
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        """Embed the query and retrieve top-k similar chunks.

        Args:
            query: Raw query string.
            top_k: Number of results to return.
            filters: Optional metadata filters.
            collection: Collection to search.
            trace: Optional trace context.

        Returns:
            List of RetrievalResult sorted by score descending.
        """
        vectors = self._embedding.embed([query])
        query_vector = vectors[0]

        query_results = self._store.query(
            query_vector,
            top_k=top_k,
            filters=filters,
            collection=collection,
            trace=trace,
        )

        return [
            RetrievalResult(
                chunk_id=r.id,
                score=r.score,
                text=r.text,
                metadata=r.metadata,
                source="dense",
            )
            for r in query_results
        ]
