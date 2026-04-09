"""ChromaDB-backed vector store implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from core.settings import Settings
from libs.vector_store.base_vector_store import BaseVectorStore, QueryResult, VectorRecord


class ChromaStore(BaseVectorStore):
    """Persistent vector store backed by ChromaDB."""

    def __init__(self, settings: Settings) -> None:
        self._persist_path = Path(settings.vector_store.persist_path)
        self._persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_path))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Return an existing collection or create a new one."""
        return self._client.get_or_create_collection(name=name)

    # ------------------------------------------------------------------
    # BaseVectorStore interface
    # ------------------------------------------------------------------

    def upsert(
        self,
        records: list[VectorRecord],
        *,
        collection: str = "default",
        trace: Any | None = None,
    ) -> int:
        col = self._get_or_create_collection(collection)
        if not records:
            return 0

        ids = [r.id for r in records]
        embeddings = [r.vector for r in records]
        documents = [r.text for r in records]
        metadatas = [r.metadata for r in records]

        col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        return len(records)

    def query(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        collection: str = "default",
        trace: Any | None = None,
    ) -> list[QueryResult]:
        col = self._get_or_create_collection(collection)

        kwargs: dict[str, Any] = {
            "query_embeddings": [vector],
            "n_results": top_k,
        }
        if filters:
            kwargs["where"] = filters

        result = col.query(**kwargs)

        # ChromaDB returns lists of lists; we queried with a single embedding.
        ids_batch = result.get("ids", [[]])[0]
        distances_batch = result.get("distances", [[]])[0]
        documents_batch = result.get("documents", [[]])[0]
        metadatas_batch = result.get("metadatas", [[]])[0]

        results: list[QueryResult] = []
        for i, doc_id in enumerate(ids_batch):
            results.append(
                QueryResult(
                    id=doc_id,
                    score=1.0 - distances_batch[i] if distances_batch else 0.0,
                    text=documents_batch[i] if documents_batch else "",
                    metadata=metadatas_batch[i] if metadatas_batch else {},
                )
            )
        return results

    def get_by_ids(
        self,
        ids: list[str],
        *,
        collection: str = "default",
    ) -> list[dict[str, Any]]:
        col = self._get_or_create_collection(collection)
        if not ids:
            return []

        result = col.get(ids=ids)
        out: list[dict[str, Any]] = []
        for i, doc_id in enumerate(result.get("ids", [])):
            out.append(
                {
                    "id": doc_id,
                    "text": result["documents"][i] if result.get("documents") else "",
                    "metadata": result["metadatas"][i] if result.get("metadatas") else {},
                }
            )
        return out

    def delete_by_metadata(
        self,
        filter: dict[str, Any],
        *,
        collection: str = "default",
    ) -> int:
        col = self._get_or_create_collection(collection)
        # First count matching records
        peek = col.get(where=filter)
        count = len(peek.get("ids", []))
        if count:
            col.delete(where=filter)
        return count

    def get_collection_stats(
        self,
        collection: str = "default",
    ) -> dict[str, Any]:
        col = self._get_or_create_collection(collection)
        return {
            "name": collection,
            "doc_count": col.count(),
        }
