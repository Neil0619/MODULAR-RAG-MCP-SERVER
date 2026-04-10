"""VectorUpserter — write dense vectors to the vector store.

Receives ChunkRecords with dense vectors, generates deterministic IDs,
and upserts them into the vector store via ``BaseVectorStore``.
Same content always produces the same ID, ensuring idempotency.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from core.types import ChunkRecord
from libs.vector_store.base_vector_store import VectorRecord
from libs.vector_store.vector_store_factory import VectorStoreFactory

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext


def _generate_stable_id(record: ChunkRecord) -> str:
    """Generate a deterministic ID from source_path + chunk_index + content hash.

    Format: ``{source_path_hash}_{chunk_index}_{content_hash[:8]}``
    """
    source = record.metadata.get("source_path", "")
    chunk_index = record.metadata.get("chunk_index", 0)
    content_hash = hashlib.sha256(record.text.encode()).hexdigest()[:8]
    source_hash = hashlib.sha256(source.encode()).hexdigest()[:8]
    return f"{source_hash}_{chunk_index:04d}_{content_hash}"


class VectorUpserter:
    """Upsert ChunkRecords with dense vectors into the vector store.

    Args:
        settings: Application settings (used to create the vector store).
        store: Optional pre-created vector store instance.
    """

    def __init__(
        self,
        settings: Settings,
        store: Any | None = None,
    ) -> None:
        if store is not None:
            self._store = store
        else:
            self._store = VectorStoreFactory.create(settings)

    def upsert(
        self,
        records: list[ChunkRecord],
        collection: str = "default",
        trace: TraceContext | None = None,
    ) -> int:
        """Upsert ChunkRecords into the vector store.

        Args:
            records: Records with ``dense_vector`` populated.
            collection: Collection name.
            trace: Optional trace context.

        Returns:
            Number of records upserted.

        Raises:
            ValueError: If records is empty.
        """
        if not records:
            raise ValueError("records must not be empty")

        vector_records: list[VectorRecord] = []
        for rec in records:
            if rec.dense_vector is None:
                continue

            stable_id = _generate_stable_id(rec)
            vector_records.append(
                VectorRecord(
                    id=stable_id,
                    vector=rec.dense_vector,
                    metadata=rec.metadata,
                    text=rec.text,
                )
            )

        if not vector_records:
            return 0

        return self._store.upsert(vector_records, collection=collection, trace=trace)

    def get_stable_id(self, record: ChunkRecord) -> str:
        """Get the stable ID for a record (useful for testing)."""
        return _generate_stable_id(record)
