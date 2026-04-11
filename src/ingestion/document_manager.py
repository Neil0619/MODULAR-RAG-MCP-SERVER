"""DocumentManager — cross-storage document lifecycle management.

Coordinates list / delete / stats operations across four storage backends:

1. **ChromaStore** (vector store) — chunk vectors + metadata
2. **BM25Indexer** (sparse index) — inverted index entries
3. **ImageStorage** — image files + SQLite index
4. **FileIntegrityChecker** — ingestion history records
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import FileIntegrityChecker
from libs.vector_store.base_vector_store import BaseVectorStore


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class DocumentInfo:
    """Summary of an ingested document."""

    source_path: str
    collection: str
    chunk_count: int = 0
    image_count: int = 0
    doc_hash: str = ""


@dataclass
class DeleteResult:
    """Result of a cross-storage document deletion."""

    source_path: str
    collection: str
    chunks_deleted: int = 0
    bm25_postings_removed: int = 0
    images_deleted: int = 0
    integrity_removed: bool = False


@dataclass
class CollectionStats:
    """Aggregate statistics for a collection."""

    name: str
    document_count: int = 0
    chunk_count: int = 0
    image_count: int = 0


# ---------------------------------------------------------------------------
# DocumentManager
# ---------------------------------------------------------------------------


class DocumentManager:
    """Coordinate document lifecycle across storage backends.

    Args:
        chroma_store: Vector store (ChromaDB).
        bm25_indexer: BM25 inverted-index manager.
        image_storage: Image file + index storage.
        file_integrity: File-hash history tracker.
    """

    def __init__(
        self,
        chroma_store: BaseVectorStore,
        bm25_indexer: BM25Indexer,
        image_storage: ImageStorage,
        file_integrity: FileIntegrityChecker,
    ) -> None:
        self._chroma = chroma_store
        self._bm25 = bm25_indexer
        self._images = image_storage
        self._integrity = file_integrity

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_documents(
        self,
        collection: str = "default",
    ) -> list[DocumentInfo]:
        """List all ingested documents in a collection.

        Groups chunks by ``metadata.source_path`` and enriches each entry
        with image counts from ImageStorage.
        """
        all_records = self._chroma.get_all(collection=collection)

        # Group by source_path
        doc_map: dict[str, dict[str, Any]] = {}
        for rec in all_records:
            meta = rec.get("metadata", {})
            source = meta.get("source_path", "")
            if not source:
                continue
            if source not in doc_map:
                doc_map[source] = {
                    "chunk_count": 0,
                    "doc_hash": meta.get("doc_hash", ""),
                }
            doc_map[source]["chunk_count"] += 1

        # Build DocumentInfo list with image counts
        result: list[DocumentInfo] = []
        for source, info in doc_map.items():
            doc_hash = info["doc_hash"]
            image_count = 0
            if doc_hash:
                imgs = self._images.get_by_doc_hash(doc_hash)
                image_count = len(imgs)

            result.append(
                DocumentInfo(
                    source_path=source,
                    collection=collection,
                    chunk_count=info["chunk_count"],
                    image_count=image_count,
                    doc_hash=doc_hash,
                )
            )

        return result

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_document(
        self,
        source_path: str,
        collection: str = "default",
    ) -> DeleteResult:
        """Delete a document and all its data across all storage backends.

        Steps:
            1. Fetch chunk IDs from ChromaStore by ``source_path``.
            2. Delete chunks from ChromaStore.
            3. Remove corresponding entries from BM25 index.
            4. Delete associated images from ImageStorage.
            5. Remove ingestion history record from FileIntegrity.
        """
        result = DeleteResult(source_path=source_path, collection=collection)

        # 1. Get chunk IDs and doc_hash for this source
        all_records = self._chroma.get_all(collection=collection)
        chunk_ids: list[str] = []
        doc_hash = ""
        for rec in all_records:
            meta = rec.get("metadata", {})
            if meta.get("source_path") == source_path:
                chunk_ids.append(rec["id"])
                if not doc_hash:
                    doc_hash = meta.get("doc_hash", "")

        if not chunk_ids:
            return result

        # 2. Delete from ChromaStore
        result.chunks_deleted = self._chroma.delete_by_metadata(
            {"source_path": source_path},
            collection=collection,
        )

        # 3. Remove from BM25 index
        result.bm25_postings_removed = self._bm25.remove_chunks(chunk_ids)
        # Persist updated index
        self._bm25.save(collection)

        # 4. Delete images
        if doc_hash:
            result.images_deleted = self._images.delete_by_doc_hash(
                doc_hash, collection
            )

        # 5. Remove integrity record
        if doc_hash:
            result.integrity_removed = self._integrity.remove_record(doc_hash)

        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_collection_stats(
        self,
        collection: str = "default",
    ) -> CollectionStats:
        """Return aggregate statistics for a collection."""
        all_records = self._chroma.get_all(collection=collection)

        # Count unique documents by source_path
        sources: set[str] = set()
        for rec in all_records:
            meta = rec.get("metadata", {})
            sp = meta.get("source_path", "")
            if sp:
                sources.add(sp)

        # Count images in collection
        images = self._images.get_by_collection(collection)

        return CollectionStats(
            name=collection,
            document_count=len(sources),
            chunk_count=len(all_records),
            image_count=len(images),
        )
