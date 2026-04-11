"""DataService — encapsulates reads from ChromaStore, ImageStorage, and DocumentManager.

Provides a unified interface for the Data Browser page to list documents,
fetch chunk details, and retrieve associated images.
"""

from __future__ import annotations

from typing import Any

from ingestion.document_manager import DocumentInfo


class DataService:
    """Read-only data access for the Dashboard.

    Args:
        document_manager: A configured :class:`DocumentManager`.
        image_storage: An :class:`ImageStorage` instance for image lookup.
    """

    def __init__(
        self,
        document_manager: Any,
        image_storage: Any,
    ) -> None:
        self._dm = document_manager
        self._images = image_storage

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def list_documents(
        self,
        collection: str = "default",
    ) -> list[DocumentInfo]:
        """List all documents in a collection with chunk/image counts."""
        return self._dm.list_documents(collection)

    def list_collections(self) -> list[str]:
        """Return distinct collection names that have documents."""
        docs = self._dm.list_documents("default")
        # Try a few common collection names; expand if needed
        collections: set[str] = set()
        for candidate in ("default", "test", "research"):
            try:
                result = self._dm.list_documents(candidate)
                if result:
                    collections.add(candidate)
            except Exception:
                pass
        # Always include default
        collections.add("default")
        return sorted(collections)

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def get_chunks_for_document(
        self,
        source_path: str,
        collection: str = "default",
    ) -> list[dict[str, Any]]:
        """Return all chunks for a given source document.

        Each dict has ``id``, ``text``, and ``metadata`` keys.
        """
        from core.settings import load_settings
        from libs.vector_store.vector_store_factory import VectorStoreFactory

        settings = load_settings()
        store = VectorStoreFactory.create(settings)

        all_records = store.get_all(collection=collection)
        return [
            rec for rec in all_records
            if rec.get("metadata", {}).get("source_path") == source_path
        ]

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def get_images_for_doc(
        self,
        doc_hash: str,
        collection: str = "default",
    ) -> list[dict[str, Any]]:
        """Return image records for a document hash."""
        return self._images.get_by_doc_hash(doc_hash)

    def get_image_path(self, image_id: str) -> str | None:
        """Look up file path for an image_id."""
        return self._images.get_path(image_id)
