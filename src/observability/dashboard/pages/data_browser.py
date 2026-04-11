"""Data Browser page — browse documents, chunks, and images."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import streamlit as st

from observability.dashboard.services.data_service import DataService


def _init_service() -> DataService:
    """Create a DataService with real backend connections."""
    from core.settings import load_settings
    from ingestion.document_manager import DocumentManager
    from ingestion.storage.bm25_indexer import BM25Indexer
    from ingestion.storage.image_storage import ImageStorage
    from libs.loader.file_integrity import SQLiteIntegrityChecker
    from libs.vector_store.vector_store_factory import VectorStoreFactory

    settings = load_settings()
    store = VectorStoreFactory.create(settings)
    bm25 = BM25Indexer()
    images = ImageStorage()
    integrity = SQLiteIntegrityChecker()
    dm = DocumentManager(store, bm25, images, integrity)
    return DataService(document_manager=dm, image_storage=images)


def render() -> None:
    """Render the Data Browser page."""
    st.header("Data Browser")

    try:
        service = _init_service()
    except Exception as exc:
        st.error(f"Failed to initialise data service: {exc}")
        return

    # --- Collection selector ---
    docs = service.list_documents("default")
    collections = service.list_collections()

    selected_col = st.selectbox(
        "Collection",
        options=collections,
        index=0,
    )

    # Refresh docs for selected collection
    if selected_col != "default":
        docs = service.list_documents(selected_col)

    if not docs:
        st.info(
            f"No documents found in collection **{selected_col}**. "
            "Run `python scripts/ingest.py` to add documents."
        )
        return

    st.metric("Documents", len(docs))

    # --- Document list ---
    st.subheader("Documents")

    for doc in docs:
        label = f"{doc.source_path}  ({doc.chunk_count} chunks, {doc.image_count} images)"
        with st.expander(label):
            _render_document_detail(service, doc, selected_col)


def _render_document_detail(
    service: DataService,
    doc: Any,
    collection: str,
) -> None:
    """Show chunks and images for a single document."""
    # Document metadata
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Source:** `{doc.source_path}`")
        st.markdown(f"**Doc Hash:** `{doc.doc_hash or 'N/A'}`")
    with col2:
        st.markdown(f"**Chunks:** {doc.chunk_count}")
        st.markdown(f"**Images:** {doc.image_count}")

    # --- Images ---
    if doc.doc_hash:
        images = service.get_images_for_doc(doc.doc_hash, collection)
        if images:
            st.subheader("Images")
            img_cols = st.columns(min(len(images), 4))
            for idx, img_rec in enumerate(images):
                col = img_cols[idx % len(img_cols)]
                img_path = img_rec.get("file_path", "")
                if img_path and Path(img_path).exists():
                    with col:
                        _show_image(img_path, img_rec.get("image_id", ""))
                else:
                    with col:
                        st.caption(f"[Image not found: {img_rec.get('image_id', '')}]")

    # --- Chunks ---
    st.subheader("Chunks")
    chunks = service.get_chunks_for_document(doc.source_path, collection)

    if not chunks:
        st.info("No chunks found.")
        return

    for chunk in chunks:
        chunk_idx = chunk.get("metadata", {}).get("chunk_index", "?")
        chunk_id = chunk.get("id", "?")
        text = chunk.get("text", "")
        title = chunk.get("metadata", {}).get("title", "")

        header = f"Chunk {chunk_idx}"
        if title:
            header += f" — {title}"

        with st.expander(header):
            # Text content (truncated preview)
            st.markdown("**Content:**")
            st.text(text[:500] + ("..." if len(text) > 500 else ""))

            # Metadata
            meta = chunk.get("metadata", {})
            display_meta = {
                k: v for k, v in meta.items()
                if k not in ("source_path",) and v is not None
            }
            if display_meta:
                st.markdown("**Metadata:**")
                st.json(display_meta)


def _show_image(img_path: str, image_id: str) -> None:
    """Display an image from local file system."""
    try:
        data = Path(img_path).read_bytes()
        encoded = base64.b64encode(data).decode()
        # Guess MIME from extension
        ext = Path(img_path).suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/png")
        st.image(f"data:{mime};base64,{encoded}", caption=image_id)
    except Exception as exc:
        st.caption(f"[Error loading image: {exc}]")
