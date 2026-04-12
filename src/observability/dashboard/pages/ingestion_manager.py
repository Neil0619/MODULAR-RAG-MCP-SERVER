"""Ingestion Manager page — upload files, trigger ingestion, delete documents."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st


def render() -> None:
    """Render the Ingestion Manager page."""
    st.header("Ingestion Manager")

    _render_upload_section()
    st.divider()
    _render_document_list()


# ---------------------------------------------------------------------------
# Upload & Ingest
# ---------------------------------------------------------------------------


def _render_upload_section() -> None:
    """File upload widget and ingestion trigger."""
    st.subheader("Upload & Ingest")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Select PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
    with col2:
        collection = st.text_input("Collection", value="default")

    if not uploaded_files:
        st.info("Select one or more PDF files to ingest.")
        return

    if st.button("Start Ingestion", type="primary"):
        _run_ingestion(uploaded_files, collection)


def _run_ingestion(
    uploaded_files: list[Any],
    collection: str,
) -> None:
    """Run IngestionPipeline on each uploaded file with progress."""
    from core.settings import load_settings
    from ingestion.pipeline import IngestionPipeline

    settings = load_settings()
    total_files = len(uploaded_files)
    overall_progress = st.progress(0.0, text="Preparing…")

    for file_idx, uploaded in enumerate(uploaded_files):
        file_label = uploaded.name

        # Write uploaded bytes to a temp file
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False, prefix="ingest_"
        ) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        try:
            pipeline = IngestionPipeline(settings, collection=collection)

            # Stage-level progress bar
            stage_bar = st.progress(0.0, text=f"[{file_idx+1}/{total_files}] {file_label} — starting…")

            def _on_progress(
                stage_name: str,
                current: int,
                total: int,
                _label: str = file_label,
                _bar: Any = stage_bar,
            ) -> None:
                frac = current / total if total > 0 else 0.0
                _bar.progress(
                    frac,
                    text=f"[{file_idx+1}/{total_files}] {_label} — {stage_name} ({current}/{total})",
                )

            summary = pipeline.run(tmp_path, force=True, on_progress=_on_progress)

            if summary.get("skipped"):
                stage_bar.progress(1.0, text=f"[{file_idx+1}/{total_files}] {file_label} — skipped (already ingested)")
            else:
                stage_bar.progress(1.0, text=f"[{file_idx+1}/{total_files}] {file_label} — done ✓")

        except Exception as exc:
            st.error(f"Failed to ingest {file_label}: {exc}")
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

        overall_progress.progress(
            (file_idx + 1) / total_files,
            text=f"Completed {file_idx + 1}/{total_files} files",
        )

    st.success(f"Ingestion complete: {total_files} file(s) processed.")


# ---------------------------------------------------------------------------
# Document list & delete
# ---------------------------------------------------------------------------


def _render_document_list() -> None:
    """List existing documents with delete buttons."""
    st.subheader("Existing Documents")

    collection = st.text_input("Filter collection", value="default", key="del_col")

    try:
        docs = _list_docs(collection)
    except Exception as exc:
        st.warning(f"Could not list documents: {exc}")
        return

    if not docs:
        st.info(f"No documents in collection **{collection}**.")
        return

    for doc in docs:
        col1, col2, col3 = st.columns([5, 2, 1])
        with col1:
            st.markdown(f"`{doc.source_path}`")
        with col2:
            st.caption(f"{doc.chunk_count} chunks · {doc.image_count} images")
        with col3:
            if st.button("🗑", key=f"del_{doc.source_path}_{collection}", help=f"Delete {doc.source_path}"):
                _delete_doc(doc.source_path, collection)
                st.rerun()


def _list_docs(collection: str) -> list[Any]:
    """List documents via DocumentManager."""
    from ingestion.document_manager import DocumentManager
    from ingestion.storage.bm25_indexer import BM25Indexer
    from ingestion.storage.image_storage import ImageStorage
    from libs.loader.file_integrity import SQLiteIntegrityChecker
    from libs.vector_store.vector_store_factory import VectorStoreFactory
    from core.settings import load_settings

    settings = load_settings()
    store = VectorStoreFactory.create(settings)
    bm25 = BM25Indexer()
    images = ImageStorage()
    integrity = SQLiteIntegrityChecker()
    dm = DocumentManager(store, bm25, images, integrity)
    return dm.list_documents(collection)


def _delete_doc(source_path: str, collection: str) -> None:
    """Delete a document across all storage backends."""
    from ingestion.document_manager import DocumentManager
    from ingestion.storage.bm25_indexer import BM25Indexer
    from ingestion.storage.image_storage import ImageStorage
    from libs.loader.file_integrity import SQLiteIntegrityChecker
    from libs.vector_store.vector_store_factory import VectorStoreFactory
    from core.settings import load_settings

    settings = load_settings()
    store = VectorStoreFactory.create(settings)
    bm25 = BM25Indexer()
    images = ImageStorage()
    integrity = SQLiteIntegrityChecker()
    dm = DocumentManager(store, bm25, images, integrity)

    result = dm.delete_document(source_path, collection)
    st.success(
        f"Deleted **{source_path}**: "
        f"{result.chunks_deleted} chunks, "
        f"{result.images_deleted} images removed."
    )


render()
