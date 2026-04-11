"""get_document_summary — MCP tool to retrieve document metadata.

Looks up document chunks by source_path or doc_hash in the vector store
and returns aggregated summary information (title, summary, tags, stats).
"""

from __future__ import annotations

import json
from typing import Any

from core.settings import load_settings

_settings = None
_store = None


def _get_store() -> Any:
    """Lazily initialize vector store."""
    global _store, _settings
    if _store is None:
        _settings = load_settings()
        from libs.vector_store.vector_store_factory import VectorStoreFactory
        _store = VectorStoreFactory.create(_settings)
    return _store


async def get_document_summary(
    *,
    source_path: str = "",
    collection: str = "default",
) -> str:
    """Get summary info for a document.

    Queries the vector store for chunks matching the source_path,
    then aggregates metadata from the first chunk found.

    Args:
        source_path: Path of the source document.
        collection: Collection to search in.

    Returns:
        JSON string with document summary or error.
    """
    if not source_path:
        return json.dumps({"error": "source_path is required"}, ensure_ascii=False)

    store = _get_store()

    # Query with a metadata filter for source_path
    # Use a zero-vector to get all matching chunks (score doesn't matter)
    try:
        stats = store.get_collection_stats(collection)
        doc_count = stats.get("doc_count", 0)
        if doc_count == 0:
            return json.dumps({
                "error": f"No documents found in collection '{collection}'. "
                         "Please run ingest.py first.",
            }, ensure_ascii=False)
    except Exception:
        pass  # Continue anyway — the query may still work

    # Use a dummy vector to filter by metadata
    # We need dimensions — get from first chunk or use a default
    try:
        import hashlib
        # Generate a deterministic dummy vector (won't affect results since we filter)
        dummy_vec = [0.0] * 8  # minimal dimensions

        results = store.query(
            dummy_vec,
            top_k=100,
            filters={"source_path": source_path},
            collection=collection,
        )
    except Exception as exc:
        return json.dumps({
            "error": f"Failed to query store: {exc}",
        }, ensure_ascii=False)

    if not results:
        return json.dumps({
            "error": f"Document not found: {source_path}",
            "suggestion": "Check the source_path or try list_collections first.",
        }, ensure_ascii=False)

    # Aggregate info from all matching chunks
    first = results[0]
    titles = set()
    tags: list[str] = []
    summaries = set()
    seen_tags: set[str] = set()

    for r in results:
        meta = r.metadata if hasattr(r, "metadata") else {}
        t = meta.get("title", "")
        if t and t not in titles:
            titles.add(t)
        s = meta.get("summary", "")
        if s and s not in summaries:
            summaries.add(s)
        for tag in meta.get("tags", []):
            if tag not in seen_tags:
                seen_tags.add(tag)
                tags.append(tag)

    summary = {
        "source_path": source_path,
        "collection": collection,
        "chunk_count": len(results),
        "titles": list(titles),
        "summaries": list(summaries)[:3],
        "tags": tags[:10],
        "doc_hash": first.metadata.get("doc_hash", ""),
        "page_count": first.metadata.get("page_count"),
        "doc_type": first.metadata.get("doc_type", ""),
    }

    return json.dumps(summary, ensure_ascii=False, indent=2)
