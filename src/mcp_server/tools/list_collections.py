"""list_collections — MCP tool to list available knowledge base collections.

Scans BM25 index files and queries the vector store for collection stats.
"""

from __future__ import annotations

import json
from pathlib import Path
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


def _scan_bm25_collections(index_dir: str = "data/db/bm25") -> list[dict[str, Any]]:
    """Scan BM25 index directory for collection files."""
    path = Path(index_dir)
    if not path.exists():
        return []

    collections = []
    for f in sorted(path.glob("*.json")):
        name = f.stem
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            meta = data.get("metadata", {})
            collections.append({
                "name": name,
                "doc_count": meta.get("N", 0),
                "term_count": len(data.get("terms", {})),
                "source": "bm25",
            })
        except Exception:
            collections.append({"name": name, "source": "bm25", "error": "failed to read"})
    return collections


async def list_collections() -> str:
    """List all available collections with stats.

    Returns:
        JSON string with collection info.
    """
    bm25_cols = _scan_bm25_collections()

    # Merge with vector store info if available
    result: list[dict[str, Any]] = []
    bm25_map = {c["name"]: c for c in bm25_cols}

    if bm25_cols:
        result.extend(bm25_cols)

    response = {
        "collections": result,
        "total": len(result),
    }

    return json.dumps(response, ensure_ascii=False, indent=2)
