"""BM25Indexer — build and persist an inverted index with IDF.

Receives sparse term weights from SparseEncoder, computes IDF, builds
an inverted index, and serializes to ``data/db/bm25/`` for later
query by SparseRetriever (D3).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from core.types import ChunkRecord


class BM25Indexer:
    """Build and persist a BM25 inverted index.

    Index structure::

        {
            "metadata": {"N": total_docs, "avg_dl": average_doc_length},
            "terms": {
                "word": {
                    "idf": float,
                    "postings": [{"id": str, "tf": float, "dl": int}]
                }
            }
        }

    Args:
        index_dir: Directory for index persistence.
            Defaults to ``data/db/bm25``.
    """

    def __init__(self, index_dir: str = "data/db/bm25") -> None:
        self._index_dir = Path(index_dir)
        self._index: dict[str, Any] = {"metadata": {}, "terms": {}}

    def build(
        self,
        records: list[ChunkRecord],
    ) -> None:
        """Build the inverted index from ChunkRecords.

        Args:
            records: Records with ``sparse_vector`` populated.
        """
        N = len(records)
        if N == 0:
            self._index = {"metadata": {"N": 0, "avg_dl": 0.0}, "terms": {}}
            return

        # Compute doc lengths and total
        total_dl = 0
        doc_lengths: dict[str, int] = {}
        for rec in records:
            sv = rec.sparse_vector or {}
            dl = sum(int(v * 10) for v in sv.values())  # approximate token count
            if dl == 0 and sv:
                dl = len(sv)  # fallback: number of unique terms
            doc_lengths[rec.id] = dl
            total_dl += dl

        avg_dl = total_dl / N

        # Build inverted index: term -> {chunk_id, tf, dl}
        term_postings: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            sv = rec.sparse_vector or {}
            for term, tf in sv.items():
                posting = {
                    "id": rec.id,
                    "tf": tf,
                    "dl": doc_lengths[rec.id],
                }
                if term not in term_postings:
                    term_postings[term] = []
                term_postings[term].append(posting)

        # Compute IDF for each term
        terms_index: dict[str, Any] = {}
        for term, postings in term_postings.items():
            df = len(postings)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            terms_index[term] = {
                "idf": idf,
                "postings": postings,
            }

        self._index = {
            "metadata": {"N": N, "avg_dl": avg_dl},
            "terms": terms_index,
        }

    def save(self, collection: str = "default") -> Path:
        """Save index to disk.

        Args:
            collection: Collection name for the index file.

        Returns:
            Path to the saved index file.
        """
        self._index_dir.mkdir(parents=True, exist_ok=True)
        path = self._index_dir / f"{collection}.json"
        path.write_text(json.dumps(self._index, ensure_ascii=False), encoding="utf-8")
        return path

    def load(self, collection: str = "default") -> None:
        """Load index from disk.

        Args:
            collection: Collection name to load.
        """
        path = self._index_dir / f"{collection}.json"
        if path.exists():
            self._index = json.loads(path.read_text(encoding="utf-8"))

    def query(
        self,
        terms: list[str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Score and rank chunks by BM25 for the given query terms.

        Args:
            terms: Query terms (pre-tokenized).
            top_k: Number of results to return.

        Returns:
            List of (chunk_id, score) tuples, sorted descending by score.
        """
        metadata = self._index.get("metadata", {})
        N = metadata.get("N", 0)
        avg_dl = metadata.get("avg_dl", 1.0)
        terms_index = self._index.get("terms", {})

        k1 = 1.5  # BM25 parameter
        b = 0.75  # BM25 parameter

        scores: dict[str, float] = {}

        for term in terms:
            entry = terms_index.get(term)
            if not entry:
                continue

            idf = entry["idf"]
            for posting in entry["postings"]:
                chunk_id = posting["id"]
                tf = posting["tf"]
                dl = posting.get("dl", 1)

                # BM25 score component
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / avg_dl) if avg_dl > 0 else tf + k1
                score = idf * numerator / denominator

                scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def remove_chunks(self, chunk_ids: list[str]) -> int:
        """Remove all postings for the given chunk IDs.

        After removal, stale terms (no remaining postings) are pruned and
        the metadata (N, avg_dl) is recalculated.

        Args:
            chunk_ids: IDs of chunks to remove from the index.

        Returns:
            Total number of postings removed.
        """
        if not chunk_ids:
            return 0

        id_set = set(chunk_ids)
        removed = 0
        stale_terms: list[str] = []

        terms = self._index.get("terms", {})
        for term, entry in list(terms.items()):
            original = entry["postings"]
            filtered = [p for p in original if p["id"] not in id_set]
            removed += len(original) - len(filtered)
            if filtered:
                entry["postings"] = filtered
                # Recompute IDF
                N = self._index["metadata"].get("N", 0) - len(chunk_ids)
                df = len(filtered)
                entry["idf"] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            else:
                stale_terms.append(term)

        for term in stale_terms:
            terms.pop(term, None)

        # Recalculate metadata
        all_postings = []
        for entry in terms.values():
            all_postings.extend(entry["postings"])

        self._index["metadata"] = {
            "N": len({p["id"] for p in all_postings}),
            "avg_dl": (sum(p["dl"] for p in all_postings) / len(all_postings))
            if all_postings
            else 0.0,
        }

        return removed

    @property
    def term_count(self) -> int:
        """Number of unique terms in the index."""
        return len(self._index.get("terms", {}))

    @property
    def doc_count(self) -> int:
        """Number of documents in the index."""
        return self._index.get("metadata", {}).get("N", 0)

    @property
    def index(self) -> dict[str, Any]:
        """Raw index data (for testing)."""
        return self._index
