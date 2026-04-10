"""ImageStorage — save image files and track them in a SQLite index.

Stores images under ``data/images/{collection}/`` and maintains an
``image_index`` table in ``data/db/image_index.db`` for lookup by
image_id, collection, or doc_hash.
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ImageStorage:
    """Store image files with a SQLite index for fast lookup.

    Args:
        base_dir: Root directory for image files.
            Defaults to ``data/images``.
        db_path: Path to the SQLite index database.
            Defaults to ``data/db/image_index.db``.
    """

    def __init__(
        self,
        base_dir: str = "data/images",
        db_path: str = "data/db/image_index.db",
    ) -> None:
        self._base_dir = Path(base_dir)
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS image_index (
                image_id   TEXT PRIMARY KEY,
                file_path  TEXT NOT NULL,
                collection TEXT,
                doc_hash   TEXT,
                page_num   INTEGER,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash)"
        )
        self._conn.commit()

    def save(
        self,
        image_id: str,
        data: bytes,
        filename: str,
        collection: str = "default",
        doc_hash: str = "",
        page_num: int | None = None,
    ) -> str:
        """Save image data to disk and record in the index.

        Args:
            image_id: Unique image identifier.
            data: Raw image bytes.
            filename: Filename (e.g. ``img_001.png``).
            collection: Collection subdirectory.
            doc_hash: Parent document hash.
            page_num: Page number in source document.

        Returns:
            Absolute path to the saved file.
        """
        dest_dir = self._base_dir / collection
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename
        dest_path.write_bytes(data)

        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO image_index (image_id, file_path, collection, doc_hash, page_num, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                file_path = excluded.file_path,
                collection = excluded.collection,
                doc_hash = excluded.doc_hash,
                page_num = excluded.page_num,
                created_at = excluded.created_at
            """,
            (image_id, str(dest_path), collection, doc_hash, page_num, now),
        )
        self._conn.commit()
        return str(dest_path)

    def save_from_path(
        self,
        image_id: str,
        source_path: str,
        collection: str = "default",
        doc_hash: str = "",
        page_num: int | None = None,
    ) -> str:
        """Copy an image file from a source path into storage.

        Args:
            image_id: Unique image identifier.
            source_path: Path to the existing image file.
            collection: Collection subdirectory.
            doc_hash: Parent document hash.
            page_num: Page number in source document.

        Returns:
            Path to the stored file.
        """
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Image file not found: {source_path}")
        data = src.read_bytes()
        return self.save(image_id, data, src.name, collection, doc_hash, page_num)

    def get_path(self, image_id: str) -> str | None:
        """Look up the file path for an image_id.

        Returns:
            File path string, or None if not found.
        """
        row = self._conn.execute(
            "SELECT file_path FROM image_index WHERE image_id = ?",
            (image_id,),
        ).fetchone()
        return row[0] if row else None

    def get_by_collection(self, collection: str) -> list[dict[str, Any]]:
        """Return all image records for a collection."""
        rows = self._conn.execute(
            "SELECT image_id, file_path, doc_hash, page_num FROM image_index WHERE collection = ?",
            (collection,),
        ).fetchall()
        return [
            {"image_id": r[0], "file_path": r[1], "doc_hash": r[2], "page_num": r[3]}
            for r in rows
        ]

    def get_by_doc_hash(self, doc_hash: str) -> list[dict[str, Any]]:
        """Return all image records for a document hash."""
        rows = self._conn.execute(
            "SELECT image_id, file_path, collection, page_num FROM image_index WHERE doc_hash = ?",
            (doc_hash,),
        ).fetchall()
        return [
            {"image_id": r[0], "file_path": r[1], "collection": r[2], "page_num": r[3]}
            for r in rows
        ]

    def exists(self, image_id: str) -> bool:
        """Check if an image_id is indexed."""
        row = self._conn.execute(
            "SELECT 1 FROM image_index WHERE image_id = ?",
            (image_id,),
        ).fetchone()
        return row is not None

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
