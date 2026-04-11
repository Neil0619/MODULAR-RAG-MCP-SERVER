"""File integrity checking using SHA256 hashes.

Provides an abstract interface and SQLite-backed default implementation
for tracking which files have already been successfully ingested.

Database file: ``data/db/ingestion_history.db`` (auto-created).
"""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FileIntegrityChecker(ABC):
    """Abstract interface for file integrity tracking."""

    @staticmethod
    def compute_sha256(path: str) -> str:
        """Compute the SHA256 hex digest of a file.

        Args:
            path: Absolute or relative file path.

        Returns:
            64-character hex string.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """Return ``True`` if this file hash was already processed successfully."""
        ...

    @abstractmethod
    def mark_success(self, file_hash: str, file_path: str, **extra: Any) -> None:
        """Record a successful ingestion."""
        ...

    @abstractmethod
    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """Record a failed ingestion."""
        ...

    @abstractmethod
    def remove_record(self, file_hash: str) -> bool:
        """Remove a record by hash. Returns True if a row was deleted."""
        ...

    @abstractmethod
    def list_processed(self) -> list[dict[str, Any]]:
        """Return all successfully processed records as dicts."""
        ...


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """SQLite-backed file integrity tracker.

    Uses WAL mode for safe concurrent access. The database and parent
    directories are created automatically on first use.

    Args:
        db_path: Path to the SQLite database file.
            Defaults to ``data/db/ingestion_history.db``.
    """

    def __init__(self, db_path: str = "data/db/ingestion_history.db") -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_history (
                file_hash   TEXT PRIMARY KEY,
                file_path   TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'success',
                error_msg   TEXT,
                ingested_at TEXT NOT NULL,
                extra       TEXT
            )
            """
        )
        self._conn.commit()

    def should_skip(self, file_hash: str) -> bool:
        """Check if a file hash was already successfully ingested."""
        row = self._conn.execute(
            "SELECT status FROM ingestion_history WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        return row is not None and row[0] == "success"

    def mark_success(self, file_hash: str, file_path: str, **extra: Any) -> None:
        """Mark a file as successfully ingested.

        If the hash already exists (e.g. from a prior failure), it is
        updated to ``success``.
        """
        import json

        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO ingestion_history (file_hash, file_path, status, ingested_at, extra)
            VALUES (?, ?, 'success', ?, ?)
            ON CONFLICT(file_hash) DO UPDATE SET
                status = 'success',
                file_path = excluded.file_path,
                ingested_at = excluded.ingested_at,
                error_msg = NULL,
                extra = excluded.extra
            """,
            (file_hash, file_path, now, json.dumps(extra) if extra else None),
        )
        self._conn.commit()

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """Mark a file as failed during ingestion."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO ingestion_history (file_hash, file_path, status, error_msg, ingested_at)
            VALUES (?, ?, 'failed', ?, ?)
            ON CONFLICT(file_hash) DO UPDATE SET
                status = 'failed',
                error_msg = excluded.error_msg,
                ingested_at = excluded.ingested_at
            """,
            (file_hash, "", error_msg, now),
        )
        self._conn.commit()

    def remove_record(self, file_hash: str) -> bool:
        """Delete a record by hash. Returns True if a row was removed."""
        cursor = self._conn.execute(
            "DELETE FROM ingestion_history WHERE file_hash = ?",
            (file_hash,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def list_processed(self) -> list[dict[str, Any]]:
        """Return all successfully processed records."""
        rows = self._conn.execute(
            "SELECT file_hash, file_path, ingested_at FROM ingestion_history WHERE status = 'success' ORDER BY ingested_at DESC"
        ).fetchall()
        return [
            {"file_hash": r[0], "file_path": r[1], "ingested_at": r[2]}
            for r in rows
        ]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    @property
    def db_path(self) -> str:
        return self._db_path
