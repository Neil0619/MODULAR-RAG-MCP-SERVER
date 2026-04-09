"""Unit tests for file integrity checker (C2)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from libs.loader.file_integrity import (
    FileIntegrityChecker,
    SQLiteIntegrityChecker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def checker(tmp_path: Path) -> SQLiteIntegrityChecker:
    """Create a checker with a temp database."""
    db = tmp_path / "test.db"
    return SQLiteIntegrityChecker(str(db))


@pytest.fixture()
def sample_file(tmp_path: Path) -> str:
    """Create a sample file and return its path."""
    f = tmp_path / "sample.txt"
    f.write_text("Hello, world!")
    return str(f)


# ---------------------------------------------------------------------------
# SHA256 computation
# ---------------------------------------------------------------------------


class TestComputeSHA256:
    def test_consistent_hash(self, sample_file: str) -> None:
        h1 = FileIntegrityChecker.compute_sha256(sample_file)
        h2 = FileIntegrityChecker.compute_sha256(sample_file)
        assert h1 == h2

    def test_hash_is_64_hex_chars(self, sample_file: str) -> None:
        h = FileIntegrityChecker.compute_sha256(sample_file)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_files_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")
        assert FileIntegrityChecker.compute_sha256(str(f1)) != FileIntegrityChecker.compute_sha256(str(f2))

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="File not found"):
            FileIntegrityChecker.compute_sha256("/nonexistent/file.txt")


# ---------------------------------------------------------------------------
# SQLiteIntegrityChecker
# ---------------------------------------------------------------------------


class TestSQLiteIntegrityChecker:
    def test_creates_db_file(self, tmp_path: Path) -> None:
        db = tmp_path / "subdir" / "test.db"
        checker = SQLiteIntegrityChecker(str(db))
        assert Path(str(db)).exists()
        checker.close()

    def test_should_skip_false_initially(self, checker: SQLiteIntegrityChecker) -> None:
        assert checker.should_skip("abc123") is False

    def test_mark_success_then_skip(self, checker: SQLiteIntegrityChecker) -> None:
        h = "deadbeef" * 8
        checker.mark_success(h, "/data/file.pdf")
        assert checker.should_skip(h) is True

    def test_mark_failed_not_skipped(self, checker: SQLiteIntegrityChecker) -> None:
        h = "cafebabe" * 8
        checker.mark_failed(h, "parse error")
        assert checker.should_skip(h) is False

    def test_mark_failed_then_success_skips(self, checker: SQLiteIntegrityChecker) -> None:
        h = "abc" + "0" * 61
        checker.mark_failed(h, "temp error")
        assert checker.should_skip(h) is False
        checker.mark_success(h, "/data/file.pdf")
        assert checker.should_skip(h) is True

    def test_re_ingest_updates_status(self, checker: SQLiteIntegrityChecker) -> None:
        h = "111" + "0" * 61
        checker.mark_success(h, "/v1/file.pdf")
        checker.mark_success(h, "/v2/file.pdf")
        assert checker.should_skip(h) is True

    def test_extra_metadata_stored(self, checker: SQLiteIntegrityChecker, sample_file: str) -> None:
        h = FileIntegrityChecker.compute_sha256(sample_file)
        checker.mark_success(h, sample_file, chunks=10, pages=5)
        assert checker.should_skip(h) is True

    def test_close_is_idempotent(self, checker: SQLiteIntegrityChecker) -> None:
        checker.close()
        # Second close should not raise
        checker.close()

    def test_db_path_property(self, tmp_path: Path) -> None:
        db = str(tmp_path / "my.db")
        checker = SQLiteIntegrityChecker(db)
        assert checker.db_path == db
        checker.close()
