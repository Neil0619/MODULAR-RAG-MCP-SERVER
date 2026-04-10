"""Unit tests for ImageStorage (C13)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.storage.image_storage import ImageStorage


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImageStorage:
    @pytest.fixture()
    def storage(self, tmp_path: Path) -> ImageStorage:
        return ImageStorage(
            base_dir=str(tmp_path / "images"),
            db_path=str(tmp_path / "db" / "image_index.db"),
        )

    def test_save_creates_file(self, storage: ImageStorage) -> None:
        path = storage.save("img_001", b"\x89PNG\r\n", "img_001.png", collection="test")
        assert Path(path).exists()
        assert Path(path).read_bytes() == b"\x89PNG\r\n"

    def test_save_records_in_index(self, storage: ImageStorage) -> None:
        storage.save("img_001", b"data", "img_001.png", collection="default")
        assert storage.exists("img_001")

    def test_get_path_returns_saved_path(self, storage: ImageStorage) -> None:
        path = storage.save("img_002", b"data", "img_002.png")
        result = storage.get_path("img_002")
        assert result == path

    def test_get_path_unknown_returns_none(self, storage: ImageStorage) -> None:
        assert storage.get_path("nonexistent") is None

    def test_save_from_path(self, storage: ImageStorage, tmp_path: Path) -> None:
        src = tmp_path / "source.png"
        src.write_bytes(b"\x89PNG source data")
        path = storage.save_from_path("img_003", str(src), collection="docs")
        assert Path(path).exists()
        assert Path(path).read_bytes() == b"\x89PNG source data"

    def test_save_from_path_not_found(self, storage: ImageStorage) -> None:
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            storage.save_from_path("img_x", "/nonexistent/file.png")

    def test_get_by_collection(self, storage: ImageStorage) -> None:
        storage.save("img_a", b"a", "a.png", collection="col1")
        storage.save("img_b", b"b", "b.png", collection="col1")
        storage.save("img_c", b"c", "c.png", collection="col2")

        results = storage.get_by_collection("col1")
        assert len(results) == 2
        ids = [r["image_id"] for r in results]
        assert "img_a" in ids
        assert "img_b" in ids

    def test_get_by_doc_hash(self, storage: ImageStorage) -> None:
        storage.save("img_a", b"a", "a.png", doc_hash="hash1")
        storage.save("img_b", b"b", "b.png", doc_hash="hash1")
        storage.save("img_c", b"c", "c.png", doc_hash="hash2")

        results = storage.get_by_doc_hash("hash1")
        assert len(results) == 2

    def test_save_with_page_num(self, storage: ImageStorage) -> None:
        storage.save("img_p3", b"data", "img.png", page_num=3)
        results = storage.get_by_collection("default")
        assert results[0]["page_num"] == 3

    def test_upsert_same_id_updates(self, storage: ImageStorage) -> None:
        storage.save("img_dup", b"v1", "img.png")
        storage.save("img_dup", b"v2", "img_v2.png")
        results = storage.get_by_collection("default")
        assert len(results) == 1

    def test_exists_false_for_unknown(self, storage: ImageStorage) -> None:
        assert storage.exists("no_such_image") is False

    def test_close_is_idempotent(self, storage: ImageStorage) -> None:
        storage.close()
        storage.close()
