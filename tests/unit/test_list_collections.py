"""Unit tests for list_collections tool (E4)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_server.tools.list_collections import _scan_bm25_collections, list_collections


class TestListCollections:
    """Tests for list_collections tool."""

    def test_scan_bm25_with_data(self, tmp_path: Path) -> None:
        """Scanning BM25 dir with index files returns collection info."""
        index_dir = tmp_path / "bm25"
        index_dir.mkdir()
        (index_dir / "research.json").write_text(json.dumps({
            "metadata": {"N": 42, "avg_dl": 150.0},
            "terms": {"machine": {"idf": 1.5, "postings": []}, "learning": {"idf": 2.0, "postings": []}},
        }))
        (index_dir / "default.json").write_text(json.dumps({
            "metadata": {"N": 10, "avg_dl": 100.0},
            "terms": {},
        }))

        result = _scan_bm25_collections(str(index_dir))
        assert len(result) == 2
        assert result[0]["name"] == "default"
        assert result[0]["doc_count"] == 10
        assert result[1]["name"] == "research"
        assert result[1]["doc_count"] == 42
        assert result[1]["term_count"] == 2

    def test_scan_bm25_empty_dir(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "bm25"
        index_dir.mkdir()
        result = _scan_bm25_collections(str(index_dir))
        assert result == []

    def test_scan_bm25_nonexistent_dir(self, tmp_path: Path) -> None:
        result = _scan_bm25_collections(str(tmp_path / "nonexistent"))
        assert result == []

    def test_scan_bm25_corrupt_file(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "bm25"
        index_dir.mkdir()
        (index_dir / "bad.json").write_text("not valid json{{{")

        result = _scan_bm25_collections(str(index_dir))
        assert len(result) == 1
        assert result[0]["name"] == "bad"
        assert "error" in result[0]

    def test_list_collections_returns_json(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "bm25"
        index_dir.mkdir()
        (index_dir / "test.json").write_text(json.dumps({
            "metadata": {"N": 5}, "terms": {},
        }))

        with patch("mcp_server.tools.list_collections._scan_bm25_collections") as mock_scan:
            mock_scan.return_value = [{"name": "test", "doc_count": 5, "source": "bm25"}]
            import asyncio
            result = asyncio.run(list_collections())

        parsed = json.loads(result)
        assert "collections" in parsed
        assert parsed["total"] == 1
        assert parsed["collections"][0]["name"] == "test"
