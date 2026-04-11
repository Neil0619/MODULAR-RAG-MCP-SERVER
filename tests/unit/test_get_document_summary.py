"""Unit tests for get_document_summary tool (E5)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from mcp_server.tools.get_document_summary import get_document_summary


def _make_mock_store(results: list) -> MagicMock:
    mock = MagicMock()
    mock.get_collection_stats.return_value = {"doc_count": len(results)}
    mock.query.return_value = results
    return mock


def _query_result(cid: str, meta: dict) -> MagicMock:
    r = MagicMock()
    r.id = cid
    r.text = "chunk text"
    r.score = 0.9
    r.metadata = meta
    return r


class TestGetDocumentSummary:
    """Tests for get_document_summary tool."""

    def test_missing_source_path(self) -> None:
        result = asyncio.run(get_document_summary(source_path=""))
        parsed = json.loads(result)
        assert "error" in parsed

    def test_document_found(self) -> None:
        results = [
            _query_result("c1", {
                "source_path": "doc.pdf",
                "doc_hash": "abc123",
                "title": "Test Document",
                "summary": "A test doc",
                "tags": ["test", "pdf"],
                "page_count": 5,
                "doc_type": "pdf",
            }),
            _query_result("c2", {
                "source_path": "doc.pdf",
                "doc_hash": "abc123",
                "title": "Test Document",
                "tags": ["rag"],
            }),
        ]
        mock_store = _make_mock_store(results)

        with patch("mcp_server.tools.get_document_summary._get_store", return_value=mock_store):
            result = asyncio.run(get_document_summary(source_path="doc.pdf"))
            parsed = json.loads(result)

        assert parsed["source_path"] == "doc.pdf"
        assert parsed["chunk_count"] == 2
        assert "Test Document" in parsed["titles"]
        assert "test" in parsed["tags"]
        assert "rag" in parsed["tags"]
        assert parsed["doc_hash"] == "abc123"
        assert parsed["page_count"] == 5

    def test_document_not_found(self) -> None:
        mock_store = _make_mock_store([])
        mock_store.get_collection_stats.return_value = {"doc_count": 100}
        mock_store.query.return_value = []

        with patch("mcp_server.tools.get_document_summary._get_store", return_value=mock_store):
            result = asyncio.run(get_document_summary(source_path="nonexistent.pdf"))
            parsed = json.loads(result)

        assert "error" in parsed
        assert "not found" in parsed["error"].lower()

    def test_empty_collection(self) -> None:
        mock_store = MagicMock()
        mock_store.get_collection_stats.return_value = {"doc_count": 0}

        with patch("mcp_server.tools.get_document_summary._get_store", return_value=mock_store):
            result = asyncio.run(get_document_summary(source_path="doc.pdf"))
            parsed = json.loads(result)

        assert "error" in parsed

    def test_tags_deduplicated(self) -> None:
        results = [
            _query_result("c1", {"source_path": "a.pdf", "tags": ["ml", "ai"]}),
            _query_result("c2", {"source_path": "a.pdf", "tags": ["ai", "nlp"]}),
        ]
        mock_store = _make_mock_store(results)

        with patch("mcp_server.tools.get_document_summary._get_store", return_value=mock_store):
            result = asyncio.run(get_document_summary(source_path="a.pdf"))
            parsed = json.loads(result)

        assert parsed["tags"] == ["ml", "ai", "nlp"]
