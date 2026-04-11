"""Unit tests for ResponseBuilder and CitationGenerator (E3)."""

from __future__ import annotations

import json

import pytest

from core.response.citation_generator import Citation, CitationGenerator
from core.response.response_builder import ResponseBuilder
from core.types import RetrievalResult


def _rr(cid: str, score: float, text: str = "", meta: dict | None = None) -> RetrievalResult:
    return RetrievalResult(chunk_id=cid, score=score, text=text, metadata=meta or {}, source="fusion")


class TestCitationGenerator:
    def test_basic_generation(self) -> None:
        results = [
            _rr("c1", 0.95, "Hello world", {"source_path": "a.pdf", "page": 1}),
            _rr("c2", 0.85, "Another chunk", {"source_path": "b.pdf"}),
        ]
        citations = CitationGenerator.generate(results)
        assert len(citations) == 2
        assert citations[0].index == 1
        assert citations[0].source == "a.pdf"
        assert citations[0].page == 1
        assert citations[1].index == 2
        assert citations[1].source == "b.pdf"

    def test_empty_results(self) -> None:
        assert CitationGenerator.generate([]) == []

    def test_text_preview_truncated(self) -> None:
        long_text = "x" * 300
        results = [_rr("c1", 0.9, long_text)]
        citations = CitationGenerator.generate(results)
        assert len(citations[0].text_preview) < len(long_text)
        assert citations[0].text_preview.endswith("...")

    def test_to_dict(self) -> None:
        c = Citation(index=1, source="a.pdf", page=3, chunk_id="c1", score=0.95, text_preview="hi")
        d = c.to_dict()
        assert d["index"] == 1
        assert d["source"] == "a.pdf"
        assert d["page"] == 3
        assert d["score"] == 0.95

    def test_to_dict_no_page(self) -> None:
        c = Citation(index=1, source="a.pdf", chunk_id="c1", score=0.9)
        d = c.to_dict()
        assert "page" not in d


class TestResponseBuilder:
    def test_build_with_results(self) -> None:
        results = [
            _rr("c1", 0.95, "Azure config guide", {"source_path": "a.pdf", "title": "Azure Setup"}),
            _rr("c2", 0.80, "RAG overview", {"source_path": "b.pdf"}),
        ]
        response = ResponseBuilder.build(results, "how to configure Azure")

        assert "text" in response
        assert "citations" in response
        assert len(response["citations"]) == 2

        text = response["text"]
        assert "how to configure Azure" in text
        assert "[1]" in text
        assert "[2]" in text
        assert "Azure Setup" in text
        assert "a.pdf" in text

    def test_build_empty_results(self) -> None:
        response = ResponseBuilder.build([], "test query")
        assert "No relevant documents found" in response["text"]
        assert response["citations"] == []

    def test_citations_structure(self) -> None:
        results = [_rr("c1", 0.9, "text", {"source_path": "a.pdf"})]
        response = ResponseBuilder.build(results, "q")
        c = response["citations"][0]
        assert "index" in c
        assert "source" in c
        assert "chunk_id" in c
        assert "score" in c

    def test_citation_markers_match(self) -> None:
        results = [
            _rr("c1", 0.95, "First"),
            _rr("c2", 0.85, "Second"),
            _rr("c3", 0.75, "Third"),
        ]
        response = ResponseBuilder.build(results, "test")
        text = response["text"]
        assert "[1]" in text
        assert "[2]" in text
        assert "[3]" in text

    def test_score_in_citations_section(self) -> None:
        results = [_rr("c1", 0.9123, "text", {"source_path": "doc.pdf"})]
        response = ResponseBuilder.build(results, "q")
        assert "0.9123" in response["text"]
