"""Unit tests for MetadataEnricher (C6)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from core.settings import (
    EmbeddingSettings,
    IngestionSettings,
    LLMSettings,
    RetrievalSettings,
    Settings,
    VectorStoreSettings,
)
from core.types import Chunk
from ingestion.transform.metadata_enricher import MetadataEnricher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(use_llm: bool = False) -> Settings:
    ingestion = IngestionSettings(
        metadata_enricher=IngestionSettings.MetadataEnricherSettings(use_llm=use_llm),
    )
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
        ingestion=ingestion,
    )


def _make_chunk(text: str, **meta: Any) -> Chunk:
    return Chunk(id="test", text=text, metadata={"source_path": "/test.pdf", **meta})


# ---------------------------------------------------------------------------
# Rule-based enrichment
# ---------------------------------------------------------------------------


class TestRuleBasedEnrichment:
    @pytest.fixture()
    def enricher(self) -> MetadataEnricher:
        return MetadataEnricher(_make_settings(use_llm=False))

    def test_adds_title_summary_tags(self, enricher: MetadataEnricher) -> None:
        chunk = _make_chunk("Introduction to Machine Learning\n\nML is a branch of AI.")
        result = enricher.transform([chunk])
        assert "title" in result[0].metadata
        assert "summary" in result[0].metadata
        assert "tags" in result[0].metadata

    def test_title_is_first_line(self, enricher: MetadataEnricher) -> None:
        chunk = _make_chunk("Chapter 3: Neural Networks\n\nDetailed content here.")
        result = enricher.transform([chunk])
        assert result[0].metadata["title"] == "Chapter 3: Neural Networks"

    def test_title_truncated_long_line(self, enricher: MetadataEnricher) -> None:
        long_line = "A" * 200
        chunk = _make_chunk(long_line)
        result = enricher.transform([chunk])
        assert len(result[0].metadata["title"]) <= 100

    def test_summary_is_text_preview(self, enricher: MetadataEnricher) -> None:
        text = "Short text"
        chunk = _make_chunk(text)
        result = enricher.transform([chunk])
        assert result[0].metadata["summary"] == text

    def test_summary_truncated_long_text(self, enricher: MetadataEnricher) -> None:
        text = "B" * 300
        chunk = _make_chunk(text)
        result = enricher.transform([chunk])
        assert len(result[0].metadata["summary"]) <= 200

    def test_tags_extracted_from_words(self, enricher: MetadataEnricher) -> None:
        chunk = _make_chunk("Machine learning algorithms process data efficiently")
        result = enricher.transform([chunk])
        tags = result[0].metadata["tags"]
        assert isinstance(tags, list)
        assert len(tags) >= 1
        # Words >3 chars
        assert all(len(t) > 3 for t in tags)

    def test_tags_max_five(self, enricher: MetadataEnricher) -> None:
        chunk = _make_chunk("alpha beta gamma delta epsilon zeta eta theta")
        result = enricher.transform([chunk])
        assert len(result[0].metadata["tags"]) <= 5

    def test_metadata_marks_rule(self, enricher: MetadataEnricher) -> None:
        chunk = _make_chunk("Some text here")
        result = enricher.transform([chunk])
        assert result[0].metadata["enriched_by"] == "rule"

    def test_empty_text_handled(self, enricher: MetadataEnricher) -> None:
        chunk = _make_chunk("")
        result = enricher.transform([chunk])
        assert result[0].metadata["title"] == ""

    def test_preserves_existing_metadata(self, enricher: MetadataEnricher) -> None:
        chunk = _make_chunk("Some text", source_path="/test.pdf", doc_type="pdf")
        result = enricher.transform([chunk])
        assert result[0].metadata["source_path"] == "/test.pdf"
        assert result[0].metadata["doc_type"] == "pdf"


# ---------------------------------------------------------------------------
# LLM enrichment (mocked)
# ---------------------------------------------------------------------------


class TestLLMEnrichment:
    def _make_enricher_with_mock(self) -> tuple[MetadataEnricher, MagicMock]:
        mock_llm = MagicMock()
        mock_llm.chat_str.return_value = (
            "TITLE: Neural Network Overview\n"
            "SUMMARY: This text introduces neural network concepts.\n"
            "TAGS: neural networks, deep learning, AI, machine learning"
        )
        settings = _make_settings(use_llm=True)
        return MetadataEnricher(settings, llm=mock_llm), mock_llm

    def test_llm_enrichment_used(self) -> None:
        enricher, _ = self._make_enricher_with_mock()
        chunk = _make_chunk("Neural networks are computational models...")
        result = enricher.transform([chunk])
        assert result[0].metadata["enriched_by"] == "llm"
        assert result[0].metadata["title"] == "Neural Network Overview"
        assert "neural networks" in result[0].metadata["tags"]

    def test_llm_called_with_text(self) -> None:
        enricher, mock_llm = self._make_enricher_with_mock()
        chunk = _make_chunk("Some input text for enrichment")
        enricher.transform([chunk])
        call_arg = mock_llm.chat_str.call_args[0][0]
        assert "Some input text" in call_arg

    def test_llm_failure_falls_back(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat_str.side_effect = Exception("API timeout")
        settings = _make_settings(use_llm=True)
        enricher = MetadataEnricher(settings, llm=mock_llm)

        chunk = _make_chunk("Machine learning is important")
        result = enricher.transform([chunk])
        assert result[0].metadata["enriched_by"] == "rule"
        assert result[0].metadata["enrichment_fallback"] == "llm_failed"
        # Should still have title/summary/tags from rules
        assert "title" in result[0].metadata
        assert "tags" in result[0].metadata

    def test_llm_empty_response_falls_back(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat_str.return_value = ""
        settings = _make_settings(use_llm=True)
        enricher = MetadataEnricher(settings, llm=mock_llm)

        chunk = _make_chunk("Some content")
        result = enricher.transform([chunk])
        assert result[0].metadata["enriched_by"] == "rule"

    def test_llm_malformed_response_falls_back(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat_str.return_value = "This is not structured metadata"
        settings = _make_settings(use_llm=True)
        enricher = MetadataEnricher(settings, llm=mock_llm)

        chunk = _make_chunk("Content here")
        result = enricher.transform([chunk])
        assert result[0].metadata["enriched_by"] == "rule"


# ---------------------------------------------------------------------------
# Config switch
# ---------------------------------------------------------------------------


class TestConfigSwitch:
    def test_use_llm_false_skips_llm(self) -> None:
        mock_llm = MagicMock()
        settings = _make_settings(use_llm=False)
        enricher = MetadataEnricher(settings, llm=mock_llm)
        chunk = _make_chunk("Text")
        enricher.transform([chunk])
        mock_llm.chat_str.assert_not_called()

    def test_use_llm_true_calls_llm(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat_str.return_value = "TITLE: Test\nSUMMARY: S\nTAGS: t"
        settings = _make_settings(use_llm=True)
        enricher = MetadataEnricher(settings, llm=mock_llm)
        chunk = _make_chunk("Text")
        enricher.transform([chunk])
        mock_llm.chat_str.assert_called_once()


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_parse_valid_response(self) -> None:
        resp = "TITLE: My Title\nSUMMARY: A summary.\nTAGS: a, b, c"
        result = MetadataEnricher._parse_llm_response(resp)
        assert result is not None
        assert result["title"] == "My Title"
        assert result["summary"] == "A summary."
        assert result["tags"] == ["a", "b", "c"]

    def test_parse_partial_response(self) -> None:
        resp = "TITLE: Only Title"
        result = MetadataEnricher._parse_llm_response(resp)
        assert result is not None
        assert result["title"] == "Only Title"
        assert "summary" not in result

    def test_parse_empty_returns_none(self) -> None:
        assert MetadataEnricher._parse_llm_response("") is None
        assert MetadataEnricher._parse_llm_response("   ") is None

    def test_parse_no_title_returns_none(self) -> None:
        resp = "SUMMARY: Something\nTAGS: x, y"
        assert MetadataEnricher._parse_llm_response(resp) is None


# ---------------------------------------------------------------------------
# Multiple chunks
# ---------------------------------------------------------------------------


class TestMultipleChunks:
    def test_all_chunks_enriched(self) -> None:
        enricher = MetadataEnricher(_make_settings(use_llm=False))
        chunks = [
            _make_chunk("First chunk text"),
            _make_chunk("Second chunk text"),
            _make_chunk("Third chunk text"),
        ]
        result = enricher.transform(chunks)
        assert len(result) == 3
        for c in result:
            assert "title" in c.metadata
            assert "summary" in c.metadata
            assert "tags" in c.metadata
