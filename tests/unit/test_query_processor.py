"""Unit tests for QueryProcessor (D1)."""

from __future__ import annotations

import pytest

from core.query_engine.query_processor import QueryProcessor
from core.types import ProcessedQuery


class TestQueryProcessor:
    """Tests for keyword extraction and filter parsing."""

    @pytest.fixture()
    def processor(self) -> QueryProcessor:
        return QueryProcessor(use_stop_words=True)

    @pytest.fixture()
    def processor_no_stop(self) -> QueryProcessor:
        return QueryProcessor(use_stop_words=False)

    # -- Basic keyword extraction --

    def test_basic_english_query(self, processor: QueryProcessor) -> None:
        pq = processor.process("How does Azure OpenAI work?")
        assert "azure" in pq.keywords
        assert "openai" in pq.keywords
        assert "work" in pq.keywords
        # Stop words should be filtered
        assert "how" not in pq.keywords
        assert "does" not in pq.keywords

    def test_keywords_non_empty_for_non_empty_input(self, processor: QueryProcessor) -> None:
        pq = processor.process("machine learning algorithms")
        assert len(pq.keywords) > 0

    def test_empty_query_returns_empty_keywords(self, processor: QueryProcessor) -> None:
        pq = processor.process("")
        assert pq.keywords == []

    def test_whitespace_only_query(self, processor: QueryProcessor) -> None:
        pq = processor.process("   \n\t  ")
        assert pq.keywords == []

    def test_only_stop_words(self, processor: QueryProcessor) -> None:
        pq = processor.process("the is at which on and or")
        assert pq.keywords == []

    def test_preserves_original_query(self, processor: QueryProcessor) -> None:
        query = "What is RAG pipeline?"
        pq = processor.process(query)
        assert pq.original_query == query

    # -- Chinese support --

    def test_chinese_query(self, processor: QueryProcessor) -> None:
        pq = processor.process("如何配置 Azure OpenAI 服务")
        assert "azure" in pq.keywords
        assert "openai" in pq.keywords
        # Chinese segments
        assert any(k for k in pq.keywords if len(k) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in k))

    def test_mixed_chinese_english(self, processor: QueryProcessor) -> None:
        pq = processor.process("RAG系统的混合检索 hybrid search")
        assert "hybrid" in pq.keywords
        assert "search" in pq.keywords
        assert "rag" in pq.keywords

    # -- Deduplication --

    def test_deduplication_preserves_order(self, processor: QueryProcessor) -> None:
        pq = processor.process("embedding embedding vector vector embedding")
        assert pq.keywords == ["embedding", "vector"]

    # -- Stop words toggle --

    def test_no_stop_words_keeps_all(self, processor_no_stop: QueryProcessor) -> None:
        pq = processor_no_stop.process("the is at which")
        assert "the" in pq.keywords
        assert "is" in pq.keywords
        assert "at" in pq.keywords
        assert "which" in pq.keywords

    # -- Filters --

    def test_filters_default_empty(self, processor: QueryProcessor) -> None:
        pq = processor.process("test query")
        assert pq.filters == {}

    def test_filters_passed_through(self, processor: QueryProcessor) -> None:
        filters = {"collection": "research", "doc_type": "pdf"}
        pq = processor.process("test query", filters=filters)
        assert pq.filters == filters
        assert pq.filters["collection"] == "research"

    def test_filters_none_becomes_empty(self, processor: QueryProcessor) -> None:
        pq = processor.process("test query", filters=None)
        assert pq.filters == {}

    # -- ProcessedQuery serialization --

    def test_processed_query_to_dict(self, processor: QueryProcessor) -> None:
        pq = processor.process("Azure config", filters={"collection": "test"})
        d = pq.to_dict()
        assert d["original_query"] == "Azure config"
        assert isinstance(d["keywords"], list)
        assert d["filters"] == {"collection": "test"}

    def test_processed_query_from_dict(self) -> None:
        data = {
            "original_query": "test",
            "keywords": ["test", "query"],
            "filters": {"col": "default"},
        }
        pq = ProcessedQuery.from_dict(data)
        assert pq.original_query == "test"
        assert pq.keywords == ["test", "query"]
        assert pq.filters == {"col": "default"}

    def test_processed_query_roundtrip(self, processor: QueryProcessor) -> None:
        pq = processor.process("vector database query", filters={"collection": "docs"})
        restored = ProcessedQuery.from_dict(pq.to_dict())
        assert restored.original_query == pq.original_query
        assert restored.keywords == pq.keywords
        assert restored.filters == pq.filters

    # -- Short tokens filtered --

    def test_single_char_tokens_filtered(self, processor: QueryProcessor) -> None:
        pq = processor.process("a b c d")
        assert pq.keywords == []  # Single chars don't match the 2+ regex

    def test_two_char_tokens_kept(self, processor: QueryProcessor) -> None:
        pq = processor.process("AI ML NLP")
        assert "ai" in pq.keywords
        assert "ml" in pq.keywords
        assert "nlp" in pq.keywords

    # -- Special characters --

    def test_special_characters_stripped(self, processor: QueryProcessor) -> None:
        pq = processor.process("RAG, pipeline! (with) brackets & symbols")
        assert "rag" in pq.keywords
        assert "pipeline" in pq.keywords
        assert "brackets" in pq.keywords
        assert "symbols" in pq.keywords
