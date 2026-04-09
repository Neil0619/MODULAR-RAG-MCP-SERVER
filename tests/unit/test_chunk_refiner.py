"""Unit tests for ChunkRefiner (C5).

Uses mocked LLM — no real API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.settings import (
    EmbeddingSettings,
    IngestionSettings,
    LLMSettings,
    RetrievalSettings,
    Settings,
    VectorStoreSettings,
)
from core.trace.trace_context import TraceContext
from core.types import Chunk, Document
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.chunk_refiner import ChunkRefiner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "noisy_chunks.json"


def _load_fixtures() -> list[dict]:
    return json.loads(_FIXTURES.read_text())


def _make_settings(use_llm: bool = False) -> Settings:
    ingestion = IngestionSettings(
        chunk_refiner=IngestionSettings.ChunkRefinerSettings(use_llm=use_llm),
    )
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
        ingestion=ingestion,
    )


def _make_chunk(text: str, **metadata: Any) -> Chunk:
    return Chunk(id="test", text=text, metadata={"source_path": "/test.pdf", **metadata})


# ---------------------------------------------------------------------------
# BaseTransform contract
# ---------------------------------------------------------------------------


class TestBaseTransform:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseTransform()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Rule-based refinement with fixtures
# ---------------------------------------------------------------------------


class TestRuleBasedRefinement:
    @pytest.fixture()
    def refiner(self) -> ChunkRefiner:
        return ChunkRefiner(_make_settings(use_llm=False))

    @pytest.mark.parametrize("fixture", _load_fixtures(), ids=lambda f: f["name"])
    def test_fixture(self, refiner: ChunkRefiner, fixture: dict) -> None:
        chunk = _make_chunk(fixture["input"])
        result = refiner.transform([chunk])
        assert len(result) == 1
        output = result[0].text
        expected = fixture["expected_patterns"]

        if "contains" in expected:
            for phrase in expected["contains"]:
                assert phrase in output, f"Missing '{phrase}' in output"

        if expected.get("no_page_numbers"):
            assert "Page " not in output or "of " not in output
            assert "P. " not in output

        if expected.get("no_separators"):
            assert "---" not in output
            assert "***" not in output

        if expected.get("no_html_comments"):
            assert "<!--" not in output

        if "max_consecutive_newlines" in expected:
            max_n = expected["max_consecutive_newlines"]
            assert "\n" * (max_n + 1) not in output

    def test_excessive_whitespace_collapsed(self, refiner: ChunkRefiner) -> None:
        chunk = _make_chunk("Hello\n\n\n\n\nWorld")
        result = refiner.transform([chunk])
        assert "\n\n\n" not in result[0].text

    def test_html_comments_removed(self, refiner: ChunkRefiner) -> None:
        chunk = _make_chunk("Before<!-- comment -->After")
        result = refiner.transform([chunk])
        assert "<!--" not in result[0].text
        assert "Before" in result[0].text

    def test_page_numbers_removed(self, refiner: ChunkRefiner) -> None:
        chunk = _make_chunk("Content\n\nPage 5 of 20\n\nMore content")
        result = refiner.transform([chunk])
        assert "Page 5 of 20" not in result[0].text

    def test_horizontal_rules_removed(self, refiner: ChunkRefiner) -> None:
        chunk = _make_chunk("Above\n\n--------\n\nBelow")
        result = refiner.transform([chunk])
        assert "----" not in result[0].text

    def test_clean_text_preserved(self, refiner: ChunkRefiner) -> None:
        original = "This is clean text.\nNo noise here."
        chunk = _make_chunk(original)
        result = refiner.transform([chunk])
        assert result[0].text == original

    def test_code_blocks_preserved(self, refiner: ChunkRefiner) -> None:
        code = "    def hello():\n        print('world')\n        return True"
        chunk = _make_chunk(f"Intro\n\n{code}\n\nEnd")
        result = refiner.transform([chunk])
        assert "def hello():" in result[0].text
        assert "print('world')" in result[0].text

    def test_metadata_marks_rule(self, refiner: ChunkRefiner) -> None:
        chunk = _make_chunk("Some text")
        result = refiner.transform([chunk])
        assert result[0].metadata["refined_by"] == "rule"


# ---------------------------------------------------------------------------
# LLM refinement (mocked)
# ---------------------------------------------------------------------------


class TestLLMRefinement:
    def _make_refiner_with_mock_llm(self) -> tuple[ChunkRefiner, MagicMock]:
        mock_llm = MagicMock()
        mock_llm.chat_str.return_value = "Clean refined text from LLM"
        settings = _make_settings(use_llm=True)
        refiner = ChunkRefiner(settings, llm=mock_llm)
        return refiner, mock_llm

    def test_llm_refinement_used(self) -> None:
        refiner, mock_llm = self._make_refiner_with_mock_llm()
        chunk = _make_chunk("Noisy text")
        result = refiner.transform([chunk])
        assert result[0].text == "Clean refined text from LLM"
        assert result[0].metadata["refined_by"] == "llm"

    def test_llm_called_with_prompt(self) -> None:
        refiner, mock_llm = self._make_refiner_with_mock_llm()
        chunk = _make_chunk("Some noisy input")
        refiner.transform([chunk])
        call_arg = mock_llm.chat_str.call_args[0][0]
        assert "Some noisy input" in call_arg

    def test_llm_failure_falls_back_to_rule(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat_str.side_effect = Exception("API error")
        settings = _make_settings(use_llm=True)
        refiner = ChunkRefiner(settings, llm=mock_llm)

        chunk = _make_chunk("Noisy  \n\n\n  text")
        result = refiner.transform([chunk])
        assert result[0].metadata["refined_by"] == "rule"
        assert result[0].metadata["refinement_fallback"] == "llm_failed"

    def test_llm_empty_response_falls_back(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat_str.return_value = "   "
        settings = _make_settings(use_llm=True)
        refiner = ChunkRefiner(settings, llm=mock_llm)

        chunk = _make_chunk("Some text")
        result = refiner.transform([chunk])
        assert result[0].metadata["refined_by"] == "rule"


# ---------------------------------------------------------------------------
# Config switch
# ---------------------------------------------------------------------------


class TestConfigSwitch:
    def test_use_llm_false_skips_llm(self) -> None:
        mock_llm = MagicMock()
        settings = _make_settings(use_llm=False)
        refiner = ChunkRefiner(settings, llm=mock_llm)

        chunk = _make_chunk("Text")
        refiner.transform([chunk])
        mock_llm.chat_str.assert_not_called()

    def test_use_llm_true_calls_llm(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat_str.return_value = "Refined"
        settings = _make_settings(use_llm=True)
        refiner = ChunkRefiner(settings, llm=mock_llm)

        chunk = _make_chunk("Text")
        refiner.transform([chunk])
        mock_llm.chat_str.assert_called_once()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_single_chunk_exception_does_not_block_others(self) -> None:
        refiner = ChunkRefiner(_make_settings(use_llm=False))
        chunks = [
            _make_chunk("Normal text"),
            _make_chunk(""),  # Empty — edge case
            _make_chunk("More normal text"),
        ]
        result = refiner.transform(chunks)
        assert len(result) == 3
        assert result[0].metadata["refined_by"] == "rule"
        assert result[2].metadata["refined_by"] == "rule"


# ---------------------------------------------------------------------------
# TraceContext integration
# ---------------------------------------------------------------------------


class TestTraceContext:
    def test_trace_context_created(self) -> None:
        trace = TraceContext()
        assert len(trace.trace_id) == 16

    def test_trace_context_record_stage(self) -> None:
        trace = TraceContext()
        trace.record_stage("refiner", {"chunks_processed": 5})
        assert trace.stages["refiner"]["chunks_processed"] == 5

    def test_refiner_accepts_trace(self) -> None:
        refiner = ChunkRefiner(_make_settings(use_llm=False))
        trace = TraceContext()
        chunk = _make_chunk("Text")
        result = refiner.transform([chunk], trace=trace)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


class TestPromptLoading:
    def test_default_prompt_loaded(self) -> None:
        refiner = ChunkRefiner(_make_settings(use_llm=False))
        assert "{text}" in refiner._prompt_template

    def test_custom_prompt_path(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "custom.txt"
        prompt_file.write_text("Custom: {text}")
        refiner = ChunkRefiner(
            _make_settings(use_llm=False),
            prompt_path=str(prompt_file),
        )
        assert refiner._prompt_template == "Custom: {text}"

    def test_missing_prompt_uses_default(self) -> None:
        refiner = ChunkRefiner(
            _make_settings(use_llm=False),
            prompt_path="/nonexistent/prompt.txt",
        )
        assert "{text}" in refiner._prompt_template
