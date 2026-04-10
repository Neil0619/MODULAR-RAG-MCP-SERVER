"""Integration tests for ChunkRefiner with real LLM (C5).

⚠️  These tests require a valid LLM configuration in config/settings.yaml
    and will make real API calls (costs apply).

Run:  pytest tests/integration/test_chunk_refiner_llm.py -v -s

Marked with @pytest.mark.integration so they are skipped by default
in fast test runs (pytest -q without -m integration).
"""

from __future__ import annotations

import pytest

from core.settings import load_settings
from core.types import Chunk
from ingestion.transform.chunk_refiner import ChunkRefiner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str) -> Chunk:
    return Chunk(id="test", text=text, metadata={"source_path": "/test.pdf"})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def settings():
    """Load real settings from config/settings.yaml, force LLM on."""
    s = load_settings()
    s.ingestion.chunk_refiner.use_llm = True
    return s


@pytest.fixture(scope="module")
def refiner(settings):
    """Create ChunkRefiner with LLM enabled (real LLM from settings)."""
    return ChunkRefiner(settings)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRealLLMRefinement:
    """Verify real LLM connectivity and refinement quality."""

    def test_llm_refine_noisy_text(self, refiner: ChunkRefiner) -> None:
        """LLM should clean noisy text while preserving content."""
        noisy = (
            "Introduction to Neural Networks\n\n"
            "Page 3 of 45\n\n"
            "Neural networks are computational models  \n"
            "inspired by biological neurons.   \n\n\n\n"
            "<!-- OCR artifact -->\n"
            "They consist of layers of interconnected nodes."
        )
        chunk = _make_chunk(noisy)
        result = refiner.transform([chunk])

        assert len(result) == 1
        text = result[0].text

        # Content must be preserved
        assert "neural network" in text.lower() or "Neural" in text
        assert "layer" in text.lower() or "computational" in text.lower()

        # Noise should be reduced (page number artifacts removed by rules at minimum)
        assert "Page 3 of 45" not in text
        assert "<!--" not in text

        # Check metadata — must be LLM refined (not just rule-based)
        assert result[0].metadata["refined_by"] == "llm"

    def test_llm_refine_clean_text(self, refiner: ChunkRefiner) -> None:
        """Clean text should be preserved (not corrupted) by LLM refinement."""
        clean = (
            "Machine learning is a subset of artificial intelligence "
            "that enables systems to learn from data."
        )
        chunk = _make_chunk(clean)
        result = refiner.transform([chunk])

        assert len(result) == 1
        text = result[0].text.lower()
        # Core meaning preserved
        assert "machine learning" in text or "learn" in text

    def test_llm_refine_multiple_chunks(self, refiner: ChunkRefiner) -> None:
        """Multiple chunks should all be processed without errors."""
        chunks = [
            _make_chunk("Deep learning uses neural networks with multiple layers."),
            _make_chunk("Natural language processing handles text data."),
            _make_chunk(
                "Computer vision   \n\n\n   processes images and video.\n"
                "Page 12 of 30\n"
                "<!-- footer -->"
            ),
        ]
        result = refiner.transform(chunks)

        assert len(result) == 3
        for chunk in result:
            assert "refined_by" in chunk.metadata
            assert chunk.metadata["refined_by"] == "llm"
            # Text should not be empty
            assert len(chunk.text.strip()) > 0


@pytest.mark.integration
class TestRealLLMDegradation:
    """Verify graceful degradation when LLM has issues."""

    def test_degradation_with_invalid_model(self) -> None:
        """Using an invalid model should degrade to rule-based, not crash."""
        settings = load_settings()
        # Force an invalid model to trigger API error → fallback
        from core.settings import LLMSettings

        settings.llm = LLMSettings(
            provider=settings.llm.provider,
            model="nonexistent-model-xyz-123",
            api_key=settings.llm.api_key,
            azure_endpoint=settings.llm.azure_endpoint,
            api_version=settings.llm.api_version,
            deployment_name=settings.llm.deployment_name,
            temperature=0.0,
            max_tokens=100,
            base_url=getattr(settings.llm, "base_url", ""),
        )
        # Enable LLM so it attempts the call
        settings.ingestion.chunk_refiner.use_llm = True

        refiner = ChunkRefiner(settings)
        chunk = _make_chunk("Some text\n\nPage 1 of 10\n\nMore content")
        result = refiner.transform([chunk])

        # Should degrade to rule-based, not crash
        assert len(result) == 1
        assert result[0].metadata["refined_by"] in ("llm", "rule")
        # Page number removed by rules at minimum
        assert "Page 1 of 10" not in result[0].text
