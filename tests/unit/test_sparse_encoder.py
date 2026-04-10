"""Unit tests for SparseEncoder (C9)."""

from __future__ import annotations

from core.types import Chunk
from ingestion.embedding.sparse_encoder import SparseEncoder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSparseEncoder:
    @property
    def encoder(self) -> SparseEncoder:
        return SparseEncoder(use_stop_words=True)

    def test_encode_returns_correct_count(self) -> None:
        enc = self.encoder
        chunks = [
            Chunk(id="c1", text="Hello world"),
            Chunk(id="c2", text="Machine learning"),
        ]
        result = enc.encode(chunks)
        assert len(result) == 2

    def test_encode_returns_dict_of_floats(self) -> None:
        enc = self.encoder
        chunks = [Chunk(id="c1", text="document retrieval system")]
        result = enc.encode(chunks)
        weights = result[0]
        assert isinstance(weights, dict)
        for v in weights.values():
            assert isinstance(v, float)

    def test_tf_weights_normalized(self) -> None:
        enc = self.encoder
        # "hello" appears twice, others once
        chunks = [Chunk(id="c1", text="hello hello world")]
        result = enc.encode(chunks)
        weights = result[0]
        assert weights["hello"] == 1.0  # max freq normalized to 1.0
        assert weights["world"] == 0.5  # freq=1 / max_freq=2

    def test_stop_words_filtered(self) -> None:
        enc = self.encoder
        chunks = [Chunk(id="c1", text="the quick brown fox")]
        result = enc.encode(chunks)
        weights = result[0]
        assert "the" not in weights
        assert "quick" in weights

    def test_empty_text_returns_empty_dict(self) -> None:
        enc = self.encoder
        chunks = [Chunk(id="c1", text="")]
        result = enc.encode(chunks)
        assert result[0] == {}

    def test_whitespace_only_returns_empty(self) -> None:
        enc = self.encoder
        chunks = [Chunk(id="c1", text="   \n\t  ")]
        result = enc.encode(chunks)
        assert result[0] == {}

    def test_short_tokens_filtered(self) -> None:
        enc = self.encoder
        chunks = [Chunk(id="c1", text="I am a big person")]
        result = enc.encode(chunks)
        weights = result[0]
        # Single-char tokens should be filtered (< 2 chars)
        assert "i" not in weights
        assert "a" not in weights
        assert "am" in weights

    def test_encode_texts(self) -> None:
        enc = self.encoder
        result = enc.encode_texts(["hello world", "machine learning"])
        assert len(result) == 2
        assert "hello" in result[0]
        assert "machine" in result[1]

    def test_case_insensitive(self) -> None:
        enc = self.encoder
        chunks = [Chunk(id="c1", text="Hello HELLO hello")]
        result = enc.encode(chunks)
        weights = result[0]
        assert "hello" in weights
        assert weights["hello"] == 1.0

    def test_no_stop_words_mode(self) -> None:
        enc = SparseEncoder(use_stop_words=False)
        chunks = [Chunk(id="c1", text="the document system")]
        result = enc.encode(chunks)
        weights = result[0]
        assert "the" in weights
        assert "document" in weights

    def test_encode_empty_list(self) -> None:
        enc = self.encoder
        result = enc.encode([])
        assert result == []

    def test_encode_texts_empty_list(self) -> None:
        enc = self.encoder
        result = enc.encode_texts([])
        assert result == []
