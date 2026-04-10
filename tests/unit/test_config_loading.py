"""Unit tests for config loading and validation (A3)."""

import os
import textwrap
from pathlib import Path

import pytest
import yaml

from core.settings import (
    EmbeddingSettings,
    LLMSettings,
    Settings,
    SettingsError,
    load_settings,
    validate_settings,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def _write_config(tmp_path: Path, content: str) -> Path:
    """Write a YAML config to a temp file and return its path."""
    p = tmp_path / "settings.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


MINIMAL_VALID = """\
    llm:
      provider: openai
      model: gpt-4o
    embedding:
      provider: openai
      model: text-embedding-3-small
    vector_store:
      backend: chroma
    retrieval:
      sparse_backend: bm25
"""

# ---------------------------------------------------------------------------
# load_settings — success path
# ---------------------------------------------------------------------------


def test_load_settings_default_config() -> None:
    """The shipped config/settings.yaml should load without errors."""
    settings = load_settings()
    assert isinstance(settings, Settings)


def test_load_settings_returns_correct_types() -> None:
    """Loaded settings must have correct sub-dataclass types."""
    settings = load_settings()
    assert isinstance(settings.llm, LLMSettings)
    assert isinstance(settings.embedding, EmbeddingSettings)


def test_load_settings_fields_populated() -> None:
    """Key fields from the default config must be populated."""
    settings = load_settings()
    assert settings.llm.provider in {"azure", "openai", "ollama", "deepseek", "doubao"}
    assert settings.llm.model  # non-empty
    assert settings.embedding.provider in {"openai", "azure", "ollama", "doubao"}
    assert settings.embedding.model
    assert settings.vector_store.backend == "chroma"


def test_load_settings_minimal_config(tmp_path: Path) -> None:
    """A minimal valid config loads successfully."""
    p = _write_config(tmp_path, MINIMAL_VALID)
    settings = load_settings(p)
    assert settings.llm.provider == "openai"
    assert settings.embedding.model == "text-embedding-3-small"


# ---------------------------------------------------------------------------
# load_settings — failure path
# ---------------------------------------------------------------------------


def test_load_settings_missing_file(tmp_path: Path) -> None:
    """Loading a non-existent file raises SettingsError."""
    with pytest.raises(SettingsError, match="Config file not found"):
        load_settings(tmp_path / "nonexistent.yaml")


def test_load_settings_missing_llm_provider(tmp_path: Path) -> None:
    """Missing llm.provider must fail with a readable error."""
    config = """\
        embedding:
          provider: openai
          model: text-embedding-3-small
        vector_store:
          backend: chroma
        retrieval:
          sparse_backend: bm25
    """
    p = _write_config(tmp_path, config)
    with pytest.raises(SettingsError, match="llm.provider"):
        load_settings(p)


def test_load_settings_missing_embedding_model(tmp_path: Path) -> None:
    """Missing embedding.model must fail with a readable error."""
    config = """\
        llm:
          provider: openai
          model: gpt-4o
        embedding:
          provider: openai
        vector_store:
          backend: chroma
        retrieval:
          sparse_backend: bm25
    """
    p = _write_config(tmp_path, config)
    with pytest.raises(SettingsError, match="embedding.model"):
        load_settings(p)


def test_load_settings_invalid_rerank_backend(tmp_path: Path) -> None:
    """Invalid rerank backend must fail validation."""
    config = """\
        llm:
          provider: openai
          model: gpt-4o
        embedding:
          provider: openai
          model: text-embedding-3-small
        vector_store:
          backend: chroma
        retrieval:
          sparse_backend: bm25
        rerank:
          backend: invalid_option
    """
    p = _write_config(tmp_path, config)
    with pytest.raises(SettingsError, match="rerank.backend"):
        load_settings(p)


# ---------------------------------------------------------------------------
# validate_settings — direct tests
# ---------------------------------------------------------------------------


def test_validate_settings_ok() -> None:
    """A properly constructed Settings object validates cleanly."""
    s = Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
    )
    validate_settings(s)  # should not raise


def test_validate_settings_multiple_errors() -> None:
    """Multiple missing fields produce a multi-line error message."""
    s = Settings()  # all defaults → empty strings
    with pytest.raises(SettingsError, match=r"(?s)llm\.provider.*embedding\.provider"):
        validate_settings(s)


# ---------------------------------------------------------------------------
# env var expansion
# ---------------------------------------------------------------------------


def test_env_var_expansion(tmp_path: Path) -> None:
    """${VAR} in config values should be expanded from environment."""
    os.environ["TEST_RAG_API_KEY"] = "sk-test-123"
    try:
        config = """\
            llm:
              provider: openai
              model: gpt-4o
              api_key: "${TEST_RAG_API_KEY}"
            embedding:
              provider: openai
              model: text-embedding-3-small
            vector_store:
              backend: chroma
            retrieval:
              sparse_backend: bm25
        """
        p = _write_config(tmp_path, config)
        settings = load_settings(p)
        assert settings.llm.api_key == "sk-test-123"
    finally:
        del os.environ["TEST_RAG_API_KEY"]


# ---------------------------------------------------------------------------
# Import fix: VectorStoreSettings and RetrievalSettings used above
# ---------------------------------------------------------------------------

from core.settings import VectorStoreSettings, RetrievalSettings  # noqa: E402
