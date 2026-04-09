"""Configuration loading and validation.

Reads ``config/settings.yaml`` and exposes a validated :class:`Settings` object.
Environment variables in the form ``${VAR_NAME}`` are expanded at load time.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from observability.logger import get_logger

logger = get_logger("settings")

# Pattern for ${ENV_VAR} substitution
_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ``${ENV_VAR}`` references in config values."""
    if isinstance(value, str):
        def _replacer(m: re.Match) -> str:
            env_name = m.group(1)
            env_val = os.environ.get(env_name, "")
            if not env_val:
                logger.warning("Environment variable %s is not set", env_name)
            return env_val
        return _ENV_VAR_RE.sub(_replacer, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Settings dataclass — mirrors config/settings.yaml structure
# ---------------------------------------------------------------------------


@dataclass
class LLMSettings:
    provider: str = ""
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    # Azure-specific
    azure_endpoint: str = ""
    api_key: str = ""
    api_version: str = ""
    deployment_name: str = ""
    # OpenAI-specific
    base_url: str = ""


@dataclass
class EmbeddingSettings:
    provider: str = ""
    model: str = ""
    api_key: str = ""
    dimensions: int = 1536
    batch_size: int = 100
    # Azure-specific
    azure_endpoint: str = ""
    api_version: str = ""
    deployment_name: str = ""


@dataclass
class VisionLLMSettings:
    provider: str = ""
    model: str = ""
    azure_endpoint: str = ""
    api_key: str = ""
    api_version: str = ""
    deployment_name: str = ""
    max_image_size: int = 2048


@dataclass
class VectorStoreSettings:
    backend: str = "chroma"
    persist_path: str = "./data/db/chroma"


@dataclass
class SplitterSettings:
    provider: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])


@dataclass
class RetrievalSettings:
    sparse_backend: str = "bm25"
    fusion_algorithm: str = "rrf"
    rrf_k: int = 60
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 10


@dataclass
class RerankSettings:
    backend: str = "none"
    model: str = ""
    top_m: int = 30


@dataclass
class IngestionSettings:
    """Nested ingestion feature flags."""

    @dataclass
    class ChunkRefinerSettings:
        use_llm: bool = False

    @dataclass
    class MetadataEnricherSettings:
        use_llm: bool = True

    @dataclass
    class ImageCaptionerSettings:
        enabled: bool = False

    chunk_refiner: ChunkRefinerSettings = field(default_factory=ChunkRefinerSettings)
    metadata_enricher: MetadataEnricherSettings = field(default_factory=MetadataEnricherSettings)
    image_captioner: ImageCaptionerSettings = field(default_factory=ImageCaptionerSettings)


@dataclass
class EvaluationSettings:
    backends: list[str] = field(default_factory=lambda: ["custom"])
    golden_test_set: str = "./tests/fixtures/golden_test_set.json"


@dataclass
class ObservabilitySettings:
    enabled: bool = True

    @dataclass
    class LoggingSettings:
        log_file: str = "logs/traces.jsonl"
        log_level: str = "INFO"

    logging: LoggingSettings = field(default_factory=LoggingSettings)
    detail_level: str = "standard"


@dataclass
class DashboardSettings:
    enabled: bool = True
    port: int = 8501
    traces_dir: str = "./logs"
    auto_refresh: bool = True
    refresh_interval: int = 5


@dataclass
class Settings:
    """Root settings object — mirrors config/settings.yaml."""

    llm: LLMSettings = field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    vision_llm: VisionLLMSettings = field(default_factory=VisionLLMSettings)
    vector_store: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    splitter: SplitterSettings = field(default_factory=SplitterSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    rerank: RerankSettings = field(default_factory=RerankSettings)
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)
    dashboard: DashboardSettings = field(default_factory=DashboardSettings)


# ---------------------------------------------------------------------------
# _dict_to_dataclass helper
# ---------------------------------------------------------------------------


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively populate a dataclass from a dict, ignoring unknown keys."""
    import dataclasses
    from typing import get_type_hints

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            value = data[f.name]
            field_type = hints.get(f.name)
            if isinstance(value, dict) and field_type and hasattr(field_type, "__dataclass_fields__"):
                kwargs[f.name] = _dict_to_dataclass(field_type, value)
            else:
                kwargs[f.name] = value
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    """Load YAML config, expand env vars, parse into :class:`Settings`.

    Raises :class:`SettingsError` on missing file or validation failure.
    """
    path = Path(path)
    if not path.exists():
        raise SettingsError(f"Config file not found: {path}")

    with open(path) as f:
        raw: dict = yaml.safe_load(f) or {}

    expanded = _expand_env_vars(raw)
    settings = _dict_to_dataclass(Settings, expanded)
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    """Validate that required fields are present and non-empty.

    Raises :class:`SettingsError` with a clear field-path message.
    """
    errors: list[str] = []

    # LLM
    if not settings.llm.provider:
        errors.append("llm.provider is required")
    if not settings.llm.model:
        errors.append("llm.model is required")

    # Embedding
    if not settings.embedding.provider:
        errors.append("embedding.provider is required")
    if not settings.embedding.model:
        errors.append("embedding.model is required")

    # Vector store
    if not settings.vector_store.backend:
        errors.append("vector_store.backend is required")

    # Retrieval
    if not settings.retrieval.sparse_backend:
        errors.append("retrieval.sparse_backend is required")

    # Rerank backend must be one of the allowed values
    valid_rerank = {"none", "cross_encoder", "llm"}
    if settings.rerank.backend and settings.rerank.backend not in valid_rerank:
        errors.append(f"rerank.backend must be one of {valid_rerank}, got '{settings.rerank.backend}'")

    if errors:
        msg = "Settings validation failed:\n  - " + "\n  - ".join(errors)
        raise SettingsError(msg)

    logger.info("Settings validated successfully (llm.provider=%s)", settings.llm.provider)


class SettingsError(Exception):
    """Raised when settings are missing or invalid."""
