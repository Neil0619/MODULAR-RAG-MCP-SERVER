"""Factory for creating Embedding instances based on configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from core.settings import Settings

_PROVIDER_REGISTRY: dict[str, str] = {
    "openai": "libs.embedding.openai_embedding.OpenAIEmbedding",
    "azure": "libs.embedding.azure_embedding.AzureEmbedding",
    "ollama": "libs.embedding.ollama_embedding.OllamaEmbedding",
}


class EmbeddingFactory:
    """Create embedding instances based on ``settings.embedding.provider``."""

    @staticmethod
    def create(settings: Settings) -> BaseEmbedding:
        provider = settings.embedding.provider.lower()
        if provider not in _PROVIDER_REGISTRY:
            available = ", ".join(sorted(_PROVIDER_REGISTRY))
            raise ValueError(
                f"Unknown embedding provider: '{provider}'. Available: {available}"
            )

        module_path, class_name = _PROVIDER_REGISTRY[provider].rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(settings)

    @staticmethod
    def register_provider(name: str, class_path: str) -> None:
        _PROVIDER_REGISTRY[name.lower()] = class_path
