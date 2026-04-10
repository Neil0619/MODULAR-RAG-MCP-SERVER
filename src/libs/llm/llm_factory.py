"""Factory for creating LLM instances based on configuration.

Usage::

    from libs.llm.llm_factory import LLMFactory
    llm = LLMFactory.create(settings)
    response = llm.chat_str("Hello!")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from libs.llm.base_llm import BaseLLM

if TYPE_CHECKING:
    from core.settings import Settings

# Registry: provider_name -> module_path.ClassName
_PROVIDER_REGISTRY: dict[str, str] = {
    "openai": "libs.llm.openai_llm.OpenAILLM",
    "azure": "libs.llm.azure_llm.AzureLLM",
    "ollama": "libs.llm.ollama_llm.OllamaLLM",
    "deepseek": "libs.llm.deepseek_llm.DeepSeekLLM",
    "doubao": "libs.llm.doubao_llm.DoubaoLLM",
}

# Vision LLM registry
_VISION_REGISTRY: dict[str, str] = {
    "azure": "libs.llm.azure_vision_llm.AzureVisionLLM",
    "doubao": "libs.llm.doubao_vision_llm.DoubaoVisionLLM",
}


def _instantiate(class_path: str, settings: Settings) -> BaseLLM:
    """Lazy-import and instantiate a class from its dotted path."""
    import importlib

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(settings)


class LLMFactory:
    """Create LLM instances based on ``settings.llm.provider``."""

    @staticmethod
    def create(settings: Settings) -> BaseLLM:
        """Instantiate an LLM based on configuration.

        Args:
            settings: Application settings containing ``llm`` section.

        Returns:
            A concrete :class:`BaseLLM` implementation.

        Raises:
            ValueError: If the provider is unknown.
        """
        provider = settings.llm.provider.lower()
        if provider not in _PROVIDER_REGISTRY:
            available = ", ".join(sorted(_PROVIDER_REGISTRY))
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. Available: {available}"
            )
        return _instantiate(_PROVIDER_REGISTRY[provider], settings)

    @staticmethod
    def create_vision_llm(settings: Settings) -> BaseLLM:
        """Instantiate a Vision LLM based on ``settings.vision_llm.provider``.

        Raises:
            ValueError: If the provider is unknown.
        """
        provider = settings.vision_llm.provider.lower()
        if provider not in _VISION_REGISTRY:
            available = ", ".join(sorted(_VISION_REGISTRY))
            raise ValueError(
                f"Unknown Vision LLM provider: '{provider}'. Available: {available}"
            )
        return _instantiate(_VISION_REGISTRY[provider], settings)

    @staticmethod
    def register_provider(name: str, class_path: str) -> None:
        """Register a custom LLM provider at runtime."""
        _PROVIDER_REGISTRY[name.lower()] = class_path

    @staticmethod
    def register_vision_provider(name: str, class_path: str) -> None:
        """Register a custom Vision LLM provider at runtime."""
        _VISION_REGISTRY[name.lower()] = class_path
