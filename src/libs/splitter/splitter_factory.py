"""Factory for creating Splitter instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from libs.splitter.base_splitter import BaseSplitter

if TYPE_CHECKING:
    from core.settings import Settings

_PROVIDER_REGISTRY: dict[str, str] = {
    "recursive": "libs.splitter.recursive_splitter.RecursiveSplitter",
}


class SplitterFactory:
    @staticmethod
    def create(settings: Settings) -> BaseSplitter:
        provider = settings.splitter.provider.lower()
        if provider not in _PROVIDER_REGISTRY:
            available = ", ".join(sorted(_PROVIDER_REGISTRY))
            raise ValueError(f"Unknown splitter: '{provider}'. Available: {available}")
        module_path, class_name = _PROVIDER_REGISTRY[provider].rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(settings)

    @staticmethod
    def register_provider(name: str, class_path: str) -> None:
        _PROVIDER_REGISTRY[name.lower()] = class_path
