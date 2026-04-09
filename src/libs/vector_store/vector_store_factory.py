"""Factory for creating VectorStore instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from core.settings import Settings

_PROVIDER_REGISTRY: dict[str, str] = {
    "chroma": "libs.vector_store.chroma_store.ChromaStore",
}


class VectorStoreFactory:
    @staticmethod
    def create(settings: Settings) -> BaseVectorStore:
        backend = settings.vector_store.backend.lower()
        if backend not in _PROVIDER_REGISTRY:
            available = ", ".join(sorted(_PROVIDER_REGISTRY))
            raise ValueError(f"Unknown vector store: '{backend}'. Available: {available}")
        module_path, class_name = _PROVIDER_REGISTRY[backend].rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(settings)

    @staticmethod
    def register_provider(name: str, class_path: str) -> None:
        _PROVIDER_REGISTRY[name.lower()] = class_path
