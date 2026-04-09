"""Factory for creating Reranker instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from libs.reranker.base_reranker import BaseReranker, NoneReranker

if TYPE_CHECKING:
    from core.settings import Settings

_PROVIDER_REGISTRY: dict[str, str] = {
    "none": "",  # special case — no import needed
    "cross_encoder": "libs.reranker.cross_encoder_reranker.CrossEncoderReranker",
    "llm": "libs.reranker.llm_reranker.LLMReranker",
}


class RerankerFactory:
    @staticmethod
    def create(settings: Settings) -> BaseReranker:
        backend = settings.rerank.backend.lower()
        if backend == "none":
            return NoneReranker()
        if backend not in _PROVIDER_REGISTRY:
            available = ", ".join(sorted(_PROVIDER_REGISTRY))
            raise ValueError(f"Unknown reranker: '{backend}'. Available: {available}")
        module_path, class_name = _PROVIDER_REGISTRY[backend].rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(settings)

    @staticmethod
    def register_provider(name: str, class_path: str) -> None:
        _PROVIDER_REGISTRY[name.lower()] = class_path
