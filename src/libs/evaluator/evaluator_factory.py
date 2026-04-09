"""Factory for creating Evaluator instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from libs.evaluator.base_evaluator import BaseEvaluator

if TYPE_CHECKING:
    from core.settings import Settings

_PROVIDER_REGISTRY: dict[str, str] = {
    "custom": "libs.evaluator.custom_evaluator.CustomEvaluator",
    "ragas": "libs.evaluator.ragas_evaluator.RagasEvaluator",
}


class EvaluatorFactory:
    @staticmethod
    def create(settings: Settings) -> BaseEvaluator:
        from libs.evaluator.custom_evaluator import CustomEvaluator
        return CustomEvaluator()

    @staticmethod
    def create_from_name(name: str, settings: Settings) -> BaseEvaluator:
        name = name.lower()
        if name not in _PROVIDER_REGISTRY:
            available = ", ".join(sorted(_PROVIDER_REGISTRY))
            raise ValueError(f"Unknown evaluator: '{name}'. Available: {available}")
        module_path, class_name = _PROVIDER_REGISTRY[name].rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(settings)

    @staticmethod
    def register_provider(name: str, class_path: str) -> None:
        _PROVIDER_REGISTRY[name.lower()] = class_path
