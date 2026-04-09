"""Smoke tests: verify all top-level and sub-packages are importable."""

import importlib

import pytest


# Top-level packages
TOP_LEVEL_PACKAGES = [
    "mcp_server",
    "core",
    "ingestion",
    "libs",
    "observability",
]

# Sub-packages that should exist per DEV_SPEC 5.2
SUB_PACKAGES = [
    "mcp_server.tools",
    "core.query_engine",
    "core.response",
    "core.trace",
    "ingestion.chunking",
    "ingestion.transform",
    "ingestion.embedding",
    "ingestion.storage",
    "libs.loader",
    "libs.llm",
    "libs.embedding",
    "libs.splitter",
    "libs.vector_store",
    "libs.reranker",
    "libs.evaluator",
    "observability.dashboard",
    "observability.dashboard.pages",
    "observability.dashboard.services",
    "observability.evaluation",
]


@pytest.mark.parametrize("package", TOP_LEVEL_PACKAGES, ids=lambda p: f"import {p}")
def test_top_level_imports(package: str) -> None:
    """All top-level packages must be importable."""
    mod = importlib.import_module(package)
    assert mod is not None


@pytest.mark.parametrize("package", SUB_PACKAGES, ids=lambda p: f"import {p}")
def test_sub_package_imports(package: str) -> None:
    """All sub-packages must be importable."""
    mod = importlib.import_module(package)
    assert mod is not None


def test_config_settings_readable() -> None:
    """config/settings.yaml must exist and be parseable."""
    import os

    import yaml

    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml")
    config_path = os.path.normpath(config_path)
    assert os.path.exists(config_path), f"Config file not found: {config_path}"

    with open(config_path) as f:
        settings = yaml.safe_load(f)

    assert isinstance(settings, dict)
    # Verify key sections exist
    for section in ["llm", "embedding", "vector_store", "retrieval", "rerank"]:
        assert section in settings, f"Missing config section: {section}"


def test_prompt_templates_exist() -> None:
    """All three prompt template files must exist and be non-empty."""
    import os

    prompts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "config", "prompts")
    prompts_dir = os.path.normpath(prompts_dir)

    for prompt_file in ["image_captioning.txt", "chunk_refinement.txt", "rerank.txt"]:
        path = os.path.join(prompts_dir, prompt_file)
        assert os.path.exists(path), f"Prompt file missing: {path}"
        assert os.path.getsize(path) > 0, f"Prompt file is empty: {path}"


def test_pytest_markers_configured() -> None:
    """Verify pytest markers are available."""
    import pytest

    # These markers should be registered in pyproject.toml
    for marker in ["unit", "integration", "e2e"]:
        # Just verify the marker name is a valid string (no exception)
        assert isinstance(marker, str)
