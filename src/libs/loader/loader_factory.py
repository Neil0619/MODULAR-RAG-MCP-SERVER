"""Factory for creating document loaders based on file extension.

Unlike other factories in this project that route on ``settings.provider``,
``LoaderFactory`` routes on **file extension** — the user selects a file,
not a loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from libs.loader.base_loader import BaseLoader

_EXTENSION_REGISTRY: dict[str, str] = {
    # Text
    "txt": "libs.loader.txt_loader.TxtLoader",
    "md": "libs.loader.markdown_loader.MarkdownLoader",
    "markdown": "libs.loader.markdown_loader.MarkdownLoader",
    # Office
    "docx": "libs.loader.docx_loader.DocxLoader",
    "pptx": "libs.loader.pptx_loader.PptxLoader",
    # Structured
    "html": "libs.loader.html_loader.HtmlLoader",
    "htm": "libs.loader.html_loader.HtmlLoader",
    "csv": "libs.loader.csv_loader.CsvLoader",
    # PDF
    "pdf": "libs.loader.pdf_loader.PdfLoader",
    # Video
    "mp4": "libs.loader.video_loader.VideoLoader",
    "avi": "libs.loader.video_loader.VideoLoader",
    "mov": "libs.loader.video_loader.VideoLoader",
    "mkv": "libs.loader.video_loader.VideoLoader",
    "webm": "libs.loader.video_loader.VideoLoader",
}


class LoaderFactory:
    """Create loaders based on file extension."""

    @staticmethod
    def create_from_path(file_path: str) -> BaseLoader:
        """Create a loader based on file extension.

        Args:
            file_path: Path to the file to load.

        Returns:
            A :class:`BaseLoader` instance for the given file type.

        Raises:
            ValueError: If the file extension is not supported.
        """
        ext = Path(file_path).suffix.lower().lstrip(".")
        if ext not in _EXTENSION_REGISTRY:
            available = ", ".join(sorted(_EXTENSION_REGISTRY.keys()))
            raise ValueError(
                f"Unsupported file type: '.{ext}'. "
                f"Supported extensions: {available}"
            )
        module_path, class_name = _EXTENSION_REGISTRY[ext].rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls()

    @staticmethod
    def create_from_name(name: str) -> BaseLoader:
        """Create a loader by registered extension name (e.g. 'pdf', 'docx').

        Args:
            name: File extension without the dot.

        Returns:
            A :class:`BaseLoader` instance.

        Raises:
            ValueError: If the name is not registered.
        """
        name = name.lower().lstrip(".")
        if name not in _EXTENSION_REGISTRY:
            available = ", ".join(sorted(_EXTENSION_REGISTRY.keys()))
            raise ValueError(
                f"Unsupported file type: '{name}'. "
                f"Supported extensions: {available}"
            )
        module_path, class_name = _EXTENSION_REGISTRY[name].rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls()

    @staticmethod
    def register_extension(ext: str, class_path: str) -> None:
        """Register a new extension at runtime.

        Args:
            ext: File extension without the dot (e.g. 'json').
            class_path: Fully qualified class path (e.g. 'libs.loader.json_loader.JsonLoader').
        """
        _EXTENSION_REGISTRY[ext.lower().lstrip(".")] = class_path

    @staticmethod
    def supported_extensions() -> list[str]:
        """Return sorted list of all registered file extensions."""
        return sorted(_EXTENSION_REGISTRY.keys())
