"""Video loader — extracts audio + key frames (placeholder for J5)."""

from __future__ import annotations

from typing import Any

from core.types import Document
from libs.loader.base_loader import BaseLoader

try:
    import whisper
except ImportError:
    whisper = None  # type: ignore[assignment]

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]


class VideoLoader(BaseLoader):
    """Load a video file into a :class:`Document`."""

    def load(self, path: str, **kwargs: Any) -> Document:
        if whisper is None:
            raise RuntimeError(
                "openai-whisper is required for video loading. "
                "Install with: pip install openai-whisper"
            )
        if cv2 is None:
            raise RuntimeError(
                "opencv-python is required for video loading. "
                "Install with: pip install opencv-python"
            )
        raise NotImplementedError("VideoLoader will be implemented in J5")
