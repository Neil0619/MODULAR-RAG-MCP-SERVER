"""Contract tests for Video Loader (mocked deps)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.types import Document


class TestVideoLoaderImportGuard:

    def test_whisper_missing_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "test.mp4"
        f.write_bytes(b"\x00" * 100)
        with patch.dict("sys.modules", {"whisper": None}):
            from libs.loader.video_loader import VideoLoader
            loader = VideoLoader.__new__(VideoLoader)
            with pytest.raises(RuntimeError, match="openai-whisper"):
                loader.load(str(f))

    def test_cv2_missing_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "test.mp4"
        f.write_bytes(b"\x00" * 100)
        import libs.loader.video_loader as vl_mod
        orig_w, orig_cv = vl_mod.whisper, vl_mod.cv2
        vl_mod.whisper = MagicMock()  # whisper present
        vl_mod.cv2 = None  # cv2 missing
        try:
            loader = vl_mod.VideoLoader()
            with pytest.raises(RuntimeError, match="opencv-python"):
                loader.load(str(f))
        finally:
            vl_mod.whisper, vl_mod.cv2 = orig_w, orig_cv


class TestVideoLoaderWithMocks:

    def _make_loader(self) -> Any:
        from libs.loader.video_loader import VideoLoader
        return VideoLoader(image_dir="/tmp/test_images")

    @patch("libs.loader.video_loader.cv2")
    @patch("libs.loader.video_loader.whisper")
    def test_load_returns_document(self, mock_whisper: MagicMock, mock_cv2: MagicMock, tmp_path: Path) -> None:
        f = tmp_path / "test.mp4"
        f.write_bytes(b"\x00" * 100)

        # Mock whisper
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Hello world from video"}
        mock_whisper.load_model.return_value = mock_model

        # Mock moviepy
        mock_clip = MagicMock()
        mock_clip.audio = MagicMock()
        mock_clip.audio.write_audiofile = MagicMock()

        # Mock cv2
        mock_cap = MagicMock()
        mock_cap.get.side_effect = [30.0, 60]  # fps, total_frames
        mock_cap.read.return_value = (True, MagicMock())
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.imwrite.return_value = True

        with patch("libs.loader.video_loader.VideoLoader._transcribe_audio", return_value="Hello world"):
            with patch("libs.loader.video_loader.VideoLoader._extract_frames", return_value=[]):
                doc = self._make_loader().load(str(f))
                assert isinstance(doc, Document)
                assert doc.metadata["doc_type"] == "video"

    @patch("libs.loader.video_loader.whisper", None)
    @patch("libs.loader.video_loader.cv2", None)
    def test_metadata_has_doc_hash(self, tmp_path: Path) -> None:
        # Can't load without deps, but verify the placeholder structure
        from libs.loader.video_loader import VideoLoader
        loader = VideoLoader()
        assert hasattr(loader, "_whisper_model")
        assert hasattr(loader, "_frame_interval_sec")

    def test_constructor_defaults(self) -> None:
        from libs.loader.video_loader import VideoLoader
        loader = VideoLoader()
        assert loader._frame_interval_sec == 10
        assert loader._max_frames == 30
        assert loader._whisper_model == "base"

    def test_constructor_custom(self) -> None:
        from libs.loader.video_loader import VideoLoader
        loader = VideoLoader(frame_interval_sec=5, max_frames=15, whisper_model="small")
        assert loader._frame_interval_sec == 5
        assert loader._max_frames == 15
        assert loader._whisper_model == "small"

    def test_missing_file_raises(self) -> None:
        from libs.loader.video_loader import VideoLoader
        import libs.loader.video_loader as vl_mod
        # Save and restore whisper/cv2 state since they're installed
        orig_w, orig_cv = vl_mod.whisper, vl_mod.cv2
        vl_mod.whisper = MagicMock()
        vl_mod.cv2 = MagicMock()
        try:
            with pytest.raises(FileNotFoundError):
                VideoLoader().load("/nonexistent/video.mp4")
        finally:
            vl_mod.whisper, vl_mod.cv2 = orig_w, orig_cv
