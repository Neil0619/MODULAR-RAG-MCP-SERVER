"""Video loader — extracts audio transcription + key frames."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from core.types import Document, ImageRef
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
    """Load a video file into a :class:`Document`.

    Three-stage pipeline:
    1. Audio extraction → Whisper transcription
    2. Key frame extraction at regular intervals
    3. Combine into Document with ``[IMAGE: {id}]`` placeholders

    Requires ``openai-whisper`` and ``opencv-python``.
    """

    def __init__(
        self,
        image_dir: str = "data/images",
        frame_interval_sec: int = 10,
        max_frames: int = 30,
        whisper_model: str = "base",
    ) -> None:
        self._image_dir = image_dir
        self._frame_interval_sec = frame_interval_sec
        self._max_frames = max_frames
        self._whisper_model = whisper_model

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

        file_path = self._validate_path(path)
        doc_hash = self._file_hash(file_path)
        collection_dir = Path(self._image_dir) / doc_hash
        collection_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Audio transcription
        transcript = self._transcribe_audio(file_path)

        # Stage 2: Frame extraction
        frames = self._extract_frames(file_path, collection_dir, doc_hash)

        # Stage 3: Combine into Document
        text_parts: list[str] = []
        if transcript:
            text_parts.append("## Transcript\n\n" + transcript)
        if frames:
            frame_section = "## Key Frames\n\n"
            for frame in frames:
                placeholder = f"[IMAGE: {frame['id']}]"
                timestamp = frame.get("position", {}).get("timestamp_sec", 0)
                frame_section += f"Frame at {timestamp:.1f}s: {placeholder}\n\n"
            text_parts.append(frame_section)

        full_text = "\n\n".join(text_parts)

        # Fix text offsets
        for frame in frames:
            placeholder = f"[IMAGE: {frame['id']}]"
            offset = full_text.find(placeholder)
            if offset >= 0:
                frame["text_offset"] = offset

        return Document(
            text=full_text,
            metadata={
                "source_path": str(file_path),
                "doc_hash": doc_hash,
                "doc_type": "video",
                "transcript_length": len(transcript) if transcript else 0,
                "frame_count": len(frames),
                "images": frames,
            },
        )

    def _transcribe_audio(self, video_path: Path) -> str:
        """Extract audio from video and transcribe with Whisper."""
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            raise RuntimeError(
                "moviepy is required for video audio extraction. "
                "Install with: pip install moviepy"
            )

        tmp_audio_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_audio_path = tmp.name

            clip = VideoFileClip(str(video_path))
            if clip.audio is not None:
                clip.audio.write_audiofile(tmp_audio_path, verbose=False, logger=None)
                clip.close()

                model = whisper.load_model(self._whisper_model)
                result = model.transcribe(tmp_audio_path)
                return result["text"].strip()
            else:
                clip.close()
                return ""
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Audio transcription failed for %s: %s", video_path, exc
            )
            return ""
        finally:
            if tmp_audio_path:
                Path(tmp_audio_path).unlink(missing_ok=True)

    def _extract_frames(
        self, video_path: Path, output_dir: Path, doc_hash: str
    ) -> list[dict[str, Any]]:
        """Extract key frames from video at regular intervals."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            cap.release()
            return []

        frame_interval = int(fps * self._frame_interval_sec)
        frames: list[dict[str, Any]] = []
        seq = 0

        for frame_idx in range(0, total_frames, max(frame_interval, 1)):
            if seq >= self._max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            timestamp = frame_idx / fps
            image_id = f"{doc_hash}_frame_{seq}"
            img_path = output_dir / f"{image_id}.jpg"
            cv2.imwrite(str(img_path), frame)

            placeholder = f"[IMAGE: {image_id}]"
            frames.append(
                ImageRef(
                    id=image_id,
                    path=str(img_path),
                    page=int(timestamp),
                    text_offset=0,
                    text_length=len(placeholder),
                    position={"timestamp_sec": round(timestamp, 2)},
                ).to_dict()
            )
            seq += 1

        cap.release()
        return frames

    @staticmethod
    def _file_hash(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
