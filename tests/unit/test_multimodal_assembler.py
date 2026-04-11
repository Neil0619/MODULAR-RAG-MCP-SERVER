"""Unit tests for MultimodalAssembler (E6)."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from core.response.multimodal_assembler import MultimodalAssembler, _guess_mime
from core.types import RetrievalResult


def _rr(cid: str, text: str = "chunk text", meta: dict | None = None) -> RetrievalResult:
    return RetrievalResult(chunk_id=cid, score=0.9, text=text, metadata=meta or {}, source="fusion")


class TestMultimodalAssembler:
    """Tests for multimodal content assembly."""

    def test_text_only_results(self) -> None:
        results = [_rr("c1", "Hello world"), _rr("c2", "Another chunk")]
        content = MultimodalAssembler.assemble(results)
        assert len(content) == 2
        assert all(c["type"] == "text" for c in content)

    def test_image_included_from_disk(self, tmp_path: Path) -> None:
        # Create a fake image file
        img_path = tmp_path / "test_img.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        results = [_rr("c1", "See image", {
            "image_refs": ["img_001"],
            "images": [{"id": "img_001", "path": str(img_path)}],
        })]
        content = MultimodalAssembler.assemble(results)
        types = [c["type"] for c in content]
        assert "text" in types
        assert "image" in types

        # Verify image content
        img_item = next(c for c in content if c["type"] == "image")
        assert img_item["mimeType"] == "image/png"
        # Verify base64 is valid
        decoded = base64.b64decode(img_item["data"])
        assert decoded.startswith(b"\x89PNG")

    def test_missing_image_file_skipped(self, tmp_path: Path) -> None:
        results = [_rr("c1", "text", {
            "image_refs": ["img_999"],
            "images": [{"id": "img_999", "path": str(tmp_path / "nonexistent.png")}],
        })]
        content = MultimodalAssembler.assemble(results)
        assert all(c["type"] == "text" for c in content)

    def test_no_image_refs(self) -> None:
        results = [_rr("c1", "plain text")]
        content = MultimodalAssembler.assemble(results)
        assert all(c["type"] == "text" for c in content)

    def test_max_images_limit(self, tmp_path: Path) -> None:
        results = []
        for i in range(5):
            img_path = tmp_path / f"img_{i}.png"
            img_path.write_bytes(b"\x89PNG" + bytes([i]) * 10)
            results.append(_rr(f"c{i}", f"text {i}", {
                "image_refs": [f"img_{i}"],
                "images": [{"id": f"img_{i}", "path": str(img_path)}],
            }))

        content = MultimodalAssembler.assemble(results, max_images=2)
        images = [c for c in content if c["type"] == "image"]
        assert len(images) <= 2

    def test_include_text_false(self, tmp_path: Path) -> None:
        img_path = tmp_path / "img.png"
        img_path.write_bytes(b"\x89PNG\x00" * 20)

        results = [_rr("c1", "text", {
            "image_refs": ["img_0"],
            "images": [{"id": "img_0", "path": str(img_path)}],
        })]
        content = MultimodalAssembler.assemble(results, include_text=False)
        types = [c["type"] for c in content]
        assert "text" not in types
        assert "image" in types

    def test_has_images(self) -> None:
        r1 = _rr("c1", "text", {"image_refs": ["img_1"]})
        assert MultimodalAssembler.has_images([r1]) is True

    def test_has_images_false(self) -> None:
        r1 = _rr("c1", "text")
        assert MultimodalAssembler.has_images([r1]) is False

    def test_guess_mime(self) -> None:
        assert _guess_mime("photo.png") == "image/png"
        assert _guess_mime("photo.jpg") == "image/jpeg"
        assert _guess_mime("photo.jpeg") == "image/jpeg"
        assert _guess_mime("photo.gif") == "image/gif"
        assert _guess_mime("photo.webp") == "image/webp"
        assert _guess_mime("photo.unknown") == "application/octet-stream"

    def test_empty_results(self) -> None:
        content = MultimodalAssembler.assemble([])
        assert content == []
