"""PDF loader using PyMuPDF (fitz).

Extracts text and embedded images from PDF files. Images are saved to
``data/images/{doc_hash}/`` and marked with ``[IMAGE: {image_id}]``
placeholders in the document text.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

from core.types import Document, ImageRef
from libs.loader.base_loader import BaseLoader

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore[assignment]


class PdfLoader(BaseLoader):
    """Load a PDF file into a :class:`Document`.

    Extracts page text and embedded images. Images are saved to disk
    and referenced via ``[IMAGE: {image_id}]`` placeholders.

    If PyMuPDF is not installed, :meth:`load` will raise a
    ``RuntimeError`` with an installation hint.

    Args:
        image_dir: Root directory for extracted images.
            Defaults to ``data/images``.
    """

    def __init__(self, image_dir: str = "data/images") -> None:
        self._image_dir = image_dir

    def load(self, path: str, **kwargs: Any) -> Document:
        """Load a PDF file.

        Args:
            path: Path to the PDF file.

        Returns:
            Document with text, images metadata, and source_path.

        Raises:
            RuntimeError: If PyMuPDF is not installed.
            FileNotFoundError: If the file does not exist.
        """
        if fitz is None:
            raise RuntimeError(
                "PyMuPDF is required for PDF loading. "
                "Install with: pip install pymupdf"
            )

        file_path = self._validate_path(path)
        doc_hash = self._file_hash(file_path)
        collection_dir = Path(self._image_dir) / doc_hash
        collection_dir.mkdir(parents=True, exist_ok=True)

        try:
            doc = fitz.open(str(file_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to open PDF {path}: {exc}") from exc

        try:
            full_text_parts: list[str] = []
            images: list[dict[str, Any]] = []
            image_seq = 0

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text") or ""

                # Extract images from this page
                img_list = page.get_images(full=True)
                for img_info in img_list:
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        if base_image is None:
                            continue
                        img_bytes = base_image["image"]
                        img_ext = base_image.get("ext", "png")

                        image_id = f"{doc_hash}_{page_num}_{image_seq}"
                        img_filename = f"{image_id}.{img_ext}"
                        img_path = collection_dir / img_filename
                        img_path.write_bytes(img_bytes)

                        # Compute placeholder
                        placeholder = f"[IMAGE: {image_id}]"

                        # Get image position on page
                        rects = page.get_image_rects(xref)
                        position: dict[str, Any] = {}
                        if rects:
                            r = rects[0]
                            position = {
                                "x": round(r.x0, 2),
                                "y": round(r.y0, 2),
                                "width": round(r.width, 2),
                                "height": round(r.height, 2),
                            }

                        # Append placeholder after page text
                        page_text += "\n" + placeholder + "\n"

                        images.append(
                            ImageRef(
                                id=image_id,
                                path=str(img_path),
                                page=page_num,
                                text_offset=0,  # Will be computed after join
                                text_length=len(placeholder),
                                position=position,
                            ).to_dict()
                        )
                        image_seq += 1
                    except Exception:
                        # Image extraction failure should not block text parsing
                        import logging

                        logging.getLogger(__name__).warning(
                            "Failed to extract image xref=%s from %s",
                            xref,
                            path,
                            exc_info=True,
                        )
                        continue

                full_text_parts.append(page_text)

            full_text = "\n".join(full_text_parts)

            # Fix text_offset for each image by finding placeholder in text
            for img in images:
                placeholder = f"[IMAGE: {img['id']}]"
                offset = full_text.find(placeholder)
                if offset >= 0:
                    img["text_offset"] = offset

            page_count = len(doc)
        finally:
            doc.close()

        return Document(
            text=full_text,
            metadata={
                "source_path": str(file_path),
                "doc_hash": doc_hash,
                "doc_type": "pdf",
                "page_count": page_count,
                "images": images,
            },
        )

    @staticmethod
    def _file_hash(path: Path) -> str:
        """Compute a short hash of the file for directory naming."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
