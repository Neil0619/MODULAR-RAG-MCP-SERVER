"""DOCX loader — reads .docx files using python-docx."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from core.types import Document
from libs.loader.base_loader import BaseLoader

try:
    import docx  # python-docx
except ImportError:
    docx = None  # type: ignore[assignment]


class DocxLoader(BaseLoader):
    """Load a DOCX file into a :class:`Document`.

    Extracts paragraphs, tables, and images. Images are saved to
    ``data/images/{doc_hash}/`` with ``[IMAGE: {id}]`` placeholders.

    Requires the ``python-docx`` package.
    """

    def __init__(self, image_dir: str = "data/images") -> None:
        self._image_dir = image_dir

    def load(self, path: str, **kwargs: Any) -> Document:
        if docx is None:
            raise RuntimeError(
                "python-docx is required for DOCX loading. "
                "Install with: pip install python-docx"
            )

        file_path = self._validate_path(path)
        doc_hash = self._file_hash(file_path)
        collection_dir = Path(self._image_dir) / doc_hash
        collection_dir.mkdir(parents=True, exist_ok=True)

        document = docx.Document(str(file_path))

        text_parts: list[str] = []
        images: list[dict[str, Any]] = []
        image_seq = 0

        # Extract paragraphs
        for para in document.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        # Extract tables
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                text_parts.append(" | ".join(cells))

        # Extract images from inline shapes
        for rel_id, rel in document.part.rels.items():
            if "image" in rel.reltype:
                try:
                    image_data = rel.target_part.blob
                    ext = rel.target_part.content_type.split("/")[-1]
                    if ext == "jpeg":
                        ext = "jpg"

                    image_id = f"{doc_hash}_img_{image_seq}"
                    img_filename = f"{image_id}.{ext}"
                    img_path = collection_dir / img_filename
                    img_path.write_bytes(image_data)

                    placeholder = f"[IMAGE: {image_id}]"
                    text_parts.append(placeholder)

                    images.append({
                        "id": image_id,
                        "path": str(img_path),
                        "page": None,
                        "text_offset": 0,
                        "text_length": len(placeholder),
                        "position": {},
                    })
                    image_seq += 1
                except Exception:
                    continue

        full_text = "\n\n".join(text_parts)

        # Fix text_offsets
        for img in images:
            offset = full_text.find(f"[IMAGE: {img['id']}]")
            if offset >= 0:
                img["text_offset"] = offset

        return Document(
            text=full_text,
            metadata={
                "source_path": str(file_path),
                "doc_hash": doc_hash,
                "doc_type": "docx",
                "paragraph_count": len(document.paragraphs),
                "table_count": len(document.tables),
                "images": images,
            },
        )

    @staticmethod
    def _file_hash(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
