"""DocumentChunker — adapter between libs.splitter and ingestion pipeline.

Converts a :class:`Document` into a list of :class:`Chunk` objects with
proper IDs, metadata inheritance, offset tracking, and image reference
distribution.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from core.types import Chunk, Document
from libs.splitter.splitter_factory import SplitterFactory

if TYPE_CHECKING:
    from core.settings import Settings

# Pattern to find [IMAGE: {image_id}] placeholders
_IMAGE_PLACEHOLDER_RE = re.compile(r"\[IMAGE:\s*([^\]]+)\]")


class DocumentChunker:
    """Split a Document into Chunks using the configured splitter.

    This is an adapter layer that adds business logic on top of the
    pure-text ``libs.splitter``:

    1. Deterministic Chunk IDs (``{doc_id}_{index:04d}_{hash_8}``)
    2. Metadata inheritance from parent Document
    3. ``chunk_index`` in metadata
    4. ``source_ref`` linking back to Document.id
    5. Image reference distribution per chunk
    6. Type conversion: ``str`` → ``Chunk``
    """

    def __init__(self, settings: Settings) -> None:
        self._splitter = SplitterFactory.create(settings)

    def split_document(self, document: Document) -> list[Chunk]:
        """Split a Document into Chunks.

        Args:
            document: The source Document.

        Returns:
            List of Chunk objects with IDs, metadata, and offsets.
        """
        if not document.text.strip():
            return []

        raw_chunks = self._splitter.split_text(document.text)
        if not raw_chunks:
            return []

        # Build a list of (start_offset, end_offset) for each raw chunk
        offsets = self._compute_offsets(document.text, raw_chunks)

        # Get document-level images for distribution
        doc_images = document.metadata.get("images", [])

        chunks: list[Chunk] = []
        for idx, (text, (start, end)) in enumerate(zip(raw_chunks, offsets)):
            chunk_id = _generate_chunk_id(document.id, idx, text)
            metadata = self._inherit_metadata(document, idx, text, doc_images)

            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    metadata=metadata,
                    start_offset=start,
                    end_offset=end,
                    source_ref=document.id,
                )
            )

        return chunks

    def _inherit_metadata(
        self,
        document: Document,
        chunk_index: int,
        chunk_text: str,
        doc_images: list[dict],
    ) -> dict:
        """Build chunk metadata from document metadata + image distribution."""
        # Start with a copy of document metadata (exclude images list)
        metadata = {
            k: v
            for k, v in document.metadata.items()
            if k != "images"
        }
        metadata["chunk_index"] = chunk_index

        # Distribute image references to this chunk
        referenced_ids = _IMAGE_PLACEHOLDER_RE.findall(chunk_text)
        if referenced_ids:
            image_map = {img["id"]: img for img in doc_images if "id" in img}
            chunk_images = [
                image_map[img_id]
                for img_id in referenced_ids
                if img_id in image_map
            ]
            if chunk_images:
                metadata["images"] = chunk_images
            metadata["image_refs"] = referenced_ids

        return metadata

    @staticmethod
    def _compute_offsets(full_text: str, raw_chunks: list[str]) -> list[tuple[int, int]]:
        """Compute (start, end) offsets for each chunk in the full text.

        Uses a sliding search approach — chunks are expected to be
        substrings of the full text in order.
        """
        offsets = []
        search_start = 0
        for chunk_text in raw_chunks:
            pos = full_text.find(chunk_text, search_start)
            if pos == -1:
                # Fallback: append with estimated offset
                start = search_start
                end = start + len(chunk_text)
            else:
                start = pos
                end = pos + len(chunk_text)
                search_start = end
            offsets.append((start, end))
        return offsets


def _generate_chunk_id(doc_id: str, index: int, text: str) -> str:
    """Generate a deterministic chunk ID.

    Format: ``{doc_id}_{index:04d}_{hash_8chars}``
    """
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:8]
    return f"{doc_id}_{index:04d}_{text_hash}"
