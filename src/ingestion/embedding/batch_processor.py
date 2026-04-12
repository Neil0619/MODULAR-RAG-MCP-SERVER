"""BatchProcessor — split chunks into batches and drive encoding.

Orchestrates dense + sparse encoding across batches of chunks,
recording per-batch timing for trace/observability.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from core.types import Chunk, ChunkRecord
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext

logger = logging.getLogger("rag.embedding.batch")


class BatchProcessor:
    """Process chunks in batches through dense + sparse encoders.

    Args:
        settings: Application settings.
        batch_size: Number of chunks per encoding batch. Defaults to 32.
        dense_encoder: Optional pre-created DenseEncoder.
        sparse_encoder: Optional pre-created SparseEncoder.
    """

    def __init__(
        self,
        settings: Settings,
        batch_size: int = 32,
        dense_encoder: DenseEncoder | None = None,
        sparse_encoder: SparseEncoder | None = None,
    ) -> None:
        self._batch_size = batch_size
        self._dense = dense_encoder or DenseEncoder(settings)
        self._sparse = sparse_encoder or SparseEncoder()

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def process(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[ChunkRecord]:
        """Encode all chunks in batches, producing ChunkRecords.

        Args:
            chunks: Input chunks to encode.
            trace: Optional trace context for timing.

        Returns:
            List of ChunkRecords with dense + sparse vectors.
        """
        if not chunks:
            return []

        records: list[ChunkRecord] = []
        total_batches = (len(chunks) + self._batch_size - 1) // self._batch_size
        logger.info(
            "BatchProcessor.process: %d chunks → %d batches (batch_size=%d)",
            len(chunks), total_batches, self._batch_size,
        )

        for batch_idx in range(total_batches):
            start = batch_idx * self._batch_size
            end = min(start + self._batch_size, len(chunks))
            batch = chunks[start:end]

            batch_start = time.monotonic()

            logger.info(
                "Encoding batch %d/%d: %d chunks", batch_idx + 1, total_batches, len(batch),
            )
            dense_vectors = self._dense.encode(batch)
            sparse_vectors = self._sparse.encode(batch)

            batch_elapsed = time.monotonic() - batch_start
            logger.info(
                "Batch %d/%d done: elapsed=%.2fs, dense=%d, sparse=%d",
                batch_idx + 1, total_batches, batch_elapsed,
                len(dense_vectors), len(sparse_vectors),
            )

            for chunk, dense_vec, sparse_vec in zip(batch, dense_vectors, sparse_vectors):
                records.append(
                    ChunkRecord.from_chunk(
                        chunk,
                        dense_vector=dense_vec,
                        sparse_vector=sparse_vec,
                    )
                )

            if trace is not None:
                trace.record_stage(
                    f"batch_{batch_idx}",
                    {
                        "chunk_count": len(batch),
                        "elapsed_seconds": round(batch_elapsed, 4),
                    },
                )

        return records

    def _make_batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        """Split chunks into batches (exposed for testing)."""
        batches = []
        for i in range(0, len(chunks), self._batch_size):
            batches.append(chunks[i : i + self._batch_size])
        return batches
