"""Ingestion Pipeline — orchestrates the full document ingestion flow.

Pipeline stages:
1. **Integrity check** — skip already-processed files (SHA256)
2. **Load** — extract text + images from PDF
3. **Split** — chunk document into segments
4. **Transform** — refine text + enrich metadata + caption images
5. **Encode** — dense + sparse vector encoding (batched)
6. **Store** — upsert vectors + build BM25 index
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from core.trace.trace_context import TraceContext
from core.trace.trace_collector import TraceCollector
from core.types import Document
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.vector_upserter import VectorUpserter
from ingestion.transform.chunk_refiner import ChunkRefiner
from ingestion.transform.image_captioner import ImageCaptioner
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.loader.pdf_loader import PdfLoader

if TYPE_CHECKING:
    from core.settings import Settings

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when a pipeline stage fails."""


class IngestionPipeline:
    """Orchestrate the full document ingestion pipeline.

    Args:
        settings: Application settings.
        collection: Collection name for storage.
    """

    def __init__(self, settings: Settings, collection: str = "default") -> None:
        self._settings = settings
        self._collection = collection
        self._trace = TraceContext(trace_type="ingestion")
        self._collector = TraceCollector()

    @property
    def trace(self) -> TraceContext:
        return self._trace

    def run(
        self,
        file_path: str,
        force: bool = False,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Execute the full ingestion pipeline for a single file.

        Args:
            file_path: Path to the document file (PDF).
            force: Skip integrity check and re-ingest.
            on_progress: Optional callback ``(stage_name, current, total)``
                invoked at the end of each stage.  When *None*, no callback
                is made (existing behaviour unchanged).

        Returns:
            Summary dict with stats from each stage.

        Raises:
            PipelineError: If any stage fails.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise PipelineError(f"File not found: {file_path}")

        summary: dict[str, Any] = {
            "file": str(path),
            "collection": self._collection,
            "stages": {},
        }

        # Total number of main stages (used for progress reporting)
        _TOTAL_STAGES = 5
        _stage_index = 0

        def _advance(stage_name: str) -> None:
            nonlocal _stage_index
            _stage_index += 1
            if on_progress is not None:
                on_progress(stage_name, _stage_index, _TOTAL_STAGES)

        # Stage 1: Integrity check
        if not force:
            checker = SQLiteIntegrityChecker()
            file_hash = checker.compute_sha256(str(path))
            if checker.should_skip(file_hash):
                logger.info("Skipping already-ingested file: %s", path)
                summary["skipped"] = True
                return summary
            summary["file_hash"] = file_hash

        # Stage 2: Load
        try:
            logger.info("Loading: %s", path)
            self._trace.start_stage("load")
            loader = PdfLoader()
            document = loader.load(str(path))
            summary["stages"]["load"] = {
                "method": "markitdown",
                "text_length": len(document.text),
                "page_count": document.metadata.get("page_count", 0),
                "image_count": len(document.metadata.get("images", [])),
            }
            self._trace.record_stage("load", summary["stages"]["load"])
            _advance("load")
        except Exception as exc:
            raise PipelineError(f"Load stage failed: {exc}") from exc

        # Stage 3: Split
        try:
            logger.info("Splitting document into chunks")
            self._trace.start_stage("split")
            chunker = DocumentChunker(self._settings)
            chunks = chunker.split_document(document)
            summary["stages"]["split"] = {
                "method": "recursive",
                "chunk_count": len(chunks),
            }
            self._trace.record_stage("split", summary["stages"]["split"])
            _advance("split")
        except Exception as exc:
            raise PipelineError(f"Split stage failed: {exc}") from exc

        if not chunks:
            summary["stages"]["skip_reason"] = "no_chunks"
            return summary

        # Stage 4: Transform (refine → enrich → caption)
        try:
            logger.info("Transforming chunks")
            self._trace.start_stage("transform")
            refiner = ChunkRefiner(self._settings)
            chunks = refiner.transform(chunks, trace=self._trace)

            enricher = MetadataEnricher(self._settings)
            chunks = enricher.transform(chunks, trace=self._trace)

            captioner = ImageCaptioner(self._settings)
            chunks = captioner.transform(chunks, trace=self._trace)

            summary["stages"]["transform"] = {"chunk_count": len(chunks)}
            self._trace.record_stage("transform", summary["stages"]["transform"])
            _advance("transform")
        except Exception as exc:
            raise PipelineError(f"Transform stage failed: {exc}") from exc

        # Stage 5: Encode (dense + sparse)
        try:
            logger.info("Encoding chunks (dense + sparse)")
            self._trace.start_stage("encode")
            processor = BatchProcessor(self._settings)
            records = processor.process(chunks, trace=self._trace)
            summary["stages"]["encode"] = {
                "record_count": len(records),
                "method": "dense+sparse",
            }
            self._trace.record_stage("encode", summary["stages"]["encode"])
            _advance("encode")
        except Exception as exc:
            raise PipelineError(f"Encode stage failed: {exc}") from exc

        # Stage 6: Store (vectors + BM25)
        try:
            logger.info("Storing vectors and BM25 index")
            self._trace.start_stage("store")
            upserter = VectorUpserter(self._settings)
            upserted = upserter.upsert(records, collection=self._collection)
            summary["stages"]["store"] = {"method": "chroma", "upserted": upserted}

            indexer = BM25Indexer()
            indexer.build(records)
            indexer.save(self._collection)
            summary["stages"]["store"]["bm25_terms"] = indexer.term_count
            self._trace.record_stage("store", summary["stages"]["store"])
            _advance("store")
        except Exception as exc:
            raise PipelineError(f"Store stage failed: {exc}") from exc

        # Mark success
        if not force and "file_hash" in summary:
            checker = SQLiteIntegrityChecker()
            checker.mark_success(summary["file_hash"], str(path), chunks=len(chunks))

        self._trace.finish()
        self._collector.collect(self._trace)
        logger.info("Pipeline complete: %d chunks ingested", len(chunks))
        return summary
