#!/usr/bin/env python3
"""Ingest documents into the knowledge base.

Usage::

    python scripts/ingest.py --path docs/paper.pdf --collection research
    python scripts/ingest.py --path docs/ --collection research
    python scripts/ingest.py --path docs/paper.pdf --force

Accepts a single file or a directory. When a directory is given, all
``.pdf`` files inside it (non-recursive) are ingested.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable when running from project root
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.settings import load_settings
from ingestion.pipeline import IngestionPipeline, PipelineError


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the knowledge base",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to a PDF file or directory containing PDF files.",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="Target collection name (default: default).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if the file has already been processed.",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings YAML (default: config/settings.yaml).",
    )
    return parser.parse_args(argv)


def _collect_files(path: str) -> list[Path]:
    """Return a list of PDF files from the given path."""
    p = Path(path).resolve()
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() == ".pdf")
        if not files:
            print(f"No PDF files found in {p}", file=sys.stderr)
        return files
    print(f"Path not found: {p}", file=sys.stderr)
    return []


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ingest CLI."""
    args = _parse_args(argv)
    settings = load_settings(args.config)

    files = _collect_files(args.path)
    if not files:
        return 1

    pipeline = IngestionPipeline(settings, collection=args.collection)

    success = 0
    for fp in files:
        print(f"\n{'='*60}")
        print(f"Processing: {fp.name}")
        print(f"{'='*60}")
        try:
            summary = pipeline.run(str(fp), force=args.force)

            if summary.get("skipped"):
                print(f"  SKIPPED (already ingested)")
                success += 1
                continue

            stages = summary.get("stages", {})
            load_info = stages.get("load", {})
            split_info = stages.get("split", {})
            encode_info = stages.get("encode", {})
            store_info = stages.get("store", {})

            print(f"  Load:   {load_info.get('text_length', 0)} chars, "
                  f"{load_info.get('page_count', 0)} pages, "
                  f"{load_info.get('image_count', 0)} images")
            print(f"  Split:  {split_info.get('chunk_count', 0)} chunks")
            print(f"  Encode: {encode_info.get('record_count', 0)} records")
            print(f"  Store:  {store_info.get('upserted', 0)} upserted, "
                  f"{store_info.get('bm25_terms', 0)} BM25 terms")

            skip = stages.get("skip_reason")
            if skip:
                print(f"  NOTE: {skip}")

            success += 1

        except PipelineError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"  UNEXPECTED: {exc}", file=sys.stderr)

    print(f"\n{'='*60}")
    print(f"Done: {success}/{len(files)} files processed successfully")
    return 0 if success == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
