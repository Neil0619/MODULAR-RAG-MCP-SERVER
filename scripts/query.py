#!/usr/bin/env python3
"""Query the knowledge base from the command line.

Usage::

    python scripts/query.py --query "如何配置 Azure？" --top-k 5
    python scripts/query.py --query "RAG pipeline" --collection research --verbose
    python scripts/query.py --query "test" --no-rerank

Runs the full retrieval pipeline: query processing → dense + sparse →
RRF fusion → optional reranking → formatted output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable when running from project root
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.reranker import Reranker
from core.settings import load_settings
from core.trace.trace_context import TraceContext
from core.trace.trace_collector import TraceCollector
from core.types import RetrievalResult


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the Modular RAG knowledge base",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Query text to search for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10).",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="Target collection name (default: default).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show intermediate results from each stage.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the reranking stage.",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings YAML (default: config/settings.yaml).",
    )
    return parser.parse_args(argv)


def _format_result(idx: int, r: RetrievalResult, verbose: bool = False) -> str:
    """Format a single retrieval result for display."""
    source = r.metadata.get("source_path", "unknown")
    chunk_idx = r.metadata.get("chunk_index", "?")
    title = r.metadata.get("title", "")

    lines = [
        f"  [{idx+1}] score={r.score:.4f}  source={source}  chunk={chunk_idx}",
    ]
    if title:
        lines.append(f"      title: {title}")

    # Text preview
    text = r.text[:200].replace("\n", " ")
    if len(r.text) > 200:
        text += "..."
    lines.append(f"      text: {text}")

    if verbose:
        tags = r.metadata.get("tags", [])
        if tags:
            lines.append(f"      tags: {tags}")
        reranked = r.metadata.get("reranked", False)
        fallback = r.metadata.get("fallback", False)
        flags = []
        if reranked:
            flags.append("reranked")
        if fallback:
            flags.append("fallback")
        if flags:
            lines.append(f"      flags: {', '.join(flags)}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the query CLI."""
    args = _parse_args(argv)
    settings = load_settings(args.config)

    # Build pipeline components
    qp = QueryProcessor()
    hybrid = HybridSearch(settings, query_processor=qp)
    reranker = Reranker(settings) if not args.no_rerank else None
    collector = TraceCollector()

    trace = TraceContext(trace_type="query")

    if args.verbose:
        pq = qp.process(args.query)
        print(f"Keywords: {pq.keywords}", file=sys.stderr)
        print(f"Filters: {pq.filters}", file=sys.stderr)

    # Stage 1: Hybrid search
    results = hybrid.search(
        args.query,
        top_k=args.top_k,
        collection=args.collection,
        trace=trace,
    )

    if not results:
        print("\n未找到相关文档，请先运行 ingest.py 摄取数据。")
        return 0

    if args.verbose:
        print(f"\n--- Fusion results: {len(results)} candidates ---", file=sys.stderr)
        for i, r in enumerate(results[:3]):
            print(f"  {i+1}. {r.chunk_id} score={r.score:.4f}", file=sys.stderr)

    # Stage 2: Reranking (optional)
    if reranker is not None:
        results = reranker.rerank(args.query, results, top_k=args.top_k, trace=trace)
        if args.verbose:
            fb = sum(1 for r in results if r.metadata.get("fallback"))
            print(f"\n--- Reranked: {len(results)} results", file=sys.stderr)
            if fb:
                print(f"    (fallback: {fb} results)", file=sys.stderr)

    # Persist trace
    trace.finish()
    collector.collect(trace)

    # Output
    print(f"\n{'='*60}")
    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")
    print(f"{'='*60}")

    for i, r in enumerate(results):
        print(_format_result(i, r, verbose=args.verbose))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
