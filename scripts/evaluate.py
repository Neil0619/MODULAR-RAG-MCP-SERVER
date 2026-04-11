#!/usr/bin/env python3
"""Evaluate RAG retrieval quality against a golden test set.

Usage::

    python scripts/evaluate.py
    python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.settings import load_settings
from core.query_engine.hybrid_search import HybridSearch
from observability.evaluation.composite_evaluator import CompositeEvaluator
from observability.evaluation.eval_runner import EvalRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG quality")
    parser.add_argument(
        "--test-set",
        default="tests/fixtures/golden_test_set.json",
        help="Path to golden test set JSON",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval top-k")
    args = parser.parse_args()

    settings = load_settings()

    # Build hybrid search (same as scripts/query.py)
    hybrid = HybridSearch(settings)

    # Build evaluator from config
    evaluator = CompositeEvaluator.from_settings(settings)

    runner = EvalRunner(settings=settings, hybrid_search=hybrid, evaluator=evaluator)
    report = runner.run(args.test_set, top_k=args.top_k)

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Report  ({report.successful_cases}/{report.total_cases} cases)")
    print(f"{'='*60}")

    for cr in report.case_results:
        status = "OK" if cr.error is None else f"FAIL: {cr.error}"
        print(f"\n  Query: {cr.query}")
        print(f"  Status: {status}")
        if cr.metrics:
            for k, v in cr.metrics.items():
                print(f"    {k}: {v:.4f}")

    print(f"\n{'='*60}")
    print("Aggregate Metrics")
    print(f"{'='*60}")
    for k, v in report.aggregate_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
