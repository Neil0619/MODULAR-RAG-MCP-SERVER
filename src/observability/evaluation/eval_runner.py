"""EvalRunner — orchestrates evaluation over a golden test set."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from observability.logger import get_logger

logger = get_logger("eval_runner")


@dataclass
class TestCase:
    """A single test case from the golden test set."""

    query: str
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    """Result of evaluating a single test case."""

    query: str
    retrieved_ids: list[str] = field(default_factory=list)
    golden_ids: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass
class EvalReport:
    """Aggregated evaluation report."""

    total_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    case_results: list[CaseResult] = field(default_factory=list)


class EvalRunner:
    """Run evaluation over a golden test set.

    For each test case, runs hybrid search to get ``top_k`` results,
    then evaluates against golden IDs using the configured evaluator.

    Usage::

        runner = EvalRunner(settings=settings, hybrid_search=search, evaluator=evaluator)
        report = runner.run("tests/fixtures/golden_test_set.json")
        print(report.aggregate_metrics)
    """

    def __init__(
        self,
        settings: Any,
        hybrid_search: Any,
        evaluator: Any,
    ) -> None:
        self._settings = settings
        self._hybrid_search = hybrid_search
        self._evaluator = evaluator

    def run(
        self,
        test_set_path: str | Path,
        top_k: int = 10,
    ) -> EvalReport:
        """Execute evaluation and return an :class:`EvalReport`.

        Args:
            test_set_path: Path to golden test set JSON file.
            top_k: Number of results to retrieve per query.

        Returns:
            :class:`EvalReport` with per-case and aggregate metrics.
        """
        test_cases = self._load_test_set(test_set_path)
        report = EvalReport(total_cases=len(test_cases))

        for tc in test_cases:
            case_result = self._evaluate_case(tc, top_k)
            report.case_results.append(case_result)
            if case_result.error is None:
                report.successful_cases += 1
            else:
                report.failed_cases += 1

        report.aggregate_metrics = self._aggregate_metrics(report.case_results)
        return report

    def _evaluate_case(self, tc: TestCase, top_k: int) -> CaseResult:
        """Run retrieval + evaluation for a single test case."""
        result = CaseResult(
            query=tc.query,
            golden_ids=tc.expected_chunk_ids,
        )

        try:
            search_results = self._hybrid_search.search(
                tc.query, top_k=top_k
            )
            result.retrieved_ids = [r.chunk_id for r in search_results]

            metrics = self._evaluator.evaluate(
                query=tc.query,
                retrieved_ids=result.retrieved_ids,
                golden_ids=result.golden_ids,
            )
            result.metrics = metrics
        except Exception as exc:
            logger.error("Evaluation failed for query '%s': %s", tc.query, exc)
            result.error = str(exc)

        return result

    @staticmethod
    def _aggregate_metrics(case_results: list[CaseResult]) -> dict[str, float]:
        """Average all numeric metrics across successful cases."""
        from collections import defaultdict

        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)

        for cr in case_results:
            if cr.error is not None:
                continue
            for key, value in cr.metrics.items():
                sums[key] += value
                counts[key] += 1

        return {
            key: round(sums[key] / counts[key], 4)
            for key in sums
            if counts[key] > 0
        }

    @staticmethod
    def _load_test_set(path: str | Path) -> list[TestCase]:
        """Load golden test set from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Golden test set not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        cases = data.get("test_cases", [])
        return [
            TestCase(
                query=c["query"],
                expected_chunk_ids=c.get("expected_chunk_ids", []),
                expected_sources=c.get("expected_sources", []),
            )
            for c in cases
        ]
