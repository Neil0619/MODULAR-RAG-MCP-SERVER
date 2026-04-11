"""Unit tests for EvalRunner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from observability.evaluation.eval_runner import EvalRunner, TestCase, CaseResult, EvalReport


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeSearchResult:
    def __init__(self, chunk_id: str) -> None:
        self.chunk_id = chunk_id
        self.text = "dummy"
        self.score = 0.5
        self.metadata: dict[str, Any] = {}


class _FakeHybridSearch:
    def __init__(self, results: list[list[_FakeSearchResult]] | None = None) -> None:
        self._results = results or []
        self._call_idx = 0

    def search(self, query: str, top_k: int = 10, **kwargs: Any) -> list[_FakeSearchResult]:
        if self._call_idx < len(self._results):
            r = self._results[self._call_idx]
            self._call_idx += 1
            return r
        return []


class _FakeEvaluator:
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        **kwargs: Any,
    ) -> dict[str, float]:
        golden_set = set(golden_ids)
        hits = sum(1 for rid in retrieved_ids if rid in golden_set)
        hit_rate = float(bool(hits)) if golden_ids else 0.0
        return {"hit_rate": hit_rate, "test_metric": 0.75}


class _FailingSearch:
    def search(self, query: str, **kwargs: Any) -> list[Any]:
        raise RuntimeError("search failed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_test_set(path: Path, cases: list[dict[str, Any]]) -> None:
    data = {"test_cases": cases}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadTestSet:

    def test_load_valid_file(self, tmp_path: Path) -> None:
        f = tmp_path / "golden.json"
        _write_test_set(f, [
            {"query": "q1", "expected_chunk_ids": ["a", "b"]},
            {"query": "q2", "expected_sources": ["doc.pdf"]},
        ])
        cases = EvalRunner._load_test_set(f)
        assert len(cases) == 2
        assert cases[0].query == "q1"
        assert cases[0].expected_chunk_ids == ["a", "b"]
        assert cases[1].expected_sources == ["doc.pdf"]

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            EvalRunner._load_test_set(tmp_path / "nope.json")

    def test_load_empty_test_cases(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.json"
        f.write_text(json.dumps({"test_cases": []}))
        cases = EvalRunner._load_test_set(f)
        assert cases == []


class TestEvalRunner:

    def test_run_single_case(self, tmp_path: Path) -> None:
        f = tmp_path / "golden.json"
        _write_test_set(f, [
            {"query": "what is RAG", "expected_chunk_ids": ["c1", "c2"]},
        ])
        search = _FakeHybridSearch([[_FakeSearchResult("c1"), _FakeSearchResult("c3")]])
        evaluator = _FakeEvaluator()
        runner = EvalRunner(settings=None, hybrid_search=search, evaluator=evaluator)
        report = runner.run(f)
        assert report.total_cases == 1
        assert report.successful_cases == 1
        assert report.failed_cases == 0
        assert report.case_results[0].retrieved_ids == ["c1", "c3"]

    def test_run_multiple_cases(self, tmp_path: Path) -> None:
        f = tmp_path / "golden.json"
        _write_test_set(f, [
            {"query": "q1", "expected_chunk_ids": ["a"]},
            {"query": "q2", "expected_chunk_ids": ["b"]},
        ])
        search = _FakeHybridSearch([
            [_FakeSearchResult("a")],
            [_FakeSearchResult("b")],
        ])
        evaluator = _FakeEvaluator()
        runner = EvalRunner(settings=None, hybrid_search=search, evaluator=evaluator)
        report = runner.run(f)
        assert report.total_cases == 2
        assert report.successful_cases == 2

    def test_search_failure_captured(self, tmp_path: Path) -> None:
        f = tmp_path / "golden.json"
        _write_test_set(f, [
            {"query": "q1", "expected_chunk_ids": ["a"]},
        ])
        search = _FailingSearch()
        evaluator = _FakeEvaluator()
        runner = EvalRunner(settings=None, hybrid_search=search, evaluator=evaluator)
        report = runner.run(f)
        assert report.failed_cases == 1
        assert "search failed" in report.case_results[0].error

    def test_aggregate_metrics(self, tmp_path: Path) -> None:
        f = tmp_path / "golden.json"
        _write_test_set(f, [
            {"query": "q1", "expected_chunk_ids": ["a"]},
            {"query": "q2", "expected_chunk_ids": ["b"]},
        ])
        search = _FakeHybridSearch([
            [_FakeSearchResult("a")],
            [_FakeSearchResult("x")],
        ])
        evaluator = _FakeEvaluator()
        runner = EvalRunner(settings=None, hybrid_search=search, evaluator=evaluator)
        report = runner.run(f)
        # hit_rate: 1.0 + 0.0 = 0.5 average
        assert report.aggregate_metrics["hit_rate"] == 0.5
        # test_metric always 0.75
        assert report.aggregate_metrics["test_metric"] == 0.75

    def test_top_k_forwarded(self, tmp_path: Path) -> None:
        f = tmp_path / "golden.json"
        _write_test_set(f, [
            {"query": "q1", "expected_chunk_ids": []},
        ])
        search = _FakeHybridSearch([[_FakeSearchResult("x")]])
        evaluator = _FakeEvaluator()
        runner = EvalRunner(settings=None, hybrid_search=search, evaluator=evaluator)
        report = runner.run(f, top_k=3)
        assert report.total_cases == 1


class TestAggregateMetrics:

    def test_empty_results(self) -> None:
        assert EvalRunner._aggregate_metrics([]) == {}

    def test_all_failed(self) -> None:
        results = [CaseResult(query="q", error="fail")]
        assert EvalRunner._aggregate_metrics(results) == {}

    def test_mixed_success_failure(self) -> None:
        results = [
            CaseResult(query="q1", metrics={"a": 1.0}),
            CaseResult(query="q2", error="fail"),
            CaseResult(query="q3", metrics={"a": 0.5}),
        ]
        agg = EvalRunner._aggregate_metrics(results)
        assert agg["a"] == pytest.approx(0.75)


class TestDataClasses:

    def test_eval_report_defaults(self) -> None:
        r = EvalReport()
        assert r.total_cases == 0
        assert r.aggregate_metrics == {}

    def test_case_result_serializable(self) -> None:
        cr = CaseResult(query="q", metrics={"hit_rate": 1.0})
        d = {"query": cr.query, "metrics": cr.metrics}
        assert json.dumps(d)
