"""Evaluation Panel page — run evaluations and view metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from observability.evaluation.eval_runner import EvalRunner


def render() -> None:
    """Render the Evaluation Panel page."""
    st.header("Evaluation Panel")

    # --- Configuration ---
    st.subheader("Configuration")

    default_test_set = "tests/fixtures/golden_test_set.json"
    test_set_path = st.text_input("Golden Test Set path", value=default_test_set)

    top_k = st.number_input("Top-K", min_value=1, max_value=100, value=10)

    # Check test set exists
    ts_path = Path(test_set_path)
    if not ts_path.exists():
        st.warning(f"Test set not found: `{test_set_path}`. Create it first.")
        return

    # Show test case count
    try:
        with open(ts_path, encoding="utf-8") as f:
            data = json.load(f)
        n_cases = len(data.get("test_cases", []))
        st.metric("Test Cases", n_cases)
    except Exception as exc:
        st.error(f"Failed to read test set: {exc}")
        return

    # --- Run Evaluation ---
    if st.button("Run Evaluation", type="primary"):
        _run_evaluation(test_set_path, top_k)

    # --- Previous Results ---
    st.subheader("Previous Results")
    _show_previous_results()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EVAL_RESULTS_DIR = Path("logs/eval_results")


def _run_evaluation(test_set_path: str, top_k: int) -> None:
    """Execute the evaluation pipeline and display results."""
    with st.spinner("Running evaluation..."):
        try:
            from core.settings import load_settings
            from core.query_engine.hybrid_search import HybridSearch
            from observability.evaluation.composite_evaluator import CompositeEvaluator

            settings = load_settings()
            hybrid = HybridSearch(settings)
            evaluator = CompositeEvaluator.from_settings(settings)
            runner = EvalRunner(settings=settings, hybrid_search=hybrid, evaluator=evaluator)
            report = runner.run(test_set_path, top_k=top_k)
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
            return

    # --- Display results ---
    st.success(
        f"Completed: {report.successful_cases}/{report.total_cases} cases "
        f"({report.failed_cases} failed)"
    )

    # Aggregate metrics
    if report.aggregate_metrics:
        st.subheader("Aggregate Metrics")

        import pandas as pd

        df = pd.DataFrame(
            [{"Metric": k, "Score": v} for k, v in report.aggregate_metrics.items()]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Per-case results
    if report.case_results:
        st.subheader("Per-Case Results")
        for cr in report.case_results:
            status = "OK" if cr.error is None else f"FAIL: {cr.error}"
            icon = "green" if cr.error is None else "red"
            with st.expander(f":{icon}[{status}] — {cr.query}"):
                if cr.retrieved_ids:
                    st.markdown(f"**Retrieved IDs:** `{', '.join(cr.retrieved_ids[:10])}`")
                if cr.golden_ids:
                    st.markdown(f"**Golden IDs:** `{', '.join(cr.golden_ids[:10])}`")
                if cr.metrics:
                    for k, v in cr.metrics.items():
                        st.markdown(f"**{k}:** {v:.4f}")

    # Save results
    _save_result(report)


def _save_result(report: Any) -> None:
    """Persist evaluation report to JSONL for historical comparison."""
    _EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_dict = {
        "total_cases": report.total_cases,
        "successful_cases": report.successful_cases,
        "failed_cases": report.failed_cases,
        "aggregate_metrics": report.aggregate_metrics,
        "case_results": [
            {
                "query": cr.query,
                "metrics": cr.metrics,
                "error": cr.error,
            }
            for cr in report.case_results
        ],
    }
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = _EVAL_RESULTS_DIR / f"eval_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    st.caption(f"Results saved to `{out_file}`")


def _show_previous_results() -> None:
    """Display list of past evaluation results."""
    if not _EVAL_RESULTS_DIR.exists():
        st.caption("No previous evaluation results found.")
        return

    result_files = sorted(_EVAL_RESULTS_DIR.glob("eval_*.json"), reverse=True)
    if not result_files:
        st.caption("No previous evaluation results found.")
        return

    for rf in result_files[:10]:  # Show last 10
        try:
            with open(rf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        agg = data.get("aggregate_metrics", {})
        hit = agg.get("hit_rate", "N/A")
        ts = rf.stem.replace("eval_", "").replace("_", " ")
        label = f"{ts} — {data.get('successful_cases', '?')}/{data.get('total_cases', '?')} cases — hit_rate={hit}"

        with st.expander(label):
            if agg:
                import pandas as pd

                df = pd.DataFrame(
                    [{"Metric": k, "Score": v} for k, v in agg.items()]
                )
                st.dataframe(df, use_container_width=True, hide_index=True)


render()
