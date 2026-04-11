"""Ingestion Traces page — view ingestion history and per-trace stage waterfall."""

from __future__ import annotations

from typing import Any

import streamlit as st

from observability.dashboard.services.trace_service import TraceService


def render() -> None:
    """Render the Ingestion Traces page."""
    st.header("Ingestion Traces")

    trace_service = TraceService()
    traces = trace_service.list_traces(trace_type="ingestion")

    if not traces:
        st.info(
            "No ingestion traces found. Run `python scripts/ingest.py` to generate traces."
        )
        return

    st.metric("Total Ingestions", len(traces))

    # --- Trace list ---
    for trace in traces:
        trace_id = trace.get("trace_id", "?")
        total_ms = trace.get("total_elapsed_ms", 0)
        stages = trace.get("stages", {})
        stage_names = list(stages.keys())

        # Find source info from the first stage that has it
        source = _extract_source(trace)

        label = f"`{trace_id}` — {source} — {total_ms:.0f}ms ({len(stage_names)} stages)"

        with st.expander(label):
            _render_trace_detail(trace_service, trace)


def _extract_source(trace: dict[str, Any]) -> str:
    """Try to extract source file name from trace metadata."""
    for stage_data in trace.get("stages", {}).values():
        if isinstance(stage_data, dict):
            # not stored directly, use what we have
            break
    return trace.get("source_path", "unknown")


def _render_trace_detail(
    trace_service: TraceService,
    trace: dict[str, Any],
) -> None:
    """Show stage waterfall chart and details for a single trace."""
    stages_info = trace_service.get_stage_elapsed(trace)

    if not stages_info:
        st.caption("No stage data recorded.")
        return

    # --- Waterfall chart (horizontal bar) ---
    st.subheader("Stage Elapsed Time")

    import pandas as pd

    df = pd.DataFrame(stages_info)
    df = df[["stage", "elapsed_ms"]].rename(columns={"stage": "Stage", "elapsed_ms": "Time (ms)"})
    df = df.sort_values("Time (ms)", ascending=True)

    st.bar_chart(df.set_index("Stage"))

    # --- Stage details table ---
    st.subheader("Stage Details")

    for info in stages_info:
        stage_name = info.pop("stage", "?")
        elapsed = info.pop("elapsed_ms", 0)
        with st.expander(f"**{stage_name}** — {elapsed:.1f}ms"):
            if info:
                for k, v in info.items():
                    st.markdown(f"**{k}:** `{v}`")
            else:
                st.caption("No additional details.")
