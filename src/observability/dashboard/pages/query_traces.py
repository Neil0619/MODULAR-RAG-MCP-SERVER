"""Query Traces page — query history, Dense vs Sparse comparison, Rerank diff."""

from __future__ import annotations

from typing import Any

import streamlit as st

from observability.dashboard.services.trace_service import TraceService


def render() -> None:
    """Render the Query Traces page."""
    st.header("Query Traces")

    trace_service = TraceService()
    traces = trace_service.list_traces(trace_type="query")

    if not traces:
        st.info(
            "No query traces found. Run `python scripts/query.py` or use "
            "the MCP tool `query_knowledge_hub` to generate traces."
        )
        return

    # --- Search filter ---
    search = st.text_input("Search queries", placeholder="Filter by query text…")
    if search:
        search_lower = search.lower()
        traces = [
            t for t in traces
            if search_lower in _get_query_text(t).lower()
        ]

    st.metric("Query Traces", len(traces))

    # --- Trace list ---
    for trace in traces:
        trace_id = trace.get("trace_id", "?")
        query_text = _get_query_text(trace)
        total_ms = trace.get("total_elapsed_ms", 0)
        stage_count = len(trace.get("stages", {}))

        preview = query_text[:60] + ("…" if len(query_text) > 60 else "")
        label = f"`{trace_id}` — \"{preview}\" — {total_ms:.0f}ms ({stage_count} stages)"

        with st.expander(label):
            _render_trace_detail(trace_service, trace)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_query_text(trace: dict[str, Any]) -> str:
    """Extract the original query text from the trace."""
    qp = trace.get("stages", {}).get("query_processing", {})
    if isinstance(qp, dict):
        return qp.get("original_query", "")
    return ""


def _render_trace_detail(
    trace_service: TraceService,
    trace: dict[str, Any],
) -> None:
    """Show waterfall chart, Dense vs Sparse comparison, Rerank diff."""
    stages_info = trace_service.get_stage_elapsed(trace)

    if not stages_info:
        st.caption("No stage data recorded.")
        return

    # --- Waterfall chart ---
    st.subheader("Stage Elapsed Time")

    import pandas as pd

    df = pd.DataFrame(stages_info)
    df_chart = df[["stage", "elapsed_ms"]].rename(
        columns={"stage": "Stage", "elapsed_ms": "Time (ms)"}
    )
    df_chart = df_chart.sort_values("Time (ms)", ascending=True)
    st.bar_chart(df_chart.set_index("Stage"))

    # --- Dense vs Sparse comparison ---
    stages = trace.get("stages", {})
    dense = stages.get("dense_retrieval", {})
    sparse = stages.get("sparse_retrieval", {})

    if isinstance(dense, dict) and isinstance(sparse, dict):
        st.subheader("Dense vs Sparse Retrieval")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Dense (Embedding)**")
            st.metric("Hits", dense.get("hit_count", 0))
            st.caption(f"Method: {dense.get('method', 'N/A')}")
        with col2:
            st.markdown("**Sparse (BM25)**")
            st.metric("Hits", sparse.get("hit_count", 0))
            st.caption(f"Method: {sparse.get('method', 'N/A')}")

    # --- Fusion summary ---
    fusion = stages.get("fusion", {})
    if isinstance(fusion, dict):
        st.subheader("Fusion")
        st.markdown(
            f"Algorithm: `{fusion.get('algorithm', 'N/A')}` | "
            f"Results: **{fusion.get('result_count', 0)}**"
        )

    # --- Rerank diff ---
    rerank = stages.get("rerank", {})
    if isinstance(rerank, dict) and rerank:
        st.subheader("Rerank")
        fallback = rerank.get("fallback", False)
        if fallback:
            st.warning("Reranker fell back to fusion ranking.")
        else:
            st.success(
                f"Method: `{rerank.get('method', 'N/A')}` | "
                f"Input: {rerank.get('input_count', '?')} → "
                f"Output: {rerank.get('output_count', '?')}"
            )

    # --- Stage details ---
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
