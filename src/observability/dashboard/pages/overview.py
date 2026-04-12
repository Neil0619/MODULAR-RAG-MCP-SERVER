"""Overview page — system configuration and data statistics."""

from __future__ import annotations

import streamlit as st

from observability.dashboard.services.config_service import ConfigService


def render() -> None:
    """Render the system overview page."""
    st.header("System Overview")

    # --- Component Configuration ---
    st.subheader("Pluggable Components")

    try:
        service = ConfigService()
        overview = service.get_overview()

        for comp in overview.components:
            with st.expander(f"**{comp.name}** — {comp.provider}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Provider:** `{comp.provider}`")
                    if comp.model:
                        st.markdown(f"**Model:** `{comp.model}`")
                with col2:
                    if comp.details:
                        for key, val in comp.details.items():
                            st.markdown(f"**{key}:** `{val}`")

    except Exception as exc:
        st.warning(f"Could not load settings: {exc}")

    # --- Data Statistics ---
    st.subheader("Data Statistics")

    try:
        from libs.vector_store.vector_store_factory import VectorStoreFactory

        service = ConfigService()
        store = VectorStoreFactory.create(service.settings)
        stats = store.get_collection_stats("default")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats.get("doc_count", 0))
        with col2:
            st.metric("Collection", "default")

    except Exception as exc:
        st.info(f"No data loaded yet. Run `ingest.py` to add documents. ({exc})")

    # --- Recent Activity ---
    st.subheader("Recent Activity")

    try:
        trace_file = st.secrets.get("traces_dir", "logs")
    except Exception:
        trace_file = "logs"
    import os
    trace_path = os.path.join(str(trace_file), "traces.jsonl")

    if os.path.exists(trace_path):
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-5:]  # last 5 traces
            if lines:
                import json
                for line in reversed(lines):
                    entry = json.loads(line)
                    msg = entry.get("message", "{}")
                    try:
                        trace_data = json.loads(msg)
                    except (json.JSONDecodeError, TypeError):
                        trace_data = {}
                    trace_type = trace_data.get("trace_type", "unknown")
                    trace_id = trace_data.get("trace_id", "?")
                    elapsed = trace_data.get("total_elapsed_ms", 0)
                    st.markdown(
                        f"- `{trace_id}` — **{trace_type}** — {elapsed:.1f}ms"
                    )
            else:
                st.info("No trace records found.")
        except Exception as exc:
            st.info(f"Could not read traces: {exc}")
    else:
        st.info("No trace file found. Run a query or ingestion to generate traces.")


render()
