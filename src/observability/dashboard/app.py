"""Dashboard app — Streamlit multi-page application for RAG system management.

Run with: streamlit run src/observability/dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

from observability.dashboard.pages.overview import render as render_overview
from observability.dashboard.pages.data_browser import render as render_data_browser
from observability.dashboard.pages.ingestion_manager import render as render_ingestion_manager
from observability.dashboard.pages.placeholder import render_placeholder


def main() -> None:
    """Main entry point for the Dashboard."""
    st.set_page_config(
        page_title="RAG Dashboard",
        page_icon="🔍",
        layout="wide",
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "System Overview",
            "Data Browser",
            "Ingestion Manager",
            "Ingestion Traces",
            "Query Traces",
            "Evaluation Panel",
        ],
    )

    if page == "System Overview":
        render_overview()
    elif page == "Data Browser":
        render_data_browser()
    elif page == "Ingestion Manager":
        render_ingestion_manager()
    elif page == "Ingestion Traces":
        render_placeholder("Ingestion Traces", "Will be implemented in G5.")
    elif page == "Query Traces":
        render_placeholder("Query Traces", "Will be implemented in G6.")
    elif page == "Evaluation Panel":
        render_placeholder("Evaluation Panel", "Will be implemented in H4.")


if __name__ == "__main__":
    main()
