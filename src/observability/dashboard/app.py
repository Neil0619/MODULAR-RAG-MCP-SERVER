"""Dashboard app — Streamlit multi-page application for RAG system management.

Run with: streamlit run src/observability/dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# Initialise the root "rag" logger so that all rag.* child loggers
# (rag.embedding.doubao, rag.dashboard.ingestion, etc.) inherit
# the stderr + file handlers automatically.
from observability.logger import get_logger

get_logger("rag")

# Pages are relative to this script's directory
_PAGES_DIR = Path(__file__).parent / "pages"


def main() -> None:
    """Main entry point for the Dashboard."""
    pages = {
        "": [
            st.Page(
                str(_PAGES_DIR / "overview.py"),
                title="System Overview",
                icon=":material/dashboard:",
                url_path="overview",
            ),
            st.Page(
                str(_PAGES_DIR / "data_browser.py"),
                title="Data Browser",
                icon=":material/storage:",
                url_path="data_browser",
            ),
            st.Page(
                str(_PAGES_DIR / "ingestion_manager.py"),
                title="Ingestion Manager",
                icon=":material/upload:",
                url_path="ingestion_manager",
            ),
            st.Page(
                str(_PAGES_DIR / "ingestion_traces.py"),
                title="Ingestion Traces",
                icon=":material/timeline:",
                url_path="ingestion_traces",
            ),
            st.Page(
                str(_PAGES_DIR / "query_traces.py"),
                title="Query Traces",
                icon=":material/search:",
                url_path="query_traces",
            ),
            st.Page(
                str(_PAGES_DIR / "evaluation_panel.py"),
                title="Evaluation Panel",
                icon=":material/assessment:",
                url_path="evaluation_panel",
            ),
        ],
    }
    pg = st.navigation(pages)
    st.set_page_config(
        page_title="RAG Dashboard",
        page_icon=":material/search:",
        layout="wide",
    )
    pg.run()


if __name__ == "__main__":
    main()
