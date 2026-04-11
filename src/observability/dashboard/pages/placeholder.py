"""Placeholder page for unimplemented dashboard sections."""

from __future__ import annotations

import streamlit as st


def render_placeholder(title: str, description: str = "") -> None:
    """Render a placeholder page."""
    st.header(title)
    st.info(f"This page is not yet implemented.{(' ' + description) if description else ''}")
