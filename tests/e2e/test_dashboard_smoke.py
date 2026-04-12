"""E2E: Dashboard smoke tests using Streamlit AppTest.

Verifies all 6 Dashboard pages can render without Python exceptions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Streamlit AppTest
from streamlit.testing.v1 import AppTest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_APP_SCRIPT = _PROJECT_ROOT / "src" / "observability" / "dashboard" / "app.py"


@pytest.fixture(scope="module")
def app() -> AppTest:
    """Create a shared AppTest instance for the dashboard."""
    assert _APP_SCRIPT.exists(), f"Dashboard app not found: {_APP_SCRIPT}"
    at = AppTest.from_file(str(_APP_SCRIPT))
    return at


class TestDashboardSmoke:
    """Verify all 6 dashboard pages render without errors."""

    def test_app_loads(self, app: AppTest) -> None:
        """Dashboard loads without import or startup errors."""
        app.run(timeout=30)
        # No exceptions in the main block
        assert not app.exception, f"Dashboard failed to load: {app.exception}"

    def test_system_overview_page(self, app: AppTest) -> None:
        """System Overview page renders."""
        app.run(timeout=30)
        assert not app.exception

    def test_data_browser_page(self, app: AppTest) -> None:
        """Data Browser page renders after navigation."""
        app.run(timeout=30)
        # Select Data Browser via sidebar radio
        if app.sidebar.radio:
            app.sidebar.radio[0].set_value("Data Browser")
            app.run(timeout=30)
        assert not app.exception

    def test_ingestion_manager_page(self, app: AppTest) -> None:
        """Ingestion Manager page renders."""
        app.run(timeout=30)
        if app.sidebar.radio:
            app.sidebar.radio[0].set_value("Ingestion Manager")
            app.run(timeout=30)
        assert not app.exception

    def test_ingestion_traces_page(self, app: AppTest) -> None:
        """Ingestion Traces page renders."""
        app.run(timeout=30)
        if app.sidebar.radio:
            app.sidebar.radio[0].set_value("Ingestion Traces")
            app.run(timeout=30)
        assert not app.exception

    def test_query_traces_page(self, app: AppTest) -> None:
        """Query Traces page renders."""
        app.run(timeout=30)
        if app.sidebar.radio:
            app.sidebar.radio[0].set_value("Query Traces")
            app.run(timeout=30)
        assert not app.exception

    def test_evaluation_panel_page(self, app: AppTest) -> None:
        """Evaluation Panel page renders."""
        app.run(timeout=30)
        if app.sidebar.radio:
            app.sidebar.radio[0].set_value("Evaluation Panel")
            app.run(timeout=30)
        assert not app.exception
