#!/usr/bin/env python
"""Start the Streamlit Dashboard for RAG system management.

Usage: python scripts/start_dashboard.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DASHBOARD_PATH = Path(__file__).resolve().parent.parent / "src" / "observability" / "dashboard" / "app.py"


def main() -> None:
    if not DASHBOARD_PATH.exists():
        print(f"Error: Dashboard entry point not found: {DASHBOARD_PATH}", file=sys.stderr)
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, "-m", "streamlit", "run", str(DASHBOARD_PATH), "--server.port=8501"]
    print(f"Starting Dashboard: {' '.join(cmd)}")
    print(f"Working directory: {project_root}")
    subprocess.run(cmd, cwd=str(project_root))


if __name__ == "__main__":
    main()
