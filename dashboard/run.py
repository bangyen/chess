#!/usr/bin/env python3
"""
Template Dashboard Runner

Simple script to run the template dashboard with proper configuration.
Supports both development and production modes.
"""

import sys
import uvicorn
from pathlib import Path

if __name__ == "__main__":
    # Add project root to Python path if not already present
    project_root = Path(__file__).parent.parent.resolve()
    project_root_str = str(project_root)

    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    # Change to dashboard directory for proper static file serving
    dashboard_dir = Path(__file__).parent

    try:
        # Check if we're in development mode
        is_development = "--dev" in sys.argv or "development" in sys.argv

        if is_development:
            print("Starting template dashboard in development mode...")
            uvicorn.run(
                "dashboard.main:app",
                host="0.0.0.0",
                port=5050,
                reload=True,
                reload_dirs=[str(dashboard_dir)],
                log_level="info",
                server_header=False,
                forwarded_allow_ips="*",
            )
        else:
            print("Starting template dashboard in production mode...")
            uvicorn.run(
                "dashboard.main:app",
                host="0.0.0.0",
                port=5050,
                reload=False,
                log_level="info",
                server_header=False,
                forwarded_allow_ips="*",
            )
    except Exception as e:
        print(f"Failed to start dashboard server: {e}", file=sys.stderr)
        sys.exit(1)
