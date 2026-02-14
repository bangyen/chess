#!/usr/bin/env python3
"""
Main CLI dispatcher for Chess AI tools.

This provides a unified entry point for all chess AI functionality.
"""

import argparse
import sys
from typing import List, Optional

from .audit import main as audit_main
from .explainable import main as explainable_main


def main(args: Optional[List[str]] = None) -> None:
    """Main CLI entry point that dispatches to subcommands."""
    if args is None:
        args = sys.argv[1:]

    if not args or args[0] in ["help", "-h", "--help"]:
        parser = argparse.ArgumentParser(
            description="Chess AI Tools - Feature analysis and explainable gameplay",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Available commands:
  audit     Run feature explainability audit against Stockfish
  play      Play interactive chess with AI explanations
  help      Show this help message

Examples:
  chess-ai audit --baseline_features --positions 100
  chess-ai play --strength intermediate
  chess-ai play --depth 20
            """,
        )
        parser.print_help()
        return

    command = args[0]
    remaining_args = args[1:]

    # Dispatch to appropriate subcommand
    if command == "audit":
        # Set sys.argv for the audit command
        original_argv = sys.argv
        sys.argv = ["chess-ai-audit", *remaining_args]
        try:
            audit_main()
        finally:
            sys.argv = original_argv
    elif command == "play":
        # Set sys.argv for the explainable command
        original_argv = sys.argv
        sys.argv = ["chess-ai-play", *remaining_args]
        try:
            explainable_main()
        finally:
            sys.argv = original_argv
    else:
        print(f"Unknown command: {command}")
        print("Available commands: audit, play, help")
        sys.exit(1)


if __name__ == "__main__":
    main()
