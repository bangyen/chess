#!/usr/bin/env python3
"""
Main CLI dispatcher for Chess AI tools.

This provides a unified entry point for all chess AI functionality.
"""

import sys
from typing import List, Optional

from .audit import main as audit_main
from .explainable import main as explainable_main


def main(args: Optional[List[str]] = None) -> None:
    """Main CLI entry point that dispatches to subcommands."""
    if args is None:
        args = sys.argv[1:]

    if not args or args[0] in ["help", "-h", "--help"]:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = Console()

        # Header
        console.print(
            Panel(
                Text(
                    "Chess AI Tools - Feature analysis and explainable gameplay",
                    style="bold cyan",
                ),
                expand=False,
                border_style="cyan",
            )
        )

        # Commands Table
        commands_table = Table(box=None, padding=(0, 2))
        commands_table.add_column("Command", style="bold magenta")
        commands_table.add_column("Description", style="white")

        commands_table.add_row(
            "audit", "Run feature explainability audit against Stockfish"
        )
        commands_table.add_row("play", "Play interactive chess with AI explanations")
        commands_table.add_row("help", "Show this help message")

        console.print("\n[bold]Available commands:[/bold]")
        console.print(commands_table)

        # Examples
        examples_text = Text.assemble(
            ("chess-ai audit --baseline_features --positions 100", "green"),
            "\n",
            ("chess-ai play --strength intermediate", "green"),
            "\n",
            ("chess-ai play --depth 20", "green"),
        )
        console.print(Panel(examples_text, title="Examples", border_style="dim"))
        console.print("")
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
