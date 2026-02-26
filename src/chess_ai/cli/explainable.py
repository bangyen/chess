#!/usr/bin/env python3
"""
CLI for the Explainable Chess Engine

Usage: python -m src.chess_feature_audit.cli_explainable
"""

import argparse
import sys

from ..explainable_engine import ExplainableChessEngine
from ..utils.engine_discovery import find_stockfish


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Explainable Chess Engine - Learn chess with AI explanations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.chess_ai.cli.explainable
  python -m src.chess_ai.cli.explainable --engine /path/to/stockfish
  python -m src.chess_ai.cli.explainable --depth 20 --strength intermediate
        """,
    )

    parser.add_argument(
        "--engine",
        help="Path to Stockfish engine (auto-detected if not specified)",
    )

    parser.add_argument(
        "--depth", type=int, default=16, help="Search depth for analysis (default: 16)"
    )

    parser.add_argument(
        "--strength",
        choices=["beginner", "novice", "intermediate", "advanced", "expert"],
        default="beginner",
        help="Stockfish opponent strength (default: beginner)",
    )

    args = parser.parse_args()

    # Find Stockfish path
    if args.engine:
        stockfish_path = args.engine
    else:
        try:
            stockfish_path = find_stockfish()
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    try:
        with ExplainableChessEngine(
            stockfish_path, args.depth, args.strength
        ) as engine:
            engine.play_interactive_game()
    except KeyboardInterrupt:
        print("\n👋 Thanks for playing!")
    except FileNotFoundError:
        print(f"❌ Error: Could not find Stockfish at {stockfish_path}")
        print("Please install Stockfish or specify the correct path with --engine")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
