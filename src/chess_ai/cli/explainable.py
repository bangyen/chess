#!/usr/bin/env python3
"""
CLI for the Explainable Chess Engine

Usage: python -m src.chess_feature_audit.cli_explainable
"""

import argparse
import sys

from ..explainable_engine import ExplainableChessEngine


def main():
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
        default="/opt/homebrew/bin/stockfish",
        help="Path to Stockfish engine (default: /opt/homebrew/bin/stockfish)",
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

    print("🎯 Explainable Chess Engine")
    print("=" * 50)
    print(
        "This engine will analyze your moves and explain what you should have done instead."
    )
    print("Make moves in standard algebraic notation (e4, Nf3, O-O, etc.)")
    print("Type 'help' for commands, 'quit' to exit")
    print("=" * 50)

    try:
        with ExplainableChessEngine(args.engine, args.depth, args.strength) as engine:
            engine.play_interactive_game()
    except KeyboardInterrupt:
        print("\n👋 Thanks for playing!")
    except FileNotFoundError:
        print(f"❌ Error: Could not find Stockfish at {args.engine}")
        print("Please install Stockfish or specify the correct path with --engine")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
