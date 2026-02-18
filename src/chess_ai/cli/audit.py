"""Command-line interface for the chess feature audit tool."""

import argparse
import json
import os
import random
import sys

import numpy as np

from ..audit import audit_feature_set
from ..engine import SFConfig, sf_open
from ..features import baseline_extract_features, load_feature_module
from ..utils import (
    sample_positions_from_pgn,
    sample_random_positions,
    sample_stratified_positions,
)
from ..utils.sampling import DEFAULT_PHASE_WEIGHTS


def _parse_phase_weights(raw: str) -> dict:
    """Parse a JSON string into a phase-weights dict.

    Validates that every key is a recognised game phase and that
    values are positive numbers.

    Raises:
        argparse.ArgumentTypeError: On invalid input.
    """
    try:
        weights: dict[str, float] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"--phase-weights must be valid JSON: {exc}"
        ) from exc

    valid_phases = {"opening", "middlegame", "endgame"}
    unknown = set(weights.keys()) - valid_phases
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown phase(s) in --phase-weights: {unknown}. "
            f"Valid phases: {valid_phases}"
        )
    for k, v in weights.items():
        if not isinstance(v, (int, float)) or v <= 0:
            raise argparse.ArgumentTypeError(
                f"Phase weight for '{k}' must be a positive number, got {v!r}"
            )
    return weights


def main() -> None:
    """Main CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Audit explainability of a chess feature set against Stockfish."
    )
    ap.add_argument(
        "--engine",
        type=str,
        default=os.environ.get("STOCKFISH_PATH", ""),
        help="Path to Stockfish binary",
    )
    cpu_count = os.cpu_count() or 2
    ap.add_argument("--threads", type=int, default=max(1, cpu_count // 2))
    ap.add_argument(
        "--depth",
        type=int,
        default=12,
        help="Fixed search depth (set 0 to use movetime)",
    )
    ap.add_argument(
        "--movetime", type=int, default=0, help="Milliseconds per evaluation if depth=0"
    )
    ap.add_argument(
        "--multipv", type=int, default=3, help="Max MultiPV used for ranking metric"
    )
    ap.add_argument("--positions", type=int, default=100)
    ap.add_argument(
        "--pgn", type=str, default="", help="Optional PGN file to sample positions"
    )
    ap.add_argument(
        "--ply-skip",
        type=int,
        default=8,
        help="Keep every Nth ply when sampling from PGN",
    )
    ap.add_argument(
        "--features_module",
        type=str,
        default="",
        help="Path to a Python module that defines extract_features(board)",
    )
    ap.add_argument(
        "--baseline_features",
        action="store_true",
        help="Use built-in small baseline features",
    )
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="L1 regularization strength for surrogate",
    )
    ap.add_argument(
        "--gap",
        type=float,
        default=50.0,
        help="CP gap to treat best vs second as decisive for faithfulness",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable phase-stratified sampling and use uniform random positions",
    )
    ap.add_argument(
        "--phase-weights",
        type=_parse_phase_weights,
        default=None,
        help=(
            "JSON dict of phase weights for stratified sampling, e.g. "
            '\'{"opening": 0.25, "middlegame": 0.50, "endgame": 0.25}\'. '
            "Ignored when --no-stratify is set."
        ),
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.engine:
        print(
            "Please provide --engine PATH or set STOCKFISH_PATH env var.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Feature extractor — checked early so we fail fast before expensive
    # position sampling when neither flag is provided.
    if args.baseline_features:
        extract_fn = baseline_extract_features
    elif args.features_module:
        mod = load_feature_module(args.features_module)
        extract_fn = mod.extract_features  # type: ignore
    else:
        print(
            "Provide --features_module PATH or use --baseline_features to run.",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = SFConfig(
        engine_path=args.engine,
        depth=args.depth if args.depth > 0 else 0,
        movetime=args.movetime if args.depth == 0 else 0,
        multipv=max(2, args.multipv),
        threads=args.threads,
    )

    # Determine whether to use stratified sampling (on by default).
    use_stratify = not args.no_stratify
    phase_weights = args.phase_weights if use_stratify else None

    # Positions
    if args.pgn:
        boards = sample_positions_from_pgn(
            args.pgn,
            args.positions,
            ply_skip=args.ply_skip,
            phase_weights=phase_weights,
        )
        if len(boards) < args.positions:
            # Supplement shortfall with generated positions.
            shortfall = args.positions - len(boards)
            if use_stratify:
                boards += sample_stratified_positions(
                    shortfall,
                    phase_weights=phase_weights or DEFAULT_PHASE_WEIGHTS,
                )
            else:
                boards += sample_random_positions(shortfall)
    else:
        if use_stratify:
            boards = sample_stratified_positions(
                args.positions,
                phase_weights=phase_weights or DEFAULT_PHASE_WEIGHTS,
            )
        else:
            boards = sample_random_positions(args.positions)
    if not boards:
        print("No positions sampled.", file=sys.stderr)
        sys.exit(1)

    # Engine
    engine = sf_open(cfg)

    try:
        res = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            multipv_for_ranking=args.multipv,
            test_size=args.test_size,
            l1_alpha=args.alpha,
            gap_threshold_cp=args.gap,
        )
    finally:
        engine.quit()

    # Report
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    # Create Header Panel
    header_text = Text.assemble(
        ("Explainability Audit Report", "bold cyan"),
        "\n",
        (f"Positions: {len(boards)}", "white"),
        " | ",
        (f"Depth: {cfg.depth or 'movetime'}", "white"),
        " | ",
        (f"MultiPV: {cfg.multipv}", "white"),
    )
    console.print(Panel(header_text, border_style="cyan"))

    # Main Metrics Table
    metrics_table = Table(title="Validation Metrics", box=None, show_header=False)
    metrics_table.add_column("Metric", style="bold white")
    metrics_table.add_column("Value", style="bold green")
    metrics_table.add_column("Target", style="dim")

    metrics_table.add_row(
        "Fidelity (Delta-R²)",
        f"{res.r2:0.3f}",
        "≥ 0.35",
        style="green" if res.r2 >= 0.35 else "yellow",
    )
    metrics_table.add_row(
        "Move ranking (Kendall τ)",
        f"{res.tau_mean:0.3f} ({res.tau_covered}/{res.n_tau})",
        "≥ 0.45",
        style="green" if res.tau_mean >= 0.45 else "yellow",
    )
    metrics_table.add_row(
        "Local faithfulness (best vs 2)",
        f"{res.local_faithfulness*100:0.1f}%",
        "≥ 80%",
        style="green" if res.local_faithfulness >= 0.8 else "yellow",
    )
    metrics_table.add_row(
        "Local faithfulness (decisive)",
        f"{res.local_faithfulness_decisive*100:0.1f}%",
        "≥ 80%",
        style="green" if res.local_faithfulness_decisive >= 0.8 else "yellow",
    )
    metrics_table.add_row(
        "Sparsity (explanation reasons)",
        f"{res.sparsity_mean:0.2f}",
        "≤ 4.0",
        style="green" if res.sparsity_mean <= 4.0 else "yellow",
    )
    metrics_table.add_row(
        "Position Coverage",
        f"{res.coverage_ratio*100:0.1f}%",
        "≥ 70%",
        style="green" if res.coverage_ratio >= 0.7 else "yellow",
    )

    # Top Features Table
    features_table = Table(title="Top Driving Features", header_style="bold magenta")
    features_table.add_column("Feature", style="cyan")
    features_table.add_column("Impact (Coef)", justify="right")

    for name, coef in res.top_features_by_coef:
        if abs(coef) > 1e-8:
            color = "green" if coef > 0 else "red"
            features_table.add_row(name, Text(f"{coef:+.4f}", style=color))

    # Stable Features Table
    stable_table = Table(
        title="Stable Features (Bootstrap)", box=None, show_header=False
    )
    stable_table.add_column("Feature", style="green")

    if res.stable_features:
        for name in res.stable_features:
            stable_table.add_row(f"• {name}")
    else:
        stable_table.add_row("(none reached threshold)", style="dim")

    # Layout: side-by-side using a parent grid table for perfect alignment
    layout_table = Table.grid(padding=(0, 4))
    layout_table.add_column()
    layout_table.add_column()
    layout_table.add_row(features_table, stable_table)

    console.print(metrics_table)
    console.print(Panel(layout_table, border_style="magenta"))

    # Guidance Panel
    guidance_text = Text.assemble(
        ("• ", "bold green"),
        "Fidelity: How well features track engine eval changes.\n",
        ("• ", "bold green"),
        "Tau: Correlation between feature-predicted vs SF rankings.\n",
        ("• ", "bold green"),
        "Faithfulness: Consistency with engine's top-move preference.\n",
        ("• ", "bold green"),
        "Sparsity/Coverage: Explanation quality vs robustness.",
    )
    console.print(Panel(guidance_text, title="Guidance", border_style="dim"))
    console.print("")


if __name__ == "__main__":
    main()
