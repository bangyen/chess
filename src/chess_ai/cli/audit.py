"""Command-line interface for the chess feature audit tool."""

import argparse
import os
import random
import sys

import numpy as np

from ..audit import audit_feature_set
from ..engine import SFConfig, sf_open
from ..features import baseline_extract_features, load_feature_module
from ..utils import sample_positions_from_pgn, sample_random_positions


def main():
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
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.engine:
        print(
            "Please provide --engine PATH or set STOCKFISH_PATH env var.",
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

    # Positions
    if args.pgn:
        boards = sample_positions_from_pgn(
            args.pgn, args.positions, ply_skip=args.ply_skip
        )
        if len(boards) < args.positions:
            # supplement with randoms
            boards += sample_random_positions(args.positions - len(boards))
    else:
        boards = sample_random_positions(args.positions)
    if not boards:
        print("No positions sampled.", file=sys.stderr)
        sys.exit(1)

    # Feature extractor
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
    print("\n=== Explainability Audit Report ===")
    print(
        f"Positions: {len(boards)}  |  Depth: {cfg.depth or 'movetime'}  |  MultiPV: {cfg.multipv}"
    )
    print(f"Fidelity (Delta-R^2):          {res.r2:0.3f}")
    print(
        f"Move ranking (Kendall tau):    {res.tau_mean:0.3f}  (positions covered: {res.tau_covered}/{res.n_tau})"
    )
    print(
        f"Local faithfulness (best vs 2): {res.local_faithfulness*100:0.1f}% (gap ≥ {args.gap} cp)"
    )
    print(
        f"Local faithfulness (decisive): {res.local_faithfulness_decisive*100:0.1f}% (gap ≥ 80.0 cp)"
    )
    print(
        f"Sparsity (reasons to cover 80% contribution for best move): {res.sparsity_mean:0.2f}"
    )
    print(f"Coverage (≥2 strong reasons):  {res.coverage_ratio*100:0.1f}%")
    print("\nTop features by |coef|:")
    for name, coef in res.top_features_by_coef:
        print(f"  {name:30s}  coef={coef:.4f}")
    if res.stable_features:
        print(f"\nStable features (picked ≥{100 * 0.7:.0f}% of bootstraps):")
        for name in res.stable_features:
            print(f"  - {name}")
    else:
        print("\nStable features: (none reached threshold)")

    print("\nGuidance:")
    print(
        " - Aim for Delta-R^2 ≥ 0.60 on mixed middlegames at depth ~16 as a healthy baseline."
    )
    print(" - Tau ≥ 0.45 for top-3 move ranking is decent; higher is better.")
    print(
        " - Local faithfulness ≥ 80% on decisive positions shows explanations track preferences."
    )
    print(" - Sparsity around 3–5 suggests crisp, narratable reasons.")
    print(
        " - Coverage ≥ 70% with ≥2 strong reasons means you can explain most positions.\n"
    )


if __name__ == "__main__":
    main()
