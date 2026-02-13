"""Compare explainability-audit metrics across two git commits.

Automates: checkout → build Rust extension → run audit → parse metrics → restore.
The dominant cost is Stockfish analysis, so the defaults (100 positions, depth 12,
multi-threaded) are tuned for fast iteration (~3 min).  Override with
``--positions``, ``--depth``, or ``--threads`` when you need higher fidelity.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from typing import Dict


def run_command(
    command: str, check: bool = True, capture_output: bool = True
) -> subprocess.CompletedProcess:
    """Run a shell command and return the completed process."""
    result = subprocess.run(
        command, shell=True, check=check, capture_output=capture_output, text=True
    )
    return result


def get_current_commit() -> str:
    """Get the current commit hash."""
    return str(run_command("git rev-parse HEAD").stdout).strip()


def check_clean_state():
    """Ensure the git working directory is clean before switching commits."""
    status = run_command("git status --porcelain").stdout.strip()
    if status:
        print("Error: Working directory is not clean. Please commit or stash changes.")
        sys.exit(1)


def run_audit(
    commit: str,
    positions: int,
    seed: int,
    engine: str,
    depth: int,
    threads: int,
) -> str:
    """Run the audit tool on a specific commit.

    Streams stderr (tqdm progress bars) to the terminal in real time
    while capturing stdout for metric parsing.
    """
    print(
        f"  Running audit on {commit[:8]}  "
        f"(positions={positions}, depth={depth}, threads={threads})"
    )
    cmd = (
        f"uv run python -m chess_ai.cli.audit "
        f"--engine {engine} "
        f"--baseline_features "
        f"--positions {positions} "
        f"--seed {seed} "
        f"--depth {depth} "
        f"--threads {threads}"
    )
    # Capture stdout for parsing; let stderr (tqdm) flow to terminal
    result = subprocess.run(
        cmd,
        shell=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=None,  # inherit — tqdm progress is visible live
        text=True,
    )
    if result.returncode != 0:
        print(f"Error running audit on {commit}:")
        print(result.stdout)
        sys.exit(1)
    return str(result.stdout)


def parse_metrics(output: str) -> Dict[str, float]:
    """Parse metrics from the audit tool output."""
    metrics: Dict[str, float] = {}

    m = re.search(r"Fidelity \(Delta-R\^2\):\s+([0-9.]+)", output)
    if m:
        metrics["r2"] = float(m.group(1))

    m = re.search(r"Move ranking \(Kendall tau\):\s+([0-9.]+)", output)
    if m:
        metrics["tau"] = float(m.group(1))

    m = re.search(r"Local faithfulness \(best vs 2\):\s+([0-9.]+)%", output)
    if m:
        metrics["faithfulness"] = float(m.group(1))

    m = re.search(r"Sparsity .*:\s+([0-9.]+)", output)
    if m:
        metrics["sparsity"] = float(m.group(1))

    m = re.search(r"Coverage .*:\s+([0-9.]+)%", output)
    if m:
        metrics["coverage"] = float(m.group(1))

    return metrics


def fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as a human-readable string."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark chess-ai-audit across two commits"
    )
    parser.add_argument("old_commit", help="Baseline commit hash/ref")
    parser.add_argument("new_commit", help="Candidate commit hash/ref")

    cpu_count = os.cpu_count() or 2

    parser.add_argument(
        "--positions",
        type=int,
        default=100,
        help="Number of positions to test (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=12,
        help="Stockfish search depth (default: 12)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, cpu_count // 2),
        help=f"Stockfish threads (default: {max(1, cpu_count // 2)})",
    )
    parser.add_argument(
        "--engine",
        default=run_command("which stockfish").stdout.strip()
        or "/opt/homebrew/bin/stockfish",
        help="Path to Stockfish engine",
    )

    args = parser.parse_args()

    positions = args.positions
    depth = args.depth
    threads = args.threads

    check_clean_state()

    original_branch = run_command("git branch --show-current").stdout.strip()
    original_commit = get_current_commit()

    print(f"Starting benchmark: {args.old_commit} vs {args.new_commit}")
    print(
        f"  positions={positions}  depth={depth}  threads={threads}  seed={args.seed}"
    )
    print()

    wall_start = time.monotonic()

    try:
        # ---- New commit ----------------------------------------------------
        print(f"[1/2] Switching to new commit: {args.new_commit}")
        run_command(f"git checkout {args.new_commit}")
        print("  Building Rust extension...")
        run_command("uv run maturin develop --release", check=False)
        t0 = time.monotonic()
        new_output = run_audit(
            args.new_commit,
            positions,
            args.seed,
            args.engine,
            depth,
            threads,
        )
        new_elapsed = time.monotonic() - t0
        new_metrics = parse_metrics(new_output)
        print(f"  Done in {fmt_elapsed(new_elapsed)}\n")

        # ---- Old commit ----------------------------------------------------
        print(f"[2/2] Switching to old commit: {args.old_commit}")
        run_command(f"git checkout {args.old_commit}")
        print("  Building Rust extension...")
        run_command("uv run maturin develop --release", check=False)
        t0 = time.monotonic()
        old_output = run_audit(
            args.old_commit,
            positions,
            args.seed,
            args.engine,
            depth,
            threads,
        )
        old_elapsed = time.monotonic() - t0
        old_metrics = parse_metrics(old_output)
        print(f"  Done in {fmt_elapsed(old_elapsed)}\n")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

    finally:
        # Restore state
        print(f"Restoring state to {original_branch or original_commit}...")
        if original_branch:
            run_command(f"git checkout {original_branch}")
        else:
            run_command(f"git checkout {original_commit}")

    # Print comparison -------------------------------------------------------
    wall_total = time.monotonic() - wall_start
    print(f"\n=== Benchmark Comparison  (total {fmt_elapsed(wall_total)}) ===")
    print(
        f"  positions={positions}  depth={depth}  threads={threads}  seed={args.seed}"
    )
    print(f"{'Metric':<30} | {'Old':<10} | {'New':<10} | {'Delta':<10}")
    print("-" * 70)

    keys = [
        ("r2", "Fidelity (R^2)"),
        ("tau", "Kendall Tau"),
        ("faithfulness", "Faithfulness (%)"),
        ("sparsity", "Sparsity"),
        ("coverage", "Coverage (%)"),
    ]

    for key, label in keys:
        old_val = old_metrics.get(key, 0.0)
        new_val = new_metrics.get(key, 0.0)
        delta = new_val - old_val
        print(f"{label:<30} | {old_val:<10.3f} | {new_val:<10.3f} | {delta:<+10.3f}")


if __name__ == "__main__":
    main()
