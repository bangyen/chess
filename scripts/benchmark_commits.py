import argparse
import re
import subprocess
import sys
from typing import Dict


def run_command(
    command: str, check: bool = True, capture_output: bool = True
) -> subprocess.CompletedProcess:
    """Run a shell command."""
    result = subprocess.run(
        command, shell=True, check=check, capture_output=capture_output, text=True
    )
    return result


def get_current_commit() -> str:
    """Get the current commit hash."""
    return str(run_command("git rev-parse HEAD").stdout).strip()


def check_clean_state():
    """Ensure the git working directory is clean."""
    status = run_command("git status --porcelain").stdout.strip()
    if status:
        print("Error: Working directory is not clean. Please commit or stash changes.")
        sys.exit(1)


def run_audit(commit: str, positions: int, seed: int, engine: str) -> str:
    """Run the audit tool on a specific commit."""
    print(f"Running audit on commit {commit}...")
    cmd = (
        f"uv run python -m chess_ai.cli.audit "
        f"--engine {engine} "
        f"--baseline_features "
        f"--positions {positions} "
        f"--seed {seed}"
    )
    # We set check=False so we can handle errors gracefully
    result = run_command(cmd, check=False)
    if result.returncode != 0:
        print(f"Error running audit on {commit}:")
        print(result.stderr)
        print(result.stdout)
        sys.exit(1)
    return str(result.stdout)


def parse_metrics(output: str) -> Dict[str, float]:
    """Parse metrics from the audit tool output."""
    metrics = {}

    # Fidelity (Delta-R^2):          0.124
    m = re.search(r"Fidelity \(Delta-R\^2\):\s+([0-9.]+)", output)
    if m:
        metrics["r2"] = float(m.group(1))

    # Move ranking (Kendall tau):    0.387
    m = re.search(r"Move ranking \(Kendall tau\):\s+([0-9.]+)", output)
    if m:
        metrics["tau"] = float(m.group(1))

    # Local faithfulness (best vs 2): 87.5%
    m = re.search(r"Local faithfulness \(best vs 2\):\s+([0-9.]+)%", output)
    if m:
        metrics["faithfulness"] = float(m.group(1))

    # Sparsity (reasons...): 1.25
    m = re.search(r"Sparsity .*:\s+([0-9.]+)", output)
    if m:
        metrics["sparsity"] = float(m.group(1))

    # Coverage (>=2 strong reasons):  93.8%
    m = re.search(r"Coverage .*:\s+([0-9.]+)%", output)
    if m:
        metrics["coverage"] = float(m.group(1))

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark chess-ai-audit across two commits"
    )
    parser.add_argument("old_commit", help="Baseline commit hash/ref")
    parser.add_argument("new_commit", help="Candidate commit hash/ref")
    parser.add_argument(
        "--positions", type=int, default=400, help="Number of positions to test"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--engine",
        default=run_command("which stockfish").stdout.strip()
        or "/opt/homebrew/bin/stockfish",
        help="Path to Stockfish engine",
    )

    args = parser.parse_args()

    check_clean_state()

    original_branch = run_command("git branch --show-current").stdout.strip()
    original_commit = get_current_commit()

    print(f"Starting benchmark: {args.old_commit} vs {args.new_commit}")
    print(f"Positions: {args.positions}, Seed: {args.seed}")

    try:
        # Run on new commit first
        print(f"Switching to new commit: {args.new_commit}")
        run_command(f"git checkout {args.new_commit}")
        new_output = run_audit(args.new_commit, args.positions, args.seed, args.engine)
        new_metrics = parse_metrics(new_output)

        # Run on old commit
        print(f"Switching to old commit: {args.old_commit}")
        run_command(f"git checkout {args.old_commit}")
        old_output = run_audit(args.old_commit, args.positions, args.seed, args.engine)
        old_metrics = parse_metrics(old_output)

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

    # Print comparison
    print("\n=== Benchmark Comparison ===")
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
