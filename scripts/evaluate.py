"""
Post-training evaluation for PLCM continual learning experiments.

The trainer writes a results JSON per model (``runs/<model>_results.json``)
containing the full accuracy matrix R[i,j] and the summary metrics. This
script loads one or more of those files, re-renders the per-model metrics
table, and prints a side-by-side comparison — the analysis counterpart to
``scripts/train.py``.

Usage:
    # Evaluate a single run
    python scripts/evaluate.py --results runs/plcm_results.json

    # Compare several runs
    python scripts/evaluate.py --results runs/plcm_results.json runs/lstm_results.json

    # Auto-discover every *_results.json in a directory
    python scripts/evaluate.py --runs-dir runs/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.metrics import ContinualMetrics


def load_results(path: str) -> dict:
    """Load a results JSON written by ContinualTrainer.save_results.

    Args:
        path: Path to a *_results.json file

    Returns:
        Parsed results dict
    """
    with open(path) as f:
        # allow_nan=True (the default) parses the NaN tokens the trainer emits
        # for not-yet-evaluated (i, j) cells of the accuracy matrix.
        return json.load(f)


def discover_results(runs_dir: str) -> list[str]:
    """Find all *_results.json files in a directory.

    Args:
        runs_dir: Directory to scan

    Returns:
        Sorted list of results file paths
    """
    return sorted(str(p) for p in Path(runs_dir).glob("*_results.json"))


def metrics_from_matrix(matrix: list[list[float]]) -> ContinualMetrics:
    """Rebuild a ContinualMetrics from a saved accuracy matrix.

    The metrics are recomputed from R[i,j] rather than trusting the stored
    scalars, so the printed table and summary always agree with the matrix.

    Args:
        matrix: Saved accuracy matrix (list of lists, possibly with NaN)

    Returns:
        A ContinualMetrics populated with the matrix
    """
    arr = np.array(matrix, dtype=float)  # [T, T]
    num_tasks = arr.shape[0]
    metrics = ContinualMetrics(num_tasks)
    metrics.accuracy_matrix = arr
    return metrics


def summarize_run(path: str, results: dict) -> dict:
    """Print a single run's metrics table and return its summary scalars.

    Args:
        path: Source file path (for display)
        results: Parsed results dict

    Returns:
        Dict of the four headline metrics plus the model label
    """
    model_type = results.get("model_type", Path(path).stem.replace("_results", ""))

    print(f"\n{'#' * 60}")
    print(f"# {model_type}   ({path})")
    print(f"{'#' * 60}")

    matrix = results.get("accuracy_matrix")
    if matrix is not None:
        metrics = metrics_from_matrix(matrix)
        metrics.print_summary()
        summary = metrics.summary()
    else:
        # No matrix stored (e.g. a combined comparison file) — fall back to
        # the pre-computed scalars.
        print("  (no accuracy matrix stored — showing saved scalars)")
        summary = {
            k: results.get(k, float("nan"))
            for k in (
                "average_accuracy",
                "forgetting",
                "backward_transfer",
                "forward_transfer",
            )
        }
        for k, v in summary.items():
            print(f"  {k:<20} {v:.4f}")

    if "training_time" in results:
        print(f"\n  Training time: {float(results['training_time']):.1f}s")

    return {
        "model_type": model_type,
        "average_accuracy": summary["average_accuracy"],
        "forgetting": summary["forgetting"],
        "backward_transfer": summary["backward_transfer"],
        "forward_transfer": summary["forward_transfer"],
    }


def print_comparison(rows: list[dict]) -> None:
    """Print a comparison table across runs.

    Args:
        rows: Per-run summary dicts from summarize_run
    """
    print("\n" + "=" * 64)
    print("COMPARISON SUMMARY")
    print("=" * 64)
    print(f"{'Model':<20} {'Avg Acc':>10} {'Forgetting':>12} {'BWT':>10} {'FWT':>10}")
    print("-" * 64)
    for r in rows:
        print(
            f"{r['model_type']:<20} "
            f"{r['average_accuracy']:>10.4f} "
            f"{r['forgetting']:>12.4f} "
            f"{r['backward_transfer']:>10.4f} "
            f"{r['forward_transfer']:>10.4f}"
        )
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(description="PLCM post-training evaluation")
    parser.add_argument(
        "--results", type=str, nargs="+", default=None,
        help="One or more results JSON files to evaluate",
    )
    parser.add_argument(
        "--runs-dir", type=str, default="runs/",
        help="Directory to auto-discover *_results.json in (if --results omitted)",
    )
    args = parser.parse_args()

    paths = args.results if args.results else discover_results(args.runs_dir)

    if not paths:
        print(
            f"No results files found. Looked for --results or "
            f"*_results.json in '{args.runs_dir}'.\n"
            f"Run scripts/train.py first to produce results."
        )
        sys.exit(1)

    rows = []
    for path in paths:
        if not Path(path).exists():
            print(f"Warning: {path} does not exist — skipping.")
            continue
        results = load_results(path)
        rows.append(summarize_run(path, results))

    if len(rows) > 1:
        print_comparison(rows)


if __name__ == "__main__":
    main()
