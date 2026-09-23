"""
Mechanical completion check for training runs (local or Modal fan-out).

Catch 18: launch artifacts read as completion evidence. A results directory is
created by os.makedirs at container start, BEFORE training — so directory
existence proves a job launched, not that it finished. Under preemption-and-
restart, a directory can also hold a truncated result. Counting directories is
the infrastructure cousin of the n=1 problem: a cheap signal standing in for the
expensive one it does not contain.

STANDING DEFINITION OF "a run finished" — all three must hold:
  (a) accuracy matrix is num_tasks x num_tasks, fully populated, no NaN
  (b) task_history has num_tasks entries, each with its full epoch count
  (c) file mtime postdates any preemption/restart event you are checking against

On PASS the checker prints the run's headline numbers, not just a checkmark, so
the verification output IS the raw table and no transcription step sits between
"verified" and "reported".

Usage:
    python scripts/verify_runs.py 'runs/e1_*'
    python scripts/verify_runs.py 'runs/e1_*' --tasks 5 --epochs 10 --after 2026-07-31T23:40
"""

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import sys

import numpy as np


def check(path: str, n_tasks: int, n_epochs: int, after_ts: float | None) -> dict:
    """Return a verdict dict for one results JSON."""
    name = os.path.basename(os.path.dirname(path))
    out = {"run": name, "path": path, "ok": False, "problems": []}
    try:
        d = json.load(open(path))
    except Exception as e:
        out["problems"].append(f"unreadable: {type(e).__name__}")
        return out

    # (a) accuracy matrix complete
    M = np.array(d.get("accuracy_matrix", []), dtype=float)
    if M.shape != (n_tasks, n_tasks):
        out["problems"].append(f"matrix shape {M.shape} != ({n_tasks},{n_tasks})")
    elif np.isnan(M).any():
        out["problems"].append(f"matrix has {int(np.isnan(M).sum())} NaN cells")

    # (b) task history complete
    th = d.get("task_history", [])
    if len(th) != n_tasks:
        out["problems"].append(f"task_history has {len(th)} tasks, expected {n_tasks}")
    else:
        for t in th:
            got = len(t.get("epochs", []))
            # task 0 may legitimately run epochs_first_task; accept >= n_epochs
            if got < n_epochs:
                out["problems"].append(
                    f"task {t.get('task_id')} has {got} epochs, expected >= {n_epochs}")

    # (c) mtime postdates the reference event
    mtime = os.path.getmtime(path)
    if after_ts is not None and mtime < after_ts:
        out["problems"].append(
            f"mtime {dt.datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M} predates --after")

    out["ok"] = not out["problems"]
    if M.size and M.shape == (n_tasks, n_tasks) and not np.isnan(M).any():
        final = M[-1]
        out.update({
            "avg": float(d.get("average_accuracy", np.nan)),
            "forget": float(d.get("forgetting", np.nan)),
            "diag": float(np.diag(M).mean()),
            "ret": float(final[:-1].mean()),
            # checksum of the final row: two runs that agree here agree everywhere
            # that matters for the results table
            "final_row_sha": hashlib.sha1(
                np.round(final, 6).tobytes()).hexdigest()[:8],
            "mtime": dt.datetime.fromtimestamp(mtime).strftime("%m-%d %H:%M"),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description="Mechanical run-completion verifier")
    ap.add_argument("patterns", nargs="+", help="glob(s) over run directories")
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--after", type=str, default=None,
                    help="ISO timestamp; results must be newer (e.g. a preemption event)")
    args = ap.parse_args()

    after_ts = None
    if args.after:
        after_ts = dt.datetime.fromisoformat(args.after).timestamp()

    paths = []
    for pat in args.patterns:
        for d in sorted(glob.glob(pat)):
            paths += sorted(glob.glob(os.path.join(d, "*_results.json")))
    if not paths:
        print("No results JSON found under:", args.patterns)
        print("(Empty directories mean runs LAUNCHED, not finished — see catch 18.)")
        sys.exit(1)

    print(f"{'run':<34}{'AVG':>8}{'RET':>8}{'DIAG':>8}{'forget':>8}{'final-row':>11}{'mtime':>13}")
    print("-" * 90)
    results = [check(p, args.tasks, args.epochs, after_ts) for p in paths]
    for r in results:
        if r["ok"]:
            print(f"{r['run']:<34}{r['avg']:>8.4f}{r['ret']:>8.4f}{r['diag']:>8.4f}"
                  f"{r['forget']:>8.4f}{r['final_row_sha']:>11}{r['mtime']:>13}")
        else:
            print(f"{r['run']:<34}  FAIL: {'; '.join(r['problems'])}")

    ok = [r for r in results if r["ok"]]
    print("-" * 90)
    print(f"  {len(ok)}/{len(results)} complete")
    if len(ok) != len(results):
        print("\n  Re-run only the failing cells; the table waits for complete data.")
        sys.exit(2)


if __name__ == "__main__":
    main()
