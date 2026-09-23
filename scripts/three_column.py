"""
E11 Step 2 — three-column scope check (catch 28).

Contract: docs/E11_instrument_correction.md

For every (arm, seed, old task k) prints, side by side:

  matrix    the accuracy matrix's final row R[4,k] -- what was REPORTED
  fwd_full  PLCM.forward over the FULL test loader -- the deployed path, re-run
  instr_v4  the corrected instrument (deployed path, instrument's test subsample)
  instr_v3  the OLD instrument (hand-rebuilt readout on raw c_t) -- the defect

matrix vs fwd_full isolates STATE/DATA differences (step 3's residual: same code
path, so any gap is which checkpoint or which examples).
instr_v4 vs instr_v3 isolates the CATCH-28 DEFECT itself (same examples, same
checkpoint, different tensor).

Verdicts are computed from the arrays printed beside them (catch 22): no verdict
string is written to mirror an expectation.

Usage:
    python scripts/three_column.py --benchmark mnist
    python scripts/three_column.py --benchmark har_shift      # needs the HAR zip
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.channel_decomp import (load, load_task_data, features_and_logits,
                                    assert_path_identity, N_TEST)
from src.data.har_shift import HARShiftBenchmark
from src.data.har_subject import HARSubjectBenchmark
from src.data.permuted_mnist import PermutedMNISTBenchmark

TASKS = [0, 1, 2, 3]

# benchmark -> {arm: (ckpt template or dict, use_adapter, results-json template)}
ARMS = {
    "mnist": {
        "E4/OFF": ({s: f"checkpoints/e4_off_seed{s}/mafc_seed{s}" for s in (42, 1337, 2024)},
                   False, {s: f"runs/e4_off_seed{s}/mafc_results.json" for s in (42, 1337, 2024)}),
        "E4/ON":  ({42: "checkpoints/fullrank_ref/mafc_seed42",
                    1337: "checkpoints/e4_on_seed1337/mafc_seed1337",
                    2024: "checkpoints/e4_on_seed2024/mafc_seed2024"},
                   True, {42: "runs/fullrank_ref/mafc_results.json",
                          1337: "runs/e4_on_seed1337/mafc_results.json",
                          2024: "runs/e4_on_seed2024/mafc_results.json"}),
    },
    "har_shift": {
        "E5/OFF":   ("runs/ckpt_e5_off_seed{s}/mafc_seed{s}", False,
                     "runs/e5_har_noadapt_seed{s}/mafc_results.json"),
        "E5/v1-ON": ("runs/ckpt_e5_seed{s}/mafc_seed{s}", True,
                     "runs/e5diag_har_adapt_seed{s}/mafc_results.json"),
    },
    "har_subject": {
        "E10/OFF": ("runs/ckpt_e10_off_seed{s}/mafc_seed{s}", False,
                    "runs/e10_off_seed{s}/mafc_results.json"),
        "E10/ON":  ("runs/ckpt_e10_on_seed{s}/mafc_seed{s}", True,
                    "runs/e10_on_seed{s}/mafc_results.json"),
    },
}


def fmt(t, s):
    return t[s] if isinstance(t, dict) else t.format(s=s)


def build_bench(key, seed, batch_size=256):
    if key == "har_subject":
        return HARSubjectBenchmark(num_tasks=5, root=".", batch_size=batch_size)
    if key == "har_shift":
        return HARShiftBenchmark(num_tasks=5, root=".", batch_size=batch_size)
    return PermutedMNISTBenchmark(num_tasks=5, batch_size=batch_size, seed=seed)


def collect_full(loader):
    """Whole test split as tensors, in loader order (single pass, pairs intact)."""
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
    return torch.cat(xs), torch.cat(ys)


@torch.no_grad()
def deployed_full(model, loader, task_k, device):
    """Exactly what metrics.evaluate_task does: the deployed path, full test set."""
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, store_memories=False, task_hint=task_k)["logits"]
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.shape[0]
    return correct / total, total


@torch.no_grad()
def instrument_v3(model, x, y, task_k, use_adapter, device, bs=256):
    """The SUPERSEDED instrument, verbatim: readout hand-rebuilt on raw c_t.

    Kept only to size the defect. Never used for a reported number again.
    """
    key = str(task_k)
    correct = 0
    for i in range(0, x.shape[0], bs):
        xb = x[i:i + bs].to(device)
        if use_adapter and key in model.task_adapters:
            xb = model._apply_adapter(xb, key)
        lstm_out, (_, c_n) = model.lstm(xb)
        c_t = c_n[-1]
        o_t = torch.sigmoid(model.output_gate(torch.cat([lstm_out[:, -1, :], c_t], -1)))
        feat = o_t * torch.tanh(c_t)
        head = (model.task_classifiers[key]
                if getattr(model, "use_task_heads", False) and key in model.task_classifiers
                else model.classifier)
        correct += (head(feat).argmax(1).cpu() == y[i:i + bs]).sum().item()
    return correct / x.shape[0]


def main():
    ap = argparse.ArgumentParser(description="E11 step 2: three-column scope check")
    ap.add_argument("--benchmark", default="mnist",
                    choices=["mnist", "har_shift", "har_subject"])
    ap.add_argument("--seeds", default="42,1337,2024")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    # The runs used configs/mafc_phase1.yaml (batch_size 128). Evaluating the
    # same weights on the same examples at a DIFFERENT batch size changes GEMM
    # blocking and so the last bits of every logit — which flips argmaxes for
    # samples sitting near a tie. At near-chance accuracy there are many.
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]

    rows, missing = [], []
    for arm, (ckpt_t, use_ad, res_t) in ARMS[args.benchmark].items():
        for seed in seeds:
            d, rj = fmt(ckpt_t, seed), fmt(res_t, seed)
            if not Path(d).exists() or not Path(rj).exists():
                missing.append((arm, seed, d if not Path(d).exists() else rj))
                continue
            mat = np.array(json.load(open(rj))["accuracy_matrix"], dtype=float)
            bench = build_bench(args.benchmark, seed, args.batch_size)
            data = load_task_data(bench, n_tasks=5)
            model = load(d, 4)
            for k in TASKS:
                xtr, ytr, xte, yte = data[k]
                # The permanent gate, on every cell this script reports.
                assert_path_identity(model, xte, k, use_ad, device,
                                     label=f"{arm}/s{seed}/T{k}", verbose=False)
                _, te_loader = bench.get_task_loaders(k)
                acc_full, n_full = deployed_full(model, te_loader, k, device)
                _, l4 = features_and_logits(model, xte, yte, k, use_ad, device)
                # The instrument on the FULL test set. Separates "wrong tensor"
                # from "fewer examples": instr_full vs matrix isolates the
                # instrument, instr_v4 vs instr_full isolates the subsample.
                xf, yf = collect_full(te_loader)
                _, lf = features_and_logits(model, xf, yf, k, use_ad, device)
                rows.append(dict(
                    arm=arm, seed=seed, task=k,
                    matrix=float(mat[4, k]),
                    fwd_full=acc_full, n_full=n_full,
                    instr_v4=float((l4.argmax(1) == yte).float().mean()),
                    instr_full=float((lf.argmax(1) == yf).float().mean()),
                    instr_v3=instrument_v3(model, xte, yte, k, use_ad, device),
                    n_sub=int(xte.shape[0]),
                ))

    print("=" * 104)
    print(f"E11 STEP 2 — THREE-COLUMN SCOPE CHECK   benchmark={args.benchmark}")
    print("=" * 104)
    if missing:
        for a, s, p in missing:
            print(f"  MISSING  {a}/seed{s}: {p}")
    print(f"\n  {'arm':<10}{'seed':>6}{'task':>5}{'matrix':>10}{'fwd_full':>10}"
          f"{'instr_full':>11}{'instr_v4':>10}{'instr_v3':>10}{'|ifull-mat|':>12}"
          f"{'|v3-mat|':>10}{'n_full':>8}{'n_sub':>7}")
    for r in rows:
        print(f"  {r['arm']:<10}{r['seed']:>6}{r['task']:>5}{r['matrix']:>10.4f}"
              f"{r['fwd_full']:>10.4f}{r['instr_full']:>11.4f}{r['instr_v4']:>10.4f}"
              f"{r['instr_v3']:>10.4f}"
              f"{abs(r['instr_full']-r['matrix']):>12.4f}"
              f"{abs(r['instr_v3']-r['matrix']):>10.4f}"
              f"{r['n_full']:>8}{r['n_sub']:>7}")

    print("\n" + "=" * 104)
    print("READ")
    print("=" * 104)
    for arm in dict.fromkeys(r["arm"] for r in rows):
        rs = [r for r in rows if r["arm"] == arm]
        d_fwd = float(np.mean([abs(r["fwd_full"] - r["matrix"]) for r in rs]))
        d_if = float(np.mean([abs(r["instr_full"] - r["matrix"]) for r in rs]))
        d_v3 = float(np.mean([abs(r["instr_v3"] - r["matrix"]) for r in rs]))
        d_def = float(np.mean([abs(r["instr_full"] - r["instr_v3"]) for r in rs]))
        d_samp = float(np.mean([abs(r["instr_v4"] - r["instr_full"]) for r in rs]))
        sub_only = all(r["n_sub"] >= r["n_full"] for r in rs)
        # Verdicts derived from the means printed on this same line.
        v_state = ("state/data CLEAN (fwd reproduces the matrix)" if d_fwd < 1e-6
                   else f"state/data GAP {d_fwd:+.4f}")
        v_instr = ("instrument EXACT (corrected instrument == deployed path)"
                   if d_if < 1e-6 else f"instrument GAP {d_if:.4f} on the same examples")
        v_floor = ("floor SOUND (defect moved nothing)" if d_def < 1e-6
                   else f"floor AFFECTED (defect moves acc by {d_def:.4f} mean abs)")
        print(f"  {arm:<10} mean|fwd-matrix| {d_fwd:.4f}   mean|instr_full-matrix| {d_if:.4f}   "
              f"mean|v3-matrix| {d_v3:.4f}   mean|corrected-v3| {d_def:.4f}   "
              f"mean|subsample effect| {d_samp:.4f}")
        print(f"  {'':<10} -> {v_state}; {v_instr}; {v_floor}"
              f"{'; instrument used the FULL test set' if sub_only else '; instrument SUBSAMPLED'}")

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
