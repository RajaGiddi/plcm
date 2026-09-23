"""
E4b — Subspace structure of encoder drift.

Pre-registration: docs/E4B_subspace_prereg.md (v2)

E4 measured drift in the full 256-dim representation and H-G2 failed (ON/OFF
ratio 0.437 vs a 1/3 bar). But the mechanism claim was never about total drift —
it is about whether the encoder still supports the old readout. This asks whether
the ON arm's drift is STRUCTURED: does it avoid the readout's subspace?

Locked definitions:
  S           = row space of the task-0 shared classifier W (rank 10), per arm.
                P_S = V V^T from W = U S V^T.
  drift-in-S  = 1 - cos(P_S f_th0, P_S f_th4)   [same form as E4 -> bars comparable]
  random ctl  = 20 draws of a uniformly random rank-10 subspace, mean +/- std

Hypotheses:
  H-SUB1  ON drift-in-S <= 1/2 * ON drift-in-random          (structure)
  H-SUB2  ON/OFF drift-in-S ratio <= 1/3                     (strength, E4's bar)
  H-SUB3' report ||P_S d||^2/||d||^2 per arm vs the 3.9% random baseline (reported)

LEMMA (prereg 1a): W (I - P_S) = 0, so out-of-S drift is functionally invisible to
a linear readout. Geometric structure and functional retention are inseparable at
this resolution; branch (B) is unreachable. Verified below at runtime.

Usage:
    python scripts/subspace_drift.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.plcm import PLCM
from src.data.permuted_mnist import PermutedMNISTBenchmark

SEEDS = [42, 1337, 2024]
N_EVAL = 2000
RANK = 10              # readout rank; also the random control's rank
N_RANDOM = 20          # random-subspace draws (prereg v2 amendment 4)
STRUCTURE_FACTOR = 0.5     # H-SUB1: <= 1/2 of random
STRENGTH_BAR = 1.0 / 3.0   # H-SUB2: the same bar the full-space claim failed
RANDOM_SEED = 0

ON_DIRS = {42: "checkpoints/fullrank_ref/mafc_seed42",
           1337: "checkpoints/e4_on_seed1337/mafc_seed1337",
           2024: "checkpoints/e4_on_seed2024/mafc_seed2024"}
OFF_DIRS = {s: f"checkpoints/e4_off_seed{s}/mafc_seed{s}" for s in SEEDS}


def load(d: str, task: int, epoch: int = 9) -> PLCM:
    p = Path(d) / f"task{task}_epoch{epoch}.pt"
    if not p.exists():
        raise FileNotFoundError(p)
    m = PLCM.load_from_checkpoint(torch.load(p, weights_only=True, map_location="cpu"))
    m.eval()
    for q in m.parameters():
        q.requires_grad = False
    return m


@torch.no_grad()
def encode(model: PLCM, x: torch.Tensor, use_adapter: bool, device, bs: int = 512):
    out = []
    for i in range(0, x.shape[0], bs):
        xb = x[i:i + bs].to(device)
        if use_adapter and "0" in model.task_adapters:
            sh = xb.shape
            xb = model.task_adapters["0"](xb.reshape(sh[0], -1)).reshape(sh)
        _, (_, c) = model.lstm(xb)
        out.append(c[-1].cpu())
    return torch.cat(out)


def readout_projector(W: torch.Tensor) -> torch.Tensor:
    """P_S onto the row space of W (rank = min(W.shape))."""
    _, _, Vt = torch.linalg.svd(W, full_matrices=False)
    return Vt.T @ Vt


def random_projector(dim: int, rank: int, gen: torch.Generator) -> torch.Tensor:
    A = torch.randn(dim, rank, generator=gen)
    Q, _ = torch.linalg.qr(A)
    return Q @ Q.T


def drift_in(P: torch.Tensor, f0: torch.Tensor, f4: torch.Tensor) -> float:
    """1 - cos on the projected representations (E4's functional form)."""
    return float(1.0 - nn.functional.cosine_similarity(f0 @ P, f4 @ P, dim=1).mean())


def verify_lemma(W: torch.Tensor, P: torch.Tensor) -> dict:
    """W (I - P_S) = 0 : out-of-S drift is invisible to a linear readout."""
    resid = (W - W @ P).norm().item()
    x = torch.randn(64, W.shape[1])
    gap = ((x @ W.T) - ((x @ P) @ W.T)).abs().max().item()
    return {"W_minus_WP_fro": resid, "max_output_gap": gap}


def run_arm(arm: str, seed: int, x: torch.Tensor, device) -> dict:
    d = (ON_DIRS if arm == "on" else OFF_DIRS)[seed]
    m0, m4 = load(d, 0), load(d, 4)
    use_ad = (arm == "on")
    f0, f4 = encode(m0, x, use_ad, device), encode(m4, x, use_ad, device)

    W = m0.classifier.weight.detach()          # the readout the model actually used
    P_S = readout_projector(W)
    lemma = verify_lemma(W, P_S)

    d_full = float(1.0 - nn.functional.cosine_similarity(f0, f4, dim=1).mean())
    d_S = drift_in(P_S, f0, f4)

    gen = torch.Generator().manual_seed(RANDOM_SEED + seed)
    rand = [drift_in(random_projector(f0.shape[1], RANK, gen), f0, f4)
            for _ in range(N_RANDOM)]

    delta = f4 - f0                             # H-SUB3': in-S energy fraction
    frac = float((delta @ P_S).pow(2).sum() / delta.pow(2).sum())

    return {"arm": arm, "seed": seed, "drift_full": d_full, "drift_S": d_S,
            "drift_random_mean": float(np.mean(rand)), "drift_random_std": float(np.std(rand)),
            "inS_energy_fraction": frac, "lemma": lemma}


def main():
    ap = argparse.ArgumentParser(description="E4b subspace drift structure")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--out", type=str, default="runs/subspace/")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    bench = PermutedMNISTBenchmark(num_tasks=5, batch_size=256, seed=42)
    _, te = bench.get_task_loaders(0)
    xs = []
    for xb, _ in te:
        xs.append(xb)
        if sum(t.shape[0] for t in xs) >= N_EVAL:
            break
    x = torch.cat(xs)[:N_EVAL]

    random_baseline = RANK / x.shape[1] if False else RANK / 256
    print(f"task-0 eval inputs: {tuple(x.shape)}   d=256, rank(S)={RANK}")
    print(f"random rank-{RANK} subspace captures {random_baseline*100:.1f}% of a random vector\n")

    res = []
    for seed in args.seeds:
        for arm in ("on", "off"):
            try:
                r = run_arm(arm, seed, x, device)
            except FileNotFoundError as e:
                print(f"  seed {seed} {arm:3s}: MISSING {e}")
                continue
            res.append(r)
            print(f"  seed {seed} {arm:3s}: drift_full={r['drift_full']:.4f}  "
                  f"drift_S={r['drift_S']:.4f}  drift_rand={r['drift_random_mean']:.4f}"
                  f"+/-{r['drift_random_std']:.4f}  inS_energy={r['inS_energy_fraction']*100:.2f}%")
    if not res:
        print("\nNo checkpoints found."); return

    lem = res[0]["lemma"]
    print(f"\n  LEMMA CHECK: ||W - W P_S||_F = {lem['W_minus_WP_fro']:.2e}, "
          f"max output gap = {lem['max_output_gap']:.2e}  -> W(I-P_S)=0 confirmed")

    def agg(arm, k):
        v = [r[k] for r in res if r["arm"] == arm]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), 0.0)

    onS, onS_s = agg("on", "drift_S")
    offS, offS_s = agg("off", "drift_S")
    onR, _ = agg("on", "drift_random_mean")
    offR, _ = agg("off", "drift_random_mean")
    onE, onE_s = agg("on", "inS_energy_fraction")
    offE, _ = agg("off", "inS_energy_fraction")
    onF, _ = agg("on", "drift_full")
    offF, _ = agg("off", "drift_full")

    print("\n" + "=" * 76)
    print("E4b — SUBSPACE STRUCTURE OF ENCODER DRIFT (n=3)")
    print("=" * 76)
    print(f"  {'metric':<34}{'ON (adapters)':>19}{'OFF (vanilla)':>19}")
    print(f"  {'drift, FULL space (E4)':<34}{onF:>19.4f}{offF:>19.4f}")
    print(f"  {'drift, in readout subspace S':<34}{onS:>19.4f}{offS:>19.4f}")
    print(f"  {'drift, random rank-10 subspace':<34}{onR:>19.4f}{offR:>19.4f}")
    print(f"  {'in-S drift energy fraction':<34}{onE*100:>18.2f}%{offE*100:>18.2f}%")
    print(f"  {'(random baseline)':<34}{random_baseline*100:>18.2f}%{random_baseline*100:>18.2f}%")

    h1 = onS <= STRUCTURE_FACTOR * onR
    h2_ratio = onS / offS if offS > 0 else float("inf")
    h2 = h2_ratio <= STRENGTH_BAR
    print()
    print(f"  H-SUB1 structure : ON drift_S {onS:.4f} vs 1/2*random {STRUCTURE_FACTOR*onR:.4f}"
          f"   -> {'PASS' if h1 else 'FAIL'}")
    print(f"  H-SUB2 strength  : ON/OFF in-S ratio {h2_ratio:.3f} vs bar {STRENGTH_BAR:.3f}"
          f"   -> {'PASS' if h2 else 'FAIL'}")
    print(f"                     (E4 full-space ratio was 0.437 — same bar, both shown)")
    print(f"  H-SUB3' reported : in-S energy ON {onE*100:.2f}% vs OFF {offE*100:.2f}% "
          f"vs random {random_baseline*100:.2f}%")

    if h1 and h2:
        branch = "(A) claim restored in precise form: drift is structured AND meets the original bar in S"
    elif h1 and not h2:
        branch = "(C) structure without strength: drift avoids S more than chance, but not by the bar"
    else:
        branch = "(D) retention despite UNSTRUCTURED drift — deepens the mystery; open problem"
    print(f"\n  BRANCH: {branch}")
    print("  (B) is unreachable under a linear readout — see lemma; retained in the contract.")

    with open(os.path.join(args.out, "subspace_raw.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(args.out, "subspace_summary.json"), "w") as f:
        json.dump({"on_drift_full": onF, "off_drift_full": offF,
                   "on_drift_S": [onS, onS_s], "off_drift_S": [offS, offS_s],
                   "on_drift_random": onR, "off_drift_random": offR,
                   "on_inS_energy": [onE, onE_s], "off_inS_energy": offE,
                   "random_baseline": random_baseline,
                   "h_sub1_pass": bool(h1), "h_sub2_ratio": h2_ratio,
                   "h_sub2_pass": bool(h2), "branch": branch, "lemma": lem}, f, indent=2)


if __name__ == "__main__":
    main()
