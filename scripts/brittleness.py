"""
Frozen-feature brittleness control: LSTM vs MLP (+ permutation-invariant reference).

Pre-registration: docs/BRITTLENESS_frozen_prereg.md

Design (no adapters, no memory, no replay anywhere):
    1. Train encoder + head on task 0.
    2. FREEZE the encoder.
    3. For each task k, train a FRESH linear head on the frozen features of
       task k and measure test accuracy.
    Plus a permutation-invariant reference: a fresh linear head on raw 784-dim
    pixels per task, which absorbs any permutation by construction (~0.90).

Metric: mean accuracy over tasks 1-4 ("frozen transfer"). Task 0 is reported
separately as an in-distribution sanity check.

Usage:
    python scripts/brittleness.py                      # all 3 seeds, both arms
    python scripts/brittleness.py --seeds 42           # smoke
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

from src.data.permuted_mnist import PermutedMNISTBenchmark

# ---- locked protocol constants (docs/BRITTLENESS_frozen_prereg.md §5) ----
SEEDS = [42, 1337, 2024]
NUM_TASKS = 5
ENCODER_EPOCHS = 10
HEAD_EPOCHS = 5
BATCH = 128
LR = 1e-3
HEAD_LR = 1e-3
HIDDEN = 256
RECIPE_SEED = 0          # fixed init seed for every head, so arms differ only in features
# -------------------------------------------------------------------------


class LSTMEncoder(nn.Module):
    """28 timesteps x 28 features -> final cell state (the PLCM readout path)."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.rnn = nn.LSTM(28, hidden, batch_first=True)

    def forward(self, x):                      # x: [B, 28, 28]
        _, (_, c) = self.rnn(x)
        return c[-1]                           # [B, hidden]


class MLPEncoderStandalone(nn.Module):
    """784 -> hidden -> hidden (flattened; sees all pixels simultaneously)."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
        )

    def forward(self, x):                      # x: [B, 28, 28]
        return self.net(x.reshape(x.shape[0], -1))


def build_encoder(arch: str) -> nn.Module:
    return LSTMEncoder() if arch == "lstm" else MLPEncoderStandalone()


def train_encoder(enc, loader, device, epochs=ENCODER_EPOCHS):
    """Train encoder + a throwaway head on task 0, then the encoder is frozen."""
    head = nn.Linear(HIDDEN, 10).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=LR)
    enc.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            nn.functional.cross_entropy(head(enc(x)), y).backward()
            opt.step()
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


@torch.no_grad()
def features(enc, loader, device, raw=False):
    F, Y = [], []
    for x, y in loader:
        x = x.to(device)
        F.append((x.reshape(x.shape[0], -1) if raw else enc(x)).cpu())
        Y.append(y)
    return torch.cat(F), torch.cat(Y)


def probe_accuracy(ftr_tr, y_tr, ftr_te, y_te, device):
    """Fresh linear head, fixed recipe, never reused across tasks."""
    torch.manual_seed(RECIPE_SEED)
    head = nn.Linear(ftr_tr.shape[1], 10).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
    ftr_tr, y_tr = ftr_tr.to(device), y_tr.to(device)
    n = ftr_tr.shape[0]
    for _ in range(HEAD_EPOCHS):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            nn.functional.cross_entropy(head(ftr_tr[idx]), y_tr[idx]).backward()
            opt.step()
    with torch.no_grad():
        pred = head(ftr_te.to(device)).argmax(1).cpu()
    return float((pred == y_te).float().mean())


def run_arm(arch: str, seed: int, device) -> dict:
    """arch in {'lstm', 'mlp', 'raw'}; 'raw' skips the encoder entirely."""
    torch.manual_seed(seed)
    bench = PermutedMNISTBenchmark(num_tasks=NUM_TASKS, batch_size=BATCH, seed=seed)

    enc = None
    if arch != "raw":
        enc = build_encoder(arch).to(device)
        tr0, _ = bench.get_task_loaders(0)
        enc = train_encoder(enc, tr0, device)

    accs = []
    for k in range(NUM_TASKS):
        tr, te = bench.get_task_loaders(k)
        f_tr, y_tr = features(enc, tr, device, raw=(arch == "raw"))
        f_te, y_te = features(enc, te, device, raw=(arch == "raw"))
        accs.append(probe_accuracy(f_tr, y_tr, f_te, y_te, device))

    return {
        "arch": arch, "seed": seed,
        "per_task": accs,
        "task0": accs[0],
        "frozen_transfer": float(np.mean(accs[1:])),  # tasks 1-4
    }


def main():
    ap = argparse.ArgumentParser(description="Frozen-feature brittleness control")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--out", type=str, default="runs/brittleness/")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )
    print(f"Device: {device}")
    os.makedirs(args.out, exist_ok=True)

    results = []
    for seed in args.seeds:
        for arch in ("lstm", "mlp", "raw"):
            r = run_arm(arch, seed, device)
            results.append(r)
            print(f"  seed {seed} {arch:4s}: task0={r['task0']:.4f}  "
                  f"frozen_transfer(T1-4)={r['frozen_transfer']:.4f}  "
                  f"per_task={[round(a,3) for a in r['per_task']]}")
        with open(os.path.join(args.out, "brittleness.json"), "w") as f:
            json.dump(results, f, indent=2)

    def agg(arch):
        v = [r["frozen_transfer"] for r in results if r["arch"] == arch]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), 0.0)

    lstm_m, lstm_s = agg("lstm")
    mlp_m, mlp_s = agg("mlp")
    raw_m, _ = agg("raw")
    gap = (mlp_m - lstm_m) * 100

    print("\n" + "=" * 66)
    print("FROZEN-FEATURE BRITTLENESS (mean over seeds, tasks 1-4)")
    print("=" * 66)
    print(f"  LSTM frozen transfer : {lstm_m:.4f} +/- {lstm_s:.4f}")
    print(f"  MLP  frozen transfer : {mlp_m:.4f} +/- {mlp_s:.4f}")
    print(f"  raw-pixel REFERENCE  : {raw_m:.4f}   (permutation-invariant by construction)")
    print(f"  GAP (MLP - LSTM)     : {gap:+.1f}pp")
    branch = ("STANDS (>= +15pp)" if gap >= 15 else
              ("WEAK (+5 to +15pp)" if gap > 5 else "DIES (<= +5pp) -> strike the claim"))
    print(f"  PRE-REGISTERED BRANCH: {branch}")

    # Amended prereg 3: the raw line is a permutation-invariant REFERENCE, not a
    # floor. Falling below it means that encoder traded permutation-robustness
    # for task-0 specialization. Both below => the joint finding supersedes the
    # architecture attribution.
    below = {"LSTM": lstm_m < raw_m, "MLP": mlp_m < raw_m}
    print(f"  below the invariant reference?  LSTM: {below['LSTM']}   MLP: {below['MLP']}")
    joint = below["LSTM"] and below["MLP"]
    if joint:
        print()
        print("  *** JOINT FINDING (supersedes the architecture attribution) ***")
        print(f"      BOTH frozen encoders transfer below the permutation-invariant")
        print(f"      reference (LSTM {lstm_m:.4f}, MLP {mlp_m:.4f} vs {raw_m:.4f}).")
        print("      The failure is representational COMMITMENT generally, not")
        print("      recurrence specifically. Report with equal prominence.")
    elif any(below.values()):
        arm = "LSTM" if below["LSTM"] else "MLP"
        print(f"    -> {arm} traded permutation-robustness for task-0 specialization.")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"lstm": [lstm_m, lstm_s], "mlp": [mlp_m, mlp_s],
                   "raw_invariant_reference": raw_m, "gap_pp": gap,
                   "branch": branch, "below_reference": below,
                   "joint_finding": joint}, f, indent=2)


if __name__ == "__main__":
    main()
