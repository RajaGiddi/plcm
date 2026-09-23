"""
E13 — tier-0 stack: re-derivation + cross-regime MNIST cell.

Pre-registration: signed contract, regenerated for execution 2026-08-10.
Analysis-only. No training. Existing checkpoints.

TWO QUESTIONS, ONE PASS
  H-S1  Re-derive the tier-0 stack's rho on HAR/OFF. The frozen 0.347 is a
        hardcoded literal with no computation behind it. **No pass bar** — the
        output IS the ledger value, and a bar here would be re-baring against a
        number we already distrust.
  H-S2  Cross-regime: stack pooled rho >= 0.30 on MNIST/OFF (E8's own H-R4 bar,
        reused unchanged).
  H-S3  Composition with adapters on MNIST/ON, with the EVALUABILITY
        PRE-COMMITMENT: if more than half the cells fall under the 0.02
        denominator guard, the verdict is "composition unevaluable at this
        residual scale" — reported, never massaged into a number.

THREE CURES, EACH MEASURED ALONE (composite rule). Stored per task: the era
classifier head and the class prototypes — C x d floats each, no raw data.
  era_head    read theta_T features through the STORED ERA HEAD
  proj        project theta_T features onto the STORED ERA-PROTOTYPE SPAN,
              read with the deployed head
  stack       projection THEN era head  == transport_estimate's `pipeline(I)`,
              which is the object the 0.347 literal names

The stack's definition is IMPORTED, not re-derived: `span_of` and the era head
come from the scripts that already own them. A parallel implementation is how
catch 32 happened.

GATES (contract sec 2, plus the sec 2a amendment)
  * catch 28 — assert_path_identity on every arm x task cell
  * catch 29 — load_era restores task_stats era-correctly
  * catch 32 — label alignment: data comes through the AUDITED loader
    (`load_task_data`, single-pass), and a position-vs-dataset bitwise check
    prints PASS with a shuffled-loader positive control that must FAIL
  * platform gate — `fwd_full` must reproduce each arm's recorded final row
    EXACTLY (0.0000/cell) in the environment the analysis runs in
  * full test split, no N_TEST cap
  * per-cell guard, never post-pooling; forced-inclusion printed beside every
    guarded pooled value

Usage:
    python scripts/e13_stack.py --datasets mnist          # reproduces locally
    modal run --detach modal_runner.py::spawn_analysis --experiment e13
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.channel_decomp import (load, load_task_data, features_and_logits,
                                    refit_probe, assert_path_identity, SEEDS)
from scripts.cure_screen import era_head_of
from scripts.transport_estimate import span_of
from src.data.har_shift import HARShiftBenchmark
from src.data.permuted_mnist import PermutedMNISTBenchmark

CUR = 4                      # current task; old tasks are 0..3
OLD = list(range(CUR))
GUARD = 0.02                 # RHO_MIN_DENOM, applied PER CELL
R_BAR = 0.05
FROZEN_TIER0 = 0.347         # the literal H-S1 replaces
HS2_BAR = 0.30

ARMS = {
    "HAR/OFF":   dict(dirs="runs/ckpt_e5_off_seed{s}/mafc_seed{s}", ad=False,
                      matrix="runs/e5_har_noadapt_seed{s}/mafc_results.json",
                      ds="har", n_classes=6),
    "HAR/v1-ON": dict(dirs="runs/ckpt_e5_seed{s}/mafc_seed{s}", ad=True,
                      matrix="runs/e5diag_har_adapt_seed{s}/mafc_results.json",
                      ds="har", n_classes=6),
    "MNIST/OFF": dict(dirs={s: f"checkpoints/e4_off_seed{s}/mafc_seed{s}" for s in SEEDS},
                      ad=False,
                      matrix={s: f"runs/e4_off_seed{s}/mafc_results.json" for s in SEEDS},
                      ds="mnist", n_classes=10),
    "MNIST/ON":  dict(dirs={42: "checkpoints/fullrank_ref/mafc_seed42",
                            1337: "checkpoints/e4_on_seed1337/mafc_seed1337",
                            2024: "checkpoints/e4_on_seed2024/mafc_seed2024"},
                      ad=True,
                      matrix={42: "runs/fullrank_ref/mafc_results.json",
                              1337: "runs/e4_on_seed1337/mafc_results.json",
                              2024: "runs/e4_on_seed2024/mafc_results.json"},
                      ds="mnist", n_classes=10),
}


def fmt(t, s):
    return t[s] if isinstance(t, dict) else t.format(s=s)


def build_bench(ds, seed, bs=256):
    return (HARShiftBenchmark(num_tasks=5, root=".", batch_size=bs) if ds == "har"
            else PermutedMNISTBenchmark(num_tasks=5, batch_size=bs, seed=seed))


# ------------------------------------------------------------------ gates --
def alignment_gate(bench) -> bool:
    """Catch-32 (contract sec 2a). Position-vs-dataset, bitwise, with a
    shuffled-loader POSITIVE CONTROL that must fail."""
    _, te = bench.get_task_loaders(0)
    ds = te.dataset
    xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=64, shuffle=False):
        xs.append(x); ys.append(y)
        if sum(t.shape[0] for t in xs) >= 64:
            break
    xl, yl = torch.cat(xs)[:64], torch.cat(ys)[:64]
    xd = torch.stack([ds[i][0] for i in range(64)])
    yd = torch.tensor([ds[i][1] for i in range(64)])
    ok = torch.equal(xl, xd) and torch.equal(yl, yd)

    g = torch.Generator().manual_seed(0)
    sx, sy = next(iter(DataLoader(ds, batch_size=64, shuffle=True, generator=g)))
    fired = not (torch.equal(sx, xd) and torch.equal(sy, yd))
    print(f"  catch-32 alignment: sequential order == dataset indexing "
          f"-> {'PASS' if ok else 'FAIL'} | shuffled control "
          f"{'FIRED' if fired else 'DID NOT FIRE'}")
    return ok and fired


@torch.no_grad()
def platform_gate(arm, cfg, device) -> tuple[bool, float]:
    """`fwd_full` must reproduce the recorded final row EXACTLY, here."""
    worst = 0.0
    for s in SEEDS:
        mp = fmt(cfg["matrix"], s)
        if not Path(mp).exists():
            print(f"  platform gate {arm}/s{s}: MATRIX ABSENT {mp}")
            return False, float("nan")
        M = np.array(json.load(open(mp))["accuracy_matrix"], float)
        bench = build_bench(cfg["ds"], s)
        m4 = load(fmt(cfg["dirs"], s), CUR)
        for k in OLD:
            _, te = bench.get_task_loaders(k)
            c = t = 0
            for x, y in te:
                p = m4(x.to(device), store_memories=False, task_hint=k)["logits"]
                c += (p.argmax(-1).cpu() == y).sum().item(); t += y.shape[0]
            worst = max(worst, abs(c / t - float(M[CUR, k])))
    print(f"  platform gate {arm}: max |fwd_full - matrix| = {worst:.4f} -> "
          f"{'PASS' if worst == 0.0 else 'FAIL (run where the data reproduces)'}")
    return worst == 0.0, worst


# ------------------------------------------------------------------ cures --
def score_arm(arm, cfg, device, out_rows):
    n_cls = cfg["n_classes"]
    for s in SEEDS:
        d = fmt(cfg["dirs"], s)
        bench = build_bench(cfg["ds"], s)
        # AUDITED loader, full split (caps raised, single-pass discipline kept).
        # Contract: "Full test split, no N_TEST cap". The TRAIN cap (N_TRAIN=4000)
        # is NOT lifted — it is identical in every cell and is what makes the
        # refit recipe comparable across the program. Uncapping it would change
        # the recipe, not just the sample.
        data = load_task_data(bench, n_tasks=5, n_test=10**9)
        m4 = load(d, CUR)
        for k in OLD:
            xtr, ytr, xte, yte = data[k]
            m_era = load(d, k)
            assert_path_identity(m_era, xte, k, cfg["ad"], device,
                                 label=f"{arm}/s{s}/era{k}", verbose=False)
            assert_path_identity(m4, xte, k, cfg["ad"], device,
                                 label=f"{arm}/s{s}/t4", verbose=False)
            eh = era_head_of(m_era, k)          # STORED era head
            dh = era_head_of(m4, k)             # deployed head at theta_T

            f_e_tr, _ = features_and_logits(m_era, xtr, ytr, k, cfg["ad"], device)
            f_e_te, l_e = features_and_logits(m_era, xte, yte, k, cfg["ad"], device)
            f_4_tr, _ = features_and_logits(m4, xtr, ytr, k, cfg["ad"], device)
            f_4_te, l_4 = features_and_logits(m4, xte, yte, k, cfg["ad"], device)

            acc_ceiling = float((l_e.argmax(1) == yte).float().mean())
            acc_orig = float((l_4.argmax(1) == yte).float().mean())
            acc_rc = refit_probe(f_e_tr, ytr, f_e_te, yte, n_cls)
            acc_r4 = refit_probe(f_4_tr, ytr, f_4_te, yte, n_cls)
            R = acc_rc - acc_ceiling

            # STORED era prototypes -> their span. Same construction as
            # transport_estimate; span_of is imported, not re-derived.
            present = sorted(set(int(v) for v in ytr))
            B = torch.stack([f_e_tr[ytr == c].mean(0) for c in present]
                            ).numpy().astype(np.float64)
            Pb = span_of(B)
            P = torch.from_numpy((Pb @ Pb.T).astype(np.float32))

            with torch.no_grad():
                cures = {
                    "era_head": float((eh(f_4_te).argmax(1) == yte).float().mean()),
                    "proj": float((dh(f_4_te @ P).argmax(1) == yte).float().mean()),
                    "stack": float((eh(f_4_te @ P).argmax(1) == yte).float().mean()),
                }
            out_rows.append(dict(arm=arm, seed=s, task=k, n_test=int(yte.shape[0]),
                                 acc_orig=acc_orig, acc_ceiling=acc_ceiling,
                                 acc_refit=acc_r4, R=R, **cures))
            print(f"    {arm:<10} s{s} T{k}  floor {acc_orig:.4f} refit {acc_r4:.4f} "
                  f"R {R:+.4f} | era_head {cures['era_head']:.4f} "
                  f"proj {cures['proj']:.4f} stack {cures['stack']:.4f}", flush=True)
        del m4


def rho_cells(rows, cure):
    """Per-cell rho against the BIAS-ADJUSTED denominator (era-head family)."""
    out = []
    for r in rows:
        D = r["acc_refit"] - r["R"] - r["acc_orig"]
        rho = (r[cure] - r["acc_orig"]) / D if abs(D) > 1e-12 else float("nan")
        se = math.sqrt(max(r[cure] * (1 - r[cure]), 0) / r["n_test"]) / max(abs(D), 1e-9)
        out.append({"seed": r["seed"], "task": r["task"], "D": D, "rho": rho, "se": se})
    return out


def pooled(cells, guarded=True):
    keep = [c for c in cells if (not guarded or c["D"] > GUARD) and c["rho"] == c["rho"]]
    if not keep:
        return float("nan"), float("nan"), 0, len(cells)
    m = sum(c["rho"] for c in keep) / len(keep)
    se = math.sqrt(sum(c["se"] ** 2 for c in keep)) / len(keep)
    return m, se, len(keep), len(cells)


def main():
    ap = argparse.ArgumentParser(description="E13 tier-0 stack")
    ap.add_argument("--datasets", default="mnist,har")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/e13_stack.json")
    args = ap.parse_args()
    device = torch.device(args.device)
    want = set(args.datasets.split(","))

    print("=" * 104)
    print("E13 — TIER-0 STACK: re-derivation (H-S1) + cross-regime MNIST (H-S2/H-S3)")
    print("=" * 104)

    rows, gated_out = [], {}
    for arm, cfg in ARMS.items():
        if cfg["ds"] not in want:
            continue
        print(f"\n{'-'*104}\n{arm}\n{'-'*104}")
        if not Path(fmt(cfg["dirs"], SEEDS[0])).exists():
            print(f"  ABSENT — checkpoints missing at {fmt(cfg['dirs'], SEEDS[0])}")
            gated_out[arm] = "checkpoints absent"
            continue
        if not alignment_gate(build_bench(cfg["ds"], SEEDS[0])):
            gated_out[arm] = "catch-32 alignment gate failed"; continue
        ok, worst = platform_gate(arm, cfg, device)
        if not ok:
            gated_out[arm] = f"platform gate failed (max |d| {worst:.4f})"; continue
        score_arm(arm, cfg, device, rows)

    print("\n" + "=" * 104)
    print("PER-ARM RESULTS   (guarded pooled rho | forced-inclusion beside it)")
    print("=" * 104)
    summary = {}
    for arm in ARMS:
        ar = [r for r in rows if r["arm"] == arm]
        if not ar:
            print(f"  {arm:<10} ABSENT — {gated_out.get(arm, 'not run')}")
            continue
        Rm = float(np.mean([r["R"] for r in ar]))
        print(f"  {arm:<10} pooled R {Rm:+.4f} -> {'PASS' if abs(Rm) <= R_BAR else 'FAIL'}"
              f"   ({len(ar)} cells)")
        summary[arm] = {"R": Rm, "cures": {}}
        for cure in ("era_head", "proj", "stack"):
            cells = rho_cells(ar, cure)
            g, gse, gk, n = pooled(cells, True)
            f, fse, fk, _ = pooled(cells, False)
            print(f"    {cure:<10} guarded {g:+.3f} [{g-1.96*gse:+.3f},{g+1.96*gse:+.3f}] "
                  f"({gk}/{n})   forced {f:+.3f} ({fk}/{n})")
            summary[arm]["cures"][cure] = {"guarded": g, "ci": [g-1.96*gse, g+1.96*gse],
                                           "kept": gk, "n": n, "forced": f}

    print("\n" + "=" * 104)
    print("HYPOTHESES")
    print("=" * 104)
    if "HAR/OFF" in summary:
        st = summary["HAR/OFF"]["cures"]["stack"]
        print(f"  H-S1  tier-0 stack on HAR/OFF = {st['guarded']:+.3f} "
              f"[{st['ci'][0]:+.3f}, {st['ci'][1]:+.3f}]  (forced {st['forced']:+.3f})")
        print(f"        vs the frozen literal {FROZEN_TIER0} -> delta "
              f"{st['guarded']-FROZEN_TIER0:+.3f}. NO BAR — this value IS the ledger "
              f"entry, superseding the literal.")
    else:
        print("  H-S1  ABSENT — HAR/OFF did not clear its gates here")
    if "MNIST/OFF" in summary:
        st = summary["MNIST/OFF"]["cures"]["stack"]
        met = st["guarded"] >= HS2_BAR
        print(f"  H-S2  MNIST/OFF stack {st['guarded']:+.3f} vs {HS2_BAR} -> "
              f"{'MET' if met else 'NOT MET'}  (forced {st['forced']:+.3f})")
    if "MNIST/ON" in summary:
        st = summary["MNIST/ON"]["cures"]["stack"]
        frac_excl = 1 - st["kept"] / max(st["n"], 1)
        if frac_excl > 0.5:
            print(f"  H-S3  {st['kept']}/{st['n']} cells survive the {GUARD} guard "
                  f"({frac_excl*100:.0f}% excluded) -> **COMPOSITION UNEVALUABLE AT "
                  f"THIS RESIDUAL SCALE** (pre-committed verdict)")
        else:
            print(f"  H-S3  MNIST/ON stack {st['guarded']:+.3f} vs {HS2_BAR} -> "
                  f"{'MET' if st['guarded'] >= HS2_BAR else 'NOT MET'} "
                  f"({st['kept']}/{st['n']} cells)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"rows": rows, "summary": summary, "absent": gated_out},
              open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
